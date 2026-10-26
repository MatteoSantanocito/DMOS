#!/usr/bin/env bash
# =============================================================================
# preflight_runs.sh — gate GO/NO-GO prima di una sessione di run del paper
# =============================================================================
#
# SCOPO
# -----
# Dopo una manutenzione / reboot dei cluster, parecchio stato VOLATILE si
# perde silenziosamente e invalida gli esperimenti geo-aware SENZA dare errori
# evidenti. Questo script e' il cancello da superare PRIMA di lanciare i run:
# verifica, con verdetto unico GO / NO-GO, tutto cio' che un reboot rompe.
#
# COSA CONTROLLA (in ordine, accumula problemi e da' un verdetto finale)
#   [0] Raggiungibilita'  ingress + Prometheus dei 6 cluster   (--watch per
#                         attendere che la manutenzione finisca)
#   [1] Nodi Ready        kubectl get nodes: tutti Ready, nessuno cordonato
#   [2] netem / geo-RTT   IL controllo chiave: SSH ai worker, `tc qdisc show`,
#                         confronto col delay atteso (test-env.json). Un reboot
#                         AZZERA netem -> la RTT geografica sparisce e l'esperimento
#                         misura una LAN piatta. --fix riapplica netem.
#   [3] ping_rtt sanity   Prometheus: ping_rtt_mean_seconds deve mostrare RTT
#                         shaped (>50ms da qualche parte), non LAN pura.
#   [4] Cilium annotations  i 5 Service di online-boutique devono essere global
#                         (service.cilium.io/global, o la chiave legacy
#                         io.cilium/global-service) con un'affinity.
#   [5] Stato pod         online-boutique: nessun pod non-Running/Pending/
#                         CrashLoop; frontend Ready. (Reset di stato fra i run.)
#   [6] Layer DMOS        delega a scripts/dmos-ready-check.py (config, metriche
#                         Hubble/kube-state, client Prometheus/K8s).
#
# USO
#   bash experiments/preflight_runs.sh                # gate completo, una volta
#   bash experiments/preflight_runs.sh --watch        # attende che i 6 cluster
#                                                      # tornino su, poi il gate
#   bash experiments/preflight_runs.sh --fix          # se netem e' resettato,
#                                                      # lo riapplica e ricontrolla
#   bash experiments/preflight_runs.sh --skip-dmos    # salta il check [6]
#
# EXIT
#   0 = GO     (nessun FAIL; warning non bloccanti ammessi)
#   1 = NO-GO  (almeno un FAIL critico: NON lanciare i run)
#   2 = prerequisiti mancanti (tool/file)
# =============================================================================

set -uo pipefail   # NB: niente -e: vogliamo accumulare i fallimenti, non abortire

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache"
ENV_JSON="${CACHE_DIR}/test-env.json"
NETEM_SCRIPT="${REPO_ROOT}/scripts/setup-netem-and-ping.sh"
READY_CHECK="${REPO_ROOT}/scripts/dmos-ready-check.py"

# Service gestiti, devono avere le annotation Cilium global-service
SERVICES=(frontend cartservice productcatalogservice checkoutservice recommendationservice)
NS="online-boutique"

# Tolleranza nel confronto del delay netem (ms) e soglia "RTT shaped"
NETEM_TOL_MS=3
RTT_SHAPED_MIN_S=0.05     # almeno una serie ping_rtt > 50ms => netem attivo

# ── Flag ──────────────────────────────────────────────────────────────────
WATCH=0; FIX=0; SKIP_DMOS=0; WATCH_INTERVAL=15
while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)     WATCH=1; shift ;;
        --watch-interval) WATCH_INTERVAL="$2"; shift 2 ;;
        --fix)       FIX=1; shift ;;
        --skip-dmos) SKIP_DMOS=1; shift ;;
        -h|--help)
            sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Argomento sconosciuto: $1"; exit 2 ;;
    esac
done

# ── Colori ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'
    YELLOW=$'\e[33m'; RED=$'\e[31m'; DIM=$'\e[2m'; RESET=$'\e[0m'
else
    BOLD=''; CYAN=''; GREEN=''; YELLOW=''; RED=''; DIM=''; RESET=''
fi
hdr()  { printf '\n%s%s=== %s%s\n' "${BOLD}" "${CYAN}" "$*" "${RESET}"; }
step() { printf '  %s->%s %s\n' "${CYAN}" "${RESET}" "$*"; }
okk()  { printf '  %s[OK]%s   %s\n' "${GREEN}" "${RESET}" "$*"; }
info() { printf '  %s%s%s\n' "${DIM}" "$*" "${RESET}"; }

# ── Accumulo problemi (no exit immediato) ────────────────────────────────────
FAILS=(); WARNS=()
crit() { printf '  %s[FAIL]%s %s\n' "${RED}"    "${RESET}" "$*"; FAILS+=("$*"); }
warn() { printf '  %s[WARN]%s %s\n' "${YELLOW}" "${RESET}" "$*"; WARNS+=("$*"); }
die()  { printf '\n  %s[x] %s%s\n' "${RED}" "$*" "${RESET}"; exit 2; }

# ── Prerequisiti ─────────────────────────────────────────────────────────────
for t in jq kubectl ssh curl python; do
    command -v "$t" >/dev/null 2>&1 || die "Tool mancante nel PATH: $t"
done
[[ -f "$ENV_JSON" ]] || die ".cache/test-env.json assente — lancia scripts/render-runtime-files.sh"

SSH_USER="$(jq -r '.ssh.user' "$ENV_JSON" | tr -d '\r')"
SSH_KEY="$(jq -r '.ssh.key'  "$ENV_JSON" | tr -d '\r')"
SSH_KEY="${SSH_KEY/#\~/$HOME}"
[[ -f "$SSH_KEY" ]] || die "SSH key non trovata: $SSH_KEY"

SSH_OPTS=(-n -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=6)

kc_of() {  # risolve il kubeconfig_path (assoluto / ~ / relativo a REPO_ROOT)
    local kc="$1"
    kc="${kc/#\~/$HOME}"
    if [[ "$kc" != /* && "$kc" != [A-Za-z]:* ]]; then kc="${REPO_ROOT}/${kc}"; fi
    printf '%s' "$kc"
}

# =============================================================================
# [0] Raggiungibilita' (con eventuale --watch)
# =============================================================================
reachable_count() {  # stampa "<up> <total>"
    local up=0 tot=0 url
    while IFS=$'\t' read -r ing prom; do
        tot=$((tot+1))
        local ic pc
        ic=$(curl -s -o /dev/null --max-time 6 -w "%{http_code}" "${ing}/" 2>/dev/null || echo 000)
        pc=$(curl -s -o /dev/null --max-time 6 -w "%{http_code}" "${prom}/-/healthy" 2>/dev/null || echo 000)
        [[ "$ic" == "200" && "$pc" == "200" ]] && up=$((up+1))
    done < <(jq -r '.clusters[] | [.ingress_url, .prometheus_url] | @tsv' "$ENV_JSON" | tr -d '\r')
    printf '%s %s' "$up" "$tot"
}

if [[ "$WATCH" == "1" ]]; then
    hdr "[0] Attesa ritorno cluster (--watch, ogni ${WATCH_INTERVAL}s, Ctrl-C per uscire)"
    while :; do
        read -r up tot < <(reachable_count)
        ts="$(date +%H:%M:%S)"
        if [[ "$up" -eq "$tot" ]]; then
            okk "${ts}  tutti i ${tot} cluster raggiungibili — proseguo col gate"
            break
        fi
        printf '  %s%s%s  %s/%s cluster su (attendo)...\n' "${DIM}" "$ts" "${RESET}" "$up" "$tot"
        sleep "$WATCH_INTERVAL"
    done
fi

hdr "[0] Raggiungibilita' ingress + Prometheus"
while IFS=$'\t' read -r name ing prom; do
    ic=$(curl -s -o /dev/null --max-time 6 -w "%{http_code}" "${ing}/" 2>/dev/null || echo 000)
    pc=$(curl -s -o /dev/null --max-time 6 -w "%{http_code}" "${prom}/-/healthy" 2>/dev/null || echo 000)
    if [[ "$ic" == "200" && "$pc" == "200" ]]; then
        okk "$name  ingress+prometheus 200"
    else
        crit "$name  ingress=${ic} prometheus=${pc} (VPN giu' o cluster non pronto?)"
    fi
done < <(jq -r '.clusters[] | [.name, .ingress_url, .prometheus_url] | @tsv' "$ENV_JSON" | tr -d '\r')

# =============================================================================
# [1] Nodi Ready
# =============================================================================
hdr "[1] Nodi Ready su ogni cluster"
while IFS=$'\t' read -r name kcp; do
    kc="$(kc_of "$kcp")"
    if [[ ! -f "$kc" ]]; then crit "$name  kubeconfig mancante: $kc"; continue; fi
    # jsonpath: per ogni nodo, la condition Ready + eventuale unschedulable
    out=$(kubectl --kubeconfig "$kc" get nodes \
            -o jsonpath='{range .items[*]}{.metadata.name}{"="}{.status.conditions[?(@.type=="Ready")].status}{":"}{.spec.unschedulable}{"\n"}{end}' \
            2>/dev/null)
    if [[ -z "$out" ]]; then crit "$name  kubectl get nodes non risponde"; continue; fi
    total=0; ready=0; cordoned=0
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        total=$((total+1))
        [[ "$line" == *"=True:"* ]] && ready=$((ready+1))
        [[ "$line" == *":true"* ]] && cordoned=$((cordoned+1))
    done <<< "$out"
    if [[ "$ready" -eq "$total" && "$cordoned" -eq 0 && "$total" -gt 0 ]]; then
        okk "$name  ${ready}/${total} nodi Ready, 0 cordonati"
    else
        crit "$name  ${ready}/${total} Ready, ${cordoned} cordonati"
    fi
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')

# =============================================================================
# [2] netem / geo-RTT  (il controllo chiave dopo un reboot)
# =============================================================================
netem_reset=0
hdr "[2] netem (tc qdisc) sui worker — confronto col delay atteso"
while IFS=$'\t' read -r name ip exp iface; do
    if [[ -z "$iface" || "$iface" == "null" ]]; then
        info "$name  interfaccia non specificata — skip"; continue
    fi
    if [[ -z "$exp" || "$exp" == "0" || "$exp" == "null" ]]; then
        # cluster baseline: ci aspettiamo NESSUN netem (o delay 0)
        q=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "tc qdisc show dev ${iface}" 2>/dev/null || echo "SSHERR")
        if [[ "$q" == "SSHERR" ]]; then warn "$name  SSH fallito (baseline, non bloccante)"; continue; fi
        if echo "$q" | grep -q "netem"; then
            warn "$name  baseline 0ms ma netem PRESENTE: $(echo "$q" | grep -oE 'delay [0-9.]+ms' | head -1)"
        else
            okk "$name  baseline 0ms, nessun netem (atteso)"
        fi
        continue
    fi
    q=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${ip}" "tc qdisc show dev ${iface}" 2>/dev/null || echo "SSHERR")
    if [[ "$q" == "SSHERR" ]]; then crit "$name  SSH al worker fallito — impossibile verificare netem"; continue; fi
    if ! echo "$q" | grep -q "netem"; then
        crit "$name  netem ASSENTE su ${iface} (atteso ${exp}ms) — RESET da reboot!"
        netem_reset=1
        continue
    fi
    meas=$(echo "$q" | grep -oE 'delay [0-9.]+ms' | head -1 | grep -oE '[0-9.]+')
    if [[ -z "$meas" ]]; then
        warn "$name  netem presente ma delay non parsabile: ${q}"
        continue
    fi
    # |meas - exp| <= tol ?
    if awk -v m="$meas" -v e="$exp" -v t="$NETEM_TOL_MS" 'BEGIN{d=m-e; if(d<0)d=-d; exit !(d<=t)}'; then
        okk "$name  netem delay ${meas}ms (atteso ${exp}ms)"
    else
        warn "$name  netem delay ${meas}ms != atteso ${exp}ms (tol ${NETEM_TOL_MS}ms)"
    fi
done < <(jq -r '.clusters[] | [.name, .ip, (.netem.delay_ms // 0), (.netem.interface // "")] | @tsv' "$ENV_JSON" | tr -d '\r')

# Auto-fix se richiesto e netem risulta resettato
if [[ "$netem_reset" == "1" && "$FIX" == "1" ]]; then
    hdr "[2b] --fix: riapplico netem (setup-netem-and-ping.sh netem-only)"
    if [[ -f "$NETEM_SCRIPT" ]]; then
        bash "$NETEM_SCRIPT" netem-only && okk "netem riapplicato — RILANCIA il preflight per riverificare" \
            || crit "riapplicazione netem fallita"
    else
        crit "script netem non trovato: $NETEM_SCRIPT"
    fi
fi

# =============================================================================
# [3] ping_rtt sanity (Prometheus) — netem davvero attivo end-to-end?
# =============================================================================
hdr "[3] ping_rtt_mean_seconds — RTT shaped (non LAN piatta)"
# Basta interrogare un cluster: ping_exporter misura verso tutti i peer.
prom0=$(jq -r '.clusters[0].prometheus_url' "$ENV_JSON" | tr -d '\r')
resp=$(curl -s --max-time 8 "${prom0}/api/v1/query?query=max(ping_rtt_mean_seconds)" 2>/dev/null || echo "")
maxrtt=$(echo "$resp" | jq -r '.data.result[0].value[1] // empty' 2>/dev/null)
if [[ -z "$maxrtt" ]]; then
    warn "ping_rtt_mean_seconds assente su ${prom0} (ping_exporter non ancora scrappato? attendi 30-60s)"
else
    if awk -v v="$maxrtt" -v th="$RTT_SHAPED_MIN_S" 'BEGIN{exit !(v>=th)}'; then
        okk "max ping_rtt = $(awk -v v="$maxrtt" 'BEGIN{printf "%.0f", v*1000}')ms — netem sta shapando"
    else
        crit "max ping_rtt = $(awk -v v="$maxrtt" 'BEGIN{printf "%.1f", v*1000}')ms < ${RTT_SHAPED_MIN_S}s — sembra LAN PURA (netem off?)"
    fi
fi

# =============================================================================
# [4] Annotazioni Cilium global sui 5 Service (frontend: affinity none, paper)
# =============================================================================
hdr "[4] Annotazioni Cilium sui Service di ${NS}"
while IFS=$'\t' read -r name kcp; do
    kc="$(kc_of "$kcp")"
    [[ -f "$kc" ]] || { crit "$name  kubeconfig mancante"; continue; }
    missing=""
    for svc in "${SERVICES[@]}"; do
        ann=$(kubectl --kubeconfig "$kc" -n "$NS" get svc "$svc" \
                -o jsonpath='{.metadata.annotations.service\.cilium\.io/global}{"|"}{.metadata.annotations.io\.cilium/global-service}' \
                2>/dev/null || echo "")
        # chiave attuale (service.cilium.io/global) o legacy (io.cilium/global-service)
        if [[ "${ann%%|*}" != "true" && "${ann#*|}" != "true" ]]; then
            missing="${missing} ${svc}"
        fi
    done
    if [[ -z "$missing" ]]; then
        okk "$name  tutti i ${#SERVICES[@]} Service sono global"
    else
        warn "$name  global-service mancante su:${missing}"
    fi
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')

# =============================================================================
# [4b] ClusterMesh — connettivita' mesh, endpoint-sync, global services
# =============================================================================
# Informativo: lo stato del mesh non blocca i run (il serving e' locale di
# proposito), ma e' utile sapere dopo un reboot se il mesh e' tornato su e
# che configurazione global/affinity hanno i Service.
hdr "[4b] ClusterMesh (connettivita', endpoint-sync, global/affinity)"
first_kc="$(kc_of "$(jq -r '.clusters[0].kubeconfig_path' "$ENV_JSON" | tr -d '\r')")"
first_name="$(jq -r '.clusters[0].name' "$ENV_JSON" | tr -d '\r')"

# endpoint-sync flag (determina se i global service condividono endpoint cross-cluster)
es="$(kubectl --kubeconfig "$first_kc" -n kube-system get cm cilium-config \
        -o jsonpath='{.data.clustermesh-enable-endpoint-sync}' 2>/dev/null)"
es="${es:-<assente>}"
if [[ "$es" == "true" ]]; then
    okk "endpoint-sync = true → i global service condividono endpoint cross-cluster"
else
    warn "endpoint-sync = ${es} → global service INERTI: serving locale (nessun overflow cross-cluster)"
fi

# Connettivita' mesh per cluster: "N/M remote clusters ready"
while IFS=$'\t' read -r name kcp; do
    kc="$(kc_of "$kcp")"
    [[ -f "$kc" ]] || { crit "$name  kubeconfig mancante"; continue; }
    st=$(kubectl --kubeconfig "$kc" -n kube-system exec ds/cilium -- \
            cilium-dbg status 2>/dev/null | grep -i "ClusterMesh:" | head -1)
    nm=$(echo "$st" | grep -oE '[0-9]+/[0-9]+ remote clusters ready' | head -1)
    if [[ -z "$nm" ]]; then
        warn "$name  ClusterMesh: stato non leggibile"
        continue
    fi
    n="${nm%%/*}"; rest="${nm#*/}"; m="${rest%% *}"
    if [[ "$n" == "$m" && "$n" -gt 0 ]]; then
        okk "$name  ClusterMesh ${nm}"
    else
        warn "$name  ClusterMesh ${nm} (mesh incompleto — peering rotto dopo reboot?)"
    fi
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')

# Riepilogo global/affinity dei Service (dal primo cluster, uguale sugli altri)
info "Global services e affinity in ${NS} (da ${first_name}):"
kubectl --kubeconfig "$first_kc" -n "$NS" get svc \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.annotations.service\.cilium\.io/global}{.metadata.annotations.io\.cilium/global-service}{"\t"}{.metadata.annotations.service\.cilium\.io/affinity}{"\n"}{end}' \
    2>/dev/null | while IFS=$'\t' read -r svc g a; do
        [[ "$g" == true* ]] && printf '    %sglobal%s %-24s affinity=%s\n' \
            "${CYAN}" "${RESET}" "$svc" "${a:-none}"
    done

# Test definitivo: il frontend ha endpoint REMOTI? (riga del Service + backend
# remoti = cross-cluster attivo). Conta i backend SOLO dell'entry frontend
# (dalla riga con la sua ClusterIP fino alla riga del Service successivo).
# NB: con affinity=local i backend remoti sono in STANDBY e NON compaiono a
# riposo (peso 0): 1 backend attivo è il valore atteso e corretto. Compaiono
# solo quando il locale è unavailable/espulso (failover). Verificato col test
# scale-to-0: i 5 frontend remoti diventano 'active'. Quindi qui interpretiamo
# il conteggio in base all'affinity invece di allarmare.
faff=$(kubectl --kubeconfig "$first_kc" -n "$NS" get svc frontend \
        -o jsonpath='{.metadata.annotations.service\.cilium\.io/affinity}' 2>/dev/null)
fip=$(kubectl --kubeconfig "$first_kc" -n "$NS" get svc frontend \
        -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
if [[ -n "$fip" ]]; then
    nbk=$(kubectl --kubeconfig "$first_kc" -n kube-system exec ds/cilium -- \
            cilium-dbg service list 2>/dev/null | awk -v ip="$fip" '
                # nuova entry di servizio = riga che inizia con un ID numerico
                /^[0-9]+[[:space:]]/ { inblk = ($0 ~ ip) ? 1 : 0 }
                inblk && /=>/ { c++ }
                END { print c+0 }')
    # Paper NCA: affinity none -> il pool del frontend contiene i pod pronti di
    # TUTTI i cluster (>= 6 con 1 replica per cluster).
    if [[ "${nbk:-0}" -gt 1 ]]; then
        okk "frontend: ${nbk} backend nel pool cross-cluster (affinity=${faff:-none})"
    elif [[ "$faff" == "local" ]]; then
        warn "frontend: affinity=local (1 backend attivo + remoti in standby) — il paper usa affinity=none: kubectl apply -f deployments/global-service-frontend.yaml"
    else
        warn "frontend: ${nbk:-0} backend nel datapath con affinity=${faff:-none}: il pool NON e' condiviso fra i cluster (vedi docs/CILIUM-CLUSTERMESH-NOTES.md §4)"
    fi
fi

# =============================================================================
# [5] Stato pod online-boutique (reset / salute)
# =============================================================================
hdr "[5] Stato pod in ${NS} (no Pending/CrashLoop, frontend Ready)"
while IFS=$'\t' read -r name kcp; do
    kc="$(kc_of "$kcp")"
    [[ -f "$kc" ]] || { crit "$name  kubeconfig mancante"; continue; }
    # Conta pod non sani (fase != Running e != Succeeded)
    bad=$(kubectl --kubeconfig "$kc" -n "$NS" get pods --no-headers 2>/dev/null \
            | awk '$3!="Running" && $3!="Completed" && $3!="Succeeded" {print $1":"$3}' | tr '\n' ' ')
    fe=$(kubectl --kubeconfig "$kc" -n "$NS" get deploy frontend \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
    fe="${fe:-0}"
    if [[ -n "$bad" ]]; then
        warn "$name  pod non-Running: ${bad}"
    elif [[ "$fe" -lt 1 ]]; then
        crit "$name  frontend non Ready (readyReplicas=${fe})"
    else
        okk "$name  pod tutti Running, frontend Ready (${fe})"
    fi
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')

# =============================================================================
# [6] Layer DMOS (delega al checker esistente)
# =============================================================================
if [[ "$SKIP_DMOS" == "0" ]]; then
    hdr "[6] Layer DMOS — scripts/dmos-ready-check.py"
    if [[ -f "$READY_CHECK" ]]; then
        ( cd "$REPO_ROOT" && python "$READY_CHECK" )
        rc=$?
        if [[ "$rc" -eq 0 ]]; then okk "dmos-ready-check: READY"
        elif [[ "$rc" -eq 2 ]]; then warn "dmos-ready-check: ready con warning"
        else crit "dmos-ready-check: NOT READY (rc=${rc})"; fi
    else
        warn "dmos-ready-check.py non trovato — skip layer DMOS"
    fi
else
    info "[6] layer DMOS saltato (--skip-dmos)"
fi

# =============================================================================
# Verdetto finale
# =============================================================================
hdr "VERDETTO"
printf '  %sFAIL: %d   WARN: %d%s\n' "${BOLD}" "${#FAILS[@]}" "${#WARNS[@]}" "${RESET}"
if [[ "${#WARNS[@]}" -gt 0 ]]; then
    printf '\n  %sWARNING (non bloccanti):%s\n' "${YELLOW}" "${RESET}"
    for w in "${WARNS[@]}"; do printf '    %s- %s%s\n' "${YELLOW}" "$w" "${RESET}"; done
fi
if [[ "${#FAILS[@]}" -gt 0 ]]; then
    printf '\n  %sFAIL (bloccanti):%s\n' "${RED}" "${RESET}"
    for f in "${FAILS[@]}"; do printf '    %s- %s%s\n' "${RED}" "$f" "${RESET}"; done
    printf '\n  %s%sNO-GO — NON lanciare i run finche'\'' i FAIL non sono risolti.%s\n' "${BOLD}" "${RED}" "${RESET}"
    [[ "$netem_reset" == "1" && "$FIX" == "0" ]] && \
        printf '  %sSuggerimento: rilancia con --fix per riapplicare netem.%s\n' "${DIM}" "${RESET}"
    exit 1
fi
printf '\n  %s%sGO — puoi lanciare i run.%s\n' "${BOLD}" "${GREEN}" "${RESET}"
[[ "${#WARNS[@]}" -gt 0 ]] && printf '  %s(controlla i warning sopra: non bloccano ma vale la pena guardarli)%s\n' "${DIM}" "${RESET}"
exit 0
