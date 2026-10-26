#!/usr/bin/env bash
# =============================================================================
# run_capacity.sh — orchestratore del capacity test k6
# =============================================================================
#
# Cosa fa, in ordine:
#   1. Pre-flight: verifica k6, kubectl, kubeconfig del cluster target
#   2. Scala il deployment a 1 replica SUL CLUSTER TARGET e a 0 sugli altri
#      (per evitare contaminazione via Cilium Global Service)
#   3. Aspetta che le repliche siano pronte (kubectl rollout status)
#   4. Sanity check Ingress (3 retry curl, abort se non risponde 200)
#   5. Snapshot Prometheus PRE: cpu/mem del worker + replica count
#   6. Lancia k6_capacity_test.js con env + JSON output
#   7. Snapshot Prometheus POST
#   8. Restore: scale tutti i cluster -> 1 replica (riporta fleet alla baseline)
#   9. Stampa "next step": riga YAML da incollare in services.yaml
#
# PREREQUISITO: il fleet deve avere service.cilium.io/affinity=local applicata
# sui Service global (decisione architetturale 2026-05-18). Lo state finale dei
# Service global e' definito in deployments/global-service-{frontend,backends}.yaml
# e applicato da setup-dmos-on-clusters.ps1.
#
# IMPORTANTE — eseguire SENZA netem:
#   Il verdetto del capacity test si basa sulla latency p95 LATO CLIENT.
#   Il tc netem aggiunge millisecondi a ogni risposta -> p95 gonfiato ->
#   knee-point falso. Su un cluster con netem attivo (es. ms-cluster-03 con
#   50ms) il test puo' dare "knee N/A" perche' lo SLA p95<200ms diventa
#   irraggiungibile gia' a basso carico. Prima del capacity test:
#       bash scripts/setup-netem-and-ping.sh remove-netem
#   e dopo, per ripristinare la simulazione geografica:
#       bash scripts/setup-netem-and-ping.sh netem-only
#   (NB: il throttle_check.sh invece e' indifferente al netem — misura CPU,
#    non latency. Vedi experiments/THROTTLE-CHECK.md sezione 9.1)
#
# Uso base:
#   bash experiments/run_capacity.sh                       # default cluster-01, frontend
#   bash experiments/run_capacity.sh --cluster ms-cluster-05
#
# Variabili env (opzionali):
#   DMOS_CLUSTER          : default ms-cluster-01
#   DMOS_SERVICE          : default frontend
#   DMOS_STEPS            : default "25,50,75,100,125,150,175,200,250,300"
#   DMOS_STEP_DURATION_S  : default 90
#   DMOS_SETTLE_S         : default 15
#   DMOS_SLA_P95_MS       : default 200
#   DMOS_SLA_ERR_RATE     : default 0.02
#   DMOS_NO_RESTORE       : "1" = NON ripristina (per debug post-mortem)
# =============================================================================

set -euo pipefail

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/experiments/results"
mkdir -p "$RESULTS_DIR"

# ── Defaults / env ───────────────────────────────────────────────────────────
CLUSTER="${DMOS_CLUSTER:-ms-cluster-01}"
SERVICE="${DMOS_SERVICE:-frontend}"
NS="online-boutique"
# KUBECONFIG_DIR: da env; altrimenti derivata da .cache/test-env.json
# (dirname dei kubeconfig_path renderizzati); fallback $HOME/dmos-iac.
if [[ -z "${KUBECONFIG_DIR:-}" && -f "${REPO_ROOT}/.cache/test-env.json" ]]; then
    KUBECONFIG_DIR="$(sed -n 's|.*"kubeconfig_path"[^"]*"\(.*\)/kubeconfig-[^"]*".*|\1|p' \
        "${REPO_ROOT}/.cache/test-env.json" | head -n1 | tr -d '\r')"
fi
KUBECONFIG_DIR="${KUBECONFIG_DIR:-$HOME/dmos-iac}"

# Parse args (override env)
LABEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)    CLUSTER="$2"; shift 2 ;;
        --service)    SERVICE="$2"; shift 2 ;;
        --label-this) LABEL="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,33p' "$0"; exit 0 ;;
        *) echo "Argomento sconosciuto: $1"; exit 2 ;;
    esac
done

LABEL="${LABEL:-${CLUSTER}}"
TS=$(date +%Y%m%d_%H%M)
RESULT_FILE_JSON="${RESULTS_DIR}/capacity_${LABEL}_${TS}.json"

# ── Colors ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'
    YELLOW=$'\e[33m'; RED=$'\e[31m'; DIM=$'\e[2m'; RESET=$'\e[0m'
else
    BOLD=''; CYAN=''; GREEN=''; YELLOW=''; RED=''; DIM=''; RESET=''
fi
hdr()  { printf '\n%s%s%s %s%s\n' "${BOLD}" "${CYAN}" "===" "$*" "${RESET}"; }
step() { printf '  %s→%s %s\n' "${CYAN}" "${RESET}" "$*"; }
ok()   { printf '  %s✓%s %s\n' "${GREEN}" "${RESET}" "$*"; }
warn() { printf '  %s!%s %s\n' "${YELLOW}" "${RESET}" "$*"; }
fail() { printf '  %s✗%s %s\n' "${RED}"    "${RESET}" "$*"; exit 1; }

# ── Helpers ──────────────────────────────────────────────────────────────────
kc_of() {
    # Map ms-cluster-XX -> kubeconfig path
    echo "${KUBECONFIG_DIR}/kubeconfig-$1"
}

scale_to() {
    local cluster=$1 replicas=$2
    local kc; kc=$(kc_of "$cluster")
    if [[ ! -f "$kc" ]]; then
        warn "kubeconfig non trovato per $cluster, skip"
        return 0
    fi
    kubectl --kubeconfig "$kc" -n "$NS" scale deployment "$SERVICE" --replicas="$replicas" >/dev/null 2>&1
    return $?
}

prom_query() {
    # Esegue una query Prometheus e ritorna il valore numerico del primo result.
    # Args: $1 = ip (di un cluster con Prom NodePort 32090), $2 = promql
    local ip=$1 q=$2
    curl -s --max-time 5 "http://${ip}:32090/api/v1/query?query=$(printf '%s' "$q" | jq -sRr @uri)" \
        | jq -r '.data.result[0].value[1] // "n/a"' 2>/dev/null | tr -d '\r' || echo "n/a"
}

# ── 1. Pre-flight ────────────────────────────────────────────────────────────
hdr "Pre-flight"

# Auto-discover di tool installati in path comuni su Windows
# (winget, scoop, chocolatey, Program Files) e li mette nel PATH di sessione.
# Cosi' lo script gira anche se l'utente non ha sistemato il PATH manualmente.
ensure_in_path() {
    local tool=$1
    shift
    command -v "$tool" >/dev/null 2>&1 && return 0
    for candidate in "$@"; do
        if [[ -x "$candidate" ]]; then
            export PATH="$(dirname "$candidate"):$PATH"
            return 0
        fi
    done
    return 1
}

# jq paths comuni
ensure_in_path jq \
    "$HOME/bin/jq.exe" \
    "/c/Users/$(whoami)/AppData/Local/Microsoft/WinGet/Packages/jqlang.jq_Microsoft.Winget.Source_8wekyb3d8bbwe/jq.exe" \
    "/c/Users/$(whoami)/scoop/shims/jq.exe" \
    "/c/ProgramData/chocolatey/bin/jq.exe" \
    "/c/Program Files/jq/bin/jq.exe"

# k6 paths comuni
ensure_in_path k6 \
    "/c/Program Files/k6/k6.exe" \
    "$HOME/scoop/shims/k6.exe" \
    "/c/Users/$(whoami)/AppData/Local/Microsoft/WinGet/Packages/k6.k6_Microsoft.Winget.Source_8wekyb3d8bbwe/k6.exe"

for tool in k6 kubectl jq curl; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail "Tool mancante: $tool (riapri Git Bash dopo install, o aggiungi al PATH)"
    fi
done
ok "Tool presenti (jq=$(which jq), k6=$(which k6))"

KC_TARGET=$(kc_of "$CLUSTER")
[[ -f "$KC_TARGET" ]] || fail "Kubeconfig del cluster target non trovato: $KC_TARGET"
ok "Cluster target: $CLUSTER ($KC_TARGET)"

# Verifica deployment esiste sul target
kubectl --kubeconfig "$KC_TARGET" -n "$NS" get deployment "$SERVICE" >/dev/null 2>&1 \
    || fail "Deployment $SERVICE non trovato in $NS su $CLUSTER"
ok "Deployment $SERVICE presente"

# Verifica .cache/test-env.json
ENV_JSON="${REPO_ROOT}/.cache/test-env.json"
[[ -f "$ENV_JSON" ]] || fail ".cache/test-env.json mancante, lancia render-runtime-files.sh"
ok ".cache/test-env.json pronto"

# Lista cluster dal cache (per scale 0)
ALL_CLUSTERS=$(jq -r '.clusters[].name' "$ENV_JSON" | tr -d '\r')
N_CLUSTERS=$(echo "$ALL_CLUSTERS" | wc -l | tr -d ' ')
echo "$ALL_CLUSTERS" | grep -qx "$CLUSTER" || fail "Cluster $CLUSTER non in test-env.json"
ok "Fleet: $N_CLUSTERS cluster (target $CLUSTER)"

# IP del target per query Prometheus
TARGET_IP=$(jq -r --arg n "$CLUSTER" '.clusters[] | select(.name==$n) | .ip' "$ENV_JSON" | tr -d '\r')
ok "IP target (per Prom snapshot): $TARGET_IP"

# ── 2. Scale: 1 replica su target, 0 sugli altri ─────────────────────────────
hdr "Scale deployment"

step "Scale $CLUSTER → 1 replica"
scale_to "$CLUSTER" 1
sleep 1

for c in $ALL_CLUSTERS; do
    if [[ "$c" != "$CLUSTER" ]]; then
        step "Scale $c → 0 repliche (no contamination)"
        scale_to "$c" 0
    fi
done
ok "Scale richieste inviate"

# ── 3. Wait rollout su target ────────────────────────────────────────────────
hdr "Aspetto $SERVICE Ready su $CLUSTER (max 2 min)"
if kubectl --kubeconfig "$KC_TARGET" -n "$NS" rollout status deployment "$SERVICE" --timeout=2m; then
    ok "$SERVICE Ready su $CLUSTER"
else
    fail "$SERVICE non e' tornato Ready entro 2 min"
fi

# ── 3.5 Verifica precondizione: affinity=local applicata sui Service global ──
# La decisione architetturale (2026-05-18) e' di mantenere affinity=local
# permanentemente nei manifest deployments/global-service-*.yaml. Lo script
# verifica solo che lo stato live sia coerente; se manca, warna ma procede.
hdr "Verifica precondizione: affinity=local sui Service global"
SERVICES_FOR_AFFINITY="frontend cartservice productcatalogservice recommendationservice checkoutservice"
missing_affinity=0
for svc in $SERVICES_FOR_AFFINITY; do
    aff=$(kubectl --kubeconfig "$KC_TARGET" -n "$NS" get svc "$svc" \
        -o jsonpath='{.metadata.annotations.service\.cilium\.io/affinity}' 2>/dev/null)
    if [[ "$aff" != "local" ]]; then
        warn "$svc su $CLUSTER non ha service.cilium.io/affinity=local (e' '$aff')"
        missing_affinity=$((missing_affinity+1))
    fi
done
if [[ "$missing_affinity" -gt 0 ]]; then
    warn "$missing_affinity Service non hanno affinity=local. Possibili 500 spurii sotto carico."
    warn "Re-apply: kubectl apply -f deployments/global-service-{frontend,backends}.yaml"
else
    ok "Tutti i Service global hanno affinity=local"
fi

# ── 4. Snapshot Prometheus PRE ───────────────────────────────────────────────
hdr "Snapshot Prometheus (PRE-test)"
PRE_CPU=$(prom_query "$TARGET_IP" "sum(rate(container_cpu_usage_seconds_total{namespace=\"$NS\",pod=~\"$SERVICE-.*\"}[1m]))")
PRE_MEM=$(prom_query "$TARGET_IP" "sum(container_memory_working_set_bytes{namespace=\"$NS\",pod=~\"$SERVICE-.*\"})")
PRE_REPLICAS=$(prom_query "$TARGET_IP" "kube_deployment_status_replicas{namespace=\"$NS\",deployment=\"$SERVICE\"}")
echo "  CPU rate (1m):  $PRE_CPU cores"
echo "  Memory:         $PRE_MEM bytes"
echo "  Replicas:       $PRE_REPLICAS"

# ── 4.5 Sanity check: l'Ingress NodePort 30080 risponde 200? ────────────────
# Evitiamo di sprecare 17 min di k6 se l'Ingress e' rotto. Faccio 3 retry per
# coprire il caso "frontend appena restartato, riconnessione gRPC in corso".
hdr "Sanity check Ingress (curl pre-test)"
INGRESS_URL=$(jq -r --arg n "$CLUSTER" '.clusters[] | select(.name==$n) | .ingress_url' "$ENV_JSON" | tr -d '\r')
step "Probe $INGRESS_URL/ (max 3 tentativi, timeout 5s ognuno)"
sanity_ok=0
for attempt in 1 2 3; do
    code=$(curl -s -o /dev/null --max-time 5 -w "%{http_code}" "$INGRESS_URL/" 2>/dev/null)
    if [[ "$code" == "200" ]]; then
        ok "Ingress risponde 200 al tentativo $attempt"
        sanity_ok=1
        break
    fi
    warn "Tentativo $attempt: HTTP=$code (aspetto 5s)"
    sleep 5
done
if [[ "$sanity_ok" != "1" ]]; then
    fail "L'Ingress non risponde 200. Stop prima di sprecare 17 min di k6."
    echo "  Debug:"
    echo "    1. kubectl --kubeconfig $KC_TARGET -n $NS get pods"
    echo "    2. kubectl --kubeconfig $KC_TARGET -n kube-system get ingressclass cilium"
    echo "    3. curl -v $INGRESS_URL/"
    echo "    4. Se HTTP 500 su /: forse cartservice gRPC e' giu'. Rollout restart deploy frontend"
    echo "    5. Se TCP timeout: vedi DMOS-DEPLOYMENT-PIPELINE.md sezione troubleshooting Cilium Ingress"
    exit 1
fi

# ── 5. Run k6 ────────────────────────────────────────────────────────────────
# Default DMOS_STEPS alzato il 2026-05-18 dopo run preliminare (knee@60 rps
# con affinity=none, ma 100 rps ancora gestiti). Con affinity=local atteso
# knee piu' alto. Il nuovo range 25..300 cerca il knee VERO della HW puro.
DEFAULT_STEPS="25,50,75,100,125,150,175,200,250,300"
hdr "Run k6 capacity test (target $CLUSTER)"
echo "  Steps:         ${DMOS_STEPS:-$DEFAULT_STEPS}"
echo "  Step duration: ${DMOS_STEP_DURATION_S:-90}s (settle 15s + measure 90s)"
echo "  Output JSON:   $RESULT_FILE_JSON"
echo "  Custom summary:experiments/results/capacity_summary.json"
echo ""

cd "$REPO_ROOT"

# k6 e' gia' stato risolto in pre-flight via ensure_in_path
K6_BIN="$(command -v k6)"

"$K6_BIN" run \
    -e DMOS_CLUSTER="$CLUSTER" \
    -e DMOS_STEPS="${DMOS_STEPS:-$DEFAULT_STEPS}" \
    -e DMOS_STEP_DURATION_S="${DMOS_STEP_DURATION_S:-90}" \
    -e DMOS_SLA_P95_MS="${DMOS_SLA_P95_MS:-200}" \
    -e DMOS_SLA_ERR_RATE="${DMOS_SLA_ERR_RATE:-0.02}" \
    --out "json=${RESULT_FILE_JSON}" \
    experiments/k6_capacity_test.js
K6_EXIT=$?

# ── 6. Snapshot Prometheus POST ──────────────────────────────────────────────
hdr "Snapshot Prometheus (POST-test)"
POST_CPU=$(prom_query "$TARGET_IP" "sum(rate(container_cpu_usage_seconds_total{namespace=\"$NS\",pod=~\"$SERVICE-.*\"}[1m]))")
POST_MEM=$(prom_query "$TARGET_IP" "sum(container_memory_working_set_bytes{namespace=\"$NS\",pod=~\"$SERVICE-.*\"})")
echo "  CPU rate (1m):  $POST_CPU cores (was $PRE_CPU)"
echo "  Memory:         $POST_MEM bytes (was $PRE_MEM)"

# ── 7. Restore: tutti i cluster a 1 replica ──────────────────────────────────
# Lo affinity NON va toccato qui (e' parte del setup permanente, vedi
# deployments/global-service-*.yaml).
if [[ "${DMOS_NO_RESTORE:-0}" != "1" ]]; then
    hdr "Restore: scale tutti i cluster → 1 replica"
    for c in $ALL_CLUSTERS; do
        step "Scale $c → 1 replica"
        scale_to "$c" 1
    done
    ok "Restore completato (l'effettivo Ready dei pod puo' richiedere 30-60s)"
else
    warn "DMOS_NO_RESTORE=1 → stato NON ripristinato. Esegui manualmente:"
    for c in $ALL_CLUSTERS; do
        kc=$(kc_of "$c")
        echo "  kubectl --kubeconfig $kc -n $NS scale deployment $SERVICE --replicas=1"
    done
fi

# ── 8. Next step suggerito ───────────────────────────────────────────────────
hdr "Risultato"
SUMMARY_JSON="${REPO_ROOT}/experiments/results/capacity_summary.json"
if [[ -f "$SUMMARY_JSON" ]]; then
    RECOMMENDED=$(jq -r '.knee.recommended_capacity_req_per_sec // "n/a"' "$SUMMARY_JSON" | tr -d '\r')
    KNEE_RPS=$(jq -r '.knee.rps_per_replica // "n/a"' "$SUMMARY_JSON" | tr -d '\r')

    if [[ "$RECOMMENDED" != "n/a" && "$RECOMMENDED" != "null" ]]; then
        echo "  Knee-point misurato:    $KNEE_RPS rps/replica"
        echo "  Raccomandazione:        $RECOMMENDED rps/replica (80% del knee)"
        echo ""
        echo "  Per applicare, modifica config/services.yaml:"
        echo ""
        echo "    services:"
        echo "      $SERVICE:"
        echo "        capacity_req_per_sec: $RECOMMENDED"
        echo ""
        echo "  Risultati archiviati in: $RESULT_FILE_JSON"
        echo "  Summary JSON: $SUMMARY_JSON"
    else
        warn "Knee non determinato — controlla la tabella del k6 summary sopra"
    fi
else
    warn "Summary JSON non generato (k6 fallito?)"
fi

exit ${K6_EXIT:-0}
