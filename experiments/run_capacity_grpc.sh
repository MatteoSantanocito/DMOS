#!/usr/bin/env bash
# =============================================================================
# run_capacity_grpc.sh — capacity test gRPC dei BACKEND, eseguito IN-CLUSTER
# =============================================================================
#
# SCOPO
# -----
# Misurare empiricamente `capacity_req_per_sec` dei microservizi BACKEND gRPC di
# Online Boutique (cartservice, productcatalogservice, recommendationservice,
# checkoutservice), valore finora STIMATO a occhio in config/services.yaml.
#
# PERCHE' IN-CLUSTER (e non piu' via kubectl port-forward)
# -------------------------------------------------------
# La prima versione lanciava k6 sul PC e raggiungeva il backend con
# `kubectl port-forward`. Misura SBAGLIATA, dimostrata dai dati:
#   1. Tassa fissa ~47ms per richiesta (round-trip PC -> API server -> pod):
#      p50 inchiodato a ~47-54ms su TUTTI i backend, a qualsiasi carico ->
#      cartservice/productcatalog (SLA p95 50ms) fallivano gia' a pod scarico.
#   2. Il listener del port-forward CROLLAVA sotto carico: i log mostravano
#      "dial tcp 127.0.0.1:PORT: connection actively refused" -> i "knee" di
#      recommendation/checkout erano il limite del port-forward, non del pod.
# Soluzione: k6 gira come Job DENTRO il cluster e colpisce il backend via
# ClusterIP (es. cartservice:7070). gRPC pod-to-pod e' sub-millisecondo, regge
# migliaia di connessioni, e attraversa lo stesso proxy L7 Cilium del traffico
# reale frontend->backend (le CNP l7-visibility-* ammettono fromEndpoints:[{}],
# cioe' qualsiasi pod del namespace online-boutique).
#
# COSA FA, per ogni backend
# -------------------------
#   1. loadgenerator -> 0 su tutti i cluster (niente traffico di fondo)
#   2. backend target -> 1 replica sul cluster scelto (misura 1 pod)
#   3. dipendenze del backend -> 4 repliche sul cluster target (non strozzano)
#   4. crea un Job k6 (immagine grafana/k6) nel namespace online-boutique,
#      con lo script + il proto montati da una ConfigMap
#   5. segue `kubectl logs -f` del pod k6 (output pulito, righe appese)
#   6. estrae il JSON di summary dai log (fra marcatori) -> results/
#   7. cancella il Job, ripristina le repliche delle dipendenze
#   8. stampa il `capacity_req_per_sec` consigliato (80% del knee)
#
# CAVEAT DI ISOLAMENTO (vedi experiments/CAPACITY-GRPC.md sez. 4)
#   cartservice            pulito (solo redis-cart, veloce)
#   productcatalogservice  pulito (in-memory)
#   recommendationservice  chiama productcatalog -> potenziato a 4 repliche
#   checkoutservice        fan-out su 6 servizi -> tutti potenziati; PlaceOrder
#                          testato con carrelli vuoti (vedi doc)
#
# PREREQUISITI
#   - Git Bash su Windows; kubectl e jq nel PATH
#   - experiments/k6_capacity_grpc.js e experiments/demo.proto presenti
#   - kubeconfig dei cluster in $KUBECONFIG_DIR
#   - il cluster deve poter scaricare l'immagine grafana/k6 da Docker Hub
#   - netem indifferente: il traffico k6->backend e' intra-cluster
#
# USO
#   bash experiments/run_capacity_grpc.sh                       # cartservice, cluster-01
#   bash experiments/run_capacity_grpc.sh --backend productcatalogservice
#   bash experiments/run_capacity_grpc.sh --all                 # tutti e 4
#   bash experiments/run_capacity_grpc.sh --backend cartservice --cluster ms-cluster-04
#
# ENV (override opzionali)
#   DMOS_BACKEND DMOS_CLUSTER DMOS_STEPS DMOS_STEP_DURATION_S DMOS_SETTLE_S
#   DMOS_SLA_P95_MS DMOS_SLA_ERR_RATE DMOS_NO_RESTORE=1 DMOS_K6_IMAGE
#   KUBECONFIG_DIR
# =============================================================================

set -euo pipefail

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/experiments/results"
mkdir -p "$RESULTS_DIR"
cd "$REPO_ROOT"

# ── Defaults / env ───────────────────────────────────────────────────────────
NS="online-boutique"
# KUBECONFIG_DIR: da env; altrimenti derivata da .cache/test-env.json
# (dirname dei kubeconfig_path renderizzati); fallback $HOME/dmos-iac.
if [[ -z "${KUBECONFIG_DIR:-}" && -f "${REPO_ROOT}/.cache/test-env.json" ]]; then
    KUBECONFIG_DIR="$(sed -n 's|.*"kubeconfig_path"[^"]*"\(.*\)/kubeconfig-[^"]*".*|\1|p' \
        "${REPO_ROOT}/.cache/test-env.json" | head -n1 | tr -d '\r')"
fi
KUBECONFIG_DIR="${KUBECONFIG_DIR:-$HOME/dmos-iac}"
BACKEND="${DMOS_BACKEND:-cartservice}"
CLUSTER="${DMOS_CLUSTER:-ms-cluster-01}"
K6_IMAGE="${DMOS_K6_IMAGE:-grafana/k6:latest}"
CM_NAME="dmos-k6-cap-scripts"
BOOST_REPLICAS=4
RUN_ALL=0

# ── Colors ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD=$'\e[1m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'
    YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
    BOLD=''; CYAN=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi
hdr()  { printf '\n%s%s%s %s%s\n' "${BOLD}" "${CYAN}" "===" "$*" "${RESET}"; }
step() { printf '  %s->%s %s\n' "${CYAN}" "${RESET}" "$*"; }
ok()   { printf '  %s[ok]%s %s\n' "${GREEN}" "${RESET}" "$*"; }
warn() { printf '  %s[!]%s %s\n'  "${YELLOW}" "${RESET}" "$*"; }
fail() { printf '  %s[x]%s %s\n'  "${RED}"    "${RESET}" "$*"; exit 1; }

usage() {
    cat <<'EOF'
run_capacity_grpc.sh — capacity test gRPC dei backend di Online Boutique (in-cluster)

USO:
  bash experiments/run_capacity_grpc.sh [--backend NAME] [--cluster ms-cluster-XX]
  bash experiments/run_capacity_grpc.sh --all

OPZIONI:
  --backend NAME   cartservice | productcatalogservice |
                   recommendationservice | checkoutservice   (default cartservice)
  --cluster NAME   cluster su cui eseguire il test            (default ms-cluster-01)
  --all            testa tutti e 4 i backend in sequenza
  -h, --help       questo aiuto

ENV: DMOS_STEPS DMOS_STEP_DURATION_S DMOS_SETTLE_S DMOS_SLA_P95_MS
     DMOS_SLA_ERR_RATE DMOS_NO_RESTORE=1 DMOS_K6_IMAGE KUBECONFIG_DIR
EOF
}

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --all)     RUN_ALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Argomento sconosciuto: $1"; echo; usage; exit 2 ;;
    esac
done

# ── Per-backend config ───────────────────────────────────────────────────────
# Imposta: B_PORT  B_METHOD  B_STEPS  B_SLA_P95  B_DEPS
backend_config() {
    case "$1" in
        cartservice)
            B_PORT=7070
            B_METHOD="hipstershop.CartService/GetCart"
            # Range FINE per affinare il knee (2026-05-21): il run precedente
            # dava knee 686 ma a 700 rps il p99 era gia' 698ms (CFS throttling
            # sul limite 300m CPU). Step da 100 rps fra 300 e 800 per trovare
            # il knee p99-pulito (col guard p99 < 4xSLA in k6_capacity_grpc.js).
            B_STEPS="300,400,500,600,700,800,1000,1300"
            B_SLA_P95=50
            B_DEPS=""        # solo redis-cart (veloce, non potenziato)
            ;;
        productcatalogservice)
            B_PORT=3550
            B_METHOD="hipstershop.ProductCatalogService/ListProducts"
            # Range affinato (2026-05-21): knee misurato ~2000 rps (p95 48ms).
            # Step da 500 rps attorno al ginocchio per delimitarlo meglio.
            B_STEPS="500,1000,1500,2000,2500,3000,4000,5000"
            B_SLA_P95=50
            B_DEPS=""        # in-memory, nessuna dipendenza
            ;;
        recommendationservice)
            B_PORT=8080
            B_METHOD="hipstershop.RecommendationService/ListRecommendations"
            # Range affinato (2026-05-21): cliff Python netto fra 125 (p95 3ms)
            # e 150 (p95 1100ms). Step da 10 rps per delimitare il ginocchio.
            B_STEPS="100,110,120,130,140,150,170,200"
            B_SLA_P95=120
            B_DEPS="productcatalogservice"
            ;;
        checkoutservice)
            B_PORT=5050
            B_METHOD="hipstershop.CheckoutService/PlaceOrder"
            # Range affinato (2026-05-21): knee fra 200 (p95 73ms) e 350
            # (p95 949ms). Step da 50 rps per delimitare il ginocchio.
            B_STEPS="100,150,200,250,300,350,450,600"
            B_SLA_P95=500
            B_DEPS="cartservice productcatalogservice currencyservice paymentservice shippingservice emailservice"
            ;;
        *)
            fail "Backend sconosciuto: '$1' (validi: cartservice productcatalogservice recommendationservice checkoutservice)"
            ;;
    esac
}

# ── Helpers ──────────────────────────────────────────────────────────────────
kc_of() { echo "${KUBECONFIG_DIR}/kubeconfig-$1"; }

ensure_in_path() {
    local tool=$1; shift
    command -v "$tool" >/dev/null 2>&1 && return 0
    local candidate
    for candidate in "$@"; do
        if [[ -x "$candidate" ]]; then
            export PATH="$(dirname "$candidate"):$PATH"
            return 0
        fi
    done
    return 1
}

# ── Stato globale (per il cleanup trap) ──────────────────────────────────────
TARGET_KC=""
BOOSTED_DEPS=""
CURRENT_JOB=""
CM_CREATED=0
declare -a RESULTS_SUMMARY=()

# Cleanup: rete di sicurezza per Ctrl-C / errori. Nel flusso normale ogni
# backend cancella il proprio Job e azzera BOOSTED_DEPS, e la ConfigMap viene
# rimossa a fine script; qui si raccolgono solo i casi di uscita anomala.
cleanup() {
    local ec=$?
    if [[ -n "${CURRENT_JOB:-}" && -n "${TARGET_KC:-}" ]]; then
        kubectl --kubeconfig "$TARGET_KC" -n "$NS" delete job "$CURRENT_JOB" \
            --ignore-not-found >/dev/null 2>&1 || true
    fi
    if [[ "${DMOS_NO_RESTORE:-0}" != "1" && -n "${BOOSTED_DEPS:-}" && -n "${TARGET_KC:-}" ]]; then
        printf '\n  %s[!]%s Cleanup: ripristino dipendenze -> 1 replica\n' "${YELLOW}" "${RESET}"
        local d
        for d in $BOOSTED_DEPS; do
            kubectl --kubeconfig "$TARGET_KC" -n "$NS" scale deployment "$d" --replicas=1 >/dev/null 2>&1 || true
        done
    fi
    if [[ "${CM_CREATED:-0}" == "1" && -n "${TARGET_KC:-}" ]]; then
        kubectl --kubeconfig "$TARGET_KC" -n "$NS" delete configmap "$CM_NAME" \
            --ignore-not-found >/dev/null 2>&1 || true
    fi
    return $ec
}
# EXIT: cleanup normale. INT/TERM: cleanup e poi STOP davvero — senza questo,
# un Ctrl-C (o un segnale spurio) eseguiva cleanup ma lasciava PROSEGUIRE lo
# script, col Job appena cancellato e i dati troncati.
trap cleanup EXIT
trap 'echo; warn "Interrotto (SIGINT): pulizia e stop"; trap - EXIT; cleanup; exit 130' INT
trap 'echo; warn "Ricevuto SIGTERM: pulizia e stop";  trap - EXIT; cleanup; exit 143' TERM

# ── Pre-flight ───────────────────────────────────────────────────────────────
hdr "Pre-flight"

ensure_in_path jq \
    "$HOME/bin/jq.exe" \
    "/c/Users/$(whoami)/AppData/Local/Microsoft/WinGet/Packages/jqlang.jq_Microsoft.Winget.Source_8wekyb3d8bbwe/jq.exe" \
    "/c/Users/$(whoami)/scoop/shims/jq.exe" \
    "/c/ProgramData/chocolatey/bin/jq.exe"

for tool in kubectl jq; do
    command -v "$tool" >/dev/null 2>&1 \
        || fail "Tool mancante: $tool (riapri Git Bash dopo l'install, o aggiungi al PATH)"
done
ok "Tool presenti (kubectl, jq=$(command -v jq))"

K6_SCRIPT="${REPO_ROOT}/experiments/k6_capacity_grpc.js"
PROTO="${REPO_ROOT}/experiments/demo.proto"
[[ -f "$K6_SCRIPT" ]] || fail "Script k6 mancante: $K6_SCRIPT"
[[ -f "$PROTO" ]]     || fail "Proto mancante: $PROTO"
ok "Script k6 + proto presenti (eseguiti in-cluster come ConfigMap)"
ok "Immagine k6: $K6_IMAGE"

TARGET_KC="$(kc_of "$CLUSTER")"
[[ -f "$TARGET_KC" ]] || fail "Kubeconfig del cluster target non trovato: $TARGET_KC"
ok "Cluster target: $CLUSTER ($TARGET_KC)"

# Scopri i cluster dai file kubeconfig
declare -a ALL_CLUSTERS=()
for kc in "${KUBECONFIG_DIR}"/kubeconfig-ms-cluster-*; do
    [[ -f "$kc" ]] || continue
    ALL_CLUSTERS+=("$(basename "$kc" | sed 's/^kubeconfig-//')")
done
[[ ${#ALL_CLUSTERS[@]} -gt 0 ]] || fail "Nessun kubeconfig-ms-cluster-* in $KUBECONFIG_DIR"
ok "Fleet: ${#ALL_CLUSTERS[@]} cluster (${ALL_CLUSTERS[*]})"

# ── Disattiva loadgenerator ovunque ─────────────────────────────────────────
hdr "Disattivo loadgenerator su tutti i cluster"
for c in "${ALL_CLUSTERS[@]}"; do
    kc="$(kc_of "$c")"
    if kubectl --kubeconfig "$kc" -n "$NS" scale deployment loadgenerator --replicas=0 >/dev/null 2>&1; then
        ok "$c: loadgenerator -> 0"
    else
        warn "$c: loadgenerator non scalato (assente o cluster irraggiungibile)"
    fi
done
warn "loadgenerator resta a 0 -> riaccendilo prima della fase di traffico:"
warn "  for c in 01 02 03 04 05 06; do kubectl --kubeconfig $KUBECONFIG_DIR/kubeconfig-ms-cluster-\$c -n $NS scale deploy loadgenerator --replicas=1; done"

# ── ConfigMap con script k6 + proto (una volta sola) ────────────────────────
hdr "Creo la ConfigMap $CM_NAME (script k6 + proto)"
kubectl --kubeconfig "$TARGET_KC" -n "$NS" delete configmap "$CM_NAME" \
    --ignore-not-found >/dev/null 2>&1 || true
kubectl --kubeconfig "$TARGET_KC" -n "$NS" create configmap "$CM_NAME" \
    --from-file=k6_capacity_grpc.js=experiments/k6_capacity_grpc.js \
    --from-file=demo.proto=experiments/demo.proto >/dev/null
CM_CREATED=1
ok "ConfigMap creata su $CLUSTER"

# =============================================================================
# run_one_backend BACKEND — capacity test gRPC di un singolo backend (in-cluster)
# =============================================================================
run_one_backend() {
    local backend="$1"
    backend_config "$backend"

    local steps="${DMOS_STEPS:-$B_STEPS}"
    local sla_p95="${DMOS_SLA_P95_MS:-$B_SLA_P95}"
    local measure_s="${DMOS_STEP_DURATION_S:-60}"
    local settle_s="${DMOS_SETTLE_S:-12}"
    local err_rate="${DMOS_SLA_ERR_RATE:-0.02}"
    local target="${backend}:${B_PORT}"
    local jobname="dmos-k6-cap-${backend}"

    local n_steps est_min
    n_steps="$(echo "$steps" | awk -F, '{print NF}')"
    est_min="$(awk "BEGIN{printf \"%.0f\", ${n_steps}*(${measure_s}+${settle_s})/60}")"

    hdr "######## Backend: $backend ########"
    echo "  Metodo gRPC:    $B_METHOD"
    echo "  Target (DNS):   $target  (ClusterIP, in-cluster)"
    echo "  Cluster:        $CLUSTER"
    echo "  Steps (rps):    $steps   (~${est_min} min di test)"
    echo "  SLA:            p95 < ${sla_p95}ms , err < $(awk "BEGIN{print ${err_rate}*100}")%"
    echo "  Dipendenze:     ${B_DEPS:-(nessuna)}"

    kubectl --kubeconfig "$TARGET_KC" -n "$NS" get deployment "$backend" >/dev/null 2>&1 \
        || fail "Deployment $backend non trovato in $NS su $CLUSTER"

    # --- scala il backend a 1 replica ---------------------------------------
    hdr "Scale $backend -> 1 replica su $CLUSTER"
    kubectl --kubeconfig "$TARGET_KC" -n "$NS" scale deployment "$backend" --replicas=1 >/dev/null
    ok "$backend scalato a 1"

    # --- potenzia le dipendenze ---------------------------------------------
    BOOSTED_DEPS=""
    if [[ -n "$B_DEPS" ]]; then
        hdr "Potenzio le dipendenze -> $BOOST_REPLICAS repliche su $CLUSTER"
        local d
        for d in $B_DEPS; do
            if kubectl --kubeconfig "$TARGET_KC" -n "$NS" get deployment "$d" >/dev/null 2>&1; then
                kubectl --kubeconfig "$TARGET_KC" -n "$NS" scale deployment "$d" --replicas="$BOOST_REPLICAS" >/dev/null
                BOOSTED_DEPS="$BOOSTED_DEPS $d"
                ok "$d -> $BOOST_REPLICAS repliche"
            else
                warn "Dipendenza $d non trovata, skip"
            fi
        done
    fi

    # --- attendi Ready: backend + dipendenze --------------------------------
    hdr "Aspetto Ready: $backend + dipendenze (timeout 3 min ciascuno)"
    local d
    for d in "$backend" $BOOSTED_DEPS; do
        if kubectl --kubeconfig "$TARGET_KC" -n "$NS" rollout status deployment "$d" --timeout=3m >/dev/null 2>&1; then
            ok "$d Ready"
        else
            fail "$d non e' tornato Ready entro 3 min"
        fi
    done

    # --- cancella eventuale Job stantio + attendi che i suoi pod spariscano --
    kubectl --kubeconfig "$TARGET_KC" -n "$NS" delete job "$jobname" \
        --ignore-not-found --wait=true >/dev/null 2>&1 || true
    local leftover
    for i in $(seq 1 30); do
        leftover="$(kubectl --kubeconfig "$TARGET_KC" -n "$NS" get pod \
            -l job-name="$jobname" --no-headers 2>/dev/null | wc -l | tr -d ' ')"
        [[ "${leftover:-0}" -eq 0 ]] && break
        sleep 1
    done

    # --- crea il Job k6 ------------------------------------------------------
    hdr "Creo il Job k6 ($jobname)"
    CURRENT_JOB="$jobname"
    kubectl --kubeconfig "$TARGET_KC" -n "$NS" apply -f - >/dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${jobname}
  namespace: ${NS}
  labels:
    app: dmos-k6-cap
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 900
  template:
    metadata:
      labels:
        app: dmos-k6-cap
    spec:
      restartPolicy: Never
      containers:
      - name: k6
        image: ${K6_IMAGE}
        command: ["k6", "run", "/scripts/k6_capacity_grpc.js"]
        env:
        - name: DMOS_GRPC_TARGET
          value: "${target}"
        - name: DMOS_GRPC_METHOD
          value: "${B_METHOD}"
        - name: DMOS_SVC_NAME
          value: "${backend}"
        - name: DMOS_STEPS
          value: "${steps}"
        - name: DMOS_STEP_DURATION_S
          value: "${measure_s}"
        - name: DMOS_SETTLE_S
          value: "${settle_s}"
        - name: DMOS_SLA_P95_MS
          value: "${sla_p95}"
        - name: DMOS_SLA_ERR_RATE
          value: "${err_rate}"
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "4"
            memory: "4Gi"
        volumeMounts:
        - name: scripts
          mountPath: /scripts
      volumes:
      - name: scripts
        configMap:
          name: ${CM_NAME}
EOF
    ok "Job applicato"

    # --- attendi che il pod del Job sia creato ------------------------------
    local podname=""
    for i in $(seq 1 30); do
        podname="$(kubectl --kubeconfig "$TARGET_KC" -n "$NS" get pod \
            -l job-name="$jobname" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null | tr -d '\r')"
        [[ -n "$podname" ]] && break
        sleep 2
    done
    [[ -n "$podname" ]] || fail "Pod del Job $jobname non creato"
    step "Pod: $podname — attendo Running (pull immagine $K6_IMAGE, max ~6 min)..."

    # --- attendi Running, intercetta problemi di pull immagine --------------
    local phase="" waiting=""
    for i in $(seq 1 120); do
        phase="$(kubectl --kubeconfig "$TARGET_KC" -n "$NS" get pod "$podname" \
            -o jsonpath='{.status.phase}' 2>/dev/null | tr -d '\r')"
        case "$phase" in
            Running|Succeeded|Failed) break ;;
        esac
        waiting="$(kubectl --kubeconfig "$TARGET_KC" -n "$NS" get pod "$podname" \
            -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null | tr -d '\r')"
        if [[ "$waiting" == "ImagePullBackOff" || "$waiting" == "ErrImagePull" ]]; then
            kubectl --kubeconfig "$TARGET_KC" -n "$NS" describe pod "$podname" 2>/dev/null | tail -15
            fail "Immagine k6 non scaricabile: $K6_IMAGE ($waiting). Imposta DMOS_K6_IMAGE con un'immagine raggiungibile."
        fi
        sleep 3
    done
    [[ -n "$phase" ]] || fail "Pod $podname senza stato dopo l'attesa"
    ok "Pod $podname: $phase"

    # --- segui i log live + attendi il completamento del Job ----------------
    # `kubectl logs -f` da' la vista live; su follow lunghi (10 min via VPN)
    # puo' cadere. Per questo, dopo il follow, attendiamo comunque che il Job
    # raggiunga uno stato TERMINALE e poi rileggiamo il log COMPLETO con
    # `kubectl logs` (senza -f): l'estrazione del JSON resta affidabile anche
    # se il follow e' caduto a meta'.
    local ts; ts="$(date +%Y%m%d_%H%M)"
    local k6log="${RESULTS_DIR}/capacity_grpc_${backend}_${ts}.k6log.txt"
    local k6live; k6live="$(mktemp)"
    hdr "k6 in esecuzione — log live dal pod (~${est_min} min)"
    echo "  Log: $k6log"
    echo "  La tabella capacity + knee-point compare alla fine."
    set +e
    kubectl --kubeconfig "$TARGET_KC" -n "$NS" logs -f "$podname" 2>&1 | tee "$k6live"
    set -e

    hdr "Attendo lo stato terminale del Job $jobname"
    local succeeded="" failed="" terminal=0
    for i in $(seq 1 360); do
        succeeded="$(kubectl --kubeconfig "$TARGET_KC" -n "$NS" get job "$jobname" \
            -o jsonpath='{.status.succeeded}' 2>/dev/null | tr -d '\r')"
        failed="$(kubectl --kubeconfig "$TARGET_KC" -n "$NS" get job "$jobname" \
            -o jsonpath='{.status.failed}' 2>/dev/null | tr -d '\r')"
        if [[ "$succeeded" == "1" || -n "$failed" ]]; then terminal=1; break; fi
        if [[ $((i % 12)) -eq 0 ]]; then step "...Job ancora in corso ($((i / 12)) min)"; fi
        sleep 5
    done
    if [[ "$succeeded" == "1" ]]; then
        ok "Job $jobname completato"
    elif [[ "$terminal" == "1" ]]; then
        warn "Job $jobname risulta fallito (failed='$failed')"
    else
        warn "Job $jobname non terminato entro l'attesa (30 min)"
    fi

    # Log COMPLETO e autorevole (no -f); fallback alla cattura live del follow.
    if ! kubectl --kubeconfig "$TARGET_KC" -n "$NS" logs "$podname" >"$k6log" 2>/dev/null \
         || [[ ! -s "$k6log" ]]; then
        cp "$k6live" "$k6log" 2>/dev/null || true
    fi
    rm -f "$k6live" 2>/dev/null || true

    # --- estrai il JSON di summary dai log ----------------------------------
    local archived="${RESULTS_DIR}/capacity_grpc_${backend}_${ts}.json"
    awk '/^===DMOS_CAPACITY_JSON_BEGIN===/{f=1;next} /^===DMOS_CAPACITY_JSON_END===/{f=0} f' \
        "$k6log" | tr -d '\r' > "$archived"
    local rec="n/a" knee="n/a"
    if [[ -s "$archived" ]] && jq empty "$archived" >/dev/null 2>&1; then
        rec="$(jq -r '.knee.recommended_capacity_req_per_sec // "n/a"' "$archived" | tr -d '\r')"
        knee="$(jq -r '.knee.rps_per_replica // "n/a"' "$archived" | tr -d '\r')"
        ok "Summary archiviato: results/capacity_grpc_${backend}_${ts}.json"
    else
        warn "JSON di summary non estratto dai log (marcatori assenti? k6 fallito?)"
        rm -f "$archived" 2>/dev/null || true
    fi

    # --- cancella il Job ----------------------------------------------------
    kubectl --kubeconfig "$TARGET_KC" -n "$NS" delete job "$jobname" \
        --ignore-not-found --wait=false >/dev/null 2>&1 || true
    CURRENT_JOB=""

    # --- ripristina le dipendenze -------------------------------------------
    if [[ "${DMOS_NO_RESTORE:-0}" != "1" ]]; then
        if [[ -n "$BOOSTED_DEPS" ]]; then
            hdr "Restore dipendenze -> 1 replica"
            for d in $BOOSTED_DEPS; do
                kubectl --kubeconfig "$TARGET_KC" -n "$NS" scale deployment "$d" --replicas=1 >/dev/null 2>&1 || true
                ok "$d -> 1 replica"
            done
        fi
        BOOSTED_DEPS=""
    else
        warn "DMOS_NO_RESTORE=1 -> dipendenze lasciate a $BOOST_REPLICAS repliche"
    fi

    # --- registra il risultato ----------------------------------------------
    RESULTS_SUMMARY+=("${backend}|${knee}|${rec}")
    if [[ "$rec" != "n/a" && "$rec" != "null" ]]; then
        ok "$backend: knee ${knee} rps/replica -> capacity_req_per_sec consigliato = ${rec}"
    else
        warn "$backend: knee non determinato (controlla la tabella k6 sopra)"
    fi
}

# =============================================================================
# MAIN
# =============================================================================
if [[ "$RUN_ALL" == "1" ]]; then
    for b in cartservice productcatalogservice recommendationservice checkoutservice; do
        run_one_backend "$b" || warn "$b: run terminata con warning"
    done
else
    run_one_backend "$BACKEND" || warn "$BACKEND: run terminata con warning"
fi

# ── ConfigMap cleanup ────────────────────────────────────────────────────────
kubectl --kubeconfig "$TARGET_KC" -n "$NS" delete configmap "$CM_NAME" \
    --ignore-not-found >/dev/null 2>&1 || true
CM_CREATED=0

# ── Riepilogo finale ─────────────────────────────────────────────────────────
hdr "Riepilogo capacity backend gRPC"
printf '  %-24s | %-16s | %-22s\n' "Backend" "knee (rps/rep)" "capacity_req_per_sec"
printf '  %s\n' "--------------------------------------------------------------------------"
if [[ ${#RESULTS_SUMMARY[@]} -gt 0 ]]; then
    for row in "${RESULTS_SUMMARY[@]}"; do
        IFS='|' read -r r_backend r_knee r_rec <<< "$row"
        printf '  %-24s | %-16s | %-22s\n' "$r_backend" "$r_knee" "$r_rec"
    done
fi
echo ""
echo "  Snippet per config/services.yaml (sezione services:):"
echo ""
if [[ ${#RESULTS_SUMMARY[@]} -gt 0 ]]; then
    for row in "${RESULTS_SUMMARY[@]}"; do
        IFS='|' read -r r_backend r_knee r_rec <<< "$row"
        if [[ "$r_rec" != "n/a" && "$r_rec" != "null" ]]; then
            echo "    ${r_backend}:"
            echo "      capacity_req_per_sec: ${r_rec}"
        else
            echo "    # ${r_backend}: knee non determinato, rilancia con DMOS_STEPS piu' fitti/alti"
        fi
    done
fi
echo ""
echo "  Dettaglio per-backend: results/capacity_grpc_<backend>_<timestamp>.json"
echo "  NB: loadgenerator e' rimasto a 0 su tutti i cluster (riaccendilo per la fase di traffico)."
echo ""
