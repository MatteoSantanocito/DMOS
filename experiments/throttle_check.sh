#!/usr/bin/env bash
# =============================================================================
# throttle_check.sh — verifica CFS throttling a max_replicas
# =============================================================================
#
# SCOPO
# -----
# Verificare che, scalando il frontend a `max_replicas` su un cluster, le
# repliche NON vadano in CFS throttling. Se il throttling e' ~0, il modello
# DMOS "capacita cluster = capacity_req_per_sec x max_replicas" e' LINEARE
# ed esatto (vedi dmos-iac/HETEROGENEOUS-CLUSTER-SETUP.md sezione 4).
#
# NON e' un capacity test (quello misura la capacita' di 1 replica isolata,
# vedi run_capacity.sh). Questo e' un check leggero (~4 min) post-reprovisioning
# per validare che il `max_replicas_per_cluster` scelto sia sotto la soglia
# fisica di throttling del worker.
#
# COSA FA
# -------
#   1. Pre-flight: tool, kubeconfig, .cache
#   2. Legge max_replicas del cluster target da config/services.yaml
#   3. Scala il frontend a max_replicas sul cluster target, aspetta Ready
#   4. Snapshot Prometheus PRE: contatori CFS throttling
#   5. Genera carico costante all'80% della capacita' (k6, ~3 min)
#   6. Snapshot Prometheus POST: contatori CFS + CPU usage
#   7. Calcola il throttle ratio e da' il verdetto
#   8. Restore: scala il frontend a 1 replica
#
# USO
# ---
#   bash experiments/throttle_check.sh                          # default ms-cluster-01
#   bash experiments/throttle_check.sh --cluster ms-cluster-03  # tier LOW
#   bash experiments/throttle_check.sh --cluster ms-cluster-01 --load-pct 90
#
# Per validare tutti i 3 tier dopo il re-provisioning:
#   bash experiments/throttle_check.sh --cluster ms-cluster-01   # HIGH
#   bash experiments/throttle_check.sh --cluster ms-cluster-02   # MEDIUM
#   bash experiments/throttle_check.sh --cluster ms-cluster-03   # LOW
#
# VARIABILI ENV (opzionali)
# -------------------------
#   DMOS_CLUSTER          : default ms-cluster-01
#   DMOS_LOAD_PCT         : % della capacita' da generare (default 80)
#   DMOS_LOAD_DURATION_S  : durata del carico in s (default 180)
#   DMOS_CAP_PER_REPLICA  : capacita' per replica (default 60, da capacity test)
#   DMOS_THROTTLE_WARN    : soglia ratio sopra cui warna (default 0.05 = 5%)
#
# EXIT CODE
#   0 = nessun throttling significativo (modello lineare confermato)
#   1 = throttling sopra soglia (max_replicas troppo alto per quel worker)
#   2 = errore di prerequisiti
# =============================================================================

set -euo pipefail

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults / env ───────────────────────────────────────────────────────────
CLUSTER="${DMOS_CLUSTER:-ms-cluster-01}"
LOAD_PCT="${DMOS_LOAD_PCT:-80}"
LOAD_DURATION_S="${DMOS_LOAD_DURATION_S:-180}"
CAP_PER_REPLICA="${DMOS_CAP_PER_REPLICA:-60}"
THROTTLE_WARN="${DMOS_THROTTLE_WARN:-0.05}"
NS="online-boutique"
SERVICE="frontend"
# KUBECONFIG_DIR: da env; altrimenti derivata da .cache/test-env.json
# (dirname dei kubeconfig_path renderizzati); fallback $HOME/dmos-iac.
if [[ -z "${KUBECONFIG_DIR:-}" && -f "${REPO_ROOT}/.cache/test-env.json" ]]; then
    KUBECONFIG_DIR="$(sed -n 's|.*"kubeconfig_path"[^"]*"\(.*\)/kubeconfig-[^"]*".*|\1|p' \
        "${REPO_ROOT}/.cache/test-env.json" | head -n1 | tr -d '\r')"
fi
KUBECONFIG_DIR="${KUBECONFIG_DIR:-$HOME/dmos-iac}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cluster)   CLUSTER="$2"; shift 2 ;;
        --load-pct)  LOAD_PCT="$2"; shift 2 ;;
        --duration)  LOAD_DURATION_S="$2"; shift 2 ;;
        -h|--help)   sed -n '3,45p' "$0"; exit 0 ;;
        *) echo "Argomento sconosciuto: $1"; exit 2 ;;
    esac
done

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
fail() { printf '  %s✗%s %s\n' "${RED}" "${RESET}" "$*"; exit 2; }

# ── Auto-discover tool (winget/scoop/choco/Program Files) ────────────────────
ensure_in_path() {
    local tool=$1; shift
    command -v "$tool" >/dev/null 2>&1 && return 0
    for candidate in "$@"; do
        if [[ -x "$candidate" ]]; then
            export PATH="$(dirname "$candidate"):$PATH"
            return 0
        fi
    done
    return 1
}
ensure_in_path jq \
    "$HOME/bin/jq.exe" \
    "/c/Users/$(whoami)/AppData/Local/Microsoft/WinGet/Packages/jqlang.jq_Microsoft.Winget.Source_8wekyb3d8bbwe/jq.exe" \
    "/c/ProgramData/chocolatey/bin/jq.exe"
ensure_in_path k6 \
    "/c/Program Files/k6/k6.exe" \
    "$HOME/scoop/shims/k6.exe"

# ── Helpers ──────────────────────────────────────────────────────────────────
kc_of() { echo "${KUBECONFIG_DIR}/kubeconfig-$1"; }

# Esegue una query PromQL e ritorna il valore scalare del primo result.
prom_query() {
    local prom_url=$1 q=$2
    curl -s --max-time 8 "${prom_url}/api/v1/query?query=$(printf '%s' "$q" | jq -sRr @uri)" \
        | jq -r '.data.result[0].value[1] // "n/a"' 2>/dev/null | tr -d '\r' || echo "n/a"
}

# ── 1. Pre-flight ────────────────────────────────────────────────────────────
hdr "Pre-flight — throttle check su $CLUSTER"
for tool in k6 kubectl jq curl python; do
    command -v "$tool" >/dev/null 2>&1 || fail "Tool mancante: $tool"
done
ok "Tool presenti"

KC_TARGET=$(kc_of "$CLUSTER")
[[ -f "$KC_TARGET" ]] || fail "Kubeconfig non trovato: $KC_TARGET"
ok "Cluster target: $CLUSTER"

ENV_JSON="${REPO_ROOT}/.cache/test-env.json"
[[ -f "$ENV_JSON" ]] || fail ".cache/test-env.json mancante — lancia render-runtime-files.sh"

# IP + Prometheus URL del cluster target
TARGET_IP=$(jq -r --arg n "$CLUSTER" '.clusters[] | select(.name==$n) | .ip' "$ENV_JSON" | tr -d '\r')
PROM_URL=$(jq -r --arg n "$CLUSTER" '.clusters[] | select(.name==$n) | .prometheus_url' "$ENV_JSON" | tr -d '\r')
[[ -n "$TARGET_IP" ]] || fail "IP del cluster $CLUSTER non trovato in test-env.json"
ok "IP=$TARGET_IP  Prometheus=$PROM_URL"

# ── 2. Leggi max_replicas del cluster da services.yaml ───────────────────────
hdr "Lettura max_replicas per $CLUSTER"
# NOTA: python e' un eseguibile Windows nativo, non capisce i path stile
# Git Bash (/c/Users/...). Quindi facciamo `cd` nella repo root (cd capisce
# /c/...) e passiamo a python un path RELATIVO: python eredita il CWD in
# formato Windows e lo risolve correttamente.
# `|| true` + check esplicito: senza, un errore di python con set -e
# ucciderebbe lo script in silenzio (prima del messaggio di fail).
MAX_REPLICAS=$(cd "$REPO_ROOT" && python -c "
import yaml
d = yaml.safe_load(open('config/services.yaml'))
fe = d['services']['frontend']
mrpc = fe.get('max_replicas_per_cluster', {})
glob = fe.get('autoscaling', {}).get('max_replicas', 4)
print(mrpc.get('${CLUSTER}', glob))
" 2>&1 | tr -d '\r') || true
if ! [[ "$MAX_REPLICAS" =~ ^[0-9]+$ ]]; then
    fail "Non riesco a leggere max_replicas da services.yaml. Output: $MAX_REPLICAS"
fi

# Carico target = LOAD_PCT% della capacita' = LOAD_PCT/100 x CAP_PER_REPLICA x max_replicas
CAPACITY=$((CAP_PER_REPLICA * MAX_REPLICAS))
TARGET_RATE=$(python -c "print(int($CAPACITY * $LOAD_PCT / 100))")
ok "max_replicas=$MAX_REPLICAS  capacita=${CAPACITY} rps  carico target (${LOAD_PCT}%)=${TARGET_RATE} rps"

# ── 3. Scale frontend a max_replicas ─────────────────────────────────────────
hdr "Scale $SERVICE → $MAX_REPLICAS repliche su $CLUSTER"
kubectl --kubeconfig "$KC_TARGET" -n "$NS" scale deployment "$SERVICE" \
    --replicas="$MAX_REPLICAS" >/dev/null 2>&1 || fail "kubectl scale fallito"
step "Attendo rollout (max 3 min)..."
if kubectl --kubeconfig "$KC_TARGET" -n "$NS" rollout status deployment "$SERVICE" --timeout=3m; then
    ok "$SERVICE Ready con $MAX_REPLICAS repliche"
else
    warn "Rollout non completato entro 3 min — proseguo comunque (potrebbero esserci pod Pending)"
fi

# Cleanup trap: ripristina sempre 1 replica anche se lo script crasha/Ctrl+C
restore_replicas() {
    printf '\n'
    step "Restore: scale $SERVICE → 1 replica su $CLUSTER"
    kubectl --kubeconfig "$KC_TARGET" -n "$NS" scale deployment "$SERVICE" \
        --replicas=1 >/dev/null 2>&1 || true
}
trap restore_replicas EXIT

# ── 4. Snapshot Prometheus PRE ───────────────────────────────────────────────
hdr "Snapshot Prometheus PRE-carico"
# Contatori CFS cumulativi (li useremo come baseline)
PRE_THROTTLED=$(prom_query "$PROM_URL" "sum(container_cpu_cfs_throttled_periods_total{namespace=\"$NS\",pod=~\"$SERVICE-.*\"})")
PRE_PERIODS=$(prom_query "$PROM_URL" "sum(container_cpu_cfs_periods_total{namespace=\"$NS\",pod=~\"$SERVICE-.*\"})")
echo "  CFS throttled periods (cumul): $PRE_THROTTLED"
echo "  CFS total periods (cumul):     $PRE_PERIODS"

# ── 5. Genera carico ─────────────────────────────────────────────────────────
hdr "Genero carico ${TARGET_RATE} rps per ${LOAD_DURATION_S}s"
K6_BIN="$(command -v k6)"
cd "$REPO_ROOT"
"$K6_BIN" run \
    -e DMOS_CLUSTER="$CLUSTER" \
    -e DMOS_RATE="$TARGET_RATE" \
    -e DMOS_DURATION="${LOAD_DURATION_S}s" \
    --quiet \
    experiments/k6_throttle_load.js 2>&1 | tail -20 || warn "k6 ha ritornato un errore (continuo con la misura)"

# ── 6. Snapshot Prometheus POST + calcolo throttle ratio ─────────────────────
hdr "Snapshot Prometheus POST-carico"

# Throttle ratio sulla finestra del carico: frazione di periodi CFS che sono
# stati throttled negli ultimi 2 min (= steady-state del carico).
WINDOW="2m"
THROTTLE_RATIO=$(prom_query "$PROM_URL" \
    "sum(rate(container_cpu_cfs_throttled_periods_total{namespace=\"$NS\",pod=~\"$SERVICE-.*\"}[$WINDOW])) / sum(rate(container_cpu_cfs_periods_total{namespace=\"$NS\",pod=~\"$SERVICE-.*\"}[$WINDOW]))")

# CPU usata da tutti i pod applicativi+monitoring sul worker (proxy del carico worker)
WORKER_CPU=$(prom_query "$PROM_URL" \
    "sum(rate(container_cpu_usage_seconds_total{namespace=~\"$NS|observability|kube-system\"}[$WINDOW]))")

# CPU usata dai soli pod frontend
FRONTEND_CPU=$(prom_query "$PROM_URL" \
    "sum(rate(container_cpu_usage_seconds_total{namespace=\"$NS\",pod=~\"$SERVICE-.*\"}[$WINDOW]))")

echo "  Throttle ratio (frontend, ${WINDOW}): $THROTTLE_RATIO"
echo "  CPU usata da frontend:                $FRONTEND_CPU cores"
echo "  CPU usata da tutti i pod sul worker:  $WORKER_CPU cores"

# ── 7. Verdetto ──────────────────────────────────────────────────────────────
hdr "Verdetto — $CLUSTER (max_replicas=$MAX_REPLICAS)"

EXIT_CODE=0
if [[ "$THROTTLE_RATIO" == "n/a" || -z "$THROTTLE_RATIO" ]]; then
    warn "Throttle ratio non disponibile (Prometheus non ha la metrica CFS?)."
    warn "Verifica che cAdvisor esponga container_cpu_cfs_throttled_periods_total."
    EXIT_CODE=2
else
    # Confronto numerico via python (bash non fa float)
    VERDICT=$(python -c "
ratio = float('$THROTTLE_RATIO')
warn  = float('$THROTTLE_WARN')
if ratio < warn:
    print('OK')
elif ratio < warn * 3:
    print('WARN')
else:
    print('FAIL')
")
    PCT=$(python -c "print(f'{float(\"$THROTTLE_RATIO\")*100:.2f}')")
    case "$VERDICT" in
        OK)
            ok "Throttle ratio ${PCT}% < soglia $(python -c "print(float('$THROTTLE_WARN')*100)")%"
            ok "MODELLO LINEARE CONFERMATO: capacita = ${CAP_PER_REPLICA} x ${MAX_REPLICAS} = ${CAPACITY} rps"
            echo ""
            echo "  Le $MAX_REPLICAS repliche frontend non sono CPU-throttled a ${LOAD_PCT}% di carico."
            echo "  max_replicas_per_cluster=$MAX_REPLICAS e' sotto la soglia fisica del worker. OK."
            ;;
        WARN)
            warn "Throttle ratio ${PCT}% — throttling lieve ma non critico"
            echo ""
            echo "  Il worker e' vicino alla saturazione a ${LOAD_PCT}% di carico."
            echo "  Considera di ridurre max_replicas_per_cluster[$CLUSTER] di 1, oppure"
            echo "  accetta una lieve sublinearita' (catturata comunque da Phi_cap a runtime)."
            EXIT_CODE=0
            ;;
        FAIL)
            fail_msg="Throttle ratio ${PCT}% — throttling SIGNIFICATIVO"
            printf '  %s✗%s %s\n' "${RED}" "${RESET}" "$fail_msg"
            echo ""
            echo "  Le repliche frontend sono CPU-throttled: il modello lineare"
            echo "  capacita = cap x max_replicas SOVRASTIMA la capacita reale."
            echo "  AZIONE: riduci max_replicas_per_cluster[$CLUSTER] in config/services.yaml"
            echo "  (prova max_replicas-1 e ri-lancia questo check)."
            EXIT_CODE=1
            ;;
    esac
fi

echo ""
echo "  ${DIM}Nota: il restore (scale → 1 replica) parte automaticamente ora.${RESET}"

exit $EXIT_CODE
