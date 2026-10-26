#!/usr/bin/env bash
# =============================================================================
# run_flash_crowd.sh — orchestratore di UN run del test Flash Crowd
# =============================================================================
#
# SCOPO
# -----
# Esegue un singolo "arm" della valutazione Flash Crowd (dmos-off / dmos-l1 /
# dmos-l1l2 / round-robin) e raccoglie, in una cartella per-run:
#   - metriche client-side da k6  (latenza/errori per-fase e per-cluster)
#   - metriche scheduler+cluster nel tempo (collect_dmos_live.py -> JSONL):
#     score Φ, current/target replicas, scaling events, decision latency,
#     CPU/mem/pod per cluster
#   - manifest riproducibile (git commit, config, timestamp)
#
# DUAL-MODE — dove gira k6
# ------------------------
#   DMOS_K6_REMOTE vuota         -> k6 gira IN LOCALE (sul laptop). Per prove.
#   DMOS_K6_REMOTE=ubuntu@IP     -> k6 gira sulla VM dedicata via SSH (run veri).
# Il resto (manifest, collector, archiviazione) gira sempre sul laptop.
#
# Il collector collect_dmos_live.py gira SEMPRE sul laptop: scrapa il /metrics
# di DMOS (localhost:9090) + Prometheus di ogni cluster.
#
# PREREQUISITI
#   - Git Bash su Windows; jq, kubectl, python (con `requests`) nel PATH
#   - modo locale: anche k6 nel PATH
#   - modo remoto: SSH key per la VM (default: .ssh.key da .cache/test-env.json,
#     override con DMOS_K6_SSH_KEY), k6 gia' installato sulla VM
#   - DMOS gia' avviato (tranne per --no-dmos): il collector lo scrapa
#   - i 6 cluster up e raggiungibili
#
# USO
#   # arm dmos-off (DMOS spento, repliche fisse):
#   bash experiments/run_flash_crowd.sh --label dmos-off --no-dmos
#
#   # arm dmos-l1l2 (DMOS completo gia' avviato):
#   bash experiments/run_flash_crowd.sh --label dmos-l1l2
#
#   # k6 sulla VM dedicata:
#   DMOS_K6_REMOTE=ubuntu@<K6_VM_IP> \
#       bash experiments/run_flash_crowd.sh --label dmos-l1l2
#
# ENV (override opzionali)
#   DMOS_K6_REMOTE DMOS_K6_SSH_KEY DMOS_PEAK_RPS DMOS_BASELINE_RPS
#   DMOS_SPIKE_REGION DMOS_PHASE_TIMES DMOS_VU_FACTOR DMOS_RAW_OUT=0
#   KUBECONFIG_DIR
# =============================================================================

set -euo pipefail

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/experiments/results"
ENV_JSON="${REPO_ROOT}/.cache/test-env.json"
K6_SCRIPT="${REPO_ROOT}/experiments/k6_flash_crowd.js"
COLLECTOR="${REPO_ROOT}/experiments/collect_dmos_live.py"

# ── Defaults / env ───────────────────────────────────────────────────────────
LABEL=""
NO_DMOS=0
PRESET_RUN_DIR=""
K6_VERSION=""
MANIFEST_PY="${REPO_ROOT}/experiments/write_manifest.py"
DMOS_CONFIG_URL="http://localhost:8081/config"
K6_REMOTE="${DMOS_K6_REMOTE:-}"
# SSH key per la VM k6: da env; se vuota viene risolta nel pre-flight
# (modo remoto) da .ssh.key di .cache/test-env.json, come preflight_runs.sh.
K6_SSH_KEY="${DMOS_K6_SSH_KEY:-}"
PEAK_RPS="${DMOS_PEAK_RPS:-250}"
BASELINE_RPS="${DMOS_BASELINE_RPS:-50}"
SPIKE_REGION="${DMOS_SPIKE_REGION:-ms-cluster-01}"
PHASE_TIMES="${DMOS_PHASE_TIMES:-180,300,600,720,900}"
VU_FACTOR="${DMOS_VU_FACTOR:-20}"
RAW_OUT="${DMOS_RAW_OUT:-1}"
DMOS_METRICS_URL="http://localhost:9090/metrics"
RDIR="dmos-fc-run"   # cartella di lavoro sulla VM remota

# ── Regime multi-modale: override pesi per-cluster ───────────────────────────
# Inoltra esplicitamente DMOS_SPIKE_W{1..6} (+ INGRESS_W, SPIKE_WEIGHT) come
# argomenti -e a k6, cosi' raggiungono anche la k6 REMOTA (SSH non porta l'env
# del laptop). Costruisce anche la stringa per il manifest (tracciabilita').
SPIKE_EARGS=""
SPIKE_W_REC=""
for i in 1 2 3 4 5 6; do
    _vS="DMOS_SPIKE_W${i}";  _valS="${!_vS:-}"
    [[ -n "$_valS" ]] && SPIKE_EARGS="${SPIKE_EARGS} -e ${_vS}=${_valS}"
    SPIKE_W_REC="${SPIKE_W_REC}${SPIKE_W_REC:+,}W${i}=${_valS:-}"
    _vI="DMOS_INGRESS_W${i}"; _valI="${!_vI:-}"
    [[ -n "$_valI" ]] && SPIKE_EARGS="${SPIKE_EARGS} -e ${_vI}=${_valI}"
done
[[ -n "${DMOS_SPIKE_WEIGHT:-}" ]] && SPIKE_EARGS="${SPIKE_EARGS} -e DMOS_SPIKE_WEIGHT=${DMOS_SPIKE_WEIGHT}"

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
run_flash_crowd.sh — un run del test Flash Crowd per DMOS

USO:
  bash experiments/run_flash_crowd.sh --label NAME [--no-dmos] [--peak N] [--run-dir DIR]

OPZIONI:
  --label NAME   etichetta del run (dmos-off | dmos-l1 | dmos-l1l2 | round-robin)
  --no-dmos      DMOS non e' in esecuzione (arm dmos-off): il collector salta /metrics
  --peak N       override del peak rps (default 250)
  --run-dir DIR  usa una cartella di run gia' creata (dall'orchestratore, che ci
                 ha fatto scrivere dmos.log / dmos_config.json / env_state_pre.json)
                 invece di crearne una nuova. Deve chiamarsi <YYYYmmdd_HHMMSS>_<label>.
  -h, --help     questo aiuto

ENV: DMOS_K6_REMOTE (vuota=locale, ubuntu@IP=VM via SSH) DMOS_K6_SSH_KEY
     DMOS_BASELINE_RPS DMOS_SPIKE_REGION DMOS_PHASE_TIMES DMOS_VU_FACTOR
     DMOS_RAW_OUT=0 KUBECONFIG_DIR
EOF
}

# ── Parse args ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label)   LABEL="$2"; shift 2 ;;
        --no-dmos) NO_DMOS=1; shift ;;
        --peak)    PEAK_RPS="$2"; shift 2 ;;
        --run-dir) PRESET_RUN_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Argomento sconosciuto: $1"; echo; usage; exit 2 ;;
    esac
done
[[ -n "$LABEL" ]] || { echo "Manca --label"; echo; usage; exit 2; }

TEST_DURATION="${PHASE_TIMES##*,}"            # ultimo valore = durata totale
COLLECTOR_DURATION=$(( TEST_DURATION + 90 ))  # un po' oltre, per il cooldown

# ── Stato globale per il cleanup ─────────────────────────────────────────────
COLLECTOR_PID=""
cleanup() {
    local ec=$?
    if [[ -n "${COLLECTOR_PID:-}" ]] && kill -0 "$COLLECTOR_PID" 2>/dev/null; then
        kill "$COLLECTOR_PID" 2>/dev/null || true
    fi
    return $ec
}
trap cleanup EXIT
trap 'echo; warn "Interrotto"; trap - EXIT; cleanup; exit 130' INT
trap 'echo; warn "SIGTERM";   trap - EXIT; cleanup; exit 143' TERM

# ── Helper SSH (modo remoto) ─────────────────────────────────────────────────
# La chiave (-i) e' aggiunta al momento della chiamata: K6_SSH_KEY puo' essere
# risolta solo nel pre-flight (da .cache/test-env.json).
SSH_OPTS=(-o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10
          -o ServerAliveInterval=30 -o BatchMode=yes)
rssh() { ssh -i "$K6_SSH_KEY" "${SSH_OPTS[@]}" "$K6_REMOTE" "$@"; }
rscp() { scp -i "$K6_SSH_KEY" "${SSH_OPTS[@]}" "$@"; }

# ── 1. Pre-flight ────────────────────────────────────────────────────────────
hdr "Pre-flight — run '$LABEL'"

for tool in jq kubectl python curl; do
    command -v "$tool" >/dev/null 2>&1 || fail "Tool mancante: $tool"
done
python -c "import requests" 2>/dev/null \
    || fail "Modulo python 'requests' assente — attiva il venv del progetto"
[[ -f "$K6_SCRIPT" ]]  || fail "Script k6 mancante: $K6_SCRIPT"
[[ -f "$COLLECTOR" ]]  || fail "Collector mancante: $COLLECTOR"
[[ -f "$MANIFEST_PY" ]] || fail "write_manifest.py mancante: $MANIFEST_PY"
[[ -f "$ENV_JSON" ]]   || fail ".cache/test-env.json mancante — lancia render-runtime-files.sh"
ok "Tool e file presenti"

if [[ -z "$K6_REMOTE" ]]; then
    K6_MODE="locale"
    command -v k6 >/dev/null 2>&1 || fail "k6 non nel PATH (serve per il modo locale)"
    K6_VERSION="$(k6 version 2>/dev/null | head -1 | tr -d '\r' || true)"
    ok "Modo k6: LOCALE (sul laptop) — ${K6_VERSION:-versione ignota}"
else
    K6_MODE="remoto ($K6_REMOTE)"
    # Risolvi la SSH key se non passata via env (pattern di preflight_runs.sh)
    if [[ -z "$K6_SSH_KEY" ]]; then
        K6_SSH_KEY="$(jq -r '.ssh.key // empty' "$ENV_JSON" | tr -d '\r')"
        K6_SSH_KEY="${K6_SSH_KEY/#\~/$HOME}"
    fi
    [[ -n "$K6_SSH_KEY" ]] || fail "SSH key non determinata — imposta DMOS_K6_SSH_KEY o .ssh.key in .cache/test-env.json"
    [[ -f "$K6_SSH_KEY" ]] || fail "SSH key non trovata: $K6_SSH_KEY"
    rssh 'true' >/dev/null 2>&1 \
        || fail "SSH alla VM '$K6_REMOTE' fallito — la VM e' accesa? IP/chiave giusti?"
    rssh 'command -v k6 >/dev/null 2>&1' \
        || fail "k6 non installato sulla VM '$K6_REMOTE' — installalo prima (vedi doc)"
    K6_VERSION="$(rssh 'k6 version' 2>/dev/null | head -1 | tr -d '\r' || true)"
    ok "Modo k6: REMOTO — VM raggiungibile, k6 presente (${K6_VERSION:-versione ignota})"
fi

# DMOS reachable? (obbligatorio per gli arm dmos-on)
if [[ "$NO_DMOS" == "1" ]]; then
    ok "Arm senza DMOS (--no-dmos): il collector non scrapera' /metrics"
else
    curl -s --max-time 5 "$DMOS_METRICS_URL" >/dev/null 2>&1 \
        && ok "DMOS raggiungibile su :9090" \
        || fail "DMOS non risponde su $DMOS_METRICS_URL — avvialo prima del run (o usa --no-dmos)"
fi

# I 6 cluster rispondono?
hdr "Verifica raggiungibilita' dei cluster"
mapfile -t INGRESS_URLS < <(jq -r '.clusters[].ingress_url' "$ENV_JSON" | tr -d '\r')
down=0
for url in "${INGRESS_URLS[@]}"; do
    # 3 tentativi: un blip transitorio della VPN non deve abortire il run.
    code=""
    for attempt in 1 2 3; do
        code=$(curl -s -o /dev/null --max-time 8 -w "%{http_code}" "$url/" 2>/dev/null || true)
        [[ "$code" == "200" ]] && break
        sleep 2
    done
    if [[ "$code" == "200" ]]; then
        ok "$url -> 200"
    else
        warn "$url -> HTTP ${code:-000} (3 tentativi falliti)"
        down=$((down + 1))
    fi
done
[[ "$down" -eq 0 ]] || fail "$down cluster non rispondono 200 — sistemali prima di un run da ~$((TEST_DURATION/60)) min"

# ── 2. Cartella del run + manifest (provenienza) ────────────────────────────
if [[ -n "$PRESET_RUN_DIR" ]]; then
    # Cartella creata dall'orchestratore (che ci ha gia' scritto dmos.log,
    # dmos_config.json, env_state_pre.json): la riuso.
    RUN_DIR="$PRESET_RUN_DIR"
    [[ -d "$RUN_DIR" ]] || fail "--run-dir non esiste: $RUN_DIR"
    [[ "$(basename "$RUN_DIR")" =~ ^[0-9]{8}_[0-9]{6}_.+$ ]] \
        || fail "--run-dir deve chiamarsi <YYYYmmdd_HHMMSS>_<label>: $RUN_DIR"
else
    TS="$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="${RESULTS_DIR}/${TS}_${LABEL}"
    mkdir -p "$RUN_DIR"
fi
hdr "Cartella del run: $(basename "$RUN_DIR")"

# Argomenti -e per k6 (valori senza spazi -> nessun quoting). Costruiti qui
# perche' finiscono anche nel manifest.
K6_EARGS="-e DMOS_RUN_LABEL=${LABEL}"
K6_EARGS="${K6_EARGS} -e DMOS_PEAK_RPS=${PEAK_RPS}"
K6_EARGS="${K6_EARGS} -e DMOS_BASELINE_RPS=${BASELINE_RPS}"
K6_EARGS="${K6_EARGS} -e DMOS_SPIKE_REGION=${SPIKE_REGION}"
K6_EARGS="${K6_EARGS} -e DMOS_PHASE_TIMES=${PHASE_TIMES}"
K6_EARGS="${K6_EARGS} -e DMOS_VU_FACTOR=${VU_FACTOR}"
K6_EARGS="${K6_EARGS}${SPIKE_EARGS}"   # pesi multi-modali (anche per k6 remota)
RAW_ARG=""
[[ "$RAW_OUT" == "1" ]] && RAW_ARG="--out json=k6_events.json"

NO_DMOS_ARG=""; [[ "$NO_DMOS" == "1" ]] && NO_DMOS_ARG="--no-dmos"
python "$MANIFEST_PY" start --run-dir "$RUN_DIR" --label "$LABEL" $NO_DMOS_ARG \
    --k6-mode "$K6_MODE" --k6-remote "$K6_REMOTE" --k6-version "$K6_VERSION" \
    --k6-args "$K6_EARGS" --raw-out "$RAW_OUT" \
    --peak-rps "$PEAK_RPS" --baseline-rps "$BASELINE_RPS" \
    --spike-region "$SPIKE_REGION" --spike-weights "$SPIKE_W_REC" \
    --phase-times "$PHASE_TIMES" --test-duration "$TEST_DURATION" --vu-factor "$VU_FACTOR" \
    || fail "write_manifest.py start fallito"
ok "manifest.json scritto (config k6, git, host, config_snapshot/)"

# Configurazione EFFETTIVA di DMOS: se l'orchestratore l'ha gia' fatta
# scrivere (--config-dump) la teniamo; altrimenti la chiediamo al processo
# DMOS in esecuzione (GET :8081/config), cosi' anche un DMOS avviato a mano
# lascia traccia dell'algoritmo con cui ha girato.
DMOS_CONFIG_SOURCE="none"
if [[ "$NO_DMOS" != "1" ]]; then
    if [[ -s "${RUN_DIR}/dmos_config.json" ]]; then
        DMOS_CONFIG_SOURCE="dump"
    elif curl -sf --max-time 5 "$DMOS_CONFIG_URL" -o "${RUN_DIR}/dmos_config.json"; then
        DMOS_CONFIG_SOURCE="http"
    else
        rm -f "${RUN_DIR}/dmos_config.json"
        warn "Configurazione effettiva di DMOS non disponibile (DMOS senza provenance?)"
    fi
    if [[ "$DMOS_CONFIG_SOURCE" != "none" ]]; then
        ok "Config DMOS ($DMOS_CONFIG_SOURCE): $(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('summary','?'))" "${RUN_DIR}/dmos_config.json" 2>/dev/null)"
    fi
fi

# ── 3. Lancio il collector (background, sul laptop) ─────────────────────────
hdr "Avvio collector metriche DMOS+cluster (collect_dmos_live.py)"
COLLECTOR_OUT="${RUN_DIR}/dmos_live.jsonl"
NO_DMOS_FLAG=""
[[ "$NO_DMOS" == "1" ]] && NO_DMOS_FLAG="--no-dmos"
python "$COLLECTOR" "$COLLECTOR_DURATION" \
    --scenario "$LABEL" --interval 5 --output "$COLLECTOR_OUT" $NO_DMOS_FLAG \
    > "${RUN_DIR}/collector_stdout.txt" 2>&1 &
COLLECTOR_PID=$!
sleep 2
kill -0 "$COLLECTOR_PID" 2>/dev/null \
    || fail "Il collector e' morto subito — vedi ${RUN_DIR}/collector_stdout.txt"
ok "Collector attivo (PID $COLLECTOR_PID), campiona ogni 5s per ${COLLECTOR_DURATION}s"

# ── 4. Eseguo k6 ─────────────────────────────────────────────────────────────
EST_MIN=$(( TEST_DURATION / 60 ))
K6_EXIT=0

if [[ -z "$K6_REMOTE" ]]; then
    # ---- modo LOCALE -------------------------------------------------------
    hdr "k6 in LOCALE — durata ~${EST_MIN} min"
    echo "  Output live qui sotto; il riepilogo per-fase compare alla fine."
    cd "$REPO_ROOT"
    rm -f flash_crowd_summary.json
    LOCAL_RAW=""
    [[ "$RAW_OUT" == "1" ]] && LOCAL_RAW="--out json=${RUN_DIR}/k6_events.json"
    set +e
    k6 run $K6_EARGS $LOCAL_RAW experiments/k6_flash_crowd.js 2>&1 | tee "${RUN_DIR}/k6_stdout.txt"
    K6_EXIT=${PIPESTATUS[0]}
    set -e
    [[ -f flash_crowd_summary.json ]] && mv flash_crowd_summary.json "${RUN_DIR}/"
else
    # ---- modo REMOTO (k6 sulla VM via SSH) --------------------------------
    hdr "k6 sulla VM ${K6_REMOTE} — durata ~${EST_MIN} min"
    step "Preparo la cartella di lavoro e copio script + test-env"
    rssh "rm -rf ~/${RDIR} && mkdir -p ~/${RDIR}" || fail "Impossibile preparare ~/${RDIR} sulla VM"
    rscp "$K6_SCRIPT" "${K6_REMOTE}:${RDIR}/k6_flash_crowd.js" >/dev/null || fail "scp dello script k6 fallito"
    rscp "$ENV_JSON"  "${K6_REMOTE}:${RDIR}/test-env.json"     >/dev/null || fail "scp di test-env.json fallito"

    # k6 lanciato DETACHED (setsid): sopravvive a una caduta della SSH.
    REMOTE_CMD="cd ~/${RDIR} && setsid bash -c 'k6 run ${K6_EARGS} -e DMOS_TEST_ENV=test-env.json ${RAW_ARG} k6_flash_crowd.js >k6_stdout.txt 2>&1' </dev/null >/dev/null 2>&1 & echo launched"
    step "Lancio k6 in modo detached sulla VM"
    rssh "$REMOTE_CMD" >/dev/null || fail "Lancio di k6 sulla VM fallito"
    sleep 5

    # Poll: ogni 10s controllo se il processo k6 e' ancora vivo sulla VM.
    hdr "Attendo la fine di k6 sulla VM"
    terminated=0
    max_iter=$(( (TEST_DURATION + 360) / 10 ))
    for ((i = 0; i < max_iter; i++)); do
        st=$(rssh "pgrep -x k6 >/dev/null 2>&1 && echo RUN || echo DONE" 2>/dev/null || echo SSHERR)
        if [[ "$st" == "DONE" ]]; then terminated=1; break; fi
        if [[ $((i % 6)) -eq 0 ]]; then step "...k6 in corso sulla VM ($((i / 6)) min)"; fi
        sleep 10
    done
    [[ "$terminated" == "1" ]] || warn "k6 sulla VM non risulta terminato entro l'attesa"

    step "Recupero i risultati dalla VM"
    rscp "${K6_REMOTE}:${RDIR}/k6_stdout.txt"            "${RUN_DIR}/" >/dev/null 2>&1 || warn "k6_stdout.txt non recuperato"
    rscp "${K6_REMOTE}:${RDIR}/flash_crowd_summary.json" "${RUN_DIR}/" >/dev/null 2>&1 || warn "flash_crowd_summary.json non recuperato"
    if [[ "$RAW_OUT" == "1" ]]; then
        rscp "${K6_REMOTE}:${RDIR}/k6_events.json" "${RUN_DIR}/" >/dev/null 2>&1 || warn "k6_events.json non recuperato"
    fi
fi
ok "k6 terminato (exit ${K6_EXIT})"

# ── 6. Attendo la fine del collector ─────────────────────────────────────────
hdr "Attendo la fine del collector"
if [[ -n "${COLLECTOR_PID:-}" ]]; then
    wait "$COLLECTOR_PID" 2>/dev/null || true
    COLLECTOR_PID=""
fi
ok "Collector terminato — timeline in dmos_live.jsonl"

# ── 7. Verdetto di validita' + riepilogo ─────────────────────────────────────
hdr "Riepilogo run '${LABEL}'"
SUMMARY="${RUN_DIR}/flash_crowd_summary.json"
if [[ -f "$SUMMARY" ]] && jq empty "$SUMMARY" >/dev/null 2>&1; then
    run_valid=$(jq -r '.validity.run_valid // "?"' "$SUMMARY" | tr -d '\r')
    dropped=$(jq -r '.validity.dropped_iterations // 0' "$SUMMARY" | tr -d '\r')
    g_p95=$(jq -r '.results.global.p95_ms // "n/a"' "$SUMMARY" | tr -d '\r')
    g_fail=$(jq -r '.results.global.http_failure_rate // "n/a"' "$SUMMARY" | tr -d '\r')
    echo "  p95 globale:         ${g_p95} ms"
    echo "  HTTP failure rate:   ${g_fail}"
    echo "  Iterazioni droppate: ${dropped}"
    if [[ "$run_valid" == "true" ]]; then
        ok "RUN VALIDO — k6 ha sostenuto il rate offerto"
    else
        warn "RUN NON VALIDO — k6 VU-starved (dropped=${dropped}). Rilancia con DMOS_VU_FACTOR piu' alto."
    fi
else
    warn "flash_crowd_summary.json assente o non valido — controlla k6_stdout.txt"
fi

# ── 8. Manifest: provenienza completa + controlli ───────────────────────────
# (l'orchestratore lo richiama dopo lo snapshot post-run: e' idempotente)
hdr "Provenienza del run"
python "$MANIFEST_PY" finish --run-dir "$RUN_DIR" --k6-exit "$K6_EXIT" \
    --dmos-config-source "$DMOS_CONFIG_SOURCE" || warn "write_manifest.py finish fallito"

echo ""
echo "  Artefatti del run in: ${RUN_DIR}/"
echo "    manifest.json            provenienza: config k6, git, host, controlli"
echo "    dmos_config.json         configurazione EFFETTIVA di DMOS (config_id)"
echo "    config_snapshot/         copia delle config/*.yaml usate"
echo "    flash_crowd_summary.json metriche per-fase / per-cluster"
echo "    dmos_live.jsonl          timeline scheduler+cluster (ogni 5s, con config_id)"
echo "    k6_stdout.txt            output completo di k6"
[[ "$RAW_OUT" == "1" ]] && echo "    k6_events.json           eventi k6 grezzi (per analisi fine)"
echo "RUN_DIR=${RUN_DIR}"
echo ""
echo "  Prossimo arm: cambia stato/config di DMOS e rilancia con --label diverso."
echo ""
