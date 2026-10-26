#!/usr/bin/env bash
# =============================================================================
# experiments/lib/run_common.sh — logica comune agli orchestratori dei run
#   (run_one.sh, run_matrix_session.sh, run_skew_sweep.sh)
# =============================================================================
# Da usare con `source` DOPO essersi spostati nella root del repo:
#
#   cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
#   source experiments/lib/run_common.sh
#   rc_init
#   run_named <arm> <regime>          # es. run_named nca con-sg
#
# I NOMI dei test (arm e regime) si traducono in configurazione in UN solo
# punto, experiments/run_naming.py: label = dmos-<arm>__<regime>, identica in
# tutti gli script e nell'analisi. Elenco:  python experiments/run_naming.py list
# run_cell resta disponibile per celle fuori schema:
#   run_cell <label> off  ""             ""          "<env k6>" ["<arg run>"]
#   run_cell <label> dmos "<arg DMOS>"   "<env DMOS>" "<env k6>" ["<arg run>"]
#
# run_cell rende TRACCIABILE ogni run (vedi docs/RUN-PROVENANCE.md):
#   1. prep: ferma DMOS (solo il suo processo) → reset a 1 replica (loadgenerator
#      a 0) → settle → health-gate
#   2. crea la cartella  experiments/results/<ts>_<label>/
#   3. snapshot del TESTBED pre-run  → env_state_pre.json
#      con GATE: se netem manca su un cluster che lo prevede, se il frontend non
#      ha l'affinity attesa o se il loadgenerator gira, il run NON parte (la
#      cartella va in results/_aborted/).
#   4. avvia DMOS con log e configurazione EFFETTIVA dentro la cartella:
#      dmos.log, dmos_config.json (config_id)
#   5. run k6 (run_flash_crowd.sh --run-dir ...) → manifest.json, summary, timeline
#   6. ferma DMOS, snapshot post-run → env_state_post.json
#   7. completa il manifest (write_manifest.py finish): controlli di provenienza
#
# Variabili: SETTLE (default 75), DMOS_K6_REMOTE, DMOS_REQUIRE_NETEM (default 1),
#            DMOS_ENV_CAPTURE (default 1; 0 = nessuno snapshot del testbed),
#            DMOS_EXPECT_FRONTEND_AFFINITY (default none; vuota = nessun controllo),
#            DMOS_ALLOW_LOADGENERATOR (default 0: la cella non parte se il
#            loadgenerator di Online Boutique ha repliche > 0).
# =============================================================================

rc_init() {
  PY="venv/Scripts/python.exe"; [ -f "$PY" ] || PY="python"
  ENV_JSON=".cache/test-env.json"
  RESULTS_DIR="experiments/results"
  command -v jq >/dev/null 2>&1 || { echo "ERRORE: jq mancante nel PATH"; exit 2; }
  [ -f "$ENV_JSON" ] || { echo "ERRORE: $ENV_JSON assente — genera l'ambiente con scripts/render-runtime-files.sh"; exit 2; }
  # k6 remoto: env DMOS_K6_REMOTE, altrimenti .k6.remote da test-env.json
  if [ -z "${DMOS_K6_REMOTE:-}" ]; then
    DMOS_K6_REMOTE="$(jq -r '.k6.remote // empty' "$ENV_JSON" | tr -d '\r')"
  fi
  [ -n "${DMOS_K6_REMOTE:-}" ] || { echo "ERRORE: DMOS_K6_REMOTE non settata e .k6.remote assente in $ENV_JSON (es.: export DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>)"; exit 2; }
  export DMOS_K6_REMOTE
  SETTLE="${SETTLE:-75}"
  REQUIRE_NETEM="${DMOS_REQUIRE_NETEM:-1}"
  ENV_CAPTURE="${DMOS_ENV_CAPTURE:-1}"
  # Data plane atteso (paper NCA 2026: frontend global, affinity none). Vuoto =
  # nessun controllo; "local" per i bracci che lo richiedono (es. HPA local).
  EXPECT_FE_AFFINITY="${DMOS_EXPECT_FRONTEND_AFFINITY-none}"
  # Il loadgenerator interno di Online Boutique deve essere spento (1 = consenti).
  ALLOW_LOADGEN="${DMOS_ALLOW_LOADGENERATOR:-0}"
  # Cluster (numero:IP) e kubeconfig letti da test-env.json (niente IP hardcoded)
  CLUSTERS=(); declare -gA KCFG
  while IFS=$'\t' read -r cname cip ckcp; do
    cn="${cname##*-}"; ckcp="${ckcp/#\~/$HOME}"
    CLUSTERS+=( "$cn:$cip" ); KCFG["$cn"]="$ckcp"
  done < <(jq -r '.clusters[] | [.name, .ip, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')
  [ ${#CLUSTERS[@]} -gt 0 ] || { echo "ERRORE: nessun cluster in $ENV_JSON"; exit 2; }
}

# ── arresto di DMOS: SOLO il processo DMOS, mai altri processi Python ─────────
# Non serve conoscere il nome del processo: DMOS si riconosce dalla sua riga
# di comando, che contiene sempre "-m src.dmos_main" (python -m src.dmos_main).
#   - Windows: si interrogano i processi python* (Win32_Process) e si fermano
#     solo quelli con "-m src.dmos_main" nella command line — compreso il
#     processo figlio creato dal launcher python.exe del venv, e un DMOS
#     lanciato a mano o rimasto da una sessione interrotta. Shell, editor o
#     altri programmi che citano la stringa non vengono toccati.
#   - Linux/macOS: il PID salvato al lancio (DMOS_PIDFILE), verificato su
#     /proc/<pid>/cmdline, più `pkill -f` sullo stesso pattern.
# Gli altri processi Python della macchina non vengono toccati. Un PID salvato
# non viene mai usato senza verificarne l'identità (potrebbe essere stato
# riassegnato a un altro processo).
DMOS_PIDFILE="${DMOS_PIDFILE:-.cache/dmos.pid}"

record_dmos_pid() {   # $1 = PID ($!) del processo DMOS appena lanciato
  mkdir -p "$(dirname "$DMOS_PIDFILE")"
  echo "$1" > "$DMOS_PIDFILE"
}

_stop_dmos_processes() {
  if command -v pwsh >/dev/null 2>&1; then
    pwsh -NoProfile -Command 'Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "-m\s+src\.dmos_main" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }' >/dev/null 2>&1
  else
    local p=""
    [ -s "$DMOS_PIDFILE" ] && p="$(tr -d '\r\n ' < "$DMOS_PIDFILE")"
    if [ -n "$p" ] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q 'python.* -m src[.]dmos_main'; then
      kill "$p" 2>/dev/null
    fi
    command -v pkill >/dev/null 2>&1 && pkill -f 'python[0-9.]* .*-m src[.]dmos_main' 2>/dev/null
  fi
  rm -f "$DMOS_PIDFILE"
  return 0
}

kill_dmos() {
  _stop_dmos_processes
  for i in $(seq 1 15); do
    [ "$(curl -s -o /dev/null -m2 -w '%{http_code}' http://localhost:9090/metrics 2>/dev/null)" = "000" ] && return 0
    sleep 1
  done
  echo "  !!! :9090 risponde ancora: c'e' un processo non-DMOS sulla porta o DMOS non si ferma"; return 1
}

# ── recovery di un cluster con ingress non responsivo ───────────────────────
recover_cluster() {
  local n="$1"; local kc="${KCFG[$n]:-}"
  [ -n "$kc" ] || { echo "    [recovery] cluster-$n: kubeconfig assente in $ENV_JSON"; return 1; }
  echo "    [recovery] cluster-$n: restart frontend + cilium-envoy"
  kubectl --kubeconfig "$kc" -n online-boutique rollout restart deploy/frontend  >/dev/null 2>&1
  kubectl --kubeconfig "$kc" -n kube-system     rollout restart ds/cilium-envoy   >/dev/null 2>&1
  kubectl --kubeconfig "$kc" -n kube-system     rollout status  ds/cilium-envoy  --timeout=120s >/dev/null 2>&1
  kubectl --kubeconfig "$kc" -n online-boutique rollout status  deploy/frontend  --timeout=120s >/dev/null 2>&1
  sleep 8
}

# ── health-gate: tutti i cluster devono dare 200; recovery+retry su quelli giù ─
health_gate() {
  local attempt
  for attempt in 1 2 3 4 5; do
    local bad=() entry n ip c
    for entry in "${CLUSTERS[@]}"; do
      n="${entry%%:*}"; ip="${entry#*:}"
      c=$(curl -s -o /dev/null -m5 -w '%{http_code}' "http://$ip:30080/" 2>/dev/null)
      [ "$c" = "200" ] || bad+=("$n")
    done
    [ ${#bad[@]} -eq 0 ] && { echo "  [health] tutti i ${#CLUSTERS[@]} cluster 200 OK"; return 0; }
    echo "  [health] cluster giù: ${bad[*]} (tentativo $attempt/5) → recovery"
    for n in "${bad[@]}"; do recover_cluster "$n"; done
    sleep 20   # quiescenza extra dopo il restart prima di ri-controllare
  done
  echo "  [health] FALLITO: cluster ancora giù dopo recovery"; return 1
}

# ── stato pristino: kill DMOS → reset → settle → health-gate ─────────────────
prep() {
  kill_dmos >/dev/null 2>&1
  pwsh scripts/reset_test.ps1 || true
  echo "  [settle] reset+quiescenza ${SETTLE}s"; sleep "$SETTLE"
  health_gate
}

# ── attende che DMOS esponga /metrics e abbia scritto la sua configurazione ──
wait_dmos_ready() {
  local cfg="$1" i
  for i in $(seq 1 35); do
    if [ "$(curl -s -o /dev/null -m2 -w '%{http_code}' http://localhost:9090/metrics 2>/dev/null)" = "200" ] \
       && [ -s "$cfg" ]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# ── nomi dei test → configurazione (experiments/run_naming.py) ───────────────
# resolve_cell ARM REGIME imposta CELL_LABEL CELL_KIND CELL_DMOS_ARGS
# CELL_DMOS_ENV CELL_K6_ENV CELL_RUN_ARGS CELL_DESC; rc 2 se il nome non vale.
resolve_cell() {
  local out
  out="$("$PY" experiments/run_naming.py --env-json "$ENV_JSON" shell "$1" "$2")" || return 2
  eval "$(printf '%s\n' "$out" | tr -d '\r')"
}

# run_named ARM REGIME: una cella tracciabile dal suo nome.
run_named() {
  resolve_cell "$1" "$2" || return 2
  run_cell "$CELL_LABEL" "$CELL_KIND" "$CELL_DMOS_ARGS" "$CELL_DMOS_ENV" "$CELL_K6_ENV" "$CELL_RUN_ARGS"
}

# ── UNA cella tracciabile ────────────────────────────────────────────────────
# run_cell LABEL KIND DMOS_ARGS DMOS_ENV K6_ENV [RUN_ARGS]
#   KIND      off | dmos
#   DMOS_ARGS argomenti di src.dmos_main (es. --no-scheduling)
#   DMOS_ENV  variabili d'ambiente SOLO per DMOS (es. DMOS_SCORE_GATE=1 ...)
#   K6_ENV    variabili per run_flash_crowd/k6 (pesi regime, picco, VU factor)
#   RUN_ARGS  argomenti extra di run_flash_crowd.sh (es. --peak 400)
run_cell() {
  local label="$1" kind="$2" dmos_args="${3:-}" dmos_env="${4:-}" k6_env="${5:-}" run_args="${6:-}"
  echo "----- $label ($kind) -----"

  # 0) gate di CONFIGURAZIONE (offline, prima di qualsiasi reset): la
  #    configurazione dell'arm deve costruirsi (modello esistente, pesi validi).
  local expected_cfg=""
  if [ "$kind" = "dmos" ]; then
    expected_cfg="$(mktemp -t dmos_expected_XXXXXX.json)"
    if ! env $dmos_env "$PY" -m src.dmos_main $dmos_args --dump-config-only "$expected_cfg" \
         > "${expected_cfg}.out" 2>&1; then
      echo "  [ABORT] $label: configurazione DMOS non valida:"
      grep -E "Error|error|not found" "${expected_cfg}.out" | tail -3 | sed 's/^/          /'
      rm -f "$expected_cfg" "${expected_cfg}.out"
      return 1
    fi
    echo "  [config] $(tail -1 "${expected_cfg}.out")"
    rm -f "${expected_cfg}.out"
  fi

  prep || { echo "  [SKIP] $label: health-gate fallito"; rm -f "$expected_cfg"; return 1; }

  local ts run_dir rc pid=""
  ts="$(date +%Y%m%d_%H%M%S)"
  run_dir="${RESULTS_DIR}/${ts}_${label}"
  mkdir -p "$run_dir"
  [ -n "$expected_cfg" ] && mv "$expected_cfg" "$run_dir/dmos_config_expected.json"

  # 1) testbed PRIMA del run (+ gate netem)
  if [ "$ENV_CAPTURE" = "1" ]; then
    local req=""; [ "$REQUIRE_NETEM" = "1" ] && req="--require-netem"
    local exp_aff=""; [ -n "$EXPECT_FE_AFFINITY" ] && exp_aff="--expect-affinity $EXPECT_FE_AFFINITY --affinity-services frontend"
    local exp_lg="";  [ "$ALLOW_LOADGEN" != "1" ] && exp_lg="--expect-no-loadgenerator"
    "$PY" experiments/capture_env_state.py --phase pre --out "$run_dir/env_state_pre.json" $req $exp_aff $exp_lg
    rc=$?
    if [ "$rc" -eq 3 ]; then
      mkdir -p "${RESULTS_DIR}/_aborted"
      mv "$run_dir" "${RESULTS_DIR}/_aborted/" 2>/dev/null
      echo "  [ABORT] $label: testbed non nello stato atteso — snapshot in ${RESULTS_DIR}/_aborted/$(basename "$run_dir")/"
      echo "          (violazioni elencate sopra: netem -> bash experiments/preflight_runs.sh --fix;"
      echo "           affinity del frontend -> kubectl apply -f deployments/global-service-frontend.yaml;"
      echo "           loadgenerator -> kubectl -n online-boutique scale deploy loadgenerator --replicas=0)"
      return 1
    elif [ "$rc" -ne 0 ]; then
      echo "  [warn] snapshot pre-run non riuscito (rc=$rc): il run prosegue senza"
    fi
  fi

  # 2) DMOS (log + configurazione effettiva nella cartella del run)
  local nodmos=""
  if [ "$kind" = "dmos" ]; then
    env $dmos_env DMOS_CONFIG_DUMP="$run_dir/dmos_config.json" \
        "$PY" -m src.dmos_main $dmos_args > "$run_dir/dmos.log" 2>&1 &
    pid=$!
    record_dmos_pid "$pid"
    if ! wait_dmos_ready "$run_dir/dmos_config.json"; then
      echo "  [warn] DMOS non pronto dopo 35s (vedi $run_dir/dmos.log)"
    fi
    if [ -s "$run_dir/dmos_config.json" ]; then
      echo "  [dmos] $("$PY" -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['summary'])" "$run_dir/dmos_config.json" 2>/dev/null)"
    fi
  else
    nodmos="--no-dmos"
  fi

  # 3) run k6 nella cartella preparata
  env $k6_env bash experiments/run_flash_crowd.sh --label "$label" --run-dir "$run_dir" $nodmos $run_args
  rc=$?

  # 4) ferma DMOS, poi testbed DOPO il run
  if [ "$kind" = "dmos" ]; then kill_dmos >/dev/null 2>&1; wait "$pid" 2>/dev/null; fi
  if [ "$ENV_CAPTURE" = "1" ]; then
    "$PY" experiments/capture_env_state.py --phase post --out "$run_dir/env_state_post.json" --quiet \
      || echo "  [warn] snapshot post-run non riuscito"
  fi

  # 5) manifest completo (idempotente: aggiunge snapshot post e log DMOS chiuso)
  "$PY" experiments/write_manifest.py finish --run-dir "$run_dir" \
    || echo "  [warn] write_manifest.py finish fallito"
  return $rc
}
