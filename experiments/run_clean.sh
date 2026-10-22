#!/usr/bin/env bash
# run_clean.sh — UN run di Studio A "pulito", end-to-end, con overflow GSLB sano.
#
# Fa, in sequenza:
#   1. clean_reset.sh                  (stato pristino: tutto a 1 replica)
#   2. gslb_check.py                   (RICREA la CEC + CONTROLLI SERRATI: endpoint
#                                       freschi, healthy, sul path, nessun timeout).
#                                       Se fallisce -> ABORT (non spreca 18 min).
#   3. refresher CEC in background     (ri-applica la CEC a ogni cambio di pod,
#                                       così resta fresca mentre DMOS scala nel run)
#   4. run_one.sh <arm> <regime>       (prep + health-gate + DMOS + k6; nomi
#                                       di experiments/run_naming.py)
#   5. stop refresher + archive_run    (elimina k6_events.json, archivia il run)
#
# USO:
#   bash experiments/run_clean.sh l1l2 uni
#   bash experiments/run_clean.sh off con
#   GSLB_WEIGHTS=01=1,02=1,03=1,04=1,05=1,06=1  (default uniforme)
#   DMOS_IAC=$HOME/dmos-iac                     (dove stanno gslb_*.py + kubeconfig;
#                                                default: ricavata da .cache/test-env.json)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ARM="${1:-l1l2}"; REGIME="${2:-uni}"
PY="venv/Scripts/python.exe"; [ -f "$PY" ] || PY="python"
# label del run (e validazione dei nomi PRIMA di toccare il testbed)
ENVJ=(); [ -f .cache/test-env.json ] && ENVJ=(--env-json .cache/test-env.json)
LABEL="$("$PY" experiments/run_naming.py "${ENVJ[@]}" shell "$ARM" "$REGIME" | tr -d '\r' \
         | sed -n 's/^CELL_LABEL=//p')"
[ -n "$LABEL" ] || exit 2
# DMOS_IAC: env, altrimenti ricavata dal kubeconfig_path in .cache/test-env.json
# (gitignored, vedi preflight_runs.sh). Errore esplicito se manca entrambe.
DMOS_IAC="${DMOS_IAC:-}"
if [ -z "$DMOS_IAC" ] && [ -f .cache/test-env.json ] && command -v jq >/dev/null 2>&1; then
  DMOS_IAC="$(dirname "$(jq -r '.clusters[0].kubeconfig_path' .cache/test-env.json | tr -d '\r')")"
fi
[ -n "$DMOS_IAC" ] || { echo "[ERR] DMOS_IAC non impostata e .cache/test-env.json assente — esporta DMOS_IAC=<dir con gslb_*.py e kubeconfig> (es. \$HOME/dmos-iac)"; exit 2; }
GSLB_WEIGHTS="${GSLB_WEIGHTS:-01=1,02=1,03=1,04=1,05=1,06=1}"
KCBASE="$DMOS_IAC/kubeconfig-ms-cluster"
CLUSTERS=(01 02 03 04 05 06)

live_ips() {  # set di IP frontend vivi, ordinato (per rilevare i cambi)
  for c in "${CLUSTERS[@]}"; do
    kubectl --kubeconfig "$KCBASE-$c" -n online-boutique get pods -l app=frontend \
      -o 'go-template={{range .items}}{{.status.podIP}} {{end}}' 2>/dev/null
  done | tr ' ' '\n' | grep -v '^$' | sort | tr '\n' ' '
}

echo "### run_clean — $ARM $REGIME ###"

echo "== [1/5] clean_reset =="
bash experiments/clean_reset.sh || { echo "[ABORT] clean_reset fallito"; exit 1; }

echo "== [2/5] gslb_check (controlli serrati) =="
( cd "$DMOS_IAC" && python gslb_check.py --weights "$GSLB_WEIGHTS" ) \
  || { echo "[ABORT] overflow NON sano — niente run finche' non e' verde"; exit 2; }

echo "== [3/5] avvio refresher CEC (fresca durante lo scaling) =="
REFLOG="/tmp/gslb_refresh_${ARM}_${REGIME}.log"
( prev=""
  while true; do
    cur="$(live_ips)"
    if [ -n "$cur" ] && [ "$cur" != "$prev" ]; then
      ( cd "$DMOS_IAC" && python gslb_cec_writer.py --weights "$GSLB_WEIGHTS" --apply ) \
        >>"$REFLOG" 2>&1
      echo "$(date +%H:%M:%S) re-apply: $cur" >>"$REFLOG"; prev="$cur"
    fi
    sleep 15
  done ) & REF_PID=$!
trap 'kill $REF_PID 2>/dev/null' EXIT
echo "  refresher PID $REF_PID (log: $REFLOG)"

echo "== [4/5] run_one $ARM $REGIME =="
bash experiments/run_one.sh "$ARM" "$REGIME"
RC=$?

echo "== [5/5] stop refresher + archive =="
kill $REF_PID 2>/dev/null; trap - EXIT
LAST=$(ls -td experiments/results/*_${LABEL} 2>/dev/null | head -1)
[ -n "$LAST" ] && bash experiments/archive_run.sh "$LAST"

echo "### run_clean FATTO ($ARM $REGIME, run_one rc=$RC) ###"
echo "  refresher re-apply totali: $(grep -c re-apply "$REFLOG" 2>/dev/null || echo 0)"
exit $RC
