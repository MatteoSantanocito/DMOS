#!/usr/bin/env bash
# run_one.sh — UNA singola cella (arm × regime), con prep pristino + health-gate.
# Per un pilot (F0) o per rifare un run contaminato senza lanciare una sessione.
#
#   bash experiments/run_one.sh ARM REGIME            # esegue la cella
#   bash experiments/run_one.sh ARM REGIME --dry-run  # mostra cosa eseguirebbe
#   bash experiments/run_one.sh --list                # arm e regimi validi
#
# I nomi sono definiti in experiments/run_naming.py e valgono uguali in tutti
# gli script; la cartella del run si chiama <ts>_dmos-<arm>__<regime>.
#   ARM     off | l2 | l1l2 | <modello di score> (nca, gate_g0, gate_g1, gate_g3,
#           additive_dem50, locality, latency_first, capacity_first,
#           carbon_first, ...) | profile-<profilo> | legacy-default |
#           legacy-gate1 | legacy-nofloor
#           l1l2 = la configurazione della campagna NCA 2026 (gate g=2, pesi
#           0.25/0.20/0.25/0.30, floor off), letta dai log DMOS di giugno
#   REGIME  <domanda>[-<cluster caldo>][-p<picco>]
#           domanda uni|bal|s50|s65|s75|con; cluster caldo de (default) fr pl
#           us sg no; picco in req/s (default 350)
# Es.:  bash experiments/run_one.sh nca con
#       bash experiments/run_one.sh gate_g0 s50
#       bash experiments/run_one.sh l1l2 con-sg
#       bash experiments/run_one.sh nca bal-p500
#
# Il run e' TRACCIABILE: la cartella contiene dmos_config.json (configurazione
# effettiva di DMOS, config_id), dmos.log, env_state_pre/post.json (netem,
# affinity, Cilium, HPA, loadgenerator) e manifest.json con i controlli.
# Ispeziona con:  python experiments/run_provenance.py show <cartella_run>
# NB: per un reset profondo del fleet prima di una sessione:
#     bash experiments/clean_reset.sh [--deep]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PY="venv/Scripts/python.exe"; [ -f "$PY" ] || PY="python"
ENVJ=(); [ -f .cache/test-env.json ] && ENVJ=(--env-json .cache/test-env.json)

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
  echo; "$PY" experiments/run_naming.py "${ENVJ[@]}" list
  exit 0
fi
if [ $# -lt 2 ]; then
  echo "uso: bash experiments/run_one.sh ARM REGIME [--dry-run]   (elenco: --list)"; exit 2
fi
ARM="$1"; REGIME="$2"

if [ "${3:-}" = "--dry-run" ]; then
  "$PY" experiments/run_naming.py "${ENVJ[@]}" show "$ARM" "$REGIME"; exit $?
fi

source experiments/lib/run_common.sh
rc_init
resolve_cell "$ARM" "$REGIME" || exit 2

echo "### run_one — $CELL_LABEL ###"
echo "    $CELL_DESC"
run_cell "$CELL_LABEL" "$CELL_KIND" "$CELL_DMOS_ARGS" "$CELL_DMOS_ENV" "$CELL_K6_ENV" "$CELL_RUN_ARGS"
rc=$?
echo "### FATTO: $CELL_LABEL (rc=$rc) ###"
exit $rc
