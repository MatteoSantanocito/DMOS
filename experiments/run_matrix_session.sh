#!/usr/bin/env bash
# =============================================================================
# run_matrix_session.sh — UNA ripetizione PULITA di una matrice arm × regime.
# =============================================================================
#   bash experiments/run_matrix_session.sh [--arms "A B ..."] [--regimes "R ..."] [--dry-run]
#
# Default: il core del paper NCA 2026 (fase F1) — arm "off l2 l1l2" ×
# regimi "uni bal s50 s65 s75 con" = 18 celle (~25 min l'una, reset compreso).
# I nomi di arm e regimi sono quelli di experiments/run_naming.py
# (elenco:  python experiments/run_naming.py list). Esempi:
#   # F2 / E2 — gate sì o no e quanto forte, a BAL, 0.50, CON
#   bash experiments/run_matrix_session.sh --arms "gate_g0 gate_g1 nca gate_g3 additive_dem50" \
#        --regimes "bal s50 con"
#   # E3 — profili di pesi a BAL e CON
#   bash experiments/run_matrix_session.sh --arms "nca latency_first capacity_first carbon_first" \
#        --regimes "bal con"
#   # F3 / E4 — cluster caldo PL e SG a CON
#   bash experiments/run_matrix_session.sh --arms "off l2 nca" --regimes "con-pl con-sg"
#   # E4 — sweep del volume a skew fisso
#   bash experiments/run_matrix_session.sh --arms "off l2 nca" --regimes "con-p200 con con-p500"
#   # solo il piano, senza toccare il testbed
#   bash experiments/run_matrix_session.sh --arms "off l2 nca" --regimes "bal con" --dry-run
#
# Filosofia: UNA rep per sessione. Rilancia lo stesso comando in sessioni
# separate per accumulare ripetizioni: i run con la stessa label in timestamp
# diversi sono ripetizioni per l'analisi (paper_generate_all), e il config_id
# registrato garantisce che lo siano davvero (run_provenance.py audit).
#
#   1) ORDINE EQUO: gli arm girano consecutivi PER REGIME → condividono lo
#      stato del cluster → confronto leale.
#   2) TUTTI i nomi sono validati PRIMA di partire (un nome sbagliato non fa
#      perdere mezza sessione).
#   3) STATO PRISTINO per run: arresto di DMOS → reset (loadgenerator a 0) →
#      settle → health-gate con auto-recovery (frontend + cilium-envoy).
#   4) TRACCIABILITA' (experiments/lib/run_common.sh): configurazione effettiva
#      di DMOS, log, stato del testbed pre/post, manifest con i controlli.
#      Gate: la cella non parte se netem manca, se il frontend non ha
#      l'affinity attesa o se il loadgenerator gira.
#
# Variabili opzionali: SETTLE (sec quiescenza post-reset, default 75),
#   DMOS_K6_REMOTE (default: .k6.remote da .cache/test-env.json),
#   DMOS_REQUIRE_NETEM (default 1), DMOS_ENV_CAPTURE (default 1),
#   DMOS_EXPECT_FRONTEND_AFFINITY (default none), DMOS_ALLOW_LOADGENERATOR (default 0).
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1   # repo root

ARMS="off l2 l1l2"
REGIMES="uni bal s50 s65 s75 con"
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --arms)    ARMS="$2"; shift 2 ;;
    --regimes) REGIMES="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "argomento sconosciuto: $1 (vedi --help)"; exit 2 ;;
  esac
done

PY="venv/Scripts/python.exe"; [ -f "$PY" ] || PY="python"
ENVJ=(); [ -f .cache/test-env.json ] && ENVJ=(--env-json .cache/test-env.json)

# Validazione di tutti i nomi prima di toccare il testbed
if [ "$DRY" = "1" ]; then
  "$PY" experiments/run_naming.py "${ENVJ[@]}" plan --arms "$ARMS" --regimes "$REGIMES"; exit $?
fi
"$PY" experiments/run_naming.py "${ENVJ[@]}" plan --arms "$ARMS" --regimes "$REGIMES" --labels-only \
  > /dev/null || exit 2

source experiments/lib/run_common.sh
rc_init

N=$(( $(wc -w <<< "$ARMS") * $(wc -w <<< "$REGIMES") ))
echo "### run_matrix_session — UNA rep, $N celle ###"
echo "  arm: $ARMS"
echo "  regimi: $REGIMES"
echo "  K6=$DMOS_K6_REMOTE  SETTLE=${SETTLE}s  netem-gate=${REQUIRE_NETEM}"
grep -E '^[[:space:]]*provider:' config/carbon.yaml || echo "  [warn] provider carbon?"
kill_dmos >/dev/null 2>&1

FAILED=()
for regime in $REGIMES; do
  for arm in $ARMS; do
    run_named "$arm" "$regime" || FAILED+=("${arm}__${regime}")
  done
done

kill_dmos >/dev/null 2>&1
echo "############ SESSIONE FINITA ############"
echo "Celle fallite: ${FAILED[*]:-NESSUNA}"
echo "Verifica la provenienza: $PY experiments/run_provenance.py audit"
echo "Rilancia lo stesso comando in una nuova sessione per la rep successiva."
