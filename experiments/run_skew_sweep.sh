#!/usr/bin/env bash
# =============================================================================
# run_skew_sweep.sh — i punti di concentrazione intermedia (0.50 / 0.65 / 0.75)
# =============================================================================
# Scorciatoia di run_matrix_session.sh con  --regimes "s50 s65 s75":
# fra bal (0.40) e con (0.85), a picco fisso 350 req/s (varia SOLO la
# concentrazione). Default arm "off l2 l1l2" (le tre curve di Fig. 2 del
# paper NCA); gli altri argomenti passano invariati:
#   bash experiments/run_skew_sweep.sh
#   bash experiments/run_skew_sweep.sh --arms "off l2 nca"
#   bash experiments/run_skew_sweep.sh --dry-run
# 9 celle × ~25 min (reset compreso). Ogni run e' tracciabile come in
# run_matrix_session.sh.
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
exec bash experiments/run_matrix_session.sh --regimes "s50 s65 s75" "$@"
