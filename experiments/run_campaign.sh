#!/usr/bin/env bash
# run_campaign.sh — an unattended campaign: the cells of a plan file, one at a time.
#
#   bash experiments/run_campaign.sh experiments/plans/<name>.plan
#
# Plan file, one line per cell (# = comment):
#   <block> <arm> <regime> [VAR=value ...]      e.g.  A reach-pool-ps con-p250 DMOS_RAW_OUT=1
#   @prestop                                    apply preStop/grace (apply_pod_lifecycle.sh),
#                                               then require it in every following run
#   @apply <file.patch>                         git apply a code patch between cells, then
#                                               run the unit tests; stop if either fails
#                                               (the runs record the uncommitted diff)
# For every cell:
#   - waits until it cannot overlap a Cilium re-apply window (campaign_ctl.py wait)
#   - runs run_one.sh: reset to 1 replica (cell does not start unless every
#     Deployment is at 1 ready pod), Hubble labels / ClusterMesh checked before
#     and after the load (DMOS_FLEET_DIR)
#   - checks the run (campaign_ctl.py check); a run that is not valid moves to
#     results/_retried/ and the cell is run again, up to DMOS_CAMPAIGN_TRIES (3)
#   - two cells in a row with no valid run stop the campaign (fleet to inspect)
# Output: results/_campaign_<name>.log (progress), .tsv (one line per attempt),
#         results/_<name>_<block>_<arm>_<regime>_t<n>.log (one log per attempt).
# Keeps Windows awake while it runs (no settings changed).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PLAN="${1:?uso: bash experiments/run_campaign.sh <file .plan>}"
[ -f "$PLAN" ] || { echo "ERRORE: $PLAN non esiste"; exit 2; }
: "${DMOS_FLEET_DIR:?ERRORE: DMOS_FLEET_DIR non settata (controllo di ClusterMesh ed etichette Hubble a ogni run)}"
: "${DMOS_K6_REMOTE:?ERRORE: DMOS_K6_REMOTE non settata (VM di k6, es. ubuntu@<K6_VM_IP>)}"
export DMOS_FLEET_DIR DMOS_K6_REMOTE DMOS_RAW_OUT="${DMOS_RAW_OUT:-0}" DMOS_EXPECT_RESET=1
PY="venv/Scripts/python.exe"; [ -f "$PY" ] || PY="python"
TRIES="${DMOS_CAMPAIGN_TRIES:-3}"
RUN_MIN="${DMOS_CAMPAIGN_RUN_MIN:-24}"   # upper bound of one cell, for the window check
NAME="$(basename "$PLAN" .plan)"
R=experiments/results
LOG="$R/_campaign_${NAME}.log"; TSV="$R/_campaign_${NAME}.tsv"
mkdir -p "$R/_retried"
[ -f "$TSV" ] || printf 'time\tblock\tarm\tregime\ttry\tstatus\trun\treasons\n' > "$TSV"
say() { echo "$(date '+%d/%m %H:%M') $*" | tee -a "$LOG"; }

# Keep the PC awake while the file exists (SetThreadExecutionState, like "caffeine")
AWAKE="$(mktemp -t dmos_awake_XXXXXX)"
if command -v pwsh >/dev/null 2>&1; then
  pwsh -NoProfile -Command "\$k = Add-Type -Name P -Namespace W -PassThru -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'; while (Test-Path '$(cygpath -w "$AWAKE" 2>/dev/null || echo "$AWAKE")') { [void]\$k::SetThreadExecutionState([uint32]2147483649); Start-Sleep 30 }" >/dev/null 2>&1 &
fi
trap 'rm -f "$AWAKE"' EXIT

PRESTOP=""; fails_in_row=0; n_ok=0; n_bad=0
# a plan saved with Windows line endings (core.autocrlf, Notepad) would read "con\r": strip \r
say "=== campagna $NAME: $(tr -d '\r' < "$PLAN" | grep -cvE '^\s*(#|$|@)') celle, fino a $TRIES tentativi ciascuna ==="
exec 3< <(tr -d '\r' < "$PLAN")
while read -r -u 3 block arm regime extra; do
  [[ -z "${block:-}" || "$block" == \#* ]] && continue
  if [ "$block" = "@prestop" ]; then
    "$PY" experiments/campaign_ctl.py wait --minutes 15 | tee -a "$LOG"
    say "preStop: applico (rolling update di frontend e servizi co-scalati)"
    bash experiments/apply_pod_lifecycle.sh >> "$LOG" 2>&1 \
      || { say "STOP: apply_pod_lifecycle.sh fallito (vedi $LOG)"; exit 1; }
    n=$(bash experiments/apply_pod_lifecycle.sh --verify 2>&1 | tee -a "$LOG" | grep -o '=30s/preStop15s' | wc -l)
    want=$(( $(jq '.clusters | length' .cache/test-env.json) * $("$PY" -c "import sys; sys.path.insert(0,'experiments'); import campaign_ctl as c; print(len(c.lifecycle_services(c.ROOT)))") ))
    [ "$n" -eq "$want" ] || { say "STOP: preStop verificato su $n deployment su $want"; exit 1; }
    say "preStop ok su $n deployment: richiesto in ogni run da qui in poi"
    PRESTOP="--prestop"; continue
  fi
  if [ "$block" = "@apply" ]; then
    say "patch: applico $arm"
    { git apply --check "$arm" && git apply "$arm"; } >> "$LOG" 2>&1 \
      || { say "STOP: patch $arm non applicabile (vedi $LOG)"; exit 1; }
    "$PY" -m pytest -q tests >> "$LOG" 2>&1 \
      || { say "STOP: test falliti dopo la patch $arm (vedi $LOG)"; exit 1; }
    say "patch applicata, test ok"
    continue
  fi
  ok=0
  for try in $(seq 1 "$TRIES"); do
    "$PY" experiments/campaign_ctl.py wait --minutes "$RUN_MIN" | tee -a "$LOG"
    clog="$R/_${NAME}_${block}_${arm}_${regime}_t${try}.log"
    since=$(date +%s)
    say "[$block] $arm $regime (tentativo $try) ${extra:-}"
    env ${extra:-} bash experiments/run_one.sh "$arm" "$regime" > "$clog" 2>&1 < /dev/null
    rc=$?
    res="$("$PY" experiments/campaign_ctl.py check --arm "$arm" --regime "$regime" --since "$since" \
            --log "$clog" --rc "$rc" $PRESTOP)"
    IFS=$'\t' read -r run status reasons <<< "$res"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$block" "$arm" "$regime" "$try" \
      "$status" "$(basename "$run")" "$reasons" >> "$TSV"
    if [ "$status" = "OK" ]; then
      say "    OK  $(basename "$run")"; ok=1; break
    fi
    say "    NON VALIDO: $reasons"
    [ -d "$run" ] && mv "$run" "$R/_retried/" && say "    spostato in $R/_retried/$(basename "$run")"
  done
  if [ "$ok" = 1 ]; then n_ok=$((n_ok+1)); fails_in_row=0
  else
    n_bad=$((n_bad+1)); fails_in_row=$((fails_in_row+1))
    say "    CELLA PERSA dopo $TRIES tentativi: $block $arm $regime"
    if [ "$fails_in_row" -ge 2 ]; then
      say "STOP: due celle di fila senza run valido, la flotta va controllata"; exit 1
    fi
  fi
done
say "=== fine: $n_ok celle valide, $n_bad perse (dettagli in $TSV) ==="
[ "$n_bad" -eq 0 ]
