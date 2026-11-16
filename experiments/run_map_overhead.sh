#!/usr/bin/env bash
# =============================================================================
# run_map_overhead.sh — does the live flow map change the measurements?
# =============================================================================
#
# The same cell (default: reach-pool-ps con) run N times with the live flow
# map open (experiments/live_flow_map.py: Hubble streams of every cluster +
# the page in the browser) and N times without, in ABBA order
# (on off off on on off off on ...) so that a drift of the testbed over the
# hours does not look like an effect of the map.
#
# Each run folder gets LIVE_MAP_on.txt or LIVE_MAP_off.txt; the "on" runs also
# get live_map_state.json (relay status and number of flows read: proof that
# the map was streaming) and live_map.log. A cell does not start while a
# hubble process is still running (a leftover stream would load the "off" runs).
#
# Usage (repository root, Git Bash, same environment as run_one.sh):
#   bash experiments/run_map_overhead.sh                 # 4 pairs, reach-pool-ps con
#   PAIRS=3 ARM=locality-ps REGIME=con-tosg bash experiments/run_map_overhead.sh
#   OPEN_BROWSER=0 ...                                   # streams only, no page
# =============================================================================
set -u

ARM="${ARM:-reach-pool-ps}"
REGIME="${REGIME:-con}"
PAIRS="${PAIRS:-4}"
PORT="${MAP_PORT:-8765}"
OPEN_BROWSER="${OPEN_BROWSER:-1}"
RESULTS="experiments/results"
PY="venv/Scripts/python.exe"; [ -f "$PY" ] || PY="python"
mkdir -p "$RESULTS"

hubble_running() {
  if command -v tasklist >/dev/null 2>&1; then
    tasklist 2>/dev/null | grep -qiE '^hubble(\.exe)?[[:space:]]'
  else
    pgrep -x hubble >/dev/null 2>&1
  fi
}

open_page() {
  [ "$OPEN_BROWSER" = "1" ] || return 0
  local url="http://127.0.0.1:${PORT}/"
  if command -v cmd >/dev/null 2>&1; then cmd //c start "" "$url" >/dev/null 2>&1
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1
  fi
}

# ABBA: pair p is (on, off) for odd p and (off, on) for even p
order=()
for ((p = 1; p <= PAIRS; p++)); do
  if (( p % 2 == 1 )); then order+=(on off); else order+=(off on); fi
done
echo "map overhead: ${ARM} ${REGIME}, order ${order[*]} (~$(( ${#order[@]} * 20 )) min)"

k=0
for map in "${order[@]}"; do
  k=$((k + 1))
  if hubble_running; then
    echo "$(date +%H:%M) [$k] STOP: a hubble process is still running (taskkill //F //IM hubble.exe)"
    exit 1
  fi
  stamp="$RESULTS/.map_overhead_stamp_$k"
  touch "$stamp"
  if [ "$map" = "on" ]; then
    "$PY" experiments/live_flow_map.py --port "$PORT" --duration 2400 \
      > "$RESULTS/_map_overhead_${k}_map.log" 2>&1 &
    map_pid=$!
    for _ in $(seq 1 60); do                       # wait for the page (CiliumNode read + streams)
      curl -s -o /dev/null "http://127.0.0.1:${PORT}/state" && break
      sleep 2
    done
    open_page
    sleep 20                                       # streams connected before the load
  fi

  bash experiments/run_one.sh "$ARM" "$REGIME" > "$RESULTS/_map_overhead_${k}.log" 2>&1
  rc=$?

  run_dir="$(find "$RESULTS" -maxdepth 1 -type d -name "*_dmos-${ARM}__${REGIME}" -newer "$stamp" \
             | sort | tail -1)"
  if [ "$map" = "on" ]; then
    [ -n "$run_dir" ] && curl -s "http://127.0.0.1:${PORT}/state" > "$run_dir/live_map_state.json"
    curl -s -o /dev/null "http://127.0.0.1:${PORT}/quit"
    for _ in $(seq 1 15); do kill -0 "$map_pid" 2>/dev/null || break; sleep 1; done
    [ -n "$run_dir" ] && cp "$RESULTS/_map_overhead_${k}_map.log" "$run_dir/live_map.log"
  fi
  if [ -n "$run_dir" ]; then
    echo "$map" > "$run_dir/LIVE_MAP_${map}.txt"
  fi
  rm -f "$stamp"
  echo "$(date +%H:%M) [$k] map=$map rc=$rc ${run_dir:-NESSUNA CARTELLA (run abortito? vedi $RESULTS/_map_overhead_${k}.log)}"
done
