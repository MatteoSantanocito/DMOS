#!/usr/bin/env bash
# archive_run.sh — archivia i run ed elimina i file pesanti.
#
# Per ogni run dir: elimina k6_events.json (~1.2 GB, raw k6 inutile dopo il
# summary), comprime dmos_live.jsonl, e sposta la cartella in experiments/archive/.
# Tiene: flash_crowd_summary.json, manifest.json, dmos_live.jsonl.gz, *_stdout.txt.
#
# USO:
#   bash experiments/archive_run.sh                 # TUTTI i run in results/
#   bash experiments/archive_run.sh <run_dir>       # solo quello
#   bash experiments/archive_run.sh --prune-only    # solo elimina pesanti, non sposta
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
RESULTS="experiments/results"; ARCHIVE="experiments/archive"
mkdir -p "$ARCHIVE"

PRUNE_ONLY=0; TARGETS=()
for a in "$@"; do
  case "$a" in
    --prune-only) PRUNE_ONLY=1 ;;
    *) TARGETS+=("$a") ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && mapfile -t TARGETS < <(find "$RESULTS" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)

freed=0
for d in "${TARGETS[@]}"; do
  [ -d "$d" ] || { echo "  [skip] non esiste: $d"; continue; }
  base=$(basename "$d")
  # elimina il raw pesante
  if [ -f "$d/k6_events.json" ]; then
    sz=$(stat -c%s "$d/k6_events.json" 2>/dev/null || echo 0)
    rm -f "$d/k6_events.json"; freed=$((freed + sz))
    echo "  [prune] $base/k6_events.json ($((sz/1024/1024)) MB)"
  fi
  # comprimi la timeline
  [ -f "$d/dmos_live.jsonl" ] && gzip -f "$d/dmos_live.jsonl" 2>/dev/null
  # sposta in archive
  if [ "$PRUNE_ONLY" -eq 0 ]; then
    mv -f "$d" "$ARCHIVE/" 2>/dev/null && echo "  [arch]  -> $ARCHIVE/$base"
  fi
done
echo "  spazio liberato: ~$((freed/1024/1024)) MB"
echo "  results/ ora: $(du -sh "$RESULTS" 2>/dev/null | cut -f1)  |  archive/: $(du -sh "$ARCHIVE" 2>/dev/null | cut -f1)"
