#!/usr/bin/env bash
# validate_run.sh — verdetto automatico CLEAN / SUSPECT / DEGRADED su un run.
# Su questo testbed il data-plane può degradare a metà run (timeout su pod sani):
# non si previene, si DETECTA e si scarta. Questo script ti dà il verdetto subito
# dopo ogni run, così tieni i puliti e rifai i degradati.
#
# USO:
#   bash experiments/validate_run.sh              # valida l'ULTIMO run
#   bash experiments/validate_run.sh <run_dir>    # valida un run specifico
#
# Verdetto (euristica sulla firma del soft-degrado che abbiamo osservato):
#   DEGRADED  peak err > 10%  OPPURE  un cluster NON-hot ha err > 2%
#             (un cluster che non dovrebbe fallire fallisce = data-plane rotto)
#   SUSPECT   peak err 6-10%  OPPURE  hot p95 > 6000ms con molti timeout
#   CLEAN     altrimenti
# Exit: 0=CLEAN, 2=SUSPECT, 3=DEGRADED, 1=errore.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY="venv/Scripts/python.exe"; [ -f "$PY" ] || PY="python"

RUN="${1:-}"
if [ -z "$RUN" ]; then
  RUN=$(ls -dt experiments/results/*_dmos-* 2>/dev/null | grep -v _quarantine | head -1)
fi
[ -d "$RUN" ] || { echo "run dir non trovata: $RUN"; exit 1; }

export PYTHONIOENCODING=utf-8
"$PY" - "$RUN" <<'PY'
import json, sys, os, re
run = sys.argv[1]
label = os.path.basename(run)
summ = json.load(open(os.path.join(run, "flash_crowd_summary.json"), encoding="utf-8"))
phases = summ["results"]["per_phase"]
ph = phases["peak"]
peak_err = ph["errorRate"] * 100
peak_p95 = ph["p95"]
# baseline p95: a 50 rps idle dev'essere basso (~400ms). Un valore alto indica
# che il fleet era GIA' degradato all'avvio (coda di latenza che lo 0%-err non
# cattura). NB: il cooldown e' SEMPRE alto (churn di scale-down) -> non lo usiamo.
baseline_p95 = phases.get("baseline", {}).get("p95", 0)
pc = summ["results"].get("per_cluster", {})

# hot cluster = quello con più richieste
hot = max(pc.items(), key=lambda kv: kv[1].get("requests", 0))[0] if pc else None
hot_err = pc.get(hot, {}).get("errorRate", 0) * 100 if hot else 0
hot_p95 = pc.get(hot, {}).get("p95", 0) if hot else 0
nonhot_max = max((v.get("errorRate", 0) * 100 for k, v in pc.items() if k != hot), default=0)

# frazione di 'request timeout' nel k6 stdout (firma del soft-degrado data-plane)
tmo = 0; total_req = sum(v.get("requests", 0) for v in pc.values()) or 1
ks = os.path.join(run, "k6_stdout.txt")
if os.path.exists(ks):
    txt = open(ks, encoding="utf-8", errors="ignore").read()
    tmo = txt.count("request timeout")
tmo_frac = 100.0 * tmo / total_req

# verdetto — DRIVER: metriche di PICCO (la fase che riportiamo). La baseline alta
# da sola e' una coda cosmetica (cold-start dei pod appena resettati) e NON
# scarta il run; conta solo se la degradazione raggiunge ANCHE il picco.
if peak_err > 10 or nonhot_max > 2:
    verdict, code = "DEGRADED", 3
elif peak_err > 6 or peak_p95 > 3000 or (baseline_p95 > 4000 and peak_p95 > 1500):
    verdict, code = "SUSPECT", 2
else:
    verdict, code = "CLEAN", 0

bar = "─" * 60
print(bar)
print(f"  RUN     : {label}")
print(f"  peak     : p95={peak_p95:.0f}ms  err={peak_err:.1f}%")
print(f"  baseline : p95={baseline_p95:.0f}ms  (idle 50rps, atteso ~400; alto = fleet degradato all'avvio)")
print(f"  hot={hot}: err={hot_err:.1f}%  p95={hot_p95:.0f}ms (whole-run, include code baseline/cooldown)")
print(f"  non-hot : err max={nonhot_max:.1f}%   (dovrebbe essere ~0)")
print(f"  timeout : {tmo} su {total_req} req = {tmo_frac:.1f}%")
print(bar)
mark = {"CLEAN":"✓", "SUSPECT":"?", "DEGRADED":"✗"}[verdict]
print(f"  VERDETTO: [{mark}] {verdict}")
if verdict == "DEGRADED":
    print("  → SCARTA e rifai (clean_reset.sh --check, poi rilancia il regime)")
elif verdict == "SUSPECT":
    print("  → borderline: tienilo ma assicurati di avere ≥2 rep CLEAN del regime")
else:
    print("  → tienilo: rep valida")
print(bar)
sys.exit(code)
PY
