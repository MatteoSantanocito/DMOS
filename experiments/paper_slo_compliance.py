"""
paper_slo_compliance.py — Post-hoc SLO violation analysis from k6 raw events
============================================================================

PERCHE' QUESTO SCRIPT
---------------------
Nelle 9 run a 250 rps, `http_req_failed` e' 0% in tutti gli arm — k6 ha
timeout di default 60s, quindi le request lente (~5% nella coda saturata
di cluster-01 a 6.5s) vengono comunque servite e non contano come errori.

Per il paper questo e' un punto debole: un reviewer potrebbe dire "se non
ci sono errori, perche' DMOS aiuta?". La risposta corretta secondo la
letteratura (Dean & Barroso 2013, "The Tail at Scale") e' inquadrare il
problema su SLO COMPLIANCE: una request servita in 6.5s viola lo SLO di
una web app interattiva tanto quanto un 5xx.

Questo script calcola POST-HOC dai `k6_events.json` gia' raccolti la
percentuale di request che hanno violato lo SLO. Soglie default: 1000ms
(SLO standard interactive web) e 2000ms (SLO largo / mobile).

OUTPUT
------
- Per ogni run: JSON cached in `<run_dir>/slo_compliance.json`
- Tabella aggregata (mean +/- CI95 su 3 ripetizioni) per arm, stampata
  in markdown pronto per essere incollato in Table 1 del paper.

USO
---
  python experiments/paper_slo_compliance.py
  python experiments/paper_slo_compliance.py --thresholds 500,1000,2000
  python experiments/paper_slo_compliance.py --force-rescan  # ignora cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Import paper_lib for run discovery + stats
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from paper_lib import (  # noqa: E402
    SCENARIOS, SCENARIO_LABELS, discover_runs, bootstrap_ci,
)

DEFAULT_THRESHOLDS_MS = [1000, 2000]


def scan_k6_events(events_path: Path, thresholds_ms: List[float]) -> Dict:
    """
    Scansiona k6_events.json line-by-line, conta http_req_duration Points
    sopra ogni soglia, globalmente e per-phase.

    Returns:
        {
          'n_requests_total': int,
          'thresholds_ms': [1000, 2000],
          'global': {
              '1000': {'n_above': int, 'pct_above': float},
              '2000': {...},
          },
          'per_phase': {
              'baseline': {'n_requests': int, 'thresholds': {...}},
              'spike':    {...},
              ...
          },
        }
    """
    # Inizializzo contatori
    phases = ['baseline', 'spike', 'peak', 'decay', 'cooldown']
    n_total = 0
    n_above = {str(t): 0 for t in thresholds_ms}
    per_phase: Dict[str, Dict] = {
        ph: {
            'n_requests': 0,
            'n_above': {str(t): 0 for t in thresholds_ms},
        }
        for ph in phases
    }

    # Substring per skippare velocemente righe che non servono
    needle = b'"metric":"http_req_duration"'

    with events_path.open('rb') as f:
        for line in f:
            # Fast-path: la maggioranza delle righe NON e' http_req_duration
            if needle not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Filtra: solo Points (non la definizione del Metric)
            if obj.get('type') != 'Point':
                continue
            data = obj.get('data', {})
            val = data.get('value')
            if val is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue

            tags = data.get('tags', {})
            phase = tags.get('phase', '')
            ok = tags.get('expected_response', '') == 'true'

            n_total += 1
            phase_bucket = per_phase.get(phase)
            if phase_bucket is not None:
                phase_bucket['n_requests'] += 1

            for t in thresholds_ms:
                # Violazione SLO = richiesta LENTA (durata>soglia) OPPURE FALLITA.
                # Senza l'OR-failed, un 503 servito in 26ms NON conterebbe come
                # violazione → premierebbe gli arm che scaricano carico (fail-fast).
                if (val_f > t) or (not ok):
                    n_above[str(t)] += 1
                    if phase_bucket is not None:
                        phase_bucket['n_above'][str(t)] += 1

    # Computo percentuali
    result = {
        'n_requests_total': n_total,
        'thresholds_ms': thresholds_ms,
        'global': {},
        'per_phase': {},
    }
    for t in thresholds_ms:
        n = n_above[str(t)]
        pct = (100.0 * n / n_total) if n_total > 0 else 0.0
        result['global'][str(t)] = {'n_above': n, 'pct_above': pct}

    for ph, bucket in per_phase.items():
        nph = bucket['n_requests']
        ph_out = {'n_requests': nph, 'thresholds': {}}
        for t in thresholds_ms:
            n = bucket['n_above'][str(t)]
            pct = (100.0 * n / nph) if nph > 0 else 0.0
            ph_out['thresholds'][str(t)] = {'n_above': n, 'pct_above': pct}
        result['per_phase'][ph] = ph_out

    return result


def compliance_per_arm(groups, threshold_ms: float, phase: Optional[str] = None,
                       cache_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """
    Per ogni arm, array di SLO violation rate (%) — uno per ripetizione.
    Se `phase` e' None, prende il valore globale; altrimenti la phase indicata.
    `cache_data[run_dir_str]` deve essere il dict ritornato da scan_k6_events.
    """
    out: Dict[str, np.ndarray] = {}
    cache_data = cache_data or {}
    for scenario, group in groups.items():
        vals = []
        for run in group.runs:
            d = cache_data.get(str(run.run_dir))
            if not d:
                continue
            if phase is None:
                src = d.get('global', {}).get(str(threshold_ms), {})
            else:
                src = d.get('per_phase', {}).get(phase, {}) \
                       .get('thresholds', {}).get(str(threshold_ms), {})
            pct = src.get('pct_above')
            if pct is not None:
                vals.append(float(pct))
        out[scenario] = np.array(vals)
    return out


def fmt_mean_ci(vals: np.ndarray, fmt: str = '{:.2f}') -> str:
    if len(vals) == 0:
        return 'n/a'
    if len(vals) == 1:
        return fmt.format(vals[0])
    m, lo, hi = bootstrap_ci(vals)
    half = (hi - lo) / 2.0
    return f"{fmt.format(m)} ± {fmt.format(half)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', type=Path,
                    default=SCRIPT_DIR / 'results',
                    help='Cartella con i run (default: experiments/results)')
    ap.add_argument('--thresholds', type=str,
                    default=','.join(str(int(t)) for t in DEFAULT_THRESHOLDS_MS),
                    help='Soglie SLO in ms, comma-separated (default 1000,2000)')
    ap.add_argument('--force-rescan', action='store_true',
                    help='Ignora la cache slo_compliance.json e riscanna')
    ap.add_argument('--out-json', type=Path, default=None,
                    help='Se settato, salva il dict aggregato qui')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',')]

    print(f"[discover] scanning {args.results_dir}")
    groups = discover_runs(args.results_dir, load_timeline=False)

    if not groups:
        print("[err] nessun run trovato")
        sys.exit(2)

    # ── Step 1: per ogni run, calcola (o riusa cache) la compliance ─────────
    cache_data: Dict[str, Dict] = {}

    for scenario, group in groups.items():
        print(f"\n[{scenario}] {len(group.runs)} run")
        for run in group.runs:
            events_path = run.run_dir / 'k6_events.json'
            cache_path = run.run_dir / 'slo_compliance.json'

            if not events_path.is_file():
                print(f"  [warn] {run.run_dir.name}: k6_events.json mancante")
                continue

            # Riusa cache se le soglie coincidono
            need_rescan = args.force_rescan
            if cache_path.is_file() and not need_rescan:
                try:
                    cached = json.loads(cache_path.read_text())
                    cached_thr = set(cached.get('thresholds_ms', []))
                    if set(thresholds).issubset(cached_thr):
                        cache_data[str(run.run_dir)] = cached
                        print(f"  [cache] {run.run_dir.name}: "
                              f"n={cached['n_requests_total']:,}")
                        continue
                    else:
                        need_rescan = True
                except (json.JSONDecodeError, KeyError):
                    need_rescan = True

            # Scan vero e proprio (richiede ~30s per file da ~880MB)
            size_mb = events_path.stat().st_size / 1024 / 1024
            print(f"  [scan]  {run.run_dir.name}: {size_mb:.0f} MB ... ", end='', flush=True)
            result = scan_k6_events(events_path, thresholds)
            cache_path.write_text(json.dumps(result, indent=2))
            cache_data[str(run.run_dir)] = result
            print(f"n={result['n_requests_total']:,}  "
                  f"global>{int(thresholds[-1])}ms="
                  f"{result['global'][str(thresholds[-1])]['pct_above']:.2f}%")

    # ── Step 2: aggregato per arm (mean +/- CI95) ──────────────────────────
    print("\n" + "=" * 72)
    print("SLO COMPLIANCE — aggregato per arm (mean ± CI95 across reps)")
    print("=" * 72)

    # Markdown table compatibile col paper
    header = "| Arm |" + "".join(
        f" Global > {int(t)} ms |" for t in thresholds
    )
    sep = "|" + "---|" * (1 + len(thresholds))
    print()
    print(header)
    print(sep)
    for scenario in SCENARIOS:
        if scenario not in groups:
            continue
        cells = [SCENARIO_LABELS.get(scenario, scenario)]
        for t in thresholds:
            arr = compliance_per_arm(groups, t, phase=None,
                                     cache_data=cache_data)[scenario]
            cells.append(fmt_mean_ci(arr, '{:.2f}') + '%')
        print("| " + " | ".join(cells) + " |")

    # Per-phase (solo per threshold piu' interessante = il primo)
    primary_t = thresholds[0]
    print()
    print(f"### Per-phase, SLO > {int(primary_t)} ms")
    phases = ['baseline', 'spike', 'peak', 'decay', 'cooldown']
    header2 = "| Arm |" + "".join(f" {ph} |" for ph in phases)
    sep2 = "|" + "---|" * (1 + len(phases))
    print()
    print(header2)
    print(sep2)
    for scenario in SCENARIOS:
        if scenario not in groups:
            continue
        cells = [SCENARIO_LABELS.get(scenario, scenario)]
        for ph in phases:
            arr = compliance_per_arm(groups, primary_t, phase=ph,
                                     cache_data=cache_data)[scenario]
            cells.append(fmt_mean_ci(arr, '{:.2f}') + '%')
        print("| " + " | ".join(cells) + " |")

    # Headline numbers per il paper
    print()
    print("=" * 72)
    print("HEADLINE NUMBERS per paper:")
    print("=" * 72)
    for t in thresholds:
        per_arm = compliance_per_arm(groups, t, cache_data=cache_data)
        print(f"\nSLO > {int(t)} ms — % di request che violano lo SLO:")
        for scenario in SCENARIOS:
            if scenario not in per_arm:
                continue
            arr = per_arm[scenario]
            print(f"  {SCENARIO_LABELS[scenario]:<25}: {fmt_mean_ci(arr, '{:.2f}'):>15}%   "
                  f"(reps={len(arr)})")
        # Reduction relativo
        if 'dmos-off' in per_arm and 'dmos-l1l2' in per_arm \
                and len(per_arm['dmos-off']) > 0 and len(per_arm['dmos-l1l2']) > 0:
            off_mean = per_arm['dmos-off'].mean()
            l1l2_mean = per_arm['dmos-l1l2'].mean()
            if off_mean > 0:
                red = 100.0 * (off_mean - l1l2_mean) / off_mean
                print(f"  -> L1+L2 vs OFF: riduzione {red:+.1f}% delle violazioni")

    if args.out_json:
        agg = {
            'thresholds_ms': thresholds,
            'per_arm': {},
        }
        for t in thresholds:
            per_arm = compliance_per_arm(groups, t, cache_data=cache_data)
            agg['per_arm'][str(t)] = {
                sc: arr.tolist() for sc, arr in per_arm.items()
            }
        args.out_json.write_text(json.dumps(agg, indent=2))
        print(f"\n[saved] {args.out_json}")


if __name__ == '__main__':
    main()
