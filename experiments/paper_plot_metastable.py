#!/usr/bin/env python3
"""
paper_plot_metastable.py — Paper Figure 11 (cooldown adaptation transients)
============================================================================

Genera una figura "design rationale" che mostra come l'autoscaler senza
geo-awareness (L2) sostenga spike di latenza ricorrenti durante la fase
di cooldown, mentre L1+L2 recupera in modo pulito.

NOTA: la precedente Fig 10 ("Hubble inflation") e' stata rimossa perche'
il fattore ~10x tra Hubble observed rate e k6 offered rate riflette il
naturale fanout dei microservizi Online Boutique (frontend invoca ~10
servizi via gRPC per ogni richiesta esterna), non retry amplification.
Mantenerla nel paper avrebbe esposto a una critica fondata di reviewer
con esperienza microservizi.

Output (in --output-dir, default: paper_artifacts/<timestamp>/figures/):
  - fig11_cooldown_adaptation.{pdf,png}
       p95 latency nel tempo per il run "L2-only" (transient di adaptation
       nel cooldown) vs il run "L1+L2" (recovery pulito).

Usage
-----
    python paper_plot_metastable.py --auto
    python paper_plot_metastable.py \\
        --naive-run experiments/results/<dmos-l2_run> \\
        --fixed-run experiments/results/<dmos-l1l2_run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paper_lib import (
    PHASE_DEFS, setup_matplotlib, save_figure, shade_phases,
)
from paper_plot_trajectories import _load_k6_events


# ── Helpers ─────────────────────────────────────────────────────────────────

def _p95_over_time(k6_events_path: Path,
                   bin_sec: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    p95 della latency per bin di `bin_sec` secondi.

    bin_sec=30 (era 15): allarghiamo i bin per smussare il sawtooth artifact
    al 5° percentile della coda saturata, che con bin troppo piccoli oscilla
    fra valori fast e slow.
    """
    df = _load_k6_events(k6_events_path, include_status=False)
    if df is None or df.empty:
        return np.array([]), np.array([])
    df = df.copy()
    df['bin_s'] = (df['elapsed_s'] // bin_sec * bin_sec).astype(int)
    g = df.groupby('bin_s')['value'].quantile(0.95).reset_index()
    return g['bin_s'].to_numpy(), g['value'].to_numpy()


# ── Figure 11 — Cooldown adaptation transients ──────────────────────────────

def plot_fig11_collapse_recovery(naive_run: Path,
                                 fixed_run: Optional[Path],
                                 out_dir: Path,
                                 formats=('pdf', 'png')) -> List[Path]:
    """
    p95 latency over time: L2-only (residual transients in cooldown) vs L1+L2
    (clean recovery).

    L'arm L2-only mostra adaptation transients ricorrenti nel cooldown — un
    indicatore di residual saturation che il piping reattivo dell'autoscaler
    non riesce a smorzare senza una decisione di placement informata (L1).
    """
    import matplotlib.pyplot as plt

    if not (naive_run / 'k6_events.json').is_file():
        print(f"  [fig11] manca k6_events.json in {naive_run} → skip")
        return []

    fig, ax = plt.subplots(figsize=(11, 5.5))
    shade_phases(ax, x_unit='min', y_frac_label=0.97)

    # L2-only arm
    t_n, p95_n = _p95_over_time(naive_run / 'k6_events.json')
    if len(t_n) > 0:
        ax.plot(t_n / 60.0, p95_n, color='#F57C00', linewidth=2.0,
                label='DMOS L2 (scaling only)')

    # L1+L2 arm (opzionale)
    if fixed_run is not None and (fixed_run / 'k6_events.json').is_file():
        t_f, p95_f = _p95_over_time(fixed_run / 'k6_events.json')
        if len(t_f) > 0:
            ax.plot(t_f / 60.0, p95_f, color='#1565C0', linewidth=2.0,
                    label='DMOS L1+L2 (full)')

    # Reference SLO line invece del fittizio k6 timeout
    ax.axhline(1000, color='gray', linestyle=':', linewidth=1.2, alpha=0.8,
               label='1 s SLO')

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("p95 Latency (ms, 30-s bins)")
    ax.set_xlim(0, 15)
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.set_title(
        "Figure 11 — Cooldown adaptation transients in autoscaling without "
        "geo-aware placement\n"
        "(L2-only sustains residual saturation transients into cooldown; "
        "L1+L2 recovers cleanly)",
        fontsize=11, fontweight='bold', pad=12,
    )

    return save_figure(fig, out_dir, 'fig11_cooldown_adaptation', formats)


# ── Auto-discovery dei run da usare ─────────────────────────────────────────

def _auto_discover(results_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Heuristica:
      naive_run = il piu' vecchio dmos-l2 (presumibilmente con vecchio codice).
      fixed_run = il piu' recente dmos-l1l2 se esiste, altrimenti il piu' recente
                  dmos-off (come "control" che NON ha il collapse).
    """
    runs = sorted(results_dir.iterdir())
    l2_runs = [r for r in runs if r.is_dir() and r.name.endswith('_dmos-l2')]
    l1l2_runs = [r for r in runs if r.is_dir() and r.name.endswith('_dmos-l1l2')]
    off_runs = [r for r in runs if r.is_dir() and r.name.endswith('_dmos-off')]
    naive = l2_runs[0] if l2_runs else None
    fixed = (l1l2_runs[-1] if l1l2_runs else
             (off_runs[-1] if off_runs else None))
    return naive, fixed


# ── Main CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figs 10-11 (metastable failure analysis) for the DMOS paper.",
    )
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent / 'results',
                        help="Directory con i run dirs (default: experiments/results)")
    parser.add_argument('--naive-run', type=Path, default=None,
                        help='Cartella di un run "naive" (collapse atteso)')
    parser.add_argument('--fixed-run', type=Path, default=None,
                        help='Cartella di un run "fixed" (recovery atteso)')
    parser.add_argument('--auto', action='store_true',
                        help='Auto-discover naive (oldest dmos-l2) e fixed (latest dmos-l1l2 o dmos-off)')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg'])
    parser.add_argument('--dpi', type=int, default=600)
    args = parser.parse_args()

    setup_matplotlib(target_dpi=args.dpi)

    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = Path(__file__).parent.parent / f'paper_artifacts/{ts}/figures'
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    naive = args.naive_run
    fixed = args.fixed_run
    if args.auto or (naive is None):
        a_naive, a_fixed = _auto_discover(args.results_dir)
        naive = naive or a_naive
        fixed = fixed or a_fixed

    if naive is None or not naive.is_dir():
        print(f"  [error] naive run non trovato. Specificare con --naive-run.")
        return 1

    print(f"\n  Naive run: {naive.name}")
    if fixed is not None:
        print(f"  Fixed run: {fixed.name}")
    else:
        print(f"  Fixed run: (assente — Fig 11 mostrera' solo la naive)")
    print(f"  Output dir: {args.output_dir}\n")

    formats = tuple(args.formats)
    written = []
    for name, fn in [
        ("Fig 11 (cooldown adaptation)",      lambda: plot_fig11_collapse_recovery(naive, fixed, args.output_dir, formats)),
    ]:
        print(f"  {name}...")
        try:
            paths = fn()
            for p in paths:
                print(f"    ✓ {p.name}  ({p.stat().st_size/1024:.1f} KB)")
            written.extend(paths)
        except Exception as e:
            print(f"    ✗ failed: {e}")
            import traceback; traceback.print_exc()

    print(f"\n  Done. {len(written)} files written.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
