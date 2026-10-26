#!/usr/bin/env python3
"""
paper_plot_skew_robustness.py — "p95 vs demand skew" robustness figure (Study A)
================================================================================

COSA MOSTRA (la figura-chiave della robustezza di L1)
-----------------------------------------------------
Asse X = skew della domanda (quanto il flash crowd e' concentrato su pochi
cluster): Uniform → Balanced → Concentrated. Asse Y = tail latency p95 (o %
violazioni SLO) nella fase di picco. Una linea per ogni arm di scheduling
({DMOS off, DMOS L2, DMOS L1+L2}), con banda CI95 bootstrap.

La tesi che la figura deve rendere visiva: man mano che la domanda si
concentra (skew cresce), gli arm senza placement geo-aware (off, L2) degradano
— la coda esplode perche' i pod restano dove la domanda NON e'. DMOS L1+L2
resta piatto perche' L1 ri-piazza la capacita' verso il cluster sotto stress.
Una curva piatta vs due curve che salgono = robustezza allo skew.

I run vanno nominati  dmos-<arm>__<regime>  (experiments/run_naming.py):
    arm    default dmos-off, dmos-l2, dmos-l1l2 (core NCA); qualsiasi altro arm
           con --arms (es. dmos-nca, dmos-gate_g0)
    regime default uni bal s50 s65 s75 con (skew crescente); le varianti di
           cluster caldo / picco (con-sg, con-p400) con --regimes
A PARITA' DI CONFIGURAZIONE DMOS e di intensita' di carico: l'unica variabile
e' il vettore di domanda per cluster (DMOS_SPIKE_W). Questo isola l'effetto
dello scheduling dallo skew, evitando l'accusa di pesi hand-tuned.

OUTPUT (in --output-dir, default paper_artifacts/<ts>/figures/)
  - fig_skew_robustness.{pdf,png}   2 pannelli affiancati: (a) p95, (b) %SLO.
  - skew_robustness_data.json       i numeri (mean+CI95) dietro la figura.

USO
---
  # prerequisito per il pannello SLO: paper_slo_compliance.py (popola le cache)
  python experiments/paper_plot_skew_robustness.py
  python experiments/paper_plot_skew_robustness.py --arms dmos-off dmos-l2 dmos-l1l2 \
      --regimes uni bal con
  python experiments/paper_plot_skew_robustness.py --metric p95   # solo un pannello
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paper_lib import (  # noqa: E402
    ScenarioGroup, discover_runs, print_group_summary,
    bootstrap_ci, parse_arm_regime,
    ARM_LABELS, ARM_COLORS, ARM_HATCHES, regime_label, regime_skew,
    scenario_color, setup_matplotlib, save_figure,
)
from paper_stat_compare import (  # noqa: E402
    SLO_THRESHOLD_MS, slo_violations_per_arm,
)

DEFAULT_ARMS = ['dmos-off', 'dmos-l2', 'dmos-l1l2']
DEFAULT_REGIMES = ['uni', 'bal', 's50', 's65', 's75', 'con']
P95_KEY = 'results.per_phase.peak.p95'


def _ci_for(group: Optional[ScenarioGroup], key_path: str):
    """(mean, lo, hi) bootstrap da k6_summary, o None se mancano dati."""
    if group is None:
        return None
    vals = group.metric_values_per_run(key_path)
    if len(vals) == 0:
        return None
    return bootstrap_ci(vals)


def _ci_from_array(vals: np.ndarray):
    if vals is None or len(vals) == 0:
        return None
    return bootstrap_ci(vals)


def collect(groups: Dict[str, ScenarioGroup],
            arms: List[str], regimes: List[str]) -> Dict:
    """
    Costruisce, per ogni arm e metrica, le serie ordinate per skew crescente.
    Ritorna {
      'arms': [...], 'regimes': [...], 'skew': [...],
      'p95': {arm: [{regime, skew, mean, lo, hi} | None per regime]},
      'slo': {arm: [...]},
    }
    """
    slo_data = slo_violations_per_arm(groups, SLO_THRESHOLD_MS, phase='peak')
    regimes_sorted = sorted(regimes, key=lambda r: regime_skew(r, 0.0))
    skew_axis = [regime_skew(r, i) for i, r in enumerate(regimes_sorted)]

    out: Dict = {'arms': arms, 'regimes': regimes_sorted, 'skew': skew_axis,
                 'p95': {}, 'slo': {}, 'goodput': {}}
    for arm in arms:
        p95_series, slo_series, gp_series = [], [], []
        for regime in regimes_sorted:
            label = f"{arm}__{regime}"
            grp = groups.get(label)
            # p95 dal k6_summary
            ci = _ci_for(grp, P95_KEY)
            p95_series.append(None if ci is None else
                              {'regime': regime, 'skew': regime_skew(regime),
                               'mean': ci[0], 'lo': ci[1], 'hi': ci[2],
                               'n': grp.n_repetitions})
            # SLO% dalla cache slo_compliance.json
            sci = _ci_from_array(slo_data.get(label))
            slo_series.append(None if sci is None else
                              {'regime': regime, 'skew': regime_skew(regime),
                               'mean': sci[0], 'lo': sci[1], 'hi': sci[2]})
            # goodput = rps di picco * (1 - errorRate), per run, poi CI95 bootstrap
            gci = None
            if grp is not None:
                rps = grp.metric_values_per_run('results.per_phase.peak.rps')
                err = grp.metric_values_per_run('results.per_phase.peak.errorRate')
                if len(rps) > 0 and len(rps) == len(err):
                    gci = bootstrap_ci(np.asarray(rps) * (1.0 - np.asarray(err)))
            gp_series.append(None if gci is None else
                             {'regime': regime, 'skew': regime_skew(regime),
                              'mean': gci[0], 'lo': gci[1], 'hi': gci[2]})
        out['p95'][arm] = p95_series
        out['slo'][arm] = slo_series
        out['goodput'][arm] = gp_series
    return out


def _plot_panel(ax, data: Dict, arms: List[str], metric_key: str,
                ylabel: str, title: str):
    """Barre raggruppate: x = regimi (test isolati), una barra per arm.
    Le barre evitano la falsa continuità delle linee fra punti distinti."""
    regimes = data['regimes']
    n_reg, n_arm = len(regimes), len(arms)
    x = np.arange(n_reg)
    width = 0.8 / max(1, n_arm)
    plotted_any = False
    for ai, arm in enumerate(arms):
        series = data[metric_key].get(arm, [])
        ys, los, his = [], [], []
        for pt in series:
            if pt is None:
                ys.append(np.nan); los.append(0.0); his.append(0.0)
            else:
                ys.append(pt['mean'])
                los.append(max(0.0, pt['mean'] - pt['lo']))
                his.append(max(0.0, pt['hi'] - pt['mean']))
        if all(v != v for v in ys):   # tutti NaN
            continue
        plotted_any = True
        color = ARM_COLORS.get(arm, scenario_color(arm))
        offset = (ai - (n_arm - 1) / 2.0) * width
        ax.bar(x + offset, ys, width, yerr=[los, his], capsize=3,
               color=color, label=ARM_LABELS.get(arm, arm),
               edgecolor='black', linewidth=0.4,
               error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([regime_label(r) for r in regimes],
                       rotation=15, ha='right', fontsize=8)
    ax.set_xlabel('Demand skew — share on hottest cluster')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return plotted_any


def plot(data: Dict, arms: List[str], out_dir: Path,
         which: str = 'both') -> Optional[List[Path]]:
    import matplotlib.pyplot as plt
    panels = []
    if which in ('p95', 'both', 'all'):
        panels.append(('p95', 'p95 latency (ms)', 'Tail latency vs demand skew'))
    if which in ('slo', 'both', 'all'):
        panels.append(('slo', f'requests > {int(SLO_THRESHOLD_MS)} ms (%)',
                       'SLO violations vs demand skew'))
    if which in ('goodput', 'all'):
        panels.append(('goodput', 'goodput (successful req/s)',
                       'Goodput vs demand skew'))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.6),
                             squeeze=False)
    any_plotted = False
    for ax, (key, ylab, title) in zip(axes[0], panels):
        if _plot_panel(ax, data, arms, key, ylab, title):
            any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return None
    # legenda condivisa
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(arms),
               frameon=True, fontsize=8, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    return save_figure(fig, out_dir, 'fig_skew_robustness')


def main():
    ap = argparse.ArgumentParser(
        description="Study A — p95/SLO vs demand-skew robustness figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--results-dir', type=Path, default=SCRIPT_DIR / 'results')
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--arms', nargs='+', default=DEFAULT_ARMS)
    ap.add_argument('--regimes', nargs='+', default=DEFAULT_REGIMES)
    ap.add_argument('--metric', choices=['p95', 'slo', 'goodput', 'both', 'all'],
                    default='all')
    args = ap.parse_args()

    # filtro discovery: tutte le combinazioni arm×regime
    wanted = [f"{a}__{r}" for a in args.arms for r in args.regimes]

    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = SCRIPT_DIR.parent / f'paper_artifacts/{ts}/figures'
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Discovering runs in: {args.results_dir}")
    print(f"  Arms   : {args.arms}")
    print(f"  Regimes: {args.regimes}  (skew order: "
          f"{sorted(args.regimes, key=lambda r: regime_skew(r, 0.0))})")
    groups = discover_runs(args.results_dir, scenario_filter=wanted,
                           load_timeline=False)
    print_group_summary(groups)
    if not groups:
        print("  No matching <arm>__<regime> runs found.")
        print(f"  Attesi dir tipo: <ts>_{args.arms[0]}__{args.regimes[0]}")
        return 1

    setup_matplotlib()
    data = collect(groups, args.arms, args.regimes)

    fig_paths = plot(data, args.arms, args.output_dir, which=args.metric)
    (args.output_dir / 'skew_robustness_data.json').write_text(
        json.dumps(data, indent=2, default=str), encoding='utf-8')

    print(f"\n  Writing artifacts in: {args.output_dir}\n")
    print("    ✓ skew_robustness_data.json")
    if fig_paths:
        for p in fig_paths:
            print(f"    ✓ {p.name}")
        print("\n  Done. Nel paper:")
        print(f"      \\includegraphics{{{args.output_dir.name}/fig_skew_robustness.pdf}}\n")
    else:
        print("    [warn] nessun dato sufficiente per tracciare la figura.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
