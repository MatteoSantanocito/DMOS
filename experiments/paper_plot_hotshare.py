#!/usr/bin/env python3
"""
paper_plot_hotshare.py — "frontend replica share on hottest cluster vs demand skew"
===================================================================================
Figura-meccanismo: mostra DOVE va la capacità. X = skew della domanda
(quota sul cluster più caldo), Y = % di repliche frontend collocate su quel
cluster (default c01/EU-DE), una barra per arm. Si appaia con
fig_skew_robustness (stesso asse X).

Storia: DMOS (l1l2) fa salire la quota col crescere dello skew (il placement
insegue la domanda, plateau al cap), mentre off resta uniforme (1/6 = 16.7%)
e l2 scala il totale ma lo divide ~uniforme → resta basso. Il riferimento
tratteggiato y=16.7% = "demand-blind / uniforme".

USO:  python experiments/paper_plot_hotshare.py
      → paper/figures/figpaper_hotshare_skew.{pdf,png}   (Fig. 3 del paper NCA)
      python experiments/paper_plot_hotshare.py --dmos-arm dmos-nca
      python experiments/paper_plot_hotshare.py --dmos-arm dmos-nca --hot sg
      → run dmos-<arm>__<regime>-sg, quota sul cluster SG;
        file figpaper_hotshare_skew_sg.{pdf,png}
I nomi di arm e regimi sono quelli di experiments/run_naming.py.
"""
from __future__ import annotations
import argparse, json, glob, os, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_lib  # noqa: E402,F401  (console UTF-8 on Windows)
import run_naming  # noqa: E402

REG_SKEW = {'uni': 0.167, 'bal': 0.40, 's50': 0.50, 's65': 0.65, 's75': 0.75, 'con': 0.85}
REG_ORDER = ['uni', 'bal', 's50', 's65', 's75', 'con']
REG_LABEL = {'uni': 'Uniform', 'bal': 'Balanced', 's50': 'Skew 50%',
             's65': 'Skew 65%', 's75': 'Skew 75%', 'con': 'Concentrated'}
ARMS = [('dmos-off', 'DMOS off (static)', '#C62828'),
        ('dmos-l2', 'DMOS L2 (scaling only)', '#F57C00'),
        ('dmos-l1l2', 'DMOS L1+L2 (full)', '#1565C0')]
CL = [f'ms-cluster-0{i}' for i in range(1, 7)]
RESULTS = Path('experiments/results')


def hot_share(run: str, hot_idx: int = 0):
    """% di repliche frontend sul cluster caldo (indice nel fleet; picco
    360-600 s), o None."""
    f = os.path.join(run, 'dmos_live.jsonl')
    if not os.path.exists(f):
        return None
    rows = []
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            cl = d.get('services', {}).get('frontend', {}).get('clusters', {})
            if cl:
                rows.append((d['elapsed_s'],
                             [cl.get(c, {}).get('target_replicas', 0) for c in CL]))
    peak = [r for r in rows if 360 <= r[0] <= 600]
    if not peak:
        return None
    mean = [statistics.mean(r[1][i] for r in peak) for i in range(6)]
    tot = sum(mean)
    if tot <= 0:
        return None
    return 100.0 * mean[hot_idx] / tot


def collect(hot_suffix: str = '', hot_idx: int = 0):
    out = {}
    for arm, _, _ in ARMS:
        out[arm] = {}
        for reg in REG_ORDER:
            vals = [hot_share(d, hot_idx)
                    for d in glob.glob(str(RESULTS / f'*_{arm}__{reg}{hot_suffix}'))]
            vals = [v for v in vals if v is not None]
            # off è statico: 1 replica/cluster → 1/6 = 16.7% (il suo collector non
            # popola il frontend perché DMOS è spento → fallback al valore esatto)
            if arm == 'dmos-off' or not vals:
                if arm == 'dmos-off':
                    out[arm][reg] = (100.0 / 6.0, 0.0, 1)
                continue
            m = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            out[arm][reg] = (m, sd, len(vals))
    return out


def main():
    global RESULTS
    ap = argparse.ArgumentParser(description='Frontend replica share on the hot cluster vs demand skew',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--dmos-arm', default='dmos-l1l2',
                    help='braccio DMOS completo (default dmos-l1l2, core NCA; es. dmos-nca)')
    ap.add_argument('--hot', default=None,
                    help='cluster caldo dei regimi: de (default), fr, pl, us, sg, no')
    ap.add_argument('--results-dir', type=Path, default=RESULTS)
    ap.add_argument('--output-dir', type=Path, default=Path('paper/figures'))
    args = ap.parse_args()
    RESULTS = args.results_dir
    if args.dmos_arm != ARMS[2][0]:
        ARMS[2] = (args.dmos_arm, paper_lib.ARM_LABELS.get(args.dmos_arm, args.dmos_arm),
                   paper_lib.ARM_COLORS.get(args.dmos_arm, '#1565C0'))
    shorts = [run_naming.region_short(r) for _, r in run_naming.fleet()]
    hot = (args.hot or shorts[0]).lower()
    if hot not in shorts:
        print(f'  cluster caldo sconosciuto: {hot} (validi: {shorts})')
        return 2
    hot_idx = shorts.index(hot)
    hot_suffix = '' if hot_idx == 0 else f'-{hot}'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight',
                         'savefig.dpi': 600})
    from matplotlib.ticker import MultipleLocator
    data = collect(hot_suffix, hot_idx)
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    # Barre raggruppate: x = regimi (test isolati), una barra per arm.
    regs = list(REG_ORDER)
    x = np.arange(len(regs))
    n_arm = len(ARMS)
    width = 0.8 / max(1, n_arm)

    # riferimento: quota uniforme (demand-blind) = 1/6
    ax.axhline(100.0 / 6.0, ls=':', lw=1.2, color='gray', alpha=0.7)
    ax.text(len(regs) - 0.5, 100.0 / 6.0 + 1.2, 'uniform (1/6)', color='gray',
            fontsize=8.5, ha='right', style='italic')

    for ai, (arm, label, color) in enumerate(ARMS):
        ys, errs = [], []
        for r in regs:
            if r in data[arm]:
                ys.append(data[arm][r][0]); errs.append(data[arm][r][1])
            else:
                ys.append(np.nan); errs.append(0.0)
        offset = (ai - (n_arm - 1) / 2.0) * width
        ax.bar(x + offset, ys, width, yerr=errs, capsize=3, color=color,
               label=label, edgecolor='black', linewidth=0.4,
               error_kw={'linewidth': 0.8})

    ax.set_xlabel('Demand skew — share on hottest cluster'
                  + (f' ({hot.upper()})' if hot_suffix else ''))
    ax.set_ylabel('Frontend replicas on hottest\ncluster (% of total)')
    ax.set_xticks(x)
    ax.set_xticklabels([REG_LABEL[r] for r in regs], rotation=15, ha='right',
                       fontsize=8.5)
    ax.set_ylim(0, 60)
    ax.yaxis.set_major_locator(MultipleLocator(5))   # tick ogni 5% -> più risoluzione
    ax.tick_params(axis='y', labelsize=8.5)
    ax.grid(alpha=0.25, ls='--', axis='y')           # righe orizzontali di riferimento
    ax.set_axisbelow(True)                            # griglia dietro le barre
    ax.legend(frameon=True, fontsize=9, loc='upper left')
    fig.tight_layout()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    stem = 'figpaper_hotshare_skew' + (f'_{hot}' if hot_suffix else '')
    for ext in ('pdf', 'png'):
        p = outdir / f'{stem}.{ext}'
        fig.savefig(p)
        print(f'  ✓ {p}')
    # dump numerico
    print('\n  Numeri (mean% [n]):')
    for arm, label, _ in ARMS:
        s = '  '.join(f'{r}={data[arm][r][0]:.1f}' for r in REG_ORDER if r in data[arm])
        print(f'    {label:28s} {s}')


if __name__ == '__main__':
    sys.exit(main())
