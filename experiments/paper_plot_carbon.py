#!/usr/bin/env python3
"""
paper_plot_carbon.py — Paper Figure 9 (carbon emissions)
=========================================================

Genera la figura carbon per il paper a partire dai run packaging-ed da
`run_flash_crowd.sh`. Il calcolo e' post-hoc dai dati gia' raccolti:
- `dmos_live.jsonl` -> CPU utilization per cluster per snapshot
- `config/local.yaml` -> carbon intensity (gCO2/kWh) per cluster

Output (in --output-dir, default: paper_artifacts/<timestamp>/figures/):
  - fig9_carbon_emissions.{pdf,png}
      Bar chart: gCO2 totali per run per arm.
      Stacked per-cluster -> mostra dove vengono prodotte le emissioni.
      Error bars: deviazione standard tra ripetizioni.

Usage
-----
    python paper_plot_carbon.py
    python paper_plot_carbon.py --results-dir experiments/results --output-dir paper_artifacts/v1/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paper_lib import (
    SCENARIOS, SCENARIO_LABELS, SCENARIO_COLORS, SCENARIO_HATCHES,
    CLUSTER_COLORS, CLUSTER_REGIONS,
    ScenarioGroup, discover_runs, setup_matplotlib, save_figure,
    compute_carbon_per_run, load_cluster_carbon_config, print_group_summary,
)


def plot_fig9_carbon_emissions(groups: Dict[str, ScenarioGroup],
                               out_dir: Path,
                               formats=('pdf', 'png')) -> List[Path]:
    """
    Fig 9 — Cumulative CO2 emissions per arm, stacked per-cluster.

    Bar chart: una barra per arm, alta = somma gCO2 della run media,
    composta da segmenti colorati uno per cluster.
    Error bar verticale: deviazione standard del totale tra ripetizioni.
    """
    import matplotlib.pyplot as plt

    cfg = load_cluster_carbon_config()
    cluster_order = list(cfg.keys())   # ordine deterministico

    # Per ogni arm: matrice (n_reps x n_clusters) di gCO2
    per_arm: Dict[str, np.ndarray] = {}
    per_arm_totals: Dict[str, np.ndarray] = {}
    for scen, group in groups.items():
        rows = []
        for run in group.runs:
            res = compute_carbon_per_run(run, cluster_carbon=cfg)
            if res is None:
                continue
            row = [res['per_cluster'].get(c, {}).get('co2_g', 0.0)
                   for c in cluster_order]
            rows.append(row)
        if rows:
            per_arm[scen] = np.array(rows, dtype=float)
            per_arm_totals[scen] = per_arm[scen].sum(axis=1)

    if not per_arm:
        print("  [fig9] no carbon data -> skip")
        return []

    # Largo figura per ospitare la legenda a destra fuori dal plot
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    # Asse X: arm in ordine SCENARIOS, poi eventuali altri
    arm_names = [s for s in SCENARIOS if s in per_arm] + \
                [s for s in per_arm if s not in SCENARIOS]
    x_pos = np.arange(len(arm_names))

    # Per ogni cluster sovrappongo la sua media (un segmento dello stacked bar)
    bottoms = np.zeros(len(arm_names))
    for i, cluster in enumerate(cluster_order):
        means = np.array([per_arm[a][:, i].mean() for a in arm_names])
        color = CLUSTER_COLORS.get(cluster, f'C{i}')
        label = CLUSTER_REGIONS.get(cluster, cluster).split(' ')[0]
        ax.bar(x_pos, means, bottom=bottoms, color=color, edgecolor='black',
               linewidth=0.6, label=f'{label}  ({cfg[cluster]["gco2_per_kwh"]:.0f} gCO₂/kWh)')
        bottoms += means

    # Error bar sul totale (std tra ripetizioni)
    total_means = np.array([per_arm_totals[a].mean() for a in arm_names])
    total_stds = np.array([per_arm_totals[a].std(ddof=1) if len(per_arm_totals[a]) > 1 else 0
                          for a in arm_names])
    ax.errorbar(x_pos, total_means, yerr=total_stds, fmt='none',
                ecolor='black', capsize=5, capthick=1.5, linewidth=1.5,
                label=f'±1σ across {min(len(v) for v in per_arm.values())} reps')

    # Annota i totali sopra ogni barra + delta % vs baseline (dmos-off)
    # Il delta racconta la storia "+X% carbon overhead vs static" che e'
    # invisibile a colpo d'occhio perche' i bar sono quasi uguali.
    baseline_arm = 'dmos-off' if 'dmos-off' in per_arm else arm_names[0]
    baseline_co2 = per_arm_totals[baseline_arm].mean()
    for i, (arm, m, s) in enumerate(zip(arm_names, total_means, total_stds)):
        # Label primaria: valore in g + kg
        ax.text(i, m + s + max(total_means) * 0.025,
                f'{m:.2f} g\n({m/1000:.3f} kg)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Sub-label: delta % vs baseline (skip per il baseline stesso)
        if arm != baseline_arm and baseline_co2 > 0:
            delta_pct = 100.0 * (m - baseline_co2) / baseline_co2
            delta_color = '#2E7D32' if delta_pct < 10 else '#F57C00'  # verde se <10%
            sign = '+' if delta_pct >= 0 else ''
            ax.text(i, m + s + max(total_means) * 0.115,
                    f'{sign}{delta_pct:.1f}% vs off',
                    ha='center', va='bottom', fontsize=9,
                    color=delta_color, fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SCENARIO_LABELS.get(a, a) for a in arm_names], rotation=8)
    ax.set_ylabel("Cumulative CO₂ emissions (g) over the 15-min flash crowd")
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    # Legenda FUORI dal plot area, a destra. Evita di nascondere le label
    # dei totali sopra le barre (bug della versione precedente).
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=8, framealpha=0.9, ncol=1)
    ax.set_ylim(0, max(total_means + total_stds) * 1.30)

    ax.set_title(
        "Figure 9 — Cumulative CO₂ emissions per scheduling strategy "
        "(stacked by cluster region)\n"
        "Linear power model on node CPU; per-region grid carbon intensity from config/local.yaml",
        fontsize=11, fontweight='bold', pad=12,
    )

    # tight_layout-style padding ma rispetta bbox_to_anchor della legenda
    fig.subplots_adjust(right=0.78)

    return save_figure(fig, out_dir, 'fig9_carbon_emissions', formats)


# ── Main CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 9 (carbon emissions) for the DMOS paper.",
    )
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent / 'results')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--scenarios', nargs='+', default=None)
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

    print(f"\n  Discovering runs in: {args.results_dir}")
    groups = discover_runs(args.results_dir, scenario_filter=args.scenarios,
                           load_timeline=True)
    print_group_summary(groups)
    if not groups:
        print("  No runs found. Bye.")
        return 1

    formats = tuple(args.formats)
    print(f"  Generating Fig 9 in: {args.output_dir}\n")
    try:
        paths = plot_fig9_carbon_emissions(groups, args.output_dir, formats)
        for p in paths:
            print(f"    ✓ {p.name}  ({p.stat().st_size/1024:.1f} KB)")
    except Exception as e:
        print(f"    ✗ failed: {e}")
        import traceback; traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
