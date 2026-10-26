#!/usr/bin/env python3
"""
paper_plot_supplementary.py — Paper Figures 5-8 (supplementary)
================================================================

Genera le figure supplementari del paper. Sono importanti per:
- evidence H3 (geo-fairness): Fig 5 + Fig 8
- sanity check del workload: Fig 6
- distribuzione per-cluster: Fig 7

Output (in --output-dir, default: paper_artifacts/<timestamp>/figures/):
  - fig5_error_heatmap.{pdf,png}
       Griglia 2x2 (1 panel per scenario). Heatmap Cluster x Phase, colore = error%.
       Mostra geo-fairness: DMOS-l1l2 dovrebbe avere colori uniformi tra cluster.
  - fig6_throughput_achieved.{pdf,png}
       Singolo grafico, RPS misurato vs target. Bande di fasi. Verifica che k6
       sta mandando il workload previsto.
  - fig7_boxplot_p95_per_cluster.{pdf,png}
       Box plot: per ciascun cluster, distribuzione p95 negli scenari.
       Mostra dispersione tra cluster.
  - fig8_geo_fairness.{pdf,png}
       Per ogni scenario: varianza error rate tra cluster nelle 3 fasi
       chiave (spike, peak, decay). Lower = more fair.

Usage
-----
    python paper_plot_supplementary.py
    python paper_plot_supplementary.py --results-dir experiments/results \\
                                       --output-dir paper_artifacts/v1/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from paper_lib import (
    SCENARIOS, SCENARIO_LABELS, SCENARIO_COLORS, SCENARIO_HATCHES,
    CLUSTER_COLORS, CLUSTER_REGIONS, PHASE_DEFS, PHASES,
    ScenarioGroup, discover_runs, setup_matplotlib, shade_phases, save_figure,
    bin_timeline, BIN_SEC_DEFAULT, print_group_summary,
    compute_cluster_phase_stats,
)

# Soglia SLO usata per Fig 5/7/8 (coerente con Fig 3/3b nel main set)
SLO_THRESHOLD_MS = 1000.0


# ── Fig 5: SLO violation heatmap (Cluster × Phase) per scenario ─────────────

def plot_fig5_error_heatmap(groups: Dict[str, ScenarioGroup],
                            out_dir: Path,
                            formats=('pdf', 'png'),
                            slo_ms: float = SLO_THRESHOLD_MS) -> List[Path]:
    """
    Heatmap Cluster × Phase di SLO violation rate (% requests > slo_ms),
    una per arm. Sostituisce la vecchia "error rate heatmap": l'errore
    rate e' 0% in tutti gli arm (k6 timeout=60s, nessun service crash)
    quindi quella figura non discriminava nulla. Le violazioni SLO sono
    il vero proxy della qualita' percepita.

    Aspetto atteso: cluster-01 (EU-DE, dove il flash crowd e' diretto)
    rosso fuoco per dmos-off/L2 durante spike+peak; blu pallido per
    dmos-L1+L2.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    n_scenarios = len(groups)
    if n_scenarios == 0:
        return []

    # Phase order: rispetta PHASE_DEFS
    phase_order = [p[2] for p in PHASE_DEFS]

    # Step 1: per ogni (scenario, cluster, phase) costruisci mean(pct_violations) sui rep
    arm_matrix: Dict[str, pd.DataFrame] = {}
    all_clusters: set = set()
    for scenario, group in groups.items():
        per_run_mats: List[pd.DataFrame] = []
        for run in group.runs:
            stats = compute_cluster_phase_stats(run.run_dir, slo_ms=slo_ms)
            cells = stats.get('cells', {})
            if not cells:
                continue
            tmp: Dict[str, Dict[str, float]] = {}
            for _, c in cells.items():
                cl = c['cluster']
                ph = c['phase']
                if ph not in phase_order:
                    continue
                tmp.setdefault(cl, {})[ph] = c['pct_violations']
            if not tmp:
                continue
            df = pd.DataFrame(tmp).T   # rows=cluster, cols=phase
            df = df.reindex(columns=phase_order, fill_value=np.nan)
            per_run_mats.append(df)
        if not per_run_mats:
            continue
        all_idx = sorted(set().union(*(m.index for m in per_run_mats)))
        all_col = phase_order
        aligned = [m.reindex(index=all_idx, columns=all_col) for m in per_run_mats]
        arm_matrix[scenario] = pd.concat(aligned).groupby(level=0).mean()
        arm_matrix[scenario] = arm_matrix[scenario].reindex(columns=phase_order)
        all_clusters.update(all_idx)

    if not arm_matrix:
        print("  [fig5] no SLO data → skip")
        return []

    cluster_order = sorted(all_clusters)
    for sc in list(arm_matrix.keys()):
        arm_matrix[sc] = arm_matrix[sc].reindex(index=cluster_order,
                                                columns=phase_order)

    # Colormap green→yellow→orange→red. vmax dal 95° percentile globale
    cmap = LinearSegmentedColormap.from_list(
        'slo_traffic', ['#2E7D32', '#FFEB3B', '#F57C00', '#C62828'], N=256)
    all_vals = np.concatenate([m.values[~np.isnan(m.values)].flatten()
                               for m in arm_matrix.values()])
    vmax = max(5.0, float(np.nanpercentile(all_vals, 98))) if all_vals.size else 50.0

    n = len(arm_matrix)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n + 0.6, 4.5),
                             squeeze=False, sharey=True)

    im = None
    panel_idx = 0
    for scenario, mat in arm_matrix.items():
        ax = axes[0, panel_idx]
        vals = mat.values
        im = ax.imshow(vals, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
        # Annotazioni
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                txt_color = 'white' if v > vmax * 0.55 else 'black'
                ax.text(j, i, f"{v:.1f}", ha='center', va='center',
                        color=txt_color, fontsize=9)
        ax.set_xticks(range(len(phase_order)))
        ax.set_xticklabels([p.capitalize() for p in phase_order],
                           rotation=30, ha='right', fontsize=9)
        if panel_idx == 0:
            ax.set_yticks(range(len(cluster_order)))
            ax.set_yticklabels([CLUSTER_REGIONS.get(c, c).split(' ')[0]
                                for c in cluster_order], fontsize=9)
            ax.set_ylabel("Cluster (ingress region)")
        else:
            ax.set_yticks(range(len(cluster_order)))
            ax.set_yticklabels([])
        ax.set_title(
            f"{SCENARIO_LABELS.get(scenario, scenario)}  (n={groups[scenario].n_repetitions})",
            fontsize=11, fontweight='bold',
            color=SCENARIO_COLORS.get(scenario, 'black'))
        panel_idx += 1

    fig.subplots_adjust(right=0.91, bottom=0.18, top=0.84)
    cbar_ax = fig.add_axes([0.93, 0.18, 0.012, 0.65])
    fig.colorbar(im, cax=cbar_ax,
                 label=f'SLO violations (% requests > {int(slo_ms)} ms)')

    fig.suptitle(
        f"Figure 5 — SLO violation rate by Cluster $\\times$ Phase "
        f"(SLO = {int(slo_ms)} ms)\n"
        "(uniform colour rows = good geo-fairness; "
        "hot rows = saturated ingress)",
        fontsize=12, fontweight='bold', y=0.99,
    )

    return save_figure(fig, out_dir, 'fig5_slo_heatmap', formats)


# ── Fig 6: throughput achieved vs target ────────────────────────────────────

def plot_fig6_throughput_achieved(groups: Dict[str, ScenarioGroup],
                                  out_dir: Path,
                                  formats=('pdf', 'png')) -> List[Path]:
    """
    Verifica che il workload generato corrisponda al target (5 phases).

    Plot: 1 grafico, 4 linee scenario, ognuna mostra rps achieved. Linea
    tratteggiata = target ideale.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Target ideale: ricostruito dalla config del manifest del primo run trovato
    sample_run = None
    for group in groups.values():
        if group.runs:
            sample_run = group.runs[0]
            break

    if sample_run and sample_run.manifest:
        cfg = sample_run.manifest.get('config', {})
        # Default coerente con run_flash_crowd.sh corrente (250 rps peak)
        baseline_rps = cfg.get('baseline_rps', 50)
        peak_rps = cfg.get('peak_rps', 250)
        # Trapezoidal shape: ramp during spike, hold during peak, ramp during decay.
        # k6 puo' produrre leggermente di piu' a regime per via del per-VU pacing
        # (overshoot ~10-20%), quindi mostriamo "configured target" non "expected".
        target_x = [0, 180, 300, 600, 720, 900]
        target_y = [baseline_rps, baseline_rps, peak_rps, peak_rps,
                    baseline_rps, baseline_rps]
        ax.plot(np.array(target_x) / 60, target_y,
                'k--', alpha=0.5, linewidth=1.5,
                label=f'Configured target ({peak_rps} rps peak)')

    # Per ogni scenario, rps achieved da k6_events.json (bin 10s)
    for scenario, group in groups.items():
        from paper_plot_trajectories import _load_k6_events
        all_events: List[pd.DataFrame] = []
        for run in group.runs:
            evp = run.run_dir / 'k6_events.json'
            if evp.is_file():
                df = _load_k6_events(evp)
                if df is not None and not df.empty:
                    all_events.append(df)
        if not all_events:
            continue
        df = pd.concat(all_events, ignore_index=True)
        df['bin_s'] = (df['elapsed_s'] // 10 * 10).astype(int)
        rps = df.groupby('bin_s').size() / (10 * group.n_repetitions)
        ax.plot(rps.index / 60.0, rps.values,
                color=group.color, label=group.label,
                linewidth=1.6, alpha=0.85)

    shade_phases(ax, x_unit='min', y_frac_label=0.97)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Throughput (rps)")
    ax.set_xlim(0, 15)
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_title(
        "Figure 6 — Sanity check: all strategies sustain the same offered load\n"
        "(curves overlap by design — the latency gap in Figures 2–4 is "
        "at iso-throughput, NOT throughput shedding)",
        fontsize=11, fontweight='bold', pad=12,
    )

    return save_figure(fig, out_dir, 'fig6_throughput_achieved', formats)


# ── Fig 7: boxplot p95 per cluster ──────────────────────────────────────────

def plot_fig7_boxplot_p95_per_cluster(groups: Dict[str, ScenarioGroup],
                                      out_dir: Path,
                                      formats=('pdf', 'png'),
                                      phase: str = 'peak',
                                      slo_ms: float = SLO_THRESHOLD_MS) -> List[Path]:
    """
    Box plot p95 per cluster RESTRINGENDO ALLA FASE PEAK.

    La versione precedente usava `results.per_cluster` di k6_summary, che
    e' aggregato sul *tutto il run* (15 min) — diluiva il segnale di peak
    con baseline/cooldown e i 3 arm sembravano indistinguibili. Qui invece
    leggiamo p95 per (cluster, phase) dalle cache `cluster_phase_stats.json`.

    Aggiungiamo anche una linea orizzontale tratteggiata allo SLO target.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # data[cluster][scenario] = list of p95 (uno per run) nella phase scelta
    data: Dict[str, Dict[str, List[float]]] = {}
    for scenario, group in groups.items():
        for run in group.runs:
            stats = compute_cluster_phase_stats(run.run_dir, slo_ms=slo_ms)
            cells = stats.get('cells', {})
            for _, c in cells.items():
                if c['phase'] != phase:
                    continue
                data.setdefault(c['cluster'], {}).setdefault(scenario, []).append(c['p95'])

    if not data:
        print(f"  [fig7] no per-cluster peak-phase data → skip")
        return []

    clusters_sorted = sorted(data.keys())
    n_clusters = len(clusters_sorted)
    n_scenarios = max(1, len([s for s in SCENARIOS if s in groups]))

    fig, ax = plt.subplots(figsize=(2 * n_clusters + 2, 6))

    width = 0.85 / n_scenarios
    for i, cluster in enumerate(clusters_sorted):
        for j, scenario in enumerate(SCENARIOS):
            if scenario not in groups:
                continue
            vals = data.get(cluster, {}).get(scenario, [])
            if not vals:
                continue
            pos = i + j * width - 0.425 + width / 2
            box = ax.boxplot(vals, positions=[pos], widths=width * 0.85,
                             patch_artist=True, showfliers=False)
            for patch in box['boxes']:
                patch.set_facecolor(SCENARIO_COLORS.get(scenario, '#888'))
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')

    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels([CLUSTER_REGIONS.get(c, c).split(' ')[0]
                        for c in clusters_sorted], rotation=15, ha='right')
    ax.set_ylabel("p95 Latency in peak phase (ms)")
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    # SLO reference line
    ax.axhline(slo_ms, color='gray', linestyle=':', linewidth=1.2, alpha=0.8)
    ax.text(n_clusters - 0.5, slo_ms * 1.08, f'SLO {int(slo_ms)} ms',
            fontsize=8, color='gray', ha='right', va='bottom')

    handles = [Patch(facecolor=SCENARIO_COLORS[s], edgecolor='black',
                     label=SCENARIO_LABELS[s], alpha=0.7)
               for s in SCENARIOS if s in groups]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)

    # Annotazione "Nx gap" sul cluster del traffico (EU-DE / cluster-01): le
    # box di dmos-off e dmos-l2 si sovrappongono (entrambe sature), L1+L2 e'
    # ~10x sotto. Senza annotazione il messaggio non emerge a colpo d'occhio.
    try:
        # Trova cluster-01 / EU-DE per indice
        target_cluster = None
        for tc_name in ('ms-cluster-01', 'cluster-01'):
            if tc_name in clusters_sorted:
                target_cluster = tc_name
                break
        if target_cluster is None and clusters_sorted:
            # Fallback: cluster col p95 piu' alto in dmos-off
            best_p95 = -1
            for c in clusters_sorted:
                v = data.get(c, {}).get('dmos-off', [])
                if v and float(np.mean(v)) > best_p95:
                    best_p95 = float(np.mean(v))
                    target_cluster = c
        if target_cluster is not None:
            tc_idx = clusters_sorted.index(target_cluster)
            off_v = data.get(target_cluster, {}).get('dmos-off', [])
            l1l2_v = data.get(target_cluster, {}).get('dmos-l1l2', [])
            if off_v and l1l2_v:
                off_mean = float(np.mean(off_v))
                l1l2_mean = float(np.mean(l1l2_v))
                if l1l2_mean > 0:
                    gap = off_mean / l1l2_mean
                    # Posiziona il testo sopra il box di L1+L2 (blu), con freccia verso off (rosso)
                    ax.annotate(
                        f"  ~{gap:.0f}× gap\n  (geo-aware\n   placement)",
                        xy=(tc_idx + 0.15, off_mean * 0.85),
                        xytext=(tc_idx + 0.6, np.sqrt(off_mean * l1l2_mean)),
                        fontsize=9, color='#1565C0', fontweight='bold',
                        ha='left', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD',
                                  ec='#1565C0', linewidth=0.8),
                        arrowprops=dict(arrowstyle='<->',
                                        color='#1565C0', lw=1.4),
                    )
    except Exception:
        pass

    ax.set_title(
        f"Figure 7 — p95 latency per cluster during {phase} phase\n"
        f"(boxes show median + IQR across $n$ repetitions; "
        f"dashed line = {int(slo_ms)}ms SLO; the gap on the ingress cluster "
        f"is where L1's geo-aware placement matters)",
        fontsize=11, fontweight='bold', pad=12,
    )

    return save_figure(fig, out_dir, 'fig7_boxplot_p95_per_cluster', formats)


# ── Fig 8: geo-fairness (variance error rate across clusters) ───────────────

def plot_fig8_geo_fairness(groups: Dict[str, ScenarioGroup],
                           out_dir: Path,
                           formats=('pdf', 'png'),
                           phase: str = 'peak',
                           slo_ms: float = SLO_THRESHOLD_MS) -> List[Path]:
    """
    Bar chart: per ogni arm, varianza della SLO violation rate (>slo_ms)
    fra cluster, nella phase scelta.

    Riformulazione: la vecchia versione usava Var(error_rate)≈0 ovunque
    (errors=0% in tutti gli arm) — bar tutti a zero, nessuna informazione.
    SLO violations DISTRIBUISCONO il segnale tra cluster: dmos-off
    concentra tutto su cluster-01 → varianza alta → low geo-fairness.
    dmos-L1+L2 spalma il carico → varianza bassa → high geo-fairness.

    Asse Y: variance in unita' [(% requests)²] — interpretabile.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_x: List[str] = []
    bars_y: List[float] = []
    bars_err: List[float] = []
    bars_color: List[str] = []
    bars_hatch: List[str] = []
    bars_n: List[int] = []

    for scenario in SCENARIOS:
        if scenario not in groups:
            continue
        group = groups[scenario]
        run_variances: List[float] = []
        for run in group.runs:
            stats = compute_cluster_phase_stats(run.run_dir, slo_ms=slo_ms)
            cells = stats.get('cells', {})
            # raccogli pct_violations per ogni cluster (filtrando alla phase)
            per_cluster_pcts = [c['pct_violations'] for _, c in cells.items()
                                if c['phase'] == phase]
            if len(per_cluster_pcts) >= 2:
                run_variances.append(float(np.var(per_cluster_pcts, ddof=1)))
        if not run_variances:
            continue
        bars_x.append(SCENARIO_LABELS.get(scenario, scenario))
        bars_y.append(float(np.mean(run_variances)))
        bars_err.append(float(np.std(run_variances, ddof=1))
                        if len(run_variances) > 1 else 0.0)
        bars_color.append(SCENARIO_COLORS.get(scenario, '#888'))
        bars_hatch.append(SCENARIO_HATCHES.get(scenario, ''))
        bars_n.append(len(run_variances))

    if not bars_x:
        print("  [fig8] no per-cluster SLO data → skip")
        return []

    x_pos = np.arange(len(bars_x))
    bars = ax.bar(x_pos, bars_y, yerr=bars_err,
                  color=bars_color, edgecolor='black', linewidth=1.2,
                  capsize=5, alpha=0.85)
    for bar, h in zip(bars, bars_hatch):
        bar.set_hatch(h)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bars_x, rotation=10, ha='right')
    ax.set_ylabel(f"Var(SLO violation %) across clusters [(% req)²]")
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # Annota valori sopra le barre
    ymax = max((b + e for b, e in zip(bars_y, bars_err)), default=1.0) * 1.18
    ax.set_ylim(0, ymax)
    for bar, val, err in zip(bars, bars_y, bars_err):
        ax.text(bar.get_x() + bar.get_width() / 2, val + err + ymax * 0.02,
                f"{val:.1f}", ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.set_title(
        f"Figure 8 — Geo-fairness: variance of SLO violations across clusters "
        f"during {phase} phase\n"
        f"(SLO = {int(slo_ms)} ms; lower bar = more geo-fair allocation; "
        f"error bars = ±1σ across reps)",
        fontsize=11, fontweight='bold', pad=12,
    )

    return save_figure(fig, out_dir, 'fig8_geo_fairness', formats)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera Figure 5-8 supplementari del paper DMOS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        return 1

    formats = tuple(args.formats)
    print(f"  Generating supplementary figures in: {args.output_dir}\n")
    written = []
    for name, fn in [
        ("Fig 5 (error heatmap)",           plot_fig5_error_heatmap),
        ("Fig 6 (throughput achieved)",     plot_fig6_throughput_achieved),
        ("Fig 7 (boxplot p95/cluster)",     plot_fig7_boxplot_p95_per_cluster),
        ("Fig 8 (geo-fairness)",            plot_fig8_geo_fairness),
    ]:
        print(f"  {name}...")
        try:
            paths = fn(groups, args.output_dir, formats)
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
