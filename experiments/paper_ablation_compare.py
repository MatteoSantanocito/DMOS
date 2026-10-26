#!/usr/bin/env python3
"""
paper_ablation_compare.py — Objective-ablation comparison (E3 / Study B)
========================================================================

PERCHE' QUESTO SCRIPT
---------------------
`paper_stat_compare.py` confronta i 3 arm di SCHEDULING ({off, l2, l1l2}) e
itera su una lista SCENARIOS hardcoded: ignora qualsiasi arm fuori da quella
lista. Lo Study B confronta invece PROFILI DI PESI a parita' di workload e di
architettura (tutti full L1+L2), per dimostrare che la combinazione
multi-obiettivo di DMOS domina ciascun obiettivo singolo:

  default (E3 del piano di test: modelli di score a gate g=2, weights.yaml)
    dmos-latency_first    obiettivo dominante: latenza
    dmos-capacity_first   obiettivo dominante: capacita'
    dmos-carbon_first     obiettivo dominante: carbonio
    dmos-nca              pesi del paper NCA (multi-obiettivo, baseline)
  campagna precedente (Study B, profili legacy; con --arms)
    dmos-profile-latonly  omega_latency = 1   ≈ LAIS (latency-aware)
    dmos-profile-netonly  omega_geo_load = 1  ≈ NetMARKS / Diktyo (locality-aware)
    dmos-profile-carbon   omega_carbon = 1    ≈ CASPER (carbon-aware)
    (archivi: dmos-latonly / dmos-netonly / dmos-carbon / dmos-l1l2)

Tutti girano sotto lo stesso regime di domanda (default: __bal) e la stessa
intensita' di carico. L'unica variabile e' il profilo di pesi. Nomi degli
arm: experiments/run_naming.py.

La narrativa attesa (da verificare coi dati): ogni baseline single-objective
vince la SUA metrica ma paga le altre (latonly: miglior p95 ma piu' CO2;
carbon: miglior CO2 ma peggior coda di latenza); il profilo multi-obiettivo
ottiene il miglior trade-off complessivo.

OUTPUT (in --output-dir, default paper_artifacts/<ts>/ablation/)
  - ablation_metrics.tex       Tabella LaTeX booktabs: una riga per profilo,
                               colonne p50/p95/p99/SLO/throughput/replicas/CO2,
                               mean [CI95]. La cella migliore per colonna e' in
                               grassetto.
  - ablation_stat_tests.tex    Welch t-test + Cohen's d di ogni single-objective
                               vs il profilo multi-obiettivo (baseline).
  - ablation.md                Versione markdown.
  - ablation_data.json         Dump numeri raw.
  - fig_ablation_tradeoff.{pdf,png}   Bar chart comparativo (p95 + SLO%).

USO
---
  # prerequisito: lanciare prima paper_slo_compliance.py per popolare le cache
  python experiments/paper_ablation_compare.py
  python experiments/paper_ablation_compare.py --regime con
  python experiments/paper_ablation_compare.py \
      --arms dmos-profile-latonly__bal dmos-profile-netonly__bal \
             dmos-profile-carbon__bal dmos-l1l2__bal \
      --baseline dmos-l1l2__bal
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
    bootstrap_ci, welch_ttest, cohen_d, cohen_d_label,
    scenario_label, scenario_color, scenario_hatch,
    parse_arm_regime, carbon_per_arm, setup_matplotlib, save_figure,
)
# Riusiamo i helper gia' scritti per Table 1/2 (SLO da cache, replicas, METRICS).
from paper_stat_compare import (  # noqa: E402
    METRICS, SLO_THRESHOLD_MS,
    slo_violations_per_arm, avg_replicas_per_scenario,
    _fmt_metric_latex, _fmt_metric_md, _fmt_cohen_d, _latex_escape,
)

# Default E3: profili di pesi (modelli di score) contro i pesi del paper,
# tutti sotto regime "balanced".
DEFAULT_ARMS = [
    'dmos-latency_first__bal',
    'dmos-capacity_first__bal',
    'dmos-carbon_first__bal',
    'dmos-nca__bal',
]
DEFAULT_BASELINE = 'dmos-nca__bal'    # multi-obiettivo = arm di riferimento


# ── Aggregazione metriche per arm ───────────────────────────────────────────

def build_rows(groups: Dict[str, ScenarioGroup],
               arms_order: List[str]) -> Dict:
    """
    Per ogni arm calcola mean+CI95 di p50/p95/p99 (peak), SLO viol, throughput,
    avg replicas (peak), CO2 totale. Stessa struttura di paper_stat_compare
    ma iterando sugli arm REALMENTE scoperti (non sulla lista hardcoded).
    """
    repl_data = avg_replicas_per_scenario(groups, phase='peak')
    carbon_data = carbon_per_arm(groups)
    slo_data = slo_violations_per_arm(groups, SLO_THRESHOLD_MS, phase='peak')

    rows: Dict[str, dict] = {}
    for arm in arms_order:
        group = groups.get(arm)
        if group is None:
            continue
        row = {
            'scenario': arm,
            'label':    scenario_label(arm),
            'n_reps':   group.n_repetitions,
            'metrics':  {},
        }
        for key_path, name, unit, fmt_kind, *_ in METRICS:
            vals = group.metric_values_per_run(key_path)
            if len(vals) > 0:
                mean, low, high = bootstrap_ci(vals)
                row['metrics'][key_path] = {
                    'name': name, 'unit': unit, 'fmt': fmt_kind,
                    'mean': mean, 'ci_low': low, 'ci_high': high,
                    'raw': vals.tolist(),
                }
        if arm in slo_data:
            vals = slo_data[arm]
            mean, low, high = bootstrap_ci(vals)
            row['metrics']['slo_viol_1s'] = {
                'name': f'SLO viol. >{int(SLO_THRESHOLD_MS/1000)}s (%)',
                'unit': '%', 'fmt': 'slo_pct',
                'mean': mean, 'ci_low': low, 'ci_high': high,
                'raw': vals.tolist(),
            }
        if arm in repl_data:
            vals = repl_data[arm]
            mean, low, high = bootstrap_ci(vals)
            row['metrics']['avg_replicas_peak'] = {
                'name': 'Avg replicas (peak)', 'unit': '', 'fmt': 'rep',
                'mean': mean, 'ci_low': low, 'ci_high': high,
                'raw': vals.tolist(),
            }
        if arm in carbon_data:
            vals = carbon_data[arm]
            mean, low, high = bootstrap_ci(vals)
            row['metrics']['carbon_total_g'] = {
                'name': 'Total CO₂ (g)', 'unit': 'g', 'fmt': 'co2',
                'mean': mean, 'ci_low': low, 'ci_high': high,
                'raw': vals.tolist(),
            }
        rows[arm] = row

    metric_order = [
        'results.per_phase.peak.p50',
        'results.per_phase.peak.p95',
        'results.per_phase.peak.p99',
        'slo_viol_1s',
        'results.global.avg_rps',
        'avg_replicas_peak',
        'carbon_total_g',
    ]
    # lower_is_better per evidenziare la cella migliore in ogni colonna
    lower_better = {
        'results.per_phase.peak.p50': True,
        'results.per_phase.peak.p95': True,
        'results.per_phase.peak.p99': True,
        'slo_viol_1s': True,
        'results.global.avg_rps': False,
        'avg_replicas_peak': True,
        'carbon_total_g': True,
    }
    return {'rows': rows, 'metric_order': metric_order,
            'lower_better': lower_better, 'arms_order': arms_order}


def _best_arm_per_metric(table: Dict) -> Dict[str, str]:
    """Per ogni metrica, ritorna l'arm con il valore migliore (per bold)."""
    rows = table['rows']
    best: Dict[str, str] = {}
    for key_path in table['metric_order']:
        cand = [(arm, r['metrics'][key_path]['mean'])
                for arm, r in rows.items() if key_path in r['metrics']]
        if not cand:
            continue
        lb = table['lower_better'].get(key_path, True)
        best[key_path] = (min if lb else max)(cand, key=lambda kv: kv[1])[0]
    return best


# ── Render Table (metrics) ──────────────────────────────────────────────────

def render_metrics_latex(table: Dict) -> str:
    rows = table['rows']
    metric_order = table['metric_order']
    best = _best_arm_per_metric(table)

    any_row = next(iter(rows.values()), None)
    if any_row is None:
        return "% ablation table empty (no runs discovered)"
    metric_names = [any_row['metrics'][k]['name'] if k in any_row['metrics']
                    else k.split('.')[-1] for k in metric_order]
    col_spec = 'l' + 'c' * len(metric_names)

    lines = [
        "% Study B — Objective ablation: single-objective vs multi-objective",
        "% Generated by paper_ablation_compare.py",
        "\\begin{table*}[!t]", "\\centering",
        ("\\caption{Objective ablation under an identical demand regime "
         "(balanced flash crowd). All arms use the full L1+L2 architecture; "
         "the only variable is the scoring weight profile. Single-objective "
         "profiles emulate latency-aware (LAIS), locality-aware "
         "(NetMARKS/Diktyo) and carbon-aware (CASPER) schedulers. Each cell is "
         "the mean over $n$ repetitions with bootstrap 95\\% CI; the best value "
         "per column is in \\textbf{bold}.}"),
        "\\label{tab:ablation-objectives}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Weight profile & " + " & ".join(metric_names) + " \\\\",
        "\\midrule",
    ]
    for arm in table['arms_order']:
        if arm not in rows:
            continue
        r = rows[arm]
        cells = [_latex_escape(r['label']) + f" ($n={r['n_reps']}$)"]
        for key_path in metric_order:
            m = r['metrics'].get(key_path)
            if m is None:
                cells.append("--")
                continue
            txt = _fmt_metric_latex(m)
            if best.get(key_path) == arm:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def render_metrics_markdown(table: Dict) -> str:
    rows = table['rows']
    metric_order = table['metric_order']
    best = _best_arm_per_metric(table)
    any_row = next(iter(rows.values()), None)
    if any_row is None:
        return "### Study B — (no runs discovered)\n"
    metric_names = [any_row['metrics'][k]['name'] if k in any_row['metrics']
                    else k.split('.')[-1] for k in metric_order]
    lines = ["### Study B — Objective ablation (peak phase, mean [CI95]; **bold** = best)\n"]
    lines.append("| Weight profile | n | " + " | ".join(metric_names) + " |")
    lines.append("|" + "---|" * (len(metric_names) + 2))
    for arm in table['arms_order']:
        if arm not in rows:
            continue
        r = rows[arm]
        cells = [r['label'], str(r['n_reps'])]
        for key_path in metric_order:
            m = r['metrics'].get(key_path)
            if m is None:
                cells.append("--")
                continue
            txt = _fmt_metric_md(m)
            if best.get(key_path) == arm:
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ── Pairwise stat tests vs multi-objective baseline ─────────────────────────

def build_stat_tests(groups: Dict[str, ScenarioGroup],
                     arms_order: List[str],
                     baseline: str) -> List[Dict]:
    if baseline not in groups:
        baseline = next((a for a in arms_order if a in groups), None)
    if baseline is None:
        return []
    base = groups[baseline]
    slo_data = slo_violations_per_arm(groups, SLO_THRESHOLD_MS, phase='peak')

    pairs: List[Dict] = []
    for arm in arms_order:
        if arm == baseline or arm not in groups:
            continue
        group = groups[arm]
        for key_path, name, unit, fmt_kind, lower_better, in_t2 in METRICS:
            if not in_t2:
                continue
            a = group.metric_values_per_run(key_path)
            b = base.metric_values_per_run(key_path)
            if len(a) < 2 or len(b) < 2:
                continue
            t, p = welch_ttest(a, b)
            d = cohen_d(a, b)
            # Delta = quanto il baseline (multi-obj) e' MIGLIORE dell'arm
            # single-objective (positivo = baseline migliore). Per metriche
            # lower-is-better cio' significa (arm - base); si inverte per le
            # higher-is-better.
            improvement = (a.mean() - b.mean()) / b.mean() * 100 if b.mean() else 0
            if not lower_better:
                improvement = -improvement
            pairs.append({
                'baseline': baseline, 'scenario': arm, 'metric': name, 'unit': unit,
                'mean_arm': float(a.mean()), 'mean_base': float(b.mean()),
                'improvement_pct': improvement,
                't_stat': t, 'p_value': p, 'cohen_d': d,
                'magnitude': cohen_d_label(d),
                'significant': (p < 0.05) if not np.isnan(p) else False,
            })
        if arm in slo_data and baseline in slo_data:
            a, b = slo_data[arm], slo_data[baseline]
            if len(a) >= 2 and len(b) >= 2:
                t, p = welch_ttest(a, b)
                d = cohen_d(a, b)
                # SLO viol e' lower-is-better: (arm - base) → positivo se il
                # baseline multi-obj viola meno (cioe' e' migliore).
                improvement = (a.mean() - b.mean()) / b.mean() * 100 if b.mean() else 0
                pairs.append({
                    'baseline': baseline, 'scenario': arm,
                    'metric': f'SLO viol. >{int(SLO_THRESHOLD_MS/1000)}s (%)',
                    'unit': '%', 'mean_arm': float(a.mean()), 'mean_base': float(b.mean()),
                    'improvement_pct': improvement,
                    't_stat': t, 'p_value': p, 'cohen_d': d,
                    'magnitude': cohen_d_label(d),
                    'significant': (p < 0.05) if not np.isnan(p) else False,
                })
    return pairs


def render_stat_tests_latex(pairs: List[Dict], arms_order: List[str]) -> str:
    if not pairs:
        return "% Study B stat tests empty"
    base_label = scenario_label(pairs[0]['baseline'])
    lines = [
        "% Study B — Pairwise Welch t-test + Cohen's d vs multi-objective",
        "% Generated by paper_ablation_compare.py",
        "\\begin{table*}[!t]", "\\centering",
        ("\\caption{Each single-objective profile compared against the "
         f"multi-objective DMOS profile (\\emph{{{_latex_escape(base_label)}}}). "
         "\\enquote{$\\Delta$ vs multi-obj} is the relative degradation of the "
         "single-objective arm w.r.t.\\ the multi-objective baseline "
         "(positive = baseline is better). Welch's $t$-test ($p<0.05$ marked "
         "*); Cohen's $d$ effect size.}"),
        "\\label{tab:ablation-stat-tests}",
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        ("Profile & Metric & Profile & Multi-obj & $\\Delta$ vs multi-obj & "
         "$p$-value & Cohen's $d$ \\\\"),
        "\\midrule",
    ]
    by_arm: Dict[str, List[Dict]] = {}
    for p in pairs:
        by_arm.setdefault(p['scenario'], []).append(p)
    for arm in arms_order:
        if arm not in by_arm:
            continue
        for i, p in enumerate(by_arm[arm]):
            cells = []
            cells.append(f"\\textbf{{{_latex_escape(scenario_label(arm))}}}" if i == 0 else "")
            cells.append(p['metric'])
            cells.append(f"{p['mean_arm']:.1f}")
            cells.append(f"{p['mean_base']:.1f}")
            sign = '+' if p['improvement_pct'] > 0 else ''
            cells.append(f"{sign}{p['improvement_pct']:.1f}\\%")
            if np.isnan(p['p_value']):
                cells.append("n/a")
            else:
                star = '$^*$' if p['significant'] else ''
                cells.append(f"{p['p_value']:.3g}{star}")
            cells.append(_fmt_cohen_d(p['cohen_d'], p['magnitude'])
                         if not np.isnan(p['cohen_d']) else "n/a")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def render_stat_tests_markdown(pairs: List[Dict], arms_order: List[str]) -> str:
    if not pairs:
        return "### Study B — stat tests (no comparisons available)\n"
    base_label = scenario_label(pairs[0]['baseline'])
    lines = [f"### Study B — Pairwise tests vs multi-objective `{base_label}`\n"]
    lines.append("| Profile | Metric | Profile | Multi-obj | Δ vs multi-obj | p-value | Cohen's d |")
    lines.append("|---|---|---|---|---|---|---|")
    by_arm: Dict[str, List[Dict]] = {}
    for p in pairs:
        by_arm.setdefault(p['scenario'], []).append(p)
    for arm in arms_order:
        if arm not in by_arm:
            continue
        for i, p in enumerate(by_arm[arm]):
            label = f"**{scenario_label(arm)}**" if i == 0 else ""
            pv = "n/a" if np.isnan(p['p_value']) else f"{p['p_value']:.3g}"
            if not np.isnan(p['p_value']) and p['significant']:
                pv += " *"
            d = "n/a" if np.isnan(p['cohen_d']) else _fmt_cohen_d(p['cohen_d'], p['magnitude'])
            sign = '+' if p['improvement_pct'] > 0 else ''
            lines.append(
                f"| {label} | {p['metric']} | {p['mean_arm']:.1f} | {p['mean_base']:.1f} | "
                f"{sign}{p['improvement_pct']:.1f}% | {pv} | {d} |"
            )
    lines.append("\n*pvalue < 0.05 marked with `*`.*\n")
    return "\n".join(lines)


# ── Trade-off figure (p95 + SLO%) ───────────────────────────────────────────

def plot_tradeoff(table: Dict, out_dir: Path) -> Optional[List[Path]]:
    import matplotlib.pyplot as plt
    rows = table['rows']
    arms = [a for a in table['arms_order'] if a in rows]
    if not arms:
        return None

    p95_key = 'results.per_phase.peak.p95'
    slo_key = 'slo_viol_1s'

    def series(key):
        means, los, his = [], [], []
        for a in arms:
            m = rows[a]['metrics'].get(key)
            if m is None:
                means.append(np.nan); los.append(0); his.append(0)
            else:
                means.append(m['mean'])
                los.append(m['mean'] - m['ci_low'])
                his.append(m['ci_high'] - m['mean'])
        return np.array(means), np.array([los, his])

    p95_m, p95_err = series(p95_key)
    slo_m, slo_err = series(slo_key)

    labels = [scenario_label(a).split(' — ')[0] for a in arms]
    colors = [scenario_color(a) for a in arms]
    hatches = [scenario_hatch(a) for a in arms]
    x = np.arange(len(arms))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (m, err, title, ylab) in zip(
            (ax1, ax2),
            [(p95_m, p95_err, 'Tail latency (peak)', 'p95 latency (ms)'),
             (slo_m, slo_err, 'SLO violations (peak)', f'requests > {int(SLO_THRESHOLD_MS)} ms (%)')]):
        bars = ax.bar(x, m, yerr=err, capsize=4, color=colors,
                      edgecolor='black', linewidth=0.6)
        for bar, h in zip(bars, hatches):
            if h:
                bar.set_hatch(h)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    fig.suptitle('Objective ablation: single-objective baselines vs multi-objective DMOS',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return save_figure(fig, out_dir, 'fig_ablation_tradeoff')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Study B — objective ablation comparison for the DMOS paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--results-dir', type=Path, default=SCRIPT_DIR / 'results')
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--arms', nargs='+', default=None,
                    help="Etichette arm da confrontare (default: profili E3 sotto __bal)")
    ap.add_argument('--regime', default=None,
                    help="Se --arms non e' dato, costruisce gli arm di default con questo regime "
                         "(es. --regime con → dmos-latency_first__con ...)")
    ap.add_argument('--baseline', default=None,
                    help="Arm multi-obiettivo di riferimento (default: dmos-nca__<regime>, "
                         "altrimenti dmos-l1l2__<regime>)")
    args = ap.parse_args()

    if args.arms:
        arms_order = list(args.arms)
    elif args.regime:
        suff = f"__{args.regime}"
        arms_order = [a.split('__')[0] + suff for a in DEFAULT_ARMS]
    else:
        arms_order = list(DEFAULT_ARMS)

    baseline = args.baseline
    if baseline is None:
        # preferisci l'arm full L1+L2 tra quelli in lista
        baseline = next((a for a in arms_order if a.startswith('dmos-nca__')),
                        next((a for a in arms_order if a.startswith('dmos-l1l2__')),
                             arms_order[-1] if arms_order else DEFAULT_BASELINE))

    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = SCRIPT_DIR.parent / f'paper_artifacts/{ts}/ablation'
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Discovering runs in: {args.results_dir}")
    print(f"  Study B arms       : {arms_order}")
    print(f"  Multi-obj baseline : {baseline}")
    groups = discover_runs(args.results_dir, scenario_filter=arms_order,
                           load_timeline=True)
    print_group_summary(groups)
    if not groups:
        print("  No matching runs found. Hai gia' lanciato gli arm E3?")
        print("  (Servono dir tipo  <ts>_dmos-latency_first__bal,  <ts>_dmos-nca__bal, ...)")
        return 1

    missing = [a for a in arms_order if a not in groups]
    if missing:
        print(f"  [warn] arm senza run: {missing} — verranno omessi dalle tabelle.")

    setup_matplotlib()

    table = build_rows(groups, arms_order)
    tex_metrics = render_metrics_latex(table)
    md_metrics = render_metrics_markdown(table)

    pairs = build_stat_tests(groups, arms_order, baseline)
    tex_stats = render_stat_tests_latex(pairs, arms_order)
    md_stats = render_stat_tests_markdown(pairs, arms_order)

    fig_paths = plot_tradeoff(table, args.output_dir)

    print(f"\n  Writing artifacts in: {args.output_dir}\n")
    (args.output_dir / 'ablation_metrics.tex').write_text(tex_metrics, encoding='utf-8')
    print("    ✓ ablation_metrics.tex")
    (args.output_dir / 'ablation_stat_tests.tex').write_text(tex_stats, encoding='utf-8')
    print("    ✓ ablation_stat_tests.tex")
    (args.output_dir / 'ablation.md').write_text(md_metrics + "\n\n" + md_stats, encoding='utf-8')
    print("    ✓ ablation.md")
    (args.output_dir / 'ablation_data.json').write_text(
        json.dumps({'table': table, 'stat_tests': pairs, 'baseline': baseline},
                   indent=2, default=str), encoding='utf-8')
    print("    ✓ ablation_data.json")
    if fig_paths:
        for p in fig_paths:
            print(f"    ✓ {p.name}")

    print("\n  Done. Nel paper:")
    print(f"      \\input{{{args.output_dir.name}/ablation_metrics.tex}}")
    print(f"      \\input{{{args.output_dir.name}/ablation_stat_tests.tex}}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
