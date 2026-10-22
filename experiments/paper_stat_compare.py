#!/usr/bin/env python3
"""
paper_stat_compare.py — Statistical comparison tables for the DMOS paper
=========================================================================

Genera due tabelle LaTeX paper-ready (booktabs style) e una markdown
leggibile dal repo.

Output (in --output-dir, default: paper_artifacts/<timestamp>/tables/):
  - table1_metrics.tex
        Una riga per scenario, colonne: p95 [CI95], p99 [CI95], err% [CI95],
        avg replicas [CI95]. Tutti calcolati su fase 'peak' (300-600s) media
        delle 3 ripetizioni.
  - table2_stat_tests.tex
        Per ciascuna coppia (DMOS-off vs DMOS-l1l2, ecc.) e ciascuna metrica:
        Welch's t-test pvalue, Cohen's d (effect size + label).
  - tables.md
        Versione markdown leggibile (per repo + slide quickreview).
  - tables_data.json
        Dump JSON dei numeri raw, riusabile da altri script (es. paper_report).

Usage
-----
    python paper_stat_compare.py
    python paper_stat_compare.py --results-dir experiments/results --output-dir paper_artifacts/v1/tables
    python paper_stat_compare.py --baseline dmos-off    # confronto vs questo scenario
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paper_lib import (
    SCENARIOS, SCENARIO_LABELS, PHASES, PHASE_DEFS,
    ScenarioGroup, discover_runs,
    bootstrap_ci, welch_ttest, cohen_d, cohen_d_label,
    fmt_ms, fmt_pct, fmt_ci, print_group_summary,
    carbon_per_arm, scenario_label,
)


def _ordered_scenarios(keys) -> List[str]:
    """Ordina gli scenari scoperti: prima quelli canonici (SCENARIOS), poi
    gli altri (arm regime-suffissi / custom) nell'ordine di scoperta. Cosi'
    le tabelle includono anche arm fuori dalla lista hardcoded."""
    keys = list(keys)
    head = [s for s in SCENARIOS if s in keys]
    tail = [s for s in keys if s not in head]
    return head + tail


# ── Metric definitions (peak phase) ─────────────────────────────────────────
# (key_path nel k6_summary, label, unit, formatter, lower_is_better, in_table2)
#
# - errorRate rimossa: con k6 timeout=60s e nessun service crash, e' 0% in
#   tutti gli arm. La sostituiamo con SLO violations >1s (special-cased,
#   letta da slo_compliance.json).
# - avg_rps (throughput) tenuta in Table 1 come sanity check ("tutti gli arm
#   sostengono il carico"), ma esclusa da Table 2 (invariante per design:
#   metterla nei pairwise test mostra "+0.0% / p~0.5" che confonde).

SLO_THRESHOLD_MS = 2000.0   # soglia SLO (2s: sensata per la catena gRPC di Online Boutique)

METRICS = [
    # (key_path, label, unit, fmt, lower_is_better, in_table2)
    ('results.per_phase.peak.p50',  'p50 (ms)',         'ms',  'ms',  True,  True),
    ('results.per_phase.peak.p95',  'p95 (ms)',         'ms',  'ms',  True,  True),
    ('results.per_phase.peak.p99',  'p99 (ms)',         'ms',  'ms',  True,  True),
    ('results.global.avg_rps',      'Throughput (rps)', 'rps', 'rps', False, False),
]


# ── Aggregation ──────────────────────────────────────────────────────────────

def collect_metric_values(groups: Dict[str, ScenarioGroup],
                          key_path: str) -> Dict[str, np.ndarray]:
    """{scenario: array di N valori (uno per ripetizione)}"""
    out = {}
    for scenario, group in groups.items():
        vals = group.metric_values_per_run(key_path)
        if len(vals) > 0:
            out[scenario] = vals
    return out


def slo_violations_per_arm(groups: Dict[str, ScenarioGroup],
                           threshold_ms: float = SLO_THRESHOLD_MS,
                           phase: str = 'peak') -> Dict[str, np.ndarray]:
    """
    Per ogni arm, array di SLO violation rate (%) — uno per ripetizione.
    Legge dai file `slo_compliance.json` cached da paper_slo_compliance.py.
    Se i cache file mancano, l'arm viene omesso (chiamare prima
    paper_slo_compliance.py o lo Step 0 della pipeline).
    """
    out: Dict[str, np.ndarray] = {}
    # Le chiavi possono essere serializzate come '1000' o '1000.0' a seconda
    # di come e' stato cachato. Proviamo entrambe.
    candidate_keys = [str(threshold_ms), str(float(threshold_ms)), str(int(threshold_ms))]
    for scenario, group in groups.items():
        vals: List[float] = []
        for run in group.runs:
            cache_path = run.run_dir / 'slo_compliance.json'
            if not cache_path.is_file():
                continue
            try:
                d = json.loads(cache_path.read_text())
            except Exception:
                continue
            if phase in (None, 'global'):
                bucket = d.get('global', {})
            else:
                bucket = d.get('per_phase', {}).get(phase, {}).get('thresholds', {})
            pct = None
            for k in candidate_keys:
                if k in bucket:
                    pct = bucket[k].get('pct_above')
                    if pct is not None:
                        break
            if pct is not None:
                vals.append(float(pct))
        if vals:
            out[scenario] = np.array(vals)
    return out


def avg_replicas_per_scenario(groups: Dict[str, ScenarioGroup],
                              phase: str = 'peak') -> Dict[str, np.ndarray]:
    """Media replicas per scenario durante una fase, una serie per run."""
    out = {}
    phase_window = next(((t0, t1) for (t0, t1, n, _) in PHASE_DEFS if n == phase), None)
    if phase_window is None:
        return out
    t0, t1 = phase_window
    for scenario, group in groups.items():
        vals = []
        for run in group.runs:
            if run.timeline_df is None or run.timeline_df.empty:
                continue
            df = run.timeline_df
            sub = df[(df['elapsed_s'] >= t0) & (df['elapsed_s'] < t1)]
            if not sub.empty and 'replicas' in sub.columns:
                # Mean across all clusters and time bins → "total replicas"
                # then mean across the phase
                per_ts = sub.groupby('ts')['replicas'].sum()
                vals.append(float(per_ts.mean()))
        if vals:
            out[scenario] = np.array(vals)
    return out


# ── Table 1: Metrics with CI95 ──────────────────────────────────────────────

def build_table1(groups: Dict[str, ScenarioGroup]) -> Dict:
    """
    Costruisce dati per Table 1: per ogni scenario, mean + CI95 di:
    p50, p95, p99, error rate, throughput, avg replicas, total CO2.
    """
    table = {}
    repl_data = avg_replicas_per_scenario(groups, phase='peak')
    carbon_data = carbon_per_arm(groups)
    slo_data = slo_violations_per_arm(groups, SLO_THRESHOLD_MS, phase='peak')

    for scenario, group in groups.items():
        row = {
            'scenario':    scenario,
            'label':       scenario_label(scenario),
            'n_reps':      group.n_repetitions,
            'metrics':     {},
        }
        for key_path, name, unit, fmt_kind, *_ in METRICS:
            vals = group.metric_values_per_run(key_path)
            if len(vals) > 0:
                mean, low, high = bootstrap_ci(vals)
                row['metrics'][key_path] = {
                    'name':  name, 'unit': unit, 'fmt': fmt_kind,
                    'mean':  mean, 'ci_low': low, 'ci_high': high,
                    'raw':   vals.tolist(),
                }

        # SLO violations (>1s) durante peak — special-cased da slo_compliance.json
        if scenario in slo_data:
            vals = slo_data[scenario]
            mean, low, high = bootstrap_ci(vals)
            row['metrics']['slo_viol_1s'] = {
                'name': f'SLO viol. >{int(SLO_THRESHOLD_MS/1000)}s (%)',
                'unit': '%', 'fmt': 'slo_pct',
                'mean': mean, 'ci_low': low, 'ci_high': high,
                'raw':  vals.tolist(),
            }

        # avg replicas (peak)
        if scenario in repl_data:
            vals = repl_data[scenario]
            mean, low, high = bootstrap_ci(vals)
            row['metrics']['avg_replicas_peak'] = {
                'name': 'Avg replicas (peak)', 'unit': '', 'fmt': 'rep',
                'mean': mean, 'ci_low': low, 'ci_high': high,
                'raw':  vals.tolist(),
            }

        # Total CO2 emissions per run (g) — calcolato post-hoc dal timeline
        if scenario in carbon_data:
            vals = carbon_data[scenario]
            mean, low, high = bootstrap_ci(vals)
            row['metrics']['carbon_total_g'] = {
                'name': 'Total CO₂ (g)', 'unit': 'g', 'fmt': 'co2',
                'mean': mean, 'ci_low': low, 'ci_high': high,
                'raw':  vals.tolist(),
            }
        table[scenario] = row

    # Ordine colonne: latency stats, SLO, throughput (sanity), replicas, CO2
    metric_order = (['results.per_phase.peak.p50',
                     'results.per_phase.peak.p95',
                     'results.per_phase.peak.p99',
                     'slo_viol_1s',
                     'results.global.avg_rps',
                     'avg_replicas_peak',
                     'carbon_total_g'])
    return {'rows': table, 'metric_order': metric_order}


def render_table1_latex(table: Dict) -> str:
    """
    Render Table 1 come LaTeX booktabs. Una riga per scenario, una colonna
    per metrica. Mostra mean [low, high].
    """
    metric_order = table['metric_order']
    rows = table['rows']

    # Header
    metric_names = []
    for key_path in metric_order:
        any_row = next(iter(rows.values()))
        m_info = any_row['metrics'].get(key_path) if any_row else None
        if m_info:
            metric_names.append(m_info['name'])
        else:
            metric_names.append(key_path.split('.')[-1])

    n_cols = 1 + len(metric_names)
    col_spec = 'l' + 'c' * len(metric_names)

    lines = []
    lines.append("% Table 1 — Aggregate metrics with bootstrap CI95 (peak phase)")
    lines.append("% Generated by paper_stat_compare.py")
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{End-to-end performance metrics during the \\emph{peak} phase "
                 "(300-600 s of the flash-crowd test). "
                 "Each cell shows the mean across the $n$ repetitions "
                 "with bootstrap 95\\% confidence interval in brackets.}")
    lines.append("\\label{tab:metrics-peak}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row
    header = "Strategy & " + " & ".join(metric_names) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Data rows — itera sugli scenari REALMENTE scoperti (gia' ordinati
    # SCENARIOS-first da discover_runs), cosi' funziona anche con arm
    # regime-suffissi (es. dmos-l1l2__con) fuori dalla lista hardcoded.
    for scenario in _ordered_scenarios(rows.keys()):
        if scenario not in rows:
            continue
        r = rows[scenario]
        cells = [_latex_escape(r['label']) + f" ($n={r['n_reps']}$)"]
        for key_path in metric_order:
            m = r['metrics'].get(key_path)
            if m is None:
                cells.append("--")
                continue
            cells.append(_fmt_metric_latex(m))
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def render_table1_markdown(table: Dict) -> str:
    """Markdown version per repo/slide."""
    metric_order = table['metric_order']
    rows = table['rows']

    any_row = next(iter(rows.values()))
    metric_names = [any_row['metrics'][k]['name'] if k in any_row['metrics'] else k.split('.')[-1]
                    for k in metric_order]

    lines = ["### Table 1 — Aggregate metrics (peak phase, mean [CI95])\n"]
    lines.append("| Strategy | n | " + " | ".join(metric_names) + " |")
    lines.append("|" + "---|" * (len(metric_names) + 2))
    for scenario in _ordered_scenarios(rows.keys()):
        if scenario not in rows:
            continue
        r = rows[scenario]
        row = [f"**{r['label']}**", str(r['n_reps'])]
        for key_path in metric_order:
            m = r['metrics'].get(key_path)
            if m is None:
                row.append("--")
                continue
            row.append(_fmt_metric_md(m))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ── Table 2: Statistical pairwise tests ─────────────────────────────────────

def build_table2(groups: Dict[str, ScenarioGroup],
                 baseline: str = 'dmos-off') -> List[Dict]:
    """
    Per ogni metrica e ogni scenario diverso da `baseline`, calcola:
        Welch's t-test (pvalue), Cohen's d (effect size, magnitudine).

    Returns: lista di dict {scenario_a, scenario_b, metric, pvalue, d, mag}
    """
    if baseline not in groups:
        # Fallback: usa il primo scenario disponibile
        baseline = next(iter(groups.keys()))

    pairs = []
    base_group = groups[baseline]

    # SLO violations (special-cased: dato extra-METRICS, da slo_compliance.json)
    slo_data = slo_violations_per_arm(groups, SLO_THRESHOLD_MS, phase='peak')

    for scenario, group in groups.items():
        if scenario == baseline:
            continue

        # METRICS standard, filtrando quelli con in_table2=False (es. throughput
        # invariante per design — confonde i pairwise tests).
        for key_path, name, unit, fmt_kind, lower_better, in_table2 in METRICS:
            if not in_table2:
                continue
            a = base_group.metric_values_per_run(key_path)
            b = group.metric_values_per_run(key_path)
            if len(a) < 2 or len(b) < 2:
                continue
            t, p = welch_ttest(a, b)
            d = cohen_d(a, b)
            mag = cohen_d_label(d)
            improvement = (a.mean() - b.mean()) / a.mean() * 100 if a.mean() != 0 else 0
            if not lower_better:
                improvement = -improvement   # higher is better → flip sign
            pairs.append({
                'baseline':       baseline,
                'scenario':       scenario,
                'metric':         name,
                'unit':           unit,
                'mean_a':         a.mean(),
                'mean_b':         b.mean(),
                'improvement_pct': improvement,
                't_stat':         t,
                'p_value':        p,
                'cohen_d':        d,
                'magnitude':      mag,
                'significant':    p < 0.05 if not np.isnan(p) else False,
            })

        # SLO violations row (lower is better)
        if baseline in slo_data and scenario in slo_data:
            a = slo_data[baseline]
            b = slo_data[scenario]
            if len(a) >= 2 and len(b) >= 2:
                t, p = welch_ttest(a, b)
                d = cohen_d(a, b)
                mag = cohen_d_label(d)
                improvement = (a.mean() - b.mean()) / a.mean() * 100 if a.mean() != 0 else 0
                pairs.append({
                    'baseline':        baseline,
                    'scenario':        scenario,
                    'metric':          f'SLO viol. >{int(SLO_THRESHOLD_MS/1000)}s (%)',
                    'unit':            '%',
                    'mean_a':          float(a.mean()),
                    'mean_b':          float(b.mean()),
                    'improvement_pct': improvement,
                    't_stat':          t,
                    'p_value':         p,
                    'cohen_d':         d,
                    'magnitude':       mag,
                    'significant':     p < 0.05 if not np.isnan(p) else False,
                })
    return pairs


def render_table2_latex(pairs: List[Dict]) -> str:
    """Table 2 LaTeX. Una riga per pair (scenario, metric)."""
    if not pairs:
        return "% Table 2 empty (no comparisons available)"

    lines = []
    lines.append("% Table 2 — Pairwise Welch's t-test + Cohen's d effect size")
    lines.append("% Generated by paper_stat_compare.py")
    lines.append("\\begin{table*}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{Pairwise statistical comparison against the baseline strategy. "
                 "Welch's $t$-test $p$-value evaluates statistical significance "
                 "($p<0.05$ marked with *); Cohen's $d$ quantifies effect size "
                 "(magnitude: $|d|<0.2$ negligible, $<0.5$ small, $<0.8$ medium, $\\geq 0.8$ large).}")
    lines.append("\\label{tab:stat-tests}")
    lines.append("\\begin{tabular}{llrrrrl}")
    lines.append("\\toprule")
    lines.append("Strategy vs.\\ baseline & Metric & Baseline & Strategy & "
                 "Improv. & $p$-value & Cohen's $d$ \\\\")
    lines.append("\\midrule")

    # Group rows by scenario for nicer formatting
    by_scenario: Dict[str, List[Dict]] = {}
    for p in pairs:
        by_scenario.setdefault(p['scenario'], []).append(p)

    for scenario in _ordered_scenarios(by_scenario.keys()):
        if scenario not in by_scenario:
            continue
        rows = by_scenario[scenario]
        for i, p in enumerate(rows):
            cells = []
            if i == 0:
                label = scenario_label(scenario)
                cells.append(f"\\textbf{{{_latex_escape(label)}}}")
            else:
                cells.append("")
            cells.append(p['metric'])
            cells.append(f"{p['mean_a']:.1f}")
            cells.append(f"{p['mean_b']:.1f}")
            sign = '+' if p['improvement_pct'] > 0 else ''
            cells.append(f"{sign}{p['improvement_pct']:.1f}\\%")
            if np.isnan(p['p_value']):
                cells.append("n/a")
            else:
                star = '$^*$' if p['significant'] else ''
                cells.append(f"{p['p_value']:.3g}{star}")
            if np.isnan(p['cohen_d']):
                cells.append("n/a")
            else:
                cells.append(_fmt_cohen_d(p['cohen_d'], p['magnitude']))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")

    # Rimuovi ultimo \midrule e metti \bottomrule
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def render_table2_markdown(pairs: List[Dict]) -> str:
    """Markdown version."""
    if not pairs:
        return "### Table 2 — (no comparisons available)\n"

    lines = [f"### Table 2 — Pairwise statistical tests vs baseline `{pairs[0]['baseline']}`\n"]
    lines.append("| Strategy | Metric | Baseline | Strategy | Improv. | p-value | Cohen's d |")
    lines.append("|---|---|---|---|---|---|---|")

    by_scenario: Dict[str, List[Dict]] = {}
    for p in pairs:
        by_scenario.setdefault(p['scenario'], []).append(p)

    for scenario in _ordered_scenarios(by_scenario.keys()):
        if scenario not in by_scenario:
            continue
        rows = by_scenario[scenario]
        for i, p in enumerate(rows):
            label = f"**{SCENARIO_LABELS.get(scenario, scenario)}**" if i == 0 else ""
            pv = "n/a" if np.isnan(p['p_value']) else f"{p['p_value']:.3g}"
            if not np.isnan(p['p_value']) and p['significant']:
                pv += " *"
            d = "n/a" if np.isnan(p['cohen_d']) else _fmt_cohen_d(p['cohen_d'], p['magnitude'])
            sign = '+' if p['improvement_pct'] > 0 else ''
            lines.append(
                f"| {label} | {p['metric']} | {p['mean_a']:.1f} | {p['mean_b']:.1f} | "
                f"{sign}{p['improvement_pct']:.1f}% | {pv} | {d} |"
            )
    lines.append("\n*pvalue < 0.05 marked with `*`.*\n")
    return "\n".join(lines)


# ── Formatting helpers ──────────────────────────────────────────────────────

def _fmt_metric_latex(m: Dict) -> str:
    """Format una cella della tabella LaTeX con mean + CI95."""
    if m['fmt'] == 'pct':
        return f"{m['mean']*100:.2f}\\% [{m['ci_low']*100:.2f}, {m['ci_high']*100:.2f}]"
    if m['fmt'] == 'slo_pct':
        # SLO violations gia' in % (pct_above), no moltiplicazione 100
        return f"{m['mean']:.2f}\\% [{m['ci_low']:.2f}, {m['ci_high']:.2f}]"
    if m['fmt'] == 'ms':
        return f"{m['mean']:.0f} [{m['ci_low']:.0f}, {m['ci_high']:.0f}]"
    if m['fmt'] == 'rps':
        return f"{m['mean']:.1f} [{m['ci_low']:.1f}, {m['ci_high']:.1f}]"
    if m['fmt'] == 'rep':
        return f"{m['mean']:.1f} [{m['ci_low']:.1f}, {m['ci_high']:.1f}]"
    if m['fmt'] == 'co2':
        return f"{m['mean']:.1f} [{m['ci_low']:.1f}, {m['ci_high']:.1f}]"
    return f"{m['mean']:.2f}"


def _fmt_metric_md(m: Dict) -> str:
    """Format una cella markdown."""
    if m['fmt'] == 'pct':
        return f"{m['mean']*100:.2f}% [{m['ci_low']*100:.2f}, {m['ci_high']*100:.2f}]"
    if m['fmt'] == 'slo_pct':
        return f"{m['mean']:.2f}% [{m['ci_low']:.2f}, {m['ci_high']:.2f}]"
    if m['fmt'] == 'ms':
        return f"{m['mean']:.0f} [{m['ci_low']:.0f}, {m['ci_high']:.0f}]"
    if m['fmt'] in ('rps', 'rep', 'co2'):
        return f"{m['mean']:.1f} [{m['ci_low']:.1f}, {m['ci_high']:.1f}]"
    return f"{m['mean']:.2f}"


def _fmt_cohen_d(d: float, magnitude: str) -> str:
    """
    Cohen's d con cap visivo a |d|=20. Sopra quel valore la varianza
    intra-arm e' di fatto zero rispetto al differenziale mean-vs-mean: il
    valore numerico esatto (es. +239) e' tecnicamente corretto ma sembra
    un bug a un reviewer. Mostrare ">20 (saturated)" e' onesto e leggibile.
    """
    if np.isnan(d):
        return "n/a"
    abs_d = abs(d)
    if abs_d >= 20:
        sign = '+' if d > 0 else '-'
        return f"{sign}>20 (saturated)"
    sign = '+' if d >= 0 else ''
    return f"{sign}{d:.2f} ({magnitude})"


def _latex_escape(s: str) -> str:
    """Escape LaTeX special chars."""
    return (s
            .replace('\\', r'\textbackslash{}')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('$', r'\$')
            .replace('#', r'\#')
            .replace('_', r'\_')
            .replace('{', r'\{')
            .replace('}', r'\}'))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera Table 1 (metrics) + Table 2 (stat tests) per il paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent / 'results')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--baseline', default='dmos-off',
                        help="Scenario di baseline per i confronti pairwise di Table 2")
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help="Filtra scenari (default: tutti)")
    args = parser.parse_args()

    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = Path(__file__).parent.parent / f'paper_artifacts/{ts}/tables'
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Discovering runs in: {args.results_dir}")
    groups = discover_runs(args.results_dir, scenario_filter=args.scenarios,
                           load_timeline=True)
    print_group_summary(groups)
    if not groups:
        print("  No runs found, exiting.")
        return 1

    # Build tables
    print(f"  Building Table 1 (metrics)...")
    table1 = build_table1(groups)
    tex1 = render_table1_latex(table1)
    md1 = render_table1_markdown(table1)

    print(f"  Building Table 2 (stat tests, baseline={args.baseline})...")
    pairs = build_table2(groups, baseline=args.baseline)
    tex2 = render_table2_latex(pairs)
    md2 = render_table2_markdown(pairs)

    # Write files
    print(f"\n  Writing tables in: {args.output_dir}\n")

    f1 = args.output_dir / 'table1_metrics.tex'
    f1.write_text(tex1, encoding='utf-8')
    print(f"    ✓ {f1.name}")

    f2 = args.output_dir / 'table2_stat_tests.tex'
    f2.write_text(tex2, encoding='utf-8')
    print(f"    ✓ {f2.name}")

    md = args.output_dir / 'tables.md'
    md.write_text(md1 + "\n\n" + md2, encoding='utf-8')
    print(f"    ✓ {md.name}")

    raw = args.output_dir / 'tables_data.json'
    raw.write_text(json.dumps({
        'table1': table1,
        'table2': pairs,
        'baseline': args.baseline,
    }, indent=2, default=str), encoding='utf-8')
    print(f"    ✓ {raw.name}")

    print(f"\n  Done. Include nel paper LaTeX:")
    print(f"      \\input{{{args.output_dir.name}/table1_metrics.tex}}")
    print(f"      \\input{{{args.output_dir.name}/table2_stat_tests.tex}}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
