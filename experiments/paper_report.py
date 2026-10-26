#!/usr/bin/env python3
"""
paper_report.py — Generate self-contained HTML report for the DMOS paper
=========================================================================

Genera un singolo file HTML che contiene:
  - Tutti i grafici Figure 1-8 (embedded come base64 PNG)
  - Tabelle 1 e 2 (markdown converted in HTML)
  - "Hypothesis check": automatic pass/fail per H1, H2, H3 in base ai numeri
  - Manifest del run (config, git commit, timestamp, ripetizioni)

Output: paper_report.html in --output-dir (default: paper_artifacts/<ts>/).

Usage
-----
    # Auto: presume che le figure siano in paper_artifacts/<ts>/figures/ e
    # le tabelle in paper_artifacts/<ts>/tables/ (genera-le prima!)
    python paper_report.py --paper-dir paper_artifacts/<ts>/

    # Per generare TUTTO da zero (richiama plot + tables + report):
    python paper_generate_all.py    (vedi script master)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paper_lib import (
    SCENARIOS, SCENARIO_LABELS, SCENARIO_COLORS,
    PHASE_DEFS, ScenarioGroup, discover_runs,
    bootstrap_ci, fmt_pct, print_group_summary,
    carbon_per_arm,
)


# ── Hypothesis check (H1, H2, H3) ───────────────────────────────────────────

# Soglie quantitative per declare H_i = PASS / FAIL.
# Vedi FLASH-CROWD-EVAL-PLAN.md sezione 2.

H1_P95_THRESHOLD_MS = 1000        # DMOS-l1l2 deve avere p95 (peak) < 1000ms
H1_BASELINE_THRESHOLD_MS = 3000   # DMOS-off deve essere > 3000ms (per H1 valida)
# H2: rifrasata su SLO compliance. Nelle nostre run k6 ha timeout default 60s,
# quindi le request tardive (~5-20% sopra 1s) vengono COMUNQUE servite e non
# contano come errori HTTP. Misurare "error rate" da' 0% in tutti gli arm,
# cancellando il differenziale vero. Usiamo invece SLO violations a 1s
# (Dean & Barroso 2013), letto dal cache di paper_slo_compliance.py.
H2_SLO_MS = 1000                  # SLO target: 1 secondo (interactive web)
H2_SLO_THRESHOLD_PCT = 2.0        # DMOS-l1l2 deve avere SLO violations < 2%
H2_SLO_BASELINE_PCT = 10.0        # DMOS-off deve avere SLO violations > 10%
H3_FAIRNESS_THRESHOLD = 0.0002    # variance(err) < 0.02pp = 0.0002
H4_CARBON_REDUCTION_PCT = 0.10    # DMOS-l1l2 deve ridurre CO2 di almeno il 10%
                                  # rispetto a dmos-off (per H4 PASS)

# Regime e braccio DMOS su cui valutare le ipotesi di stress (H1-H4). I run
# usano il naming dmos-<arm>__<regime> (experiments/run_naming.py), quindi qui
# risolviamo gli scenari per il regime scelto (default 'con' = flash crowd
# estremo) e per il braccio DMOS completo (default dmos-l1l2, core NCA; es.
# dmos-nca). Override via env DMOS_H_REGIME / DMOS_H_ARM.
import os
H_REGIME = os.environ.get('DMOS_H_REGIME', 'con')
H_ARM = os.environ.get('DMOS_H_ARM', 'dmos-l1l2')
H_L1L2 = f'{H_ARM}__{H_REGIME}'
H_OFF = f'dmos-off__{H_REGIME}'


def check_hypothesis_1(groups: Dict[str, ScenarioGroup]) -> Dict:
    """H1: durante peak, dmos-l1l2 < 1000ms, dmos-off > 3000ms."""
    result = {'name': 'H1 (latency)',
              'description': 'DMOS-L1+L2 maintains p95 < 1000ms during peak; '
                             'DMOS-off exceeds 3000ms.',
              'status': 'PENDING'}
    if H_L1L2 not in groups or H_OFF not in groups:
        result['note'] = f'Required scenarios missing ({H_L1L2}, {H_OFF})'
        result['status'] = 'INCOMPLETE'
        return result
    l1l2_p95 = groups[H_L1L2].metric_values_per_run('results.per_phase.peak.p95')
    off_p95 = groups[H_OFF].metric_values_per_run('results.per_phase.peak.p95')
    if len(l1l2_p95) == 0 or len(off_p95) == 0:
        result['status'] = 'INCOMPLETE'
        result['note'] = 'No p95 data available'
        return result

    # Usiamo median(mean) sui rep: piu' robusto agli outlier transient di
    # adaptation del controller (1 rep su N puo' avere un picco di settling
    # senza che il "tipico" comportamento sia compromesso). Riportiamo
    # anche il mean per trasparenza.
    l1l2_median = float(np.median(l1l2_p95))
    l1l2_mean = float(l1l2_p95.mean())
    off_median = float(np.median(off_p95))
    off_mean = float(off_p95.mean())
    h1_pass = (l1l2_median < H1_P95_THRESHOLD_MS) and (off_median > H1_BASELINE_THRESHOLD_MS)
    result.update({
        'status':  'PASS' if h1_pass else 'FAIL',
        'measured_l1l2_p95_ms_median': l1l2_median,
        'measured_l1l2_p95_ms_mean':   l1l2_mean,
        'measured_off_p95_ms_median':  off_median,
        'measured_off_p95_ms_mean':    off_mean,
        'threshold_l1l2_ms':           H1_P95_THRESHOLD_MS,
        'threshold_off_ms':            H1_BASELINE_THRESHOLD_MS,
        'aggregate':                   'median of per-run p95 (robust to outliers)',
    })
    return result


def check_hypothesis_2(groups: Dict[str, ScenarioGroup]) -> Dict:
    """H2 (rifrasata): DMOS-L1+L2 ha SLO violations < 2%, dmos-off > 10%.

    Legge i `slo_compliance.json` cached da paper_slo_compliance.py.
    Se non esistono, lo step e' INCOMPLETE con un hint al comando per generarli.
    """
    result = {
        'name': 'H2 (SLO compliance)',
        'description': f'DMOS-L1+L2 keeps SLO violations (>{H2_SLO_MS} ms) '
                       f'< {H2_SLO_THRESHOLD_PCT}%; DMOS-off exceeds '
                       f'{H2_SLO_BASELINE_PCT}%.',
        'status': 'PENDING'
    }
    if H_L1L2 not in groups or H_OFF not in groups:
        result['status'] = 'INCOMPLETE'
        result['note'] = f'Required scenarios missing ({H_L1L2}, {H_OFF})'
        return result

    def _collect_slo_pcts(group, threshold_ms: float) -> List[float]:
        vals: List[float] = []
        for run in group.runs:
            cache_path = run.run_dir / 'slo_compliance.json'
            if not cache_path.is_file():
                continue
            try:
                d = json.loads(cache_path.read_text())
                pct = d.get('global', {}).get(str(float(threshold_ms)), {}) \
                       .get('pct_above')
                if pct is None:
                    # Anche serializzato come int (caso common)
                    pct = d.get('global', {}).get(str(int(threshold_ms)), {}) \
                           .get('pct_above')
                if pct is not None:
                    vals.append(float(pct))
            except Exception:
                continue
        return vals

    l1l2_pcts = _collect_slo_pcts(groups[H_L1L2], H2_SLO_MS)
    off_pcts = _collect_slo_pcts(groups[H_OFF], H2_SLO_MS)

    if not l1l2_pcts or not off_pcts:
        result['status'] = 'INCOMPLETE'
        result['note'] = ('slo_compliance.json non trovati nei run dirs. '
                          'Esegui: python experiments/paper_slo_compliance.py')
        return result

    # Median: robusta a 1 outlier (es. run l1l2 con transient)
    l1l2_median = float(np.median(l1l2_pcts))
    off_median = float(np.median(off_pcts))
    h2_pass = (l1l2_median < H2_SLO_THRESHOLD_PCT) and \
              (off_median > H2_SLO_BASELINE_PCT)
    result.update({
        'status': 'PASS' if h2_pass else 'FAIL',
        'measured_l1l2_slo_pct_median': l1l2_median,
        'measured_l1l2_slo_pct_mean':   float(np.mean(l1l2_pcts)),
        'measured_off_slo_pct_median':  off_median,
        'measured_off_slo_pct_mean':    float(np.mean(off_pcts)),
        'threshold_l1l2_pct':           H2_SLO_THRESHOLD_PCT,
        'threshold_off_pct':            H2_SLO_BASELINE_PCT,
        'slo_threshold_ms':             H2_SLO_MS,
        'aggregate':                    'median of per-run violation rates',
    })
    return result


def check_hypothesis_3(groups: Dict[str, ScenarioGroup]) -> Dict:
    """H3: dmos-l1l2 ha varianza error rate tra cluster < 0.0002."""
    result = {'name': 'H3 (geo-fairness)',
              'description': 'DMOS-L1+L2 distributes load fairly: '
                             'variance(error rate) across clusters < 0.0002.',
              'status': 'PENDING'}
    if H_L1L2 not in groups:
        result['status'] = 'INCOMPLETE'
        return result
    group = groups[H_L1L2]
    variances = []
    for run in group.runs:
        per_cluster = run.k6_summary.get('results', {}).get('per_cluster', {})
        errs = [stats.get('errorRate', 0) for stats in per_cluster.values()
                if stats.get('errorRate') is not None]
        if len(errs) >= 2:
            variances.append(np.var(errs, ddof=1))
    if not variances:
        result['status'] = 'INCOMPLETE'
        result['note'] = 'No per-cluster error data'
        return result

    measured = float(np.mean(variances))
    h3_pass = measured < H3_FAIRNESS_THRESHOLD
    result.update({
        'status': 'PASS' if h3_pass else 'FAIL',
        'measured_variance': measured,
        'threshold': H3_FAIRNESS_THRESHOLD,
    })
    return result


def check_hypothesis_4(groups: Dict[str, ScenarioGroup]) -> Dict:
    """H4: DMOS-L1+L2 riduce le emissioni totali di CO2 di almeno il 10% vs dmos-off."""
    result = {'name': 'H4 (carbon)',
              'description': f'DMOS-L1+L2 reduces total CO2 emissions by at least '
                             f'{H4_CARBON_REDUCTION_PCT*100:.0f}% over the run vs DMOS-off.',
              'status': 'PENDING'}
    if H_L1L2 not in groups or H_OFF not in groups:
        result['status'] = 'INCOMPLETE'
        result['note'] = f'Required scenarios missing ({H_L1L2}, {H_OFF})'
        return result
    carb = carbon_per_arm(groups)          # keyed by scenario label (<arm>__<regime>)
    if H_L1L2 not in carb or H_OFF not in carb:
        result['status'] = 'INCOMPLETE'
        result['note'] = 'No carbon data computed (timeline_df missing?)'
        return result
    l1l2_mean = float(carb[H_L1L2].mean())
    off_mean = float(carb[H_OFF].mean())
    if off_mean <= 0:
        result['status'] = 'INCOMPLETE'
        result['note'] = 'dmos-off CO2 baseline non-positive'
        return result
    reduction = (off_mean - l1l2_mean) / off_mean
    h4_pass = reduction >= H4_CARBON_REDUCTION_PCT
    result.update({
        'status': 'PASS' if h4_pass else 'FAIL',
        'measured_l1l2_g':     l1l2_mean,
        'measured_off_g':      off_mean,
        'measured_reduction':  reduction,
        'threshold_reduction': H4_CARBON_REDUCTION_PCT,
    })
    return result


def all_hypothesis_checks(groups: Dict[str, ScenarioGroup]) -> List[Dict]:
    return [
        check_hypothesis_1(groups),
        check_hypothesis_2(groups),
        check_hypothesis_3(groups),
        check_hypothesis_4(groups),
    ]


# ── HTML rendering ──────────────────────────────────────────────────────────

CSS = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px;
    line-height: 1.6;
    color: #1a202c;
    background: #fafafa;
}
h1 { color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 8px; }
h2 { color: #1565C0; margin-top: 40px; border-bottom: 1px solid #cbd5e0; }
h3 { color: #2d3748; margin-top: 30px; }
header.meta {
    background: #fff;
    padding: 16px 24px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}
header.meta dl {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 6px 16px;
    margin: 0;
}
header.meta dt { font-weight: 600; color: #4a5568; }
header.meta dd { margin: 0; }
.fig {
    background: #fff;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}
.fig img { max-width: 100%; height: auto; display: block; }
.fig .caption { margin-top: 8px; font-style: italic; color: #4a5568; }
table.metrics { border-collapse: collapse; margin: 16px 0; width: 100%; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
table.metrics th { background: #1565C0; color: #fff; padding: 10px; text-align: left; }
table.metrics td { padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }
table.metrics tr:hover { background: #f7fafc; }
.hyp { padding: 12px 16px; border-radius: 6px; margin: 8px 0; border-left: 4px solid; }
.hyp.PASS { background: #f0fff4; border-color: #38a169; }
.hyp.FAIL { background: #fff5f5; border-color: #e53e3e; }
.hyp.INCOMPLETE { background: #fffaf0; border-color: #dd6b20; }
.hyp .name { font-weight: 600; font-size: 1.05em; }
.hyp .status { float: right; padding: 2px 12px; border-radius: 999px; font-weight: 600; color: #fff; font-size: 0.9em; }
.hyp.PASS .status { background: #38a169; }
.hyp.FAIL .status { background: #e53e3e; }
.hyp.INCOMPLETE .status { background: #dd6b20; }
.hyp .desc { color: #4a5568; margin: 4px 0; }
.hyp .data { font-family: 'Fira Code', monospace; font-size: 0.9em; color: #2d3748; background: #f7fafc; padding: 6px 10px; border-radius: 4px; margin-top: 6px; }
code { background: #edf2f7; padding: 2px 6px; border-radius: 3px; font-family: 'Fira Code', monospace; font-size: 0.9em; }
.legend { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle; border-radius: 2px; }
footer { margin-top: 60px; padding-top: 16px; border-top: 1px solid #cbd5e0; color: #718096; font-size: 0.9em; }
</style>
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DMOS Paper Report — {title}</title>
{css}
</head>
<body>
<h1>DMOS Flash-Crowd Evaluation Report</h1>

<header class="meta">
<dl>
<dt>Generated</dt><dd>{generated}</dd>
<dt>Run directory</dt><dd><code>{run_dir}</code></dd>
<dt>Scenarios</dt><dd>{scenarios_summary}</dd>
<dt>Git commit (first run)</dt><dd><code>{git_commit}</code></dd>
</dl>
</header>

<h2>Hypothesis check (automatic)</h2>
{hypothesis_html}

<h2>Main figures (1-4)</h2>
{main_figs_html}

<h2>Supplementary figures (5-8)</h2>
{supp_figs_html}

<h2>Tables</h2>
{tables_html}

<h2>Scenarios summary</h2>
{scenario_table_html}

<footer>
Generated by <code>paper_report.py</code>. For the LaTeX-ready figures,
see the corresponding <code>.pdf</code> files alongside this report.
</footer>
</body>
</html>
"""


def render_hypothesis_html(checks: List[Dict]) -> str:
    parts = []
    for c in checks:
        status = c.get('status', 'PENDING')
        parts.append(f'<div class="hyp {status}">')
        parts.append(f'  <span class="status">{status}</span>')
        parts.append(f'  <div class="name">{c["name"]}</div>')
        parts.append(f'  <div class="desc">{c["description"]}</div>')
        # Detail data
        data_parts = []
        for k, v in c.items():
            if k in ('name', 'description', 'status', 'note'):
                continue
            if isinstance(v, float):
                if 'err' in k.lower() or 'threshold' in k.lower() and 'rate' not in k:
                    data_parts.append(f"{k}={fmt_pct(v)}")
                else:
                    data_parts.append(f"{k}={v:.3g}")
            else:
                data_parts.append(f"{k}={v}")
        if data_parts:
            parts.append(f'  <div class="data">{"  ·  ".join(data_parts)}</div>')
        if c.get('note'):
            parts.append(f'  <div class="data">note: {c["note"]}</div>')
        parts.append('</div>')
    return "\n".join(parts)


def render_figure_html(fig_path: Path, caption: str) -> str:
    if not fig_path.is_file():
        return f'<div class="fig"><em>Figure not found: <code>{fig_path}</code></em></div>'
    try:
        data = fig_path.read_bytes()
        b64 = base64.b64encode(data).decode('ascii')
        return (f'<div class="fig">'
                f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
                f'<div class="caption">{caption}</div>'
                f'</div>')
    except Exception as e:
        return f'<div class="fig"><em>Failed to embed {fig_path.name}: {e}</em></div>'


def render_scenario_table_html(groups: Dict[str, ScenarioGroup]) -> str:
    parts = ['<table class="metrics">']
    parts.append('<thead><tr><th>Scenario</th><th>Label</th><th>Repetitions</th><th>Time range</th></tr></thead>')
    parts.append('<tbody>')
    for scenario, group in groups.items():
        color = SCENARIO_COLORS.get(scenario, '#888')
        label = SCENARIO_LABELS.get(scenario, scenario)
        if group.runs:
            time_range = f"{group.runs[0].timestamp} → {group.runs[-1].timestamp}"
        else:
            time_range = "—"
        parts.append(
            f'<tr>'
            f'<td><span class="legend" style="background:{color}"></span><code>{scenario}</code></td>'
            f'<td>{label}</td><td>{group.n_repetitions}</td>'
            f'<td><code>{time_range}</code></td>'
            f'</tr>'
        )
    parts.append('</tbody></table>')
    return "\n".join(parts)


def render_tables_html(tables_dir: Path) -> str:
    md_file = tables_dir / 'tables.md'
    if not md_file.is_file():
        return '<p><em>tables.md not found. Run <code>paper_stat_compare.py</code> first.</em></p>'
    md_text = md_file.read_text(encoding='utf-8')
    return _markdown_to_html_simple(md_text)


def _markdown_to_html_simple(md: str) -> str:
    """Markdown → HTML minimale: headers, tabelle GFM, bold, code."""
    lines = md.split('\n')
    html_lines = []
    in_table = False
    table_buffer = []

    def flush_table():
        nonlocal in_table, table_buffer
        if not table_buffer:
            return
        html_lines.append('<table class="metrics">')
        for i, row in enumerate(table_buffer):
            cells = [c.strip() for c in row.strip('|').split('|')]
            tag = 'th' if i == 0 else 'td'
            html_lines.append('<tr>' + ''.join(
                f'<{tag}>{_inline_md(c)}</{tag}>' for c in cells
            ) + '</tr>')
        html_lines.append('</table>')
        table_buffer = []
        in_table = False

    for raw in lines:
        line = raw.rstrip()
        if line.startswith('### '):
            flush_table()
            html_lines.append(f'<h3>{line[4:]}</h3>')
        elif line.startswith('## '):
            flush_table()
            html_lines.append(f'<h2>{line[3:]}</h2>')
        elif line.startswith('# '):
            flush_table()
            html_lines.append(f'<h1>{line[2:]}</h1>')
        elif line.startswith('|') and line.endswith('|'):
            # Riga di tabella (ma salta separator line "|---|---|...")
            if all(c in '-:|' for c in line.replace(' ', '')):
                in_table = True
                continue
            table_buffer.append(line)
            in_table = True
        elif in_table and not line.strip():
            flush_table()
        elif line.strip() == '':
            html_lines.append('')
        else:
            html_lines.append(f'<p>{_inline_md(line)}</p>')

    flush_table()
    return "\n".join(html_lines)


def _inline_md(s: str) -> str:
    """Inline markdown: **bold**, *italic*, `code`."""
    import re
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
    s = re.sub(r'`([^`]+?)`', r'<code>\1</code>', s)
    return s


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate self-contained HTML report.")
    parser.add_argument('--paper-dir', type=Path, default=None,
                        help='Directory paper_artifacts/<ts>/ con sotto figures/ e tables/.')
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).parent / 'results')
    parser.add_argument('--output', type=Path, default=None,
                        help='Path output HTML (default: <paper-dir>/paper_report.html)')
    args = parser.parse_args()

    if args.paper_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.paper_dir = Path(__file__).parent.parent / f'paper_artifacts/{ts}'
    args.paper_dir = Path(args.paper_dir)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    figs_dir = args.paper_dir / 'figures'
    tables_dir = args.paper_dir / 'tables'

    if args.output is None:
        args.output = args.paper_dir / 'paper_report.html'

    # Discover runs
    print(f"\n  Loading runs from: {args.results_dir}")
    groups = discover_runs(args.results_dir, load_timeline=False)
    print_group_summary(groups)
    if not groups:
        return 1

    # Hypothesis check
    print(f"  Running hypothesis checks...")
    checks = all_hypothesis_checks(groups)
    for c in checks:
        print(f"    {c['name']}: {c['status']}")

    # Build HTML
    print(f"\n  Building HTML report → {args.output}")

    # First run's manifest
    git_commit = "n/a"
    for group in groups.values():
        if group.runs and group.runs[0].manifest:
            mf = group.runs[0].manifest
            # manifest.json di run_flash_crowd.sh: git e' annidato in {git:{commit}}.
            # Fallback al vecchio campo flat git_commit.
            git_commit = (mf.get('git', {}) or {}).get('commit') \
                or mf.get('git_commit', 'n/a')
            break

    main_figs = [
        ('fig1_replicas_heatmap.png',     'Figure 1 — Per-service × per-cluster replica placement during peak (mean across reps). Row totals reveal asymmetric placement under L1.'),
        ('fig2_p95_latency_overlay.png',  'Figure 2 — p95 latency over time per scheduling strategy.'),
        ('fig3_slo_violation_overlay.png','Figure 3 — SLO violation rate (% requests > 1 s) over time.'),
        ('fig3b_slo_per_phase_bars.png',  'Figure 3b — SLO violations per test phase (mean ± CI95 over reps).'),
        ('fig4_latency_cdf_per_phase.png','Figure 4 — Cumulative latency distribution per phase; dashed lines = SLO thresholds.'),
        ('fig9_carbon_emissions.png',     'Figure 9 — Cumulative CO₂ emissions per strategy (stacked by cluster).'),
    ]
    supp_figs = [
        ('fig5_slo_heatmap.png',            'Figure 5 — SLO violation rate by Cluster × Phase per strategy (>1s threshold). Uniform rows = geo-fair.'),
        ('fig6_throughput_achieved.png',    'Figure 6 — Sanity check (NOT a DMOS-wins figure): all strategies sustain the same offered load → the latency gap is at iso-throughput, not throughput shedding.'),
        ('fig7_boxplot_p95_per_cluster.png','Figure 7 — p95 latency per cluster during peak phase; dashed line = 1s SLO.'),
        ('fig8_geo_fairness.png',           'Figure 8 — Geo-fairness: variance of SLO violations across clusters (peak phase). Lower = more fair.'),
        ('fig11_cooldown_adaptation.png',
                                            'Figure 11 — Cooldown adaptation: L2-only exhibits residual saturation transients; L1+L2 recovers cleanly.'),
    ]

    html = HTML_TEMPLATE.format(
        title=args.paper_dir.name,
        css=CSS,
        generated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        run_dir=str(args.results_dir),
        scenarios_summary=", ".join(f"<code>{s}</code> (n={g.n_repetitions})"
                                    for s, g in groups.items()),
        git_commit=git_commit,
        hypothesis_html=render_hypothesis_html(checks),
        main_figs_html="\n".join(render_figure_html(figs_dir / f, c) for f, c in main_figs),
        supp_figs_html="\n".join(render_figure_html(figs_dir / f, c) for f, c in supp_figs),
        tables_html=render_tables_html(tables_dir),
        scenario_table_html=render_scenario_table_html(groups),
    )

    args.output.write_text(html, encoding='utf-8')
    size_kb = args.output.stat().st_size / 1024
    print(f"    ✓ {args.output.name}  ({size_kb:.1f} KB)\n")

    # Anche dump JSON delle hypothesis checks (machine-readable)
    json_path = args.paper_dir / 'hypothesis_checks.json'
    json_path.write_text(json.dumps(checks, indent=2, default=str), encoding='utf-8')
    print(f"    ✓ {json_path.name}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
