#!/usr/bin/env python3
"""
paper_generate_all.py — Master script for the DMOS paper artifacts
====================================================================

Genera TUTTI gli artefatti del paper in una singola cartella:
  paper_artifacts/<YYYYMMDD_HHMMSS>/
  ├── figures/
  │   ├── fig1_replicas_trajectory.{pdf,png}
  │   ├── fig2_p95_latency_overlay.{pdf,png}
  │   ├── fig3_error_rate_overlay.{pdf,png}
  │   ├── fig4_latency_cdf_per_phase.{pdf,png}
  │   ├── fig5_error_heatmap.{pdf,png}
  │   ├── fig6_throughput_achieved.{pdf,png}
  │   ├── fig7_boxplot_p95_per_cluster.{pdf,png}
  │   └── fig8_geo_fairness.{pdf,png}
  ├── tables/
  │   ├── table1_metrics.tex           (booktabs, LaTeX-ready)
  │   ├── table2_stat_tests.tex
  │   ├── tables.md
  │   └── tables_data.json
  ├── paper_report.html                (self-contained, embedded base64 figs)
  ├── hypothesis_checks.json           (machine-readable H1/H2/H3 status)
  └── manifest.json                    (config + git commit + run inventory)

Esegue in sequenza:
  0. paper_slo_compliance.py       (cache SLO per H2 e Table 1)
  1. paper_plot_trajectories.py    (Figure 1-4, main)
  2. paper_plot_supplementary.py   (Figure 5-8, supplementary)
  2b. paper_plot_carbon.py         (Figure 9, carbon)
  2c. paper_plot_metastable.py     (Figure 10-11, Study C metastabile, best-effort)
  2d. paper_plot_skew_robustness.py(Study A: p95/SLO vs demand skew, best-effort)
  2e. paper_ablation_compare.py    (E3: profili di pesi, best-effort)
  3. paper_stat_compare.py         (Table 1, 2; markdown; JSON)
  4. paper_report.py               (HTML standalone master report)

Gli step Study A/B/C sono best-effort: se i run <arm>__<regime> o
<obiettivo>__bal non esistono ancora, lo step viene saltato senza far
fallire l'intera pipeline.

Validazione: warning se uno scenario ha != 3 ripetizioni (3 e' il default
target del FLASH-CROWD-EVAL-PLAN.md).

Usage
-----
    # All-in-one con auto-timestamp:
    python paper_generate_all.py

    # Custom destination + tag:
    python paper_generate_all.py --tag "v1-final" --output-base paper_artifacts/

    # Sub-selezione (skippa qualcosa):
    python paper_generate_all.py --skip-supplementary
    python paper_generate_all.py --no-report

    # Filtra scenari (es. solo dmos-l1l2 vs dmos-off):
    python paper_generate_all.py --scenarios dmos-off dmos-l1l2

Variabili env (alternative a CLI):
    DMOS_PAPER_TAG          : tag del run (default: timestamp)
    DMOS_RESULTS_DIR        : input dir (default: experiments/results)
    DMOS_PAPER_BASELINE     : baseline per stat tests (default: dmos-off)
    DMOS_TARGET_REPS        : numero atteso di ripetizioni (default: 3)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Setup path
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from paper_lib import (
    SCENARIOS, SCENARIO_LABELS, ScenarioGroup, discover_runs, print_group_summary,
)


# ── Colors per console output ──────────────────────────────────────────────

C_RESET = '\033[0m'
C_BOLD = '\033[1m'
C_CYAN = '\033[36m'
C_GREEN = '\033[32m'
C_YELLOW = '\033[33m'
C_RED = '\033[31m'
C_DIM = '\033[2m'


def hdr(msg: str):
    line = "─" * 72
    print(f"\n{C_DIM}{line}{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}  {msg}{C_RESET}")
    print(f"{C_DIM}{line}{C_RESET}")


def step(msg: str):
    print(f"  {C_CYAN}→{C_RESET} {msg}")


def ok(msg: str):
    print(f"  {C_GREEN}✓{C_RESET} {msg}")


def warn(msg: str):
    print(f"  {C_YELLOW}!{C_RESET} {msg}")


def fail(msg: str):
    print(f"  {C_RED}✗{C_RESET} {msg}")


# ── Validation: scenari + ripetizioni attese ───────────────────────────────

def validate_groups(groups: Dict[str, ScenarioGroup], target_reps: int) -> List[str]:
    """Ritorna lista di warning (es. scenari con < target_reps)."""
    warnings_list = []
    if not groups:
        warnings_list.append("No scenarios found in results-dir.")
        return warnings_list

    # Tutti gli scenari attesi presenti?
    expected_scenarios = set(SCENARIOS)
    found_scenarios = set(groups.keys())
    missing = expected_scenarios - found_scenarios
    if missing:
        warnings_list.append(
            f"Missing scenarios (atteso da SCENARIOS list): {sorted(missing)}. "
            f"Il paper avra' confronti incompleti.")

    # Ripetizioni
    for scenario, group in groups.items():
        if group.n_repetitions < target_reps:
            warnings_list.append(
                f"Scenario '{scenario}' ha {group.n_repetitions} ripetizioni "
                f"(target: {target_reps}). CI95 sara' wide.")
        elif group.n_repetitions > target_reps:
            warnings_list.append(
                f"Scenario '{scenario}' ha {group.n_repetitions} ripetizioni "
                f"(target: {target_reps}). Verifica che non ci siano run sporche.")

    return warnings_list


# ── Sub-script invocations ─────────────────────────────────────────────────

def run_subscript(script: str, args: List[str], cwd: Path = None) -> int:
    """Esegue uno script in subprocess. Ritorna exit code."""
    script_path = HERE / script
    if not script_path.is_file():
        fail(f"Script non trovato: {script_path}")
        return 1
    cmd = [sys.executable, str(script_path)] + args
    print(f"  {C_DIM}$ {' '.join(cmd)}{C_RESET}")
    env = os.environ.copy()
    # Forza UTF-8 per evitare cp1252 errors su Windows
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    try:
        proc = subprocess.run(cmd, cwd=cwd or HERE.parent, env=env)
        return proc.returncode
    except KeyboardInterrupt:
        warn("Interrupted by user")
        return 130
    except Exception as e:
        fail(f"Errore lanciando {script}: {e}")
        return 1


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate all paper artifacts (figures, tables, report) in one folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--results-dir', type=Path,
                        default=Path(os.environ.get('DMOS_RESULTS_DIR',
                                                    str(HERE / 'results'))),
                        help='Dove sono i run dirs (default: experiments/results)')
    parser.add_argument('--output-base', type=Path,
                        default=HERE.parent / 'paper_artifacts',
                        help='Directory padre (default: paper_artifacts/)')
    parser.add_argument('--tag',
                        default=os.environ.get('DMOS_PAPER_TAG', ''),
                        help='Suffix opzionale alla cartella (default: solo timestamp)')
    parser.add_argument('--baseline',
                        default=os.environ.get('DMOS_PAPER_BASELINE', 'dmos-off'),
                        help='Scenario di baseline per Table 2 (default: dmos-off)')
    parser.add_argument('--target-reps', type=int,
                        default=int(os.environ.get('DMOS_TARGET_REPS', '3')),
                        help='Ripetizioni attese per scenario (default: 3)')
    parser.add_argument('--scenarios', nargs='+', default=None,
                        help='Filtra scenari (default: tutti)')
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg'],
                        help='Formati per le figure (default: pdf png)')
    parser.add_argument('--dpi', type=int, default=600,
                        help='DPI per i grafici (default: 600)')
    parser.add_argument('--skip-main', action='store_true',
                        help='Skip Figure 1-4 (paper_plot_trajectories)')
    parser.add_argument('--skip-supplementary', action='store_true',
                        help='Skip Figure 5-8 (paper_plot_supplementary)')
    parser.add_argument('--skip-detailed', action='store_true',
                        help='Skip grafici dettagliati per-regime (paper_plot_detailed, ~10min)')
    parser.add_argument('--skip-tables', action='store_true',
                        help='Skip Table 1-2 (paper_stat_compare)')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip HTML report (paper_report)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Solo validation, no script execution')
    args = parser.parse_args()

    # ── Build output dir ────────────────────────────────────────────────────
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{ts}_{args.tag}" if args.tag else ts
    paper_dir = args.output_base / folder_name
    figs_dir = paper_dir / 'figures'
    tables_dir = paper_dir / 'tables'

    hdr("DMOS Paper Artifact Generator")
    print(f"  Results input:  {args.results_dir}")
    print(f"  Paper output:   {paper_dir}")
    print(f"  Baseline:       {args.baseline}")
    print(f"  Target reps:    {args.target_reps}")
    print(f"  Formats:        {args.formats}, DPI={args.dpi}")

    # ── Validation pre-flight ───────────────────────────────────────────────
    hdr("Pre-flight validation")

    if not args.results_dir.is_dir():
        fail(f"Results dir non esiste: {args.results_dir}")
        return 1

    groups = discover_runs(args.results_dir, scenario_filter=args.scenarios,
                           load_timeline=False)
    if not groups:
        fail("Nessun run scoperto. Lancia prima i flash-crowd run con "
             "`bash experiments/run_flash_crowd.sh --label <scenario>`")
        return 1

    print_group_summary(groups)

    issues = validate_groups(groups, target_reps=args.target_reps)
    if issues:
        for w in issues:
            warn(w)
        warn(f"Ci sono {len(issues)} warning. Procedere?")
    else:
        ok("All scenarios complete, target repetitions met.")

    if args.dry_run:
        print("\n  [dry-run] No artifacts will be produced. Bye.")
        return 0

    paper_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Manifest globale ────────────────────────────────────────────────────
    git_commit = "n/a"
    git_branch = "n/a"
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                cwd=HERE.parent, capture_output=True, text=True)
        if result.returncode == 0:
            git_commit = result.stdout.strip()
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                cwd=HERE.parent, capture_output=True, text=True)
        if result.returncode == 0:
            git_branch = result.stdout.strip()
    except FileNotFoundError:
        pass

    manifest = {
        'generated_at':    datetime.now().isoformat(),
        'tag':             args.tag,
        'paper_dir':       str(paper_dir),
        'results_dir':     str(args.results_dir),
        'baseline':        args.baseline,
        'target_reps':     args.target_reps,
        'formats':         args.formats,
        'dpi':             args.dpi,
        'git_commit':      git_commit,
        'git_branch':      git_branch,
        'scenarios_found': {scen: g.n_repetitions for scen, g in groups.items()},
        'validation_warnings': issues,
        'python_version':  sys.version,
    }
    manifest_path = paper_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    ok(f"manifest.json scritto ({git_commit}, branch {git_branch})")

    # ── Step 1: Figure 1-4 ──────────────────────────────────────────────────
    common_args = [
        '--results-dir', str(args.results_dir),
        '--output-dir', str(figs_dir),
        '--formats'] + args.formats + [
        '--dpi', str(args.dpi),
    ]
    if args.scenarios:
        common_args += ['--scenarios'] + args.scenarios

    overall_status = []

    # ── Step 0: SLO compliance cache ────────────────────────────────────────
    # Scansiona i k6_events.json e cacha slo_compliance.json in ogni run dir.
    # H2 (in paper_report) e Table 1 (in paper_stat_compare) leggono da qui.
    # Best-effort: se fallisce, il resto della pipeline funziona comunque
    # (H2 dira' INCOMPLETE, Table 1 omettera' la riga SLO).
    hdr("Step 0/4 — SLO compliance cache (paper_slo_compliance)")
    slo_args = ['--results-dir', str(args.results_dir)]
    rc = run_subscript('paper_slo_compliance.py', slo_args)
    if rc != 0:
        warn(f"paper_slo_compliance.py exit={rc} (best-effort, non fatal)")
        overall_status.append(('slo_cache', 'fail'))
    else:
        ok("SLO compliance cache ready")
        overall_status.append(('slo_cache', 'ok'))

    if not args.skip_main:
        hdr("Step 1/4 — Main figures (Fig 1-4)")
        rc = run_subscript('paper_plot_trajectories.py', common_args)
        if rc != 0:
            fail(f"paper_plot_trajectories.py exit={rc}")
            overall_status.append(('main_figures', 'fail'))
        else:
            ok("Main figures generated")
            overall_status.append(('main_figures', 'ok'))
    else:
        warn("Skipping main figures (--skip-main)")
        overall_status.append(('main_figures', 'skip'))

    # ── Step 2: Figure 5-8 ──────────────────────────────────────────────────
    if not args.skip_supplementary:
        hdr("Step 2a/4 — Supplementary figures (Fig 5-8)")
        rc = run_subscript('paper_plot_supplementary.py', common_args)
        if rc != 0:
            fail(f"paper_plot_supplementary.py exit={rc}")
            overall_status.append(('supp_figures', 'fail'))
        else:
            ok("Supplementary figures generated")
            overall_status.append(('supp_figures', 'ok'))
    else:
        warn("Skipping supplementary figures (--skip-supplementary)")
        overall_status.append(('supp_figures', 'skip'))

    # ── Step 2b: Fig 9 carbon ───────────────────────────────────────────────
    if not args.skip_supplementary:
        hdr("Step 2b/4 — Carbon figure (Fig 9)")
        rc = run_subscript('paper_plot_carbon.py', common_args)
        if rc != 0:
            fail(f"paper_plot_carbon.py exit={rc}")
            overall_status.append(('carbon_figure', 'fail'))
        else:
            ok("Carbon figure generated")
            overall_status.append(('carbon_figure', 'ok'))

    # ── Step 2f: Grafici dettagliati per-regime (uni/bal/con) ───────────────
    # Stile analyze_three_way portato a NDJSON/6-cluster/3-arm. Output in
    # <figs>/detailed/{uni,bal,con}/ + 3 figure SLO cross-regime. Pesante
    # (~10min: parsa i k6_events.json del rep rappresentativo di ogni cella).
    if not args.skip_detailed:
        hdr("Step 2f/4 — Detailed per-regime figures (paper_plot_detailed)")
        det_args = ['--results-dir', str(args.results_dir),
                    '--output-dir', str(figs_dir / 'detailed')]
        rc = run_subscript('paper_plot_detailed.py', det_args)
        if rc != 0:
            warn(f"paper_plot_detailed.py exit={rc} (best-effort, non fatal)")
            overall_status.append(('detailed_figures', 'fail'))
        else:
            ok("Detailed per-regime figures generated")
            overall_status.append(('detailed_figures', 'ok'))
    else:
        warn("Skipping detailed figures (--skip-detailed)")
        overall_status.append(('detailed_figures', 'skip'))

    # ── Step 2c: Figs 10-11 metastable ──────────────────────────────────────
    # Best-effort: serve almeno un run "naive" dmos-l2 (vecchio). Se non c'e',
    # lo step viene saltato senza fallire la pipeline.
    if not args.skip_supplementary:
        hdr("Step 2c/4 — Metastable failure figures (Fig 10-11)")
        meta_args = [
            '--results-dir', str(args.results_dir),
            '--output-dir', str(figs_dir),
            '--formats'] + args.formats + ['--dpi', str(args.dpi), '--auto']
        rc = run_subscript('paper_plot_metastable.py', meta_args)
        if rc != 0:
            warn(f"paper_plot_metastable.py exit={rc} (best-effort, non fatal)")
            overall_status.append(('metastable_figs', 'skip'))
        else:
            ok("Metastable failure figures generated")
            overall_status.append(('metastable_figs', 'ok'))

    # ── Step 2d: Study A — skew robustness (p95/SLO vs demand skew) ──────────
    # Best-effort: servono i run <arm>__<regime> (dmos-off/l2/l1l2 × uni/bal/con).
    # Se non ci sono ancora, lo step viene saltato senza far fallire la pipeline.
    if not args.skip_supplementary:
        hdr("Step 2d/4 — Study A: skew robustness figure")
        skew_args = [
            '--results-dir', str(args.results_dir),
            '--output-dir', str(figs_dir),
        ]
        rc = run_subscript('paper_plot_skew_robustness.py', skew_args)
        if rc != 0:
            warn(f"paper_plot_skew_robustness.py exit={rc} "
                 f"(best-effort: mancano i run <arm>__<regime>?)")
            overall_status.append(('skew_robustness', 'skip'))
        else:
            ok("Skew robustness figure generated")
            overall_status.append(('skew_robustness', 'ok'))

    # ── Step 2e: E3 — profili di pesi (latency/capacity/carbon-first vs nca) ──
    # Best-effort: servono i run dmos-<profilo>__bal (paper_ablation_compare.py
    # --arms per altri arm). Produce figura e tabelle LaTeX in ablation/.
    if not args.skip_tables:
        hdr("Step 2e/4 — E3: objective ablation")
        ablation_dir = paper_dir / 'ablation'
        ablation_dir.mkdir(parents=True, exist_ok=True)
        abl_args = [
            '--results-dir', str(args.results_dir),
            '--output-dir', str(ablation_dir),
        ]
        rc = run_subscript('paper_ablation_compare.py', abl_args)
        if rc != 0:
            warn(f"paper_ablation_compare.py exit={rc} "
                 f"(best-effort: mancano i run dmos-<profilo>__bal?)")
            overall_status.append(('objective_ablation', 'skip'))
        else:
            ok("Objective ablation tables+figure generated")
            overall_status.append(('objective_ablation', 'ok'))

    # ── Step 3: Tables ──────────────────────────────────────────────────────
    if not args.skip_tables:
        hdr("Step 3/4 — Tables 1-2 (LaTeX + markdown)")
        tbl_args = [
            '--results-dir', str(args.results_dir),
            '--output-dir', str(tables_dir),
            '--baseline', args.baseline,
        ]
        if args.scenarios:
            tbl_args += ['--scenarios'] + args.scenarios
        rc = run_subscript('paper_stat_compare.py', tbl_args)
        if rc != 0:
            fail(f"paper_stat_compare.py exit={rc}")
            overall_status.append(('tables', 'fail'))
        else:
            ok("Tables generated")
            overall_status.append(('tables', 'ok'))
    else:
        warn("Skipping tables (--skip-tables)")
        overall_status.append(('tables', 'skip'))

    # ── Step 4: HTML report ─────────────────────────────────────────────────
    if not args.no_report:
        hdr("Step 4/4 — HTML report")
        rep_args = [
            '--paper-dir', str(paper_dir),
            '--results-dir', str(args.results_dir),
        ]
        rc = run_subscript('paper_report.py', rep_args)
        if rc != 0:
            fail(f"paper_report.py exit={rc}")
            overall_status.append(('report', 'fail'))
        else:
            ok(f"HTML report: {paper_dir / 'paper_report.html'}")
            overall_status.append(('report', 'ok'))
    else:
        warn("Skipping report (--no-report)")
        overall_status.append(('report', 'skip'))

    # ── Riepilogo ───────────────────────────────────────────────────────────
    hdr("Summary")
    for name, status in overall_status:
        if status == 'ok':
            ok(f"{name:18s} ✓")
        elif status == 'fail':
            fail(f"{name:18s} ✗")
        else:
            warn(f"{name:18s} (skipped)")

    n_fail = sum(1 for _, s in overall_status if s == 'fail')
    print()
    if n_fail == 0:
        print(f"  {C_GREEN}{C_BOLD}All artifacts ready in:{C_RESET} {paper_dir}")
        print(f"\n  Open in browser:")
        print(f"    file:///{(paper_dir / 'paper_report.html').as_posix()}")
        print(f"\n  Include nel paper LaTeX (assumendo working dir = paper_artifacts/{folder_name}/):")
        print(f"    \\begin{{figure}}[!t]")
        print(f"      \\centering")
        print(f"      \\includegraphics[width=\\linewidth]{{figures/fig1_replicas_heatmap.pdf}}")
        print(f"      \\caption{{...}}")
        print(f"      \\label{{fig:replicas-trajectory}}")
        print(f"    \\end{{figure}}")
        print(f"    \\input{{tables/table1_metrics.tex}}")
        print(f"    \\input{{tables/table2_stat_tests.tex}}\n")
        return 0
    else:
        fail(f"{n_fail} step(s) hanno fallito. Vedi log sopra.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
