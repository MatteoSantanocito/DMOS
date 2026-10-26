#!/usr/bin/env python3
"""
paper_plot_trajectories.py — Paper Figures 1-4 (paper-critical)
================================================================

Genera le 4 figure principali del paper a partire dai run packaging-ed da
`run_flash_crowd.sh`. Ogni figure ha PDF (vector, per LaTeX) + PNG (raster,
per slide/poster).

Output (in --output-dir, default: paper_artifacts/<timestamp>/figures/):
  - fig1_replicas_trajectory.{pdf,png}
       4-panel (1 per scenario), 6 linee per panel (1 per cluster).
       Mostra come DMOS sposta capacita' geograficamente nel tempo.
  - fig2_p95_latency_overlay.{pdf,png}
       1 grafico, 4 linee (1 per scenario), p95 vs tempo. Bande fasi sullo
       sfondo. IL grafico killer per H1.
  - fig3_error_rate_overlay.{pdf,png}
       1 grafico, 4 linee, error rate %, log scale opzionale. Evidence per H2.
  - fig4_latency_cdf_per_phase.{pdf,png}
       5-panel (1 per fase), CDF di latency, 4 curve per panel (1 per scenario).
       Mostra tail behavior: dove DMOS-l1l2 batte gli altri.

Usage
-----
    # Auto-discovery: trova run in experiments/results/, genera tutto
    python paper_plot_trajectories.py

    # Custom paths
    python paper_plot_trajectories.py \\
        --results-dir experiments/results \\
        --output-dir paper_artifacts/v1/figures \\
        --scenarios dmos-off dmos-l1 dmos-l1l2

    # PDF only (skippa PNG):
    python paper_plot_trajectories.py --formats pdf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Import shared library
sys.path.insert(0, str(Path(__file__).parent))
from paper_lib import (
    SCENARIOS, SCENARIO_LABELS, SCENARIO_COLORS, SCENARIO_HATCHES,
    CLUSTER_COLORS, CLUSTER_REGIONS, PHASE_DEFS, PHASES,
    Run, ScenarioGroup,
    discover_runs, setup_matplotlib, shade_phases, save_figure,
    bin_timeline, BIN_SEC_DEFAULT, print_group_summary,
)


# ── Figure 1: Replica trajectory per cluster, 4-panel ──────────────────────

def plot_fig1_replicas_trajectory(groups: Dict[str, ScenarioGroup],
                                  out_dir: Path,
                                  formats=('pdf', 'png'),
                                  clean: bool = False,
                                  name: str | None = None) -> List[Path]:
    """
    Per-service × per-cluster replica heatmap durante il peak.

    1 riga × N colonne (una per arm). Ogni panel:
      - Y: servizi managed (frontend, productcatalog, cart, ...)
      - X: cluster (ms-cluster-01 ... ms-cluster-06)
      - Colore = mean replicas in fase peak (media sui rep)
      - Annotazione = valore numerico

    Perche' sostituisce la vecchia trajectory: il sum-over-services
    nascondeva l'asimmetria di placement di L1. Con questa heatmap si
    vede direttamente DOVE l'arm posiziona ogni servizio — il segnale
    principale del livello L1 multi-obiettivo.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from paper_lib import load_per_service_per_cluster

    n_scenarios = len(groups)
    if n_scenarios == 0:
        print("  [fig1] no scenarios → skip")
        return []

    # Snapshot phase: usiamo SPIKE (180-300s), non PEAK.
    # Motivo: durante il peak entrambi L2 e L1+L2 hanno gia' raggiunto la
    # configurazione finale di repliche (~10 su cluster-01), quindi visualmente
    # appaiono identici e nascondono il vero contributo di L1. Durante lo spike
    # invece L2 sta ancora rampando reattivamente mentre L1+L2 ha gia' piazzato
    # le repliche dove serve — il differenziale di timing diventa visibile.
    snapshot_phase = 'spike'
    snap_window = next(((tt0, tt1) for (tt0, tt1, n, _) in PHASE_DEFS
                        if n == snapshot_phase),
                      (180, 300))
    t0, t1 = snap_window

    # ── Step 1: per ogni arm, aggrega per-service per-cluster ────────────────
    # arm_matrix[scenario] = DataFrame (rows=service, cols=cluster) di mean replicas
    arm_matrix: Dict[str, pd.DataFrame] = {}
    all_services: List[str] = []
    all_clusters: List[str] = []

    for scenario, group in groups.items():
        per_run_matrices: List[pd.DataFrame] = []
        for run in group.runs:
            jsonl = run.run_dir / 'dmos_live.jsonl'
            if not jsonl.is_file():
                continue
            df = load_per_service_per_cluster(jsonl)
            if df is None or df.empty:
                continue
            # Filtra peak phase
            sub = df[(df['elapsed_s'] >= t0) & (df['elapsed_s'] < t1)]
            if sub.empty:
                continue
            # Media sul tempo per ogni (service, cluster)
            mat = (sub.groupby(['service', 'cluster'])['replicas']
                      .mean().unstack(fill_value=0))
            per_run_matrices.append(mat)

        if not per_run_matrices:
            print(f"  [fig1] {scenario}: no per-service data → skip")
            continue

        # Media sui rep, allineando index/colonne
        all_idx = sorted(set().union(*(m.index for m in per_run_matrices)))
        all_col = sorted(set().union(*(m.columns for m in per_run_matrices)))
        aligned = [m.reindex(index=all_idx, columns=all_col, fill_value=0)
                   for m in per_run_matrices]
        mean_mat = sum(aligned) / len(aligned)
        arm_matrix[scenario] = mean_mat
        all_services = sorted(set(all_services) | set(all_idx))
        all_clusters = sorted(set(all_clusters) | set(all_col))

    if not arm_matrix:
        print("  [fig1] no data available across any arm → skip")
        return []

    # Allineo tutte le matrici allo stesso shape per confronto visivo
    for sc, mat in arm_matrix.items():
        arm_matrix[sc] = mat.reindex(index=all_services,
                                     columns=all_clusters, fill_value=0)

    # ── Step 2: range colore comune per confronto cross-panel ───────────────
    vmax = max(mat.values.max() for mat in arm_matrix.values())
    vmax = max(1.0, vmax)
    cmap = plt.get_cmap('YlOrRd')

    # ── Step 3: layout 1xN panels + colorbar ────────────────────────────────
    n = len(arm_matrix)
    fig = plt.figure(figsize=(5.0 * n + 0.7, 0.65 * len(all_services) + 1.8))
    gs = gridspec.GridSpec(1, n + 1, figure=fig, wspace=0.08,
                           width_ratios=[1.0] * n + [0.04])

    # Cluster pretty labels (region code)
    cluster_labels = [CLUSTER_REGIONS.get(c, c).split(' ')[0]
                      if c in CLUSTER_REGIONS else c.replace('ms-cluster-', 'C')
                      for c in all_clusters]

    panel_idx = 0
    im_last = None
    for scenario, group in groups.items():
        if scenario not in arm_matrix:
            continue
        ax = fig.add_subplot(gs[0, panel_idx])
        mat = arm_matrix[scenario]
        data = mat.values

        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
        im_last = im

        # Annotazioni nelle celle (valore replicas)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                # contrasto: testo bianco se cella scura, nero se chiara
                txt_color = 'white' if val > 0.55 * vmax else 'black'
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                        fontsize=9, color=txt_color)

        ax.set_xticks(range(len(all_clusters)))
        ax.set_xticklabels(cluster_labels, rotation=30, ha='right', fontsize=9)
        if panel_idx == 0:
            ax.set_yticks(range(len(all_services)))
            ax.set_yticklabels(all_services, fontsize=9)
            ax.set_ylabel("Service")
        else:
            ax.set_yticks(range(len(all_services)))
            ax.set_yticklabels([])
        ax.set_xlabel("Cluster")
        ax.set_title(f"{SCENARIO_LABELS.get(scenario, scenario)}  (n={group.n_repetitions})",
                     fontsize=11, fontweight='bold', pad=10,
                     color=SCENARIO_COLORS.get(scenario, 'black'))
        # Totale per cluster SOTTO l'asse X (era sopra → sovrapposizione col title).
        # Posizionato a y = len(services) + 0.4 = subito sotto l'ultima riga.
        col_totals = data.sum(axis=0)
        bottom_y = data.shape[0] + 0.05
        for j, tot in enumerate(col_totals):
            ax.text(j, bottom_y, f"Σ{tot:.0f}", ha='center', va='top',
                    fontsize=8, color='#555', style='italic')
        # Aggiorna xlabel per fare spazio ai Σ
        ax.set_xlabel(f"Cluster   (Σ = total replicas in column)")
        panel_idx += 1

    # Colorbar condivisa
    cax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im_last, cax=cax)
    cbar.set_label("Mean replicas (peak phase)", fontsize=10)

    if not clean:
        fig.suptitle(
            f"Figure 1 — Per-service × per-cluster replica placement during {snapshot_phase} phase\n"
            f"(mean across $n$ repetitions; "
            f"L1 places replicas on cluster-01 immediately, L2 is still ramping reactively → "
            f"the timing differential drives the latency gap)",
            fontsize=11, fontweight='bold', y=1.04,
        )

    return save_figure(fig, out_dir, name or 'fig1_replicas_heatmap', formats)


# ── Figure 2: p95 latency overlay ───────────────────────────────────────────

def plot_fig2_p95_overlay(groups: Dict[str, ScenarioGroup],
                          out_dir: Path,
                          formats=('pdf', 'png'),
                          clean: bool = False,
                          name: str | None = None) -> List[Path]:
    """
    Singolo grafico, X=tempo, Y=p95 latency.
    Una linea per scenario, colore = SCENARIO_COLORS.

    Nota: il p95 nel tempo non e' direttamente nel k6_summary (che lo aggrega
    per fase). Lo ricostruiamo dal k6_events.json se presente, oppure dal
    timeline.csv (proxy = `hubble_reqs_rate` se non c'e' latency timestream).

    Approccio: estraiamo p95 per *bin* (es. 5s) dai k6 events. Se nessun
    events.json, fallback su per-phase summary (5 punti totali).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    plotted_any = False
    for scenario, group in groups.items():
        # Strategia: leggi k6_events.json (JSONL) e calcola p95 per bin di 10s.
        # Aggrega tutte le ripetizioni in un unico DataFrame, poi groupby bin.
        all_events: List[pd.DataFrame] = []
        for run in group.runs:
            events_path = run.run_dir / 'k6_events.json'
            if events_path.is_file():
                df = _load_k6_events(events_path)
                if df is not None and not df.empty:
                    all_events.append(df)

        if all_events:
            df = pd.concat(all_events, ignore_index=True)
            # Bin 10s, p95 per bin
            df['bin_s'] = (df['elapsed_s'] // 10 * 10).astype(int)
            # Conta sample per bin; richiediamo >= MIN_SAMPLES_PER_BIN per
            # evitare p95 instabile alla coda del test (cooldown finale dove
            # k6 sta chiudendo i VU e i bin hanno pochissimi sample, generando
            # spike artificiali a >10s che confondono il reviewer).
            counts = df.groupby('bin_s').size()
            MIN_SAMPLES_PER_BIN = 50
            p95 = df.groupby('bin_s')['value'].quantile(0.95).reset_index()
            p95 = p95[p95['bin_s'].isin(counts[counts >= MIN_SAMPLES_PER_BIN].index)]
            # Trim ultimi 30s del test: i VU stanno chiudendo, p95 spurio.
            test_end_s = max(b for b, *_ in PHASE_DEFS) if False else PHASE_DEFS[-1][1]
            p95 = p95[p95['bin_s'] < test_end_s - 30]
            ax.plot(p95['bin_s'] / 60.0, p95['value'],
                    color=group.color, label=group.label,
                    linewidth=2.0, alpha=0.95)
            plotted_any = True
        else:
            # Fallback: usa per-phase summary se k6_events non disponibile
            phase_x = []
            phase_y = []
            for t0, t1, name, _ in PHASE_DEFS:
                vals = group.metric_values_per_run(f'results.per_phase.{name}.p95')
                if len(vals) > 0:
                    phase_x.append((t0 + t1) / 2 / 60.0)
                    phase_y.append(vals.mean())
            if phase_x:
                ax.plot(phase_x, phase_y, marker='o',
                        color=group.color, label=group.label,
                        linewidth=2.0, alpha=0.95)
                plotted_any = True

    if not plotted_any:
        print("  [fig2] nessun dato disponibile → skip")
        return []

    shade_phases(ax, x_unit='min', y_frac_label=0.97)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_xlim(0, 15)
    ax.set_yscale('log')
    ax.legend(loc='best' if clean else 'upper right',
              framealpha=0.9, fontsize=12 if clean else 10)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    if not clean:
        ax.set_title(
            "Figure 2 — End-to-end p95 latency over time per scheduling strategy\n"
            "(log scale; bands = test phases; lower is better)",
            fontsize=12, fontweight='bold', pad=12,
        )

    return save_figure(fig, out_dir, name or 'fig2_p95_latency_overlay', formats)


# ── Figure 3: SLO violation rate over time ──────────────────────────────────
#
# Sostituisce la vecchia "HTTP error rate" perche' nei nostri esperimenti
# (k6 con timeout default 60s, no service crash) l'error rate e' ~0% in
# tutti gli arm. Il segnale vero e' la TAIL LATENCY: la quota di request
# che superano lo SLO target (default 1000 ms, "interactive web service",
# Dean & Barroso 2013). Cosi' rendiamo visibile il differenziale del 90%
# che la sola p95 nasconde dietro un singolo numero scalare.

SLO_THRESHOLD_MS = float(os.environ.get("DMOS_SLO_MS", "1000"))  # soglia SLO (override via env; paper usa 2000)


def plot_fig3_slo_violation_overlay(groups: Dict[str, ScenarioGroup],
                                    out_dir: Path,
                                    formats=('pdf', 'png'),
                                    slo_ms: float = SLO_THRESHOLD_MS,
                                    bin_sec: int = 10,
                                    clean: bool = False,
                                    name: str | None = None) -> List[Path]:
    """
    X=tempo, Y=% request con latency > slo_ms (in bin di bin_sec secondi).
    Una linea per scenario.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    plotted_any = False
    for scenario, group in groups.items():
        all_events: List[pd.DataFrame] = []
        for run in group.runs:
            events_path = run.run_dir / 'k6_events.json'
            if events_path.is_file():
                df = _load_k6_events(events_path)
                if df is not None and not df.empty:
                    all_events.append(df)

        if all_events:
            df = pd.concat(all_events, ignore_index=True)
            df['bin_s'] = (df['elapsed_s'] // bin_sec * bin_sec).astype(int)
            df['violates'] = (df['value'] > slo_ms).astype(int)
            agg = df.groupby('bin_s').agg(viol=('violates', 'mean'),
                                          total=('violates', 'count')).reset_index()
            # Soglia sample minimi: 50 anziche' 20 + trim ultimi 30s — evita
            # gli spike spurii a fine test (k6 sta chiudendo i VU, bin
            # praticamente vuoti, viol_rate rumoroso o saturo a 100%).
            agg = agg[agg['total'] >= 50]
            test_end_s = PHASE_DEFS[-1][1]
            agg = agg[agg['bin_s'] < test_end_s - 30]
            ax.plot(agg['bin_s'] / 60.0, agg['viol'] * 100,
                    color=group.color, label=group.label,
                    linewidth=2.0, alpha=0.95)
            plotted_any = True

    if not plotted_any:
        print("  [fig3] nessun dato disponibile → skip")
        return []

    shade_phases(ax, x_unit='min', y_frac_label=0.97)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"SLO violations (% requests > {int(slo_ms)} ms)")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, None)
    ax.legend(loc='best' if clean else 'upper right',
              framealpha=0.9, fontsize=12 if clean else 10)
    ax.grid(True, alpha=0.3, linestyle='--')
    if not clean:
        ax.set_title(
            f"Figure 3 — SLO violation rate over time per scheduling strategy\n"
            f"(SLO = {int(slo_ms)} ms; rolling {bin_sec}s bins; lower is better)",
            fontsize=12, fontweight='bold', pad=12,
        )

    return save_figure(fig, out_dir, name or 'fig3_slo_violation_overlay', formats)


# ── Figure 3b: SLO compliance per phase (bar chart) ─────────────────────────

def plot_fig3b_slo_per_phase_bars(groups: Dict[str, ScenarioGroup],
                                  out_dir: Path,
                                  formats=('pdf', 'png'),
                                  slo_ms: float = SLO_THRESHOLD_MS) -> List[Path]:
    """
    Bar chart raggruppato: 5 phase sull'asse X, 3 arm (barre side-by-side).
    Y = % request con latency > slo_ms. Errorbar = CI95 bootstrap sui rep.
    """
    import matplotlib.pyplot as plt

    # Calcolo per ogni (arm, phase, run) la % violazioni
    phases = [p[2] for p in PHASE_DEFS]
    # data[scenario][phase] = array di violation% (uno per run)
    data: Dict[str, Dict[str, List[float]]] = {
        sc: {ph: [] for ph in phases} for sc in groups.keys()
    }

    for scenario, group in groups.items():
        for run in group.runs:
            events_path = run.run_dir / 'k6_events.json'
            if not events_path.is_file():
                continue
            df = _load_k6_events(events_path)
            if df is None or df.empty:
                continue
            for t0, t1, ph, _ in PHASE_DEFS:
                sub = df[(df['elapsed_s'] >= t0) & (df['elapsed_s'] < t1)]
                if len(sub) == 0:
                    continue
                pct = 100.0 * (sub['value'] > slo_ms).mean()
                data[scenario][ph].append(pct)

    # Calcolo mean + CI95 per ogni cella
    from paper_lib import bootstrap_ci
    means: Dict[str, List[float]] = {sc: [] for sc in groups.keys()}
    err_lo: Dict[str, List[float]] = {sc: [] for sc in groups.keys()}
    err_hi: Dict[str, List[float]] = {sc: [] for sc in groups.keys()}
    for scenario in groups.keys():
        for ph in phases:
            vals = np.asarray(data[scenario][ph])
            if len(vals) == 0:
                m, lo, hi = 0.0, 0.0, 0.0
            elif len(vals) == 1:
                m, lo, hi = float(vals[0]), float(vals[0]), float(vals[0])
            else:
                m, lo, hi = bootstrap_ci(vals)
            means[scenario].append(m)
            err_lo[scenario].append(max(0.0, m - lo))
            err_hi[scenario].append(max(0.0, hi - m))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    n_arms = len(groups)
    bar_w = 0.8 / max(n_arms, 1)
    x = np.arange(len(phases))

    from paper_lib import SCENARIO_HATCHES
    for i, (scenario, group) in enumerate(groups.items()):
        offset = (i - (n_arms - 1) / 2) * bar_w
        ax.bar(x + offset, means[scenario], bar_w,
               yerr=[err_lo[scenario], err_hi[scenario]],
               color=group.color, label=group.label,
               hatch=SCENARIO_HATCHES.get(scenario, ''),
               edgecolor='black', linewidth=0.6,
               error_kw=dict(ecolor='black', lw=1.0, capsize=3))

    ax.set_xticks(x)
    ax.set_xticklabels([ph.capitalize() for ph in phases])
    ax.set_xlabel("Test phase")
    ax.set_ylabel(f"SLO violations (% requests > {int(slo_ms)} ms)")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_title(
        f"Figure 3b — SLO compliance per test phase (SLO = {int(slo_ms)} ms)\n"
        f"Error bars: CI95 bootstrap over repetitions; lower is better",
        fontsize=12, fontweight='bold', pad=12,
    )

    return save_figure(fig, out_dir, 'fig3b_slo_per_phase_bars', formats)


# Alias backward-compat: il vecchio nome e' usato in paper_generate_all
plot_fig3_error_rate_overlay = plot_fig3_slo_violation_overlay


# ── Figure 4: CDF latency per phase ─────────────────────────────────────────

def plot_fig4_cdf_per_phase(groups: Dict[str, ScenarioGroup],
                            out_dir: Path,
                            formats=('pdf', 'png')) -> List[Path]:
    """
    Griglia 1x5 (1 panel per fase). Ogni panel ha N curve CDF (1 per scenario).
    Asse X: latency (ms), log scale. Asse Y: P(latency <= X), [0,1].
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)

    for i, (t0, t1, phase, _) in enumerate(PHASE_DEFS):
        ax = axes[i]
        plotted = False
        for scenario, group in groups.items():
            # Aggrega latency values di tutti i run nella fase
            vals = []
            for run in group.runs:
                events_path = run.run_dir / 'k6_events.json'
                if events_path.is_file():
                    df = _load_k6_events(events_path)
                    if df is None or df.empty:
                        continue
                    in_phase = df[(df['elapsed_s'] >= t0) & (df['elapsed_s'] < t1)]
                    vals.extend(in_phase['value'].tolist())
            if vals:
                vals_arr = np.sort(np.asarray(vals))
                ax.plot(vals_arr,
                        np.arange(1, len(vals_arr) + 1) / len(vals_arr),
                        color=group.color, label=group.label,
                        linewidth=1.8, alpha=0.95)
                plotted = True

        ax.set_title(f"{phase.capitalize()}\n({t0}-{t1}s)", fontsize=11)
        ax.set_xlabel("Latency (ms)")
        if i == 0:
            ax.set_ylabel("P(X ≤ latency)")
        ax.set_xscale('log')
        ax.set_ylim(0, 1.0)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        # Linee verticali alle soglie SLO: chiunque legga la figura vede
        # immediatamente dove ogni CDF "taglia" lo SLO (ordinata = compliance).
        ax.axvline(SLO_THRESHOLD_MS, color='gray', linestyle=':',
                   linewidth=1.0, alpha=0.7)
        ax.axvline(2 * SLO_THRESHOLD_MS, color='gray', linestyle=':',
                   linewidth=0.8, alpha=0.5)
        if i == 0:
            # Annotazione leggera solo sul primo panel per non sporcare
            ax.text(SLO_THRESHOLD_MS, 0.02, f' SLO\n {int(SLO_THRESHOLD_MS)}ms',
                    fontsize=7, color='gray', rotation=0, va='bottom', ha='left')
        if plotted and i == 4:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Figure 4 — Cumulative distribution of end-to-end latency, per test phase\n"
        "(log-scaled x-axis; curves further to the left = better)",
        fontsize=12, fontweight='bold', y=1.05,
    )

    return save_figure(fig, out_dir, 'fig4_latency_cdf_per_phase', formats)


# ── Helper: load k6 JSON events (JSONL format) ──────────────────────────────

def _load_k6_events(path: Path, include_status: bool = False) -> pd.DataFrame:
    """
    Carica `k6_events.json` (JSONL) e ritorna un DataFrame con:
      - elapsed_s : secondi da inizio test
      - value     : valore della metrica (per http_req_duration: ms)
      - status    : status code HTTP (opzionale, se include_status)

    k6 events.json e' JSONL: ogni riga e' un JSON object tipo:
      {"type":"Point","data":{"time":"2026-01-01T00:00:00.123Z","value":123.4,
                              "tags":{"status":"200",...}},"metric":"http_req_duration"}

    OTTIMIZZAZIONI:
    - Cache parquet su disco accanto al .json (5 MB vs 880 MB). La prima
      chiamata fa il parsing pesante (~30-60s per file da 880MB), tutte le
      successive sono istantanee.
    - Skip rapido delle righe via substring-match prima di json.loads.
    - Parser ISO 8601 manuale, ~20x piu' veloce di pd.to_datetime().
    """
    import json
    import os
    from datetime import datetime

    # Cache: prova parquet (se pyarrow/fastparquet installati), altrimenti pickle
    # compresso (sempre disponibile, ~stessa dimensione, leggermente piu' lento).
    def _has_parquet_engine() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except ImportError:
            try:
                import fastparquet  # noqa: F401
                return True
            except ImportError:
                return False

    use_parquet = _has_parquet_engine()
    if use_parquet:
        cache_suffix = '_with_status.parquet' if include_status else '.parquet'
    else:
        cache_suffix = '_with_status.pkl.gz' if include_status else '.pkl.gz'
    cache_path = path.with_suffix(cache_suffix)

    # Cache hit: leggi (~50ms vs 30+s)
    if cache_path.is_file():
        try:
            src_mtime = path.stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            if cache_mtime >= src_mtime:
                if use_parquet:
                    return pd.read_parquet(cache_path)
                else:
                    return pd.read_pickle(cache_path, compression='gzip')
        except Exception:
            pass  # cache corrotta -> ricostruisci

    # Cache miss: scan completo
    print(f"  [k6cache] parsing {path.name} (~880MB, primo accesso)...", flush=True)

    needle = b'"metric":"http_req_duration"'
    times = []
    values = []
    statuses = []
    t_start_unix = None

    def _parse_iso(s: str) -> float:
        # Format esatto k6: "2026-05-25T19:28:13.605518886Z"
        # Tronchiamo i nanosecondi a microsecondi (Python max precisione)
        try:
            # Fast-path: assumendo formato fisso
            if s.endswith('Z'):
                s = s[:-1]
            # tronca ns -> us
            if '.' in s:
                base, frac = s.split('.', 1)
                frac = frac[:6]
                s = f"{base}.{frac}"
            dt = datetime.fromisoformat(s)
            return dt.timestamp()
        except Exception:
            try:
                return pd.to_datetime(s).timestamp()
            except Exception:
                return float('nan')

    try:
        with path.open('rb') as f:
            for line in f:
                # Fast-path: la maggior parte delle righe non e' utile
                if needle not in line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get('type') != 'Point':
                    continue
                d = ev.get('data', {})
                t_iso = d.get('time')
                if not t_iso:
                    continue
                ts = _parse_iso(t_iso)
                if ts != ts:  # NaN check
                    continue
                if t_start_unix is None:
                    t_start_unix = ts
                times.append(ts - t_start_unix)
                values.append(float(d.get('value', 0.0)))
                if include_status:
                    s = d.get('tags', {}).get('status', '0')
                    try:
                        statuses.append(int(s))
                    except Exception:
                        statuses.append(0)

        if not times:
            return None

        df = pd.DataFrame({'elapsed_s': times, 'value': values})
        if include_status:
            df['status'] = statuses

        # Salva cache (non-fatale se fallisce)
        try:
            if use_parquet:
                df.to_parquet(cache_path, index=False, compression='snappy')
                fmt = 'parquet'
            else:
                df.to_pickle(cache_path, compression='gzip')
                fmt = 'pickle.gz'
            sz = os.path.getsize(cache_path) / 1024 / 1024
            print(f"  [k6cache] saved {cache_path.name} ({fmt}, {sz:.1f} MB, {len(df):,} rows)",
                  flush=True)
        except Exception as e:
            print(f"  [k6cache] warning: cache save failed: {e}", flush=True)

        return df
    except Exception as e:
        print(f"  [warn] failed to load k6 events {path.name}: {e}")
        return None


# ── Main CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera Figure 1-4 paper-quality del paper DMOS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--results-dir', type=Path,
        default=Path(__file__).parent / 'results',
        help='Directory con i run dirs <YYYYMMDD>_<HHMMSS>_<scenario>/',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=None,
        help='Directory output (default: paper_artifacts/<timestamp>/figures/)',
    )
    parser.add_argument(
        '--scenarios', nargs='+', default=None,
        help='Filtra solo questi scenari (default: tutti scoperti)',
    )
    parser.add_argument(
        '--formats', nargs='+', default=['pdf', 'png'],
        choices=['pdf', 'png', 'svg'],
        help='Formati output (default: pdf png)',
    )
    parser.add_argument(
        '--dpi', type=int, default=600,
        help='DPI per PNG/PDF raster (default: 600)',
    )
    args = parser.parse_args()

    # Style setup
    setup_matplotlib(target_dpi=args.dpi)

    # Output dir auto: paper_artifacts/<timestamp>/figures/
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = Path(__file__).parent.parent / f'paper_artifacts/{ts}/figures'
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discovery
    print(f"\n  Discovering runs in: {args.results_dir}")
    groups = discover_runs(args.results_dir, scenario_filter=args.scenarios,
                           load_timeline=True)
    print_group_summary(groups)
    if not groups:
        print("  No runs found. Bye.")
        return 1

    formats = tuple(args.formats)

    # Genera le 4 figure
    print(f"  Generating figures in: {args.output_dir}\n")
    written = []
    for name, fn in [
        ("Fig 1 (per-service placement heatmap)", plot_fig1_replicas_trajectory),
        ("Fig 2 (p95 latency overlay)",  plot_fig2_p95_overlay),
        ("Fig 3 (SLO violation overlay)", plot_fig3_slo_violation_overlay),
        ("Fig 3b (SLO per-phase bars)",  plot_fig3b_slo_per_phase_bars),
        ("Fig 4 (CDF + SLO lines)",      plot_fig4_cdf_per_phase),
    ]:
        print(f"  {name}...")
        try:
            paths = fn(groups, args.output_dir, formats)
            for p in paths:
                size_kb = p.stat().st_size / 1024
                print(f"    ✓ {p.name}  ({size_kb:.1f} KB)")
            written.extend(paths)
        except Exception as e:
            print(f"    ✗ failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  Done. {len(written)} files written in {args.output_dir}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
