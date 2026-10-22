#!/usr/bin/env python3
"""
paper_plot_detailed.py — Grafici dettagliati "thesis-quality" per regime.
=========================================================================
Porta lo stile di analyze_three_way.py (i grafici ricchi a 3 cluster) al
dataset attuale: k6_events.json (NDJSON), 6 cluster, 3 arm (off/l2/l1l2),
organizzato in 3 SEZIONI per regime — uni / bal / con.

Per OGNI regime genera (in <out>/detailed/<regime>/):
  1. key_metrics.png        — 4 barre: error rate, p95, throughput, fail+dropped
                              (media ± dev su tutte le reps)
  2. latency_timeseries.png — p50/p95/p99 nel tempo (rep rappresentativo)
  3. error_throughput.png   — error rate / throughput / VUs nel tempo
  4. per_cluster.png        — p95 + error per ciascuno dei 6 cluster
  5. error_heatmap.png      — error rate arm×fase  e  arm×cluster (picco)
  6. cdf.png                — CDF latenza globale + per-cluster

E 3 figure cross-regime (in <out>/detailed/):
  - slo_violation.png       — %SLO>soglia (picco), 3 pannelli uni/bal/con
  - slo_per_phase.png       — %SLO per fase × arm, 3 pannelli uni/bal/con
  - skew_robustness.png     — %SLO (picco) vs skew, una linea per arm

Sorgente per i time-series: il rep RAPPRESENTATIVO (mediano per peak-error) di
ogni (regime, arm). Le barre key_metrics usano la MEDIA su tutte le reps.

Uso:
  python experiments/paper_plot_detailed.py
  python experiments/paper_plot_detailed.py --results-dir experiments/results \
      --output-dir paper_artifacts/detailed --slo-ms 2000
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import paper_lib as PL

# ── Config ────────────────────────────────────────────────────────────────────
REGIMES = ['uni', 'bal', 's50', 's65', 's75', 'con']   # ordinati per skew crescente (curva)
ARMS = ['dmos-l1l2', 'dmos-l2', 'dmos-off']        # best → worst (legenda)
ARM_SHORT = {'dmos-l1l2': 'DMOS (L1 + L2)', 'dmos-l2': 'DMOS (L2)', 'dmos-off': 'DMOS OFF'}
LINE_STYLES = {'dmos-l1l2': '-', 'dmos-l2': '-.', 'dmos-off': ':'}


def set_dmos_arm(arm: str) -> None:
    """Usa `arm` (es. dmos-nca) al posto di dmos-l1l2 come braccio DMOS completo."""
    if arm == ARMS[0]:
        return
    old = ARMS[0]
    ARMS[0] = arm
    ARM_SHORT[arm] = PL.ARM_LABELS.get(arm, arm)
    LINE_STYLES[arm] = LINE_STYLES.pop(old)
METRICS_KEEP = ('http_req_duration', 'vus')        # duration → lat+tput+err; vus → carico
BIN_SEC = 5
DPI = 300

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': DPI,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.labelsize': 10, 'legend.fontsize': 9,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15, 'lines.linewidth': 1.2,
})


# ── Parsing k6_events.json (NDJSON) ─────────────────────────────────────────────
def parse_k6_events(path: Path) -> pd.DataFrame:
    """Stream NDJSON → DataFrame. Tiene solo http_req_duration + vus.
    Colonne: elapsed_s, metric, value, cluster, ok(bool), status, phase."""
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '"type":"Point"' not in line:
                continue
            if ('"http_req_duration"' not in line) and ('"vus"' not in line):
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('type') != 'Point':
                continue
            m = o.get('metric')
            if m not in METRICS_KEEP:
                continue
            d = o.get('data', {})
            t = d.get('tags', {})
            rows.append((d.get('time'), m, d.get('value'),
                         t.get('cluster', ''), t.get('expected_response', '') == 'true',
                         t.get('status', ''), t.get('phase', '')))
    if not rows:
        return pd.DataFrame(columns=['elapsed_s', 'metric', 'value', 'cluster', 'ok', 'status', 'phase'])
    df = pd.DataFrame(rows, columns=['time', 'metric', 'value', 'cluster', 'ok', 'status', 'phase'])
    df['dt'] = pd.to_datetime(df['time'], format='ISO8601', utc=True, errors='coerce')
    t0 = df['dt'].min()
    df['elapsed_s'] = (df['dt'] - t0).dt.total_seconds()
    return df.drop(columns=['time', 'dt'])


def dur(df):  return df[df['metric'] == 'http_req_duration']
def vus(df):  return df[df['metric'] == 'vus']


# ── Selezione rep rappresentativo ───────────────────────────────────────────────
def representative_run(group, results_dir):
    """Rep con peak-error MEDIANO (più rappresentativo, non cherry-pick)."""
    scored = []
    for r in group.runs:
        try:
            pe = r.k6_summary['results']['per_phase']['peak']['errorRate']
        except Exception:
            pe = float('nan')
        scored.append((pe, r))
    scored = [s for s in scored if not np.isnan(s[0])]
    if not scored:
        return group.runs[0] if group.runs else None
    scored.sort(key=lambda x: x[0])
    med = scored[len(scored) // 2][1]   # mediano (per n dispari = centro)
    return med


def run_dir_of(results_dir, run):
    return Path(results_dir) / f"{run.timestamp}_{run.scenario}"


# ── Shading fasi ────────────────────────────────────────────────────────────────
def shade(ax):
    for (s, e, _, col) in PL.PHASE_DEFS:
        ax.axvspan(s / 60, e / 60, alpha=0.12, color=col, zorder=0)

def phase_labels(ax):
    for (s, e, name, _) in PL.PHASE_DEFS:
        ax.text((s + e) / 2 / 60, 0.97, name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, alpha=0.5, fontstyle='italic')

def clusters_in(dfs):
    cs = set()
    for d in dfs.values():
        if d is not None and not d.empty:
            cs |= set(dur(d)['cluster'].unique())
    cs.discard('')
    return sorted(cs)


# ══ FIGURE PER-REGIME ═══════════════════════════════════════════════════════════
def fig_key_metrics(group_by_arm, dfs, regime, out):
    """4 barre: error rate / p95(tutte+riuscite) / throughput / fail+dropped.
    Error/throughput/fail = media±dev su tutte le reps (summary); p95 = dal rep
    rappresentativo, separando 'tutte' vs 'solo riuscite' per mostrare il bias."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    arms = [a for a in ARMS if a in group_by_arm]
    x = np.arange(len(arms)); cols = [PL.scenario_color(a) for a in arms]
    labels = [ARM_SHORT[a] for a in arms]

    def collect(path):
        out_m, out_s = [], []
        for a in arms:
            v = group_by_arm[a].metric_values_per_run(path)
            out_m.append(np.mean(v) if len(v) else 0.0)
            out_s.append(np.std(v) if len(v) > 1 else 0.0)
        return np.array(out_m), np.array(out_s)

    # Error rate (peak, %)
    m, s = collect('results.per_phase.peak.errorRate'); m, s = m * 100, s * 100
    ax = axes[0, 0]; b = ax.bar(x, m, 0.6, yerr=s, color=cols, alpha=0.85, capsize=4)
    for bi, v in zip(b, m): ax.text(bi.get_x() + bi.get_width() / 2, v, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Error Rate (peak)'); ax.set_ylabel('Error Rate (%)')

    # p95 (peak, ms): TUTTE le richieste vs SOLO riuscite (dal rep rappresentativo).
    # Le 'tutte' sono deflazionate dai fallimenti-lampo degli arm che scaricano
    # carico → mostriamo entrambe per onestà.
    ax = axes[0, 1]; bw = 0.38; p95all, p95ok = [], []
    for a in arms:
        d = dfs.get(a)
        if d is not None and not d.empty:
            dd = dur(d); dd = dd[dd['phase'] == 'peak']
            okv = dd[dd['ok']]['value']
            p95all.append(np.percentile(dd['value'], 95) if len(dd) else 0)
            p95ok.append(np.percentile(okv, 95) if len(okv) else 0)
        else:
            p95all.append(0); p95ok.append(0)
    b1 = ax.bar(x - bw / 2, p95all, bw, color=cols, alpha=0.45, hatch='//', label='all requests')
    b2 = ax.bar(x + bw / 2, p95ok, bw, color=cols, alpha=0.9, label='successful only')
    for bi, v in zip(b1, p95all): ax.text(bi.get_x() + bi.get_width() / 2, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    for bi, v in zip(b2, p95ok): ax.text(bi.get_x() + bi.get_width() / 2, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_title('p95 Response Latency (peak) — all vs successful'); ax.set_ylabel('p95 (ms)'); ax.legend(fontsize=8)

    # Effective throughput (successful rps, global)
    rps, _ = collect('results.global.avg_rps')
    fr, _ = collect('results.global.http_failure_rate')
    eff = rps * (1 - fr)
    ax = axes[1, 0]; b = ax.bar(x, eff, 0.6, color=cols, alpha=0.85)
    for bi, v in zip(b, eff): ax.text(bi.get_x() + bi.get_width() / 2, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    ax.set_title('Effective Throughput'); ax.set_ylabel('Successful RPS')

    # Failed requests + dropped iterations (count, global)
    tot, _ = collect('results.global.total_requests')
    failed = tot * fr
    drop, _ = collect('validity.dropped_iterations')
    ax = axes[1, 1]; bw = 0.38
    ax.bar(x - bw / 2, failed, bw, color=cols, alpha=0.7, label='Failed requests')
    ax.bar(x + bw / 2, drop, bw, color=cols, alpha=0.35, hatch='//', label='Dropped iterations')
    for i, (fv, dv) in enumerate(zip(failed, drop)):
        ax.text(i - bw / 2, fv, f'{fv:,.0f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + bw / 2, dv, f'{dv:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_title('Failed Requests & Dropped Iterations'); ax.set_ylabel('Count'); ax.legend()

    for ax in axes.flat:
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.grid(True, alpha=0.2, axis='y')
    fig.suptitle(f'Key Metrics Comparison — {PL.regime_label(regime)} (skew {PL.regime_skew(regime, 0.0):.2f})', fontsize=14)
    _save(fig, out, 'key_metrics.png')


def fig_latency_ts(dfs, regime, out):
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
    for qi, (q, ql) in enumerate([(0.50, 'p50'), (0.95, 'p95'), (0.99, 'p99')]):
        ax = axes[qi]
        for a in ARMS:
            d = dfs.get(a)
            if d is None or d.empty: continue
            dd = dur(d).copy(); dd['bin'] = (dd['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = dd.groupby('bin')['value'].quantile(q).reset_index()
            ax.plot(g['bin'] / 60, g['value'], label=ARM_SHORT[a], color=PL.scenario_color(a))
        ax.set_ylabel(f'{ql} (ms)'); ax.set_title(f'{ql} Response Latency')
        ax.legend(); ax.grid(True, alpha=0.2); shade(ax); phase_labels(ax)
    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Latency Time Series — {PL.regime_label(regime)}', fontsize=14, y=1.0)
    _save(fig, out, 'latency_timeseries.png')


def fig_error_throughput(dfs, regime, out):
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=True)
    # Error rate
    ax = axes[0]
    for a in ARMS:
        d = dfs.get(a)
        if d is None or d.empty: continue
        dd = dur(d).copy(); dd['bin'] = (dd['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = dd.groupby('bin')['ok'].agg(['count', 'sum']).reset_index()
        g['err'] = (g['count'] - g['sum']) / g['count'] * 100
        ax.plot(g['bin'] / 60, g['err'], label=ARM_SHORT[a], color=PL.scenario_color(a))
    ax.set_ylabel('Error Rate (%)'); ax.set_title('Error Rate'); ax.legend(); ax.grid(True, alpha=0.2); shade(ax); phase_labels(ax)
    # Throughput (successful)
    ax = axes[1]
    for a in ARMS:
        d = dfs.get(a)
        if d is None or d.empty: continue
        dd = dur(d); ok = dd[dd['ok']].copy(); ok['bin'] = (ok['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = ok.groupby('bin').size().reset_index(name='n'); g['rps'] = g['n'] / BIN_SEC
        ax.plot(g['bin'] / 60, g['rps'], label=ARM_SHORT[a], color=PL.scenario_color(a))
    ax.set_ylabel('Successful RPS'); ax.set_title('Throughput'); ax.legend(); ax.grid(True, alpha=0.2); shade(ax)
    # VUs
    ax = axes[2]
    for a in ARMS:
        d = dfs.get(a)
        if d is None or d.empty: continue
        vv = vus(d)
        if vv.empty: continue
        vv = vv.sort_values('elapsed_s')
        ax.plot(vv['elapsed_s'] / 60, vv['value'], label=ARM_SHORT[a], color=PL.scenario_color(a))
    ax.set_ylabel('VUs'); ax.set_title('Virtual Users'); ax.legend(); ax.grid(True, alpha=0.2); shade(ax)
    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Error, Throughput & VUs — {PL.regime_label(regime)}', fontsize=14, y=1.0)
    _save(fig, out, 'error_throughput.png')


def fig_per_cluster(dfs, regime, out):
    cs = clusters_in(dfs)
    if not cs: return
    n = len(cs)
    fig, axes = plt.subplots(n, 2, figsize=(18, 3.2 * n), sharex='col', squeeze=False)
    err_axes = []; emax = 0.0
    for ci, cl in enumerate(cs):
        cname = PL.CLUSTER_REGIONS.get(cl, cl)
        axL, axR = axes[ci, 0], axes[ci, 1]
        for a in ARMS:
            d = dfs.get(a)
            if d is None or d.empty: continue
            cd = dur(d); cd = cd[cd['cluster'] == cl].copy()
            if cd.empty: continue
            cd['bin'] = (cd['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = cd.groupby('bin')['value'].quantile(0.95).reset_index()
            axL.plot(g['bin'] / 60, g['value'], label=ARM_SHORT[a], color=PL.scenario_color(a), lw=1.0)
            ge = cd.groupby('bin')['ok'].agg(['count', 'sum']).reset_index()
            ge['err'] = (ge['count'] - ge['sum']) / ge['count'] * 100
            axR.plot(ge['bin'] / 60, ge['err'], label=ARM_SHORT[a], color=PL.scenario_color(a), lw=1.0)
            if len(ge): emax = max(emax, float(ge['err'].max()))
        axL.set_ylabel('p95 (ms)'); axL.set_title(f'{cname} — p95 Latency'); axL.legend(fontsize=7); axL.grid(True, alpha=0.2); shade(axL)
        axR.set_ylabel('Error %'); axR.set_title(f'{cname} — Error Rate'); axR.legend(fontsize=7); axR.grid(True, alpha=0.2); shade(axR)
        err_axes.append(axR)
    # Asse Y error CONDIVISO fra i cluster: i cluster a 0% mostrano una linea piatta
    # pulita contro la stessa scala del cluster caldo (non un ±0.04 fuorviante che
    # "sembra vuoto"). Rende ovvio che gli altri cluster sono sani sotto con.
    emax = max(emax, 2.0)
    for axR in err_axes:
        axR.set_ylim(-emax * 0.04, emax * 1.08)
    axes[-1, 0].set_xlabel('Time (min)'); axes[-1, 1].set_xlabel('Time (min)')
    fig.suptitle(f'Per-Cluster Detail — {PL.regime_label(regime)}', fontsize=14, y=1.0)
    _save(fig, out, 'per_cluster.png')


def fig_error_heatmap(dfs, regime, out):
    arms = [a for a in ARMS if dfs.get(a) is not None and not dfs[a].empty]
    if not arms: return
    cs = clusters_in(dfs)
    fig, axes = plt.subplots(1, 2, figsize=(20, 1.4 + 1.1 * len(arms)))
    # arm × phase
    ax = axes[0]; phn = [p[2] for p in PL.PHASE_DEFS]
    M = np.zeros((len(arms), len(phn)))
    for i, a in enumerate(arms):
        dd = dur(dfs[a])
        for j, ph in enumerate(phn):
            pr = dd[dd['phase'] == ph]
            M[i, j] = (~pr['ok']).sum() / len(pr) * 100 if len(pr) else 0
    im = ax.imshow(M, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=max(20, M.max()))
    ax.set_xticks(range(len(phn))); ax.set_xticklabels(phn, rotation=15)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([ARM_SHORT[a] for a in arms])
    for i in range(len(arms)):
        for j in range(len(phn)):
            ax.text(j, i, f'{M[i,j]:.2f}%', ha='center', va='center', fontsize=9, color='white' if M[i, j] > 12 else 'black')
    ax.set_title('Error Rate: Arm × Phase'); fig.colorbar(im, ax=ax, shrink=0.7)
    # arm × cluster (peak)
    ax = axes[1]
    M2 = np.zeros((len(arms), len(cs)))
    for i, a in enumerate(arms):
        dd = dur(dfs[a]); dd = dd[dd['phase'] == 'peak']
        for j, cl in enumerate(cs):
            cr = dd[dd['cluster'] == cl]
            M2[i, j] = (~cr['ok']).sum() / len(cr) * 100 if len(cr) else 0
    im2 = ax.imshow(M2, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=max(20, M2.max()))
    ax.set_xticks(range(len(cs))); ax.set_xticklabels([PL.CLUSTER_REGIONS.get(c, c).split(' ')[0] for c in cs], rotation=20, fontsize=8)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([ARM_SHORT[a] for a in arms])
    for i in range(len(arms)):
        for j in range(len(cs)):
            ax.text(j, i, f'{M2[i,j]:.1f}%', ha='center', va='center', fontsize=8, color='white' if M2[i, j] > 12 else 'black')
    ax.set_title('Error Rate: Arm × Cluster (peak)'); fig.colorbar(im2, ax=ax, shrink=0.7)
    fig.suptitle(f'Error Heatmaps — {PL.regime_label(regime)}', fontsize=14, y=1.04)
    _save(fig, out, 'error_heatmap.png')


def fig_cdf(dfs, regime, out):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    ax = axes[0]
    for a in ARMS:
        d = dfs.get(a)
        if d is None or d.empty: continue
        lat = np.sort(dur(d)['value'].dropna().values)
        if not len(lat): continue
        ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat), label=ARM_SHORT[a], color=PL.scenario_color(a), lw=1.5)
    ax.set_xlim(0, 3000); ax.axhline(0.95, color='gray', ls='--', alpha=0.4); ax.axhline(0.99, color='gray', ls=':', alpha=0.4)
    ax.set_xlabel('Latency (ms)'); ax.set_ylabel('CDF'); ax.set_title('Global Latency CDF'); ax.legend(); ax.grid(True, alpha=0.2)
    ax = axes[1]; cs = clusters_in(dfs)
    styles = LINE_STYLES
    for a in ARMS:
        d = dfs.get(a)
        if d is None or d.empty: continue
        dd = dur(d)
        for cl in cs:
            v = np.sort(dd[dd['cluster'] == cl]['value'].dropna().values)
            if not len(v): continue
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=PL.CLUSTER_COLORS.get(cl, '#888'),
                    ls=styles.get(a, '-'), lw=1.0, alpha=0.8)
    ax.set_xlim(0, 3000); ax.set_xlabel('Latency (ms)'); ax.set_ylabel('CDF')
    ax.set_title('Per-Cluster Latency CDF (line=arm, color=cluster)'); ax.grid(True, alpha=0.2)
    fig.suptitle(f'Latency CDF — {PL.regime_label(regime)}', fontsize=14, y=1.02)
    _save(fig, out, 'cdf.png')


# ══ FIGURE CROSS-REGIME (SLO) ═══════════════════════════════════════════════════
def fig_slo_violation(slo, out, thr_ms):
    """3 pannelli uni/bal/con: %SLO-violation (picco) per arm."""
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(6 * len(REGIMES), 5), sharey=True)
    for k, reg in enumerate(REGIMES):
        ax = axes[k]; arms = [a for a in ARMS if (reg, a) in slo]
        x = np.arange(len(arms)); vals = [slo[(reg, a)]['peak'] for a in arms]
        b = ax.bar(x, vals, 0.6, color=[PL.scenario_color(a) for a in arms], alpha=0.85)
        for bi, v in zip(b, vals): ax.text(bi.get_x() + bi.get_width() / 2, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([ARM_SHORT[a] for a in arms], rotation=10)
        ax.set_title(PL.regime_label(reg)); ax.grid(True, alpha=0.2, axis='y')
        if k == 0: ax.set_ylabel(f'SLO violation (peak, >{thr_ms:.0f}ms o fail) %')
    fig.suptitle(f'SLO Violation (peak phase, threshold {thr_ms:.0f}ms o fail)', fontsize=14, y=1.02)
    _save(fig, out, 'slo_violation.png')


def fig_slo_per_phase(slo, out, thr_ms):
    """3 pannelli uni/bal/con: heatmap %SLO per fase × arm."""
    phn = [p[2] for p in PL.PHASE_DEFS]
    fig, axes = plt.subplots(1, len(REGIMES), figsize=(6.5 * len(REGIMES), 4.5))
    vmax = max((slo[k]['per_phase'][ph] for k in slo for ph in phn), default=20)
    vmax = max(20, vmax)
    for kk, reg in enumerate(REGIMES):
        ax = axes[kk]; arms = [a for a in ARMS if (reg, a) in slo]
        M = np.array([[slo[(reg, a)]['per_phase'][ph] for ph in phn] for a in arms]) if arms else np.zeros((0, len(phn)))
        im = ax.imshow(M, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(phn))); ax.set_xticklabels(phn, rotation=20, fontsize=8)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([ARM_SHORT[a] for a in arms])
        for i in range(len(arms)):
            for j in range(len(phn)):
                ax.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center', fontsize=8, color='white' if M[i, j] > vmax / 2 else 'black')
        ax.set_title(PL.regime_label(reg))
    fig.suptitle(f'SLO Violation per Phase × Arm (%, threshold {thr_ms:.0f}ms o fail)', fontsize=14, y=1.04)
    _save(fig, out, 'slo_per_phase.png')


def fig_skew_robustness(slo, out, thr_ms):
    """%SLO (picco) vs skew, una linea per arm."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for a in ARMS:
        xs, ys = [], []
        for reg in REGIMES:
            if (reg, a) in slo:
                xs.append(PL.regime_skew(reg, 0.0)); ys.append(slo[(reg, a)]['peak'])
        if xs:
            ax.plot(xs, ys, 'o-', label=ARM_SHORT[a], color=PL.scenario_color(a), lw=2, ms=8)
    ax.set_xlabel('Demand skew (hottest-cluster share)'); ax.set_ylabel(f'SLO violation (peak, >{thr_ms:.0f}ms o fail) %')
    ax.set_title('Skew Robustness — SLO violation vs demand concentration')
    ax.legend(); ax.grid(True, alpha=0.25)
    for reg in REGIMES:
        ax.axvline(PL.regime_skew(reg, 0.0), color='gray', ls=':', alpha=0.3)
        ax.text(PL.regime_skew(reg, 0.0), ax.get_ylim()[1] * 0.98, PL.regime_label(reg), rotation=90, va='top', ha='right', fontsize=7, alpha=0.5)
    _save(fig, out, 'skew_robustness.png')


# ── util ────────────────────────────────────────────────────────────────────────
def _save(fig, d, name):
    Path(d).mkdir(parents=True, exist_ok=True)
    p = Path(d) / name
    fig.savefig(p, dpi=DPI); plt.close(fig)
    print(f"    saved {p} ({os.path.getsize(p)/1024:.0f} KB)")


def slo_from_df(df, thr_ms):
    """%richieste che VIOLANO lo SLO, per-fase. Violazione = (durata>soglia) OR (fallita):
    un 503 servito in 26ms NON è 'veloce', è un fallimento → conta come violazione,
    altrimenti gli arm che scaricano carico (fail-fast) verrebbero premiati."""
    d = dur(df)
    res = {'peak': 0.0, 'per_phase': {}}
    for (_, _, ph, _) in PL.PHASE_DEFS:
        pr = d[d['phase'] == ph]
        if len(pr):
            viol = ((pr['value'] > thr_ms) | (~pr['ok'])).sum()
            res['per_phase'][ph] = viol / len(pr) * 100
        else:
            res['per_phase'][ph] = 0.0
    res['peak'] = res['per_phase'].get('peak', 0.0)
    return res


def slo_mean_from_caches(group, results_dir, thr_ms):
    """MEDIA su TUTTE le reps della cella dai `slo_compliance.json` cachati
    (coerente con le tabelle del paper). Ritorna {'peak':, 'per_phase':{ph:}}
    o None se nessuna cache disponibile."""
    phases = [p[2] for p in PL.PHASE_DEFS]
    acc = {ph: [] for ph in phases}
    tkey = str(int(thr_ms))   # es. '2000' → match chiave '2000.0'
    for run in group.runs:
        f = run_dir_of(results_dir, run) / 'slo_compliance.json'
        if not f.exists():
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        pp = d.get('per_phase', {})
        for ph in phases:
            th = pp.get(ph, {}).get('thresholds', {})
            key = next((k for k in th if k.startswith(tkey)), None)
            if key is not None and th[key].get('pct_above') is not None:
                acc[ph].append(float(th[key]['pct_above']))
    if not any(acc.values()):
        return None
    per_phase = {ph: (float(np.mean(acc[ph])) if acc[ph] else 0.0) for ph in phases}
    return {'peak': per_phase.get('peak', 0.0), 'per_phase': per_phase}


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Grafici dettagliati per regime (uni/bal/con)")
    ap.add_argument('--results-dir', type=Path, default=Path('experiments/results'))
    ap.add_argument('--output-dir', type=Path, default=Path('paper_artifacts/detailed'))
    ap.add_argument('--slo-ms', type=float, default=2000.0)
    ap.add_argument('--dmos-arm', default='dmos-l1l2',
                    help='braccio DMOS completo da confrontare con off e l2 '
                         '(default dmos-l1l2, core NCA; es. dmos-nca)')
    ap.add_argument('--regimes', nargs='+', default=None,
                    help='solo questi regimi (rigenerazione selettiva, es. --regimes con). '
                         'Con questo flag le figure SLO cross-regime vengono SALTATE '
                         '(servono tutti i regimi).')
    args = ap.parse_args()
    set_dmos_arm(args.dmos_arm)

    print(f"=== Detailed plots → {args.output_dir} ===")
    groups = PL.discover_runs(args.results_dir, load_timeline=False)

    regimes = args.regimes if args.regimes else REGIMES
    slo = {}   # (regime, arm) → {peak, per_phase}
    for reg in regimes:
        print(f"\n── regime {reg} ──")
        dfs = {}
        gba = {}   # arm → ScenarioGroup (per le barre, tutte le reps)
        for a in ARMS:
            scen = f"{a}__{reg}"
            g = groups.get(scen)
            if not g or not g.runs:
                print(f"  [skip] {scen}: nessun run"); continue
            gba[a] = g
            rep = representative_run(g, args.results_dir)
            ev = run_dir_of(args.results_dir, rep) / 'k6_events.json'
            if ev.exists():
                print(f"  parse {a}: rep {rep.timestamp} ({ev.stat().st_size/1e9:.1f} GB) ...", flush=True)
                dfs[a] = parse_k6_events(ev)
            else:
                print(f"  [warn] {a}: k6_events.json assente (rep {rep.timestamp}) — solo barre")
                dfs[a] = None
            # SLO per le figure cross-regime: MEDIA su tutte le reps dalle cache
            # slo_compliance.json (coerente con Table 1); fallback al rep parsato.
            sc = slo_mean_from_caches(g, args.results_dir, args.slo_ms)
            if sc is None and dfs.get(a) is not None and not dfs[a].empty:
                sc = slo_from_df(dfs[a], args.slo_ms)
            if sc is not None:
                slo[(reg, a)] = sc
        outdir = args.output_dir / reg
        if gba:
            fig_key_metrics(gba, dfs, reg, outdir)
        if any(v is not None and not v.empty for v in dfs.values()):
            fig_latency_ts(dfs, reg, outdir)
            fig_error_throughput(dfs, reg, outdir)
            fig_per_cluster(dfs, reg, outdir)
            fig_error_heatmap(dfs, reg, outdir)
            fig_cdf(dfs, reg, outdir)
        del dfs   # libera memoria prima del regime successivo

    # Cross-regime SLO (solo su run completo: servono tutti i regimi)
    if slo and not args.regimes:
        print("\n── figure SLO cross-regime ──")
        fig_slo_violation(slo, args.output_dir, args.slo_ms)
        fig_slo_per_phase(slo, args.output_dir, args.slo_ms)
        fig_skew_robustness(slo, args.output_dir, args.slo_ms)
        with open(args.output_dir / 'slo_data.json', 'w') as f:
            json.dump({f"{r}|{a}": v for (r, a), v in slo.items()}, f, indent=1)
    print("\nDone.")


if __name__ == '__main__':
    main()
