#!/usr/bin/env python3
"""
analyze_three_way.py — DMOS ON vs HPA vs DMOS OFF (3-way comparison)
=====================================================================
Generates high-resolution thesis-quality plots comparing all 3 scenarios.

Usage:
  python analyze_three_way.py ^
    --k6-on risultati_DMOS_ON_c300new_150.csv ^
    --k6-off risultati_DMOS_OFF_c300new_150.csv ^
    --k6-hpa risultati_HPA_c300new_150.csv ^
    --post-on ..\results\104315_20260408_flash_crowd_DMOS_ON_post.jsonl ^
    --post-off ..\results\110349_20260408_flash_crowd_DMOS_OFF_post.jsonl ^
    --post-hpa ..\results\134457_20260408_flash_crowd_HPA_post.jsonl ^
    --rps 150 --output-dir plots_3way_150rps
"""

import os, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Style ────────────────────────────────────────────────────────────────────
TARGET_DPI = 600
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': TARGET_DPI,
    'font.size': 10, 'font.family': 'sans-serif',
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.labelsize': 10, 'legend.fontsize': 9,
    'figure.figsize': (18, 10), 'lines.linewidth': 1.2,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

SCENARIOS = ["DMOS ON", "HPA", "DMOS OFF"]
COLORS = {"DMOS ON": "#1565C0", "HPA": "#F57C00", "DMOS OFF": "#C62828"}
COLORS_LIGHT = {"DMOS ON": "#90CAF9", "HPA": "#FFE0B2", "DMOS OFF": "#EF9A9A"}
HATCHES = {"DMOS ON": "", "HPA": "//", "DMOS OFF": "\\\\"}

CLUSTER_COLORS = {"cluster1": "#2E7D32", "cluster2": "#E65100", "cluster3": "#6A1B9A"}
CLUSTER_NAMES = {
    "cluster1": "C1-Frankfurt (DE)",
    "cluster2": "C2-Paris (FR)",
    "cluster3": "C3-Warsaw (PL)",
}
INGRESS_TO_CLUSTER = {"c1-DE": "cluster1", "c2-FR": "cluster2", "c3-PL": "cluster3"}

SERVICES = ["frontend", "cartservice", "productcatalogservice",
            "checkoutservice", "recommendationservice"]

PHASES = [
    (0,   120, "Warm-up"),
    (120, 180, "Ramp"),
    (180, 660, "Sustained"),
    (660, 780, "Decline"),
    (780, 840, "Cool"),
]
PHASE_COLORS = ['#E3F2FD', '#FFF9C4', '#FFCDD2', '#FFF9C4', '#E8F5E9']

BIN_SEC = 5


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_k6(path, label):
    print(f"  Loading k6 {label}: {path} ...", end=" ", flush=True)
    dtype_map = {
        'metric_name': 'category', 'method': 'category',
        'scenario': 'category', 'status': 'str',
        'proto': 'category', 'extra_tags': 'str', 'expected_response': 'str',
    }
    df = pd.read_csv(path, low_memory=False, dtype=dtype_map)
    df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    t_start = df['datetime'].min()
    df['elapsed_s'] = (df['datetime'] - t_start).dt.total_seconds()
    df['ingress'] = df['extra_tags'].str.extract(r'ingress=(\S+)', expand=False).astype('category')
    df['cluster'] = df['ingress'].map(INGRESS_TO_CLUSTER).astype('category')
    df['expected_response'] = df['expected_response'].astype(str).str.lower() == 'true'
    df['status_code'] = pd.to_numeric(df['status'], errors='coerce')
    df['test'] = label
    print(f"({len(df):,} rows)")
    return df


def load_post_jsonl(path, label):
    """Load Prometheus POST JSON → list of per-timestamp snapshots."""
    print(f"  Loading POST {label}: {path} ...", end=" ", flush=True)
    with open(path) as f:
        data = json.loads(f.read().strip())

    timestamps = set()
    for cdata in data.get('clusters', {}).values():
        for svc_data in cdata.get('services', {}).values():
            for ts, val in svc_data.get('cpu_cores', []):
                timestamps.add(ts)
            break
        break
    timestamps = sorted(timestamps)
    if not timestamps:
        print("(empty)")
        return []

    t0 = timestamps[0]
    td_lookup = {td['timestamp']: td for td in data.get('traffic_distribution', [])}

    snapshots = []
    for ts in timestamps:
        snap = {
            '_elapsed_s': ts - t0,
            'resources': {},
            'replicas': {},   # {service: {cluster: n}}
        }
        for cname, cdata in data.get('clusters', {}).items():
            snap['resources'][cname] = {}
            for svc, sdata in cdata.get('services', {}).items():
                replicas = _find_val(sdata.get('ready_pods', []), ts)
                cpu = _find_val(sdata.get('cpu_cores', []), ts)
                snap['resources'][cname][svc] = {
                    'ready_pods': int(replicas) if replicas else 0,
                    'cpu_cores': cpu,
                }
                if svc not in snap['replicas']:
                    snap['replicas'][svc] = {}
                snap['replicas'][svc][cname] = int(replicas) if replicas else 0

            td = td_lookup.get(ts, {})
            snap['resources'][cname]['_traffic_pct'] = td.get(cname, 0)

        snapshots.append(snap)

    print(f"({len(snapshots)} snapshots)")
    return snapshots


def _find_val(timeseries, target_ts, tolerance=16.0):
    if not timeseries:
        return None
    best_val, best_diff = None, float('inf')
    for ts, val in timeseries:
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_val = val
        elif diff > best_diff:
            break
    return best_val if best_diff <= tolerance else None


# ── Helpers ──────────────────────────────────────────────────────────────────

def k6_dur(df): return df[df['metric_name'] == 'http_req_duration'].copy()
def k6_reqs(df): return df[df['metric_name'] == 'http_reqs'].copy()
def k6_vus(df): return df[df['metric_name'] == 'vus'].copy()

def _shade(ax):
    for (s, e, _), col in zip(PHASES, PHASE_COLORS):
        ax.axvspan(s/60, e/60, alpha=0.12, color=col, zorder=0)

def _phase_labels(ax, y_frac=0.97):
    for (s, e, name), _ in zip(PHASES, PHASE_COLORS):
        ax.text((s+e)/2/60, y_frac, name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, alpha=0.5, fontstyle='italic')

def _save(fig, d, name):
    p = d / name
    fig.savefig(p, dpi=TARGET_DPI)
    print(f"    Saved: {name} ({os.path.getsize(p)/1024/1024:.1f} MB)")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary_table(k6s, rps, out):
    """Page 0: Summary comparison table."""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')

    rows = []
    for label in SCENARIOS:
        if label not in k6s:
            continue
        df = k6s[label]
        dur = k6_dur(df)
        reqs = k6_reqs(df)
        total = len(reqs)
        failed = reqs[~reqs['expected_response']].shape[0]
        ok = total - failed
        lat = dur['metric_value']
        vus = k6_vus(df)
        max_vus = int(vus['metric_value'].max()) if not vus.empty else 0
        rows.append([
            label, f"{total:,}", f"{ok:,}", f"{failed:,}",
            f"{failed/total*100:.2f}%",
            f"{lat.mean():.0f}", f"{lat.quantile(0.50):.0f}",
            f"{lat.quantile(0.95):.0f}", f"{lat.quantile(0.99):.0f}",
            f"{max_vus}",
        ])

    cols = ['Scenario', 'Total Reqs', 'Success', 'Failed', 'Error %',
            'Avg (ms)', 'p50 (ms)', 'p95 (ms)', 'p99 (ms)', 'Max VUs']
    table = ax.table(cellText=rows, colLabels=cols, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.15, 2.2)

    # Color header
    for j in range(len(cols)):
        table[0, j].set_facecolor('#263238')
        table[0, j].set_text_props(color='white', weight='bold')
    # Color rows
    row_colors = {'DMOS ON': '#E3F2FD', 'HPA': '#FFF3E0', 'DMOS OFF': '#FFEBEE'}
    for i, label in enumerate(SCENARIOS):
        if label in k6s:
            idx = [r[0] for r in rows].index(label) + 1
            for j in range(len(cols)):
                table[idx, j].set_facecolor(row_colors.get(label, 'white'))

    # Per-cluster sub-table
    pc_rows = []
    for cname, ckey in INGRESS_TO_CLUSTER.items():
        row = [CLUSTER_NAMES[ckey]]
        for label in SCENARIOS:
            if label not in k6s:
                continue
            df = k6s[label]
            reqs = k6_reqs(df)
            dur_df = k6_dur(df)
            cr = reqs[reqs['ingress'] == cname]
            cl_dur = dur_df[dur_df['ingress'] == cname]['metric_value']
            t = len(cr)
            f = cr[~cr['expected_response']].shape[0]
            row.append(f"{f/t*100:.2f}%" if t > 0 else "N/A")
            row.append(f"{cl_dur.quantile(0.95):.0f}" if len(cl_dur) > 0 else "N/A")
        pc_rows.append(row)

    pc_cols = ['Cluster']
    for label in SCENARIOS:
        if label in k6s:
            pc_cols.extend([f'{label} err%', f'{label} p95'])

    ax.text(0.5, 0.52, 'Per-Cluster Breakdown', ha='center',
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    t2 = ax.table(cellText=pc_rows, colLabels=pc_cols, loc='center', cellLoc='center',
                   bbox=[0.05, 0.15, 0.9, 0.35])
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    for j in range(len(pc_cols)):
        t2[0, j].set_facecolor('#455A64')
        t2[0, j].set_text_props(color='white', weight='bold')

    fig.suptitle(f'3-Way Comparison Summary — {rps} rps (tc netem 300ms on C3)',
                 fontsize=14, y=0.98)
    _save(fig, out, '3way_00_summary.png')


def plot_latency_timeseries(k6s, rps, out):
    """Page 1: p50/p95/p99 over time."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
    for qi, (q, ql) in enumerate([(0.50, 'p50'), (0.95, 'p95'), (0.99, 'p99')]):
        ax = axes[qi]
        for label in SCENARIOS:
            if label not in k6s:
                continue
            dur = k6_dur(k6s[label])
            dur['bin'] = (dur['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = dur.groupby('bin')['metric_value'].quantile(q).reset_index()
            ax.plot(g['bin']/60, g['metric_value'], label=label,
                    color=COLORS[label], lw=1.2, alpha=0.9)
        ax.set_ylabel(f'{ql} (ms)')
        ax.set_title(f'{ql} Response Latency')
        ax.legend()
        ax.grid(True, alpha=0.2)
        _shade(ax)
        _phase_labels(ax)
    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Latency Time Series — {rps} rps', fontsize=14, y=1.01)
    _save(fig, out, '3way_01_latency_timeseries.png')


def plot_error_throughput(k6s, rps, out):
    """Page 2: Error rate + throughput + VUs."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

    ax = axes[0]
    for label in SCENARIOS:
        if label not in k6s:
            continue
        reqs = k6_reqs(k6s[label])
        reqs['bin'] = (reqs['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = reqs.groupby('bin').agg(
            total=('expected_response', 'count'),
            failed=('expected_response', lambda x: (~x).sum())
        ).reset_index()
        g['err_pct'] = g['failed'] / g['total'] * 100
        ax.plot(g['bin']/60, g['err_pct'], label=label, color=COLORS[label], lw=1.0)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)
    _phase_labels(ax)

    ax = axes[1]
    for label in SCENARIOS:
        if label not in k6s:
            continue
        reqs = k6_reqs(k6s[label])
        ok = reqs[reqs['expected_response']]
        ok_bin = (ok['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = ok_bin.value_counts().sort_index().reset_index()
        g.columns = ['bin', 'n']
        g['rps'] = g['n'] / BIN_SEC
        ax.plot(g['bin']/60, g['rps'], label=label, color=COLORS[label], lw=1.0)
    ax.axhline(rps, color='gray', ls='--', alpha=0.5, label=f'Target {rps} rps')
    ax.set_ylabel('Successful RPS')
    ax.set_title('Throughput')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)

    ax = axes[2]
    for label in SCENARIOS:
        if label not in k6s:
            continue
        vus = k6_vus(k6s[label])
        if not vus.empty:
            ax.plot(vus['elapsed_s']/60, vus['metric_value'],
                    label=label, color=COLORS[label], lw=1.0)
    ax.set_ylabel('VUs')
    ax.set_title('Virtual Users')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)
    ax.set_xlabel('Time (min)')

    fig.suptitle(f'Error, Throughput & VUs — {rps} rps', fontsize=14, y=1.01)
    _save(fig, out, '3way_02_error_throughput.png')


def plot_cdf(k6s, rps, out):
    """Page 3: CDF global + per-cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    for label in SCENARIOS:
        if label not in k6s:
            continue
        dur = k6_dur(k6s[label])
        lat = np.sort(dur['metric_value'].dropna().values)
        cdf = np.arange(1, len(lat)+1) / len(lat)
        ax.plot(lat, cdf, label=label, color=COLORS[label], lw=1.5)
    ax.set_xlim(0, 3000)
    ax.axhline(0.95, color='gray', ls='--', alpha=0.4, label='p95')
    ax.axhline(0.99, color='gray', ls=':', alpha=0.4, label='p99')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Global Latency CDF')
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    styles = {"DMOS ON": "-", "HPA": "--", "DMOS OFF": ":"}
    for label in SCENARIOS:
        if label not in k6s:
            continue
        dur = k6_dur(k6s[label])
        for cname, ckey in INGRESS_TO_CLUSTER.items():
            vals = dur[dur['ingress'] == cname]['metric_value'].dropna().values
            if len(vals) > 0:
                lat = np.sort(vals)
                cdf = np.arange(1, len(lat)+1) / len(lat)
                ax.plot(lat, cdf, label=f'{label} {CLUSTER_NAMES[ckey][:2]}',
                        color=CLUSTER_COLORS[ckey], ls=styles[label], lw=1.0, alpha=0.8)
    ax.set_xlim(0, 3000)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Per-Cluster Latency CDF')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.2)

    fig.suptitle(f'Latency CDF — {rps} rps', fontsize=14, y=1.01)
    _save(fig, out, '3way_03_cdf.png')


def plot_per_cluster(k6s, rps, out):
    """Page 4: Per-cluster p95 + error rate."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 16), sharex='col')
    clusters = ["cluster1", "cluster2", "cluster3"]

    for ci, ckey in enumerate(clusters):
        cname = [k for k,v in INGRESS_TO_CLUSTER.items() if v == ckey][0]

        ax = axes[ci, 0]
        for label in SCENARIOS:
            if label not in k6s:
                continue
            dur = k6_dur(k6s[label])
            cl = dur[dur['ingress'] == cname].copy()
            cl['bin'] = (cl['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = cl.groupby('bin')['metric_value'].quantile(0.95).reset_index()
            ax.plot(g['bin']/60, g['metric_value'], label=label,
                    color=COLORS[label], lw=1.0)
        ax.set_ylabel('p95 (ms)')
        ax.set_title(f'{CLUSTER_NAMES[ckey]} — p95 Latency')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)

        ax = axes[ci, 1]
        for label in SCENARIOS:
            if label not in k6s:
                continue
            reqs = k6_reqs(k6s[label])
            cl = reqs[reqs['ingress'] == cname].copy()
            cl['bin'] = (cl['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = cl.groupby('bin').agg(
                total=('expected_response', 'count'),
                failed=('expected_response', lambda x: (~x).sum())
            ).reset_index()
            g['err_pct'] = g['failed'] / g['total'] * 100
            ax.plot(g['bin']/60, g['err_pct'], label=label,
                    color=COLORS[label], lw=1.0)
        ax.set_ylabel('Error %')
        ax.set_title(f'{CLUSTER_NAMES[ckey]} — Error Rate')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)

    axes[-1,0].set_xlabel('Time (min)')
    axes[-1,1].set_xlabel('Time (min)')
    fig.suptitle(f'Per-Cluster Detail — {rps} rps', fontsize=14, y=1.01)
    _save(fig, out, '3way_04_per_cluster.png')


def plot_bar_comparison(k6s, rps, out):
    """Page 5: Bar chart comparison — key metrics side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    labels_present = [l for l in SCENARIOS if l in k6s]
    x = np.arange(len(labels_present))
    w = 0.6

    # Error rate
    ax = axes[0, 0]
    errs = []
    for label in labels_present:
        reqs = k6_reqs(k6s[label])
        t = len(reqs)
        f = reqs[~reqs['expected_response']].shape[0]
        errs.append(f/t*100)
    bars = ax.bar(x, errs, w, color=[COLORS[l] for l in labels_present], alpha=0.85)
    for b, v in zip(bars, errs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
                f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate')
    ax.grid(True, alpha=0.2, axis='y')

    # p95
    ax = axes[0, 1]
    p95s = []
    for label in labels_present:
        dur = k6_dur(k6s[label])
        p95s.append(dur['metric_value'].quantile(0.95))
    bars = ax.bar(x, p95s, w, color=[COLORS[l] for l in labels_present], alpha=0.85)
    for b, v in zip(bars, p95s):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+20,
                f'{v:.0f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present)
    ax.set_ylabel('p95 Latency (ms)')
    ax.set_title('p95 Response Latency')
    ax.grid(True, alpha=0.2, axis='y')

    # Throughput
    ax = axes[1, 0]
    tputs = []
    for label in labels_present:
        reqs = k6_reqs(k6s[label])
        ok = reqs[reqs['expected_response']]
        duration = ok['elapsed_s'].max() - ok['elapsed_s'].min()
        tputs.append(len(ok) / duration if duration > 0 else 0)
    bars = ax.bar(x, tputs, w, color=[COLORS[l] for l in labels_present], alpha=0.85)
    for b, v in zip(bars, tputs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present)
    ax.set_ylabel('Successful RPS')
    ax.set_title('Effective Throughput')
    ax.grid(True, alpha=0.2, axis='y')

    # Connection errors + dropped
    ax = axes[1, 1]
    conn_errs = []
    dropped = []
    for label in labels_present:
        reqs = k6_reqs(k6s[label])
        # Connection errors: status == 0 or NaN
        ce = reqs[reqs['status_code'].isna() | (reqs['status_code'] == 0)].shape[0]
        conn_errs.append(ce)
        # Dropped iterations from VUs data
        df = k6s[label]
        di = df[df['metric_name'] == 'dropped_iterations']
        dropped.append(int(di['metric_value'].sum()) if not di.empty else 0)
    bw = 0.3
    ax.bar(x - bw/2, conn_errs, bw, label='Connection Errors',
           color=[COLORS[l] for l in labels_present], alpha=0.7)
    ax.bar(x + bw/2, dropped, bw, label='Dropped Iterations',
           color=[COLORS[l] for l in labels_present], alpha=0.4, hatch='//')
    for i, (ce, dr) in enumerate(zip(conn_errs, dropped)):
        ax.text(i - bw/2, ce + 50, f'{ce:,}', ha='center', fontsize=9)
        ax.text(i + bw/2, dr + 50, f'{dr:,}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_present)
    ax.set_ylabel('Count')
    ax.set_title('Connection Errors & Dropped Iterations')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle(f'Key Metrics Comparison — {rps} rps', fontsize=14, y=1.01)
    _save(fig, out, '3way_05_bar_comparison.png')


def plot_replicas_comparison(posts, rps, out):
    """Page 6: Frontend replicas over time per scenario."""
    available = {l: s for l, s in posts.items() if s}
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5*n), sharex=True)
    if n == 1:
        axes = [axes]

    clusters = ["cluster1", "cluster2", "cluster3"]

    for idx, (label, snaps) in enumerate(available.items()):
        ax = axes[idx]
        elapsed = [s['_elapsed_s']/60 for s in snaps]

        for cl in clusters:
            reps = [s.get('replicas', {}).get('frontend', {}).get(cl, 0) for s in snaps]
            ax.step(elapsed, reps, where='post', color=CLUSTER_COLORS[cl],
                    lw=1.5, label=CLUSTER_NAMES[cl])

        total = [sum(s.get('replicas', {}).get('frontend', {}).values()) for s in snaps]
        ax.plot(elapsed, total, 'k-', lw=1, alpha=0.5, label='Total')

        ax.set_ylabel('Replicas')
        ax.set_title(f'{label} — Frontend Replicas per Cluster')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, ncol=4)
        ax.grid(True, alpha=0.2)
        _shade(ax)

    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'Replica Distribution — {rps} rps', fontsize=14, y=1.01)
    _save(fig, out, '3way_06_replicas.png')


def plot_cpu_comparison(posts, rps, out):
    """Page 7: Total CPU per cluster per scenario."""
    available = {l: s for l, s in posts.items() if s}
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(18, 5*n), sharex=True)
    if n == 1:
        axes = [axes]

    clusters = ["cluster1", "cluster2", "cluster3"]

    for idx, (label, snaps) in enumerate(available.items()):
        ax = axes[idx]
        elapsed = [s['_elapsed_s']/60 for s in snaps]

        for cl in clusters:
            cpu_total = []
            for s in snaps:
                cpu_sum = 0
                for svc in SERVICES:
                    val = s.get('resources', {}).get(cl, {}).get(svc, {}).get('cpu_cores')
                    if val is not None:
                        cpu_sum += val
                cpu_total.append(cpu_sum * 1000)
            ax.plot(elapsed, cpu_total, color=CLUSTER_COLORS[cl],
                    lw=1.2, label=CLUSTER_NAMES[cl])

        ax.axhline(4000, color='red', ls='--', alpha=0.4, label='Node limit')
        ax.set_ylabel('CPU (millicores)')
        ax.set_title(f'{label} — Total CPU Usage per Cluster')
        ax.legend(fontsize=8, ncol=4)
        ax.grid(True, alpha=0.2)
        _shade(ax)

    axes[-1].set_xlabel('Time (min)')
    fig.suptitle(f'CPU Utilization — {rps} rps', fontsize=14, y=1.01)
    _save(fig, out, '3way_07_cpu.png')


def plot_phase_error_heatmap(k6s, rps, out):
    """Page 8: Error heatmap — scenario x phase and scenario x cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    labels_present = [l for l in SCENARIOS if l in k6s]
    pnames = [p[2] for p in PHASES]

    # Scenario x Phase
    ax = axes[0]
    matrix = np.zeros((len(labels_present), len(PHASES)))
    for i, label in enumerate(labels_present):
        reqs = k6_reqs(k6s[label])
        for j, (s, e, _) in enumerate(PHASES):
            pr = reqs[(reqs['elapsed_s'] >= s) & (reqs['elapsed_s'] < e)]
            t = len(pr)
            f = pr[~pr['expected_response']].shape[0]
            matrix[i, j] = f/t*100 if t > 0 else 0

    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0,
                   vmax=max(20, matrix.max()))
    ax.set_xticks(range(len(pnames)))
    ax.set_xticklabels(pnames, fontsize=9, rotation=15)
    ax.set_yticks(range(len(labels_present)))
    ax.set_yticklabels(labels_present, fontsize=10)
    for i in range(len(labels_present)):
        for j in range(len(PHASES)):
            ax.text(j, i, f'{matrix[i,j]:.2f}%', ha='center', va='center',
                    fontsize=9, color='white' if matrix[i,j] > 10 else 'black')
    ax.set_title('Error Rate: Scenario x Phase')
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Scenario x Cluster
    ax = axes[1]
    ckeys = list(INGRESS_TO_CLUSTER.keys())
    matrix2 = np.zeros((len(labels_present), len(ckeys)))
    for i, label in enumerate(labels_present):
        reqs = k6_reqs(k6s[label])
        # Sustained phase only (most relevant)
        reqs_s = reqs[(reqs['elapsed_s'] >= 180) & (reqs['elapsed_s'] < 660)]
        for j, cname in enumerate(ckeys):
            cr = reqs_s[reqs_s['ingress'] == cname]
            t = len(cr)
            f = cr[~cr['expected_response']].shape[0]
            matrix2[i, j] = f/t*100 if t > 0 else 0

    im2 = ax.imshow(matrix2, cmap='RdYlGn_r', aspect='auto', vmin=0,
                    vmax=max(20, matrix2.max()))
    clabels = [CLUSTER_NAMES[INGRESS_TO_CLUSTER[c]][:12] for c in ckeys]
    ax.set_xticks(range(len(ckeys)))
    ax.set_xticklabels(clabels, fontsize=9)
    ax.set_yticks(range(len(labels_present)))
    ax.set_yticklabels(labels_present, fontsize=10)
    for i in range(len(labels_present)):
        for j in range(len(ckeys)):
            ax.text(j, i, f'{matrix2[i,j]:.2f}%', ha='center', va='center',
                    fontsize=9, color='white' if matrix2[i,j] > 10 else 'black')
    ax.set_title('Error Rate: Scenario x Cluster (Sustained Phase)')
    fig.colorbar(im2, ax=ax, shrink=0.7)

    fig.suptitle(f'Error Heatmaps — {rps} rps', fontsize=14, y=1.05)
    _save(fig, out, '3way_08_error_heatmap.png')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-Way Comparison: DMOS ON vs HPA vs DMOS OFF")
    parser.add_argument("--k6-on", required=True)
    parser.add_argument("--k6-off", required=True)
    parser.add_argument("--k6-hpa", required=True)
    parser.add_argument("--post-on", help="Prometheus POST JSONL for DMOS ON")
    parser.add_argument("--post-off", help="Prometheus POST JSONL for DMOS OFF")
    parser.add_argument("--post-hpa", help="Prometheus POST JSONL for HPA")
    parser.add_argument("--rps", type=int, default=150)
    parser.add_argument("--output-dir", default="plots_3way")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading k6 CSV files ===")
    k6s = {
        "DMOS ON": load_k6(args.k6_on, "DMOS ON"),
        "HPA": load_k6(args.k6_hpa, "HPA"),
        "DMOS OFF": load_k6(args.k6_off, "DMOS OFF"),
    }

    posts = {}
    if args.post_on or args.post_off or args.post_hpa:
        print("\n=== Loading POST JSONL files ===")
        if args.post_on:
            posts["DMOS ON"] = load_post_jsonl(args.post_on, "DMOS ON")
        if args.post_hpa:
            posts["HPA"] = load_post_jsonl(args.post_hpa, "HPA")
        if args.post_off:
            posts["DMOS OFF"] = load_post_jsonl(args.post_off, "DMOS OFF")

    print(f"\n=== Generating plots → {out}/ ===")
    plot_summary_table(k6s, args.rps, out)
    plot_latency_timeseries(k6s, args.rps, out)
    plot_error_throughput(k6s, args.rps, out)
    plot_cdf(k6s, args.rps, out)
    plot_per_cluster(k6s, args.rps, out)
    plot_bar_comparison(k6s, args.rps, out)

    if posts:
        plot_replicas_comparison(posts, args.rps, out)
        plot_cpu_comparison(posts, args.rps, out)

    plot_phase_error_heatmap(k6s, args.rps, out)

    print(f"\nDone! 9 plot pages saved to {out}/")


if __name__ == '__main__':
    main()
