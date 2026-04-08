#!/usr/bin/env python3
"""
analyze_dmos_hires.py — Analisi DMOS ON vs B ad alta risoluzione
===================================================================
Versione potenziata per tesi con:
  - DPI 600 (stampa accademica)
  - Bin temporali 5s (alta granularità)
  - 10+ pagine di grafici separati
  - Violin plot, heatmap temporale, carbon/energy, VU usage
  - Supporto GPU via matplotlib backend Agg (no display)
  - --label-b per scegliere etichetta confronto (default "DMOS OFF", oppure "HPA")

Uso DMOS ON vs OFF:
  python analyze_dmos_hires.py ^
    --k6-on  risultati_DMOS_ON_c300new_150.csv ^
    --k6-off risultati_DMOS_OFF_c300new_150.csv ^
    --jsonl-on  ../results/104315_post.jsonl ^
    --jsonl-off ../results/110349_post.jsonl ^
    --rps 150 --output-dir plots_on_vs_off

Uso DMOS ON vs HPA:
  python analyze_dmos_hires.py ^
    --k6-on  risultati_DMOS_ON_c300new_150.csv ^
    --k6-off risultati_HPA_c300new_150.csv ^
    --jsonl-on  ../results/104315_post.jsonl ^
    --jsonl-off ../results/134457_post.jsonl ^
    --label-b HPA --rps 150 --output-dir plots_on_vs_hpa
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Force Agg backend for large renders (no GUI needed)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Style (high resolution) ─────────────────────────────────────────────────
TARGET_DPI = 600

plt.rcParams.update({
    'figure.dpi': 150,       # screen preview
    'savefig.dpi': TARGET_DPI,
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'legend.framealpha': 0.9,
    'figure.figsize': (18, 10),
    'lines.linewidth': 1.2,
    'lines.antialiased': True,
    'patch.antialiased': True,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Default colors — will be updated by main() based on --label-b
LABEL_A = "DMOS ON"
LABEL_B = "DMOS OFF"
COLORS = {LABEL_A: "#1565C0", LABEL_B: "#C62828"}
COLORS_LIGHT = {LABEL_A: "#90CAF9", LABEL_B: "#EF9A9A"}
COLORS_B_MAP = {
    "DMOS OFF": ("#C62828", "#EF9A9A"),   # red
    "HPA":      ("#F57C00", "#FFE0B2"),   # orange
}
CLUSTER_COLORS = {"cluster1": "#2E7D32", "cluster2": "#E65100", "cluster3": "#6A1B9A"}
CLUSTER_NAMES = {
    "cluster1": "C1 — Frankfurt (DE)",
    "cluster2": "C2 — Paris (FR)",
    "cluster3": "C3 — Warsaw (PL)",
}
INGRESS_TO_CLUSTER = {"c1-DE": "cluster1", "c2-FR": "cluster2", "c3-PL": "cluster3"}

SERVICES = ["frontend", "cartservice", "productcatalogservice",
            "checkoutservice", "recommendationservice"]
SERVICE_SHORT = {
    "frontend": "Frontend", "cartservice": "Cart",
    "productcatalogservice": "ProductCatalog",
    "checkoutservice": "Checkout", "recommendationservice": "Recommendation",
}

PHASES = [
    (0,   120, "Warm-up"),
    (120, 180, "Ramp"),
    (180, 660, "Sustained"),
    (660, 780, "Decline"),
    (780, 840, "Cool"),
]
PHASE_COLORS = ['#E3F2FD', '#FFF9C4', '#FFCDD2', '#FFF9C4', '#E8F5E9']

BIN_SEC = 5   # 5-second bins for max resolution


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_k6_csv(path, label):
    """Load k6 CSV with optimized dtypes for large files."""
    print(f"  Loading k6 {label}: {path} ...", end=" ", flush=True)
    dtype_map = {
        'metric_name': 'category',
        'method': 'category',
        'scenario': 'category',
        'status': 'str',
        'proto': 'category',
        'extra_tags': 'str',
        'expected_response': 'str',
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


def load_jsonl(path, label):
    """Load DMOS JSONL (line-delimited) or Prometheus POST JSON (single object).

    Detects the format automatically:
      - If the file starts with '{' and is a single JSON object with 'clusters' key,
        it's a Prometheus POST snapshot → converted to a list of per-timestamp snapshots.
      - Otherwise, it's classic line-delimited JSONL with DMOS scheduler snapshots.
    """
    print(f"  Loading JSONL {label}: {path} ...", end=" ", flush=True)

    with open(path) as f:
        content = f.read().strip()

    if not content:
        print("(empty)")
        return None

    # Try single-object Prometheus POST format
    first_char = content[0]
    if first_char == '{':
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = None

        if data and 'clusters' in data:
            snapshots = _convert_prom_post_to_snapshots(data)
            print(f"({len(snapshots)} snapshots from Prometheus POST)")
            return snapshots

    # Fallback: line-delimited JSONL (DMOS scheduler snapshots)
    snapshots = []
    for line in content.split('\n'):
        line = line.strip()
        if line:
            try:
                snapshots.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not snapshots:
        print("(no valid lines)")
        return None

    for snap in snapshots:
        snap['_dt'] = datetime.fromisoformat(snap['timestamp'])
    t0 = snapshots[0]['_dt']
    for snap in snapshots:
        snap['_elapsed_s'] = (snap['_dt'] - t0).total_seconds()
    print(f"({len(snapshots)} DMOS snapshots)")
    return snapshots


def _convert_prom_post_to_snapshots(data):
    """Convert a Prometheus POST JSON into a list of per-timestamp snapshots.

    Input format:
      {
        "clusters": {
          "cluster1": {
            "services": {
              "frontend": {
                "cpu_cores": [[ts, val], ...],
                "ready_pods": [[ts, val], ...],
                "memory_bytes": [[ts, val], ...],
                ...
              }
            },
            "node": {"cpu_usage_pct": [...], "memory_usage_pct": [...]},
            "hubble_http_rate": [[ts, val], ...],
            "frontend_net_bytes_s": [[ts, val], ...],
          }
        },
        "traffic_distribution": [{"timestamp": ts, "cluster1": pct, ...}, ...],
        "metadata": {"start_time": ..., "duration_s": ..., "step": "15s"}
      }

    Output: list of snapshot dicts compatible with extract_timeseries().
    """
    meta = data.get('metadata', {})
    start_time = meta.get('start_time', '')

    # Collect all unique timestamps from the first service's cpu_cores
    timestamps = set()
    for cname, cdata in data.get('clusters', {}).items():
        for svc_name, svc_data in cdata.get('services', {}).items():
            for ts, val in svc_data.get('cpu_cores', []):
                timestamps.add(ts)
            break
        break
    timestamps = sorted(timestamps)
    if not timestamps:
        return []

    t0 = timestamps[0]

    # Build traffic_distribution lookup
    td_lookup = {}
    for td in data.get('traffic_distribution', []):
        td_lookup[td['timestamp']] = td

    # Build per-timestamp snapshots
    snapshots = []
    for ts in timestamps:
        snap = {
            '_dt': datetime.fromtimestamp(ts),
            '_elapsed_s': ts - t0,
            'timestamp': datetime.fromtimestamp(ts).isoformat(),
            'resources': {},
            'dmos': {'services': {}},
        }

        total_traffic_per_svc = {}

        for cname, cdata in data.get('clusters', {}).items():
            snap['resources'][cname] = {}

            for svc_name, svc_data in cdata.get('services', {}).items():
                # Find value at this timestamp
                cpu = _find_val(svc_data.get('cpu_cores', []), ts)
                mem_bytes = _find_val(svc_data.get('memory_bytes', []), ts)
                net = _find_val(svc_data.get('network_recv_bytes_s', []), ts)
                replicas = _find_val(svc_data.get('ready_pods', []), ts)

                snap['resources'][cname][svc_name] = {
                    'cpu_cores': cpu,
                    'memory_mb': mem_bytes / (1024 * 1024) if mem_bytes else None,
                    'network_recv_bytes_s': net,
                }

                # Build dmos-compatible structure for replicas
                if svc_name not in snap['dmos']['services']:
                    snap['dmos']['services'][svc_name] = {
                        'actual_traffic': 0,
                        'predicted_traffic_total': 0,
                        'total_replicas': 0,
                        'clusters': {},
                    }
                svc_snap = snap['dmos']['services'][svc_name]
                svc_snap['clusters'][cname] = {
                    'current_replicas': int(replicas) if replicas else 0,
                    'score': 0,
                    'predicted_traffic': 0,
                }
                svc_snap['total_replicas'] += int(replicas) if replicas else 0

            # Traffic distribution
            td_entry = td_lookup.get(ts, {})
            pct = td_entry.get(cname)  # keys are "cluster1", "cluster2", "cluster3"
            snap['resources'][cname]['_traffic_pct'] = {'frontend': pct}

        snapshots.append(snap)

    return snapshots


def _find_val(timeseries, target_ts, tolerance=16.0):
    """Find value in [[ts, val], ...] closest to target_ts."""
    if not timeseries:
        return None
    # Binary-ish search (series is sorted)
    best_val = None
    best_diff = float('inf')
    for ts, val in timeseries:
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_val = val
        elif diff > best_diff:
            break  # Past best, stop
    return best_val if best_diff <= tolerance else None


# ── Helpers ──────────────────────────────────────────────────────────────────

def k6_duration(df):
    return df[df['metric_name'] == 'http_req_duration'].copy()

def k6_reqs(df):
    return df[df['metric_name'] == 'http_reqs'].copy()

def k6_vus(df):
    return df[df['metric_name'] == 'vus'].copy()

def _shade(ax):
    """Draw phase background colors."""
    for (s, e, _), col in zip(PHASES, PHASE_COLORS):
        ax.axvspan(s / 60, e / 60, alpha=0.12, color=col, zorder=0)

def _phase_labels(ax, y_frac=0.97):
    """Overlay phase names at top."""
    for (s, e, name), _ in zip(PHASES, PHASE_COLORS):
        mid = (s + e) / 2 / 60
        ax.text(mid, y_frac, name, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, alpha=0.5, fontstyle='italic')

def _savefig(fig, plot_dir, name):
    path = plot_dir / name
    fig.savefig(path, dpi=TARGET_DPI)
    print(f"    Saved: {name} ({os.path.getsize(path)/1024/1024:.1f} MB)")
    plt.close(fig)


def url_to_path(url):
    if pd.isna(url):
        return 'unknown'
    parts = str(url).split(':30080')
    return parts[1].split('?')[0] if len(parts) > 1 else str(url)[-20:]


# ── JSONL Extraction ─────────────────────────────────────────────────────────

def extract_timeseries(snapshots, service="frontend"):
    data = {
        'elapsed_s': [], 'actual_traffic': [], 'predicted_traffic': [],
        'total_replicas': [],
    }
    for c in ["cluster1", "cluster2", "cluster3"]:
        data[f'{c}_replicas'] = []
        data[f'{c}_score'] = []
        data[f'{c}_predicted'] = []
        data[f'{c}_cpu'] = []
        data[f'{c}_memory_mb'] = []
        data[f'{c}_traffic_pct'] = []
        data[f'{c}_network_bytes_s'] = []

    for snap in snapshots:
        svc = snap.get('dmos', {}).get('services', {}).get(service, {})
        data['elapsed_s'].append(snap['_elapsed_s'])
        data['actual_traffic'].append(svc.get('actual_traffic', 0))
        data['predicted_traffic'].append(svc.get('predicted_traffic_total', 0))
        data['total_replicas'].append(svc.get('total_replicas', 0))
        for c in ["cluster1", "cluster2", "cluster3"]:
            cd = svc.get('clusters', {}).get(c, {})
            data[f'{c}_replicas'].append(cd.get('current_replicas', 0))
            data[f'{c}_score'].append(cd.get('score', 0))
            data[f'{c}_predicted'].append(cd.get('predicted_traffic', 0))
            res = snap.get('resources', {}).get(c, {}).get(service, {})
            data[f'{c}_cpu'].append(res.get('cpu_cores'))
            data[f'{c}_memory_mb'].append(res.get('memory_mb'))
            data[f'{c}_network_bytes_s'].append(res.get('network_recv_bytes_s'))
            tpct = snap.get('resources', {}).get(c, {}).get('_traffic_pct', {}).get('frontend')
            data[f'{c}_traffic_pct'].append(tpct)

    return pd.DataFrame(data)


def compute_jain_index(snapshots, service="frontend"):
    jain_data = {'elapsed_s': [], 'jain_index': [], 'jain_index_score': []}
    for snap in snapshots:
        svc = snap.get('dmos', {}).get('services', {}).get(service, {})
        replicas, scores = [], []
        for c in ["cluster1", "cluster2", "cluster3"]:
            cd = svc.get('clusters', {}).get(c, {})
            replicas.append(cd.get('current_replicas', 0))
            scores.append(cd.get('score', 0))
        r = np.array(replicas, dtype=float)
        s = np.array(scores, dtype=float)
        n = len(r)
        sum_sq = np.sum(r ** 2)
        jain_r = (np.sum(r) ** 2) / (n * sum_sq) if sum_sq > 0 else 1.0
        sum_sq_s = np.sum(s ** 2)
        jain_s = (np.sum(s) ** 2) / (n * sum_sq_s) if sum_sq_s > 0 else 1.0
        jain_data['elapsed_s'].append(snap['_elapsed_s'])
        jain_data['jain_index'].append(jain_r)
        jain_data['jain_index_score'].append(jain_s)
    return pd.DataFrame(jain_data)


def compute_prediction_accuracy(snapshots, service="frontend"):
    actual, predicted = [], []
    for snap in snapshots:
        svc = snap.get('dmos', {}).get('services', {}).get(service, {})
        a = svc.get('actual_traffic', 0)
        p = svc.get('predicted_traffic_total', 0)
        if a > 1:
            actual.append(a)
            predicted.append(p)
    if not actual:
        return None
    actual = np.array(actual)
    predicted = np.array(predicted)
    errors = predicted - actual
    abs_pct = np.abs(errors / actual) * 100
    return {
        'mape': np.mean(abs_pct),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(np.abs(errors)),
        'r_squared': 1 - np.sum(errors**2) / np.sum((actual - np.mean(actual))**2) if np.var(actual) > 0 else 0,
        'directional_accuracy': np.mean(np.sign(np.diff(predicted)) == np.sign(np.diff(actual))) * 100 if len(actual) > 1 else 0,
        'mean_actual': np.mean(actual),
        'mean_predicted': np.mean(predicted),
        'actual': actual, 'predicted': predicted,
    }


def compute_scaling_stats(snapshots):
    stats = {}
    for svc in SERVICES:
        svc_stats = {'total_scale_up': 0, 'total_scale_down': 0, 'per_cluster': {}}
        for c in ["cluster1", "cluster2", "cluster3"]:
            if snapshots:
                last = snapshots[-1]
                cd = last.get('dmos', {}).get('services', {}).get(svc, {}).get('clusters', {}).get(c, {})
                su = cd.get('scale_up_events_cumulative', 0)
                sd = cd.get('scale_down_events_cumulative', 0)
                svc_stats['per_cluster'][c] = {'scale_up': su, 'scale_down': sd}
                svc_stats['total_scale_up'] += su
                svc_stats['total_scale_down'] += sd
        stats[svc] = svc_stats
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# PLOT PAGES
# ══════════════════════════════════════════════════════════════════════════════

def plot_p1_latency_timeseries(k6_data, target_rps, plot_dir):
    """Page 1: p50/p95/p99 latency over time at 5s resolution."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

    for qi, (q, q_label) in enumerate([(0.50, 'p50'), (0.95, 'p95'), (0.99, 'p99')]):
        ax = axes[qi]
        for label, df in k6_data.items():
            dur = k6_duration(df)
            dur['bin'] = (dur['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = dur.groupby('bin')['metric_value'].quantile(q).reset_index()
            ax.plot(g['bin'] / 60, g['metric_value'], label=label,
                    color=COLORS[label], lw=1.0, alpha=0.9)
        ax.set_ylabel(f'{q_label} Latency (ms)')
        ax.set_title(f'{q_label} Response Latency ({BIN_SEC}s bins)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        _shade(ax)
        _phase_labels(ax)

    axes[-1].set_xlabel('Time (minutes)')
    fig.suptitle(f'Page 1: Latency Time Series — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p01_latency_timeseries.png')


def plot_p2_error_throughput(k6_data, target_rps, plot_dir):
    """Page 2: Error rate + successful throughput + VUs."""
    fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

    # Error rate
    ax = axes[0]
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        reqs['bin'] = (reqs['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = reqs.groupby('bin').agg(
            total=('expected_response', 'count'),
            failed=('expected_response', lambda x: (~x).sum())
        ).reset_index()
        g['err_pct'] = g['failed'] / g['total'] * 100
        ax.plot(g['bin'] / 60, g['err_pct'], label=label, color=COLORS[label], lw=1.0)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title(f'Error Rate ({BIN_SEC}s bins)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)
    _phase_labels(ax)

    # Successful RPS
    ax = axes[1]
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        ok = reqs[reqs['expected_response']]
        ok_bin = (ok['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = ok_bin.value_counts().sort_index().reset_index()
        g.columns = ['bin', 'n']
        g['rps'] = g['n'] / BIN_SEC
        ax.plot(g['bin'] / 60, g['rps'], label=label, color=COLORS[label], lw=1.0)
    ax.axhline(target_rps, color='gray', ls='--', alpha=0.5, label=f'Target {target_rps} rps')
    ax.set_ylabel('Successful RPS')
    ax.set_title('Successful Throughput')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)

    # VUs
    ax = axes[2]
    for label, df in k6_data.items():
        vus = k6_vus(df)
        if not vus.empty:
            ax.plot(vus['elapsed_s'] / 60, vus['metric_value'],
                    label=label, color=COLORS[label], lw=1.0)
    ax.set_ylabel('Active VUs')
    ax.set_title('Virtual Users Over Time')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)
    ax.set_xlabel('Time (minutes)')

    fig.suptitle(f'Page 2: Error Rate, Throughput & VUs — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p02_error_throughput_vus.png')


def plot_p3_latency_cdf_violin(k6_data, target_rps, plot_dir):
    """Page 3: CDF + Violin plots per cluster."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # CDF global
    ax = fig.add_subplot(gs[0, 0])
    for label, df in k6_data.items():
        dur = k6_duration(df)
        lat = np.sort(dur['metric_value'].dropna().values)
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat, cdf, label=label, color=COLORS[label], lw=1.2)
    ax.set_xlim(0, 3000)
    ax.axhline(0.95, color='gray', ls='--', alpha=0.4, label='p95')
    ax.axhline(0.99, color='gray', ls=':', alpha=0.4, label='p99')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Global Latency CDF')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # CDF per cluster
    ax = fig.add_subplot(gs[0, 1])
    for label, df in k6_data.items():
        dur = k6_duration(df)
        ls_style = '-' if label == LABEL_A else '--'
        for cname, ckey in INGRESS_TO_CLUSTER.items():
            cl_dur = dur[dur['ingress'] == cname]['metric_value'].dropna().values
            if len(cl_dur) > 0:
                lat = np.sort(cl_dur)
                cdf = np.arange(1, len(lat) + 1) / len(lat)
                ax.plot(lat, cdf, label=f'{label} {CLUSTER_NAMES[ckey][:2]}',
                        color=CLUSTER_COLORS[ckey], ls=ls_style, lw=1.0, alpha=0.8)
    ax.set_xlim(0, 3000)
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Per-Cluster Latency CDF')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)

    # Violin plot per cluster — DMOS ON
    ax = fig.add_subplot(gs[1, 0])
    _violin_per_cluster(ax, k6_data, LABEL_A)
    ax.set_title(f'{LABEL_A} — Latency Distribution per Cluster')

    # Violin plot per cluster — B
    ax = fig.add_subplot(gs[1, 1])
    _violin_per_cluster(ax, k6_data, LABEL_B)
    ax.set_title(f'{LABEL_B} — Latency Distribution per Cluster')

    fig.suptitle(f'Page 3: Latency CDF & Distribution — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p03_cdf_violin.png')


def _violin_per_cluster(ax, k6_data, label):
    if label not in k6_data:
        return
    df = k6_data[label]
    dur = k6_duration(df)
    data_list = []
    labels_list = []
    colors_list = []
    for cname, ckey in INGRESS_TO_CLUSTER.items():
        vals = dur[dur['ingress'] == cname]['metric_value'].dropna().values
        if len(vals) > 10:
            # Cap at 5000ms for visualization
            vals = vals[vals < 5000]
            data_list.append(vals)
            labels_list.append(CLUSTER_NAMES[ckey])
            colors_list.append(CLUSTER_COLORS[ckey])

    if data_list:
        vp = ax.violinplot(data_list, showmeans=True, showmedians=True, showextrema=False)
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors_list[i])
            body.set_alpha(0.6)
        vp['cmeans'].set_color('black')
        vp['cmedians'].set_color('red')
        ax.set_xticks(range(1, len(labels_list) + 1))
        ax.set_xticklabels(labels_list, fontsize=9)
        ax.set_ylabel('Latency (ms)')
        ax.grid(True, alpha=0.2, axis='y')

        # Add p95/p99 annotations
        for i, vals in enumerate(data_list):
            p95 = np.percentile(vals, 95)
            p99 = np.percentile(vals, 99)
            ax.text(i + 1, p95, f'p95={p95:.0f}', ha='center', va='bottom', fontsize=7, color='blue')
            ax.text(i + 1, p99, f'p99={p99:.0f}', ha='center', va='bottom', fontsize=7, color='red')


def plot_p4_per_cluster_detail(k6_data, target_rps, plot_dir):
    """Page 4: Per-cluster p95 latency + error rate time series."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 16), sharex='col')

    clusters_ordered = ["cluster1", "cluster2", "cluster3"]

    for ci, ckey in enumerate(clusters_ordered):
        cname = [k for k, v in INGRESS_TO_CLUSTER.items() if v == ckey][0]

        # Left: p95 latency per cluster
        ax = axes[ci, 0]
        for label, df in k6_data.items():
            dur = k6_duration(df)
            cl = dur[dur['ingress'] == cname]
            cl = cl.copy()
            cl['bin'] = (cl['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = cl.groupby('bin')['metric_value'].quantile(0.95).reset_index()
            ax.plot(g['bin'] / 60, g['metric_value'], label=label,
                    color=COLORS[label], lw=1.0)
        ax.set_ylabel(f'p95 (ms)')
        ax.set_title(f'{CLUSTER_NAMES[ckey]} — p95 Latency')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)

        # Right: error rate per cluster
        ax = axes[ci, 1]
        for label, df in k6_data.items():
            reqs = k6_reqs(df)
            cl = reqs[reqs['ingress'] == cname].copy()
            cl['bin'] = (cl['elapsed_s'] // BIN_SEC) * BIN_SEC
            g = cl.groupby('bin').agg(
                total=('expected_response', 'count'),
                failed=('expected_response', lambda x: (~x).sum())
            ).reset_index()
            g['err_pct'] = g['failed'] / g['total'] * 100
            ax.plot(g['bin'] / 60, g['err_pct'], label=label,
                    color=COLORS[label], lw=1.0)
        ax.set_ylabel('Error %')
        ax.set_title(f'{CLUSTER_NAMES[ckey]} — Error Rate')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)

    axes[-1, 0].set_xlabel('Time (min)')
    axes[-1, 1].set_xlabel('Time (min)')
    fig.suptitle(f'Page 4: Per-Cluster Latency & Errors — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p04_per_cluster_detail.png')


def plot_p5_status_endpoint(k6_data, target_rps, plot_dir):
    """Page 5: Status codes, per-endpoint error, per-phase breakdown."""
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35)

    # Status codes
    for idx, (label, df) in enumerate(k6_data.items()):
        ax = fig.add_subplot(gs[0, idx])
        reqs = k6_reqs(df)
        sc = reqs['status_code'].dropna().astype(int).value_counts().sort_index()
        cmap = ['#4CAF50' if s < 300 else '#2196F3' if s < 400
                else '#FF9800' if s < 500 else '#F44336' for s in sc.index]
        bars = ax.bar([str(s) for s in sc.index], sc.values, color=cmap, alpha=0.85)
        for b, v in zip(bars, sc.values):
            pct = v / len(reqs) * 100
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=7)
        ax.set_title(f'{label} — Status Codes')
        ax.grid(True, alpha=0.2, axis='y')

    # Per-endpoint error rate
    ax = fig.add_subplot(gs[1, :])
    all_paths = set()
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        reqs = reqs.copy()
        reqs['path'] = reqs['url'].apply(url_to_path)
        all_paths.update(reqs['path'].unique())
    paths = sorted([p for p in all_paths if p != 'unknown'])[:15]
    x = np.arange(len(paths))
    w = 0.35
    for i, (label, df) in enumerate(k6_data.items()):
        reqs = k6_reqs(df).copy()
        reqs['path'] = reqs['url'].apply(url_to_path)
        rates = []
        for p in paths:
            pr = reqs[reqs['path'] == p]
            t = len(pr)
            f = pr[~pr['expected_response']].shape[0]
            rates.append(f / t * 100 if t > 0 else 0)
        ax.barh(x - w/2 + i*w, rates, w, label=label, color=COLORS[label], alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(paths, fontsize=7)
    ax.set_xlabel('Error Rate (%)')
    ax.set_title('Error Rate per Endpoint')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='x')

    # Per-phase summary
    ax = fig.add_subplot(gs[2, 0])
    pnames = [p[2] for p in PHASES]
    x = np.arange(len(pnames))
    w = 0.35
    for i, (label, df) in enumerate(k6_data.items()):
        reqs = k6_reqs(df)
        rates = []
        for s, e, _ in PHASES:
            pr = reqs[(reqs['elapsed_s'] >= s) & (reqs['elapsed_s'] < e)]
            t = len(pr)
            f = pr[~pr['expected_response']].shape[0]
            rates.append(f / t * 100 if t > 0 else 0)
        bars = ax.bar(x - w/2 + i*w, rates, w, label=label, color=COLORS[label], alpha=0.8)
        for b, v in zip(bars, rates):
            if v > 0.1:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                        f'{v:.2f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(pnames, fontsize=9)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate per Phase')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # Heatmap: cluster x phase (DMOS ON)
    ax = fig.add_subplot(gs[2, 1])
    if LABEL_A in k6_data:
        reqs = k6_reqs(k6_data[LABEL_A])
        clusters = list(INGRESS_TO_CLUSTER.keys())
        matrix = np.zeros((len(clusters), len(PHASES)))
        for j, (s, e, _) in enumerate(PHASES):
            for i, cl in enumerate(clusters):
                cr = reqs[(reqs['ingress'] == cl) & (reqs['elapsed_s'] >= s) & (reqs['elapsed_s'] < e)]
                t = len(cr)
                f = cr[~cr['expected_response']].shape[0]
                matrix[i, j] = f / t * 100 if t > 0 else 0
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0,
                       vmax=max(20, matrix.max()))
        ax.set_xticks(range(len(pnames)))
        ax.set_xticklabels(pnames, fontsize=8, rotation=15, ha='right')
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels([CLUSTER_NAMES[INGRESS_TO_CLUSTER[c]] for c in clusters], fontsize=9)
        for i in range(len(clusters)):
            for j in range(len(PHASES)):
                ax.text(j, i, f'{matrix[i,j]:.2f}%', ha='center', va='center', fontsize=8,
                        color='white' if matrix[i,j] > 10 else 'black')
        ax.set_title(f'{LABEL_A} — Error Heatmap (Cluster x Phase)')
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f'Page 5: Status Codes & Endpoint Analysis — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p05_status_endpoint.png')


def plot_p6_latency_heatmap(k6_data, target_rps, plot_dir):
    """Page 6: Temporal latency heatmap (time x latency bucket)."""
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)

    lat_max = 3000
    lat_bins = np.linspace(0, lat_max, 120)  # 25ms buckets
    time_bins = np.arange(0, 860, BIN_SEC)

    for idx, (label, df) in enumerate(k6_data.items()):
        ax = axes[idx]
        dur = k6_duration(df)
        lat = dur['metric_value'].clip(upper=lat_max).values
        t = dur['elapsed_s'].values

        h, xedges, yedges = np.histogram2d(t, lat, bins=[time_bins, lat_bins])
        # Normalize per time bin
        h_norm = h / h.sum(axis=1, keepdims=True)
        h_norm = np.nan_to_num(h_norm)

        im = ax.pcolormesh(xedges / 60, yedges, h_norm.T,
                           cmap='hot_r', vmin=0, vmax=0.15, shading='auto')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'{label} — Latency Density Over Time')
        fig.colorbar(im, ax=ax, shrink=0.7, label='Density')
        _shade(ax)

    axes[-1].set_xlabel('Time (minutes)')
    fig.suptitle(f'Page 6: Latency Heatmap — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p06_latency_heatmap.png')


def plot_p7_dmos_internals(jsonl_on, target_rps, plot_dir):
    """Page 7: DMOS internals — traffic, replicas, scores."""
    if not jsonl_on:
        return

    ts = extract_timeseries(jsonl_on, "frontend")
    if ts.empty:
        return

    fig, axes = plt.subplots(3, 2, figsize=(20, 16))

    # Actual vs Predicted
    ax = axes[0, 0]
    ax.plot(ts['elapsed_s'] / 60, ts['actual_traffic'], label='Actual', color='#1565C0', lw=1.5)
    ax.plot(ts['elapsed_s'] / 60, ts['predicted_traffic'], label='Predicted', color='#E65100', lw=1.5, ls='--')
    ax.fill_between(ts['elapsed_s'] / 60, ts['actual_traffic'], ts['predicted_traffic'],
                    alpha=0.1, color='#FF9800')
    ax.set_ylabel('Traffic (req/s)')
    ax.set_title('Frontend: Actual vs Predicted Traffic')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)

    # Stacked replicas
    ax = axes[0, 1]
    bottom = np.zeros(len(ts))
    for c in ["cluster1", "cluster2", "cluster3"]:
        vals = ts[f'{c}_replicas'].values.astype(float)
        ax.bar(ts['elapsed_s'] / 60, vals, bottom=bottom, width=0.2,
               label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], alpha=0.8)
        bottom += vals
    ax.set_ylabel('Replicas')
    ax.set_title('Frontend Replicas (Stacked)')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    _shade(ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Per-cluster replicas (line)
    ax = axes[1, 0]
    for c in ["cluster1", "cluster2", "cluster3"]:
        ax.step(ts['elapsed_s'] / 60, ts[f'{c}_replicas'],
                label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.5, where='post')
    ax.set_ylabel('Replicas')
    ax.set_title('Frontend Replicas per Cluster')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Cluster scores
    ax = axes[1, 1]
    for c in ["cluster1", "cluster2", "cluster3"]:
        ax.plot(ts['elapsed_s'] / 60, ts[f'{c}_score'],
                label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.5)
    ax.set_ylabel('Multi-Objective Score')
    ax.set_title('DMOS Cluster Scores (Frontend)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)

    # Per-cluster predicted traffic
    ax = axes[2, 0]
    for c in ["cluster1", "cluster2", "cluster3"]:
        ax.plot(ts['elapsed_s'] / 60, ts[f'{c}_predicted'],
                label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.2)
    ax.set_ylabel('Predicted Traffic (req/s)')
    ax.set_title('Per-Cluster Predicted Traffic')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)
    ax.set_xlabel('Time (minutes)')

    # Prediction scatter
    ax = axes[2, 1]
    mask = ts['actual_traffic'] > 1
    if mask.any():
        ax.scatter(ts.loc[mask, 'actual_traffic'], ts.loc[mask, 'predicted_traffic'],
                   alpha=0.5, s=15, color='#1565C0', edgecolors='none')
        max_val = max(ts.loc[mask, 'actual_traffic'].max(),
                      ts.loc[mask, 'predicted_traffic'].max())
        ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, label='Perfect')
        ax.set_xlabel('Actual Traffic (req/s)')
        ax.set_ylabel('Predicted Traffic (req/s)')
    ax.set_title('Prediction Accuracy Scatter')
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.suptitle(f'Page 7: DMOS Internals — Frontend — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p07_dmos_internals.png')


def plot_p8_fairness(jsonl_on, target_rps, plot_dir):
    """Page 8: Jain index, scaling events, all services replicas."""
    if not jsonl_on:
        return

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Jain index over time (frontend)
    ax = fig.add_subplot(gs[0, 0])
    jain = compute_jain_index(jsonl_on, "frontend")
    ax.plot(jain['elapsed_s'] / 60, jain['jain_index'], color='#2E7D32', lw=1.5, label='Replica Fairness')
    ax.plot(jain['elapsed_s'] / 60, jain['jain_index_score'], color='#E65100', lw=1.5, ls='--', label='Score Fairness')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4, label='Perfect (1.0)')
    ax.axhline(1/3, color='red', ls=':', alpha=0.4, label='Worst (1/N)')
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title("Jain's Index Over Time (Frontend)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)

    # Jain box per service
    ax = fig.add_subplot(gs[0, 1])
    jain_per_svc = {}
    for svc in SERVICES:
        j = compute_jain_index(jsonl_on, svc)
        jain_per_svc[SERVICE_SHORT[svc]] = j['jain_index'].values
    bp = ax.boxplot(jain_per_svc.values(), labels=jain_per_svc.keys(), patch_artist=True)
    box_colors = ['#4CAF50', '#FF9800', '#9C27B0', '#2196F3', '#F44336']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax.set_ylabel("Jain's Index")
    ax.set_title("Jain's Index Distribution per Service")
    ax.grid(True, alpha=0.2, axis='y')

    # Scaling events
    ax = fig.add_subplot(gs[1, 0])
    stats = compute_scaling_stats(jsonl_on)
    svc_list = [s for s in SERVICES if stats[s]['total_scale_up'] + stats[s]['total_scale_down'] > 0]
    if not svc_list:
        svc_list = SERVICES
    x = np.arange(len(svc_list))
    w = 0.25
    for i, c in enumerate(["cluster1", "cluster2", "cluster3"]):
        su = [stats[s]['per_cluster'].get(c, {}).get('scale_up', 0) for s in svc_list]
        sd = [-stats[s]['per_cluster'].get(c, {}).get('scale_down', 0) for s in svc_list]
        ax.bar(x + i * w, su, w, label=f'{CLUSTER_NAMES[c][:2]} up', color=CLUSTER_COLORS[c], alpha=0.8)
        ax.bar(x + i * w, sd, w, color=CLUSTER_COLORS[c], alpha=0.3)
    ax.set_xticks(x + w)
    ax.set_xticklabels([SERVICE_SHORT[s] for s in svc_list], fontsize=8)
    ax.set_ylabel('Events (+up / -down)')
    ax.set_title('Scaling Events per Cluster')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(0, color='black', lw=0.5)

    # All services total replicas
    ax = fig.add_subplot(gs[1, 1])
    for svc in SERVICES:
        ts = extract_timeseries(jsonl_on, svc)
        if not ts.empty:
            ax.step(ts['elapsed_s'] / 60, ts['total_replicas'],
                    label=SERVICE_SHORT[svc], lw=1.2, where='post')
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Total Replicas')
    ax.set_title('Total Replicas per Service')
    ax.legend()
    ax.grid(True, alpha=0.2)
    _shade(ax)

    fig.suptitle(f'Page 8: Fairness & Scaling — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p08_fairness_scaling.png')


def plot_p9_resources(jsonl_on, jsonl_off, target_rps, plot_dir):
    """Page 9: CPU, memory, network per cluster."""
    data_sets = {}
    if jsonl_on:
        data_sets[LABEL_A] = jsonl_on
    if jsonl_off:
        data_sets[LABEL_B] = jsonl_off
    if not data_sets:
        return

    fig, axes = plt.subplots(3, 2, figsize=(20, 16))

    # Frontend CPU per cluster
    for idx, (label, snaps) in enumerate(data_sets.items()):
        ax = axes[0, idx]
        ts = extract_timeseries(snaps, "frontend")
        for c in ["cluster1", "cluster2", "cluster3"]:
            vals = pd.to_numeric(pd.Series(ts[f'{c}_cpu'].values), errors='coerce')
            ax.plot(ts['elapsed_s'] / 60, vals * 1000, label=CLUSTER_NAMES[c],
                    color=CLUSTER_COLORS[c], lw=1.2)
        ax.set_ylabel('CPU (millicores)')
        ax.set_title(f'{label} — Frontend CPU')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)

    # Total CPU all services
    for idx, (label, snaps) in enumerate(data_sets.items()):
        ax = axes[1, idx]
        for c in ["cluster1", "cluster2", "cluster3"]:
            total_cpu = []
            for snap in snaps:
                cpu_sum = 0
                for svc in SERVICES:
                    val = snap.get('resources', {}).get(c, {}).get(svc, {}).get('cpu_cores')
                    if val is not None:
                        cpu_sum += val
                total_cpu.append(cpu_sum * 1000)
            elapsed = [s['_elapsed_s'] / 60 for s in snaps]
            ax.plot(elapsed, total_cpu, label=CLUSTER_NAMES[c],
                    color=CLUSTER_COLORS[c], lw=1.2)
        ax.axhline(4000, color='red', ls='--', alpha=0.4, label='Node limit (4000m)')
        ax.set_ylabel('CPU (millicores)')
        ax.set_title(f'{label} — Total CPU per Cluster')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)

    # Peak memory bar
    ax = axes[2, 0]
    clusters = ["cluster1", "cluster2", "cluster3"]
    x = np.arange(len(clusters))
    w = 0.35
    for i, (label, snaps) in enumerate(data_sets.items()):
        peak_mem = []
        for c in clusters:
            max_mem = 0
            for snap in snaps:
                mem_sum = 0
                for svc in SERVICES:
                    val = snap.get('resources', {}).get(c, {}).get(svc, {}).get('memory_mb')
                    if val is not None:
                        mem_sum += val
                max_mem = max(max_mem, mem_sum)
            peak_mem.append(max_mem)
        ax.bar(x - w/2 + i*w, peak_mem, w, label=label, color=COLORS[label], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CLUSTER_NAMES[c] for c in clusters], fontsize=8)
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Peak Memory per Cluster')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # Traffic distribution
    ax = axes[2, 1]
    if jsonl_on:
        ts = extract_timeseries(jsonl_on, "frontend")
        for c in ["cluster1", "cluster2", "cluster3"]:
            vals = pd.to_numeric(pd.Series(ts[f'{c}_traffic_pct'].values), errors='coerce')
            ax.plot(ts['elapsed_s'] / 60, vals, label=CLUSTER_NAMES[c],
                    color=CLUSTER_COLORS[c], lw=1.2)
        ax.set_ylabel('Traffic Share (%)')
        ax.set_title(f'{LABEL_A} — Traffic Distribution')
        ax.legend()
        ax.grid(True, alpha=0.2)
        _shade(ax)
    ax.set_xlabel('Time (minutes)')

    fig.suptitle(f'Page 9: Resource Utilization — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p09_resources.png')


def plot_p10_backend_services(jsonl_on, target_rps, plot_dir):
    """Page 10: Backend services replicas + prediction accuracy."""
    if not jsonl_on:
        return

    backend = [s for s in SERVICES if s != "frontend"]
    fig, axes = plt.subplots(len(backend), 2, figsize=(20, 5 * len(backend)))

    for si, svc in enumerate(backend):
        ts = extract_timeseries(jsonl_on, svc)
        if ts.empty:
            continue

        # Replicas per cluster
        ax = axes[si, 0]
        for c in ["cluster1", "cluster2", "cluster3"]:
            ax.step(ts['elapsed_s'] / 60, ts[f'{c}_replicas'],
                    label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.2, where='post')
        ax.set_ylabel('Replicas')
        ax.set_title(f'{SERVICE_SHORT[svc]} — Replicas per Cluster')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Traffic
        ax = axes[si, 1]
        ax.plot(ts['elapsed_s'] / 60, ts['actual_traffic'], label='Actual', color='#1565C0', lw=1.2)
        ax.plot(ts['elapsed_s'] / 60, ts['predicted_traffic'], label='Predicted', color='#E65100', lw=1.2, ls='--')
        ax.set_ylabel('Traffic (req/s)')
        ax.set_title(f'{SERVICE_SHORT[svc]} — Traffic')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        _shade(ax)

    axes[-1, 0].set_xlabel('Time (minutes)')
    axes[-1, 1].set_xlabel('Time (minutes)')
    fig.suptitle(f'Page 10: Backend Services — {target_rps} rps', fontsize=14, y=1.01)
    _savefig(fig, plot_dir, 'p10_backend_services.png')


def plot_p0_summary(k6_data, jsonl_on, target_rps, plot_dir):
    """Page 0: Summary table with all key metrics."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')

    rows = []
    for label, df in k6_data.items():
        dur = k6_duration(df)
        reqs = k6_reqs(df)
        total = len(reqs)
        failed = reqs[~reqs['expected_response']].shape[0]
        lat = dur['metric_value']
        row = [label, f"{total:,}", f"{total - failed:,}", f"{failed:,}",
               f"{failed/total*100:.2f}%",
               f"{lat.mean():.0f}", f"{lat.quantile(0.50):.0f}",
               f"{lat.quantile(0.95):.0f}", f"{lat.quantile(0.99):.0f}",
               f"{lat.max():.0f}"]
        rows.append(row)

    if len(rows) == 2:
        delta = ["DELTA"]
        on_reqs = k6_reqs(k6_data[LABEL_A])
        off_reqs = k6_reqs(k6_data[LABEL_B])
        on_total = len(on_reqs)
        off_total = len(off_reqs)
        on_fail = on_reqs[~on_reqs['expected_response']].shape[0]
        off_fail = off_reqs[~off_reqs['expected_response']].shape[0]
        delta.append(f"{on_total - off_total:+,}")
        delta.append(f"{(on_total - on_fail) - (off_total - off_fail):+,}")
        delta.append(f"{on_fail - off_fail:+,}")
        on_err = on_fail / on_total * 100
        off_err = off_fail / off_total * 100
        delta.append(f"{on_err - off_err:+.2f}pp")
        on_dur = k6_duration(k6_data[LABEL_A])['metric_value']
        off_dur = k6_duration(k6_data[LABEL_B])['metric_value']
        for q in [lambda x: x.mean(), lambda x: x.quantile(0.50),
                  lambda x: x.quantile(0.95), lambda x: x.quantile(0.99),
                  lambda x: x.max()]:
            d = q(on_dur) - q(off_dur)
            delta.append(f"{d:+.0f}ms")
        rows.append(delta)

    cols = ['Test', 'Total', 'Success', 'Failed', 'Error%',
            'Avg(ms)', 'p50(ms)', 'p95(ms)', 'p99(ms)', 'Max(ms)']

    table = ax.table(cellText=rows, colLabels=cols, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.0)

    for j in range(len(cols)):
        table[0, j].set_facecolor('#263238')
        table[0, j].set_text_props(color='white', weight='bold')
    table[1, 0].set_facecolor('#E3F2FD')
    table[2, 0].set_facecolor('#FFEBEE')
    if len(rows) > 2:
        for j in range(len(cols)):
            table[3, j].set_facecolor('#FFF9C4')

    # Per-cluster summary
    if len(k6_data) == 2:
        pc_rows = []
        for cname, ckey in INGRESS_TO_CLUSTER.items():
            row = [CLUSTER_NAMES[ckey]]
            for label in [LABEL_A, LABEL_B]:
                df = k6_data[label]
                reqs = k6_reqs(df)
                dur_df = k6_duration(df)
                cr = reqs[reqs['ingress'] == cname]
                cl_dur = dur_df[dur_df['ingress'] == cname]['metric_value']
                t = len(cr)
                f = cr[~cr['expected_response']].shape[0]
                row.extend([
                    f"{f/t*100:.2f}%" if t > 0 else "N/A",
                    f"{cl_dur.quantile(0.95):.0f}" if len(cl_dur) > 0 else "N/A",
                ])
            pc_rows.append(row)

        y_pos = 0.58
        ax.text(0.5, y_pos + 0.04, 'Per-Cluster Summary', ha='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)

        short_b = LABEL_B.replace("DMOS ", "")
        pc_cols = ['Cluster', 'ON err%', 'ON p95', f'{short_b} err%', f'{short_b} p95']
        table2 = ax.table(cellText=pc_rows, colLabels=pc_cols,
                          loc='center', cellLoc='center',
                          bbox=[0.15, y_pos - 0.20, 0.7, 0.22])
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        for j in range(len(pc_cols)):
            table2[0, j].set_facecolor('#455A64')
            table2[0, j].set_text_props(color='white', weight='bold')

    # DMOS metrics
    if jsonl_on:
        dmos_rows = []
        jain = compute_jain_index(jsonl_on, "frontend")
        dmos_rows.append(["Jain Index (mean)", f"{jain['jain_index'].mean():.3f}"])
        dmos_rows.append(["Jain Index (min)", f"{jain['jain_index'].min():.3f}"])
        acc = compute_prediction_accuracy(jsonl_on, "frontend")
        if acc:
            dmos_rows.append(["Prediction MAPE", f"{acc['mape']:.1f}%"])
            dmos_rows.append(["Prediction R²", f"{acc['r_squared']:.3f}"])
        stats = compute_scaling_stats(jsonl_on)
        total_su = sum(s['total_scale_up'] for s in stats.values())
        total_sd = sum(s['total_scale_down'] for s in stats.values())
        dmos_rows.append(["Total Scale-up Events", str(total_su)])
        dmos_rows.append(["Total Scale-down Events", str(total_sd)])

        ax.text(0.5, 0.30, 'DMOS Internal Metrics', ha='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)

        table3 = ax.table(cellText=dmos_rows,
                          colLabels=['Metric', 'Value'],
                          loc='lower center', cellLoc='center',
                          bbox=[0.25, 0.02, 0.5, 0.27])
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        for j in range(2):
            table3[0, j].set_facecolor('#455A64')
            table3[0, j].set_text_props(color='white', weight='bold')

    ax.set_title(f'DMOS Evaluation Summary — {target_rps} rps', fontsize=16, pad=30)
    _savefig(fig, plot_dir, 'p00_summary.png')


def plot_p11_exclude_c3(k6_data, target_rps, plot_dir):
    """Page 11: C1+C2 only — isolates DMOS effect from high-latency cluster.

    Cluster3 (Warsaw/PL, 350ms netem) has ~86% error rate in BOTH scenarios,
    dominated by gRPC timeout due to RTT. Excluding it shows the true DMOS
    contribution on the healthy clusters.
    """
    # Filter: keep only cluster1 (c1-DE) and cluster2 (c2-FR)
    LOW_LAT = ['c1-DE', 'c2-FR']
    LOW_LAT_NAMES = {
        'c1-DE': 'C1 — Frankfurt (DE)',
        'c2-FR': 'C2 — Paris (FR)',
    }

    fig = plt.figure(figsize=(22, 22))
    gs = gridspec.GridSpec(4, 2, hspace=0.40, wspace=0.30)

    # ── 1. Summary table C1+C2 only ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')

    rows = []
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        dur = k6_duration(df)
        reqs_f = reqs[reqs['ingress'].isin(LOW_LAT)]
        dur_f = dur[dur['ingress'].isin(LOW_LAT)]
        total = len(reqs_f)
        failed = reqs_f[~reqs_f['expected_response']].shape[0]
        lat = dur_f['metric_value']
        rows.append([
            label,
            f"{total:,}", f"{total - failed:,}", f"{failed:,}",
            f"{failed/total*100:.2f}%" if total > 0 else "N/A",
            f"{lat.mean():.0f}", f"{lat.quantile(0.50):.0f}",
            f"{lat.quantile(0.95):.0f}", f"{lat.quantile(0.99):.0f}",
        ])

    # Delta row
    if len(rows) == 2:
        on_reqs = k6_reqs(k6_data[LABEL_A])
        off_reqs = k6_reqs(k6_data[LABEL_B])
        on_f = on_reqs[on_reqs['ingress'].isin(LOW_LAT)]
        off_f = off_reqs[off_reqs['ingress'].isin(LOW_LAT)]
        on_dur = k6_duration(k6_data[LABEL_A])
        off_dur = k6_duration(k6_data[LABEL_B])
        on_dur_f = on_dur[on_dur['ingress'].isin(LOW_LAT)]['metric_value']
        off_dur_f = off_dur[off_dur['ingress'].isin(LOW_LAT)]['metric_value']
        on_total = len(on_f)
        off_total = len(off_f)
        on_fail = on_f[~on_f['expected_response']].shape[0]
        off_fail = off_f[~off_f['expected_response']].shape[0]
        on_err = on_fail / on_total * 100 if on_total > 0 else 0
        off_err = off_fail / off_total * 100 if off_total > 0 else 0
        delta = ["DELTA"]
        delta.append(f"{on_total - off_total:+,}")
        delta.append(f"{(on_total - on_fail) - (off_total - off_fail):+,}")
        delta.append(f"{on_fail - off_fail:+,}")
        delta.append(f"{on_err - off_err:+.2f}pp")
        for qfn in [lambda x: x.mean(), lambda x: x.quantile(0.50),
                     lambda x: x.quantile(0.95), lambda x: x.quantile(0.99)]:
            d = qfn(on_dur_f) - qfn(off_dur_f)
            delta.append(f"{d:+.0f}ms")
        rows.append(delta)

    cols = ['Test', 'Total', 'Success', 'Failed', 'Error%',
            'Avg(ms)', 'p50(ms)', 'p95(ms)', 'p99(ms)']
    table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 2.0)
    for j in range(len(cols)):
        table[0, j].set_facecolor('#263238')
        table[0, j].set_text_props(color='white', weight='bold')
    table[1, 0].set_facecolor('#E3F2FD')
    table[2, 0].set_facecolor('#FFEBEE')
    if len(rows) > 2:
        for j in range(len(cols)):
            table[3, j].set_facecolor('#FFF9C4')
    ax.set_title('Summary — C1 (Frankfurt) + C2 (Paris) only, excluding C3 (Warsaw, 350ms RTT)',
                 fontsize=12, pad=15)

    # ── 2. Error rate over time: All 3 clusters vs C1+C2 only ───────────────
    ax = fig.add_subplot(gs[1, 0])
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        # All clusters
        reqs_all = reqs.copy()
        reqs_all['bin'] = (reqs_all['elapsed_s'] // BIN_SEC) * BIN_SEC
        g_all = reqs_all.groupby('bin').agg(
            total=('expected_response', 'count'),
            failed=('expected_response', lambda x: (~x).sum())
        ).reset_index()
        g_all['err_pct'] = g_all['failed'] / g_all['total'] * 100
        ax.plot(g_all['bin'] / 60, g_all['err_pct'],
                color=COLORS[label], lw=0.8, alpha=0.35, ls='--')

        # C1+C2 only
        reqs_f = reqs[reqs['ingress'].isin(LOW_LAT)].copy()
        reqs_f['bin'] = (reqs_f['elapsed_s'] // BIN_SEC) * BIN_SEC
        g_f = reqs_f.groupby('bin').agg(
            total=('expected_response', 'count'),
            failed=('expected_response', lambda x: (~x).sum())
        ).reset_index()
        g_f['err_pct'] = g_f['failed'] / g_f['total'] * 100
        ax.plot(g_f['bin'] / 60, g_f['err_pct'],
                label=f'{label} (C1+C2)', color=COLORS[label], lw=1.2)

    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate: C1+C2 (solid) vs All Clusters (dashed)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    _shade(ax)
    _phase_labels(ax)

    # ── 3. p95 latency: C1+C2 only ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for label, df in k6_data.items():
        dur = k6_duration(df)
        dur_f = dur[dur['ingress'].isin(LOW_LAT)].copy()
        dur_f['bin'] = (dur_f['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = dur_f.groupby('bin')['metric_value'].quantile(0.95).reset_index()
        ax.plot(g['bin'] / 60, g['metric_value'],
                label=f'{label} (C1+C2)', color=COLORS[label], lw=1.2)
    ax.set_ylabel('p95 Latency (ms)')
    ax.set_title('p95 Latency — C1+C2 Only')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    _shade(ax)
    _phase_labels(ax)

    # ── 4. Per-phase error rate comparison: C1+C2 only ───────────────────────
    ax = fig.add_subplot(gs[2, 0])
    pnames = [p[2] for p in PHASES]
    x = np.arange(len(pnames))
    w = 0.35
    for i, (label, df) in enumerate(k6_data.items()):
        reqs = k6_reqs(df)
        reqs_f = reqs[reqs['ingress'].isin(LOW_LAT)]
        rates = []
        for s, e, _ in PHASES:
            pr = reqs_f[(reqs_f['elapsed_s'] >= s) & (reqs_f['elapsed_s'] < e)]
            t = len(pr)
            f = pr[~pr['expected_response']].shape[0]
            rates.append(f / t * 100 if t > 0 else 0)
        bars = ax.bar(x - w/2 + i*w, rates, w, label=label, color=COLORS[label], alpha=0.8)
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pnames, fontsize=9)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate per Phase — C1+C2 Only')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # ── 5. Per-phase error: All clusters (for comparison) ────────────────────
    ax = fig.add_subplot(gs[2, 1])
    for i, (label, df) in enumerate(k6_data.items()):
        reqs = k6_reqs(df)
        rates = []
        for s, e, _ in PHASES:
            pr = reqs[(reqs['elapsed_s'] >= s) & (reqs['elapsed_s'] < e)]
            t = len(pr)
            f = pr[~pr['expected_response']].shape[0]
            rates.append(f / t * 100 if t > 0 else 0)
        bars = ax.bar(x - w/2 + i*w, rates, w, label=label, color=COLORS[label], alpha=0.8)
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pnames, fontsize=9)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate per Phase — All 3 Clusters')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    # ── 6. CDF: C1+C2 only ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    for label, df in k6_data.items():
        dur = k6_duration(df)
        dur_f = dur[dur['ingress'].isin(LOW_LAT)]
        lat = np.sort(dur_f['metric_value'].dropna().values)
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat, cdf, label=f'{label} (C1+C2)', color=COLORS[label], lw=1.2)
    ax.set_xlim(0, 2000)
    ax.axhline(0.95, color='gray', ls='--', alpha=0.4, label='p95')
    ax.axhline(0.99, color='gray', ls=':', alpha=0.4, label='p99')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Latency CDF — C1+C2 Only')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── 7. Successful RPS: C1+C2 only ───────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        ok = reqs[(reqs['expected_response']) & (reqs['ingress'].isin(LOW_LAT))].copy()
        ok_bin = (ok['elapsed_s'] // BIN_SEC) * BIN_SEC
        g = ok_bin.value_counts().sort_index().reset_index()
        g.columns = ['bin', 'n']
        g['rps'] = g['n'] / BIN_SEC
        ax.plot(g['bin'] / 60, g['rps'], label=f'{label} (C1+C2)',
                color=COLORS[label], lw=1.2)
    # C1+C2 get ~67% of traffic
    ax.axhline(target_rps * 2/3, color='gray', ls='--', alpha=0.5,
               label=f'Expected ~{target_rps*2/3:.0f} rps (2/3 of {target_rps})')
    ax.set_ylabel('Successful RPS')
    ax.set_title('Successful Throughput — C1+C2 Only')
    ax.set_xlabel('Time (minutes)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    _shade(ax)

    fig.suptitle(
        f'Page 11: Impact of High-Latency Cluster — C1+C2 vs All — {target_rps} rps\n'
        f'C3 (Warsaw, 350ms RTT) has ~86% error rate in both ON and OFF — excluding it isolates DMOS effect',
        fontsize=13, y=1.01)
    _savefig(fig, plot_dir, 'p11_exclude_c3_analysis.png')


# ══════════════════════════════════════════════════════════════════════════════
# JSON + CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(k6_data, jsonl_on, jsonl_off, target_rps, plot_dir):
    report = {"generated_at": datetime.now().isoformat(), "target_rps": target_rps, "tests": {}}

    for label, df in k6_data.items():
        dur = k6_duration(df)
        reqs = k6_reqs(df)
        total = len(reqs)
        failed = reqs[~reqs['expected_response']].shape[0]
        lat = dur['metric_value']

        test_report = {
            "total_requests": int(total),
            "successful_requests": int(total - failed),
            "failed_requests": int(failed),
            "error_rate_pct": round(failed / total * 100, 4) if total > 0 else 0,
            "latency": {
                "avg_ms": round(float(lat.mean()), 1),
                "p50_ms": round(float(lat.quantile(0.50)), 1),
                "p90_ms": round(float(lat.quantile(0.90)), 1),
                "p95_ms": round(float(lat.quantile(0.95)), 1),
                "p99_ms": round(float(lat.quantile(0.99)), 1),
                "max_ms": round(float(lat.max()), 1),
            },
            "per_cluster": {},
            "per_phase": {},
        }

        for ing, cl in INGRESS_TO_CLUSTER.items():
            cr = reqs[reqs['ingress'] == ing]
            cl_dur = dur[dur['ingress'] == ing]['metric_value']
            t = len(cr)
            f = cr[~cr['expected_response']].shape[0]
            test_report["per_cluster"][cl] = {
                "requests": int(t),
                "error_rate_pct": round(f / t * 100, 4) if t > 0 else 0,
                "p95_ms": round(float(cl_dur.quantile(0.95)), 1) if len(cl_dur) > 0 else None,
                "p99_ms": round(float(cl_dur.quantile(0.99)), 1) if len(cl_dur) > 0 else None,
            }

        for start, end, name in PHASES:
            pr = reqs[(reqs['elapsed_s'] >= start) & (reqs['elapsed_s'] < end)]
            pd_dur = dur[(dur['elapsed_s'] >= start) & (dur['elapsed_s'] < end)]
            t = len(pr)
            f = pr[~pr['expected_response']].shape[0]
            test_report["per_phase"][name] = {
                "requests": int(t),
                "error_rate_pct": round(f / t * 100, 4) if t > 0 else 0,
                "p95_ms": round(float(pd_dur['metric_value'].quantile(0.95)), 1) if len(pd_dur) > 0 else None,
            }

        report["tests"][label] = test_report

    if jsonl_on:
        dmos_report = {"services": {}}
        for svc in SERVICES:
            acc = compute_prediction_accuracy(jsonl_on, svc)
            jain = compute_jain_index(jsonl_on, svc)
            svc_report = {
                "jain_index_mean": round(float(jain['jain_index'].mean()), 4),
                "jain_index_min": round(float(jain['jain_index'].min()), 4),
            }
            if acc:
                svc_report["prediction"] = {
                    "mape": round(acc['mape'], 2),
                    "rmse": round(acc['rmse'], 3),
                    "r_squared": round(acc['r_squared'], 4),
                    "directional_accuracy_pct": round(acc['directional_accuracy'], 1),
                }
            dmos_report["services"][svc] = svc_report

        stats = compute_scaling_stats(jsonl_on)
        dmos_report["scaling_events"] = {}
        for svc, s in stats.items():
            dmos_report["scaling_events"][svc] = {
                "total_scale_up": s['total_scale_up'],
                "total_scale_down": s['total_scale_down'],
                "per_cluster": s['per_cluster'],
            }
        report["dmos"] = dmos_report

    path = plot_dir / 'analysis_report.json'
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"    Saved: analysis_report.json")
    return report


def print_report(report):
    print("\n" + "=" * 72)
    print("  COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 72)

    for label, data in report.get("tests", {}).items():
        print(f"\n  {'─' * 52}")
        print(f"  {label}")
        print(f"  {'─' * 52}")
        print(f"    Total requests.......... {data['total_requests']:,}")
        print(f"    Successful.............. {data['successful_requests']:,}")
        print(f"    Failed.................. {data['failed_requests']:,}")
        print(f"    Error rate.............. {data['error_rate_pct']:.4f}%")
        lat = data['latency']
        print(f"    Avg latency............. {lat['avg_ms']}ms")
        print(f"    p50..................... {lat['p50_ms']}ms")
        print(f"    p95..................... {lat['p95_ms']}ms")
        print(f"    p99..................... {lat['p99_ms']}ms")
        print(f"    max..................... {lat['max_ms']}ms")

        print(f"\n    Per Cluster:")
        for cl, cd in data.get('per_cluster', {}).items():
            print(f"      {CLUSTER_NAMES.get(cl, cl):.<35s} err={cd['error_rate_pct']:.4f}%  "
                  f"p95={cd['p95_ms']}ms  p99={cd['p99_ms']}ms")

        print(f"\n    Per Phase:")
        for phase, pd_data in data.get('per_phase', {}).items():
            print(f"      {phase:.<30s} err={pd_data['error_rate_pct']:.4f}%  "
                  f"p95={pd_data['p95_ms']}ms  reqs={pd_data['requests']:,}")

    tests = list(report.get("tests", {}).values())
    if len(tests) == 2:
        on, off = tests[0], tests[1]
        print(f"\n  {'═' * 52}")
        print(f"  DELTA ({LABEL_A} - {LABEL_B})")
        print(f"  {'═' * 52}")
        err_d = on['error_rate_pct'] - off['error_rate_pct']
        err_red = ((off['error_rate_pct'] - on['error_rate_pct']) / off['error_rate_pct'] * 100) if off['error_rate_pct'] > 0 else 0
        print(f"    Error rate delta........ {err_d:+.4f} pp")
        print(f"    Error reduction......... {err_red:+.1f}%")
        print(f"    Extra successful reqs... {on['successful_requests'] - off['successful_requests']:+,}")
        print(f"    p95 delta............... {on['latency']['p95_ms'] - off['latency']['p95_ms']:+.0f}ms")
        print(f"    p99 delta............... {on['latency']['p99_ms'] - off['latency']['p99_ms']:+.0f}ms")

    dmos = report.get("dmos", {})
    if dmos:
        print(f"\n  {'═' * 52}")
        print(f"  DMOS INTERNAL METRICS")
        print(f"  {'═' * 52}")
        for svc, sd in dmos.get("services", {}).items():
            print(f"\n    {svc}:")
            print(f"      Jain Index (mean)...... {sd['jain_index_mean']}")
            print(f"      Jain Index (min)....... {sd['jain_index_min']}")
            pred = sd.get("prediction", {})
            if pred:
                print(f"      MAPE................... {pred['mape']}%")
                print(f"      R²..................... {pred['r_squared']}")
                print(f"      Dir. Accuracy.......... {pred['directional_accuracy_pct']}%")

        print(f"\n    Scaling Events:")
        for svc, se in dmos.get("scaling_events", {}).items():
            print(f"      {svc:.<30s} up={se['total_scale_up']}  down={se['total_scale_down']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DMOS High-Resolution Analysis (DPI 600)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--k6-on", required=True, help="k6 CSV for DMOS ON")
    parser.add_argument("--k6-off", required=True, help="k6 CSV for comparison (DMOS OFF or HPA)")
    parser.add_argument("--jsonl-on", help="DMOS/Prometheus JSONL for DMOS ON")
    parser.add_argument("--jsonl-off", help="DMOS/Prometheus JSONL for comparison")
    parser.add_argument("--label-b", default="DMOS OFF",
                        help="Label for the comparison scenario (default: 'DMOS OFF', use 'HPA' for HPA)")
    parser.add_argument("--rps", type=int, default=100, help="Target RPS (for labels)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI (default 600)")
    args = parser.parse_args()

    global TARGET_DPI, LABEL_B, COLORS, COLORS_LIGHT
    TARGET_DPI = args.dpi
    LABEL_B = args.label_b
    # Update color maps with the correct label
    b_color, b_light = COLORS_B_MAP.get(LABEL_B, ("#C62828", "#EF9A9A"))
    COLORS = {LABEL_A: "#1565C0", LABEL_B: b_color}
    COLORS_LIGHT = {LABEL_A: "#90CAF9", LABEL_B: b_light}

    if args.output_dir:
        plot_dir = Path(args.output_dir)
    else:
        plot_dir = Path(args.k6_on).parent / f"plots_{args.rps}rps"
    plot_dir.mkdir(parents=True, exist_ok=True)

    target_rps = args.rps

    print("=" * 62)
    print(f"  DMOS High-Resolution Analysis ({TARGET_DPI} DPI)")
    print(f"  Comparison: {LABEL_A} vs {LABEL_B}")
    print(f"  Target: {target_rps} rps")
    print("=" * 62)

    # Load data
    print("\n[1/4] Loading k6 CSV data...")
    k6_data = {
        LABEL_A: load_k6_csv(Path(args.k6_on), LABEL_A),
        LABEL_B: load_k6_csv(Path(args.k6_off), LABEL_B),
    }

    jsonl_on = None
    jsonl_off = None
    print("\n[2/4] Loading JSONL metrics...")
    if args.jsonl_on:
        jsonl_on = load_jsonl(Path(args.jsonl_on), LABEL_A)
    if args.jsonl_off:
        jsonl_off = load_jsonl(Path(args.jsonl_off), LABEL_B)

    # Generate all plots
    print(f"\n[3/4] Generating plots in {plot_dir}/ (DPI={TARGET_DPI})...")

    # k6 plots (always)
    plot_p0_summary(k6_data, jsonl_on, target_rps, plot_dir)
    plot_p1_latency_timeseries(k6_data, target_rps, plot_dir)
    plot_p2_error_throughput(k6_data, target_rps, plot_dir)
    plot_p3_latency_cdf_violin(k6_data, target_rps, plot_dir)
    plot_p4_per_cluster_detail(k6_data, target_rps, plot_dir)
    plot_p5_status_endpoint(k6_data, target_rps, plot_dir)
    plot_p6_latency_heatmap(k6_data, target_rps, plot_dir)
    plot_p11_exclude_c3(k6_data, target_rps, plot_dir)

    # DMOS plots (require JSONL)
    if jsonl_on:
        plot_p7_dmos_internals(jsonl_on, target_rps, plot_dir)
        plot_p8_fairness(jsonl_on, target_rps, plot_dir)
        plot_p10_backend_services(jsonl_on, target_rps, plot_dir)

    if jsonl_on or jsonl_off:
        plot_p9_resources(jsonl_on, jsonl_off, target_rps, plot_dir)

    # Report
    print(f"\n[4/4] Generating reports...")
    report = generate_report(k6_data, jsonl_on, jsonl_off, target_rps, plot_dir)
    print_report(report)

    n_plots = len(list(plot_dir.glob('*.png')))
    total_mb = sum(f.stat().st_size for f in plot_dir.glob('*.png')) / 1024 / 1024
    print(f"\n{'=' * 62}")
    print(f"  DONE! {n_plots} plots ({total_mb:.1f} MB) + JSON report")
    print(f"  Output: {plot_dir}")
    print(f"{'=' * 62}")


if __name__ == '__main__':
    main()
