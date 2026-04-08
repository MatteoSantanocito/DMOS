#!/usr/bin/env python3
"""
analyze_dmos_complete.py — Analisi completa DMOS ON vs DMOS OFF
================================================================
Combina dati da:
  1. k6 CSV (latenza, errori, throughput per endpoint/cluster)
  2. DMOS JSONL (repliche, predittore, score, scaling events)
  3. Prometheus/cAdvisor via JSONL (CPU, memoria, network)
  4. Cilium Hubble via JSONL (cross-cluster flows)

Output: grafici PNG + JSON report + console summary

Uso:
  python analyze_dmos_complete.py \\
    --k6-on  risultati_DMOS_ON.csv \\
    --k6-off risultati_DMOS_OFF.csv \\
    --jsonl-on  results/xxxxx_flash_crowd_DMOS_ON.jsonl \\
    --jsonl-off results/xxxxx_flash_crowd_DMOS_OFF.jsonl

  Se non hai i JSONL, puoi usare solo i k6 CSV:
  python analyze_dmos_complete.py \\
    --k6-on  risultati_DMOS_ON.csv \\
    --k6-off risultati_DMOS_OFF.csv
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from datetime import datetime

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300, 'savefig.dpi': 300,
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'legend.fontsize': 9, 'figure.figsize': (16, 8),
})

COLORS = {"DMOS ON": "#2196F3", "DMOS OFF": "#F44336"}
CLUSTER_COLORS = {"cluster1": "#4CAF50", "cluster2": "#FF9800", "cluster3": "#9C27B0"}
CLUSTER_NAMES = {"cluster1": "C1 (Frankfurt/DE)", "cluster2": "C2 (Paris/FR)", "cluster3": "C3 (Warsaw/PL)"}
INGRESS_TO_CLUSTER = {"c1-DE": "cluster1", "c2-FR": "cluster2", "c3-PL": "cluster3"}

SERVICES = ["frontend", "cartservice", "productcatalogservice", "checkoutservice", "recommendationservice"]
SERVICE_SHORT = {
    "frontend": "frontend", "cartservice": "cart", "productcatalogservice": "prodcat",
    "checkoutservice": "checkout", "recommendationservice": "recommend"
}

PHASES = [
    (0, 120, "Warm-up"), (120, 180, "Ramp-up"),
    (180, 660, "Sustained Peak"), (660, 780, "Decline"), (780, 840, "Cooldown"),
]
PHASE_COLORS = ['#E3F2FD', '#FFF9C4', '#FFEBEE', '#FFF9C4', '#E8F5E9']


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_k6_csv(path, label):
    """Load k6 CSV and preprocess."""
    print(f"  Loading k6 {label}: {path}...")
    df = pd.read_csv(path, low_memory=False)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['t_start'] = df['datetime'].min()
    df['elapsed_s'] = (df['datetime'] - df['t_start']).dt.total_seconds()
    df['ingress'] = df['extra_tags'].str.extract(r'ingress=(\S+)', expand=False)
    df['cluster'] = df['ingress'].map(INGRESS_TO_CLUSTER)
    df['expected_response'] = df['expected_response'].astype(str).str.lower() == 'true'
    df['status'] = pd.to_numeric(df['status'], errors='coerce')
    df['test'] = label
    return df


def load_jsonl(path, label):
    """Load DMOS JSONL metrics file."""
    print(f"  Loading JSONL {label}: {path}...")
    snapshots = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))

    if not snapshots:
        return None

    # Parse timestamps and compute elapsed
    for snap in snapshots:
        snap['_dt'] = datetime.fromisoformat(snap['timestamp'])

    t0 = snapshots[0]['_dt']
    for snap in snapshots:
        snap['_elapsed_s'] = (snap['_dt'] - t0).total_seconds()

    return snapshots


# ══════════════════════════════════════════════════════════════════════════════
# K6-ONLY ANALYSIS (always available)
# ══════════════════════════════════════════════════════════════════════════════

def k6_duration(df):
    return df[df['metric_name'] == 'http_req_duration'].copy()

def k6_reqs(df):
    return df[df['metric_name'] == 'http_reqs'].copy()

def k6_vus(df):
    return df[df['metric_name'] == 'vus'].copy()


# ══════════════════════════════════════════════════════════════════════════════
# JSONL ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def extract_timeseries_from_jsonl(snapshots, service="frontend"):
    """Extract time-aligned metrics from JSONL snapshots."""
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
    """Compute Jain's fairness index over time for replica distribution."""
    jain_data = {'elapsed_s': [], 'jain_index': [], 'jain_index_score': []}

    for snap in snapshots:
        svc = snap.get('dmos', {}).get('services', {}).get(service, {})
        replicas = []
        scores = []
        for c in ["cluster1", "cluster2", "cluster3"]:
            cd = svc.get('clusters', {}).get(c, {})
            replicas.append(cd.get('current_replicas', 0))
            scores.append(cd.get('score', 0))

        # Jain index: (sum(x_i))^2 / (n * sum(x_i^2))
        r = np.array(replicas, dtype=float)
        s = np.array(scores, dtype=float)
        n = len(r)

        # Replica fairness
        sum_sq = np.sum(r ** 2)
        jain_r = (np.sum(r) ** 2) / (n * sum_sq) if sum_sq > 0 else 1.0

        # Score fairness
        sum_sq_s = np.sum(s ** 2)
        jain_s = (np.sum(s) ** 2) / (n * sum_sq_s) if sum_sq_s > 0 else 1.0

        jain_data['elapsed_s'].append(snap['_elapsed_s'])
        jain_data['jain_index'].append(jain_r)
        jain_data['jain_index_score'].append(jain_s)

    return pd.DataFrame(jain_data)


def compute_prediction_accuracy(snapshots, service="frontend"):
    """Compute prediction accuracy metrics."""
    actual = []
    predicted = []

    for snap in snapshots:
        svc = snap.get('dmos', {}).get('services', {}).get(service, {})
        a = svc.get('actual_traffic', 0)
        p = svc.get('predicted_traffic_total', 0)
        if a > 1:  # Only when there's meaningful traffic
            actual.append(a)
            predicted.append(p)

    if not actual:
        return None

    actual = np.array(actual)
    predicted = np.array(predicted)

    errors = predicted - actual
    abs_pct_errors = np.abs(errors / actual) * 100

    return {
        'mape': np.mean(abs_pct_errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(np.abs(errors)),
        'r_squared': 1 - np.sum(errors ** 2) / np.sum((actual - np.mean(actual)) ** 2) if np.var(actual) > 0 else 0,
        'directional_accuracy': np.mean(np.sign(np.diff(predicted)) == np.sign(np.diff(actual))) * 100 if len(actual) > 1 else 0,
        'mean_actual': np.mean(actual),
        'mean_predicted': np.mean(predicted),
        'actual': actual,
        'predicted': predicted,
    }


def compute_scaling_stats(snapshots):
    """Compute scaling event statistics from JSONL."""
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


def extract_cross_cluster_flows(snapshots):
    """Extract Cilium cross-cluster flow data over time."""
    data = {'elapsed_s': []}
    for c in ["cluster1", "cluster2", "cluster3"]:
        data[f'{c}_total'] = []
        data[f'{c}_remote'] = []

    for snap in snapshots:
        data['elapsed_s'].append(snap['_elapsed_s'])
        flows = snap.get('cilium_flows', {})
        for c in ["cluster1", "cluster2", "cluster3"]:
            fd = flows.get(c, {})
            data[f'{c}_total'].append(fd.get('total_http_flows'))
            data[f'{c}_remote'].append(fd.get('remote_source_flows'))

    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _shade(ax):
    for (s, e, name), col in zip(PHASES, PHASE_COLORS):
        ax.axvspan(s / 60, e / 60, alpha=0.15, color=col)


# ── PAGE 1: k6 Core Comparison (always available) ───────────────────────────

def plot_page1_k6(k6_data, plot_dir):
    """k6: Latency timeseries, error rate, throughput, CDF."""
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    W = 10  # window seconds

    # 1. p95 Latency over time
    ax = fig.add_subplot(gs[0, 0])
    for label, df in k6_data.items():
        dur = k6_duration(df)
        dur['bin'] = (dur['elapsed_s'] // W) * W
        g = dur.groupby('bin')['metric_value'].quantile(0.95).reset_index()
        ax.plot(g['bin'] / 60, g['metric_value'], label=label, color=COLORS[label], lw=1.5)
    ax.set_ylabel('p95 Latency (ms)'); ax.set_title('p95 Latency Over Time')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 2. p99 Latency over time
    ax = fig.add_subplot(gs[0, 1])
    for label, df in k6_data.items():
        dur = k6_duration(df)
        dur['bin'] = (dur['elapsed_s'] // W) * W
        g = dur.groupby('bin')['metric_value'].quantile(0.99).reset_index()
        ax.plot(g['bin'] / 60, g['metric_value'], label=label, color=COLORS[label], lw=1.5)
    ax.set_ylabel('p99 Latency (ms)'); ax.set_title('p99 Latency Over Time')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 3. Error Rate over time
    ax = fig.add_subplot(gs[1, 0])
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        reqs['bin'] = (reqs['elapsed_s'] // W) * W
        g = reqs.groupby('bin').agg(total=('expected_response', 'count'),
                                     failed=('expected_response', lambda x: (~x).sum())).reset_index()
        g['err_pct'] = g['failed'] / g['total'] * 100
        ax.plot(g['bin'] / 60, g['err_pct'], label=label, color=COLORS[label], lw=1.5)
    ax.set_ylabel('Error Rate (%)'); ax.set_title('Error Rate Over Time')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)
    ax.yaxis.set_major_formatter(PercentFormatter())

    # 4. Successful RPS
    ax = fig.add_subplot(gs[1, 1])
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        ok = reqs[reqs['expected_response']]
        ok['bin'] = (ok['elapsed_s'] // W) * W
        g = ok.groupby('bin').size().reset_index(name='n')
        g['rps'] = g['n'] / W
        ax.plot(g['bin'] / 60, g['rps'], label=label, color=COLORS[label], lw=1.5)
    ax.axhline(75, color='gray', ls='--', alpha=0.4, label='Target 75 rps')
    ax.set_ylabel('Successful RPS'); ax.set_title('Successful Throughput')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 5. CDF
    ax = fig.add_subplot(gs[2, 0])
    for label, df in k6_data.items():
        dur = k6_duration(df)
        lat = np.sort(dur['metric_value'].values)
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat, cdf, label=label, color=COLORS[label], lw=1.5)
    ax.set_xlim(0, 2000); ax.set_xlabel('Latency (ms)'); ax.set_ylabel('CDF')
    ax.axhline(0.95, color='gray', ls='--', alpha=0.4)
    ax.axhline(0.99, color='gray', ls=':', alpha=0.4)
    ax.set_title('Latency CDF'); ax.legend(); ax.grid(True, alpha=0.3)

    # 6. Per-cluster error rate bar chart
    ax = fig.add_subplot(gs[2, 1])
    clusters = list(INGRESS_TO_CLUSTER.keys())
    x = np.arange(len(clusters))
    w = 0.35
    for i, (label, df) in enumerate(k6_data.items()):
        reqs = k6_reqs(df)
        rates = []
        for cl in clusters:
            cr = reqs[reqs['ingress'] == cl]
            total = len(cr)
            fail = cr[~cr['expected_response']].shape[0]
            rates.append(fail / total * 100 if total > 0 else 0)
        bars = ax.bar(x - w/2 + i*w, rates, w, label=label, color=COLORS[label], alpha=0.8)
        for b, v in zip(bars, rates):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([CLUSTER_NAMES[INGRESS_TO_CLUSTER[c]] for c in clusters], fontsize=9)
    ax.set_ylabel('Error Rate (%)'); ax.set_title('Error Rate per Cluster')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Page 1: k6 Load Test Comparison — DMOS ON vs OFF', fontsize=16, y=1.01)
    fig.savefig(plot_dir / 'page1_k6_comparison.png')
    print(f"  Saved: page1_k6_comparison.png")
    plt.close()


# ── PAGE 2: k6 Per-Endpoint + Status + Phases ───────────────────────────────

def plot_page2_k6_detail(k6_data, plot_dir):
    """k6 details: status codes, per-endpoint, per-phase, traffic distribution."""
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.35)

    def url_to_path(url):
        if pd.isna(url): return 'unknown'
        parts = str(url).split(':30080')
        return parts[1] if len(parts) > 1 else str(url)[-20:]

    # 1. Status code distribution (side by side)
    for idx, (label, df) in enumerate(k6_data.items()):
        ax = fig.add_subplot(gs[0, idx])
        reqs = k6_reqs(df)
        sc = reqs['status'].value_counts().sort_index()
        colors_map = ['#4CAF50' if s < 300 else '#2196F3' if s < 400 else '#FF9800' if s < 500 else '#F44336' for s in sc.index]
        bars = ax.bar([str(int(s)) for s in sc.index], sc.values, color=colors_map, alpha=0.8)
        for b, v in zip(bars, sc.values):
            pct = v / len(reqs) * 100
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'{label} — Status Codes'); ax.grid(True, alpha=0.3, axis='y')

    # 2. Per-endpoint error rate
    ax = fig.add_subplot(gs[1, :])
    all_paths = set()
    for label, df in k6_data.items():
        reqs = k6_reqs(df)
        reqs['path'] = reqs['url'].apply(url_to_path)
        all_paths.update(reqs['path'].unique())
    paths = sorted([p for p in all_paths if p != 'unknown'])
    x = np.arange(len(paths))
    w = 0.35
    for i, (label, df) in enumerate(k6_data.items()):
        reqs = k6_reqs(df)
        reqs['path'] = reqs['url'].apply(url_to_path)
        rates = []
        for p in paths:
            pr = reqs[reqs['path'] == p]
            total = len(pr)
            fail = pr[~pr['expected_response']].shape[0]
            rates.append(fail / total * 100 if total > 0 else 0)
        ax.barh(x - w/2 + i*w, rates, w, label=label, color=COLORS[label], alpha=0.8)
    ax.set_yticks(x); ax.set_yticklabels(paths, fontsize=8)
    ax.set_xlabel('Error Rate (%)'); ax.set_title('Error Rate per Endpoint')
    ax.legend(); ax.grid(True, alpha=0.3, axis='x')

    # 3. Per-phase summary
    ax = fig.add_subplot(gs[2, 0])
    pnames = [p[2] for p in PHASES]
    x = np.arange(len(pnames))
    w = 0.35
    for i, (label, df) in enumerate(k6_data.items()):
        reqs = k6_reqs(df)
        rates = []
        for s, e, _ in PHASES:
            pr = reqs[(reqs['elapsed_s'] >= s) & (reqs['elapsed_s'] < e)]
            total = len(pr)
            fail = pr[~pr['expected_response']].shape[0]
            rates.append(fail / total * 100 if total > 0 else 0)
        bars = ax.bar(x - w/2 + i*w, rates, w, label=label, color=COLORS[label], alpha=0.8)
        for b, v in zip(bars, rates):
            if v > 0.5:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(pnames, fontsize=9)
    ax.set_ylabel('Error Rate (%)'); ax.set_title('Error Rate per Phase')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # 4. Heatmap error cluster × phase
    ax = fig.add_subplot(gs[2, 1])
    # Only for DMOS ON
    label = "DMOS ON"
    if label in k6_data:
        df = k6_data[label]
        reqs = k6_reqs(df)
        clusters = list(INGRESS_TO_CLUSTER.keys())
        matrix = np.zeros((len(clusters), len(PHASES)))
        for j, (s, e, _) in enumerate(PHASES):
            for i, cl in enumerate(clusters):
                cr = reqs[(reqs['ingress'] == cl) & (reqs['elapsed_s'] >= s) & (reqs['elapsed_s'] < e)]
                total = len(cr)
                fail = cr[~cr['expected_response']].shape[0]
                matrix[i, j] = fail / total * 100 if total > 0 else 0
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
        ax.set_xticks(range(len(pnames))); ax.set_xticklabels(pnames, fontsize=8, rotation=15, ha='right')
        ax.set_yticks(range(len(clusters)))
        ax.set_yticklabels([CLUSTER_NAMES[INGRESS_TO_CLUSTER[c]] for c in clusters], fontsize=9)
        for i in range(len(clusters)):
            for j in range(len(PHASES)):
                ax.text(j, i, f'{matrix[i,j]:.1f}%', ha='center', va='center', fontsize=8,
                        color='white' if matrix[i,j] > 25 else 'black')
        ax.set_title('DMOS ON — Error Heatmap (Cluster x Phase)')
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle('Page 2: k6 Detailed Analysis', fontsize=16, y=1.01)
    fig.savefig(plot_dir / 'page2_k6_detail.png')
    print(f"  Saved: page2_k6_detail.png")
    plt.close()


# ── PAGE 3: DMOS Internals (requires JSONL) ─────────────────────────────────

def plot_page3_dmos(jsonl_on, plot_dir):
    """DMOS internals: replicas, traffic, predictor, scores."""
    ts = extract_timeseries_from_jsonl(jsonl_on, "frontend")
    if ts.empty:
        print("  [skip] Page 3: no JSONL data")
        return

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # 1. Actual vs Predicted Traffic
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ts['elapsed_s'] / 60, ts['actual_traffic'], label='Actual', color='#2196F3', lw=2)
    ax.plot(ts['elapsed_s'] / 60, ts['predicted_traffic'], label='Predicted', color='#FF9800', lw=2, ls='--')
    ax.fill_between(ts['elapsed_s'] / 60, ts['actual_traffic'], ts['predicted_traffic'], alpha=0.15, color='#FF9800')
    ax.set_ylabel('Traffic (req/s)'); ax.set_title('Frontend: Actual vs Predicted Traffic')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 2. Total Replicas over time
    ax = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(ts))
    for c in ["cluster1", "cluster2", "cluster3"]:
        vals = ts[f'{c}_replicas'].values
        ax.bar(ts['elapsed_s'] / 60, vals, bottom=bottom, width=0.3,
               label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], alpha=0.8)
        bottom += vals
    ax.set_ylabel('Replicas'); ax.set_title('Frontend Replicas per Cluster (Stacked)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y'); _shade(ax)

    # 3. Per-cluster replicas (line)
    ax = fig.add_subplot(gs[1, 0])
    for c in ["cluster1", "cluster2", "cluster3"]:
        ax.plot(ts['elapsed_s'] / 60, ts[f'{c}_replicas'],
                label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=2, marker='o', markersize=3)
    ax.set_ylabel('Replicas'); ax.set_title('Frontend Replicas per Cluster')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)
    ax.set_ylim(bottom=0)

    # 4. Cluster Scores over time
    ax = fig.add_subplot(gs[1, 1])
    for c in ["cluster1", "cluster2", "cluster3"]:
        scores = ts[f'{c}_score'].values
        ax.plot(ts['elapsed_s'] / 60, scores,
                label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=2)
    ax.set_ylabel('Multi-Objective Score'); ax.set_title('DMOS Cluster Scores (Frontend)')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 5. Per-cluster predicted traffic
    ax = fig.add_subplot(gs[2, 0])
    for c in ["cluster1", "cluster2", "cluster3"]:
        ax.plot(ts['elapsed_s'] / 60, ts[f'{c}_predicted'],
                label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.5)
    ax.set_ylabel('Predicted Traffic (req/s)'); ax.set_title('Per-Cluster Predicted Traffic')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)
    ax.set_xlabel('Time (minutes)')

    # 6. Prediction scatter (actual vs predicted)
    ax = fig.add_subplot(gs[2, 1])
    mask = ts['actual_traffic'] > 1
    if mask.any():
        ax.scatter(ts.loc[mask, 'actual_traffic'], ts.loc[mask, 'predicted_traffic'],
                   alpha=0.6, s=20, color='#2196F3')
        max_val = max(ts.loc[mask, 'actual_traffic'].max(), ts.loc[mask, 'predicted_traffic'].max())
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Perfect prediction')
        ax.set_xlabel('Actual Traffic (req/s)'); ax.set_ylabel('Predicted Traffic (req/s)')
    ax.set_title('Prediction Accuracy Scatter'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('Page 3: DMOS Internals — Frontend Service', fontsize=16, y=1.01)
    fig.savefig(plot_dir / 'page3_dmos_internals.png')
    print(f"  Saved: page3_dmos_internals.png")
    plt.close()


# ── PAGE 4: Jain Index + Fairness ───────────────────────────────────────────

def plot_page4_fairness(jsonl_on, plot_dir):
    """Jain index, scaling events, fairness analysis."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # 1. Jain Index over time (frontend)
    ax = fig.add_subplot(gs[0, 0])
    jain = compute_jain_index(jsonl_on, "frontend")
    ax.plot(jain['elapsed_s'] / 60, jain['jain_index'], color='#4CAF50', lw=2, label='Replica Fairness')
    ax.plot(jain['elapsed_s'] / 60, jain['jain_index_score'], color='#FF9800', lw=2, ls='--', label='Score Fairness')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4, label='Perfect (1.0)')
    ax.axhline(1/3, color='red', ls=':', alpha=0.4, label='Worst (1/N)')
    ax.set_ylabel("Jain's Fairness Index"); ax.set_title("Jain's Index Over Time (Frontend)")
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 2. Jain Index per service (box plot of all snapshots)
    ax = fig.add_subplot(gs[0, 1])
    jain_per_svc = {}
    for svc in SERVICES:
        j = compute_jain_index(jsonl_on, svc)
        jain_per_svc[SERVICE_SHORT[svc]] = j['jain_index'].values

    bp = ax.boxplot(jain_per_svc.values(), labels=jain_per_svc.keys(), patch_artist=True)
    colors_box = ['#4CAF50', '#FF9800', '#9C27B0', '#2196F3', '#F44336']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(1.0, color='gray', ls=':', alpha=0.4)
    ax.set_ylabel("Jain's Index"); ax.set_title("Jain's Index Distribution per Service")
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Scaling events per service per cluster
    ax = fig.add_subplot(gs[1, 0])
    stats = compute_scaling_stats(jsonl_on)
    services_plot = [s for s in SERVICES if stats[s]['total_scale_up'] + stats[s]['total_scale_down'] > 0]
    if not services_plot:
        services_plot = SERVICES

    x = np.arange(len(services_plot))
    w = 0.25
    for i, c in enumerate(["cluster1", "cluster2", "cluster3"]):
        su = [stats[s]['per_cluster'].get(c, {}).get('scale_up', 0) for s in services_plot]
        sd = [-stats[s]['per_cluster'].get(c, {}).get('scale_down', 0) for s in services_plot]
        ax.bar(x + i * w, su, w, label=f'{CLUSTER_NAMES[c]} up', color=CLUSTER_COLORS[c], alpha=0.8)
        ax.bar(x + i * w, sd, w, color=CLUSTER_COLORS[c], alpha=0.4)

    ax.set_xticks(x + w); ax.set_xticklabels([SERVICE_SHORT[s] for s in services_plot], fontsize=9)
    ax.set_ylabel('Events (up +, down -)'); ax.set_title('Scaling Events per Cluster')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', lw=0.5)

    # 4. All services total replicas over time
    ax = fig.add_subplot(gs[1, 1])
    for svc in SERVICES:
        ts = extract_timeseries_from_jsonl(jsonl_on, svc)
        if not ts.empty:
            ax.plot(ts['elapsed_s'] / 60, ts['total_replicas'],
                    label=SERVICE_SHORT[svc], lw=1.5)
    ax.set_xlabel('Time (minutes)'); ax.set_ylabel('Total Replicas')
    ax.set_title('Total Replicas per Service Over Time')
    ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    fig.suptitle('Page 4: Fairness & Scaling Analysis', fontsize=16, y=1.01)
    fig.savefig(plot_dir / 'page4_fairness_scaling.png')
    print(f"  Saved: page4_fairness_scaling.png")
    plt.close()


# ── PAGE 5: Resource Utilization (CPU, Memory) ──────────────────────────────

def plot_page5_resources(jsonl_on, jsonl_off, plot_dir):
    """CPU and memory utilization per cluster, ON vs OFF."""
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    data_sets = {}
    if jsonl_on:
        data_sets["DMOS ON"] = jsonl_on
    if jsonl_off:
        data_sets["DMOS OFF"] = jsonl_off

    # 1-2. Frontend CPU per cluster over time (ON and OFF)
    for idx, (label, snaps) in enumerate(data_sets.items()):
        ax = fig.add_subplot(gs[0, idx])
        ts = extract_timeseries_from_jsonl(snaps, "frontend")
        for c in ["cluster1", "cluster2", "cluster3"]:
            vals = ts[f'{c}_cpu'].values
            # Replace None with NaN for plotting
            vals = pd.to_numeric(pd.Series(vals), errors='coerce')
            ax.plot(ts['elapsed_s'] / 60, vals * 1000,  # cores → millicores
                    label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.5)
        ax.set_ylabel('CPU (millicores)'); ax.set_title(f'{label} — Frontend CPU per Cluster')
        ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 3-4. Total CPU (all services) per cluster
    for idx, (label, snaps) in enumerate(data_sets.items()):
        ax = fig.add_subplot(gs[1, idx])
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
            ax.plot(elapsed, total_cpu, label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.5)
        ax.axhline(4000, color='red', ls='--', alpha=0.4, label='Node limit (4000m)')
        ax.set_ylabel('CPU (millicores)'); ax.set_title(f'{label} — Total CPU per Cluster')
        ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    # 5. Memory comparison (bar chart, peak values)
    ax = fig.add_subplot(gs[2, 0])
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
    ax.set_xticks(x); ax.set_xticklabels([CLUSTER_NAMES[c] for c in clusters], fontsize=9)
    ax.set_ylabel('Peak Memory (MB)'); ax.set_title('Peak Memory Usage per Cluster')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # 6. Traffic distribution over time
    ax = fig.add_subplot(gs[2, 1])
    if jsonl_on:
        ts = extract_timeseries_from_jsonl(jsonl_on, "frontend")
        for c in ["cluster1", "cluster2", "cluster3"]:
            vals = pd.to_numeric(pd.Series(ts[f'{c}_traffic_pct'].values), errors='coerce')
            ax.plot(ts['elapsed_s'] / 60, vals,
                    label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], lw=1.5)
        ax.set_xlabel('Time (minutes)'); ax.set_ylabel('Traffic Share (%)')
        ax.set_title('DMOS ON — Traffic Distribution Over Time')
        ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)

    fig.suptitle('Page 5: Resource Utilization', fontsize=16, y=1.01)
    fig.savefig(plot_dir / 'page5_resources.png')
    print(f"  Saved: page5_resources.png")
    plt.close()


# ── PAGE 6: Cross-Cluster + Prediction Accuracy ─────────────────────────────

def plot_page6_advanced(jsonl_on, k6_on, plot_dir):
    """Cross-cluster flows, prediction accuracy, backend services."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # 1. Prediction accuracy per service
    ax = fig.add_subplot(gs[0, 0])
    if jsonl_on:
        mapes = []
        svc_labels = []
        for svc in SERVICES:
            acc = compute_prediction_accuracy(jsonl_on, svc)
            if acc:
                mapes.append(acc['mape'])
                svc_labels.append(SERVICE_SHORT[svc])

        if mapes:
            colors_bar = ['#4CAF50', '#FF9800', '#9C27B0', '#2196F3', '#F44336'][:len(mapes)]
            bars = ax.bar(svc_labels, mapes, color=colors_bar, alpha=0.8)
            for b, v in zip(bars, mapes):
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
            ax.axhline(20, color='green', ls='--', alpha=0.4, label='Good (<20%)')
            ax.axhline(50, color='red', ls='--', alpha=0.4, label='Poor (>50%)')

    ax.set_ylabel('MAPE (%)'); ax.set_title('Prediction Accuracy (MAPE) per Service')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # 2. Prediction accuracy detailed (R², directional)
    ax = fig.add_subplot(gs[0, 1])
    if jsonl_on:
        table_data = []
        for svc in SERVICES:
            acc = compute_prediction_accuracy(jsonl_on, svc)
            if acc:
                table_data.append([
                    SERVICE_SHORT[svc],
                    f"{acc['mape']:.1f}%",
                    f"{acc['rmse']:.2f}",
                    f"{acc['r_squared']:.3f}",
                    f"{acc['directional_accuracy']:.0f}%",
                    f"{acc['mean_actual']:.1f}",
                    f"{acc['mean_predicted']:.1f}",
                ])

        if table_data:
            ax.axis('off')
            table = ax.table(
                cellText=table_data,
                colLabels=['Service', 'MAPE', 'RMSE', 'R²', 'Dir.Acc', 'Avg Act.', 'Avg Pred.'],
                loc='center', cellLoc='center'
            )
            table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 1.5)
            for j in range(7):
                table[0, j].set_facecolor('#E0E0E0')
                table[0, j].set_text_props(weight='bold')
            ax.set_title('Prediction Accuracy Details', pad=20)

    # 3. Cross-cluster flows (if available)
    ax = fig.add_subplot(gs[1, 0])
    if jsonl_on:
        flows_df = extract_cross_cluster_flows(jsonl_on)
        has_data = False
        for c in ["cluster1", "cluster2", "cluster3"]:
            total = pd.to_numeric(flows_df[f'{c}_total'], errors='coerce')
            remote = pd.to_numeric(flows_df[f'{c}_remote'], errors='coerce')
            if total.notna().any():
                has_data = True
                # Compute delta (rate of new flows)
                total_rate = total.diff() / 15  # per second (15s interval)
                ax.plot(flows_df['elapsed_s'] / 60, total_rate,
                        label=f'{CLUSTER_NAMES[c]}', color=CLUSTER_COLORS[c], lw=1.5)
        if has_data:
            ax.set_ylabel('HTTP Flows/s'); ax.set_title('Cilium HTTP Flow Rate per Cluster')
            ax.legend(); ax.grid(True, alpha=0.3); _shade(ax)
        else:
            ax.text(0.5, 0.5, 'No Cilium Hubble data available\n(hubble_http_requests_total)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, alpha=0.5)
            ax.set_title('Cilium Cross-Cluster Flows')

    # 4. Backend services replica comparison
    ax = fig.add_subplot(gs[1, 1])
    if jsonl_on:
        backend_svcs = [s for s in SERVICES if s != "frontend"]
        x = np.arange(len(backend_svcs))
        w = 0.2
        for i, c in enumerate(["cluster1", "cluster2", "cluster3"]):
            peak_replicas = []
            for svc in backend_svcs:
                max_r = 0
                for snap in jsonl_on:
                    r = snap.get('dmos', {}).get('services', {}).get(svc, {}).get('clusters', {}).get(c, {}).get('current_replicas', 0)
                    max_r = max(max_r, r)
                peak_replicas.append(max_r)
            ax.bar(x + i * w, peak_replicas, w, label=CLUSTER_NAMES[c], color=CLUSTER_COLORS[c], alpha=0.8)
        ax.set_xticks(x + w); ax.set_xticklabels([SERVICE_SHORT[s] for s in backend_svcs], fontsize=9)
        ax.set_ylabel('Peak Replicas'); ax.set_title('Peak Backend Replicas per Cluster')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Page 6: Advanced Analysis — Prediction & Cross-Cluster', fontsize=16, y=1.01)
    fig.savefig(plot_dir / 'page6_advanced.png')
    print(f"  Saved: page6_advanced.png")
    plt.close()


# ── SUMMARY TABLE IMAGE ─────────────────────────────────────────────────────

def plot_summary_image(k6_data, jsonl_on, plot_dir):
    """Generate summary table image with all key metrics."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')

    rows = []

    # k6 metrics
    for label, df in k6_data.items():
        dur = k6_duration(df)
        reqs = k6_reqs(df)
        total = len(reqs)
        failed = reqs[~reqs['expected_response']].shape[0]
        lat = dur['metric_value']

        row = [label, str(total), str(total - failed), str(failed),
               f"{failed/total*100:.2f}%",
               f"{lat.mean():.0f}", f"{lat.quantile(0.50):.0f}",
               f"{lat.quantile(0.95):.0f}", f"{lat.quantile(0.99):.0f}"]
        rows.append(row)

    # Add delta row
    if len(rows) == 2:
        delta = []
        delta.append("DELTA")
        for i in range(1, 4):
            d = int(rows[0][i]) - int(rows[1][i])
            delta.append(f"{d:+d}")
        # Error rate delta
        on_err = float(rows[0][4].rstrip('%'))
        off_err = float(rows[1][4].rstrip('%'))
        delta.append(f"{on_err - off_err:+.2f}pp")
        # Latency deltas
        for i in range(5, 9):
            d = int(rows[0][i]) - int(rows[1][i])
            delta.append(f"{d:+d}ms")
        rows.append(delta)

    cols = ['Test', 'Total Req', 'Success', 'Failed', 'Error %',
            'Avg(ms)', 'p50(ms)', 'p95(ms)', 'p99(ms)']

    table = ax.table(cellText=rows, colLabels=cols, loc='upper center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.8)

    # Style header
    for j in range(len(cols)):
        table[0, j].set_facecolor('#37474F')
        table[0, j].set_text_props(color='white', weight='bold')

    # Style rows
    table[1, 0].set_facecolor('#E3F2FD')  # DMOS ON
    table[2, 0].set_facecolor('#FFEBEE')  # DMOS OFF
    if len(rows) > 2:
        for j in range(len(cols)):
            table[3, j].set_facecolor('#FFF9C4')  # Delta

    # Add DMOS-specific metrics if available
    if jsonl_on:
        dmos_rows = []

        # Jain index
        jain = compute_jain_index(jsonl_on, "frontend")
        dmos_rows.append(['Jain Index (mean)', f"{jain['jain_index'].mean():.3f}",
                          '', '', '', '', '', '', ''])
        dmos_rows.append(['Jain Index (min)', f"{jain['jain_index'].min():.3f}",
                          '', '', '', '', '', '', ''])

        # Prediction accuracy
        acc = compute_prediction_accuracy(jsonl_on, "frontend")
        if acc:
            dmos_rows.append(['Prediction MAPE', f"{acc['mape']:.1f}%", '', '', '', '', '', '', ''])
            dmos_rows.append(['Prediction R²', f"{acc['r_squared']:.3f}", '', '', '', '', '', '', ''])

        # Scaling events
        stats = compute_scaling_stats(jsonl_on)
        total_su = sum(s['total_scale_up'] for s in stats.values())
        total_sd = sum(s['total_scale_down'] for s in stats.values())
        dmos_rows.append(['Total Scale-up Events', str(total_su), '', '', '', '', '', '', ''])
        dmos_rows.append(['Total Scale-down Events', str(total_sd), '', '', '', '', '', '', ''])

        # Add DMOS table below
        y_offset = 0.45
        ax.text(0.5, y_offset + 0.05, 'DMOS Internal Metrics', ha='center',
                fontsize=13, fontweight='bold', transform=ax.transAxes)

        table2 = ax.table(cellText=dmos_rows,
                          colLabels=['Metric', 'Value', '', '', '', '', '', '', ''],
                          loc='lower center', cellLoc='center')
        table2.auto_set_font_size(False); table2.set_fontsize(10); table2.scale(1.2, 1.5)
        for j in range(9):
            table2[0, j].set_facecolor('#37474F')
            table2[0, j].set_text_props(color='white', weight='bold')

    ax.set_title('DMOS Evaluation Summary — DMOS ON vs DMOS OFF', fontsize=16, pad=30)
    fig.savefig(plot_dir / 'page0_summary.png')
    print(f"  Saved: page0_summary.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# JSON REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_json_report(k6_data, jsonl_on, jsonl_off, plot_dir):
    """Generate comprehensive JSON report."""
    report = {"generated_at": datetime.now().isoformat(), "tests": {}}

    for label, df in k6_data.items():
        dur = k6_duration(df)
        reqs = k6_reqs(df)
        total = len(reqs)
        failed = reqs[~reqs['expected_response']].shape[0]
        lat = dur['metric_value']

        test_report = {
            "total_requests": total,
            "successful_requests": total - failed,
            "failed_requests": failed,
            "error_rate_pct": round(failed / total * 100, 2) if total > 0 else 0,
            "latency": {
                "avg_ms": round(lat.mean(), 1),
                "p50_ms": round(lat.quantile(0.50), 1),
                "p90_ms": round(lat.quantile(0.90), 1),
                "p95_ms": round(lat.quantile(0.95), 1),
                "p99_ms": round(lat.quantile(0.99), 1),
                "max_ms": round(lat.max(), 1),
            },
            "per_cluster": {},
            "per_phase": {},
        }

        # Per cluster
        for ing, cl in INGRESS_TO_CLUSTER.items():
            cr = reqs[reqs['ingress'] == ing]
            cl_dur = dur[dur['ingress'] == ing]['metric_value']
            t = len(cr)
            f = cr[~cr['expected_response']].shape[0]
            test_report["per_cluster"][cl] = {
                "requests": t,
                "error_rate_pct": round(f / t * 100, 2) if t > 0 else 0,
                "p95_ms": round(cl_dur.quantile(0.95), 1) if len(cl_dur) > 0 else None,
                "p99_ms": round(cl_dur.quantile(0.99), 1) if len(cl_dur) > 0 else None,
            }

        # Per phase
        for start, end, name in PHASES:
            pr = reqs[(reqs['elapsed_s'] >= start) & (reqs['elapsed_s'] < end)]
            pd_dur = dur[(dur['elapsed_s'] >= start) & (dur['elapsed_s'] < end)]
            t = len(pr)
            f = pr[~pr['expected_response']].shape[0]
            test_report["per_phase"][name] = {
                "requests": t,
                "error_rate_pct": round(f / t * 100, 2) if t > 0 else 0,
                "p95_ms": round(pd_dur['metric_value'].quantile(0.95), 1) if len(pd_dur) > 0 else None,
            }

        report["tests"][label] = test_report

    # DMOS metrics
    if jsonl_on:
        dmos_report = {"services": {}}
        for svc in SERVICES:
            acc = compute_prediction_accuracy(jsonl_on, svc)
            jain = compute_jain_index(jsonl_on, svc)
            svc_report = {
                "jain_index_mean": round(jain['jain_index'].mean(), 4),
                "jain_index_min": round(jain['jain_index'].min(), 4),
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

    # Save
    report_path = plot_dir / 'analysis_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved: analysis_report.json")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_console_report(report):
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 70)

    for label, data in report.get("tests", {}).items():
        print(f"\n  {'─' * 50}")
        print(f"  {label}")
        print(f"  {'─' * 50}")
        print(f"    Total requests.......... {data['total_requests']}")
        print(f"    Successful.............. {data['successful_requests']}")
        print(f"    Failed.................. {data['failed_requests']}")
        print(f"    Error rate.............. {data['error_rate_pct']}%")
        lat = data['latency']
        print(f"    Avg latency............. {lat['avg_ms']}ms")
        print(f"    p50..................... {lat['p50_ms']}ms")
        print(f"    p95..................... {lat['p95_ms']}ms")
        print(f"    p99..................... {lat['p99_ms']}ms")

        print(f"\n    Per Cluster:")
        for cl, cd in data.get('per_cluster', {}).items():
            print(f"      {CLUSTER_NAMES.get(cl, cl):.<30s} err={cd['error_rate_pct']}%  p95={cd['p95_ms']}ms")

        print(f"\n    Per Phase:")
        for phase, pd_data in data.get('per_phase', {}).items():
            print(f"      {phase:.<30s} err={pd_data['error_rate_pct']}%  p95={pd_data['p95_ms']}ms  reqs={pd_data['requests']}")

    # Delta
    tests = list(report.get("tests", {}).values())
    if len(tests) == 2:
        on, off = tests[0], tests[1]
        print(f"\n  {'═' * 50}")
        print(f"  DELTA (DMOS ON - DMOS OFF)")
        print(f"  {'═' * 50}")
        err_delta = on['error_rate_pct'] - off['error_rate_pct']
        err_reduction = ((off['error_rate_pct'] - on['error_rate_pct']) / off['error_rate_pct'] * 100) if off['error_rate_pct'] > 0 else 0
        print(f"    Error rate delta........ {err_delta:+.2f} percentage points")
        print(f"    Error reduction......... {err_reduction:.1f}%")
        print(f"    Extra successful reqs... {on['successful_requests'] - off['successful_requests']:+d}")
        p95_delta = on['latency']['p95_ms'] - off['latency']['p95_ms']
        p99_delta = on['latency']['p99_ms'] - off['latency']['p99_ms']
        print(f"    p95 delta............... {p95_delta:+.0f}ms")
        print(f"    p99 delta............... {p99_delta:+.0f}ms")

    # DMOS metrics
    dmos = report.get("dmos", {})
    if dmos:
        print(f"\n  {'═' * 50}")
        print(f"  DMOS INTERNAL METRICS")
        print(f"  {'═' * 50}")
        for svc, sd in dmos.get("services", {}).items():
            print(f"\n    {svc}:")
            print(f"      Jain Index (mean)...... {sd['jain_index_mean']}")
            print(f"      Jain Index (min)....... {sd['jain_index_min']}")
            pred = sd.get("prediction", {})
            if pred:
                print(f"      Prediction MAPE........ {pred['mape']}%")
                print(f"      Prediction R².......... {pred['r_squared']}")
                print(f"      Directional Accuracy... {pred['directional_accuracy_pct']}%")

        print(f"\n    Scaling Events:")
        for svc, se in dmos.get("scaling_events", {}).items():
            print(f"      {svc:.<25s} up={se['total_scale_up']}  down={se['total_scale_down']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DMOS Complete Analysis")
    parser.add_argument("--k6-on", required=True, help="k6 CSV for DMOS ON test")
    parser.add_argument("--k6-off", required=True, help="k6 CSV for DMOS OFF test")
    parser.add_argument("--jsonl-on", help="DMOS JSONL metrics for DMOS ON test")
    parser.add_argument("--jsonl-off", help="DMOS JSONL metrics for DMOS OFF test")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    args = parser.parse_args()

    # Output directory
    if args.output_dir:
        plot_dir = Path(args.output_dir)
    else:
        plot_dir = Path(args.k6_on).parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DMOS Complete Analysis")
    print("=" * 60)

    # Load k6 data
    print("\n[1/4] Loading k6 CSV data...")
    k6_data = {
        "DMOS ON": load_k6_csv(Path(args.k6_on), "DMOS ON"),
        "DMOS OFF": load_k6_csv(Path(args.k6_off), "DMOS OFF"),
    }

    # Load JSONL data (optional)
    jsonl_on = None
    jsonl_off = None
    if args.jsonl_on:
        print("\n[2/4] Loading JSONL metrics...")
        jsonl_on = load_jsonl(Path(args.jsonl_on), "DMOS ON")
    if args.jsonl_off:
        jsonl_off = load_jsonl(Path(args.jsonl_off), "DMOS OFF")

    # Generate plots
    print(f"\n[3/4] Generating plots in {plot_dir}/...")

    # Always available (k6 only)
    plot_page1_k6(k6_data, plot_dir)
    plot_page2_k6_detail(k6_data, plot_dir)
    plot_summary_image(k6_data, jsonl_on, plot_dir)

    # DMOS-specific (require JSONL)
    if jsonl_on:
        plot_page3_dmos(jsonl_on, plot_dir)
        plot_page4_fairness(jsonl_on, plot_dir)
        plot_page6_advanced(jsonl_on, k6_data.get("DMOS ON"), plot_dir)

    if jsonl_on or jsonl_off:
        plot_page5_resources(jsonl_on, jsonl_off, plot_dir)

    # JSON report + console
    print(f"\n[4/4] Generating reports...")
    report = generate_json_report(k6_data, jsonl_on, jsonl_off, plot_dir)
    print_console_report(report)

    n_plots = len(list(plot_dir.glob('*.png')))
    print(f"\n{'=' * 60}")
    print(f"  DONE! {n_plots} plots + 1 JSON report saved to:")
    print(f"  {plot_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
