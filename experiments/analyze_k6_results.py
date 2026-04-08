#!/usr/bin/env python3
"""
analyze_k6_results.py — Analisi comparativa DMOS ON vs DMOS OFF
================================================================
Genera grafici e tabelle da CSV k6 per la tesi.

Uso:
  python analyze_k6_results.py

Output:
  - Grafici PNG nella cartella experiments/plots/
  - Tabella summary stampata a console
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from datetime import datetime, timedelta

# ── Configurazione ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# File CSV da confrontare
FILES = {
    "DMOS ON":  BASE_DIR / "risultati_k6_on_global_DMOS_ON_SERVICES.csv",
    "DMOS OFF": BASE_DIR / "risultati_k6_on_global_DMOS_OFF800.csv",
}

# Mapping ingress tag → cluster name
INGRESS_MAP = {
    "c1-DE": "Cluster 1 (Frankfurt/DE)",
    "c2-FR": "Cluster 2 (Paris/FR)",
    "c3-PL": "Cluster 3 (Warsaw/PL)",
}

# Fasi del test (offset in secondi dall'inizio)
PHASES = [
    (0,   120, "Warm-up\n(10 rps)"),
    (120, 180, "Ramp-up\n(10→75 rps)"),
    (180, 660, "Sustained Peak\n(75 rps)"),
    (660, 780, "Decline\n(75→10 rps)"),
    (780, 840, "Cooldown\n(10 rps)"),
]

# Stile grafici
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

COLORS = {"DMOS ON": "#2196F3", "DMOS OFF": "#F44336"}
CLUSTER_COLORS = {"c1-DE": "#4CAF50", "c2-FR": "#FF9800", "c3-PL": "#9C27B0"}


# ── Caricamento e preprocessing ──────────────────────────────────────────────

def load_k6_csv(filepath, label):
    """Carica CSV k6 e preprocessa."""
    print(f"  Caricamento {label}: {filepath.name}...")
    df = pd.read_csv(filepath, low_memory=False)

    # Timestamp → datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['t_start'] = df['datetime'].min()
    df['elapsed_s'] = (df['datetime'] - df['t_start']).dt.total_seconds()

    # Estrai ingress dal campo extra_tags
    df['ingress'] = df['extra_tags'].str.extract(r'ingress=(\S+)', expand=False)

    # Normalizza expected_response
    df['expected_response'] = df['expected_response'].astype(str).str.lower() == 'true'

    # Converte status a numerico
    df['status'] = pd.to_numeric(df['status'], errors='coerce')

    # Label del test
    df['test'] = label

    return df


def extract_http_duration(df):
    """Estrae solo le righe http_req_duration (la metrica principale di latenza)."""
    return df[df['metric_name'] == 'http_req_duration'].copy()


def extract_http_reqs(df):
    """Estrae solo le righe http_reqs (1 per ogni richiesta HTTP)."""
    return df[df['metric_name'] == 'http_reqs'].copy()


def extract_vus(df):
    """Estrae le righe vus (numero di virtual users attivi)."""
    return df[df['metric_name'] == 'vus'].copy()


# ── 1. Summary Table ────────────────────────────────────────────────────────

def print_summary_table(datasets):
    """Stampa tabella riassuntiva comparativa."""
    print("\n" + "=" * 80)
    print("  SUMMARY COMPARATIVO: DMOS ON vs DMOS OFF")
    print("=" * 80)

    rows = []
    for label, df in datasets.items():
        dur = extract_http_duration(df)
        reqs = extract_http_reqs(df)

        total = len(reqs)
        failed = reqs[~reqs['expected_response']].shape[0]
        success = total - failed
        error_rate = (failed / total * 100) if total > 0 else 0

        # Latenza globale (ms)
        lat = dur['metric_value']
        lat_ok = dur[dur['expected_response']]['metric_value']

        # Status code distribution
        status_dist = reqs['status'].value_counts().sort_index()

        # Solo fase sustained peak (180-660s)
        dur_peak = dur[(dur['elapsed_s'] >= 180) & (dur['elapsed_s'] <= 660)]
        lat_peak = dur_peak['metric_value']
        reqs_peak = reqs[(reqs['elapsed_s'] >= 180) & (reqs['elapsed_s'] <= 660)]
        failed_peak = reqs_peak[~reqs_peak['expected_response']].shape[0]
        total_peak = len(reqs_peak)
        error_peak = (failed_peak / total_peak * 100) if total_peak > 0 else 0

        vus = extract_vus(df)

        rows.append({
            'Test': label,
            'Total Requests': total,
            'Successful': success,
            'Failed': failed,
            'Error Rate (%)': round(error_rate, 2),
            'Avg (ms)': round(lat.mean(), 1),
            'p50 (ms)': round(lat.quantile(0.50), 1),
            'p90 (ms)': round(lat.quantile(0.90), 1),
            'p95 (ms)': round(lat.quantile(0.95), 1),
            'p99 (ms)': round(lat.quantile(0.99), 1),
            'Max (ms)': round(lat.max(), 1),
            'p95 success-only (ms)': round(lat_ok.quantile(0.95), 1) if len(lat_ok) > 0 else 'N/A',
            'p99 success-only (ms)': round(lat_ok.quantile(0.99), 1) if len(lat_ok) > 0 else 'N/A',
            'Peak Error Rate (%)': round(error_peak, 2),
            'Peak p95 (ms)': round(lat_peak.quantile(0.95), 1) if len(lat_peak) > 0 else 'N/A',
            'Peak p99 (ms)': round(lat_peak.quantile(0.99), 1) if len(lat_peak) > 0 else 'N/A',
            'Max VUs': int(vus['metric_value'].max()) if len(vus) > 0 else 'N/A',
            'Status Codes': dict(status_dist),
        })

    # Stampa formattata
    for r in rows:
        print(f"\n  {'─' * 40}")
        print(f"  {r['Test']}")
        print(f"  {'─' * 40}")
        for k, v in r.items():
            if k == 'Test':
                continue
            print(f"    {k:.<35s} {v}")

    # Differenza percentuale
    if len(rows) == 2:
        on, off = rows[0], rows[1]
        print(f"\n  {'═' * 40}")
        print(f"  DELTA (DMOS ON vs OFF)")
        print(f"  {'═' * 40}")
        err_delta = on['Error Rate (%)'] - off['Error Rate (%)']
        print(f"    Error Rate delta................ {err_delta:+.2f} pp")
        err_reduction = ((off['Error Rate (%)'] - on['Error Rate (%)']) / off['Error Rate (%)'] * 100)
        print(f"    Error reduction................ {err_reduction:.1f}%")
        print(f"    Extra successful requests...... {on['Successful'] - off['Successful']:+d}")

    return rows


# ── 2. Latency Over Time (Time Series) ──────────────────────────────────────

def plot_latency_timeseries(datasets, window_s=10):
    """p50/p95/p99 rolling nel tempo, DMOS ON vs OFF sovrapposti."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    percentiles = [('p50', 0.50), ('p95', 0.95), ('p99', 0.99)]

    for ax, (pname, pval) in zip(axes, percentiles):
        for label, df in datasets.items():
            dur = extract_http_duration(df)
            # Bin per intervalli di window_s secondi
            dur['time_bin'] = (dur['elapsed_s'] // window_s) * window_s
            grouped = dur.groupby('time_bin')['metric_value'].quantile(pval).reset_index()
            grouped.columns = ['elapsed_s', 'latency']
            ax.plot(grouped['elapsed_s'] / 60, grouped['latency'],
                    label=label, color=COLORS[label], alpha=0.85, linewidth=1.5)

        ax.set_ylabel(f'{pname} Latency (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{pname} Latency Over Time ({window_s}s rolling window)')

        # Shading fasi
        _shade_phases(ax)

    axes[-1].set_xlabel('Time (minutes)')
    fig.suptitle('Latency Time Series — DMOS ON vs DMOS OFF', fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '01_latency_timeseries.png')
    print(f"  Salvato: {PLOT_DIR / '01_latency_timeseries.png'}")
    plt.close()


# ── 3. Error Rate Over Time ─────────────────────────────────────────────────

def plot_error_rate_timeseries(datasets, window_s=10):
    """Error rate % nel tempo per DMOS ON vs OFF."""
    fig, ax = plt.subplots(figsize=(16, 6))

    for label, df in datasets.items():
        reqs = extract_http_reqs(df)
        reqs['time_bin'] = (reqs['elapsed_s'] // window_s) * window_s
        grouped = reqs.groupby('time_bin').agg(
            total=('expected_response', 'count'),
            failed=('expected_response', lambda x: (~x).sum())
        ).reset_index()
        grouped['error_pct'] = grouped['failed'] / grouped['total'] * 100
        ax.plot(grouped['time_bin'] / 60, grouped['error_pct'],
                label=label, color=COLORS[label], alpha=0.85, linewidth=1.5)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title(f'Error Rate Over Time ({window_s}s window) — DMOS ON vs DMOS OFF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(PercentFormatter())
    _shade_phases(ax)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '02_error_rate_timeseries.png')
    print(f"  Salvato: {PLOT_DIR / '02_error_rate_timeseries.png'}")
    plt.close()


# ── 4. Throughput (RPS) Over Time ────────────────────────────────────────────

def plot_throughput_timeseries(datasets, window_s=10):
    """Requests per second nel tempo."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Total RPS
    ax = axes[0]
    for label, df in datasets.items():
        reqs = extract_http_reqs(df)
        reqs['time_bin'] = (reqs['elapsed_s'] // window_s) * window_s
        grouped = reqs.groupby('time_bin').size().reset_index(name='count')
        grouped['rps'] = grouped['count'] / window_s
        ax.plot(grouped['time_bin'] / 60, grouped['rps'],
                label=label, color=COLORS[label], alpha=0.85, linewidth=1.5)

    ax.set_ylabel('Requests/s (total)')
    ax.set_title(f'Total Throughput ({window_s}s window)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=75, color='gray', linestyle='--', alpha=0.5, label='Target 75 rps')
    _shade_phases(ax)

    # Successful RPS only
    ax = axes[1]
    for label, df in datasets.items():
        reqs = extract_http_reqs(df)
        reqs_ok = reqs[reqs['expected_response']]
        reqs_ok['time_bin'] = (reqs_ok['elapsed_s'] // window_s) * window_s
        grouped = reqs_ok.groupby('time_bin').size().reset_index(name='count')
        grouped['rps'] = grouped['count'] / window_s
        ax.plot(grouped['time_bin'] / 60, grouped['rps'],
                label=f'{label} (success only)', color=COLORS[label], alpha=0.85, linewidth=1.5)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Successful Requests/s')
    ax.set_title(f'Successful Throughput ({window_s}s window)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _shade_phases(ax)

    fig.suptitle('Throughput — DMOS ON vs DMOS OFF', fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '03_throughput_timeseries.png')
    print(f"  Salvato: {PLOT_DIR / '03_throughput_timeseries.png'}")
    plt.close()


# ── 5. Latency Distribution (Histogram / CDF) ───────────────────────────────

def plot_latency_distribution(datasets):
    """Istogramma e CDF della latenza HTTP."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax = axes[0]
    for label, df in datasets.items():
        dur = extract_http_duration(df)
        lat = dur['metric_value'].clip(upper=2000)  # Clip per leggibilita
        ax.hist(lat, bins=100, alpha=0.5, label=label, color=COLORS[label], density=True)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Latency Distribution (clipped at 2000ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # CDF
    ax = axes[1]
    for label, df in datasets.items():
        dur = extract_http_duration(df)
        lat = np.sort(dur['metric_value'].values)
        cdf = np.arange(1, len(lat) + 1) / len(lat)
        ax.plot(lat, cdf, label=label, color=COLORS[label], linewidth=1.5)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution Function (CDF)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2000)
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5)
    ax.text(1800, 0.955, 'p95', fontsize=9, color='gray')
    ax.text(1800, 0.995, 'p99', fontsize=9, color='gray')

    fig.suptitle('Latency Distribution — DMOS ON vs DMOS OFF', fontsize=16)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '04_latency_distribution.png')
    print(f"  Salvato: {PLOT_DIR / '04_latency_distribution.png'}")
    plt.close()


# ── 6. Per-Cluster Analysis ─────────────────────────────────────────────────

def plot_per_cluster_comparison(datasets):
    """Error rate e p95 per cluster, DMOS ON vs OFF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cluster_labels = list(INGRESS_MAP.keys())
    x = np.arange(len(cluster_labels))
    width = 0.35

    # Error rate per cluster
    ax = axes[0]
    for i, (label, df) in enumerate(datasets.items()):
        reqs = extract_http_reqs(df)
        error_rates = []
        for cl in cluster_labels:
            cl_reqs = reqs[reqs['ingress'] == cl]
            total = len(cl_reqs)
            failed = cl_reqs[~cl_reqs['expected_response']].shape[0]
            error_rates.append((failed / total * 100) if total > 0 else 0)

        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, error_rates, width, label=label, color=COLORS[label], alpha=0.8)
        for bar, val in zip(bars, error_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([INGRESS_MAP[c] for c in cluster_labels], fontsize=9)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate per Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # p95 per cluster
    ax = axes[1]
    for i, (label, df) in enumerate(datasets.items()):
        dur = extract_http_duration(df)
        p95s = []
        for cl in cluster_labels:
            cl_dur = dur[dur['ingress'] == cl]['metric_value']
            p95s.append(cl_dur.quantile(0.95) if len(cl_dur) > 0 else 0)

        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, p95s, width, label=label, color=COLORS[label], alpha=0.8)
        for bar, val in zip(bars, p95s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([INGRESS_MAP[c] for c in cluster_labels], fontsize=9)
    ax.set_ylabel('p95 Latency (ms)')
    ax.set_title('p95 Latency per Cluster')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Per-Cluster Comparison — DMOS ON vs DMOS OFF', fontsize=16)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '05_per_cluster_comparison.png')
    print(f"  Salvato: {PLOT_DIR / '05_per_cluster_comparison.png'}")
    plt.close()


# ── 7. Per-Cluster Latency Time Series ──────────────────────────────────────

def plot_per_cluster_timeseries(datasets, window_s=15):
    """p95 latency per cluster nel tempo, un subplot per test."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)

    for ax, (label, df) in zip(axes, datasets.items()):
        dur = extract_http_duration(df)
        for cl, cl_name in INGRESS_MAP.items():
            cl_dur = dur[dur['ingress'] == cl].copy()
            if len(cl_dur) == 0:
                continue
            cl_dur['time_bin'] = (cl_dur['elapsed_s'] // window_s) * window_s
            grouped = cl_dur.groupby('time_bin')['metric_value'].quantile(0.95).reset_index()
            grouped.columns = ['elapsed_s', 'p95']
            ax.plot(grouped['elapsed_s'] / 60, grouped['p95'],
                    label=cl_name, color=CLUSTER_COLORS[cl], alpha=0.85, linewidth=1.5)

        ax.set_ylabel('p95 Latency (ms)')
        ax.set_title(f'{label} — p95 per Cluster ({window_s}s window)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        _shade_phases(ax)

    axes[-1].set_xlabel('Time (minutes)')
    fig.suptitle('Per-Cluster p95 Latency Over Time', fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '06_per_cluster_latency_timeseries.png')
    print(f"  Salvato: {PLOT_DIR / '06_per_cluster_latency_timeseries.png'}")
    plt.close()


# ── 8. Per-Cluster Error Rate Time Series ────────────────────────────────────

def plot_per_cluster_error_timeseries(datasets, window_s=15):
    """Error rate per cluster nel tempo."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, sharey=True)

    for ax, (label, df) in zip(axes, datasets.items()):
        reqs = extract_http_reqs(df)
        for cl, cl_name in INGRESS_MAP.items():
            cl_reqs = reqs[reqs['ingress'] == cl].copy()
            if len(cl_reqs) == 0:
                continue
            cl_reqs['time_bin'] = (cl_reqs['elapsed_s'] // window_s) * window_s
            grouped = cl_reqs.groupby('time_bin').agg(
                total=('expected_response', 'count'),
                failed=('expected_response', lambda x: (~x).sum())
            ).reset_index()
            grouped['error_pct'] = grouped['failed'] / grouped['total'] * 100
            ax.plot(grouped['time_bin'] / 60, grouped['error_pct'],
                    label=cl_name, color=CLUSTER_COLORS[cl], alpha=0.85, linewidth=1.5)

        ax.set_ylabel('Error Rate (%)')
        ax.set_title(f'{label} — Error Rate per Cluster ({window_s}s window)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(PercentFormatter())
        _shade_phases(ax)

    axes[-1].set_xlabel('Time (minutes)')
    fig.suptitle('Per-Cluster Error Rate Over Time', fontsize=16, y=1.01)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '07_per_cluster_error_timeseries.png')
    print(f"  Salvato: {PLOT_DIR / '07_per_cluster_error_timeseries.png'}")
    plt.close()


# ── 9. Status Code Distribution ─────────────────────────────────────────────

def plot_status_codes(datasets):
    """Distribuzione status code, DMOS ON vs OFF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (label, df) in zip(axes, datasets.items()):
        reqs = extract_http_reqs(df)
        status_counts = reqs['status'].value_counts().sort_index()

        colors_map = []
        for s in status_counts.index:
            if s < 300:
                colors_map.append('#4CAF50')  # green = 2xx
            elif s < 400:
                colors_map.append('#2196F3')  # blue = 3xx (redirect)
            elif s < 500:
                colors_map.append('#FF9800')  # orange = 4xx
            else:
                colors_map.append('#F44336')  # red = 5xx

        bars = ax.bar([str(int(s)) for s in status_counts.index],
                      status_counts.values, color=colors_map, alpha=0.8)

        for bar, val in zip(bars, status_counts.values):
            pct = val / len(reqs) * 100
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('HTTP Status Code')
        ax.set_ylabel('Count')
        ax.set_title(f'{label} — Status Code Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('HTTP Status Code Distribution', fontsize=16)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '08_status_codes.png')
    print(f"  Salvato: {PLOT_DIR / '08_status_codes.png'}")
    plt.close()


# ── 10. Per-Endpoint Analysis ────────────────────────────────────────────────

def plot_per_endpoint(datasets):
    """p95 e error rate per endpoint, DMOS ON vs OFF."""
    # Normalizza URL a path
    def url_to_path(url):
        if pd.isna(url):
            return 'unknown'
        parts = str(url).split(':30080')
        return parts[1] if len(parts) > 1 else url

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    all_paths = set()
    for label, df in datasets.items():
        reqs = extract_http_reqs(df)
        reqs['path'] = reqs['url'].apply(url_to_path)
        all_paths.update(reqs['path'].unique())

    paths = sorted([p for p in all_paths if p != 'unknown'])
    x = np.arange(len(paths))
    width = 0.35

    # p95 per endpoint
    ax = axes[0]
    for i, (label, df) in enumerate(datasets.items()):
        dur = extract_http_duration(df)
        dur['path'] = dur['url'].apply(url_to_path)
        p95s = []
        for p in paths:
            p_dur = dur[dur['path'] == p]['metric_value']
            p95s.append(p_dur.quantile(0.95) if len(p_dur) > 0 else 0)

        offset = -width / 2 + i * width
        ax.barh(x + offset, p95s, width, label=label, color=COLORS[label], alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(paths, fontsize=9)
    ax.set_xlabel('p95 Latency (ms)')
    ax.set_title('p95 Latency per Endpoint')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Error rate per endpoint
    ax = axes[1]
    for i, (label, df) in enumerate(datasets.items()):
        reqs = extract_http_reqs(df)
        reqs['path'] = reqs['url'].apply(url_to_path)
        error_rates = []
        for p in paths:
            p_reqs = reqs[reqs['path'] == p]
            total = len(p_reqs)
            failed = p_reqs[~p_reqs['expected_response']].shape[0]
            error_rates.append((failed / total * 100) if total > 0 else 0)

        offset = -width / 2 + i * width
        ax.barh(x + offset, error_rates, width, label=label, color=COLORS[label], alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(paths, fontsize=9)
    ax.set_xlabel('Error Rate (%)')
    ax.set_title('Error Rate per Endpoint')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Per-Endpoint Analysis — DMOS ON vs DMOS OFF', fontsize=16)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '09_per_endpoint.png')
    print(f"  Salvato: {PLOT_DIR / '09_per_endpoint.png'}")
    plt.close()


# ── 11. VUs Over Time ───────────────────────────────────────────────────────

def plot_vus_timeseries(datasets):
    """Virtual Users attivi nel tempo."""
    fig, ax = plt.subplots(figsize=(16, 5))

    for label, df in datasets.items():
        vus = extract_vus(df)
        if len(vus) == 0:
            continue
        ax.plot(vus['elapsed_s'] / 60, vus['metric_value'],
                label=label, color=COLORS[label], alpha=0.85, linewidth=1.5)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Active VUs')
    ax.set_title('Virtual Users Over Time — DMOS ON vs DMOS OFF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _shade_phases(ax)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '10_vus_timeseries.png')
    print(f"  Salvato: {PLOT_DIR / '10_vus_timeseries.png'}")
    plt.close()


# ── 12. Per-Phase Summary Bar Chart ─────────────────────────────────────────

def plot_per_phase_summary(datasets):
    """Error rate e p95 per fase del test."""
    phase_names = [p[2].replace('\n', ' ') for p in PHASES]
    x = np.arange(len(phase_names))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Error rate per fase
    ax = axes[0]
    for i, (label, df) in enumerate(datasets.items()):
        reqs = extract_http_reqs(df)
        error_rates = []
        for start, end, _ in PHASES:
            phase_reqs = reqs[(reqs['elapsed_s'] >= start) & (reqs['elapsed_s'] < end)]
            total = len(phase_reqs)
            failed = phase_reqs[~phase_reqs['expected_response']].shape[0]
            error_rates.append((failed / total * 100) if total > 0 else 0)

        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, error_rates, width, label=label, color=COLORS[label], alpha=0.8)
        for bar, val in zip(bars, error_rates):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(phase_names, fontsize=9)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rate per Test Phase')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # p95 per fase
    ax = axes[1]
    for i, (label, df) in enumerate(datasets.items()):
        dur = extract_http_duration(df)
        p95s = []
        for start, end, _ in PHASES:
            phase_dur = dur[(dur['elapsed_s'] >= start) & (dur['elapsed_s'] < end)]
            p95s.append(phase_dur['metric_value'].quantile(0.95) if len(phase_dur) > 0 else 0)

        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, p95s, width, label=label, color=COLORS[label], alpha=0.8)
        for bar, val in zip(bars, p95s):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f'{val:.0f}ms', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(phase_names, fontsize=9)
    ax.set_ylabel('p95 Latency (ms)')
    ax.set_title('p95 Latency per Test Phase')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Per-Phase Analysis — DMOS ON vs DMOS OFF', fontsize=16)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '11_per_phase_summary.png')
    print(f"  Salvato: {PLOT_DIR / '11_per_phase_summary.png'}")
    plt.close()


# ── 13. Heatmap: Error Rate per Cluster × Phase ─────────────────────────────

def plot_heatmap_errors(datasets):
    """Heatmap: error rate per cluster e fase."""
    phase_names = [p[2].replace('\n', ' ') for p in PHASES]
    cluster_labels = list(INGRESS_MAP.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (label, df) in zip(axes, datasets.items()):
        reqs = extract_http_reqs(df)
        matrix = np.zeros((len(cluster_labels), len(PHASES)))

        for j, (start, end, _) in enumerate(PHASES):
            for i, cl in enumerate(cluster_labels):
                cl_reqs = reqs[(reqs['ingress'] == cl) &
                               (reqs['elapsed_s'] >= start) &
                               (reqs['elapsed_s'] < end)]
                total = len(cl_reqs)
                failed = cl_reqs[~cl_reqs['expected_response']].shape[0]
                matrix[i, j] = (failed / total * 100) if total > 0 else 0

        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=50)
        ax.set_xticks(range(len(phase_names)))
        ax.set_xticklabels(phase_names, fontsize=8, rotation=20, ha='right')
        ax.set_yticks(range(len(cluster_labels)))
        ax.set_yticklabels([INGRESS_MAP[c] for c in cluster_labels], fontsize=9)
        ax.set_title(f'{label}')

        # Annotazioni
        for i in range(len(cluster_labels)):
            for j in range(len(PHASES)):
                ax.text(j, i, f'{matrix[i, j]:.1f}%',
                        ha='center', va='center', fontsize=9,
                        color='white' if matrix[i, j] > 25 else 'black')

    fig.colorbar(im, ax=axes, label='Error Rate (%)', shrink=0.8)
    fig.suptitle('Error Rate Heatmap: Cluster × Phase — DMOS ON vs DMOS OFF', fontsize=14)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '12_heatmap_errors.png')
    print(f"  Salvato: {PLOT_DIR / '12_heatmap_errors.png'}")
    plt.close()


# ── 14. Traffic Distribution per Cluster ─────────────────────────────────────

def plot_traffic_distribution(datasets):
    """Pie chart: distribuzione traffico per cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (label, df) in zip(axes, datasets.items()):
        reqs = extract_http_reqs(df)
        counts = reqs['ingress'].value_counts()

        labels_pie = [INGRESS_MAP.get(c, c) for c in counts.index]
        colors_pie = [CLUSTER_COLORS.get(c, 'gray') for c in counts.index]

        wedges, texts, autotexts = ax.pie(
            counts.values, labels=labels_pie, autopct='%1.1f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 9}
        )
        ax.set_title(f'{label}\n({counts.sum()} requests)')

    fig.suptitle('Traffic Distribution per Cluster', fontsize=14)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '13_traffic_distribution.png')
    print(f"  Salvato: {PLOT_DIR / '13_traffic_distribution.png'}")
    plt.close()


# ── 15. Comprehensive Summary Table (saved as image) ─────────────────────────

def plot_summary_table_image(summary_rows):
    """Crea immagine della tabella summary per inserimento diretto nella tesi."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Prepara dati tabella
    metrics = [
        'Total Requests', 'Successful', 'Failed', 'Error Rate (%)',
        'Avg (ms)', 'p50 (ms)', 'p90 (ms)', 'p95 (ms)', 'p99 (ms)', 'Max (ms)',
        'p95 success-only (ms)', 'p99 success-only (ms)',
        'Peak Error Rate (%)', 'Peak p95 (ms)', 'Peak p99 (ms)', 'Max VUs'
    ]

    cell_text = []
    for m in metrics:
        row = [m]
        for r in summary_rows:
            row.append(str(r.get(m, 'N/A')))
        # Delta
        if len(summary_rows) == 2:
            try:
                v_on = float(summary_rows[0][m])
                v_off = float(summary_rows[1][m])
                delta = v_on - v_off
                pct = ((v_on - v_off) / v_off * 100) if v_off != 0 else 0
                row.append(f'{delta:+.1f} ({pct:+.1f}%)')
            except (ValueError, TypeError):
                row.append('-')
        cell_text.append(row)

    col_labels = ['Metric', 'DMOS ON', 'DMOS OFF', 'Delta']

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Colora header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#E0E0E0')
        table[0, j].set_text_props(weight='bold')

    # Colora righe con error rate
    for i, m in enumerate(metrics):
        if 'Error' in m or 'Failed' in m:
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor('#FFF3E0')

    ax.set_title('DMOS ON vs DMOS OFF — Comprehensive Comparison', fontsize=14, pad=20)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / '00_summary_table.png')
    print(f"  Salvato: {PLOT_DIR / '00_summary_table.png'}")
    plt.close()


# ── Helper: shade phases ─────────────────────────────────────────────────────

def _shade_phases(ax):
    """Aggiunge sfondo colorato per le fasi del test."""
    colors_phase = ['#E3F2FD', '#FFF9C4', '#FFEBEE', '#FFF9C4', '#E8F5E9']
    for (start, end, name), color in zip(PHASES, colors_phase):
        ax.axvspan(start / 60, end / 60, alpha=0.15, color=color)
        # Label solo se abbastanza largo
        if (end - start) > 50:
            ax.text((start + end) / 2 / 60, ax.get_ylim()[1] * 0.97,
                    name.replace('\n', ' '), ha='center', va='top',
                    fontsize=7, alpha=0.6)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  k6 Results Analyzer — DMOS ON vs DMOS OFF")
    print("=" * 60)

    # Verifica file
    for label, path in FILES.items():
        if not path.exists():
            print(f"\n  ERRORE: File non trovato: {path}")
            print(f"  Assicurati che i CSV siano nella cartella experiments/")
            sys.exit(1)

    # Caricamento
    print("\n[1/2] Caricamento CSV...")
    datasets = {}
    for label, path in FILES.items():
        datasets[label] = load_k6_csv(path, label)

    # Summary
    print("\n[2/2] Generazione analisi...")
    summary_rows = print_summary_table(datasets)

    # Grafici
    print(f"\nGenerazione grafici in {PLOT_DIR}/...")
    plot_summary_table_image(summary_rows)
    plot_latency_timeseries(datasets)
    plot_error_rate_timeseries(datasets)
    plot_throughput_timeseries(datasets)
    plot_latency_distribution(datasets)
    plot_per_cluster_comparison(datasets)
    plot_per_cluster_timeseries(datasets)
    plot_per_cluster_error_timeseries(datasets)
    plot_status_codes(datasets)
    plot_per_endpoint(datasets)
    plot_vus_timeseries(datasets)
    plot_per_phase_summary(datasets)
    plot_heatmap_errors(datasets)
    plot_traffic_distribution(datasets)

    print(f"\n{'=' * 60}")
    print(f"  COMPLETO! {len(os.listdir(PLOT_DIR))} grafici salvati in:")
    print(f"  {PLOT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
