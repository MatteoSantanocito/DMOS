#!/usr/bin/env python3
"""
Plot Per-Cluster Latency Comparison
====================================
Reads the time-series CSV produced by locustfile_multiingress.py
and generates comparison graphs similar to Romano's thesis (Tables 5-1/5-2).

Produces:
  1. Per-cluster p95 latency over time (3 lines, one per cluster)
  2. Per-cluster average latency bar chart (with error bars)
  3. Request distribution pie chart

Usage:
  python plot_cluster_latency.py results/multiingress/flash_crowd_timeseries_*.csv
  python plot_cluster_latency.py results/multiingress/  # processes all CSVs in dir
"""

import sys
import csv
import glob
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# Cluster colors (consistent with DMOS analyzer)
COLORS = {
    'cluster1': '#1f77b4',  # blue  — DE
    'cluster2': '#ff7f0e',  # orange — FR
    'cluster3': '#2ca02c',  # green — PL
}
REGIONS = {'cluster1': 'DE', 'cluster2': 'FR', 'cluster3': 'PL'}
CLUSTERS = ['cluster1', 'cluster2', 'cluster3']


def load_timeseries(csv_path):
    """Load time-series CSV into structured data."""
    data = {c: {'timestamps': [], 'counts': [], 'avg_ms': [], 'p95_ms': []} for c in CLUSTERS}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row['timestamp']
            for c in CLUSTERS:
                count = int(row.get(f'{c}_count', 0))
                avg = float(row.get(f'{c}_avg_ms', 0))
                p95 = float(row.get(f'{c}_p95_ms', 0))
                if count > 0:
                    data[c]['timestamps'].append(ts)
                    data[c]['counts'].append(count)
                    data[c]['avg_ms'].append(avg)
                    data[c]['p95_ms'].append(p95)

    return data


def load_summary(csv_path):
    """Load summary CSV into list of dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_p95_over_time(data, scenario_name, output_path):
    """Plot 1: Per-cluster p95 latency over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(f'DMOS Multi-Ingress — Per-Cluster p95 Latency\n{scenario_name}',
                 fontsize=14, fontweight='bold')

    for c in CLUSTERS:
        if data[c]['timestamps']:
            times = range(len(data[c]['timestamps']))
            ax.plot(times, data[c]['p95_ms'],
                    color=COLORS[c], lw=1.5, alpha=0.8,
                    label=f"{c} ({REGIONS[c]})")

    ax.set_xlabel('Time (windows)')
    ax.set_ylabel('p95 Response Time (ms)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add tick labels every N windows
    if data['cluster1']['timestamps']:
        ts_list = data['cluster1']['timestamps']
        n = len(ts_list)
        step = max(1, n // 20)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([ts_list[i] for i in range(0, n, step)], rotation=45, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {output_path}")


def plot_avg_comparison(summary_rows, scenario_name, output_path):
    """Plot 2: Bar chart comparing avg/p50/p95/p99 per cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'DMOS Multi-Ingress — Cluster Latency Comparison\n{scenario_name}',
                 fontsize=14, fontweight='bold')

    clusters = [r['cluster'] for r in summary_rows]
    regions = [REGIONS.get(r['cluster'], '?') for r in summary_rows]
    labels = [f"{c}\n({reg})" for c, reg in zip(clusters, regions)]

    # -- Left: Latency bars --
    ax = axes[0]
    x = np.arange(len(clusters))
    width = 0.2

    avg_vals = [float(r['avg_ms']) for r in summary_rows]
    p50_vals = [float(r['p50_ms']) for r in summary_rows]
    p95_vals = [float(r['p95_ms']) for r in summary_rows]
    p99_vals = [float(r['p99_ms']) for r in summary_rows]

    bars1 = ax.bar(x - 1.5*width, avg_vals, width, label='Avg', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, p50_vals, width, label='p50', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, p95_vals, width, label='p95', color='#e74c3c', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, p99_vals, width, label='p99', color='#9b59b6', alpha=0.8)

    ax.set_ylabel('Response Time (ms)')
    ax.set_title('Latency per Cluster')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)

    # -- Right: Request distribution pie --
    ax2 = axes[1]
    req_counts = [int(r['requests']) for r in summary_rows]
    colors = [COLORS.get(r['cluster'], '#999') for r in summary_rows]

    if sum(req_counts) > 0:
        wedges, texts, autotexts = ax2.pie(
            req_counts, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'fontsize': 10}
        )
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
    ax2.set_title('Request Distribution')

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {output_path}")


def plot_request_rate_over_time(data, scenario_name, output_path):
    """Plot 3: Per-cluster request rate (RPS) over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'DMOS Multi-Ingress — Per-Cluster Request Rate\n{scenario_name}',
                 fontsize=14, fontweight='bold')

    for c in CLUSTERS:
        if data[c]['timestamps']:
            # Convert counts per window to RPS (window = 10s)
            rps = [count / 10.0 for count in data[c]['counts']]
            ax.plot(range(len(rps)), rps,
                    color=COLORS[c], lw=1.5, alpha=0.8,
                    label=f"{c} ({REGIONS[c]})")

    ax.set_xlabel('Time (windows)')
    ax.set_ylabel('Requests per Second')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if data['cluster1']['timestamps']:
        ts_list = data['cluster1']['timestamps']
        n = len(ts_list)
        step = max(1, n // 20)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([ts_list[i] for i in range(0, n, step)], rotation=45, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {output_path}")


def process_scenario(ts_csv_path, summary_csv_path=None):
    """Process one scenario's CSV files and generate all plots."""
    ts_path = Path(ts_csv_path)
    scenario_name = ts_path.stem.split("_timeseries")[0].replace("_", " ").title()
    output_dir = ts_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    base = ts_path.stem.replace("_timeseries", "").rsplit("_", 1)[0]

    print(f"\n  Processing: {scenario_name}")
    print(f"  {'─'*50}")

    # Load time-series data
    data = load_timeseries(ts_csv_path)

    # Plot 1: p95 over time
    plot_p95_over_time(data, scenario_name, output_dir / f"{base}_p95_per_cluster.png")

    # Plot 3: Request rate over time
    plot_request_rate_over_time(data, scenario_name, output_dir / f"{base}_rps_per_cluster.png")

    # Plot 2: Summary comparison (if summary CSV available)
    if summary_csv_path and Path(summary_csv_path).exists():
        summary = load_summary(summary_csv_path)
        if summary:
            plot_avg_comparison(summary, scenario_name, output_dir / f"{base}_latency_comparison.png")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_cluster_latency.py <timeseries.csv> [summary.csv]")
        print("       python plot_cluster_latency.py <results_dir>/")
        sys.exit(1)

    target = sys.argv[1]

    if Path(target).is_dir():
        # Process all scenarios in directory
        ts_files = sorted(glob.glob(f"{target}/*_timeseries_*.csv"))
        if not ts_files:
            print(f"No time-series CSV files found in {target}")
            sys.exit(1)

        for ts_file in ts_files:
            # Try to find matching summary file
            summary_pattern = ts_file.replace("_timeseries_", "_cluster_latency_")
            summary_files = glob.glob(summary_pattern)
            summary = summary_files[0] if summary_files else None
            process_scenario(ts_file, summary)
    else:
        # Process single file
        summary = sys.argv[2] if len(sys.argv) > 2 else None
        process_scenario(target, summary)

    print(f"\n  All plots generated!\n")


if __name__ == "__main__":
    main()
