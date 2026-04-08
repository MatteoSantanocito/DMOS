#!/usr/bin/env python3
"""
analyze_dmos_live.py — DMOS Live Metrics Analyzer
===================================================
Reads JSONL from collect_dmos_live.py and generates high-resolution plots
showing how DMOS worked: scaling, scheduling, traffic prediction, cluster scores.

Usage:
  python analyze_dmos_live.py --input ..\results\XXXX_dmos_live.jsonl --output-dir plots_dmos
  python analyze_dmos_live.py --on ..\results\ON_dmos_live.jsonl --off ..\results\OFF_dmos_live.jsonl --output-dir plots_dmos_compare
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from datetime import datetime

# ─── Style ──────────────────────────────────────────────────────────────────

DPI = 600
FIGSIZE_FULL = (16, 10)
FIGSIZE_HALF = (16, 6)

CLUSTER_COLORS = {
    "cluster1": "#2196F3",  # blue  (DE)
    "cluster2": "#4CAF50",  # green (FR)
    "cluster3": "#FF9800",  # orange (PL)
}
CLUSTER_LABELS = {
    "cluster1": "C1-DE",
    "cluster2": "C2-FR",
    "cluster3": "C3-PL",
}

SERVICE_COLORS = {
    "frontend": "#E91E63",
    "cartservice": "#9C27B0",
    "productcatalogservice": "#3F51B5",
    "checkoutservice": "#009688",
    "recommendationservice": "#FF5722",
}

KNOWN_SERVICES = [
    "frontend", "cartservice", "productcatalogservice",
    "checkoutservice", "recommendationservice"
]
KNOWN_CLUSTERS = ["cluster1", "cluster2", "cluster3"]


# ─── Data Loader ────────────────────────────────────────────────────────────

def load_live_jsonl(path: str) -> list[dict]:
    """Load line-delimited JSONL snapshots."""
    snapshots = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"  Loaded {len(snapshots)} snapshots from {path}")
    return snapshots


def extract_series(snapshots: list[dict]) -> dict:
    """Extract time-series arrays from snapshots for easy plotting."""
    ts = np.array([s["elapsed_s"] for s in snapshots])
    d = {"elapsed": ts}

    # Per-service series
    for svc in KNOWN_SERVICES:
        prefix = svc
        actual = []
        predicted = []
        total_reps = []
        total_tgt = []
        sched_inv = []

        for s in snapshots:
            sd = s.get("services", {}).get(svc, {})
            actual.append(sd.get("actual_traffic_rps", 0))
            predicted.append(sd.get("predicted_traffic_total_rps", 0))
            total_reps.append(sd.get("total_current_replicas", 0))
            total_tgt.append(sd.get("total_target_replicas", 0))
            sched_inv.append(sd.get("scheduling_invocations", 0))

        d[f"{prefix}_actual_rps"] = np.array(actual)
        d[f"{prefix}_predicted_rps"] = np.array(predicted)
        d[f"{prefix}_total_replicas"] = np.array(total_reps)
        d[f"{prefix}_total_target"] = np.array(total_tgt)
        d[f"{prefix}_sched_invocations"] = np.array(sched_inv)

        # Per-cluster per-service
        for cl in KNOWN_CLUSTERS:
            cr = []
            tr = []
            sc = []
            pt = []
            su = []
            sd_ev = []
            cpu = []
            for s in snapshots:
                cd = s.get("services", {}).get(svc, {}).get("clusters", {}).get(cl, {})
                cr.append(cd.get("current_replicas", 0))
                tr.append(cd.get("target_replicas", 0))
                sc.append(cd.get("score", 0))
                pt.append(cd.get("predicted_traffic", 0))
                su.append(cd.get("scale_up_events", 0))
                sd_ev.append(cd.get("scale_down_events", 0))
                cpu.append(cd.get("cpu_cores") if cd.get("cpu_cores") is not None else np.nan)

            d[f"{prefix}_{cl}_replicas"] = np.array(cr)
            d[f"{prefix}_{cl}_target"] = np.array(tr)
            d[f"{prefix}_{cl}_score"] = np.array(sc)
            d[f"{prefix}_{cl}_predicted"] = np.array(pt)
            d[f"{prefix}_{cl}_scaleup"] = np.array(su)
            d[f"{prefix}_{cl}_scaledown"] = np.array(sd_ev)
            d[f"{prefix}_{cl}_cpu"] = np.array(cpu)

    # Cluster-level
    for cl in KNOWN_CLUSTERS:
        hub = []
        ncpu = []
        tpct = []
        for s in snapshots:
            cs = s.get("clusters", {}).get(cl, {})
            hub.append(cs.get("hubble_http_rps") if cs.get("hubble_http_rps") is not None else np.nan)
            ncpu.append(cs.get("node_cpu_pct") if cs.get("node_cpu_pct") is not None else np.nan)
            tpct.append(cs.get("traffic_pct", 0))

        d[f"{cl}_hubble_rps"] = np.array(hub)
        d[f"{cl}_node_cpu_pct"] = np.array(ncpu)
        d[f"{cl}_traffic_pct"] = np.array(tpct)

    return d


# ─── Plot Functions ─────────────────────────────────────────────────────────

def plot_01_traffic_prediction(d: dict, out_dir: Path):
    """Page 1: Actual vs Predicted traffic per service."""
    fig, axes = plt.subplots(len(KNOWN_SERVICES), 1, figsize=FIGSIZE_FULL, sharex=True)
    fig.suptitle("DMOS — Actual vs Predicted Traffic", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60  # minutes

    for i, svc in enumerate(KNOWN_SERVICES):
        ax = axes[i]
        actual = d[f"{svc}_actual_rps"]
        predicted = d[f"{svc}_predicted_rps"]

        ax.plot(t, actual, color=SERVICE_COLORS[svc], linewidth=1.5, label="Actual", alpha=0.9)
        ax.plot(t, predicted, color=SERVICE_COLORS[svc], linewidth=1.2, linestyle="--",
                label="Predicted", alpha=0.7)
        ax.fill_between(t, actual, predicted, alpha=0.15, color=SERVICE_COLORS[svc])

        # Error band
        err = np.abs(actual - predicted)
        mean_err = np.nanmean(err[actual > 0])
        ax.set_ylabel("req/s", fontsize=9)
        ax.set_title(f"{svc}  (MAE={mean_err:.1f} rps)", fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out_dir / "dmos_01_traffic_prediction.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [1/8] Traffic prediction")


def plot_02_replicas_per_cluster(d: dict, out_dir: Path):
    """Page 2: Stacked area — replicas per cluster for each service."""
    fig, axes = plt.subplots(len(KNOWN_SERVICES), 1, figsize=FIGSIZE_FULL, sharex=True)
    fig.suptitle("DMOS — Replica Distribution per Cluster", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60

    for i, svc in enumerate(KNOWN_SERVICES):
        ax = axes[i]
        stacks = []
        labels = []
        colors = []
        for cl in KNOWN_CLUSTERS:
            stacks.append(d[f"{svc}_{cl}_replicas"])
            labels.append(CLUSTER_LABELS[cl])
            colors.append(CLUSTER_COLORS[cl])

        ax.stackplot(t, *stacks, labels=labels, colors=colors, alpha=0.7)
        # Total target line
        total_tgt = d[f"{svc}_total_target"]
        ax.plot(t, total_tgt, 'k--', linewidth=1, alpha=0.5, label="Target")

        ax.set_ylabel("Replicas", fontsize=9)
        ax.set_title(svc, fontsize=10, loc="left")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if i == 0:
            ax.legend(fontsize=8, loc="upper right", ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out_dir / "dmos_02_replicas_per_cluster.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [2/8] Replicas per cluster")


def plot_03_cluster_scores(d: dict, out_dir: Path):
    """Page 3: Cluster scores over time (per service)."""
    fig, axes = plt.subplots(len(KNOWN_SERVICES), 1, figsize=FIGSIZE_FULL, sharex=True)
    fig.suptitle("DMOS — Multi-Objective Cluster Scores (higher = better)", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60

    for i, svc in enumerate(KNOWN_SERVICES):
        ax = axes[i]
        for cl in KNOWN_CLUSTERS:
            scores = d[f"{svc}_{cl}_score"]
            ax.plot(t, scores, color=CLUSTER_COLORS[cl], linewidth=1.3,
                    label=CLUSTER_LABELS[cl], alpha=0.9)

        ax.set_ylabel("Score", fontsize=9)
        ax.set_title(svc, fontsize=10, loc="left")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out_dir / "dmos_03_cluster_scores.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [3/8] Cluster scores")


def plot_04_scaling_events(d: dict, out_dir: Path):
    """Page 4: Cumulative scaling events + delta markers."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_HALF, sharex=True)
    fig.suptitle("DMOS — Scaling Events Timeline", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60

    # Top: frontend detail
    ax = axes[0]
    ax.set_title("Frontend — Replicas vs Traffic", fontsize=11, loc="left")
    ax2 = ax.twinx()

    for cl in KNOWN_CLUSTERS:
        reps = d[f"frontend_{cl}_replicas"]
        ax.step(t, reps, where='post', color=CLUSTER_COLORS[cl],
                linewidth=1.5, label=f"Reps {CLUSTER_LABELS[cl]}")

    actual = d["frontend_actual_rps"]
    predicted = d["frontend_predicted_rps"]
    ax2.plot(t, actual, 'k-', linewidth=1, alpha=0.6, label="Actual RPS")
    ax2.plot(t, predicted, 'k--', linewidth=1, alpha=0.4, label="Predicted RPS")

    ax.set_ylabel("Replicas")
    ax2.set_ylabel("req/s")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left", ncol=3)
    ax.grid(True, alpha=0.3)

    # Bottom: cumulative scale events for all services
    ax = axes[1]
    ax.set_title("Cumulative Scale Events (all services)", fontsize=11, loc="left")

    for svc in KNOWN_SERVICES:
        for cl in KNOWN_CLUSTERS:
            up = d[f"{svc}_{cl}_scaleup"]
            down = d[f"{svc}_{cl}_scaledown"]
            total_ev = up + down
            if np.max(total_ev) > 0:
                ax.plot(t, total_ev, linewidth=1, alpha=0.7,
                        color=CLUSTER_COLORS[cl],
                        label=f"{svc[:8]}@{CLUSTER_LABELS[cl]}")

    ax.set_ylabel("Events (cumulative)")
    ax.set_xlabel("Time (min)")
    # Only show legend if not too many entries
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 15:
        ax.legend(fontsize=6, loc="upper left", ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "dmos_04_scaling_events.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [4/8] Scaling events")


def plot_05_traffic_distribution(d: dict, out_dir: Path):
    """Page 5: Traffic distribution % across clusters + Hubble RPS."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_HALF, sharex=True)
    fig.suptitle("DMOS — Traffic Distribution & Hubble HTTP Rate", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60

    # Stacked area: traffic %
    ax = axes[0]
    pcts = [d[f"{cl}_traffic_pct"] for cl in KNOWN_CLUSTERS]
    ax.stackplot(t, *pcts,
                 labels=[CLUSTER_LABELS[cl] for cl in KNOWN_CLUSTERS],
                 colors=[CLUSTER_COLORS[cl] for cl in KNOWN_CLUSTERS],
                 alpha=0.7)
    ax.set_ylabel("Traffic Share (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Hubble HTTP RPS per cluster
    ax = axes[1]
    for cl in KNOWN_CLUSTERS:
        hub = d[f"{cl}_hubble_rps"]
        ax.plot(t, hub, color=CLUSTER_COLORS[cl], linewidth=1.3,
                label=CLUSTER_LABELS[cl])
    ax.set_ylabel("Hubble HTTP RPS")
    ax.set_xlabel("Time (min)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "dmos_05_traffic_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [5/8] Traffic distribution")


def plot_06_node_cpu(d: dict, out_dir: Path):
    """Page 6: Node CPU + per-service CPU per cluster."""
    fig, axes = plt.subplots(len(KNOWN_CLUSTERS), 1, figsize=FIGSIZE_FULL, sharex=True)
    fig.suptitle("DMOS — Node CPU & Frontend CPU per Cluster", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60

    for i, cl in enumerate(KNOWN_CLUSTERS):
        ax = axes[i]
        # Node CPU
        ncpu = d[f"{cl}_node_cpu_pct"]
        ax.fill_between(t, 0, ncpu, alpha=0.2, color=CLUSTER_COLORS[cl])
        ax.plot(t, ncpu, color=CLUSTER_COLORS[cl], linewidth=1.5,
                label=f"Node CPU %", alpha=0.8)

        # Per-service CPU on this cluster
        ax2 = ax.twinx()
        for svc in KNOWN_SERVICES:
            cpu = d[f"{svc}_{cl}_cpu"]
            if not np.all(np.isnan(cpu)):
                ax2.plot(t, cpu, linewidth=1, alpha=0.7,
                         color=SERVICE_COLORS[svc], linestyle=":",
                         label=f"{svc[:12]} CPU")

        ax.set_ylabel(f"{CLUSTER_LABELS[cl]} Node CPU %")
        ax2.set_ylabel("Service CPU (cores)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        if i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6,
                      loc="upper right", ncol=3)

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out_dir / "dmos_06_node_cpu.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [6/8] Node CPU")


def plot_07_frontend_detail(d: dict, out_dir: Path):
    """Page 7: Frontend deep dive — traffic, replicas, scores, CPU in one view."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 1, hspace=0.35)
    fig.suptitle("DMOS — Frontend Deep Dive", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60

    # Row 1: Actual vs Predicted traffic
    ax = fig.add_subplot(gs[0])
    actual = d["frontend_actual_rps"]
    predicted = d["frontend_predicted_rps"]
    ax.plot(t, actual, 'k-', linewidth=1.5, label="Actual")
    ax.plot(t, predicted, 'r--', linewidth=1.2, label="Predicted")
    ax.fill_between(t, actual, predicted, alpha=0.15, color="red")
    ax.set_ylabel("req/s")
    ax.set_title("Traffic: Actual vs Predicted", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2: Per-cluster replicas (step)
    ax = fig.add_subplot(gs[1])
    for cl in KNOWN_CLUSTERS:
        reps = d[f"frontend_{cl}_replicas"]
        ax.step(t, reps, where='post', color=CLUSTER_COLORS[cl],
                linewidth=1.5, label=CLUSTER_LABELS[cl])
    total = d["frontend_total_replicas"]
    ax.plot(t, total, 'k-', linewidth=1, alpha=0.5, label="Total")
    ax.set_ylabel("Replicas")
    ax.set_title("Replicas per Cluster", fontsize=10, loc="left")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # Row 3: Cluster scores
    ax = fig.add_subplot(gs[2])
    for cl in KNOWN_CLUSTERS:
        sc = d[f"frontend_{cl}_score"]
        ax.plot(t, sc, color=CLUSTER_COLORS[cl], linewidth=1.3,
                label=CLUSTER_LABELS[cl])
    ax.set_ylabel("Score")
    ax.set_title("Multi-Objective Cluster Scores", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 4: Per-cluster CPU for frontend
    ax = fig.add_subplot(gs[3])
    for cl in KNOWN_CLUSTERS:
        cpu = d[f"frontend_{cl}_cpu"]
        ax.plot(t, cpu, color=CLUSTER_COLORS[cl], linewidth=1.3,
                label=CLUSTER_LABELS[cl])
    ax.set_ylabel("CPU (cores)")
    ax.set_xlabel("Time (min)")
    ax.set_title("Frontend CPU Usage per Cluster", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(out_dir / "dmos_07_frontend_detail.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [7/8] Frontend detail")


def plot_08_scheduling_stats(d: dict, out_dir: Path):
    """Page 8: Scheduling invocation count + Jain fairness (estimated from replicas)."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_HALF, sharex=True)
    fig.suptitle("DMOS — Scheduling Activity & Fairness", fontsize=14, fontweight="bold")
    t = d["elapsed"] / 60

    # Scheduling invocations (delta per interval)
    ax = axes[0]
    ax.set_title("Scheduling Invocations (cumulative)", fontsize=10, loc="left")
    for svc in KNOWN_SERVICES:
        inv = d[f"{svc}_sched_invocations"]
        ax.plot(t, inv, color=SERVICE_COLORS[svc], linewidth=1.2, label=svc[:15])
    ax.set_ylabel("Invocations")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)

    # Jain's Fairness Index for frontend replica distribution
    ax = axes[1]
    ax.set_title("Frontend — Jain's Fairness Index (replica distribution)", fontsize=10, loc="left")

    jain_vals = []
    for idx in range(len(t)):
        reps = []
        for cl in KNOWN_CLUSTERS:
            r = d[f"frontend_{cl}_replicas"][idx]
            if r > 0:
                reps.append(r)
        if len(reps) >= 2:
            s = sum(reps)
            s2 = sum(r**2 for r in reps)
            n = len(reps)
            jain = (s**2) / (n * s2) if s2 > 0 else 1.0
        else:
            jain = 1.0
        jain_vals.append(jain)

    ax.plot(t, jain_vals, 'k-', linewidth=1.5)
    ax.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label="Perfect fairness")
    ax.axhline(y=1/3, color='r', linestyle=':', alpha=0.5, label="Worst (all on 1 cluster)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Jain Index")
    ax.set_xlabel("Time (min)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "dmos_08_scheduling_stats.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [8/8] Scheduling stats")


# ─── Compare ON vs OFF ──────────────────────────────────────────────────────

def plot_compare_replicas(d_on: dict, d_off: dict, out_dir: Path):
    """Comparison: frontend replicas ON vs OFF."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_HALF, sharex=False)
    fig.suptitle("DMOS ON vs OFF — Frontend Replicas per Cluster", fontsize=14, fontweight="bold")

    for idx, (label, d) in enumerate([("DMOS ON", d_on), ("DMOS OFF", d_off)]):
        ax = axes[idx]
        t = d["elapsed"] / 60
        for cl in KNOWN_CLUSTERS:
            reps = d[f"frontend_{cl}_replicas"]
            ax.step(t, reps, where='post', color=CLUSTER_COLORS[cl],
                    linewidth=1.5, label=CLUSTER_LABELS[cl])
        total = d["frontend_total_replicas"]
        ax.plot(t, total, 'k-', linewidth=1, alpha=0.5, label="Total")
        ax.set_ylabel("Replicas")
        ax.set_title(label, fontsize=11, loc="left")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out_dir / "dmos_compare_replicas.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [+] Compare replicas ON vs OFF")


def plot_compare_traffic_dist(d_on: dict, d_off: dict, out_dir: Path):
    """Comparison: traffic distribution ON vs OFF."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_HALF, sharex=False)
    fig.suptitle("DMOS ON vs OFF — Traffic Distribution", fontsize=14, fontweight="bold")

    for idx, (label, d) in enumerate([("DMOS ON", d_on), ("DMOS OFF", d_off)]):
        ax = axes[idx]
        t = d["elapsed"] / 60
        pcts = [d[f"{cl}_traffic_pct"] for cl in KNOWN_CLUSTERS]
        ax.stackplot(t, *pcts,
                     labels=[CLUSTER_LABELS[cl] for cl in KNOWN_CLUSTERS],
                     colors=[CLUSTER_COLORS[cl] for cl in KNOWN_CLUSTERS],
                     alpha=0.7)
        ax.set_ylabel("Traffic %")
        ax.set_ylim(0, 100)
        ax.set_title(label, fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out_dir / "dmos_compare_traffic_dist.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [+] Compare traffic distribution ON vs OFF")


def plot_compare_node_cpu(d_on: dict, d_off: dict, out_dir: Path):
    """Comparison: node CPU ON vs OFF."""
    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_HALF, sharex=False)
    fig.suptitle("DMOS ON vs OFF — Node CPU Usage", fontsize=14, fontweight="bold")

    for idx, (label, d) in enumerate([("DMOS ON", d_on), ("DMOS OFF", d_off)]):
        ax = axes[idx]
        t = d["elapsed"] / 60
        for cl in KNOWN_CLUSTERS:
            ncpu = d[f"{cl}_node_cpu_pct"]
            ax.plot(t, ncpu, color=CLUSTER_COLORS[cl], linewidth=1.3,
                    label=CLUSTER_LABELS[cl])
        ax.set_ylabel("Node CPU %")
        ax.set_ylim(0, 100)
        ax.set_title(label, fontsize=11, loc="left")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out_dir / "dmos_compare_node_cpu.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [+] Compare node CPU ON vs OFF")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DMOS Live Metrics Analyzer")
    parser.add_argument("--input", "-i", help="Single JSONL (DMOS ON)")
    parser.add_argument("--on", help="DMOS ON JSONL (for comparison)")
    parser.add_argument("--off", help="DMOS OFF JSONL (for comparison)")
    parser.add_argument("--output-dir", "-o", default="plots_dmos_live",
                        help="Output directory for plots")
    args = parser.parse_args()

    if not args.input and not args.on:
        parser.error("Provide --input (single) or --on/--off (comparison)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single analysis (DMOS ON)
    if args.input or args.on:
        src = args.input or args.on
        print(f"\nLoading DMOS ON data: {src}")
        snaps = load_live_jsonl(src)
        d = extract_series(snaps)

        print(f"\nGenerating plots → {out_dir}/")
        plot_01_traffic_prediction(d, out_dir)
        plot_02_replicas_per_cluster(d, out_dir)
        plot_03_cluster_scores(d, out_dir)
        plot_04_scaling_events(d, out_dir)
        plot_05_traffic_distribution(d, out_dir)
        plot_06_node_cpu(d, out_dir)
        plot_07_frontend_detail(d, out_dir)
        plot_08_scheduling_stats(d, out_dir)

    # Comparison (ON vs OFF)
    if args.on and args.off:
        print(f"\nLoading DMOS OFF data: {args.off}")
        snaps_off = load_live_jsonl(args.off)
        d_off = extract_series(snaps_off)
        d_on = d  # already loaded

        print(f"\nGenerating comparison plots...")
        plot_compare_replicas(d_on, d_off, out_dir)
        plot_compare_traffic_dist(d_on, d_off, out_dir)
        plot_compare_node_cpu(d_on, d_off, out_dir)

    print(f"\nDone! All plots saved to {out_dir}/")


if __name__ == '__main__':
    main()
