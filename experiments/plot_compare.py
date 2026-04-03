"""
plot_compare.py — DMOS ON vs OFF Comparative Analysis
======================================================
Generates a side-by-side comparison of two Locust timeseries CSVs (ON vs OFF).

Usage:
    python experiments/plot_compare.py \\
        --on  results/multiingress/flash_crowd_timeseries_ON.csv \\
        --off results/multiingress/flash_crowd_timeseries_OFF.csv \\
        --on-jsonl  results/flash_crowd_on.jsonl \\
        --label "Flash Crowd"

Output:
    results/multiingress/plots/compare_<scenario>_timeseries.png
    results/multiingress/plots/compare_<scenario>_summary.png
    results/multiingress/plots/compare_<scenario>_advanced.png
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
KNOWN_CLUSTERS = ["cluster1", "cluster2", "cluster3"]
CLUSTER_REGIONS = {"cluster1": "DE", "cluster2": "FR", "cluster3": "PL"}
COLORS = {
    "cluster1": "#1f77b4",
    "cluster2": "#ff7f0e",
    "cluster3": "#2ca02c",
}
SLO_MS = 1000
OUTPUT_DIR = Path("results/multiingress/plots")
WINDOW_SECONDS = 10   # timeseries CSV bucket size

# CO₂ config (from clusters.yaml + services.yaml)
POWER_WATTS_PER_REPLICA = 50.0   # frontend power_watts in services.yaml
GCO2_KWH = {
    "cluster1": 350,   # DE — mixed grid
    "cluster2": 80,    # FR — nuclear
    "cluster3": 650,   # PL — coal
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_timeseries(csv_path: str) -> dict:
    """Load per-cluster timeseries CSV produced by locustfile_multiingress.py."""
    data = {cn: {"timestamps": [], "p95_ms": [], "slo_pct": [], "count": [], "fail_pct": []}
            for cn in KNOWN_CLUSTERS}

    path = Path(csv_path)
    if not path.exists():
        print(f"  ⚠  File not found: {csv_path}")
        return data

    day_offset = 0
    prev_t = None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "").strip()
            if not ts_str:
                continue
            try:
                t = datetime.strptime(ts_str, "%H:%M:%S")
                if prev_t and t < prev_t:
                    day_offset += 1
                t += timedelta(days=day_offset)
                prev_t = t
            except ValueError:
                continue

            for cn in KNOWN_CLUSTERS:
                try:
                    p95  = float(row.get(f"{cn}_p95_ms",   0) or 0)
                    slo  = float(row.get(f"{cn}_slo_pct",  0) or 0)
                    cnt  = float(row.get(f"{cn}_count",    0) or 0)
                    fail = float(row.get(f"{cn}_fail_pct", 0) or 0)
                    data[cn]["timestamps"].append(t)
                    data[cn]["p95_ms"].append(p95)
                    data[cn]["slo_pct"].append(slo)
                    data[cn]["count"].append(cnt)
                    data[cn]["fail_pct"].append(fail)
                except (ValueError, KeyError):
                    pass

    return data


def load_scale_events(jsonl_path: str) -> list:
    """Extract scale-up events from JSONL collector file. Returns list of (timestamp, from, to)."""
    if not jsonl_path:
        return []
    path = Path(jsonl_path)
    if not path.exists():
        return []

    events = []
    prev_replicas = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
                ts_str = snap.get("timestamp", "")
                t = datetime.fromisoformat(ts_str)

                # Total frontend replicas
                fe = snap.get("dmos", {}).get("services", {}).get("frontend", {})
                clusters = fe.get("clusters", {})
                total = sum(
                    cl.get("current_replicas", 0)
                    for cl in clusters.values()
                )
                if prev_replicas is not None and total > prev_replicas:
                    events.append((t, prev_replicas, total))
                prev_replicas = total
            except Exception:
                pass
    return events


# ── Advanced metric helpers ───────────────────────────────────────────────────

def load_jsonl_replicas(jsonl_path: str) -> dict:
    """
    Load per-cluster frontend replica counts over time from JSONL.
    Returns: {cluster: {"timestamps": [...], "replicas": [...]}}
    """
    result = {cn: {"timestamps": [], "replicas": []} for cn in KNOWN_CLUSTERS}
    if not jsonl_path:
        return result
    path = Path(jsonl_path)
    if not path.exists():
        return result

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
                t  = datetime.fromisoformat(snap["timestamp"])
                fe = snap.get("dmos", {}).get("services", {}).get("frontend", {})
                for cn in KNOWN_CLUSTERS:
                    r = fe.get("clusters", {}).get(cn, {}).get("current_replicas", 0)
                    result[cn]["timestamps"].append(t)
                    result[cn]["replicas"].append(r)
            except Exception:
                pass
    return result


def compute_goodput(ts_data: dict) -> dict:
    """
    Per-cluster goodput per window: requests actually served within SLO.
      goodput[i] = count[i] × (100 − slo_pct[i]) / 100
    slo_pct is the fraction (0–100) of requests in the window that exceeded SLO.
    """
    result = {}
    for cn in KNOWN_CLUSTERS:
        counts  = ts_data[cn]["count"]
        slo_pct = ts_data[cn]["slo_pct"]
        result[cn] = [
            c * max(0.0, (100.0 - s) / 100.0)
            for c, s in zip(counts, slo_pct)
        ]
    return result


def compute_co2_cumulative(ts_data: dict, jsonl_replicas: dict | None = None) -> tuple:
    """
    Compute cumulative CO₂ (grams) over time.

    If jsonl_replicas provided (DMOS ON): uses actual replica counts.
    Otherwise (DMOS OFF baseline): assumes 1 replica per cluster at all times.

    Returns: (minutes_list, cumulative_co2_list, total_g)
    """
    ref_ts = ts_data[KNOWN_CLUSTERS[0]]["timestamps"]
    if not ref_ts:
        return [], [], 0.0

    t0 = ref_ts[0]
    mins  = []
    co2_c = []
    total = 0.0

    for i, t in enumerate(ref_ts):
        for cn in KNOWN_CLUSTERS:
            if jsonl_replicas:
                # Find closest JSONL sample by time
                jts = jsonl_replicas[cn]["timestamps"]
                jrp = jsonl_replicas[cn]["replicas"]
                if jts:
                    # Binary-search closest timestamp
                    idx = min(range(len(jts)), key=lambda k: abs((jts[k] - t).total_seconds()))
                    replicas = jrp[idx]
                else:
                    replicas = 1
            else:
                replicas = 1   # constant baseline (DMOS OFF, no scaling)

            dt_h = WINDOW_SECONDS / 3600.0
            total += replicas * POWER_WATTS_PER_REPLICA * dt_h * GCO2_KWH[cn] / 1000.0

        mins.append((t - t0).total_seconds() / 60.0)
        co2_c.append(total)

    return mins, co2_c, total


def _aggregate_slo_compliance(ts_data: dict, agg_n: int = 3) -> dict:
    """
    Aggregate per-window slo_pct into SLO compliance (%) over wider windows.
    agg_n=3 → 30s windows (3 × 10s each).
    Returns: {cluster: {"mins": [...], "compliance_pct": [...]}}
    """
    result = {}
    for cn in KNOWN_CLUSTERS:
        mins_raw = ts_data[cn]["_min"]
        slo      = ts_data[cn]["slo_pct"]
        if not mins_raw:
            result[cn] = {"mins": [], "compliance_pct": []}
            continue
        agg_mins, agg_comp = [], []
        for i in range(0, len(mins_raw) - agg_n + 1, agg_n):
            chunk = slo[i:i + agg_n]
            agg_comp.append(100.0 - (sum(chunk) / len(chunk)))
            agg_mins.append(mins_raw[i + len(chunk) // 2])
        result[cn] = {"mins": agg_mins, "compliance_pct": agg_comp}
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────
def _to_minutes(timestamps: list) -> list:
    """Convert list of datetime to minutes-since-start (float)."""
    if not timestamps:
        return []
    t0 = timestamps[0]
    return [(t - t0).total_seconds() / 60.0 for t in timestamps]


def _scale_event_minutes(ev_ts: datetime, ts_raw: list) -> float | None:
    """Convert a scale event timestamp to minutes relative to test start."""
    if not ts_raw:
        return None
    return (ev_ts - ts_raw[0]).total_seconds() / 60.0


def plot_compare(on_data: dict, off_data: dict,
                 scale_events_on: list,
                 label: str, output_path: Path,
                 align_time: bool = True):
    """
    3-row comparison plot using minutes-since-start on X axis.
      Row 0: p95 latency per cluster (ON solid, OFF dashed)
      Row 1: SLO violation % per cluster
      Row 2: Request throughput (count/window) per cluster
    """
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=False)
    fig.suptitle(f"DMOS ON vs OFF — {label}", fontsize=14, fontweight="bold")

    # _min already computed in main() before this call.
    # ── Row 0: p95 latency ────────────────────────────────────────────────────
    ax = axes[0]
    for cn in KNOWN_CLUSTERS:
        color  = COLORS[cn]
        region = CLUSTER_REGIONS.get(cn, "")
        min_on  = on_data[cn]["_min"]
        min_off = off_data[cn]["_min"]
        p95_on  = on_data[cn]["p95_ms"]
        p95_off = off_data[cn]["p95_ms"]
        if min_on and any(v > 0 for v in p95_on):
            ax.plot(min_on, p95_on, lw=2.2, color=color,
                    label=f"{cn} ({region}) ON")
        if min_off and any(v > 0 for v in p95_off):
            ax.plot(min_off, p95_off, lw=2.2, color=color, ls="--", alpha=0.7,
                    label=f"{cn} ({region}) OFF")

    # Scale-up annotations (DMOS ON only)
    # Use time-only comparison: extract HH:MM:SS from full ISO timestamp
    ts_raw_on = on_data[KNOWN_CLUSTERS[0]]["timestamps"]  # datetime(1900,1,1, H,M,S)
    if ts_raw_on:
        t0_raw = ts_raw_on[0]  # datetime(1900,1,1, H,M,S)
        for ev_ts, frm, to in scale_events_on:
            # Re-anchor ev_ts to same base date as timeseries (1900-01-01)
            ev_rebased = t0_raw.replace(
                hour=ev_ts.hour, minute=ev_ts.minute, second=ev_ts.second
            )
            ev_min = (ev_rebased - t0_raw).total_seconds() / 60.0
            if ev_min < 0:  # midnight rollover
                ev_min += 24 * 60
            ax.axvline(ev_min, color="purple", ls=":", lw=1.5, alpha=0.7)
            ax.text(ev_min + 0.1, 800,
                    f"↑{frm}→{to}", fontsize=7, color="purple", alpha=0.85)

    ax.axhline(SLO_MS, color="red", ls=":", lw=1.5, alpha=0.7, label="SLO 1000ms")
    ax.set_ylabel("p95 Latency (ms)", fontsize=10)
    ax.set_title("p95 Latency per Cluster — ON (solid) vs OFF (dashed)", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Minutes since test start", fontsize=9)

    # ── Row 1: SLO violation % ────────────────────────────────────────────────
    ax = axes[1]
    for cn in KNOWN_CLUSTERS:
        color  = COLORS[cn]
        region = CLUSTER_REGIONS.get(cn, "")
        if on_data[cn]["_min"] and any(v > 0 for v in on_data[cn]["slo_pct"]):
            ax.plot(on_data[cn]["_min"], on_data[cn]["slo_pct"],
                    lw=2, color=color, label=f"{cn} ({region}) ON")
        if off_data[cn]["_min"] and any(v > 0 for v in off_data[cn]["slo_pct"]):
            ax.plot(off_data[cn]["_min"], off_data[cn]["slo_pct"],
                    lw=2, color=color, ls="--", alpha=0.7,
                    label=f"{cn} ({region}) OFF")

    ax.axhline(5, color="orange", ls="--", lw=1.3, alpha=0.7, label="5% threshold")
    ax.set_ylabel("SLO Violation Rate (%)", fontsize=10)
    ax.set_title("SLO Violations (>1000ms) — ON vs OFF", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_xlabel("Minutes since test start", fontsize=9)

    # ── Row 2: Throughput ─────────────────────────────────────────────────────
    ax = axes[2]
    for cn in KNOWN_CLUSTERS:
        color  = COLORS[cn]
        region = CLUSTER_REGIONS.get(cn, "")
        if on_data[cn]["_min"] and any(v > 0 for v in on_data[cn]["count"]):
            ax.plot(on_data[cn]["_min"], on_data[cn]["count"],
                    lw=2, color=color, label=f"{cn} ({region}) ON")
        if off_data[cn]["_min"] and any(v > 0 for v in off_data[cn]["count"]):
            ax.plot(off_data[cn]["_min"], off_data[cn]["count"],
                    lw=2, color=color, ls="--", alpha=0.7,
                    label=f"{cn} ({region}) OFF")

    ax.set_ylabel("Requests / 10s window", fontsize=10)
    ax.set_title("Request Throughput per Cluster — ON vs OFF", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_xlabel("Minutes since test start", fontsize=9)

    # ── Row 3: Failure rate % ─────────────────────────────────────────────────
    ax = axes[3]
    for cn in KNOWN_CLUSTERS:
        color  = COLORS[cn]
        region = CLUSTER_REGIONS.get(cn, "")
        fail_on  = on_data[cn].get("fail_pct", [])
        fail_off = off_data[cn].get("fail_pct", [])
        if on_data[cn]["_min"] and any(v > 0 for v in fail_on):
            ax.plot(on_data[cn]["_min"], fail_on, lw=2, color=color,
                    label=f"{cn} ({region}) ON")
        if off_data[cn]["_min"] and any(v > 0 for v in fail_off):
            ax.plot(off_data[cn]["_min"], fail_off, lw=2, color=color, ls="--", alpha=0.7,
                    label=f"{cn} ({region}) OFF")

    ax.set_ylabel("Failure Rate (%)", fontsize=10)
    ax.set_title("Request Failure Rate per Cluster — ON vs OFF\n"
                 "(includes ConnectTimeout=10s → visible pod crashes/restarts)", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_xlabel("Minutes since test start", fontsize=9)

    # ── Legend patches ────────────────────────────────────────────────────────
    on_patch  = mpatches.Patch(color="gray",        label="— DMOS ON")
    off_patch = mpatches.Patch(color="gray", alpha=0.45, label="-- DMOS OFF")
    fig.legend(handles=[on_patch, off_patch], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {output_path}")


def plot_summary_bars(on_data_flat: dict, off_data_flat: dict,
                      label: str, output_path: Path):
    """
    Bar chart summary: p95 and SLO violations per cluster, ON vs OFF.
    on_data_flat / off_data_flat: {cluster: {p95_ms, slo_pct, peak_p95, slo_duration_s}}
    """
    clusters = KNOWN_CLUSTERS
    regions  = [CLUSTER_REGIONS.get(cn, "") for cn in clusters]
    x = np.arange(len(clusters))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"DMOS ON vs OFF Summary — {label}", fontsize=13, fontweight="bold")

    metrics = [
        ("p95_ms",          "p95 Latency (ms)",          "p95 Latency per Cluster"),
        ("peak_p95_ms",     "Peak p95 Latency (ms)",     "Peak Spike p95 per Cluster"),
        ("slo_duration_s",  "SLO Violation Duration (s)","SLO Violation Duration per Cluster"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        vals_on  = [on_data_flat.get(cn, {}).get(key, 0) or 0  for cn in clusters]
        vals_off = [off_data_flat.get(cn, {}).get(key, 0) or 0 for cn in clusters]

        bars_on  = ax.bar(x - width/2, vals_on,  width, label="DMOS ON",
                          color=[COLORS[cn] for cn in clusters], alpha=0.9)
        bars_off = ax.bar(x + width/2, vals_off, width, label="DMOS OFF",
                          color=[COLORS[cn] for cn in clusters], alpha=0.45,
                          edgecolor="gray", hatch="//")

        # Value labels
        for bar in bars_on:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + max(vals_on + vals_off)*0.01,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=8)
        for bar in bars_off:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + max(vals_on + vals_off)*0.01,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=8)

        if key == "p95_ms":
            ax.axhline(SLO_MS, color="red", ls=":", lw=1.5, alpha=0.7, label="SLO")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{cn}\n({r})" for cn, r in zip(clusters, regions)], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {output_path}")


def plot_advanced_metrics(on_data: dict, off_data: dict,
                          on_jsonl_replicas: dict,
                          label: str, output_path: Path):
    """
    2 × 2 advanced comparison figure:
      [0,0] Goodput (req within SLO per 10s) — ON vs OFF per cluster
      [0,1] SLO compliance % (30s windows)   — ON vs OFF per cluster
      [1,0] Traffic distribution pie          — ON (left) vs OFF (right)
      [1,1] Cumulative CO₂ (g)               — ON vs OFF
    Requires on_data / off_data to already have "_min" lists computed.
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"DMOS ON vs OFF — Advanced Metrics — {label}",
                 fontsize=13, fontweight="bold")

    ax_gp  = fig.add_subplot(2, 2, 1)
    ax_slo = fig.add_subplot(2, 2, 2)
    ax_pie = fig.add_subplot(2, 2, 3)
    ax_co2 = fig.add_subplot(2, 2, 4)

    on_goodput  = compute_goodput(on_data)
    off_goodput = compute_goodput(off_data)

    # ── [0,0] Goodput timeline ────────────────────────────────────────────────
    for cn in KNOWN_CLUSTERS:
        color  = COLORS[cn]
        region = CLUSTER_REGIONS.get(cn, "")
        if on_data[cn]["_min"] and any(v > 0 for v in on_goodput[cn]):
            ax_gp.plot(on_data[cn]["_min"], on_goodput[cn],
                       lw=2, color=color, label=f"{cn} ({region}) ON")
        if off_data[cn]["_min"] and any(v > 0 for v in off_goodput[cn]):
            ax_gp.plot(off_data[cn]["_min"], off_goodput[cn],
                       lw=2, color=color, ls="--", alpha=0.65,
                       label=f"{cn} ({region}) OFF")

    # Total goodput summary
    total_gp_on  = sum(sum(on_goodput[cn])  for cn in KNOWN_CLUSTERS)
    total_gp_off = sum(sum(off_goodput[cn]) for cn in KNOWN_CLUSTERS)
    delta_pct = ((total_gp_on - total_gp_off) / total_gp_off * 100
                 if total_gp_off > 0 else 0)
    sign = "+" if delta_pct >= 0 else ""
    ax_gp.text(0.97, 0.97,
               f"Total ON:  {total_gp_on:.0f} req\nTotal OFF: {total_gp_off:.0f} req\n"
               f"DMOS advantage: {sign}{delta_pct:.0f}%",
               transform=ax_gp.transAxes, ha="right", va="top", fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))
    ax_gp.set_title("Goodput (Requests Served Within SLO per 10s)", fontsize=10)
    ax_gp.set_ylabel("Good Requests / window", fontsize=9)
    ax_gp.set_xlabel("Minutes since start", fontsize=9)
    ax_gp.set_ylim(bottom=0)
    ax_gp.legend(fontsize=7, ncol=2)

    # ── [0,1] SLO compliance % (30s windows) ─────────────────────────────────
    on_comp  = _aggregate_slo_compliance(on_data,  agg_n=3)
    off_comp = _aggregate_slo_compliance(off_data, agg_n=3)
    for cn in KNOWN_CLUSTERS:
        color  = COLORS[cn]
        region = CLUSTER_REGIONS.get(cn, "")
        if on_comp[cn]["mins"]:
            ax_slo.plot(on_comp[cn]["mins"], on_comp[cn]["compliance_pct"],
                        lw=2, color=color, label=f"{cn} ({region}) ON")
        if off_comp[cn]["mins"]:
            ax_slo.plot(off_comp[cn]["mins"], off_comp[cn]["compliance_pct"],
                        lw=2, color=color, ls="--", alpha=0.65,
                        label=f"{cn} ({region}) OFF")
    ax_slo.axhline(95, color="orange", ls="--", lw=1.2, alpha=0.7, label="95% target")
    ax_slo.set_title("SLO Compliance % (30s windows)", fontsize=10)
    ax_slo.set_ylabel("Requests within SLO (%)", fontsize=9)
    ax_slo.set_xlabel("Minutes since start", fontsize=9)
    ax_slo.set_ylim(0, 105)
    ax_slo.legend(fontsize=7, ncol=2)

    # ── [1,0] Traffic distribution pies ──────────────────────────────────────
    ax_pie.axis("off")
    ax_pie.set_title("Traffic Distribution per Cluster (total requests)",
                     fontsize=10, pad=16)

    pie_colors = [COLORS[cn] for cn in KNOWN_CLUSTERS]
    pie_labels = [f"{cn}\n({CLUSTER_REGIONS.get(cn, '')})" for cn in KNOWN_CLUSTERS]
    on_totals  = [sum(on_data[cn]["count"])  for cn in KNOWN_CLUSTERS]
    off_totals = [sum(off_data[cn]["count"]) for cn in KNOWN_CLUSTERS]

    ax_l = ax_pie.inset_axes([0.01, 0.05, 0.44, 0.88])
    ax_r = ax_pie.inset_axes([0.55, 0.05, 0.44, 0.88])

    def _draw_pie(ax, totals, title):
        s = sum(totals)
        if s > 0:
            ax.pie(totals, labels=pie_labels, colors=pie_colors,
                   autopct="%1.0f%%", startangle=90,
                   textprops={"fontsize": 8}, pctdistance=0.75)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
        ax.set_title(title, fontsize=9, pad=4)

    _draw_pie(ax_l, on_totals,  "DMOS ON")
    _draw_pie(ax_r, off_totals, "DMOS OFF")

    # ── [1,1] Cumulative CO₂ ─────────────────────────────────────────────────
    mins_on,  co2_on,  total_on  = compute_co2_cumulative(on_data,  on_jsonl_replicas)
    mins_off, co2_off, total_off = compute_co2_cumulative(off_data, None)  # 1 replica/cluster

    if mins_on:
        ax_co2.plot(mins_on,  co2_on,  color="#2c7bb6", lw=2.5, label="DMOS ON")
    if mins_off:
        ax_co2.plot(mins_off, co2_off, color="#d7191c", lw=2.5, ls="--", label="DMOS OFF")

    if total_on > 0 and total_off > 0:
        # CO2 per good request — efficiency metric
        eff_on  = total_on  / total_gp_on  if total_gp_on  > 0 else 0
        eff_off = total_off / total_gp_off if total_gp_off > 0 else 0
        savings = total_off - total_on
        savings_pct = savings / total_off * 100 if total_off > 0 else 0
        ax_co2.text(
            0.03, 0.97,
            f"Total ON:  {total_on:.1f}g  ({eff_on:.3f} g/good-req)\n"
            f"Total OFF: {total_off:.1f}g  ({eff_off:.3f} g/good-req)\n"
            f"Δ absolute: {savings:+.1f}g ({savings_pct:+.0f}%)",
            transform=ax_co2.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85)
        )

    ax_co2.set_title("Cumulative CO₂ Equivalent (g)\n"
                     "[OFF assumes 1 replica/cluster; ON uses JSONL replica counts]",
                     fontsize=9)
    ax_co2.set_ylabel("CO₂ (g)", fontsize=9)
    ax_co2.set_xlabel("Minutes since start", fontsize=9)
    ax_co2.set_ylim(bottom=0)
    ax_co2.legend(fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {output_path}")


# ── Advanced metrics from timeseries ─────────────────────────────────────────
def _compute_flat_metrics(ts_data: dict, window_seconds: int = 10) -> dict:
    """Compute p95 (mean), peak p95, SLO duration from timeseries data."""
    result = {}
    for cn in KNOWN_CLUSTERS:
        p95_vals = ts_data[cn]["p95_ms"]
        slo_vals = ts_data[cn]["slo_pct"]
        active   = [v for v in p95_vals if v > 0]
        if not active:
            result[cn] = {}
            continue

        peak_p95     = max(active)
        mean_p95     = sum(active) / len(active)
        slo_duration = sum(1 for v in p95_vals if v > SLO_MS) * window_seconds
        mean_slo     = sum(slo_vals) / len(slo_vals) if slo_vals else 0

        # Baseline (first 20%)
        n_warm = max(1, len(active) // 5)
        baseline = sum(active[:n_warm]) / n_warm

        # Recovery time
        peak_idx = p95_vals.index(peak_p95)
        recovery_s = None
        for i in range(peak_idx + 1, len(p95_vals)):
            if 0 < p95_vals[i] <= baseline * 2:
                recovery_s = (i - peak_idx) * window_seconds
                break

        result[cn] = {
            "p95_ms":          mean_p95,
            "peak_p95_ms":     peak_p95,
            "slo_duration_s":  slo_duration,
            "slo_pct_mean":    mean_slo,
            "baseline_p95_ms": baseline,
            "recovery_s":      recovery_s,
        }
    return result


# ── CLI Entry Point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare DMOS ON vs OFF from Locust timeseries CSVs"
    )
    parser.add_argument("--on",  required=True, help="Timeseries CSV for DMOS ON run")
    parser.add_argument("--off", required=True, help="Timeseries CSV for DMOS OFF run")
    parser.add_argument("--on-jsonl",  default=None, help="Collector JSONL for DMOS ON (for scale-up annotations)")
    parser.add_argument("--label",     default="Scenario", help="Scenario label (e.g. 'Flash Crowd')")
    parser.add_argument("--scenario",  default=None, help="Scenario name for output filename")
    parser.add_argument("--no-align",  action="store_true", help="Don't align timelines (use real timestamps)")
    args = parser.parse_args()

    scenario_slug = (args.scenario or args.label).lower().replace(" ", "_")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading ON  timeseries: {args.on}")
    on_data  = load_timeseries(args.on)
    print(f"  Loading OFF timeseries: {args.off}")
    off_data = load_timeseries(args.off)

    # Pre-compute _min for all consumers (plot_compare + plot_advanced_metrics)
    for cn in KNOWN_CLUSTERS:
        on_data[cn]["_min"]  = _to_minutes(on_data[cn]["timestamps"])
        off_data[cn]["_min"] = _to_minutes(off_data[cn]["timestamps"])

    scale_events    = []
    jsonl_replicas  = {}
    if args.on_jsonl:
        print(f"  Loading scale events:   {args.on_jsonl}")
        scale_events   = load_scale_events(args.on_jsonl)
        jsonl_replicas = load_jsonl_replicas(args.on_jsonl)
        print(f"  Found {len(scale_events)} scale-up events")

    # Timeseries comparison plot
    out_ts = OUTPUT_DIR / f"compare_{scenario_slug}_timeseries.png"
    print(f"\n  Generating timeseries comparison...")
    plot_compare(on_data, off_data, scale_events,
                 label=args.label,
                 output_path=out_ts,
                 align_time=not args.no_align)

    # Summary bar chart
    print(f"  Generating summary bar chart...")
    metrics_on  = _compute_flat_metrics(on_data)
    metrics_off = _compute_flat_metrics(off_data)
    out_bar = OUTPUT_DIR / f"compare_{scenario_slug}_summary.png"
    plot_summary_bars(metrics_on, metrics_off,
                      label=args.label,
                      output_path=out_bar)

    # Advanced metrics plot
    print(f"  Generating advanced metrics plot...")
    out_adv = OUTPUT_DIR / f"compare_{scenario_slug}_advanced.png"
    plot_advanced_metrics(on_data, off_data, jsonl_replicas,
                          label=args.label,
                          output_path=out_adv)

    # ── Console summary ───────────────────────────────────────────────────────
    on_gp  = compute_goodput(on_data)
    off_gp = compute_goodput(off_data)
    total_gp_on  = sum(sum(on_gp[cn])  for cn in KNOWN_CLUSTERS)
    total_gp_off = sum(sum(off_gp[cn]) for cn in KNOWN_CLUSTERS)
    gp_delta = ((total_gp_on - total_gp_off) / total_gp_off * 100
                if total_gp_off > 0 else 0)

    _, co2_on_v,  co2_on_total  = compute_co2_cumulative(on_data,  jsonl_replicas)
    _, co2_off_v, co2_off_total = compute_co2_cumulative(off_data, None)
    co2_savings_pct = ((co2_off_total - co2_on_total) / co2_off_total * 100
                       if co2_off_total > 0 else 0)

    print(f"\n{'═'*80}")
    print(f"  COMPARISON SUMMARY — {args.label}")
    print(f"{'═'*80}")
    print(f"  {'Cluster':<12} {'ON p95':>8}  {'OFF p95':>8}  {'Δ p95':>8}  "
          f"{'ON peak':>8}  {'OFF peak':>8}  {'ON SLO':>8}  {'OFF SLO':>8}")
    print(f"  {'─'*80}")
    for cn in KNOWN_CLUSTERS:
        mo = metrics_on.get(cn, {})
        mf = metrics_off.get(cn, {})
        p95_on  = mo.get("p95_ms")
        p95_off = mf.get("p95_ms")
        delta   = ((p95_off - p95_on) / p95_on * 100
                   if (p95_on and p95_off and p95_on > 0) else None)
        delta_s = (f"{delta:+.0f}%" if delta is not None else "N/A")
        print(f"  {cn:<12} "
              f"{(f'{p95_on:.0f}ms' if p95_on else 'N/A'):>8}  "
              f"{(f'{p95_off:.0f}ms' if p95_off else 'N/A'):>8}  "
              f"{delta_s:>8}  "
              f"{mo.get('peak_p95_ms', 0):.0f}ms  "
              f"{mf.get('peak_p95_ms', 0):.0f}ms  "
              f"{mo.get('slo_duration_s', 0):.0f}s  "
              f"{mf.get('slo_duration_s', 0):.0f}s")
    print(f"  {'─'*80}")
    sign_gp = "+" if gp_delta >= 0 else ""
    print(f"  Goodput total  ON={total_gp_on:.0f}  OFF={total_gp_off:.0f}  "
          f"Δ={sign_gp}{gp_delta:.1f}%")
    print(f"  CO₂ total      ON={co2_on_total:.1f}g  OFF={co2_off_total:.1f}g  "
          f"Δ={co2_off_total - co2_on_total:+.1f}g ({co2_savings_pct:+.0f}%)")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()
