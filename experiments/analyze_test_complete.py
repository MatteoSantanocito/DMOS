"""
DMOS Comprehensive Test Analysis v2
====================================
Analyzes JSONL files produced by collect_metrics_simple.py v2.
Also supports legacy .txt files for backward compatibility.

Metrics computed:
  1. Traffic Analysis (actual, predicted, per-service)
  2. Replica Distribution (per cluster, per service, total)
  3. Scaling Activity (events, transitions, oscillation/flapping)
  4. Prediction Accuracy (MAPE, RMSE, R², directional accuracy)
  5. Resource Efficiency (utilization, over/under-provisioning ratio timeline)
  6. Time to Scale (TtS) — proactive vs reactive scaling
  7. Cluster Fairness (Jain Index over time)
  8. Response Time Correlation (Locust p95 vs replicas/traffic)
  9. Multi-Objective Score Evolution
 10. Backend Services Impact Analysis
 11. Scheduling Duration Analysis

Outputs:
  - results/plots/dmos_page1_core.png      (traffic, replicas, correlation, distribution)
  - results/plots/dmos_page2_scaling.png    (TtS, oscillation, over/under-prov, events)
  - results/plots/dmos_page3_quality.png    (prediction scatter, fairness, scores, scheduling)
  - results/plots/dmos_page4_response.png   (Locust RT, RT vs replicas, backend, summary)
  - results/<file>.analysis.json            (full statistics JSON)
  - Console report with all KPIs
"""

import re
import json
import math
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

KNOWN_SERVICES = [
    "frontend", "cartservice", "productcatalogservice",
    "checkoutservice", "recommendationservice"
]
KNOWN_CLUSTERS = ["cluster1", "cluster2", "cluster3"]

# From config/services.yaml
SERVICE_CAPACITY = {
    "frontend": 50,
    "cartservice": 100,
    "productcatalogservice": 80,
    "checkoutservice": 30,
    "recommendationservice": 60,
}
SERVICE_MIN_REPLICAS = {
    "frontend": 2,
    "cartservice": 1,
    "productcatalogservice": 1,
    "checkoutservice": 1,
    "recommendationservice": 1,
}

HIGH_THRESHOLD = 30  # req/s
LOW_THRESHOLD = 10   # req/s

# Plot style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'legend.fontsize': 8,
})
COLORS = {
    'cluster1': '#1f77b4',
    'cluster2': '#ff7f0e',
    'cluster3': '#2ca02c',
    'traffic': '#1a5276',
    'predicted': '#27ae60',
    'replicas': '#c0392b',
    'p95': '#8e44ad',
    'overprov': '#e67e22',
    'underprov': '#e74c3c',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(filepath: str) -> List[dict]:
    """Load structured JSONL data from v2 collector."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    obj["_ts"] = datetime.fromisoformat(obj["timestamp"])
                    data.append(obj)
                except (json.JSONDecodeError, KeyError):
                    continue
    return data


def load_legacy_txt(filepath: str) -> List[dict]:
    """
    Parse legacy .txt metrics file into same structure as JSONL.
    Only supports frontend metrics (original format).
    """
    with open(filepath, 'r') as f:
        content = f.read()

    pattern = r'Timestamp: ([\d\-T:\.]+)\s*={70,}\s*(.*?)(?=Timestamp:|$)'
    matches = re.findall(pattern, content, re.DOTALL)

    data = []
    for ts_str, block in matches:
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            continue

        snap = {
            "timestamp": ts_str,
            "_ts": ts,
            "dmos": {"services": {}},
            "locust": {"available": False},
        }

        # Parse frontend metrics from raw prometheus text
        fe = {
            "actual_traffic": 0.0,
            "predicted_traffic_total": 0.0,
            "total_replicas": 0,
            "clusters": {},
            "avg_scheduling_duration_s": 0.0,
            "scheduling_invocations": 0,
        }
        scale_up_total = 0
        scale_down_total = 0

        for line in block.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                if 'dmos_actual_traffic{service="frontend"}' in line:
                    m = re.search(r'([\d\.eE\+\-]+)$', line)
                    if m: fe["actual_traffic"] = float(m.group(1))

                elif 'dmos_current_replicas{' in line and 'service="frontend"' in line:
                    mc = re.search(r'cluster="([^"]+)"', line)
                    v = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                    if mc and v:
                        c = mc.group(1)
                        reps = int(float(v.group(1)))
                        if c not in fe["clusters"]:
                            fe["clusters"][c] = {}
                        fe["clusters"][c]["current_replicas"] = reps

                elif 'dmos_target_replicas{' in line and 'service="frontend"' in line:
                    mc = re.search(r'cluster="([^"]+)"', line)
                    v = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                    if mc and v:
                        c = mc.group(1)
                        if c not in fe["clusters"]:
                            fe["clusters"][c] = {}
                        fe["clusters"][c]["target_replicas"] = int(float(v.group(1)))

                elif 'dmos_cluster_score{' in line and 'service="frontend"' in line:
                    mc = re.search(r'cluster="([^"]+)"', line)
                    v = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                    if mc and v:
                        c = mc.group(1)
                        if c not in fe["clusters"]:
                            fe["clusters"][c] = {}
                        fe["clusters"][c]["score"] = float(v.group(1))

                elif 'dmos_predicted_traffic{' in line and 'service="frontend"' in line:
                    mc = re.search(r'cluster="([^"]+)"', line)
                    v = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                    if mc and v:
                        c = mc.group(1)
                        if c not in fe["clusters"]:
                            fe["clusters"][c] = {}
                        fe["clusters"][c]["predicted_traffic"] = float(v.group(1))
                        fe["predicted_traffic_total"] += float(v.group(1))

                elif 'dmos_scaling_events_total{' in line and 'service="frontend"' in line:
                    ma = re.search(r'action="([^"]+)"', line)
                    v = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                    if ma and v:
                        if ma.group(1) == "scale_up":
                            scale_up_total += int(float(v.group(1)))
                        else:
                            scale_down_total += int(float(v.group(1)))

            except Exception:
                continue

        # Fill in missing cluster data
        for c in KNOWN_CLUSTERS:
            if c not in fe["clusters"]:
                fe["clusters"][c] = {
                    "current_replicas": 0, "target_replicas": 0,
                    "score": 0.0, "predicted_traffic": 0.0,
                    "scale_up_events_cumulative": 0,
                    "scale_down_events_cumulative": 0,
                }
            else:
                cd = fe["clusters"][c]
                cd.setdefault("current_replicas", 0)
                cd.setdefault("target_replicas", 0)
                cd.setdefault("score", 0.0)
                cd.setdefault("predicted_traffic", 0.0)
                cd.setdefault("scale_up_events_cumulative", 0)
                cd.setdefault("scale_down_events_cumulative", 0)

        fe["total_replicas"] = sum(
            fe["clusters"][c].get("current_replicas", 0) for c in KNOWN_CLUSTERS
        )

        snap["dmos"]["services"]["frontend"] = fe
        data.append(snap)

    return data


def load_data(filepath: str) -> List[dict]:
    """Auto-detect format and load data."""
    if filepath.endswith('.jsonl'):
        data = load_jsonl(filepath)
        print(f"📂 Loaded {len(data)} snapshots from JSONL")
    else:
        data = load_legacy_txt(filepath)
        print(f"📂 Loaded {len(data)} snapshots from legacy TXT")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: extract timeseries for a specific service
# ═══════════════════════════════════════════════════════════════════════════════

def extract_service_ts(data: List[dict], svc: str) -> dict:
    """Extract time series for a given service."""
    ts = {
        "timestamps": [],
        "traffic": [],
        "predicted": [],
        "total_replicas": [],
        "per_cluster_replicas": {c: [] for c in KNOWN_CLUSTERS},
        "per_cluster_scores": {c: [] for c in KNOWN_CLUSTERS},
        "per_cluster_target": {c: [] for c in KNOWN_CLUSTERS},
    }
    for snap in data:
        svc_data = snap["dmos"]["services"].get(svc)
        if svc_data is None:
            continue
        ts["timestamps"].append(snap["_ts"])
        ts["traffic"].append(svc_data.get("actual_traffic", 0))
        ts["predicted"].append(svc_data.get("predicted_traffic_total", 0))
        ts["total_replicas"].append(svc_data.get("total_replicas", 0))
        for c in KNOWN_CLUSTERS:
            cd = svc_data.get("clusters", {}).get(c, {})
            ts["per_cluster_replicas"][c].append(cd.get("current_replicas", 0))
            ts["per_cluster_scores"][c].append(cd.get("score", 0.0))
            ts["per_cluster_target"][c].append(cd.get("target_replicas", 0))
    return ts


def extract_locust_ts(data: List[dict]) -> dict:
    """Extract Locust response time timeseries."""
    ts = {
        "timestamps": [],
        "avg_rt": [],
        "median_rt": [],
        "p95_rt": [],
        "p99_rt": [],
        "rps": [],
        "fail_ratio": [],
        "users": [],
    }
    for snap in data:
        loc = snap.get("locust", {})
        if not loc.get("available", False):
            continue
        ts["timestamps"].append(snap["_ts"])
        ts["avg_rt"].append(loc.get("avg_response_time_ms", 0))
        ts["median_rt"].append(loc.get("median_response_time_ms", 0))
        ts["p95_rt"].append(loc.get("p95_response_time_ms", 0))
        ts["p99_rt"].append(loc.get("p99_response_time_ms", 0))
        ts["rps"].append(loc.get("total_rps", 0))
        ts["fail_ratio"].append(loc.get("total_fail_ratio", 0))
        ts["users"].append(loc.get("current_users", 0))
    return ts


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Computations
# ═══════════════════════════════════════════════════════════════════════════════

def compute_prediction_accuracy(traffic: list, predicted: list) -> dict:
    """Compute MAPE, RMSE, R², directional accuracy."""
    pairs = [(a, p) for a, p in zip(traffic, predicted) if a > 1 and p > 0]
    if len(pairs) < 3:
        return {"mape": None, "rmse": None, "r2": None, "directional_accuracy": None, "n": len(pairs)}

    actuals = [a for a, _ in pairs]
    preds = [p for _, p in pairs]

    # MAPE
    ape = [abs(p - a) / a * 100 for a, p in pairs]
    mape = sum(ape) / len(ape)

    # RMSE
    sq_errors = [(p - a) ** 2 for a, p in pairs]
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

    # R²
    mean_a = sum(actuals) / len(actuals)
    ss_res = sum((a - p) ** 2 for a, p in pairs)
    ss_tot = sum((a - mean_a) ** 2 for a in actuals)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy (did prediction go in the same direction as actual?)
    dir_correct = 0
    for i in range(1, len(pairs)):
        a_dir = pairs[i][0] - pairs[i - 1][0]
        p_dir = pairs[i][1] - pairs[i - 1][1]
        if (a_dir >= 0 and p_dir >= 0) or (a_dir < 0 and p_dir < 0):
            dir_correct += 1
    dir_acc = dir_correct / (len(pairs) - 1) * 100 if len(pairs) > 1 else 0

    return {"mape": mape, "rmse": rmse, "r2": r2, "directional_accuracy": dir_acc, "n": len(pairs)}


def compute_provisioning_ratio(traffic: list, replicas: list, capacity_per_replica: float) -> dict:
    """
    Compute over/under provisioning ratio per snapshot.
    ratio = provisioned_capacity / actual_traffic
    > 1 = over-provisioned, < 1 = under-provisioned, ~1.15 = ideal
    """
    ratios = []
    over_count = 0
    under_count = 0
    for t, r in zip(traffic, replicas):
        if t > 0 and r > 0:
            cap = r * capacity_per_replica
            ratio = cap / t
            ratios.append(ratio)
            if ratio > 1.5:
                over_count += 1
            elif ratio < 1.0:
                under_count += 1

    if not ratios:
        return {"ratios": [], "mean": 0, "over_pct": 0, "under_pct": 0}

    return {
        "ratios": ratios,
        "mean": sum(ratios) / len(ratios),
        "median": sorted(ratios)[len(ratios) // 2],
        "over_pct": over_count / len(ratios) * 100,
        "under_pct": under_count / len(ratios) * 100,
        "ideal_pct": (len(ratios) - over_count - under_count) / len(ratios) * 100,
    }


def compute_time_to_scale(timestamps: list, traffic: list, replicas: list) -> dict:
    """
    Compute Time to Scale (TtS).
    For each traffic threshold crossing, find when replicas actually changed.
    Negative TtS = proactive (scaled before traffic spike) — the DMOS value proposition.
    Positive TtS = reactive.
    """
    tts_events = []

    for i in range(1, len(traffic)):
        # Detect threshold crossing: traffic crosses HIGH_THRESHOLD upward
        if traffic[i - 1] <= HIGH_THRESHOLD < traffic[i]:
            cross_time = timestamps[i]

            # Find the nearest previous or next replica increase
            # Search backward: did replicas increase BEFORE the crossing?
            best_scale_time = None
            best_delta = None

            # Look backward up to 10 samples (~150s)
            for j in range(max(0, i - 10), i):
                if replicas[j] < replicas[min(j + 1, len(replicas) - 1)]:
                    best_scale_time = timestamps[j + 1] if j + 1 < len(timestamps) else timestamps[j]
                    break

            # If not found backward, look forward
            if best_scale_time is None:
                for j in range(i, min(len(replicas) - 1, i + 10)):
                    if replicas[j] < replicas[j + 1]:
                        best_scale_time = timestamps[j + 1]
                        break

            if best_scale_time is not None:
                tts_seconds = (best_scale_time - cross_time).total_seconds()
                tts_events.append({
                    "cross_time": cross_time,
                    "scale_time": best_scale_time,
                    "tts_seconds": tts_seconds,
                    "proactive": tts_seconds < 0,
                })

    proactive = [e for e in tts_events if e["proactive"]]
    reactive = [e for e in tts_events if not e["proactive"]]

    return {
        "events": tts_events,
        "count": len(tts_events),
        "proactive_count": len(proactive),
        "reactive_count": len(reactive),
        "proactive_pct": len(proactive) / len(tts_events) * 100 if tts_events else 0,
        "avg_tts": sum(e["tts_seconds"] for e in tts_events) / len(tts_events) if tts_events else 0,
        "avg_proactive_tts": sum(e["tts_seconds"] for e in proactive) / len(proactive) if proactive else 0,
        "avg_reactive_tts": sum(e["tts_seconds"] for e in reactive) / len(reactive) if reactive else 0,
    }


def compute_scaling_oscillation(replicas: list, timestamps: list, window_samples: int = 20) -> dict:
    """
    Detect scaling oscillation (flapping).
    Count direction changes (up→down or down→up) in sliding windows.
    High oscillation = unstable system.
    """
    if len(replicas) < 3:
        return {"direction_changes": 0, "max_changes_in_window": 0, "flapping_windows": 0}

    # Compute direction changes
    directions = []
    for i in range(1, len(replicas)):
        if replicas[i] > replicas[i - 1]:
            directions.append(1)   # up
        elif replicas[i] < replicas[i - 1]:
            directions.append(-1)  # down
        else:
            directions.append(0)   # stable

    # Count direction reversals
    total_changes = 0
    for i in range(1, len(directions)):
        if directions[i] != 0 and directions[i - 1] != 0 and directions[i] != directions[i - 1]:
            total_changes += 1

    # Sliding window analysis
    max_window_changes = 0
    flapping_windows = 0
    window_changes_list = []
    for start in range(0, len(directions) - window_samples + 1):
        window = directions[start:start + window_samples]
        changes = 0
        for j in range(1, len(window)):
            if window[j] != 0 and window[j - 1] != 0 and window[j] != window[j - 1]:
                changes += 1
        window_changes_list.append(changes)
        max_window_changes = max(max_window_changes, changes)
        if changes >= 3:  # 3+ reversals in 5min window = flapping
            flapping_windows += 1

    return {
        "direction_changes": total_changes,
        "max_changes_in_window": max_window_changes,
        "flapping_windows": flapping_windows,
        "window_changes_timeline": window_changes_list,
    }


def compute_jain_fairness(data: List[dict], svc: str = "frontend") -> Tuple[list, list]:
    """
    Compute Jain Fairness Index over time for replica distribution across clusters.
    J(x) = (Σx_i)² / (n * Σx_i²)   where x_i = replicas on cluster i.
    J=1 means perfectly fair, J=1/n means all on one cluster.
    """
    timestamps = []
    jain_values = []

    for snap in data:
        svc_data = snap["dmos"]["services"].get(svc)
        if svc_data is None:
            continue

        reps = []
        for c in KNOWN_CLUSTERS:
            cd = svc_data.get("clusters", {}).get(c, {})
            reps.append(cd.get("current_replicas", 0))

        total = sum(reps)
        if total > 0:
            sum_sq = sum(x ** 2 for x in reps)
            n = len([r for r in reps if r > 0])  # only count active clusters
            if n > 0 and sum_sq > 0:
                jain = (total ** 2) / (n * sum_sq)
            else:
                jain = 0.0
        else:
            jain = 0.0

        timestamps.append(snap["_ts"])
        jain_values.append(jain)

    return timestamps, jain_values


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Report (Console)
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(data: List[dict], fe_ts: dict, locust_ts: dict, stats: dict):
    """Print comprehensive statistical report to console."""
    W = 78

    def header(title):
        print(f"\n{'═' * W}")
        print(f"  {title}")
        print(f"{'═' * W}")

    def kv(label, value, indent=4):
        print(f"{' ' * indent}{label:<36s} {value}")

    header("📊 DMOS COMPREHENSIVE TEST REPORT")

    # ── Test Info ──
    header("1. TEST INFO")
    dur = (data[-1]["_ts"] - data[0]["_ts"]).total_seconds()
    kv("Start:", data[0]["timestamp"][:19])
    kv("End:", data[-1]["timestamp"][:19])
    kv("Duration:", f"{dur / 60:.1f} minutes ({dur:.0f}s)")
    kv("Snapshots:", f"{len(data)} (every ~15s)")
    locust_available = any(s.get("locust", {}).get("available") for s in data)
    kv("Locust data:", "✅ Available" if locust_available else "❌ Not available")

    # ── Traffic ──
    header("2. TRAFFIC ANALYSIS (frontend)")
    t = [v for v in fe_ts["traffic"] if v > 0]
    if t:
        kv("Samples with traffic:", f"{len(t)}/{len(fe_ts['traffic'])}")
        kv("Min:", f"{min(t):.2f} req/s")
        kv("Max:", f"{max(t):.2f} req/s")
        kv("Mean:", f"{sum(t)/len(t):.2f} req/s")
        st = sorted(t)
        kv("Median (p50):", f"{st[len(st)//2]:.2f} req/s")
        kv("p95:", f"{st[int(len(st)*0.95)]:.2f} req/s")
        kv("p99:", f"{st[int(len(st)*0.99)]:.2f} req/s")

    # ── Replicas ──
    header("3. REPLICA DISTRIBUTION")
    reps = fe_ts["total_replicas"]
    if any(r > 0 for r in reps):
        kv("Total — Min:", str(min(reps)))
        kv("Total — Max:", str(max(reps)))
        kv("Total — Mean:", f"{sum(reps)/len(reps):.1f}")
        for c in KNOWN_CLUSTERS:
            cr = [v for v in fe_ts["per_cluster_replicas"][c] if v > 0]
            if cr:
                kv(f"{c}:", f"avg={sum(cr)/len(cr):.1f}, max={max(cr)}, active {len(cr)}/{len(reps)} snapshots")
            else:
                kv(f"{c}:", "never active")

    # ── Scaling Activity ──
    header("4. SCALING ACTIVITY")
    transitions = []
    for i in range(1, len(reps)):
        if reps[i] != reps[i - 1]:
            transitions.append({
                "time": fe_ts["timestamps"][i],
                "from": reps[i - 1],
                "to": reps[i],
                "delta": reps[i] - reps[i - 1],
                "traffic": fe_ts["traffic"][i],
            })
    kv("Scaling transitions:", str(len(transitions)))
    scale_ups = [t for t in transitions if t["delta"] > 0]
    scale_downs = [t for t in transitions if t["delta"] < 0]
    kv("Scale-up events:", str(len(scale_ups)))
    kv("Scale-down events:", str(len(scale_downs)))
    if transitions:
        print(f"\n    {'Time':>10s}  {'From':>4s} → {'To':>4s}  {'Δ':>4s}  {'Traffic':>10s}")
        print(f"    {'─'*45}")
        for t in transitions[:15]:
            d = "↗️" if t["delta"] > 0 else "↘️"
            print(f"    {t['time'].strftime('%H:%M:%S'):>10s}  {t['from']:>4d} → {t['to']:>4d}  {t['delta']:>+4d}  {t['traffic']:>8.1f} rps  {d}")
        if len(transitions) > 15:
            print(f"    ... and {len(transitions)-15} more")

    # ── Oscillation ──
    osc = stats.get("oscillation", {})
    header("5. SCALING OSCILLATION (FLAPPING)")
    kv("Direction reversals:", str(osc.get("direction_changes", 0)))
    kv("Max reversals in 5min window:", str(osc.get("max_changes_in_window", 0)))
    kv("Flapping windows (≥3 reversals):", str(osc.get("flapping_windows", 0)))
    if osc.get("flapping_windows", 0) == 0:
        print(f"    ✅ No flapping detected — PD controller is stable")
    else:
        print(f"    ⚠️  Flapping detected — consider tuning PD gains or debounce")

    # ── Prediction ──
    header("6. PREDICTION ACCURACY")
    pa = stats.get("prediction_accuracy", {})
    if pa.get("mape") is not None:
        kv("Samples:", str(pa["n"]))
        kv("MAPE:", f"{pa['mape']:.1f}%")
        kv("RMSE:", f"{pa['rmse']:.2f} req/s")
        kv("R²:", f"{pa['r2']:.4f}")
        kv("Directional accuracy:", f"{pa['directional_accuracy']:.1f}%")
        if pa["mape"] < 15:
            print(f"    ✅ Excellent prediction accuracy (<15% MAPE)")
        elif pa["mape"] < 25:
            print(f"    ✓  Good prediction accuracy")
        elif pa["mape"] < 40:
            print(f"    ⚠️  Fair prediction accuracy — may need tuning")
        else:
            print(f"    ❌ Poor prediction accuracy — review predictor config")
    else:
        print(f"    ⚠️  Not enough prediction data")

    # ── Resource Efficiency ──
    header("7. RESOURCE EFFICIENCY")
    prov = stats.get("provisioning", {})
    if prov.get("mean"):
        kv("Avg provisioning ratio:", f"{prov['mean']:.2f}x  (ideal ~1.15x)")
        kv("Median provisioning ratio:", f"{prov['median']:.2f}x")
        kv("Over-provisioned (>1.5x):", f"{prov['over_pct']:.1f}% of time")
        kv("Under-provisioned (<1.0x):", f"{prov['under_pct']:.1f}% of time")
        kv("In ideal range:", f"{prov['ideal_pct']:.1f}% of time")

        traffic_with = [v for v in fe_ts["traffic"] if v > 0]
        if traffic_with:
            avg_t = sum(traffic_with) / len(traffic_with)
            avg_r = sum(fe_ts["total_replicas"]) / len(fe_ts["total_replicas"])
            cap = avg_r * SERVICE_CAPACITY["frontend"]
            util = avg_t / cap * 100 if cap > 0 else 0
            kv("Avg utilization:", f"{util:.1f}%")

    # ── Time to Scale ──
    header("8. TIME TO SCALE (TtS)")
    tts = stats.get("tts", {})
    kv("Threshold crossings detected:", str(tts.get("count", 0)))
    if tts.get("count", 0) > 0:
        kv("Proactive scalings:", f"{tts['proactive_count']}  ({tts['proactive_pct']:.0f}%)")
        kv("Reactive scalings:", str(tts["reactive_count"]))
        kv("Avg TtS:", f"{tts['avg_tts']:.1f}s")
        if tts["proactive_count"] > 0:
            kv("Avg proactive TtS:", f"{tts['avg_proactive_tts']:.1f}s  (negative = before spike)")
        if tts["reactive_count"] > 0:
            kv("Avg reactive TtS:", f"{tts['avg_reactive_tts']:.1f}s")
        if tts["proactive_pct"] > 50:
            print(f"    ✅ DMOS is predominantly proactive — thesis value demonstrated!")
        else:
            print(f"    ⚠️  DMOS is mostly reactive — check predictor training data")
    else:
        print(f"    ⚠️  No threshold crossings in this test — try higher load variance")

    # ── Fairness ──
    header("9. CLUSTER FAIRNESS (Jain Index)")
    jain_ts, jain_vals = stats.get("jain_data", ([], []))
    if jain_vals:
        nonzero = [j for j in jain_vals if j > 0]
        if nonzero:
            kv("Mean Jain Index:", f"{sum(nonzero)/len(nonzero):.3f}  (1.0 = perfect)")
            kv("Min Jain Index:", f"{min(nonzero):.3f}")

    # ── Locust Response Time ──
    header("10. END-USER RESPONSE TIME (Locust)")
    if locust_ts["timestamps"]:
        rt = locust_ts["p95_rt"]
        rt_pos = [v for v in rt if v > 0]
        if rt_pos:
            kv("p95 response time — Min:", f"{min(rt_pos):.0f} ms")
            kv("p95 response time — Max:", f"{max(rt_pos):.0f} ms")
            kv("p95 response time — Mean:", f"{sum(rt_pos)/len(rt_pos):.0f} ms")
        rps = locust_ts["rps"]
        rps_pos = [v for v in rps if v > 0]
        if rps_pos:
            kv("Throughput (RPS) — Mean:", f"{sum(rps_pos)/len(rps_pos):.1f}")
        fr = locust_ts["fail_ratio"]
        if fr:
            kv("Failure ratio — Max:", f"{max(fr)*100:.2f}%")
    else:
        print(f"    ⚠️  No Locust data — run with --locust-host to capture")

    # ── Backend Services ──
    header("11. BACKEND SERVICES")
    for svc in KNOWN_SERVICES:
        if svc == "frontend":
            continue
        svc_ts = extract_service_ts(data, svc)
        if any(r > 0 for r in svc_ts["total_replicas"]):
            t_pos = [v for v in svc_ts["traffic"] if v > 0]
            avg_r = sum(svc_ts["total_replicas"]) / len(svc_ts["total_replicas"]) if svc_ts["total_replicas"] else 0
            traffic_str = f"avg={sum(t_pos)/len(t_pos):.1f} rps" if t_pos else "no traffic data"
            kv(f"{svc}:", f"avg_replicas={avg_r:.1f}, {traffic_str}")
        else:
            kv(f"{svc}:", "no replicas observed (not yet scaled by DMOS)")

    print(f"\n{'═' * W}")
    print(f"  ✅ Report complete")
    print(f"{'═' * W}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt_time(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

def generate_page1_core(fe_ts: dict, output_dir: Path):
    """Page 1: Core metrics — Traffic, Replicas, Correlation, Distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DMOS Analysis — Core Metrics', fontsize=14, fontweight='bold')
    timestamps = fe_ts["timestamps"]

    # 1. Traffic over time
    ax = axes[0, 0]
    ax.plot(timestamps, fe_ts["traffic"], color=COLORS['traffic'], lw=2, label='Actual Traffic')
    if any(p > 0 for p in fe_ts["predicted"]):
        ax.plot(timestamps, fe_ts["predicted"], '--', color=COLORS['predicted'], lw=1.5, alpha=0.7, label='Predicted Traffic')
    ax.axhline(y=HIGH_THRESHOLD, color='orange', ls='--', alpha=0.5, label=f'High Threshold ({HIGH_THRESHOLD})')
    ax.axhline(y=LOW_THRESHOLD, color='cyan', ls='--', alpha=0.5, label=f'Low Threshold ({LOW_THRESHOLD})')
    ax.set_ylabel('Traffic (req/s)')
    ax.set_title('Traffic Over Time')
    ax.legend(loc='upper left', fontsize=7)
    _fmt_time(ax)

    # 2. Replicas over time
    ax = axes[0, 1]
    ax.plot(timestamps, fe_ts["total_replicas"], color=COLORS['replicas'], lw=2, marker='o', ms=3, label='Total Replicas')
    ax.set_ylabel('Replica Count')
    ax.set_title('Total Replica Count Over Time')
    ax.legend()
    _fmt_time(ax)

    # 3. Traffic vs Replicas (dual axis)
    ax = axes[1, 0]
    ax.plot(timestamps, fe_ts["traffic"], color=COLORS['traffic'], lw=2, label='Traffic')
    ax.set_ylabel('Traffic (req/s)', color=COLORS['traffic'])
    ax.tick_params(axis='y', labelcolor=COLORS['traffic'])
    ax2 = ax.twinx()
    ax2.plot(timestamps, fe_ts["total_replicas"], color=COLORS['replicas'], lw=2, marker='o', ms=2)
    ax2.set_ylabel('Replicas', color=COLORS['replicas'])
    ax2.tick_params(axis='y', labelcolor=COLORS['replicas'])
    ax.set_title('Traffic vs Replicas Correlation')
    _fmt_time(ax)

    # 4. Cluster distribution (stacked area)
    ax = axes[1, 1]
    c1 = fe_ts["per_cluster_replicas"]["cluster1"]
    c2 = fe_ts["per_cluster_replicas"]["cluster2"]
    c3 = fe_ts["per_cluster_replicas"]["cluster3"]
    ax.fill_between(timestamps, 0, c1, label='Cluster1', alpha=0.7, color=COLORS['cluster1'])
    c1c2 = [a + b for a, b in zip(c1, c2)]
    ax.fill_between(timestamps, c1, c1c2, label='Cluster2', alpha=0.7, color=COLORS['cluster2'])
    total = [a + b + c for a, b, c in zip(c1, c2, c3)]
    ax.fill_between(timestamps, c1c2, total, label='Cluster3', alpha=0.7, color=COLORS['cluster3'])
    ax.set_ylabel('Replicas')
    ax.set_title('Replica Distribution Across Clusters')
    ax.legend()
    _fmt_time(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "dmos_page1_core.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {path.name}")


def generate_page2_scaling(fe_ts: dict, stats: dict, output_dir: Path):
    """Page 2: Scaling analysis — TtS, Oscillation, Provisioning, Events."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DMOS Analysis — Scaling Quality', fontsize=14, fontweight='bold')
    timestamps = fe_ts["timestamps"]

    # 1. Over/Under-provisioning ratio timeline
    ax = axes[0, 0]
    prov = stats.get("provisioning", {})
    ratios = prov.get("ratios", [])
    if ratios:
        # Align ratios with timestamps (only where traffic > 0)
        ratio_ts = [t for t, tr in zip(timestamps, fe_ts["traffic"]) if tr > 0]
        min_len = min(len(ratio_ts), len(ratios))
        ratio_ts = ratio_ts[:min_len]
        ratios_plot = ratios[:min_len]
        ax.plot(ratio_ts, ratios_plot, color=COLORS['overprov'], lw=1.5, alpha=0.8)
        ax.fill_between(ratio_ts, 1.0, ratios_plot,
                        where=[r > 1.0 for r in ratios_plot],
                        color=COLORS['overprov'], alpha=0.2, label='Over-provisioned')
        ax.fill_between(ratio_ts, ratios_plot, 1.0,
                        where=[r < 1.0 for r in ratios_plot],
                        color=COLORS['underprov'], alpha=0.2, label='Under-provisioned')
        ax.axhline(y=1.0, color='black', ls='-', lw=1)
        ax.axhline(y=1.15, color='green', ls='--', alpha=0.5, label='Ideal (1.15x)')
    ax.set_ylabel('Provisioning Ratio')
    ax.set_title('Over/Under-Provisioning Ratio')
    ax.legend(fontsize=7)
    _fmt_time(ax)

    # 2. Time to Scale events
    ax = axes[0, 1]
    tts = stats.get("tts", {})
    tts_events = tts.get("events", [])
    if tts_events:
        pro_times = [e["cross_time"] for e in tts_events if e["proactive"]]
        pro_vals = [e["tts_seconds"] for e in tts_events if e["proactive"]]
        rea_times = [e["cross_time"] for e in tts_events if not e["proactive"]]
        rea_vals = [e["tts_seconds"] for e in tts_events if not e["proactive"]]
        if pro_times:
            ax.bar(pro_times, pro_vals, width=0.001, color='green', alpha=0.7, label=f'Proactive ({len(pro_times)})')
        if rea_times:
            ax.bar(rea_times, rea_vals, width=0.001, color='red', alpha=0.7, label=f'Reactive ({len(rea_times)})')
        ax.axhline(y=0, color='black', ls='-', lw=0.8)
    ax.set_ylabel('Time to Scale (seconds)')
    ax.set_title('Time to Scale (TtS) — Negative = Proactive')
    ax.legend()
    _fmt_time(ax)

    # 3. Scaling oscillation (direction changes in sliding window)
    ax = axes[1, 0]
    osc = stats.get("oscillation", {})
    wc = osc.get("window_changes_timeline", [])
    if wc:
        # Create timestamps for windows
        wc_ts = timestamps[:len(wc)]
        ax.plot(wc_ts, wc, color='#9b59b6', lw=1.5)
        ax.fill_between(wc_ts, 0, wc, alpha=0.2, color='#9b59b6')
        ax.axhline(y=3, color='red', ls='--', alpha=0.5, label='Flapping threshold')
    ax.set_ylabel('Direction Changes')
    ax.set_title('Scaling Oscillation (5-min sliding window)')
    ax.legend()
    _fmt_time(ax)

    # 4. Scaling events timeline
    ax = axes[1, 1]
    reps = fe_ts["total_replicas"]
    up_t, up_d = [], []
    dn_t, dn_d = [], []
    for i in range(1, len(reps)):
        delta = reps[i] - reps[i - 1]
        if delta > 0:
            up_t.append(timestamps[i]); up_d.append(delta)
        elif delta < 0:
            dn_t.append(timestamps[i]); dn_d.append(delta)
    if up_t:
        ax.scatter(up_t, up_d, color='green', s=80, marker='^', label='Scale Up', zorder=5)
    if dn_t:
        ax.scatter(dn_t, dn_d, color='red', s=80, marker='v', label='Scale Down', zorder=5)
    ax.axhline(y=0, color='black', ls='-', lw=0.5)
    ax.set_ylabel('Replica Change (Δ)')
    ax.set_title('Scaling Events Timeline')
    ax.legend()
    _fmt_time(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "dmos_page2_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {path.name}")


def generate_page3_quality(data: List[dict], fe_ts: dict, stats: dict, output_dir: Path):
    """Page 3: Quality metrics — Prediction scatter, Fairness, Scores, Duration."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DMOS Analysis — Quality Metrics', fontsize=14, fontweight='bold')
    timestamps = fe_ts["timestamps"]

    # 1. Prediction accuracy scatter
    ax = axes[0, 0]
    pairs = [(a, p) for a, p in zip(fe_ts["traffic"], fe_ts["predicted"]) if a > 1 and p > 0]
    if pairs:
        a_vals, p_vals = zip(*pairs)
        ax.scatter(a_vals, p_vals, alpha=0.5, s=25, color='#2980b9')
        mx = max(max(a_vals), max(p_vals))
        ax.plot([0, mx], [0, mx], 'r--', lw=2, label='Perfect Prediction')
        pa = stats.get("prediction_accuracy", {})
        if pa.get("mape") is not None:
            ax.text(0.05, 0.95, f"MAPE={pa['mape']:.1f}%\nR²={pa['r2']:.3f}",
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Actual Traffic (req/s)')
    ax.set_ylabel('Predicted Traffic (req/s)')
    ax.set_title('Prediction Accuracy')
    ax.legend()

    # 2. Jain Fairness Index over time
    ax = axes[0, 1]
    jain_ts, jain_vals = stats.get("jain_data", ([], []))
    if jain_ts:
        ax.plot(jain_ts, jain_vals, color='#16a085', lw=2)
        ax.fill_between(jain_ts, 0, jain_vals, alpha=0.15, color='#16a085')
        ax.axhline(y=1.0, color='green', ls='--', alpha=0.5, label='Perfect fairness')
        ax.axhline(y=1/3, color='red', ls='--', alpha=0.5, label='Worst case (1/3)')
        ax.set_ylim(0, 1.1)
    ax.set_ylabel('Jain Fairness Index')
    ax.set_title('Cluster Fairness Over Time')
    ax.legend(fontsize=7)
    _fmt_time(ax)

    # 3. Multi-objective scores evolution
    ax = axes[1, 0]
    for c in KNOWN_CLUSTERS:
        scores = fe_ts["per_cluster_scores"][c]
        if any(s > 0 for s in scores):
            ax.plot(timestamps, scores, lw=1.5, label=c, color=COLORS[c])
    ax.set_ylabel('Multi-Objective Score')
    ax.set_title('Cluster Scores Over Time')
    ax.legend()
    _fmt_time(ax)

    # 4. Scheduling duration over time
    ax = axes[1, 1]
    # Extract from raw data
    sched_counts = []
    sched_avgs = []
    sched_ts = []
    prev_count = 0
    prev_sum_val = 0
    for snap in data:
        fe = snap["dmos"]["services"].get("frontend", {})
        cnt = fe.get("scheduling_invocations", 0)
        avg = fe.get("avg_scheduling_duration_s", 0)
        if cnt > prev_count:
            sched_ts.append(snap["_ts"])
            sched_avgs.append(avg)
            prev_count = cnt
    if sched_ts:
        ax.bar(sched_ts, sched_avgs, width=0.0005, color='#e67e22', alpha=0.7)
        avg_dur = sum(sched_avgs) / len(sched_avgs)
        ax.axhline(y=avg_dur, color='red', ls='--', alpha=0.7, label=f'Avg: {avg_dur:.3f}s')
    ax.set_ylabel('Duration (seconds)')
    ax.set_title('Scheduling Duration')
    ax.legend()
    _fmt_time(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "dmos_page3_quality.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {path.name}")


def generate_page4_response(data: List[dict], fe_ts: dict, locust_ts: dict, stats: dict, output_dir: Path):
    """Page 4: Response Time & Backend — Locust RT, RT vs Replicas, Backend svcs, KPI summary."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DMOS Analysis — End-User & Backend Impact', fontsize=14, fontweight='bold')

    # 1. Locust Response Time over time
    ax = axes[0, 0]
    if locust_ts["timestamps"]:
        ts = locust_ts["timestamps"]
        if any(v > 0 for v in locust_ts["p95_rt"]):
            ax.plot(ts, locust_ts["p95_rt"], color=COLORS['p95'], lw=2, label='p95')
        if any(v > 0 for v in locust_ts["p99_rt"]):
            ax.plot(ts, locust_ts["p99_rt"], color='#c0392b', lw=1.5, ls='--', label='p99')
        if any(v > 0 for v in locust_ts["median_rt"]):
            ax.plot(ts, locust_ts["median_rt"], color='#2ecc71', lw=1.5, label='Median')
        ax.legend()
        _fmt_time(ax)
    else:
        ax.text(0.5, 0.5, 'No Locust data\nRun with --locust-host',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')
    ax.set_ylabel('Response Time (ms)')
    ax.set_title('End-User Response Time (Locust)')

    # 2. p95 RT vs Total Replicas scatter
    ax = axes[0, 1]
    if locust_ts["timestamps"]:
        # Align by nearest timestamp
        rt_map = {}
        for i, t in enumerate(locust_ts["timestamps"]):
            rt_map[t] = locust_ts["p95_rt"][i]
        
        aligned_rt = []
        aligned_reps = []
        for i, t in enumerate(fe_ts["timestamps"]):
            # Find nearest locust timestamp
            best = None
            best_diff = float('inf')
            for lt in locust_ts["timestamps"]:
                diff = abs((t - lt).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best = lt
            if best and best_diff < 20 and rt_map.get(best, 0) > 0:
                aligned_rt.append(rt_map[best])
                aligned_reps.append(fe_ts["total_replicas"][i])
        
        if aligned_rt:
            sc = ax.scatter(aligned_reps, aligned_rt, alpha=0.5, s=30, c=COLORS['p95'])
            ax.set_xlabel('Total Replicas')
            ax.set_ylabel('p95 Response Time (ms)')
    ax.set_title('Response Time vs Replica Count')

    # 3. Backend services replicas over time (stacked)
    ax = axes[1, 0]
    has_backend = False
    for svc in KNOWN_SERVICES:
        if svc == "frontend":
            continue
        svc_ts = extract_service_ts(data, svc)
        if any(r > 0 for r in svc_ts["total_replicas"]):
            ax.plot(svc_ts["timestamps"], svc_ts["total_replicas"], lw=1.5, label=svc, alpha=0.8)
            has_backend = True
    if not has_backend:
        ax.text(0.5, 0.5, 'Backend services not yet\nscaled by DMOS',
                transform=ax.transAxes, ha='center', va='center', fontsize=11, color='gray')
    else:
        ax.legend(fontsize=7)
        _fmt_time(ax)
    ax.set_ylabel('Total Replicas')
    ax.set_title('Backend Services Scaling')

    # 4. KPI Summary table
    ax = axes[1, 1]
    ax.axis('off')
    pa = stats.get("prediction_accuracy", {})
    prov = stats.get("provisioning", {})
    tts = stats.get("tts", {})
    osc = stats.get("oscillation", {})

    kpis = [
        ("Metric", "Value", "Status"),
        ("MAPE", f"{pa.get('mape', 'N/A'):.1f}%" if pa.get('mape') else "N/A",
         "✅" if pa.get('mape') and pa['mape'] < 20 else "⚠️" if pa.get('mape') else "—"),
        ("R²", f"{pa.get('r2', 'N/A'):.3f}" if pa.get('r2') is not None else "N/A",
         "✅" if pa.get('r2') and pa['r2'] > 0.7 else "⚠️" if pa.get('r2') else "—"),
        ("Proactive %", f"{tts.get('proactive_pct', 0):.0f}%",
         "✅" if tts.get('proactive_pct', 0) > 50 else "⚠️"),
        ("Avg TtS", f"{tts.get('avg_tts', 0):.1f}s",
         "✅" if tts.get('avg_tts', 0) < 0 else "⚠️"),
        ("Over-prov %", f"{prov.get('over_pct', 0):.1f}%",
         "✅" if prov.get('over_pct', 0) < 30 else "⚠️"),
        ("Under-prov %", f"{prov.get('under_pct', 0):.1f}%",
         "✅" if prov.get('under_pct', 0) < 10 else "⚠️"),
        ("Flapping", str(osc.get("flapping_windows", 0)),
         "✅" if osc.get("flapping_windows", 0) == 0 else "⚠️"),
    ]

    table = ax.table(
        cellText=[row for row in kpis],
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.3, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Style data rows
    for i in range(1, len(kpis)):
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    ax.set_title('KPI Summary Dashboard', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "dmos_page4_response.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_analysis_json(data: List[dict], fe_ts: dict, locust_ts: dict, stats: dict, output_file: Path):
    """Export comprehensive analysis as JSON."""

    traffic_vals = [v for v in fe_ts["traffic"] if v > 0]
    reps = fe_ts["total_replicas"]

    summary = {
        "test_info": {
            "start": data[0]["timestamp"],
            "end": data[-1]["timestamp"],
            "duration_seconds": (data[-1]["_ts"] - data[0]["_ts"]).total_seconds(),
            "snapshots": len(data),
            "locust_available": any(s.get("locust", {}).get("available") for s in data),
        },
        "traffic": {
            "min": min(traffic_vals) if traffic_vals else 0,
            "max": max(traffic_vals) if traffic_vals else 0,
            "mean": sum(traffic_vals) / len(traffic_vals) if traffic_vals else 0,
            "samples": len(traffic_vals),
        },
        "replicas": {
            "min": min(reps) if reps else 0,
            "max": max(reps) if reps else 0,
            "mean": sum(reps) / len(reps) if reps else 0,
        },
        "prediction_accuracy": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in stats.get("prediction_accuracy", {}).items()
        },
        "provisioning": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in stats.get("provisioning", {}).items()
            if k != "ratios"
        },
        "time_to_scale": {
            k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in stats.get("tts", {}).items()
            if k != "events"
        },
        "oscillation": {
            k: v for k, v in stats.get("oscillation", {}).items()
            if k != "window_changes_timeline"
        },
        "jain_fairness": {
            "mean": 0,
            "min": 0,
        },
    }

    jain_ts, jain_vals = stats.get("jain_data", ([], []))
    nonzero_jain = [j for j in jain_vals if j > 0]
    if nonzero_jain:
        summary["jain_fairness"]["mean"] = round(sum(nonzero_jain) / len(nonzero_jain), 4)
        summary["jain_fairness"]["min"] = round(min(nonzero_jain), 4)

    if locust_ts["p95_rt"]:
        rt = [v for v in locust_ts["p95_rt"] if v > 0]
        summary["response_time"] = {
            "p95_min_ms": min(rt) if rt else 0,
            "p95_max_ms": max(rt) if rt else 0,
            "p95_mean_ms": round(sum(rt) / len(rt), 1) if rt else 0,
        }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  📄 {output_file.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(filepath: str):
    """Main analysis pipeline."""
    print("=" * 78)
    print("🔬 DMOS Comprehensive Test Analysis v2")
    print("=" * 78)
    print(f"\n  Input: {filepath}\n")

    # Load
    data = load_data(filepath)
    if not data:
        print("❌ No data found!")
        return

    # Extract timeseries
    fe_ts = extract_service_ts(data, "frontend")
    locust_ts = extract_locust_ts(data)

    # Compute all metrics
    print("  Computing metrics...")
    stats = {}
    stats["prediction_accuracy"] = compute_prediction_accuracy(fe_ts["traffic"], fe_ts["predicted"])
    stats["provisioning"] = compute_provisioning_ratio(
        fe_ts["traffic"], fe_ts["total_replicas"], SERVICE_CAPACITY["frontend"]
    )
    stats["tts"] = compute_time_to_scale(fe_ts["timestamps"], fe_ts["traffic"], fe_ts["total_replicas"])
    stats["oscillation"] = compute_scaling_oscillation(fe_ts["total_replicas"], fe_ts["timestamps"])
    stats["jain_data"] = compute_jain_fairness(data, "frontend")

    # Console report
    print_report(data, fe_ts, locust_ts, stats)

    # Generate plots
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    print("  Generating plots...")
    generate_page1_core(fe_ts, output_dir)
    generate_page2_scaling(fe_ts, stats, output_dir)
    generate_page3_quality(data, fe_ts, stats, output_dir)
    generate_page4_response(data, fe_ts, locust_ts, stats, output_dir)

    # Export JSON
    json_path = Path(filepath).with_suffix('.analysis.json')
    export_analysis_json(data, fe_ts, locust_ts, stats, json_path)

    print(f"\n{'=' * 78}")
    print(f"✅ Analysis complete!")
    print(f"   Plots:    results/plots/dmos_page{{1..4}}_*.png")
    print(f"   JSON:     {json_path}")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-find latest file
        results_dir = Path("results")
        # Prefer JSONL
        files = list(results_dir.glob("metrics_*.jsonl"))
        if not files:
            files = list(results_dir.glob("metrics_timeseries_*.txt"))
        if not files:
            print("❌ No metrics files found in results/")
            print("Usage: python analyze_test_complete.py <metrics_file>")
            sys.exit(1)

        latest = max(files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Using latest file: {latest.name}\n")
        analyze(str(latest))
    else:
        analyze(sys.argv[1])