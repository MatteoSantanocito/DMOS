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

# From config/services.yaml — aggiornato con capacity test 27/02/2026
# frontend: knee point gradual_ramp ~45 rps/pod, burst ~25 rps/pod
# Per analisi post-hoc (provisioning ratio): usa il valore graduale (osservato)
# Per DMOS config/services.yaml (scaling trigger): usa valore conservativo (30)
SERVICE_CAPACITY = {
    "frontend": 45,   # gradual_ramp: 3 pod @ 131 rps = p95 200ms → 43.7 rps/pod
    "cartservice": 10,
    "productcatalogservice": 15,
    "checkoutservice": 5,
    "recommendationservice": 10,
}
SERVICE_MIN_REPLICAS = {
    "frontend": 1,
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
CLUSTER_REGIONS = {"cluster1": "DE", "cluster2": "FR", "cluster3": "PL"}


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
    """Compute MAPE, RMSE, R², directional accuracy.

    Two MAPE variants are computed:
    - mape_overall:     All snapshots where actual > 1 rps and pred > 0.
                        Inflated by tail (actual→0, pred still EMA-high).
    - mape_active:      Only snapshots where actual >= max(5, peak*0.10).
                        Tail excluded → more meaningful for thesis.
    The primary metric reported is mape_active (clearly labelled).
    """
    peak = max(traffic) if traffic else 1.0

    # --- Overall pairs (same filter as before for backward compat) ---
    pairs_all = [(a, p) for a, p in zip(traffic, predicted) if a > 1 and p > 0]

    # --- Active-phase pairs (exclude tail + warmup) ---
    active_threshold = max(5.0, peak * 0.10)
    pairs_active = [(a, p) for a, p in zip(traffic, predicted)
                    if a >= active_threshold and p > 0]

    if len(pairs_all) < 3:
        return {
            "mape": None, "mape_active": None,
            "rmse": None, "r2": None, "directional_accuracy": None,
            "n": len(pairs_all), "n_active": len(pairs_active),
            "active_threshold": active_threshold,
        }

    def _compute_mape(pairs):
        return sum(abs(p - a) / a * 100 for a, p in pairs) / len(pairs)

    mape_overall = _compute_mape(pairs_all)
    mape_active = _compute_mape(pairs_active) if len(pairs_active) >= 3 else None

    actuals = [a for a, _ in pairs_all]
    preds   = [p for _, p in pairs_all]

    # RMSE (overall)
    sq_errors = [(p - a) ** 2 for a, p in pairs_all]
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

    # R² (overall)
    mean_a = sum(actuals) / len(actuals)
    ss_res = sum((a - p) ** 2 for a, p in pairs_all)
    ss_tot = sum((a - mean_a) ** 2 for a in actuals)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy (active pairs preferred for accuracy)
    pairs_dir = pairs_active if len(pairs_active) >= 3 else pairs_all
    dir_correct = 0
    for i in range(1, len(pairs_dir)):
        a_dir = pairs_dir[i][0] - pairs_dir[i - 1][0]
        p_dir = pairs_dir[i][1] - pairs_dir[i - 1][1]
        if (a_dir >= 0 and p_dir >= 0) or (a_dir < 0 and p_dir < 0):
            dir_correct += 1
    dir_acc = dir_correct / (len(pairs_dir) - 1) * 100 if len(pairs_dir) > 1 else 0

    # Primary reported MAPE = active (if available), else overall
    mape_primary = mape_active if mape_active is not None else mape_overall

    return {
        "mape": mape_primary,           # primary (active-phase, tail excluded)
        "mape_overall": mape_overall,   # incl. tail — for reference
        "mape_active": mape_active,     # same as mape (explicit alias)
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": dir_acc,
        "n": len(pairs_all),
        "n_active": len(pairs_active),
        "active_threshold": active_threshold,
        "peak_traffic": peak,
    }

def compute_provisioning_ratio(traffic: list, replicas: list, capacity_per_replica: float,
                                min_replicas_total: int = 6) -> dict:
    """
    Compute over/under provisioning ratio per snapshot.
    
    ratio = provisioned_capacity / max(actual_traffic, min_capacity)
    
    where min_capacity = min_replicas_total * capacity_per_replica
    
    This prevents the HA floor (e.g. 2 replicas × 3 clusters = 6 minimum)
    from artificially inflating the over-provisioning ratio during low-traffic
    periods. The minimum replica count is an architectural constraint for
    high availability, not wasteful over-provisioning.
    
    > 1 = over-provisioned, < 1 = under-provisioned, ~1.15 = ideal

    Warmup exclusion: snapshots where traffic < 10% of max traffic are excluded
    from summary statistics (but kept in the ratios list for plotting).
    """
    ratios = []
    over_count = 0
    under_count = 0

    max_traffic = max(traffic) if traffic else 1
    warmup_threshold = max(max_traffic * 0.10, HIGH_THRESHOLD)  # At least HIGH_THRESHOLD
    
    # Minimum capacity floor: the HA baseline that the system always maintains
    min_capacity = min_replicas_total * capacity_per_replica

    # All ratios (for plotting)
    for t, r in zip(traffic, replicas):
        if t > 0 and r > 0:
            cap = r * capacity_per_replica
            # Use max(traffic, min_capacity) as denominator to avoid
            # penalizing HA floor during low-traffic periods
            effective_demand = max(t, min_capacity)
            ratio = cap / effective_demand
            ratios.append(ratio)
        else:
            ratios.append(0)

    # Active-phase ratios (for statistics) — exclude warmup
    active_ratios = []
    for t, r in zip(traffic, replicas):
        if t >= warmup_threshold and r > 0:
            cap = r * capacity_per_replica
            effective_demand = max(t, min_capacity)
            ratio = cap / effective_demand
            active_ratios.append(ratio)
            if ratio > 1.5:
                over_count += 1
            elif ratio < 1.0:
                under_count += 1

    if not active_ratios:
        active_ratios = [r for r in ratios if r > 0]  # Fallback to all
        over_count = sum(1 for r in active_ratios if r > 1.5)
        under_count = sum(1 for r in active_ratios if r < 1.0)

    n = len(active_ratios) if active_ratios else 1

    return {
        "ratios": ratios,  # Full list for plotting
        "active_ratios": active_ratios,  # Only active phase
        "mean": sum(active_ratios) / n if active_ratios else 0,
        "median": sorted(active_ratios)[n // 2] if active_ratios else 0,
        "over_pct": over_count / n * 100,
        "under_pct": under_count / n * 100,
        "ideal_pct": (n - over_count - under_count) / n * 100,
        "active_snapshots": n,
    }

def compute_time_to_scale(timestamps: list, traffic: list, replicas: list,
                          predicted: list = None, capacity_per_replica: float = 50) -> dict:
    """
    Compute Time to Scale (TtS) with two complementary methods:

    Method 1 — Threshold-based (original):
      For each traffic upward crossing of HIGH_THRESHOLD, find the nearest replica increase.
      Negative TtS = proactive (scaled before traffic spike).

    Method 2 — Demand-driven (new):
      For each scale-up event, determine whether the scaling was driven by
      predicted traffic exceeding current demand (proactive) or by current
      traffic already exceeding provisioned capacity (reactive).

      A scale-up is PROACTIVE if, at the time of the event:
        predicted_traffic > current_traffic * 1.05  (prediction anticipates growth)
      AND
        needed_replicas(predicted) > current_replicas  (prediction justified scaling)

      A scale-up is REACTIVE if current traffic alone already required more replicas.

    The two methods are complementary. Method 2 is the primary metric for DMOS
    because it directly measures whether the predictor drove the scaling decision.
    """
    # ── Method 1: Threshold crossings (keep for backward compat) ──
    threshold_events = []
    for i in range(1, len(traffic)):
        if traffic[i - 1] <= HIGH_THRESHOLD < traffic[i]:
            cross_time = timestamps[i]
            best_scale_time = None
            for j in range(max(0, i - 10), i):
                if replicas[j] < replicas[min(j + 1, len(replicas) - 1)]:
                    best_scale_time = timestamps[j + 1] if j + 1 < len(timestamps) else timestamps[j]
                    break
            if best_scale_time is None:
                for j in range(i, min(len(replicas) - 1, i + 10)):
                    if replicas[j] < replicas[j + 1]:
                        best_scale_time = timestamps[j + 1]
                        break
            if best_scale_time is not None:
                tts_seconds = (best_scale_time - cross_time).total_seconds()
                threshold_events.append({
                    "cross_time": cross_time,
                    "scale_time": best_scale_time,
                    "tts_seconds": tts_seconds,
                    "proactive": tts_seconds < 0,
                    "method": "threshold",
                })

    # ── Method 2: Demand-driven proactivity ──
    demand_events = []
    if predicted is not None and len(predicted) == len(traffic):
        for i in range(1, len(replicas)):
            delta = replicas[i] - replicas[i - 1]
            if delta > 0:  # Scale-up event detected
                t_cur = traffic[i]
                t_pred = predicted[i]
                r_before = replicas[i - 1]
                r_after = replicas[i]

                # How many replicas does the *current* traffic require?
                needed_now = max(1, math.ceil(t_cur / capacity_per_replica * 1.15))
                # How many does the *predicted* traffic require?
                needed_pred = max(1, math.ceil(t_pred / capacity_per_replica * 1.15))

                # Classification:
                # In a multi-cluster system like DMOS, the actual replica count is
                # much higher than the theoretical minimum (due to fairness, latency,
                # and per-cluster distribution). So comparing needed_replicas vs
                # current_replicas doesn't capture proactivity.
                #
                # Instead, proactivity means: the PREDICTION drove the scaling decision
                # by anticipating traffic growth. This is true when:
                #   predicted_traffic > current_traffic (prediction sees growth coming)
                #
                # The threshold is small (>3%) because the floor guarantee means
                # pred >= current always, so any pred > current means the trend
                # component was positive (i.e., the predictor detected a ramp).
                prediction_anticipates = t_pred > t_cur * 1.03

                if prediction_anticipates:
                    classification = "proactive"
                else:
                    classification = "reactive"

                is_proactive = (classification == "proactive")

                # Estimate TtS: how far ahead did the predictor see?
                # Look forward to find when traffic actually reaches predicted level
                tts_seconds = 0.0
                if is_proactive and t_pred > t_cur:
                    for j in range(i + 1, min(len(traffic), i + 20)):
                        if traffic[j] >= t_pred * 0.9:  # traffic reached ~90% of prediction
                            tts_seconds = -(timestamps[j] - timestamps[i]).total_seconds()
                            break
                    else:
                        # Traffic never reached prediction level in window → very proactive
                        tts_seconds = -60.0  # Conservative estimate

                demand_events.append({
                    "cross_time": timestamps[i],
                    "scale_time": timestamps[i],
                    "tts_seconds": tts_seconds,
                    "proactive": is_proactive,
                    "method": "demand",
                    "classification": classification,
                    "traffic": t_cur,
                    "predicted": t_pred,
                    "replicas_before": r_before,
                    "replicas_after": r_after,
                    "needed_now": needed_now,
                    "needed_pred": needed_pred,
                    "delta": delta,
                })

    # ── Combine: use demand_events as primary, threshold as supplementary ──
    all_events = demand_events if demand_events else threshold_events

    proactive = [e for e in all_events if e["proactive"]]
    reactive = [e for e in all_events if not e["proactive"]]

    return {
        "events": all_events,
        "threshold_events": threshold_events,
        "demand_events": demand_events,
        "count": len(all_events),
        "proactive_count": len(proactive),
        "reactive_count": len(reactive),
        "proactive_pct": len(proactive) / len(all_events) * 100 if all_events else 0,
        "avg_tts": sum(e["tts_seconds"] for e in all_events) / len(all_events) if all_events else 0,
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
            n = len(KNOWN_CLUSTERS)  # sempre 3: fairness rispetto a tutti i cluster previsti
            # J=1 → distribuzione perfetta, J=1/3 → tutto su un cluster
            if sum_sq > 0:
                jain = (total ** 2) / (n * sum_sq)
            else:
                jain = 0.0
        else:
            jain = 0.0

        timestamps.append(snap["_ts"])
        jain_values.append(jain)

    return timestamps, jain_values


def load_locust_cluster_csv(scenario: str = None, results_dir: Path = None) -> dict:
    """
    Legge il timeseries CSV per-cluster prodotto da locustfile_multiingress.py.
    Compatibile sia con formato vecchio (3 colonne/cluster) che nuovo (4 colonne/cluster con slo_pct).

    Returns dict:
        {cluster_name: {"timestamps": [...], "p95_ms": [...], "slo_pct": [...], "count": [...]}}
    """
    empty = {c: {"timestamps": [], "p95_ms": [], "slo_pct": [], "count": []} for c in KNOWN_CLUSTERS}

    search_dir = results_dir or Path("results/multiingress")
    if not search_dir.exists():
        print(f"  ℹ  Locust cluster dir not found: {search_dir}")
        return empty

    pattern = f"{scenario}_timeseries_*.csv" if scenario else "*_timeseries_*.csv"
    candidates = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates and scenario:
        # Fallback: strip _off / _on suffix (collector uses _off/_on, Locust CSV doesn't)
        import re as _re
        base_scenario = _re.sub(r'_(off|on)$', '', scenario)
        if base_scenario != scenario:
            pattern = f"{base_scenario}_timeseries_*.csv"
            candidates = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print(f"  ℹ  No Locust timeseries CSV found (pattern: {pattern})")
        return empty

    csv_path = candidates[0]
    print(f"  📄 Locust cluster CSV: {csv_path.name}")

    result = {c: {"timestamps": [], "p95_ms": [], "slo_pct": [], "count": []} for c in KNOWN_CLUSTERS}
    today = datetime.now().date()

    try:
        import csv as _csv
        with open(csv_path, "r", newline="") as f:
            reader = _csv.DictReader(f)
            prev_ts = None
            day_offset = 0
            for row in reader:
                try:
                    t = datetime.strptime(row["timestamp"], "%H:%M:%S")
                    ts = t.replace(year=today.year, month=today.month, day=today.day)
                    # Handle midnight rollover
                    if prev_ts and ts < prev_ts:
                        day_offset += 1
                    from datetime import timedelta as _td
                    ts = ts + _td(days=day_offset)
                    prev_ts = ts
                except ValueError:
                    continue

                for cn in KNOWN_CLUSTERS:
                    try:
                        count = int(float(row.get(f"{cn}_count", 0) or 0))
                        p95   = float(row.get(f"{cn}_p95_ms", 0) or 0)
                        slo   = float(row.get(f"{cn}_slo_pct", 0) or 0)
                        result[cn]["timestamps"].append(ts)
                        result[cn]["count"].append(count)
                        result[cn]["p95_ms"].append(p95)
                        result[cn]["slo_pct"].append(slo)
                    except (ValueError, KeyError):
                        pass
    except Exception as e:
        print(f"  ⚠  Error reading Locust cluster CSV: {e}")
        return empty

    total_rows = len(result[KNOWN_CLUSTERS[0]]["timestamps"])
    print(f"  ✅ Loaded {total_rows} windows from Locust per-cluster CSV")
    return result


def load_locust_cluster_latency_csv(scenario: str = None, results_dir: Path = None) -> dict:
    """
    Legge il cluster_latency CSV prodotto da locustfile_multiingress.py.
    Contiene statistiche aggregate per-cluster sull'INTERA durata del test
    (equivalente al P95 globale flat di k6 / Romano):
      cluster, region, requests, failures, fail_pct, avg_ms, p50_ms, p90_ms, p95_ms, p99_ms, slo_pct

    Returns dict:
        {cluster_name: {"requests": int, "failures": int, "avg_ms": float,
                        "p50_ms": float, "p90_ms": float, "p95_ms": float,
                        "p99_ms": float, "slo_pct": float}}
    """
    empty = {c: {"requests": 0, "failures": 0, "avg_ms": 0.0,
                 "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0,
                 "p99_ms": 0.0, "slo_pct": 0.0} for c in KNOWN_CLUSTERS}

    search_dir = results_dir or Path("results/multiingress")
    if not search_dir.exists():
        return empty

    pattern = f"{scenario}_cluster_latency_*.csv" if scenario else "*_cluster_latency_*.csv"
    candidates = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates and scenario:
        # Fallback: strip _off / _on suffix (collector uses _off/_on, Locust CSV doesn't)
        import re as _re
        base_scenario = _re.sub(r'_(off|on)$', '', scenario)
        if base_scenario != scenario:
            pattern = f"{base_scenario}_cluster_latency_*.csv"
            candidates = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print(f"  ℹ  No Locust cluster_latency CSV found (pattern: {pattern})")
        return empty

    csv_path = candidates[0]
    print(f"  📄 Locust latency CSV: {csv_path.name}")

    result = {c: dict(empty[c]) for c in KNOWN_CLUSTERS}
    try:
        import csv as _csv
        with open(csv_path, "r", newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                cn = row.get("cluster", "").strip()
                if cn not in KNOWN_CLUSTERS:
                    continue
                result[cn] = {
                    "requests": int(float(row.get("requests", 0) or 0)),
                    "failures": int(float(row.get("failures", 0) or 0)),
                    "avg_ms":   float(row.get("avg_ms",  0) or 0),
                    "p50_ms":   float(row.get("p50_ms",  0) or 0),
                    "p90_ms":   float(row.get("p90_ms",  0) or 0),
                    "p95_ms":   float(row.get("p95_ms",  0) or 0),
                    "p99_ms":   float(row.get("p99_ms",  0) or 0),
                    "slo_pct":  float(row.get("slo_pct", 0) or 0),
                }
    except Exception as e:
        print(f"  ⚠  Error reading Locust latency CSV: {e}")
        return empty

    loaded = [cn for cn in KNOWN_CLUSTERS if result[cn]["requests"] > 0]
    print(f"  ✅ Loaded cluster_latency for: {', '.join(loaded)}")
    return result


def compute_global_p95_flat(cluster_latency: dict) -> Optional[float]:
    """
    P95 globale flat (stile k6/Romano): media pesata dei p95 per-cluster,
    pesata per numero di request totali.
    Approssima il vero p95 globale — comparabile con i risultati di Romano.
    """
    total_req = sum(v.get("requests", 0) for v in cluster_latency.values())
    if total_req == 0:
        return None
    weighted = sum(
        v.get("p95_ms", 0.0) * v.get("requests", 0)
        for v in cluster_latency.values()
        if v.get("requests", 0) > 0
    )
    return weighted / total_req


# ═══════════════════════════════════════════════════════════════════════════════
# Statistical Report (Console)
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(data: List[dict], fe_ts: dict, locust_ts: dict, stats: dict,
                 cluster_latency: dict = None):
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
        # Replica distribution % per cluster (media snapshot con traffico)
        active_snaps = [i for i, r in enumerate(reps) if r > 0]
        if active_snaps:
            print(f"\n    {'Cluster':<12} {'Avg reps':>10} {'Share %':>10} {'Max':>6}")
            print(f"    {'─'*42}")
            total_avg = sum(reps[i] for i in active_snaps) / len(active_snaps) if active_snaps else 1
            for c in KNOWN_CLUSTERS:
                cr_all = [fe_ts["per_cluster_replicas"][c][i] for i in active_snaps]
                avg_c = sum(cr_all) / len(cr_all) if cr_all else 0
                share = avg_c / max(total_avg, 1) * 100
                region = CLUSTER_REGIONS.get(c, "")
                print(f"    {c:<12} {avg_c:>10.1f} {share:>9.1f}% {max(cr_all) if cr_all else 0:>6d}")
            print()

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
        active_thr = pa.get("active_threshold", 0)
        peak = pa.get("peak_traffic", 0)
        n_total = pa.get("n", 0)
        n_active = pa.get("n_active", 0)
        kv("Peak traffic:", f"{peak:.1f} req/s")
        kv("Active-phase threshold:", f"≥ {active_thr:.1f} req/s  (max(5, peak×10%))")
        kv("Samples (total / active):", f"{n_total} / {n_active}")
        # Primary metric: active-only MAPE
        mape_act = pa.get("mape_active")
        mape_ov  = pa.get("mape_overall")
        if mape_act is not None:
            kv("MAPE (active-phase):", f"{mape_act:.1f}%  ← primary metric")
        if mape_ov is not None:
            kv("MAPE (incl. tail):", f"{mape_ov:.1f}%  (inflated by EMA decay after traffic ends)")
        kv("RMSE:", f"{pa['rmse']:.2f} req/s")
        kv("R²:", f"{pa['r2']:.4f}")
        kv("Directional accuracy:", f"{pa['directional_accuracy']:.1f}%")
        primary = mape_act if mape_act is not None else mape_ov
        if primary < 15:
            print(f"    ✅ Excellent prediction accuracy (<15% active-MAPE)")
        elif primary < 25:
            print(f"    ✓  Good prediction accuracy")
        elif primary < 40:
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
        kv("Active snapshots:", f"{prov.get('active_snapshots', '?')}  (warmup excluded)")

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
    demand_events = tts.get("demand_events", [])
    threshold_events = tts.get("threshold_events", [])

    if demand_events:
        kv("Scale-up events analyzed:", str(len(demand_events)))
        kv("Proactive scalings:", f"{tts['proactive_count']}  ({tts['proactive_pct']:.0f}%)")
        kv("Reactive scalings:", str(tts["reactive_count"]))
        kv("Avg TtS:", f"{tts['avg_tts']:.1f}s")
        if tts["proactive_count"] > 0:
            kv("Avg proactive TtS:", f"{tts['avg_proactive_tts']:.1f}s  (negative = before spike)")
        if tts["reactive_count"] > 0:
            kv("Avg reactive TtS:", f"{tts['avg_reactive_tts']:.1f}s")
        if tts["proactive_pct"] > 50:
            print(f"    ✅ DMOS is predominantly proactive — thesis value demonstrated!")
        elif tts["proactive_pct"] > 30:
            print(f"    ✓ DMOS shows proactive behavior in {tts['proactive_pct']:.0f}% of scale-ups")
        else:
            print(f"    ⚠️  DMOS is mostly reactive — check predictor tuning")

        print(f"\n    Demand-Driven Scale-Up Analysis:")
        for e in demand_events[:10]:
            cls = e.get("classification", "?")
            tag = "🟢 PROACTIVE" if e["proactive"] else "🔴 REACTIVE"
            print(f"      {e['cross_time'].strftime('%H:%M:%S')} | {e['replicas_before']}→{e['replicas_after']} "
                  f"(+{e['delta']}) | traffic={e['traffic']:.0f} pred={e['predicted']:.0f} "
                  f"| need_now={e['needed_now']} need_pred={e['needed_pred']} | {tag}")
        if len(demand_events) > 10:
            print(f"      ... and {len(demand_events) - 10} more")
    elif threshold_events:
        kv("Threshold crossings detected:", str(len(threshold_events)))
        kv("Proactive scalings:", f"{tts['proactive_count']}  ({tts['proactive_pct']:.0f}%)")
        kv("Reactive scalings:", str(tts["reactive_count"]))
    else:
        print(f"    ⚠️  No scaling events detected — try higher load variance")

    # ── Fairness ──
    header("9. CLUSTER FAIRNESS (Jain Index)")
    jain_ts, jain_vals = stats.get("jain_data", ([], []))
    if jain_vals:
        nonzero = [j for j in jain_vals if j > 0]
        if nonzero:
            mean_j = sum(nonzero) / len(nonzero)
            kv("Mean Jain Index:", f"{mean_j:.3f}  (1.0 = perfect, 0.333 = worst)")
            kv("Min Jain Index:", f"{min(nonzero):.3f}")
            status = "✅ Good" if mean_j > 0.85 else "⚠️  Unbalanced" if mean_j > 0.6 else "❌ Poor"
            kv("Fairness status:", status)
    else:
        print(f"    ⚠️  No Jain data available")

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

    # ── SLO Violation Rate + Global P95 flat (da cluster_latency CSV) ──
    header("10b. GLOBAL P95 FLAT  (stile k6 / Romano — aggregato sull'intero test)")
    kv("SLO threshold:", "1000 ms  (allineato Romano p95~1.0–1.5s, Cilantro 2s)")
    if cluster_latency:
        flat = compute_global_p95_flat(cluster_latency)
        if flat is not None:
            status = "✅" if flat < 1000 else ("⚠️" if flat < 2000 else "❌")
            kv("p95 globale flat (pesato):", f"{flat:.0f} ms  {status}")
            kv("  (Romano Scenario 2 rif.):", "~1320 ms  ON / ~2140 ms OFF")
            print()
            print(f"    {'Cluster':<12} {'Requests':>9}  {'p50':>7}  {'p90':>7}  {'p95':>7}  {'p99':>7}  {'SLO%':>7}")
            print(f"    {'─'*58}")
            for cn in KNOWN_CLUSTERS:
                cl = cluster_latency[cn]
                reg = CLUSTER_REGIONS.get(cn, "")
                print(f"    {cn+' ('+reg+')':<17} {cl['requests']:>7}  "
                      f"{cl['p50_ms']:>6.0f}ms  {cl['p90_ms']:>6.0f}ms  "
                      f"{cl['p95_ms']:>6.0f}ms  {cl['p99_ms']:>6.0f}ms  "
                      f"{cl['slo_pct']:>6.1f}%")
        else:
            kv("p95 globale flat:", "N/A  (cluster_latency CSV senza request)")
    else:
        kv("p95 globale flat:", "N/A  (cluster_latency CSV non trovato)")

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

def generate_page_scaling(fe_ts: dict, stats: dict, output_dir: Path, prefix: str = "dmos"):
    """
    Thesis Page 1 — Scaling & Resource Allocation.
    Plots: Traffic+Predicted | Replica distribution % | Provisioning ratio | TtS proactive%.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DMOS Analysis — Scaling & Resource Allocation', fontsize=14, fontweight='bold')
    timestamps = fe_ts["timestamps"]

    # ── [0,0] Traffic + Predicted ──
    ax = axes[0, 0]
    ax.plot(timestamps, fe_ts["traffic"], color=COLORS['traffic'], lw=2, label='Actual Traffic')
    if any(p > 0 for p in fe_ts["predicted"]):
        ax.plot(timestamps, fe_ts["predicted"], '--', color=COLORS['predicted'],
                lw=1.5, alpha=0.8, label='Predicted Traffic')
    ax.set_ylabel('Traffic (req/s)')
    ax.set_title('Traffic & Prediction Over Time')
    ax.legend(fontsize=8)
    _fmt_time(ax)

    # ── [0,1] Replica distribution % per cluster (normalized stacked area) ──
    ax = axes[0, 1]
    c1 = fe_ts["per_cluster_replicas"]["cluster1"]
    c2 = fe_ts["per_cluster_replicas"]["cluster2"]
    c3 = fe_ts["per_cluster_replicas"]["cluster3"]
    totals = [a + b + c for a, b, c in zip(c1, c2, c3)]

    def _pct(vals, tots):
        return [v / t * 100 if t > 0 else 0 for v, t in zip(vals, tots)]

    p1, p2, p3 = _pct(c1, totals), _pct(c2, totals), _pct(c3, totals)
    ax.stackplot(timestamps, p1, p2, p3,
                 labels=[f'cluster1 ({CLUSTER_REGIONS.get("cluster1","DE")})',
                         f'cluster2 ({CLUSTER_REGIONS.get("cluster2","FR")})',
                         f'cluster3 ({CLUSTER_REGIONS.get("cluster3","PL")})'],
                 colors=[COLORS['cluster1'], COLORS['cluster2'], COLORS['cluster3']],
                 alpha=0.75)
    ax.axhline(y=33.3, color='gray', ls='--', alpha=0.5, lw=1, label='Equal share (33%)')
    ax.set_ylim(0, 100)
    ax.set_ylabel('Replica Share (%)')
    ax.set_title('Replica Distribution % per Cluster')
    ax.legend(fontsize=7)
    _fmt_time(ax)

    # ── [1,0] Provisioning ratio (over/under-prov timeline) ──
    ax = axes[1, 0]
    prov = stats.get("provisioning", {})
    ratios = prov.get("ratios", [])
    if ratios:
        min_len = min(len(timestamps), len(ratios))
        ratio_ts = timestamps[:min_len]
        ratios_plot = ratios[:min_len]
        plot_ts     = [t for t, r in zip(ratio_ts, ratios_plot) if r > 0]
        plot_ratios = [r for r in ratios_plot if r > 0]
        if plot_ts:
            ax.plot(plot_ts, plot_ratios, color=COLORS['overprov'], lw=1.5, alpha=0.9)
            ax.fill_between(plot_ts, 1.0, plot_ratios,
                            where=[r > 1.0 for r in plot_ratios],
                            color=COLORS['overprov'], alpha=0.25, label='Over-provisioned')
            ax.fill_between(plot_ts, plot_ratios, 1.0,
                            where=[r < 1.0 for r in plot_ratios],
                            color=COLORS['underprov'], alpha=0.25, label='Under-provisioned')
    ax.axhline(y=1.0, color='black', ls='-', lw=1)
    ax.axhline(y=1.15, color='green', ls='--', alpha=0.5, lw=1, label='Ideal (1.15×)')
    over_p  = prov.get('over_pct', 0)
    under_p = prov.get('under_pct', 0)
    ideal_p = prov.get('ideal_pct', 0)
    ax.text(0.02, 0.97,
            f"Over: {over_p:.1f}%   Under: {under_p:.1f}%   Ideal: {ideal_p:.1f}%",
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_ylabel('Provisioning Ratio  (capacity / demand)')
    ax.set_title('Resource Utilization — Over/Under-Provisioning')
    ax.legend(fontsize=7)
    _fmt_time(ax)

    # ── [1,1] Time to Scale — proactive vs reactive bar chart ──
    ax = axes[1, 1]
    tts = stats.get("tts", {})
    tts_events = tts.get("events", [])
    if tts_events:
        pro_times = [e["cross_time"] for e in tts_events if e["proactive"]]
        pro_vals  = [e["tts_seconds"] for e in tts_events if e["proactive"]]
        rea_times = [e["cross_time"] for e in tts_events if not e["proactive"]]
        rea_vals  = [e["tts_seconds"] for e in tts_events if not e["proactive"]]
        if pro_times:
            ax.bar(pro_times, pro_vals, width=0.0012, color='#27ae60', alpha=0.8,
                   label=f'Proactive ({len(pro_times)})')
        if rea_times:
            ax.bar(rea_times, rea_vals, width=0.0012, color='#e74c3c', alpha=0.8,
                   label=f'Reactive ({len(rea_times)})')
        ax.axhline(y=0, color='black', ls='-', lw=0.8)
        pro_pct = tts.get("proactive_pct", 0)
        color_pct = '#27ae60' if pro_pct > 50 else '#e74c3c'
        n_total = len(tts_events)
        n_pro   = tts.get("proactive_count", 0)
        ax.text(0.02, 0.97,
                f"Proactive: {pro_pct:.0f}%  ({n_pro}/{n_total} events)",
                transform=ax.transAxes, fontsize=9, fontweight='bold', va='top',
                color=color_pct,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No scaling events detected',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_ylabel('Time to Scale (s)   [negative = before spike]')
    ax.set_title('Time to Scale — Proactive vs Reactive')
    _fmt_time(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / f"{prefix}_page1_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {path.name}")


def generate_page_qos(fe_ts: dict, locust_ts: dict, locust_cluster: dict,
                      stats: dict, output_dir: Path, prefix: str = "dmos",
                      cluster_latency: dict = None):
    """
    Thesis Page 2 — Quality of Service & Fairness.
    Plots: p95 per cluster | SLO violation rate | Jain FI | KPI summary table.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DMOS Analysis — Quality of Service & Fairness', fontsize=14, fontweight='bold')

    # ── [0,0] p95 latency per cluster over time ──
    ax = axes[0, 0]
    has_data = False
    for cn in KNOWN_CLUSTERS:
        ts_c  = locust_cluster[cn]["timestamps"]
        p95_c = locust_cluster[cn]["p95_ms"]
        if ts_c and any(v > 0 for v in p95_c):
            ax.plot(ts_c, p95_c, lw=2, color=COLORS[cn],
                    label=f"{cn} ({CLUSTER_REGIONS.get(cn, '')})")
            has_data = True
    if has_data:
        ax.axhline(y=1000, color='red', ls=':', lw=1.5, alpha=0.7, label='SLO 1 s')
        ax.legend(fontsize=7)
        _fmt_time(ax)
    else:
        ax.text(0.5, 0.5, 'No per-cluster latency data\n(run locustfile_multiingress.py)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_ylabel('p95 Latency (ms)')
    ax.set_title('p95 Latency per Cluster Over Time')

    # ── [0,1] SLO violation rate per cluster ──
    ax = axes[0, 1]
    has_slo = False
    for cn in KNOWN_CLUSTERS:
        ts_c  = locust_cluster[cn]["timestamps"]
        slo_c = locust_cluster[cn]["slo_pct"]
        if ts_c and any(v > 0 for v in slo_c):
            ax.plot(ts_c, slo_c, lw=2, color=COLORS[cn],
                    label=f"{cn} ({CLUSTER_REGIONS.get(cn, '')})")
            has_slo = True
    if has_slo:
        ax.axhline(y=5, color='orange', ls='--', alpha=0.7, lw=1.5, label='5% threshold')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
        _fmt_time(ax)
    else:
        ax.text(0.5, 0.5, 'SLO data not available\n(update locustfile and re-run)',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_ylabel('SLO Violation Rate (%)')
    ax.set_title('SLO Violation Rate per Cluster  (req > 1 s)')

    # ── [1,0] Jain Fairness Index over time ──
    ax = axes[1, 0]
    jain_ts, jain_vals = stats.get("jain_data", ([], []))
    if jain_ts:
        ax.plot(jain_ts, jain_vals, color='#16a085', lw=2)
        ax.fill_between(jain_ts, 0, jain_vals, alpha=0.15, color='#16a085')
        ax.axhline(y=1.0, color='green',  ls='--', alpha=0.6, lw=1, label='Perfect (1.0)')
        ax.axhline(y=1/3, color='red',    ls='--', alpha=0.5, lw=1, label='Worst (0.333)')
        nonzero_j = [j for j in jain_vals if j > 0]
        if nonzero_j:
            mean_j = sum(nonzero_j) / len(nonzero_j)
            ax.text(0.02, 0.05, f"Mean Jain: {mean_j:.3f}",
                    transform=ax.transAxes, fontsize=9, fontweight='bold',
                    color='#16a085',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
        _fmt_time(ax)
    else:
        ax.text(0.5, 0.5, 'No Jain data available',
                transform=ax.transAxes, ha='center', va='center', fontsize=10, color='gray')
    ax.set_ylabel('Jain Fairness Index  (n = 3 clusters)')
    ax.set_title('Cluster Fairness — Jain Index Over Time')

    # ── [1,1] KPI Summary table (6 thesis metrics) ──
    ax = axes[1, 1]
    ax.axis('off')
    prov = stats.get("provisioning", {})
    tts  = stats.get("tts", {})
    jain_ts_v, jain_vals_v = stats.get("jain_data", ([], []))
    nonzero_j = [j for j in jain_vals_v if j > 0]
    jain_mean = sum(nonzero_j) / len(nonzero_j) if nonzero_j else 0

    # Global p95: mean of per-cluster p95 values from Locust CSV (fallback: global locust)
    all_p95 = []
    for cn in KNOWN_CLUSTERS:
        all_p95.extend(v for v in locust_cluster[cn]["p95_ms"] if v > 0)
    if all_p95:
        global_p95_mean: Optional[float] = sum(all_p95) / len(all_p95)
    else:
        rt_vals = [v for v in locust_ts["p95_rt"] if v > 0]
        global_p95_mean = sum(rt_vals) / len(rt_vals) if rt_vals else None

    # SLO violation: mean across all clusters/windows
    all_slo = []
    for cn in KNOWN_CLUSTERS:
        all_slo.extend(v for v in locust_cluster[cn]["slo_pct"] if v >= 0)
    global_slo_mean: Optional[float] = sum(all_slo) / len(all_slo) if all_slo else None

    over_p  = prov.get("over_pct", 0)
    under_p = prov.get("under_pct", 0)
    pro_pct = tts.get("proactive_pct", 0)
    n_pro   = tts.get("proactive_count", 0)
    n_total_events = n_pro + tts.get("reactive_count", 0)

    def _slo_status(v: Optional[float]) -> str:
        if v is None: return "—"
        return "✅" if v < 5 else "⚠️" if v < 15 else "❌"

    # p95 flat globale (Romano/k6 style)
    p95_flat: Optional[float] = compute_global_p95_flat(cluster_latency) if cluster_latency else None

    kpis = [
        ("Metric", "Value", "✓"),
        ("p95 latency (windowed mean)",
         f"{global_p95_mean:.0f} ms" if global_p95_mean else "N/A",
         "✅" if global_p95_mean and global_p95_mean < 1000 else ("⚠️" if global_p95_mean else "—")),
        ("p95 latency (flat / k6-style)",
         f"{p95_flat:.0f} ms" if p95_flat is not None else "N/A",
         "✅" if p95_flat is not None and p95_flat < 1000 else ("⚠️" if p95_flat is not None and p95_flat < 2000 else ("❌" if p95_flat else "—"))),
        ("SLO violation rate (mean)",
         f"{global_slo_mean:.1f}%" if global_slo_mean is not None else "N/A",
         _slo_status(global_slo_mean)),
        ("Jain Fairness Index (mean)",
         f"{jain_mean:.3f}   (1.0 = perfect)",
         "✅" if jain_mean > 0.85 else ("⚠️" if jain_mean > 0 else "—")),
        ("Over-provisioned time",
         f"{over_p:.1f}% of snapshots",
         "✅" if over_p < 30 else "⚠️"),
        ("Under-provisioned time",
         f"{under_p:.1f}% of snapshots",
         "✅" if under_p < 10 else "⚠️"),
        ("Proactive scaling %",
         f"{pro_pct:.0f}%   ({n_pro}/{n_total_events} scale-ups)",
         "✅" if pro_pct > 50 else "⚠️"),
    ]

    table = ax.table(
        cellText=kpis,
        cellLoc='center',
        loc='center',
        colWidths=[0.44, 0.34, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.75)

    # Style header row
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Style data rows (alternating)
    for i in range(1, len(kpis)):
        for j in range(3):
            cell = table[i, j]
            cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    ax.set_title('KPI Summary — Thesis Metrics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / f"{prefix}_page2_qos.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Export
# ═══════════════════════════════════════════════════════════════════════════════

def export_analysis_json(data: List[dict], fe_ts: dict, locust_ts: dict, stats: dict,
                         locust_cluster: dict, output_file: Path,
                         cluster_latency: dict = None):
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
            if k not in ("ratios", "active_ratios")
        },
        "time_to_scale": {
            k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in stats.get("tts", {}).items()
            if k not in ("events", "demand_events", "threshold_events")
        },
        "oscillation": {
            k: v for k, v in stats.get("oscillation", {}).items()
            if k != "window_changes_timeline"
        },
        "jain_fairness": {
            "mean": 0,
            "min": 0,
        },
        "slo_violation": {
            "global_mean_pct": None,
        },
    }

    jain_ts, jain_vals = stats.get("jain_data", ([], []))
    nonzero_jain = [j for j in jain_vals if j > 0]
    if nonzero_jain:
        summary["jain_fairness"]["mean"] = round(sum(nonzero_jain) / len(nonzero_jain), 4)
        summary["jain_fairness"]["min"] = round(min(nonzero_jain), 4)

    # Per-cluster p95 and SLO stats from Locust CSV
    per_cluster_p95: dict = {}
    per_cluster_slo: dict = {}
    all_slo_vals: list = []
    all_p95_vals: list = []
    for cn in KNOWN_CLUSTERS:
        c_p95 = [v for v in locust_cluster[cn]["p95_ms"] if v > 0]
        c_slo = [v for v in locust_cluster[cn]["slo_pct"] if v >= 0]
        per_cluster_p95[cn] = round(sum(c_p95) / len(c_p95), 1) if c_p95 else None
        per_cluster_slo[cn] = round(sum(c_slo) / len(c_slo), 2) if c_slo else None
        all_p95_vals.extend(c_p95)
        all_slo_vals.extend(c_slo)

    summary["slo_violation"]["global_mean_pct"] = (
        round(sum(all_slo_vals) / len(all_slo_vals), 2) if all_slo_vals else None
    )
    summary["slo_violation"]["per_cluster_mean_pct"] = per_cluster_slo

    # p95 response time
    p95_flat = compute_global_p95_flat(cluster_latency) if cluster_latency else None
    if all_p95_vals:
        summary["response_time"] = {
            "p95_global_mean_ms": round(sum(all_p95_vals) / len(all_p95_vals), 1),
            "p95_global_flat_ms": round(p95_flat, 1) if p95_flat is not None else None,
            "p95_per_cluster_mean_ms": per_cluster_p95,
        }
        if cluster_latency:
            summary["response_time"]["p95_per_cluster_flat_ms"] = {
                cn: round(cluster_latency[cn]["p95_ms"], 1)
                for cn in KNOWN_CLUSTERS if cluster_latency[cn]["requests"] > 0
            }
    elif locust_ts["p95_rt"]:
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
def _build_output_paths(input_file: str) -> tuple:
    """
    Estrae scenario e prefix dal nome del file input.
    Ritorna (output_dir, file_prefix).

    Nuovo formato:  092224_20260217_flash_crowd.jsonl → ("results/flash_crowd", "092224_20260217")
    Vecchio formato: metrics_20260217_092224.jsonl     → ("results/test", "092224_20260217")
    """
    input_path = Path(input_file)
    stem = input_path.stem          # es: "092224_20260217_flash_crowd"
    parent = input_path.parent      # es: "results"
    parts = stem.split("_")

    # Nuovo formato: HHMMSS_YYYYMMDD_scenario[_parts]
    if len(parts) >= 3 and len(parts[0]) == 6 and parts[0].isdigit() and len(parts[1]) == 8 and parts[1].isdigit():
        time_str = parts[0]
        date_str = parts[1]
        scenario = "_".join(parts[2:])
        prefix = f"{time_str}_{date_str}"
    # Vecchio formato: metrics_YYYYMMDD_HHMMSS
    elif parts[0] in ("metrics", "metrics_timeseries") and len(parts) >= 3:
        date_str = parts[-2] if len(parts[-2]) == 8 else parts[1]
        time_str = parts[-1] if len(parts[-1]) == 6 else parts[2]
        scenario = "test"
        prefix = f"{time_str}_{date_str}"
    else:
        scenario = "test"
        prefix = stem

    output_dir = parent / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, prefix, scenario

def analyze(filepath: str):
    """Main analysis pipeline."""
    print("=" * 78)
    print("🔬 DMOS Comprehensive Test Analysis v2")
    print("=" * 78)
    print(f"\n  Input: {filepath}\n")

    # Calcola output paths
    output_dir, prefix, scenario = _build_output_paths(filepath)
    print(f"  📁 Output: {output_dir}/")
    print(f"  📝 Prefix: {prefix}\n")

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
        fe_ts["traffic"], fe_ts["total_replicas"], SERVICE_CAPACITY["frontend"],
        min_replicas_total=3   # 1 replica × 3 cluster = floor HA minimo
        # (era 6, ma 6×45=270 > max_traffic=179 → ratio sempre <1 = artefatto)
    )
    stats["tts"] = compute_time_to_scale(fe_ts["timestamps"], fe_ts["traffic"], fe_ts["total_replicas"],
                                          predicted=fe_ts.get("predicted"), capacity_per_replica=SERVICE_CAPACITY["frontend"])
    stats["oscillation"] = compute_scaling_oscillation(fe_ts["total_replicas"], fe_ts["timestamps"])
    stats["jain_data"] = compute_jain_fairness(data, "frontend")

    # Load per-cluster Locust CSV (prodotto da locustfile_multiingress.py)
    locust_cluster = load_locust_cluster_csv(scenario=scenario)

    # Load cluster_latency CSV (aggregato sull'intero test — stile k6/Romano)
    cluster_latency = load_locust_cluster_latency_csv(scenario=scenario)

    # Console report
    print_report(data, fe_ts, locust_ts, stats, cluster_latency=cluster_latency)

    # Generate plots (2 thesis pages)
    print("  Generating plots...")
    generate_page_scaling(fe_ts, stats, output_dir, prefix)
    generate_page_qos(fe_ts, locust_ts, locust_cluster, stats, output_dir, prefix,
                      cluster_latency=cluster_latency)

    # Export JSON
    json_path = output_dir / f"{prefix}_analysis.json"
    export_analysis_json(data, fe_ts, locust_ts, stats, locust_cluster, json_path,
                         cluster_latency=cluster_latency)

    print(f"\n{'=' * 78}")
    print(f"✅ Analysis complete!")
    print(f"   Output:   {output_dir}/")
    print(f"   Plots:    {prefix}_page1_scaling.png")
    print(f"             {prefix}_page2_qos.png")
    print(f"   JSON:     {json_path.name}")
    print(f"{'=' * 78}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Auto-find latest file
        results_dir = Path("results")
        # Prefer JSONL
        files = list(results_dir.glob("*.jsonl"))
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