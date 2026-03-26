"""
DMOS Comprehensive Metrics Collector v2
=======================================
Collects:
  1. DMOS internal metrics (all services, all clusters) via /metrics endpoint
  2. Locust stats CSV (response times p50/p95/p99, failure rate) via Locust REST API
  3. Scheduling duration from histogram
  
Outputs a structured JSON-lines file (.jsonl) + legacy txt for backward compat.

Usage:
  python collect_metrics_simple.py [duration_sec] [--locust-host http://localhost:8089]
"""

import requests
import time
import json
import csv
import io
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

DMOS_METRICS_URL = "http://localhost:9090/metrics"
LOCUST_API_URL   = "http://localhost:8089"   # Locust web UI
OUTPUT_DIR       = Path("results")
SCRAPE_INTERVAL  = 15   # seconds

# Known services from config/services.yaml
KNOWN_SERVICES = [
    "frontend", "cartservice", "productcatalogservice",
    "checkoutservice", "recommendationservice"
]
KNOWN_CLUSTERS = ["cluster1", "cluster2", "cluster3"]


# ─── DMOS Metrics Parser ─────────────────────────────────────────────────────

def scrape_dmos_metrics() -> dict:
    """Scrape and parse all DMOS Prometheus metrics into structured dict."""
    try:
        r = requests.get(DMOS_METRICS_URL, timeout=5)
        raw_text = r.text
    except Exception as e:
        print(f"  ⚠ DMOS metrics error: {e}")
        return {"raw": "", "parsed": {}}

    parsed = {
        "actual_traffic": {},       # {service: value}
        "current_replicas": {},     # {(cluster,service): value}
        "target_replicas": {},      # {(cluster,service): value}
        "cluster_score": {},        # {(cluster,service): value}
        "predicted_traffic": {},    # {(cluster,service): value}
        "scaling_events": {},       # {(cluster,service,action): value}
        "scheduling_duration_sum": {},   # {service: value}
        "scheduling_duration_count": {}, # {service: value}
    }

    for line in raw_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            # dmos_actual_traffic{service="frontend"} 123.45
            if 'dmos_actual_traffic{' in line:
                m = re.search(r'service="([^"]+)"', line)
                v = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if m and v:
                    parsed["actual_traffic"][m.group(1)] = float(v.group(1))

            # dmos_current_replicas{cluster="cluster1",service="frontend"} 3.0
            elif 'dmos_current_replicas{' in line:
                mc = re.search(r'cluster="([^"]+)"', line)
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if mc and ms and v:
                    parsed["current_replicas"][(mc.group(1), ms.group(1))] = int(float(v.group(1)))

            elif 'dmos_target_replicas{' in line:
                mc = re.search(r'cluster="([^"]+)"', line)
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if mc and ms and v:
                    parsed["target_replicas"][(mc.group(1), ms.group(1))] = int(float(v.group(1)))

            elif 'dmos_cluster_score{' in line:
                mc = re.search(r'cluster="([^"]+)"', line)
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if mc and ms and v:
                    parsed["cluster_score"][(mc.group(1), ms.group(1))] = float(v.group(1))

            elif 'dmos_predicted_traffic{' in line:
                mc = re.search(r'cluster="([^"]+)"', line)
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if mc and ms and v:
                    parsed["predicted_traffic"][(mc.group(1), ms.group(1))] = float(v.group(1))

            elif 'dmos_scaling_events_total{' in line:
                ma = re.search(r'action="([^"]+)"', line)
                mc = re.search(r'cluster="([^"]+)"', line)
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if ma and mc and ms and v:
                    parsed["scaling_events"][(mc.group(1), ms.group(1), ma.group(1))] = int(float(v.group(1)))

            elif 'dmos_scheduling_duration_seconds_sum{' in line:
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if ms and v:
                    parsed["scheduling_duration_sum"][ms.group(1)] = float(v.group(1))

            elif 'dmos_scheduling_duration_seconds_count{' in line:
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if ms and v:
                    parsed["scheduling_duration_count"][ms.group(1)] = int(float(v.group(1)))

        except (ValueError, AttributeError):
            continue

    return {"raw": raw_text, "parsed": parsed}


# ─── Locust Stats Fetcher ────────────────────────────────────────────────────

def scrape_locust_stats(locust_url: str) -> dict:
    """
    Fetch Locust statistics via REST API.
    Returns aggregated stats: avg/median/p95/p99 response time, rps, failure count.
    """
    stats = {
        "available": False,
        "total_rps": 0.0,
        "total_fail_ratio": 0.0,
        "avg_response_time_ms": 0.0,
        "median_response_time_ms": 0.0,
        "p95_response_time_ms": 0.0,
        "p99_response_time_ms": 0.0,
        "current_users": 0,
        "per_endpoint": {}
    }

    try:
        # /stats/requests returns JSON with per-endpoint stats
        r = requests.get(f"{locust_url}/stats/requests", timeout=3)
        if r.status_code != 200:
            return stats

        data = r.json()
        stats["available"] = True
        stats["current_users"] = data.get("user_count", 0)

        # [FIX] When Locust has zero active users, force rps=0.
        # Without this, Locust's rolling average keeps returning the last
        # measured rps (e.g. 24.5) for 30-60s after users_count drops to 0.
        # This causes analyze_test_complete to report non-zero throughput
        # during the idle phase, inflating success metrics artificially.
        _zero_users = (stats["current_users"] == 0)

        # In Locust 2.x, aggregated stats are in stats_total (not in stats[])
        agg = data.get("stats_total", {})
        if agg:
            raw_rps = agg.get("current_rps", 0)
            stats["total_rps"] = 0.0 if _zero_users else raw_rps
            stats["avg_response_time_ms"] = agg.get("avg_response_time", 0)
            stats["median_response_time_ms"] = agg.get("median_response_time", 0)
            fail_ratio = 0
            if agg.get("num_requests", 0) > 0:
                fail_ratio = agg["num_failures"] / agg["num_requests"]
            stats["total_fail_ratio"] = fail_ratio

        # p95/p99 are available directly in the root JSON (Locust 2.x)
        p95_direct = data.get("current_response_time_percentile_95", 0)
        p99_direct = data.get("current_response_time_percentile_99", 0)
        if p95_direct:
            stats["p95_response_time_ms"] = p95_direct
        if p99_direct:
            stats["p99_response_time_ms"] = p99_direct

        # Per-endpoint stats
        for entry in data.get("stats", []):
            name = entry.get("name", "")
            method = entry.get("method", "")
            endpoint_data = {
                "num_requests": entry.get("num_requests", 0),
                "num_failures": entry.get("num_failures", 0),
                "avg_response_time": entry.get("avg_response_time", 0),
                "median_response_time": entry.get("median_response_time", 0),
                "current_rps": entry.get("current_rps", 0),
                "current_fail_per_sec": entry.get("current_fail_per_sec", 0),
            }
            if name and name != "Aggregated":
                stats["per_endpoint"][f"{method} {name}"] = endpoint_data

        # Fetch percentiles from /stats/requests (response_time_percentiles)
        # These are in the entries as well but need the extended stats
        # Use /stats/report endpoint for p95/p99
        try:
            r2 = requests.get(f"{locust_url}/stats/requests", timeout=3)
            data2 = r2.json()
            for entry in data2.get("stats", []):
                if entry.get("name") == "Aggregated":
                    # Locust provides percentiles in response_times dict
                    percentiles = entry.get("response_times", {})
                    # These are {ms: count} pairs, but current_response_time_percentile
                    # is available directly
                    pass
        except Exception:
            pass

        # Alternative: get percentiles from /percentile endpoint
        try:
            r3 = requests.get(f"{locust_url}/stats/requests", timeout=3)
            data3 = r3.json()
            # Check for percentile_response_time
            for entry in data3.get("stats", []):
                if entry.get("name") == "Aggregated":
                    # Some Locust versions expose these directly
                    if "avg_response_time" in entry:
                        stats["avg_response_time_ms"] = entry["avg_response_time"]
            
            # Get current percentiles from percentile endpoint
            r4 = requests.get(f"{locust_url}/stats/requests/csv", timeout=3)
            if r4.status_code == 200:
                reader = csv.DictReader(io.StringIO(r4.text))
                for row in reader:
                    if row.get("Name", "").strip() == "Aggregated":
                        for key in ["50%", "66%", "75%", "80%", "90%", "95%", "98%", "99%", "99.9%", "99.99%", "100%"]:
                            if key in row and row[key]:
                                try:
                                    val = float(row[key])
                                    if key == "50%":
                                        stats["median_response_time_ms"] = val
                                    elif key == "95%":
                                        stats["p95_response_time_ms"] = val
                                    elif key == "99%":
                                        stats["p99_response_time_ms"] = val
                                except ValueError:
                                    pass
        except Exception:
            pass

    except requests.ConnectionError:
        # Locust not running — that's fine
        pass
    except Exception as e:
        print(f"  ⚠ Locust stats error: {e}")

    return stats


# ─── Snapshot Builder ────────────────────────────────────────────────────────

def build_snapshot(dmos: dict, locust: dict, timestamp: str) -> dict:
    """
    Build a unified snapshot combining DMOS + Locust data.
    All services, all clusters, all metrics in one JSON object.
    Robust to empty/failed DMOS scrape: all missing fields default to 0.
    """
    p = dmos.get("parsed", {})  # {} when DMOS unreachable — don't crash

    # Per-service metrics
    services_data = {}
    for svc in KNOWN_SERVICES:
        svc_data = {
            "actual_traffic": p.get("actual_traffic", {}).get(svc, 0.0),
            "predicted_traffic_total": 0.0,
            "clusters": {}
        }

        total_replicas = 0
        for cluster in KNOWN_CLUSTERS:
            key = (cluster, svc)
            cr = p.get("current_replicas", {}).get(key, 0)
            tr = p.get("target_replicas", {}).get(key, 0)
            sc = p.get("cluster_score", {}).get(key, 0.0)
            pt = p.get("predicted_traffic", {}).get(key, 0.0)
            su = p.get("scaling_events", {}).get((cluster, svc, "scale_up"), 0)
            sd = p.get("scaling_events", {}).get((cluster, svc, "scale_down"), 0)

            svc_data["clusters"][cluster] = {
                "current_replicas": cr,
                "target_replicas": tr,
                "score": sc,
                "predicted_traffic": pt,
                "scale_up_events_cumulative": su,
                "scale_down_events_cumulative": sd,
            }
            total_replicas += cr
            svc_data["predicted_traffic_total"] += pt

        svc_data["total_replicas"] = total_replicas

        # Scheduling duration (average from histogram)
        sched_sum = p.get("scheduling_duration_sum", {}).get(svc, 0.0)
        sched_count = p.get("scheduling_duration_count", {}).get(svc, 0)
        svc_data["avg_scheduling_duration_s"] = (
            sched_sum / sched_count if sched_count > 0 else 0.0
        )
        svc_data["scheduling_invocations"] = sched_count

        services_data[svc] = svc_data

    snapshot = {
        "timestamp": timestamp,
        "dmos": {
            "services": services_data,
        },
        "locust": locust,
    }

    return snapshot


# ─── Console Summary ─────────────────────────────────────────────────────────

def print_summary(snap: dict, iteration: int, total: int):
    """Print a detailed multi-line summary with per-cluster info."""
    ts = snap["timestamp"]
    services = snap["dmos"]["services"]
    
    fe = services.get("frontend", {})
    fe_traffic = fe.get("actual_traffic", 0)
    fe_predicted = fe.get("predicted_traffic_total", 0)
    fe_total = fe.get("total_replicas", 0)
    
    # Frontend per-cluster
    fe_parts = []
    for c in KNOWN_CLUSTERS:
        cd = fe.get("clusters", {}).get(c, {})
        r = cd.get("current_replicas", 0)
        fe_parts.append(str(r))
    fe_cluster_str = "/".join(fe_parts)
    
    # Locust data
    loc = snap.get("locust", {})
    if loc.get("available"):
        p95 = loc.get("p95_response_time_ms", 0)
        rps = loc.get("total_rps", 0)
        users = loc.get("current_users", 0)
        fail = loc.get("total_fail_ratio", 0)
        locust_str = f"p95={p95:.0f}ms rps={rps:.1f} users={users} fail={fail:.1%}"
    else:
        locust_str = "locust: waiting..."
    
    # Backend services summary: name(total: c1/c2/c3)
    backend_parts = []
    for svc_name in KNOWN_SERVICES:
        if svc_name == "frontend":
            continue
        svc = services.get(svc_name, {})
        svc_total = svc.get("total_replicas", 0)
        parts = []
        for c in KNOWN_CLUSTERS:
            cd = svc.get("clusters", {}).get(c, {})
            parts.append(str(cd.get("current_replicas", 0)))
        # Short name: cartservice → cart, productcatalogservice → catalog, etc.
        short = svc_name.replace("service", "").replace("productcatalog", "catalog")
        backend_parts.append(f"{short}={svc_total}({'/'.join(parts)})")
    backend_str = " ".join(backend_parts)
    
    # Print
    print(
        f"  [{iteration:3d}/{total}] {ts[11:19]} | "
        f"FE: {fe_traffic:5.1f}rps→{fe_predicted:5.1f}pred "
        f"replicas={fe_total}(c1/c2/c3={fe_cluster_str}) | "
        f"{locust_str}"
    )
    if iteration == 1 or iteration % 5 == 0:
        # Show backend detail every 5 snapshots
        print(f"           backends: {backend_str}")


# ─── Main Collector Loop ─────────────────────────────────────────────────────

def run_collector(duration_seconds: int = 300, locust_url: str = LOCUST_API_URL, scenario: str = "test", skip_dmos: bool = False):
    """
    Main collection loop.
    Outputs:
      - results/metrics_YYYYMMDD_HHMMSS.jsonl  (structured, one JSON per line)
      - results/metrics_timeseries_YYYYMMDD_HHMMSS.txt  (legacy raw format)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    time_str = datetime.now().strftime("%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")
    jsonl_file = OUTPUT_DIR / f"{time_str}_{date_str}_{scenario}.jsonl"
    legacy_file = OUTPUT_DIR / f"{time_str}_{date_str}_{scenario}.txt"

    iterations = duration_seconds // SCRAPE_INTERVAL

    print("=" * 80)
    print("🚀 DMOS Comprehensive Metrics Collector v2")
    print("=" * 80)
    print(f"  Duration:     {duration_seconds} seconds ({duration_seconds // 60}m {duration_seconds % 60}s)")
    print(f"  Interval:     {SCRAPE_INTERVAL} seconds")
    print(f"  Iterations:   {iterations}")
    print(f"  DMOS URL:     {DMOS_METRICS_URL}")
    print(f"  Locust URL:   {locust_url}")
    print(f"  Output JSONL: {jsonl_file}")
    print(f"  Output TXT:   {legacy_file}")
    print("-" * 80)
    print("  Legend:")
    print("    FE: X rps → Y pred   = Prometheus traffic (actual → DMOS predicted)")
    print("    replicas=N(c1/c2/c3) = frontend replicas total (per cluster)")
    print("    rps=X                = Locust measured throughput (actual end-user)")
    print("    backends: svc=N(x/y/z) = backend replicas total (per cluster)")
    print("    FE traffic ≠ Locust rps: FE is from Prometheus (network bytes estimate),")
    print("                             Locust rps is actual HTTP requests measured")
    print("=" * 80)
    print()

    # ── Wait for DMOS to be reachable (max 60s) ──
    if skip_dmos:
        print("  DMOS endpoint skipped (--no-dmos) — baseline/OFF run, DMOS metrics = 0")
    else:
        print("  Waiting for DMOS metrics endpoint...", end="", flush=True)
        for _attempt in range(20):
            try:
                r = __import__("requests").get(DMOS_METRICS_URL, timeout=3)
                if r.status_code == 200:
                    print(" ✅ ready")
                    break
            except Exception:
                pass
            print(".", end="", flush=True)
            time.sleep(3)
        else:
            print(" ⚠ DMOS not reachable after 60s — collecting anyway (zeros until it starts)")

    # Auto-stop con cooldown post-test:
    #   1. Rileva fine Locust: users=0 e rps≈0 per LOCUST_IDLE_DETECT campioni
    #   2. Continua a raccogliere per COOLDOWN_SAMPLES campioni extra (per vedere
    #      DMOS scalare giù — ogni ciclo di scheduling è 30s, quindi 8×15s = 120s
    #      copre ~4 cicli di scale-down)
    #   3. Poi ferma
    LOCUST_IDLE_DETECT  = 3   # campioni per confermare che Locust è fermo
    COOLDOWN_SAMPLES    = 8   # campioni extra dopo rilevamento idle (= 120s extra)
    _locust_ever_active = False
    _locust_idle_streak = 0
    _cooldown_remaining = -1  # -1 = cooldown non ancora iniziato

    with open(jsonl_file, 'w') as fj, open(legacy_file, 'w') as ft:
        for i in range(iterations):
            timestamp = datetime.now().isoformat()

            # Scrape both sources
            dmos = {"raw": "", "parsed": {}} if skip_dmos else scrape_dmos_metrics()
            locust = scrape_locust_stats(locust_url)

            # Build unified snapshot
            snap = build_snapshot(dmos, locust, timestamp)

            # Write structured JSONL
            fj.write(json.dumps(snap) + "\n")
            fj.flush()

            # Write legacy TXT (raw Prometheus)
            ft.write(f"\n{'='*70}\n")
            ft.write(f"Timestamp: {timestamp}\n")
            ft.write(f"{'='*70}\n")
            ft.write(dmos.get("raw", ""))
            ft.write("\n")
            ft.flush()

            # Console
            print_summary(snap, i + 1, iterations)

            # ── Auto-stop con cooldown post-test ──────────────────────
            loc = snap.get("locust", {})
            if _cooldown_remaining > 0:
                # Siamo in fase di cooldown: aspettiamo che DMOS scala giù
                _cooldown_remaining -= 1
                remaining_skip = iterations - i - 1 - _cooldown_remaining
                print(f"           [cooldown] {_cooldown_remaining * SCRAPE_INTERVAL}s rimasti "
                      f"— osservazione scale-down DMOS "
                      f"({remaining_skip} iter risparmiate al termine)")
            elif _cooldown_remaining == 0:
                # Cooldown finito → ferma il collector
                remaining = iterations - i - 1
                print(f"\n  [auto-stop] Cooldown completato — "
                      f"fine collezione ({remaining} iter risparmiate)")
                break
            elif loc.get("available"):
                users = loc.get("current_users", 0)
                rps   = loc.get("total_rps", 0.0)
                if users > 0 or rps > 1.0:
                    _locust_ever_active = True
                    _locust_idle_streak = 0
                elif _locust_ever_active:
                    # Locust era attivo e ora è fermo (users=0, rps≈0)
                    _locust_idle_streak += 1
                    if _locust_idle_streak >= LOCUST_IDLE_DETECT:
                        # Test confermato finito → avvia cooldown
                        _cooldown_remaining = COOLDOWN_SAMPLES
                        print(f"\n  [cooldown] Locust terminato — "
                              f"raccolta per {COOLDOWN_SAMPLES * SCRAPE_INTERVAL}s extra "
                              f"per osservare scale-down DMOS...")

            if i < iterations - 1:
                time.sleep(SCRAPE_INTERVAL)

    print()
    print("=" * 80)
    print("  Collection complete!")
    print(f"  JSONL: {jsonl_file}")
    print(f"  TXT:   {legacy_file}")
    print("=" * 80)

    return str(jsonl_file)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMOS Metrics Collector v2")
    parser.add_argument("duration", nargs="?", type=int, default=300,
                        help="Collection duration in seconds (default: 300)")
    parser.add_argument("--scenario", "-s", type=str, default="test",
                        help="Nome scenario per il file output (es. flash_crowd, sinusoidal)")
    parser.add_argument("--locust-host", default=LOCUST_API_URL,
                        help="Locust web UI URL (default: http://localhost:8089)")
    parser.add_argument("--no-dmos", action="store_true",
                        help="Skip DMOS endpoint wait (use for baseline/DMOS-OFF runs)")
    args = parser.parse_args()

    print(f"\nStarting in 3 seconds...")
    time.sleep(3)
    run_collector(duration_seconds=args.duration, locust_url=args.locust_host,
                  scenario=args.scenario, skip_dmos=args.no_dmos)