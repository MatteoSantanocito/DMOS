#!/usr/bin/env python3
"""
collect_metrics_post.py — Post-test Prometheus Metrics Collector
================================================================
Runs AFTER the k6 test completes. Queries Prometheus range API to get
ALL data points (every 15s scrape interval) for the entire test duration.

Much more data than the live collector (every data point vs 9-10 snapshots).

Usage:
  # Collect last 15 minutes of data
  python collect_metrics_post.py --duration 900 --scenario flash_crowd_DMOS_ON

  # Collect specific time range (ISO format)
  python collect_metrics_post.py --start 2026-04-06T11:37:00 --end 2026-04-06T11:51:00 --scenario test

  # Collect with DMOS metrics
  python collect_metrics_post.py --duration 900 --scenario flash_crowd_DMOS_ON --dmos
"""

import requests
import json
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ───────────────────────────────────────────────────────────

CLUSTER_PROMETHEUS = {
    "cluster1": "http://192.168.1.245:30090",
    "cluster2": "http://192.168.1.246:30090",
    "cluster3": "http://192.168.1.247:30090",
}

DMOS_METRICS_URL = "http://localhost:9090/metrics"
RESOURCE_NAMESPACE = "online-boutique"
OUTPUT_DIR = Path(__file__).parent.parent / "results"

KNOWN_SERVICES = [
    "frontend", "cartservice", "productcatalogservice",
    "checkoutservice", "recommendationservice"
]

ALL_SERVICES = KNOWN_SERVICES + [
    "currencyservice", "emailservice", "paymentservice",
    "shippingservice", "adservice", "redis-cart"
]

STEP = "15s"  # Match Prometheus scrape interval


# ─── Prometheus Range Query ──────────────────────────────────────────────────

def prom_range_query(prom_url: str, query: str, start: float, end: float, step: str = STEP) -> list:
    """Execute Prometheus range query, return list of {timestamp, value} dicts."""
    try:
        r = requests.get(
            f"{prom_url}/api/v1/query_range",
            params={"query": query, "start": start, "end": end, "step": step},
            timeout=30,
        )
        data = r.json()
        if data.get("status") == "success":
            results = data["data"]["result"]
            return results
    except Exception as e:
        print(f"  [warn] Query failed: {e}")
    return []


def prom_range_scalar(prom_url: str, query: str, start: float, end: float, step: str = STEP) -> list:
    """Range query returning [(timestamp, value), ...] for single-series result."""
    results = prom_range_query(prom_url, query, start, end, step)
    if results and len(results) > 0:
        return [(float(ts), float(val)) for ts, val in results[0]["values"]]
    return []


def prom_range_multi(prom_url: str, query: str, start: float, end: float, step: str = STEP) -> dict:
    """Range query returning {label_key: [(ts, val), ...]} for multi-series."""
    results = prom_range_query(prom_url, query, start, end, step)
    output = {}
    for series in results:
        # Use the most descriptive label as key
        metric = series.get("metric", {})
        key = metric.get("pod", metric.get("service", metric.get("container", str(metric))))
        output[key] = [(float(ts), float(val)) for ts, val in series["values"]]
    return output


# ─── Collect All Metrics ─────────────────────────────────────────────────────

def collect_cluster_metrics(cluster: str, prom_url: str, start: float, end: float) -> dict:
    """Collect all metrics for a single cluster using range queries."""
    ns = RESOURCE_NAMESPACE
    cluster_data = {"services": {}, "node": {}, "traffic_pct": {}}

    print(f"  Collecting {cluster}...")

    for svc in KNOWN_SERVICES:
        svc_data = {}

        # CPU usage (rate over 1m window)
        cpu_q = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",container="server",pod=~"{svc}-.*"}}[1m]))'
        svc_data["cpu_cores"] = prom_range_scalar(prom_url, cpu_q, start, end)

        # Memory (working set)
        mem_q = f'sum(container_memory_working_set_bytes{{namespace="{ns}",container="server",pod=~"{svc}-.*"}})'
        svc_data["memory_bytes"] = prom_range_scalar(prom_url, mem_q, start, end)

        # Network receive bytes/s
        net_q = f'sum(rate(container_network_receive_bytes_total{{namespace="{ns}",pod=~"{svc}-.*"}}[1m]))'
        svc_data["network_recv_bytes_s"] = prom_range_scalar(prom_url, net_q, start, end)

        # Ready pods count
        ready_q = f'count(kube_pod_status_ready{{namespace="{ns}",pod=~"{svc}-.*",condition="true"}})'
        svc_data["ready_pods"] = prom_range_scalar(prom_url, ready_q, start, end)

        # Restart count
        restarts_q = f'sum(kube_pod_container_status_restarts_total{{namespace="{ns}",pod=~"{svc}-.*"}})'
        svc_data["restarts_total"] = prom_range_scalar(prom_url, restarts_q, start, end)

        cluster_data["services"][svc] = svc_data

    # Node CPU/Memory
    node_cpu_q = f'1 - avg(rate(node_cpu_seconds_total{{mode="idle"}}[1m]))'
    cluster_data["node"]["cpu_usage_pct"] = prom_range_scalar(prom_url, node_cpu_q, start, end)

    node_mem_q = f'1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)'
    cluster_data["node"]["memory_usage_pct"] = prom_range_scalar(prom_url, node_mem_q, start, end)

    # Hubble HTTP traffic (if available)
    hubble_q = f'sum(rate(hubble_http_requests_total{{reporter="destination",destination_namespace="{ns}"}}[1m]))'
    cluster_data["hubble_http_rate"] = prom_range_scalar(prom_url, hubble_q, start, end)

    # Frontend traffic for distribution calc
    fe_net_q = f'sum(rate(container_network_receive_bytes_total{{namespace="{ns}",pod=~"frontend-.*"}}[1m]))'
    cluster_data["frontend_net_bytes_s"] = prom_range_scalar(prom_url, fe_net_q, start, end)

    return cluster_data


def collect_dmos_metrics(start: float, end: float) -> dict:
    """
    DMOS exposes Prometheus metrics at localhost:9090/metrics.
    Since it's a custom endpoint (not a real Prometheus server with range query support),
    we can only get the CURRENT state. For historical data, we'd need DMOS to push
    metrics to a real Prometheus. Instead, we parse the DMOS log file.
    """
    dmos_data = {"available": False}

    # Try to read DMOS log for historical decisions
    log_path = Path(__file__).parent.parent / "logs" / "DMOSMain.log"
    if log_path.exists():
        dmos_data["available"] = True
        dmos_data["log_entries"] = []

        start_dt = datetime.fromtimestamp(start)
        end_dt = datetime.fromtimestamp(end)

        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Parse timestamp from log: "2026-04-06 11:37:27 | INFO | ..."
                try:
                    ts_str = line[:19]
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    if start_dt <= ts <= end_dt:
                        # Only keep scaling-related entries
                        if any(kw in line.lower() for kw in [
                            'scaling', 'scale_up', 'scale_down', 'target_replicas',
                            'predicted', 'actual_traffic', 'scheduling', 'jain',
                            'score', 'allocation', 'replicas'
                        ]):
                            dmos_data["log_entries"].append({
                                "timestamp": ts.isoformat(),
                                "message": line.strip()
                            })
                except (ValueError, IndexError):
                    continue

        print(f"  DMOS log: {len(dmos_data['log_entries'])} relevant entries found")

    return dmos_data


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-test Prometheus Metrics Collector")
    parser.add_argument("--duration", type=int, default=900, help="Look back N seconds from now (default 900)")
    parser.add_argument("--start", help="Start time (ISO format, e.g. 2026-04-06T11:37:00)")
    parser.add_argument("--end", help="End time (ISO format, e.g. 2026-04-06T11:51:00)")
    parser.add_argument("--scenario", default="test", help="Scenario name for output file")
    parser.add_argument("--dmos", action="store_true", help="Also collect DMOS log entries")
    parser.add_argument("--step", default="15s", help="Query step/resolution (default 15s)")
    args = parser.parse_args()

    global STEP
    STEP = args.step

    # Determine time range
    if args.start and args.end:
        start_ts = datetime.fromisoformat(args.start).timestamp()
        end_ts = datetime.fromisoformat(args.end).timestamp()
    else:
        end_ts = time.time()
        start_ts = end_ts - args.duration

    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)
    duration_min = (end_ts - start_ts) / 60
    expected_points = int((end_ts - start_ts) / 15)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("%H%M%S_%Y%m%d")
    out_file = OUTPUT_DIR / f"{prefix}_{args.scenario}_post.jsonl"

    print(f"\n{'='*60}")
    print(f"  Post-Test Prometheus Metrics Collector")
    print(f"  Scenario:   {args.scenario}")
    print(f"  Time range: {start_dt.strftime('%H:%M:%S')} → {end_dt.strftime('%H:%M:%S')}")
    print(f"  Duration:   {duration_min:.0f} minutes")
    print(f"  Resolution: {STEP} (~{expected_points} data points)")
    print(f"  Output:     {out_file}")
    print(f"{'='*60}")

    # Collect from all clusters in parallel
    print(f"\n[1/3] Collecting cluster metrics from Prometheus...")
    all_data = {"clusters": {}}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(collect_cluster_metrics, cluster, url, start_ts, end_ts): cluster
            for cluster, url in CLUSTER_PROMETHEUS.items()
        }
        for future in as_completed(futures):
            cluster = futures[future]
            try:
                all_data["clusters"][cluster] = future.result()
            except Exception as e:
                print(f"  [error] {cluster}: {e}")
                all_data["clusters"][cluster] = {"error": str(e)}

    # DMOS log entries
    if args.dmos:
        print(f"\n[2/3] Collecting DMOS log entries...")
        all_data["dmos"] = collect_dmos_metrics(start_ts, end_ts)
    else:
        print(f"\n[2/3] Skipping DMOS (use --dmos to include)")

    # Compute traffic distribution over time
    print(f"\n[3/3] Computing derived metrics...")
    clusters = list(CLUSTER_PROMETHEUS.keys())
    fe_traffic = {}
    for c in clusters:
        ts_data = all_data["clusters"].get(c, {}).get("frontend_net_bytes_s", [])
        fe_traffic[c] = {ts: val for ts, val in ts_data}

    # Get all timestamps
    all_timestamps = set()
    for c in clusters:
        all_timestamps.update(fe_traffic[c].keys())
    all_timestamps = sorted(all_timestamps)

    traffic_distribution = []
    for ts in all_timestamps:
        total = sum(fe_traffic[c].get(ts, 0) for c in clusters)
        pcts = {}
        for c in clusters:
            val = fe_traffic[c].get(ts, 0)
            pcts[c] = round(val / total * 100, 1) if total > 0 else 0
        traffic_distribution.append({"timestamp": ts, **pcts})

    all_data["traffic_distribution"] = traffic_distribution

    # Metadata
    all_data["metadata"] = {
        "scenario": args.scenario,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "duration_s": end_ts - start_ts,
        "step": STEP,
        "collected_at": datetime.now().isoformat(),
    }

    # Summary stats
    for cluster, cdata in all_data["clusters"].items():
        for svc, sdata in cdata.get("services", {}).items():
            cpu_points = sdata.get("cpu_cores", [])
            if cpu_points:
                vals = [v for _, v in cpu_points]
                sdata["cpu_summary"] = {
                    "mean": round(sum(vals) / len(vals), 4),
                    "max": round(max(vals), 4),
                    "points": len(vals),
                }
            ready_points = sdata.get("ready_pods", [])
            if ready_points:
                vals = [v for _, v in ready_points]
                sdata["replicas_summary"] = {
                    "mean": round(sum(vals) / len(vals), 1),
                    "max": int(max(vals)),
                    "min": int(min(vals)),
                }
            restart_points = sdata.get("restarts_total", [])
            if restart_points:
                first_val = restart_points[0][1]
                last_val = restart_points[-1][1]
                sdata["new_restarts"] = int(last_val - first_val)

    # Write output
    with open(out_file, 'w') as f:
        f.write(json.dumps(all_data, indent=2, default=str))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*60}")

    for cluster in clusters:
        cdata = all_data["clusters"].get(cluster, {})
        print(f"\n  {cluster}:")
        for svc in KNOWN_SERVICES:
            sdata = cdata.get("services", {}).get(svc, {})
            cpu_s = sdata.get("cpu_summary", {})
            rep_s = sdata.get("replicas_summary", {})
            restarts = sdata.get("new_restarts", 0)
            points = cpu_s.get("points", 0)
            print(f"    {svc:<25s} CPU avg={cpu_s.get('mean','?'):.4f} max={cpu_s.get('max','?'):.4f}  "
                  f"replicas={rep_s.get('min','?')}-{rep_s.get('max','?')}  "
                  f"restarts={restarts}  ({points} points)")

    print(f"\n  Traffic Distribution (first/middle/last):")
    td = all_data["traffic_distribution"]
    if td:
        for idx in [0, len(td)//2, -1]:
            entry = td[idx]
            ts_str = datetime.fromtimestamp(entry["timestamp"]).strftime("%H:%M:%S")
            print(f"    {ts_str}: C1={entry.get('cluster1',0):.1f}% C2={entry.get('cluster2',0):.1f}% C3={entry.get('cluster3',0):.1f}%")

    total_points = sum(
        cdata.get("services", {}).get("frontend", {}).get("cpu_summary", {}).get("points", 0)
        for cdata in all_data["clusters"].values()
    )
    print(f"\n  Total data points: ~{total_points} per metric per cluster")
    print(f"  Saved to: {out_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
