"""
DMOS Metrics Collector for k6 Tests
====================================
Collects DMOS + Prometheus metrics during k6 load tests.
(k6 does not expose a REST API like Locust, so we skip Locust and rely on
 DMOS /metrics + Prometheus cAdvisor for cluster resources.)

Outputs: results/{timestamp}_{scenario}.jsonl

Usage:
  python collect_metrics_k6.py [duration_sec] [--scenario name] [--no-dmos]

Examples:
  # DMOS ON test (collects DMOS metrics + cluster resources)
  python collect_metrics_k6.py 900 --scenario flash_crowd_DMOS_ON

  # DMOS OFF test (collects only cluster resources, no DMOS)
  python collect_metrics_k6.py 900 --scenario flash_crowd_DMOS_OFF --no-dmos
"""

import requests
import time
import json
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ─── Configuration ───────────────────────────────────────────────────────────

DMOS_METRICS_URL = "http://localhost:9090/metrics"
SCRAPE_INTERVAL  = 15   # seconds
OUTPUT_DIR       = Path(__file__).parent.parent / "results"

KNOWN_SERVICES = [
    "frontend", "cartservice", "productcatalogservice",
    "checkoutservice", "recommendationservice"
]
KNOWN_CLUSTERS = ["cluster1", "cluster2", "cluster3"]

CLUSTER_PROMETHEUS = {
    "cluster1": "http://192.168.1.245:30090",
    "cluster2": "http://192.168.1.246:30090",
    "cluster3": "http://192.168.1.247:30090",
}
RESOURCE_NAMESPACE = "online-boutique"


# ─── DMOS Metrics Parser ────────────────────────────────────────────────────

def scrape_dmos_metrics() -> dict:
    """Scrape DMOS Prometheus endpoint."""
    try:
        r = requests.get(DMOS_METRICS_URL, timeout=5)
        raw = r.text
    except Exception as e:
        print(f"  [warn] DMOS metrics unavailable: {e}")
        return {"parsed": {}}

    parsed = {
        "actual_traffic": {},
        "current_replicas": {},
        "target_replicas": {},
        "cluster_score": {},
        "predicted_traffic": {},
        "scaling_events": {},
        "scheduling_duration_sum": {},
        "scheduling_duration_count": {},
    }

    for line in raw.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            if 'dmos_actual_traffic{' in line:
                m = re.search(r'service="([^"]+)"', line)
                v = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if m and v: parsed["actual_traffic"][m.group(1)] = float(v.group(1))

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
                if ms and v: parsed["scheduling_duration_sum"][ms.group(1)] = float(v.group(1))

            elif 'dmos_scheduling_duration_seconds_count{' in line:
                ms = re.search(r'service="([^"]+)"', line)
                v  = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
                if ms and v: parsed["scheduling_duration_count"][ms.group(1)] = int(float(v.group(1)))

        except (ValueError, AttributeError):
            continue

    return {"parsed": parsed}


# ─── Prometheus Query Helper ─────────────────────────────────────────────────

def _prom_query(prom_url: str, query: str) -> float | None:
    try:
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=5)
        data = r.json()
        if data.get("status") == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
    except Exception:
        pass
    return None


def _prom_query_multi(prom_url: str, query: str) -> list:
    """Execute a Prometheus query that returns multiple results (one per label)."""
    try:
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=5)
        data = r.json()
        if data.get("status") == "success":
            return data["data"]["result"]
    except Exception:
        pass
    return []


# ─── Cluster Resources (CPU, Memory, Network, Replicas) ─────────────────────

def scrape_cluster_resources() -> dict:
    """Query each cluster's Prometheus for pod-level metrics."""
    ns = RESOURCE_NAMESPACE
    result = {}

    for cluster, prom_url in CLUSTER_PROMETHEUS.items():
        cluster_data = {}
        for svc in KNOWN_SERVICES:
            cpu_q = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",container="server",pod=~"{svc}-.*"}}[1m]))'
            mem_q = f'sum(container_memory_working_set_bytes{{namespace="{ns}",container="server",pod=~"{svc}-.*"}})'
            net_q = f'sum(rate(container_network_receive_bytes_total{{namespace="{ns}",pod=~"{svc}-.*"}}[1m]))'
            # Count running pods
            pods_q = f'count(kube_pod_status_ready{{namespace="{ns}",pod=~"{svc}-.*",condition="true"}})'

            cpu_val = _prom_query(prom_url, cpu_q)
            mem_val = _prom_query(prom_url, mem_q)
            net_val = _prom_query(prom_url, net_q)
            pods_val = _prom_query(prom_url, pods_q)

            cluster_data[svc] = {
                "cpu_cores": round(cpu_val, 4) if cpu_val is not None else None,
                "memory_mb": round(mem_val / (1024 * 1024), 1) if mem_val is not None else None,
                "network_recv_bytes_s": round(net_val, 1) if net_val is not None else None,
                "ready_pods": int(pods_val) if pods_val is not None else None,
            }
        result[cluster] = cluster_data

    # Frontend traffic distribution (%) based on network bytes received
    fe_bytes = {}
    for cn in KNOWN_CLUSTERS:
        val = result.get(cn, {}).get("frontend", {}).get("network_recv_bytes_s")
        if val is not None:
            fe_bytes[cn] = val
    total_bytes = sum(fe_bytes.values()) if fe_bytes else 0
    for cn in KNOWN_CLUSTERS:
        pct = round(fe_bytes.get(cn, 0) / total_bytes * 100, 1) if total_bytes > 0 else None
        result.setdefault(cn, {})["_traffic_pct"] = {"frontend": pct}

    return result


# ─── Pod Health (Restarts, Status) ───────────────────────────────────────────

def scrape_pod_health() -> dict:
    """
    Query each cluster's Prometheus for pod restart counts and status.
    Uses kube-state-metrics which exposes:
      - kube_pod_container_status_restarts_total
      - kube_pod_status_phase
      - kube_pod_status_ready
    """
    ns = RESOURCE_NAMESPACE
    # All services to monitor (not just KNOWN_SERVICES — include support services)
    all_services = KNOWN_SERVICES + [
        "currencyservice", "emailservice", "paymentservice",
        "shippingservice", "adservice", "redis-cart"
    ]
    # Deduplicate
    all_services = list(dict.fromkeys(all_services))

    result = {}
    for cluster, prom_url in CLUSTER_PROMETHEUS.items():
        cluster_pods = {}

        for svc in all_services:
            # Total restarts (sum across all containers in all pods of this service)
            restarts_q = (
                f'sum(kube_pod_container_status_restarts_total{{'
                f'namespace="{ns}",pod=~"{svc}-.*"}})'
            )
            # Count of pods in Running phase
            running_q = (
                f'count(kube_pod_status_phase{{'
                f'namespace="{ns}",pod=~"{svc}-.*",phase="Running"}})'
            )
            # Count of pods Ready
            ready_q = (
                f'count(kube_pod_status_ready{{'
                f'namespace="{ns}",pod=~"{svc}-.*",condition="true"}})'
            )
            # Count of pods NOT ready but still in Running phase
            # (filters out old ReplicaSet ghosts that kube-state-metrics retains)
            notready_q = (
                f'count(kube_pod_status_ready{{'
                f'namespace="{ns}",pod=~"{svc}-.*",condition="false"}} '
                f'and kube_pod_status_phase{{'
                f'namespace="{ns}",pod=~"{svc}-.*",phase="Running"}})'
            )

            restarts = _prom_query(prom_url, restarts_q)
            running = _prom_query(prom_url, running_q)
            ready = _prom_query(prom_url, ready_q)
            notready = _prom_query(prom_url, notready_q)

            cluster_pods[svc] = {
                "restarts_total": int(restarts) if restarts is not None else None,
                "running": int(running) if running is not None else None,
                "ready": int(ready) if ready is not None else None,
                "not_ready": int(notready) if notready is not None else None,
            }

        result[cluster] = cluster_pods

    return result


# ─── Hubble/Cilium Cross-Cluster Metrics ────────────────────────────────────

def scrape_cilium_flows() -> dict:
    """
    Query Hubble metrics from each cluster's Prometheus for cross-cluster
    request counts. Uses hubble_flows_processed_total with source/dest labels.
    """
    flows = {}
    for cluster, prom_url in CLUSTER_PROMETHEUS.items():
        # Total HTTP flows processed (L7)
        local_q = f'sum(hubble_http_requests_total{{reporter="destination",destination_namespace="{RESOURCE_NAMESPACE}"}})'
        # Flows from remote clusters (source is a remote cluster identity)
        remote_q = f'sum(hubble_http_requests_total{{reporter="destination",destination_namespace="{RESOURCE_NAMESPACE}",source_namespace!~"{RESOURCE_NAMESPACE}"}})'

        local_val = _prom_query(prom_url, local_q)
        remote_val = _prom_query(prom_url, remote_q)

        flows[cluster] = {
            "total_http_flows": local_val,
            "remote_source_flows": remote_val,
        }
    return flows


# ─── Snapshot Builder ───────────────────────────────────────────────────────

def build_snapshot(dmos: dict, resources: dict, cilium: dict, pod_health: dict, timestamp: str, no_dmos: bool) -> dict:
    """Build unified snapshot."""
    p = dmos.get("parsed", {})

    services_data = {}
    for svc in KNOWN_SERVICES:
        svc_data = {
            "actual_traffic": p.get("actual_traffic", {}).get(svc, 0.0),
            "predicted_traffic_total": 0.0,
            "clusters": {},
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

        sched_sum = p.get("scheduling_duration_sum", {}).get(svc, 0.0)
        sched_count = p.get("scheduling_duration_count", {}).get(svc, 0)
        svc_data["avg_scheduling_duration_s"] = sched_sum / sched_count if sched_count > 0 else 0.0
        svc_data["scheduling_invocations"] = sched_count

        services_data[svc] = svc_data

    return {
        "timestamp": timestamp,
        "dmos_enabled": not no_dmos,
        "dmos": {"services": services_data},
        "resources": resources,
        "cilium_flows": cilium,
        "pod_health": pod_health,
    }


# ─── Console Summary ────────────────────────────────────────────────────────

def print_summary(snap: dict, iteration: int, total_iter: int, baseline_restarts: dict):
    ts = snap["timestamp"]
    svcs = snap["dmos"]["services"]
    fe = svcs.get("frontend", {})

    # Header
    pct_done = iteration / total_iter * 100 if total_iter > 0 else 0
    print(f"\n{'='*70}")
    print(f"  [{iteration}/{total_iter}] {ts}  ({pct_done:.0f}% complete)")
    print(f"{'='*70}")

    # DMOS metrics
    if snap["dmos_enabled"]:
        print(f"  {'Service':<25s} {'Traffic':>8s} {'Predicted':>10s} {'Replicas':>10s} {'Target':>8s}")
        print(f"  {'-'*65}")
        for svc_name in KNOWN_SERVICES:
            s = svcs.get(svc_name, {})
            traffic = s.get("actual_traffic", 0)
            predicted = s.get("predicted_traffic_total", 0)
            total_r = s.get("total_replicas", 0)
            per_cluster = []
            target_per_cluster = []
            for c in KNOWN_CLUSTERS:
                cd = s.get("clusters", {}).get(c, {})
                per_cluster.append(str(cd.get("current_replicas", 0)))
                target_per_cluster.append(str(cd.get("target_replicas", 0)))
            r_str = "/".join(per_cluster)
            t_str = "/".join(target_per_cluster)
            print(f"  {svc_name:<25s} {traffic:>7.1f}  {predicted:>9.1f}  {r_str:>10s} {t_str:>8s}")

    # Resources
    res = snap.get("resources", {})
    print(f"\n  {'Cluster':<12s} {'Frontend CPU':>14s} {'Frontend Mem':>14s} {'Traffic %':>10s}")
    print(f"  {'-'*55}")
    for c in KNOWN_CLUSTERS:
        cd = res.get(c, {})
        fe_r = cd.get("frontend", {})
        cpu = fe_r.get("cpu_cores")
        mem = fe_r.get("memory_mb")
        tpct = cd.get("_traffic_pct", {}).get("frontend")
        cpu_s = f"{cpu:.3f}" if cpu is not None else "N/A"
        mem_s = f"{mem:.0f}MB" if mem is not None else "N/A"
        tpct_s = f"{tpct:.1f}%" if tpct is not None else "N/A"
        print(f"  {c:<12s} {cpu_s:>14s} {mem_s:>14s} {tpct_s:>10s}")

    # Pod health (restarts) — show only NEW restarts since test started
    ph = snap.get("pod_health", {})

    # On first snapshot, record baseline
    if not baseline_restarts:
        for c in KNOWN_CLUSTERS:
            for svc_name, svc_health in ph.get(c, {}).items():
                r = svc_health.get("restarts_total")
                baseline_restarts[(c, svc_name)] = r if r is not None else 0

    # Compute deltas
    has_new_restarts = False
    delta_restarts = {}
    for c in KNOWN_CLUSTERS:
        for svc_name, svc_health in ph.get(c, {}).items():
            r = svc_health.get("restarts_total")
            if r is None:
                r = 0
            base = baseline_restarts.get((c, svc_name), r)
            delta = r - base
            delta_restarts[(c, svc_name)] = delta
            if delta > 0:
                has_new_restarts = True

    if has_new_restarts:
        print(f"\n  ⚠ NEW Pod Restarts (during this test):")
        print(f"  {'Service':<25s} {'C1':>6s} {'C2':>6s} {'C3':>6s}")
        print(f"  {'-'*45}")
        all_svcs = set()
        for c in KNOWN_CLUSTERS:
            all_svcs.update(ph.get(c, {}).keys())
        for svc_name in sorted(all_svcs):
            deltas = [delta_restarts.get((c, svc_name), 0) for c in KNOWN_CLUSTERS]
            if sum(deltas) > 0:
                d_strs = [f"+{d}" if d > 0 else "." for d in deltas]
                print(f"  {svc_name:<25s} {d_strs[0]:>6s} {d_strs[1]:>6s} {d_strs[2]:>6s}")

    # Cilium flows
    flows = snap.get("cilium_flows", {})
    if any(v.get("total_http_flows") is not None for v in flows.values()):
        print(f"\n  Cilium HTTP Flows:")
        for c in KNOWN_CLUSTERS:
            f = flows.get(c, {})
            total = f.get("total_http_flows", "N/A")
            remote = f.get("remote_source_flows", "N/A")
            print(f"    {c}: total={total}, remote={remote}")


# ─── Main Loop ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DMOS Metrics Collector for k6")
    parser.add_argument("duration", nargs="?", type=int, default=900, help="Collection duration in seconds (default 900)")
    parser.add_argument("--scenario", default="flash_crowd", help="Scenario name for output file")
    parser.add_argument("--no-dmos", action="store_true", help="Skip DMOS metrics collection (for baseline tests)")
    parser.add_argument("--interval", type=int, default=15, help="Scrape interval in seconds (default 15)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    prefix = now.strftime("%H%M%S_%Y%m%d")
    out_file = OUTPUT_DIR / f"{prefix}_{args.scenario}.jsonl"

    total_iterations = args.duration // args.interval
    print(f"\n{'='*60}")
    print(f"  DMOS Metrics Collector for k6")
    print(f"  Scenario: {args.scenario}")
    print(f"  Duration: {args.duration}s ({total_iterations} snapshots @ {args.interval}s)")
    print(f"  DMOS:     {'Enabled' if not args.no_dmos else 'Disabled (baseline)'}")
    print(f"  Output:   {out_file}")
    print(f"{'='*60}")

    # Track baseline restarts from first snapshot to show only NEW restarts
    baseline_restarts = {}  # {(cluster, svc): count}

    with open(out_file, 'w') as f:
        for i in range(1, total_iterations + 1):
            ts = datetime.now().isoformat()

            # Collect DMOS metrics
            if not args.no_dmos:
                dmos = scrape_dmos_metrics()
            else:
                dmos = {"parsed": {}}

            # Collect cluster resources
            try:
                resources = scrape_cluster_resources()
            except Exception as e:
                print(f"  [warn] Resource collection failed: {e}")
                resources = {}

            # Collect Cilium flows
            try:
                cilium = scrape_cilium_flows()
            except Exception as e:
                print(f"  [warn] Cilium flow collection failed: {e}")
                cilium = {}

            # Collect pod health (restarts, status)
            try:
                pod_health = scrape_pod_health()
            except Exception as e:
                print(f"  [warn] Pod health collection failed: {e}")
                pod_health = {}

            snap = build_snapshot(dmos, resources, cilium, pod_health, ts, args.no_dmos)

            # Write to JSONL
            f.write(json.dumps(snap) + '\n')
            f.flush()

            # Console output
            print_summary(snap, i, total_iterations, baseline_restarts)

            # Wait for next interval
            if i < total_iterations:
                time.sleep(args.interval)

    print(f"\n{'='*60}")
    print(f"  Collection complete! {total_iterations} snapshots saved to:")
    print(f"  {out_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
