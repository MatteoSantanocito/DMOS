#!/usr/bin/env python3
"""
collect_dmos_live.py — Real-Time DMOS Internals Collector
=========================================================
Runs IN PARALLEL with DMOS during k6 tests.
Scrapes DMOS /metrics endpoint + per-cluster Prometheus every N seconds,
producing a structured JSONL perfect for analysis plots.

Output: results/{timestamp}_{scenario}_dmos_live.jsonl
  - One JSON line per snapshot (every 10-15s)
  - Contains: actual/predicted traffic, cluster scores, replicas (current/target),
    scaling events (cumulative), scheduling duration, per-cluster CPU/mem/pods,
    Hubble HTTP rates, traffic distribution

Usage:
  # Run alongside DMOS ON test (default 900s)
  python collect_dmos_live.py 900 --scenario flash_crowd_DMOS_ON

  # Custom interval (10s for higher resolution)
  python collect_dmos_live.py 900 --scenario flash_crowd_DMOS_ON --interval 10

  # DMOS OFF baseline (no DMOS endpoint, only cluster resources)
  python collect_dmos_live.py 900 --scenario flash_crowd_DMOS_OFF --no-dmos
"""

import requests
import time
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ───────────────────────────────────────────────────────────

DMOS_METRICS_URL = "http://localhost:9090/metrics"

KNOWN_SERVICES = [
    "frontend", "cartservice", "productcatalogservice",
    "checkoutservice", "recommendationservice"
]
# Cluster topology read from .cache/test-env.json (rendered from config/local.yaml).
import json as _json
_env_path = Path(__file__).resolve().parent.parent / ".cache" / "test-env.json"
try:
    with open(_env_path, "r", encoding="utf-8") as _f:
        _env = _json.load(_f)
except FileNotFoundError:
    raise SystemExit(
        f"{_env_path} not found. Run 'bash scripts/render-runtime-files.sh' "
        "first (it generates this file from config/local.yaml)."
    )
KNOWN_CLUSTERS = [c["name"] for c in _env["clusters"]]
CLUSTER_PROMETHEUS = {c["name"]: c["prometheus_url"] for c in _env["clusters"]}
NAMESPACE = "online-boutique"
OUTPUT_DIR = Path(__file__).parent.parent / "results"


# ─── Prometheus Helper ──────────────────────────────────────────────────────

def _prom_query(prom_url: str, query: str, timeout: float = 5) -> float | None:
    try:
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=timeout)
        data = r.json()
        if data.get("status") == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
    except Exception:
        pass
    return None


def _prom_query_series(prom_url: str, query: str, timeout: float = 8) -> list:
    """Come _prom_query ma ritorna TUTTE le serie: [(labels_dict, value), ...].

    Serve per le metriche con label multiple (es. L7 duration per coppia
    source_workload→destination_workload, ping_rtt per target), dove
    _prom_query scarterebbe tutto tranne la prima serie.
    """
    out = []
    try:
        r = requests.get(f"{prom_url}/api/v1/query", params={"query": query}, timeout=timeout)
        data = r.json()
        if data.get("status") == "success":
            for s in data["data"]["result"]:
                try:
                    val = float(s["value"][1])
                except (ValueError, KeyError, IndexError):
                    continue
                # scarta NaN/Inf (es. histogram_quantile su bucket vuoti)
                if val != val or val in (float("inf"), float("-inf")):
                    continue
                out.append((s.get("metric", {}), val))
    except Exception:
        pass
    return out


# ─── DMOS /metrics Scraper ──────────────────────────────────────────────────

def scrape_dmos() -> dict:
    """Parse DMOS Prometheus text exposition format."""
    try:
        r = requests.get(DMOS_METRICS_URL, timeout=5)
        raw = r.text
    except Exception as e:
        return {"error": str(e)}

    data = {
        "actual_traffic": {},       # {service: rps}
        "predicted_traffic": {},    # {(cluster,service): rps}
        "current_replicas": {},     # {(cluster,service): int}
        "target_replicas": {},      # {(cluster,service): int}
        "cluster_score": {},        # {(cluster,service): float}
        "scaling_events": {},       # {(cluster,service,action): int}
        "sched_duration_sum": {},   # {service: float}
        "sched_duration_count": {}, # {service: int}
        "config_id": None,          # provenance: dmos_config_info{config_id=...}
    }

    for line in raw.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            val_m = re.search(r'}\s+([\d\.eE\+\-]+)$', line)
            if not val_m:
                continue
            val = float(val_m.group(1))

            if 'dmos_actual_traffic{' in line:
                s = re.search(r'service="([^"]+)"', line)
                if s:
                    data["actual_traffic"][s.group(1)] = val

            elif 'dmos_predicted_traffic{' in line:
                c = re.search(r'cluster="([^"]+)"', line)
                s = re.search(r'service="([^"]+)"', line)
                if c and s:
                    data["predicted_traffic"][f"{c.group(1)}|{s.group(1)}"] = val

            elif 'dmos_current_replicas{' in line:
                c = re.search(r'cluster="([^"]+)"', line)
                s = re.search(r'service="([^"]+)"', line)
                if c and s:
                    data["current_replicas"][f"{c.group(1)}|{s.group(1)}"] = int(val)

            elif 'dmos_target_replicas{' in line:
                c = re.search(r'cluster="([^"]+)"', line)
                s = re.search(r'service="([^"]+)"', line)
                if c and s:
                    data["target_replicas"][f"{c.group(1)}|{s.group(1)}"] = int(val)

            elif 'dmos_cluster_score{' in line:
                c = re.search(r'cluster="([^"]+)"', line)
                s = re.search(r'service="([^"]+)"', line)
                if c and s:
                    data["cluster_score"][f"{c.group(1)}|{s.group(1)}"] = round(val, 4)

            elif 'dmos_scaling_events_total{' in line:
                a = re.search(r'action="([^"]+)"', line)
                c = re.search(r'cluster="([^"]+)"', line)
                s = re.search(r'service="([^"]+)"', line)
                if a and c and s:
                    data["scaling_events"][f"{c.group(1)}|{s.group(1)}|{a.group(1)}"] = int(val)

            elif 'dmos_scheduling_duration_seconds_sum{' in line:
                s = re.search(r'service="([^"]+)"', line)
                if s:
                    data["sched_duration_sum"][s.group(1)] = val

            elif 'dmos_scheduling_duration_seconds_count{' in line:
                s = re.search(r'service="([^"]+)"', line)
                if s:
                    data["sched_duration_count"][s.group(1)] = int(val)

            elif 'dmos_config_info{' in line:
                cid = re.search(r'config_id="([^"]+)"', line)
                if cid and val == 1.0:
                    data["config_id"] = cid.group(1)

        except (ValueError, AttributeError):
            continue

    return data


# ─── Per-Cluster Resource Scraper ───────────────────────────────────────────

def scrape_cluster(cluster: str, prom_url: str) -> dict:
    """Query one cluster's Prometheus for resource + traffic metrics.

    Le 23 query (5 servizi x 4 metriche + 3 cluster-level) sono indipendenti:
    si lanciano TUTTE in parallelo. Le PromQL e la struttura del risultato
    sono identiche alla versione sequenziale precedente — cambia solo la
    concorrenza, che porta lo scrape per-cluster da ~12s a ~1-2s.
    """
    ns = NAMESPACE

    # Costruisci la mappa {chiave: PromQL}. chiave: ('svc',svc,metric) | ('cl',metric)
    tasks: dict = {}
    for svc in KNOWN_SERVICES:
        tasks[('svc', svc, 'cpu')] = (
            f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",container="server",pod=~"{svc}-.*"}}[1m]))')
        tasks[('svc', svc, 'mem')] = (
            f'sum(container_memory_working_set_bytes{{namespace="{ns}",container="server",pod=~"{svc}-.*"}})')
        tasks[('svc', svc, 'pods')] = (
            f'count(kube_pod_status_ready{{namespace="{ns}",pod=~"{svc}-.*",condition="true"}})')
        tasks[('svc', svc, 'restarts')] = (
            f'sum(kube_pod_container_status_restarts_total{{namespace="{ns}",pod=~"{svc}-.*"}})')

    # Hubble HTTP rate (total L7 traffic hitting this cluster)
    # reporter="server" (non "destination") — Cilium Hubble label convention
    tasks[('cl', 'hubble')] = (
        f'sum(rate(hubble_http_requests_total{{reporter="server",destination_namespace="{ns}"}}[1m]))')
    # Frontend HTTP ingress rate (for traffic distribution).
    # hubble_http_requests_total con filtro frontend + ingress direction;
    # source_workload="" → solo traffico esterno (k6 via Cilium Ingress),
    # esclude health probes e gRPC inter-service.
    tasks[('cl', 'fe_ingress')] = (
        f'sum(rate(hubble_http_requests_total{{'
        f'destination_workload="frontend",'
        f'destination_namespace="{ns}",'
        f'reporter="server",'
        f'traffic_direction="ingress",'
        f'source_workload=""'
        f'}}[1m]))')
    # Node-level CPU
    tasks[('cl', 'node_cpu')] = '1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))'

    # Esegui tutte le query in parallelo
    vals: dict = {}
    with ThreadPoolExecutor(max_workers=min(12, len(tasks))) as ex:
        futs = {ex.submit(_prom_query, prom_url, q): k for k, q in tasks.items()}
        for f in as_completed(futs):
            try:
                vals[futs[f]] = f.result()
            except Exception:
                vals[futs[f]] = None

    # Assembla il risultato (stessa identica struttura di prima)
    result = {"services": {}}
    for svc in KNOWN_SERVICES:
        cpu = vals.get(('svc', svc, 'cpu'))
        mem = vals.get(('svc', svc, 'mem'))
        pods = vals.get(('svc', svc, 'pods'))
        restarts = vals.get(('svc', svc, 'restarts'))
        result["services"][svc] = {
            "cpu_cores": round(cpu, 4) if cpu is not None else None,
            "memory_mb": round(mem / (1024**2), 1) if mem is not None else None,
            "ready_pods": int(pods) if pods is not None else None,
            "restarts": int(restarts) if restarts is not None else None,
        }

    hubble = vals.get(('cl', 'hubble'))
    result["hubble_http_rps"] = round(hubble, 2) if hubble is not None else None
    fe_ingress = vals.get(('cl', 'fe_ingress'))
    result["frontend_net_bytes_s"] = round(fe_ingress, 1) if fe_ingress is not None else None
    node_cpu = vals.get(('cl', 'node_cpu'))
    result["node_cpu_pct"] = round(node_cpu * 100, 1) if node_cpu is not None else None

    # ── L7 latency decomposition data (per la figura "latency budget") ──────
    # (a) processing: p95 L7 per coppia source→destination workload interna
    #     (Hubble httpV2, labelsContext gia' espone source/destination_workload).
    #     Servizio-a-servizio = source_workload!="" (esclude l'ingress esterno,
    #     che non ha source_workload — cfr score_functions.py:109).
    l7_q = (
        f'histogram_quantile(0.95, sum(rate('
        f'hubble_http_request_duration_seconds_bucket{{destination_namespace="{ns}"}}'
        f'[1m])) by (le, source_workload, destination_workload)) * 1000'
    )
    l7_edges = {}
    for labels, p95 in _prom_query_series(prom_url, l7_q):
        src = labels.get("source_workload", "")
        dst = labels.get("destination_workload", "")
        if not dst:
            continue
        # src vuoto = traffico ingress esterno: lo teniamo come "ingress->frontend"
        edge = f"{src or 'ingress'}->{dst}"
        l7_edges[edge] = round(p95, 2)
    result["l7_edge_p95_ms"] = l7_edges

    # (b) geo/network: RTT media verso ogni peer (ping_exporter, ICMP).
    #     by (target) -> una serie per IP peer; e' il segnale che alimenta Phi_net.
    rtt_q = 'avg(ping_rtt_mean_seconds) by (target) * 1000'
    ping_rtt = {}
    for labels, rtt in _prom_query_series(prom_url, rtt_q):
        tgt = labels.get("target", "")
        if tgt:
            ping_rtt[tgt] = round(rtt, 2)
    result["ping_rtt_ms"] = ping_rtt

    return result


def scrape_all_clusters() -> dict:
    """Scrape all clusters in parallel.

    max_workers = numero di cluster: tutti gli scrape partono in UNA sola
    ondata. Con max_workers=3 e 6 cluster servivano 2 ondate (~14s); con una
    ondata il tempo totale e' ~quello del singolo cluster piu' lento (~7s),
    cosi' la cadenza del collector dimezza.
    """
    result = {}
    with ThreadPoolExecutor(max_workers=max(1, len(CLUSTER_PROMETHEUS))) as ex:
        futs = {ex.submit(scrape_cluster, c, url): c for c, url in CLUSTER_PROMETHEUS.items()}
        for f in as_completed(futs):
            c = futs[f]
            try:
                result[c] = f.result()
            except Exception as e:
                result[c] = {"error": str(e)}
    return result


# ─── Snapshot Builder ───────────────────────────────────────────────────────

def build_snapshot(ts_iso: str, elapsed_s: float, dmos: dict, clusters: dict, no_dmos: bool) -> dict:
    """Build a single unified snapshot."""
    snap = {
        "timestamp": ts_iso,
        "elapsed_s": round(elapsed_s, 1),
        "dmos_enabled": not no_dmos,
        # config_id exported by the DMOS process serving this snapshot (None
        # for --no-dmos arms or a DMOS build predating provenance)
        "dmos_config_id": dmos.get("config_id"),
    }

    # DMOS internals (per-service, per-cluster)
    services = {}
    for svc in KNOWN_SERVICES:
        svc_d = {
            "actual_traffic_rps": dmos.get("actual_traffic", {}).get(svc, 0.0),
            "clusters": {},
        }
        total_predicted = 0.0
        total_current = 0
        total_target = 0

        for cl in KNOWN_CLUSTERS:
            key = f"{cl}|{svc}"
            cr = dmos.get("current_replicas", {}).get(key, 0)
            tr = dmos.get("target_replicas", {}).get(key, 0)
            sc = dmos.get("cluster_score", {}).get(key, 0.0)
            pt = dmos.get("predicted_traffic", {}).get(key, 0.0)
            su = dmos.get("scaling_events", {}).get(f"{key}|scale_up", 0)
            sd = dmos.get("scaling_events", {}).get(f"{key}|scale_down", 0)

            # Merge with Prometheus cluster data
            cl_prom = clusters.get(cl, {}).get("services", {}).get(svc, {})

            svc_d["clusters"][cl] = {
                "current_replicas": cr,
                "target_replicas": tr,
                "score": sc,
                "predicted_traffic": pt,
                "scale_up_events": su,
                "scale_down_events": sd,
                # from Prometheus
                "cpu_cores": cl_prom.get("cpu_cores"),
                "memory_mb": cl_prom.get("memory_mb"),
                "ready_pods": cl_prom.get("ready_pods"),
                "restarts": cl_prom.get("restarts"),
            }
            total_predicted += pt
            total_current += cr
            total_target += tr

        svc_d["predicted_traffic_total_rps"] = round(total_predicted, 2)
        svc_d["total_current_replicas"] = total_current
        svc_d["total_target_replicas"] = total_target

        # Scheduling stats
        sc_sum = dmos.get("sched_duration_sum", {}).get(svc, 0)
        sc_cnt = dmos.get("sched_duration_count", {}).get(svc, 0)
        svc_d["scheduling_avg_ms"] = round(sc_sum / sc_cnt * 1000, 1) if sc_cnt > 0 else 0
        svc_d["scheduling_invocations"] = sc_cnt

        services[svc] = svc_d

    snap["services"] = services

    # Cluster-level summaries
    cluster_summary = {}
    fe_bytes = {}
    for cl in KNOWN_CLUSTERS:
        cd = clusters.get(cl, {})
        cluster_summary[cl] = {
            "hubble_http_rps": cd.get("hubble_http_rps"),
            "node_cpu_pct": cd.get("node_cpu_pct"),
            "frontend_net_bytes_s": cd.get("frontend_net_bytes_s"),
            # Dati per la figura latency-budget (rete vs processing):
            "l7_edge_p95_ms": cd.get("l7_edge_p95_ms", {}),
            "ping_rtt_ms": cd.get("ping_rtt_ms", {}),
        }
        if cd.get("frontend_net_bytes_s") is not None:
            fe_bytes[cl] = cd["frontend_net_bytes_s"]

    # Traffic distribution %
    total_bytes = sum(fe_bytes.values()) if fe_bytes else 0
    for cl in KNOWN_CLUSTERS:
        pct = round(fe_bytes.get(cl, 0) / total_bytes * 100, 1) if total_bytes > 0 else 0
        cluster_summary[cl]["traffic_pct"] = pct

    snap["clusters"] = cluster_summary

    return snap


# ─── Console Output ─────────────────────────────────────────────────────────

def print_snapshot(snap: dict, idx: int, total: int, duration: float = None):
    el = snap["elapsed_s"]
    n = len(KNOWN_CLUSTERS)
    # Progresso basato sul wall-clock se conosciamo la durata target,
    # altrimenti sul conteggio iterazioni.
    if duration and duration > 0:
        pct = min(100.0, el / duration * 100.0)
        progress = f"[{idx}] t={el:.0f}/{duration:.0f}s ({pct:.0f}%)"
    else:
        pct = idx / total * 100 if total > 0 else 0
        progress = f"[{idx}/{total}] t={el:.0f}s ({pct:.0f}%)"
    # Larghezze dinamiche: i valori per-cluster sono stringhe "/"-joinate,
    # cosi' la tabella scala da N=3 (tesi) a N=6 (setup attuale) senza tagli.
    reps_w = max(14, 3 * n)
    scores_w = max(20, 5 * n)
    print(f"\n{'='*72}")
    print(f"  {progress}  {snap['timestamp']}")
    print(f"{'='*72}")

    # Legenda: C1..Cn -> nome cluster (l'ordine vale per le colonne sotto)
    legend = "  ".join(f"C{i+1}={cl}" for i, cl in enumerate(KNOWN_CLUSTERS))
    print(f"  clusters:  {legend}")

    # Services table — reps/score per-cluster come stringhe compatte
    print(f"  {'Service':<22s} {'Actual':>7s} {'Predict':>8s} {'Reps':>5s} {'Tgt':>5s} "
          f"{'Reps/cluster':<{reps_w}s} {'Score/cluster':<{scores_w}s}")
    print(f"  {'-'*(54 + reps_w + scores_w)}")

    for svc in KNOWN_SERVICES:
        s = snap["services"].get(svc, {})
        actual = s.get("actual_traffic_rps", 0)
        pred = s.get("predicted_traffic_total_rps", 0)
        tot_r = s.get("total_current_replicas", 0)
        tot_t = s.get("total_target_replicas", 0)

        reps = []
        scores = []
        for cl in KNOWN_CLUSTERS:
            cd = s.get("clusters", {}).get(cl, {})
            reps.append(str(cd.get("current_replicas", 0)))
            scores.append(f"{cd.get('score', 0):.2f}")

        r_str = "/".join(reps)
        s_str = "/".join(scores)
        short = svc[:22]
        print(f"  {short:<22s} {actual:>7.1f} {pred:>8.1f} {tot_r:>5d} {tot_t:>5d} "
              f"{r_str:<{reps_w}s} {s_str:<{scores_w}s}")

    # Cluster summary
    print(f"\n  {'Cluster':<10s} {'Hubble RPS':>11s} {'Node CPU':>9s} {'Traffic%':>9s}")
    print(f"  {'-'*42}")
    for cl in KNOWN_CLUSTERS:
        cs = snap["clusters"].get(cl, {})
        hub = cs.get("hubble_http_rps")
        cpu = cs.get("node_cpu_pct")
        tpct = cs.get("traffic_pct", 0)
        hub_s = f"{hub:.1f}" if hub is not None else "N/A"
        cpu_s = f"{cpu:.1f}%" if cpu is not None else "N/A"
        print(f"  {cl:<10s} {hub_s:>11s} {cpu_s:>9s} {tpct:>8.1f}%")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-Time DMOS Internals Collector")
    parser.add_argument("duration", nargs="?", type=int, default=900,
                        help="Collection duration in seconds (default 900)")
    parser.add_argument("--scenario", default="flash_crowd_DMOS_ON",
                        help="Scenario name for output file")
    parser.add_argument("--no-dmos", action="store_true",
                        help="Skip DMOS /metrics (for DMOS OFF baseline)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Scrape interval in seconds (default 10)")
    parser.add_argument("--output", default=None,
                        help="Percorso esatto del file JSONL di output "
                             "(default: results/<ts>_<scenario>_dmos_live.jsonl)")
    args = parser.parse_args()

    now = datetime.now()
    prefix = now.strftime("%H%M%S_%Y%m%d")
    if args.output:
        out_file = Path(args.output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUT_DIR / f"{prefix}_{args.scenario}_dmos_live.jsonl"

    # Numero di snapshot ATTESI (solo per display): se gli scrape sono piu'
    # lenti di --interval ne verranno raccolti meno, ma la durata wall-clock
    # resta quella richiesta (vedi loop sotto).
    target_samples = max(1, args.duration // args.interval)

    print(f"\n{'='*60}")
    print(f"  DMOS Live Collector")
    print(f"  Scenario:  {args.scenario}")
    print(f"  Duration:  {args.duration}s (~{target_samples} snapshot @ {args.interval}s)")
    print(f"  DMOS:      {'ON' if not args.no_dmos else 'OFF (baseline)'}")
    print(f"  Output:    {out_file}")
    print(f"{'='*60}")

    t0 = time.time()
    i = 0

    # Loop guidato dal WALL-CLOCK, non da un conteggio fisso di iterazioni.
    # Cosi' il collector termina sempre a ~args.duration secondi anche se i
    # singoli scrape Prometheus sono lenti. Prima: 198 iterazioni fisse, con
    # scrape ~16s l'una -> ~53 min invece dei 990s richiesti.
    with open(out_file, 'w') as f:
        while True:
            elapsed = time.time() - t0
            if elapsed >= args.duration:
                break
            i += 1
            ts = datetime.now().isoformat()

            # Scrape DMOS + clusters
            dmos_data = {}
            if not args.no_dmos:
                dmos_data = scrape_dmos()
                if "error" in dmos_data:
                    print(f"  [warn] DMOS: {dmos_data['error']}")
                    dmos_data = {}

            clusters_data = scrape_all_clusters()

            snap = build_snapshot(ts, elapsed, dmos_data, clusters_data, args.no_dmos)

            f.write(json.dumps(snap) + '\n')
            f.flush()

            print_snapshot(snap, i, target_samples, duration=args.duration)

            # Dormi fino al prossimo tick, ma MAI oltre la durata totale.
            # Se gli scrape sono piu' lenti di --interval, `wait` e' <=0 e si
            # passa subito allo snapshot successivo (cadenza = tempo di scrape).
            next_tick = t0 + i * args.interval
            wait = min(next_tick - time.time(), args.duration - (time.time() - t0))
            if wait > 0:
                time.sleep(wait)

    print(f"\n{'='*60}")
    print(f"  Done! {i} snapshot in {time.time()-t0:.0f}s → {out_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
