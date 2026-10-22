#!/usr/bin/env python3
"""
Backend Capacity Observer — Runs alongside k6/Locust capacity tests
====================================================================

Scopo: monitorare il throughput, la latenza p95 e il CPU di TUTTI i 5 servizi
       (frontend + 4 backend gRPC) durante un test di capacita, tramite
       query Prometheus/Hubble.

Perche serve:
  Il capacity test esistente (k6_capacity_test.js, locustfile_capacity.py)
  misura solo il frontend dal lato client (latenza HTTP, rps).
  Ma DMOS scala TUTTI i 5 servizi — servono i knee point anche dei backend.
  I backend comunicano via gRPC intra-cluster, invisibili a k6.
  Hubble (Cilium L7) vede il traffico gRPC e lo espone come metriche Prometheus.

Differenza rispetto a ghz_backend_capacity.py:
  - ghz: misura la capacita ISOLATA di un singolo servizio (Phase 1)
         Invia richieste gRPC direttamente al backend via port-forward.
  - observer (questo): misura la capacita SYSTEM-LEVEL (Phase 2)
         Il traffico arriva dal frontend (via k6/Locust) e questo script
         OSSERVA le metriche di tutti i servizi via Prometheus.

  Phase 1 (ghz) risponde a: "quante rps puo gestire 1 replica di cartservice?"
  Phase 2 (observer) risponde a: "sotto carico reale, quale servizio satura per primo?"

Call graph monitorato:
  k6 HTTP -> frontend -> cartservice -----------> redis-cart
                       -> productcatalogservice   (in-memory, zero dipendenze)
                       -> recommendationservice -> productcatalogservice
                       -> checkoutservice -------> 6 servizi downstream

  L'observer vede il traffico su OGNI arco di questo grafo tramite Hubble:
  hubble_http_requests_total{destination_workload="<servizio>"}

Metodologia (system-level observation):
  1. Lancia il k6 capacity test (o Locust) normalmente
  2. In parallelo, lancia questo script
  3. Lo script interroga Prometheus ogni POLL_INTERVAL secondi per ogni servizio
  4. A fine test (Ctrl+C), produce:
     - Tabella per-servizio con rps max, latency p95, CPU usage
     - Identificazione del servizio collo di bottiglia (primo a violare SLA)
     - Valori di capacita raccomandati per services.yaml
     - CSV con serie temporale completa

Riferimenti scientifici:
  - Gan et al. (ASPLOS 2019): microservice dependency-aware profiling
  - Sriraman & Dhanotia (ASPLOS 2020): end-to-end tail latency in microservices
  - Universal Scalability Law (Gunther): throughput saturation detection

Uso:
  # Terminal 1: lancia il test di carico
  k6 run -e CLUSTER=cluster1 experiments/k6_capacity_test.js

  # Terminal 2: lancia l'observer
  python experiments/backend_capacity_observer.py

  # Oppure osserva un singolo cluster:
  python experiments/backend_capacity_observer.py --cluster cluster1

  # Con SLA personalizzati:
  python experiments/backend_capacity_observer.py --sla-p95 300

Output:
  results/capacity/backend_observer_YYYYMMDD_HHMMSS.csv
  results/capacity/backend_observer_YYYYMMDD_HHMMSS.json
  Console: tabella live + summary finale
"""

import argparse
import csv
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERRORE: 'requests' non installato. Esegui: pip install requests")
    sys.exit(1)

# ============================================================================
# Configurazione
# ============================================================================

# Cluster topology read from .cache/test-env.json (rendered from config/local.yaml).
import json as _json
import os as _os
_env_path = _os.path.join(
    _os.path.dirname(__file__), "..", ".cache", "test-env.json"
)
try:
    with open(_env_path, "r", encoding="utf-8") as _f:
        _env = _json.load(_f)
except FileNotFoundError:
    raise SystemExit(
        f"{_env_path} not found. Run 'bash scripts/render-runtime-files.sh' "
        "first (it generates this file from config/local.yaml)."
    )

CLUSTERS = {
    c["name"]: {
        "prometheus": c["prometheus_url"],
        "label": f'{c["region"]} ({(c.get("netem") or {}).get("delay_ms", 0)}ms)',
    }
    for c in _env["clusters"]
}

# Servizi monitorati (tutti quelli scalati da DMOS)
SERVICES = {
    "frontend": {
        "sla_p95_ms": 200,   # da services.yaml
        "sla_err": 0.02,
        "estimated_capacity": 39,
    },
    "cartservice": {
        "sla_p95_ms": 50,
        "sla_err": 0.02,
        "estimated_capacity": 50,
    },
    "productcatalogservice": {
        "sla_p95_ms": 75,
        "sla_err": 0.02,
        "estimated_capacity": 80,
    },
    "checkoutservice": {
        "sla_p95_ms": 200,
        "sla_err": 0.02,
        "estimated_capacity": 20,
    },
    "recommendationservice": {
        "sla_p95_ms": 100,
        "sla_err": 0.02,
        "estimated_capacity": 40,
    },
}

NAMESPACE = "online-boutique"
POLL_INTERVAL = 15          # secondi tra ogni campionamento
PROM_TIMEOUT = 5            # timeout query Prometheus
RATE_WINDOW = "1m"          # finestra rate() — allineata a prometheus_client.py
SAFETY_MARGIN = 0.80        # 80% del knee point come capacita raccomandata

# ============================================================================
# Query Prometheus
# ============================================================================

def prom_query(base_url: str, query: str, timeout: int = PROM_TIMEOUT) -> Optional[float]:
    """Esegue una query Prometheus e restituisce il valore scalare."""
    try:
        resp = requests.get(
            f"{base_url}/api/v1/query",
            params={"query": query},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "success":
            result = data.get("data", {}).get("result", [])
            if result and len(result) > 0:
                val = float(result[0]["value"][1])
                if math.isfinite(val) and val >= 0:
                    return val
    except Exception:
        pass
    return None


def get_service_traffic(prom_url: str, service: str) -> Optional[float]:
    """Traffico in req/s per un servizio (Hubble L7)."""
    query = (
        f'sum(rate(hubble_http_requests_total{{'
        f'destination_workload="{service}",'
        f'destination_namespace="{NAMESPACE}"'
        f'}}[{RATE_WINDOW}]))'
    )
    return prom_query(prom_url, query)


def get_service_latency_p95(prom_url: str, service: str) -> Optional[float]:
    """Latenza p95 in ms per un servizio (Hubble L7 histogram)."""
    query = (
        f'histogram_quantile(0.95,'
        f'sum(rate(hubble_http_request_duration_seconds_bucket{{'
        f'destination_workload="{service}",'
        f'destination_namespace="{NAMESPACE}"'
        f'}}[{RATE_WINDOW}])) by (le)'
        f') * 1000'
    )
    val = prom_query(prom_url, query)
    # Guard +Inf (histogram_quantile edge case)
    if val is not None and val > 60000:
        return None
    return val


def get_service_error_rate(prom_url: str, service: str) -> Optional[float]:
    """Error rate (5xx fraction) per un servizio."""
    total_query = (
        f'sum(rate(hubble_http_requests_total{{'
        f'destination_workload="{service}",'
        f'destination_namespace="{NAMESPACE}"'
        f'}}[{RATE_WINDOW}]))'
    )
    error_query = (
        f'sum(rate(hubble_http_requests_total{{'
        f'destination_workload="{service}",'
        f'destination_namespace="{NAMESPACE}",'
        f'http_status_code=~"5.."'
        f'}}[{RATE_WINDOW}]))'
    )
    total = prom_query(prom_url, total_query)
    errors = prom_query(prom_url, error_query)
    if total and total > 0:
        err = (errors or 0) / total
        return err
    return None


def get_service_cpu(prom_url: str, service: str) -> Optional[float]:
    """CPU usage totale (cores) per tutti i pod di un servizio."""
    query = (
        f'sum(rate(container_cpu_usage_seconds_total{{'
        f'namespace="{NAMESPACE}",'
        f'pod=~"{service}.*",'
        f'container!=""'
        f'}}[{RATE_WINDOW}]))'
    )
    return prom_query(prom_url, query)


def get_pod_count(prom_url: str, service: str) -> Optional[int]:
    """Numero di pod running per un servizio."""
    query = (
        f'count(kube_pod_info{{'
        f'namespace="{NAMESPACE}",'
        f'created_by_name=~"{service}-.*"'
        f'}})'
    )
    val = prom_query(prom_url, query)
    return int(val) if val is not None else None


# ============================================================================
# Snapshot: raccolta metriche per tutti i servizi su un cluster
# ============================================================================

def collect_snapshot(cluster_name: str, prom_url: str) -> Dict:
    """Raccoglie un singolo snapshot di metriche per tutti i servizi."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "cluster": cluster_name,
        "services": {},
    }

    for svc_name in SERVICES:
        svc_data = {}
        svc_data["traffic_rps"] = get_service_traffic(prom_url, svc_name)
        svc_data["latency_p95_ms"] = get_service_latency_p95(prom_url, svc_name)
        svc_data["error_rate"] = get_service_error_rate(prom_url, svc_name)
        svc_data["cpu_cores"] = get_service_cpu(prom_url, svc_name)
        svc_data["pod_count"] = get_pod_count(prom_url, svc_name)

        # Calcola rps/replica
        pods = svc_data["pod_count"]
        rps = svc_data["traffic_rps"]
        if pods and pods > 0 and rps and rps > 0:
            svc_data["rps_per_replica"] = rps / pods
        else:
            svc_data["rps_per_replica"] = None

        # Valuta SLA
        sla_cfg = SERVICES[svc_name]
        p95 = svc_data["latency_p95_ms"]
        err = svc_data["error_rate"]
        sla_ok = True
        if p95 is not None and p95 > sla_cfg["sla_p95_ms"]:
            sla_ok = False
        if err is not None and err > sla_cfg["sla_err"]:
            sla_ok = False
        svc_data["sla_ok"] = sla_ok

        snapshot["services"][svc_name] = svc_data

    return snapshot


# ============================================================================
# Display: tabella live a console
# ============================================================================

def print_header():
    """Stampa l'header della tabella."""
    print()
    print("=" * 100)
    print("  BACKEND CAPACITY OBSERVER — Live Metrics")
    print("=" * 100)
    print(
        f"  {'Service':<25s} | {'rps':>6s} | {'rps/rep':>7s} | "
        f"{'p95':>7s} | {'err%':>5s} | {'CPU':>5s} | {'pods':>4s} | SLA"
    )
    print("  " + "-" * 90)


def print_snapshot(snapshot: Dict, compact: bool = False):
    """Stampa uno snapshot a console."""
    ts = snapshot["timestamp"][11:19]  # HH:MM:SS
    cluster = snapshot["cluster"]

    if not compact:
        print(f"\n  [{ts}] {cluster}")

    for svc_name, svc in snapshot["services"].items():
        rps = svc.get("traffic_rps")
        rps_rep = svc.get("rps_per_replica")
        p95 = svc.get("latency_p95_ms")
        err = svc.get("error_rate")
        cpu = svc.get("cpu_cores")
        pods = svc.get("pod_count")
        sla_ok = svc.get("sla_ok", True)

        rps_str = f"{rps:6.1f}" if rps is not None else "   N/A"
        rps_rep_str = f"{rps_rep:7.1f}" if rps_rep is not None else "    N/A"
        p95_str = f"{p95:6.0f}ms" if p95 is not None else "    N/A"
        err_str = f"{err*100:4.1f}%" if err is not None else "  N/A"
        cpu_str = f"{cpu:5.3f}" if cpu is not None else "  N/A"
        pods_str = f"{pods:4d}" if pods is not None else " N/A"
        sla_str = " OK " if sla_ok else "FAIL"

        # Evidenzia se SLA violato
        prefix = "  "
        if not sla_ok:
            prefix = ">>"

        print(
            f"{prefix}{svc_name:<25s} | {rps_str} | {rps_rep_str} | "
            f"{p95_str} | {err_str} | {cpu_str} | {pods_str} | {sla_str}"
        )


# ============================================================================
# Analisi finale: knee point per servizio
# ============================================================================

def analyze_results(
    all_snapshots: List[Dict],
) -> Dict:
    """
    Analizza tutti gli snapshot raccolti e identifica:
    - Peak rps/replica per servizio (ultimo step con SLA ok)
    - Bottleneck service (primo a violare SLA)
    - Capacita raccomandata per services.yaml
    """
    # Raggruppa per (cluster, servizio) e trova il max rps/replica con SLA ok
    per_service_cluster: Dict[Tuple[str, str], Dict] = {}

    for snap in all_snapshots:
        cluster = snap["cluster"]
        for svc_name, svc_data in snap["services"].items():
            key = (cluster, svc_name)
            if key not in per_service_cluster:
                per_service_cluster[key] = {
                    "max_rps_sla_ok": 0.0,
                    "max_rps_overall": 0.0,
                    "first_sla_fail_ts": None,
                    "samples": 0,
                    "sla_violations": 0,
                }

            entry = per_service_cluster[key]
            entry["samples"] += 1

            rps_rep = svc_data.get("rps_per_replica")
            if rps_rep is not None and rps_rep > entry["max_rps_overall"]:
                entry["max_rps_overall"] = rps_rep

            if svc_data.get("sla_ok") and rps_rep is not None:
                if rps_rep > entry["max_rps_sla_ok"]:
                    entry["max_rps_sla_ok"] = rps_rep
            elif not svc_data.get("sla_ok"):
                entry["sla_violations"] += 1
                if entry["first_sla_fail_ts"] is None:
                    entry["first_sla_fail_ts"] = snap["timestamp"]

    # Genera report per servizio
    report = {}
    for (cluster, svc_name), data in sorted(per_service_cluster.items()):
        if cluster not in report:
            report[cluster] = {}

        knee_rps = data["max_rps_sla_ok"]
        recommended = max(1, round(knee_rps * SAFETY_MARGIN)) if knee_rps > 0 else None
        estimated = SERVICES[svc_name]["estimated_capacity"]

        report[cluster][svc_name] = {
            "knee_rps_per_replica": round(knee_rps, 1),
            "max_rps_per_replica": round(data["max_rps_overall"], 1),
            "recommended_capacity": recommended,
            "current_estimated": estimated,
            "delta_pct": (
                round((knee_rps - estimated) / estimated * 100, 1)
                if knee_rps > 0 and estimated > 0
                else None
            ),
            "samples": data["samples"],
            "sla_violations": data["sla_violations"],
            "first_sla_fail": data["first_sla_fail_ts"],
        }

    return report


def print_final_report(report: Dict):
    """Stampa il report finale con knee points e raccomandazioni."""
    print("\n" + "=" * 100)
    print("  CAPACITY ANALYSIS — Per-Service Knee Points")
    print("=" * 100)

    for cluster_name in sorted(report.keys()):
        cluster_report = report[cluster_name]
        print(f"\n  [{cluster_name}]")
        print(
            f"  {'Service':<25s} | {'Knee rps/rep':>12s} | {'Max rps/rep':>11s} | "
            f"{'Recommended':>11s} | {'Current':>7s} | {'Delta':>7s} | {'Violations':>10s}"
        )
        print("  " + "-" * 100)

        bottleneck_svc = None
        bottleneck_knee = float("inf")

        for svc_name, data in cluster_report.items():
            knee = data["knee_rps_per_replica"]
            max_rps = data["max_rps_per_replica"]
            rec = data["recommended_capacity"]
            est = data["current_estimated"]
            delta = data["delta_pct"]
            viols = data["sla_violations"]

            knee_str = f"{knee:10.1f}" if knee > 0 else "       N/A"
            max_str = f"{max_rps:9.1f}" if max_rps > 0 else "      N/A"
            rec_str = f"{rec:9d}" if rec is not None else "      N/A"
            est_str = f"{est:5d}" if est else "  N/A"
            delta_str = f"{delta:+6.1f}%" if delta is not None else "    N/A"
            viols_str = f"{viols:8d}" if viols > 0 else "       0"

            # Identifica servizio bottleneck
            if knee > 0 and knee < bottleneck_knee:
                bottleneck_knee = knee
                bottleneck_svc = svc_name

            # Flag se la stima corrente e' molto diversa dal misurato
            flag = ""
            if delta is not None and abs(delta) > 30:
                flag = " <<"

            print(
                f"  {svc_name:<25s} | {knee_str} | {max_str} | "
                f"{rec_str} | {est_str} | {delta_str} | {viols_str}{flag}"
            )

        if bottleneck_svc:
            print(f"\n  Bottleneck: {bottleneck_svc} "
                  f"(knee={bottleneck_knee:.1f} rps/replica)")

    # Suggerimenti per services.yaml
    print("\n" + "=" * 100)
    print("  RACCOMANDAZIONI per config/services.yaml")
    print("=" * 100)

    for cluster_name in sorted(report.keys()):
        has_recommendations = False
        for svc_name, data in report[cluster_name].items():
            rec = data["recommended_capacity"]
            est = data["current_estimated"]
            delta = data["delta_pct"]
            if rec is not None and delta is not None and abs(delta) > 15:
                if not has_recommendations:
                    print(f"\n  # {cluster_name}:")
                    has_recommendations = True
                direction = "troppo alta" if delta < 0 else "troppo bassa"
                print(
                    f"  {svc_name}: capacity {est} -> {rec} "
                    f"(stima corrente {direction}, delta {delta:+.1f}%)"
                )

    print("\n" + "=" * 100)


# ============================================================================
# Salvataggio risultati
# ============================================================================

def save_results(all_snapshots: List[Dict], report: Dict, output_dir: str):
    """Salva CSV (serie temporale) e JSON (summary)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV — serie temporale piatta
    csv_path = out_path / f"backend_observer_{ts}.csv"
    csv_rows = []
    for snap in all_snapshots:
        for svc_name, svc_data in snap["services"].items():
            row = {
                "timestamp": snap["timestamp"],
                "cluster": snap["cluster"],
                "service": svc_name,
                "traffic_rps": svc_data.get("traffic_rps"),
                "rps_per_replica": svc_data.get("rps_per_replica"),
                "latency_p95_ms": svc_data.get("latency_p95_ms"),
                "error_rate": svc_data.get("error_rate"),
                "cpu_cores": svc_data.get("cpu_cores"),
                "pod_count": svc_data.get("pod_count"),
                "sla_ok": svc_data.get("sla_ok"),
            }
            csv_rows.append(row)

    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    # JSON — summary
    json_path = out_path / f"backend_observer_{ts}.json"
    summary = {
        "timestamp": ts,
        "duration_seconds": len(all_snapshots) * POLL_INTERVAL,
        "total_snapshots": len(all_snapshots),
        "services_monitored": list(SERVICES.keys()),
        "report": report,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  File CSV:  {csv_path}")
    print(f"  File JSON: {json_path}")


# ============================================================================
# Main loop
# ============================================================================

_running = True


def _signal_handler(sig, frame):
    global _running
    print("\n\n  Ricevuto Ctrl+C — elaborazione risultati...")
    _running = False


def main():
    global _running

    parser = argparse.ArgumentParser(
        description="Backend Capacity Observer — monitora tutti i servizi via Prometheus"
    )
    parser.add_argument(
        "--cluster",
        choices=list(CLUSTERS.keys()),
        help="Osserva solo un cluster specifico (default: tutti)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_INTERVAL,
        help=f"Intervallo di polling in secondi (default: {POLL_INTERVAL})",
    )
    parser.add_argument(
        "--sla-p95",
        type=int,
        help="Override SLA p95 globale per tutti i servizi (ms)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/capacity",
        help="Directory di output (default: results/capacity)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Durata massima in secondi (0 = fino a Ctrl+C)",
    )
    args = parser.parse_args()

    # Override SLA se specificato
    if args.sla_p95:
        for svc in SERVICES.values():
            svc["sla_p95_ms"] = args.sla_p95

    # Seleziona cluster
    if args.cluster:
        target_clusters = {args.cluster: CLUSTERS[args.cluster]}
    else:
        target_clusters = CLUSTERS

    signal.signal(signal.SIGINT, _signal_handler)

    # Banner
    print("\n" + "=" * 100)
    print("  BACKEND CAPACITY OBSERVER")
    print("=" * 100)
    print(f"  Cluster:   {', '.join(target_clusters.keys())}")
    print(f"  Servizi:   {', '.join(SERVICES.keys())}")
    print(f"  Intervallo: {args.interval}s")
    print(f"  Output:    {args.output_dir}")
    if args.duration > 0:
        print(f"  Durata:    {args.duration}s")
    else:
        print(f"  Durata:    fino a Ctrl+C")
    print()
    print("  Assicurati che il test di carico (k6/Locust) sia in esecuzione!")
    print("  Premi Ctrl+C per terminare e generare il report.")
    print("=" * 100)

    # Verifica connettivita Prometheus
    print("\n  Verifica connettivita Prometheus...")
    for cname, cdata in target_clusters.items():
        try:
            resp = requests.get(
                f"{cdata['prometheus']}/api/v1/query",
                params={"query": "up"},
                timeout=3,
            )
            if resp.status_code == 200:
                print(f"    {cname}: OK ({cdata['prometheus']})")
            else:
                print(f"    {cname}: HTTP {resp.status_code} ({cdata['prometheus']})")
        except Exception as e:
            print(f"    {cname}: ERRORE - {e}")
            print(f"    Verifica che la VPN sia attiva e Prometheus raggiungibile")

    print_header()

    all_snapshots = []
    start_time = time.time()
    iteration = 0

    while _running:
        if args.duration > 0 and (time.time() - start_time) > args.duration:
            print(f"\n  Durata massima raggiunta ({args.duration}s)")
            break

        for cname, cdata in target_clusters.items():
            snapshot = collect_snapshot(cname, cdata["prometheus"])
            all_snapshots.append(snapshot)
            print_snapshot(snapshot)

        iteration += 1
        if iteration % 10 == 0:
            # Ri-stampa header ogni 10 iterazioni per leggibilita
            print_header()

        # Wait
        for _ in range(args.interval):
            if not _running:
                break
            time.sleep(1)

    # Analisi finale
    if all_snapshots:
        report = analyze_results(all_snapshots)
        print_final_report(report)
        save_results(all_snapshots, report, args.output_dir)
    else:
        print("\n  Nessun dato raccolto.")


if __name__ == "__main__":
    main()
