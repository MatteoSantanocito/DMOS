"""
check_prometheus.py — Verifica che ogni Prometheus per-cluster
abbia le metriche necessarie per DMOS.

Eseguire PRIMA di ogni esperimento per validare il setup.

Usage:
    python scripts/check_prometheus.py
"""

import requests
import json
import sys
from typing import Optional

# ─── Configurazione cluster ────────────────────────────────────────────────
CLUSTERS = {
    "cluster1": "http://192.168.1.245:30090",
    "cluster2": "http://192.168.1.246:30090",
    "cluster3": "http://192.168.1.247:30090",
}
NAMESPACE = "online-boutique"
TIMEOUT = 5

# ─── Colori terminale ──────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")


def prom_query(url: str, query: str) -> Optional[list]:
    """Esegue una query PromQL, restituisce la lista result o None."""
    try:
        r = requests.get(f"{url}/api/v1/query", params={"query": query}, timeout=TIMEOUT)
        data = r.json()
        if data.get("status") == "success":
            result = data.get("data", {}).get("result", [])
            return result
        return None
    except Exception as e:
        return None


def check_cluster(name: str, url: str):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Cluster: {name}  ({url}){RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # ── 1. Connettività base ────────────────────────────────────────────
    print(f"\n  [1] Connettività Prometheus")
    try:
        r = requests.get(f"{url}/-/healthy", timeout=TIMEOUT)
        if r.status_code == 200:
            ok(f"Prometheus raggiungibile ({url})")
        else:
            fail(f"HTTP {r.status_code} da {url}/-/healthy")
            return
    except Exception as e:
        fail(f"Connessione fallita: {e}")
        return

    # ── 2. kube-state-metrics ───────────────────────────────────────────
    print(f"\n  [2] kube-state-metrics (necessario per CPU/memoria disponibile)")

    q = 'sum(kube_node_status_capacity{resource="cpu"})'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = float(res[0]["value"][1])
        ok(f"kube_node_status_capacity cpu = {val:.1f} cores")
    else:
        fail(f"kube_node_status_capacity NON trovata — kube-state-metrics non installato?")
        warn("FIX: helm install kube-state-metrics prometheus-community/kube-state-metrics -n monitoring")

    q = 'sum(kube_node_status_capacity{resource="memory"})'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = float(res[0]["value"][1]) / (1024**3)
        ok(f"kube_node_status_capacity memory = {val:.1f} GB")
    else:
        fail("kube_node_status_capacity memory NON trovata")

    # ── 3. kubelet cAdvisor (sempre necessario) ─────────────────────────
    print(f"\n  [3] kubelet cAdvisor (CPU usage, memoria, network bytes)")

    q = f'sum(rate(container_cpu_usage_seconds_total{{image!="", namespace="{NAMESPACE}"}}[2m]))'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = float(res[0]["value"][1])
        ok(f"container_cpu_usage_seconds_total ({NAMESPACE}) = {val:.3f} cores/s")
    else:
        warn(f"container_cpu_usage (namespace={NAMESPACE}) = 0 o non trovato (OK se nessun pod attivo)")

    q = f'sum(container_memory_working_set_bytes{{image!="", namespace="{NAMESPACE}"}})'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = float(res[0]["value"][1]) / (1024**2)
        ok(f"container_memory_working_set_bytes ({NAMESPACE}) = {val:.0f} Mi")
    else:
        warn(f"container_memory_working_set_bytes = 0 (OK se nessun pod)")

    q = f'sum(rate(container_network_receive_bytes_total{{namespace="{NAMESPACE}"}}[1m]))'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = float(res[0]["value"][1])
        ok(f"container_network_receive_bytes_total ({NAMESPACE}) = {val:.0f} B/s  (~{val/4000:.1f} req/s stima)")
    else:
        warn(f"container_network_receive_bytes = 0 (OK se nessun traffico attivo)")

    # ── 4. Hubble L7 HTTP metrics ───────────────────────────────────────
    print(f"\n  [4] Hubble HTTP metrics (Try 1 in catena fallback — OTTIMALE per rps)")

    q = 'hubble_http_requests_total'
    res = prom_query(url, q)
    if res:
        ok(f"hubble_http_requests_total presente ({len(res)} serie)")

        # Check traffico INGRESS verso online-boutique (Cilium 1.16.x httpV2)
        # traffic_direction="INGRESS" = traffico entrante verso i pod
        q2 = f'sum(rate(hubble_http_requests_total{{destination_namespace="{NAMESPACE}",traffic_direction="INGRESS"}}[1m]))'
        res2 = prom_query(url, q2)
        if res2 and res2[0].get("value"):
            val2 = float(res2[0]["value"][1])
            if val2 > 0:
                ok(f"Traffico HTTP INGRESS verso {NAMESPACE}: {val2:.2f} rps ✓")
            else:
                # prova senza filtro direction (magari il traffico è EGRESS o altro)
                q2b = f'sum(rate(hubble_http_requests_total{{destination_namespace="{NAMESPACE}"}}[1m]))'
                res2b = prom_query(url, q2b)
                val2b = float(res2b[0]["value"][1]) if res2b and res2b[0].get("value") else 0
                if val2b > 0:
                    ok(f"Traffico HTTP verso {NAMESPACE}: {val2b:.2f} rps (direction=any) ✓")
                else:
                    warn(f"hubble_http_requests_total rate = 0 — CiliumNetworkPolicy L7 applicata?")
                    warn("FIX: kubectl apply -f deployments/cilium-l7-visibility.yaml")

        # Mostra breakdown per traffic_direction
        q3 = f'sum by (traffic_direction) (rate(hubble_http_requests_total{{destination_namespace="{NAMESPACE}"}}[1m]))'
        res3 = prom_query(url, q3)
        if res3:
            for r in res3:
                td  = r.get("metric", {}).get("traffic_direction", "?")
                val = float(r.get("value", [0, 0])[1])
                if val > 0:
                    ok(f"  traffic_direction={td}: {val:.2f} rps")
    else:
        fail("hubble_http_requests_total NON presente — Hubble HTTP non abilitato")
        warn("FIX: abilita Hubble metrics su Cilium:")
        warn("  helm upgrade cilium ... --set hubble.enabled=true \\")
        warn("       --set hubble.metrics.enabled='{http,drop,tcp,flow,icmp,dns}'")
        warn("  Oppure: kubectl -n kube-system edit configmap cilium-config")
        warn("  e aggiungi: hubble-metrics-server: ':9091'")
        warn("              hubble-metrics: http drop tcp flow icmp dns")

    # ── 5. Hubble latency histogram ─────────────────────────────────────
    print(f"\n  [5] Hubble latency histogram (per p95 latency score)")

    q = 'hubble_http_request_duration_seconds_bucket'
    res = prom_query(url, q)
    if res:
        ok(f"hubble_http_request_duration_seconds_bucket presente ({len(res)} serie)")
        # Test query p95 — usa label Cilium 1.16.x (destination_workload, traffic_direction)
        p95_queries = [
            # Formato corretto per Cilium 1.16.x httpV2
            f'histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{{destination_workload="frontend",traffic_direction="INGRESS"}}[5m])) by (le)) * 1000',
            # Senza filtro traffic_direction
            f'histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{{destination_workload="frontend"}}[5m])) by (le)) * 1000',
            # Per tutto il namespace
            f'histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{{destination_namespace="{NAMESPACE}"}}[5m])) by (le)) * 1000',
        ]
        p95_ok = False
        for p95q in p95_queries:
            res2 = prom_query(url, p95q)
            if res2 and res2[0].get("value"):
                val = float(res2[0]["value"][1])
                if 0 < val < 60000:
                    ok(f"p95 latency frontend via Hubble = {val:.0f} ms ✓")
                    p95_ok = True
                    break
        if not p95_ok:
            # Mostra labels disponibili per debug
            res4 = prom_query(url, 'hubble_http_request_duration_seconds_bucket')
            if res4:
                labels = list(res4[0].get("metric", {}).keys())
                warn(f"p95 = 0 o NaN (nessun traffico HTTP attivo o L7 policy mancante)")
                warn(f"Labels disponibili: {labels}")
            else:
                warn("p95 latency = nessun dato (avvia traffico con Locust per generare metriche)")
    else:
        fail("hubble_http_request_duration_seconds_bucket NON presente")
        warn("La latency p95 userà il fallback config (baseline_ms=2.0) — phi_lat ≈ 0.10 su tutti i cluster")
        warn("→ La componente latency del score non discrimina tra cluster (tutti uguali)")

    # ── 6. Kube pod info (per get_pod_count — legacy, verifica comunque) ─
    print(f"\n  [6] kube_pod_info (per conteggio pod)")

    q = f'count(kube_pod_info{{namespace="{NAMESPACE}"}})'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = int(float(res[0]["value"][1]))
        ok(f"kube_pod_info ({NAMESPACE}) = {val} pod totali")
    else:
        warn("kube_pod_info non trovato — OK se kube-state-metrics non ha ancora scraped")

    # ── 7. Verifica metriche per servizio specifico (frontend) ──────────
    print(f"\n  [7] Verifica per servizio 'frontend'")

    # CPU usage per frontend
    q = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}", pod=~"frontend-.*"}}[5m]))'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = float(res[0]["value"][1])
        ok(f"CPU usage frontend = {val*100:.1f}%")
    else:
        warn("CPU usage frontend = non disponibile (nessun pod frontend attivo?)")

    # Network bytes per frontend
    q = f'sum(rate(container_network_receive_bytes_total{{namespace="{NAMESPACE}", pod=~"frontend.*"}}[1m]))'
    res = prom_query(url, q)
    if res and res[0].get("value"):
        val = float(res[0]["value"][1])
        ok(f"Network bytes frontend = {val:.0f} B/s (~{val/4000:.2f} rps stimati)")
    else:
        warn("Network bytes frontend = 0")

    # ── 8. DMOS metrics (porta 9090 su localhost) ───────────────────────
    # (solo per cluster1 che è dove gira DMOS)
    if "245" in url:
        print(f"\n  [8] DMOS metrics (localhost:9090 — solo cluster1)")
        try:
            r = requests.get("http://localhost:9090/metrics", timeout=3)
            if r.status_code == 200:
                lines = [l for l in r.text.split("\n") if "dmos_" in l and not l.startswith("#")]
                if lines:
                    ok(f"DMOS /metrics raggiungibile — {len(lines)} metriche DMOS")
                    # Mostra sample
                    for l in lines[:5]:
                        print(f"    {l}")
                else:
                    warn("DMOS /metrics risponde ma nessuna metrica dmos_ presente (DMOS non ancora avviato?)")
            else:
                fail(f"DMOS /metrics: HTTP {r.status_code}")
        except Exception as e:
            fail(f"DMOS metrics non raggiungibile: {e}")

    # ── 9. Locust API ───────────────────────────────────────────────────
    if "245" in url:
        print(f"\n  [9] Locust API (localhost:8089)")
        try:
            r = requests.get("http://localhost:8089/stats/requests", timeout=2)
            if r.status_code == 200:
                data = r.json()
                rps = data.get("total_rps", 0)
                users = data.get("user_count", 0)
                ok(f"Locust API raggiungibile — {rps:.1f} rps, {users} users")
            else:
                warn(f"Locust API: HTTP {r.status_code}")
        except Exception:
            warn("Locust non raggiungibile (OK se test non è ancora partito)")


def main():
    print(f"\n{BOLD}{'='*60}")
    print("  DMOS Prometheus Health Check")
    print(f"{'='*60}{RESET}")
    print("  Verifica metriche necessarie per ogni cluster.")
    print("  Eseguire PRIMA di ogni esperimento.\n")

    results = {}
    for name, url in CLUSTERS.items():
        check_cluster(name, url)

    print(f"\n{BOLD}{'='*60}")
    print("  Fine check. Rivedi i ⚠️  e ❌ sopra.")
    print(f"{'='*60}{RESET}\n")

    print("  Metriche critiche per DMOS:")
    print("  ✅ kube_node_status_capacity   → capacity score (hard constraint)")
    print("  ✅ container_cpu_usage         → CPU disponibile per scheduling")
    print("  ✅ hubble_http_requests_total  → rps L7 reali (Cilium 1.16.x httpV2)")
    print("  ✅ hubble_http_request_duration → p95 latency L7 (Cilium 1.16.x httpV2)")
    print()
    print("  Query corrette per Cilium 1.16.x (httpV2 labelsContext):")
    print(f"    RPS:     sum(rate(hubble_http_requests_total{{destination_namespace=\"online-boutique\",traffic_direction=\"INGRESS\"}}[1m]))")
    print(f"    p95 lat: histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{{destination_namespace=\"online-boutique\"}}[5m])) by (le)) * 1000")
    print()
    print("  Note: le metriche HTTP L7 richiedono traffico attivo + CiliumNetworkPolicy L7.")
    print("  Se assenti: DMOS usa fallback network_bytes/4000 (rps) e baseline_latency_ms=2.0.")
    print()


if __name__ == "__main__":
    main()
