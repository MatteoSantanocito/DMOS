"""
Frontend Capacity Test — Multi-Ingress Edition
================================================
Calibra capacity_req_per_sec testando UNA replica isolata su CIASCUN cluster,
per misurare la capacità reale sotto traffico multi-ingress.

DIFFERENZE RISPETTO AL CAPACITY TEST SINGLE-INGRESS:
  - Il test originale testava solo cluster1 (192.168.1.245)
  - Questo test può essere eseguito su ciascun cluster singolarmente,
    oppure in modalità sequenziale automatica (tutti e 3 in serie)
  - Misura la capacità reale per-cluster, tenendo conto che il traffico
    multi-ingress attraversa percorsi di rete diversi
  - Produce un report comparativo tra i cluster per identificare
    eventuali differenze di performance

PERCHÉ SERVE:
  Il capacity_req_per_sec=25 è stato calibrato con test single-ingress
  (tutto il traffico su cluster1). Con multi-ingress, le richieste che
  entrano da cluster2/3 possono avere latenza diversa perché:
    1. I backend (cart, checkout, ecc.) potrebbero non essere co-locati
    2. I tunnel Cilium cross-cluster aggiungono latenza
    3. Le risorse hardware dei 3 cluster possono differire in pratica
  Ricalibrazione necessaria per avere provisioning ratio accurato.

MODALITÀ DI ESECUZIONE:

  Modalità 1 — Test singolo cluster (raccomandato per rigore scientifico):
  =========================================================================
  Testare un cluster alla volta, con 1 replica SOLO su quel cluster.

    # Test cluster1 (DE):
    kubectl --context cluster1 scale deploy/frontend -n online-boutique --replicas=1
    kubectl --context cluster2 scale deploy/frontend -n online-boutique --replicas=0
    kubectl --context cluster3 scale deploy/frontend -n online-boutique --replicas=0
    # STOP DMOS!
    $env:DMOS_TARGET_CLUSTER="cluster1"
    locust -f capacity_test_multiingress.py --users 200 --spawn-rate 5 --run-time 10m

    # Test cluster2 (FR):
    kubectl --context cluster1 scale deploy/frontend -n online-boutique --replicas=0
    kubectl --context cluster2 scale deploy/frontend -n online-boutique --replicas=1
    kubectl --context cluster3 scale deploy/frontend -n online-boutique --replicas=0
    $env:DMOS_TARGET_CLUSTER="cluster2"
    locust -f capacity_test_multiingress.py --users 200 --spawn-rate 5 --run-time 10m

    # Test cluster3 (PL):
    kubectl --context cluster1 scale deploy/frontend -n online-boutique --replicas=0
    kubectl --context cluster2 scale deploy/frontend -n online-boutique --replicas=0
    kubectl --context cluster3 scale deploy/frontend -n online-boutique --replicas=1
    $env:DMOS_TARGET_CLUSTER="cluster3"
    locust -f capacity_test_multiingress.py --users 200 --spawn-rate 5 --run-time 10m

  Modalità 2 — Test con backend distribuiti (scenario realistico):
  =================================================================
  Testa un cluster alla volta MA con i backend attivi su TUTTI i cluster
  (come in produzione). Questo misura la capacità includendo le chiamate
  gRPC cross-cluster.

    kubectl --context cluster1 scale deploy/frontend -n online-boutique --replicas=1
    kubectl --context cluster2 scale deploy/frontend -n online-boutique --replicas=0
    kubectl --context cluster3 scale deploy/frontend -n online-boutique --replicas=0
    # NON toccare i backend! Restano distribuiti su tutti i cluster.
    $env:DMOS_TARGET_CLUSTER="cluster1"
    $env:DMOS_BACKEND_MODE="distributed"
    locust -f capacity_test_multiingress.py --users 200 --spawn-rate 5 --run-time 10m

  Modalità 3 — Test con backend co-locati (best-case):
  =====================================================
  Frontend + tutti i backend su UN SOLO cluster. Misura la capacità
  senza overhead cross-cluster (uguale al test originale single-ingress).

    # Scala tutti i backend a 0 su cluster2/3
    for svc in cartservice productcatalogservice checkoutservice recommendationservice currencyservice shippingservice paymentservice emailservice adservice; do
      kubectl --context cluster2 scale deploy/$svc -n online-boutique --replicas=0
      kubectl --context cluster3 scale deploy/$svc -n online-boutique --replicas=0
    done
    $env:DMOS_TARGET_CLUSTER="cluster1"
    $env:DMOS_BACKEND_MODE="colocated"
    locust -f capacity_test_multiingress.py --users 200 --spawn-rate 5 --run-time 10m

OUTPUT:
  - Report in tempo reale con SLO breach detection
  - CSV con risultati per-finestra per analisi successiva
  - Raccomandazione capacity_req_per_sec per ogni soglia SLO
  - Confronto con il valore attuale (25 rps)
"""

import os
import csv
import time
import random
import datetime
from pathlib import Path
from collections import defaultdict

from locust import HttpUser, task, between, events


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CLUSTER_CONFIG = {
    "cluster1": {"host": "http://192.168.1.245:30007", "region": "DE (Frankfurt)"},
    "cluster2": {"host": "http://192.168.1.246:30007", "region": "FR (Parigi)"},
    "cluster3": {"host": "http://192.168.1.247:30007", "region": "PL (Varsavia)"},
}

# Quale cluster testare — impostato via env o default a cluster1
TARGET_CLUSTER = os.environ.get("DMOS_TARGET_CLUSTER", "cluster1")
BACKEND_MODE = os.environ.get("DMOS_BACKEND_MODE", "isolated")  # isolated | distributed | colocated

if TARGET_CLUSTER not in CLUSTER_CONFIG:
    raise ValueError(f"DMOS_TARGET_CLUSTER='{TARGET_CLUSTER}' non valido. "
                     f"Usa: {list(CLUSTER_CONFIG.keys())}")

TARGET_HOST = CLUSTER_CONFIG[TARGET_CLUSTER]["host"]
TARGET_REGION = CLUSTER_CONFIG[TARGET_CLUSTER]["region"]

# SLO thresholds to check (ms)
SLO_THRESHOLDS = [200, 300, 400, 500]

# Current value to compare against
CURRENT_CAPACITY = 25  # capacity_req_per_sec attuale

# Metrics collection
WINDOW_SIZE = 10  # seconds per window
window_stats = defaultdict(lambda: {
    "requests": 0,
    "failures": 0,
    "response_times": [],
})

# SLO breach tracking
capacity_found = {}        # {slo_ms: rps_at_breach}
capacity_sustained = {}    # {slo_ms: max_rps_before_sustained_breach}

# Products and currencies for realistic behavior
PRODUCTS = [
    "OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O", "L9ECAV7KIM",
    "2ZYFJ3GM2N", "0PUK6V6EV0", "LS4PSXUNUM", "9SIQT8TOJO", "6E92ZMYYFZ",
]
CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CAD"]


# ═══════════════════════════════════════════════════════════════════════════════
# User Definition
# ═══════════════════════════════════════════════════════════════════════════════

class FrontendUser(HttpUser):
    """
    Simula un utente Online Boutique.
    L'host è impostato dinamicamente in base al cluster target.
    Il mix di task replica quello del locustfile_multiingress.py per coerenza.
    """
    host = TARGET_HOST
    wait_time = between(1, 3)  # Stesso wait_time del multi-ingress

    @task(40)
    def browse_homepage(self):
        """Homepage — trigger productcatalog + recommendation + ad."""
        with self.client.get("/", catch_response=True) as resp:
            _track(resp)
            if resp.status_code != 200:
                resp.failure(f"Homepage: {resp.status_code}")

    @task(25)
    def view_product(self):
        """Dettaglio prodotto — trigger productcatalog + recommendation."""
        pid = random.choice(PRODUCTS)
        with self.client.get(f"/product/{pid}", catch_response=True,
                             name="GET /product/[id]") as resp:
            _track(resp)
            if resp.status_code != 200:
                resp.failure(f"Product: {resp.status_code}")

    @task(15)
    def add_to_cart(self):
        """Aggiungi al carrello — trigger cartservice."""
        pid = random.choice(PRODUCTS)
        with self.client.post("/cart",
                              data={"product_id": pid, "quantity": random.randint(1, 3)},
                              catch_response=True, name="POST /cart") as resp:
            _track(resp)
            if resp.status_code not in (200, 302):
                resp.failure(f"Cart: {resp.status_code}")

    @task(10)
    def view_cart(self):
        """Visualizza carrello — trigger cartservice + recommendation."""
        with self.client.get("/cart", catch_response=True) as resp:
            _track(resp)
            if resp.status_code != 200:
                resp.failure(f"Cart view: {resp.status_code}")

    @task(5)
    def set_currency(self):
        """Cambia valuta — trigger currencyservice."""
        currency = random.choice(CURRENCIES)
        with self.client.post("/setCurrency",
                              data={"currency_code": currency},
                              catch_response=True, name="POST /setCurrency") as resp:
            _track(resp)
            if resp.status_code not in (200, 302):
                resp.failure(f"Currency: {resp.status_code}")

    @task(5)
    def checkout(self):
        """Checkout completo — trigger checkout + payment + shipping + email."""
        with self.client.post("/cart/checkout",
                              data={
                                  "email": "test@example.com",
                                  "street_address": "123 Test St",
                                  "zip_code": "10001",
                                  "city": "New York",
                                  "state": "NY",
                                  "country": "US",
                                  "credit_card_number": "4432801561520454",
                                  "credit_card_expiration_month": "1",
                                  "credit_card_expiration_year": "2030",
                                  "credit_card_cvv": "672",
                              },
                              catch_response=True, name="POST /cart/checkout") as resp:
            _track(resp)
            if resp.status_code not in (200, 302):
                resp.failure(f"Checkout: {resp.status_code}")


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def _track(resp):
    """Track response in the current time window."""
    window = int(time.time()) // WINDOW_SIZE
    rt_ms = resp.elapsed.total_seconds() * 1000
    window_stats[window]["requests"] += 1
    window_stats[window]["response_times"].append(rt_ms)
    if resp.status_code >= 400:
        window_stats[window]["failures"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Real-time Reporter
# ═══════════════════════════════════════════════════════════════════════════════

_last_window = 0
_consecutive_breaches = defaultdict(int)  # {slo_ms: count of consecutive breaches}
SUSTAINED_BREACH_THRESHOLD = 3  # finestre consecutive per considerare il breach "stabile"


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"\n{'='*80}")
    print(f"  FRONTEND CAPACITY TEST — Multi-Ingress Edition")
    print(f"{'='*80}")
    print(f"  Target cluster:    {TARGET_CLUSTER} — {TARGET_REGION}")
    print(f"  Target host:       {TARGET_HOST}")
    print(f"  Backend mode:      {BACKEND_MODE}")
    print(f"  Current capacity:  {CURRENT_CAPACITY} rps (da ricalibriare)")
    print(f"  SLO thresholds:    {SLO_THRESHOLDS} ms (p95)")
    print(f"  Window size:       {WINDOW_SIZE}s")
    print(f"{'='*80}")
    print(f"\n  {'Time':>8} | {'Users':>5} | {'RPS':>6} | {'p50':>6} | {'p95':>6} | "
          f"{'p99':>6} | {'Fail%':>6} | SLO Status")
    print(f"  {'-'*78}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    global _last_window

    current_window = int(time.time()) // WINDOW_SIZE

    if current_window > _last_window and _last_window in window_stats:
        stats = window_stats[_last_window]
        rts = sorted(stats["response_times"])

        if len(rts) >= 5:
            n = len(rts)
            p50 = rts[int(n * 0.50)]
            p95 = rts[int(n * 0.95)]
            p99 = rts[min(int(n * 0.99), n - 1)]
            rps = stats["requests"] / WINDOW_SIZE
            fail_pct = (stats["failures"] / stats["requests"] * 100) if stats["requests"] > 0 else 0

            # Check SLO breaches
            slo_parts = []
            for slo in SLO_THRESHOLDS:
                if p95 <= slo:
                    slo_parts.append(f"\u2705{slo}")
                    _consecutive_breaches[slo] = 0
                else:
                    slo_parts.append(f"\u274C{slo}")
                    _consecutive_breaches[slo] += 1

                    # First breach
                    if slo not in capacity_found:
                        capacity_found[slo] = rps
                        print(f"\n  \u26A0\uFE0F  FIRST BREACH: p95={p95:.0f}ms > {slo}ms "
                              f"at {rps:.1f} rps ({_last_window})\n")

                    # Sustained breach (3+ consecutive windows)
                    if (_consecutive_breaches[slo] >= SUSTAINED_BREACH_THRESHOLD
                            and slo not in capacity_sustained):
                        capacity_sustained[slo] = rps
                        print(f"\n  \U0001F6A8  SUSTAINED BREACH: p95 > {slo}ms "
                              f"for {SUSTAINED_BREACH_THRESHOLD} consecutive windows "
                              f"at ~{rps:.1f} rps\n")

            # Get user count
            try:
                users = on_request._runner.user_count if on_request._runner else "?"
            except Exception:
                users = "?"

            ts = time.strftime("%H:%M:%S", time.localtime(_last_window * WINDOW_SIZE))
            print(f"  {ts:>8} | {str(users):>5} | {rps:6.1f} | {p50:6.0f} | "
                  f"{p95:6.0f} | {p99:6.0f} | {fail_pct:5.1f}% | {' '.join(slo_parts)}")

        _last_window = current_window
    elif current_window > _last_window:
        _last_window = current_window


@events.init.add_listener
def on_init(environment, **kwargs):
    on_request._runner = environment.runner


# ═══════════════════════════════════════════════════════════════════════════════
# Final Report
# ═══════════════════════════════════════════════════════════════════════════════

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    output_dir = Path("results/capacity_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*80}")
    print(f"  CAPACITY TEST RESULTS — {TARGET_CLUSTER} ({TARGET_REGION})")
    print(f"  Backend mode: {BACKEND_MODE}")
    print(f"{'='*80}")

    # ── Capacity per SLO ──
    print(f"\n  Capacit\u00e0 per replica a diverse soglie SLO:")
    print(f"  {'SLO (p95)':>12} | {'1st Breach':>12} | {'Sustained':>12} | {'Raccomandato':>14} | {'vs Attuale':>12}")
    print(f"  {'-'*68}")

    recommendations = {}
    for slo in SLO_THRESHOLDS:
        first = capacity_found.get(slo)
        sustained = capacity_sustained.get(slo)

        # Use sustained breach if available, otherwise first breach
        ref_rps = sustained if sustained else first
        if ref_rps:
            # Safety factor 80% (come nell'originale)
            recommended = int(ref_rps * 0.8)
            recommendations[slo] = recommended
            delta = recommended - CURRENT_CAPACITY
            delta_str = f"{delta:+d} rps" if delta != 0 else "uguale"
            print(f"  < {slo:>4}ms   | {first:>10.1f} rps | "
                  f"{sustained if sustained else 'N/A':>10} | "
                  f"{recommended:>10} rps   | {delta_str:>10}")
        else:
            print(f"  < {slo:>4}ms   | {'N/A':>12} | {'N/A':>12} | "
                  f"{'> max test':>14} | {'N/A':>12}")

    # ── Recommendation ──
    print(f"\n  {'='*68}")
    print(f"  RACCOMANDAZIONE per {TARGET_CLUSTER}:")

    if 300 in recommendations:
        rec = recommendations[300]
        print(f"\n    capacity_req_per_sec: {rec}  (SLO p95 < 300ms, safety \u00d70.8)")
        if rec > CURRENT_CAPACITY:
            print(f"    \u2191 Aumento da {CURRENT_CAPACITY} a {rec} (+{rec - CURRENT_CAPACITY} rps)")
            print(f"    Effetto: provisioning ratio salir\u00e0, under-provisioning diminuir\u00e0")
        elif rec < CURRENT_CAPACITY:
            print(f"    \u2193 Riduzione da {CURRENT_CAPACITY} a {rec} ({rec - CURRENT_CAPACITY} rps)")
            print(f"    Effetto: pi\u00f9 repliche allocate, minore rischio saturazione")
        else:
            print(f"    \u2192 Valore attuale {CURRENT_CAPACITY} confermato")
    elif 200 in recommendations:
        rec = recommendations[200]
        print(f"\n    capacity_req_per_sec: {rec}  (SLO p95 < 200ms, safety \u00d70.8)")
    else:
        print(f"\n    Nessun breach rilevato! Il frontend regge tutto il carico testato.")
        print(f"    Eseguire con pi\u00f9 utenti (--users 300+) per trovare il limite.")

    # ── CSV output ──
    csv_file = output_dir / f"capacity_{TARGET_CLUSTER}_{BACKEND_MODE}_{timestamp}.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "window", "requests", "rps",
            "p50_ms", "p90_ms", "p95_ms", "p99_ms",
            "fail_pct", "cluster", "region", "backend_mode",
        ])
        for w in sorted(window_stats.keys()):
            s = window_stats[w]
            rts = sorted(s["response_times"])
            if len(rts) >= 2:
                n = len(rts)
                writer.writerow([
                    time.strftime("%H:%M:%S", time.localtime(w * WINDOW_SIZE)),
                    w,
                    s["requests"],
                    round(s["requests"] / WINDOW_SIZE, 1),
                    round(rts[int(n * 0.50)], 1),
                    round(rts[int(n * 0.90)], 1),
                    round(rts[int(n * 0.95)], 1),
                    round(rts[min(int(n * 0.99), n - 1)], 1),
                    round(s["failures"] / s["requests"] * 100, 2) if s["requests"] > 0 else 0,
                    TARGET_CLUSTER,
                    TARGET_REGION,
                    BACKEND_MODE,
                ])
    print(f"\n  CSV salvato: {csv_file}")

    # ── Quick comparison table ──
    print(f"\n  Tabella per-finestra (ultime 10):")
    print(f"  {'Time':>8} | {'RPS':>6} | {'p50':>6} | {'p95':>6} | {'p99':>6} | {'Fail%':>6}")
    print(f"  {'-'*52}")
    sorted_windows = sorted(window_stats.keys())
    for w in sorted_windows[-10:]:
        s = window_stats[w]
        rts = sorted(s["response_times"])
        if len(rts) >= 2:
            n = len(rts)
            ts = time.strftime("%H:%M:%S", time.localtime(w * WINDOW_SIZE))
            rps = s["requests"] / WINDOW_SIZE
            p50 = rts[int(n * 0.50)]
            p95 = rts[int(n * 0.95)]
            p99 = rts[min(int(n * 0.99), n - 1)]
            fail = s["failures"] / s["requests"] * 100 if s["requests"] > 0 else 0
            print(f"  {ts:>8} | {rps:6.1f} | {p50:6.0f} | {p95:6.0f} | {p99:6.0f} | {fail:5.1f}%")

    print(f"\n{'='*80}")
    print(f"  Test completato per {TARGET_CLUSTER}.")
    if TARGET_CLUSTER == "cluster1":
        print(f"  Prossimo: $env:DMOS_TARGET_CLUSTER=\"cluster2\"")
    elif TARGET_CLUSTER == "cluster2":
        print(f"  Prossimo: $env:DMOS_TARGET_CLUSTER=\"cluster3\"")
    else:
        print(f"  Tutti i cluster testati! Confronta i CSV in results/capacity_test/")
    print(f"{'='*80}\n")