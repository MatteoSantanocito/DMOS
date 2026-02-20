"""
Frontend Capacity Test for DMOS Calibration
============================================
Scales ONE replica of frontend and gradually increases load
to find the true capacity_req_per_sec at different SLO thresholds.

USAGE:
  1. STOP DMOS (altrimenti ri-scala le repliche):
     Ctrl+C su DMOS

  2. Scale frontend a 1 replica su UN SOLO cluster:
     kubectl scale deploy/frontend -n online-boutique --replicas=1 --kubeconfig .kube/cluster1.yaml
     kubectl scale deploy/frontend -n online-boutique --replicas=0 --kubeconfig .kube/cluster2.yaml
     kubectl scale deploy/frontend -n online-boutique --replicas=0 --kubeconfig .kube/cluster3.yaml

  3. Verifica:
     kubectl get deploy frontend -n online-boutique --kubeconfig .kube/cluster1.yaml

  4. Lancia il test (PowerShell):
     locust -f capacity_test_locustfile.py --host http://192.168.1.245:30007 ^
            --users 200 --spawn-rate 5 --run-time 10m --html capacity_report.html

  5. Analizza i risultati: guarda dove p95 supera la soglia SLO.

  6. Dopo il test, rilancia DMOS e lascia che gestisca le repliche.
"""

import time
import random
import datetime
from locust import FastHttpUser, task, between, events
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# Metrics tracking
# ═══════════════════════════════════════════════════════════════════════════════

window_stats = defaultdict(lambda: {"requests": 0, "failures": 0, "response_times": []})
WINDOW_SIZE = 10  # seconds
SLO_THRESHOLDS = [200, 300, 400, 500]  # ms - p95 targets to check

# Product IDs from Online Boutique catalog
PRODUCTS = [
    '0PUK6V6EV0', '1YMWWN1N4O', '2ZYFJ3GM2N', '66VCHSJNUP',
    '6E92ZMYYFZ', '9SIQT8TOJO', 'L9ECAV7KIM', 'LS4PSXUNUM', 'OLJCESPC7Z'
]
CURRENCIES = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY']


# ═══════════════════════════════════════════════════════════════════════════════
# User (identico al locustfile_scenarios.py — basato sull'ufficiale Google)
# ═══════════════════════════════════════════════════════════════════════════════

class FrontendUser(FastHttpUser):
    """Task mix identico al locustfile_scenarios.py per calibrazione accurata."""
    wait_time = between(1, 5)

    @task(10)
    def browse_product(self):
        self.client.get("/product/" + random.choice(PRODUCTS))

    @task(5)
    def browse_homepage(self):
        self.client.get("/")

    @task(3)
    def view_cart(self):
        self.client.get("/cart")

    @task(2)
    def set_currency(self):
        self.client.post("/setCurrency", {
            'currency_code': random.choice(CURRENCIES)
        })

    @task(2)
    def add_to_cart(self):
        product = random.choice(PRODUCTS)
        self.client.get("/product/" + product)
        self.client.post("/cart", {
            'product_id': product,
            'quantity': random.randint(1, 5)
        })

    @task(1)
    def checkout(self):
        product = random.choice(PRODUCTS)
        self.client.get("/product/" + product)
        self.client.post("/cart", {
            'product_id': product,
            'quantity': 1
        })
        current_year = datetime.datetime.now().year + 1
        self.client.post("/cart/checkout", {
            'email': f'test{random.randint(1,9999)}@example.com',
            'street_address': '123 Test Street',
            'zip_code': '10001',
            'city': 'New York',
            'state': 'NY',
            'country': 'US',
            'credit_card_number': '4432-8015-6152-0454',
            'credit_card_expiration_month': str(random.randint(1, 12)),
            'credit_card_expiration_year': str(random.randint(current_year, current_year + 5)),
            'credit_card_cvv': str(random.randint(100, 999)),
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Event listeners per tracking e reporting
# ═══════════════════════════════════════════════════════════════════════════════

def track_response(response_time, status_code):
    """Track response time in current window"""
    window = int(time.time()) // WINDOW_SIZE
    window_stats[window]["requests"] += 1
    window_stats[window]["response_times"].append(response_time)
    if status_code >= 400:
        window_stats[window]["failures"] += 1


last_reported_window = 0
capacity_found = {}


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "=" * 80)
    print("  FRONTEND CAPACITY TEST — Single Replica Calibration")
    print("  Task mix: browse(10) + homepage(5) + cart(3) + currency(2)")
    print("            + add_to_cart(2) + checkout(1) = 23 total")
    print("  wait_time: between(1, 5) — identico a locustfile_scenarios.py")
    print("=" * 80)
    print(f"SLO thresholds: {SLO_THRESHOLDS} ms (p95)")
    print(f"Window size: {WINDOW_SIZE}s")
    print("-" * 80)
    print(f"{'Time':>8} | {'Users':>5} | {'RPS':>6} | {'p50':>6} | {'p95':>6} | {'p99':>6} | {'Fail%':>6} | SLO Status")
    print("-" * 80)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    global last_reported_window

    # Track this request
    status_code = 500 if exception else 200
    track_response(response_time, status_code)

    current_window = int(time.time()) // WINDOW_SIZE

    if current_window > last_reported_window and last_reported_window in window_stats:
        stats = window_stats[last_reported_window]
        rts = sorted(stats["response_times"])

        if len(rts) >= 5:
            n = len(rts)
            p50 = rts[int(n * 0.50)]
            p95 = rts[int(n * 0.95)]
            p99 = rts[int(n * 0.99)] if n > 10 else rts[-1]
            rps = stats["requests"] / WINDOW_SIZE
            fail_pct = (stats["failures"] / stats["requests"] * 100) if stats["requests"] > 0 else 0

            # Check SLO breaches
            slo_status = []
            for slo in SLO_THRESHOLDS:
                if p95 <= slo:
                    slo_status.append(f"  OK<{slo}")
                else:
                    slo_status.append(f"  XX>{slo}")
                    if slo not in capacity_found:
                        capacity_found[slo] = rps
                        print(f"\n  >>> SLO BREACH: p95={p95:.0f}ms > {slo}ms at {rps:.1f} rps\n")

            # Get user count
            try:
                users = on_request._runner.user_count if on_request._runner else '?'
            except:
                users = '?'

            ts = time.strftime("%H:%M:%S", time.localtime(last_reported_window * WINDOW_SIZE))
            print(f"{ts:>8} | {str(users):>5} | {rps:6.1f} | {p50:6.0f} | {p95:6.0f} | {p99:6.0f} | {fail_pct:5.1f}% | {' '.join(slo_status)}")

        last_reported_window = current_window
    elif current_window > last_reported_window:
        last_reported_window = current_window


@events.init.add_listener
def on_init(environment, **kwargs):
    on_request._runner = environment.runner


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\n" + "=" * 80)
    print("  CAPACITY TEST RESULTS — Single Replica")
    print("=" * 80)

    if capacity_found:
        print("\nCapacity per replica at different SLO thresholds:")
        for slo_ms in sorted(capacity_found.keys()):
            rps = capacity_found[slo_ms]
            print(f"  p95 < {slo_ms}ms : ~{rps:.0f} req/s per replica")

        print("\nRecommended capacity_req_per_sec for config/services.yaml:")
        for slo_ms in sorted(capacity_found.keys()):
            rps = capacity_found[slo_ms]
            recommended = int(rps * 0.8)  # 80% safety factor
            print(f"  SLO p95<{slo_ms}ms  -->  capacity_req_per_sec: {recommended}")
    else:
        print("\n  No SLO breaches detected! Frontend handled all load.")
        print("  Try running with more --users to find the limit.")

    # Detailed window analysis
    print("\n\nDetailed window analysis:")
    print(f"{'Window':>8} | {'RPS':>6} | {'p50':>6} | {'p95':>6} | {'Reqs':>6} | {'Fail':>5}")
    print("-" * 55)
    for w in sorted(window_stats.keys()):
        s = window_stats[w]
        rts = sorted(s["response_times"])
        if len(rts) >= 5:
            p50 = rts[int(len(rts) * 0.50)]
            p95 = rts[int(len(rts) * 0.95)]
            rps = s["requests"] / WINDOW_SIZE
            fail = s["failures"]
            print(f"{w:>8} | {rps:6.1f} | {p50:6.0f} | {p95:6.0f} | {s['requests']:>6} | {fail:>5}")

    print("=" * 80)