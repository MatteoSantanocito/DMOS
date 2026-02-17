"""
Frontend Capacity Test for DMOS Calibration
============================================
Scales ONE replica of frontend and gradually increases load
to find the true capacity_req_per_sec at different SLO thresholds.

USAGE:
  1. Scale frontend to exactly 1 replica on ONE cluster:
     kubectl --context cluster1 scale deploy/frontend -n online-boutique --replicas=1
     kubectl --context cluster2 scale deploy/frontend -n online-boutique --replicas=0
     kubectl --context cluster3 scale deploy/frontend -n online-boutique --replicas=0

  2. STOP DMOS temporarily (otherwise it will re-scale):
     Ctrl+C on DMOS, or scale dmos-scheduler to 0

  3. Run this test:
     locust -f capacity_test_locustfile.py --host http://192.168.1.245:30007 \
            --users 200 --spawn-rate 5 --run-time 10m --html capacity_report.html

  4. Analyze results to find where p95 crosses your SLO threshold.
     The script prints capacity breakpoints in real-time.

  5. After test, restart DMOS and let it manage replicas again.
"""

import time
from locust import HttpUser, task, between, events
from collections import defaultdict

# Track metrics per time window
window_stats = defaultdict(lambda: {"requests": 0, "failures": 0, "response_times": []})
WINDOW_SIZE = 10  # seconds
SLO_THRESHOLDS = [200, 300, 400, 500]  # ms - p95 targets to check

class FrontendUser(HttpUser):
    wait_time = between(0.5, 1.5)
    
    @task(10)
    def browse_home(self):
        with self.client.get("/", catch_response=True) as resp:
            track_response(resp)
    
    @task(5)
    def browse_product(self):
        products = [
            "OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O",
            "L9ECAV7KIM", "2ZYFJ3GM2N", "0PUK6V6EV0",
            "LS4PSXUNUM", "9SIQT8TOJO", "6E92ZMYYFZ"
        ]
        import random
        pid = random.choice(products)
        with self.client.get(f"/product/{pid}", catch_response=True) as resp:
            track_response(resp)
    
    @task(3)
    def add_to_cart(self):
        with self.client.post("/cart", json={
            "product_id": "OLJCESPC7Z",
            "quantity": 1
        }, catch_response=True) as resp:
            track_response(resp)
    
    @task(1)
    def view_cart(self):
        with self.client.get("/cart", catch_response=True) as resp:
            track_response(resp)


def track_response(resp):
    """Track response time in current window"""
    window = int(time.time()) // WINDOW_SIZE
    window_stats[window]["requests"] += 1
    window_stats[window]["response_times"].append(resp.elapsed.total_seconds() * 1000)
    if resp.status_code >= 400:
        window_stats[window]["failures"] += 1


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "=" * 80)
    print("🔬 FRONTEND CAPACITY TEST — Single Replica Calibration")
    print("=" * 80)
    print(f"SLO thresholds being tested: {SLO_THRESHOLDS} ms (p95)")
    print(f"Window size: {WINDOW_SIZE}s")
    print("-" * 80)
    print(f"{'Time':>8} | {'Users':>5} | {'RPS':>6} | {'p50':>6} | {'p95':>6} | {'p99':>6} | {'Fail%':>6} | SLO Status")
    print("-" * 80)


# Periodic reporter
last_reported_window = 0
capacity_found = {}  # {slo_ms: rps_at_breach}

@events.request.add_listener  
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    global last_reported_window
    
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
                    slo_status.append(f"✅{slo}")
                else:
                    slo_status.append(f"❌{slo}")
                    if slo not in capacity_found:
                        capacity_found[slo] = rps
                        print(f"\n  ⚠️  SLO BREACH: p95={p95:.0f}ms > {slo}ms at {rps:.1f} rps\n")
            
            # Get user count from environment
            try:
                from locust.runners import MasterRunner, LocalRunner
                runner = kwargs.get('runner') or on_request._runner
                users = runner.user_count if runner else '?'
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
    print("📊 CAPACITY TEST RESULTS — Single Replica")
    print("=" * 80)
    
    if capacity_found:
        print("\nCapacity per replica at different SLO thresholds:")
        for slo_ms in sorted(capacity_found.keys()):
            rps = capacity_found[slo_ms]
            print(f"  p95 < {slo_ms}ms : ~{rps:.0f} req/s per replica")
        
        print("\n📋 Recommended capacity_req_per_sec values for services.yaml:")
        for slo_ms in sorted(capacity_found.keys()):
            rps = capacity_found[slo_ms]
            # Apply 80% safety factor for sustained load
            recommended = int(rps * 0.8)
            print(f"  SLO p95<{slo_ms}ms → capacity_req_per_sec: {recommended}")
    else:
        print("\n  No SLO breaches detected! Frontend can handle all tested load.")
        print("  Consider running with more users to find the limit.")
    
    # Also compute from all windows
    print("\n\nDetailed window analysis:")
    print(f"{'Window':>8} | {'RPS':>6} | {'p95':>6} | {'Reqs':>6}")
    for w in sorted(window_stats.keys()):
        s = window_stats[w]
        rts = sorted(s["response_times"])
        if len(rts) >= 5:
            p95 = rts[int(len(rts) * 0.95)]
            rps = s["requests"] / WINDOW_SIZE
            print(f"{w:>8} | {rps:6.1f} | {p95:6.0f} | {s['requests']:>6}")
    
    print("=" * 80)