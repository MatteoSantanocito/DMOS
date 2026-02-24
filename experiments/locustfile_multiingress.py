"""
DMOS Multi-Ingress Scenario Load Test — v3
============================================
Uses TaskSet + 3 separate User classes for proper connection pooling.

Usage (PowerShell):
  $env:DMOS_SCENARIO="flash_crowd"
  locust -f locustfile_multiingress.py
"""

import os
import csv
import math
import time
import random
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from locust import HttpUser, TaskSet, task, between, LoadTestShape, events

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO = os.environ.get("DMOS_SCENARIO", "flash_crowd").lower()
_raw_weights = os.environ.get("DMOS_WEIGHTS", "40,35,25")
WEIGHTS = [int(w) for w in _raw_weights.split(",")]

PRODUCTS = [
    "OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O", "L9ECAV7KIM",
    "2ZYFJ3GM2N", "0PUK6V6EV0", "LS4PSXUNUM", "9SIQT8TOJO", "6E92ZMYYFZ",
]
CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CAD"]

# ═══════════════════════════════════════════════════════════════════════════════
# Per-Cluster Latency Tracking
# ═══════════════════════════════════════════════════════════════════════════════

_stats_lock = threading.Lock()
_cluster_stats = defaultdict(lambda: {"count": 0, "failures": 0, "response_times": []})
_WINDOW_SIZE = 10
_windowed_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "response_times": []}))
CLUSTER_NAMES = ["cluster1", "cluster2", "cluster3"]
CLUSTER_REGIONS = {"cluster1": "DE", "cluster2": "FR", "cluster3": "PL"}


def track_request(cluster_name, response_time_ms, is_failure=False):
    window = int(time.time()) // _WINDOW_SIZE
    with _stats_lock:
        _cluster_stats[cluster_name]["count"] += 1
        _cluster_stats[cluster_name]["response_times"].append(response_time_ms)
        if is_failure:
            _cluster_stats[cluster_name]["failures"] += 1
        _windowed_stats[window][cluster_name]["count"] += 1
        _windowed_stats[window][cluster_name]["response_times"].append(response_time_ms)


# ═══════════════════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "flash_crowd": [
        {"duration": 180, "users_start": 50, "users_end": 50, "spawn_rate": 10, "label": "warm-up"},
        {"duration": 240, "users_start": 50, "users_end": 350, "spawn_rate": 50, "label": "flash-spike"},
        {"duration": 300, "users_start": 350, "users_end": 350, "spawn_rate": 10, "label": "sustained-peak"},
        {"duration": 180, "users_start": 350, "users_end": 200, "spawn_rate": 5, "label": "partial-decline"},
        {"duration": 180, "users_start": 200, "users_end": 50, "spawn_rate": 5, "label": "full-decline"},
        {"duration": 120, "users_start": 50, "users_end": 50, "spawn_rate": 10, "label": "cooldown"},
    ],
    "double_wave": [
        {"duration": 120, "users_start": 50, "users_end": 50, "spawn_rate": 10, "label": "warm-up"},
        {"duration": 180, "users_start": 50, "users_end": 300, "spawn_rate": 10, "label": "wave1-ramp"},
        {"duration": 240, "users_start": 300, "users_end": 300, "spawn_rate": 10, "label": "wave1-peak"},
        {"duration": 180, "users_start": 300, "users_end": 100, "spawn_rate": 5, "label": "valley-descent"},
        {"duration": 120, "users_start": 100, "users_end": 100, "spawn_rate": 10, "label": "valley"},
        {"duration": 180, "users_start": 100, "users_end": 350, "spawn_rate": 10, "label": "wave2-ramp"},
        {"duration": 240, "users_start": 350, "users_end": 350, "spawn_rate": 10, "label": "wave2-peak"},
        {"duration": 180, "users_start": 350, "users_end": 50, "spawn_rate": 5, "label": "final-decline"},
        {"duration": 120, "users_start": 50, "users_end": 50, "spawn_rate": 10, "label": "cooldown"},
    ],
    "sinusoidal": [
        {"duration": 120, "users_start": 50, "users_end": 50, "spawn_rate": 10, "label": "warm-up"},
        {"duration": 1800, "users_start": 80, "users_end": 300, "spawn_rate": 15, "label": "sinusoidal",
         "type": "sinusoidal", "period": 360, "min_users": 80, "max_users": 300},
        {"duration": 120, "users_start": 50, "users_end": 50, "spawn_rate": 10, "label": "cooldown"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Traffic Shape
# ═══════════════════════════════════════════════════════════════════════════════

class DMOSScenarioShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        self.phases = SCENARIOS.get(SCENARIO, SCENARIOS["flash_crowd"])
        self._phase_offsets = []
        offset = 0
        for p in self.phases:
            self._phase_offsets.append(offset)
            offset += p["duration"]
        self._total_duration = offset
        weight_strs = [f"cluster{i+1}={WEIGHTS[i]}" for i in range(3)]
        print(f"\n{'='*70}")
        print(f"  DMOS Multi-Ingress Load Test v3")
        print(f"  Scenario: {SCENARIO} | Duration: {self._total_duration/60:.0f} min")
        print(f"  Weights:  {', '.join(weight_strs)}")
        print(f"{'='*70}\n")

    def tick(self):
        run_time = self.get_run_time()
        if run_time >= self._total_duration:
            return None
        for i, phase in enumerate(self.phases):
            phase_start = self._phase_offsets[i]
            phase_end = phase_start + phase["duration"]
            if run_time < phase_end:
                t_in_phase = run_time - phase_start
                if t_in_phase < 1:
                    print(f"\n  Phase: {phase['label']} | "
                          f"Users: {phase['users_start']} -> {phase['users_end']} | "
                          f"Duration: {phase['duration']}s")
                if phase.get("type") == "sinusoidal":
                    period = phase["period"]
                    amp = (phase["max_users"] - phase["min_users"]) / 2
                    mid = (phase["max_users"] + phase["min_users"]) / 2
                    return (int(mid + amp * math.sin(2 * math.pi * t_in_phase / period)),
                            phase["spawn_rate"])
                progress = t_in_phase / phase["duration"]
                current = int(phase["users_start"] +
                              (phase["users_end"] - phase["users_start"]) * progress)
                return (current, phase["spawn_rate"])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TaskSet with browsing behavior
# ═══════════════════════════════════════════════════════════════════════════════

class BrowsingTasks(TaskSet):
    """Online Boutique browsing behavior. Tracks latency per cluster."""

    def _get_cluster_name(self):
        return getattr(self.user, "cluster_name", "unknown")

    def _track(self, resp):
        rt_ms = resp.elapsed.total_seconds() * 1000
        track_request(self._get_cluster_name(), rt_ms, resp.status_code >= 400)

    @task(40)
    def browse_homepage(self):
        with self.client.get("/", catch_response=True) as resp:
            self._track(resp)
            if resp.status_code != 200:
                resp.failure(f"{self._get_cluster_name()}: {resp.status_code}")

    @task(25)
    def view_product(self):
        pid = random.choice(PRODUCTS)
        with self.client.get(f"/product/{pid}", catch_response=True,
                             name="GET /product/[id]") as resp:
            self._track(resp)
            if resp.status_code != 200:
                resp.failure(f"{self._get_cluster_name()}: {resp.status_code}")

    @task(15)
    def add_to_cart(self):
        pid = random.choice(PRODUCTS)
        with self.client.post("/cart",
                              data={"product_id": pid, "quantity": random.randint(1, 3)},
                              catch_response=True, name="POST /cart") as resp:
            self._track(resp)
            if resp.status_code not in (200, 302):
                resp.failure(f"{self._get_cluster_name()}: {resp.status_code}")

    @task(10)
    def view_cart(self):
        with self.client.get("/cart", catch_response=True) as resp:
            self._track(resp)
            if resp.status_code != 200:
                resp.failure(f"{self._get_cluster_name()}: {resp.status_code}")

    @task(5)
    def set_currency(self):
        currency = random.choice(CURRENCIES)
        with self.client.post("/setCurrency",
                              data={"currency_code": currency},
                              catch_response=True, name="POST /setCurrency") as resp:
            self._track(resp)
            if resp.status_code not in (200, 302):
                resp.failure(f"{self._get_cluster_name()}: {resp.status_code}")

    @task(5)
    def checkout(self):
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
            self._track(resp)
            if resp.status_code not in (200, 302):
                resp.failure(f"{self._get_cluster_name()}: {resp.status_code}")


# ═══════════════════════════════════════════════════════════════════════════════
# Three User classes — one per cluster
# ═══════════════════════════════════════════════════════════════════════════════

class Cluster1User(HttpUser):
    host = "http://192.168.1.245:30007"
    cluster_name = "cluster1"
    weight = WEIGHTS[0]
    wait_time = between(1, 3)
    tasks = [BrowsingTasks]


class Cluster2User(HttpUser):
    host = "http://192.168.1.246:30007"
    cluster_name = "cluster2"
    weight = WEIGHTS[1]
    wait_time = between(1, 3)
    tasks = [BrowsingTasks]


class Cluster3User(HttpUser):
    host = "http://192.168.1.247:30007"
    cluster_name = "cluster3"
    weight = WEIGHTS[2]
    wait_time = between(1, 3)
    tasks = [BrowsingTasks]


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"\n  Test started: {SCENARIO} | Per-cluster tracking: ACTIVE\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    output_dir = Path("results/multiingress")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*78}")
    print(f"  PER-CLUSTER LATENCY COMPARISON — {SCENARIO}")
    print(f"{'='*78}")
    print(f"\n  {'Cluster':<12} {'Region':<6} {'Requests':>10} {'Fail%':>8} "
          f"{'Avg':>8} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8}")
    print(f"  {'-'*72}")

    summary_rows = []
    for cname in CLUSTER_NAMES:
        region = CLUSTER_REGIONS[cname]
        stats = _cluster_stats.get(cname)
        if not stats or not stats["response_times"]:
            print(f"  {cname:<12} {region:<6} {'No data':>10}")
            continue
        rts = sorted(stats["response_times"])
        n = len(rts)
        avg = sum(rts) / n
        p50 = rts[int(n * 0.50)]
        p90 = rts[int(n * 0.90)]
        p95 = rts[int(n * 0.95)]
        p99 = rts[min(int(n * 0.99), n - 1)]
        fail_pct = stats["failures"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"  {cname:<12} {region:<6} {stats['count']:>10} "
              f"{fail_pct:>7.1f}% {avg:>7.0f}ms {p50:>7.0f}ms "
              f"{p90:>7.0f}ms {p95:>7.0f}ms {p99:>7.0f}ms")
        summary_rows.append({
            "cluster": cname, "region": region,
            "requests": stats["count"], "failures": stats["failures"],
            "fail_pct": round(fail_pct, 2),
            "avg_ms": round(avg, 1), "p50_ms": round(p50, 1),
            "p90_ms": round(p90, 1), "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
        })

    all_rts = []
    total_reqs = total_fails = 0
    for s in _cluster_stats.values():
        all_rts.extend(s["response_times"])
        total_reqs += s["count"]
        total_fails += s["failures"]
    if all_rts:
        all_rts.sort()
        n = len(all_rts)
        print(f"  {'-'*72}")
        print(f"  {'GLOBAL':<12} {'ALL':<6} {total_reqs:>10} "
              f"{total_fails/total_reqs*100:>7.1f}% "
              f"{sum(all_rts)/n:>7.0f}ms {all_rts[int(n*0.50)]:>7.0f}ms "
              f"{all_rts[int(n*0.90)]:>7.0f}ms {all_rts[int(n*0.95)]:>7.0f}ms "
              f"{all_rts[min(int(n*0.99),n-1)]:>7.0f}ms")
    print(f"\n{'='*78}\n")

    summary_file = output_dir / f"{SCENARIO}_cluster_latency_{timestamp}.csv"
    if summary_rows:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Summary CSV:    {summary_file}")

    ts_file = output_dir / f"{SCENARIO}_timeseries_{timestamp}.csv"
    with open(ts_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp"]
        for cn in CLUSTER_NAMES:
            header.extend([f"{cn}_count", f"{cn}_avg_ms", f"{cn}_p95_ms"])
        writer.writerow(header)
        for window_ts in sorted(_windowed_stats.keys()):
            row = [datetime.fromtimestamp(window_ts * _WINDOW_SIZE).strftime("%H:%M:%S")]
            for cn in CLUSTER_NAMES:
                ws = _windowed_stats[window_ts].get(cn)
                if ws and ws["response_times"]:
                    rts = sorted(ws["response_times"])
                    nw = len(rts)
                    row.extend([nw, round(sum(rts)/nw, 1),
                                round(rts[int(nw*0.95)], 1) if nw >= 2 else round(rts[-1], 1)])
                else:
                    row.extend([0, 0, 0])
            writer.writerow(row)
    print(f"  Time-series CSV: {ts_file}")
    print(f"\n  Done! Use plot_cluster_latency.py to generate graphs.\n")