"""
DMOS Multi-Ingress Scenario Load Test — v3
============================================
Uses TaskSet + 3 separate User classes for proper connection pooling.

Usage (PowerShell):
  $env:DMOS_SCENARIO="flash_crowd"
  locust -f locustfile_multiingress.py
"""

import netrc  # pre-import per evitare AssertionError gevent al primo uso di requests
import os
import csv
import math
import time
import random
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import requests as _requests
from gevent.threadpool import ThreadPoolExecutor as _GeventThreadPool
from locust import HttpUser, task, between, LoadTestShape, events

# Thread pool: HTTP in thread OS reali → r.elapsed accurato su Windows
# (immune al gevent select() polling delay che azzerava le latenze)
_request_pool = _GeventThreadPool(max_workers=800)
_thread_local = threading.local()


def _get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _requests.Session()
    return _thread_local.session


def _blocking_get(url):
    resp = _get_session().get(url, timeout=10, allow_redirects=True)
    return resp.elapsed.total_seconds() * 1000, resp.status_code


def _blocking_post(url, data):
    resp = _get_session().post(url, data=data, timeout=10, allow_redirects=True)
    return resp.elapsed.total_seconds() * 1000, resp.status_code

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
_cluster_stats = defaultdict(lambda: {"count": 0, "failures": 0, "slo_violations": 0, "response_times": []})
_WINDOW_SIZE = 10
_windowed_stats = defaultdict(lambda: defaultdict(lambda: {"count": 0, "slo_violations": 0, "response_times": []}))

# SLO threshold: 1s (allineato con Romano p95 ~1.0–1.5s "buono", Cilantro latency SLO 2s)
SLO_THRESHOLD_MS = 1000
CLUSTER_NAMES = ["cluster1", "cluster2", "cluster3"]
CLUSTER_REGIONS = {"cluster1": "DE", "cluster2": "FR", "cluster3": "PL"}

# Safety: ferma il test se il p95 globale supera questa soglia per N check consecutivi
SAFETY_P95_LIMIT_MS = float(os.environ.get("SAFETY_P95_MS", "8000"))  # default 8s
SAFETY_CONSECUTIVE  = int(os.environ.get("SAFETY_CONSECUTIVE", "3"))  # 3 × 30s = 90s
SAFETY_CHECK_SEC    = 30  # intervallo check (secondi)


def track_request(cluster_name, response_time_ms, is_failure=False):
    window = int(time.time()) // _WINDOW_SIZE
    with _stats_lock:
        _cluster_stats[cluster_name]["count"] += 1
        _cluster_stats[cluster_name]["response_times"].append(response_time_ms)
        if is_failure:
            _cluster_stats[cluster_name]["failures"] += 1
        if response_time_ms > SLO_THRESHOLD_MS:
            _cluster_stats[cluster_name]["slo_violations"] += 1
        _windowed_stats[window][cluster_name]["count"] += 1
        _windowed_stats[window][cluster_name]["response_times"].append(response_time_ms)
        if response_time_ms > SLO_THRESHOLD_MS:
            _windowed_stats[window][cluster_name]["slo_violations"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "gradual_ramp": [
        # Picco 300 utenti calibrato per netem (capacity_per_sec=5):
        #   c1(DE) → ~12 repliche, c2(FR) → ~7, c3(PL) → ~4
        # Durata totale: 120+600+300+300+120 = 1440s = 24min
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 600, "users_start": 10,  "users_end": 300, "spawn_rate": 5,  "label": "ramp-up"},
        {"duration": 300, "users_start": 300, "users_end": 300, "spawn_rate": 10, "label": "peak"},
        {"duration": 300, "users_start": 300, "users_end": 10,  "spawn_rate": 5,  "label": "ramp-down"},
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "cooldown"},
    ],
    "flash_crowd": [
        # Spike 320 utenti in 60s → test reattività DMOS
        # Durata totale: 180+60+300+180+180+120 = 1020s ≈ 17min
        {"duration": 180, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 60,  "users_start": 10,  "users_end": 320, "spawn_rate": 60, "label": "flash-spike"},
        {"duration": 300, "users_start": 320, "users_end": 320, "spawn_rate": 10, "label": "sustained-peak"},
        {"duration": 180, "users_start": 320, "users_end": 160, "spawn_rate": 8,  "label": "partial-decline"},
        {"duration": 180, "users_start": 160, "users_end": 10,  "spawn_rate": 8,  "label": "full-decline"},
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "cooldown"},
    ],
    "double_wave": [
        # Due ondate: wave1 picco 200, valley 50, wave2 picco 250
        # Durata totale: 120+180+240+180+120+180+240+180+120 = 1560s = 26min
        {"duration": 120, "users_start": 20, "users_end": 20,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 180, "users_start": 20, "users_end": 200, "spawn_rate": 8,  "label": "wave1-ramp"},
        {"duration": 240, "users_start": 200, "users_end": 200, "spawn_rate": 10, "label": "wave1-peak"},
        {"duration": 180, "users_start": 200, "users_end": 50,  "spawn_rate": 5,  "label": "valley-descent"},
        {"duration": 120, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "valley"},
        {"duration": 180, "users_start": 50,  "users_end": 250, "spawn_rate": 8,  "label": "wave2-ramp"},
        {"duration": 240, "users_start": 250, "users_end": 250, "spawn_rate": 10, "label": "wave2-peak"},
        {"duration": 180, "users_start": 250, "users_end": 20,  "spawn_rate": 5,  "label": "final-decline"},
        {"duration": 120, "users_start": 20,  "users_end": 20,  "spawn_rate": 5,  "label": "cooldown"},
    ],
    "sinusoidal": [
        # Sinusoide: min 40, max 200, periodo 360s
        # Durata totale: 120+1800+120 = 2040s = 34min
        {"duration": 120, "users_start": 40, "users_end": 40, "spawn_rate": 5, "label": "warm-up"},
        {"duration": 1800, "users_start": 40, "users_end": 200, "spawn_rate": 10, "label": "sinusoidal",
         "type": "sinusoidal", "period": 360, "min_users": 40, "max_users": 200},
        {"duration": 120, "users_start": 40, "users_end": 40, "spawn_rate": 5, "label": "cooldown"},
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

_CHECKOUT_DATA = {
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
}

# (path, weight, method, data)
_ENDPOINTS = [
    ("/",                       0.40, "GET",  None),
    ("/product/OLJCESPC7Z",     0.15, "GET",  None),
    ("/product/66VCHSJNUP",     0.10, "GET",  None),
    ("/cart",                   0.10, "GET",  None),
    ("/cart",                   0.15, "POST", {"product_id": "OLJCESPC7Z", "quantity": 1}),
    ("/setCurrency",            0.05, "POST", {"currency_code": "EUR"}),
    ("/cart/checkout",          0.05, "POST", _CHECKOUT_DATA),
]


def _weighted_choice():
    r = random.random()
    cumulative = 0.0
    for path, weight, method, data in _ENDPOINTS:
        cumulative += weight
        if r <= cumulative:
            return path, method, data
    path, _, method, data = _ENDPOINTS[-1]
    return path, method, data


def _do_request(cluster_name, base_url):
    """Richiesta in thread OS reale — r.elapsed accurato su Windows."""
    path, method, data = _weighted_choice()
    url = base_url + path
    try:
        if method == "POST":
            elapsed_ms, status = _request_pool.submit(_blocking_post, url, data).result()
        else:
            elapsed_ms, status = _request_pool.submit(_blocking_get, url).result()
    except Exception:
        elapsed_ms, status = 10000.0, 500
    track_request(cluster_name, elapsed_ms, status >= 500)


# ═══════════════════════════════════════════════════════════════════════════════
# Three User classes — one per cluster
# ═══════════════════════════════════════════════════════════════════════════════

class Cluster1User(HttpUser):
    host = "http://192.168.1.245:30080"
    weight = WEIGHTS[0]
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request("cluster1", "http://192.168.1.245:30080")


class Cluster2User(HttpUser):
    host = "http://192.168.1.246:30080"
    weight = WEIGHTS[1]
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request("cluster2", "http://192.168.1.246:30080")


class Cluster3User(HttpUser):
    host = "http://192.168.1.247:30080"
    weight = WEIGHTS[2]
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request("cluster3", "http://192.168.1.247:30080")


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def _safety_watchdog(environment):
    """Ferma il test se il p95 globale supera SAFETY_P95_LIMIT_MS per SAFETY_CONSECUTIVE check."""
    consecutive = 0
    while not environment.runner.state == "stopped":
        time.sleep(SAFETY_CHECK_SEC)
        with _stats_lock:
            all_rts = []
            for s in _cluster_stats.values():
                all_rts.extend(s["response_times"])
        if len(all_rts) < 50:
            continue  # troppo pochi dati, salta
        p95 = sorted(all_rts)[int(len(all_rts) * 0.95)]
        if p95 > SAFETY_P95_LIMIT_MS:
            consecutive += 1
            print(f"\n  ⚠️  SAFETY: p95 globale = {p95:.0f}ms > {SAFETY_P95_LIMIT_MS:.0f}ms "
                  f"({consecutive}/{SAFETY_CONSECUTIVE})")
            if consecutive >= SAFETY_CONSECUTIVE:
                print(f"\n  🛑 SAFETY STOP: p95 troppo alto per {SAFETY_CONSECUTIVE} check consecutivi "
                      f"— test fermato per proteggere le VM\n")
                environment.runner.quit()
                return
        else:
            if consecutive > 0:
                print(f"  ✅ SAFETY: p95 tornato a {p95:.0f}ms — reset contatore")
            consecutive = 0


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"\n  Test started: {SCENARIO} | Per-cluster tracking: ACTIVE\n")
    print(f"  Safety watchdog: p95 limit={SAFETY_P95_LIMIT_MS:.0f}ms, "
          f"stop after {SAFETY_CONSECUTIVE} consecutive checks ({SAFETY_CONSECUTIVE*SAFETY_CHECK_SEC}s)\n")
    t = threading.Thread(target=_safety_watchdog, args=(environment,), daemon=True)
    t.start()


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    output_dir = Path(__file__).parent.parent / "results" / "multiingress"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*78}")
    print(f"  PER-CLUSTER LATENCY COMPARISON — {SCENARIO}")
    print(f"{'='*78}")
    print(f"\n  {'Cluster':<12} {'Region':<6} {'Requests':>10} {'Fail%':>8} "
          f"{'Avg':>8} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {f'SLO>{SLO_THRESHOLD_MS}ms':>12}")
    print(f"  {'-'*86}")

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
        slo_pct = stats["slo_violations"] / stats["count"] * 100 if stats["count"] > 0 else 0
        print(f"  {cname:<12} {region:<6} {stats['count']:>10} "
              f"{fail_pct:>7.1f}% {avg:>7.0f}ms {p50:>7.0f}ms "
              f"{p90:>7.0f}ms {p95:>7.0f}ms {p99:>7.0f}ms "
              f"SLO>{SLO_THRESHOLD_MS}ms:{slo_pct:>5.1f}%")
        summary_rows.append({
            "cluster": cname, "region": region,
            "requests": stats["count"], "failures": stats["failures"],
            "fail_pct": round(fail_pct, 2),
            "avg_ms": round(avg, 1), "p50_ms": round(p50, 1),
            "p90_ms": round(p90, 1), "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
            "slo_pct": round(slo_pct, 2),   # % richieste > SLO_THRESHOLD_MS
        })

    all_rts = []
    total_reqs = total_fails = total_slo_viol = 0
    for s in _cluster_stats.values():
        all_rts.extend(s["response_times"])
        total_reqs += s["count"]
        total_fails += s["failures"]
        total_slo_viol += s["slo_violations"]
    if all_rts:
        all_rts.sort()
        n = len(all_rts)
        global_slo_pct = total_slo_viol / total_reqs * 100 if total_reqs > 0 else 0
        print(f"  {'-'*86}")
        print(f"  {'GLOBAL':<12} {'ALL':<6} {total_reqs:>10} "
              f"{total_fails/total_reqs*100:>7.1f}% "
              f"{sum(all_rts)/n:>7.0f}ms {all_rts[int(n*0.50)]:>7.0f}ms "
              f"{all_rts[int(n*0.90)]:>7.0f}ms {all_rts[int(n*0.95)]:>7.0f}ms "
              f"{all_rts[min(int(n*0.99),n-1)]:>7.0f}ms {global_slo_pct:>11.1f}%")
    print(f"\n{'='*86}\n")

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
            # count, avg_ms, p95_ms, slo_pct (% req > SLO_THRESHOLD_MS in this window)
            header.extend([f"{cn}_count", f"{cn}_avg_ms", f"{cn}_p95_ms", f"{cn}_slo_pct"])
        writer.writerow(header)
        for window_ts in sorted(_windowed_stats.keys()):
            row = [datetime.fromtimestamp(window_ts * _WINDOW_SIZE).strftime("%H:%M:%S")]
            for cn in CLUSTER_NAMES:
                ws = _windowed_stats[window_ts].get(cn)
                if ws and ws["response_times"]:
                    rts = sorted(ws["response_times"])
                    nw = len(rts)
                    p95_w = round(rts[int(nw * 0.95)], 1) if nw >= 2 else round(rts[-1], 1)
                    slo_pct_w = round(ws["slo_violations"] / nw * 100, 1)
                    row.extend([nw, round(sum(rts) / nw, 1), p95_w, slo_pct_w])
                else:
                    row.extend([0, 0, 0, 0])
            writer.writerow(row)
    print(f"  Time-series CSV: {ts_file}")
    print(f"\n  Done! Use plot_cluster_latency.py to generate graphs.\n")