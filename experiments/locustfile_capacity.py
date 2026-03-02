"""
Capacity Test — Frontend per replica
=====================================
Scopo: misurare la capacità massima sostenibile del frontend (rps/replica)
       a latenza p95 < soglia SLA.

Come funziona:
  1. Tiene 1 replica per cluster (NESSUN autoscaling DMOS durante il test)
  2. Sale a gradini (steps): ogni step aggiunge 20 utenti per 90 secondi
  3. Misura p95, throughput e fail rate a fine di ogni step
  4. Si ferma quando p95 supera SLA_P95_MS o fail rate > MAX_FAIL_RATE
  5. Riporta il knee point = rps/replica al passo precedente

Setup richiesto:
  # 1. Scala frontend a 1 replica per cluster (disabilita DMOS)
  kubectl scale deployment frontend -n online-boutique --replicas=1 --context cluster1
  kubectl scale deployment frontend -n online-boutique --replicas=1 --context cluster2
  kubectl scale deployment frontend -n online-boutique --replicas=1 --context cluster3

  # 2. Avvia Locust in modalità web
  locust -f experiments/locustfile_capacity.py --host http://ignored

  # 3. Oppure headless (raccomandato per capacity test):
  locust -f experiments/locustfile_capacity.py --host http://ignored --headless

  # 4. Dopo il test, ripristina il min autoscaling
  kubectl scale deployment frontend -n online-boutique --replicas=1 --context cluster1
  (DMOS riprende da sè)

Output:
  results/capacity/capacity_YYYYMMDD_HHMMSS.csv   (per-step metrics)
  results/capacity/capacity_YYYYMMDD_HHMMSS.json  (summary con knee point)
  Console: knee point e valore raccomandato per services.yaml

Parametri configurabili in CAPACITY_CONFIG sotto.
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner

import time
import csv
import json
import math
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict


# ─── Configurazione ──────────────────────────────────────────────────────────

CAPACITY_CONFIG = {
    # Target cluster IPs + porta Nginx Ingress NodePort
    # NOTA: 30080 è la porta NodePort di Nginx Ingress (non più 30007 frontend diretto).
    # Verifica con: kubectl get svc -n ingress-nginx ingress-nginx-controller
    "clusters": {
        "cluster1": "http://192.168.1.245:30080",
        "cluster2": "http://192.168.1.246:30080",
        "cluster3": "http://192.168.1.247:30080",
    },

    # Step configuration
    "users_start": 20,          # Utenti al primo step
    "users_step": 20,           # Utenti aggiunti ad ogni step
    "users_max": 300,           # Stop se si raggiunge questo valore
    "step_duration_s": 90,      # Durata di ogni step (secondi)
    "spawn_rate": 10,           # Utenti/secondo durante lo spawn

    # SLA thresholds — il test si ferma quando vengono violati
    "sla_p95_ms": 300,          # ms — più tollerante del SLA di produzione (100ms)
                                 # per trovare la capacità reale del sistema
    "sla_fail_rate": 0.02,      # 2% max fail

    # Output
    "output_dir": "results/capacity",
    "num_replicas_per_cluster": 1,  # Quante repliche sono attive durante il test
}

# ─── Variabili globali per raccolta metriche ─────────────────────────────────

_stats_lock = threading.Lock()
_step_stats = defaultdict(lambda: {
    "requests": 0,
    "failures": 0,
    "total_rt": 0.0,
    "response_times": [],
})
_current_step = 0
_test_results = []
_knee_point_rps = None
_knee_point_users = None

# ─── Endpoint mix per Online Boutique ────────────────────────────────────────

ENDPOINTS = [
    ("/", 0.35),
    ("/product/OLJCESPC7Z", 0.25),
    ("/product/66VCHSJNUP", 0.10),
    ("/cart", 0.15),
    ("/setCurrency", 0.05),
    ("/cart/checkout", 0.10),
]

def _weighted_choice(items):
    """Sceglie un endpoint pesato."""
    r = __import__("random").random()
    cumulative = 0
    for item, weight in items:
        cumulative += weight
        if r <= cumulative:
            return item
    return items[-1][0]


# ─── Locust User Classes ──────────────────────────────────────────────────────

class Cluster1User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster1"]
    weight = 40  # 40% del traffico → cluster1 (DE)
    wait_time = between(0.5, 1.5)

    @task
    def browse(self):
        endpoint = _weighted_choice(ENDPOINTS)
        with self.client.get(endpoint, catch_response=True, name=endpoint) as r:
            with _stats_lock:
                _step_stats[_current_step]["requests"] += 1
                _step_stats[_current_step]["total_rt"] += r.elapsed.total_seconds() * 1000
                _step_stats[_current_step]["response_times"].append(
                    r.elapsed.total_seconds() * 1000
                )
                if r.status_code >= 500:
                    _step_stats[_current_step]["failures"] += 1
                    r.failure(f"HTTP {r.status_code}")
                else:
                    r.success()


class Cluster2User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster2"]
    weight = 35  # 35% → cluster2 (FR)
    wait_time = between(0.5, 1.5)

    @task
    def browse(self):
        endpoint = _weighted_choice(ENDPOINTS)
        with self.client.get(endpoint, catch_response=True, name=endpoint) as r:
            with _stats_lock:
                _step_stats[_current_step]["requests"] += 1
                _step_stats[_current_step]["total_rt"] += r.elapsed.total_seconds() * 1000
                _step_stats[_current_step]["response_times"].append(
                    r.elapsed.total_seconds() * 1000
                )
                if r.status_code >= 500:
                    _step_stats[_current_step]["failures"] += 1
                    r.failure(f"HTTP {r.status_code}")
                else:
                    r.success()


class Cluster3User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster3"]
    weight = 25  # 25% → cluster3 (PL)
    wait_time = between(0.5, 1.5)

    @task
    def browse(self):
        endpoint = _weighted_choice(ENDPOINTS)
        with self.client.get(endpoint, catch_response=True, name=endpoint) as r:
            with _stats_lock:
                _step_stats[_current_step]["requests"] += 1
                _step_stats[_current_step]["total_rt"] += r.elapsed.total_seconds() * 1000
                _step_stats[_current_step]["response_times"].append(
                    r.elapsed.total_seconds() * 1000
                )
                if r.status_code >= 500:
                    _step_stats[_current_step]["failures"] += 1
                    r.failure(f"HTTP {r.status_code}")
                else:
                    r.success()


# ─── LoadTestShape per stepped load ──────────────────────────────────────────

from locust import LoadTestShape

class SteppedCapacityShape(LoadTestShape):
    """
    Stepped load: ogni step aggiunge utenti e aspetta step_duration_s secondi.
    Si ferma automaticamente quando SLA viene violato o si raggiunge users_max.
    """
    cfg = CAPACITY_CONFIG
    step_time = cfg["step_duration_s"]
    step_users = cfg["users_step"]
    spawn_rate = cfg["spawn_rate"]
    start_users = cfg["users_start"]
    max_users = cfg["users_max"]

    _last_step_end = 0
    _current_step_users = 0
    _stop_flag = False

    def tick(self):
        global _current_step, _test_results, _knee_point_rps, _knee_point_users

        if self._stop_flag:
            return None  # Stop test

        run_time = self.get_run_time()

        # Calcola lo step corrente
        step_num = int(run_time / self.step_time)
        target_users = self.start_users + step_num * self.step_users

        # Cambio di step: analizza il passo precedente
        if step_num != _current_step and _current_step > 0:
            self._analyze_step(_current_step)

        _current_step = step_num

        if target_users > self.max_users:
            print(f"\n⚠️  Reached max users ({self.max_users}), stopping test")
            self._finalize()
            return None

        return (target_users, self.spawn_rate)

    def _analyze_step(self, step_idx: int):
        """Analizza le metriche dell'ultimo step completato."""
        global _test_results, _knee_point_rps, _knee_point_users

        with _stats_lock:
            stats = dict(_step_stats[step_idx])

        if not stats["response_times"]:
            return

        users = self.start_users + step_idx * self.step_users
        n = stats["requests"]
        duration_s = self.step_time
        rps = n / duration_s
        rps_per_replica = rps / (
            self.cfg["num_replicas_per_cluster"] * len(self.cfg["clusters"])
        )

        rt_sorted = sorted(stats["response_times"])
        p50 = rt_sorted[int(len(rt_sorted) * 0.50)]
        p95 = rt_sorted[int(len(rt_sorted) * 0.95)]
        p99 = rt_sorted[int(len(rt_sorted) * 0.99)] if len(rt_sorted) > 100 else p95
        avg = stats["total_rt"] / n if n > 0 else 0
        fail_rate = stats["failures"] / n if n > 0 else 0

        result = {
            "step": step_idx,
            "users": users,
            "requests": n,
            "rps": round(rps, 1),
            "rps_per_replica": round(rps_per_replica, 1),
            "avg_ms": round(avg, 1),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
            "fail_rate": round(fail_rate, 4),
            "sla_ok": p95 < self.cfg["sla_p95_ms"] and fail_rate < self.cfg["sla_fail_rate"],
        }
        _test_results.append(result)

        # Log
        status = "✅" if result["sla_ok"] else "❌"
        print(
            f"  Step {step_idx:2d} | users={users:3d} | "
            f"rps={rps:5.1f} ({rps_per_replica:.1f}/pod) | "
            f"p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms | "
            f"fail={fail_rate*100:.1f}% | {status}"
        )

        # Aggiorna knee point: ultimo step ok
        if result["sla_ok"]:
            _knee_point_rps = rps_per_replica
            _knee_point_users = users
        else:
            print(f"\n  🛑 SLA VIOLATO (p95={p95:.0f}ms > {self.cfg['sla_p95_ms']}ms "
                  f"o fail={fail_rate*100:.1f}% > {self.cfg['sla_fail_rate']*100:.0f}%)")
            print(f"  📌 Knee point: {_knee_point_rps:.1f} rps/replica "
                  f"({_knee_point_users} users)\n")
            self._stop_flag = True

    def _finalize(self):
        """Salva i risultati e stampa il summary finale."""
        global _test_results, _knee_point_rps

        cfg = self.cfg
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_dir / f"capacity_{ts}.csv"
        json_path = output_dir / f"capacity_{ts}.json"

        # CSV
        if _test_results:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_test_results[0].keys())
                writer.writeheader()
                writer.writerows(_test_results)

        # JSON summary
        summary = {
            "timestamp": ts,
            "config": cfg,
            "knee_point_rps_per_replica": _knee_point_rps,
            "knee_point_users": _knee_point_users,
            "recommended_capacity_req_per_sec": (
                int(_knee_point_rps * 0.80) if _knee_point_rps else None
            ),  # 80% del knee = margine di sicurezza
            "steps": _test_results,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 65)
        print("📊 CAPACITY TEST RESULTS")
        print("=" * 65)
        if _knee_point_rps:
            rec = int(_knee_point_rps * 0.80)
            print(f"  Knee point:            {_knee_point_rps:.1f} rps/replica")
            print(f"  Recommended capacity:  {rec} rps/replica (80% di {_knee_point_rps:.1f})")
            print(f"")
            print(f"  → Aggiorna config/services.yaml:")
            print(f"    frontend:")
            print(f"      capacity_req_per_sec: {rec}")
            print(f"")
            print(f"  → Aggiorna analyze_test_complete.py:")
            print(f"    SERVICE_CAPACITY['frontend'] = {int(_knee_point_rps)}")
        else:
            print("  ⚠️  Nessun knee point trovato — prova ad aumentare users_max")
        print(f"")
        print(f"  Output: {csv_path}")
        print(f"  Output: {json_path}")
        print("=" * 65 + "\n")


# ─── Event hooks ─────────────────────────────────────────────────────────────

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    cfg = CAPACITY_CONFIG
    total_clusters = len(cfg["clusters"])
    total_replicas = total_clusters * cfg["num_replicas_per_cluster"]

    print("\n" + "=" * 65)
    print("🚀 DMOS CAPACITY TEST")
    print("=" * 65)
    print(f"  Clusters:      {total_clusters} ({', '.join(cfg['clusters'].keys())})")
    print(f"  Replicas:      {cfg['num_replicas_per_cluster']} per cluster → {total_replicas} totali")
    print(f"  Steps:         ogni {cfg['step_duration_s']}s, +{cfg['users_step']} utenti")
    print(f"  User range:    {cfg['users_start']} → {cfg['users_max']}")
    print(f"  SLA threshold: p95 < {cfg['sla_p95_ms']}ms, fail < {cfg['sla_fail_rate']*100:.0f}%")
    print(f"")
    print(f"  IMPORTANTE: assicurati che DMOS sia FERMO e")
    print(f"  frontend abbia esattamente {cfg['num_replicas_per_cluster']} replica/cluster")
    print("=" * 65)
    print(f"\n  {'Step':>4s} | {'Users':>5s} | {'RPS':>6s} | {'rps/pod':>7s} | "
          f"{'p50':>5s} | {'p95':>5s} | {'p99':>5s} | {'Fail':>5s} | SLA")
    print(f"  " + "-" * 60)
