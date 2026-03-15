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

import csv
import json
import math
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Sopprime la tabella finale di Locust (per-endpoint breakdown)
# che verrebbe stampata dopo il summary personalizzato.
import locust.stats
locust.stats.print_stats = lambda *args, **kwargs: None
locust.stats.print_percentile_stats = lambda *args, **kwargs: None
locust.stats.print_error_report = lambda *args, **kwargs: None

# NOTA: non usare r.elapsed.total_seconds() — in Locust con catch_response=True
# restituisce 0 in alcune versioni. Usa r.request_meta["response_time"] che è la
# misura interna di Locust (time.perf_counter(), senza overhead gevent su Windows).


# ─── Configurazione ──────────────────────────────────────────────────────────

CAPACITY_CONFIG = {
    # Target cluster IPs + porta NodePort Nginx Ingress
    "clusters": {
        "cluster1": "http://192.168.1.245:30080",
        "cluster2": "http://192.168.1.246:30080",
        "cluster3": "http://192.168.1.247:30080",
    },

    # Step configuration
    "users_start": 150,         # Utenti al primo step — abbastanza da tenere gevent attivo su Windows
    "users_step": 5,            # Utenti aggiunti ad ogni step
    "users_max": 700,           # Stop se si raggiunge questo valore
    "step_duration_s": 90,      # Durata di ogni step (secondi)
    "spawn_rate": 5,            # Utenti/secondo durante lo spawn
    "warmup_duration_s": 120,   # Riscaldamento iniziale non misurato (2 min a users_start)
                                 # Evita falsi SLA fail da Major GC JVM dopo test precedente

    # SLA thresholds — il test si ferma quando vengono violati
    "sla_p95_ms": 200,          # ms — 200ms = risposta percepita istantanea (baseline ~70ms)
    "sla_fail_rate": 0.02,      # 2% max fail
    "sla_consecutive": 2,       # Stop solo dopo N step consecutivi in violazione (evita falsi stop da GC spike)

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
_warmup_done = False       # Flag: warmup completato, metriche ora valide
_consecutive_violations = 0

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

def _do_request(user_self, endpoint):
    """Esegue la richiesta usando la latenza interna di Locust (request_meta).

    NON usa time.time() perché su Windows include ~2600ms di overhead gevent
    (scheduling delay quando il greenlet ritorna dalla sleep). La misura corretta
    è r.request_meta["response_time"] che Locust calcola con time.perf_counter()
    dentro la macchina HTTP, prima di restituire il controllo al greenlet.
    """
    with user_self.client.get(endpoint, catch_response=True, name=endpoint) as r:
        # request_meta["response_time"] è in millisecondi (float), misurato da Locust
        # con perf_counter() — stessa metrica mostrata dalla tabella built-in di Locust.
        elapsed_ms = r.request_meta["response_time"]
        with _stats_lock:
            _step_stats[_current_step]["requests"] += 1
            _step_stats[_current_step]["total_rt"] += elapsed_ms
            _step_stats[_current_step]["response_times"].append(elapsed_ms)
            if r.status_code >= 500:
                _step_stats[_current_step]["failures"] += 1
                r.failure(f"HTTP {r.status_code}")
            else:
                r.success()


class Cluster1User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster1"]
    weight = 40  # 40% del traffico → cluster1 (DE)
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request(self, _weighted_choice(ENDPOINTS))


class Cluster2User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster2"]
    weight = 35  # 35% → cluster2 (FR)
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request(self, _weighted_choice(ENDPOINTS))


class Cluster3User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster3"]
    weight = 25  # 25% → cluster3 (PL)
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request(self, _weighted_choice(ENDPOINTS))


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
        global _current_step, _test_results, _knee_point_rps, _knee_point_users, _warmup_done

        if self._stop_flag:
            return None  # Stop test

        run_time = self.get_run_time()
        warmup = self.cfg.get("warmup_duration_s", 0)

        # ── Fase di warmup (non misurata) ────────────────────────────────────
        # Tiene users_start utenti attivi ma NON analizza le metriche.
        # Serve a scaldare le JVM, svuotare gli heap da GC post-test, stabilizzare
        # i connection pool — evita falsi SLA fail al primo step misurato.
        if run_time < warmup:
            remaining = warmup - run_time
            if int(run_time) % 30 == 0 and run_time > 0:
                # Log ogni 30s durante warmup per confermare che è attivo
                print(f"  🔥 Warmup in corso... ancora {remaining:.0f}s "
                      f"({self.start_users} utenti, metriche non misurate)")
            return (self.start_users, self.spawn_rate)

        # ── Fase di test (misurata) ───────────────────────────────────────────
        if not _warmup_done:
            # Prima volta che usciamo dal warmup: svuota le stats accumulate
            # durante il riscaldamento così non inquinano la fase misurata
            with _stats_lock:
                _step_stats.clear()
            _warmup_done = True
            print(f"\n  ✅ Warmup completato — inizio fase di misura\n"
                  f"  {'Step':>4s} | {'Utenti':>6s} | {'RPS':>6s} | {'rps/pod':>7s} | "
                  f"{'p50':>5s} | {'p95':>5s} | {'p99':>5s} | {'Fail':>5s} | SLA")
            print(f"  " + "-" * 62)

        effective_time = run_time - warmup
        step_num = int(effective_time / self.step_time)
        target_users = self.start_users + step_num * self.step_users

        # Cambio di step: analizza il passo precedente
        if step_num != _current_step and _current_step > 0:
            self._analyze_step(_current_step)

        _current_step = step_num

        if target_users > self.max_users:
            print(f"\n  ⚠️  Limite utenti raggiunto ({self.max_users}), test terminato")
            self._finalize()
            return None

        return (target_users, self.spawn_rate)

    def _analyze_step(self, step_idx: int):
        """Analizza le metriche dell'ultimo step completato."""
        global _test_results, _knee_point_rps, _knee_point_users, _consecutive_violations

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

        sla_ok = p95 < self.cfg["sla_p95_ms"] and fail_rate < self.cfg["sla_fail_rate"]
        max_consec = self.cfg.get("sla_consecutive", 1)

        if sla_ok:
            _consecutive_violations = 0
            stato = "✅"
        else:
            _consecutive_violations += 1
            stato = f"⚠️ ({_consecutive_violations}/{max_consec})" if _consecutive_violations < max_consec else "❌"

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
            "sla_ok": sla_ok,
        }
        _test_results.append(result)

        print(
            f"  Step {step_idx:2d} | utenti={users:4d} | "
            f"rps={rps:5.1f} ({rps_per_replica:.1f}/pod) | "
            f"p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms | "
            f"fail={fail_rate*100:.1f}% | {stato}"
        )

        if sla_ok:
            _knee_point_rps = rps_per_replica
            _knee_point_users = users
        elif _consecutive_violations == 1:
            # Primo step in violazione — potrebbe essere spike GC, aspetta conferma
            print(f"  ⚠️  p95={p95:.0f}ms > {self.cfg['sla_p95_ms']}ms — "
                  f"possibile spike, aspetto conferma al prossimo step...")
        else:
            # N step consecutivi → saturazione confermata
            print(f"\n  🛑 SLA VIOLATO per {_consecutive_violations} step consecutivi "
                  f"→ saturazione confermata")
            if _knee_point_rps is not None:
                print(f"  📌 Knee point: {_knee_point_rps:.1f} rps/replica "
                      f"({_knee_point_users} users)\n")
            else:
                print(f"  📌 Knee point: N/A (SLA violato già al primo step — "
                      f"sistema già saturo con {users} utenti)\n")
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
                max(1, round(_knee_point_rps * 0.80)) if _knee_point_rps else None
            ),  # 80% del knee = margine di sicurezza (max(1,...) evita di ottenere 0)
            "steps": _test_results,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Controlla se il rps era già piatto dal primo step (sistema saturo a priori)
        _warn_saturation = False
        if len(_test_results) >= 3:
            rps_values = [r["rps"] for r in _test_results]
            rps_range = max(rps_values) - min(rps_values)
            if rps_range < 1.0:  # rps variato meno di 1 req/s in tutto il test
                _warn_saturation = True

        print("\n" + "=" * 65)
        print("📊 RISULTATI CAPACITY TEST")
        print("=" * 65)
        if _warn_saturation:
            print(f"  ⚠️  ATTENZIONE: throughput piatto ({rps_values[0]:.1f}→{rps_values[-1]:.1f} rps)")
            print(f"  Il sistema era già saturo dal primo step.")
            print(f"  Verifica che i pod siano attivi e il nodo abbia risorse libere.")
            print()

        # Riepilogo dell'ultimo step misurato
        if _test_results:
            last = _test_results[-1]
            stato = "✅ SLA rispettato" if last["sla_ok"] else "❌ SLA violato"
            print(f"  Ultimo step misurato:")
            print(f"    Utenti:  {last['users']}")
            print(f"    RPS:     {last['rps']:.1f} totali  ({last['rps_per_replica']:.1f}/pod)")
            print(f"    p50:     {last['p50_ms']:.0f}ms")
            print(f"    p95:     {last['p95_ms']:.0f}ms")
            print(f"    p99:     {last['p99_ms']:.0f}ms")
            print(f"    Fallimenti: {last['fail_rate']*100:.1f}%")
            print(f"    Stato:   {stato}")
            print()

        if _knee_point_rps:
            rec = max(1, round(_knee_point_rps * 0.80))
            print(f"  Knee point:          {_knee_point_rps:.1f} rps/replica  ({_knee_point_users} utenti)")
            print(f"  Capacità raccomandata: {rec} rps/replica  (80% del knee point)")
            print(f"")
            print(f"  → Aggiorna config/services.yaml:")
            print(f"    frontend:")
            print(f"      capacity_req_per_sec: {rec}")
            print(f"")
            print(f"  → Aggiorna analyze_test_complete.py:")
            print(f"    SERVICE_CAPACITY['frontend'] = {int(_knee_point_rps)}")
        else:
            print("  ⚠️  Knee point non trovato — aumenta users_max e rilancia")
        print(f"")
        print(f"  File CSV:  {csv_path}")
        print(f"  File JSON: {json_path}")
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
    warmup = cfg.get("warmup_duration_s", 0)
    print(f"  Cluster:       {total_clusters} ({', '.join(cfg['clusters'].keys())})")
    print(f"  Repliche:      {cfg['num_replicas_per_cluster']} per cluster → {total_replicas} totali")
    print(f"  Warmup:        {warmup}s a {cfg['users_start']} utenti (non misurato)")
    print(f"  Step:          ogni {cfg['step_duration_s']}s, +{cfg['users_step']} utenti")
    print(f"  Range utenti:  {cfg['users_start']} → {cfg['users_max']}")
    print(f"  Soglia SLA:    p95 < {cfg['sla_p95_ms']}ms, fallimenti < {cfg['sla_fail_rate']*100:.0f}%")
    print(f"")
    print(f"  ⚠️  Assicurati che DMOS sia FERMO e")
    print(f"  il frontend abbia esattamente {cfg['num_replicas_per_cluster']} replica/cluster")
    print("=" * 65)
    if warmup > 0:
        print(f"\n  🔥 Avvio fase di warmup ({warmup}s)...")
