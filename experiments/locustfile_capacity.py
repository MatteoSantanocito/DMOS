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
from gevent.threadpool import ThreadPoolExecutor as _GeventThreadPool

import csv
import json
import math
import threading
import requests as _requests
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Thread pool: ogni richiesta HTTP gira in un thread OS reale (non greenlet)
# → immune al gevent select() polling delay su Windows (~2700ms)
# gevent.threadpool integra i thread reali con l'event loop: il greenlet
# aspetta il risultato correttamente senza bloccare tutto Locust.
_request_pool = _GeventThreadPool(max_workers=800)
_thread_local = threading.local()

def _get_session():
    """Sessione requests per-thread (connection reuse, una per thread OS)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _requests.Session()
    return _thread_local.session

def _blocking_http_get(url):
    """Gira in un thread OS reale — r.elapsed accurato su Windows e macOS."""
    session = _get_session()
    resp = session.get(url, timeout=10)
    return resp.elapsed.total_seconds() * 1000, resp.status_code


def _blocking_http_post(url, data):
    """POST in un thread OS reale — r.elapsed accurato su Windows e macOS.
    allow_redirects=True segue i 302 che il frontend emette dopo /setCurrency
    e /cart/checkout (comportamento standard del browser).
    """
    session = _get_session()
    resp = session.post(url, data=data, timeout=10, allow_redirects=True)
    return resp.elapsed.total_seconds() * 1000, resp.status_code

# Sopprime la tabella finale di Locust (per-endpoint breakdown)
# che verrebbe stampata dopo il summary personalizzato.
import locust.stats
locust.stats.print_stats = lambda *args, **kwargs: None
locust.stats.print_percentile_stats = lambda *args, **kwargs: None
locust.stats.print_error_report = lambda *args, **kwargs: None

# Timing: r.elapsed (urllib3 interno) misurato nel thread OS reale → accurato.


# ─── Configurazione ──────────────────────────────────────────────────────────

CAPACITY_CONFIG = {
    # Target cluster IPs + porta NodePort Nginx Ingress
    "clusters": {
        "cluster1": "http://192.168.1.245:30080",
        "cluster2": "http://192.168.1.246:30080",
        "cluster3": "http://192.168.1.247:30080",
    },

    # Step configuration
    "users_start": 5,           # Utenti al primo step
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

    # Pesi del traffico per cluster (devono corrispondere ai weight delle HttpUser class)
    # Usati per calcolare il rps della replica più carica (C_replica corretto).
    # Con distribuzione uniforme (tutti uguali) il risultato coincide con total/N.
    "cluster_weights": {
        "cluster1": 40,
        "cluster2": 35,
        "cluster3": 25,
    },

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

_CHECKOUT_DATA = {
    "email":                        "test@example.com",
    "street_address":               "123 Test St",
    "zip_code":                     "10001",
    "city":                         "New York",
    "state":                        "NY",
    "country":                      "US",
    "credit_card_number":           "4432801561520454",
    "credit_card_expiration_month": "1",
    "credit_card_expiration_year":  "2030",
    "credit_card_cvv":              "672",
}

# Formato: (path, peso, metodo, data_post_o_None)
# /setCurrency e /cart/checkout richiedono POST — inviarli come GET restituisce
# 405 in ~1ms senza caricare il backend, falsando il knee point verso l'alto.
ENDPOINTS = [
    ("/",                  0.35, "GET",  None),
    ("/product/OLJCESPC7Z",0.25, "GET",  None),
    ("/product/66VCHSJNUP",0.10, "GET",  None),
    ("/cart",              0.15, "GET",  None),
    ("/setCurrency",       0.05, "POST", {"currency_code": "EUR"}),
    ("/cart/checkout",     0.10, "POST", _CHECKOUT_DATA),
]


def _weighted_choice(items):
    """Sceglie un endpoint pesato. Ritorna (path, method, data)."""
    r = __import__("random").random()
    cumulative = 0.0
    for path, weight, method, data in items:
        cumulative += weight
        if r <= cumulative:
            return path, method, data
    path, _, method, data = items[-1]
    return path, method, data


# ─── Locust User Classes ──────────────────────────────────────────────────────

# Mappa diretta classe→URL (immune a --host che Locust inietta sulle classi)
_CLUSTER_URL = {
    "Cluster1User": CAPACITY_CONFIG["clusters"]["cluster1"],
    "Cluster2User": CAPACITY_CONFIG["clusters"]["cluster2"],
    "Cluster3User": CAPACITY_CONFIG["clusters"]["cluster3"],
}

def _do_request(user_self, endpoint_tuple):
    """Esegue la richiesta in un thread OS reale via gevent.threadpool.

    Il greenlet Locust aspetta il risultato senza bloccare l'event loop.
    r.elapsed è misurato nel thread reale → nessun gevent select() delay.
    Usa GET o POST a seconda dell'endpoint per misurare il carico reale.
    """
    
   # 1. Estrae i 3 valori dall'endpoint scelto da _weighted_choice
    path, method, data = endpoint_tuple
    # es: path="/cart", method="GET", data=None
    # path="/setCurrency", method="POST", data={"currency_code":"EUR"}
    
    # 2. Costruisce URL completo per il cluster giusto
    # type(user_self).__name__ → "Cluster1User" → "http://192.168.1.245:30080"
    # url finale → "http://192.168.1.245:30080/cart"
    url = _CLUSTER_URL[type(user_self).__name__] + path

    try:
        if method == "POST":
            # 3a. Lancia la POST in un thread OS reale, il greenlet aspetta
            # senza bloccare l'event loop di gevent
            elapsed_ms, status_code = _request_pool.submit(
                _blocking_http_post, url, data
            ).result()
        else:
            # 3b. Stessa cosa per GET
            elapsed_ms, status_code = _request_pool.submit(
                _blocking_http_get, url
            ).result()
    except Exception:
         # 4. Timeout / connessione persa
        elapsed_ms, status_code = 10000.0, 500

    with _stats_lock:
         # 5. Scrittura thread-safe: più thread OS scrivono in parallelo
       # _stats_lock evita race condition sull'array condiviso
        _step_stats[_current_step]["requests"] += 1   # totale richieste
        _step_stats[_current_step]["total_rt"] += elapsed_ms # somma per calcolare media
        _step_stats[_current_step]["response_times"].append(elapsed_ms) # array per p95
        if status_code >= 500:
            _step_stats[_current_step]["failures"] += 1


class Cluster1User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster1"]
    weight = 40  # 40% del traffico → cluster1 
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request(self, _weighted_choice(ENDPOINTS))


class Cluster2User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster2"]
    weight = 35  # 35% → cluster2 
    wait_time = between(1, 3)

    @task
    def browse(self):
        _do_request(self, _weighted_choice(ENDPOINTS))


class Cluster3User(HttpUser):
    host = CAPACITY_CONFIG["clusters"]["cluster3"]
    weight = 25  # 25% → cluster3 
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
                print(f"   Warmup in corso... ancora {remaining:.0f}s "
                      f"({self.start_users} utenti, metriche non misurate)")
            return (self.start_users, self.spawn_rate)

        # ── Fase di test (misurata) ───────────────────────────────────────────
        if not _warmup_done:
            # Prima volta che usciamo dal warmup: svuota le stats accumulate
            # durante il riscaldamento così non inquinano la fase misurata
            with _stats_lock:
                _step_stats.clear()
            _warmup_done = True
            print(f"\n  Warmup completato — inizio fase di misura\n"
                  f"  {'Step':>4s} | {'Utenti':>6s} | {'RPS':>6s} | {'rps/pod':>7s} | "
                  f"{'p50':>5s} | {'p95':>5s} | {'p99':>5s} | {'Fail':>5s} | SLA")
            print(f"  " + "-" * 62)

        effective_time = run_time - warmup
        step_num = int(effective_time / self.step_time)
        target_users = self.start_users + step_num * self.step_users

        # Cambio di step: analizza il passo precedente
        #Calcolo lo step attuale, lo analizzo e passo allo step successivo 
        if step_num != _current_step and _current_step > 0:
            self._analyze_step(_current_step)

        _current_step = step_num

        if target_users > self.max_users:
            print(f"\n   Limite utenti raggiunto ({self.max_users}), test terminato")
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

        # rps_per_replica = carico della replica più carica al knee point.
        # Usa il peso massimo tra i cluster
        weights = self.cfg.get("cluster_weights", {})
        if weights:
            total_weight = sum(weights.values())
            max_fraction = max(w / total_weight for w in weights.values())
        else:
            # Fallback: distribuzione uniforme
            max_fraction = 1.0 / len(self.cfg["clusters"])
        rps_per_replica = (rps * max_fraction) / self.cfg["num_replicas_per_cluster"]

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
            stato = ""
        else:
            _consecutive_violations += 1
            stato = f" ({_consecutive_violations}/{max_consec})" if _consecutive_violations < max_consec else "❌"

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
            print(f"    p95={p95:.0f}ms > {self.cfg['sla_p95_ms']}ms — "
                  f"possibile spike, aspetto conferma al prossimo step...")
        else:
            # N step consecutivi → saturazione confermata
            print(f"\n   SLA VIOLATO per {_consecutive_violations} step consecutivi "
                  f"→ saturazione confermata")
            if _knee_point_rps is not None:
                print(f"   Knee point: {_knee_point_rps:.1f} rps/replica "
                      f"({_knee_point_users} users)\n")
            else:
                print(f"   Knee point: N/A (SLA violato già al primo step — "
                      f"sistema già saturo con {users} utenti)\n")
            self._stop_flag = True
            self._finalize()

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
                round(_knee_point_rps) if _knee_point_rps else None
            ),  # valore grezzo del knee point — il safety margin (10%) è applicato da DMOS internamente
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
            rec = round(_knee_point_rps)
            print(f"  Knee point:          {_knee_point_rps:.1f} rps/replica  ({_knee_point_users} utenti)")
            print(f"  Capacità raccomandata: {rec} rps/replica  (valore grezzo — safety margin 10% aggiunto da DMOS)")
            print(f"")
            print(f"  → Aggiorna config/services.yaml:")
            print(f"    frontend:")
            print(f"      capacity_req_per_sec: {rec}")
            print(f"")
            print(f"  → Aggiorna analyze_test_complete.py:")
            print(f"    SERVICE_CAPACITY['frontend'] = {rec}")
        else:
            print("  ⚠️  Knee point non trovato — aumenta users_max e rilancia")

        # Grafici
        chart_path = self._generate_charts(output_dir, ts)
        print(f"")
        print(f"  File CSV:   {csv_path}")
        print(f"  File JSON:  {json_path}")
        if chart_path:
            print(f"  Grafici:    {chart_path}")
        print("=" * 65 + "\n")

    def _generate_charts(self, output_dir: Path, ts: str):
        """Genera un PNG con 2 subplot: latenza e throughput vs utenti."""
        try:
            import matplotlib
            matplotlib.use("Agg")          # non-interactive, funziona senza display
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("  ⚠️  matplotlib non installato — grafici saltati (pip install matplotlib)")
            return None

        if not _test_results:
            return None

        users   = [r["users"]           for r in _test_results]
        p50     = [r["p50_ms"]          for r in _test_results]
        p95     = [r["p95_ms"]          for r in _test_results]
        p99     = [r["p99_ms"]          for r in _test_results]
        rps_pod = [r["rps_per_replica"] for r in _test_results]
        sla_ms  = self.cfg["sla_p95_ms"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("DMOS Capacity Test — Frontend", fontsize=14, fontweight="bold")

        # ── Subplot 1: Latenza ────────────────────────────────────────────────
        ax1.plot(users, p50, "o-", color="#2196F3", linewidth=2, markersize=4, label="p50")
        ax1.plot(users, p95, "s-", color="#FF9800", linewidth=2, markersize=4, label="p95")
        ax1.plot(users, p99, "^-", color="#F44336", linewidth=1.5, markersize=4,
                 label="p99", alpha=0.7)
        ax1.axhline(sla_ms, color="#F44336", linestyle="--", linewidth=1.5,
                    label=f"SLA p95 = {sla_ms}ms")

        if _knee_point_users:
            ax1.axvline(_knee_point_users, color="#9C27B0", linestyle=":", linewidth=2,
                        label=f"Knee point ({_knee_point_users} utenti)")
            # Evidenzia il knee point sulla curva p95
            knee_results = [r for r in _test_results if r["users"] == _knee_point_users]
            if knee_results:
                ax1.plot(_knee_point_users, knee_results[0]["p95_ms"],
                         "*", color="#9C27B0", markersize=14, zorder=5)

        ax1.set_ylabel("Latenza (ms)", fontsize=11)
        ax1.set_ylim(bottom=0)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Latenza per-percentile", fontsize=11)

        # Colora le righe oltre SLA in rosso chiaro
        for r in _test_results:
            if not r["sla_ok"]:
                ax1.axvspan(r["users"] - self.cfg["users_step"] / 2,
                            r["users"] + self.cfg["users_step"] / 2,
                            alpha=0.15, color="#F44336")

        # ── Subplot 2: Throughput ─────────────────────────────────────────────
        ax2.plot(users, rps_pod, "o-", color="#4CAF50", linewidth=2, markersize=4,
                 label="rps/replica")

        if _knee_point_users and _knee_point_rps:
            ax2.axvline(_knee_point_users, color="#9C27B0", linestyle=":", linewidth=2,
                        label=f"Knee point ({_knee_point_users} utenti)")
            ax2.axhline(_knee_point_rps, color="#9C27B0", linestyle="--", linewidth=1,
                        alpha=0.6, label=f"Max = {_knee_point_rps:.1f} rps/replica")
            ax2.plot(_knee_point_users, _knee_point_rps,
                     "*", color="#9C27B0", markersize=14, zorder=5)

            rec = round(_knee_point_rps)
            ax2.axhline(rec, color="#FF9800", linestyle="--", linewidth=1.5,
                        label=f"Raccomandato = {rec} rps/replica (grezzo)")

        ax2.set_xlabel("Utenti virtuali", fontsize=11)
        ax2.set_ylabel("Throughput (rps/replica)", fontsize=11)
        ax2.set_ylim(bottom=0)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Throughput sostenibile", fontsize=11)

        plt.tight_layout()
        chart_path = output_dir / f"capacity_{ts}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return chart_path


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
