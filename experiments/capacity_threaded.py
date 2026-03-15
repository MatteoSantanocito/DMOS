"""
Capacity Test con thread reali (NO gevent/Locust)
===================================================
Stesso identico design del locustfile_capacity.py ma usa
threading.Thread invece di greenlet gevent — misure accurate
su Windows senza il problema del select() polling delay.

Misura: r.elapsed.total_seconds() — timing urllib3 interno,
        non affetto da scheduling OS/gevent.

Uso:
  python experiments/capacity_threaded.py

Output:
  results/capacity/capacity_threaded_YYYYMMDD_HHMMSS.csv
  results/capacity/capacity_threaded_YYYYMMDD_HHMMSS.json
"""

import threading
import time
import random
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Configurazione ────────────────────────────────────────────────────────────

CONFIG = {
    "clusters": {
        "cluster1": "http://192.168.1.245:30080",
        "cluster2": "http://192.168.1.246:30080",
        "cluster3": "http://192.168.1.247:30080",
    },
    # Pesi traffico per cluster (devono sommare a 1.0)
    "cluster_weights": [0.40, 0.35, 0.25],

    "users_start":        150,   # Utenti iniziali (abbastanza per misure stabili)
    "users_step":         5,     # Utenti aggiunti ogni step
    "users_max":          700,   # Limite massimo utenti
    "step_duration_s":    90,    # Durata ogni step (secondi)
    "spawn_rate":         5,     # Nuovi thread/secondo durante lo spawn
    "warmup_duration_s":  120,   # Warmup iniziale non misurato

    "wait_time_min":      1.0,   # Think time minimo tra richieste (secondi)
    "wait_time_max":      3.0,   # Think time massimo

    "sla_p95_ms":         200,   # Soglia SLA (200ms = risposta percepita istantanea)
    "sla_fail_rate":      0.02,  # Max 2% errori
    "sla_consecutive":    2,     # Stop solo dopo N step consecutivi in violazione

    "output_dir":         "results/capacity",
    "num_replicas_per_cluster": 1,
}

ENDPOINTS = [
    ("/", 0.35),
    ("/product/OLJCESPC7Z", 0.25),
    ("/product/66VCHSJNUP", 0.10),
    ("/cart", 0.15),
    ("/setCurrency", 0.05),
    ("/cart/checkout", 0.10),
]

# ─── Stato globale ─────────────────────────────────────────────────────────────

_lock = threading.Lock()
_step_data = defaultdict(lambda: {"elapsed_ms": [], "failures": 0})
_current_step = 0          # Step attuale (0 = warmup, 1+ = step misurati)
_warmup_done = False
_stop_event = threading.Event()
_results = []
_knee_point_rps = None
_knee_point_users = None
_consecutive_violations = 0

# ─── Utility ───────────────────────────────────────────────────────────────────

_cluster_list = list(CONFIG["clusters"].values())
_cluster_weights = CONFIG["cluster_weights"]

def _pick_cluster():
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(_cluster_weights):
        cumulative += w
        if r <= cumulative:
            return _cluster_list[i]
    return _cluster_list[-1]

def _pick_endpoint():
    r = random.random()
    cumulative = 0.0
    for ep, w in ENDPOINTS:
        cumulative += w
        if r <= cumulative:
            return ep
    return ENDPOINTS[-1][0]

# ─── Worker thread ─────────────────────────────────────────────────────────────

def user_worker():
    """Simula un utente: richiesta → think time → ripeti."""
    session = requests.Session()
    while not _stop_event.is_set():
        base = _pick_cluster()
        endpoint = _pick_endpoint()
        url = base + endpoint

        try:
            resp = session.get(url, timeout=10)
            elapsed_ms = resp.elapsed.total_seconds() * 1000
            is_failure = resp.status_code >= 500
        except Exception:
            elapsed_ms = 10000.0
            is_failure = True

        with _lock:
            step = _current_step
            if _warmup_done and step > 0:
                _step_data[step]["elapsed_ms"].append(elapsed_ms)
                if is_failure:
                    _step_data[step]["failures"] += 1

        # Think time
        think = random.uniform(CONFIG["wait_time_min"], CONFIG["wait_time_max"])
        _stop_event.wait(think)

# ─── Analisi step ──────────────────────────────────────────────────────────────

def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]

def analyze_step(step_idx, n_users):
    global _knee_point_rps, _knee_point_users, _consecutive_violations

    with _lock:
        data = dict(_step_data[step_idx])

    times = data["elapsed_ms"]
    n_fail = data["failures"]
    n_req = len(times)

    if n_req == 0:
        print(f"  Step {step_idx:2d} | utenti={n_users:4d} | NESSUNA RICHIESTA COMPLETATA")
        return True  # Continua

    duration = CONFIG["step_duration_s"]
    rps = n_req / duration
    n_clusters = len(CONFIG["clusters"])
    n_replicas = CONFIG["num_replicas_per_cluster"]
    rps_per_pod = rps / (n_clusters * n_replicas)

    p50 = percentile(times, 50)
    p95 = percentile(times, 95)
    p99 = percentile(times, 99)
    fail_rate = n_fail / n_req

    sla_ok = p95 < CONFIG["sla_p95_ms"] and fail_rate < CONFIG["sla_fail_rate"]

    if sla_ok:
        _consecutive_violations = 0
        stato = "✅"
    else:
        _consecutive_violations += 1
        max_consec = CONFIG["sla_consecutive"]
        stato = f"⚠️ ({_consecutive_violations}/{max_consec})" if _consecutive_violations < max_consec else "❌"

    print(
        f"  Step {step_idx:2d} | utenti={n_users:4d} | "
        f"rps={rps:5.1f} ({rps_per_pod:.1f}/pod) | "
        f"p50={p50:.0f}ms p95={p95:.0f}ms p99={p99:.0f}ms | "
        f"fail={fail_rate*100:.1f}% | {stato}"
    )

    result = {
        "step": step_idx,
        "users": n_users,
        "requests": n_req,
        "rps": round(rps, 1),
        "rps_per_replica": round(rps_per_pod, 1),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "fail_rate": round(fail_rate, 4),
        "sla_ok": sla_ok,
    }
    _results.append(result)

    if sla_ok:
        _knee_point_rps = rps_per_pod
        _knee_point_users = n_users
        return True  # Continua

    # Violazione: aggiorna knee point solo se il passo precedente era ok
    if _consecutive_violations == 1:
        # Primo step in violazione — potrebbe essere spike, continua
        print(f"  ⚠️  p95={p95:.0f}ms > {CONFIG['sla_p95_ms']}ms — "
              f"possibile spike, aspetto conferma al prossimo step...")
        return True  # Continua ancora

    # N step consecutivi in violazione → knee point confermato
    print(f"\n  🛑 SLA VIOLATO per {_consecutive_violations} step consecutivi "
          f"→ saturazione confermata")
    if _knee_point_rps:
        print(f"  📌 Knee point: {_knee_point_rps:.1f} rps/replica "
              f"({_knee_point_users} utenti)\n")
    else:
        print(f"  📌 Knee point: N/A — SLA violato già al primo step\n")
    return False  # Stop

# ─── Salvataggio risultati ─────────────────────────────────────────────────────

def save_results():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"capacity_threaded_{ts}.csv"
    json_path = output_dir / f"capacity_threaded_{ts}.json"

    if _results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_results[0].keys())
            writer.writeheader()
            writer.writerows(_results)

    rec = max(1, round(_knee_point_rps * 0.80)) if _knee_point_rps else None
    summary = {
        "timestamp": ts,
        "config": CONFIG,
        "knee_point_rps_per_replica": _knee_point_rps,
        "knee_point_users": _knee_point_users,
        "recommended_capacity_req_per_sec": rec,
        "steps": _results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("📊 RISULTATI CAPACITY TEST")
    print("=" * 65)

    if _results:
        last = _results[-1]
        stato = "✅ SLA rispettato" if last["sla_ok"] else "❌ SLA violato"
        print(f"  Ultimo step:  {last['users']} utenti | "
              f"{last['rps']:.1f} RPS ({last['rps_per_replica']:.1f}/pod) | "
              f"p95={last['p95_ms']:.0f}ms | {stato}")

    if _knee_point_rps:
        print(f"  Knee point:            {_knee_point_rps:.1f} rps/replica  ({_knee_point_users} utenti)")
        print(f"  Capacità raccomandata: {rec} rps/replica  (80% knee point)")
        print(f"")
        print(f"  → config/services.yaml:")
        print(f"      capacity_req_per_sec: {rec}")
    else:
        print("  ⚠️  Knee point non trovato")

    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 65)

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _current_step, _warmup_done

    cfg = CONFIG
    n_clusters = len(cfg["clusters"])
    n_replicas = cfg["num_replicas_per_cluster"]
    warmup = cfg["warmup_duration_s"]

    print("\n" + "=" * 65)
    print("🚀 DMOS CAPACITY TEST  (thread reali, NO gevent)")
    print("=" * 65)
    print(f"  Cluster:       {n_clusters} ({', '.join(cfg['clusters'].keys())})")
    print(f"  Repliche:      {n_replicas} per cluster → {n_clusters * n_replicas} totali")
    print(f"  Warmup:        {warmup}s a {cfg['users_start']} utenti (non misurato)")
    print(f"  Step:          ogni {cfg['step_duration_s']}s, +{cfg['users_step']} utenti")
    print(f"  Range utenti:  {cfg['users_start']} → {cfg['users_max']}")
    print(f"  Soglia SLA:    p95 < {cfg['sla_p95_ms']}ms, fail < {cfg['sla_fail_rate']*100:.0f}%")
    print(f"  Misura:        r.elapsed (urllib3 interno, immune a gevent)")
    print("=" * 65)

    threads = []

    def spawn_users(target_total):
        """Porta il numero di thread attivi a target_total."""
        current = len([t for t in threads if t.is_alive()])
        to_add = target_total - current
        for _ in range(to_add):
            t = threading.Thread(target=user_worker, daemon=True)
            t.start()
            threads.append(t)
            time.sleep(1.0 / cfg["spawn_rate"])

    # ── Warmup ──────────────────────────────────────────────────────────────
    print(f"\n  🔥 Warmup: spawning {cfg['users_start']} utenti...")
    spawn_users(cfg["users_start"])
    print(f"  🔥 Warmup in corso ({warmup}s)...")

    for remaining in range(warmup, 0, -30):
        _stop_event.wait(min(30, remaining))
        if remaining > 30:
            print(f"  🔥 Warmup ancora {remaining-30}s...")

    # Fine warmup
    _current_step = 1
    _warmup_done = True
    print(f"\n  ✅ Warmup completato — inizio fase di misura")
    print(f"  {'Step':>4s} | {'Utenti':>6s} | {'RPS':>6s} | {'rps/pod':>7s} | "
          f"{'p50':>5s} | {'p95':>5s} | {'p99':>5s} | {'Fail':>5s} | SLA")
    print(f"  " + "-" * 62)

    # ── Step misurati ────────────────────────────────────────────────────────
    step_idx = 1
    n_users = cfg["users_start"]

    while n_users <= cfg["users_max"]:
        _current_step = step_idx
        step_start = time.perf_counter()

        # Aspetta la durata dello step
        _stop_event.wait(cfg["step_duration_s"])

        # Analizza step precedente
        should_continue = analyze_step(step_idx, n_users)
        if not should_continue:
            break

        # Prossimo step
        step_idx += 1
        n_users += cfg["users_step"]

        if n_users > cfg["users_max"]:
            print(f"\n  ⚠️  Limite utenti raggiunto ({cfg['users_max']}), test terminato")
            break

        # Spawn nuovi utenti
        spawn_users(n_users)

    # Fine test
    _stop_event.set()
    save_results()


if __name__ == "__main__":
    main()
