#!/usr/bin/env python3
"""
Capacity Test — asyncio/aiohttp  (Windows-safe, sostituto di locustfile_capacity.py)
======================================================================================
Perché questo file invece di Locust?
  Locust usa gevent, che su Windows usa select() con risoluzione ~2700ms.
  Ogni richiesta HTTP (risposta reale ~60ms) viene misurata come ~2700ms.
  asyncio su Windows 3.8+ usa ProactorEventLoop (IOCP) — nessun problema.

Uso:
    pip install aiohttp
    python experiments/capacity_test_async.py

Output identico a locustfile_capacity.py:
    results/capacity/capacity_YYYYMMDD_HHMMSS.csv
    results/capacity/capacity_YYYYMMDD_HHMMSS.json
"""

import asyncio
import aiohttp
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── Configurazione ────────────────────────────────────────────────────────────

CAPACITY_CONFIG = {
    "clusters": {
        "cluster1": "http://192.168.1.245:30080",
        "cluster2": "http://192.168.1.246:30080",
        "cluster3": "http://192.168.1.247:30080",
    },
    "cluster_weights": [40, 35, 25],    # % traffico: cluster1=40, cluster2=35, cluster3=25
    "users_start":     5,
    "users_step":      5,
    "users_max":       700,
    "step_duration_s": 90,
    "spawn_rate":      5,               # utenti/secondo aggiunti ad ogni step
    "warmup_duration_s": 120,
    "sla_p95_ms":      1000,
    "sla_fail_rate":   0.02,
    "output_dir":      "results/capacity",
    "num_replicas_per_cluster": 1,
}

ENDPOINTS = [
    ("/",                   0.35),
    ("/product/OLJCESPC7Z", 0.25),
    ("/product/66VCHSJNUP", 0.10),
    ("/cart",               0.15),
    ("/setCurrency",        0.05),
    ("/cart/checkout",      0.10),
]

# ── Stato condiviso ───────────────────────────────────────────────────────────

_lock         = asyncio.Lock()
_step_stats   = defaultdict(lambda: {"n": 0, "fail": 0, "rt_ms": []})
_current_step = 0
_record       = False   # True solo durante le fasi misurate (non warmup)


# ── Logica utente ─────────────────────────────────────────────────────────────

def _pick_endpoint() -> str:
    r = random.random()
    cumulative = 0.0
    for ep, w in ENDPOINTS:
        cumulative += w
        if r <= cumulative:
            return ep
    return ENDPOINTS[-1][0]


async def _user_loop(base_url: str, stop_evt: asyncio.Event) -> None:
    """Un singolo utente virtuale: richiesta → sleep(1–3s) → richiesta → ..."""
    connector = aiohttp.TCPConnector(limit=1, keepalive_timeout=60)
    timeout   = aiohttp.ClientTimeout(total=15)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while not stop_evt.is_set():
            ep = _pick_endpoint()
            t0 = time.perf_counter()
            ok = False
            try:
                async with session.get(base_url + ep) as resp:
                    await resp.read()           # consuma il body per keep-alive
                    ok = resp.status < 500
            except Exception:
                pass
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if _record:
                async with _lock:
                    s = _step_stats[_current_step]
                    s["n"] += 1
                    s["rt_ms"].append(elapsed_ms)
                    if not ok:
                        s["fail"] += 1

            # wait_time tra 1 e 3 secondi (come Locust between(1,3))
            wait = random.uniform(1.0, 3.0)
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=wait)
                return      # stop_evt scattato durante il sleep
            except asyncio.TimeoutError:
                pass        # normale: continua il loop


# ── Analisi step ──────────────────────────────────────────────────────────────

def _analyze_step(step_idx: int, n_users: int, cfg: dict):
    d = _step_stats[step_idx]
    n = d["n"]
    if n == 0:
        return None

    step_s  = cfg["step_duration_s"]
    n_pods  = len(cfg["clusters"]) * cfg["num_replicas_per_cluster"]
    rps     = n / step_s
    rps_pod = rps / n_pods

    rt = sorted(d["rt_ms"])
    p50 = rt[int(len(rt) * 0.50)]
    p95 = rt[int(len(rt) * 0.95)]
    p99 = rt[int(len(rt) * 0.99)] if len(rt) > 100 else p95
    avg = sum(d["rt_ms"]) / n
    fail_rate = d["fail"] / n
    sla_ok = p95 < cfg["sla_p95_ms"] and fail_rate < cfg["sla_fail_rate"]

    return {
        "step":            step_idx,
        "users":           n_users,
        "requests":        n,
        "rps":             round(rps, 1),
        "rps_per_replica": round(rps_pod, 1),
        "avg_ms":          round(avg, 1),
        "p50_ms":          round(p50, 1),
        "p95_ms":          round(p95, 1),
        "p99_ms":          round(p99, 1),
        "fail_rate":       round(fail_rate, 4),
        "sla_ok":          sla_ok,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_test() -> None:
    global _current_step, _record

    cfg        = CAPACITY_CONFIG
    urls       = list(cfg["clusters"].values())
    weights    = cfg["cluster_weights"]
    warmup_s   = cfg["warmup_duration_s"]
    step_s     = cfg["step_duration_s"]
    spawn_rate = cfg["spawn_rate"]
    n_pods     = len(cfg["clusters"]) * cfg["num_replicas_per_cluster"]

    results    = []
    knee_rps   = None
    knee_users = None

    print("\n" + "=" * 65)
    print("🚀 DMOS CAPACITY TEST  (asyncio — Windows IOCP)")
    print("=" * 65)
    print(f"  Cluster:   {len(urls)} ({', '.join(cfg['clusters'].keys())})")
    print(f"  Repliche:  {cfg['num_replicas_per_cluster']}/cluster → {n_pods} totali")
    print(f"  Warmup:    {warmup_s}s  |  Step: {step_s}s  +{cfg['users_step']} utenti")
    print(f"  Range:     {cfg['users_start']}→{cfg['users_max']} utenti")
    print(f"  SLA:       p95 < {cfg['sla_p95_ms']}ms,  fail < {cfg['sla_fail_rate']*100:.0f}%")
    print("=" * 65)

    stop_evt = asyncio.Event()
    tasks    = []

    def _spawn(n: int) -> None:
        for _ in range(n):
            url = random.choices(urls, weights=weights)[0]
            tasks.append(asyncio.create_task(_user_loop(url, stop_evt)))

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"\n  🔥 Warmup {warmup_s}s ({cfg['users_start']} utenti, metriche non misurate)...")
    _spawn(cfg["users_start"])

    t0_warmup = time.monotonic()
    while True:
        elapsed = time.monotonic() - t0_warmup
        remaining = warmup_s - elapsed
        if remaining <= 0.5:
            break
        chunk = min(30.0, remaining)
        await asyncio.sleep(chunk)
        remaining2 = warmup_s - (time.monotonic() - t0_warmup)
        if remaining2 > 2:
            print(f"     🔥 ancora {remaining2:.0f}s...")

    _step_stats.clear()
    _record = True
    current_users = cfg["users_start"]

    print(f"\n  ✅ Warmup completato — inizio fase di misura\n"
          f"  {'Step':>4} | {'Utenti':>6} | {'RPS':>6} | {'rps/pod':>7} | "
          f"{'p50':>5} | {'p95':>5} | {'p99':>5} | {'Fail':>5} | SLA")
    print("  " + "-" * 62)

    # ── Stepped test ──────────────────────────────────────────────────────────
    max_steps = (cfg["users_max"] - cfg["users_start"]) // cfg["users_step"] + 2

    for step_idx in range(max_steps):
        target = cfg["users_start"] + step_idx * cfg["users_step"]
        if target > cfg["users_max"]:
            print(f"\n  ⚠️  Limite utenti raggiunto ({cfg['users_max']}), test terminato")
            break

        # Aggiungi utenti per questo step (con spawn_rate utenti/secondo)
        to_add = target - current_users
        if to_add > 0:
            for i in range(0, to_add, spawn_rate):
                batch = min(spawn_rate, to_add - i)
                _spawn(batch)
                if i + spawn_rate < to_add:
                    await asyncio.sleep(1.0)
            current_users = target

        _current_step = step_idx
        await asyncio.sleep(step_s)

        r = _analyze_step(step_idx, current_users, cfg)
        if r is None:
            print(f"  Step {step_idx:2d}: nessun dato, salto")
            continue

        results.append(r)
        stato = "✅" if r["sla_ok"] else "❌"
        print(f"  Step {step_idx:2d} | utenti={current_users:4d} | "
              f"rps={r['rps']:5.1f} ({r['rps_per_replica']:.1f}/pod) | "
              f"p50={r['p50_ms']:.0f}ms p95={r['p95_ms']:.0f}ms p99={r['p99_ms']:.0f}ms | "
              f"fail={r['fail_rate']*100:.1f}% | {stato}")

        if r["sla_ok"]:
            knee_rps   = r["rps_per_replica"]
            knee_users = current_users
        else:
            print(f"\n  🛑 SLA VIOLATO  (p95={r['p95_ms']:.0f}ms > {cfg['sla_p95_ms']}ms "
                  f"o fail={r['fail_rate']*100:.1f}% > {cfg['sla_fail_rate']*100:.0f}%)")
            if knee_rps is not None:
                print(f"  📌 Knee point: {knee_rps:.1f} rps/replica  ({knee_users} utenti)\n")
            else:
                print(f"  📌 Knee point: N/A — SLA violato già al primo step\n")
            break

    # ── Stop ──────────────────────────────────────────────────────────────────
    stop_evt.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # ── Salva risultati ───────────────────────────────────────────────────────
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if results:
        csv_path = out_dir / f"capacity_{ts}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    json_path = out_dir / f"capacity_{ts}.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp":    ts,
            "config":       cfg,
            "knee_point_rps_per_replica":     knee_rps,
            "knee_point_users":               knee_users,
            "recommended_capacity_req_per_sec": (
                max(1, round(knee_rps * 0.80)) if knee_rps else None
            ),
            "steps": results,
        }, f, indent=2)

    # ── Summary finale ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 RISULTATI CAPACITY TEST")
    print("=" * 65)
    if results:
        last = results[-1]
        print(f"  Ultimo step: {last['users']} utenti | "
              f"rps={last['rps']:.1f} | p50={last['p50_ms']:.0f}ms | "
              f"p95={last['p95_ms']:.0f}ms | fail={last['fail_rate']*100:.1f}%")

    if knee_rps is not None:
        rec = max(1, round(knee_rps * 0.80))
        print(f"\n  Knee point:           {knee_rps:.1f} rps/replica  ({knee_users} utenti)")
        print(f"  Capacità raccomandata: {rec} rps/replica  (80% del knee)")
        print(f"")
        print(f"  → Aggiorna config/services.yaml:")
        print(f"      frontend:")
        print(f"        capacity_req_per_sec: {rec}")
    else:
        print("\n  ⚠️  Knee point non trovato — aumenta users_max e rilancia")

    print(f"\n  File JSON: {json_path}")
    if results:
        print(f"  File CSV:  {csv_path}")
    print("=" * 65 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Su Windows 3.8+: forza ProactorEventLoop (IOCP).
    # SelectorEventLoop usa select() con la stessa risoluzione 2700ms di gevent.
    # ProactorEventLoop usa I/O Completion Ports (IOCP): nessun problema.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(run_test())
