"""
Diagnosi latenza Windows vs Locust/gevent
==========================================
Esegue richieste concorrenti usando:
  1. Thread reali (threading.Thread) — NO gevent
  2. Misura sia r.elapsed (requests library, sincrona)
     sia wall-clock (time.perf_counter, include scheduling OS)

Confronta i due valori per isolare se il problema è:
  - Il server (entrambi alti)
  - Gevent/scheduling (solo wall-clock alto, r.elapsed basso)

Uso:
  python experiments/diagnose_latency.py
"""

import threading
import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET = "http://192.168.1.245:30080/"
ROUNDS = [1, 5, 10, 20, 50, 100]   # Numero di richieste concorrenti per ogni round
REQUESTS_PER_ROUND = 20             # Quante richieste totali per round

session_local = threading.local()

def get_session():
    if not hasattr(session_local, "s"):
        session_local.s = requests.Session()
    return session_local.s


def single_request(idx):
    """Esegue una singola richiesta e restituisce i tempi."""
    s = get_session()
    t0 = time.perf_counter()
    try:
        r = s.get(TARGET, timeout=30)
        wall_ms = (time.perf_counter() - t0) * 1000
        elapsed_ms = r.elapsed.total_seconds() * 1000
        return {
            "idx": idx,
            "status": r.status_code,
            "wall_ms": wall_ms,
            "elapsed_ms": elapsed_ms,
            "diff_ms": wall_ms - elapsed_ms,
            "error": None,
        }
    except Exception as e:
        wall_ms = (time.perf_counter() - t0) * 1000
        return {
            "idx": idx,
            "status": 0,
            "wall_ms": wall_ms,
            "elapsed_ms": 0,
            "diff_ms": wall_ms,
            "error": str(e),
        }


def run_round(concurrency, n_requests):
    """Esegue n_requests con concurrency thread paralleli."""
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request, i) for i in range(n_requests)]
        for f in as_completed(futures):
            results.append(f.result())
    return results


def percentile(data, p):
    data_sorted = sorted(data)
    idx = int(len(data_sorted) * p / 100)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def print_stats(label, values):
    if not values:
        print(f"  {label}: N/A")
        return
    print(
        f"  {label:20s}  "
        f"p50={percentile(values,50):7.1f}ms  "
        f"p95={percentile(values,95):7.1f}ms  "
        f"p99={percentile(values,99):7.1f}ms  "
        f"avg={statistics.mean(values):7.1f}ms  "
        f"max={max(values):7.1f}ms"
    )


print("\n" + "=" * 75)
print("DIAGNOSI LATENZA — thread reali vs gevent")
print(f"Target: {TARGET}")
print("=" * 75)
print()

# ── Test sequenziale prima (baseline) ────────────────────────────────────────
print("▶ Baseline sequenziale (1 richiesta alla volta, 10 richieste):")
seq_results = [single_request(i) for i in range(10)]
wall_vals = [r["wall_ms"] for r in seq_results if r["error"] is None]
elapsed_vals = [r["elapsed_ms"] for r in seq_results if r["error"] is None]
print_stats("wall-clock", wall_vals)
print_stats("r.elapsed  ", elapsed_vals)
print_stats("differenza ", [r["diff_ms"] for r in seq_results if r["error"] is None])
print()

# ── Test concorrente con N thread ─────────────────────────────────────────────
print("▶ Test concorrente (thread reali, NO gevent):")
print(f"  {'Conc':>6s} | {'wall p50':>10s} | {'wall p95':>10s} | {'elapsed p50':>12s} | {'elapsed p95':>12s} | {'diff p50':>10s}")
print(f"  " + "-" * 72)

for concurrency in ROUNDS:
    results = run_round(concurrency, REQUESTS_PER_ROUND)
    errors = [r for r in results if r["error"]]
    ok = [r for r in results if r["error"] is None]

    if not ok:
        print(f"  {concurrency:6d} | TUTTI ERRORI: {errors[0]['error']}")
        continue

    wall_p50 = percentile([r["wall_ms"] for r in ok], 50)
    wall_p95 = percentile([r["wall_ms"] for r in ok], 95)
    el_p50   = percentile([r["elapsed_ms"] for r in ok], 50)
    el_p95   = percentile([r["elapsed_ms"] for r in ok], 95)
    diff_p50 = percentile([r["diff_ms"] for r in ok], 50)
    err_count = len(errors)

    flag = ""
    if wall_p95 > 1000:
        flag = "  ⚠️  wall-clock alto!"
    if el_p95 > 500:
        flag = "  🔴 SERVER LENTO!"

    print(
        f"  {concurrency:6d} | {wall_p50:8.0f}ms | {wall_p95:8.0f}ms | "
        f"{el_p50:10.0f}ms | {el_p95:10.0f}ms | {diff_p50:8.0f}ms"
        + (f"  (err={err_count})" if err_count else "")
        + flag
    )

print()
print("Legenda:")
print("  wall-clock  = time.perf_counter() attorno alla richiesta (include scheduling OS/thread)")
print("  r.elapsed   = misura interna di requests/urllib3 (solo rete + server, sincrona)")
print("  differenza  = overhead threading/OS (idealmente < 5ms)")
print()
print("Se wall-clock >> r.elapsed → problema di scheduling (non il server)")
print("Se entrambi alti           → server genuinamente lento")
print("=" * 75)
