#!/usr/bin/env python3
"""
warmup_clusters.py — Pre-warm gRPC connection pools before Locust test

WHY:
  Online Boutique usa grpc.Dial() lazy: le connessioni ai backend (cartservice,
  productcatalogservice, ecc.) non sono stabilite al boot del pod ma alla prima
  richiesta HTTP al frontend. Con Cilium, il primo SYN su una connessione nuova
  può essere droppato mentre Cilium aggiorna il datapath → TCP timeout 10s per
  servizio → 500 dopo ~20s.

  Questo script invia alcune richieste "di riscaldamento" a ogni cluster PRIMA
  di avviare Locust. Dopo le prime 2-3 richieste riuscite, i pool gRPC sono warm
  e il test parte senza il burst iniziale di 500/504.

USAGE:
  python experiments/warmup_clusters.py

  Poi avviare Locust immediatamente dopo.
"""

import urllib.request
import urllib.error
import time
import concurrent.futures
import sys

CLUSTERS = {
    "cluster1": "http://192.168.1.245:30080",
    "cluster2": "http://192.168.1.246:30080",
    "cluster3": "http://192.168.1.247:30080",
}

WARMUP_ROUNDS = 5       # Numero di round di richieste
PAUSE_BETWEEN = 2.0     # Secondi tra un round e l'altro
TIMEOUT_SECS  = 35      # Timeout per richiesta (copre max 3 retry Nginx = 30s)


def ping_cluster(name: str, url: str) -> tuple[str, int, float]:
    """Invia una GET / e restituisce (cluster_name, status_code, latency_s)."""
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url + "/", headers={"User-Agent": "warmup/1.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
            _ = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    except Exception as e:
        status = 0  # connection error
    latency = time.monotonic() - t0
    return name, status, latency


def warmup():
    print("=" * 60)
    print("  Cluster warm-up — pre-riscaldamento pool gRPC")
    print("=" * 60)
    print(f"  Clusters: {', '.join(CLUSTERS.keys())}")
    print(f"  Round: {WARMUP_ROUNDS}  |  Pausa: {PAUSE_BETWEEN}s  |  Timeout: {TIMEOUT_SECS}s")
    print()

    all_ok = {name: False for name in CLUSTERS}

    for round_n in range(1, WARMUP_ROUNDS + 1):
        print(f"Round {round_n}/{WARMUP_ROUNDS}:", end="  ", flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(ping_cluster, name, url): name
                for name, url in CLUSTERS.items()
            }
            results = {}
            for fut in concurrent.futures.as_completed(futures):
                name, status, lat = fut.result()
                results[name] = (status, lat)

        for name in sorted(results):
            status, lat = results[name]
            icon = "[OK]" if status == 200 else ("[WARN]" if status in (500, 503) else "[ERR]")
            print(f"{icon} {name}: {status} ({lat:.2f}s)", end="  ", flush=True)
            if status == 200:
                all_ok[name] = True
        print()

        if round_n < WARMUP_ROUNDS:
            time.sleep(PAUSE_BETWEEN)

    print()
    print("=" * 60)

    failed = [n for n, ok in all_ok.items() if not ok]
    if failed:
        print(f"[WARN] Cluster non ancora warm: {', '.join(failed)}")
        print("   Attendi qualche secondo e riprova, o avvia il test comunque.")
        print("   Le prime 2-3 richieste Locust potrebbero essere lente (500/504).")
    else:
        print("[OK] Tutti i cluster warm - pool gRPC stabili.")
        print("   Avvia Locust ora per avere latenze nominali dal primo secondo.")
    print("=" * 60)
    print()

    return len(failed) == 0


if __name__ == "__main__":
    ok = warmup()
    sys.exit(0 if ok else 1)
