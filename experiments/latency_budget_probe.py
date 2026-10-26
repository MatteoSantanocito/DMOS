#!/usr/bin/env python3
"""
latency_budget_probe.py — decomposizione per-hop della latenza su UN cluster
(default cluster-01, il cluster caldo a `con`), via le histogram Hubble L7.

Per ogni edge source->destination calcola la p95 (histogram_quantile sui bucket)
e l'rps, su una finestra [WINDOW]. Da lanciare DURANTE il picco di un run con.
Identifica DOVE si perde il tempo end-to-end: frontend processing vs ciascun
downstream gRPC, incluso il fanout recommendation->productcatalog.

Uso: python experiments/latency_budget_probe.py [--prom URL] [--window 2m]

Il Prometheus target: --prom, altrimenti env DMOS_PROM_URL, altrimenti il
prometheus_url del primo cluster in .cache/test-env.json (cluster-01).
"""
import argparse, json, os, urllib.parse, urllib.request
from pathlib import Path


def default_prom():
    """Default Prometheus URL: env DMOS_PROM_URL, poi .cache/test-env.json."""
    env = os.environ.get("DMOS_PROM_URL")
    if env:
        return env
    env_json = Path(__file__).resolve().parent.parent / ".cache" / "test-env.json"
    try:
        return json.loads(env_json.read_text())["clusters"][0]["prometheus_url"]
    except Exception:
        return None

# edge = (source_workload, destination_workload). source="" → qualsiasi (ingress).
EDGES = [
    ("",                      "frontend"),               # end-to-end (frontend serving)
    ("frontend",              "productcatalogservice"),
    ("frontend",              "recommendationservice"),
    ("frontend",              "cartservice"),
    ("frontend",              "currencyservice"),
    ("frontend",              "adservice"),
    ("frontend",              "checkoutservice"),
    ("frontend",              "shippingservice"),
    ("recommendationservice", "productcatalogservice"),  # fanout di recommendation
    ("checkoutservice",       "productcatalogservice"),
]

def q(prom, expr):
    url = prom.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.load(r)
        res = d.get("data", {}).get("result", [])
        return float(res[0]["value"][1]) if res else float("nan")
    except Exception:
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom", default=None,
                    help="URL Prometheus (default: DMOS_PROM_URL o .cache/test-env.json)")
    ap.add_argument("--window", default="2m")
    args = ap.parse_args()
    if not args.prom:
        args.prom = default_prom()
    if not args.prom:
        ap.error("URL Prometheus non determinato: passa --prom URL, imposta "
                 "DMOS_PROM_URL, o genera .cache/test-env.json "
                 "(scripts/render-runtime-files.sh)")
    w = args.window
    print(f"# Latency budget per-hop @ {args.prom}  (finestra {w}, reporter=server)")
    print(f"{'edge':<42}{'p95 (ms)':>10}{'rps':>9}")
    print("-" * 61)
    for src, dst in EDGES:
        sel = f'destination_workload="{dst}",reporter="server"'
        if src:
            sel += f',source_workload="{src}"'
        p95 = q(args.prom, f'histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{{{sel}}}[{w}])) by (le)) * 1000')
        rps = q(args.prom, f'sum(rate(hubble_http_requests_total{{{sel}}}[{w}]))')
        label = f"{src or 'INGRESS'} -> {dst}"
        print(f"{label:<42}{p95:>10.0f}{rps:>9.0f}")

if __name__ == "__main__":
    main()
