#!/usr/bin/env python3
"""
ghz Backend Capacity Test — Per-Service gRPC Isolation (Python version)
========================================================================

Scopo: misurare la capacita reale (rps/replica) di ciascun backend gRPC,
       con la stessa metodologia stepped-load + knee detection usata da
       k6_capacity_test.js per il frontend HTTP.

Cosa fa ghz:
  ghz e' un tool di benchmarking gRPC — l'equivalente di k6/Locust ma per
  servizi gRPC. Manda richieste gRPC direttamente a un singolo servizio,
  bypassando il frontend. In pratica ghz SOSTITUISCE il frontend come
  generatore di richieste: invece di HTTP->frontend->gRPC->backend,
  il flusso diventa ghz->gRPC->backend.

Dipendenze e call graph:
  IMPORTANTE: ghz NON isola il servizio dalle sue dipendenze downstream.
  Quando ghz chiama un servizio, quel servizio chiama comunque tutti i
  servizi da cui dipende. Questo e' il comportamento CORRETTO: la capacity
  misurata include il tempo speso nelle dipendenze, esattamente come
  accade nel sistema reale.

  Call graph reale di Online Boutique:

    k6 HTTP -> frontend -> cartservice -----------> redis-cart
                         -> productcatalogservice   (in-memory, ZERO dipendenze)
                         -> recommendationservice -> productcatalogservice
                         -> checkoutservice -------> cartservice -> redis-cart
                                                  -> productcatalogservice
                                                  -> currencyservice
                                                  -> shippingservice
                                                  -> paymentservice
                                                  -> emailservice

  Cosa misura ghz per ogni servizio:

    Servizio                 ghz chiama...            Dipendenze colpite
    -----------------------------------------------------------------------
    cartservice              GetCart                  solo redis-cart (locale)
    productcatalogservice    GetProduct               NESSUNA (in-memory Go)
    recommendationservice    ListRecommendations      productcatalogservice
    checkoutservice          PlaceOrder               cart + catalog + currency
                                                      + shipping + payment + email

  Quindi:
    - productcatalogservice: misura quasi pura (nessuna dipendenza)
    - cartservice: dipende solo da redis-cart (locale, veloce)
    - recommendationservice: dipende da productcatalogservice (veloce)
    - checkoutservice: fan-out completo (6 servizi) — il path piu pesante

Setup critico per misure corrette:
  - Il servizio TARGET deve avere esattamente 1 replica (quello che misuri)
  - Le DIPENDENZE devono essere attive e NON saturate, altrimenti il
    bottleneck e' la dipendenza, non il servizio target
  - I servizi non scalati da DMOS (redis-cart, currencyservice,
    shippingservice, paymentservice, emailservice) hanno gia repliche
    fisse — tipicamente non sono un problema
  - DMOS deve essere FERMO durante il test

Perche Python e non bash:
  - Funziona su Windows (dove il progetto gira)
  - Parsing JSON robusto (ghz output)
  - Genera grafici automatici (matplotlib)
  - Integrazione con lo stesso output format di locustfile_capacity.py

Prerequisiti:
  1. ghz installato e nel PATH
     Windows: scaricare da https://github.com/bojand/ghz/releases
     Linux/Mac: go install github.com/bojand/ghz/cmd/ghz@latest
  2. Proto file: scaricato automaticamente da Google microservices-demo
  3. kubectl configurato con context per i 3 cluster
  4. Port-forward attivo O accesso diretto ai pod gRPC

Uso:
  # 1. Ferma DMOS

  # 2. Scala il servizio target a 1 replica, verifica dipendenze attive:
  kubectl scale deployment cartservice -n online-boutique --replicas=1 --context cluster1
  kubectl get pods -n online-boutique --context cluster1  # verifica redis-cart running

  # 3. Testa tutti i backend su cluster1:
  python experiments/ghz_backend_capacity.py --cluster cluster1

  # 4. Testa solo cartservice:
  python experiments/ghz_backend_capacity.py --cluster cluster1 --service cartservice

  # 5. Con parametri personalizzati:
  python experiments/ghz_backend_capacity.py --cluster cluster1 --step-duration 90 --max-rate 200

  # 6. Ripeti per cluster2 e cluster3 per verificare l'effetto netem
  #    sui backend gRPC intra-cluster

Output:
  results/capacity/ghz_{service}_{cluster}_{timestamp}.csv
  results/capacity/ghz_{service}_{cluster}_{timestamp}.json
  results/capacity/ghz_{service}_{cluster}_{timestamp}.png  (se matplotlib)

Riferimenti scientifici:
  - Gan et al. (ASPLOS 2019): microservice dependency-aware profiling
  - Sriraman & Dhanotia (ASPLOS 2020): end-to-end tail latency in microservices
  - Universal Scalability Law (Gunther): throughput saturation under contention
"""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Configurazione servizi backend
# ============================================================================

# Cluster IPs read from .cache/test-env.json (rendered from config/local.yaml).
import json as _json
import os as _os
_env_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", ".cache", "test-env.json"
)
try:
    with open(_env_path, "r", encoding="utf-8") as _f:
        _env = _json.load(_f)
except FileNotFoundError:
    raise SystemExit(
        f"{_env_path} not found. Run 'bash scripts/render-runtime-files.sh' "
        "first (it generates this file from config/local.yaml)."
    )
CLUSTER_IPS = {c["name"]: c["ip"] for c in _env["clusters"]}

# Servizi gRPC e i rispettivi metodi RPC + payload di test.
#
# Le RPC scelte riflettono il workload reale del frontend:
#   - CartService: GetCart e' la piu chiamata (~25% del traffico frontend)
#   - ProductCatalogService: GetProduct (~65% del traffico frontend)
#   - CheckoutService: PlaceOrder (solo ~5% ma path critico con fan-out)
#   - RecommendationService: ListRecommendations (~25% del traffico)
#
# ORDINE DI TEST CONSIGLIATO:
#   1. productcatalogservice  (zero dipendenze — misura pura)
#   2. cartservice            (dipende solo da redis-cart locale)
#   3. recommendationservice  (dipende da productcatalogservice, gia testato)
#   4. checkoutservice        (fan-out completo — testare per ultimo)
#
# Questo ordine garantisce che le dipendenze downstream siano gia state
# verificate prima di testare i servizi che le chiamano.
BACKEND_SERVICES = {
    "cartservice": {
        "port": 7070,
        "rpc": "hipstershop.CartService/GetCart",
        "data": {"user_id": "capacity-test-user"},
        "sla_p95_ms": 50,
        "estimated_capacity": 50,
        # Dipendenze: redis-cart (locale, non scalato da DMOS)
        # ghz -> cartservice -> redis-cart
        # Misura: capacita gRPC + Redis read latency (~1ms)
        "dependencies": ["redis-cart"],
        "notes": "gRPC read su Redis locale. Dipende solo da redis-cart (non scalato da DMOS, sempre 1 replica).",
    },
    "productcatalogservice": {
        "port": 3550,
        "rpc": "hipstershop.ProductCatalogService/GetProduct",
        "data": {"id": "OLJCESPC7Z"},
        "sla_p95_ms": 75,
        "estimated_capacity": 80,
        # Dipendenze: NESSUNA — catalogo in-memory Go
        # ghz -> productcatalogservice (risponde direttamente)
        # Misura: capacita pura del servizio Go senza overhead di rete
        "dependencies": [],
        "notes": "ZERO dipendenze. gRPC read-only Go, catalogo in-memory. Misura piu pulita.",
    },
    "checkoutservice": {
        "port": 5050,
        "rpc": "hipstershop.CheckoutService/PlaceOrder",
        "data": {
            "user_id": "capacity-test",
            "user_currency": "USD",
            "address": {
                "street_address": "123 Test St",
                "city": "New York",
                "state": "NY",
                "country": "US",
                "zip_code": "10001",
            },
            "email": "test@example.com",
            "credit_card": {
                "credit_card_number": "4432801561520454",
                "credit_card_cvv": 672,
                "credit_card_expiration_year": 2030,
                "credit_card_expiration_month": 1,
            },
        },
        "sla_p95_ms": 200,
        "estimated_capacity": 20,
        # Dipendenze: fan-out completo (6 servizi downstream)
        # ghz -> checkoutservice -> cartservice      -> redis-cart
        #                        -> productcatalog
        #                        -> currencyservice   (non scalato da DMOS)
        #                        -> shippingservice   (non scalato da DMOS)
        #                        -> paymentservice    (non scalato da DMOS)
        #                        -> emailservice      (non scalato da DMOS)
        # Misura: capacita del path piu pesante dell'applicazione.
        # ATTENZIONE: se una dipendenza e' satura, il bottleneck e' la
        # dipendenza, non il checkoutservice stesso.
        "dependencies": [
            "cartservice", "productcatalogservice",
            "currencyservice", "shippingservice",
            "paymentservice", "emailservice",
        ],
        "notes": "Fan-out completo: 6 dipendenze. Il path piu pesante. Testare per ULTIMO.",
    },
    "recommendationservice": {
        "port": 8080,
        "rpc": "hipstershop.RecommendationService/ListRecommendations",
        "data": {"user_id": "capacity-test", "product_ids": ["OLJCESPC7Z"]},
        "sla_p95_ms": 100,
        "estimated_capacity": 40,
        # Dipendenze: productcatalogservice (per ottenere il catalogo prodotti)
        # ghz -> recommendationservice -> productcatalogservice
        # Misura: capacita Python gRPC + overhead della call a catalog
        "dependencies": ["productcatalogservice"],
        "notes": "Python gRPC. Chiama productcatalogservice per il catalogo. Piu lento di Go.",
    },
}

PROTO_URL = "https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/protos/demo.proto"
PROTO_DIR = Path("experiments/protos")
PROTO_FILE = PROTO_DIR / "demo.proto"

SAFETY_MARGIN = 0.80  # Capacita raccomandata = 80% del knee point


# ============================================================================
# Utility
# ============================================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def download_proto():
    """Scarica demo.proto se non presente."""
    if PROTO_FILE.exists():
        return

    PROTO_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Download proto file da {PROTO_URL}...")

    try:
        import urllib.request
        urllib.request.urlretrieve(PROTO_URL, str(PROTO_FILE))
        log(f"Proto file salvato: {PROTO_FILE}")
    except Exception as e:
        print(f"ERRORE: impossibile scaricare proto file: {e}")
        print(f"Scarica manualmente: {PROTO_URL} -> {PROTO_FILE}")
        sys.exit(1)


def check_ghz():
    """Verifica che ghz sia installato."""
    try:
        result = subprocess.run(
            ["ghz", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip() or result.stderr.strip()
        log(f"ghz trovato: {version}")
        return True
    except FileNotFoundError:
        print()
        print("=" * 60)
        print("  ERRORE: ghz non trovato nel PATH")
        print("=" * 60)
        print()
        print("  ghz e' il tool standard per gRPC benchmarking.")
        print("  Installazione:")
        print()
        print("  Windows:")
        print("    1. Scarica da: https://github.com/bojand/ghz/releases")
        print("    2. Estrai ghz.exe in una cartella nel PATH")
        print()
        print("  Linux/Mac:")
        print("    go install github.com/bojand/ghz/cmd/ghz@latest")
        print()
        sys.exit(1)


def start_port_forward(
    service: str, port: int, cluster: str
) -> Optional[subprocess.Popen]:
    """Avvia kubectl port-forward in background."""
    cmd = [
        "kubectl", "port-forward",
        f"svc/{service}", f"{port}:{port}",
        "-n", "online-boutique",
        "--context", cluster,
    ]
    log(f"Port-forward: {service}:{port} -> localhost:{port}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # Attendi stabilizzazione
        if proc.poll() is not None:
            log(f"ERRORE: port-forward terminato con codice {proc.returncode}")
            return None
        return proc
    except Exception as e:
        log(f"ERRORE avvio port-forward: {e}")
        return None


def stop_port_forward(proc: Optional[subprocess.Popen]):
    """Termina il processo di port-forward."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_ghz_step(
    rpc: str, data: dict, port: int, rate: int, duration: int
) -> Optional[Dict]:
    """
    Esegue un singolo step di ghz e restituisce i risultati parsed.

    Parametri ghz:
      --rps: target request rate
      --duration: durata dello step
      --connections: connessioni gRPC (pool)
      --concurrency: worker concorrenti
      --format json: output JSON per parsing
    """
    data_json = json.dumps(data)

    cmd = [
        "ghz",
        "--proto", str(PROTO_FILE),
        "--call", rpc,
        "--data", data_json,
        "--rps", str(rate),
        "--duration", f"{duration}s",
        "--connections", "50",
        "--concurrency", "100",
        "--format", "json",
        f"localhost:{port}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 30,  # timeout con margine
        )

        if not result.stdout.strip():
            log(f"WARNING: ghz output vuoto (stderr: {result.stderr[:200]})")
            return None

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        log(f"WARNING: ghz timeout dopo {duration + 30}s")
        return None
    except json.JSONDecodeError as e:
        log(f"WARNING: ghz output non-JSON: {e}")
        return None
    except Exception as e:
        log(f"WARNING: ghz errore: {e}")
        return None


def parse_ghz_result(ghz_data: Dict) -> Dict:
    """
    Estrae metriche da ghz JSON output.

    ghz JSON format:
      {
        "count": N,
        "total": nanoseconds,
        "average": nanoseconds,
        "fastest": nanoseconds,
        "slowest": nanoseconds,
        "rps": float,
        "errorDistribution": {"rpc error: ...": N},
        "statusCodeDistribution": {"OK": N, "Unavailable": M},
        "latencyDistribution": [
          {"percentage": 10, "latency": nanoseconds},
          {"percentage": 25, "latency": nanoseconds},
          {"percentage": 50, "latency": nanoseconds},
          {"percentage": 75, "latency": nanoseconds},
          {"percentage": 90, "latency": nanoseconds},
          {"percentage": 95, "latency": nanoseconds},
          {"percentage": 99, "latency": nanoseconds}
        ]
      }
    """
    total_count = ghz_data.get("count", 0)
    rps = ghz_data.get("rps", 0)

    # Errori: conta tutte le risposte non-OK
    status_dist = ghz_data.get("statusCodeDistribution", {})
    ok_count = status_dist.get("OK", 0)
    error_count = total_count - ok_count
    error_rate = error_count / total_count if total_count > 0 else 0

    # Percentili latenza (nanoseconds -> milliseconds)
    latency_dist = ghz_data.get("latencyDistribution", [])
    percentiles = {}
    for entry in latency_dist:
        pct = entry.get("percentage", 0)
        lat_ns = entry.get("latency", 0)
        percentiles[pct] = lat_ns / 1e6  # ns -> ms

    return {
        "total_count": total_count,
        "rps": round(rps, 1),
        "error_count": error_count,
        "error_rate": round(error_rate, 4),
        "p50_ms": round(percentiles.get(50, 0), 1),
        "p90_ms": round(percentiles.get(90, 0), 1),
        "p95_ms": round(percentiles.get(95, 0), 1),
        "p99_ms": round(percentiles.get(99, 0), 1),
        "avg_ms": round(ghz_data.get("average", 0) / 1e6, 1),
    }


# ============================================================================
# Test singolo servizio
# ============================================================================

def test_service(
    service_name: str,
    cluster: str,
    step_duration: int,
    step_increment: int,
    max_rate: int,
    warmup_duration: int,
    output_dir: str,
) -> Optional[Dict]:
    """Esegue il capacity test per un singolo servizio."""

    svc_config = BACKEND_SERVICES[service_name]
    port = svc_config["port"]
    rpc = svc_config["rpc"]
    data = svc_config["data"]
    sla_p95 = svc_config["sla_p95_ms"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    deps = svc_config.get("dependencies", [])
    deps_str = ", ".join(deps) if deps else "NESSUNA (misura pura)"

    print()
    print("=" * 70)
    print(f"  CAPACITY TEST: {service_name} @ {cluster}")
    print("=" * 70)
    print(f"  RPC:           {rpc}")
    print(f"  Porta gRPC:    {port}")
    print(f"  SLA p95:       {sla_p95}ms")
    print(f"  Step:          {step_increment} rps x {step_duration}s")
    print(f"  Max rate:      {max_rate} rps")
    print(f"  Dipendenze:    {deps_str}")
    print(f"  Nota:          {svc_config['notes']}")
    print("=" * 70)
    if deps:
        print(f"  Le dipendenze [{', '.join(deps)}] verranno colpite")
        print(f"  durante il test. Assicurati che siano attive e non saturate.")
    print()

    # Port-forward
    pf_proc = start_port_forward(service_name, port, cluster)
    if pf_proc is None:
        log("ERRORE: impossibile avviare port-forward, skip servizio")
        return None

    try:
        # Warmup
        if warmup_duration > 0:
            log(f"Warmup: {warmup_duration}s a {step_increment} rps (non misurato)...")
            run_ghz_step(rpc, data, port, step_increment, warmup_duration)

        # Header tabella
        print(
            f"  {'Step':>4s} | {'Rate':>6s} | {'Actual':>8s} | "
            f"{'p50':>7s} | {'p95':>7s} | {'p99':>7s} | "
            f"{'Err%':>5s} | SLA"
        )
        print("  " + "-" * 68)

        results = []
        knee_rps = 0.0
        knee_rate = 0
        consecutive_fail = 0
        step = 0

        for rate in range(step_increment, max_rate + 1, step_increment):
            step += 1

            # Esegui step
            ghz_raw = run_ghz_step(rpc, data, port, rate, step_duration)
            if ghz_raw is None:
                log(f"Step {step}: ghz fallito, skip")
                continue

            metrics = parse_ghz_result(ghz_raw)
            metrics["step"] = step
            metrics["target_rate"] = rate

            # Valuta SLA
            sla_ok = (
                metrics["p95_ms"] < sla_p95
                and metrics["error_rate"] < 0.02
            )
            metrics["sla_ok"] = sla_ok

            if sla_ok:
                consecutive_fail = 0
                knee_rps = metrics["rps"]
                knee_rate = rate
            else:
                consecutive_fail += 1

            sla_str = " OK " if sla_ok else "FAIL"
            if not sla_ok and consecutive_fail < 2:
                sla_str = "WARN"

            print(
                f"  {step:4d} | {rate:4d}r/s | {metrics['rps']:6.1f}r/s | "
                f"{metrics['p50_ms']:5.0f}ms | {metrics['p95_ms']:5.0f}ms | "
                f"{metrics['p99_ms']:5.0f}ms | "
                f"{metrics['error_rate']*100:4.1f}% | {sla_str}"
            )

            results.append(metrics)

            # Stop se SLA violato per 2 step consecutivi
            if consecutive_fail >= 2:
                print()
                log(f"SLA violato per {consecutive_fail} step consecutivi "
                    f"-> saturazione confermata")
                break

    finally:
        stop_port_forward(pf_proc)

    # Report finale
    recommended = max(1, round(knee_rps * SAFETY_MARGIN)) if knee_rps > 0 else None
    estimated = svc_config["estimated_capacity"]

    print()
    print("=" * 70)
    if knee_rps > 0:
        delta_pct = (knee_rps - estimated) / estimated * 100
        print(f"  Knee point:            {knee_rps:.1f} rps/replica "
              f"(a {knee_rate} rps target)")
        print(f"  Capacita raccomandata: {recommended} rps/replica "
              f"(80% del knee)")
        print(f"  Stima corrente:        {estimated} rps/replica "
              f"(delta: {delta_pct:+.1f}%)")
        print()
        print(f"  -> Aggiorna config/services.yaml:")
        print(f"     {service_name}:")
        print(f"       capacity_req_per_sec: {recommended}")
    else:
        print(f"  Knee point: N/A - SLA violato gia al primo step")
        print(f"  Il servizio potrebbe essere sovraccarico o non raggiungibile")
    print("=" * 70)

    # Salva risultati
    summary = {
        "service": service_name,
        "cluster": cluster,
        "rpc": rpc,
        "sla_p95_ms": sla_p95,
        "step_duration_s": step_duration,
        "step_increment_rps": step_increment,
        "knee_rps": knee_rps,
        "knee_rate": knee_rate,
        "recommended_capacity": recommended,
        "current_estimated": estimated,
        "timestamp": timestamp,
        "steps": results,
    }

    # JSON
    json_path = out_path / f"ghz_{service_name}_{cluster}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    csv_path = out_path / f"ghz_{service_name}_{cluster}_{timestamp}.csv"
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Grafico
    chart_path = generate_chart(results, service_name, cluster, sla_p95,
                                knee_rps, knee_rate, out_path, timestamp)

    log(f"JSON: {json_path}")
    log(f"CSV:  {csv_path}")
    if chart_path:
        log(f"Chart: {chart_path}")

    return summary


def generate_chart(
    results: List[Dict],
    service_name: str,
    cluster: str,
    sla_p95: int,
    knee_rps: float,
    knee_rate: int,
    output_dir: Path,
    timestamp: str,
) -> Optional[Path]:
    """Genera un grafico latenza + throughput (come locustfile_capacity.py)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not results:
        return None

    rates = [r["target_rate"] for r in results]
    actual_rps = [r["rps"] for r in results]
    p50 = [r["p50_ms"] for r in results]
    p95 = [r["p95_ms"] for r in results]
    p99 = [r["p99_ms"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"gRPC Capacity Test - {service_name} @ {cluster}",
        fontsize=14,
        fontweight="bold",
    )

    # Subplot 1: Latenza
    ax1.plot(rates, p50, "o-", color="#2196F3", linewidth=2, markersize=4, label="p50")
    ax1.plot(rates, p95, "s-", color="#FF9800", linewidth=2, markersize=4, label="p95")
    ax1.plot(rates, p99, "^-", color="#F44336", linewidth=1.5, markersize=4,
             label="p99", alpha=0.7)
    ax1.axhline(sla_p95, color="#F44336", linestyle="--", linewidth=1.5,
                label=f"SLA p95 = {sla_p95}ms")

    if knee_rate > 0:
        ax1.axvline(knee_rate, color="#9C27B0", linestyle=":", linewidth=2,
                    label=f"Knee ({knee_rate} rps)")

    ax1.set_ylabel("Latenza (ms)", fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Latenza per percentile", fontsize=11)

    # Subplot 2: Throughput
    ax2.plot(rates, actual_rps, "o-", color="#4CAF50", linewidth=2, markersize=4,
             label="Actual rps")
    ax2.plot(rates, rates, "--", color="#9E9E9E", linewidth=1, alpha=0.5,
             label="Ideal (rate=throughput)")

    if knee_rps > 0:
        ax2.axhline(knee_rps, color="#9C27B0", linestyle="--", linewidth=1,
                    alpha=0.6, label=f"Knee = {knee_rps:.1f} rps")
        recommended = max(1, round(knee_rps * SAFETY_MARGIN))
        ax2.axhline(recommended, color="#FF9800", linestyle="--", linewidth=1.5,
                    label=f"Recommended = {recommended} rps")

    ax2.set_xlabel("Target Rate (rps)", fontsize=11)
    ax2.set_ylabel("Actual Throughput (rps)", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Throughput vs Target Rate", fontsize=11)

    plt.tight_layout()
    chart_path = output_dir / f"ghz_{service_name}_{cluster}_{timestamp}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return chart_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ghz Backend Capacity Test - misura capacita gRPC isolata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Testa tutti i backend su cluster1
  python experiments/ghz_backend_capacity.py --cluster cluster1

  # Solo cartservice con step di 90s
  python experiments/ghz_backend_capacity.py --cluster cluster1 --service cartservice --step-duration 90

  # Test conservativo con step piccoli
  python experiments/ghz_backend_capacity.py --cluster cluster1 --step-increment 5 --max-rate 100
        """,
    )
    parser.add_argument(
        "--cluster",
        choices=list(CLUSTER_IPS.keys()),
        default="cluster1",
        help="Cluster target (default: cluster1)",
    )
    parser.add_argument(
        "--service",
        choices=list(BACKEND_SERVICES.keys()) + ["all"],
        default="all",
        help="Servizio da testare (default: all)",
    )
    parser.add_argument(
        "--step-duration",
        type=int,
        default=60,
        help="Durata di ogni step in secondi (default: 60)",
    )
    parser.add_argument(
        "--step-increment",
        type=int,
        default=10,
        help="Incremento rps per step (default: 10)",
    )
    parser.add_argument(
        "--max-rate",
        type=int,
        default=300,
        help="Rate massima da testare (default: 300)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Durata warmup in secondi (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/capacity",
        help="Directory output (default: results/capacity)",
    )
    args = parser.parse_args()

    # Prerequisiti
    check_ghz()
    download_proto()

    # Banner
    print()
    print("=" * 70)
    print("  ghz BACKEND CAPACITY TEST")
    print("=" * 70)
    print(f"  Cluster:     {args.cluster} ({CLUSTER_IPS[args.cluster]})")
    print(f"  Servizi:     {args.service}")
    print(f"  Step:        {args.step_increment} rps x {args.step_duration}s")
    print(f"  Max rate:    {args.max_rate} rps")
    print(f"  Output:      {args.output_dir}")
    print("=" * 70)
    print()
    print("  PREREQUISITI:")
    print("    1. DMOS FERMO")
    print("    2. Servizio target: esattamente 1 replica sul cluster")
    print("    3. Dipendenze downstream ATTIVE e NON SATURATE:")
    print("       - redis-cart:          sempre 1 replica (non scalato da DMOS)")
    print("       - currencyservice:     sempre 1 replica (non scalato da DMOS)")
    print("       - shippingservice:     sempre 1 replica (non scalato da DMOS)")
    print("       - paymentservice:      sempre 1 replica (non scalato da DMOS)")
    print("       - emailservice:        sempre 1 replica (non scalato da DMOS)")
    print("       - productcatalogservice: 1 replica (scalato da DMOS, fissare)")
    print("       - cartservice:         1 replica (scalato da DMOS, fissare)")
    print()
    print("  NOTA: ghz NON isola il servizio dalle dipendenze. Quando")
    print("  chiami checkoutservice/PlaceOrder, il checkout chiama a sua")
    print("  volta cart, catalog, currency, shipping, payment, email.")
    print("  Questo e' CORRETTO: la capacity misurata include il tempo")
    print("  delle dipendenze, come nel sistema reale.")
    print()
    if args.service == "all":
        print("  ORDINE DI TEST: productcatalog -> cart -> recommendation -> checkout")
        print("  (dalle zero dipendenze al fan-out completo)")
        print()

    # Seleziona servizi.
    # L'ordine per "all" segue le dipendenze: prima i servizi senza
    # dipendenze, poi quelli che dipendono da servizi gia testati.
    RECOMMENDED_ORDER = [
        "productcatalogservice",   # 1. zero dipendenze — misura pura
        "cartservice",             # 2. dipende solo da redis-cart (locale)
        "recommendationservice",   # 3. dipende da productcatalog (gia testato)
        "checkoutservice",         # 4. fan-out completo — per ultimo
    ]
    if args.service == "all":
        services = RECOMMENDED_ORDER
    else:
        services = [args.service]

    # Esegui test
    all_summaries = []
    for svc_name in services:
        summary = test_service(
            service_name=svc_name,
            cluster=args.cluster,
            step_duration=args.step_duration,
            step_increment=args.step_increment,
            max_rate=args.max_rate,
            warmup_duration=args.warmup,
            output_dir=args.output_dir,
        )
        if summary:
            all_summaries.append(summary)
        print()

    # Report aggregato
    if len(all_summaries) > 1:
        print()
        print("=" * 70)
        print("  SUMMARY — Tutti i servizi")
        print("=" * 70)
        print(
            f"  {'Service':<25s} | {'Knee rps':>8s} | {'Recommended':>11s} | "
            f"{'Estimated':>9s} | {'Delta':>7s}"
        )
        print("  " + "-" * 68)

        for s in all_summaries:
            knee = s["knee_rps"]
            rec = s["recommended_capacity"]
            est = s["current_estimated"]
            delta = (
                f"{(knee - est) / est * 100:+.1f}%"
                if knee > 0 and est > 0
                else "N/A"
            )
            print(
                f"  {s['service']:<25s} | {knee:8.1f} | "
                f"{rec if rec else 'N/A':>11} | {est:9d} | {delta:>7s}"
            )

        print("=" * 70)

        # Suggerimento services.yaml
        print()
        print("  Aggiorna config/services.yaml con i valori misurati:")
        for s in all_summaries:
            if s["recommended_capacity"]:
                print(f"    {s['service']}:")
                print(f"      capacity_req_per_sec: {s['recommended_capacity']}")
        print()


if __name__ == "__main__":
    main()
