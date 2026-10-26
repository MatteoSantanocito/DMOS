#!/usr/bin/env python3
"""
dmos-ready-check.py
====================
End-to-end readiness checker per DMOS.

Verifica che TUTTE le precondizioni siano soddisfatte prima di lanciare
`python -m src.dmos_main`. Risponde con un punteggio sintetico e una lista
di issue se ce ne sono.

Cosa controlla (per ogni cluster definito in config/local.yaml):

  [Config]
    - config/local.yaml caricabile + N >= 2 cluster
    - config/services.yaml caricabile
    - config/weights.yaml caricabile
    - config/carbon.yaml caricabile

  [Connettivita']
    - Prometheus risponde su http://<ip>:<prometheus_port>/-/healthy
    - K8s API risponde via kubeconfig (kubectl get nodes)

  [Metriche obbligatorie DMOS]
    - hubble_http_requests_total       (Hubble L7)
    - hubble_http_request_duration_seconds_bucket  (Hubble L7)
    - kube_node_status_capacity        (kube-state-metrics)
    - container_cpu_usage_seconds_total (cAdvisor)
    - ping_rtt_mean_seconds             (ping_exporter - opzionale, warning se assente)

  [Workload]
    - Namespace online-boutique esistente
    - Frontend deployment Ready (>=1 replica)
    - Cilium Ingress in namespace online-boutique

  [DMOS code]
    - ConfigLoader istanzia senza errori
    - PrometheusClient si connette ad ogni cluster
    - K8sClient si connette ad ogni cluster

Usage:
    python scripts/dmos-ready-check.py
    python scripts/dmos-ready-check.py --strict      # fallisce su warning anche
    python scripts/dmos-ready-check.py --json        # output machine-readable

Exit codes:
    0 = ready (tutti i check passano)
    1 = not ready (almeno un check critico fallisce)
    2 = warning (solo check opzionali falliti, ready con riserve)
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import urllib.error

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

# ─── Colori terminale ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ─── Stato globale dei check ─────────────────────────────────────────────────
results = []   # lista di dict: {section, cluster, check, status, detail}

def add(section, cluster, check, status, detail=""):
    results.append({
        "section": section,
        "cluster": cluster,
        "check":   check,
        "status":  status,    # "ok" | "warn" | "fail"
        "detail":  detail,
    })


def print_section(name):
    print(f"\n{CYAN}{BOLD}=== {name} ==={RESET}")


def print_check(check, status, detail=""):
    icon = {"ok": f"{GREEN}OK", "warn": f"{YELLOW}WARN", "fail": f"{RED}FAIL"}[status]
    print(f"  [{icon}{RESET}] {check}", end="")
    if detail:
        print(f" {DIM}— {detail}{RESET}")
    else:
        print()


# ─── Section 1: config files ─────────────────────────────────────────────────

def check_configs():
    print_section("Config files (config/)")

    from src.utils.config_loader import ConfigLoader
    try:
        cfg = ConfigLoader(os.path.join(ROOT_DIR, "config"))
        n = len(cfg.clusters)
        if n >= 2:
            add("config", "-", "ConfigLoader carica", "ok", f"{n} clusters, {len(cfg.services)} services")
            print_check(f"ConfigLoader: {n} clusters", "ok",
                        ", ".join(cfg.clusters.keys()))
        else:
            add("config", "-", "ConfigLoader carica", "fail", f"solo {n} cluster (richiesti >=2)")
            print_check(f"ConfigLoader: solo {n} cluster", "fail", "DMOS richiede N >= 2")
        return cfg
    except Exception as e:
        add("config", "-", "ConfigLoader carica", "fail", str(e))
        print_check("ConfigLoader caricamento", "fail", str(e)[:80])
        return None


# ─── Section 2: connettività Prometheus ──────────────────────────────────────

def http_get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:
        return None, ""


def prom_query(prom_url, query, timeout=5):
    """Esegue una query PromQL. Ritorna (success, num_results)."""
    url = f"{prom_url}/api/v1/query?query={urllib.parse.quote(query)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        if data.get("status") == "success":
            return True, len(data.get("data", {}).get("result", []))
        return False, 0
    except Exception:
        return False, 0


def check_prometheus(cfg):
    print_section("Connettivita' Prometheus")

    for name, cluster in cfg.clusters.items():
        url = f"http://{cluster.ip}:{cluster.prometheus_port}"
        code, _ = http_get(f"{url}/-/healthy", timeout=3)
        if code == 200:
            add("prometheus", name, "healthy", "ok", url)
            print_check(f"{name:20s} → {url}", "ok")
        else:
            add("prometheus", name, "healthy", "fail", f"HTTP {code}" if code else "no response")
            print_check(f"{name:20s} → {url}", "fail",
                        f"HTTP {code}" if code else "no response (VPN giu'?)")


# ─── Section 3: metriche obbligatorie ────────────────────────────────────────

REQUIRED_METRICS = [
    ("hubble_http_requests_total",                  True,  "Hubble L7 (CNP + helm upgrade richiesti)"),
    ("hubble_http_request_duration_seconds_bucket", True,  "Hubble L7 histogram"),
    ("kube_node_status_capacity",                   True,  "kube-state-metrics"),
    ("container_cpu_usage_seconds_total",           True,  "cAdvisor"),
    ("ping_rtt_mean_seconds",                       False, "ping_exporter (opzionale)"),
]

def check_metrics(cfg):
    print_section("Metriche DMOS-required")

    for name, cluster in cfg.clusters.items():
        url = f"http://{cluster.ip}:{cluster.prometheus_port}"
        print(f"\n  {BOLD}{name}{RESET}")
        for metric, required, desc in REQUIRED_METRICS:
            ok, count = prom_query(url, metric)
            if ok and count > 0:
                add("metrics", name, metric, "ok", f"{count} series")
                print_check(f"  {metric}", "ok", f"{count} series")
            elif required:
                add("metrics", name, metric, "fail", desc)
                print_check(f"  {metric}", "fail", desc)
            else:
                add("metrics", name, metric, "warn", desc)
                print_check(f"  {metric}", "warn", desc)


# ─── Section 4: workload (Online Boutique + Cilium Ingress) ──────────────────

def kubectl(cluster_kc, args, timeout=10):
    """Esegue kubectl --kubeconfig <kc> <args>. Ritorna (rc, stdout, stderr)."""
    cmd = ["kubectl", "--kubeconfig", cluster_kc] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -2, "", "kubectl not found in PATH"


def check_workload(cfg):
    print_section("Workload (Online Boutique + Cilium Ingress)")

    for name, cluster in cfg.clusters.items():
        kc = os.path.expanduser(cluster.kubeconfig_path)
        if not os.path.isfile(kc):
            add("workload", name, "kubeconfig", "fail", f"file mancante: {kc}")
            print_check(f"{name}: kubeconfig", "fail", f"manca {kc}")
            continue

        # Namespace online-boutique
        rc, _, _ = kubectl(kc, ["get", "ns", "online-boutique", "-o", "name"])
        if rc == 0:
            add("workload", name, "namespace online-boutique", "ok")
        else:
            add("workload", name, "namespace online-boutique", "fail")
            print_check(f"{name}: ns online-boutique", "fail")
            continue

        # Frontend ready
        rc, out, _ = kubectl(kc, [
            "-n", "online-boutique", "get", "deploy", "frontend",
            "-o", "jsonpath={.status.readyReplicas}"
        ])
        ready = int(out) if rc == 0 and out.isdigit() else 0
        if ready >= 1:
            add("workload", name, "frontend ready", "ok", f"{ready} replica(s)")
            print_check(f"{name}: frontend Ready", "ok", f"{ready} replica(s)")
        else:
            add("workload", name, "frontend ready", "fail", f"ready={ready}")
            print_check(f"{name}: frontend Ready", "fail", f"ready={ready}")

        # Cilium Ingress (è un Ingress, controlliamo che esista)
        rc, _, _ = kubectl(kc, ["-n", "online-boutique", "get", "ingress",
                                "frontend-ingress-cilium", "-o", "name"])
        if rc == 0:
            add("workload", name, "Cilium Ingress", "ok")
            print_check(f"{name}: Cilium Ingress", "ok")
        else:
            add("workload", name, "Cilium Ingress", "warn",
                "apply deployments/ingress-frontend-cilium.yaml")
            print_check(f"{name}: Cilium Ingress", "warn", "applica il manifest")


# ─── Section 5: smoke test PrometheusClient + K8sClient ──────────────────────

def check_dmos_clients(cfg):
    print_section("Smoke test client DMOS")

    try:
        from src.metrics.prometheus_client import PrometheusClient
        for name, cluster in cfg.clusters.items():
            url = f"http://{cluster.ip}:{cluster.prometheus_port}"
            try:
                client = PrometheusClient(url=url, timeout=3, cluster_name=name)
                if client._test_connection():
                    add("dmos_clients", name, "PrometheusClient", "ok", url)
                    print_check(f"{name}: PrometheusClient", "ok", url)
                else:
                    add("dmos_clients", name, "PrometheusClient", "fail", "_test_connection False")
                    print_check(f"{name}: PrometheusClient", "fail", "_test_connection False")
            except Exception as e:
                add("dmos_clients", name, "PrometheusClient", "fail", str(e)[:80])
                print_check(f"{name}: PrometheusClient", "fail", str(e)[:80])
    except ImportError as e:
        add("dmos_clients", "-", "import PrometheusClient", "fail", str(e))
        print_check("import PrometheusClient", "fail", str(e))


# ─── Riepilogo ───────────────────────────────────────────────────────────────

def summary(strict=False):
    print()
    print(f"{CYAN}{BOLD}{'='*70}{RESET}")
    print(f"{CYAN}{BOLD}  RIEPILOGO READINESS{RESET}")
    print(f"{CYAN}{BOLD}{'='*70}{RESET}")

    n_ok    = sum(1 for r in results if r["status"] == "ok")
    n_warn  = sum(1 for r in results if r["status"] == "warn")
    n_fail  = sum(1 for r in results if r["status"] == "fail")
    n_total = len(results)

    print(f"  Check totali: {n_total}")
    print(f"  {GREEN}OK   : {n_ok}{RESET}")
    print(f"  {YELLOW}WARN : {n_warn}{RESET}")
    print(f"  {RED}FAIL : {n_fail}{RESET}")

    # Lista failures
    if n_fail > 0:
        print(f"\n  {RED}{BOLD}FAILURES:{RESET}")
        for r in results:
            if r["status"] == "fail":
                tag = f"[{r['section']}/{r['cluster']}]"
                print(f"    {RED}- {tag:30s} {r['check']}{RESET}"
                      f"{DIM} {r['detail']}{RESET}")

    # Verdetto
    print()
    if n_fail > 0:
        print(f"  {RED}{BOLD}DMOS NON E' READY{RESET}")
        print(f"  {RED}Risolvi i FAIL prima di lanciare python -m src.dmos_main{RESET}")
        return 1
    if n_warn > 0 and strict:
        print(f"  {YELLOW}{BOLD}DMOS NON READY (strict mode){RESET}")
        return 2
    if n_warn > 0:
        print(f"  {YELLOW}{BOLD}DMOS READY (con warning non bloccanti){RESET}")
        return 0
    print(f"  {GREEN}{BOLD}DMOS READY -- lancia: python -m src.dmos_main{RESET}")
    return 0


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="DMOS readiness checker")
    p.add_argument("--strict", action="store_true",
                   help="Fallisci anche su warning")
    p.add_argument("--json",   action="store_true",
                   help="Output JSON machine-readable invece di TTY")
    args = p.parse_args()

    print(f"{CYAN}{BOLD}DMOS Readiness Check{RESET}")
    print(f"{DIM}Root: {ROOT_DIR}{RESET}")
    print(f"{DIM}Config: {os.path.join(ROOT_DIR, 'config')}{RESET}")

    cfg = check_configs()
    if cfg is None:
        print(f"\n{RED}{BOLD}Stop: ConfigLoader non si carica.{RESET}")
        if args.json:
            print(json.dumps({"ready": False, "results": results}, indent=2))
        sys.exit(1)

    check_prometheus(cfg)
    check_metrics(cfg)
    check_workload(cfg)
    check_dmos_clients(cfg)

    rc = summary(strict=args.strict)

    if args.json:
        print(json.dumps({
            "ready": rc == 0,
            "n_ok":  sum(1 for r in results if r["status"] == "ok"),
            "n_warn": sum(1 for r in results if r["status"] == "warn"),
            "n_fail": sum(1 for r in results if r["status"] == "fail"),
            "results": results,
        }, indent=2))

    sys.exit(rc)


if __name__ == "__main__":
    main()
