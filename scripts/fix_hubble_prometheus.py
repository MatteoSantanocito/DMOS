#!/usr/bin/env python3
"""
fix_hubble_prometheus.py
========================
Diagnostica PERCHÉ Prometheus non scrape le metriche Hubble e applica la fix.

Controlla:
  1. Prometheus targets — hubble è già presente come target?
  2. Prometheus Operator config — serviceMonitorSelector + serviceMonitorNamespaceSelector
  3. hubble-metrics Service labels — per scrivere il ServiceMonitor corretto
  4. ServiceMonitor 'hubble' in kube-system — porta e selector corretti?

Applica automaticamente la fix se richiesto (--fix).

Usage:
  python scripts/fix_hubble_prometheus.py
  python scripts/fix_hubble_prometheus.py --fix
  python scripts/fix_hubble_prometheus.py --fix --clusters cluster1
"""

import subprocess
import json
import sys
import os
import argparse
import tempfile
import time
import urllib.request
import urllib.parse
import urllib.error

# ── Config ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTERS = ["cluster1", "cluster2", "cluster3"]
CLUSTER_IPS = {
    "cluster1": "192.168.1.245",
    "cluster2": "192.168.1.246",
    "cluster3": "192.168.1.247",
}
PROM_PORT = 30090

# ── Colori ────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"  {RED}❌ {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"  {CYAN}ℹ️  {msg}{RESET}")
def section(msg): print(f"\n  {BOLD}{msg}{RESET}")


# ── kubectl helper ─────────────────────────────────────────────────────────────
def kubectl(args: list, cluster: str) -> tuple:
    """Esegue kubectl con il kubeconfig del cluster dato. Ritorna (rc, stdout, stderr)."""
    kube = os.path.join(ROOT_DIR, ".kube", f"{cluster}.yaml")
    cmd = ["kubectl", "--kubeconfig", kube] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def kubectl_json(args: list, cluster: str):
    """kubectl … -o json → dict/list o None se errore."""
    rc, out, err = kubectl(args + ["-o", "json"], cluster)
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


# ── HTTP helpers ───────────────────────────────────────────────────────────────
def http_get_json(url: str, timeout: int = 5):
    """GET JSON da URL. Ritorna dict o None."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except Exception:
        return None


def prom_query(cluster: str, query: str) -> list:
    """Interroga Prometheus via NodePort. Ritorna list di result o []."""
    ip = CLUSTER_IPS[cluster]
    encoded = urllib.parse.quote(query)
    url = f"http://{ip}:{PROM_PORT}/api/v1/query?query={encoded}"
    data = http_get_json(url)
    if data:
        return data.get("data", {}).get("result", [])
    return []


def prom_targets(cluster: str) -> dict:
    """Ritorna i target Prometheus attivi e droppati."""
    ip = CLUSTER_IPS[cluster]
    url = f"http://{ip}:{PROM_PORT}/api/v1/targets"
    data = http_get_json(url)
    if data:
        return data.get("data", {})
    return {}


# ── Sezione 1: Targets Prometheus ─────────────────────────────────────────────
def check_prometheus_targets(cluster: str) -> bool:
    """Controlla se Prometheus ha già un target hubble attivo."""
    section("[1] Prometheus targets — hubble presente?")

    targets_data = prom_targets(cluster)
    if not targets_data:
        fail(f"Non riesco a raggiungere Prometheus su {CLUSTER_IPS[cluster]}:{PROM_PORT}")
        return False

    active  = targets_data.get("activeTargets", [])
    dropped = targets_data.get("droppedTargets", [])

    def is_hubble(t):
        url = t.get("scrapeUrl", "") + t.get("scrapePool", "")
        labels = str(t.get("labels", {})) + str(t.get("discoveredLabels", {}))
        return "hubble" in url.lower() or "hubble" in labels.lower() or "9965" in url

    hubble_active  = [t for t in active  if is_hubble(t)]
    hubble_dropped = [t for t in dropped if is_hubble(t)]

    if hubble_active:
        ok(f"Hubble target ATTIVO: {len(hubble_active)} endpoint")
        all_up = True
        for t in hubble_active[:5]:
            health = t.get("health", "?")
            url    = t.get("scrapeUrl", "?")
            color  = GREEN if health == "up" else RED
            print(f"    {color}{health}{RESET} → {url}")
            if health != "up":
                warn(f"LastError: {t.get('lastError', 'n/a')}")
                all_up = False
        return all_up
    elif hubble_dropped:
        warn(f"Hubble target DROPPATO ({len(hubble_dropped)} entry) — ServiceMonitor trovato ma target escluso")
        for t in hubble_dropped[:2]:
            dl = t.get("discoveredLabels", {})
            info(f"discoveredLabels: {json.dumps(dl, indent=6)}")
        return False
    else:
        fail("Hubble NON presente nei target Prometheus (né attivi né droppati)")
        info("→ Il Prometheus Operator non ha rilevato alcun ServiceMonitor per hubble")
        return False


# ── Sezione 2: Prometheus Operator config ─────────────────────────────────────
def check_prometheus_operator(cluster: str) -> dict:
    """Controlla serviceMonitorSelector e serviceMonitorNamespaceSelector."""
    section("[2] Prometheus Operator — configurazione selector")

    prom_obj = kubectl_json(["get", "prometheus", "-n", "monitoring"], cluster)
    if not prom_obj or not prom_obj.get("items"):
        fail("Nessun oggetto 'prometheus' trovato in namespace monitoring")
        return {"needs_fix": True, "reason": "prometheus_not_found"}

    spec = prom_obj["items"][0].get("spec", {})
    sm_selector    = spec.get("serviceMonitorSelector")
    sm_ns_selector = spec.get("serviceMonitorNamespaceSelector")

    # Mostra i valori
    if sm_selector is None:
        info("serviceMonitorSelector: NON impostato → scrape TUTTI i ServiceMonitor ✓")
    elif sm_selector == {}:
        info("serviceMonitorSelector: {} → scrape TUTTI i ServiceMonitor ✓")
    else:
        info(f"serviceMonitorSelector: {json.dumps(sm_selector)}")
        if "matchLabels" in sm_selector:
            required = sm_selector["matchLabels"]
            info(f"  → ServiceMonitor deve avere label: {required}")
            # Verifica che 'release: prometheus' sia richiesto
            if required.get("release") == "prometheus":
                ok("Il ServiceMonitor hubble ha release=prometheus ✓")

    if sm_ns_selector is None:
        # Comportamento default: guarda SOLO il namespace del Prometheus resource (monitoring)
        fail("serviceMonitorNamespaceSelector: NON impostato")
        print(f"\n  {RED}⚡ ROOT CAUSE TROVATA:{RESET}")
        print(f"    Quando 'serviceMonitorNamespaceSelector' non è impostato,")
        print(f"    il Prometheus Operator guarda ServiceMonitor SOLO nel namespace 'monitoring'.")
        print(f"    Il ServiceMonitor 'hubble' è in 'kube-system' → NON viene visto!")
        return {"needs_fix": True, "reason": "no_namespace_selector"}

    elif sm_ns_selector == {}:
        ok("serviceMonitorNamespaceSelector: {} → guarda TUTTI i namespace ✓")
        # In questo caso il ServiceMonitor in kube-system DOVREBBE essere visto
        warn("Il Prometheus Operator guarda tutti i namespace ma Hubble non è nei target")
        info("→ Possibile problema con port name o label selector")
        return {"needs_fix": True, "reason": "selector_mismatch"}

    else:
        info(f"serviceMonitorNamespaceSelector: {json.dumps(sm_ns_selector)}")
        # Controlla se include kube-system
        match_names = sm_ns_selector.get("matchNames", [])
        if "kube-system" in match_names:
            ok("kube-system è incluso nel namespace selector ✓")
            return {"needs_fix": True, "reason": "selector_mismatch"}
        else:
            fail(f"kube-system NON incluso: {match_names}")
            return {"needs_fix": True, "reason": "namespace_not_included"}


# ── Sezione 3: Service hubble-metrics labels ───────────────────────────────────
def check_hubble_service(cluster: str) -> dict:
    """Controlla il Service hubble-metrics e le sue labels."""
    section("[3] Service hubble-metrics — labels e porte")

    svc = kubectl_json(["get", "service", "hubble-metrics", "-n", "kube-system"], cluster)
    if not svc:
        fail("Service 'hubble-metrics' non trovato in kube-system")
        return {}

    labels = svc.get("metadata", {}).get("labels", {})
    ports  = svc.get("spec", {}).get("ports", [])
    cluster_ip = svc.get("spec", {}).get("clusterIP", "")

    ok("Service 'hubble-metrics' trovato")
    info(f"ClusterIP: {cluster_ip}")
    info(f"Labels: {json.dumps(labels)}")

    port_info = {}
    for p in ports:
        name     = p.get("name", "")
        port_num = p.get("port", "")
        proto    = p.get("protocol", "TCP")
        info(f"Port: name={name!r}, port={port_num}/{proto}")
        if port_num == 9965 or "hubble" in name.lower():
            port_info = {"name": name, "port": port_num}

    if not port_info:
        warn("Nessuna porta con nome 'hubble-metrics' o porta 9965 trovata")
        if ports:
            # usa il primo
            port_info = {"name": ports[0].get("name", ""), "port": ports[0].get("port", 9965)}

    return {"labels": labels, "port": port_info}


# ── Sezione 4: ServiceMonitor 'hubble' in kube-system ─────────────────────────
def check_hubble_servicemonitor(cluster: str) -> dict:
    """Controlla il ServiceMonitor creato da Cilium helm in kube-system."""
    section("[4] ServiceMonitor 'hubble' in kube-system")

    sm = kubectl_json(["get", "servicemonitor", "hubble", "-n", "kube-system"], cluster)
    if not sm:
        fail("ServiceMonitor 'hubble' non trovato in kube-system")
        # Lista tutti i ServiceMonitor in kube-system
        rc, out, _ = kubectl(["get", "servicemonitor", "-n", "kube-system"], cluster)
        if out:
            info(f"ServiceMonitor presenti in kube-system:\n{out}")
        else:
            warn("Nessun ServiceMonitor in kube-system")
        return {"found": False}

    ok("ServiceMonitor 'hubble' trovato in kube-system")
    spec   = sm.get("spec", {})
    labels = sm.get("metadata", {}).get("labels", {})
    info(f"Labels SM:           {json.dumps(labels)}")
    info(f"namespaceSelector:   {json.dumps(spec.get('namespaceSelector', {}))}")
    info(f"selector:            {json.dumps(spec.get('selector', {}))}")
    for ep in spec.get("endpoints", []):
        info(f"endpoint.port:       {ep.get('port', 'n/a')}")
        info(f"endpoint.interval:   {ep.get('interval', 'n/a')}")

    return {"found": True, "labels": labels, "spec": spec}


# ── Fix: crea ServiceMonitor in 'monitoring' ──────────────────────────────────
def build_servicemonitor_yaml(svc_info: dict) -> str:
    """Costruisce il YAML del ServiceMonitor in base alle labels reali del Service."""
    labels  = svc_info.get("labels", {})
    port    = svc_info.get("port", {})
    port_name = port.get("name", "") or "hubble-metrics"

    # Strategia per il selector: usa la label più specifica disponibile
    selector_lines = ""
    priority_keys = [
        "app.kubernetes.io/name",
        "k8s-app",
        "app",
    ]
    for key in priority_keys:
        if key in labels:
            selector_lines = f"      {key}: {labels[key]}"
            break

    # Fallback: usa k8s-app=hubble o hubble-metrics (standard Cilium)
    if not selector_lines:
        selector_lines = "      k8s-app: hubble-metrics"

    return f"""\
# ServiceMonitor per Hubble metrics nel namespace 'monitoring'
# Fix generato automaticamente da fix_hubble_prometheus.py
# Il Prometheus Operator guarda sempre il namespace 'monitoring'.
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: hubble-metrics
  namespace: monitoring
  labels:
    release: prometheus
spec:
  namespaceSelector:
    matchNames:
      - kube-system
  selector:
    matchLabels:
{selector_lines}
  endpoints:
    - port: {port_name}
      interval: 15s
      path: /metrics
      honorLabels: true
      relabelings:
        - sourceLabels: [__meta_kubernetes_pod_node_name]
          targetLabel: node
"""


def apply_servicemonitor(cluster: str, yaml_content: str) -> bool:
    """Applica il ServiceMonitor via kubectl apply su file temporaneo."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name
    try:
        kube = os.path.join(ROOT_DIR, ".kube", f"{cluster}.yaml")
        r = subprocess.run(
            ["kubectl", "apply", "-f", tmp_path, "--kubeconfig", kube],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            ok(f"ServiceMonitor applicato: {r.stdout.strip()}")
            return True
        else:
            fail(f"kubectl apply fallito: {r.stderr.strip()}")
            return False
    finally:
        os.unlink(tmp_path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Diagnosi e fix Hubble metrics → Prometheus"
    )
    parser.add_argument("--fix", action="store_true",
                        help="Applica la fix automaticamente (crea ServiceMonitor in monitoring)")
    parser.add_argument("--clusters", nargs="+", default=CLUSTERS,
                        help="Cluster da controllare (default: tutti)")
    args = parser.parse_args()

    any_needs_fix = False

    for cluster in args.clusters:
        print(f"\n{'━'*65}")
        print(f"  {BOLD}CLUSTER: {cluster}  ({CLUSTER_IPS[cluster]}){RESET}")
        print(f"{'━'*65}")

        # 1. Verifica target Prometheus
        hubble_up = check_prometheus_targets(cluster)
        if hubble_up:
            ok(f"Hubble già scraped correttamente su {cluster}!")
            results = prom_query(cluster, "count(hubble_http_requests_total)")
            if results:
                ok(f"hubble_http_requests_total presente in Prometheus ✓")
            else:
                warn("Target up ma nessun dato ancora (attendi traffico)")
            continue

        # 2. Analisi Prometheus Operator
        op_info  = check_prometheus_operator(cluster)

        # 3. Service labels
        svc_info = check_hubble_service(cluster)

        # 4. ServiceMonitor in kube-system
        sm_info  = check_hubble_servicemonitor(cluster)

        # ── Diagnosi finale ───────────────────────────────────────────────
        section("[DIAGNOSI E FIX]")

        if not svc_info:
            fail("Service hubble-metrics mancante → ri-esegui helm upgrade Cilium")
            continue

        # Genera YAML fix basato sulle label reali del Service
        yaml_content = build_servicemonitor_yaml(svc_info)

        # Salva il file aggiornato
        fix_path = os.path.join(ROOT_DIR, "deployments", "hubble-servicemonitor-monitoring.yaml")
        with open(fix_path, "w") as f:
            f.write(yaml_content)
        ok(f"Fix YAML aggiornato con label reali: deployments/hubble-servicemonitor-monitoring.yaml")

        print(f"\n  {CYAN}ServiceMonitor da applicare:{RESET}")
        for line in yaml_content.splitlines():
            print(f"    {line}")

        any_needs_fix = True

        if args.fix:
            section(f"[APPLICO FIX su {cluster}]")
            success = apply_servicemonitor(cluster, yaml_content)
            if success:
                info("Attendo 30s per il reload del Prometheus Operator...")
                time.sleep(30)
                # Verifica
                results = prom_query(cluster, "count(hubble_http_requests_total)")
                if results:
                    ok(f"✅ hubble_http_requests_total ora in Prometheus! (serie: {results})")
                else:
                    targets2 = prom_targets(cluster)
                    active2  = targets2.get("activeTargets", [])
                    hubble2  = [t for t in active2 if "hubble" in str(t).lower() or "9965" in str(t)]
                    if hubble2:
                        ok(f"Target hubble trovato ({len(hubble2)} endpoint) — attendi traffico per le metriche HTTP")
                    else:
                        warn("ServiceMonitor applicato ma metriche non ancora presenti")
                        warn("→ Aspetta 1-2 minuti e riesegui check_prometheus.py")
        else:
            print(f"\n  {YELLOW}Per applicare la fix:{RESET}")
            print(f"    python scripts/fix_hubble_prometheus.py --fix")
            print(f"\n  Oppure manualmente su tutti i cluster:")
            for c in args.clusters:
                print(f"    kubectl apply -f deployments/hubble-servicemonitor-monitoring.yaml --kubeconfig .kube/{c}.yaml")

    # ── Riepilogo finale ──────────────────────────────────────────────────
    print(f"\n{'━'*65}")
    if any_needs_fix and not args.fix:
        print(f"\n  {BOLD}{YELLOW}PROSSIMO STEP:{RESET}")
        print(f"  Esegui la fix automatica:")
        print(f"\n    python scripts/fix_hubble_prometheus.py --fix")
        print(f"\n  Poi verifica:")
        print(f"\n    python scripts/check_prometheus.py")
    elif any_needs_fix and args.fix:
        print(f"\n  Fix applicata. Verifica finale:")
        print(f"\n    python scripts/check_prometheus.py")
    print()


if __name__ == "__main__":
    main()
