#!/usr/bin/env python3
"""
kubephos_fleet.py — DMOS on a multicluster built by Kubephos (N clusters).

Helper of scripts/setup-kubephos-fleet.sh (the orchestrator); every
subcommand can also be run on its own:

  discover     read the Kubephos kubeconfig (one context per cluster), write one
               kubeconfig per cluster in --out-dir and fleet.json with what DMOS
               needs (address, app-pool CPU/RAM, Prometheus, GatewayClass,
               Cilium, ClusterMesh); exits 1 on a blocking problem
  hubble-fix   add the Hubble httpV2 labels DMOS needs where they are missing
               (helm upgrade of the installed Cilium chart, then agent restart)
  calibrate    with fixed replicas, at two steady loads (~11 min): k, the marginal
               app-node CPU per request; c_rep, the idle cost of one more replica
               set; r, the req/s one entry replica serves at 70% of its CPU
               limit; calibration.json
  write-local  write config/local.yaml for the fleet (backup of the previous one
               in --out-dir), with the Gateway NodePorts read from the clusters
               and, after a calibration, the replica cap of every cluster
  check        verify the fleet against config/local.yaml; --probe also sends a
               little traffic to the first Gateway and checks what DMOS measures
  compare      table of the k6 summaries and frontend replicas of run dirs

No Kubernetes object is changed by discover, check, compare. hubble-fix changes
the Cilium Helm release; calibrate scales Deployments (restored at the end) and
sends load; write-local changes only files.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

REPO = Path(__file__).resolve().parent.parent
LOCAL_YAML = REPO / "config" / "local.yaml"
ENV_JSON = REPO / ".cache" / "test-env.json"

APP_POOL_LABEL = "kubephos.dev/pool"
NAMESPACE = "online-boutique"
GATEWAY = "frontend-gateway"
GATEWAY_SERVICE = f"cilium-gateway-{GATEWAY}"
HUBBLE_HTTP = ("httpV2:labelsContext=source_namespace,source_workload,"
               "destination_namespace,destination_workload,traffic_direction")
HUBBLE_LABELS = ("source_namespace", "source_workload", "destination_namespace",
                 "destination_workload", "traffic_direction")
CILIUM_AGENT_METRICS_PORT = 9962
TIMEOUT = "20s"

# Regions of the reference fleet (config/local.yaml.example), in paper order.
# A region not listed here needs --baseline REGION=gCO2/kWh.
REGIONS: Dict[str, Dict[str, Any]] = {
    "DE":    {"location": "Frankfurt", "baseline": 350, "price": 0.12},
    "FR":    {"location": "Paris",     "baseline": 80,  "price": 0.15},
    "PL":    {"location": "Warsaw",    "baseline": 650, "price": 0.08},
    "US-VA": {"location": "Ashburn",   "baseline": 380, "price": 0.10},
    "SG":    {"location": "Singapore", "baseline": 480, "price": 0.11},
    "NO":    {"location": "Oslo",      "baseline": 25,  "price": 0.09},
}
# Emulated geography (tc netem, delay/jitter in ms) of the reference fleet: every
# cluster delays what leaves it, so client->cluster i ~ d_i, cluster i <-> j ~ d_i + d_j.
NETEM_MS: Dict[str, Tuple[int, int]] = {
    "DE": (0, 0), "FR": (30, 5), "PL": (50, 10), "US-VA": (100, 15), "SG": (200, 25), "NO": (40, 8),
}
DEFAULT_REGION_ORDER = ["DE", "FR", "PL", "US-VA", "SG", "NO"]


class FleetError(Exception):
    pass


# ── small helpers ───────────────────────────────────────────────────────────

def run(cmd: List[str], timeout: int = 120, check: bool = True) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise FleetError(f"{' '.join(cmd[:4])} ...: {(p.stderr or p.stdout).strip()[:400]}")
    return p.stdout


def kubectl(kc: str, *args: str, timeout: int = 60) -> str:
    return run(["kubectl", "--kubeconfig", kc, f"--request-timeout={TIMEOUT}", *args], timeout=timeout)


def kjson(kc: str, *args: str) -> Any:
    return json.loads(kubectl(kc, *args, "-o", "json"))


def http_get(url: str, timeout: float = 6.0) -> Tuple[int, str]:
    import requests
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:  # noqa: BLE001 - any network error is "unreachable"
        return 0, str(e)


def mem_gib(quantity: str) -> float:
    m = re.fullmatch(r"(\d+)(Ki|Mi|Gi|K|M|G)?", quantity.strip())
    if not m:
        raise FleetError(f"unexpected memory quantity {quantity!r}")
    n = int(m.group(1))
    factor = {"Ki": 1024, "Mi": 1024 ** 2, "Gi": 1024 ** 3, "K": 1e3, "M": 1e6, "G": 1e9, None: 1}
    return n * factor[m.group(2)] / 1024 ** 3


def cpu_cores(quantity: str) -> float:
    q = quantity.strip()
    return int(q[:-1]) / 1000 if q.endswith("m") else float(q)


# ── pure functions (tested) ─────────────────────────────────────────────────

def kubeconfig_contexts(doc: Dict[str, Any]) -> List[Tuple[str, str]]:
    """[(context, API server URL)] in kubeconfig order."""
    servers = {c["name"]: c["cluster"]["server"] for c in doc.get("clusters") or []}
    out = []
    for ctx in doc.get("contexts") or []:
        cl = ctx["context"]["cluster"]
        if cl not in servers:
            raise FleetError(f"context {ctx['name']}: cluster {cl} not in the kubeconfig")
        out.append((ctx["name"], servers[cl]))
    if not out:
        raise FleetError("the kubeconfig has no contexts")
    return out


def node_summary(nodes: Dict[str, Any], app_pool: str) -> Dict[str, Any]:
    """App-pool capacity (VM sizes: per-node rounding) and the pool of each address."""
    app, pool_of_ip, not_ready = [], {}, []
    for n in nodes.get("items") or []:
        labels = n["metadata"].get("labels") or {}
        pool = labels.get(APP_POOL_LABEL, "")
        for a in n["status"].get("addresses") or []:
            if a.get("type") == "InternalIP":
                pool_of_ip[a["address"]] = pool
        ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                    for c in n["status"].get("conditions") or [])
        if not ready:
            not_ready.append(n["metadata"]["name"])
        if pool == app_pool:
            cap = n["status"]["capacity"]
            app.append({"name": n["metadata"]["name"], "cpu": cpu_cores(cap["cpu"]),
                        "memory_gb": round(mem_gib(cap["memory"]))})
    return {
        "app_nodes": [a["name"] for a in app],
        "cpu_cores": int(round(sum(a["cpu"] for a in app))),
        "memory_gb": int(sum(a["memory_gb"] for a in app)),
        "pool_of_ip": pool_of_ip,
        "not_ready": not_ready,
    }


def prometheus_info(prometheuses: Dict[str, Any], services: Dict[str, Any]) -> Dict[str, Any]:
    items = prometheuses.get("items") or []
    if len(items) != 1:
        raise FleetError(f"expected one Prometheus (operator CR), found {len(items)}")
    p = items[0]
    ns = p["metadata"]["namespace"]
    release = ((p["spec"].get("serviceMonitorSelector") or {}).get("matchLabels") or {}).get("release")
    if not release:
        raise FleetError("the Prometheus CR selects ServiceMonitors without a `release` label")
    ports = [pt.get("nodePort") for s in services.get("items") or []
             if s["metadata"]["namespace"] == ns and s["spec"].get("type") == "NodePort"
             for pt in s["spec"]["ports"] if pt.get("port") == 9090 and pt.get("nodePort")]
    if not ports:
        raise FleetError(f"no NodePort Service on port 9090 in {ns}")
    return {"namespace": ns, "release": release, "port": int(ports[0])}


def nodeport_gateway_class(classes: Dict[str, Any], configs: Dict[str, Any],
                           preferred: str = "kubephos-cilium") -> Optional[str]:
    """A Cilium GatewayClass whose CiliumGatewayClassConfig exposes NodePort."""
    nodeport_cfgs = {(c["metadata"].get("namespace", ""), c["metadata"]["name"])
                     for c in configs.get("items") or []
                     if ((c.get("spec") or {}).get("service") or {}).get("type") == "NodePort"}
    found = []
    for gc in classes.get("items") or []:
        if gc["spec"].get("controllerName") != "io.cilium/gateway-controller":
            continue
        ref = gc["spec"].get("parametersRef") or {}
        if ref.get("kind") == "CiliumGatewayClassConfig" and \
                (ref.get("namespace", ""), ref.get("name")) in nodeport_cfgs:
            found.append(gc["metadata"]["name"])
    if preferred in found:
        return preferred
    return sorted(found)[0] if found else None


def hubble_missing(hubble_metrics: str) -> List[str]:
    """Labels DMOS needs that the httpV2 metric of cilium-config lacks."""
    http = next((m for m in hubble_metrics.split() if m.startswith("httpV2")), None)
    if http is None:
        return list(HUBBLE_LABELS)
    ctx = re.search(r"labelsContext=([^;]*)", http)
    have = set(ctx.group(1).split(",")) if ctx else set()
    return [l for l in HUBBLE_LABELS if l not in have]


def hubble_metrics_values(current: List[str]) -> List[str]:
    """The Hubble metric list with httpV2 replaced by the one DMOS needs."""
    rest = [m for m in current if not str(m).startswith("httpV2")]
    return [HUBBLE_HTTP] + (rest or ["dns", "drop", "tcp", "flow", "icmp"])


def region_plan(n: int, regions: Optional[List[str]],
                baselines: Dict[str, float]) -> List[Dict[str, Any]]:
    if regions is None:
        if n > len(DEFAULT_REGION_ORDER):
            raise FleetError(f"{n} clusters: give --regions (the defaults cover "
                             f"{len(DEFAULT_REGION_ORDER)})")
        regions = DEFAULT_REGION_ORDER[:n]
    if len(regions) != n:
        raise FleetError(f"--regions has {len(regions)} entries, the fleet {n} clusters")
    plan = []
    for r in regions:
        known = REGIONS.get(r, {})
        base = baselines.get(r, known.get("baseline"))
        if base is None:
            raise FleetError(f"region {r}: unknown carbon baseline, give --baseline {r}=<gCO2/kWh>")
        base = float(base)
        plan.append({"region": r, "location": known.get("location", r),
                     "baseline": int(base) if base.is_integer() else base,
                     "price": known.get("price", 0.10)})
    return plan


def netem_entry(region: str, enabled: bool) -> Dict[str, Any]:
    """netem block of one cluster: scope 'cluster' = all its nodes, only traffic
    to addresses outside the cluster (see scripts/setup-netem-and-ping.sh)."""
    if not enabled:
        return {"delay_ms": 0, "jitter_ms": 0, "interface": "eth0"}
    if region not in NETEM_MS:
        raise FleetError(f"no netem delay for region {region} (known: {sorted(NETEM_MS)})")
    d, j = NETEM_MS[region]
    return {"delay_ms": d, "jitter_ms": j, "interface": "auto", "scope": "cluster"}


def build_local(fleet: Dict[str, Any], plan: List[Dict[str, Any]], ports: Dict[str, int],
                ssh: Dict[str, Any], title: str,
                caps: Optional[Dict[str, Dict[str, Any]]] = None,
                netem: bool = False) -> Tuple[str, Dict[str, Any]]:
    """config/local.yaml text and data for the fleet (caps: {cluster: replica_cap()};
    netem: the delays of NETEM_MS, applied by scripts/setup-netem-and-ping.sh)."""
    caps = caps or {}
    clusters = []
    for c, reg in zip(fleet["clusters"], plan):
        entry = {
            "name": c["name"],
            "display_name": f"{reg['region']} ({c['name']})",
            "region": reg["region"],
            "location": reg["location"],
            "ip": c["ip"],
            "kubeconfig_path": c["kubeconfig"],
            "prometheus_port": c["prometheus"]["port"],
            "ingress_port": int(ports[c["name"]]),
            "ingress_demand_match": c["ingress_demand_match"],
            "resources": {"cpu_cores": c["cpu_cores"], "memory_gb": c["memory_gb"]},
            "pricing": {"cost_per_replica_hour": reg["price"]},
            "carbon": {"region_code": reg["region"], "baseline_gco2_kwh": reg["baseline"]},
            "latency": {"baseline_ms": 2.0},
            "netem": netem_entry(reg["region"], netem),
        }
        workers = sorted(ip for ip, pool in (c.get("pool_of_ip") or {}).items()
                         if pool == fleet.get("app_pool", "app"))
        if workers:
            entry["worker_ips"] = workers
        if c["name"] in caps:
            cp = caps[c["name"]]
            entry["max_replicas"] = {cp["service"]: cp["cap"]}
            if cp.get("r_calibrated"):
                entry["capacity_req_per_sec"] = {cp["service"]: int(cp["r"])}
        clusters.append(entry)
    data = {"ssh": ssh, "clusters": clusters,
            "global": {"scheduling_interval_seconds": 30, "score_request_timeout_seconds": 5},
            "alertmanager": {"webhook_url": ""}}
    head = (
        "# " + "=" * 77 + "\n"
        f"# DMOS local.yaml — {title}\n"
        f"# Generated by scripts/kubephos_fleet.py write-local on "
        f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')}. Re-run it after changing the fleet.\n"
        "#\n"
        "#   ip                    address Kubephos gives for the cluster (NodePorts answer\n"
        "#                         on every node with kube-proxy replacement)\n"
        "#   prometheus_port       NodePort of the Kubephos Prometheus\n"
        "#   ingress_port          NodePort of the Service of the Gateway (Gateway API)\n"
        "#   ingress_demand_match  namespace when the Gateway Envoy runs on a node without\n"
        "#                         frontends (docs/traffic-metrics-architecture.md)\n"
        "#   worker_ips            the app-pool nodes: DMOS measures free CPU/memory there\n"
        "#   resources             capacity of the app pool (nodes kubephos.dev/pool=app)\n"
        + ("#   netem                 emulated geography: every node of the cluster delays the\n"
           "#                         traffic leaving the cluster (scope cluster, interface auto =\n"
           "#                         the interface of the default route); apply it with\n"
           "#                         bash scripts/setup-netem-and-ping.sh\n"
           if netem else "#   netem                 0: no emulated geography on this fleet\n"))
    if caps:
        any_cap = next(iter(caps.values()))
        head += (
            "#   max_replicas          replica cap of the entry service from the app-pool capacity\n"
            "#   capacity_req_per_sec  req/s of one entry replica, calibrated (docs/KUBEPHOS.md,\n"
            "#                         'Replica cap'):\n"
            "#                         cap = min(floor((u*V - C0) / (r*k + c_rep)), floor((um*M - M0) / m))\n"
            f"#                         k = {any_cap['k_mcore']:.1f} mcore/req ({any_cap['k_source']}), "
            f"r = {any_cap['r']:.0f} req/s, c_rep = {any_cap['c_rep']:.2f} core, "
            f"u = {any_cap['u_cpu']:g}, um = {any_cap['u_mem']:g}\n")
        for name, cp in caps.items():
            head += (f"#                         {name}: V={cp['V']:g} C0={cp['C0']:.2f} -> {cp['cap_cpu']}, "
                     f"M={cp['M']:g} M0={cp['M0']:.1f} m={cp['m_unit']:.2f} -> {cp['cap_mem']}"
                     f"  => {cp['cap']}\n")
    head += "# " + "=" * 77 + "\n\n"
    return head + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), data


def demand_match_for(ip: str, pool_of_ip: Dict[str, str], app_pool: str) -> str:
    """`workload` only when the entry address is an app node (Envoy next to the
    frontends); otherwise Hubble cannot name the destination workload."""
    return "workload" if pool_of_ip.get(ip) == app_pool else "namespace"


def phase_metrics(summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    per = (summary.get("results") or {}).get("per_phase") or {}
    return {ph: {"p95": v.get("p95"), "err": v.get("errorRate"), "rps": v.get("rps")}
            for ph, v in per.items() if isinstance(v, dict)}


def max_replicas(live_lines: List[str], service: str = "frontend") -> Dict[str, int]:
    out: Dict[str, int] = {}
    for line in live_lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        clusters = (((rec.get("services") or {}).get(service) or {}).get("clusters") or {})
        for name, c in clusters.items():
            r = c.get("current_replicas")
            if isinstance(r, (int, float)):
                out[name] = max(out.get(name, 0), int(r))
    return out


# ── discover ────────────────────────────────────────────────────────────────

def discover_cluster(ctx: str, server: str, kc: str, app_pool: str) -> Dict[str, Any]:
    ip = urlparse(server).hostname
    nodes = node_summary(kjson(kc, "get", "nodes"), app_pool)
    prom = prometheus_info(kjson(kc, "get", "prometheus", "-A"), kjson(kc, "get", "svc", "-A"))
    gwc = nodeport_gateway_class(kjson(kc, "get", "gatewayclass"),
                                 kjson(kc, "get", "ciliumgatewayclassconfigs", "-A"))
    cm = kjson(kc, "-n", "kube-system", "get", "cm", "cilium-config").get("data") or {}
    rel = json.loads(run(["helm", "list", "-n", "kube-system", "--kubeconfig", kc,
                          "-f", "^cilium$", "-o", "json"]))
    chart = rel[0]["chart"] if rel else ""
    ms = kjson(kc, "get", "apiservice", "v1beta1.metrics.k8s.io")
    ms_ok = any(c.get("type") == "Available" and c.get("status") == "True"
                for c in ms.get("status", {}).get("conditions") or [])
    return {
        "name": ctx, "ip": ip, "kubeconfig": kc,
        "app_nodes": nodes["app_nodes"], "cpu_cores": nodes["cpu_cores"],
        "memory_gb": nodes["memory_gb"], "not_ready": nodes["not_ready"],
        "pool_of_ip": nodes["pool_of_ip"],
        "ingress_demand_match": demand_match_for(ip, nodes["pool_of_ip"], app_pool),
        "prometheus": prom, "gateway_class": gwc,
        "cilium_chart": chart, "cluster_id": cm.get("cluster-id"),
        "hubble_missing": hubble_missing(cm.get("hubble-metrics", "")),
        "metrics_server": ms_ok,
    }


def mesh_status(ip: str) -> Tuple[Optional[int], Dict[str, str]]:
    code, text = http_get(f"http://{ip}:{CILIUM_AGENT_METRICS_PORT}/metrics")
    if code != 200:
        return None, {}
    n = re.search(r"^cilium_clustermesh_remote_clusters(?:\{[^}]*\})? (\S+)$", text, re.M)
    ready = dict(re.findall(
        r'^cilium_clustermesh_remote_cluster_readiness_status\{[^}]*target_cluster="([^"]+)"[^}]*\} (\S+)$',
        text, re.M))
    return (int(float(n.group(1))) if n else None), ready


def fleet_problems(fleet: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(blocking problems, warnings) of a discovered fleet."""
    errs, warns = [], []
    cl = fleet["clusters"]
    ids = [c["cluster_id"] for c in cl]
    if len(set(ids)) != len(ids):
        errs.append(f"Cilium cluster-id not unique: {ids}")
    for key in ("namespace", "release"):
        vals = {c["prometheus"][key] for c in cl}
        if len(vals) != 1:
            errs.append(f"Prometheus {key} differs between clusters: {sorted(vals)}")
    for c in cl:
        n = c["name"]
        if c["not_ready"]:
            errs.append(f"{n}: nodes not Ready: {c['not_ready']}")
        if not c["app_nodes"]:
            errs.append(f"{n}: no node with {APP_POOL_LABEL}={fleet['app_pool']}")
        if not c["gateway_class"]:
            errs.append(f"{n}: no Cilium GatewayClass with a NodePort Service")
        if not c["metrics_server"]:
            errs.append(f"{n}: metrics-server (v1beta1.metrics.k8s.io) not Available")
        if not c["cilium_chart"]:
            errs.append(f"{n}: no Helm release 'cilium' in kube-system")
        if c["mesh_remote"] is None:
            warns.append(f"{n}: Cilium agent metrics not reachable on {c['ip']}:"
                         f"{CILIUM_AGENT_METRICS_PORT}, ClusterMesh not verified")
        else:
            if c["mesh_remote"] != len(cl) - 1:
                errs.append(f"{n}: ClusterMesh connected to {c['mesh_remote']} clusters, "
                            f"expected {len(cl) - 1}")
            bad = [t for t, v in c["mesh_ready"].items() if float(v) != 1]
            if bad:
                errs.append(f"{n}: ClusterMesh peers not ready: {bad}")
        if c["hubble_missing"]:
            warns.append(f"{n}: Hubble httpV2 lacks {c['hubble_missing']} (hubble-fix adds them)")
        if c["cpu_cores"] < 4:
            warns.append(f"{n}: app pool of {c['cpu_cores']} vCPU: the frontend cap (8 replicas, "
                         f"services.yaml) may not fit")
    return errs, warns


def cmd_discover(a) -> int:
    out = Path(a.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    doc = yaml.safe_load(Path(a.kubeconfig).expanduser().read_text(encoding="utf-8"))
    ctxs = kubeconfig_contexts(doc)
    if a.contexts:
        want = a.contexts.split(",")
        known = dict(ctxs)
        missing = [c for c in want if c not in known]
        if missing:
            raise FleetError(f"contexts not in the kubeconfig: {missing}")
        ctxs = [(c, known[c]) for c in want]
    for old in out.glob("kubeconfig-*"):
        if old.name.removeprefix("kubeconfig-") not in {c for c, _ in ctxs}:
            print(f"  ! {old} is not a cluster of this kubeconfig: moved to {old.name}.old")
            old.replace(old.with_name(old.name + ".old"))
    clusters = []
    for ctx, server in ctxs:
        kc = out / f"kubeconfig-{ctx}"
        kc.write_text(run(["kubectl", "--kubeconfig", str(Path(a.kubeconfig).expanduser()),
                           "config", "view", "--minify", "--flatten", "--context", ctx]),
                      encoding="utf-8")
        clusters.append((ctx, server, kc.as_posix()))
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        found = list(ex.map(lambda t: discover_cluster(t[0], t[1], t[2], a.app_pool), clusters))
    for c in found:
        c["mesh_remote"], c["mesh_ready"] = mesh_status(c["ip"])
    fleet = {"name": a.name, "kubeconfig": str(Path(a.kubeconfig).expanduser()),
             "app_pool": a.app_pool, "discovered": dt.datetime.now().isoformat(timespec="seconds"),
             "clusters": found}
    (out / "fleet.json").write_text(json.dumps(fleet, indent=2), encoding="utf-8")
    print(f"  {'cluster':<20} {'ip':<16} {'app pool':<14} {'Prometheus':<10} "
          f"{'GatewayClass':<16} {'Cilium':<14} {'mesh':<6} demand")
    for c in found:
        mesh = "?" if c["mesh_remote"] is None else f"{c['mesh_remote']}/{len(found) - 1}"
        print(f"  {c['name']:<20} {c['ip']:<16} {c['cpu_cores']:>2} vCPU {c['memory_gb']:>3} GB "
              f"{c['prometheus']['port']:<10} {str(c['gateway_class']):<16} "
              f"{c['cilium_chart']:<14} {mesh:<6} {c['ingress_demand_match']}")
    errs, warns = fleet_problems(fleet)
    for w in warns:
        print(f"  ! {w}")
    for e in errs:
        print(f"  x {e}")
    print(f"  fleet.json and {len(found)} kubeconfigs in {out}")
    return 1 if errs else 0


# ── hubble-fix ──────────────────────────────────────────────────────────────

def load_fleet(out_dir: str) -> Dict[str, Any]:
    p = Path(out_dir).expanduser() / "fleet.json"
    if not p.is_file():
        raise FleetError(f"{p} missing: run discover first")
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_hubble_fix(a) -> int:
    fleet = load_fleet(a.out_dir)
    # Read cilium-config NOW: fleet.json holds the state of the last discover,
    # and a chart re-applied afterwards (30/10/2026, ms-01-cluster-03) drops
    # the labels without fleet.json knowing.
    for c in fleet["clusters"]:
        try:
            cm = kjson(c["kubeconfig"], "-n", "kube-system", "get", "cm",
                       "cilium-config").get("data") or {}
            c["hubble_missing"] = hubble_missing(cm.get("hubble-metrics", ""))
        except Exception as e:                 # unreachable: keep what fleet.json says
            print(f"  {c['name']}: cilium-config not read ({e}); using fleet.json")
    todo = [c for c in fleet["clusters"] if c["hubble_missing"]]
    if not todo:
        print("  Hubble httpV2 already has the labels DMOS needs on every cluster")
        return 0
    run(["helm", "repo", "add", "cilium", "https://helm.cilium.io", "--force-update"])
    for c in todo:
        kc, version = c["kubeconfig"], c["cilium_chart"].removeprefix("cilium-")
        values = json.loads(run(["helm", "get", "values", "cilium", "-n", "kube-system",
                                 "--kubeconfig", kc, "-o", "json"]))
        current = ((values.get("hubble") or {}).get("metrics") or {}).get("enabled") or []
        patch = {"hubble": {"metrics": {"enabled": hubble_metrics_values(current)}}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(patch, f)
            vfile = f.name
        try:
            print(f"  {c['name']}: helm upgrade cilium {version} (Hubble labels), agent restart")
            run(["helm", "upgrade", "cilium", "cilium/cilium", "-n", "kube-system", "--version",
                 version, "--reuse-values", "-f", vfile, "--kubeconfig", kc], timeout=600)
            kubectl(kc, "-n", "kube-system", "rollout", "restart", "ds/cilium")
            kubectl(kc, "-n", "kube-system", "rollout", "status", "ds/cilium", "--timeout=5m",
                    timeout=330)
        finally:
            Path(vfile).unlink(missing_ok=True)
        cm = kjson(kc, "-n", "kube-system", "get", "cm", "cilium-config").get("data") or {}
        if hubble_missing(cm.get("hubble-metrics", "")):
            raise FleetError(f"{c['name']}: Hubble labels still missing after the upgrade")
        c["hubble_missing"] = []
    (Path(a.out_dir).expanduser() / "fleet.json").write_text(json.dumps(fleet, indent=2),
                                                             encoding="utf-8")
    return 0


# ── write-local ─────────────────────────────────────────────────────────────

def gateway_port(kc: str) -> int:
    svc = kjson(kc, "-n", NAMESPACE, "get", "svc", GATEWAY_SERVICE)
    port = svc["spec"]["ports"][0].get("nodePort")
    if not port:
        raise FleetError(f"{GATEWAY_SERVICE} has no NodePort")
    return int(port)


# ── replica cap: capacity model ─────────────────────────────────────────────
#
#   cap_cpu = floor((u * V - C0) / (r * k + c_rep))
#   cap_mem = floor((um * M - M0) / m)
#   cap     = max(1, min(cap_cpu, cap_mem))
#
# V, M  vCPU and GB of the app pool        C0, M0  CPU (cores) and memory (GB)
# u, um target utilisation (0.8, 0.85)              used on the app nodes at idle
#                                                   with one replica per service
# r     req/s of one entry replica at its target CPU (calibrated: u_pod * CPU
#       limit / CPU per request of the entry pod; else capacity_req_per_sec)
# k     marginal app-node CPU per request of the whole path (entry service,
#       co-scaled backends, Cilium agent + Envoy L7): slope between two loads
# c_rep idle app-node CPU of one more replica of the entry service and of each
#       co-scaled backend (probes, runtime, Cilium endpoint): fixed cost per unit
# m     memory limits of one entry replica plus the share of the co-scaled
#       backends it drives: sum_b calls_b * r / capacity_b * limit_b

DEFAULT_U_CPU = 0.8
DEFAULT_U_MEM = 0.85


def services_model(path: Path = REPO / "config" / "services.yaml") -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    app = raw.get("application") or {}
    svcs = raw.get("services") or {}
    entry = app.get("entry_service", "frontend")
    co = list(app.get("co_scaled_services") or [])

    def one(n):
        v = svcs.get(n) or {}
        return {"capacity": float(v.get("capacity_req_per_sec") or 0),
                "calls": float(v.get("calls_per_request") or 0),
                "deployment": v.get("deployment_name") or n}
    return {"entry": entry, "co_scaled": co, "services": {n: one(n) for n in [entry] + co}}


def memory_per_unit(model: Dict[str, Any], limits_gb: Dict[str, float]) -> float:
    """GB of memory limits per entry replica, co-scaled backend share included."""
    e = model["services"][model["entry"]]
    total = limits_gb.get(e["deployment"], 0.0)
    for n in model["co_scaled"]:
        b = model["services"][n]
        if b["capacity"] > 0:
            total += b["calls"] * e["capacity"] / b["capacity"] * limits_gb.get(b["deployment"], 0.0)
    return total


def replica_cap(V: float, M: float, C0: float, M0: float, k_mcore: float, r: float,
                m_unit: float, u_cpu: float = DEFAULT_U_CPU,
                u_mem: float = DEFAULT_U_MEM, c_rep: float = 0.0) -> Dict[str, Any]:
    if k_mcore <= 0 or r <= 0 or m_unit <= 0 or c_rep < 0:
        raise FleetError(f"capacity model: k={k_mcore}, r={r}, m={m_unit}, c_rep={c_rep}: "
                         f"k, r, m must be > 0 and c_rep >= 0")
    unit = r * k_mcore / 1000.0 + c_rep
    cap_cpu = math.floor((u_cpu * V - C0) / unit)
    cap_mem = math.floor((u_mem * M - M0) / m_unit)
    return {"V": V, "M": M, "C0": C0, "M0": M0, "k_mcore": k_mcore, "r": r, "m_unit": m_unit,
            "c_rep": c_rep, "unit_cores": unit, "u_cpu": u_cpu, "u_mem": u_mem,
            "cap_cpu": cap_cpu, "cap_mem": cap_mem, "cap": max(1, min(cap_cpu, cap_mem))}

def _prom(base: str, query: str, at: Optional[float]) -> List[float]:
    import requests
    params = {"query": query}
    if at is not None:
        params["time"] = f"{at:.3f}"
    r = requests.get(f"{base}/api/v1/query", params=params, timeout=15)
    r.raise_for_status()
    return [float(x["value"][1]) for x in r.json()["data"]["result"]]


def prom_value(base: str, query: str, at: Optional[float] = None) -> float:
    return sum(_prom(base, query, at))


def prom_max(base: str, query: str, at: Optional[float] = None) -> float:
    return max(_prom(base, query, at), default=0.0)


def app_node_regex(c: Dict[str, Any], app_pool: str) -> str:
    ips = sorted(ip for ip, pool in c["pool_of_ip"].items() if pool == app_pool)
    # Regex \. inside a PromQL double-quoted string is written \\.
    return "|".join(ip.replace(".", "\\\\.") + ":.*" for ip in ips)


def node_cpu_query(c: Dict[str, Any], app_pool: str, window: str) -> str:
    return (f'sum(rate(node_cpu_seconds_total{{mode!~"idle|iowait|steal",'
            f'instance=~"{app_node_regex(c, app_pool)}"}}[{window}]))')


def node_mem_query(c: Dict[str, Any], app_pool: str) -> str:
    sel = f'instance=~"{app_node_regex(c, app_pool)}"'
    return (f"sum(node_memory_MemTotal_bytes{{{sel}}} - node_memory_MemAvailable_bytes{{{sel}}})"
            f" / 1024^3")


def prom_base(c: Dict[str, Any]) -> str:
    return f"http://{c['ip']}:{c['prometheus']['port']}"


def deployment_mem_limits(kc: str) -> Dict[str, float]:
    out = {}
    for d in kjson(kc, "-n", NAMESPACE, "get", "deploy").get("items") or []:
        cont = d["spec"]["template"]["spec"]["containers"]
        lim = ((cont[0].get("resources") or {}).get("limits") or {}).get("memory")
        if lim:
            out[d["metadata"]["name"]] = mem_gib(lim)
    return out


def compute_caps(fleet: Dict[str, Any], calib: Dict[str, Any], u_cpu: float,
                 u_mem: float) -> Dict[str, Dict[str, Any]]:
    model = services_model()
    r = float(calib.get("r_entry") or model["services"][model["entry"]]["capacity"])
    caps = {}
    for c in fleet["clusters"]:
        idle = (calib.get("clusters") or {}).get(c["name"]) or {}
        if "C0" not in idle:
            raise FleetError(f"{c['name']}: no idle measurement in the calibration")
        m_unit = memory_per_unit(model, deployment_mem_limits(c["kubeconfig"]))
        cp = replica_cap(c["cpu_cores"], c["memory_gb"], idle["C0"], idle["M0"],
                         calib["k_mcore"], r, m_unit, u_cpu, u_mem,
                         c_rep=float(calib.get("c_rep") or 0.0))
        cp.update({"service": model["entry"], "k_source": calib.get("source", "calibration"),
                   "r_calibrated": bool(calib.get("r_entry"))})
        caps[c["name"]] = cp
    return caps


def calibration_result(idle1: Dict[str, float], idleF: Dict[str, float], F: int,
                       low: Dict[str, Dict[str, float]], high: Dict[str, Dict[str, float]],
                       entry_cpu_limit: float, u_pod: float) -> Dict[str, float]:
    """k (slope), c_rep (idle cost of one more replica set) and r (req/s of one
    entry replica at u_pod of its CPU limit) from the calibration measurements.
    low/high: {cluster: {"cpu": app-node cores, "served": entry req/s,
    "entry_cpu": cores of the entry pods}}."""
    d_cpu = sum(high[x]["cpu"] - low[x]["cpu"] for x in high)
    d_req = sum(high[x]["served"] - low[x]["served"] for x in high)
    if d_req <= 0:
        raise FleetError("the two loads served the same rate: no slope to measure")
    k = 1000.0 * d_cpu / d_req
    c_rep = max(0.0, sum(idleF[x] - idle1[x] for x in idleF) / len(idleF) / max(1, F - 1))
    entry_cpu = sum(high[x]["entry_cpu"] for x in high)
    served = sum(high[x]["served"] for x in high)
    if entry_cpu <= 0 or served <= 0:
        raise FleetError("no CPU or traffic measured on the entry pods")
    r = u_pod * entry_cpu_limit / (entry_cpu / served)
    return {"k_mcore": k, "c_rep": c_rep, "r_entry": r,
            "entry_mcore_per_req": 1000.0 * entry_cpu / served}

def cmd_calibrate(a) -> int:
    """Measure k, c_rep and r with fixed replicas at two steady loads."""
    import os
    import requests
    fleet = load_fleet(a.out_dir)
    if not ENV_JSON.is_file():
        raise FleetError(f"{ENV_JSON} missing: run write-local and render first")
    env = json.loads(ENV_JSON.read_text(encoding="utf-8"))
    if [c["name"] for c in env["clusters"]] != [c["name"] for c in fleet["clusters"]]:
        raise FleetError("test-env.json is not this fleet: run write-local and render first")
    pool = fleet["app_pool"]
    model = services_model()
    entry = model["services"][model["entry"]]
    try:
        requests.get("http://localhost:9090/metrics", timeout=2)
        dmos_up = True
    except requests.RequestException:
        dmos_up = False
    if dmos_up:
        raise FleetError("DMOS is running (localhost:9090): stop it, the calibration needs "
                         "fixed replicas")
    sys.path.insert(0, str(REPO))
    from src.metrics.prometheus_client import PrometheusClient  # noqa: E402
    def entering() -> Dict[str, float]:
        busy = {}
        for c, e in zip(fleet["clusters"], env["clusters"]):
            q = PrometheusClient.ingress_queries(match=c["ingress_demand_match"])[0]
            busy[c["name"]] = round(prom_value(e["prometheus_url"], q), 1)
        return busy

    # The rates are averaged over a few minutes: traffic that just stopped (the
    # probe of `check`) still shows for a while, so wait for it to fade.
    deadline = time.time() + a.wait_idle
    busy = entering()
    while any(v > 1.0 for v in busy.values()):
        if time.time() >= deadline:
            raise FleetError(f"traffic already entering the fleet {busy}: calibrate on an "
                             f"idle fleet (waited {a.wait_idle} s)")
        print(f"  traffic still entering the fleet {busy}: waiting for an idle fleet ...")
        time.sleep(20)
        busy = entering()

    n, F = len(fleet["clusters"]), a.replicas
    cap_F = n * F * entry["capacity"]
    r_low = a.rate_low or int(0.25 * cap_F)
    r_high = a.rate_high or int(0.5 * cap_F)
    if r_high <= r_low:
        raise FleetError("--rate-high must be above --rate-low")

    def node_cpu(window, at=None):
        return {c["name"]: prom_value(prom_base(c), node_cpu_query(c, pool, window), at)
                for c in fleet["clusters"]}

    print("  idle with 1 replica per service: app-node CPU (last 3 min) and memory")
    idle1 = node_cpu("3m")
    mem0 = {c["name"]: prom_value(prom_base(c), node_mem_query(c, pool)) for c in fleet["clusters"]}
    for x in idle1:
        print(f"    {x:<20} C0={idle1[x]:.2f} core  M0={mem0[x]:.1f} GB")

    services = [model["services"][x]["deployment"] for x in [model["entry"]] + model["co_scaled"]]
    before = {c["name"]: {d["metadata"]["name"]: d["spec"].get("replicas", 1)
                          for d in kjson(c["kubeconfig"], "-n", NAMESPACE, "get", "deploy")["items"]}
              for c in fleet["clusters"]}
    limits = {}
    for d in kjson(fleet["clusters"][0]["kubeconfig"], "-n", NAMESPACE, "get", "deploy")["items"]:
        lim = ((d["spec"]["template"]["spec"]["containers"][0].get("resources") or {})
               .get("limits") or {}).get("cpu")
        if lim:
            limits[d["metadata"]["name"]] = cpu_cores(lim)
    entry_limit = limits.get(entry["deployment"])
    if not entry_limit:
        raise FleetError(f"{entry['deployment']} has no CPU limit: r cannot be calibrated")

    workdir = Path(a.out_dir).expanduser() / "calibration"
    workdir.mkdir(parents=True, exist_ok=True)
    phases = "180,210,390,400,410"    # low 0-180 s, ramp, high 210-390 s
    entry_q = (f'sum(rate(hubble_http_requests_total{{reporter="server",'
               f'destination_workload="{entry["deployment"]}"}}[120s]))')
    entry_cpu_q = (f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",'
                   f'pod=~"{entry["deployment"]}-.*",container!="",container!="POD"}}[120s]))')
    try:
        print(f"  scaling {', '.join(services)} to {F} replicas on every cluster")
        for c in fleet["clusters"]:
            kubectl(c["kubeconfig"], "-n", NAMESPACE, "scale", "deploy", *services, f"--replicas={F}")
        for c in fleet["clusters"]:
            for d in services:
                kubectl(c["kubeconfig"], "-n", NAMESPACE, "rollout", "status", f"deploy/{d}",
                        "--timeout=180s", timeout=200)
        print("  idle with the scaled replicas: waiting 150 s, then 2 min of measurement")
        time.sleep(150)
        idleF = node_cpu("2m")
        k6env = dict(os.environ, DMOS_RUN_LABEL="calibration", DMOS_BASELINE_RPS=str(r_low),
                     DMOS_PEAK_RPS=str(r_high), DMOS_PHASE_TIMES=phases,
                     DMOS_TEST_ENV=ENV_JSON.as_posix())
        for i in range(1, n + 1):
            k6env[f"DMOS_INGRESS_W{i}"] = "1"
            k6env[f"DMOS_SPIKE_W{i}"] = "1"
        print(f"  k6: {r_low} then {r_high} req/s spread evenly over {n} clusters (~7 min, "
              f"measured over 60-180 s and 270-390 s), log in {workdir / 'k6_stdout.txt'}")
        t0 = time.time()
        with open(workdir / "k6_stdout.txt", "w", encoding="utf-8") as out:
            p = subprocess.run(["k6", "run", "--quiet",
                                (REPO / "experiments" / "k6_flash_crowd.js").as_posix()],
                               cwd=workdir, env=k6env, stdout=out, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            raise FleetError(f"k6 exited {p.returncode} (see {workdir / 'k6_stdout.txt'})")
        time.sleep(20)
        low, high, thr = {}, {}, {}
        for c in fleet["clusters"]:
            b, x = prom_base(c), c["name"]
            for store, at in ((low, t0 + 180), (high, t0 + 390)):
                store[x] = {"cpu": prom_value(b, node_cpu_query(c, pool, "120s"), at),
                            "served": prom_value(b, entry_q, at),
                            "entry_cpu": prom_value(b, entry_cpu_q, at)}
            thr[x] = prom_max(b, throttle_query("120s"), t0 + 390)
    finally:
        print("  restoring the replicas")
        for c in fleet["clusters"]:
            for d, r in before[c["name"]].items():
                if d in services:
                    kubectl(c["kubeconfig"], "-n", NAMESPACE, "scale", "deploy", d, f"--replicas={r}")
    served = sum(v["served"] for v in high.values())
    if served < 0.5 * r_high:
        raise FleetError(f"served {served:.0f} req/s for {r_high} sent: measurement not usable")
    res = calibration_result(idle1, idleF, F, low, high, entry_limit, a.u_pod)
    for c in fleet["clusters"]:
        x = c["name"]
        print(f"    {x:<20} idle {idle1[x]:.2f} -> {idleF[x]:.2f} core with {F} replicas | "
              f"low {low[x]['served']:.0f} req/s {low[x]['cpu']:.2f} core | "
              f"high {high[x]['served']:.0f} req/s {high[x]['cpu']:.2f} core | "
              f"max throttling {100 * thr[x]:.0f}%")
    print(f"  k     = {res['k_mcore']:.1f} mcore/req   (marginal app-node CPU per request, Cilium included)")
    print(f"  c_rep = {res['c_rep']:.2f} core          (idle cost of one more replica set)")
    print(f"  r     = {res['r_entry']:.0f} req/s         ({entry['deployment']}: "
          f"{res['entry_mcore_per_req']:.1f} mcore/req, limit {entry_limit:g} core, "
          f"target {a.u_pod:g}; services.yaml says {entry['capacity']:g})")
    worst = max(thr.values())
    if worst > 0.25:
        print(f"  ! a pod was throttled {100 * worst:.0f}% of the time at the high load: "
              f"if it is not a gRPC backend with pinned connections, lower --rate-high")
    calib = {"version": 2, "k_mcore": round(res["k_mcore"], 2), "c_rep": round(res["c_rep"], 3),
             "r_entry": round(res["r_entry"], 1), "entry_mcore_per_req": round(res["entry_mcore_per_req"], 2),
             "u_pod": a.u_pod, "source": f"calibrated {dt.date.today().isoformat()}",
             "rates": [r_low, r_high], "replicas": F, "max_throttling": round(worst, 3),
             "clusters": {x: {"C0": round(idle1[x], 3), "C0_scaled": round(idleF[x], 3),
                              "M0": round(mem0[x], 2), "low": low[x], "high": high[x]}
                          for x in idle1}}
    path = Path(a.out_dir).expanduser() / "calibration.json"
    path.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    print(f"  saved {path}")
    return 0

def cmd_write_local(a) -> int:
    fleet = load_fleet(a.out_dir)
    baselines = {}
    for b in a.baseline or []:
        k, _, v = b.partition("=")
        baselines[k] = float(v)
    regions = a.regions.split(",") if a.regions else fleet.get("regions")
    plan = region_plan(len(fleet["clusters"]), regions, baselines or fleet.get("baselines") or {})
    fleet["regions"] = [p["region"] for p in plan]
    fleet["baselines"] = {p["region"]: p["baseline"] for p in plan}
    (Path(a.out_dir).expanduser() / "fleet.json").write_text(json.dumps(fleet, indent=2),
                                                             encoding="utf-8")
    ports = {c["name"]: gateway_port(c["kubeconfig"]) for c in fleet["clusters"]}
    caps = None
    calib_file = Path(a.out_dir).expanduser() / "calibration.json"
    calib = json.loads(calib_file.read_text(encoding="utf-8")) if calib_file.is_file() else None
    if a.no_caps:
        pass
    elif a.k is not None:
        if not calib or not calib.get("clusters"):
            raise FleetError("--k needs the idle measurements of a calibration (run calibrate)")
        caps = compute_caps(fleet, {**calib, "k_mcore": a.k, "source": "given with --k"},
                            a.u_cpu, a.u_mem)
    elif calib:
        caps = compute_caps(fleet, calib, a.u_cpu, a.u_mem)
    else:
        print("  ! no calibration.json: replica caps from services.yaml (run calibrate, then "
              "write-local again)")
    ssh = {"user": "ubuntu", "key": "~/.ssh/dmos_nodes"}
    if LOCAL_YAML.is_file():
        old = yaml.safe_load(LOCAL_YAML.read_text(encoding="utf-8")) or {}
        ssh = old.get("ssh") or ssh
    ssh = {"user": a.ssh_user or ssh.get("user", "ubuntu"), "key": a.ssh_key or ssh.get("key")}
    title = f"Kubephos fleet '{fleet['name']}' ({len(fleet['clusters'])} clusters)"
    text, _ = build_local(fleet, plan, ports, ssh, title, caps, netem=a.netem)
    out = Path(a.out_dir).expanduser()
    (out / f"local.{fleet['name']}.yaml").write_text(text, encoding="utf-8")
    if LOCAL_YAML.is_file() and LOCAL_YAML.read_text(encoding="utf-8") != text:
        bak = out / f"local.backup-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.yaml"
        shutil.copy2(LOCAL_YAML, bak)
        print(f"  previous config/local.yaml saved as {bak}")
    LOCAL_YAML.write_text(text, encoding="utf-8")
    for c, reg in zip(fleet["clusters"], plan):
        cap = (caps or {}).get(c["name"])
        cap_txt = (f"  cap {cap['service']} {cap['cap']} (cpu {cap['cap_cpu']}, mem {cap['cap_mem']})"
                   if cap else "")
        print(f"  {c['name']:<20} {reg['region']:<6} ingress_port {ports[c['name']]:<6} "
              f"demand by {c['ingress_demand_match']:<9} {c['cpu_cores']} vCPU "
              f"{c['memory_gb']} GB{cap_txt}")
    print(f"  config/local.yaml written (copy in {out / ('local.' + fleet['name'] + '.yaml')})")
    return 0


# ── check ───────────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.errors, self.lock = 0, threading.Lock()

    def ok(self, msg):
        print(f"  ok  {msg}")

    def warn(self, msg):
        print(f"  !   {msg}")

    def fail(self, msg):
        with self.lock:
            self.errors += 1
        print(f"  x   {msg}")


def check_cluster(c: Dict[str, Any], env: Dict[str, Any], fleet: Dict[str, Any], rep: Report):
    n, kc = c["name"], c["kubeconfig"]
    lines = []
    try:
        cm = kjson(kc, "-n", "kube-system", "get", "cm", "cilium-config").get("data") or {}
        miss = hubble_missing(cm.get("hubble-metrics", ""))
        lines.append(("fail", f"{n}: Hubble httpV2 lacks {miss}") if miss
                     else ("ok", f"{n}: Hubble httpV2 labels"))
        nodes = node_summary(kjson(kc, "get", "nodes"), fleet["app_pool"])
        app = set(nodes["app_nodes"])
        deps = kjson(kc, "-n", NAMESPACE, "get", "deploy").get("items") or []
        if not deps:
            lines.append(("fail", f"{n}: no Deployment in {NAMESPACE}"))
        unavailable = [d["metadata"]["name"] for d in deps
                       if (d["spec"].get("replicas") or 0) > 0
                       and (d["status"].get("availableReplicas") or 0) < d["spec"]["replicas"]]
        lg = [d["spec"].get("replicas") for d in deps if d["metadata"]["name"] == "loadgenerator"]
        pods = kjson(kc, "-n", NAMESPACE, "get", "pods").get("items") or []
        off_pool = sorted({p["spec"].get("nodeName") for p in pods
                           if p["spec"].get("nodeName") and p["spec"]["nodeName"] not in app})
        lines.append(("fail", f"{n}: Deployments not available: {unavailable}") if unavailable
                     else ("ok", f"{n}: {len(deps)} Deployments available"))
        lines.append(("fail", f"{n}: pods outside the app pool on {off_pool}") if off_pool
                     else ("ok", f"{n}: pods only on the app pool"))
        if lg and lg[0]:
            lines.append(("warn", f"{n}: loadgenerator has {lg[0]} replica(s): "
                                  f"experiments/clean_reset.sh sets it to 0"))
        gw = kjson(kc, "-n", NAMESPACE, "get", "gateway", GATEWAY)
        prog = any(x.get("type") == "Programmed" and x.get("status") == "True"
                   for x in gw.get("status", {}).get("conditions") or [])
        lines.append(("ok", f"{n}: Gateway Programmed") if prog
                     else ("fail", f"{n}: Gateway {GATEWAY} not Programmed"))
        port = gateway_port(kc)
        ec = next((x for x in env["clusters"] if x["name"] == n), None)
        if ec is None:
            lines.append(("fail", f"{n}: not in .cache/test-env.json (run write-local + render)"))
        else:
            want = f"http://{c['ip']}:{port}"
            if ec["ingress_url"] != want:
                lines.append(("fail", f"{n}: ingress_url {ec['ingress_url']} != Gateway {want}"))
            code, _ = http_get(ec["ingress_url"] + "/", timeout=10)
            lines.append(("ok", f"{n}: Gateway {ec['ingress_url']} -> 200") if code == 200
                         else ("fail", f"{n}: Gateway {ec['ingress_url']} -> HTTP {code}"))
            prom = ec["prometheus_url"]
            code, body = http_get(f"{prom}/api/v1/targets?state=active", timeout=10)
            if code != 200:
                lines.append(("fail", f"{n}: Prometheus {prom} -> HTTP {code}"))
            else:
                t = [x for x in json.loads(body)["data"]["activeTargets"]
                     if "hubble" in x.get("scrapePool", "")]
                pools = sorted({x["scrapePool"] for x in t})
                down = [x["labels"].get("instance") for x in t if x.get("health") != "up"]
                if len(pools) != 1:
                    lines.append(("fail", f"{n}: Hubble scraped by {len(pools)} pools {pools} "
                                          f"(must be 1, otherwise rates are counted twice)"))
                elif down:
                    lines.append(("fail", f"{n}: Hubble targets down: {down}"))
                else:
                    lines.append(("ok", f"{n}: Hubble scraped once ({len(t)} targets up)"))
        remote, ready = mesh_status(c["ip"])
        if remote is None:
            lines.append(("warn", f"{n}: ClusterMesh not verified (agent metrics unreachable)"))
        elif remote != len(fleet["clusters"]) - 1 or any(float(v) != 1 for v in ready.values()):
            lines.append(("fail", f"{n}: ClusterMesh {remote} peers, readiness {ready}"))
        else:
            lines.append(("ok", f"{n}: ClusterMesh {remote}/{len(fleet['clusters']) - 1} peers ready"))
    except (FleetError, subprocess.TimeoutExpired, KeyError, ValueError) as e:
        lines.append(("fail", f"{n}: {e}"))
    for level, msg in lines:
        getattr(rep, level)(msg)


def probe_demand(env: Dict[str, Any], rep: Report, seconds: int = 45, rps: float = 6.0):
    """Traffic to the FIRST cluster only: DMOS must measure it there (client side)
    and the other clusters' frontends must serve part of it (global service)."""
    sys.path.insert(0, str(REPO))
    from src.metrics.prometheus_client import PrometheusClient  # noqa: E402
    import requests

    cfg = yaml.safe_load(LOCAL_YAML.read_text(encoding="utf-8"))
    match = {c["name"]: c.get("ingress_demand_match", "workload") for c in cfg["clusters"]}

    def client_side(c):
        q = PrometheusClient.ingress_queries(match=match.get(c["name"], "workload"))[0]
        r = PrometheusClient(c["prometheus_url"], cluster_name=c["name"], test_connection=False).query(q)
        return float(r["result"][0]["value"][1]) if r and r.get("result") else 0.0

    # Never disturb a run: no probe while traffic is already entering the fleet
    busy = {c["name"]: round(client_side(c), 1) for c in env["clusters"]}
    if any(v > 1.0 for v in busy.values()):
        rep.warn(f"probe skipped: traffic already entering the fleet {busy} "
                 f"(a run in progress, or the loadgenerator?)")
        return

    first = env["clusters"][0]
    print(f"  probe: ~{rps:.0f} req/s to {first['name']} for {seconds} s, then 25 s for Prometheus")
    stop = time.time() + seconds
    sent = [0]

    def worker():
        s = requests.Session()
        while time.time() < stop:
            try:
                s.get(first["ingress_url"] + "/", timeout=5)
                sent[0] += 1
            except Exception:  # noqa: BLE001
                pass
            time.sleep(4 / rps)
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(25)
    for i, c in enumerate(env["clusters"]):
        q_client, q_server = PrometheusClient.ingress_queries(match=match.get(c["name"], "workload"))
        prom = PrometheusClient(c["prometheus_url"], cluster_name=c["name"], test_connection=False)

        def value(q):
            r = prom.query(q)
            return float(r["result"][0]["value"][1]) if r and r.get("result") else 0.0
        client, served = value(q_client), value(q_server)
        if i == 0:
            if client > 0.3 * sent[0] / seconds:
                rep.ok(f"{c['name']}: demand measured client-side {client:.1f} req/s "
                       f"(sent {sent[0] / seconds:.1f})")
            else:
                rep.fail(f"{c['name']}: client-side demand {client:.1f} req/s for "
                         f"{sent[0] / seconds:.1f} sent: DMOS would use the server-side fallback")
        else:
            if client > 0.5:
                rep.warn(f"{c['name']}: client-side demand {client:.1f} req/s with no traffic "
                         f"sent there (loadgenerator or other clients running?)")
            if served > 0:
                rep.ok(f"{c['name']}: serves {served:.1f} req/s entered elsewhere (global frontend)")
            else:
                rep.warn(f"{c['name']}: serves nothing of the traffic entered in {first['name']}: "
                         f"is the frontend a global service with affinity none?")


def cmd_check(a) -> int:
    fleet = load_fleet(a.out_dir)
    if not ENV_JSON.is_file():
        raise FleetError(f"{ENV_JSON} missing: run write-local and scripts/render-runtime-files.sh")
    env = json.loads(ENV_JSON.read_text(encoding="utf-8"))
    names = [c["name"] for c in fleet["clusters"]]
    if [c["name"] for c in env["clusters"]] != names:
        raise FleetError(f"test-env.json clusters {[c['name'] for c in env['clusters']]} != "
                         f"fleet {names}: run write-local + render")
    rep = Report()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda c: check_cluster(c, env, fleet, rep), fleet["clusters"]))
    if a.probe and len(names) > 1:
        probe_demand(env, rep)
    print(f"  {'ALL OK' if rep.errors == 0 else str(rep.errors) + ' problem(s)'}")
    return 1 if rep.errors else 0


# ── compare ─────────────────────────────────────────────────────────────────

def run_window(summary: Dict[str, Any]) -> Optional[Tuple[float, List[int]]]:
    """(epoch of the k6 start, phase ends in s) of a run, from its k6 summary:
    timestamp_iso is written when k6 ends, i.e. at the last phase end."""
    ts = summary.get("timestamp_iso")
    phases = (summary.get("config") or {}).get("phase_times_s")
    if not ts or not phases:
        return None
    if isinstance(phases, str):
        phases = [int(x) for x in phases.split(",")]
    end = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    return end - phases[-1], [int(x) for x in phases]


def throttle_query(window: str) -> str:
    """Worst share of throttled CFS periods of an application pod."""
    sel = 'namespace="' + NAMESPACE + '"'
    return ("max(sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{" + sel + "}[" + window
            + "])) / sum by (pod) (rate(container_cpu_cfs_periods_total{" + sel + "}[" + window + "])))")


def max_replicas_query(deployment: str, window: str) -> str:
    return ('max_over_time(kube_deployment_status_replicas{namespace="' + NAMESPACE
            + '",deployment="' + deployment + '"}[' + window + '])')


def peak_node_metrics(fleet: Dict[str, Any], t0: float, phases: List[int]) -> Dict[str, Dict[str, float]]:
    """Per cluster, over the peak phase: app-node CPU used (cores) and the
    worst CPU throttling of an application pod; over the whole run: the max
    replicas of the entry service (kube-state-metrics, also without DMOS)."""
    start, end = phases[1], phases[2]
    w = f"{end - start}s"
    entry = services_model()["services"][services_model()["entry"]]["deployment"]
    out = {}
    for c in fleet["clusters"]:
        b = prom_base(c)
        try:
            cpu = prom_value(b, node_cpu_query(c, fleet["app_pool"], w), t0 + end)
            thr = prom_max(b, throttle_query(w), t0 + end)
            reps = prom_max(b, max_replicas_query(entry, f"{phases[-1]}s"), t0 + phases[-1])
        except Exception as e:  # noqa: BLE001 - Prometheus down / data expired
            print(f"  ! {c['name']}: node metrics not available ({e})")
            continue
        out[c["name"]] = {"cpu": cpu, "cpu_pct": 100 * cpu / c["cpu_cores"] if c["cpu_cores"] else 0,
                          "thr_pct": 100 * thr, "replicas": int(reps)}
    return out


def cmd_compare(a) -> int:
    fleet = None
    if a.out_dir:
        fleet = load_fleet(a.out_dir)
    rows = []
    for d in a.run_dirs:
        d = Path(d)
        s = d / "flash_crowd_summary.json"
        if not s.is_file():
            print(f"  ! {d}: flash_crowd_summary.json missing")
            continue
        summ = json.loads(s.read_text(encoding="utf-8"))
        live = d / "dmos_live.jsonl"
        reps = max_replicas(live.read_text(encoding="utf-8").splitlines()) if live.is_file() else {}
        nodes = {}
        win = run_window(summ)
        if fleet and win:
            nodes = peak_node_metrics(fleet, *win)
            if nodes:
                reps = {k: v["replicas"] for k, v in nodes.items()}
        rows.append((summ.get("run_label") or d.name, phase_metrics(summ), reps, nodes))
    if not rows:
        return 1
    phases = [p for p in ("baseline", "spike", "peak", "decay", "recovery") if p in rows[0][1]]
    print(f"  {'run':<22}" + "".join(f"{p + ' p95 ms':>18}{p + ' err%':>15}" for p in phases))
    for label, ph, _, _ in rows:
        cells = "".join(f"{(ph[p]['p95'] or 0):>18.0f}{100 * (ph[p]['err'] or 0):>15.2f}"
                        for p in phases)
        print(f"  {label:<22}{cells}")
    print("\n  max frontend replicas per cluster")
    for label, _, reps, _ in rows:
        print(f"  {label:<22}" + "  ".join(f"{k}={v}" for k, v in sorted(reps.items())))
    if any(r[3] for r in rows):
        print("\n  peak phase, per cluster: app-node CPU used (cores, % of the app pool) "
              "and worst pod CPU throttling")
        for label, _, _, nodes in rows:
            print(f"  {label:<22}" + "  ".join(
                f"{k}: {v['cpu']:.1f} ({v['cpu_pct']:.0f}%) thr {v['thr_pct']:.0f}%"
                for k, v in sorted(nodes.items())))
    elif not fleet:
        print("\n  (--out-dir <fleet dir> adds app-node CPU and throttling per cluster)")
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", required=True, help="where kubeconfigs, fleet.json, "
                        "local.<name>.yaml and backups go (outside the repository)")
    p = sub.add_parser("discover", parents=[common])
    p.add_argument("--kubeconfig", required=True, help="kubeconfig from Kubephos (one context per cluster)")
    p.add_argument("--name", required=True, help="fleet name (file names, local.yaml title)")
    p.add_argument("--contexts", help="comma-separated subset/order of the contexts (default: all, "
                   "in kubeconfig order = k6 weight order)")
    p.add_argument("--app-pool", default="app", help=f"value of {APP_POOL_LABEL} of the app nodes")
    sub.add_parser("hubble-fix", parents=[common])
    p = sub.add_parser("write-local", parents=[common])
    p.add_argument("--regions", help="comma-separated regions in cluster order "
                   f"(default: {','.join(DEFAULT_REGION_ORDER)} truncated to N)")
    p.add_argument("--baseline", action="append", help="REGION=gCO2/kWh for regions not in "
                   f"{sorted(REGIONS)} (repeatable)")
    p.add_argument("--k", type=float, help="mcore per request instead of the calibrated one "
                   "(the idle measurements of calibration.json are still used)")
    p.add_argument("--u-cpu", type=float, default=DEFAULT_U_CPU, help="target CPU utilisation")
    p.add_argument("--u-mem", type=float, default=DEFAULT_U_MEM, help="target memory utilisation")
    p.add_argument("--no-caps", action="store_true", help="no max_replicas in local.yaml "
                   "(services.yaml caps)")
    p.add_argument("--netem", action="store_true", help="emulated geography: the delays of the "
                   "reference fleet per region (DE 0, FR 30, PL 50, US-VA 100, SG 200, NO 40 ms)")
    p.add_argument("--ssh-user", help="user for SSH to the nodes (netem); default: kept from "
                   "config/local.yaml")
    p.add_argument("--ssh-key", help="private key for SSH to the nodes (netem); default: kept "
                   "from config/local.yaml")
    p = sub.add_parser("calibrate", parents=[common], help="measure k on the fleet (~8 min; "
                       "scales Deployments and sends load; DMOS must be stopped)")
    p.add_argument("--replicas", type=int, default=2, help="replicas of the entry service and "
                   "of the co-scaled backends during the measurement (default 2)")
    p.add_argument("--rate-low", type=int, help="first load, req/s (default: 25%% of the "
                   "capacity of those replicas)")
    p.add_argument("--rate-high", type=int, help="second load, req/s (default: 50%%)")
    p.add_argument("--u-pod", type=float, default=0.7, help="target CPU of one entry pod, as a "
                   "share of its CPU limit, for r (default 0.7)")
    p.add_argument("--wait-idle", type=int, default=300, help="seconds to wait for traffic "
                   "that just stopped to fade before refusing (default 300)")
    p = sub.add_parser("check", parents=[common])
    p.add_argument("--probe", action="store_true", help="also send traffic to the first Gateway "
                   "and check the demand DMOS measures (run it after clean_reset)")
    p = sub.add_parser("compare")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out-dir", help="fleet directory (fleet.json): adds app-node CPU and "
                   "pod throttling of the peak phase, read from the clusters' Prometheus")
    a = ap.parse_args(argv)
    try:
        return {"discover": cmd_discover, "hubble-fix": cmd_hubble_fix, "calibrate": cmd_calibrate,
                "write-local": cmd_write_local, "check": cmd_check,
                "compare": cmd_compare}[a.cmd](a)
    except FleetError as e:
        print(f"  x   {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
