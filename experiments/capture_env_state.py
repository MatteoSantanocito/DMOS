#!/usr/bin/env python3
"""
capture_env_state.py — snapshot of the TESTBED state around a run.

A run is valid only if the testbed was in the intended state: netem shaping
the inter-cluster RTT, the Cilium Service annotations (global / affinity /
shared) the arm assumes, no stray HPA fighting DMOS, every cluster Ready.
None of this is visible in the k6 or DMOS outputs, so it is captured here,
before and after each run, into the run directory:

    env_state_pre.json   (--phase pre)
    env_state_post.json  (--phase post)

Per cluster (in parallel): Service annotations, Deployments (replicas,
readiness, image, resources), HPAs, cilium-config + Cilium image,
clustermesh-apiserver readiness, Nodes, netem (`tc qdisc` on the worker via
SSH, same criteria as preflight_runs.sh) and ping RTT from Prometheus.

Gate mode: --require-netem / --expect-affinity / --expect-no-hpa /
--expect-no-loadgenerator / --expect-no-karmada turn violations into exit code 3, so the harness
can refuse to start a run on a testbed in the wrong state.

USAGE
  python experiments/capture_env_state.py --phase pre --out RUN_DIR/env_state_pre.json \
      [--require-netem] [--expect-affinity none|local] [--affinity-services frontend] \
      [--expect-no-hpa] [--expect-no-loadgenerator] [--expect-no-karmada]

EXIT CODES
  0 = captured, expectations met     3 = captured, expectation violated
  1 = cannot run (missing test-env.json, bad arguments)

Tool overrides (used by the tests): DMOS_KUBECTL, DMOS_SSH are command
prefixes, e.g. DMOS_KUBECTL="python fake_kubectl.py".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_JSON = REPO_ROOT / ".cache" / "test-env.json"
NAMESPACE = "online-boutique"
NETEM_TOL_MS = 3.0            # same tolerance as preflight_runs.sh

# Annotation keys (modern + legacy spellings) that decide cross-cluster serving
ANN_GLOBAL = ("service.cilium.io/global", "io.cilium/global-service")
ANN_AFFINITY = ("service.cilium.io/affinity",)
ANN_SHARED = ("service.cilium.io/shared", "io.cilium/shared-service")


# ── Small helpers ───────────────────────────────────────────────────────────

def _cmd_prefix(env_var: str, default: str) -> List[str]:
    """Command prefix from env: a JSON list (robust on Windows paths) or a
    POSIX-quoted string; the bare tool name otherwise."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return [default]
    if raw.startswith("["):
        return [str(x) for x in json.loads(raw)]
    return shlex.split(raw)


def _run(cmd: List[str], timeout: float) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        # explicit UTF-8: the Windows default (cp1252) breaks on kubectl JSON
        return (p.returncode, p.stdout.decode("utf-8", errors="replace"),
                p.stderr.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except OSError as e:
        return 127, "", str(e)


def _first(ann: Dict[str, str], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in ann:
            return ann[k]
    return None


def resolve_path(p: str) -> str:
    """kubeconfig/ssh key path: ~ expansion, relative to the repo root."""
    p = os.path.expanduser(p or "")
    if p and not os.path.isabs(p) and not re.match(r"^[A-Za-z]:", p):
        p = str(REPO_ROOT / p)
    return p


# ── Parsers (pure functions, unit-tested) ───────────────────────────────────

_DELAY_RE = re.compile(r"delay\s+([0-9.]+)(us|ms|s)(?:\s+([0-9.]+)(us|ms|s))?")


def _to_ms(value: str, unit: str) -> float:
    v = float(value)
    return v / 1000.0 if unit == "us" else (v * 1000.0 if unit == "s" else v)


def parse_tc_qdisc(text: str) -> Dict[str, Any]:
    """Parse `tc qdisc show dev X`: is netem there, with which delay/jitter."""
    present = "netem" in (text or "")
    out: Dict[str, Any] = {"present": present, "delay_ms": None, "jitter_ms": None}
    m = _DELAY_RE.search(text or "")
    if present and m:
        out["delay_ms"] = round(_to_ms(m.group(1), m.group(2)), 3)
        if m.group(3):
            out["jitter_ms"] = round(_to_ms(m.group(3), m.group(4)), 3)
    return out


FILTER_MARK = "@@DMOS_TC_FILTERS@@"
_U32_MATCH_RE = re.compile(r"^\s*match\s+([0-9a-fA-F]{8})/([0-9a-fA-F]{8})\s+at\s+16\b")
_FLOWID_RE = re.compile(r"\bflowid\s+(\S+)")


def parse_tc_filters(text: str, unshaped_class: str = "1:2") -> List[str]:
    """Destination IPs that `tc filter show` sends to the class WITHOUT delay.

    scripts/setup-netem-and-ping.sh adds one u32 filter per exempt address
    (`match ip dst A/32 flowid 1:2`); tc prints it as `match <hex ip>/ffffffff
    at 16` under the line that carries the flowid."""
    ips: List[str] = []
    flow = None
    for line in (text or "").splitlines():
        m = _FLOWID_RE.search(line)
        if m:
            flow = m.group(1)
            continue
        m = _U32_MATCH_RE.match(line)
        if m and flow == unshaped_class and m.group(2).lower() == "ffffffff":
            v = int(m.group(1), 16)
            ip = ".".join(str((v >> s) & 0xFF) for s in (24, 16, 8, 0))
            if ip not in ips:
                ips.append(ip)
    return ips


def netem_verdict(expected_ms: float, parsed: Optional[Dict[str, Any]],
                  ssh_ok: bool, tol_ms: float = NETEM_TOL_MS) -> str:
    """
    ok          netem as expected (or none, for a 0 ms baseline cluster)
    missing     delay expected but no netem qdisc  -> run INVALID (reboot?)
    mismatch    netem present, delay off by more than tol_ms
    unexpected  baseline cluster (0 ms) but a netem qdisc is installed
    unparsable  netem present but delay not readable
    ssh_error   could not inspect the worker
    """
    if not ssh_ok or parsed is None:
        return "ssh_error"
    if not expected_ms:
        return "unexpected" if parsed["present"] else "ok"
    if not parsed["present"]:
        return "missing"
    if parsed["delay_ms"] is None:
        return "unparsable"
    return "ok" if abs(parsed["delay_ms"] - float(expected_ms)) <= tol_ms else "mismatch"


def summarize_services(svc_list: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Cilium-relevant view of `kubectl get svc -o json`."""
    out = {}
    for item in svc_list.get("items", []):
        meta = item.get("metadata", {})
        ann = meta.get("annotations") or {}
        cilium_ann = {k: v for k, v in ann.items() if "cilium" in k}
        is_global = (_first(ann, ANN_GLOBAL) or "").lower() == "true"
        affinity = _first(ann, ANN_AFFINITY)
        shared = _first(ann, ANN_SHARED)
        out[meta.get("name", "?")] = {
            "type": item.get("spec", {}).get("type"),
            "global": is_global,
            # Cilium default for a global service with no annotation is "none"
            "affinity": affinity if affinity is not None else ("none" if is_global else None),
            "affinity_annotated": affinity is not None,
            "shared": shared,
            "cilium_annotations": cilium_ann,
        }
    return out


def summarize_deployments(dep_list: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in dep_list.get("items", []):
        meta, spec, status = item.get("metadata", {}), item.get("spec", {}), item.get("status", {})
        containers = spec.get("template", {}).get("spec", {}).get("containers", [])
        main = next((c for c in containers if c.get("name") == "server"),
                    containers[0] if containers else {})
        out[meta.get("name", "?")] = {
            "spec_replicas": spec.get("replicas"),
            "replicas": status.get("replicas", 0),     # pods of the Deployment, not terminating
            "ready_replicas": status.get("readyReplicas", 0),
            "available_replicas": status.get("availableReplicas", 0),
            "image": main.get("image"),
            "resources": main.get("resources", {}),
            # Karmada: label on member objects it applies (karmada_baseline.py)
            "karmada_managed": (meta.get("labels") or {}).get("karmada.io/managed") == "true",
            "revision": (meta.get("annotations") or {}).get("deployment.kubernetes.io/revision"),
            # Pod lifecycle (experiments/apply_pod_lifecycle.sh): whether a replica
            # removed by a scale-down drains before SIGTERM, and how fast a new
            # one is declared ready. Recorded since 31/10/2026: the 503 of the
            # cooldown and the latency bursts of the scaling actions depend on it.
            "lifecycle": pod_lifecycle(spec, main),
        }
    return out


def pod_lifecycle(spec: Dict[str, Any], main: Dict[str, Any]) -> Dict[str, Any]:
    pod = spec.get("template", {}).get("spec", {})
    pre = ((main.get("lifecycle") or {}).get("preStop") or {})
    pre_stop = None
    if "sleep" in pre:
        pre_stop = f"sleep {pre['sleep'].get('seconds')}s"
    elif "exec" in pre:
        pre_stop = "exec " + " ".join(pre["exec"].get("command") or [])[:80]
    elif "httpGet" in pre:
        pre_stop = "httpGet"
    rp = main.get("readinessProbe") or {}
    return {
        "termination_grace_s": pod.get("terminationGracePeriodSeconds"),
        "pre_stop": pre_stop,
        "min_ready_s": spec.get("minReadySeconds", 0),
        "readiness": ({k: rp.get(k) for k in ("initialDelaySeconds", "periodSeconds",
                                               "failureThreshold", "successThreshold")}
                      if rp else None),
    }


def summarize_hpas(hpa_list: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for item in hpa_list.get("items", []):
        spec = item.get("spec", {})
        out.append({
            "name": item.get("metadata", {}).get("name"),
            "target": spec.get("scaleTargetRef", {}).get("name"),
            "min": spec.get("minReplicas"),
            "max": spec.get("maxReplicas"),
            "metrics": spec.get("metrics", []),
        })
    return out


def summarize_nodes(node_list: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for item in node_list.get("items", []):
        st = item.get("status", {})
        ready = next((c.get("status") for c in st.get("conditions", [])
                      if c.get("type") == "Ready"), None)
        out.append({
            "name": item.get("metadata", {}).get("name"),
            "ready": ready == "True",
            "unschedulable": bool(item.get("spec", {}).get("unschedulable", False)),
            "allocatable": st.get("allocatable", {}),
            "kubelet": st.get("nodeInfo", {}).get("kubeletVersion"),
        })
    return out


# ── Collection ──────────────────────────────────────────────────────────────

class Collector:
    def __init__(self, env: Dict[str, Any], namespace: str, timeout: float,
                 skip_ssh: bool, skip_prometheus: bool):
        self.env = env
        self.ns = namespace
        self.timeout = timeout
        self.skip_ssh = skip_ssh
        self.skip_prometheus = skip_prometheus
        self.kubectl = _cmd_prefix("DMOS_KUBECTL", "kubectl")
        self.ssh = _cmd_prefix("DMOS_SSH", "ssh")
        ssh_cfg = env.get("ssh", {}) or {}
        self.ssh_user = ssh_cfg.get("user", "ubuntu")
        self.ssh_key = resolve_path(ssh_cfg.get("key", ""))

    def _kube_json(self, kc: str, args: List[str], errors: List[str], what: str):
        rc, out, err = _run(self.kubectl + ["--kubeconfig", kc, *args, "-o", "json"],
                            self.timeout)
        if rc != 0:
            errors.append(f"{what}: rc={rc} {err.strip()[:200]}")
            return None
        try:
            return json.loads(out)
        except ValueError:
            errors.append(f"{what}: invalid JSON from kubectl")
            return None

    def _netem(self, cl: Dict[str, Any]) -> Dict[str, Any]:
        cfg = cl.get("netem", {}) or {}
        expected = float(cfg.get("delay_ms") or 0)
        iface = cfg.get("interface") or ""
        res: Dict[str, Any] = {"expected_delay_ms": expected, "interface": iface,
                               "expected_jitter_ms": cfg.get("jitter_ms")}
        if self.skip_ssh:
            res["verdict"] = "skipped"
            return res
        if not iface:
            res["verdict"] = "no_interface"
            return res
        cmd = self.ssh + ["-n", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                          "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=6"]
        if self.ssh_key:
            cmd += ["-i", self.ssh_key]
        # interface "auto" (Kubephos, netem scope cluster): the interface is the one
        # of the default route of each node, so look at every device of the node.
        # The filters say which destinations are exempt from the delay (the
        # nodes of the cluster, its storage and the k6 client, see
        # scripts/setup-netem-and-ping.sh).
        if iface == "auto":
            tc = ("tc qdisc show && echo " + FILTER_MARK + " && { for d in $(ls /sys/class/net); do "
                  "tc filter show dev $d parent 1: 2>/dev/null; done; true; }")
        else:
            tc = (f"tc qdisc show dev {iface} && echo {FILTER_MARK} && "
                  f"{{ tc filter show dev {iface} parent 1: 2>/dev/null; true; }}")
        cmd += [f"{self.ssh_user}@{cl.get('ip')}", tc]
        rc, out, err = _run(cmd, self.timeout)
        ssh_ok = rc == 0
        qdisc_text, _, filter_text = (out or "").partition(FILTER_MARK)
        parsed = parse_tc_qdisc(qdisc_text) if ssh_ok else None
        res.update({"raw": qdisc_text.strip() if ssh_ok else err.strip()[:200],
                    **(parsed or {"present": None, "delay_ms": None, "jitter_ms": None})})
        res["unshaped_ips"] = parse_tc_filters(filter_text) if ssh_ok and FILTER_MARK in out else None
        res["verdict"] = netem_verdict(expected, parsed, ssh_ok)
        return res

    def _ping_rtt(self, prom_url: str) -> Dict[str, Any]:
        if self.skip_prometheus or not prom_url:
            return {"skipped": True}
        q = "avg(ping_rtt_mean_seconds) by (target) * 1000"
        url = f"{prom_url.rstrip('/')}/api/v1/query?" + urllib.parse.urlencode({"query": q})
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return {s["metric"].get("target", "?"): round(float(s["value"][1]), 2)
                    for s in data.get("data", {}).get("result", [])}
        except Exception as e:   # unreachable Prometheus must not abort the capture
            return {"error": str(e)[:200]}

    def cluster(self, cl: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        name = cl.get("name", "?")
        kc = resolve_path(cl.get("kubeconfig_path", ""))
        errors: List[str] = []
        res: Dict[str, Any] = {"errors": errors}
        if not kc or not os.path.isfile(kc):
            errors.append(f"kubeconfig not found: {kc}")
        else:
            svc = self._kube_json(kc, ["-n", self.ns, "get", "svc"], errors, "services")
            dep = self._kube_json(kc, ["-n", self.ns, "get", "deploy"], errors, "deployments")
            hpa = self._kube_json(kc, ["-n", self.ns, "get", "hpa"], errors, "hpa")
            cm = self._kube_json(kc, ["-n", "kube-system", "get", "configmap", "cilium-config"],
                                 errors, "cilium-config")
            ds = self._kube_json(kc, ["-n", "kube-system", "get", "daemonset", "cilium"],
                                 errors, "cilium daemonset")
            cma = self._kube_json(kc, ["-n", "kube-system", "get", "deploy",
                                       "clustermesh-apiserver"], errors, "clustermesh-apiserver")
            nodes = self._kube_json(kc, ["get", "nodes"], errors, "nodes")
            data = (cm or {}).get("data", {}) or {}
            containers = ((ds or {}).get("spec", {}).get("template", {})
                          .get("spec", {}).get("containers", []))
            res.update({
                "services": summarize_services(svc) if svc else None,
                "deployments": summarize_deployments(dep) if dep else None,
                "hpa": summarize_hpas(hpa) if hpa else None,
                "cilium": {
                    "image": containers[0].get("image") if containers else None,
                    "endpoint_sync": data.get("clustermesh-enable-endpoint-sync"),
                    "kube_proxy_replacement": data.get("kube-proxy-replacement"),
                    "cluster_name": data.get("cluster-name"),
                    "cluster_id": data.get("cluster-id"),
                    "config": data,
                },
                "clustermesh_apiserver": None if cma is None else {
                    "ready_replicas": cma.get("status", {}).get("readyReplicas", 0),
                    "spec_replicas": cma.get("spec", {}).get("replicas"),
                },
                "nodes": summarize_nodes(nodes) if nodes else None,
            })
        res["netem"] = self._netem(cl)
        res["ping_rtt_ms"] = self._ping_rtt(cl.get("prometheus_url", ""))
        return name, res


# ── Verdicts ────────────────────────────────────────────────────────────────

def build_verdicts(clusters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    netem = {n: c.get("netem", {}).get("verdict") for n, c in clusters.items()}
    netem_bad = {n: v for n, v in netem.items() if v in ("missing", "ssh_error")}

    # affinity/global per service, compared across clusters
    per_service: Dict[str, Dict[str, Any]] = {}
    for n, c in clusters.items():
        for svc, s in (c.get("services") or {}).items():
            d = per_service.setdefault(svc, {"global": {}, "affinity": {}, "shared": {}})
            d["global"][n] = s["global"]
            d["affinity"][n] = s["affinity"]
            d["shared"][n] = s["shared"]
    affinity_summary = {}
    for svc, d in sorted(per_service.items()):
        vals = set(d["affinity"].values())
        affinity_summary[svc] = vals.pop() if len(vals) == 1 else "MIXED"

    endpoint_sync = {n: (c.get("cilium") or {}).get("endpoint_sync") for n, c in clusters.items()}
    images = {n: (c.get("cilium") or {}).get("image") for n, c in clusters.items()}
    hpa = {n: [h["name"] for h in (c.get("hpa") or [])] for n, c in clusters.items()}
    not_ready = {}
    for n, c in clusters.items():
        bad = [d for d, s in (c.get("deployments") or {}).items()
               if (s.get("ready_replicas") or 0) < (s.get("spec_replicas") or 0)]
        if bad:
            not_ready[n] = bad
    # Online Boutique's built-in load generator: internal traffic to the
    # frontend that is not part of the experiment's load. None = not deployed.
    loadgen = {n: ((c.get("deployments") or {}).get("loadgenerator") or {}).get("spec_replicas")
               for n, c in clusters.items()}
    karmada = {n: sorted(d for d, s in (c.get("deployments") or {}).items() if s.get("karmada_managed"))
               for n, c in clusters.items()}
    karmada = {n: v for n, v in karmada.items() if v}
    nodes_not_ready = {n: [x["name"] for x in (c.get("nodes") or []) if not x["ready"]]
                       for n, c in clusters.items()}
    nodes_not_ready = {n: v for n, v in nodes_not_ready.items() if v}
    errors = {n: c.get("errors") for n, c in clusters.items() if c.get("errors")}

    # Destinations exempt from the delay on each shaped cluster (None = not read)
    unshaped = {n: (c.get("netem") or {}).get("unshaped_ips") for n, c in clusters.items()
                if float((c.get("netem") or {}).get("expected_delay_ms") or 0) > 0}

    return {
        "netem": netem,
        "netem_unshaped_ips": unshaped,
        # Same severity as preflight_runs.sh: missing / unverifiable netem on a
        # shaped cluster invalidates the run; mismatch / unexpected are warnings.
        "netem_ok": not netem_bad,
        "netem_problems": {n: v for n, v in netem.items() if v not in ("ok", "skipped")},
        "affinity": affinity_summary,
        "affinity_by_cluster": {svc: d["affinity"] for svc, d in per_service.items()},
        "global_by_cluster": {svc: d["global"] for svc, d in per_service.items()},
        "endpoint_sync": endpoint_sync,
        "endpoint_sync_consistent": len(set(endpoint_sync.values())) <= 1,
        "cilium_images": images,
        "hpa": hpa,
        "any_hpa": any(hpa.values()),
        "deployments_not_ready": not_ready,
        "loadgenerator_replicas": loadgen,
        "loadgenerator_running": any((r or 0) > 0 for r in loadgen.values()),
        "nodes_not_ready": nodes_not_ready,
        "karmada_managed": karmada,
        "any_karmada_managed": bool(karmada),
        "collection_errors": errors,
    }


def check_expectations(verdicts: Dict[str, Any], require_netem: bool,
                       expect_affinity: Optional[str], expect_no_hpa: bool,
                       services: Optional[List[str]],
                       expect_no_loadgenerator: bool = False,
                       expect_no_karmada: bool = False,
                       expect_client_local: Optional[List[str]] = None,
                       clusters: Optional[Dict[str, Any]] = None) -> List[str]:
    violations: List[str] = []
    if clusters is not None:
        # every run starts from the reset: each Deployment at exactly 1 pod,
        # ready (the loadgenerator has its own gate)
        for n, c in clusters.items():
            deps = c.get("deployments")
            if not deps:
                violations.append(f"deployments of {n} not read: cannot verify the reset to 1 replica")
                continue
            for d, s in sorted(deps.items()):
                if d == "loadgenerator":
                    continue
                got = (s.get("spec_replicas"), s.get("replicas"), s.get("ready_replicas"))
                if got != (1, 1, 1):
                    violations.append(f"{d}@{n}: not reset to 1 replica "
                                      f"(spec/pods/ready = {got[0]}/{got[1]}/{got[2]})")
    if expect_client_local:
        # every shaped cluster must exempt the load generator: its users are
        # close to the ingress they enter (journal geography, 30/10/2026)
        for n, ips in (verdicts.get("netem_unshaped_ips") or {}).items():
            if verdicts["netem"].get(n) in ("missing", "skipped", "ssh_error"):
                continue
            if ips is None:
                violations.append(f"netem filters not readable on {n}: cannot verify that "
                                  f"the k6 client {','.join(expect_client_local)} is not delayed")
                continue
            missing = [ip for ip in expect_client_local if ip not in ips]
            if missing:
                violations.append(f"netem delays the k6 client {','.join(missing)} on {n} "
                                  f"(re-apply: bash scripts/setup-netem-and-ping.sh netem-only)")
    if require_netem:
        for n, v in verdicts["netem"].items():
            if v in ("missing", "ssh_error"):
                violations.append(f"netem {v} on {n}")
    if expect_affinity:
        for svc, by_cluster in verdicts["affinity_by_cluster"].items():
            if services and svc not in services:
                continue
            for n, a in by_cluster.items():
                if a != expect_affinity:
                    violations.append(f"{svc}@{n}: affinity={a} (expected {expect_affinity})")
    if expect_no_hpa and verdicts["any_hpa"]:
        violations.append("HPA present: " + json.dumps({n: h for n, h in verdicts["hpa"].items() if h}))
    if expect_no_loadgenerator and verdicts.get("loadgenerator_running"):
        running = {n: r for n, r in verdicts["loadgenerator_replicas"].items() if (r or 0) > 0}
        violations.append("Online Boutique loadgenerator running: " + json.dumps(running))
    if expect_no_karmada and verdicts.get("any_karmada_managed"):
        violations.append("Deployments managed by Karmada (label karmada.io/managed): "
                          + json.dumps(verdicts["karmada_managed"]))
    return violations


# ── Main ────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", choices=("pre", "post", "adhoc"), default="adhoc")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--env-json", default=str(DEFAULT_ENV_JSON))
    ap.add_argument("--namespace", default=NAMESPACE)
    ap.add_argument("--timeout", type=float, default=20.0, help="per-command timeout (s)")
    ap.add_argument("--require-netem", action="store_true",
                    help="exit 3 if netem is missing / unverifiable on a shaped cluster")
    ap.add_argument("--expect-affinity", choices=("none", "local", "remote"),
                    help="exit 3 if a checked Service has a different affinity")
    ap.add_argument("--affinity-services", default="",
                    help="comma list of Services checked by --expect-affinity (default: all global)")
    ap.add_argument("--expect-no-hpa", action="store_true", help="exit 3 if any HPA exists")
    ap.add_argument("--expect-no-loadgenerator", action="store_true",
                    help="exit 3 if the Online Boutique loadgenerator has replicas > 0")
    ap.add_argument("--expect-no-karmada", action="store_true",
                    help="exit 3 if a Deployment is managed by Karmada (label karmada.io/managed=true)")
    ap.add_argument("--expect-client-local", default="",
                    help="comma list of load-generator IPs that netem must NOT delay on "
                         "any shaped cluster (exit 3 otherwise)")
    ap.add_argument("--expect-reset", action="store_true",
                    help="exit 3 unless every Deployment (but the loadgenerator) has exactly "
                         "1 replica, 1 pod and 1 ready pod on every cluster")
    ap.add_argument("--skip-ssh", action="store_true", help="do not inspect netem")
    ap.add_argument("--skip-prometheus", action="store_true", help="do not query ping RTT")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    env_path = Path(args.env_json)
    try:
        env_text = env_path.read_text(encoding="utf-8")
        env = json.loads(env_text)
    except (OSError, ValueError) as e:
        print(f"[capture_env_state] cannot read {env_path}: {e}", file=sys.stderr)
        return 1

    t0 = time.time()
    col = Collector(env, args.namespace, args.timeout, args.skip_ssh, args.skip_prometheus)
    cluster_list = env.get("clusters", [])
    with ThreadPoolExecutor(max_workers=max(1, len(cluster_list))) as ex:
        clusters = dict(ex.map(col.cluster, cluster_list))

    verdicts = build_verdicts(clusters)
    checked = [s for s in args.affinity_services.split(",") if s] or None
    if checked is None and args.expect_affinity:
        # default: every Service that is global somewhere
        checked = [svc for svc, g in verdicts["global_by_cluster"].items() if any(g.values())]
    violations = check_expectations(verdicts, args.require_netem, args.expect_affinity,
                                    args.expect_no_hpa, checked, args.expect_no_loadgenerator,
                                    args.expect_no_karmada,
                                    [x for x in args.expect_client_local.split(",") if x] or None,
                                    clusters if args.expect_reset else None)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "phase": args.phase,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - t0, 1),
        "namespace": args.namespace,
        "test_env": {"path": str(env_path),
                     "sha256": hashlib.sha256(env_text.encode("utf-8")).hexdigest()[:16]},
        "clusters": clusters,
        "verdicts": verdicts,
        "expectations": {
            "require_netem": args.require_netem,
            "expect_affinity": args.expect_affinity,
            "affinity_services": checked,
            "expect_no_hpa": args.expect_no_hpa,
            "expect_no_loadgenerator": args.expect_no_loadgenerator,
            "expect_no_karmada": args.expect_no_karmada,
            "expect_client_local": [x for x in args.expect_client_local.split(",") if x],
            "expect_reset": args.expect_reset,
            "violations": violations,
            "met": not violations,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)

    if not args.quiet:
        v = verdicts
        print(f"[env {args.phase}] netem: " + ", ".join(f"{n.split('-')[-1]}={x}"
                                                      for n, x in v["netem"].items()))
        print(f"[env {args.phase}] affinity: " + ", ".join(f"{s}={a}" for s, a in v["affinity"].items()))
        print(f"[env {args.phase}] endpoint-sync: {sorted(set(map(str, v['endpoint_sync'].values())))}"
              f" | HPA: {'yes' if v['any_hpa'] else 'none'}"
              f" | loadgenerator: {'RUNNING' if v['loadgenerator_running'] else 'off'}"
              f" | Karmada: {'MANAGES ' + str(sum(map(len, v['karmada_managed'].values()))) if v['any_karmada_managed'] else 'none'}"
              f" | errors: {sum(len(e) for e in v['collection_errors'].values())}")
        for x in violations:
            print(f"[env {args.phase}] VIOLATION: {x}")
    return 3 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
