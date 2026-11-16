#!/usr/bin/env python3
"""
hpa_baseline.py — the Kubernetes HPA baseline (arm `hpa-none`): one standard
HorizontalPodAutoscaler per service and per cluster, each cluster on its own,
scaling on CPU. DMOS does not run in this arm; nothing of DMOS changes.

What the HPAs are (from config/services.yaml, no tuning for this arm):
  services      the ones DMOS manages: application.entry_service and
                application.co_scaled_services
  min / max     the per-cluster minimum and maximum of each service (HPA
                minReplicas is at least 1)
  metric        CPU utilization, target autoscaling.target_cpu_percent (70)
  behaviour     Kubernetes defaults (immediate scale-up, 300 s scale-down
                stabilization)
  data plane    unchanged: the frontend stays a Cilium global service with
                affinity "none", so every cluster's replicas serve the pool

Every HPA carries the label app.kubernetes.io/managed-by=dmos-hpa-baseline:
`delete` removes those and only those.

Commands (the harness calls them, experiments/lib/run_common.sh):
  apply  --out FILE   create the HPAs on every cluster, check the metrics API
                      and that every HPA exists; record in FILE (exit 3 if not)
  verify --out FILE   after the load: every HPA must have read its CPU metric
                      (condition ScalingActive=True); appended to FILE
  delete [--out FILE] remove the baseline HPAs from every cluster (idempotent)
  show                print the manifests without touching the clusters

kubectl can be replaced by DMOS_KUBECTL (as in capture_env_state.py).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from capture_env_state import _cmd_prefix, resolve_path  # noqa: E402

LABEL_KEY = "app.kubernetes.io/managed-by"
LABEL_VALUE = "dmos-hpa-baseline"
SELECTOR = f"{LABEL_KEY}={LABEL_VALUE}"
NAME_PREFIX = "dmos-baseline-"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(config_dir: Optional[Path] = None):
    from loguru import logger
    from src.utils.config_loader import ConfigLoader
    logger.remove()
    logger.disable("src")
    return ConfigLoader(str(config_dir or REPO / "config"))


def hpa_specs(cfg) -> Dict[str, List[Dict[str, Any]]]:
    """{cluster: [HPA manifest, ...]} for the services DMOS manages."""
    app = cfg.application
    services = [app.entry_service] + list(app.co_scaled_services)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cluster in cfg.get_all_clusters():
        items = []
        for name in services:
            svc = cfg.get_service(name)
            if svc is None:
                continue
            target = int((svc.autoscaling or {}).get("target_cpu_percent", 70))
            lo = max(1, int(svc.get_min_replicas_for_cluster(cluster)))
            hi = max(lo, int(svc.get_max_replicas_for_cluster(cluster)))
            items.append({
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": NAME_PREFIX + svc.deployment_name,
                    "namespace": svc.namespace,
                    "labels": {LABEL_KEY: LABEL_VALUE},
                },
                "spec": {
                    "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment",
                                       "name": svc.deployment_name},
                    "minReplicas": lo,
                    "maxReplicas": hi,
                    "metrics": [{"type": "Resource", "resource": {
                        "name": "cpu", "target": {"type": "Utilization", "averageUtilization": target}}}],
                },
            })
        out[cluster] = items
    return out


class Kube:
    def __init__(self):
        self.kubectl = _cmd_prefix("DMOS_KUBECTL", "kubectl")

    def run(self, kc: str, args: List[str], stdin: Optional[str] = None,
            timeout: float = 60) -> Tuple[int, str, str]:
        try:
            p = subprocess.run(self.kubectl + ["--kubeconfig", kc, *args],
                               input=None if stdin is None else stdin.encode("utf-8"),
                               capture_output=True, timeout=timeout)
            return (p.returncode, p.stdout.decode("utf-8", "replace"),
                    p.stderr.decode("utf-8", "replace"))
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s"
        except OSError as e:
            return 127, "", str(e)


def _kubeconfigs(cfg) -> Dict[str, str]:
    return {n: resolve_path(c.kubeconfig_path) for n, c in cfg.get_all_clusters().items()}


def _namespace(cfg) -> str:
    return cfg.entry_service_config().namespace


def _list(kube: Kube, kc: str, ns: str) -> Tuple[Optional[list], str]:
    rc, out, err = kube.run(kc, ["-n", ns, "get", "hpa", "-l", SELECTOR, "-o", "json"])
    if rc != 0:
        return None, err.strip()[:300]
    try:
        return json.loads(out).get("items", []), ""
    except ValueError:
        return None, "invalid JSON from kubectl"


def cmd_apply(cfg, kube: Kube) -> Dict[str, Any]:
    specs, kcs, ns = hpa_specs(cfg), _kubeconfigs(cfg), _namespace(cfg)
    rec: Dict[str, Any] = {"applied_at": now(), "selector": SELECTOR, "clusters": {}, "ok": True}
    for cluster, items in specs.items():
        kc = kcs[cluster]
        c: Dict[str, Any] = {"expected": [i["metadata"]["name"] for i in items], "errors": []}
        rc, out, err = kube.run(kc, ["get", "apiservice", "v1beta1.metrics.k8s.io", "-o", "json"])
        try:
            conds = json.loads(out).get("status", {}).get("conditions", []) if rc == 0 else []
        except ValueError:
            conds = []
        c["metrics_api"] = any(x.get("type") == "Available" and x.get("status") == "True" for x in conds)
        if not c["metrics_api"]:
            c["errors"].append("metrics API (metrics-server) not available: HPA cannot read CPU")
        rc, out, err = kube.run(kc, ["apply", "-f", "-"],
                                stdin=json.dumps({"apiVersion": "v1", "kind": "List", "items": items}))
        if rc != 0:
            c["errors"].append("kubectl apply: " + err.strip()[:300])
        found, e = _list(kube, kc, ns)
        if found is None:
            c["errors"].append("kubectl get hpa: " + e)
            found = []
        c["present"] = sorted(i["metadata"]["name"] for i in found)
        c["specs"] = [{"name": i["metadata"]["name"], "target": i["spec"]["scaleTargetRef"]["name"],
                       "min": i["spec"]["minReplicas"], "max": i["spec"]["maxReplicas"],
                       "cpu_target_pct": i["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"]}
                      for i in items]
        missing = sorted(set(c["expected"]) - set(c["present"]))
        if missing:
            c["errors"].append(f"HPA missing after apply: {missing}")
        c["ok"] = not c["errors"]
        rec["ok"] = rec["ok"] and c["ok"]
        rec["clusters"][cluster] = c
    return rec


def cmd_verify(cfg, kube: Kube) -> Dict[str, Any]:
    kcs, ns = _kubeconfigs(cfg), _namespace(cfg)
    rec: Dict[str, Any] = {"verified_at": now(), "clusters": {}, "ok": True}
    for cluster, kc in kcs.items():
        found, e = _list(kube, kc, ns)
        c: Dict[str, Any] = {"errors": [] if found is not None else ["kubectl get hpa: " + e], "hpa": []}
        for i in found or []:
            st = i.get("status", {}) or {}
            conds = {x.get("type"): x.get("status") for x in st.get("conditions", []) or []}
            c["hpa"].append({"name": i["metadata"]["name"],
                             "current_replicas": st.get("currentReplicas"),
                             "desired_replicas": st.get("desiredReplicas"),
                             "scaling_active": conds.get("ScalingActive") == "True",
                             "able_to_scale": conds.get("AbleToScale") == "True"})
        inactive = [h["name"] for h in c["hpa"] if not h["scaling_active"]]
        if not c["hpa"]:
            c["errors"].append("no baseline HPA found")
        if inactive:
            c["errors"].append(f"HPA not reading its metric (ScalingActive != True): {inactive}")
        c["ok"] = not c["errors"]
        rec["ok"] = rec["ok"] and c["ok"]
        rec["clusters"][cluster] = c
    return rec


def cmd_delete(cfg, kube: Kube) -> Dict[str, Any]:
    kcs, ns = _kubeconfigs(cfg), _namespace(cfg)
    rec: Dict[str, Any] = {"deleted_at": now(), "clusters": {}, "ok": True}
    for cluster, kc in kcs.items():
        rc, out, err = kube.run(kc, ["-n", ns, "delete", "hpa", "-l", SELECTOR, "--ignore-not-found"])
        left, e = _list(kube, kc, ns)
        c = {"ok": rc == 0 and left == [], "remaining": [i["metadata"]["name"] for i in left or []],
             "error": (err.strip()[:300] if rc != 0 else e) or None}
        rec["ok"] = rec["ok"] and c["ok"]
        rec["clusters"][cluster] = c
    return rec


def _write(out: Optional[Path], key: str, rec: Dict[str, Any]) -> None:
    if out is None:
        return
    data = {}
    if out.is_file():
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    data[key] = rec
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("cmd", choices=["apply", "verify", "delete", "show"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    cfg = load_config(args.config_dir)
    if args.cmd == "show":
        print(json.dumps(hpa_specs(cfg), indent=1))
        return 0
    kube = Kube()
    rec = {"apply": cmd_apply, "verify": cmd_verify, "delete": cmd_delete}[args.cmd](cfg, kube)
    _write(args.out, args.cmd, rec)
    if not args.quiet or not rec["ok"]:
        bad = {c: v.get("errors") or v.get("error") or v.get("remaining")
               for c, v in rec["clusters"].items() if not v.get("ok")}
        print(f"  [hpa] {args.cmd}: {'ok' if rec['ok'] else 'PROBLEMS ' + json.dumps(bad)}")
    return 0 if rec["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
