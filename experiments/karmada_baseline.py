#!/usr/bin/env python3
"""
karmada_baseline.py — the Karmada baseline (arms `karmada-fhpa`, `karmada-agg`):
for the duration of one cell, Karmada takes over the Deployments that DMOS
scales in the 6 clusters, scales them with FederatedHPA and divides the
replicas among the clusters with its scheduler; at the end of the cell it
releases them, untouched. DMOS does not run in these arms; nothing of DMOS
changes. Setup of the Karmada control plane: docs/KARMADA-BASELINE.md.

What Karmada manages (from config/services.yaml, no tuning for this arm):
  services      the ones DMOS manages: application.entry_service and
                application.co_scaled_services (8 Deployments in Online
                Boutique). Services, Ingress, Cilium policies, redis stay local.
  templates     each Deployment as it is in the clusters (exported from the
                first cluster at every cell; the 6 clusters must have the same
                spec, otherwise Karmada's Overwrite would roll the pods out)
  FederatedHPA  one per service; min / max = sum over the clusters of the
                per-cluster min / max of services.yaml (the same limits as the
                hpa-none HPAs); CPU utilization target autoscaling.target_cpu_percent
                (70); Kubernetes default behaviour
  policy        PropagationPolicy over the 6 clusters, conflictResolution
                Overwrite (adopts the existing Deployments in place),
                preserveResourcesOnDeletion true (release keeps them), and
                  fhpa: Divided / Weighted / dynamicWeight AvailableReplicas
                  agg:  Divided / Aggregated
  data plane    unchanged: frontend Cilium global service, affinity "none"

Karmada facts this relies on (v1.19.0 docs and source, docs/KARMADA-BASELINE.md):
  - a member object applied by Karmada carries the label karmada.io/managed=true
    (karmadactl get shows it as ADOPTION=Y); with preserveResourcesOnDeletion
    the release keeps the object and removes Karmada's labels/annotations;
  - the ResourceBinding of Deployment <name> is <name>-deployment, conditions
    Scheduled and FullyApplied; FederatedHPA conditions are the autoscaling/v2
    ones (ScalingActive True = metric read);
  - by default the scheduler drops a cluster that gets 0 replicas from the
    binding and its Work is deleted: the member Deployment would then be
    deleted (or, preserved, left unmanaged with its old replicas). The
    scheduler must run with --enable-empty-workload-propagation=true, so a
    cluster given 0 replicas stays in the binding and is scaled to 0. This
    script checks the flag (host check) and, at every trace sample, that all
    clusters stay in every binding.

Commands (the harness calls them, experiments/lib/run_common.sh):
  check                       pre-flight only: control plane, clusters, metrics
                              API, scheduler flags, estimators; changes nothing
  apply --variant V --out F   adopt the Deployments (exit 3 if not done cleanly:
                              the harness then releases and aborts the cell)
  trace --out F.jsonl         sample bindings, FederatedHPAs and clusters every
                              --interval s (background during the load)
  verify --trace F.jsonl --out F   after the load
  release [--out F]           give the Deployments back (idempotent; also when
                              nothing is managed). Never deletes a template
                              unless every binding and Work preserves resources.
  status                      what Karmada manages now
  show --variant V            print PropagationPolicy and FederatedHPAs

Environment:
  DMOS_KARMADA_KUBECONFIG       kubeconfig of the Karmada apiserver (required to
                                apply; without it `release` only checks that
                                nothing in the members is managed by Karmada)
  DMOS_KARMADA_HOST_KUBECONFIG  kubeconfig of the cluster hosting Karmada (to
                                check scheduler flags, estimators, versions)
  DMOS_KARMADA_SKIP_HOST_CHECK  1 = apply without the host check (recorded as a
                                warning; the per-sample binding check stays)
  DMOS_KARMADA_VERSION          expected image tag, default v1.19.0
  DMOS_KUBECTL                  kubectl command (as in capture_env_state.py)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hpa_baseline import Kube, _kubeconfigs, _namespace, hpa_specs, load_config  # noqa: E402

LABEL_KEY = "app.kubernetes.io/managed-by"
LABEL_VALUE = "dmos-karmada-baseline"
SELECTOR = f"{LABEL_KEY}={LABEL_VALUE}"
POLICY_NAME = "dmos-karmada-baseline"
FHPA_PREFIX = "dmos-karmada-"
VARIANTS = ("fhpa", "agg")

MANAGED_LABEL = "karmada.io/managed"            # on member objects applied by Karmada
SKIP_NS_LABEL = "namespace.karmada.io/skip-auto-propagation"
REVISION_ANN = "deployment.kubernetes.io/revision"
RB_PERMANENT_ID = "resourcebinding.karmada.io/permanent-id"   # label on a binding and its Works
# template metadata that must not travel to the members
DROP_ANNOTATIONS = {REVISION_ANN, "kubectl.kubernetes.io/last-applied-configuration"}
METRICS_APISERVICE = "v1beta1.metrics.k8s.io"
REQUIRED_SCHEDULER_FLAGS = {
    "--enable-scheduler-estimator": "true",
    "--enable-empty-workload-propagation": "true",
}
KARMADA_NS = "karmada-system"
DEFAULT_VERSION = "v1.19.0"

ADOPT_TIMEOUT_S = 300
RELEASE_TIMEOUT_S = 180
POLL_S = 3.0
MEMBER_TIMEOUT_S = 20      # per kubectl call to a member (as capture_env_state.py)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_karmada_key(k: str) -> bool:
    return "karmada.io/" in k


# ── manifests (pure) ─────────────────────────────────────────────────────────

def limits(cfg) -> Dict[str, Dict[str, Any]]:
    """{deployment: {min, max, cpu_target_pct}}: sums over the clusters of the
    per-cluster limits of the hpa-none HPAs (same source, same numbers)."""
    out: Dict[str, Dict[str, Any]] = {}
    for items in hpa_specs(cfg).values():
        for i in items:
            dep = i["spec"]["scaleTargetRef"]["name"]
            d = out.setdefault(dep, {"min": 0, "max": 0, "cpu_target_pct":
                                     i["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"]})
            d["min"] += int(i["spec"]["minReplicas"])
            d["max"] += int(i["spec"]["maxReplicas"])
    return out


def spec_fingerprint(dep: Dict[str, Any]) -> str:
    """Hash of the Deployment spec without replicas: equal in every cluster, or
    adopting with Overwrite would change the pod template and roll the pods."""
    spec = dict(dep.get("spec") or {})
    spec.pop("replicas", None)
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def spec_differences(a: Dict[str, Any], b: Dict[str, Any], path: str = "spec") -> List[str]:
    """Paths where two Deployment specs differ (replicas ignored), for the error message."""
    if path == "spec":
        a = {k: v for k, v in (a or {}).items() if k != "replicas"}
        b = {k: v for k, v in (b or {}).items() if k != "replicas"}
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for k in sorted(set(a) | set(b)):
            out += spec_differences(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += spec_differences(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [path]


def clean_template(dep: Dict[str, Any], replicas: int) -> Dict[str, Any]:
    """Resource template for the Karmada control plane from a member Deployment:
    no runtime fields, no Karmada or kubectl metadata, `replicas`. No label of
    ours: template labels travel to the member Deployments and would stay
    there after the release (templates are found by name instead)."""
    meta = dep.get("metadata") or {}
    labels = {k: v for k, v in (meta.get("labels") or {}).items() if not _is_karmada_key(k)}
    ann = {k: v for k, v in (meta.get("annotations") or {}).items()
           if k not in DROP_ANNOTATIONS and not _is_karmada_key(k)}
    spec = json.loads(json.dumps(dep.get("spec") or {}))
    spec["replicas"] = int(replicas)
    md: Dict[str, Any] = {"name": meta["name"], "namespace": meta["namespace"]}
    if labels:
        md["labels"] = labels
    if ann:
        md["annotations"] = ann
    return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": md, "spec": spec}


def propagation_policy(deployments: List[str], clusters: List[str], namespace: str,
                       variant: str) -> Dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}: {VARIANTS}")
    rs: Dict[str, Any] = {"replicaSchedulingType": "Divided"}
    if variant == "fhpa":
        rs["replicaDivisionPreference"] = "Weighted"
        rs["weightPreference"] = {"dynamicWeight": "AvailableReplicas"}
    else:
        rs["replicaDivisionPreference"] = "Aggregated"
    return {
        "apiVersion": "policy.karmada.io/v1alpha1",
        "kind": "PropagationPolicy",
        "metadata": {"name": POLICY_NAME, "namespace": namespace, "labels": {LABEL_KEY: LABEL_VALUE}},
        "spec": {
            "resourceSelectors": [{"apiVersion": "apps/v1", "kind": "Deployment", "name": d}
                                  for d in deployments],
            "conflictResolution": "Overwrite",
            "preserveResourcesOnDeletion": True,
            "placement": {
                "clusterAffinity": {"clusterNames": list(clusters)},
                "replicaScheduling": rs,
            },
        },
    }


def federated_hpa(deployment: str, lim: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    return {
        "apiVersion": "autoscaling.karmada.io/v1alpha1",
        "kind": "FederatedHPA",
        "metadata": {"name": FHPA_PREFIX + deployment, "namespace": namespace,
                     "labels": {LABEL_KEY: LABEL_VALUE}},
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": deployment},
            "minReplicas": int(lim["min"]),
            "maxReplicas": int(lim["max"]),
            "metrics": [{"type": "Resource", "resource": {
                "name": "cpu", "target": {"type": "Utilization",
                                          "averageUtilization": int(lim["cpu_target_pct"])}}}],
        },
    }


def binding_name(deployment: str) -> str:
    """names.GenerateBindingName(kind, name) = lower(name + "-" + kind)."""
    return f"{deployment}-deployment".lower()


# ── state parsing (pure) ─────────────────────────────────────────────────────

def _conds(obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {c.get("type"): c for c in ((obj.get("status") or {}).get("conditions") or [])}


def member_state(dep: Dict[str, Any]) -> Dict[str, Any]:
    meta, spec, st = dep.get("metadata") or {}, dep.get("spec") or {}, dep.get("status") or {}
    return {
        "uid": meta.get("uid"),
        "revision": (meta.get("annotations") or {}).get(REVISION_ANN),
        "managed": (meta.get("labels") or {}).get(MANAGED_LABEL) == "true",
        "replicas": spec.get("replicas"),
        "ready": st.get("readyReplicas", 0) or 0,
        "updated": st.get("updatedReplicas", 0) or 0,
        "available": st.get("availableReplicas", 0) or 0,
        "observed_generation_ok": (st.get("observedGeneration") or 0) >= (meta.get("generation") or 0),
        "fingerprint": spec_fingerprint(dep),
    }


def binding_state(rb: Dict[str, Any]) -> Dict[str, Any]:
    spec = rb.get("spec") or {}
    c = _conds(rb)
    return {
        "permanent_id": ((rb.get("metadata") or {}).get("labels") or {}).get(RB_PERMANENT_ID),
        "replicas": spec.get("replicas"),
        "clusters": {x.get("name"): int(x.get("replicas") or 0) for x in spec.get("clusters") or []},
        "preserve": bool(spec.get("preserveResourcesOnDeletion")),
        "scheduled": (c.get("Scheduled") or {}).get("status") == "True",
        "fully_applied": (c.get("FullyApplied") or {}).get("status") == "True",
        "scheduled_reason": (c.get("Scheduled") or {}).get("reason"),
    }


def fhpa_state(f: Dict[str, Any]) -> Dict[str, Any]:
    st = f.get("status") or {}
    c = _conds(f)
    return {
        "current": st.get("currentReplicas"),
        "desired": st.get("desiredReplicas"),
        "scaling_active": (c.get("ScalingActive") or {}).get("status") == "True",
        "scaling_active_reason": (c.get("ScalingActive") or {}).get("reason"),
        "able_to_scale": (c.get("AbleToScale") or {}).get("status") == "True",
        "limited": (c.get("ScalingLimited") or {}).get("status") == "True",
    }


def cluster_state(cl: Dict[str, Any]) -> Dict[str, Any]:
    c = _conds(cl)
    st = cl.get("status") or {}
    return {
        "ready": (c.get("Ready") or {}).get("status") == "True",
        "sync_mode": (cl.get("spec") or {}).get("syncMode"),
        "kubernetes_version": st.get("kubernetesVersion"),
        "region": (cl.get("spec") or {}).get("region"),
    }


def parse_flags(args: List[str]) -> Dict[str, str]:
    """['--a=1', '--b', '2', '--c'] -> {'--a': '1', '--b': '2', '--c': 'true'}."""
    out: Dict[str, str] = {}
    i = 0
    while i < len(args):
        a = str(args[i])
        if a.startswith("--"):
            if "=" in a:
                k, v = a.split("=", 1)
                out[k] = v
            elif i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                out[a] = str(args[i + 1])
                i += 1
            else:
                out[a] = "true"
        i += 1
    return out


def host_findings(deploys: List[Dict[str, Any]], clusters: List[str], expected_version: str
                  ) -> Dict[str, Any]:
    """Check the Karmada components on the host cluster: scheduler flags,
    one ready estimator per member, image versions, feature gates."""
    by_name = {d["metadata"]["name"]: d for d in deploys}
    errors: List[str] = []
    warnings: List[str] = []
    images: Dict[str, str] = {}
    for n, d in by_name.items():
        cs = ((d.get("spec") or {}).get("template") or {}).get("spec", {}).get("containers") or []
        if cs:
            images[n] = cs[0].get("image", "")

    def flags_of(name: str) -> Dict[str, str]:
        d = by_name.get(name) or {}
        cs = ((d.get("spec") or {}).get("template") or {}).get("spec", {}).get("containers") or []
        return parse_flags(list(cs[0].get("command") or []) + list(cs[0].get("args") or [])) if cs else {}

    sched = flags_of("karmada-scheduler")
    if "karmada-scheduler" not in by_name:
        errors.append("karmada-scheduler deployment not found on the host cluster")
    for flag, val in REQUIRED_SCHEDULER_FLAGS.items():
        if sched.get(flag, "").lower() != val:
            errors.append(f"karmada-scheduler without {flag}={val} (see docs/KARMADA-BASELINE.md)")
    estimators = {}
    for c in clusters:
        d = by_name.get(f"karmada-scheduler-estimator-{c}")
        ready = bool(d) and ((d.get("status") or {}).get("readyReplicas") or 0) >= 1
        estimators[c] = ready
        if not ready:
            errors.append(f"scheduler estimator for {c} missing or not ready")
    cm = flags_of("karmada-controller-manager")
    gates = {"karmada-scheduler": sched.get("--feature-gates", ""),
             "karmada-controller-manager": cm.get("--feature-gates", "")}
    if "failover=true" in gates["karmada-controller-manager"].replace(" ", "").lower():
        warnings.append("Failover feature gate enabled (documented default: off)")
    if cm.get("--enable-no-execute-taint-eviction", "false").lower() == "true":
        warnings.append("NoExecute taint eviction enabled (documented default: off)")
    versions = {n: img.rsplit(":", 1)[-1] for n, img in images.items()
                if n.startswith("karmada-") and "/karmada/" in img}
    off = {n: v for n, v in versions.items() if v != expected_version}
    if off:
        warnings.append(f"Karmada images not at {expected_version}: {off}")
    return {"errors": errors, "warnings": warnings, "images": images, "versions": versions,
            "scheduler_flags": sched, "controller_manager_flags": cm, "feature_gates": gates,
            "estimators_ready": estimators}


def adoption_problems(before: Dict[str, Dict[str, Dict[str, Any]]],
                      after: Dict[str, Dict[str, Dict[str, Any]]],
                      bindings: Dict[str, Dict[str, Any]], clusters: List[str],
                      deployments: List[str]) -> Tuple[List[str], List[str]]:
    """(not-yet-done conditions, hard errors) of an adoption, from member and
    binding state. Hard errors: a Deployment recreated or rolled out, a cluster
    dropped from a binding, a Deployment missing."""
    pending: List[str] = []
    errors: List[str] = []
    for d in deployments:
        rb = bindings.get(d)
        if rb is None:
            pending.append(f"{d}: no binding yet")
            continue
        if not rb["scheduled"]:
            pending.append(f"{d}: not scheduled ({rb['scheduled_reason']})")
        if not rb["fully_applied"]:
            pending.append(f"{d}: not fully applied")
        if rb["scheduled"]:
            missing = sorted(set(clusters) - set(rb["clusters"]))
            if missing:
                errors.append(f"{d}: clusters dropped from the binding {missing} "
                              "(scheduler without --enable-empty-workload-propagation=true?)")
        for c in clusters:
            if c not in after:          # cluster not read this time (transient): wait
                pending.append(f"{c}: not read")
                continue
            m = after[c].get(d)
            b0 = (before.get(c) or {}).get(d)
            if m is None:
                errors.append(f"{d}@{c}: Deployment missing in the member")
                continue
            if b0 is not None and m["uid"] != b0["uid"]:
                errors.append(f"{d}@{c}: Deployment recreated (uid changed)")
            if b0 is not None and m["revision"] != b0["revision"]:
                errors.append(f"{d}@{c}: pod template rolled out (revision {b0['revision']} -> {m['revision']})")
            if not m["managed"]:
                pending.append(f"{d}@{c}: not managed yet")
            want = rb["clusters"].get(c) if rb["scheduled"] else None
            if want is not None and m["replicas"] != want:
                pending.append(f"{d}@{c}: replicas {m['replicas']} != {want}")
            if m["ready"] < (m["replicas"] or 0) or not m["observed_generation_ok"]:
                pending.append(f"{d}@{c}: {m['ready']}/{m['replicas']} ready")
    return pending, errors


def summarize_trace(samples: List[Dict[str, Any]], clusters: List[str]) -> Dict[str, Any]:
    """From the trace: clusters dropped from a binding, clusters not Ready,
    whether the FederatedHPAs read their metric and scaled, replica ranges."""
    dropped, not_ready, inactive = set(), set(), 0
    replicas: Dict[str, List[int]] = {}
    desired: Dict[str, set] = {}
    used: Dict[str, set] = {}
    errors_seen = 0
    for s in samples:
        if s.get("error"):
            errors_seen += 1
            continue
        for d, rb in (s.get("bindings") or {}).items():
            if rb.get("scheduled"):
                for c in set(clusters) - set(rb.get("clusters") or {}):
                    dropped.add(f"{d}@{c}")
            tot = sum((rb.get("clusters") or {}).values())
            replicas.setdefault(d, []).append(tot)
            used.setdefault(d, set()).update(c for c, r in (rb.get("clusters") or {}).items() if r > 0)
        for c, ok in (s.get("clusters_ready") or {}).items():
            if not ok:
                not_ready.add(c)
        f = s.get("fhpa") or {}
        if f and not all(x.get("scaling_active") for x in f.values()):
            inactive += 1
        for d, x in f.items():
            if x.get("desired") is not None:
                desired.setdefault(d, set()).add(x["desired"])
    n = len(samples) - errors_seen
    return {
        "samples": len(samples), "sample_errors": errors_seen,
        "clusters_dropped": sorted(dropped),
        "clusters_not_ready": sorted(not_ready),
        "fhpa_inactive_samples": inactive,
        "fhpa_active_fraction": (1 - inactive / n) if n else None,
        "scaled": {d: len(v) > 1 for d, v in desired.items()},
        "total_replicas_range": {d: [min(v), max(v)] for d, v in replicas.items() if v},
        "clusters_used": {d: sorted(v) for d, v in used.items()},
    }


# ── cluster access ───────────────────────────────────────────────────────────

class Env:
    def __init__(self, cfg, kube: Kube, karmada_kc: Optional[str], host_kc: Optional[str]):
        self.cfg, self.kube = cfg, kube
        self.kkc, self.hkc = karmada_kc, host_kc
        self.member_kcs = _kubeconfigs(cfg)
        self.clusters = list(self.member_kcs)
        self.ns = _namespace(cfg)
        self.lim = limits(cfg)
        self.deployments = list(self.lim)

    def kjson(self, args: List[str], kc: Optional[str] = None,
              timeout: float = 60) -> Tuple[Optional[Any], str]:
        rc, out, err = self.kube.run(kc or self.kkc, args + ["-o", "json"], timeout=timeout)
        if rc != 0:
            return None, (err.strip() or out.strip())[:300]
        try:
            return json.loads(out), ""
        except ValueError:
            return None, "invalid JSON from kubectl"

    def members(self) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]], List[str]]:
        """({cluster: {deployment: state}}, {cluster: {deployment: raw}}, errors)."""
        from concurrent.futures import ThreadPoolExecutor
        states, raws, errors = {}, {}, []
        names = list(self.member_kcs)
        with ThreadPoolExecutor(max_workers=max(1, len(names))) as ex:
            got = list(ex.map(lambda c: self.kjson(["-n", self.ns, "get", "deploy"],
                                                   self.member_kcs[c], timeout=MEMBER_TIMEOUT_S), names))
        for c, (data, e) in zip(names, got):
            if data is None:
                errors.append(f"{c}: kubectl get deploy: {e}")
                continue
            items = {i["metadata"]["name"]: i for i in data.get("items", [])
                     if i["metadata"]["name"] in self.lim}
            raws[c] = items
            states[c] = {n: member_state(i) for n, i in items.items()}
        return states, raws, errors

    def bindings(self) -> Tuple[Dict[str, Dict[str, Any]], str]:
        data, e = self.kjson(["-n", self.ns, "get", "resourcebindings.work.karmada.io"])
        if data is None:
            return {}, e
        names = {binding_name(d): d for d in self.deployments}
        return {names[i["metadata"]["name"]]: binding_state(i) for i in data.get("items", [])
                if i["metadata"]["name"] in names}, ""

    def fhpas(self) -> Tuple[Dict[str, Dict[str, Any]], str]:
        data, e = self.kjson(["-n", self.ns, "get", "federatedhpas.autoscaling.karmada.io", "-l", SELECTOR])
        if data is None:
            return {}, e
        return {i["spec"]["scaleTargetRef"]["name"]: fhpa_state(i) for i in data.get("items", [])}, ""

    def karmada_clusters(self) -> Tuple[Dict[str, Dict[str, Any]], str]:
        data, e = self.kjson(["get", "clusters.cluster.karmada.io"])
        if data is None:
            return {}, e
        return {i["metadata"]["name"]: cluster_state(i) for i in data.get("items", [])}, ""

    def leftovers(self) -> Tuple[Dict[str, List[str]], str]:
        out: Dict[str, List[str]] = {}
        for kind in ("deployments.apps", "propagationpolicies.policy.karmada.io",
                     "federatedhpas.autoscaling.karmada.io"):
            data, e = self.kjson(["-n", self.ns, "get", kind])
            if data is None:
                if "NotFound" in e or "not found" in e.lower():
                    continue
                return out, f"{kind}: {e}"
            names = [i["metadata"]["name"] for i in data.get("items", [])]
            if names:
                out[kind.split(".")[0]] = names
        return out, ""


def _require_karmada(env: Env) -> Optional[str]:
    if not env.kkc:
        return "DMOS_KARMADA_KUBECONFIG not set (kubeconfig of the Karmada apiserver)"
    if not os.path.isfile(env.kkc):
        return f"Karmada kubeconfig not found: {env.kkc}"
    return None


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_check(env: Env) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"checked_at": now(), "errors": [], "warnings": []}
    e = _require_karmada(env)
    if e:
        rec["errors"].append(e)
        rec["ok"] = False
        return rec
    kcl, e = env.karmada_clusters()
    if e:
        rec["errors"].append(f"Karmada apiserver not reachable: {e}")
    rec["clusters"] = kcl
    for c in env.clusters:
        s = kcl.get(c)
        if s is None:
            rec["errors"].append(f"cluster {c} not registered in Karmada (karmadactl join {c} ...)")
        else:
            if not s["ready"]:
                rec["errors"].append(f"cluster {c} not Ready in Karmada")
            if s["sync_mode"] != "Push":
                rec["errors"].append(f"cluster {c} in {s['sync_mode']} mode: the design needs Push "
                                     "(no Karmada pod on the measured nodes)")
    api, e = env.kjson(["get", "apiservice", METRICS_APISERVICE])
    avail = bool(api) and any(c.get("type") == "Available" and c.get("status") == "True"
                              for c in (api.get("status") or {}).get("conditions") or [])
    rec["metrics_api"] = avail
    if not avail:
        rec["errors"].append(f"{METRICS_APISERVICE} not Available in Karmada: "
                             "karmadactl addons enable karmada-metrics-adapter")
    ns, e = env.kjson(["get", "namespace", env.ns])
    if ns is None:
        rec["namespace"] = "absent (created by apply, with the skip label)"
    else:
        lab = (ns.get("metadata", {}).get("labels") or {}).get(SKIP_NS_LABEL, "")
        rec["namespace"] = f"present, {SKIP_NS_LABEL}={lab or '(missing)'}"
        if lab.lower() != "true":
            rec["errors"].append(f"namespace {env.ns} exists in Karmada without {SKIP_NS_LABEL}=true: "
                                 "Karmada may own the namespace in the members — fix by hand "
                                 "(docs/KARMADA-BASELINE.md), never delete it")
    if env.hkc and os.path.isfile(env.hkc):
        data, e = env.kjson(["-n", KARMADA_NS, "get", "deploy"], env.hkc)
        if data is None:
            rec["errors"].append(f"host cluster not reachable: {e}")
        else:
            hf = host_findings(data.get("items", []), env.clusters,
                               os.environ.get("DMOS_KARMADA_VERSION", DEFAULT_VERSION))
            rec["host"] = hf
            rec["errors"] += hf["errors"]
            rec["warnings"] += hf["warnings"]
    elif os.environ.get("DMOS_KARMADA_SKIP_HOST_CHECK") == "1":
        rec["warnings"].append("host check skipped (DMOS_KARMADA_SKIP_HOST_CHECK=1): scheduler flags, "
                               "estimators and versions not verified")
        rec["host"] = None
    else:
        rec["errors"].append("DMOS_KARMADA_HOST_KUBECONFIG not set: cannot verify the scheduler flags "
                             "(--enable-empty-workload-propagation, --enable-scheduler-estimator); "
                             "set it, or DMOS_KARMADA_SKIP_HOST_CHECK=1 to proceed without")
    rec["ok"] = not rec["errors"]
    return rec


def _wait(fn, timeout: float):
    """Poll fn() -> (done, payload) until done or timeout; returns (done, payload)."""
    t_end = time.time() + timeout
    done, payload = fn()
    while not done and time.time() < t_end:
        time.sleep(POLL_S)
        done, payload = fn()
    return done, payload


def cmd_apply(env: Env, variant: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"applied_at": now(), "variant": variant, "namespace": env.ns,
                           "errors": [], "warnings": [], "restarted": []}
    chk = cmd_check(env)
    rec["host_checked"] = chk.get("host") is not None
    rec["versions"] = (chk.get("host") or {}).get("versions")
    rec["version_ok"] = not any("not at v" in w for w in chk.get("warnings", []))
    rec["check"] = chk
    rec["errors"] += chk["errors"]
    rec["warnings"] += chk["warnings"]
    if rec["errors"]:
        rec["ok"] = False
        return rec

    left, e = env.leftovers()
    if e or left:
        rec["errors"].append(f"objects already in Karmada namespace {env.ns}: {left or e} "
                             "(python experiments/karmada_baseline.py release)")
    before, raws, errs = env.members()
    rec["errors"] += errs
    managed = [f"{d}@{c}" for c, ds in before.items() for d, s in ds.items() if s["managed"]]
    if managed:
        rec["errors"].append(f"Deployments already managed by Karmada: {managed}")
    for c in env.clusters:
        miss = sorted(set(env.deployments) - set((before.get(c) or {})))
        if miss:
            rec["errors"].append(f"{c}: Deployments missing {miss}")
    if rec["errors"]:
        rec["ok"] = False
        return rec

    # templates: from the first cluster, identical spec required in every cluster
    ref = env.clusters[0]
    diffs = {}
    for d in env.deployments:
        fps = {c: before[c][d]["fingerprint"] for c in env.clusters}
        if len(set(fps.values())) > 1:
            diffs[d] = {c: spec_differences(raws[ref][d]["spec"], raws[c][d]["spec"])[:5]
                        for c in env.clusters if fps[c] != fps[ref]}
    if diffs:
        rec["errors"].append("Deployment spec differs between clusters: adopting would roll the pods "
                             f"out. Align the clusters first: {json.dumps(diffs)}")
        rec["ok"] = False
        return rec
    templates = [clean_template(raws[ref][d], env.lim[d]["min"]) for d in env.deployments]
    pp = propagation_policy(env.deployments, env.clusters, env.ns, variant)
    fhpas = [federated_hpa(d, env.lim[d], env.ns) for d in env.deployments]
    rec["before"] = before
    rec["manifests"] = {"templates_fingerprint": {d: before[ref][d]["fingerprint"] for d in env.deployments},
                        "propagation_policy": pp, "federated_hpa": fhpas,
                        "template_replicas": {d: env.lim[d]["min"] for d in env.deployments}}

    # namespace in the control plane, never auto-propagated
    if env.kjson(["get", "namespace", env.ns])[0] is None:
        ns = {"apiVersion": "v1", "kind": "Namespace",
              "metadata": {"name": env.ns, "labels": {SKIP_NS_LABEL: "true"}}}
        rc, _, err = env.kube.run(env.kkc, ["create", "-f", "-"], stdin=json.dumps(ns))
        if rc != 0:
            rec["errors"].append(f"create namespace: {err.strip()[:300]}")
            rec["ok"] = False
            return rec

    # templates, then the policy that claims them (documented migration order),
    # created (not applied: no last-applied annotation travels to the members)
    for what, objs in (("templates", templates), ("propagation policy", [pp])):
        rc, _, err = env.kube.run(env.kkc, ["create", "-f", "-"],
                                  stdin=json.dumps({"apiVersion": "v1", "kind": "List", "items": objs}))
        if rc != 0:
            rec["errors"].append(f"create {what}: {err.strip()[:300]}")
            rec["ok"] = False
            return rec

    def adopted():
        rbs, e1 = env.bindings()
        after, _, e2 = env.members()
        pending, hard = adoption_problems(before, after, rbs, env.clusters, env.deployments)
        if e1:
            pending.append(f"bindings: {e1}")
        pending += e2
        return (not pending) or bool(hard), {"pending": pending, "hard": hard, "after": after, "rbs": rbs}

    done, st = _wait(adopted, ADOPT_TIMEOUT_S)
    rec["after"] = st["after"]
    rec["placement"] = {d: rb["clusters"] for d, rb in st["rbs"].items()}
    rec["errors"] += st["hard"]
    rec["restarted"] = [e for e in st["hard"] if "rolled out" in e or "recreated" in e]
    if st["pending"] and not st["hard"]:
        rec["errors"].append(f"adoption not complete after {ADOPT_TIMEOUT_S}s: {st['pending'][:10]}")
    if rec["errors"]:
        rec["ok"] = False
        return rec

    rc, _, err = env.kube.run(env.kkc, ["create", "-f", "-"],
                              stdin=json.dumps({"apiVersion": "v1", "kind": "List", "items": fhpas}))
    if rc != 0:
        rec["errors"].append(f"create FederatedHPA: {err.strip()[:300]}")
    else:
        f, e = env.fhpas()
        miss = sorted(set(env.deployments) - set(f))
        if e or miss:
            rec["errors"].append(f"FederatedHPA missing after create: {miss or e}")
    rec["adopted_at"] = now()
    rec["ok"] = not rec["errors"]
    return rec


def sample(env: Env, t0: float) -> Dict[str, Any]:
    s: Dict[str, Any] = {"epoch_s": round(time.time(), 3), "elapsed_s": round(time.time() - t0, 1)}
    rbs, e1 = env.bindings()
    f, e2 = env.fhpas()
    kcl, e3 = env.karmada_clusters()
    err = "; ".join(x for x in (e1, e2, e3) if x)
    if err:
        s["error"] = err
    s["bindings"] = rbs
    s["fhpa"] = f
    s["clusters_ready"] = {c: v["ready"] for c, v in kcl.items() if c in env.clusters}
    return s


def cmd_trace(env: Env, out: Path, interval: float, duration: float,
              stop_file: Optional[Path] = None) -> int:
    e = _require_karmada(env)
    if e:
        print(f"  [karmada] trace: {e}", file=sys.stderr)
        return 2
    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        def stopped() -> bool:
            return stop_file is not None and stop_file.exists()
        while time.time() - t0 < duration and not stopped():
            end = time.time() + interval
            fh.write(json.dumps(sample(env, t0)) + "\n")
            fh.flush()
            while time.time() < end and not stopped():
                time.sleep(min(1.0, max(0.0, end - time.time())))
    return 0


def read_trace(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def cmd_verify(env: Env, trace: Optional[Path]) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"verified_at": now(), "errors": [], "warnings": []}
    e = _require_karmada(env)
    if e:
        rec.update(ok=False, errors=[e])
        return rec
    f, e = env.fhpas()
    rec["fhpa"] = f
    if e:
        rec["errors"].append(f"FederatedHPA: {e}")
    inactive = [d for d in env.deployments if not (f.get(d) or {}).get("scaling_active")]
    rec["fhpa_inactive"] = inactive
    if inactive:
        rec["errors"].append(f"FederatedHPA not reading its metric (ScalingActive != True): {inactive}")
    states, _, errs = env.members()
    rec["errors"] += errs
    # only the clusters that were read: a read error is reported as such
    unmanaged = [f"{d}@{c}" for c in env.clusters if c in states for d in env.deployments
                 if not (states[c].get(d) or {}).get("managed")]
    rec["unmanaged"] = unmanaged
    if unmanaged:
        rec["errors"].append(f"Deployments no longer managed by Karmada: {unmanaged}")
    samples = read_trace(trace)
    tr = summarize_trace(samples, env.clusters)
    rec["trace"] = tr
    rec["clusters_dropped"] = tr["clusters_dropped"]
    rec["clusters_not_ready"] = tr["clusters_not_ready"]
    rec["never_scaled"] = bool(tr["scaled"]) and not any(tr["scaled"].values())
    if not samples:
        rec["warnings"].append("no trace samples")
    if tr["clusters_dropped"]:
        rec["errors"].append(f"clusters dropped from bindings during the run: {tr['clusters_dropped']} "
                             "(Deployments deleted or left unmanaged)")
    if tr["clusters_not_ready"]:
        rec["warnings"].append(f"clusters not Ready in Karmada during the run: {tr['clusters_not_ready']}")
    if tr["sample_errors"]:
        rec["warnings"].append(f"{tr['sample_errors']} trace samples failed")
    if tr["scaled"] and not any(tr["scaled"].values()):
        rec["warnings"].append("no FederatedHPA changed its desired replicas during the run")
    rec["ok"] = not rec["errors"]
    return rec


def cmd_release(env: Env) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"released_at": now(), "errors": [], "warnings": [], "steps": []}
    if _require_karmada(env):
        # Karmada not configured here: nothing to release, but nothing in the
        # members may be managed by a Karmada we cannot reach.
        states, _, errs = env.members()
        managed = [f"{d}@{c}" for c, ds in states.items() for d, s in ds.items() if s["managed"]]
        rec["karmada"] = "not configured"
        if managed:
            rec["errors"].append(f"Deployments managed by Karmada but DMOS_KARMADA_KUBECONFIG not set: {managed}")
        rec["warnings"] += errs
        rec["ok"] = not rec["errors"]
        return rec

    def step(msg):
        rec["steps"].append(f"{now()} {msg}")

    # 1) FederatedHPAs first: no replica change during the release
    rc, _, err = env.kube.run(env.kkc, ["-n", env.ns, "delete", "federatedhpas.autoscaling.karmada.io",
                                        "-l", SELECTOR, "--ignore-not-found"])
    step("FederatedHPAs deleted" if rc == 0 else f"delete FederatedHPA failed: {err.strip()[:200]}")
    if rc != 0:
        rec["errors"].append(f"delete FederatedHPA: {err.strip()[:300]}")

    left, e = env.leftovers()
    if e:
        rec["errors"].append(f"cannot list Karmada objects: {e}")
        rec["ok"] = False
        return rec
    ours = [d for d in left.get("deployments", []) if d in env.lim]
    foreign = [d for d in left.get("deployments", []) if d not in env.lim]
    if foreign:
        rec["warnings"].append(f"other Deployment templates in {env.ns} (not touched): {foreign}")

    if ours:
        # 2) every binding and Work of our templates must preserve resources,
        #    otherwise deleting a template deletes the Deployment in the members
        pp, _ = env.kjson(["-n", env.ns, "get", "propagationpolicies.policy.karmada.io", POLICY_NAME])
        if pp is not None and not (pp.get("spec") or {}).get("preserveResourcesOnDeletion"):
            env.kube.run(env.kkc, ["-n", env.ns, "patch", "propagationpolicies.policy.karmada.io",
                                   POLICY_NAME, "--type=merge", "-p",
                                   json.dumps({"spec": {"preserveResourcesOnDeletion": True}})])
            step("policy patched: preserveResourcesOnDeletion true")

        def preserved():
            rbs, e1 = env.bindings()
            works, e2 = env.kjson(["get", "works.work.karmada.io", "-A"])
            bad = [d for d in ours if d in rbs and not rbs[d]["preserve"]]
            ids = {rbs[d]["permanent_id"]: d for d in ours if d in rbs and rbs[d]["permanent_id"]}
            expected = sum(len(rbs[d]["clusters"]) for d in ours if d in rbs)
            matched = 0
            for w in (works or {}).get("items", []):
                if ((w.get("metadata") or {}).get("labels") or {}).get(RB_PERMANENT_ID) in ids:
                    matched += 1
                    if not (w.get("spec") or {}).get("preserveResourcesOnDeletion"):
                        bad.append(w["metadata"]["name"])
            no_id = [d for d in ours if d in rbs and not rbs[d]["permanent_id"]]
            # every Work of our bindings found and preserving: a check that
            # found nothing must not pass
            ok = not bad and not e1 and works is not None and not no_id and matched >= expected
            return ok, {"bad": bad, "works_found": matched, "works_expected": expected,
                        "bindings_without_id": no_id, "error": e1 or e2}

        ok, st = _wait(preserved, 60)
        if not ok:
            rec["errors"].append(f"bindings/Works do not preserve resources {st}: templates NOT deleted "
                                 "(deleting them would delete the Deployments in the members)")
            rec["ok"] = False
            return rec
        step("bindings and Works preserve resources")
        # 3) delete our templates: Karmada keeps the member Deployments and
        #    removes its labels and annotations from them
        rc, _, err = env.kube.run(env.kkc, ["-n", env.ns, "delete", "deployments.apps", *ours,
                                            "--ignore-not-found"])
        if rc != 0:
            rec["errors"].append(f"delete templates: {err.strip()[:300]}")
            rec["ok"] = False
            return rec
        step(f"templates deleted: {ours}")

    # 4) wait until no member Deployment is managed and all still exist
    def released():
        states, _, errs = env.members()
        managed = [f"{d}@{c}" for c, ds in states.items() for d, s in ds.items() if s["managed"]]
        missing = [f"{d}@{c}" for c in env.clusters if c in states for d in env.deployments
                   if d not in states[c]]
        return (not managed and not errs), {"managed": managed, "missing": missing, "errors": errs,
                                            "replicas": {c: {d: s["replicas"] for d, s in ds.items()}
                                                         for c, ds in states.items()}}

    ok, st = _wait(released, RELEASE_TIMEOUT_S)
    rec["members_after"] = st["replicas"]
    if st["missing"]:
        rec["errors"].append(f"Deployments MISSING in the members after release: {st['missing']} "
                             "(re-create them from the Online Boutique manifests)")
    if not ok:
        rec["errors"].append(f"still managed by Karmada after {RELEASE_TIMEOUT_S}s: {st['managed'] or st['errors']}")
    else:
        step("no member Deployment managed by Karmada")

    # 5) the policy last
    rc, _, err = env.kube.run(env.kkc, ["-n", env.ns, "delete", "propagationpolicies.policy.karmada.io",
                                        POLICY_NAME, "--ignore-not-found"])
    if rc != 0:
        rec["errors"].append(f"delete policy: {err.strip()[:300]}")
    left, e = env.leftovers()
    rest = {k: [n for n in v if n in env.lim or n == POLICY_NAME or n.startswith(FHPA_PREFIX)]
            for k, v in left.items()}
    rest = {k: v for k, v in rest.items() if v}
    if e or rest:
        rec["errors"].append(f"objects left in Karmada: {rest or e}")
    rec["ok"] = not rec["errors"]
    return rec


def cmd_status(env: Env) -> Dict[str, Any]:
    states, _, errs = env.members()
    rec: Dict[str, Any] = {
        "managed_members": sorted(f"{d}@{c}" for c, ds in states.items() for d, s in ds.items() if s["managed"]),
        "member_errors": errs,
    }
    if not _require_karmada(env):
        rec["clusters"] = env.karmada_clusters()[0]
        rec["bindings"] = env.bindings()[0]
        rec["fhpa"] = env.fhpas()[0]
        rec["karmada_objects"] = env.leftovers()[0]
    else:
        rec["karmada"] = "not configured (DMOS_KARMADA_KUBECONFIG)"
    rec["ok"] = True
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
    out.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("cmd", choices=["check", "apply", "trace", "verify", "release", "status", "show"])
    ap.add_argument("--variant", choices=VARIANTS, default="fhpa")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--trace", type=Path, default=None, help="trace file (verify)")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=1100.0)
    ap.add_argument("--stop-file", type=Path, default=None,
                    help="trace: stop when this file exists (the harness creates it after the load)")
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    cfg = load_config(args.config_dir)
    def path_env(name: str) -> Optional[str]:
        v = os.environ.get(name) or None
        return os.path.expanduser(v) if v else None
    env = Env(cfg, Kube(), path_env("DMOS_KARMADA_KUBECONFIG"), path_env("DMOS_KARMADA_HOST_KUBECONFIG"))
    if args.cmd == "show":
        print(json.dumps({"propagation_policy": propagation_policy(env.deployments, env.clusters, env.ns,
                                                                   args.variant),
                          "federated_hpa": [federated_hpa(d, env.lim[d], env.ns) for d in env.deployments]},
                         indent=1))
        return 0
    if args.cmd == "trace":
        if args.out is None:
            ap.error("trace needs --out")
        return cmd_trace(env, args.out, args.interval, args.duration, args.stop_file)
    if args.cmd == "apply":
        rec = cmd_apply(env, args.variant)
    elif args.cmd == "verify":
        rec = cmd_verify(env, args.trace)
    else:
        rec = {"check": cmd_check, "release": cmd_release, "status": cmd_status}[args.cmd](env)
    _write(args.out, args.cmd, rec)
    if args.cmd == "status":
        print(json.dumps(rec, indent=1, default=str))
    elif not args.quiet or not rec["ok"]:
        print(f"  [karmada] {args.cmd}: {'ok' if rec['ok'] else 'PROBLEMS'}")
        for x in rec.get("errors", []):
            print(f"  [karmada]   ERROR {x}")
        for x in rec.get("warnings", []):
            print(f"  [karmada]   warn  {x}")
    return 0 if rec["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
