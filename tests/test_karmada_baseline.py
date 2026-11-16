"""
Tests for experiments/karmada_baseline.py (arms karmada-fhpa / karmada-agg).

Pure parts: manifests from services.yaml, template cleaning, spec comparison,
scheduler flags, adoption and trace checks. Whole cycle: a fake, stateful
kubectl that behaves like Karmada v1.19 as read in its source (docs/
KARMADA-BASELINE.md): a policy over existing Deployments adopts them (label
karmada.io/managed=true, replicas from the binding), a cluster given 0
replicas stays in the binding only with --enable-empty-workload-propagation,
Works carry the binding's permanent-id, deleting a template with
preserveResourcesOnDeletion keeps the member Deployment and strips Karmada's
label. Plus the provenance checks of write_manifest.py and the env gate.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
SERVICES = ["frontend", "cartservice", "productcatalogservice", "checkoutservice",
            "recommendationservice", "currencyservice", "adservice", "shippingservice"]
CLUSTERS = [f"ms-cluster-0{i}" for i in range(1, 7)]

FAKE_KUBECTL = r'''
import json, os, sys
SERVICES = %(services)s
CLUSTERS = %(clusters)s
ST = os.environ["FAKE_K_STATE"]
E = os.environ.get
# one fake controller at a time: the harness calls the members in parallel
import atexit, time
_lock = os.path.join(ST, "lock")
while True:
    try:
        os.close(os.open(_lock, os.O_CREAT | os.O_EXCL)); break
    except (FileExistsError, PermissionError):   # Windows: PermissionError while the lock is being deleted
        time.sleep(0.005)
def _unlock():
    for _ in range(200):
        try:
            os.remove(_lock); return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.005)
atexit.register(_unlock)
a = sys.argv[1:]
kc = os.path.basename(a[a.index("--kubeconfig") + 1]); i = a.index("--kubeconfig"); a = a[:i] + a[i + 2:]
def load(name, default):
    p = os.path.join(ST, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else default
def save(name, obj): json.dump(obj, open(os.path.join(ST, name + ".json"), "w"))
def out(o): print(json.dumps(o)); sys.exit(0)
def fail(msg, rc=1): sys.stderr.write(msg); sys.exit(rc)

def member_default(c):
    deps = {}
    for n, s in enumerate(SERVICES):
        img = "gcr.io/ob/%%s:v0.10.3" %% s
        if E("FAKE_SPEC_DRIFT") and c == "ms-cluster-03" and s == "frontend":
            img = "gcr.io/ob/frontend:v0.10.4"
        deps[s] = {"apiVersion": "apps/v1", "kind": "Deployment",
                   "metadata": {"name": s, "namespace": "online-boutique", "uid": "%%s-%%s" %% (c, s),
                                "generation": 1, "labels": {"app": s},
                                "annotations": {"deployment.kubernetes.io/revision": "3",
                                                "kubectl.kubernetes.io/last-applied-configuration": "{}"}},
                   "spec": {"replicas": 1, "selector": {"matchLabels": {"app": s}},
                            "template": {"metadata": {"labels": {"app": s}},
                                         "spec": {"containers": [{"name": "server", "image": img,
                                                  "resources": {"requests": {"cpu": "100m"}}}]}}},
                   "status": {"replicas": 1, "readyReplicas": 1, "updatedReplicas": 1,
                              "availableReplicas": 1, "observedGeneration": 1}}
    return deps

members = {c: load("m-" + c, None) or member_default(c) for c in CLUSTERS}
k = load("karmada", {"ns": None, "templates": {}, "pp": None, "fhpa": {}, "works": {}})

def divide(total, variant):
    if variant == "agg":
        return [total] + [0] * 5
    base = [2, 2, 1, 1, 0, 0] if total == 6 else [total // 6] * 6
    return base[:6]

def reconcile():
    pp = k["pp"]
    flag = not E("FAKE_NO_EMPTY_FLAG")
    k.setdefault("bindings", {})
    for s in SERVICES:
        t = k["templates"].get(s)
        rbname = s + "-deployment"
        claimed = t is not None and pp is not None and s in [r["name"] for r in pp["spec"]["resourceSelectors"]]
        if claimed:
            variant = "agg" if pp["spec"]["placement"]["replicaScheduling"].get("replicaDivisionPreference") == "Aggregated" else "fhpa"
            reps = divide(t["spec"]["replicas"], variant)
            clusters = [{"name": c, "replicas": r} for c, r in zip(CLUSTERS, reps) if r > 0 or flag]
            preserve = bool(pp["spec"].get("preserveResourcesOnDeletion"))
            k["bindings"][rbname] = {"metadata": {"name": rbname, "labels": {"resourcebinding.karmada.io/permanent-id": "id-" + s}},
                                     "spec": {"replicas": t["spec"]["replicas"], "clusters": clusters,
                                              "preserveResourcesOnDeletion": preserve},
                                     "status": {"conditions": [{"type": "Scheduled", "status": "True", "reason": "Success"},
                                                               {"type": "FullyApplied", "status": "True"}]}}
            listed = {x["name"]: x["replicas"] for x in clusters}
            for c in CLUSTERS:
                wkey = c + "/" + s
                if c in listed:
                    k["works"][wkey] = {"metadata": {"name": s + "-w", "namespace": "karmada-es-" + c,
                                        "labels": {"resourcebinding.karmada.io/permanent-id": "id-" + s}},
                                        "spec": {"preserveResourcesOnDeletion": preserve and not E("FAKE_WORK_NOT_PRESERVED")}}
                    d = members[c].get(s)
                    if d is None:
                        continue
                    first = d["metadata"]["labels"].get("karmada.io/managed") != "true"
                    d["metadata"]["labels"]["karmada.io/managed"] = "true"
                    d["metadata"].setdefault("annotations", {})["work.karmada.io/name"] = s + "-w"
                    if d["spec"]["replicas"] != listed[c]:
                        d["metadata"]["generation"] += 1
                    d["spec"]["replicas"] = listed[c]
                    if first and E("FAKE_ROLLOUT"):
                        d["metadata"]["annotations"]["deployment.kubernetes.io/revision"] = "4"
                    d["status"].update(replicas=listed[c], readyReplicas=listed[c], updatedReplicas=listed[c],
                                       availableReplicas=listed[c], observedGeneration=d["metadata"]["generation"])
                elif wkey in k["works"]:
                    # orphan Work deleted: preserved -> unmanaged, else member object deleted
                    w = k["works"].pop(wkey)
                    if w["spec"]["preserveResourcesOnDeletion"]:
                        d = members[c].get(s)
                        if d: d["metadata"]["labels"].pop("karmada.io/managed", None)
                    else:
                        members[c].pop(s, None)
        elif rbname in k["bindings"] and t is None:
            rb = k["bindings"].pop(rbname)
            for c in CLUSTERS:
                w = k["works"].pop(c + "/" + s, None)
                if w is None:
                    continue
                if w["spec"]["preserveResourcesOnDeletion"]:
                    d = members[c].get(s)
                    if d:
                        d["metadata"]["labels"].pop("karmada.io/managed", None)
                        d["metadata"]["annotations"].pop("work.karmada.io/name", None)
                else:
                    members[c].pop(s, None)

def finish():
    save("karmada", k)
    for c, d in members.items(): save("m-" + c, d)

reconcile()
verb = next((x for x in a if x in ("get", "create", "delete", "patch")), None)
ns_i = a.index("-n") + 1 if "-n" in a else None
rest = [x for j, x in enumerate(a) if x not in (verb,) and j != ns_i and x != "-n"]

if kc == "host.config":
    flag = [] if E("FAKE_NO_EMPTY_FLAG") else ["--enable-empty-workload-propagation=true"]
    items = [{"metadata": {"name": "karmada-scheduler"}, "spec": {"template": {"spec": {"containers": [
                {"image": "docker.io/karmada/karmada-scheduler:v1.19.0", "command": ["/bin/karmada-scheduler"],
                 "args": ["--enable-scheduler-estimator=true"] + flag}]}}}, "status": {"readyReplicas": 1}},
             {"metadata": {"name": "karmada-controller-manager"}, "spec": {"template": {"spec": {"containers": [
                {"image": "docker.io/karmada/karmada-controller-manager:v1.19.0", "args": []}]}}}, "status": {"readyReplicas": 1}}]
    items += [{"metadata": {"name": "karmada-scheduler-estimator-" + c}, "spec": {"template": {"spec": {"containers": [
                {"image": "docker.io/karmada/karmada-scheduler-estimator:v1.19.0"}]}}}, "status": {"readyReplicas": 1}}
              for c in CLUSTERS]
    out({"items": items})

if kc.startswith("kubeconfig-"):
    c = kc[len("kubeconfig-"):]
    if verb == "get" and "deploy" in rest:
        finish(); out({"items": list(members[c].values())})
    fail("unsupported member call %%s" %% a)

# karmada apiserver
res = rest[0] if rest else ""
if verb == "get":
    if res.startswith("clusters"):
        out({"items": [{"metadata": {"name": c}, "spec": {"syncMode": "Push"},
                        "status": {"conditions": [{"type": "Ready", "status": "True"}], "kubernetesVersion": "v1.33.4+k3s1"}}
                       for c in CLUSTERS]})
    if res == "apiservice":
        out({"status": {"conditions": [{"type": "Available", "status": "True"}]}})
    if res == "namespace":
        if k["ns"] is None: fail('Error from server (NotFound): namespaces "online-boutique" not found')
        out(k["ns"])
    if res.startswith("resourcebindings"):
        finish(); out({"items": list(k.get("bindings", {}).values())})
    if res.startswith("federatedhpas"):
        act = "False" if E("FAKE_FHPA_INACTIVE") else "True"
        items = [dict(f, status={"currentReplicas": 6, "desiredReplicas": 6,
                                 "conditions": [{"type": "ScalingActive", "status": act, "reason": "ValidMetricFound"}]})
                 for f in k["fhpa"].values()]
        out({"items": items})
    if res.startswith("deployments"):
        finish(); out({"items": list(k["templates"].values())})
    if res.startswith("propagationpolicies"):
        if len(rest) > 1:
            if k["pp"] is None: fail("NotFound")
            out(k["pp"])
        out({"items": [k["pp"]] if k["pp"] else []})
    if res.startswith("works"):
        finish(); out({"items": list(k["works"].values())})
if verb == "create":
    obj = json.loads(sys.stdin.read())
    for o in (obj["items"] if obj.get("kind") == "List" else [obj]):
        if o["kind"] == "Namespace": k["ns"] = o
        elif o["kind"] == "Deployment": k["templates"][o["metadata"]["name"]] = o
        elif o["kind"] == "PropagationPolicy": k["pp"] = o
        elif o["kind"] == "FederatedHPA": k["fhpa"][o["metadata"]["name"]] = o
    reconcile(); finish(); sys.exit(0)
if verb == "patch":
    p = json.loads(a[a.index("-p") + 1]); k["pp"]["spec"].update(p["spec"]); reconcile(); finish(); sys.exit(0)
if verb == "delete":
    if res.startswith("federatedhpas"): k["fhpa"] = {}
    elif res.startswith("deployments"):
        for n in rest[1:]:
            if not n.startswith("--"): k["templates"].pop(n, None)
    elif res.startswith("propagationpolicies"): k["pp"] = None
    reconcile(); finish(); sys.exit(0)
fail("unsupported karmada call %%s" %% a)
''' % {"services": json.dumps(SERVICES), "clusters": json.dumps(CLUSTERS)}


def _load(name):
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def kb():
    return _load("karmada_baseline")


@pytest.fixture
def env(tmp_path, monkeypatch, kb):
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    shutil.copy(REPO / "config" / "local.yaml.example", cfg / "local.yaml")
    (tmp_path / "fake_kubectl.py").write_text(FAKE_KUBECTL, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    for f in ("karmada.config", "host.config"):
        (tmp_path / f).write_text("fake", encoding="utf-8")
    monkeypatch.setenv("DMOS_KUBECTL", json.dumps([sys.executable, str(tmp_path / "fake_kubectl.py")]))
    monkeypatch.setenv("FAKE_K_STATE", str(state))
    for k in ("FAKE_NO_EMPTY_FLAG", "FAKE_ROLLOUT", "FAKE_SPEC_DRIFT", "FAKE_FHPA_INACTIVE",
              "FAKE_WORK_NOT_PRESERVED", "DMOS_KARMADA_SKIP_HOST_CHECK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(kb, "POLL_S", 0.01)
    monkeypatch.setattr(kb, "ADOPT_TIMEOUT_S", 5)
    monkeypatch.setattr(kb, "RELEASE_TIMEOUT_S", 5)
    cfgobj = kb.load_config(cfg)

    def make(karmada=True, host=True):
        return kb.Env(cfgobj, kb.Kube(), str(tmp_path / "karmada.config") if karmada else None,
                      str(tmp_path / "host.config") if host else None)
    return make, state


def _ok(rec):
    assert rec["ok"], rec.get("errors")
    return rec


def _members(state):
    return {c: json.loads((state / f"m-{c}.json").read_text()) for c in CLUSTERS}


# ── pure parts ───────────────────────────────────────────────────────────────

def test_limits_are_the_sums_of_the_hpa_baseline(kb, env):
    make, _ = env
    e = make()
    assert list(e.lim) == SERVICES
    assert e.lim["frontend"] == {"min": 6, "max": 28, "cpu_target_pct": 70}
    assert e.lim["cartservice"]["max"] == 12 and e.lim["currencyservice"]["max"] == 26


def test_policy_variants(kb):
    pp = kb.propagation_policy(SERVICES, CLUSTERS, "online-boutique", "fhpa")
    s = pp["spec"]
    assert s["conflictResolution"] == "Overwrite" and s["preserveResourcesOnDeletion"] is True
    assert s["placement"]["clusterAffinity"]["clusterNames"] == CLUSTERS
    assert s["placement"]["replicaScheduling"] == {
        "replicaSchedulingType": "Divided", "replicaDivisionPreference": "Weighted",
        "weightPreference": {"dynamicWeight": "AvailableReplicas"}}
    agg = kb.propagation_policy(SERVICES, CLUSTERS, "online-boutique", "agg")["spec"]["placement"]
    assert agg["replicaScheduling"] == {"replicaSchedulingType": "Divided",
                                        "replicaDivisionPreference": "Aggregated"}
    assert [r["name"] for r in s["resourceSelectors"]] == SERVICES
    with pytest.raises(ValueError):
        kb.propagation_policy(SERVICES, CLUSTERS, "online-boutique", "static")


def test_federated_hpa(kb):
    f = kb.federated_hpa("frontend", {"min": 6, "max": 28, "cpu_target_pct": 70}, "online-boutique")
    assert f["apiVersion"] == "autoscaling.karmada.io/v1alpha1" and f["kind"] == "FederatedHPA"
    assert f["spec"]["scaleTargetRef"] == {"apiVersion": "apps/v1", "kind": "Deployment", "name": "frontend"}
    assert (f["spec"]["minReplicas"], f["spec"]["maxReplicas"]) == (6, 28)
    assert f["spec"]["metrics"][0]["resource"]["target"] == {"type": "Utilization", "averageUtilization": 70}
    assert "behavior" not in f["spec"]                      # Kubernetes defaults, as hpa-none


def test_clean_template(kb):
    dep = {"metadata": {"name": "frontend", "namespace": "ob", "uid": "u", "resourceVersion": "9",
                        "labels": {"app": "frontend", "karmada.io/managed": "true"},
                        "annotations": {"deployment.kubernetes.io/revision": "3", "keep": "x",
                                        "kubectl.kubernetes.io/last-applied-configuration": "{}",
                                        "work.karmada.io/name": "w"}},
           "spec": {"replicas": 1, "template": {"spec": {}}}, "status": {"replicas": 1}}
    t = kb.clean_template(dep, 6)
    assert t["metadata"] == {"name": "frontend", "namespace": "ob", "labels": {"app": "frontend"},
                             "annotations": {"keep": "x"}}
    assert t["spec"]["replicas"] == 6 and "status" not in t
    assert dep["spec"]["replicas"] == 1                     # input untouched


def test_spec_comparison_ignores_replicas(kb):
    a = {"spec": {"replicas": 1, "template": {"spec": {"containers": [{"image": "x:1"}]}}}}
    b = {"spec": {"replicas": 4, "template": {"spec": {"containers": [{"image": "x:1"}]}}}}
    c = {"spec": {"replicas": 1, "template": {"spec": {"containers": [{"image": "x:2"}]}}}}
    assert kb.spec_fingerprint(a) == kb.spec_fingerprint(b) != kb.spec_fingerprint(c)
    assert kb.spec_differences(a["spec"], c["spec"]) == ["spec.template.spec.containers[0].image"]


def test_parse_flags(kb):
    assert kb.parse_flags(["/bin/s", "--a=1", "--b", "2", "--c", "--d=x=y"]) == \
        {"--a": "1", "--b": "2", "--c": "true", "--d": "x=y"}


def test_host_findings(kb):
    def dep(name, args, image="docker.io/karmada/%s:v1.19.0", ready=1):
        return {"metadata": {"name": name}, "status": {"readyReplicas": ready},
                "spec": {"template": {"spec": {"containers": [{"image": image % name, "args": args}]}}}}
    est = [dep(f"karmada-scheduler-estimator-{c}", []) for c in CLUSTERS]
    ok = kb.host_findings([dep("karmada-scheduler", ["--enable-scheduler-estimator=true",
                                                     "--enable-empty-workload-propagation=true"])] + est,
                          CLUSTERS, "v1.19.0")
    assert ok["errors"] == [] and ok["warnings"] == []
    bad = kb.host_findings([dep("karmada-scheduler", ["--enable-scheduler-estimator=true"],
                                image="docker.io/karmada/%s:v1.18.2")] + est[:5], CLUSTERS, "v1.19.0")
    assert any("--enable-empty-workload-propagation" in e for e in bad["errors"])
    assert any("ms-cluster-06" in e for e in bad["errors"])
    assert any("v1.19.0" in w for w in bad["warnings"])


def test_adoption_problems(kb):
    st = {"uid": "u", "revision": "3", "managed": True, "replicas": 2, "ready": 2,
          "observed_generation_ok": True}
    before = {c: {"frontend": dict(st, managed=False, replicas=1)} for c in CLUSTERS}
    after = {c: {"frontend": dict(st)} for c in CLUSTERS}
    rb = {"frontend": {"scheduled": True, "fully_applied": True, "scheduled_reason": "Success",
                       "clusters": {c: 2 for c in CLUSTERS}}}
    assert kb.adoption_problems(before, after, rb, CLUSTERS, ["frontend"]) == ([], [])
    after["ms-cluster-02"]["frontend"]["revision"] = "4"
    rb["frontend"]["clusters"].pop("ms-cluster-06")
    pending, hard = kb.adoption_problems(before, after, rb, CLUSTERS, ["frontend"])
    assert any("rolled out" in h for h in hard) and any("dropped" in h for h in hard)


def test_unread_cluster_is_pending_not_missing(kb):
    # a transient read error on one member must not abort the cell as "missing"
    st = {"uid": "u", "revision": "3", "managed": True, "replicas": 1, "ready": 1,
          "observed_generation_ok": True}
    before = {c: {"frontend": dict(st, managed=False)} for c in CLUSTERS}
    after = {c: {"frontend": dict(st)} for c in CLUSTERS if c != "ms-cluster-04"}
    rb = {"frontend": {"scheduled": True, "fully_applied": True, "scheduled_reason": "Success",
                       "clusters": {c: 1 for c in CLUSTERS}}}
    pending, hard = kb.adoption_problems(before, after, rb, CLUSTERS, ["frontend"])
    assert hard == [] and pending == ["ms-cluster-04: not read"]
    after["ms-cluster-04"] = {}                                   # read, and really missing
    _, hard = kb.adoption_problems(before, after, rb, CLUSTERS, ["frontend"])
    assert hard == ["frontend@ms-cluster-04: Deployment missing in the member"]


def test_summarize_trace(kb):
    full = {c: 1 for c in CLUSTERS}
    s1 = {"bindings": {"frontend": {"scheduled": True, "clusters": full}},
          "fhpa": {"frontend": {"scaling_active": True, "desired": 6}}, "clusters_ready": {c: True for c in CLUSTERS}}
    s2 = {"bindings": {"frontend": {"scheduled": True, "clusters": dict(full, **{"ms-cluster-01": 5})}},
          "fhpa": {"frontend": {"scaling_active": True, "desired": 10}},
          "clusters_ready": dict({c: True for c in CLUSTERS}, **{"ms-cluster-04": False})}
    t = kb.summarize_trace([s1, s2, {"error": "x"}], CLUSTERS)
    assert t["clusters_dropped"] == [] and t["clusters_not_ready"] == ["ms-cluster-04"]
    assert t["scaled"] == {"frontend": True} and t["total_replicas_range"] == {"frontend": [6, 10]}
    assert t["sample_errors"] == 1 and t["fhpa_active_fraction"] == 1.0
    s3 = {"bindings": {"frontend": {"scheduled": True, "clusters": {"ms-cluster-01": 6}}}}
    assert len(kb.summarize_trace([s3], CLUSTERS)["clusters_dropped"]) == 5


# ── whole cycle with the fake Karmada ────────────────────────────────────────

def test_full_cycle(kb, env, tmp_path):
    make, state = env
    e = make()
    rec = kb.cmd_apply(e, "fhpa")
    assert rec["ok"], rec["errors"]
    assert rec["restarted"] == [] and rec["host_checked"] and rec["version_ok"]
    assert rec["placement"]["frontend"] == {"ms-cluster-01": 2, "ms-cluster-02": 2, "ms-cluster-03": 1,
                                            "ms-cluster-04": 1, "ms-cluster-05": 0, "ms-cluster-06": 0}
    m = _members(state)
    assert all(m[c][s]["metadata"]["labels"]["karmada.io/managed"] == "true" for c in CLUSTERS for s in SERVICES)
    assert m["ms-cluster-05"]["frontend"]["spec"]["replicas"] == 0          # scaled to 0, still there
    k = json.loads((state / "karmada.json").read_text())
    tpl = k["templates"]["frontend"]
    assert tpl["spec"]["replicas"] == 6 and "status" not in tpl
    assert "deployment.kubernetes.io/revision" not in (tpl["metadata"].get("annotations") or {})
    assert k["ns"]["metadata"]["labels"]["namespace.karmada.io/skip-auto-propagation"] == "true"
    assert len(k["fhpa"]) == 8

    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(kb.sample(e, 0)) + "\n", encoding="utf-8")
    v = kb.cmd_verify(e, trace)
    assert v["ok"], v["errors"]
    assert v["clusters_dropped"] == [] and v["fhpa_inactive"] == [] and v["unmanaged"] == []

    r = kb.cmd_release(e)
    assert r["ok"], r["errors"]
    m = _members(state)
    assert all(s in m[c] for c in CLUSTERS for s in SERVICES)                   # nothing deleted
    assert not any(m[c][s]["metadata"]["labels"].get("karmada.io/managed") for c in CLUSTERS for s in SERVICES)
    k = json.loads((state / "karmada.json").read_text())
    assert k["templates"] == {} and k["pp"] is None and k["fhpa"] == {}
    assert kb.cmd_release(e)["ok"]                                              # idempotent


def test_aggregated_variant(kb, env):
    make, state = env
    rec = kb.cmd_apply(make(), "agg")
    assert rec["ok"], rec["errors"]
    assert rec["placement"]["frontend"]["ms-cluster-01"] == 6
    assert sum(rec["placement"]["frontend"].values()) == 6 and len(rec["placement"]["frontend"]) == 6
    _ok(kb.cmd_release(make()))


def test_spec_drift_is_refused_before_touching_karmada(kb, env, monkeypatch):
    make, state = env
    monkeypatch.setenv("FAKE_SPEC_DRIFT", "1")
    rec = kb.cmd_apply(make(), "fhpa")
    assert not rec["ok"] and "spec differs" in rec["errors"][0]
    assert "containers[0].image" in rec["errors"][0]
    k = json.loads((state / "karmada.json").read_text()) if (state / "karmada.json").exists() else {"templates": {}}
    assert k["templates"] == {}


def test_rollout_on_adoption_is_an_error(kb, env, monkeypatch):
    make, _ = env
    monkeypatch.setenv("FAKE_ROLLOUT", "1")
    rec = kb.cmd_apply(make(), "fhpa")
    assert not rec["ok"] and rec["restarted"]
    _ok(kb.cmd_release(make()))


def test_scheduler_without_empty_propagation_is_refused(kb, env, monkeypatch):
    make, _ = env
    monkeypatch.setenv("FAKE_NO_EMPTY_FLAG", "1")
    chk = kb.cmd_check(make())
    assert not chk["ok"] and any("--enable-empty-workload-propagation" in x for x in chk["errors"])
    rec = kb.cmd_apply(make(), "fhpa")
    assert not rec["ok"]


def test_dropped_cluster_detected_when_host_check_skipped(kb, env, monkeypatch):
    # without the flag a cluster given 0 replicas leaves the binding: the
    # adoption check catches it even when the host is not checked
    make, state = env
    monkeypatch.setenv("FAKE_NO_EMPTY_FLAG", "1")
    monkeypatch.setenv("DMOS_KARMADA_SKIP_HOST_CHECK", "1")
    rec = kb.cmd_apply(make(host=False), "fhpa")
    assert not rec["ok"] and any("dropped" in x for x in rec["errors"])
    assert rec["host_checked"] is False


def test_host_kubeconfig_required_unless_skipped(kb, env):
    make, _ = env
    chk = kb.cmd_check(make(host=False))
    assert not chk["ok"] and any("DMOS_KARMADA_HOST_KUBECONFIG" in x for x in chk["errors"])


def test_release_never_deletes_templates_without_preservation(kb, env, monkeypatch):
    make, state = env
    _ok(kb.cmd_apply(make(), "fhpa"))
    monkeypatch.setenv("FAKE_WORK_NOT_PRESERVED", "1")
    r = kb.cmd_release(make())
    assert not r["ok"] and any("NOT deleted" in x for x in r["errors"])
    k = json.loads((state / "karmada.json").read_text())
    assert len(k["templates"]) == 8                                           # untouched
    assert all(s in _members(state)[c] for c in CLUSTERS for s in SERVICES)
    monkeypatch.delenv("FAKE_WORK_NOT_PRESERVED")
    _ok(kb.cmd_release(make()))


def test_inactive_fhpa_fails_verify(kb, env, monkeypatch, tmp_path):
    make, _ = env
    _ok(kb.cmd_apply(make(), "fhpa"))
    monkeypatch.setenv("FAKE_FHPA_INACTIVE", "1")
    v = kb.cmd_verify(make(), None)
    assert not v["ok"] and v["fhpa_inactive"] == SERVICES
    _ok(kb.cmd_release(make()))


def test_release_without_karmada_kubeconfig_checks_members(kb, env):
    make, state = env
    r = kb.cmd_release(make(karmada=False))
    assert r["ok"] and r["karmada"] == "not configured"
    _ok(kb.cmd_apply(make(), "fhpa"))
    r = kb.cmd_release(make(karmada=False))
    assert not r["ok"] and "DMOS_KARMADA_KUBECONFIG" in r["errors"][0]
    _ok(kb.cmd_release(make()))


def test_apply_refuses_leftovers(kb, env):
    make, _ = env
    _ok(kb.cmd_apply(make(), "fhpa"))
    rec = kb.cmd_apply(make(), "fhpa")
    assert not rec["ok"] and "already" in " ".join(rec["errors"])
    _ok(kb.cmd_release(make()))


def test_trace_stops_on_stop_file(kb, env, tmp_path):
    make, _ = env
    stop = tmp_path / "stop"
    stop.write_text("")
    out = tmp_path / "t.jsonl"
    assert kb.cmd_trace(make(), out, interval=5, duration=60, stop_file=stop) == 0
    assert not out.exists() or out.read_text() == ""


# ── harness integration: naming, env gate, provenance ───────────────────────

def test_naming_kinds(tmp_path):
    rn = _load("run_naming")
    env = tmp_path / "absent.json"   # reference 6-cluster fleet, not the local .cache
    for arm in ("karmada-fhpa", "karmada-agg"):
        c = rn.resolve(arm, "con", env_json=env)
        assert c.kind == "karmada" and c.label == f"dmos-{arm}__con" and c.dmos_env == ""
    with pytest.raises(rn.NamingError):
        rn.resolve("karmada-fhpa-min0", "con", env_json=env)
    with pytest.raises(rn.NamingError, match="reserved"):
        rn.resolve("karmada-budget", "con", env_json=env)


def test_env_gate_flags_karmada_managed():
    ces = _load("capture_env_state")
    dep = {"items": [{"metadata": {"name": "frontend", "labels": {"karmada.io/managed": "true"},
                                   "annotations": {"deployment.kubernetes.io/revision": "3"}},
                      "spec": {"replicas": 1, "template": {"spec": {"containers": [{"name": "server"}]}}},
                      "status": {}}]}
    s = ces.summarize_deployments(dep)
    assert s["frontend"]["karmada_managed"] is True and s["frontend"]["revision"] == "3"
    v = ces.build_verdicts({"ms-cluster-01": {"deployments": s}, "ms-cluster-02": {"deployments": {}}})
    assert v["karmada_managed"] == {"ms-cluster-01": ["frontend"]} and v["any_karmada_managed"]
    assert any("Karmada" in x for x in ces.check_expectations(v, False, None, False, None, False, True))
    assert ces.check_expectations(v, False, None, False, None, False, False) == []


def _manifest_for(tmp_path, label, kb_json=None, managed_pre=False):
    wm = _load("write_manifest")
    d = tmp_path / "results" / f"20261101_120000_{label}"
    d.mkdir(parents=True)
    assert wm.main(["start", "--run-dir", str(d), "--label", label, "--no-dmos",
                    "--peak-rps", "350", "--phase-times", "180,300,600,720,900"]) == 0
    snap = {"captured_at": "x", "verdicts": {"netem_ok": True, "netem_problems": {}, "affinity": {"frontend": "none"},
                                             "any_hpa": False, "any_karmada_managed": managed_pre,
                                             "karmada_managed": {"ms-cluster-01": ["frontend"]} if managed_pre else {}},
            "expectations": {"violations": []}}
    (d / "env_state_pre.json").write_text(json.dumps(snap), encoding="utf-8")
    (d / "flash_crowd_summary.json").write_text(json.dumps({"validity": {"run_valid": True, "dropped_pct": 0}}),
                                                encoding="utf-8")
    if kb_json is not None:
        (d / "karmada_baseline.json").write_text(json.dumps(kb_json), encoding="utf-8")
    assert wm.main(["finish", "--run-dir", str(d)]) == 0
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return {c["code"] for c in m["provenance"]["checks"] if c["level"] == "error"}, \
        {c["code"] for c in m["provenance"]["checks"] if c["level"] == "warn"}


GOOD = {"apply": {"ok": True, "errors": [], "warnings": [], "restarted": [], "host_checked": True,
                  "version_ok": True, "variant": "fhpa", "placement": {}},
        "verify": {"ok": True, "errors": [], "warnings": [], "clusters_dropped": [], "clusters_not_ready": [],
                   "fhpa_inactive": [], "unmanaged": [], "never_scaled": False},
        "release": {"ok": True, "errors": [], "warnings": []}}


def test_manifest_checks_for_karmada_arms(tmp_path):
    errs, warns = _manifest_for(tmp_path / "a", "dmos-karmada-fhpa__con", GOOD)
    assert not {e for e in errs if "karmada" in e or "fhpa" in e}
    bad = json.loads(json.dumps(GOOD))
    bad["apply"].update(ok=False, restarted=["frontend@ms-cluster-02: pod template rolled out"])
    bad["verify"].update(ok=False, clusters_dropped=["frontend@ms-cluster-05"], fhpa_inactive=["frontend"])
    bad["verify"]["never_scaled"] = True
    bad["release"]["ok"] = False
    errs, warns = _manifest_for(tmp_path / "b", "dmos-karmada-fhpa__con", bad)
    assert {"karmada_pod_restarted", "karmada_not_adopted", "karmada_cluster_dropped",
            "fhpa_not_active"} <= errs
    assert {"karmada_not_released", "fhpa_never_scaled"} <= warns
    errs, _ = _manifest_for(tmp_path / "c", "dmos-karmada-agg__con", None)
    assert "karmada_baseline_missing" in errs


def test_manifest_flags_karmada_in_other_arms(tmp_path):
    errs, _ = _manifest_for(tmp_path / "a", "dmos-off__con", None, managed_pre=True)
    assert "karmada_present" in errs
    errs, _ = _manifest_for(tmp_path / "b", "dmos-off__con", None, managed_pre=False)
    assert "karmada_present" not in errs
