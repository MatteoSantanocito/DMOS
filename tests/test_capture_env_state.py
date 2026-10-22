"""
Tests for experiments/capture_env_state.py (testbed snapshot + run gate).

The end-to-end tests run the real script against a fake `kubectl` and a fake
`ssh` (small Python programs returning canned output), so they need no
cluster and no network.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "capture_env_state", REPO / "experiments" / "capture_env_state.py")
ces = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ces)


# ── parse_tc_qdisc ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,present,delay,jitter", [
    ("qdisc netem 8001: root refcnt 2 limit 1000 delay 30ms  5ms\n", True, 30.0, 5.0),
    ("qdisc netem 8001: root refcnt 2 limit 1000 delay 200ms  25ms distribution normal",
     True, 200.0, 25.0),
    ("qdisc netem 8001: root refcnt 2 limit 1000 delay 50ms", True, 50.0, None),
    ("qdisc netem 8001: root refcnt 2 limit 1000 delay 1.5s", True, 1500.0, None),
    ("qdisc netem 8001: root refcnt 2 limit 1000 delay 500us", True, 0.5, None),
    ("qdisc fq_codel 0: root refcnt 2 limit 10240p flows 1024", False, None, None),
    ("", False, None, None),
])
def test_parse_tc_qdisc(text, present, delay, jitter):
    p = ces.parse_tc_qdisc(text)
    assert p["present"] is present
    assert p["delay_ms"] == delay
    assert p["jitter_ms"] == jitter


# ── netem_verdict ───────────────────────────────────────────────────────────

def test_netem_verdicts():
    shaped = {"present": True, "delay_ms": 30.0, "jitter_ms": 5.0}
    none = {"present": False, "delay_ms": None, "jitter_ms": None}
    assert ces.netem_verdict(30, shaped, True) == "ok"
    assert ces.netem_verdict(32, shaped, True) == "ok"          # within 3 ms
    assert ces.netem_verdict(100, shaped, True) == "mismatch"
    assert ces.netem_verdict(30, none, True) == "missing"
    assert ces.netem_verdict(0, none, True) == "ok"             # baseline cluster
    assert ces.netem_verdict(0, shaped, True) == "unexpected"
    assert ces.netem_verdict(30, None, False) == "ssh_error"
    assert ces.netem_verdict(30, {"present": True, "delay_ms": None}, True) == "unparsable"


# ── summarize_services ──────────────────────────────────────────────────────

def _svc(name, ann):
    return {"metadata": {"name": name, "annotations": ann}, "spec": {"type": "ClusterIP"}}


def test_summarize_services_affinity_defaults_and_legacy_keys():
    out = ces.summarize_services({"items": [
        _svc("frontend", {"io.cilium/global-service": "true"}),               # legacy key
        _svc("cart", {"service.cilium.io/global": "true",
                      "service.cilium.io/affinity": "local"}),
        _svc("redis", {}),                                                    # not global
        _svc("other", {"service.cilium.io/global": "true",
                       "io.cilium/shared-service": "false"}),
    ]})
    assert out["frontend"]["global"] is True
    assert out["frontend"]["affinity"] == "none"              # Cilium default
    assert out["frontend"]["affinity_annotated"] is False
    assert out["cart"]["affinity"] == "local"
    assert out["redis"]["global"] is False and out["redis"]["affinity"] is None
    assert out["other"]["shared"] == "false"


# ── End-to-end with fake kubectl / ssh ──────────────────────────────────────

FAKE_KUBECTL = r'''
import json, sys, os
a = sys.argv[1:]
kc = a[a.index("--kubeconfig") + 1]
cl = os.path.basename(kc)            # kc-01 / kc-02
def out(o): print(json.dumps(o)); sys.exit(0)
if "svc" in a:
    ann = {"service.cilium.io/global": "true"}
    if cl.endswith("02"):
        ann["service.cilium.io/affinity"] = "local"
    out({"items": [{"metadata": {"name": "frontend", "annotations": ann}, "spec": {}}]})
if "hpa" in a:
    items = [{"metadata": {"name": "hpa-frontend"},
              "spec": {"scaleTargetRef": {"name": "frontend"}, "minReplicas": 1, "maxReplicas": 5}}]
    out({"items": items if cl.endswith("02") else []})
if "configmap" in a:
    out({"data": {"clustermesh-enable-endpoint-sync": "false", "kube-proxy-replacement": "true",
                  "cluster-name": "c" + cl[-2:], "cluster-id": cl[-1]}})
if "daemonset" in a:
    out({"spec": {"template": {"spec": {"containers": [{"image": "quay.io/cilium/cilium:v1.19.4"}]}}}})
if "clustermesh-apiserver" in a:
    out({"spec": {"replicas": 1}, "status": {"readyReplicas": 1}})
if "deploy" in a:
    items = [{"metadata": {"name": "frontend"}, "spec": {"replicas": 2, "template": {"spec": {
        "containers": [{"name": "server", "image": "fe:v1", "resources": {"limits": {"cpu": "500m"}}}]}}},
        "status": {"readyReplicas": 2, "availableReplicas": 2}}]
    lg = os.environ.get("FAKE_LOADGEN")          # replicas of Online Boutique's loadgenerator
    if lg is not None:
        items.append({"metadata": {"name": "loadgenerator"},
                      "spec": {"replicas": int(lg) if cl.endswith("02") else 0,
                               "template": {"spec": {"containers": [{"name": "main", "image": "lg:v1"}]}}},
                      "status": {}})
    out({"items": items})
if "nodes" in a:
    out({"items": [{"metadata": {"name": "w1"}, "spec": {},
                    "status": {"conditions": [{"type": "Ready", "status": "True"}],
                               "allocatable": {"cpu": "8"}, "nodeInfo": {"kubeletVersion": "v1.32"}}}]})
sys.exit(1)
'''

FAKE_SSH = r'''
import sys, os
target = [x for x in sys.argv[1:] if "@" in x][0]
mode = os.environ.get("FAKE_NETEM_MODE", "ok")
if target.endswith("10.0.0.2"):
    if mode == "missing":
        print("qdisc fq_codel 0: root refcnt 2")
    else:
        print("qdisc netem 8001: root refcnt 2 limit 1000 delay 30ms  5ms")
else:
    print("qdisc fq_codel 0: root refcnt 2")
'''


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    (tmp_path / "fake_kubectl.py").write_text(FAKE_KUBECTL, encoding="utf-8")
    (tmp_path / "fake_ssh.py").write_text(FAKE_SSH, encoding="utf-8")
    for n in ("01", "02"):
        (tmp_path / f"kc-{n}").write_text("apiVersion: v1\n", encoding="utf-8")
    env = {
        "ssh": {"user": "ubuntu", "key": ""},
        "clusters": [
            {"name": "ms-cluster-01", "ip": "10.0.0.1", "kubeconfig_path": str(tmp_path / "kc-01"),
             "prometheus_url": "", "netem": {"delay_ms": 0, "interface": "eth0"}},
            {"name": "ms-cluster-02", "ip": "10.0.0.2", "kubeconfig_path": str(tmp_path / "kc-02"),
             "prometheus_url": "", "netem": {"delay_ms": 30, "jitter_ms": 5, "interface": "eth0"}},
        ],
    }
    env_path = tmp_path / "test-env.json"
    env_path.write_text(json.dumps(env), encoding="utf-8")
    monkeypatch.setenv("DMOS_KUBECTL", json.dumps([sys.executable, str(tmp_path / "fake_kubectl.py")]))
    monkeypatch.setenv("DMOS_SSH", json.dumps([sys.executable, str(tmp_path / "fake_ssh.py")]))
    return tmp_path, env_path


def _capture(env_path, out, *extra):
    rc = ces.main(["--phase", "pre", "--out", str(out), "--env-json", str(env_path),
                   "--skip-prometheus", "--quiet", *extra])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_capture_records_testbed_state(fake_env):
    tmp, env_path = fake_env
    rc, snap = _capture(env_path, tmp / "pre.json")
    assert rc == 0
    v = snap["verdicts"]
    assert v["netem"] == {"ms-cluster-01": "ok", "ms-cluster-02": "ok"}
    assert v["netem_ok"] is True
    assert v["affinity"]["frontend"] == "MIXED"           # none on 01, local on 02
    assert v["affinity_by_cluster"]["frontend"] == {"ms-cluster-01": "none", "ms-cluster-02": "local"}
    assert v["endpoint_sync"] == {"ms-cluster-01": "false", "ms-cluster-02": "false"}
    assert v["any_hpa"] is True and v["hpa"]["ms-cluster-02"] == ["hpa-frontend"]
    assert v["cilium_images"]["ms-cluster-01"].endswith("v1.19.4")
    c2 = snap["clusters"]["ms-cluster-02"]
    assert c2["netem"]["delay_ms"] == 30.0 and c2["netem"]["jitter_ms"] == 5.0
    assert c2["deployments"]["frontend"]["resources"]["limits"]["cpu"] == "500m"
    assert snap["expectations"]["met"] is True
    assert snap["test_env"]["sha256"]


def test_gate_netem_missing(fake_env, monkeypatch):
    tmp, env_path = fake_env
    monkeypatch.setenv("FAKE_NETEM_MODE", "missing")
    rc, snap = _capture(env_path, tmp / "pre.json", "--require-netem")
    assert rc == 3
    assert snap["verdicts"]["netem"]["ms-cluster-02"] == "missing"
    assert any("netem missing on ms-cluster-02" in x for x in snap["expectations"]["violations"])


def test_gate_affinity_and_hpa(fake_env):
    tmp, env_path = fake_env
    rc, snap = _capture(env_path, tmp / "pre.json", "--expect-affinity", "none", "--expect-no-hpa")
    assert rc == 3
    viol = " | ".join(snap["expectations"]["violations"])
    assert "frontend@ms-cluster-02: affinity=local" in viol
    assert "HPA present" in viol
    rc_ok, _ = _capture(env_path, tmp / "pre2.json", "--require-netem")
    assert rc_ok == 0                                       # netem fine, no other gate


def test_gate_loadgenerator(fake_env, monkeypatch):
    tmp, env_path = fake_env
    # not deployed at all: recorded as None, never a violation
    rc, snap = _capture(env_path, tmp / "pre0.json", "--expect-no-loadgenerator")
    assert rc == 0
    assert snap["verdicts"]["loadgenerator_replicas"] == {"ms-cluster-01": None, "ms-cluster-02": None}
    assert snap["verdicts"]["loadgenerator_running"] is False
    # deployed but scaled to 0 everywhere: fine
    monkeypatch.setenv("FAKE_LOADGEN", "0")
    rc, snap = _capture(env_path, tmp / "pre1.json", "--expect-no-loadgenerator")
    assert rc == 0 and snap["verdicts"]["loadgenerator_replicas"] == {"ms-cluster-01": 0, "ms-cluster-02": 0}
    # running on cluster 02: recorded; gate only when asked
    monkeypatch.setenv("FAKE_LOADGEN", "1")
    rc, snap = _capture(env_path, tmp / "pre2.json")
    assert rc == 0 and snap["verdicts"]["loadgenerator_running"] is True
    rc, snap = _capture(env_path, tmp / "pre3.json", "--expect-no-loadgenerator")
    assert rc == 3
    assert snap["expectations"]["expect_no_loadgenerator"] is True
    assert any("loadgenerator running" in x and "ms-cluster-02" in x
               for x in snap["expectations"]["violations"])


def test_missing_kubeconfig_is_recorded_not_fatal(fake_env):
    tmp, env_path = fake_env
    env = json.loads(env_path.read_text(encoding="utf-8"))
    env["clusters"][0]["kubeconfig_path"] = str(tmp / "does-not-exist")
    env_path.write_text(json.dumps(env), encoding="utf-8")
    rc, snap = _capture(env_path, tmp / "pre.json")
    assert rc == 0
    assert "kubeconfig not found" in snap["verdicts"]["collection_errors"]["ms-cluster-01"][0]
    assert snap["verdicts"]["netem"]["ms-cluster-01"] == "ok"   # netem still checked
