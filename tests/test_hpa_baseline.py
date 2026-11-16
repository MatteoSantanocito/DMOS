"""
Tests for experiments/hpa_baseline.py (arm hpa-none) with a fake, stateful
kubectl: manifests from services.yaml, apply / verify / delete, and the
provenance checks of write_manifest.py for HPA and non-HPA arms.
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"

FAKE_KUBECTL = r'''
import json, os, sys
a = sys.argv[1:]
cl = os.path.basename(a[a.index("--kubeconfig") + 1])
state_file = os.path.join(os.environ["FAKE_HPA_STATE"], cl + ".json")
state = json.load(open(state_file)) if os.path.exists(state_file) else []
def save(s): json.dump(s, open(state_file, "w"))
def out(o): print(json.dumps(o)); sys.exit(0)
if "apiservice" in a:
    ok = "False" if os.environ.get("FAKE_NO_METRICS") else "True"
    out({"status": {"conditions": [{"type": "Available", "status": ok}]}})
if "apply" in a:
    items = json.loads(sys.stdin.read())["items"]
    names = {i["metadata"]["name"] for i in items}
    save([s for s in state if s["metadata"]["name"] not in names] + items); sys.exit(0)
if "delete" in a:
    save([]); sys.exit(0)
if "get" in a and "hpa" in a:
    active = "False" if os.environ.get("FAKE_HPA_INACTIVE") else "True"
    for s in state:
        s["status"] = {"currentReplicas": 1, "desiredReplicas": 1,
                       "conditions": [{"type": "ScalingActive", "status": active},
                                      {"type": "AbleToScale", "status": "True"}]}
    out({"items": state})
sys.exit(1)
'''


def _load(name):
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    shutil.copy(REPO / "config" / "local.yaml.example", cfg / "local.yaml")
    (tmp_path / "fake_kubectl.py").write_text(FAKE_KUBECTL, encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("DMOS_KUBECTL", json.dumps([sys.executable, str(tmp_path / "fake_kubectl.py")]))
    monkeypatch.setenv("FAKE_HPA_STATE", str(state))
    for k in ("FAKE_NO_METRICS", "FAKE_HPA_INACTIVE"):
        monkeypatch.delenv(k, raising=False)
    return _load("hpa_baseline"), cfg, tmp_path


def test_manifests_follow_services_yaml(env):
    hb, cfg, _ = env
    specs = hb.hpa_specs(hb.load_config(cfg))
    assert len(specs) == 6
    de = {i["spec"]["scaleTargetRef"]["name"]: i for i in specs["ms-cluster-01"]}
    pl = {i["spec"]["scaleTargetRef"]["name"]: i for i in specs["ms-cluster-03"]}
    # the services DMOS manages: entry + co-scaled
    assert set(de) == {"frontend", "cartservice", "productcatalogservice", "checkoutservice",
                       "recommendationservice", "currencyservice", "adservice", "shippingservice"}
    assert de["frontend"]["spec"]["maxReplicas"] == 6 and pl["frontend"]["spec"]["maxReplicas"] == 3
    fe = de["frontend"]
    assert fe["spec"]["minReplicas"] == 1
    assert fe["spec"]["metrics"][0]["resource"]["target"]["averageUtilization"] == 70
    assert fe["metadata"]["labels"] == {hb.LABEL_KEY: hb.LABEL_VALUE}
    assert fe["metadata"]["name"].startswith(hb.NAME_PREFIX)


def test_apply_verify_delete_cycle(env):
    hb, cfg, tmp = env
    out = tmp / "hpa_baseline.json"
    assert hb.main(["apply", "--out", str(out), "--config-dir", str(cfg), "--quiet"]) == 0
    assert hb.main(["verify", "--out", str(out), "--config-dir", str(cfg), "--quiet"]) == 0
    assert hb.main(["delete", "--out", str(out), "--config-dir", str(cfg), "--quiet"]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["apply"]["ok"] and rec["verify"]["ok"] and rec["delete"]["ok"]
    assert len(rec["apply"]["clusters"]["ms-cluster-01"]["present"]) == 8
    assert all(c["remaining"] == [] for c in rec["delete"]["clusters"].values())


def test_apply_fails_without_metrics_api(env, monkeypatch):
    hb, cfg, tmp = env
    monkeypatch.setenv("FAKE_NO_METRICS", "1")
    out = tmp / "hpa_baseline.json"
    assert hb.main(["apply", "--out", str(out), "--config-dir", str(cfg), "--quiet"]) == 3
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert not rec["apply"]["ok"]
    assert "metrics API" in rec["apply"]["clusters"]["ms-cluster-01"]["errors"][0]


def test_verify_fails_when_hpa_cannot_read_cpu(env, monkeypatch):
    hb, cfg, tmp = env
    out = tmp / "hpa_baseline.json"
    assert hb.main(["apply", "--out", str(out), "--config-dir", str(cfg), "--quiet"]) == 0
    monkeypatch.setenv("FAKE_HPA_INACTIVE", "1")
    assert hb.main(["verify", "--out", str(out), "--config-dir", str(cfg), "--quiet"]) == 3


# ── provenance checks (write_manifest.py) ──────────────────────────────────

def _manifest_checks(tmp_path, label, hpa_record, any_hpa_pre=False, any_hpa_post=False):
    wm = _load("write_manifest")
    d = tmp_path / "results" / f"20261101_120000_{label}"
    d.mkdir(parents=True)
    assert wm.main(["start", "--run-dir", str(d), "--label", label, "--k6-mode", "remoto",
                    "--k6-remote", "u@x", "--k6-version", "k6", "--k6-args", "",
                    "--peak-rps", "350", "--baseline-rps", "50", "--spike-region", "ms-cluster-01",
                    "--spike-weights", "W1=85", "--phase-times", "180,300,600,720,900",
                    "--test-duration", "900", "--vu-factor", "20", "--no-dmos"]) == 0
    for phase, any_hpa in (("pre", any_hpa_pre), ("post", any_hpa_post)):
        (d / f"env_state_{phase}.json").write_text(json.dumps({
            "verdicts": {"netem_ok": True, "affinity": {"frontend": "none"}, "any_hpa": any_hpa},
            "expectations": {"violations": []}}), encoding="utf-8")
    (d / "flash_crowd_summary.json").write_text(json.dumps({"validity": {"run_valid": True}}),
                                                 encoding="utf-8")
    if hpa_record is not None:
        (d / "hpa_baseline.json").write_text(json.dumps(hpa_record), encoding="utf-8")
    assert wm.main(["finish", "--run-dir", str(d)]) == 0
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return {c["code"] for c in m["provenance"]["checks"] if c["level"] == "error"}, m


GOOD = {"apply": {"ok": True, "applied_at": "t"}, "verify": {"ok": True, "verified_at": "t"},
        "delete": {"ok": True, "deleted_at": "t"}}


def test_hpa_arm_with_active_hpas_is_valid(tmp_path):
    errs, m = _manifest_checks(tmp_path, "dmos-hpa-none__con", GOOD, any_hpa_post=True)
    assert not ({"hpa_baseline_missing", "hpa_not_applied", "hpa_not_active", "hpa_present"} & errs)
    assert m["hpa_baseline"]["verify"] == {"ok": True, "at": "t"}


def test_hpa_arm_problems_are_errors(tmp_path):
    errs, _ = _manifest_checks(tmp_path, "dmos-hpa-none__con", None)
    assert "hpa_baseline_missing" in errs
    bad = {**GOOD, "verify": {"ok": False, "verified_at": "t"}}
    errs, _ = _manifest_checks(tmp_path / "b", "dmos-hpa-none__con", bad, any_hpa_post=True)
    assert "hpa_not_active" in errs


def test_stray_hpa_in_a_dmos_arm_is_an_error(tmp_path):
    errs, _ = _manifest_checks(tmp_path, "dmos-off__con", None, any_hpa_pre=True)
    assert "hpa_present" in errs
