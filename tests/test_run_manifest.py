"""
Tests for experiments/write_manifest.py (run provenance manifest) and
experiments/run_provenance.py (show / audit / diff).
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wm = _load("write_manifest")
rp = _load("run_provenance")


def _start(run_dir, label, no_dmos=False):
    args = ["start", "--run-dir", str(run_dir), "--label", label,
            "--k6-mode", "remoto (ubuntu@x)", "--k6-remote", "ubuntu@x",
            "--k6-version", "k6 v0.54.0", "--k6-args", "-e DMOS_PEAK_RPS=350",
            "--peak-rps", "350", "--baseline-rps", "50", "--spike-region", "ms-cluster-01",
            "--spike-weights", "W1=85,W2=3", "--phase-times", "180,300,600,720,900",
            "--test-duration", "900", "--vu-factor", "20"]
    if no_dmos:
        args.append("--no-dmos")
    assert wm.main(args) == 0


def _write(p: Path, obj):
    p.write_text(json.dumps(obj), encoding="utf-8")


def _live(run_dir, ids):
    with (run_dir / "dmos_live.jsonl").open("w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({"elapsed_s": 0, "dmos_config_id": cid}) + "\n")


def _env(run_dir, phase, netem_ok=True, affinity="none"):
    _write(run_dir / f"env_state_{phase}.json", {
        "captured_at": "2026-10-21T10:00:00+00:00",
        "verdicts": {"netem_ok": netem_ok,
                     "netem_problems": {} if netem_ok else {"ms-cluster-02": "missing"},
                     "affinity": {"frontend": affinity}, "endpoint_sync": {"c1": "false"},
                     "cilium_images": {}, "any_hpa": False},
        "expectations": {"violations": []}})


def _summary(run_dir, valid=True):
    _write(run_dir / "flash_crowd_summary.json",
           {"validity": {"run_valid": valid, "dropped_pct": 0.0 if valid else 0.02}})


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "results" / "20261021_120000_dmos-l1l2__con"
    d.mkdir(parents=True)
    return d


def _finish(run_dir, *extra):
    assert wm.main(["finish", "--run-dir", str(run_dir), *extra]) == 0
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _codes(m, level="error"):
    return {c["code"] for c in m["provenance"]["checks"] if c["level"] == level}


def test_start_keeps_schema1_keys_and_adds_provenance(run_dir):
    _start(run_dir, "dmos-l1l2__con")
    m = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # schema-1 keys, same types as before (paper_lib / paper_report read them)
    assert m["run_label"] == "dmos-l1l2__con" and m["timestamp"] == "20261021_120000"
    assert m["config"]["peak_rps"] == 350 and isinstance(m["config"]["peak_rps"], int)
    assert m["config"]["spike_weights"] == "W1=85,W2=3"
    assert isinstance(m["git"]["dirty_files"], int) and m["git"]["commit"]
    # schema-2
    assert m["schema_version"] == 2
    assert (m["arm"], m["regime"]) == ("dmos-l1l2", "con")
    assert m["k6"]["version"] == "k6 v0.54.0" and m["k6"]["script_sha256"]
    snap = run_dir / "config_snapshot"
    assert (snap / "weights.yaml").is_file() and (snap / "services.yaml").is_file()


def test_sanitized_local_yaml_has_no_addresses(run_dir):
    if not (REPO / "config" / "local.yaml").is_file():
        pytest.skip("no config/local.yaml on this machine")
    _start(run_dir, "dmos-l1l2__con")
    text = (run_dir / "config_snapshot" / "local.sanitized.yaml").read_text(encoding="utf-8")
    assert "kubeconfig_path" not in text and "\nssh:" not in text
    import re
    assert not re.search(r"\bip:\s*\d+\.\d+\.\d+\.\d+", text)


def test_finish_valid_dmos_run(run_dir):
    _start(run_dir, "dmos-l1l2__con")
    _write(run_dir / "dmos_config.json", {"config_id": "abc123abc123", "summary": "[abc123abc123] L1+L2",
                                          "level1": {"warm": {"formula": "S = ..."}, "cold": {}}})
    (run_dir / "dmos.log").write_text("log", encoding="utf-8")
    _live(run_dir, ["abc123abc123"] * 5)
    _env(run_dir, "pre"); _env(run_dir, "post"); _summary(run_dir)
    m = _finish(run_dir, "--k6-exit", "0", "--dmos-config-source", "dump")
    errors = _codes(m)
    assert errors == set(), m["provenance"]["checks"]
    assert m["provenance"]["ok"] is True
    assert m["dmos"]["config_id"] == "abc123abc123"
    assert m["dmos"]["config_ids_observed"] == {"abc123abc123": 5}


@pytest.mark.parametrize("setup,expected", [
    ("no_config", "dmos_config_missing"),
    ("two_ids", "config_changed_during_run"),
    ("mismatch", "config_mismatch"),
    ("netem_pre", "netem_invalid_pre"),
    ("netem_post", "netem_lost_during_run"),
    ("starved", "k6_vu_starved"),
    ("no_summary", "k6_summary_missing"),
])
def test_finish_detects_problems(run_dir, setup, expected):
    _start(run_dir, "dmos-l1l2__con")
    if setup != "no_config":
        _write(run_dir / "dmos_config.json", {"config_id": "aaa", "level1": {}})
    ids = {"two_ids": ["aaa", "bbb"], "mismatch": ["zzz"]}.get(setup, ["aaa"])
    _live(run_dir, ids)
    _env(run_dir, "pre", netem_ok=(setup != "netem_pre"))
    _env(run_dir, "post", netem_ok=(setup != "netem_post"))
    if setup != "no_summary":
        _summary(run_dir, valid=(setup != "starved"))
    m = _finish(run_dir)
    assert expected in _codes(m)
    assert m["provenance"]["ok"] is False


def test_loadgenerator_running_is_an_error(run_dir):
    _start(run_dir, "dmos-l1l2__con")
    _write(run_dir / "dmos_config.json", {"config_id": "aaa", "level1": {}})
    _live(run_dir, ["aaa"]); _env(run_dir, "pre"); _summary(run_dir)
    snap = json.loads((run_dir / "env_state_pre.json").read_text(encoding="utf-8"))
    snap["verdicts"].update(loadgenerator_replicas={"ms-cluster-01": 1, "ms-cluster-02": 0},
                            loadgenerator_running=True)
    _write(run_dir / "env_state_post.json", snap)
    m = _finish(run_dir)
    assert "loadgenerator_running" in _codes(m)
    assert m["env_state"]["post"]["loadgenerator_running"] is True
    assert m["provenance"]["ok"] is False


def test_carbon_trace_fallback_is_a_warning(run_dir):
    _start(run_dir, "dmos-l1l2__con")
    _write(run_dir / "dmos_config.json", {
        "config_id": "aaa", "level1": {},
        "carbon": {"sources": {"ms-cluster-01": "trace:DE_2025_hourly.csv",
                               "ms-cluster-05": "realistic"}}})
    _live(run_dir, ["aaa"]); _env(run_dir, "pre"); _env(run_dir, "post"); _summary(run_dir)
    m = _finish(run_dir)
    assert m["dmos"]["carbon_sources"]["ms-cluster-05"] == "realistic"
    assert "carbon_trace_fallback" in _codes(m, "warn")
    assert "carbon_trace_fallback" not in _codes(m)          # recorded, not invalidating
    msg = next(c["message"] for c in m["provenance"]["checks"] if c["code"] == "carbon_trace_fallback")
    assert "ms-cluster-05" in msg and "ms-cluster-01" not in msg


def test_off_arm_with_dmos_alive_is_invalid(run_dir):
    _start(run_dir, "dmos-off__con", no_dmos=True)
    _live(run_dir, ["leftover-dmos"])
    _env(run_dir, "pre"); _summary(run_dir)
    m = _finish(run_dir)
    assert "dmos_running_in_off_arm" in _codes(m)


def test_finish_is_idempotent(run_dir):
    _start(run_dir, "dmos-l1l2__con")
    _write(run_dir / "dmos_config.json", {"config_id": "aaa", "level1": {}})
    _live(run_dir, ["aaa"]); _env(run_dir, "pre"); _summary(run_dir)
    _finish(run_dir, "--k6-exit", "0", "--dmos-config-source", "http")
    _env(run_dir, "post")
    m = _finish(run_dir)                       # second call, no arguments
    assert m["outcome"]["k6_exit"] == 0
    assert m["dmos"]["config_source"] == "http"
    assert m["env_state"]["post"]["file"] == "env_state_post.json"


def test_audit_flags_mixed_labels_and_invalid_runs(tmp_path, capsys):
    res = tmp_path / "results"
    for ts, cid in (("20261021_100000", "aaa"), ("20261021_110000", "bbb")):
        d = res / f"{ts}_dmos-l1l2__con"
        d.mkdir(parents=True)
        _start(d, "dmos-l1l2__con")
        _write(d / "dmos_config.json", {"config_id": cid, "level1": {}})
        _live(d, [cid]); _env(d, "pre"); _summary(d)
        _finish(d)
    legacy = res / "20260601_090000_dmos-l2__con"
    legacy.mkdir()
    _write(legacy / "manifest.json", {"run_label": "dmos-l2__con"})
    rc = rp.main(["audit", str(res)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "LABELS MIXING DIFFERENT CONFIGURATIONS" in out and "dmos-l1l2__con" in out
    assert "1 legacy" in out


def test_show_and_diff_run(tmp_path, capsys):
    res = tmp_path / "results"
    dirs = []
    for ts, gamma in (("20261021_100000", 1.0), ("20261021_110000", 2.0)):
        d = res / f"{ts}_dmos-l1l2__con"
        d.mkdir(parents=True)
        _start(d, "dmos-l1l2__con")
        cfg = {"config_id": f"id{int(gamma)}", "summary": "s",
               "modes": {"level1_scheduling": True, "demand_floor": False},
               "level1": {"score_profile": "p", "cold_start_seconds": 120,
                          "warm": {"formula": "S", "demand_roles": ["gate"],
                                   "gate": {"enabled": True, "gamma": gamma}},
                          "cold": {"formula": "S"}},
               "level2": {"pd_applied_clamp": 1, "controller": {"pd": {}, "predictor": {}}},
               "safeguards": {"scale_down_cooldown_s": 90, "scale_up_protection_s": 90,
                              "entry_budget_up": 8, "entry_budget_down": 2,
                              "backend_budget_up": 12, "backend_budget_down": 6,
                              "dead_zone_pct": 0.1, "startup_grace_s": 120},
               "coscaling": {"colocation_floor": 0.3}, "dmos_env": {}}
        _write(d / "dmos_config.json", cfg)
        _live(d, [cfg["config_id"]]); _env(d, "pre"); _summary(d)
        _finish(d)
        dirs.append(d)
    assert rp.main(["show", str(dirs[0])]) == 0
    out = capsys.readouterr().out
    assert "config_id   id1" in out and "PROVENANCE" in out and "OK" in out
    assert rp.main(["diff", str(dirs[0]), str(dirs[1])]) == 0
    out = capsys.readouterr().out
    assert "level1.warm.gate.gamma" in out
