"""campaign_ctl.py: Cilium re-apply windows and the validity check of a run."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "campaign_ctl", Path(__file__).resolve().parent.parent / "experiments" / "campaign_ctl.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

ANCHOR = datetime(2026, 10, 29, 1, 22, tzinfo=timezone.utc).timestamp()
CFG = (ANCHOR, 10 * 3600, 80 * 60)


def utc(day, h, m):
    return datetime(2026, 10, day, h, m, tzinfo=timezone.utc).timestamp()


def test_window_free_start_is_kept():
    t = utc(31, 9, 0)                                   # windows 03:22 and 13:22
    assert cc.next_safe_start(t, 24 * 60, CFG) == t


def test_cell_that_would_reach_a_window_waits_for_its_end():
    end = utc(31, 13, 22) + 80 * 60 + cc.AFTER_S       # 14:42 + 8 min
    assert cc.next_safe_start(utc(31, 13, 0), 24 * 60, CFG) == end   # would end at 13:24
    assert cc.next_safe_start(utc(31, 13, 50), 24 * 60, CFG) == end  # inside the window
    assert cc.next_safe_start(utc(31, 12, 55), 24 * 60, CFG) == utc(31, 12, 55)  # ends 13:19


def test_no_windows_when_disabled():
    assert cc.next_safe_start(123.0, 1e6, None) == 123.0


def _run(tmp, name="20261031_160000_dmos-reach-pool-ps__con-p250", **over):
    d = tmp / name
    d.mkdir()
    lc = {"pre_stop": "sleep 15s", "termination_grace_s": 30}
    files = {
        "flash_crowd_summary.json": {"validity": {"run_valid": True}},
        "manifest.json": {"outcome": {"k6_exit": 0}, "provenance": {"ok": True},
                          "dmos": {"enabled": True, "config_id": "a", "expected_config_id": "a"}},
        "env_state_pre.json": {"expectations": {"met": True, "expect_reset": True, "violations": []},
                               "clusters": {f"c{i}": {"deployments": {"frontend": {"lifecycle": lc}}}
                                            for i in range(2)}},
        "env_state_post.json": {},
    }
    files.update(over)
    for f, v in files.items():
        if v is not None:
            (d / f).write_text(json.dumps(v), encoding="utf-8")
    return d


LOG_OK = ("  [fleet] pre-run: ClusterMesh, etichette Hubble, Gateway e pod ok\n"
          "  [fleet] post-run: ClusterMesh, etichette Hubble, Gateway e pod ok\n")


def test_valid_run(tmp_path):
    d = _run(tmp_path)
    assert cc.check_run(d, LOG_OK, 0, ["frontend"], 2) == []
    assert cc.find_run(tmp_path, "reach-pool-ps", "con-p250", 0) == d
    assert cc.find_run(tmp_path, "reach-pool-ps", "con-p250", d.stat().st_mtime + 10 ** 6) is None
    assert cc.find_run(tmp_path, "pool-ps", "con-p250", 0) is None      # exact label only


def test_every_failure_is_reported(tmp_path):
    lc_old = {"pre_stop": None, "termination_grace_s": 5}
    d = _run(tmp_path,
             **{"flash_crowd_summary.json": {"validity": {"run_valid": False}},
                "env_state_pre.json": {"expectations": {"met": False, "expect_reset": False,
                                                        "violations": ["frontend@c0: not reset"]},
                                       "clusters": {"c0": {"deployments": {"frontend": {"lifecycle": lc_old}}}}},
                "env_state_post.json": None})
    (d / "INVALID_DATAPLANE.txt").write_text("x", encoding="utf-8")
    bad = " | ".join(cc.check_run(d, "  [fleet] pre-run: ClusterMesh, etichette Hubble, Gateway e pod ok\n",
                                  1, ["frontend"], 2))
    for want in ("rc=1", "INVALID_DATAPLANE", "k6 run not valid", "not reset", "reset to 1 replica not verified",
                 "1 clusters", "frontend@c0: preStop None grace 5", "no env_state_post", "fleet post-run"):
        assert want in bad, want
    assert "fleet pre-run" not in bad


def test_dmos_config_mismatch(tmp_path):
    d = _run(tmp_path, **{"manifest.json": {"outcome": {"k6_exit": 0}, "provenance": {"ok": True},
                                            "dmos": {"enabled": True, "config_id": "a",
                                                     "expected_config_id": "b"}}})
    assert any("config_id a != expected b" in x for x in cc.check_run(d, LOG_OK, 0, None, 2))
