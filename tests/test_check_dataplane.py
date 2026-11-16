"""
Tests for experiments/check_dataplane.py on synthetic runs: true
distribution (served share = replica share), local serving (served share =
demand share), and runs that cannot tell the two apart.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CL = [f"ms-cluster-0{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def cd():
    spec = importlib.util.spec_from_file_location("check_dataplane", REPO / "experiments" / "check_dataplane.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_dataplane"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path, name, demand, reps, served, cpu_per_rep):
    """A run whose peak snapshots have these replicas, served req/s and CPU."""
    d = tmp_path / name
    d.mkdir()
    summary = {"clusters": [{"name": c, "weight_spike": w} for c, w in zip(CL, demand)],
               "results": {}}
    (d / "flash_crowd_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    snaps = []
    for e in range(300, 620, 5):
        snaps.append({
            "elapsed_s": e,
            "clusters": {c: {"frontend_net_bytes_s": s} for c, s in zip(CL, served)},
            "services": {"frontend": {"clusters": {
                c: {"ready_pods": r, "cpu_cores": r * k} for c, r, k in zip(CL, reps, cpu_per_rep)}}},
        })
    (d / "dmos_live.jsonl").write_text("\n".join(json.dumps(s) for s in snaps), encoding="utf-8")
    return d


CON = [0.85, 0.03, 0.03, 0.03, 0.03, 0.03]


def test_true_distribution_is_pool(cd, tmp_path):
    reps = [2, 2, 2, 2, 2, 1]                        # l2 at con: uniform replicas
    served = [350 * r / 11 for r in reps]            # served ∝ replicas
    run = _run(tmp_path, "20261101_100000_dmos-l2__con", CON, reps, served, [0.10] * 6)
    r = cd.check_run(run)
    assert r["verdict"] == "pool" and r["verdict_served"] == "pool"
    assert r["verdict_cpu_per_replica"] == "pool"


def test_local_serving_is_local(cd, tmp_path):
    reps = [2, 2, 2, 2, 2, 1]
    served = [350 * w for w in CON]                  # served = demand
    cpu = [0.5, 0.02, 0.02, 0.02, 0.02, 0.04]        # hot replicas saturated
    run = _run(tmp_path, "20260606_100000_dmos-l2__con", CON, reps, served, cpu)
    r = cd.check_run(run)
    assert r["verdict"] == "local" and r["verdict_served"] == "local"


def test_uniform_demand_uniform_replicas_is_indistinct(cd, tmp_path):
    uni = [1 / 6] * 6
    run = _run(tmp_path, "20261101_110000_dmos-l2__uni", uni, [2] * 6, [350 / 6] * 6, [0.1] * 6)
    assert cd.check_run(run)["verdict"] == "indistinct"


def test_opposite_cpu_verdict_vetoes(cd):
    assert cd.combine("pool", "local") == "disagree"
    assert cd.combine("pool", "inconsistent") == "pool"       # CPU uncertain: no veto
    assert cd.combine("pool", None) == "pool"
    assert cd.combine("indistinct", "local") == "local"
    assert cd.combine("local", "indistinct") == "local"


def test_main_exit_status(cd, tmp_path):
    reps = [2, 2, 2, 2, 2, 1]
    _run(tmp_path, "20261101_100000_dmos-l2__con", CON, reps, [350 * r / 11 for r in reps], [0.1] * 6)
    assert cd.main(["--results-dir", str(tmp_path)]) == 0
    _run(tmp_path, "20261101_120000_dmos-off__con", CON, [1] * 6, [350 * w for w in CON],
         [0.5, 0.02, 0.02, 0.02, 0.02, 0.02])
    assert cd.main(["--results-dir", str(tmp_path)]) == 3     # a local run is flagged


def test_moving_hot_spot_is_checked_before_the_move(cd, tmp_path):
    # pool data plane; the demand moves DE -> SG at 450 s: the check must use
    # the peak before the move, where the demand is the spike weights
    reps = [2, 2, 2, 2, 2, 1]
    run = _run(tmp_path, "20261101_130000_dmos-l2__con-tosg", CON, reps,
               [350 * r / 11 for r in reps], [0.10] * 6)
    sp = run / "flash_crowd_summary.json"
    s = json.loads(sp.read_text(encoding="utf-8"))
    s["config"] = {"shift": {"enabled": True, "at_s": 450, "ramp_s": 30, "end_s": 480,
                             "region": "ms-cluster-05"}}
    sp.write_text(json.dumps(s), encoding="utf-8")
    r = cd.check_run(run)
    assert r["window_s"] == [330.0, 450.0] and r["verdict"] == "pool"
    assert cd.check_window({"config": {}}) == cd.PEAK
