"""
Tests for experiments/sim_profiles.py (offline Level-1 placement study).

Run on the template config (config/local.yaml.example) with stand-in carbon
traces, so they work on a clean clone.
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"

# clean (FR, NO) vs dirty (PL, SG) zones; file names as the carbon client expects
STANDIN_CI = {"DE": 350, "FR": 50, "PL": 650, "US-MIDA-PJM": 380, "SG": 480, "NO-NO1": 25}


@pytest.fixture(scope="module")
def sim():
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location("sim_profiles", EXP / "sim_profiles.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_profiles"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(sim, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sim")
    cfg = tmp / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    shutil.copy(REPO / "config" / "local.yaml.example", cfg / "local.yaml")
    traces = tmp / "traces"
    traces.mkdir()
    for zone, ci in STANDIN_CI.items():
        rows = ["Datetime (UTC),Carbon Intensity gCO2eq/kWh (Life cycle)"]
        rows += [f"2025-06-{1 + h // 24:02d} {h % 24:02d}:00:00,{ci}" for h in range(72)]
        (traces / f"{zone}_2025_hourly.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    from loguru import logger
    from src.metrics.carbon_client import CarbonClient
    fleet = sim.load_fleet(cfg)
    cc = dict(fleet.cfg.carbon_raw["carbon_intensity"])
    cc["provider"] = "trace"
    cc["trace"] = {**(cc.get("trace") or {}), "data_dir": str(traces)}
    carbon = CarbonClient(cc)
    logger.remove()
    return fleet, carbon


def _reps(r):
    return [r[k] for k in r if k.startswith("r_")]


def test_profile_names(sim):
    m = sim.build_model("green@nu2@g1", {"score_models": {}})
    assert m.weights["carbon"] == 0.65 and m.parameters["nu"] == 2.0 and m.gate_gamma == 1.0
    assert sim.build_model("off", {}) is None and sim.build_model("l2", {}) is None
    with pytest.raises(SystemExit):
        sim.build_model("nope", {"score_models": {}})


def test_budget_formula_and_concentrated_placement(sim, world):
    fleet, carbon = world
    r = sim.simulate(fleet, "nca", "con", carbon)
    # N = ceil(0.85*350/60)=5 on the hot cluster + 1 on each of the other five
    assert r["N"] == 10
    # hot cluster at its maximum (6 for ms-cluster-01), the others at the minimum
    assert _reps(r) == [6, 1, 1, 1, 1, 1]


def test_uniform_350_leaves_no_room_to_the_weights(sim, world):
    fleet, carbon = world
    a = sim.simulate(fleet, "perf", "uni", carbon)
    b = sim.simulate(fleet, "green", "uni", carbon)
    assert a["N"] == 6 and _reps(a) == _reps(b) == [1] * 6


def test_carbon_profile_moves_replicas_when_the_budget_allows(sim, world):
    fleet, carbon = world
    perf = sim.simulate(fleet, "perf@nu2", "uni-p1000", carbon)
    green = sim.simulate(fleet, "green@nu2", "uni-p1000", carbon)
    assert perf["N"] == green["N"] == 18
    assert green["ci_served"] < perf["ci_served"] - 20      # clearly cleaner
    # replicas move toward the clean clusters FR (index 1) and NO (index 5)
    assert _reps(green)[1] + _reps(green)[5] > _reps(perf)[1] + _reps(perf)[5]


def test_baselines(sim, world):
    fleet, carbon = world
    off = sim.simulate(fleet, "off", "con", carbon)
    l2 = sim.simulate(fleet, "l2", "con", carbon)
    assert _reps(off) == [1] * 6
    assert sum(_reps(l2)) == l2["N"]


def test_latency_model_local_vs_pool(sim, world):
    """Local serving (NCA runs): the hot cluster's replicas carry its whole
    demand, so too few of them saturate; with the shared pool every replica
    carries peak/R and the same replicas suffice."""
    fleet, carbon = world
    off_local = sim.simulate(fleet, "off", "con", carbon, dataplane="local")
    off_pool = sim.simulate(fleet, "off", "con", carbon, dataplane="pool")
    dm_local = sim.simulate(fleet, "nca", "con", carbon, dataplane="local")
    l2_pool = sim.simulate(fleet, "l2", "con", carbon, dataplane="pool")
    assert off_local["slo_viol"] > 0.4            # 1 replica for 85% of the demand
    assert dm_local["slo_viol"] < 0.02            # DMOS puts 6 replicas there
    assert off_pool["slo_viol"] < off_local["slo_viol"]
    assert l2_pool["slo_viol"] < 0.02             # pooled capacity is enough
    # local serving: nothing crosses clusters
    assert dm_local["remote_share"] == 0.0 and dm_local["extra_rtt_ms"] == 0.0


def test_proc_curve_is_monotone(sim):
    prev = (0, 0, 0)
    for lam in range(0, 400, 5):
        p50, p95, f = sim._proc_at(lam)
        assert p95 >= p50 > 0 and 0 <= f <= 1
        assert p95 >= prev[1] - 1e-9 and f >= prev[2] - 1e-9
        prev = (p50, p95, f)


def test_min_suffix_in_the_simulator(sim, world):
    fleet, carbon = world
    assert sim.split_min("green_nu2-min0") == ("green_nu2", 0)
    assert sim.split_min("nca") == ("nca", None)
    r = sim.simulate(fleet, "green_nu2-min0", "bal", carbon)
    assert r["min_replicas"] == 0
    assert min(_reps(r)) == 0                    # some cluster holds no frontend
