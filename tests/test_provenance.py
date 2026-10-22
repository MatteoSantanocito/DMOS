"""
Tests for src/utils/provenance.py and the DMOS --dump-config-only path.

The orchestrator is built for real, in dry_run mode (no port, no Kubernetes
client, no Prometheus probe), so the effective configuration is read from
the same live objects a production run uses.
"""
import json
import shutil
from pathlib import Path

import pytest

from src.utils import provenance as prov

REPO = Path(__file__).resolve().parent.parent
ALGO_ENV = ("DMOS_SCORE_GATE", "DMOS_SCORE_GATE_GAMMA", "DMOS_SCORE_GATE_EPS",
            "DMOS_LOCALITY_PURE_NET", "DMOS_WEIGHTS_COHERENT", "DMOS_DEMAND_FLOOR",
            "DMOS_SCORE_PROFILE", "DMOS_PREDICTOR_WINDOW_S", "DMOS_PREDICTOR_HORIZON_S",
            "DMOS_CONFIG_DUMP")


@pytest.fixture
def config_dir(tmp_path):
    """Repo config files + local.yaml (or the example template on a fresh clone)."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    local = REPO / "config" / "local.yaml"
    shutil.copy(local if local.is_file() else REPO / "config" / "local.yaml.example",
                cfg / "local.yaml")
    return cfg


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ALGO_ENV:
        monkeypatch.delenv(k, raising=False)


def _build(config_dir, **kw):
    from src.dmos_main import DMOSOrchestrator
    return DMOSOrchestrator(config_path=str(config_dir), dry_run=True, **kw)


def test_dry_run_has_no_side_effects(config_dir):
    orch = _build(config_dir)
    assert orch.k8s is None
    assert orch.effective_config["config_id"] == orch.config_id
    assert len(orch.config_id) == 12


def test_fingerprint_stable_and_ignores_volatile_fields(config_dir, monkeypatch):
    a = _build(config_dir).effective_config
    monkeypatch.setenv("DMOS_UNRELATED_FLAG", "1")      # recorded, not algorithmic
    b = _build(config_dir).effective_config
    assert a["config_id"] == b["config_id"]
    assert b["dmos_env"].get("DMOS_UNRELATED_FLAG") == "1"


@pytest.mark.parametrize("var,value", [
    ("DMOS_SCORE_GATE", "1"),
    ("DMOS_DEMAND_FLOOR", "0"),
    ("DMOS_LOCALITY_PURE_NET", "1"),
    ("DMOS_PREDICTOR_WINDOW_S", "30"),
])
def test_fingerprint_changes_with_the_algorithm(config_dir, monkeypatch, var, value):
    base = _build(config_dir).config_id
    monkeypatch.setenv(var, value)
    assert _build(config_dir).config_id != base


def test_gate_gamma_changes_fingerprint(config_dir, monkeypatch):
    monkeypatch.setenv("DMOS_SCORE_GATE", "1")
    g1 = _build(config_dir).config_id
    monkeypatch.setenv("DMOS_SCORE_GATE_GAMMA", "2")
    g2 = _build(config_dir).config_id
    assert g1 != g2


def test_l2_only_mode_changes_fingerprint(config_dir):
    assert _build(config_dir).config_id != _build(config_dir, no_scheduling=True).config_id


def test_snapshot_matches_live_objects(config_dir, monkeypatch):
    monkeypatch.setenv("DMOS_SCORE_GATE", "1")
    monkeypatch.setenv("DMOS_SCORE_GATE_GAMMA", "2")
    monkeypatch.setenv("DMOS_LOCALITY_PURE_NET", "1")
    monkeypatch.setenv("DMOS_WEIGHTS_COHERENT", "0.25,0.35,0.30,0.10")
    orch = _build(config_dir)
    cfg = orch.effective_config
    warm, cold = cfg["level1"]["warm"], cfg["level1"]["cold"]
    for snap, sf in ((warm, orch.scheduler.score_func_warm), (cold, orch.scheduler.score_func_cold)):
        assert snap["weights"] == {
            "latency": sf.omega_latency, "capacity": sf.omega_capacity,
            "locality": sf.omega_geo_load, "carbon": sf.omega_carbon,
            "pressure": sf.omega_demand, "demand": sf.omega_demand_share}
        assert snap["gate"] == {"enabled": True, "eps": sf.gate_eps, "gamma": sf.gate_gamma}
    assert warm["weights"] == {"latency": 0.25, "capacity": 0.35, "locality": 0.30,
                               "carbon": 0.10, "pressure": 0.0, "demand": 0.0}
    assert warm["demand_mode"] == "legacy"
    assert warm["formula"] == "S = G(d; eps=0.05, g=2) * (0.25*lat + 0.35*cap + 0.3*net + 0.1*car)"
    assert warm["demand_roles"] == ["gate"]
    assert warm["weights_source"] == "env:DMOS_WEIGHTS_COHERENT"


def test_level2_and_safeguards_recorded(config_dir):
    cfg = _build(config_dir).effective_config
    ctl = cfg["level2"]["controller"]
    assert cfg["level2"]["pd_applied_clamp"] == 1
    assert ctl["pd"] == {"kp": 2.0, "kd": 20.0, "deadband": 0.2,
                         "output_min": -3.0, "output_max": 3.0}
    assert ctl["predictor"]["window_s"] == 60 and ctl["predictor"]["horizon_s"] == 120
    assert ctl["predictor"]["trend_factor_up"] == 0.5 and ctl["predictor"]["cap_multiple"] == 2.0
    assert ctl["max_delta_up"] == 6 and ctl["max_delta_down"] == 1
    sg = cfg["safeguards"]
    assert (sg["scale_down_cooldown_s"], sg["scale_up_protection_s"]) == (90, 90)
    assert (sg["entry_budget_up"], sg["entry_budget_down"]) == (8, 2)
    assert (sg["backend_budget_up"], sg["backend_budget_down"]) == (12, 6)
    assert sg["dead_zone_pct"] == 0.10 and sg["startup_grace_s"] == 120
    assert cfg["coscaling"]["colocation_floor"] == 0.3
    assert cfg["level1"]["eligibility"]["cpu_hard_limit_pct"] == 95.0


def test_config_dump_written_atomically(config_dir, tmp_path):
    out = tmp_path / "run" / "dmos_config.json"
    orch = _build(config_dir, config_dump_path=str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["config_id"] == orch.config_id
    assert data["summary"].startswith(f"[{orch.config_id}]")
    assert not out.with_name(out.name + ".tmp").exists()


def test_l1l2_arm_reproduces_the_nca_campaign_configuration(config_dir, monkeypatch):
    """The `l1l2` arm of experiments/run_naming.py must build the score that the
    DMOS logs of the NCA 2026 campaign (15-17 June 2026) printed at every start."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("run_naming", REPO / "experiments" / "run_naming.py")
    rn = importlib.util.module_from_spec(spec)
    sys.modules["run_naming"] = rn
    spec.loader.exec_module(rn)
    for kv in rn.L1L2_ENV.split():
        k, v = kv.split("=", 1)
        monkeypatch.setenv(k, v)
    cfg = _build(config_dir).effective_config
    assert cfg["level1"]["warm"]["formula"] == \
        "S = G(d; eps=0.05, g=2) * (0.25*lat + 0.2*cap + 0.25*net + 0.3*car)"
    assert cfg["modes"]["demand_floor"] is False
    assert cfg["level1"]["warm"]["demand_roles"] == ["gate"]


def test_carbon_sources_recorded_per_cluster(config_dir):
    orch = _build(config_dir)
    cfg = orch.effective_config
    sources = cfg["carbon"]["sources"]
    assert set(sources) == set(orch.config.clusters)
    for name, c in orch.config.clusters.items():
        region = (c.carbon or {}).get("region_code")
        assert sources[name] == orch.scheduler.carbon_client.source_for(region)
        if region == "DE":
            assert sources[name].startswith("trace:DE")
        if region == "NO":                          # via trace.zone_aliases
            assert sources[name].startswith("trace:NO-NO1")
    n_trace = sum(1 for s in sources.values() if str(s).startswith("trace:"))
    assert f"carbon trace {n_trace}/{len(sources)}" in cfg["summary"]


def test_carbon_source_change_changes_fingerprint(config_dir):
    base = _build(config_dir).config_id
    text = (config_dir / "carbon.yaml").read_text(encoding="utf-8")
    # drop the NO -> NO-NO1 alias: Norway falls back to the synthetic profile
    (config_dir / "carbon.yaml").write_text(
        text.replace('"NO": "NO-NO1"', '"XX": "NO-NO1"'), encoding="utf-8")
    orch = _build(config_dir)
    regions = {(c.carbon or {}).get("region_code") for c in orch.config.clusters.values()}
    if "NO" in regions:
        assert orch.config_id != base
        assert not any(str(s).startswith("trace:NO") for s in orch.effective_config["carbon"]["sources"].values())


def test_render_formula_variants():
    class SF:
        omega_latency, omega_capacity, omega_geo_load, omega_carbon, omega_demand = .1, .2, .35, .1, .25
        locality_pure_net, gate_enabled, gate_eps, gate_gamma = False, False, 0.05, 1.0
    assert prov.render_formula(SF) == "S = 0.1*lat + 0.2*cap + 0.35*d*net + 0.1*car + 0.25*pressure"
    assert prov.demand_roles(SF) == ["locality_net", "pressure"]
    SF.gate_enabled = True
    assert prov.render_formula(SF).startswith("S = G(d; eps=0.05, g=1) * (")
    assert prov.demand_roles(SF)[0] == "gate"
