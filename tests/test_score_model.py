"""
Tests for score models (src/level1/score_model.py): the demand role in the
DMOS score — gate | additive | locality — selected with DMOS_SCORE_MODEL.
"""
import math
import shutil
from pathlib import Path

import pytest
import yaml

from src.level1.score_functions import ClusterMetrics, ScoreFunctions, ScoreParameters
from src.level1.score_model import (ScoreModel, ScoreModelError, select_model,
                                    LEGACY_ENV)

REPO = Path(__file__).resolve().parent.parent
NCA = {"demand_mode": "gate", "gate": {"eps": 0.05, "gamma": 2},
       "weights": {"latency": 0.25, "capacity": 0.35, "net": 0.30, "carbon": 0.10},
       "demand_floor": False, "cold_start": "structural"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in LEGACY_ENV + ("DMOS_SCORE_MODEL",):
        monkeypatch.delenv(k, raising=False)


def _metrics(share=0.6, rtt=100.0):
    return ClusterMetrics(
        cpu_available_cores=4.0, cpu_total_cores=8.0, memory_available_gb=6.0,
        memory_total_gb=12.0, request_rate_current=120.0, request_rate_max=360.0,
        latency_mean_ms=450.0, latency_variance_ms2=100.0,
        carbon_intensity_gco2_kwh=350.0, cost_per_replica_hour=0.1,
        network_rtt_ms=rtt, ingress_rate_rps=210.0, ingress_demand_share=share)


def _components(sf, m):
    return (sf.compute_latency_score(m), sf.compute_capacity_score(m),
            sf.compute_network_score(m), sf.compute_carbon_score(m),
            sf.compute_demand_score(m))


# ── validation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("patch,msg", [
    ({"demand_mode": "weight"}, "demand_mode must be one of"),
    ({"weights": {"latency": 0.5, "capacity": 0.4}}, "must sum to 1.0"),
    ({"weights": {"latency": 0.2, "capacity": 0.3, "net": 0.3, "carbon": 0.1, "demand": 0.1}},
     "only meaningful in additive"),
    ({"weights": {"latency": 1.0, "speed": 0.0}}, "unknown weight keys"),
    ({"gate": {"eps": 1.5, "gamma": 2}}, "gate needs"),
    ({"cold_start": "warm"}, "cold_start must be one of"),
    ({"parameters": {"rhoo": 2.0}}, "unknown parameters"),
])
def test_invalid_models_are_rejected(patch, msg):
    with pytest.raises(ScoreModelError, match=msg):
        ScoreModel.from_dict("bad", {**NCA, **patch})


def test_select_model_from_env_and_unknown_name(monkeypatch):
    raw = {"score_models": {"nca": NCA}}
    assert select_model(raw) is None                       # no env → legacy path
    monkeypatch.setenv("DMOS_SCORE_MODEL", "nca")
    m = select_model(raw)
    assert m.name == "nca" and m.demand_mode == "gate" and m.gate_gamma == 2
    monkeypatch.setenv("DMOS_SCORE_MODEL", "ghost")
    with pytest.raises(ScoreModelError, match="available: nca"):
        select_model(raw)


def test_repository_models_are_all_valid():
    raw = yaml.safe_load((REPO / "config" / "weights.yaml").read_text(encoding="utf-8"))
    names = list(raw["score_models"])
    assert {"nca", "gate_g0", "additive_dem50", "locality"} <= set(names)
    for n in names:
        ScoreModel.from_dict(n, raw["score_models"][n])    # raises if invalid
    nca = ScoreModel.from_dict("nca", raw["score_models"]["nca"])
    assert nca.weights == {"latency": 0.25, "capacity": 0.35, "net": 0.30,
                           "carbon": 0.10, "demand": 0.0}


# ── cold start ──────────────────────────────────────────────────────────────

def test_cold_variant_structural_renormalises():
    cold = ScoreModel.from_dict("nca", NCA).cold_variant()
    assert cold.phase == "cold"
    assert cold.weights["latency"] == 0 and cold.weights["capacity"] == 0
    assert math.isclose(cold.weights["net"], 0.75) and math.isclose(cold.weights["carbon"], 0.25)
    add = ScoreModel.from_dict("a", {"demand_mode": "additive", "weights": {
        "latency": 0.125, "capacity": 0.175, "net": 0.15, "carbon": 0.05, "demand": 0.5}}).cold_variant()
    assert math.isclose(sum(add.weights.values()), 1.0)
    assert math.isclose(add.weights["demand"], 0.5 / 0.7)


def test_cold_variant_without_structural_weight_keeps_warm():
    lat_only = ScoreModel.from_dict("lat", {"demand_mode": "gate",
                                            "weights": {"latency": 0.7, "capacity": 0.3}})
    cold = lat_only.cold_variant()
    assert cold.weights == lat_only.weights and "same as warm" in cold.cold_start_note


# ── the three demand modes, numerically ─────────────────────────────────────

def test_gate_mode_score():
    sf = ScoreFunctions({}, ScoreParameters(), model=ScoreModel.from_dict("nca", NCA))
    m = _metrics(share=0.6)
    lat, cap, net, car, d = _components(sf, m)
    inner = 0.25 * lat + 0.35 * cap + 0.30 * net + 0.10 * car
    gate = (0.05 + 0.95 * d) ** 2
    assert math.isclose(sf.compute_total_score(m), gate * inner, rel_tol=1e-12)
    assert math.isclose(sf.compute_score_breakdown(m)["total_score"], gate * inner, rel_tol=1e-12)


def test_gate_g0_ignores_demand():
    sf = ScoreFunctions({}, model=ScoreModel.from_dict("g0", {**NCA, "gate": {"eps": 0.05, "gamma": 0}}))
    assert math.isclose(sf.compute_total_score(_metrics(share=0.9)),
                        sf.compute_total_score(_metrics(share=0.1)), rel_tol=1e-12)


def test_additive_mode_score():
    model = ScoreModel.from_dict("add", {"demand_mode": "additive", "weights": {
        "latency": 0.125, "capacity": 0.175, "net": 0.15, "carbon": 0.05, "demand": 0.5}})
    sf = ScoreFunctions({}, model=model)
    m = _metrics(share=0.6)
    lat, cap, net, car, d = _components(sf, m)
    expected = 0.125 * lat + 0.175 * cap + 0.15 * net + 0.05 * car + 0.5 * d
    assert math.isclose(sf.compute_total_score(m), expected, rel_tol=1e-12)
    assert not sf.gate_enabled


def test_locality_mode_score():
    model = ScoreModel.from_dict("loc", {"demand_mode": "locality", "weights": NCA["weights"]})
    sf = ScoreFunctions({}, model=model)
    m = _metrics(share=0.6)
    lat, cap, net, car, d = _components(sf, m)
    expected = 0.25 * lat + 0.35 * cap + 0.30 * (d * net) + 0.10 * car
    assert math.isclose(sf.compute_total_score(m), expected, rel_tol=1e-12)
    assert not sf.gate_enabled and sf.omega_demand_share == 0


def test_legacy_path_unchanged():
    w = {"omega_latency": 0.10, "omega_capacity": 0.20, "omega_geo_load": 0.35,
         "omega_carbon": 0.10, "omega_demand": 0.25}
    sf = ScoreFunctions(w, ScoreParameters())
    m = _metrics()
    lat, cap, net, car, d = _components(sf, m)
    pressure = sf.compute_load_score(m)
    expected = 0.10 * lat + 0.20 * cap + 0.35 * (d * net) + 0.10 * car + 0.25 * pressure
    assert sf.demand_mode == "legacy" and sf.omega_demand_share == 0
    assert math.isclose(sf.compute_total_score(m), expected, rel_tol=1e-12)


# ── integration with the orchestrator (dry run) ─────────────────────────────

@pytest.fixture
def config_dir(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    local = REPO / "config" / "local.yaml"
    shutil.copy(local if local.is_file() else REPO / "config" / "local.yaml.example",
                cfg / "local.yaml")
    return cfg


def _orch(config_dir, **kw):
    from src.dmos_main import DMOSOrchestrator
    return DMOSOrchestrator(config_path=str(config_dir), dry_run=True, **kw)


def test_model_overrides_legacy_switches(config_dir, monkeypatch):
    monkeypatch.setenv("DMOS_SCORE_MODEL", "nca")
    monkeypatch.setenv("DMOS_SCORE_GATE_GAMMA", "5")
    monkeypatch.setenv("DMOS_WEIGHTS_COHERENT", "0.7,0.1,0.1,0.1")
    monkeypatch.setenv("DMOS_DEMAND_FLOOR", "1")
    orch = _orch(config_dir)
    warm = orch.scheduler.score_func_warm
    assert warm.gate_gamma == 2 and warm.omega_latency == 0.25
    assert orch.demand_floor_enabled is False                 # from the model
    cfg = orch.effective_config
    assert set(cfg["ignored_legacy_env"]) == {"DMOS_SCORE_GATE_GAMMA", "DMOS_WEIGHTS_COHERENT",
                                              "DMOS_DEMAND_FLOOR"}
    assert cfg["score_model"]["name"] == "nca"
    assert cfg["level1"]["warm"]["demand_mode"] == "gate"
    assert cfg["level1"]["cold"]["formula"] == "S = G(d; eps=0.05, g=2) * (0.75*net + 0.25*car)"


def test_config_id_depends_on_content_not_on_model_name(config_dir, monkeypatch):
    wpath = config_dir / "weights.yaml"
    raw = yaml.safe_load(wpath.read_text(encoding="utf-8"))
    raw["score_models"]["nca_copy"] = dict(raw["score_models"]["nca"], description="same algorithm")
    wpath.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    ids = {}
    for name in ("nca", "nca_copy", "locality", "additive_dem50", "gate_g0"):
        monkeypatch.setenv("DMOS_SCORE_MODEL", name)
        ids[name] = _orch(config_dir).config_id
    assert ids["nca"] == ids["nca_copy"]
    assert len({ids["nca"], ids["locality"], ids["additive_dem50"], ids["gate_g0"]}) == 4
