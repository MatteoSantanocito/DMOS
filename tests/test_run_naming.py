"""
Tests for experiments/run_naming.py: one test name = one configuration.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_naming", REPO / "experiments" / "run_naming.py")
rn = importlib.util.module_from_spec(_spec)
sys.modules["run_naming"] = rn          # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(rn)


@pytest.fixture
def env_json(tmp_path):
    regions = ["DE", "FR", "PL", "US-VA", "SG", "NO"]
    p = tmp_path / "test-env.json"
    p.write_text(json.dumps({"clusters": [
        {"name": f"ms-cluster-{i:02d}", "region": r} for i, r in enumerate(regions, start=1)]}),
        encoding="utf-8")
    return p


def _k6(cell):
    return dict(kv.split("=", 1) for kv in cell.k6_env.split())


def test_core_labels_unchanged(env_json):
    # the labels of the NCA 2026 campaign keep their exact form
    for arm in ("off", "l2", "l1l2"):
        for reg in rn.DEMAND_ORDER:
            assert rn.resolve(arm, reg, env_json).label == f"dmos-{arm}__{reg}"


def test_arms_map_to_one_configuration(env_json):
    off = rn.resolve("off", "con", env_json)
    assert off.kind == "off" and off.dmos_env == "" and off.dmos_args == ""
    l2 = rn.resolve("l2", "con", env_json)
    assert l2.kind == "dmos" and l2.dmos_args == "--no-scheduling" and l2.dmos_env == ""
    full = rn.resolve("l1l2", "con", env_json)
    assert full.kind == "dmos" and full.dmos_args == ""
    # the configuration of the NCA 2026 campaign, as read back from its DMOS logs
    assert full.dmos_env.split() == [
        "DMOS_DEMAND_FLOOR=0", "DMOS_SCORE_GATE=1", "DMOS_SCORE_GATE_GAMMA=2",
        "DMOS_LOCALITY_PURE_NET=1", "DMOS_WEIGHTS_COHERENT=0.25,0.20,0.25,0.30",
        "DMOS_ENTRY_BUDGET=per_cluster", "DMOS_FREE_RESOURCES=cluster"]
    assert rn.resolve("legacy-default", "con", env_json).dmos_env == ""
    assert rn.resolve("nca", "con", env_json).dmos_env == "DMOS_SCORE_MODEL=nca"
    assert rn.resolve("profile-latonly", "bal", env_json).dmos_env == "DMOS_SCORE_PROFILE=latonly"
    assert "DMOS_SCORE_GATE=1" in rn.resolve("legacy-gate1", "con", env_json).dmos_env


def test_every_score_model_is_an_arm(env_json):
    models = rn.score_models()
    assert "nca" in models and "gate_g0" in models
    for m in models:
        assert rn.resolve(m, "con", env_json).label == f"dmos-{m}__con"


@pytest.mark.parametrize("old,new", sorted(rn.RENAMED_ARMS.items()))
def test_renamed_arms_are_refused_with_hint(env_json, old, new):
    with pytest.raises(rn.NamingError, match=f"use '{new}'"):
        rn.resolve(old, "con", env_json)


@pytest.mark.parametrize("bad", ["hpa-local", "karmada-budget", "nosuchmodel", "a__b", "NCA", "profile-nope"])
def test_invalid_arms(env_json, bad):
    with pytest.raises(rn.NamingError):
        rn.resolve(bad, "con", env_json)


def test_regime_weights_and_defaults(env_json):
    c = rn.resolve("nca", "bal", env_json)
    k = _k6(c)
    assert [int(k[f"DMOS_SPIKE_W{i}"]) for i in range(1, 7)] == [40, 15, 12, 12, 11, 10]
    assert k["DMOS_PEAK_RPS"] == "350" and k["DMOS_SPIKE_REGION"] == "ms-cluster-01"
    assert "DMOS_VU_FACTOR" not in k


def test_hot_cluster_swaps_the_weights(env_json):
    c = rn.resolve("nca", "con-sg", env_json)
    k = _k6(c)
    assert [int(k[f"DMOS_SPIKE_W{i}"]) for i in range(1, 7)] == [3, 3, 3, 3, 85, 3]
    assert k["DMOS_SPIKE_REGION"] == "ms-cluster-05" and c.hot_region == "SG"
    c = rn.resolve("nca", "bal-pl", env_json)
    assert [int(_k6(c)[f"DMOS_SPIKE_W{i}"]) for i in range(1, 7)] == [12, 15, 40, 12, 11, 10]
    assert rn.resolve("nca", "con-us", env_json).hot_cluster == "ms-cluster-04"


def test_peak_and_canonical_form(env_json):
    c = rn.resolve("l2", "con-p400", env_json)
    assert c.label == "dmos-l2__con-p400"
    assert _k6(c)["DMOS_PEAK_RPS"] == "400" and _k6(c)["DMOS_VU_FACTOR"] == "30"
    assert rn.resolve("l2", "con-p200", env_json).peak_rps == 200
    assert "DMOS_VU_FACTOR" not in _k6(rn.resolve("l2", "con-p200", env_json))
    # defaults are dropped from the label: one cell, one name
    assert rn.resolve("nca", "con-de-p350", env_json).label == "dmos-nca__con"
    assert rn.resolve("nca", "con-sg-p500", env_json).label == "dmos-nca__con-sg-p500"


@pytest.mark.parametrize("bad", ["con400", "hot", "con-xx", "uni-sg", "con-p", "bal-sg-p500-x"])
def test_invalid_regimes(env_json, bad):
    with pytest.raises(rn.NamingError):
        rn.resolve("nca", bad, env_json)


def test_parse_regime_reads_archived_labels():
    assert rn.parse_regime("con400") == {"demand": "con", "hot": None, "to": None, "peak": 400}
    assert rn.regime_share("s65-sg") == 0.65
    assert rn.regime_label("con-sg-p500") == "Concentrated, hot SG @500rps"
    assert rn.regime_share("weird") is None


def test_plan_orders_arms_within_each_regime(env_json):
    cells = rn.plan(["off", "l2", "nca"], ["bal", "con"], env_json)
    assert [c.label for c in cells] == [
        "dmos-off__bal", "dmos-l2__bal", "dmos-nca__bal",
        "dmos-off__con", "dmos-l2__con", "dmos-nca__con"]
    with pytest.raises(rn.NamingError) as e:
        rn.plan(["off", "l1l2gate"], ["bal", "con400"], env_json)
    assert "legacy-gate1" in str(e.value) and "con-p400" in str(e.value)


def test_fleet_size_must_match(tmp_path):
    p = tmp_path / "env.json"
    p.write_text(json.dumps({"clusters": [{"name": "a", "region": "DE"}]}), encoding="utf-8")
    with pytest.raises(rn.NamingError, match="6-cluster"):
        rn.resolve("nca", "con", p)


def test_cli_shell_output_is_evaluable(env_json, capsys):
    assert rn.main(["--env-json", str(env_json), "shell", "nca", "con-sg"]) == 0
    out = capsys.readouterr().out
    assert "CELL_LABEL=dmos-nca__con-sg\n" in out and "\r" not in out
    assert rn.main(["--env-json", str(env_json), "shell", "latonly", "con"]) == 2
    assert "profile-latonly" in capsys.readouterr().err


def test_min_suffix_sets_the_entry_minimum(env_json):
    c = rn.resolve("nca-min0", "bal", env_json)
    assert c.label == "dmos-nca-min0__bal" and c.kind == "dmos"
    assert c.dmos_env.split() == ["DMOS_SCORE_MODEL=nca", "DMOS_ENTRY_MIN_REPLICAS=0"]
    l2 = rn.resolve("l2-min0", "con", env_json)
    assert l2.dmos_args == "--no-scheduling" and l2.dmos_env == "DMOS_ENTRY_MIN_REPLICAS=0"
    full = rn.resolve("l1l2-min1", "con", env_json)
    assert full.dmos_env.endswith("DMOS_ENTRY_MIN_REPLICAS=1")
    assert "DMOS_SCORE_GATE=1" in full.dmos_env          # the base arm is kept
    for arm in ("balanced_nu2-min0", "green_nu2-min0"):
        cell = rn.resolve(arm, "uni", env_json)
        assert "DMOS_ENTRY_MIN_REPLICAS=0" in cell.dmos_env
    with pytest.raises(rn.NamingError):
        rn.resolve("off-min0", "con", env_json)          # no DMOS, no minimum
    with pytest.raises(rn.NamingError):
        rn.resolve("nope-min0", "con", env_json)


def test_seeded_plan_is_a_reproducible_permutation(env_json):
    arms, regs = ["off", "l2", "nca"], ["uni", "bal", "s65", "con"]
    base = [c.label for c in rn.plan(arms, regs, env_json)]
    a = [c.label for c in rn.plan(arms, regs, env_json, seed=101)]
    b = [c.label for c in rn.plan(arms, regs, env_json, seed=101)]
    c = [c.label for c in rn.plan(arms, regs, env_json, seed=102)]
    assert a == b                                     # same seed, same order
    assert sorted(a) == sorted(base) and len(set(a)) == len(a)   # every cell once
    assert a != base or c != base                     # the seed changes the order
    # the arms of a regime still run back to back
    regimes_in_order = [lbl.split("__")[1] for lbl in a]
    blocks = [r for i, r in enumerate(regimes_in_order) if i == 0 or r != regimes_in_order[i - 1]]
    assert len(blocks) == len(regs)


def test_hpa_baseline_arm(env_json):
    c = rn.resolve("hpa-none", "con", env_json)
    assert c.label == "dmos-hpa-none__con" and c.kind == "hpa"
    assert c.dmos_env == "" and c.dmos_args == ""
    with pytest.raises(rn.NamingError):
        rn.resolve("hpa-none-min0", "con", env_json)     # no DMOS, no entry minimum
    with pytest.raises(rn.NamingError):
        rn.resolve("hpa-local", "con", env_json)         # still reserved


# ── moving hot spot (-to<hot2>) ──────────────────────────────────────────────

def test_moving_hot_spot_swaps_the_two_shares(env_json):
    c = rn.resolve("nca", "con-tosg", env_json)
    k = _k6(c)
    assert c.label == "dmos-nca__con-tosg"
    assert [int(k[f"DMOS_SPIKE_W{i}"]) for i in range(1, 7)] == [85, 3, 3, 3, 3, 3]
    assert [int(k[f"DMOS_SHIFT_W{i}"]) for i in range(1, 7)] == [3, 3, 3, 3, 85, 3]
    assert k["DMOS_SHIFT_AT"] == str(rn.SHIFT_AT_S) and k["DMOS_SHIFT_RAMP"] == str(rn.SHIFT_RAMP_S)
    assert k["DMOS_SHIFT_REGION"] == "ms-cluster-05" and c.shift_region == "SG"
    # balanced demand: only the hot cluster and the destination swap their shares
    b = rn.resolve("nca", "bal-tosg", env_json)
    assert b.shift_weights == (11, 15, 12, 12, 40, 10) and b.weights == (40, 15, 12, 12, 11, 10)
    # from a non-default hot cluster, with a peak
    r = rn.resolve("off", "con-sg-tode-p500", env_json)
    assert r.label == "dmos-off__con-sg-tode-p500"
    assert r.weights == (3, 3, 3, 3, 85, 3) and r.shift_weights == (85, 3, 3, 3, 3, 3)


def test_moving_hot_spot_is_inside_the_peak():
    # peak phase 300-600 s: the move and its ramp must fit in it
    assert 300 < rn.SHIFT_AT_S and rn.SHIFT_AT_S + rn.SHIFT_RAMP_S < 600


def test_static_regimes_have_no_shift_variables(env_json):
    for reg in ("con", "bal-sg", "con-p700"):
        c = rn.resolve("nca", reg, env_json)
        assert "DMOS_SHIFT" not in c.k6_env and c.shift_cluster == "" and c.shift_weights == ()
        assert not rn.regime_moves(reg)
    assert rn.regime_moves("con-tosg") and not rn.regime_moves("weird")


@pytest.mark.parametrize("bad", ["uni-tosg", "con-tode", "con-toxx", "con-to", "con-tosg-x"])
def test_invalid_moving_regimes(env_json, bad):
    with pytest.raises(rn.NamingError):
        rn.resolve("nca", bad, env_json)


def test_moving_regime_label_and_share():
    assert rn.regime_label("con-tosg") == "Concentrated, moves to SG"
    assert rn.parse_regime("con-tosg") == {"demand": "con", "hot": None, "to": "sg", "peak": 350}
    assert rn.regime_share("con-tosg") == 0.85
