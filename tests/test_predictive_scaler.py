"""
Tests for the predictive Level-2 scaler (src/level2/predictive_scaler.py)
and its wiring: arm naming (-ps), the fast PromQL window, the provenance.
"""
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import pytest

from src.level2.predictive_scaler import (HoltForecaster, PredictiveScaler,
                                          PredictiveScalerParams, scaler_mode)

REPO = Path(__file__).resolve().parent.parent
CL = ["a", "b", "c"]


def _ps(**kw):
    p = PredictiveScalerParams(**kw)
    p.validate()
    return PredictiveScaler(p, CL, capacity={c: 10.0 for c in CL},
                            min_replicas={c: 1 for c in CL}, max_replicas={c: 6 for c in CL})


# ── forecaster ───────────────────────────────────────────────────────────────

def test_holt_constant_series_has_no_trend_and_no_error():
    f = HoltForecaster(alpha=0.5, beta=0.3, damping_s=120, mad_ewma=0.2)
    for i in range(30):
        f.update(10.0 * i, 40.0)
    assert f.level == pytest.approx(40.0)
    assert f.trend == pytest.approx(0.0, abs=1e-9)
    assert f.sigma == pytest.approx(0.0, abs=1e-9)
    assert f.forecast(60) == pytest.approx(40.0)


def test_holt_learns_the_slope_of_a_ramp():
    f = HoltForecaster(alpha=0.5, beta=0.3, damping_s=120, mad_ewma=0.2)
    for i in range(40):                       # +2 req/s every second, sampled every 10 s
        f.update(10.0 * i, 2.0 * 10.0 * i)
    assert f.trend == pytest.approx(2.0, rel=0.05)
    h = 40.0
    expected = f.level + f.trend * 120 * (1 - math.exp(-h / 120))
    assert f.forecast(h) == pytest.approx(expected)
    assert f.forecast(h) > f.level            # a rising ramp is anticipated


def test_damped_horizon_saturates_at_tau():
    f = HoltForecaster(alpha=0.5, beta=0.3, damping_s=100, mad_ewma=0.2)
    assert f.effective_horizon(10) == pytest.approx(100 * (1 - math.exp(-0.1)))
    assert f.effective_horizon(10_000) == pytest.approx(100, rel=1e-6)
    assert f.effective_horizon(0) == 0


def test_falling_trend_is_not_extrapolated_downwards():
    f = HoltForecaster(alpha=0.5, beta=0.3, damping_s=120, mad_ewma=0.2)
    for i in range(20):
        f.update(10.0 * i, max(0.0, 200 - 10.0 * i))
    assert f.trend < 0
    assert f.forecast(60) == pytest.approx(f.level)        # rising_only
    assert f.forecast(60, rising_only=False) < f.level


# ── parameters ───────────────────────────────────────────────────────────────

def test_params_from_env_and_validation():
    p = PredictiveScalerParams.from_env({"DMOS_PS_ALPHA": "0.7", "DMOS_PS_DOWN_WINDOW_S": "90",
                                         "DMOS_PS_FAST_WINDOW": "20s"})
    assert (p.alpha, p.down_window_s, p.fast_window) == (0.7, 90.0, "20s")
    with pytest.raises(ValueError):
        PredictiveScalerParams.from_env({"DMOS_PS_ALPHA": "abc"})
    with pytest.raises(ValueError):
        PredictiveScalerParams.from_env({"DMOS_PS_ALPHA": "0"})
    with pytest.raises(ValueError):
        PredictiveScalerParams.from_env({"DMOS_PS_PANIC_RATIO": "0.5"})


def test_scaler_mode():
    assert scaler_mode({}) == "reactive"
    assert scaler_mode({"DMOS_SCALER": "Predictive"}) == "predictive"
    with pytest.raises(ValueError):
        scaler_mode({"DMOS_SCALER": "magic"})


# ── tick: sizing and decision gating ─────────────────────────────────────────

def test_steady_demand_is_sized_on_capacity_and_decided_periodically():
    ps = _ps()
    cur = {c: 3 for c in CL}
    r0 = ps.tick(0, {c: 25 for c in CL}, {c: 25 for c in CL}, cur)
    assert all(z.need == 3 for z in r0.clusters.values())     # ceil(25 / 10)
    assert r0.decide and r0.reason == "periodic"               # first tick
    ps.targets(0, {c: 3 for c in CL}, cur)                     # decision taken
    r1 = ps.tick(10, {c: 25 for c in CL}, {c: 25 for c in CL}, cur)
    assert not r1.decide                                       # 10 s < 30 s
    r3 = ps.tick(30, {c: 25 for c in CL}, {c: 25 for c in CL}, cur)
    assert r3.decide and r3.reason == "periodic"


def test_need_above_current_triggers_an_immediate_decision():
    ps = _ps()
    cur = {c: 1 for c in CL}
    ps.tick(0, {c: 5 for c in CL}, {c: 5 for c in CL}, cur)
    ps.targets(0, {c: 1 for c in CL}, cur)
    r = ps.tick(10, {"a": 25, "b": 5, "c": 5}, {"a": 12, "b": 5, "c": 5}, cur)
    assert r.need_total > r.current_total
    assert r.decide and r.reason in ("need_up", "burst")


def test_burst_detection_and_hold():
    ps = _ps(panic_hold_s=60)
    cur = {c: 1 for c in CL}
    r = ps.tick(0, {c: 45 for c in CL}, {c: 10 for c in CL}, cur)   # need 5 each vs 1
    assert r.burst and r.decide and r.reason == "burst"
    assert ps.in_burst(30) and not ps.in_burst(61)


def test_forecast_is_capped_and_noise_is_not_extrapolated():
    ps = _ps(max_forecast_multiple=2.0)
    cur = {c: 1 for c in CL}
    for i, x in enumerate([1, 10, 40, 100]):          # very steep ramp
        r = ps.tick(10.0 * i, {c: x for c in CL}, {c: x for c in CL}, cur)
    assert all(z.forecast_rps <= 2.0 * 100 + 1e-9 for z in r.clusters.values())
    quiet = _ps()
    for i, x in enumerate([0.2, 0.5, 0.9]):           # health-check noise
        rq = quiet.tick(10.0 * i, {c: x for c in CL}, {c: x for c in CL}, cur)
    assert all(z.demand_rps == pytest.approx(0.9) for z in rq.clusters.values())


def test_rising_demand_is_sized_above_the_measurement():
    ps = _ps()
    cur = {c: 1 for c in CL}
    for i in range(8):
        r = ps.tick(10.0 * i, {c: 10.0 + 5.0 * i for c in CL}, {c: 8.0 + 3.0 * i for c in CL}, cur)
    z = r.clusters["a"]
    assert z.forecast_rps > z.fast_rps
    assert z.demand_rps >= z.forecast_rps
    assert z.need >= math.ceil(z.fast_rps / 10)


# ── actuation ────────────────────────────────────────────────────────────────

def test_scale_up_is_immediate_and_bounded_by_the_cap():
    ps = _ps()
    out = ps.targets(0, {"a": 9, "b": 3, "c": 1}, {"a": 1, "b": 1, "c": 1})
    assert out == {"a": 6, "b": 3, "c": 1}                # a capped at its maximum 6


def test_up_budget_per_decision_is_respected():
    ps = _ps(up_budget_per_decision=4)
    out = ps.targets(0, {"a": 6, "b": 6, "c": 1}, {"a": 1, "b": 1, "c": 1})
    assert sum(out.values()) - 3 == 4


def test_scale_down_is_stabilized_and_stepwise():
    ps = _ps(down_window_s=120, max_down_step=1, down_budget_per_decision=3)
    cur = {"a": 5, "b": 1, "c": 1}
    assert ps.targets(0, {"a": 5, "b": 1, "c": 1}, cur)["a"] == 5
    # the recommendation drops, but 5 is still in the window: no change
    assert ps.targets(30, {"a": 2, "b": 1, "c": 1}, cur)["a"] == 5
    assert ps.targets(90, {"a": 2, "b": 1, "c": 1}, cur)["a"] == 5
    # the 5 leaves the window at t > 120: one replica at a time
    out = ps.targets(130, {"a": 2, "b": 1, "c": 1}, cur)
    assert out["a"] == 4
    cur["a"] = 4
    assert ps.targets(160, {"a": 2, "b": 1, "c": 1}, cur)["a"] == 3


def test_down_budget_is_shared_by_the_clusters():
    ps = _ps(down_window_s=10, max_down_step=2, down_budget_per_decision=3)
    cur = {"a": 6, "b": 6, "c": 6}
    ps.targets(0, cur, cur)
    out = ps.targets(20, {"a": 1, "b": 1, "c": 1}, cur)
    assert sum(cur.values()) - sum(out.values()) == 3
    assert all(cur[c] - out[c] <= 2 for c in CL)


def test_no_scale_down_during_a_burst_hold():
    ps = _ps(down_window_s=10, panic_hold_s=60)
    cur = {c: 1 for c in CL}
    ps.tick(0, {c: 45 for c in CL}, {c: 10 for c in CL}, cur)   # burst until t=60
    cur = {"a": 6, "b": 1, "c": 1}
    out = ps.targets(30, {"a": 2, "b": 1, "c": 1}, cur)
    assert out["a"] == 6
    out = ps.targets(70, {"a": 2, "b": 1, "c": 1}, cur)         # hold over
    assert out["a"] == 5


# ── actuation lag ────────────────────────────────────────────────────────────

def test_actuation_lag_is_learned_and_sets_the_horizon():
    ps = _ps(t_ready_init_s=30, tick_s=10, t_ready_ewma=0.5)
    assert ps.horizon_s == 40
    ps.record_scale_up("a", 4, t=100)
    assert ps.observe_ready("a", 3, t=110) is None         # not all ready yet
    assert ps.observe_ready("a", 4, t=120) == pytest.approx(20)
    assert ps.t_ready_s == pytest.approx(20)               # first sample replaces the prior
    assert ps.horizon_s == pytest.approx(30)
    ps.record_scale_up("b", 2, t=200)
    ps.observe_ready("b", 2, t=240)                         # 40 s
    assert ps.t_ready_s == pytest.approx(0.5 * 40 + 0.5 * 20)
    assert ps.pending_clusters() == []


def test_pending_scale_up_that_never_gets_ready_is_dropped():
    ps = _ps(t_ready_max_s=60)
    ps.record_scale_up("a", 4, t=0)
    assert ps.observe_ready("a", 2, t=70) is None
    assert ps.pending_clusters() == []
    assert ps.t_ready_samples == 0


def test_fixed_horizon_disables_learning_effect_on_horizon():
    ps = _ps(horizon_s=45)
    ps.record_scale_up("a", 2, t=0)
    ps.observe_ready("a", 2, t=100)
    assert ps.horizon_s == 45


# ── wiring ───────────────────────────────────────────────────────────────────

def test_fast_window_query_and_validation():
    from src.metrics.prometheus_client import PrometheusClient as P
    q = P.ingress_queries(match="namespace", window="30s")[0]
    assert q.endswith("[30s]))")
    assert P.ingress_queries()[0].endswith("[1m]))")           # historical default
    for bad in ("1", "30", "1m)", "-5s", "1m;drop"):
        with pytest.raises(ValueError):
            P.ingress_queries(window=bad)


def _run_naming():
    spec = importlib.util.spec_from_file_location("run_naming_ps",
                                                  REPO / "experiments" / "run_naming.py")
    rn = importlib.util.module_from_spec(spec)
    sys.modules["run_naming_ps"] = rn
    spec.loader.exec_module(rn)
    return rn


def test_ps_arm_suffix(tmp_path):
    rn = _run_naming()
    env = tmp_path / "test-env.json"
    env.write_text(json.dumps({"clusters": [
        {"name": f"ms-cluster-{i:02d}", "region": r}
        for i, r in enumerate(["DE", "FR", "PL", "US-VA", "SG", "NO"], start=1)]}), encoding="utf-8")
    cell = rn.resolve("locality-ps", "con", env)
    assert cell.label == "dmos-locality-ps__con"
    assert "DMOS_SCALER=predictive" in cell.dmos_env and "DMOS_SCORE_MODEL=locality" in cell.dmos_env
    assert "DMOS_SCALER" not in rn.resolve("locality", "con", env).dmos_env
    both = rn.resolve("locality-ps-min0", "con", env).dmos_env
    assert "DMOS_SCALER=predictive" in both and "DMOS_ENTRY_MIN_REPLICAS=0" in both
    with pytest.raises(rn.NamingError):
        rn.resolve("off-ps", "con", env)
    # uniform placement with the pooled budget and the predictive scaler (ablation of WHERE)
    l2 = rn.resolve("l2-pool-ps", "con", env)
    assert "--no-scheduling" in l2.dmos_args
    assert "DMOS_SCALER=predictive" in l2.dmos_env and "DMOS_ENTRY_BUDGET=pooled" in l2.dmos_env


@pytest.fixture
def config_dir(tmp_path):
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    shutil.copy(REPO / "config" / "local.yaml.example", cfg / "local.yaml")
    return cfg


def test_provenance_records_the_predictive_scaler(config_dir, monkeypatch):
    from src.dmos_main import DMOSOrchestrator
    monkeypatch.delenv("DMOS_SCALER", raising=False)
    reactive = DMOSOrchestrator(config_path=str(config_dir), dry_run=True)
    assert reactive.ps is None
    assert "scaler" not in reactive.effective_config["modes"]
    assert "predictive_scaler" not in reactive.effective_config["level2"]
    monkeypatch.setenv("DMOS_SCALER", "predictive")
    monkeypatch.setenv("DMOS_PS_DOWN_WINDOW_S", "90")
    pred = DMOSOrchestrator(config_path=str(config_dir), dry_run=True)
    eff = pred.effective_config
    assert eff["modes"]["scaler"] == "predictive"
    assert eff["level2"]["predictive_scaler"]["params"]["down_window_s"] == 90.0
    assert "scaler=predictive" in eff["summary"]
    assert eff["config_id"] != reactive.effective_config["config_id"]
    assert pred.debounce_seconds == pred.ps.p.min_decision_spacing_s


def test_collector_records_the_predictive_scaler(monkeypatch):
    spec = importlib.util.spec_from_file_location("collect_dmos_live_ps",
                                                  REPO / "experiments" / "collect_dmos_live.py")
    col = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(col)
    text = "\n".join([
        '# HELP dmos_ps_horizon_seconds x',
        'dmos_ps_horizon_seconds 41.0',
        'dmos_ps_actuation_lag_seconds 31.0',
        'dmos_ps_burst 1.0',
        'dmos_ps_demand_rps{cluster="c1"} 123.456',
        'dmos_ps_need_replicas{cluster="c1"} 4.0',
        'dmos_current_replicas{cluster="c1",service="frontend"} 3.0',
    ])

    class R:
        def __init__(self, t):
            self.text = t
    monkeypatch.setattr(col.requests, "get", lambda *a, **k: R(text))
    d = col.scrape_dmos()
    assert d["ps"]["horizon_s"] == 41.0 and d["ps"]["actuation_lag_s"] == 31.0
    assert d["ps"]["burst"] == 1.0
    assert d["ps"]["demand_rps"] == {"c1": 123.46} and d["ps"]["need_replicas"] == {"c1": 4.0}
    assert d["current_replicas"]["c1|frontend"] == 3


def test_replay_predictive_beats_reactive_on_a_steep_flash_crowd():
    spec = importlib.util.spec_from_file_location("scaler_replay_t",
                                                  REPO / "experiments" / "scaler_replay.py")
    sr = importlib.util.module_from_spec(spec)
    sys.modules["scaler_replay_t"] = sr
    spec.loader.exec_module(sr)
    fleet = sr.Fleet(t_ready=30, noise_seed=1)
    dem = sr.demand_function("con", 50, 500, fleet.n)
    re_ = sr.run_reactive(fleet, dem).metrics(fleet)
    pr = sr.run_predictive(fleet, dem, PredictiveScalerParams()).metrics(fleet)
    assert pr["over_target_req"] < 0.5 * re_["over_target_req"]
    assert pr["replica_min"] < 1.25 * re_["replica_min"]      # bounded extra cost


def test_need_counts_the_overflow_of_a_full_cluster():
    # 'total' budget: the excess of a cluster above its cap goes to the others,
    # so a decision is due although the hot cluster is already at its cap
    ps = _ps()                                       # caps 6, capacity 10
    cur = {"a": 6, "b": 1, "c": 1}
    r = ps.tick(0, {"a": 100, "b": 5, "c": 5}, {"a": 100, "b": 5, "c": 5}, cur)
    assert r.clusters["a"].need == 6 and r.clusters["a"].raw_need == 10
    assert r.need_total == 12 > r.current_total == 8
    assert r.decide and r.reason in ("need_up", "burst")     # 12 >= 1.5 x 8: a burst
    per = PredictiveScaler(PredictiveScalerParams(), CL, capacity={c: 10.0 for c in CL},
                           min_replicas={c: 1 for c in CL}, max_replicas={c: 6 for c in CL},
                           budget_mode="per_cluster")
    assert per.tick(0, {"a": 100, "b": 5, "c": 5}, {"a": 100, "b": 5, "c": 5}, cur).need_total == 8
    with pytest.raises(ValueError):
        PredictiveScaler(PredictiveScalerParams(), CL, capacity={c: 10.0 for c in CL},
                         min_replicas={c: 1 for c in CL}, max_replicas={c: 6 for c in CL},
                         budget_mode="weird")


def test_level1_flipping_between_clusters_starts_no_pod():
    # Level 1 keeps moving one replica between a and b (close scores), the total
    # recommendation stays 8: a move must last move_confirm_s, so no pod is
    # started or stopped, and no replica is left above the budget
    ps = _ps(down_window_s=100, move_confirm_s=60)
    cur = {"a": 4, "b": 3, "c": 1}
    flips = [{"a": 3, "b": 4, "c": 1}, {"a": 4, "b": 3, "c": 1}]
    for k in range(12):                                   # 6 minutes, a flip every 30 s
        out = ps.targets(30.0 * k, flips[k % 2], cur)
        assert out == cur
    # the total recommendation really falls: replicas are removed
    for k in range(12, 20):
        cur = ps.targets(30.0 * k, {"a": 2, "b": 2, "c": 1}, cur)
    assert cur == {"a": 2, "b": 2, "c": 1}


def test_a_lasting_move_is_made_before_it_is_broken():
    ps = _ps(down_window_s=60, move_confirm_s=60)
    cur = {"a": 4, "b": 3, "c": 1}
    move = {"a": 3, "b": 4, "c": 1}                       # same total, one replica a -> b
    assert ps.targets(0, cur, cur) == cur
    assert ps.targets(30, move, cur) == cur               # not confirmed yet
    out = ps.targets(90, move, cur)                       # recommended for 60 s: add on b
    assert out == {"a": 4, "b": 4, "c": 1}
    ps.record_scale_up("b", 4, 90)
    cur = out
    assert ps.targets(110, move, cur) == cur              # b not ready: a keeps its replica
    assert ps.observe_ready("b", 4, 115) is not None
    out = ps.targets(120, move, cur)                      # b ready, a below 4 for 90 s >= 60
    assert out == move


def test_growth_is_applied_at_once_with_the_moves():
    ps = _ps(move_confirm_s=60)
    cur = {"a": 4, "b": 3, "c": 1}
    out = ps.targets(0, {"a": 3, "b": 5, "c": 2}, cur)    # total 8 -> 10
    assert out == {"a": 4, "b": 5, "c": 2}                # ups now, the down of a later


def test_a_cluster_recommended_zero_is_emptied():
    # min 0 (arm -min0): Level 1 gives the far cluster 0 and its replica to
    # another cluster; once the new one is ready the far one is removed first
    p = PredictiveScalerParams(down_window_s=60, move_confirm_s=30)
    ps = PredictiveScaler(p, CL, capacity={c: 10.0 for c in CL},
                          min_replicas={c: 0 for c in CL}, max_replicas={c: 6 for c in CL})
    cur = {"a": 4, "b": 3, "c": 1}
    rec = {"a": 4, "b": 4, "c": 0}
    out = ps.targets(0, rec, cur)
    assert out == cur                                     # a move: wait for confirmation
    out = ps.targets(30, rec, cur)
    assert out == {"a": 4, "b": 4, "c": 1}
    ps.record_scale_up("b", 4, 30)
    ps.observe_ready("b", 4, 50)
    cur = out
    assert ps.targets(70, rec, cur) == rec                # c emptied (below 1 for 70 s)


def test_scale_down_starts_from_the_largest_surplus():
    ps = _ps(down_window_s=10, max_down_step=1, down_budget_per_decision=1)
    cur = {"a": 2, "b": 4, "c": 1}
    ps.targets(0, cur, cur)
    out = ps.targets(20, {"a": 1, "b": 1, "c": 1}, cur)
    assert out == {"a": 2, "b": 3, "c": 1}


def test_fleet_guard_allows_removal_only_below_the_stabilized_total():
    ps = _ps(down_window_s=60)
    cur = {"a": 3, "b": 3, "c": 3}
    ps.targets(0, cur, cur)                               # total 9 recommended
    assert ps.targets(30, {"a": 1, "b": 1, "c": 1}, cur) == cur   # 9 still in the window
    out = ps.targets(70, {"a": 1, "b": 1, "c": 1}, cur)
    assert sum(out.values()) == 9 - 3                     # down budget 3, one per cluster


def test_budget_hysteresis_keeps_the_ramp_surplus_on_a_plateau():
    ps = _ps(down_window_s=60, down_tolerance_frac=0.1, falling_frac=0.9)
    plateau = {"a": 45, "b": 25, "c": 25}
    ps.tick(0, plateau, plateau, {c: 3 for c in CL})
    assert ps.hold_budget(1, 12) == 12                    # end of the ramp (forecast on)
    for k in range(1, 10):                                # 10 minutes of plateau sized 10
        ps.tick(60.0 * k, plateau, plateau, {c: 4 for c in CL})
        assert ps.hold_budget(60.0 * k + 1, 10) == 11     # drop of at most 10 % (1 replica)
    falling = {"a": 20, "b": 10, "c": 10}                 # below 90 % of the window max
    ps.tick(600, falling, falling, {c: 4 for c in CL})
    assert ps.hold_budget(601, 5) == 5                    # falling: no hold
    ps.tick(700, falling, falling, {c: 2 for c in CL})    # steady again, new reference
    assert ps.hold_budget(701, 5) == 5


def test_no_budget_hold_at_the_fleet_minimum():
    ps = _ps()
    steady = {c: 5 for c in CL}
    ps.tick(0, steady, steady, {c: 1 for c in CL})
    ps.hold_budget(1, 5)
    assert ps.hold_budget(2, 3) == 3                      # 3 = sum of the minimums


def test_pooled_budget_is_sized_on_the_total_demand():
    # end of the peak: the same total moves from a to b and c. Per-cluster
    # forecasts extrapolate the rise of b and c and not the fall of a; the
    # pooled scaler forecasts the total, which does not change
    def run(mode):
        ps = PredictiveScaler(PredictiveScalerParams(), CL, capacity={c: 10.0 for c in CL},
                              min_replicas={c: 1 for c in CL}, max_replicas={c: 6 for c in CL},
                              budget_mode=mode)
        cur = {c: 3 for c in CL}
        for k in range(12):                               # steady 80/10/10
            ps.tick(10.0 * k, {"a": 80, "b": 10, "c": 10}, {"a": 80, "b": 10, "c": 10}, cur)
        for k in range(12, 18):                           # a -> b, c in 60 s, total 100
            x = (k - 11) / 6
            f = {"a": 80 - 50 * x, "b": 10 + 25 * x, "c": 10 + 25 * x}
            r = ps.tick(10.0 * k, f, f, cur)
        return r, ps
    r, ps = run("pooled")
    assert abs(ps.total_sizing.forecast_rps - 100) < 5
    assert sum(ps.demand_for_budget().values()) == pytest.approx(ps.total_sizing.demand_rps)
    assert r.need_total <= 11
    # policy 3 summed the per-cluster demands: measurement or forecast + headroom
    parts = sum(max(z.fast_rps, z.stable_rps, z.forecast_rps + z.headroom_rps)
                for z in ps.sizing.values())
    assert parts > ps.total_sizing.demand_rps + 10        # more than one replica (r = 10) above


def test_back_to_the_minimum_when_the_recommendation_is_the_minimum():
    ps = _ps(down_window_s=30)
    cur = {"a": 2, "b": 1, "c": 1}
    steady = {c: 5 for c in CL}
    for k in range(6):
        ps.tick(30.0 * k, steady, steady, cur)
        cur = ps.targets(30.0 * k + 1, {"a": 1, "b": 1, "c": 1}, cur)
    assert cur == {"a": 1, "b": 1, "c": 1}              # back to the minimum
