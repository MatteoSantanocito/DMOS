"""
Tests for the two Level-1/budget changes made after the campaign of 29/10/2026
(frontend served by the pool of all clusters, Cilium affinity 'none'):

- the 'reach' demand signal: the gate uses ψ_i = Σ_j share_j·exp(−RTT(j,i)/τ)
  instead of the share entering cluster i (src/level1/score_model.py);
- the 'pooled' entry budget: N = ceil(total demand / capacity of a replica)
  instead of the sum of the per-cluster round-ups (DMOS_ENTRY_BUDGET=pooled).

The numbers of the regression tests are those of the 29/10 runs: 33 req/s per
frontend replica, caps DE 4 / FR 3 / PL 2 / US 4 / SG 2 / NO 3, 420 req/s at
the frontend in the peak of 'con' (85 % entering DE), ping RTTs of the fleet.
"""
import importlib.util
import math
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from src.dmos_main import DMOSOrchestrator
from src.level1.score_model import (ScoreModel, ScoreModelError, reach_shares,
                                    select_model)
from src.level1.winner_determination import ClusterBid, WinnerDetermination
from src.level2.predictive_scaler import PredictiveScaler, PredictiveScalerParams
from src.metrics.prometheus_client import PrometheusClient

REPO = Path(__file__).resolve().parent.parent
NAMES = ["de", "fr", "pl", "us", "sg", "no"]
CAPS = dict(zip(NAMES, (4, 3, 2, 4, 2, 3)))
# ping RTT (ms) measured on 29/10/2026 (env_state_pre.json), source -> destination
RTT = {
    "de": {"fr": 30.6, "pl": 49.8, "us": 97.2, "sg": 201.8, "no": 37.5},
    "fr": {"de": 29.5, "pl": 78.2, "us": 131.4, "sg": 227.4, "no": 72.2},
    "pl": {"de": 51.9, "fr": 80.8, "us": 150.2, "sg": 253.5, "no": 93.8},
    "us": {"de": 100.3, "fr": 129.6, "pl": 144.5, "sg": 291.7, "no": 134.2},
    "sg": {"de": 202.5, "fr": 229.5, "pl": 251.1, "us": 292.3, "no": 244.1},
    "no": {"de": 41.8, "fr": 67.4, "pl": 87.4, "us": 140.4, "sg": 247.6},
}
CON = dict(zip(NAMES, (0.85, 0.03, 0.03, 0.03, 0.03, 0.03)))


# ── ψ ────────────────────────────────────────────────────────────────────────

def test_reach_is_at_least_the_share_and_at_most_one():
    psi = reach_shares(CON, RTT, 50)
    for c in NAMES:
        assert CON[c] <= psi[c] <= 1.0


def test_reach_ranks_clusters_by_distance_from_the_hot_spot():
    psi = reach_shares(CON, RTT, 50)
    # DE is hot; NO and FR are close to it, US and SG far away
    order = sorted(NAMES, key=psi.get, reverse=True)
    assert order[0] == "de" and set(order[1:3]) == {"fr", "no"} and order[-1] == "sg"
    assert psi["fr"] > 10 * CON["fr"]         # FR serves DE's users at +30 ms
    assert psi["sg"] < 0.05


def test_reach_tends_to_the_share_when_clusters_are_far_apart():
    far = {a: {b: 10_000.0 for b in NAMES if b != a} for a in NAMES}
    psi = reach_shares(CON, far, 50)
    for c in NAMES:
        assert psi[c] == pytest.approx(CON[c], abs=1e-12)


def test_reach_uses_the_other_direction_and_skips_unknown_pairs():
    shares = {"a": 0.5, "b": 0.5, "c": 0.0}
    psi = reach_shares(shares, {"a": {"b": 50.0}}, 50)       # only a->b measured
    e = math.exp(-1)
    assert psi["a"] == pytest.approx(0.5 + 0.5 * e)          # b->a taken from a->b
    assert psi["b"] == pytest.approx(0.5 + 0.5 * e)
    assert psi["c"] == pytest.approx(0.0)                    # no RTT: unreachable
    with pytest.raises(ValueError):
        reach_shares(shares, {}, 0)


# ── score model ──────────────────────────────────────────────────────────────

def test_reach_model_in_weights_yaml():
    raw = yaml.safe_load((REPO / "config" / "weights.yaml").read_text(encoding="utf-8"))
    m = select_model(raw, "reach")
    assert m.demand_mode == "gate" and m.demand_signal == "reach" and m.reach_tau_ms == 50
    assert m.cold_variant().demand_signal == "reach"         # same signal at cold start
    d = m.to_dict()
    assert d["demand_signal"] == "reach" and d["reach_tau_ms"] == 50
    nca = select_model(raw, "nca")
    assert nca.demand_signal == "share" and nca.to_dict()["reach_tau_ms"] is None


@pytest.mark.parametrize("patch,msg", [
    ({"demand_signal": "nearby"}, "demand_signal"),
    ({"demand_signal": "reach", "reach_tau_ms": 0}, "reach_tau_ms"),
])
def test_invalid_reach_definitions(patch, msg):
    base = {"demand_mode": "gate", "weights": {"latency": 0.25, "capacity": 0.35,
                                               "net": 0.3, "carbon": 0.1}}
    with pytest.raises(ScoreModelError, match=msg):
        ScoreModel.from_dict("x", {**base, **patch})


def _gate_scores(signal):
    """Gate g=2 of the 'reach' model on the score components logged at the
    peak of dmos-locality-ps__con (29/10): (lat, cap, Φ_net, carbon)."""
    comp = {"de": (.84, .678, .603, .826), "fr": (.98, .557, .520, .979),
            "pl": (.976, .362, .472, .621), "us": (.98, .742, .375, .720),
            "sg": (.961, .329, .229, .610), "no": (.963, .514, .502, .978)}
    w = (.25, .35, .30, .10)
    return {c: (0.05 + 0.95 * signal[c]) ** 2 * sum(a * b for a, b in zip(w, comp[c]))
            for c in NAMES}


def test_reach_gate_keeps_the_far_clusters_at_the_minimum():
    # Regression of the 29/10 finding: with the budget of the pool (14) the
    # reach gate fills DE, FR, NO, PL and leaves SG (and nearly US) out.
    scores = _gate_scores(reach_shares(CON, RTT, 50))
    bids = [ClusterBid(cluster_name=c, score=scores[c], capacity=CAPS[c]) for c in NAMES]
    alloc, ok = WinnerDetermination().allocate(bids, 14)
    got = {a.cluster_name: a.replicas for a in alloc}
    assert ok
    assert got["de"] == 4 and got["fr"] == 3 and got["no"] == 3 and got["pl"] == 2
    assert got.get("sg", 0) == 0 and got.get("us", 0) <= 2


# ── scheduler wiring ─────────────────────────────────────────────────────────

@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    for k in ("DMOS_ENTRY_BUDGET", "DMOS_SCORE_MODEL", "DMOS_SCALER", "DMOS_ENTRY_MIN_REPLICAS",
              "DMOS_FREE_RESOURCES", "DMOS_CONFIG_DUMP"):
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    shutil.copy(REPO / "config" / "local.yaml.example", cfg / "local.yaml")
    return cfg


class FakePingProm:
    """Answers get_peer_rtts_ms from a {ip: ms} row."""

    def __init__(self, row):
        self.row = row

    def get_peer_rtts_ms(self, peer_ips):
        return {ip: v for ip, v in self.row.items() if ip in peer_ips}


def _fleet_rtt(sched):
    """Fake ping rows for the clusters of the example local.yaml, built from
    RTT (first cluster = de, second = fr, ...)."""
    names = list(sched.cluster_configs)
    alias = dict(zip(names, NAMES))
    ips = {n: sched.cluster_configs[n].ip for n in names}
    return {n: FakePingProm({ips[m]: RTT[alias[n]][alias[m]] for m in names if m != n})
            for n in names}, alias


def test_scheduler_gates_on_reach_with_the_reach_model(config_dir, monkeypatch):
    monkeypatch.setenv("DMOS_SCORE_MODEL", "reach")
    orch = DMOSOrchestrator(config_path=str(config_dir), dry_run=True)
    sched = orch.scheduler
    if len(sched.cluster_configs) != 6 or len({c.ip for c in sched.cluster_configs.values()}) != 6:
        pytest.skip("the example local.yaml does not describe 6 clusters with distinct IPs")
    sched.prom_map, alias = _fleet_rtt(sched)
    shares = {n: CON[alias[n]] for n in sched.cluster_configs}
    psi = sched._reach_demand_shares(shares)
    expected = reach_shares(CON, RTT, 50)
    for n in shares:
        assert psi[n] == pytest.approx(expected[alias[n]], rel=1e-6)
    assert sched.last_reach_shares == psi
    cfg = orch.effective_config
    assert cfg["level1"]["demand_signal"] == {"signal": "reach", "tau_ms": 50.0}
    assert "demand reach tau=50ms" in cfg["summary"]


def test_scheduler_without_rtt_falls_back_to_the_share(config_dir, monkeypatch):
    monkeypatch.setenv("DMOS_SCORE_MODEL", "reach")
    sched = DMOSOrchestrator(config_path=str(config_dir), dry_run=True).scheduler
    sched.prom_map = {n: FakePingProm({}) for n in sched.cluster_configs}
    shares = {n: 1.0 / len(sched.cluster_configs) for n in sched.cluster_configs}
    assert sched._reach_demand_shares(shares) == shares


def assert_valid_promql_strings(query):
    """Every double-quoted PromQL string uses only valid escapes (Go rules)."""
    import re
    for body in re.findall(r'"((?:[^"\\]|\\.)*)"', query):
        for esc in re.findall(r"\\(.)", body):
            assert esc in 'abfnrtv\\"\'0123456789xuU', f"invalid escape \\{esc} in {body!r}"


def test_peer_rtts_query_returns_one_value_per_target():
    class P(PrometheusClient):
        def __init__(self):
            self.cluster_name = "x"
            self.q = None

        def query(self, q):
            self.q = q
            return {"result": [{"metric": {"target": "10.0.0.2"}, "value": [0, "30.6"]},
                               {"metric": {"target": "10.0.0.3"}, "value": [0, "NaN"]},
                               {"metric": {}, "value": [0, "0.1"]}]}
    p = P()
    out = p.get_peer_rtts_ms(["10.0.0.2", "10.0.0.3"])
    assert out == {"10.0.0.2": pytest.approx(30.6)}
    assert "avg by (target)" in p.q
    # PromQL string: the backslash of the regex is doubled (a bare "\." is an
    # invalid escape: Prometheus rejected the query with HTTP 400 on 30/10/2026)
    assert 'target=~"10\\\\.0\\\\.0\\\\.2|10\\\\.0\\\\.0\\\\.3"' in p.q
    assert_valid_promql_strings(p.q)
    assert p.get_peer_rtts_ms([]) == {}


def test_reach_changes_config_id_share_models_do_not(config_dir, monkeypatch):
    ids = {}
    for name in ("nca", "reach"):
        monkeypatch.setenv("DMOS_SCORE_MODEL", name)
        orch = DMOSOrchestrator(config_path=str(config_dir), dry_run=True)
        ids[name] = orch.config_id
        if name == "nca":
            assert "demand_signal" not in orch.effective_config["level1"]
    assert ids["nca"] != ids["reach"]


# ── pooled budget ────────────────────────────────────────────────────────────

class Svc:
    def __init__(self, min_replicas=1, capacity=33.0):
        self.min_replicas, self.capacity = min_replicas, capacity

    def get_capacity_for_cluster(self, c):
        return self.capacity

    def get_min_replicas_for_cluster(self, c):
        return self.min_replicas

    def get_max_replicas_for_cluster(self, c):
        return CAPS[c]


PEAK_CON = {"de": 357.0, "fr": 12.6, "pl": 12.6, "us": 12.6, "sg": 12.6, "no": 12.6}


def test_pooled_budget_on_the_29_10_peak():
    total, _ = DMOSOrchestrator.entry_budget(PEAK_CON, NAMES, Svc(), "total")
    pooled, floors = DMOSOrchestrator.entry_budget(PEAK_CON, NAMES, Svc(), "pooled")
    assert total == 16              # ceil(357/33) = 11, + 1 for each of the other 5
    assert pooled == math.ceil(sum(PEAK_CON.values()) / 33) == 13
    assert floors["de"] == 4        # floors as in the other modes


def test_pooled_budget_bounds():
    big = {c: 10_000.0 for c in NAMES}
    assert DMOSOrchestrator.entry_budget(big, NAMES, Svc(), "pooled")[0] == sum(CAPS.values())
    zero = {c: 0.0 for c in NAMES}
    assert DMOSOrchestrator.entry_budget(zero, NAMES, Svc(), "pooled")[0] == 6     # Σ min
    assert DMOSOrchestrator.entry_budget(zero, NAMES, Svc(0), "pooled")[0] == 1    # never 0


def test_pooled_budget_mean_capacity_is_weighted_by_the_caps():
    class Mixed(Svc):
        def get_capacity_for_cluster(self, c):
            return 60.0 if c == "de" else 20.0
    # r̄ = (4*60 + 14*20) / 18 = 28.9 -> ceil(300 / 28.9) = 11
    demand = {c: (300.0 if c == "de" else 0.0) for c in NAMES}
    assert DMOSOrchestrator.entry_budget(demand, NAMES, Mixed(), "pooled")[0] == 11


def test_predictive_scaler_pooled_need_matches_the_orchestrator():
    ps = PredictiveScaler(PredictiveScalerParams(), NAMES, capacity={c: 33.0 for c in NAMES},
                          min_replicas={c: 1 for c in NAMES}, max_replicas=CAPS,
                          budget_mode="pooled")
    cur = {c: 1 for c in NAMES}
    for t in (0, 10, 20):
        r = ps.tick(t, PEAK_CON, PEAK_CON, cur)
    demand = ps.demand_for_budget()
    expected, _ = DMOSOrchestrator.entry_budget(demand, NAMES, Svc(), "pooled")
    assert r.need_total == expected
    assert r.need_total < 16        # the 'total' budget of the same demand


def test_pooled_switch_is_accepted_and_recorded(config_dir, monkeypatch):
    monkeypatch.setenv("DMOS_ENTRY_BUDGET", "pooled")
    pooled = DMOSOrchestrator(config_path=str(config_dir), dry_run=True)
    monkeypatch.setenv("DMOS_ENTRY_BUDGET", "total")
    total = DMOSOrchestrator(config_path=str(config_dir), dry_run=True)
    assert pooled.effective_config["modes"]["entry_budget"] == "pooled"
    assert "budget pooled" in pooled.effective_config["summary"]
    assert pooled.config_id != total.config_id


# ── arm names ────────────────────────────────────────────────────────────────

_spec = importlib.util.spec_from_file_location("run_naming", REPO / "experiments" / "run_naming.py")
rn = importlib.util.module_from_spec(_spec)
sys.modules["run_naming"] = rn
_spec.loader.exec_module(rn)


@pytest.fixture
def env_json(tmp_path):
    import json
    p = tmp_path / "test-env.json"
    p.write_text(json.dumps({"clusters": [
        {"name": f"ms-cluster-{i:02d}", "region": r}
        for i, r in enumerate(["DE", "FR", "PL", "US-VA", "SG", "NO"], start=1)]}), encoding="utf-8")
    return p


def test_pool_suffix(env_json):
    c = rn.resolve("reach-pool-ps", "con", env_json)
    assert c.label == "dmos-reach-pool-ps__con" and c.kind == "dmos"
    assert c.dmos_env.split() == ["DMOS_SCORE_MODEL=reach", "DMOS_ENTRY_BUDGET=pooled",
                                  "DMOS_SCALER=predictive"]
    assert rn.resolve("locality-pool", "bal", env_json).dmos_env.split() == [
        "DMOS_SCORE_MODEL=locality", "DMOS_ENTRY_BUDGET=pooled"]
    assert "DMOS_ENTRY_MIN_REPLICAS=0" in rn.resolve("reach-pool-min0", "con", env_json).dmos_env
    for bad in ("off-pool", "l1l2-pool", "hpa-none-pool"):
        with pytest.raises(rn.NamingError):
            rn.resolve(bad, "con", env_json)


def test_promql_validator_catches_the_bug_of_30_10():
    with pytest.raises(AssertionError):
        assert_valid_promql_strings('avg(x{target=~"10\\.0\\.0\\.2"})')
    assert_valid_promql_strings('avg(x{target=~"10\\\\.0\\\\.0\\\\.2"})')
    assert_valid_promql_strings(PrometheusClient.node_instance_selector(["10.0.0.7", "10.0.0.8"]))


def test_reach_startup_probe_counts_the_rtt_pairs(config_dir, monkeypatch):
    monkeypatch.setenv("DMOS_SCORE_MODEL", "reach")
    sched = DMOSOrchestrator(config_path=str(config_dir), dry_run=True).scheduler
    sched.prom_map = {n: FakePingProm({}) for n in sched.cluster_configs}
    assert sched._probe_reach_rtts() == 0              # logged as ERROR
    if len(sched.cluster_configs) == 6 and len({c.ip for c in sched.cluster_configs.values()}) == 6:
        sched.prom_map, _ = _fleet_rtt(sched)
        assert sched._probe_reach_rtts() == 30


def test_manifest_flags_a_reach_run_that_fell_back(tmp_path):
    spec = importlib.util.spec_from_file_location("write_manifest", REPO / "experiments" / "write_manifest.py")
    wm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wm)
    (tmp_path / "dmos.log").write_text(
        "2026-10-30 13:35:19 | WARNING  | src.level1.dmos_scheduler:_reach_demand_shares | "
        "reach demand signal: no ping_exporter RTT between the clusters - using the raw demand shares\n",
        encoding="utf-8")
    m = {"dmos": {"config_file": "dmos_config.json", "log_file": "dmos.log",
                  "config_ids_observed": {"x": 1}, "config_id": "x"}}
    codes = [c["code"] for c in wm.compute_checks(m, tmp_path)]
    assert "reach_fallback_raw_share" in codes
    (tmp_path / "dmos.log").write_text("all good\n", encoding="utf-8")
    assert "reach_fallback_raw_share" not in [c["code"] for c in wm.compute_checks(m, tmp_path)]


def test_uniform_split_is_n_over_k_within_the_caps():
    split = DMOSOrchestrator.uniform_split
    assert split(12, NAMES, Svc()) == {c: 2 for c in NAMES}
    # naive N/K would give pl 3 (cap 2): that replica goes to us instead of being cut
    assert split(16, NAMES, Svc()) == {"de": 3, "fr": 3, "no": 3, "pl": 2, "sg": 2, "us": 3}
    assert split(99, NAMES, Svc()) == CAPS
    assert split(3, NAMES, Svc()) == {c: 1 for c in NAMES}           # never below the minimum
    assert split(3, NAMES, Svc(0)) == {"de": 1, "fr": 1, "no": 1, "pl": 0, "sg": 0, "us": 0}
