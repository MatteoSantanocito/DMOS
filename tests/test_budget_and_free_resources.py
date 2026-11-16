"""
Tests for the two fixes made after the NCA 2026 campaign:
- the entry service's budget N capped on the total (DMOS_ENTRY_BUDGET=total)
  instead of per cluster (per_cluster, the NCA behaviour);
- the free resources of Φ_cap / L0 measured on the worker node
  (DMOS_FREE_RESOURCES=worker) instead of cluster-wide (cluster, NCA).
"""
import shutil
from pathlib import Path

import pytest

from src.dmos_main import DMOSOrchestrator
from src.metrics.prometheus_client import PrometheusClient

REPO = Path(__file__).resolve().parent.parent
CLUSTERS = [f"ms-cluster-0{i}" for i in range(1, 7)]
ENV = ("DMOS_ENTRY_BUDGET", "DMOS_FREE_RESOURCES", "DMOS_ENTRY_MIN_REPLICAS",
       "DMOS_SCORE_MODEL", "DMOS_CONFIG_DUMP")


class Svc:
    """Entry service like services.yaml: 60 req/s per replica, min 1,
    max DE 6 / FR 5 / PL 3 / US 6 / SG 3 / NO 5."""
    MAX = dict(zip(CLUSTERS, (6, 5, 3, 6, 3, 5)))

    def __init__(self, min_replicas=1):
        self.min_replicas = min_replicas

    def get_capacity_for_cluster(self, c):
        return 60.0

    def get_min_replicas_for_cluster(self, c):
        return self.min_replicas

    def get_max_replicas_for_cluster(self, c):
        return self.MAX[c]


def _traffic(hot, cold):
    return {c: (hot if c == "ms-cluster-01" else cold) for c in CLUSTERS}


def test_budget_per_cluster_loses_the_excess_of_a_full_cluster():
    traffic = _traffic(625.0, 22.0)               # 'con' at 700 req/s
    n_old, floors = DMOSOrchestrator.entry_budget(traffic, CLUSTERS, Svc(), "per_cluster")
    n_new, floors_new = DMOSOrchestrator.entry_budget(traffic, CLUSTERS, Svc(), "total")
    assert n_old == 6 + 5 * 1                      # DE needs 11 but counts 6
    assert n_new == 11 + 5 * 1                     # the excess stays in the budget
    assert floors["ms-cluster-01"] == floors_new["ms-cluster-01"] == 6   # floors respect the max


def test_budget_total_is_capped_by_the_sum_of_the_maxima():
    n, _ = DMOSOrchestrator.entry_budget(_traffic(5000.0, 5000.0), CLUSTERS, Svc(), "total")
    assert n == sum(Svc.MAX.values())


def test_budgets_agree_when_no_cluster_is_full():
    traffic = _traffic(297.5, 10.5)               # 'con' at 350 req/s
    a, _ = DMOSOrchestrator.entry_budget(traffic, CLUSTERS, Svc(), "per_cluster")
    b, _ = DMOSOrchestrator.entry_budget(traffic, CLUSTERS, Svc(), "total")
    assert a == b == 10


def test_budget_keeps_one_replica_with_minimum_zero_and_no_demand():
    n, floors = DMOSOrchestrator.entry_budget(_traffic(0.0, 0.0), CLUSTERS, Svc(0), "total")
    assert n == 1 and set(floors.values()) == {0}


# ── provenance and switches ─────────────────────────────────────────────────

@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(REPO / "config" / name, cfg / name)
    local = REPO / "config" / "local.yaml"
    shutil.copy(local if local.is_file() else REPO / "config" / "local.yaml.example",
                cfg / "local.yaml")
    return cfg


def _build(cfg):
    return DMOSOrchestrator(config_path=str(cfg), dry_run=True)


def test_defaults_are_the_fixed_behaviours(config_dir):
    orch = _build(config_dir)
    modes = orch.effective_config["modes"]
    assert modes["entry_budget"] == "total" and modes["free_resources"] == "worker"
    assert "(NCA)" not in orch.effective_config["summary"]


def test_nca_behaviours_are_recorded_and_change_config_id(config_dir, monkeypatch):
    new = _build(config_dir)
    monkeypatch.setenv("DMOS_ENTRY_BUDGET", "per_cluster")
    monkeypatch.setenv("DMOS_FREE_RESOURCES", "cluster")
    old = _build(config_dir)
    assert old.effective_config["modes"]["entry_budget"] == "per_cluster"
    assert old.effective_config["modes"]["free_resources"] == "cluster"
    assert "budget per-cluster (NCA)" in old.effective_config["summary"]
    assert "free resources cluster-wide (NCA)" in old.effective_config["summary"]
    assert old.config_id != new.config_id


@pytest.mark.parametrize("var,bad", [("DMOS_ENTRY_BUDGET", "cluster"),
                                     ("DMOS_FREE_RESOURCES", "node")])
def test_bad_switch_values_are_refused(config_dir, monkeypatch, var, bad):
    monkeypatch.setenv(var, bad)
    with pytest.raises(ValueError):
        _build(config_dir)


# ── worker-node measurement ─────────────────────────────────────────────────

class FakeProm(PrometheusClient):
    """PrometheusClient answering from a dict {query substring: value}."""

    def __init__(self, answers):
        self.answers = answers
        self.cluster_name = "fake"
        self.queries = []

    def query(self, q):
        self.queries.append(q)
        for key, value in self.answers.items():
            if key in q:
                return {"result": [{"value": [0, str(value)]}]}
        return {"result": []}


def test_node_queries_select_the_worker_ip():
    sel = PrometheusClient.node_instance_selector("10.0.0.7")
    assert sel == 'instance=~"10\\\\.0\\\\.0\\\\.7(:[0-9]+)?"'
    p = FakeProm({'count(node_cpu_seconds_total{mode="idle"': 4,
                  'sum(rate(node_cpu_seconds_total{mode!="idle"': 1.5,
                  "node_memory_MemAvailable_bytes": 6 * 1024**3})
    assert p.get_node_cpu_available("10.0.0.7") == pytest.approx(2.5)
    assert p.get_node_memory_available_gb("10.0.0.7") == pytest.approx(6.0)
    assert all(sel in q for q in p.queries)


def test_node_queries_return_none_without_series():
    p = FakeProm({})
    assert p.get_node_cpu_available("10.0.0.7") is None
    assert p.get_node_memory_available_gb("10.0.0.7") is None


def test_scheduler_falls_back_to_cluster_wide_without_node_series(config_dir):
    orch = _build(config_dir)
    sched = orch.scheduler
    name = next(iter(sched.cluster_configs))
    # a Prometheus without node_exporter series but with the cluster-wide ones
    fake = FakeProm({'kube_node_status_capacity{resource="cpu"}': 6,
                     "container_cpu_usage_seconds_total": 1,
                     'kube_node_status_capacity{resource="memory"}': 12 * 1024**3,
                     "container_memory_working_set_bytes": 2 * 1024**3,
                     "ping_rtt": 10})
    fake.get_network_rtt_ms = lambda peers: 10.0
    sched.prom_map[name] = fake
    m = sched._collect_cluster_metrics(name, cold_start_mode=True)
    assert m is not None
    assert m.cpu_available_cores == pytest.approx(5.0)       # 6 - 1, cluster-wide fallback
    assert m.memory_available_gb == pytest.approx(10.0)
    assert name in sched._free_fallback_warned
