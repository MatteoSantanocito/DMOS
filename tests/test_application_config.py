"""
Tests for the application model of services.yaml (`application`): entry
service, co-scaled services, namespaces and how the entering demand is
measured. The defaults must reproduce the historical behaviour exactly.
"""
import shutil
from pathlib import Path

import pytest
import yaml

from src.metrics.prometheus_client import PrometheusClient
from src.utils.config_loader import ConfigLoader

REPO = Path(__file__).resolve().parent.parent

ONLINE_BOUTIQUE_CO_SCALED = [
    "cartservice", "productcatalogservice", "checkoutservice", "recommendationservice",
    "currencyservice", "adservice", "shippingservice",
]
# The two demand queries DMOS has always sent (client side, then server side).
HISTORICAL_INGRESS_QUERIES = [
    'sum(rate(hubble_http_requests_total{destination_workload="frontend",'
    'destination_namespace="online-boutique",reporter="client",traffic_direction="egress"}[1m]))',
    'sum(rate(hubble_http_requests_total{destination_workload="frontend",'
    'destination_namespace="online-boutique",reporter="server",traffic_direction="ingress",'
    'source_workload=""}[1m]))',
]
ALGO_ENV = ("DMOS_SCORE_MODEL", "DMOS_SCORE_GATE", "DMOS_SCORE_GATE_GAMMA", "DMOS_SCORE_GATE_EPS",
            "DMOS_LOCALITY_PURE_NET", "DMOS_WEIGHTS_COHERENT", "DMOS_DEMAND_FLOOR",
            "DMOS_SCORE_PROFILE", "DMOS_CONFIG_DUMP")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ALGO_ENV:
        monkeypatch.delenv(k, raising=False)


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


def _services(config_dir):
    return yaml.safe_load((config_dir / "services.yaml").read_text(encoding="utf-8"))


def _write_services(config_dir, data):
    (config_dir / "services.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _orchestrator(config_dir, **kw):
    from src.dmos_main import DMOSOrchestrator
    return DMOSOrchestrator(config_path=str(config_dir), dry_run=True, **kw)


# ── ConfigLoader ─────────────────────────────────────────────────────────────

def test_repo_application_matches_historical_behaviour(config_dir):
    cfg = ConfigLoader(str(config_dir))
    app = cfg.application
    assert app.entry_service == "frontend"
    assert app.co_scaled_services == ONLINE_BOUTIQUE_CO_SCALED
    assert app.namespace == "online-boutique"
    assert app.demand.source_workloads == [] and app.demand.server_side_fallback is True
    assert cfg.ingress_query_kwargs() == {
        "namespace": "online-boutique", "entry_workload": "frontend",
        "source_workloads": [], "server_side_fallback": True}


def test_without_application_section_defaults_are_identical(config_dir):
    data = _services(config_dir)
    data.pop("application")
    _write_services(config_dir, data)
    cfg = ConfigLoader(str(config_dir))
    assert cfg.application.entry_service == "frontend"
    assert cfg.application.co_scaled_services == ONLINE_BOUTIQUE_CO_SCALED
    assert cfg.application.namespace is None
    assert {s.namespace for s in cfg.services.values()} == {"online-boutique"}
    assert cfg.ingress_query_kwargs()["source_workloads"] == []


def test_default_namespace_and_per_service_override(config_dir):
    data = _services(config_dir)
    data["application"]["namespace"] = "run-42"
    for name, svc in data["services"].items():
        svc.pop("namespace")
    data["services"]["adservice"]["namespace"] = "ads"
    _write_services(config_dir, data)
    cfg = ConfigLoader(str(config_dir))
    assert cfg.services["frontend"].namespace == "run-42"
    assert cfg.services["adservice"].namespace == "ads"
    assert cfg.ingress_query_kwargs()["namespace"] == "run-42"


def test_service_without_any_namespace_is_an_error(config_dir):
    data = _services(config_dir)
    data["application"].pop("namespace")
    data["services"]["cartservice"].pop("namespace")
    _write_services(config_dir, data)
    with pytest.raises(ValueError, match="cartservice.*no namespace"):
        ConfigLoader(str(config_dir))


@pytest.mark.parametrize("mutate,message", [
    (lambda a: a.update(typo=1), "unknown key"),
    (lambda a: a.update(entry_service="nope"), "entry_service 'nope'"),
    (lambda a: a.update(co_scaled_services=["frontend"]), "cannot be co-scaled"),
    (lambda a: a.update(co_scaled_services=["ghost"]), "'ghost' is not a service"),
    (lambda a: a.update(co_scaled_services=["adservice", "adservice"]), "duplicates"),
    (lambda a: a.update(co_scaled_services="adservice"), "must be a list"),
    (lambda a: a["demand"].update(source_workloads=["Node_Proxy"]), "workload names"),
    (lambda a: a["demand"].update(source_workloads=['x"}or vector(1)']), "workload names"),
    (lambda a: a["demand"].update(source_workloads=["node-proxy", "node-proxy"]), "duplicates"),
    (lambda a: a["demand"].update(server_side_fallback="yes"), "true or false"),
    (lambda a: a["demand"].update(sources=["x"]), "unknown key"),
    (lambda a: a.update(namespace="Online_Boutique"), "namespace"),
])
def test_invalid_application_is_rejected(config_dir, mutate, message):
    data = _services(config_dir)
    mutate(data["application"])
    _write_services(config_dir, data)
    with pytest.raises(ValueError, match=message):
        ConfigLoader(str(config_dir))


# ── Demand queries ───────────────────────────────────────────────────────────

def test_default_ingress_queries_are_the_historical_ones():
    assert PrometheusClient.ingress_queries() == HISTORICAL_INGRESS_QUERIES
    assert PrometheusClient.ingress_queries("online-boutique", "frontend", [], True) \
        == HISTORICAL_INGRESS_QUERIES


def test_ingress_queries_with_sources_and_without_fallback():
    client, server = PrometheusClient.ingress_queries("run-1", "s0", ["node-proxy"], True)
    assert 'destination_workload="s0",destination_namespace="run-1"' in client
    assert client.endswith('traffic_direction="egress",source_workload=~"node-proxy"}[1m]))')
    assert 'reporter="server"' in server and 'source_workload=~"node-proxy"' in server
    assert 'source_workload=""' not in server
    (only,) = PrometheusClient.ingress_queries("ns", "fe", ["a", "b-c"], False)
    assert 'source_workload=~"a|b-c"' in only and 'reporter="client"' in only
    with pytest.raises(ValueError):
        PrometheusClient.ingress_queries("ns", 'fe"', [], True)


class _FakeProm(PrometheusClient):
    def __init__(self, answers):
        super().__init__(url="http://unused", cluster_name="c1", test_connection=False)
        self.answers, self.sent = list(answers), []

    def query(self, q):
        self.sent.append(q)
        value = self.answers.pop(0)
        return None if value is None else {"result": [{"value": [0, str(value)]}]}


def test_get_ingress_rate_order_and_fallback():
    prom = _FakeProm([12.5])
    assert prom.get_ingress_rate() == 12.5 and prom.sent == HISTORICAL_INGRESS_QUERIES[:1]
    prom = _FakeProm([0, 7.0])
    assert prom.get_ingress_rate() == 7.0 and prom.sent == HISTORICAL_INGRESS_QUERIES
    prom = _FakeProm([0])
    assert prom.get_ingress_rate(server_side_fallback=False) == 0.0 and len(prom.sent) == 1


# ── Orchestrator ─────────────────────────────────────────────────────────────

def _capture_queries(monkeypatch):
    sent = []
    monkeypatch.setattr(PrometheusClient, "query", lambda self, q: sent.append(q) or None)
    return sent


def test_orchestrator_uses_the_configured_demand(config_dir, monkeypatch):
    base = _orchestrator(config_dir).config_id
    data = _services(config_dir)
    data["application"]["demand"]["source_workloads"] = ["node-proxy"]
    _write_services(config_dir, data)
    orch = _orchestrator(config_dir)
    app = orch.effective_config["application"]
    assert app["demand"]["source_workloads"] == ["node-proxy"]
    assert orch.config_id != base                       # a different measurement
    assert "demand from node-proxy" in orch.effective_config["summary"]
    sent = _capture_queries(monkeypatch)
    orch._get_per_cluster_traffic("frontend")
    orch.scheduler.collect_scores("frontend")
    demand = [q for q in sent if "hubble_http_requests_total" in q]
    assert demand and all('source_workload=~"node-proxy"' in q for q in demand)


def test_namespace_does_not_change_config_id(config_dir):
    base = _orchestrator(config_dir)
    data = _services(config_dir)
    data["application"]["namespace"] = "kubephos-run-7"
    for svc in data["services"].values():
        svc.pop("namespace")
    _write_services(config_dir, data)
    moved = _orchestrator(config_dir)
    assert moved.effective_config["application"]["namespaces"]["frontend"] == "kubephos-run-7"
    assert moved.config_id == base.config_id            # same algorithm, other namespace


def test_another_application_without_code_changes(config_dir, monkeypatch):
    """A MuBench-like application: other names, one namespace, explicit ingress."""
    svc = {"resources_per_replica": {"requests": {"cpu": "100m", "memory": "64Mi"}},
           "capacity_req_per_sec": 50, "sla": {}, "power_watts": 10,
           "autoscaling": {"min_replicas": 1, "max_replicas": 4}}
    data = _services(config_dir)
    data["application"] = {"namespace": "mubench", "entry_service": "s0",
                           "co_scaled_services": ["s2", "s1"],
                           "demand": {"source_workloads": ["gateway"]}}
    data["services"] = {name: dict(svc, deployment_name=name, calls_per_request=1.0)
                        for name in ("s0", "s1", "s2", "db")}
    _write_services(config_dir, data)
    orch = _orchestrator(config_dir)
    assert orch.monitored_services == ["s0"]
    assert orch.service_dependencies == {"s0": ["s2", "s1"]}   # order kept, db not co-scaled
    sent = _capture_queries(monkeypatch)
    orch._get_per_cluster_traffic("s0")
    assert sent[0] == ('sum(rate(hubble_http_requests_total{destination_workload="s0",'
                       'destination_namespace="mubench",reporter="client",'
                       'traffic_direction="egress",source_workload=~"gateway"}[1m]))')
