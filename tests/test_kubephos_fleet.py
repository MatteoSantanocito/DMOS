"""Tests of scripts/kubephos_fleet.py (pure functions; no cluster needed)."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("kubephos_fleet", REPO / "scripts" / "kubephos_fleet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kf = _load()


def _node(name, pool, cpu, mem, ip, ready=True):
    return {"metadata": {"name": name, "labels": {"kubephos.dev/pool": pool}},
            "status": {"capacity": {"cpu": cpu, "memory": mem},
                       "addresses": [{"type": "InternalIP", "address": ip}],
                       "conditions": [{"type": "Ready", "status": "True" if ready else "False"}]}}


KUBEPHOS_NODES = {"items": [
    _node("c1-app-1", "app", "4", "8131772Ki", "10.0.0.5"),
    _node("c1-app-2", "app", "4", "8131768Ki", "10.0.0.6"),
    _node("c1-control-plane-1", "control-plane", "2", "4009844Ki", "10.0.0.3"),
    _node("c1-management-1", "management", "4", "8131772Ki", "10.0.0.4"),
]}


def test_kubeconfig_contexts_in_order():
    doc = {"clusters": [{"name": "a", "cluster": {"server": "https://10.0.0.3:6443"}},
                        {"name": "b", "cluster": {"server": "https://10.0.0.7:6443"}}],
           "contexts": [{"name": "b", "context": {"cluster": "b"}},
                        {"name": "a", "context": {"cluster": "a"}}]}
    assert kf.kubeconfig_contexts(doc) == [("b", "https://10.0.0.7:6443"), ("a", "https://10.0.0.3:6443")]
    with pytest.raises(kf.FleetError):
        kf.kubeconfig_contexts({"clusters": [], "contexts": []})


def test_node_summary_counts_the_app_pool_only():
    s = kf.node_summary(KUBEPHOS_NODES, "app")
    assert s["app_nodes"] == ["c1-app-1", "c1-app-2"]
    assert (s["cpu_cores"], s["memory_gb"]) == (8, 16)            # 2 x (4 vCPU, 8 GB VM)
    assert s["pool_of_ip"]["10.0.0.3"] == "control-plane" and s["not_ready"] == []
    bad = {"items": [_node("x", "app", "500m", "2Gi", "1.1.1.1", ready=False)]}
    s = kf.node_summary(bad, "app")
    assert s["not_ready"] == ["x"] and s["memory_gb"] == 2


def test_demand_match_follows_the_entry_node():
    pools = kf.node_summary(KUBEPHOS_NODES, "app")["pool_of_ip"]
    assert kf.demand_match_for("10.0.0.3", pools, "app") == "namespace"   # control plane
    assert kf.demand_match_for("10.0.0.5", pools, "app") == "workload"    # app node
    assert kf.demand_match_for("9.9.9.9", pools, "app") == "namespace"


def test_prometheus_info():
    prom = {"items": [{"metadata": {"namespace": "kubephos-observability"},
                       "spec": {"serviceMonitorSelector": {"matchLabels": {"release": "kubephos-observability"}}}}]}
    svcs = {"items": [
        {"metadata": {"namespace": "kubephos-observability", "name": "grafana"},
         "spec": {"type": "NodePort", "ports": [{"port": 80, "nodePort": 32000}]}},
        {"metadata": {"namespace": "kubephos-observability", "name": "prom"},
         "spec": {"type": "NodePort", "ports": [{"port": 9090, "nodePort": 32090}, {"port": 8080, "nodePort": 31028}]}},
    ]}
    assert kf.prometheus_info(prom, svcs) == {"namespace": "kubephos-observability",
                                              "release": "kubephos-observability", "port": 32090}
    with pytest.raises(kf.FleetError):
        kf.prometheus_info({"items": []}, svcs)


def test_nodeport_gateway_class():
    classes = {"items": [
        {"metadata": {"name": "cilium"}, "spec": {"controllerName": "io.cilium/gateway-controller"}},
        {"metadata": {"name": "kubephos-cilium"},
         "spec": {"controllerName": "io.cilium/gateway-controller",
                  "parametersRef": {"kind": "CiliumGatewayClassConfig", "name": "kubephos-nodeport",
                                    "namespace": "kube-system"}}}]}
    cfgs = {"items": [{"metadata": {"name": "kubephos-nodeport", "namespace": "kube-system"},
                       "spec": {"service": {"type": "NodePort"}}}]}
    assert kf.nodeport_gateway_class(classes, cfgs) == "kubephos-cilium"
    assert kf.nodeport_gateway_class(classes, {"items": []}) is None


def test_hubble_labels():
    assert kf.hubble_missing("dns drop tcp flow icmp httpV2") == list(kf.HUBBLE_LABELS)
    assert kf.hubble_missing(kf.HUBBLE_HTTP + " dns") == []
    assert kf.hubble_missing("httpV2:exemplars=true;labelsContext=source_namespace,"
                             "source_workload,destination_namespace,destination_workload,"
                             "traffic_direction dns") == []
    assert kf.hubble_missing("dns") == list(kf.HUBBLE_LABELS)
    assert kf.hubble_metrics_values(["dns", "drop", "httpV2", "icmp"]) == \
        [kf.HUBBLE_HTTP, "dns", "drop", "icmp"]
    assert kf.hubble_metrics_values([])[0] == kf.HUBBLE_HTTP


def test_region_plan():
    assert [p["region"] for p in kf.region_plan(3, None, {})] == ["DE", "FR", "PL"]
    with pytest.raises(kf.FleetError, match="--regions"):
        kf.region_plan(7, None, {})
    with pytest.raises(kf.FleetError, match="entries"):
        kf.region_plan(2, ["DE"], {})
    with pytest.raises(kf.FleetError, match="--baseline IT"):
        kf.region_plan(2, ["DE", "IT"], {})
    plan = kf.region_plan(2, ["DE", "IT"], {"IT": 300})
    assert plan[1] == {"region": "IT", "location": "IT", "baseline": 300, "price": 0.10}


def _fleet():
    def c(name, ip):
        return {"name": name, "ip": ip, "kubeconfig": f"C:/x/kubeconfig-{name}",
                "prometheus": {"port": 32090, "namespace": "obs", "release": "obs"},
                "ingress_demand_match": "namespace",
                "cpu_cores": 8, "memory_gb": 16, "cluster_id": name[-1], "not_ready": [],
                "app_nodes": ["n"], "gateway_class": "kubephos-cilium", "metrics_server": True,
                "cilium_chart": "cilium-1.20.3", "mesh_remote": 1, "mesh_ready": {"x": "1"},
                "hubble_missing": []}
    return {"name": "t", "app_pool": "app", "clusters": [c("t-1", "10.0.0.3"), c("t-2", "10.0.0.7")]}


def test_build_local_loads_in_dmos(tmp_path):
    from src.utils.config_loader import ConfigLoader
    fleet = _fleet()
    text, data = kf.build_local(fleet, kf.region_plan(2, ["DE", "SG"], {}),
                                {"t-1": 32052, "t-2": 31634}, {"user": "ubuntu", "key": "~/k"}, "test")
    assert data["clusters"][0]["ingress_port"] == 32052 and data["clusters"][1]["region"] == "SG"
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        (cfg / name).write_text((REPO / "config" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (cfg / "local.yaml").write_text(text, encoding="utf-8")
    loaded = ConfigLoader(str(cfg))
    c1 = loaded.clusters["t-1"]
    assert (c1.ip, c1.ingress_port, c1.ingress_demand_match, c1.cpu_cores) == ("10.0.0.3", 32052, "namespace", 8)


def test_fleet_problems():
    fleet = _fleet()
    assert kf.fleet_problems(fleet) == ([], [])
    fleet["clusters"][1]["cluster_id"] = "1"
    fleet["clusters"][1]["mesh_remote"] = 0
    fleet["clusters"][0]["hubble_missing"] = ["source_workload"]
    errs, warns = kf.fleet_problems(fleet)
    assert any("cluster-id" in e for e in errs) and any("ClusterMesh connected to 0" in e for e in errs)
    assert any("hubble-fix" in w for w in warns)


def test_compare_helpers():
    summ = {"results": {"per_phase": {"peak": {"p95": 812.5, "errorRate": 0.01, "rps": 250}}}}
    assert kf.phase_metrics(summ) == {"peak": {"p95": 812.5, "err": 0.01, "rps": 250}}
    lines = [json.dumps({"services": {"frontend": {"clusters": {"a": {"current_replicas": r}}}}})
             for r in (1, 3, 2)] + ["not json"]
    assert kf.max_replicas(lines) == {"a": 3}


# ── replica cap ──────────────────────────────────────────────────────────────

def test_replica_cap_reproduces_the_reference_high_tier():
    # 8 vCPU worker with Prometheus on it (C0 ~ 1 core), k = 14 mcore/req, r = 60:
    # floor((0.8*8 - 1) / 0.84) = 6, the cap measured on the reference testbed
    cp = kf.replica_cap(V=8, M=12, C0=1.0, M0=3.0, k_mcore=14, r=60, m_unit=0.91)
    assert (cp["cap_cpu"], cp["cap"]) == (6, 6)
    # Kubephos app pool 2 x 4 vCPU, nearly idle
    assert kf.replica_cap(8, 16, 0.1, 2.4, 14, 60, 0.91)["cap"] == 7
    # memory binds on a small pool, the cap never goes below 1
    assert kf.replica_cap(4, 4, 0.1, 2.5, 14, 60, 0.91)["cap"] == 1
    with pytest.raises(kf.FleetError):
        kf.replica_cap(8, 16, 0, 0, 0, 60, 1)


def test_memory_per_unit_uses_the_co_scaled_share():
    model = {"entry": "frontend", "co_scaled": ["reco", "cart"],
             "services": {"frontend": {"capacity": 60, "calls": 1, "deployment": "frontend"},
                          "reco": {"capacity": 40, "calls": 0.4, "deployment": "recommendationservice"},
                          "cart": {"capacity": 400, "calls": 0.65, "deployment": "cartservice"}}}
    limits = {"frontend": 0.125, "recommendationservice": 1.0, "cartservice": 0.5}
    # 0.125 + 0.4*60/40*1.0 + 0.65*60/400*0.5
    assert kf.memory_per_unit(model, limits) == pytest.approx(0.125 + 0.6 + 0.04875)


def test_services_model_reads_services_yaml():
    m = kf.services_model()
    assert m["entry"] == "frontend" and m["services"]["frontend"]["capacity"] == 60
    assert "recommendationservice" in m["co_scaled"]


def test_local_yaml_with_caps_overrides_services_yaml(tmp_path):
    from src.utils.config_loader import ConfigLoader
    fleet = _fleet()
    caps = {"t-1": dict(kf.replica_cap(8, 16, 0.1, 2.4, 14, 60, 0.91), service="frontend",
                        k_source="test"),
            "t-2": dict(kf.replica_cap(4, 8, 0.1, 2.4, 14, 60, 0.91), service="frontend",
                        k_source="test")}
    text, data = kf.build_local(fleet, kf.region_plan(2, ["DE", "SG"], {}),
                                {"t-1": 32052, "t-2": 31634}, {"user": "u", "key": "~/k"}, "t", caps)
    assert data["clusters"][0]["max_replicas"] == {"frontend": 7}
    assert "cap = min(floor((u*V - C0) / (r*k + c_rep))" in text
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        (cfg / name).write_text((REPO / "config" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (cfg / "local.yaml").write_text(text, encoding="utf-8")
    loaded = ConfigLoader(str(cfg))
    fe = loaded.get_service("frontend")
    assert fe.get_max_replicas_for_cluster("t-1") == 7
    assert fe.get_max_replicas_for_cluster("t-2") == caps["t-2"]["cap"]
    assert fe.get_max_replicas_for_cluster("ms-cluster-01") == 6        # services.yaml untouched
    assert loaded.get_service("cartservice").get_max_replicas_for_cluster("t-1") == 2


def test_bad_caps_in_local_yaml_are_rejected(tmp_path):
    from src.utils.config_loader import ConfigLoader
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        (cfg / name).write_text((REPO / "config" / name).read_text(encoding="utf-8"), encoding="utf-8")
    base = yaml.safe_load((REPO / "config" / "local.yaml.example").read_text(encoding="utf-8"))
    for bad, msg in (({"nosuch": 3}, "not a service"), ({"frontend": 0}, "integer >= 1"),
                     ({"frontend": True}, "integer >= 1")):
        base["clusters"][0]["max_replicas"] = bad
        (cfg / "local.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
        with pytest.raises(ValueError, match=msg):
            ConfigLoader(str(cfg))


def test_cluster_snapshot_records_the_caps():
    from types import SimpleNamespace
    from src.utils.provenance import cluster_snapshot
    c = SimpleNamespace(region="DE", cpu_cores=8, memory_gb=16, prometheus_port=32090,
                        carbon={}, max_replicas={})
    assert "max_replicas" not in cluster_snapshot(c)
    c.max_replicas = {"frontend": 7}
    assert cluster_snapshot(c)["max_replicas"] == {"frontend": 7}


# ── free resources on the workload nodes ─────────────────────────────────────

def test_node_selector_one_or_many_ips():
    from src.metrics.prometheus_client import PrometheusClient as P
    # PromQL string: the regex \. is written \\.
    assert P.node_instance_selector("10.0.0.5") == r'instance=~"10\\.0\\.0\\.5(:[0-9]+)?"'
    assert P.node_instance_selector(["10.0.0.5", "10.0.0.6"]) == \
        r'instance=~"(10\\.0\\.0\\.5|10\\.0\\.0\\.6)(:[0-9]+)?"'
    assert P.node_instance_selector(["10.0.0.5"]) == P.node_instance_selector("10.0.0.5")


class _FakeNodeProm:
    def __init__(self):
        self.sent = []

    def _scalar(self, q):
        self.sent.append(q)
        return 8.0 if q.startswith("count") else 2.0


def test_free_cpu_sums_the_workload_nodes():
    from src.metrics.prometheus_client import PrometheusClient as P
    fake = _FakeNodeProm()
    fake.node_instance_selector = P.node_instance_selector
    assert P.get_node_cpu_available(fake, ["10.0.0.5", "10.0.0.6"]) == 6.0
    assert all(r"10\\.0\\.0\\.6" in q for q in fake.sent)


def test_workload_nodes_in_local_yaml(tmp_path):
    from src.utils.config_loader import ConfigLoader
    fleet = _fleet()
    for c in fleet["clusters"]:
        c["pool_of_ip"] = {c["ip"]: "control-plane", "10.9.0.5": "app", "10.9.0.6": "app",
                           "10.9.0.4": "management"}
    text, data = kf.build_local(fleet, kf.region_plan(2, ["DE", "SG"], {}),
                                {"t-1": 1, "t-2": 2}, {"user": "u", "key": "~/k"}, "t")
    assert data["clusters"][0]["worker_ips"] == ["10.9.0.5", "10.9.0.6"]
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        (cfg / name).write_text((REPO / "config" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (cfg / "local.yaml").write_text(text, encoding="utf-8")
    loaded = ConfigLoader(str(cfg))
    assert loaded.clusters["t-1"].workload_node_ips == ["10.9.0.5", "10.9.0.6"]
    base = yaml.safe_load((REPO / "config" / "local.yaml.example").read_text(encoding="utf-8"))
    (cfg / "local.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    c0 = ConfigLoader(str(cfg)).clusters[base["clusters"][0]["name"]]
    assert c0.workload_node_ips == [c0.ip]                       # reference fleet unchanged
    base["clusters"][0]["worker_ips"] = "10.0.0.5"
    (cfg / "local.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ValueError, match="worker_ips"):
        ConfigLoader(str(cfg))


def test_run_window_from_the_k6_summary():
    summ = {"timestamp_iso": "2026-10-27T22:42:25.402Z",
            "config": {"phase_times_s": [180, 300, 600, 720, 900]}}
    t0, phases = kf.run_window(summ)
    assert phases == [180, 300, 600, 720, 900]
    assert t0 == pytest.approx(1793140945.402 - 900, abs=1)
    assert kf.run_window({"config": {}}) is None


def test_node_queries_for_compare():
    q = kf.throttle_query("300s")
    assert q.startswith("max(") and 'namespace="online-boutique"' in q and "[300s]" in q
    assert kf.max_replicas_query("frontend", "900s") == (
        'max_over_time(kube_deployment_status_replicas{namespace="online-boutique",'
        'deployment="frontend"}[900s])')


# ── calibration v2 ──────────────────────────────────────────────────────────

def test_calibration_result_separates_marginal_and_fixed_cost():
    # 3 clusters; idle 0.1 -> 0.4 core with 2 replicas: c_rep = 0.3
    idle1 = {"a": 0.1, "b": 0.1, "c": 0.1}
    idleF = {"a": 0.4, "b": 0.4, "c": 0.4}
    low = {x: {"cpu": 0.4 + 0.012 * 50, "served": 50, "entry_cpu": 0.16} for x in idle1}
    high = {x: {"cpu": 0.4 + 0.012 * 100, "served": 100, "entry_cpu": 0.32} for x in idle1}
    res = kf.calibration_result(idle1, idleF, 2, low, high, entry_cpu_limit=0.2, u_pod=0.7)
    assert res["k_mcore"] == pytest.approx(12.0)
    assert res["c_rep"] == pytest.approx(0.3)
    assert res["entry_mcore_per_req"] == pytest.approx(3.2)
    assert res["r_entry"] == pytest.approx(0.7 * 0.2 / 0.0032)          # ~44 req/s
    with pytest.raises(kf.FleetError):
        kf.calibration_result(idle1, idleF, 2, low, low, 0.2, 0.7)


def test_cap_with_fixed_cost_per_replica():
    # 8 vCPU pool, r = 44, k = 12, c_rep = 0.3: unit 0.83 core -> floor(6.3/0.83) = 7
    cp = kf.replica_cap(8, 16, 0.1, 2.4, 12, 44, 0.91, c_rep=0.3)
    assert cp["unit_cores"] == pytest.approx(0.828) and cp["cap"] == 7
    assert kf.replica_cap(8, 16, 0.1, 2.4, 12, 44, 0.91)["cap"] == 11     # without c_rep


def test_calibrated_capacity_reaches_dmos(tmp_path):
    from src.utils.config_loader import ConfigLoader
    fleet = _fleet()
    cp = dict(kf.replica_cap(8, 16, 0.1, 2.4, 12, 44, 0.91, c_rep=0.3), service="frontend",
              k_source="test", r_calibrated=True)
    text, data = kf.build_local(fleet, kf.region_plan(2, ["DE", "SG"], {}), {"t-1": 1, "t-2": 2},
                                {"user": "u", "key": "~/k"}, "t", {"t-1": cp})
    assert data["clusters"][0]["capacity_req_per_sec"] == {"frontend": 44}
    assert "capacity_req_per_sec" not in data["clusters"][1]
    cfg = tmp_path / "config"
    cfg.mkdir()
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        (cfg / name).write_text((REPO / "config" / name).read_text(encoding="utf-8"), encoding="utf-8")
    (cfg / "local.yaml").write_text(text, encoding="utf-8")
    fe = ConfigLoader(str(cfg)).get_service("frontend")
    assert fe.get_capacity_for_cluster("t-1") == 44.0
    assert fe.get_capacity_for_cluster("t-2") == 60.0                     # services.yaml
    assert fe.get_max_replicas_for_cluster("t-1") == 7


def test_hubble_fix_reads_the_live_cilium_config(tmp_path, monkeypatch, capsys):
    # fleet.json (last discover) says every cluster is fine, but the chart of
    # t-2 was re-applied afterwards and its cilium-config lost the labels
    # (30/10/2026): hubble-fix must look at the live configmap.
    import argparse
    fleet = _fleet()
    (tmp_path / "fleet.json").write_text(json.dumps(fleet), encoding="utf-8")
    live = {"t-1": kf.HUBBLE_HTTP + " dns", "t-2": "dns drop httpV2"}
    upgraded = []

    def fake_kjson(kc, *args):
        name = kc.rsplit("kubeconfig-", 1)[1]
        if "cilium-config" in args:
            return {"data": {"hubble-metrics": live[name]}}
        return {}

    def fake_run(cmd, timeout=120):
        if cmd[:2] == ["helm", "upgrade"]:
            kc = cmd[cmd.index("--kubeconfig") + 1]
            name = kc.rsplit("kubeconfig-", 1)[1]
            upgraded.append(name)
            live[name] = kf.HUBBLE_HTTP + " dns"       # the upgrade restores the labels
        if cmd[:2] == ["helm", "get"]:
            return json.dumps({"hubble": {"metrics": {"enabled": ["dns", "drop", "httpV2"]}}})
        return ""

    monkeypatch.setattr(kf, "kjson", fake_kjson)
    monkeypatch.setattr(kf, "run", fake_run)
    monkeypatch.setattr(kf, "kubectl", lambda *a, **k: "")
    assert kf.cmd_hubble_fix(argparse.Namespace(out_dir=str(tmp_path))) == 0
    assert upgraded == ["t-2"]
    saved = json.loads((tmp_path / "fleet.json").read_text(encoding="utf-8"))
    assert [c["hubble_missing"] for c in saved["clusters"]] == [[], []]
