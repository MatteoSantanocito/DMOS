"""
experiments/live_flow_map.py: attribution of the Hubble flows (entered in j ->
served in i), IP ownership, aggregation, DMOS log filter, web server (demo).
"""
import importlib.util
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("live_flow_map", REPO / "experiments" / "live_flow_map.py")
lfm = importlib.util.module_from_spec(_spec)
sys.modules["live_flow_map"] = lfm
_spec.loader.exec_module(lfm)

DE, FR = "ms-01-cluster-01", "ms-01-cluster-02"


@pytest.fixture
def ipmap():
    m = lfm.IpMap()
    m.add(DE, cidrs=["10.1.0.0/16"], ips=["192.168.0.120"])
    m.add(FR, cidrs=["10.2.0.0/16"], ips=["192.168.0.123"])
    return m


def _flow(server_ip, client_ip, l7_type="RESPONSE", workload="frontend", latency_ns=42_000_000,
          code=200, server_cluster=None):
    """A line of `hubble observe -o jsonpb` (proto field names)."""
    server = {"ID": 1, "identity": 1234, "namespace": "online-boutique",
              "labels": [f"k8s:app={workload}", "k8s:io.kubernetes.pod.namespace=online-boutique"],
              "pod_name": f"{workload}-7d9c6b5f4-abcde",
              "workloads": [{"name": workload, "kind": "Deployment"}]}
    if server_cluster:
        server["cluster_name"] = server_cluster
    client = {"identity": 8, "labels": ["reserved:ingress"]}
    src, dst = (server, client) if l7_type == "RESPONSE" else (client, server)
    s_ip, d_ip = (server_ip, client_ip) if l7_type == "RESPONSE" else (client_ip, server_ip)
    return {"flow": {"time": "2026-10-30T10:00:00Z", "verdict": "FORWARDED",
                     "IP": {"source": s_ip, "destination": d_ip, "ipVersion": "IPv4"},
                     "source": src, "destination": dst, "Type": "L7", "node_name": "n1",
                     "l7": {"type": l7_type, "latency_ns": str(latency_ns),
                            "http": {"code": code, "method": "GET", "url": "http://x/"}},
                     "traffic_direction": "INGRESS", "is_reply": l7_type == "RESPONSE"},
            "node_name": "n1", "time": "2026-10-30T10:00:00Z"}


def test_ipmap_nodes_and_pod_cidrs(ipmap):
    assert ipmap.lookup("10.1.3.4") == DE
    assert ipmap.lookup("10.2.0.9") == FR
    assert ipmap.lookup("192.168.0.123") == FR
    assert ipmap.lookup("8.8.8.8") is None and ipmap.lookup("not-an-ip") is None
    ipmap.add(FR, cidrs=["10.1.200.0/24"])          # most specific network wins
    assert ipmap.lookup("10.1.200.7") == FR


def test_cilium_addresses_from_ciliumnodes_and_nodes():
    items = [{"spec": {"ipam": {"podCIDRs": ["10.1.0.0/24"]},
                       "addresses": [{"type": "InternalIP", "ip": "192.168.0.120"},
                                     {"type": "CiliumInternalIP", "ip": "10.1.0.12"}]}},
             {"spec": {"podCIDRs": ["10.1.1.0/24"]},
              "status": {"addresses": [{"type": "InternalIP", "address": "192.168.0.121"},
                                       {"type": "Hostname", "address": "worker-1"}]}}]
    cidrs, ips = lfm.cilium_addresses(items)
    assert cidrs == ["10.1.0.0/24", "10.1.1.0/24"]
    assert ips == ["10.1.0.12", "192.168.0.120", "192.168.0.121"]


def test_remote_request_is_attributed_to_entry_and_serving_cluster(ipmap):
    # entered DE (gateway Envoy 10.1.0.12), served by a frontend pod in FR,
    # observed by the relay of FR
    rec = lfm.parse_flow(_flow("10.2.0.50", "10.1.0.12"), FR, ipmap)
    assert rec == {"entry": DE, "serving": FR, "lat_ms": 42.0, "code": 200, "local": False}


def test_same_response_seen_from_the_entry_cluster_is_not_counted_twice(ipmap):
    assert lfm.parse_flow(_flow("10.2.0.50", "10.1.0.12"), DE, ipmap) == {"skip": "client_side"}


def test_requests_backends_and_unknown_clients(ipmap):
    assert lfm.parse_flow(_flow("10.2.0.50", "10.1.0.12", l7_type="REQUEST"), FR, ipmap) is None
    assert lfm.parse_flow(_flow("10.2.0.50", "10.2.0.9", workload="cartservice"), FR, ipmap) is None
    assert lfm.parse_flow(_flow("10.2.0.50", "172.16.0.1"), FR, ipmap) == {"skip": "unknown_client"}
    assert lfm.parse_flow({"flow": {"Type": "L3_L4"}}, FR, ipmap) is None
    local = lfm.parse_flow(_flow("10.1.0.50", "10.1.0.12", code=503), DE, ipmap)
    assert local["local"] is True and local["code"] == 503


def test_endpoint_matching():
    assert lfm.endpoint_is({"pod_name": "frontend-6b7d8c9f4-x2k9p"}, "frontend")
    assert not lfm.endpoint_is({"pod_name": "frontend-proxy-1"}, "frontend")
    assert lfm.endpoint_is({"labels": ["k8s:app=frontend"]}, "frontend")
    assert not lfm.endpoint_is(None, "frontend")


def test_aggregator_matrix_quantiles_and_samples():
    agg = lfm.Aggregator([DE, FR], window_s=10, samples_per_s=5, seed=1)
    now = 1000.0
    for k in range(20):
        agg.add({"entry": DE, "serving": FR, "lat_ms": 100 + k, "code": 200}, now - 1 + k * 0.01)
    for k in range(10):
        agg.add({"entry": DE, "serving": DE, "lat_ms": 10, "code": 500}, now - 2)
    agg.add({"skip": "client_side"}, now)
    agg.add({"entry": FR, "serving": FR, "lat_ms": 5, "code": 200}, now - 30)   # out of the window
    snap = agg.snapshot(now=now)
    assert snap["matrix"][DE][FR] == 2.0 and snap["matrix"][DE][DE] == 1.0
    assert snap["matrix"][FR][FR] == 0.0
    assert snap["served"] == {DE: 1.0, FR: 2.0} and snap["entered"][DE] == 3.0
    assert snap["p95"][DE][FR] == pytest.approx(118.05, abs=0.1)
    assert snap["remote_share"] == pytest.approx(20 / 30, abs=1e-3)
    assert snap["errors_rps"] == {DE: 1.0}
    assert snap["skipped"] == {"client_side": 1}
    assert 5 <= len(snap["samples"]) <= 30
    later = agg.snapshot(now=now, since=snap["seq"])
    assert later["samples"] == []


def test_log_filter_and_cleanup(tmp_path):
    raw = [
        "\x1b[32m2026-10-29 12:03:46\x1b[0m | \x1b[1mINFO    \x1b[0m | __main__:schedule_service | "
        "\x1b[1mTotal replicas needed: 15 [budget pooled, predictive]\x1b[0m",
        "2026-10-29 12:03:46 | INFO     | src.level1.score_functions:compute_total_score | Score totale: 0.5",
        "2026-10-29 12:03:47 | WARNING  | __main__:x | Cluster esclusi: ['a']",
        "2026-10-29 12:03:48 | INFO     | __main__:x | Reach-weighted demand (τ=50 ms): a: d=0.8→ψ=0.9",
    ]
    log = tmp_path / "20261030_101500_dmos-reach-pool-ps__con" / "dmos.log"
    log.parent.mkdir()
    log.write_text("\n".join(raw[:2]) + "\n", encoding="utf-8")
    tail = lfm.LogTail(str(log))
    tail.poll()
    out = tail.since(0)
    assert out["run"] == "dmos-reach-pool-ps__con" and out["started"] is not None
    texts = [x["text"] for x in out["lines"]]
    assert texts[1] == "12:03:46 INFO Total replicas needed: 15 [budget pooled, predictive]"
    assert not any("Score totale" in t for t in texts)
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n".join(raw[2:]) + "\n")
    tail.poll()
    new = tail.since(out["seq"])["lines"]
    assert [x["kind"] for x in new] == ["warn", "info"]


def test_scaling_actions_from_the_real_log_format():
    last = {}
    rows = ["2026-10-29 11:42:59 | INFO     | src.k8s.client:scale_deployment | "
            "\\u2705 Scaled ms-01-cluster-01/frontend \\u2192 3 replicas",
            "2026-10-29 11:50:00 | INFO     | src.k8s.client:scale_deployment | "
            "\\u2705 Scaled ms-01-cluster-01/frontend \\u2192 2 replicas",
            "2026-10-29 11:39:22 | WARNING  | src.level2.predictor:x | Insufficient history for derivative",
            "2026-10-29 11:39:22 | WARNING  | src.metrics.prometheus_client:get_ingress_rate | "
            "Ingress rate (a): 0.3 req/s [FALLBACK server-side: loop-prone]"]
    got = [lfm.classify_log_line(lfm.clean_log_line(r), last) for r in rows]
    assert got[0] == ("up", "11:42:59 ms-01-cluster-01/frontend ? → 3")
    assert got[1] == ("down", "11:50:00 ms-01-cluster-01/frontend 3 → 2")
    assert got[2] is None and got[3] is None
    assert lfm.clean_log_line("x \\u2705 y") == "x ✅ y"


def test_parse_prom_text():
    text = ('# HELP dmos_target_replicas x\n'
            'dmos_target_replicas{cluster="a",service="frontend"} 4.0\n'
            'dmos_target_replicas{cluster="a",service="cartservice"} 2.0\n'
            'dmos_target_replicas_other 1\n')
    got = lfm.parse_prom_text(text, "dmos_target_replicas")
    assert got[(("cluster", "a"), ("service", "frontend"))] == 4.0 and len(got) == 2


def test_demo_server_serves_page_and_state():
    from http.server import ThreadingHTTPServer
    fleet = [{"name": f"c{i}", "region": r, "ip": "", "kubeconfig": "", "prometheus_url": ""}
             for i, r in enumerate(["DE", "FR", "PL", "US-VA", "SG", "NO"], start=1)]
    agg = lfm.Aggregator([c["name"] for c in fleet], window_s=5)
    metrics = lfm.Metrics(fleet, "online-boutique", "frontend", "http://127.0.0.1:1/metrics")
    log = lfm.LogTail("/nonexistent/dmos.log")
    lfm.Demo(fleet, agg, metrics, log).start()
    app = lfm.App(fleet, agg, metrics, log, [], {}, True)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), lfm.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_port}"
        time.sleep(1.0)
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "DMOS · flussi live" in page
        st = json.loads(urllib.request.urlopen(base + "/state?s=0&l=0").read())
        assert st["demo"] is True and len(st["fleet"]) == 6
        assert st["flows"]["total_rps"] > 0 and st["flows"]["samples"]
        assert set(st["metrics"]["clusters"]) == {c["name"] for c in fleet}
    finally:
        srv.shutdown()


def _run_main(args):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("rc", lfm.main(args)), daemon=True)
    th.start()
    return th, out


def test_quit_and_duration_stop_the_map():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    th, out = _run_main(["--demo", "--env-json", "/nonexistent", "--port", str(port)])
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/state", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    assert urllib.request.urlopen(f"http://127.0.0.1:{port}/quit").read() == b"bye\n"
    th.join(timeout=10)
    assert not th.is_alive() and out["rc"] == 0

    th, out = _run_main(["--demo", "--env-json", "/nonexistent", "--port", str(port),
                         "--duration", "1"])
    th.join(timeout=10)
    assert not th.is_alive() and out["rc"] == 0


def test_stop_streams_terminates_the_children():
    import subprocess

    class FakeStream(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.stop = threading.Event()
            self.procs = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])]

        def run(self):
            self.stop.wait()

    streams = [FakeStream(), FakeStream()]
    for s in streams:
        s.start()
    lfm.stop_streams(streams, timeout=5)
    assert all(p.poll() is not None for s in streams for p in s.procs)
    assert all(s.stop.is_set() and not s.is_alive() for s in streams)
