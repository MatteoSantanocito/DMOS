"""
Tests for the sustainability accounting of the journal extension:
- the shared carbon-replay anchor (DMOS_CARBON_T0_FILE) in CarbonClient;
- trace_value_at, the pure function the offline analysis shares with DMOS;
- the whole-application helpers of experiments/collect_dmos_live.py;
- experiments/paper_sustainability.run_metrics on a synthetic run.

Traces are stand-in files written in tmp_path: the real Electricity Maps
CSVs are not in the repository.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from src.metrics.carbon_client import CarbonClient

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"


def _load(name):
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ramp_trace(path: Path, start_ci=100.0, step=10.0, hours=48):
    """Hourly trace from 2025-06-01 00:00 UTC: CI = start + step * hour."""
    lines = ["Datetime (UTC),Carbon Intensity gCO2eq/kWh (Life cycle)"]
    for h in range(hours):
        lines.append(f"2025-06-{1 + h // 24:02d} {h % 24:02d}:00:00,{start_ci + step * h}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flat_trace(path: Path, ci: float, hours=48):
    _ramp_trace(path, start_ci=ci, step=0.0, hours=hours)


def _client(trace_dir: Path, speedup=1.0):
    return CarbonClient({"provider": "trace", "trace": {
        "data_dir": str(trace_dir), "window_start": "2025-06-01T00:00:00", "speedup": speedup}})


# ── carbon replay ───────────────────────────────────────────────────────────

def test_trace_value_at_interpolates_and_allows_negative_elapsed(tmp_path):
    _ramp_trace(tmp_path / "DE_2025_hourly.csv")
    c = _client(tmp_path, speedup=1.0)
    assert c.trace_value_at("DE", 0) == 100.0
    assert c.trace_value_at("DE", 3600) == 110.0
    assert c.trace_value_at("DE", 1800) == 105.0          # linear interpolation
    # before the anchor: wraps inside the trace, no error
    assert c.trace_value_at("DE", -3600) is not None
    assert c.trace_value_at("XX", 0) is None               # no trace for the zone


def test_shared_anchor_file_drives_the_replay(tmp_path, monkeypatch):
    _ramp_trace(tmp_path / "DE_2025_hourly.csv")
    anchor = tmp_path / "carbon_t0"
    anchor.write_text(f"{time.time() - 7200:.3f}\n", encoding="utf-8")   # load started 2 h ago
    monkeypatch.setenv("DMOS_CARBON_T0_FILE", str(anchor))
    c = _client(tmp_path, speedup=1.0)
    # 2 h after window_start: 120 (± the few ms between the two calls)
    assert c.get_carbon_intensity("DE") == pytest.approx(120.0, abs=0.1)
    assert c.get_carbon_intensity("DE") == pytest.approx(c.trace_value_at("DE", 7200), abs=0.1)


def test_missing_anchor_file_keeps_the_first_access_anchor_then_switches(tmp_path, monkeypatch):
    _ramp_trace(tmp_path / "DE_2025_hourly.csv")
    anchor = tmp_path / "carbon_t0"
    monkeypatch.setenv("DMOS_CARBON_T0_FILE", str(anchor))
    c = _client(tmp_path, speedup=1.0)
    assert c.get_carbon_intensity("DE") == pytest.approx(100.0, abs=0.1)   # provisional anchor
    anchor.write_text(f"{time.time() - 3600:.3f}", encoding="utf-8")       # load starts
    assert c.get_carbon_intensity("DE") == pytest.approx(110.0, abs=0.1)


def test_without_the_variable_behaviour_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("DMOS_CARBON_T0_FILE", raising=False)
    _ramp_trace(tmp_path / "DE_2025_hourly.csv")
    c = _client(tmp_path, speedup=1.0)
    assert c.get_carbon_intensity("DE") == pytest.approx(100.0, abs=0.1)
    assert c._trace_t0 is not None and c._trace_t0_shared is None


# ── collector helpers ───────────────────────────────────────────────────────

def test_pod_service_and_services_all():
    col = _load("collect_dmos_live")
    assert col.pod_service("frontend-7d9c8b6f4-x2k9p") == "frontend"
    assert col.pod_service("redis-cart-5b9c7d8f6c-abcde") == "redis-cart"
    assert col.pod_service("standalone") == "standalone"
    cpu = [({"pod": "frontend-7d9c8b6f4-x2k9p"}, 0.5), ({"pod": "frontend-7d9c8b6f4-q1w2e"}, 0.25),
           ({"pod": "currencyservice-6f7b8c9d5-zzzzz"}, 0.1)]
    ready = [({"pod": "frontend-7d9c8b6f4-x2k9p"}, 1), ({"pod": "frontend-7d9c8b6f4-q1w2e"}, 1),
             ({"pod": "currencyservice-6f7b8c9d5-zzzzz"}, 0)]
    out = col.services_all_from_series(cpu, ready)
    assert out["frontend"] == {"cpu_cores": 0.75, "ready_pods": 2}
    assert out["currencyservice"] == {"cpu_cores": 0.1, "ready_pods": 0}


def test_dmos_sampler_reports_rate_from_the_second_sample(monkeypatch):
    col = _load("collect_dmos_live")
    s = col.DmosProcessSampler()
    if s._psutil is None:
        pytest.skip("psutil not installed")
    me = s._psutil.Process()
    monkeypatch.setattr(s, "_procs", lambda: [me])      # stand-in for the DMOS process
    first = s.sample()
    assert first["cpu_cores"] is None and first["rss_mb"] > 0 and first["processes"] == 1
    sum(i * i for i in range(200_000))                     # burn a little CPU
    time.sleep(0.05)
    second = s.sample()
    assert second["cpu_cores"] is not None and second["cpu_cores"] >= 0
    monkeypatch.setattr(s, "_procs", lambda: [])
    assert s.sample() is None


# ── paper_sustainability.run_metrics ────────────────────────────────────────

def _snapshot(elapsed, epoch, a_rps, b_rps, a_cpu, b_cpu):
    def cl(rps, cpu):
        return {"frontend_net_bytes_s": rps, "app_cpu_cores": cpu, "app_ready_pods": 2,
                "app_memory_mb": 100.0, "node_cpu_busy_cores": cpu, "node_cpu_cores": 4,
                "services_all": {}}
    return {"timestamp": "x", "elapsed_s": elapsed, "epoch_s": epoch,
            "services": {}, "clusters": {"a": cl(a_rps, a_cpu), "b": cl(b_rps, b_cpu)},
            "dmos_process": {"cpu_cores": 0.05, "rss_mb": 120.0}}


def test_run_metrics_weights_carbon_by_served_traffic_and_cpu(tmp_path):
    sus = _load("paper_sustainability")
    traces = tmp_path / "traces"; traces.mkdir()
    _flat_trace(traces / "AA_2025_hourly.csv", 100.0)
    _flat_trace(traces / "BB_2025_hourly.csv", 500.0)
    replay = sus.CarbonReplay({"trace": {"window_start": "2025-06-01T00:00:00", "speedup": 96}},
                              trace_dir=traces)
    clusters = {
        "a": sus.ClusterInfo("a", "AA", cores=4, p_idle=10, p_max=50, baseline_ci=999),
        "b": sus.ClusterInfo("b", "BB", cores=4, p_idle=10, p_max=50, baseline_ci=999),
    }
    run = tmp_path / "20261101_120000_dmos-nca__bal"; run.mkdir()
    t0 = 1_800_000_000.0
    (run / "carbon_t0").write_text(f"{t0}", encoding="utf-8")
    # peak window = [300, 600): 3 cluster-a requests for each cluster-b one,
    # CPU 1 core on a and 1 core on b
    snaps = [_snapshot(e, t0 + e, 30.0, 10.0, 1.0, 1.0) for e in range(0, 900, 5)]
    (run / "dmos_live.jsonl").write_text("\n".join(json.dumps(s) for s in snaps), encoding="utf-8")
    k6 = {"results": {"per_phase": {"peak": {"requests": 12000, "errorRate": 0.0, "p95": 400}},
                      "global": {"total_requests": 30000, "http_failure_rate": 0.0, "p95_ms": 380}}}
    m = sus.run_metrics(run, clusters, replay, "peak", k6_summary=k6, dmos_config={})
    assert m["anchor"] == "carbon_t0" and m["accounting"] == "app"
    assert m["window_s"] == pytest.approx(300.0)
    assert m["ci_served_g_kwh"] == pytest.approx(0.75 * 100 + 0.25 * 500)     # 200
    assert m["ci_compute_g_kwh"] == pytest.approx(300.0)
    assert m["core_s"] == pytest.approx(600.0)                                # 2 cores x 300 s
    assert m["core_s_per_1k_ok"] == pytest.approx(50.0)
    # dynamic power 10 W per busy core: 300 s x 10 W = 0.8333 Wh per cluster
    assert m["energy_dyn_wh"] == pytest.approx(2 * 10 * 300 / 3600)
    assert m["co2_dyn_g"] == pytest.approx(10 * 300 / 3.6e6 * (100 + 500))
    assert m["p95_ms"] == 400 and m["ok_requests"] == 12000
    assert m["dmos_cpu_cores_mean"] == pytest.approx(0.05)
    assert "a=trace:AA_2025_hourly.csv" in m["ci_sources"]


def test_run_metrics_legacy_run_without_new_fields(tmp_path):
    sus = _load("paper_sustainability")
    replay = sus.CarbonReplay({"trace": {}}, trace_dir=tmp_path)     # no traces: baseline CI
    clusters = {"a": sus.ClusterInfo("a", "AA", cores=4, p_idle=10, p_max=50, baseline_ci=300)}
    run = tmp_path / "20260601_120000_dmos-l1l2__con"; run.mkdir()
    snaps = [{"timestamp": f"2026-06-01T12:{e // 60:02d}:{e % 60:02d}", "elapsed_s": e,
              "services": {"frontend": {"clusters": {"a": {"cpu_cores": 0.5, "ready_pods": 1}}}},
              "clusters": {"a": {"frontend_net_bytes_s": 10.0, "node_cpu_pct": 50.0}}}
             for e in range(0, 900, 5)]
    (run / "dmos_live.jsonl").write_text("\n".join(json.dumps(s) for s in snaps), encoding="utf-8")
    m = sus.run_metrics(run, clusters, replay, "run", k6_summary={}, dmos_config={})
    assert m["accounting"] == "partial" and m["anchor"] == "first_snapshot"
    assert m["ci_served_g_kwh"] == pytest.approx(300.0) and "a=baseline" in m["ci_sources"]
    assert m["core_s"] == pytest.approx(0.5 * 900)
    assert m["co2_node_g"] > 0
