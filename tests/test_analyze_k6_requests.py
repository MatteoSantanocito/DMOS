"""experiments/analyze_k6_requests.py: bursts of slow requests vs DMOS scaling actions."""
import csv
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ak6", REPO / "experiments" / "analyze_k6_requests.py")
ak6 = importlib.util.module_from_spec(_spec)
sys.modules["ak6"] = ak6
_spec.loader.exec_module(ak6)

T0 = 1_793_394_000.0          # load start (epoch s)


def _write_run(tmp_path):
    rows = [["t_ms", "metric", "value_ms", "status", "cluster", "phase", "endpoint", "method", "path", "shift"]]
    for i in range(3000):                       # 10 req/s for 300 s, peak phase
        t = T0 + 300 + i / 10
        slow = 400 <= t - T0 < 405              # a burst 100 s into the peak
        dur = 950.0 if slow else 80.0
        for m, v in (("http_req_duration", dur), ("http_req_blocked", 0.0),
                     ("http_req_connecting", 0.0), ("http_req_waiting", dur - 1)):
            rows.append([int(t * 1000), m, v, 200, "c1", "peak", "/", "GET", "/", ""])
    with gzip.open(tmp_path / "k6_requests.csv.gz", "wt", newline="") as f:
        csv.writer(f).writerows(rows)
    (tmp_path / "carbon_t0").write_text(f"{T0:.3f}")
    # dmos.log in local wall time of a machine 1 h ahead of UTC
    off = 3600
    def local(epoch):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(epoch + off))
    (tmp_path / "dmos.log").write_text(
        f"{local(T0 + 398)} | INFO     | src.k8s.client:scale_deployment | "
        f"\\u2705 Scaled ms-01-cluster-02/frontend \\u2192 3 replicas\n", encoding="utf-8")
    (tmp_path / "dmos_live.jsonl").write_text(json.dumps(
        {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(T0 + off)) + ".000000",
         "epoch_s": T0}) + "\n", encoding="utf-8")
    return tmp_path


def test_burst_found_and_lined_up_with_the_action(tmp_path, capsys):
    run = _write_run(tmp_path)
    assert ak6.main([str(run / "k6_requests.csv.gz"), "--run-dir", str(run), "--burst", "20"]) == 0
    out = capsys.readouterr().out
    assert "50 slower than 800 ms" in out
    assert "bursts (>= 20 slow in 10 s): 1" in out and "100% of the slow requests; at [400]" in out
    assert "waiting 949 ms" in out
    assert "t= 398.0s ms-01-cluster-02/frontend -> 3" in out
    assert "bursts with a scaling action within 15 s before: 1 of 1" in out
    assert "failed requests within 15 s after a scaling action: 0 of 0" in out
