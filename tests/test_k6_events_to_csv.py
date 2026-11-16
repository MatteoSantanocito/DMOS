"""experiments/k6_events_to_csv.py: k6 raw JSON -> compact csv.gz."""
import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("k6csv", REPO / "experiments" / "k6_events_to_csv.py")
k6csv = importlib.util.module_from_spec(_spec)
sys.modules["k6csv"] = k6csv
_spec.loader.exec_module(k6csv)


def _point(metric, value, t="2026-10-30T21:58:01.123456789+01:00", **tags):
    base = {"url": "http://192.168.0.120:31872/cart", "status": "200", "cluster": "ms-01-cluster-01",
            "phase": "peak", "endpoint": "/cart", "method": "POST"}
    base.update(tags)
    return json.dumps({"type": "Point", "metric": metric,
                       "data": {"time": t, "value": value, "tags": base}})


def test_time_parsing():
    assert k6csv.to_epoch_ms("2026-10-30T20:58:01.123456789Z") == k6csv.to_epoch_ms(
        "2026-10-30T21:58:01.123+01:00")
    assert k6csv.to_epoch_ms("2026-10-30T20:58:01Z") % 1000 == 0
    assert k6csv.to_epoch_ms("garbage") is None


def test_convert_keeps_the_request_metrics(tmp_path):
    src = tmp_path / "k6_events.json"
    src.write_text("\n".join([
        json.dumps({"type": "Metric", "metric": "http_req_duration", "data": {}}),
        _point("http_req_duration", 123.456),
        _point("http_req_connecting", 1003.2, status="0"),
        _point("vus", 40),
        _point("phase_peak_latency", 123.4),
        "not json",
    ]) + "\n", encoding="utf-8")
    assert k6csv.main([str(src)]) == 0
    with gzip.open(tmp_path / "k6_requests.csv.gz", "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["metric"] for r in rows] == ["http_req_duration", "http_req_connecting"]
    r = rows[0]
    assert r["value_ms"] == "123.46" and r["cluster"] == "ms-01-cluster-01" and r["path"] == "/cart"
    assert r["t_ms"] == str(k6csv.to_epoch_ms("2026-10-30T21:58:01.123+01:00"))
