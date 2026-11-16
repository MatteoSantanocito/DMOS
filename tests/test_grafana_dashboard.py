"""
scripts/grafana_dashboard.py: one Grafana dashboard for the whole fleet.
"""
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from src.metrics.prometheus_client import PrometheusClient

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("grafana_dashboard",
                                               REPO / "scripts" / "grafana_dashboard.py")
gd = importlib.util.module_from_spec(_spec)
sys.modules["grafana_dashboard"] = gd
_spec.loader.exec_module(gd)

REGIONS = ["DE", "FR", "PL", "US-VA", "SG", "NO"]
IPS = [f"192.168.0.{x}" for x in (120, 123, 126, 129, 132, 135)]


@pytest.fixture
def env_json(tmp_path):
    p = tmp_path / "test-env.json"
    p.write_text(json.dumps({"clusters": [
        {"name": f"ms-01-cluster-0{i}", "region": r, "ip": ip, "prometheus_url": f"http://{ip}:32090"}
        for i, (r, ip) in enumerate(zip(REGIONS, IPS), start=1)]}), encoding="utf-8")
    return p


def _targets(dash):
    for p in dash["panels"]:
        for t in p.get("targets", []):
            yield p, t


def test_render_layout_and_inputs(env_json, tmp_path):
    out = tmp_path / "d.json"
    assert gd.main(["render", "--env-json", str(env_json), "--out", str(out)]) == 0
    dash = json.loads(out.read_text(encoding="utf-8"))
    assert [i["name"] for i in dash["__inputs"]] == [f"DS_C{i}" for i in range(1, 7)]
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids))
    cells = set()
    for p in dash["panels"]:
        g = p["gridPos"]
        assert g["x"] + g["w"] <= 24
        for x in range(g["x"], g["x"] + g["w"]):
            for y in range(g["y"], g["y"] + g["h"]):
                assert (x, y) not in cells, p["title"]
                cells.add((x, y))
    # every Prometheus target points at one of the six inputs
    for p, t in _targets(dash):
        uid = t["datasource"]["uid"]
        assert uid == "__expr__" or uid in {f"${{DS_C{i}}}" for i in range(1, 7)}, p["title"]


def test_demand_panel_uses_the_query_of_dmos(env_json):
    dash = gd.build_dashboard(gd.load_fleet(env_json))
    panel = next(p for p in dash["panels"] if p["title"] == "Domanda in ingresso per cluster")
    dmos = PrometheusClient.ingress_queries("online-boutique", "frontend", match="namespace")[0]
    assert {t["expr"] for t in panel["targets"]} == {dmos}
    assert [t["legendFormat"] for t in panel["targets"]] == [f"{r} (0{i})" for i, r in
                                                            enumerate(REGIONS, start=1)]


def test_share_panels_combine_the_clusters_with_expressions(env_json):
    dash = gd.build_dashboard(gd.load_fleet(env_json))
    panel = next(p for p in dash["panels"] if p["title"] == "Quota servita (%)")
    queries = [t for t in panel["targets"] if t["datasource"]["uid"] != "__expr__"]
    exprs = [t for t in panel["targets"] if t["datasource"]["uid"] == "__expr__"]
    assert len(queries) == 6 and all(t["hide"] for t in queries)
    assert exprs[0]["refId"] == "tot" and exprs[0]["expression"] == "+".join(f"$q{i}" for i in range(1, 7))
    assert exprs[1]["expression"] == "$q1 * ($tot > 5) / ($tot * ($tot > 5)) * 100"
    assert panel["fieldConfig"]["defaults"]["custom"]["spanNulls"] is False
    reps = next(p for p in dash["panels"] if p["title"] == "Quota di repliche frontend (%)")
    assert [t for t in reps["targets"] if t["refId"].startswith("s_")][0]["expression"] == "$q1 / $tot * 100"
    for e in exprs[1:]:
        assert all(ch.isalnum() or ch == "_" for ch in e["refId"])
    names = [o["properties"][0]["value"] for o in panel["fieldConfig"]["overrides"]]
    assert names[0] == "DE (01)" and names[3] == "US-VA (04)"


def test_rtt_panels_name_the_peers(env_json):
    dash = gd.build_dashboard(gd.load_fleet(env_json))
    panel = next(p for p in dash["panels"] if p["title"] == "RTT da SG (05)")
    renames = {o["matcher"]["options"]: o["properties"][0]["value"]
               for o in panel["fieldConfig"]["overrides"]}
    assert renames["192.168.0.120"] == "DE (01)" and "192.168.0.132" not in renames


def test_bind_datasources_removes_the_placeholders(env_json):
    fleet = gd.load_fleet(env_json)
    dash = gd.bind_datasources(gd.build_dashboard(fleet),
                               {c["input"]: f"uid-{i}" for i, c in enumerate(fleet)})
    text = json.dumps(dash)
    assert "${DS_C" not in text and "__inputs" not in dash
    assert '"uid": "uid-5"' in text


class FakeGrafana(BaseHTTPRequestHandler):
    calls = []
    existing = {"dmos-ms-01-cluster-01"}

    def _reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        FakeGrafana.calls.append(("GET", self.path, self.headers.get("Authorization"), None))
        uid = self.path.rsplit("/", 1)[-1]
        self._reply(200 if uid in self.existing else 404, {})

    def do_PUT(self):
        FakeGrafana.calls.append(("PUT", self.path, None, self._body()))
        self._reply(200, {})

    def do_POST(self):
        body = self._body()
        FakeGrafana.calls.append(("POST", self.path, None, body))
        self._reply(200, {"url": "/d/dmos-fleet/x"} if "dashboards" in self.path else {})

    def log_message(self, *a):
        pass


def test_install_creates_datasources_and_dashboard(env_json, capsys):
    FakeGrafana.calls = []
    srv = HTTPServer(("127.0.0.1", 0), FakeGrafana)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}"
        assert gd.main(["install", "--env-json", str(env_json), "--grafana", url,
                        "--user", "admin", "--password", "pw"]) == 0
    finally:
        srv.shutdown()
    calls = FakeGrafana.calls
    assert calls[0][2].startswith("Basic ")
    puts = [c for c in calls if c[0] == "PUT"]
    posts_ds = [c for c in calls if c[0] == "POST" and c[1] == "/api/datasources"]
    assert len(puts) == 1 and len(posts_ds) == 5           # cluster 01 existed already
    assert posts_ds[0][3]["url"] == "http://192.168.0.123:32090"
    dash = [c for c in calls if c[1] == "/api/dashboards/db"][0][3]
    assert dash["overwrite"] is True and dash["dashboard"]["uid"] == "dmos-fleet"
    assert "${DS_C" not in json.dumps(dash)
    assert "/d/dmos-fleet/x" in capsys.readouterr().out
