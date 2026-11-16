#!/usr/bin/env python3
"""
grafana_dashboard.py — one Grafana dashboard for the whole DMOS fleet.

Every cluster has its own Prometheus; this dashboard puts them side by side
in ONE Grafana (one data source per cluster, panels with the "Mixed" data
source), so that demand, serving, replicas, latency, network and resources of
all clusters share one time axis:

  Traffic      demand entering each cluster (the query DMOS uses), requests
               served by the frontend of each cluster, and the three shares
               (demand / served / replicas): with the pooled data plane the
               served share follows the replica share, not the demand.
  Replicas     frontend ready / desired per cluster (DMOS decisions), ready pods
  Latency      p95 from the ingress (includes the hop to a remote frontend),
               p95 at the frontend, 5xx, packets dropped by Cilium (Hubble)
  Network      ping RTT from each cluster to the others (netem geography)
  Resources    busiest node CPU, frontend CPU per replica, throttling
  L7 edges     p95 of every source -> destination workload pair, per cluster

USAGE (from the repository root, after scripts/render-runtime-files.sh)
  # JSON to import by hand (Grafana > Dashboards > New > Import): Grafana asks
  # which Prometheus data source is each cluster
  python scripts/grafana_dashboard.py render

  # or everything through the Grafana API: creates/updates one Prometheus data
  # source per cluster (uid dmos-<cluster>) and the dashboard (uid dmos-fleet)
  python scripts/grafana_dashboard.py install --grafana http://<GRAFANA_HOST>:32000 \\
      --user admin --password '<password>'          # or --token <service account token>

Options: --env-json (default .cache/test-env.json), --namespace
(online-boutique), --entry (frontend), --out (render output).
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ENV_JSON = REPO / ".cache" / "test-env.json"
DEFAULT_OUT = REPO / "deployments" / "grafana" / "dmos-fleet-dashboard.json"
DASHBOARD_UID = "dmos-fleet"
MIXED = {"type": "datasource", "uid": "-- Mixed --"}
EXPR = {"type": "__expr__", "uid": "__expr__"}
# One colour per cluster, stable across panels (Grafana palette names)
COLOURS = ["blue", "green", "orange", "purple", "red", "yellow", "super-light-blue",
           "dark-green", "dark-orange", "dark-purple"]


# ── fleet ────────────────────────────────────────────────────────────────────

def load_fleet(env_json: Path) -> List[Dict[str, str]]:
    env = json.loads(Path(env_json).read_text(encoding="utf-8"))
    fleet = []
    for i, c in enumerate(env.get("clusters") or [], start=1):
        name = c["name"]
        region = c.get("region") or name
        fleet.append({
            "name": name,
            "region": region,
            "label": f"{region} ({name.split('-')[-1]})",
            "ip": c.get("ip", ""),
            "prometheus_url": c.get("prometheus_url") or f"http://{c.get('ip')}:32090",
            "input": f"DS_C{i}",
            "colour": COLOURS[(i - 1) % len(COLOURS)],
        })
    if not fleet:
        raise SystemExit(f"no clusters in {env_json}")
    return fleet


def ds_ref(c: Dict[str, str]) -> Dict[str, str]:
    return {"type": "prometheus", "uid": "${" + c["input"] + "}"}


# ── panel builders ───────────────────────────────────────────────────────────

class Layout:
    """Places panels left to right, rows of 24 columns."""

    def __init__(self):
        self.y, self.x, self.row_h = 0, 0, 0
        self.next_id = 1
        self.panels: List[Dict[str, Any]] = []

    def _pos(self, w: int, h: int) -> Dict[str, int]:
        if self.x + w > 24:
            self.y += self.row_h
            self.x, self.row_h = 0, 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def row(self, title: str) -> None:
        if self.x:
            self.y += self.row_h
            self.x, self.row_h = 0, 0
        self.panels.append({"type": "row", "title": title, "collapsed": False, "id": self.next_id,
                            "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1}, "panels": []})
        self.next_id += 1
        self.y += 1

    def add(self, panel: Dict[str, Any], w: int, h: int = 8) -> None:
        panel["id"] = self.next_id
        self.next_id += 1
        panel["gridPos"] = self._pos(w, h)
        self.panels.append(panel)


def _colour_overrides(fleet, names=None) -> List[Dict[str, Any]]:
    out = []
    for c, n in zip(fleet, names or [c["label"] for c in fleet]):
        out.append({"matcher": {"id": "byName", "options": n},
                    "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c["colour"]}}]})
    return out


def timeseries(title: str, description: str, targets: List[Dict[str, Any]], unit: str,
               datasource: Dict[str, str], stack: bool = False, overrides=None,
               decimals: Optional[int] = None, minv: Optional[float] = 0,
               maxv: Optional[float] = None, span_nulls: bool = True) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "unit": unit,
        "custom": {"drawStyle": "line", "lineWidth": 2, "fillOpacity": 25 if stack else 5,
                   "showPoints": "never", "spanNulls": span_nulls,
                   "stacking": {"mode": "normal" if stack else "none", "group": "A"}},
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    if minv is not None:
        defaults["min"] = minv
    if maxv is not None:
        defaults["max"] = maxv
    return {
        "type": "timeseries", "title": title, "description": description,
        "datasource": datasource, "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": overrides or []},
        "options": {"legend": {"displayMode": "table", "placement": "right",
                               "calcs": ["lastNotNull", "max"]},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }


def per_cluster(fleet, expr: str, ref_prefix: str = "q", hide: bool = False,
                legend: Optional[str] = None) -> List[Dict[str, Any]]:
    """One target per cluster with the same PromQL, legend = cluster label."""
    return [{"refId": f"{ref_prefix}{i}", "datasource": ds_ref(c), "expr": expr,
             "legendFormat": legend.replace("$cluster", c["label"]) if legend else c["label"],
             "range": True, "hide": hide, "interval": "15s"}
            for i, c in enumerate(fleet, start=1)]


def shares(fleet, expr: str, min_total: float = 0.0) -> List[Dict[str, Any]]:
    """Hidden per-cluster queries + server-side expressions: share of each
    cluster in the total, in %. Grafana expressions combine data sources.
    With min_total > 0 the share is left empty (0/0 = NaN) while the total is
    at or below min_total: between two runs only health checks reach the
    clusters, and their shares jump between 0 and 100 %."""
    targets = per_cluster(fleet, expr, ref_prefix="q", hide=True)
    total = "+".join(f"$q{i}" for i in range(1, len(fleet) + 1))
    targets.append({"refId": "tot", "datasource": EXPR, "type": "math",
                    "expression": total, "hide": True})
    for i, c in enumerate(fleet, start=1):
        if min_total > 0:
            on = f"($tot > {min_total:g})"
            expression = f"$q{i} * {on} / ($tot * {on}) * 100"
        else:
            expression = f"$q{i} / $tot * 100"
        targets.append({"refId": share_ref(c), "datasource": EXPR, "type": "math",
                        "expression": expression})
    return targets


def share_ref(c: Dict[str, str]) -> str:
    """refId of a share expression: letters, digits and '_' only."""
    return "s_" + re.sub(r"[^A-Za-z0-9_]", "_", c["label"]).strip("_")


def share_overrides(fleet) -> List[Dict[str, Any]]:
    """Name and colour of the share expressions (their series are named by refId)."""
    return [{"matcher": {"id": "byFrameRefID", "options": share_ref(c)},
             "properties": [{"id": "displayName", "value": c["label"]},
                            {"id": "color", "value": {"mode": "fixed", "fixedColor": c["colour"]}}]}
            for c in fleet]


# ── PromQL ───────────────────────────────────────────────────────────────────

def queries(ns: str, entry: str) -> Dict[str, str]:
    pod = f'pod=~"{entry}-.*"'
    return {
        # same query as PrometheusClient.get_ingress_rate with match "namespace"
        # (Kubephos: the Gateway Envoy is the client of the entry service)
        "demand": (f'sum(rate(hubble_http_requests_total{{destination_namespace="{ns}",'
                   f'reporter="client",traffic_direction="egress"}}[1m]))'),
        "served": (f'sum(rate(hubble_http_requests_total{{destination_namespace="{ns}",'
                   f'destination_workload="{entry}",reporter="server",'
                   f'traffic_direction="ingress"}}[1m]))'),
        "ready": f'sum(kube_deployment_status_replicas_available{{namespace="{ns}",deployment="{entry}"}})',
        "desired": f'sum(kube_deployment_spec_replicas{{namespace="{ns}",deployment="{entry}"}})',
        "pods_ready": f'sum(kube_pod_status_ready{{namespace="{ns}",condition="true"}})',
        "restarts": f'sum(increase(kube_pod_container_status_restarts_total{{namespace="{ns}"}}[5m]))',
        "p95_ingress": (f'histogram_quantile(0.95, sum by (le) (rate(hubble_http_request_duration_seconds_bucket'
                        f'{{destination_namespace="{ns}",reporter="client",traffic_direction="egress"}}[1m]))) * 1000'),
        "p95_frontend": (f'histogram_quantile(0.95, sum by (le) (rate(hubble_http_request_duration_seconds_bucket'
                         f'{{destination_namespace="{ns}",destination_workload="{entry}",reporter="server"}}[1m]))) * 1000'),
        "errors_5xx": (f'sum(rate(hubble_http_requests_total{{destination_namespace="{ns}",'
                       f'destination_workload="{entry}",reporter="server",status=~"5.."}}[1m])) or vector(0)'),
        "drops": 'sum(rate(hubble_drop_total[1m])) or vector(0)',
        "rtt": 'avg by (target) (ping_rtt_mean_seconds) * 1000',
        "node_cpu_max": '100 * max(1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])))',
        "fe_cpu_per_replica": (f'1000 * sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",{pod},'
                               f'container!="",container!="POD"}}[1m])) / '
                               f'sum(kube_deployment_status_replicas_available{{namespace="{ns}",deployment="{entry}"}})'),
        "fe_throttling": (f'100 * sum(rate(container_cpu_cfs_throttled_periods_total{{namespace="{ns}",{pod},container!=""}}[1m]))'
                          f' / sum(rate(container_cpu_cfs_periods_total{{namespace="{ns}",{pod},container!=""}}[1m]))'),
        "app_cpu": (f'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",'
                    f'container!="",container!="POD"}}[1m]))'),
        "l7_edges": (f'histogram_quantile(0.95, sum by (le, source_workload, destination_workload) '
                     f'(rate(hubble_http_request_duration_seconds_bucket{{destination_namespace="{ns}"}}[2m]))) * 1000'),
    }


# ── dashboard ────────────────────────────────────────────────────────────────

def build_dashboard(fleet: List[Dict[str, str]], ns: str = "online-boutique",
                    entry: str = "frontend") -> Dict[str, Any]:
    q = queries(ns, entry)
    L = Layout()
    colours = _colour_overrides(fleet)

    L.row("Traffico: dove entra e dove viene servito")
    L.add(timeseries("Domanda in ingresso per cluster",
                     "Richieste verso l'applicazione inviate dai client di ogni cluster (Hubble, lato "
                     "client): la stessa misura con cui DMOS calcola la domanda. Con k6 include i "
                     "redirect 302 (circa 1,2 volte le iterazioni).",
                     per_cluster(fleet, q["demand"]), "reqps", MIXED, stack=True, overrides=colours), 12)
    L.add(timeseries("Richieste servite dal frontend di ogni cluster",
                     "Richieste ricevute dai pod frontend del cluster, da qualsiasi cluster arrivino "
                     "(Hubble, lato server). Con il pool (affinity none) seguono le repliche.",
                     per_cluster(fleet, q["served"]), "reqps", MIXED, stack=True, overrides=colours), 12)
    L.add(timeseries("Quota di domanda (%)", "Domanda di ogni cluster sul totale (vuota quando il "
                     "totale è sotto 5 req/s, cioè tra un run e l'altro).",
                     shares(fleet, q["demand"], min_total=5), "percent", MIXED,
                     overrides=share_overrides(fleet), maxv=100, span_nulls=False), 8)
    L.add(timeseries("Quota servita (%)", "Richieste servite da ogni cluster sul totale (vuota "
                     "sotto 5 req/s). Con il pool segue la quota di repliche, non la domanda.",
                     shares(fleet, q["served"], min_total=5), "percent", MIXED,
                     overrides=share_overrides(fleet), maxv=100, span_nulls=False), 8)
    L.add(timeseries("Quota di repliche frontend (%)",
                     "Repliche pronte di ogni cluster sul totale: con il pool la quota servita la segue.",
                     shares(fleet, q["ready"]), "percent", MIXED, overrides=share_overrides(fleet), maxv=100), 8)

    L.row("Repliche (decisioni di DMOS)")
    L.add(timeseries("Repliche frontend pronte",
                     "Pod frontend pronti per cluster (kube-state-metrics).",
                     per_cluster(fleet, q["ready"]), "short", MIXED, stack=True, overrides=colours,
                     decimals=0), 12)
    L.add(timeseries("Repliche frontend richieste",
                     "spec.replicas del Deployment frontend: quello che DMOS ha chiesto. La distanza "
                     "dalle pronte è il tempo di avvio dei pod.",
                     per_cluster(fleet, q["desired"]), "short", MIXED, overrides=colours, decimals=0), 12)
    L.add(timeseries("Pod pronti dell'applicazione",
                     f"Tutti i pod pronti nel namespace {ns} (frontend e backend co-scalati).",
                     per_cluster(fleet, q["pods_ready"]), "short", MIXED, overrides=colours, decimals=0), 12)
    L.add(timeseries("Riavvii dei container (5 min)", "Riavvii negli ultimi 5 minuti.",
                     per_cluster(fleet, q["restarts"]), "short", MIXED, overrides=colours, decimals=0), 12)

    L.row("Latenza ed errori")
    L.add(timeseries("p95 dall'ingress",
                     "p95 delle richieste inviate dall'ingress di ogni cluster (Hubble, lato client): "
                     "include l'attraversamento verso un frontend in un altro cluster. Non include il "
                     "tratto k6 → ingress.",
                     per_cluster(fleet, q["p95_ingress"]), "ms", MIXED, overrides=colours), 12)
    L.add(timeseries("p95 al frontend",
                     "p95 delle richieste ricevute dai frontend del cluster (lato server): "
                     "elaborazione e chiamate ai backend.",
                     per_cluster(fleet, q["p95_frontend"]), "ms", MIXED, overrides=colours), 12)
    L.add(timeseries("Errori 5xx al frontend", "Risposte 5xx dei frontend di ogni cluster.",
                     per_cluster(fleet, q["errors_5xx"]), "reqps", MIXED, overrides=colours), 12)
    L.add(timeseries("Pacchetti scartati da Cilium",
                     "hubble_drop_total: pacchetti scartati (policy, endpoint non pronto, ...). Un picco "
                     "durante gli scale-up indica richieste perse e ritrasmesse. Vuoto se la metrica "
                     "drop di Hubble non è attiva.",
                     per_cluster(fleet, q["drops"]), "pps", MIXED, overrides=colours), 12)

    L.row("Rete: RTT tra i cluster (ping_exporter)")
    by_ip = {c["ip"]: c["label"] for c in fleet}
    for c in fleet:
        rename = [{"matcher": {"id": "byName", "options": ip},
                   "properties": [{"id": "displayName", "value": lbl}]}
                  for ip, lbl in by_ip.items() if ip and ip != c["ip"]]
        L.add(timeseries(f"RTT da {c['label']}",
                         "RTT medio (ms) verso i nodi degli altri cluster.",
                         [{"refId": "A", "datasource": ds_ref(c), "expr": q["rtt"],
                           "legendFormat": "{{target}}", "range": True, "interval": "15s"}],
                         "ms", ds_ref(c), overrides=rename), 8)

    L.row("Risorse")
    L.add(timeseries("CPU del nodo più carico (%)",
                     "Massimo, sui nodi del cluster, della CPU non idle (node_exporter).",
                     per_cluster(fleet, q["node_cpu_max"]), "percent", MIXED, overrides=colours,
                     maxv=100), 12)
    L.add(timeseries("CPU per replica frontend (millicore)",
                     "CPU media di un pod frontend: uguale in tutti i cluster se il traffico segue le "
                     "repliche (pool).",
                     per_cluster(fleet, q["fe_cpu_per_replica"]), "short", MIXED, overrides=colours), 12)
    L.add(timeseries("Throttling CPU frontend (%)",
                     "Periodi CFS in cui il frontend ha superato il limite di CPU.",
                     per_cluster(fleet, q["fe_throttling"]), "percent", MIXED, overrides=colours,
                     maxv=100), 12)
    L.add(timeseries("CPU totale dell'applicazione (core)",
                     f"Somma della CPU dei container del namespace {ns}.",
                     per_cluster(fleet, q["app_cpu"]), "short", MIXED, stack=True, overrides=colours), 12)

    L.row("Archi L7: p95 per coppia di workload")
    for c in fleet:
        L.add({
            "type": "table", "title": f"Archi L7 — {c['label']}",
            "description": "p95 (ms) per sorgente → destinazione, ultimi 2 minuti. Sorgente vuota = "
                           "ingresso dall'esterno del cluster.",
            "datasource": ds_ref(c),
            "targets": [{"refId": "A", "datasource": ds_ref(c), "expr": q["l7_edges"],
                         "instant": True, "range": False, "format": "table"}],
            "transformations": [{"id": "organize", "options": {
                "excludeByName": {"Time": True},
                "renameByName": {"source_workload": "sorgente",
                                 "destination_workload": "destinazione", "Value": "p95 (ms)"}}}],
            "fieldConfig": {"defaults": {"unit": "ms", "decimals": 1}, "overrides": []},
            "options": {"sortBy": [{"displayName": "p95 (ms)", "desc": True}]},
        }, 8, 9)

    return {
        "__inputs": [{"name": c["input"], "label": f"Prometheus {c['label']}",
                      "description": f"Prometheus del cluster {c['name']} ({c['prometheus_url']})",
                      "type": "datasource", "pluginId": "prometheus", "pluginName": "Prometheus"}
                     for c in fleet],
        "__requires": [{"type": "datasource", "id": "prometheus", "name": "Prometheus",
                        "version": "1.0.0"},
                       {"type": "grafana", "id": "grafana", "name": "Grafana", "version": "10.0.0"}],
        "uid": DASHBOARD_UID,
        "title": "DMOS — flotta multi-cluster",
        "description": "Domanda, servizio, repliche, latenza, rete e risorse di tutti i cluster "
                       "(generata da scripts/grafana_dashboard.py)",
        "tags": ["dmos"],
        "timezone": "browser",
        "editable": True,
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "schemaVersion": 39,
        "version": 1,
        "panels": L.panels,
        "templating": {"list": []},
        "annotations": {"list": []},
    }


def bind_datasources(dashboard: Dict[str, Any], uids: Dict[str, str]) -> Dict[str, Any]:
    """Replace the ${DS_Cx} placeholders with data source uids (what the
    Grafana import screen does) and drop the import metadata."""
    text = json.dumps({k: v for k, v in dashboard.items() if not k.startswith("__")})
    for name, uid in uids.items():
        text = text.replace("${" + name + "}", uid)
    return json.loads(text)


# ── Grafana API ──────────────────────────────────────────────────────────────

class Grafana:
    def __init__(self, url: str, user: str = "", password: str = "", token: str = "",
                 timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        if token:
            self.auth = f"Bearer {token}"
        elif user:
            self.auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
        else:
            raise SystemExit("Grafana credentials: --token, or --user and --password")

    def call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": self.auth,
                                              "Content-Type": "application/json",
                                              "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read() or b"{}")
            except ValueError:
                payload = {}
            return e.code, payload

    def upsert_datasource(self, uid: str, name: str, url: str) -> None:
        body = {"uid": uid, "name": name, "type": "prometheus", "url": url, "access": "proxy",
                "isDefault": False, "jsonData": {"timeInterval": "15s", "httpMethod": "POST"}}
        status, _ = self.call("GET", f"/api/datasources/uid/{uid}")
        if status == 200:
            status, payload = self.call("PUT", f"/api/datasources/uid/{uid}", body)
        else:
            status, payload = self.call("POST", "/api/datasources", body)
        if status not in (200, 201):
            raise SystemExit(f"data source {name}: HTTP {status} {payload}")

    def upsert_dashboard(self, dashboard: Dict[str, Any]) -> str:
        status, payload = self.call("POST", "/api/dashboards/db",
                                    {"dashboard": dashboard, "overwrite": True,
                                     "message": "scripts/grafana_dashboard.py"})
        if status != 200:
            raise SystemExit(f"dashboard: HTTP {status} {payload}")
        return payload.get("url", "")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("render", "install"))
    ap.add_argument("--env-json", type=Path, default=DEFAULT_ENV_JSON)
    ap.add_argument("--namespace", default="online-boutique")
    ap.add_argument("--entry", default="frontend")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--grafana", help="Grafana URL, e.g. http://<GRAFANA_HOST>:32000")
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--token", default="", help="service account token (instead of user/password)")
    a = ap.parse_args(argv)

    fleet = load_fleet(a.env_json)
    dash = build_dashboard(fleet, a.namespace, a.entry)
    if a.command == "render":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dash, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"dashboard for {len(fleet)} clusters written to {a.out}")
        print("Grafana > Dashboards > New > Import > Upload JSON file; choose the Prometheus "
              "data source of each cluster when asked")
        return 0

    if not a.grafana:
        ap.error("install needs --grafana URL")
    g = Grafana(a.grafana, a.user, a.password, a.token)
    uids = {}
    for c in fleet:
        uid = f"dmos-{c['name']}"[:40]
        g.upsert_datasource(uid, f"DMOS {c['label']}", c["prometheus_url"])
        uids[c["input"]] = uid
        print(f"  data source {uid:<28} -> {c['prometheus_url']}")
    url = g.upsert_dashboard(bind_datasources(dash, uids))
    print(f"dashboard: {a.grafana.rstrip('/')}{url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
