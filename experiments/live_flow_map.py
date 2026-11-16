#!/usr/bin/env python3
"""
live_flow_map.py — live map of where the requests enter and where they are served.

Reads the HTTP flows of every cluster from its Hubble Relay (hubble CLI over a
kubectl port-forward), and serves a web page (http://127.0.0.1:8765) with:

  * the fleet as a map: k6 -> entry cluster j -> cluster i whose frontend
    serves the request; moving dots are sampled requests, coloured by their
    latency; the thickness of an arc is the request rate j -> i;
  * the matrix "entered in j -> served in i" (req/s and p95), the demand and
    the served shares of every cluster;
  * the frontend replicas of every cluster (ready and desired, kube-state-
    metrics; and the target of DMOS when it runs, http://localhost:9090);
  * the DMOS log of the run in progress (newest experiments/results/*/dmos.log),
    filtered to the decisions.

How a flow is attributed: the server side of an L7 RESPONSE of the entry
service (frontend) observed by the relay of the cluster that runs that
frontend pod; the entry cluster is the cluster that owns the client IP (the
Gateway/Ingress Envoy of the cluster the request entered), found from the pod
CIDRs and node addresses of the CiliumNodes. A response seen by a relay that
does not run the frontend pod (the client side of the same request) is not
counted twice.

REQUIREMENTS: kubectl and the hubble CLI in PATH (https://github.com/cilium/hubble/releases),
Hubble Relay in every cluster (it is there if the Hubble UI works), .cache/test-env.json.

USAGE (from the repository root)
  python experiments/live_flow_map.py                 # then open http://127.0.0.1:8765
  python experiments/live_flow_map.py --demo          # synthetic data, no cluster needed
  options: --port 8765 --namespace online-boutique --entry frontend
           --relay-namespace kube-system --relay-service hubble-relay --relay-tls
           --dmos-log PATH --dmos-metrics http://localhost:9090/metrics --window 10

Reading the flows loads the Cilium agents and this machine: use it in pilot
or demonstration runs, not while measuring.
"""
from __future__ import annotations

import argparse
import glob
import ipaddress
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parent.parent
HTML = Path(__file__).resolve().parent / "live_flow_map.html"
DEFAULT_ENV_JSON = REPO / ".cache" / "test-env.json"
RESULTS = REPO / "experiments" / "results"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# DMOS log lines shown in the page: the decisions, not the per-query chatter
LOG_NOISE = re.compile(r"FALLBACK server-side|Replica scaler inizializzato|development server|"
                       r"Effective config written|Insufficient history|No derivative|"
                       r"HTTP Request|Score totale|Φ_|\\u03a6_", re.I)
SCALED_RE = re.compile(r"Scaled (\S+)/(\S+) (?:→|->) (\d+) replicas")
UNICODE_ESC = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")


# ── fleet and IP ownership ───────────────────────────────────────────────────

def load_fleet(env_json: Path) -> List[Dict[str, Any]]:
    env = json.loads(Path(env_json).read_text(encoding="utf-8"))
    out = []
    for c in env.get("clusters") or []:
        kc = c.get("kubeconfig_path", "")
        if kc and not os.path.isabs(kc) and not re.match(r"^[A-Za-z]:", kc):
            kc = str(REPO / kc)
        out.append({"name": c["name"], "region": c.get("region") or c["name"],
                    "ip": c.get("ip", ""), "kubeconfig": kc,
                    "prometheus_url": c.get("prometheus_url") or f"http://{c.get('ip')}:32090"})
    if not out:
        raise SystemExit(f"no clusters in {env_json}")
    return out


class IpMap:
    """Which cluster owns an IP: node addresses (/32) and pod CIDRs."""

    def __init__(self):
        self.nets: List[Tuple[Any, str]] = []
        self.hosts: Dict[str, str] = {}
        self._cache: Dict[str, Optional[str]] = {}

    def add(self, cluster: str, cidrs=(), ips=()) -> None:
        for c in cidrs:
            try:
                self.nets.append((ipaddress.ip_network(c, strict=False), cluster))
            except ValueError:
                pass
        for ip in ips:
            if ip:
                self.hosts[ip] = cluster
        # most specific network first
        self.nets.sort(key=lambda x: -x[0].prefixlen)
        self._cache.clear()

    def lookup(self, ip: Optional[str]) -> Optional[str]:
        if not ip:
            return None
        if ip in self._cache:
            return self._cache[ip]
        found = self.hosts.get(ip)
        if found is None:
            try:
                addr = ipaddress.ip_address(ip)
                found = next((cl for net, cl in self.nets
                              if addr.version == net.version and addr in net), None)
            except ValueError:
                found = None
        self._cache[ip] = found
        return found


def cilium_addresses(items: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Pod CIDRs and addresses of `kubectl get ciliumnodes -o json` items
    (and of `kubectl get nodes -o json` items, which have spec.podCIDRs and
    status.addresses)."""
    cidrs, ips = [], []
    for it in items:
        spec, status = it.get("spec") or {}, it.get("status") or {}
        cidrs += (spec.get("ipam") or {}).get("podCIDRs") or []
        cidrs += spec.get("podCIDRs") or []
        if spec.get("podCIDR"):
            cidrs.append(spec["podCIDR"])
        for a in (spec.get("addresses") or []) + (status.get("addresses") or []):
            ip = a.get("ip") or a.get("address")
            if ip and re.match(r"^[0-9a-fA-F:.]+$", ip):
                ips.append(ip)
    return sorted(set(cidrs)), sorted(set(ips))


# ── flows ────────────────────────────────────────────────────────────────────

def _g(d: Dict[str, Any], *keys, default=None):
    """First present key (the hubble jsonpb output uses proto names; accept
    camelCase too)."""
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
            return d[k]
    return default


def endpoint_is(ep: Dict[str, Any], workload: str) -> bool:
    if not isinstance(ep, dict):
        return False
    for w in _g(ep, "workloads", default=[]) or []:
        if w.get("name") == workload:
            return True
    labels = _g(ep, "labels", default=[]) or []
    if f"k8s:app={workload}" in labels or f"k8s:app.kubernetes.io/name={workload}" in labels:
        return True
    pod = _g(ep, "pod_name", "podName", default="") or ""
    # Deployment pod: <workload>-<replicaset hash>-<5 chars>
    return bool(re.match(rf"^{re.escape(workload)}-[a-z0-9]{{6,10}}-[a-z0-9]{{5}}$", pod))


def parse_flow(obj: Dict[str, Any], observer: str, ipmap: IpMap,
               entry: str = "frontend") -> Optional[Dict[str, Any]]:
    """One served request from a hubble jsonpb line, or None.

    -> {"entry": j, "serving": i, "lat_ms": float, "code": int, "local": bool}
    or {"skip": reason} for flows that are of the right kind but not counted."""
    flow = obj.get("flow") if isinstance(obj.get("flow"), dict) else obj
    l7 = _g(flow, "l7", "L7")
    if not isinstance(l7, dict) or str(_g(l7, "type", default="")).upper() != "RESPONSE":
        return None
    src, dst = _g(flow, "source", default={}), _g(flow, "destination", default={})
    ipb = _g(flow, "IP", "ip", default={}) or {}
    # a response goes server -> client
    if not endpoint_is(src, entry):
        return None
    server_ip, client_ip = _g(ipb, "source"), _g(ipb, "destination")
    serving = ipmap.lookup(server_ip) or _g(src, "cluster_name", "clusterName") or observer
    if serving != observer:
        return {"skip": "client_side"}          # counted by the relay of the serving cluster
    entry_cl = ipmap.lookup(client_ip) or _g(dst, "cluster_name", "clusterName")
    if not entry_cl:
        return {"skip": "unknown_client"}
    lat_ns = _g(l7, "latency_ns", "latencyNs", default=0) or 0
    http = _g(l7, "http", default={}) or {}
    return {"entry": entry_cl, "serving": serving, "lat_ms": int(lat_ns) / 1e6,
            "code": int(_g(http, "code", default=0) or 0), "local": entry_cl == serving}


# ── aggregation ──────────────────────────────────────────────────────────────

def _quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class Aggregator:
    """Sliding window of served requests + a ring of sampled requests."""

    def __init__(self, clusters: List[str], window_s: float = 10.0,
                 samples_per_s: int = 40, seed: int = 0):
        self.clusters = clusters
        self.window_s = window_s
        self.samples_per_s = samples_per_s
        self.events: Deque[Tuple[float, str, str, float, int]] = deque()
        self.samples: Deque[Dict[str, Any]] = deque(maxlen=2000)
        self.seq = 0
        self.skipped: Dict[str, int] = {}
        self._sec = -1
        self._sec_count = 0
        self._sec_seen = 0
        self._rng = random.Random(seed)
        self.lock = threading.Lock()

    def add(self, rec: Dict[str, Any], t: Optional[float] = None) -> None:
        t = time.time() if t is None else t
        with self.lock:
            if "skip" in rec:
                self.skipped[rec["skip"]] = self.skipped.get(rec["skip"], 0) + 1
                return
            self.events.append((t, rec["entry"], rec["serving"], rec["lat_ms"], rec["code"]))
            # reservoir-like sampling: at most samples_per_s dots per second
            sec = int(t)
            if sec != self._sec:
                self._sec, self._sec_count, self._sec_seen = sec, 0, 0
            self._sec_seen += 1
            if self._sec_count < self.samples_per_s or \
                    self._rng.random() < self.samples_per_s / self._sec_seen / 4:
                self._sec_count += 1
                self.seq += 1
                self.samples.append({"seq": self.seq, "t": round(t, 3), "entry": rec["entry"],
                                     "serving": rec["serving"], "lat": round(rec["lat_ms"], 1),
                                     "code": rec["code"]})

    def snapshot(self, now: Optional[float] = None, since: int = 0) -> Dict[str, Any]:
        now = time.time() if now is None else now
        with self.lock:
            while self.events and self.events[0][0] < now - self.window_s:
                self.events.popleft()
            lat: Dict[Tuple[str, str], List[float]] = {}
            errors: Dict[str, int] = {}
            for t_ev, j, i, ms, code in self.events:
                if t_ev < now - self.window_s:      # out of order and already too old
                    continue
                lat.setdefault((j, i), []).append(ms)
                if code >= 500 or code == 0:
                    errors[i] = errors.get(i, 0) + 1
            span = self.window_s
            matrix = {j: {i: 0.0 for i in self.clusters} for j in self.clusters}
            p95 = {j: {i: None for i in self.clusters} for j in self.clusters}
            for (j, i), xs in lat.items():
                matrix.setdefault(j, {})[i] = round(len(xs) / span, 2)
                p95.setdefault(j, {})[i] = round(_quantile(xs, 0.95), 1)
            served = {i: round(sum(matrix[j].get(i, 0.0) for j in matrix), 2) for i in self.clusters}
            entered = {j: round(sum(matrix[j].values()), 2) for j in matrix}
            all_lat = [x for xs in lat.values() for x in xs]
            remote = sum(len(xs) for (j, i), xs in lat.items() if j != i)
            samples = [s for s in self.samples if s["seq"] > since]
            return {
                "matrix": matrix, "p95": p95, "served": served, "entered": entered,
                "total_rps": round(len(all_lat) / span, 1),
                "p50_ms": round(_quantile(all_lat, 0.5), 1) if all_lat else None,
                "p95_ms": round(_quantile(all_lat, 0.95), 1) if all_lat else None,
                "remote_share": round(remote / len(all_lat), 3) if all_lat else None,
                "errors_rps": {i: round(n / span, 2) for i, n in errors.items()},
                "samples": samples[-400:], "seq": self.seq, "skipped": dict(self.skipped),
            }


# ── DMOS log ─────────────────────────────────────────────────────────────────

def clean_log_line(line: str) -> str:
    """'2026-10-29 12:01:08 | INFO | module:function | msg' -> '12:01:08 INFO msg',
    without colours and with the \\uXXXX escapes of the Windows log decoded."""
    line = ANSI.sub("", line).rstrip()
    line = UNICODE_ESC.sub(lambda m: chr(int(m.group(1) or m.group(2), 16)), line)
    m = re.match(r"^\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d)\s*\|\s*(\w+)\s*\|\s*[^|]*\|\s*(.*)$", line)
    if m:
        return f"{m.group(1)} {m.group(2)[:4]} {m.group(3).strip()}"
    return line


def classify_log_line(text: str, last: Dict[Tuple[str, str], int]) -> Optional[Tuple[str, str]]:
    """(kind, text to show) for a cleaned DMOS log line, or None to hide it.
    `last` remembers the replicas of every (cluster, service) to tell up from down."""
    if LOG_NOISE.search(text):
        return None
    m = SCALED_RE.search(text)
    if m:
        cl, svc, n = m.group(1), m.group(2), int(m.group(3))
        prev = last.get((cl, svc))
        last[(cl, svc)] = n
        kind = "down" if prev is not None and n < prev else "up"
        return kind, f"{text[:8]} {cl}/{svc} {prev if prev is not None else '?'} → {n}"
    if "Total replicas needed" in text:
        return "decision", text.split(" (sum of")[0]
    if "Reach-weighted demand" in text or "Scheduling di" in text or "Effective config" in text:
        return "info", text
    if re.search(r"\bburst\b|panic", text, re.I):
        return "info", text
    if " ERRO" in text[:14] or " CRIT" in text[:14]:
        return "error", text
    if " WARN" in text[:14]:
        return "warn", text
    return None


class LogTail:
    """Follows the DMOS log of the newest run (or a fixed file)."""

    def __init__(self, path: Optional[str] = None, keep: int = 300):
        self.fixed = path
        self.path: Optional[str] = None
        self.pos = 0
        self.lines: Deque[Dict[str, Any]] = deque(maxlen=keep)
        self.seq = 0
        self.run = ""
        self.started: Optional[float] = None
        self.replicas: Dict[Tuple[str, str], int] = {}
        self.lock = threading.Lock()

    def _newest(self) -> Optional[str]:
        if self.fixed:
            return self.fixed
        files = glob.glob(str(RESULTS / "*" / "dmos.log"))
        return max(files, key=os.path.getmtime) if files else None

    def poll(self) -> None:
        path = self._newest()
        if not path or not os.path.isfile(path):
            return
        if path != self.path:
            self.path, self.pos = path, 0
            self.replicas = {}
            run_dir = os.path.basename(os.path.dirname(path))
            m = re.match(r"^(\d{8}_\d{6})_(.*)$", run_dir)
            self.run = m.group(2) if m else run_dir
            try:
                self.started = time.mktime(time.strptime(m.group(1), "%Y%m%d_%H%M%S")) if m else None
            except ValueError:
                self.started = None
            with self.lock:
                self.seq += 1
                self.lines.append({"seq": self.seq, "text": f"── run {self.run} ──", "kind": "run"})
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return
        with self.lock:
            for raw in chunk.splitlines():
                got = classify_log_line(clean_log_line(raw), self.replicas)
                if got is None:
                    continue
                self.seq += 1
                self.lines.append({"seq": self.seq, "text": got[1][:400], "kind": got[0]})

    def since(self, seq: int) -> Dict[str, Any]:
        with self.lock:
            return {"run": self.run, "started": self.started, "path": self.path,
                    "lines": [x for x in self.lines if x["seq"] > seq], "seq": self.seq}


# ── Prometheus and DMOS metrics ──────────────────────────────────────────────

def prom_query(base: str, q: str, timeout: float = 4.0) -> List[Dict[str, Any]]:
    url = base.rstrip("/") + "/api/v1/query?" + urllib.parse.urlencode({"query": q})
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return (json.loads(r.read().decode("utf-8")).get("data") or {}).get("result") or []


def parse_prom_text(text: str, metric: str) -> Dict[Tuple[Tuple[str, str], ...], float]:
    """Samples of one metric in the Prometheus text format: {labels: value}."""
    out = {}
    for line in text.splitlines():
        if not line.startswith(metric + "{") and not line.startswith(metric + " "):
            continue
        m = re.match(rf"^{re.escape(metric)}(?:{{(.*)}})?\s+([^\s]+)", line)
        if not m:
            continue
        labels = tuple(sorted(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(1) or "")))
        try:
            out[labels] = float(m.group(2))
        except ValueError:
            pass
    return out


class Metrics:
    """Every few seconds: demand, ready/desired replicas, RTT per cluster
    (Prometheus of each cluster) and the DMOS target replicas (DMOS /metrics)."""

    def __init__(self, fleet, ns: str, entry: str, dmos_url: str):
        self.fleet, self.ns, self.entry, self.dmos_url = fleet, ns, entry, dmos_url
        self.data: Dict[str, Any] = {"clusters": {}, "dmos_running": False}
        self.lock = threading.Lock()
        self.ip_to_cluster = {c["ip"]: c["name"] for c in fleet if c["ip"]}

    def queries(self) -> Dict[str, str]:
        ns, e = self.ns, self.entry
        return {
            "demand": (f'sum(rate(hubble_http_requests_total{{destination_namespace="{ns}",'
                       f'reporter="client",traffic_direction="egress"}}[1m]))'),
            "ready": f'sum(kube_deployment_status_replicas_available{{namespace="{ns}",deployment="{e}"}})',
            "desired": f'sum(kube_deployment_spec_replicas{{namespace="{ns}",deployment="{e}"}})',
            "rtt": 'avg by (target) (ping_rtt_mean_seconds) * 1000',
        }

    def poll_once(self) -> None:
        qs = self.queries()
        clusters: Dict[str, Any] = {}
        for c in self.fleet:
            row: Dict[str, Any] = {}
            for key, q in qs.items():
                try:
                    res = prom_query(c["prometheus_url"], q)
                except Exception as ex:          # one cluster down must not stop the others
                    row["error"] = str(ex)[:120]
                    continue
                if key == "rtt":
                    row["rtt"] = {self.ip_to_cluster.get(s["metric"].get("target"), s["metric"].get("target")):
                                  round(float(s["value"][1]), 1) for s in res}
                else:
                    row[key] = round(float(res[0]["value"][1]), 2) if res else None
            clusters[c["name"]] = row
        dmos_target, running = {}, False
        try:
            with urllib.request.urlopen(self.dmos_url, timeout=2) as r:
                text = r.read().decode("utf-8", "replace")
            running = True
            for labels, v in parse_prom_text(text, "dmos_target_replicas").items():
                d = dict(labels)
                if d.get("service") == self.entry:
                    dmos_target[d.get("cluster")] = int(v)
        except Exception:
            pass
        with self.lock:
            self.data = {"clusters": clusters, "dmos_running": running, "dmos_target": dmos_target,
                         "updated": time.time()}

    def get(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.data)


# ── Hubble relay streams ─────────────────────────────────────────────────────

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port: int, timeout: float = 15.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


class RelayStream(threading.Thread):
    """port-forward to the Hubble Relay of one cluster + `hubble observe --follow`."""

    def __init__(self, cluster: Dict[str, Any], ipmap: IpMap, agg: Aggregator, args):
        super().__init__(daemon=True, name=f"relay-{cluster['name']}")
        self.c, self.ipmap, self.agg, self.a = cluster, ipmap, agg, args
        self.status = "starting"
        self.flows = 0
        self.stop = threading.Event()
        self.procs: List[subprocess.Popen] = []

    def _kubectl(self, *argv, timeout=20) -> str:
        cmd = ["kubectl", "--kubeconfig", self.c["kubeconfig"], *argv]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              check=True).stdout

    def _svc_port(self) -> str:
        try:
            return self._kubectl("-n", self.a.relay_namespace, "get", "svc", self.a.relay_service,
                                 "-o", "jsonpath={.spec.ports[0].port}").strip() or "80"
        except Exception:
            return "80"

    def run(self) -> None:
        backoff = 2.0
        while not self.stop.is_set():
            try:
                self._session()
                backoff = 2.0
            except Exception as ex:
                self.status = f"errore: {str(ex)[:140]}"
            finally:
                for p in self.procs:
                    if p.poll() is None:
                        p.terminate()
                self.procs = []
            self.stop.wait(backoff)
            backoff = min(30.0, backoff * 2)

    def _session(self) -> None:
        port = free_port()
        self.status = "port-forward"
        pf = subprocess.Popen(["kubectl", "--kubeconfig", self.c["kubeconfig"], "-n",
                               self.a.relay_namespace, "port-forward", f"svc/{self.a.relay_service}",
                               f"{port}:{self._svc_port()}", "--address", "127.0.0.1"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.procs.append(pf)
        if self.stop.is_set():
            return
        if not wait_port(port):
            err = (pf.stderr.read() if pf.poll() is not None else "") or "timeout"
            raise RuntimeError(f"port-forward a {self.a.relay_service}: {err.strip()[:120]}")
        cmd = ["hubble", "observe", "--server", f"127.0.0.1:{port}", "--follow",
               "--output", "jsonpb", "--type", "l7", "--namespace", self.a.namespace]
        if self.a.relay_tls:
            cmd += ["--tls", "--tls-allow-insecure"]
        if self.stop.is_set():
            return
        self.status = "connesso"
        ob = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              bufsize=1)
        self.procs.append(ob)
        for line in ob.stdout:
            if self.stop.is_set():
                return
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = parse_flow(json.loads(line), self.c["name"], self.ipmap, self.a.entry)
            except ValueError:
                continue
            if rec is not None:
                self.flows += 1
                self.agg.add(rec)
        err = ob.stderr.read() if ob.stderr else ""
        raise RuntimeError(f"hubble observe terminato: {err.strip()[:120] or 'senza errori'}")


def build_ip_map(fleet, timeout: float = 20.0) -> Tuple[IpMap, Dict[str, str]]:
    ipmap, notes = IpMap(), {}
    for c in fleet:
        cidrs, ips = [], [c["ip"]] if c["ip"] else []
        for kind in ("ciliumnodes", "nodes"):
            try:
                out = subprocess.run(["kubectl", "--kubeconfig", c["kubeconfig"], "get", kind,
                                      "-o", "json"], capture_output=True, text=True,
                                     timeout=timeout, check=True).stdout
                cc, ii = cilium_addresses(json.loads(out).get("items") or [])
                cidrs += cc
                ips += ii
            except Exception as ex:
                notes[c["name"]] = f"{kind}: {str(ex)[:100]}"
        ipmap.add(c["name"], cidrs, ips)
        notes.setdefault(c["name"], f"{len(set(cidrs))} pod CIDR, {len(set(ips))} indirizzi")
    return ipmap, notes


# ── demo mode ────────────────────────────────────────────────────────────────

class Demo(threading.Thread):
    """Synthetic flash crowd: demand 85 % DE, replicas growing from 1 per cluster,
    pool routing (a request is served where the replicas are), latency =
    processing + RTT between entry and serving cluster; fake DMOS log."""

    RTT = {"DE": 0, "FR": 30, "PL": 50, "US-VA": 100, "SG": 200, "NO": 40}

    def __init__(self, fleet, agg: Aggregator, metrics: Metrics, log: LogTail):
        super().__init__(daemon=True, name="demo")
        self.fleet, self.agg, self.metrics, self.log = fleet, agg, metrics, log
        self.rng = random.Random(1)
        self.t0 = time.time()

    def _rtt(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        return self.RTT.get(a, 80) + self.RTT.get(b, 80)

    def run(self) -> None:
        names = [c["name"] for c in self.fleet]
        region = {c["name"]: c["region"] for c in self.fleet}
        caps = dict(zip(names, [4, 3, 2, 4, 2, 3, 3, 3]))
        reps = {n: 1 for n in names}
        self.log.run, self.log.started = "demo-reach-pool-ps__con", self.t0
        last_decision = 0
        while True:
            t = time.time() - self.t0
            phase = (t % 300) / 300                       # 5-minute cycle
            rate = 60 + 300 * min(1, max(0, (phase - 0.1) / 0.3)) * (1 if phase < 0.8 else 0.3)
            hot = 0.85 if phase > 0.1 else 0.3
            demand = {n: rate * (hot if k == 0 else (1 - hot) / (len(names) - 1))
                      for k, n in enumerate(names)}
            if t - last_decision > 15:
                last_decision = t
                need = max(len(names), math.ceil(sum(demand.values()) / 20))
                order = sorted(names, key=lambda n: self._rtt(region[names[0]], region[n]))
                new = {n: 1 for n in names}
                left = need - len(names)
                for n in order:
                    add = min(caps[n] - 1, max(0, left))
                    new[n] += add
                    left -= add
                changes = {n: (reps[n], new[n]) for n in names if new[n] != reps[n]}
                reps = new
                with self.log.lock:
                    self.log.seq += 1
                    self.log.lines.append({"seq": self.log.seq, "kind": "decision",
                                           "text": time.strftime("%H:%M:%S") +
                                           f" INFO Total replicas needed: {need} [budget pooled, predictive]"})
                    for n, (a, b) in changes.items():
                        self.log.seq += 1
                        self.log.lines.append({"seq": self.log.seq, "kind": "up" if b > a else "down",
                                               "text": time.strftime("%H:%M:%S") +
                                               f" {n}/frontend {a} → {b}"})
            now = time.time()
            n_req = int(rate * 0.1)
            for _ in range(n_req):
                j = self.rng.choices(names, weights=[demand[n] for n in names])[0]
                i = self.rng.choices(names, weights=[reps[n] for n in names])[0]
                lat = 40 + self.rng.expovariate(1 / 30) + self._rtt(region[j], region[i])
                self.agg.add({"entry": j, "serving": i, "lat_ms": lat, "code": 200,
                              "local": i == j}, now)
            with self.metrics.lock:
                self.metrics.data = {
                    "clusters": {n: {"demand": round(demand[n], 1), "ready": reps[n], "desired": reps[n],
                                     "rtt": {m: self._rtt(region[n], region[m]) for m in names if m != n}}
                                 for n in names},
                    "dmos_running": True, "dmos_target": dict(reps), "updated": now}
            time.sleep(0.1)


# ── web server ───────────────────────────────────────────────────────────────

class App:
    def __init__(self, fleet, agg, metrics, log, streams, ip_notes, demo: bool):
        self.fleet, self.agg, self.metrics, self.log = fleet, agg, metrics, log
        self.streams, self.ip_notes, self.demo = streams, ip_notes, demo

    def state(self, since_sample: int, since_log: int) -> Dict[str, Any]:
        return {
            "now": time.time(),
            "demo": self.demo,
            "fleet": [{"name": c["name"], "region": c["region"]} for c in self.fleet],
            "flows": self.agg.snapshot(since=since_sample),
            "metrics": self.metrics.get(),
            "log": self.log.since(since_log),
            "relays": {s.c["name"]: {"status": s.status, "flows": s.flows,
                                     "ipmap": self.ip_notes.get(s.c["name"], "")}
                       for s in self.streams},
            "window_s": self.agg.window_s,
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/quit":
                # clean stop (port-forwards and hubble terminated), e.g. at the
                # end of a run of the overhead test
                self._send(200, b"bye\n", "text/plain")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            elif u.path == "/state":
                q = parse_qs(u.query)
                body = json.dumps(app.state(int(q.get("s", ["0"])[0]), int(q.get("l", ["0"])[0])))
                self._send(200, body.encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass
    return Handler


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-json", type=Path, default=DEFAULT_ENV_JSON)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--namespace", default="online-boutique")
    ap.add_argument("--entry", default="frontend")
    ap.add_argument("--relay-namespace", default="kube-system")
    ap.add_argument("--relay-service", default="hubble-relay")
    ap.add_argument("--relay-tls", action="store_true", help="Hubble Relay with TLS")
    ap.add_argument("--dmos-log", default=None, help="default: newest experiments/results/*/dmos.log")
    ap.add_argument("--dmos-metrics", default="http://localhost:9090/metrics")
    ap.add_argument("--window", type=float, default=10.0, help="seconds of the rate/p95 window")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no cluster")
    ap.add_argument("--duration", type=float, default=0,
                    help="stop by itself after this many seconds (0 = never; also GET /quit)")
    a = ap.parse_args(argv)

    if a.demo and not a.env_json.is_file():
        fleet = [{"name": f"ms-01-cluster-0{i}", "region": r, "ip": "", "kubeconfig": "",
                  "prometheus_url": ""}
                 for i, r in enumerate(["DE", "FR", "PL", "US-VA", "SG", "NO"], start=1)]
    else:
        fleet = load_fleet(a.env_json)
    names = [c["name"] for c in fleet]
    agg = Aggregator(names, window_s=a.window)
    metrics = Metrics(fleet, a.namespace, a.entry, a.dmos_metrics)
    log = LogTail(a.dmos_log)
    streams: List[RelayStream] = []
    notes: Dict[str, str] = {}

    if a.demo:
        Demo(fleet, agg, metrics, log).start()
    else:
        missing = [t for t in ("kubectl", "hubble") if shutil.which(t) is None]
        if missing:
            print(f"ERRORE: {', '.join(missing)} non nel PATH (hubble: "
                  f"https://github.com/cilium/hubble/releases)", file=sys.stderr)
            return 2
        print("Leggo pod CIDR e indirizzi dei nodi (CiliumNode) di ogni cluster ...")
        ipmap, notes = build_ip_map(fleet)
        for n, s in notes.items():
            print(f"  {n}: {s}")
        for c in fleet:
            st = RelayStream(c, ipmap, agg, a)
            st.start()
            streams.append(st)

        def metrics_loop():
            while True:
                try:
                    metrics.poll_once()
                except Exception:
                    pass
                time.sleep(5)
        threading.Thread(target=metrics_loop, daemon=True).start()

        def log_loop():
            while True:
                log.poll()
                time.sleep(1)
        threading.Thread(target=log_loop, daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(App(fleet, agg, metrics, log,
                                                                         streams, notes, a.demo)))
    print(f"Mappa live: http://127.0.0.1:{a.port}   (Ctrl+C per uscire)", flush=True)
    if a.duration > 0:
        threading.Timer(a.duration, srv.shutdown).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_streams(streams)
        if streams:
            print("Mappa live chiusa: port-forward e hubble terminati", flush=True)
        else:
            print("Mappa live chiusa (demo: nessuna connessione ai cluster)", flush=True)
    return 0


def stop_streams(streams: List["RelayStream"], timeout: float = 5.0) -> None:
    """Stop every relay stream and make sure no kubectl/hubble child survives."""
    for s in streams:
        s.stop.set()
    for _ in range(2):                    # a stream may start a process while we stop it
        for s in streams:
            for p in list(s.procs):
                if p.poll() is None:
                    p.terminate()
        for s in streams:
            for p in list(s.procs):
                try:
                    p.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    p.kill()
        for s in streams:
            s.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main())
