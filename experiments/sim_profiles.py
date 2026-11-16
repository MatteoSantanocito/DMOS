#!/usr/bin/env python3
"""
sim_profiles.py — offline study of how much the weight profiles of the DMOS
score can change the placement, before spending testbed time on them.

It runs the Level-1 decision of DMOS cycle by cycle on a steady peak, with
DMOS's own code wherever it exists:

  budget N      the formula of dmos_main.schedule_service:
                N = Σ_i clip(ceil(λ_in_i / c_i), n_min_i, n_max_i)
  score         ScoreFunctions built from a ScoreModel (weights.yaml format),
                ScoreParameters as the scheduler builds them (ρ from
                network_parameters, ν/CI_max/... from the model's parameters)
  allocation    WinnerDetermination.allocate (proportional, capacity-capped,
                largest remainder), then max(n_min, min(n_max, ·)) per cluster
  bid capacity  min(free CPU / cpu_request, free memory / memory_request, n_max)
  carbon        CarbonClient.trace_value_at (the hourly traces DMOS replays),
                elapsed from the start of the load

and a small model of the testbed for what DMOS observes:

  served share  s_i = r_i / Σ r (frontend global, affinity none: the eBPF
                datapath spreads connections uniformly over the ready pods)
  node CPU      busy_i = b0_i + k_i · served_i (fitted on the earlier-campaign
                timelines: ~0.2-0.35 cores + ~14 mcore per served req/s)
  free CPU      --slack measured: as DMOS reads it today (all the nodes of the
                cluster minus the usage, over the worker's cores: the share
                exceeds 1 and Φ_cap sees the hardware tier only);
                --slack worker: the worker's own free CPU
  latency       Hubble p95 seen by Φ_lat: L0 / (1 - u) on node utilisation u
  RTT           netem delays of local.yaml: RTT(i,j) = d_i + d_j; Φ_net uses
                the mean RTT of i toward its peers

Data plane (--dataplane):
  pool   true distribution (frontend global, affinity none, a Cilium without
         bug #46623): the share a cluster serves is its share of replicas
  local  each cluster serves the demand entering it — what the archived
         runs of June 2026 actually did (served share = demand share)

Client latency (Monte Carlo per cycle): netem delay of the ingress node, one
more round trip when served in another cluster, plus the processing time of
a replica at its load (PROC_CURVE, fitted on the archived runs; failures
above ~55 req/s per replica). In 'local' mode, with the legacy behaviours
(--budget cluster --slack measured), the model reproduces the earlier-campaign picture
(off/l2 collapse from s65 on, DMOS holds); it is pessimistic for the
baselines at intermediate skew (no Level-2 extra replicas). Defaults follow
the current DMOS: total budget cap, free resources of the worker node.

Not modelled: Level 2 (PD ±1 per cluster, the predictor), rate limits, the
cold-start phase, retries, the Ingress proxy load. The result is the
steady-state placement Level 1 asks for at the peak and its first-order
effect on latency, SLO and carbon.

Outputs per (profile, regime, peak), averaged over the peak cycles:
  frontend replicas per cluster, N, total frontend R, backend replicas
  (co-scaling rule), CI of the served requests, extra network RTT per request
  (requests served outside their ingress cluster pay d_k + d_i), share served
  remotely, node CPU.

Usage:
  python experiments/sim_profiles.py                                  # default study
  python experiments/sim_profiles.py --peaks "350 700" --regimes "uni bal con-sg"
  python experiments/sim_profiles.py --profiles "nca green@nu2" --csv out.csv
Profiles: score models of config/weights.yaml, the candidates defined below
(perf, equal, green, campaign), '<name>@nu<x>' to override ν, and the
baselines 'off' and 'l2'. Regimes: run_naming syntax (uni, bal-sg, con-pl,
s50-p500, ...) or 'w:<w1>,<w2>,...' for an explicit demand vector.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

from loguru import logger as _loguru  # noqa: E402

import run_naming as rn  # noqa: E402
from src.level1.score_functions import ScoreFunctions, ScoreParameters, ClusterMetrics  # noqa: E402
from src.level1.score_model import ScoreModel  # noqa: E402
from src.level1.winner_determination import WinnerDetermination, ClusterBid  # noqa: E402
from src.metrics.carbon_client import CarbonClient  # noqa: E402
from src.utils.config_loader import ConfigLoader  # noqa: E402

# ── Candidate profiles (not in weights.yaml yet) ────────────────────────────
GATE = {"eps": 0.05, "gamma": 2}
CANDIDATES: Dict[str, dict] = {
    # weights of the earlier campaign runs (arm l1l2)
    "campaign": {"weights": {"latency": .25, "capacity": .20, "net": .25, "carbon": .30}},
    # performance: user latency is mostly network → latency + proximity
    "perf":     {"weights": {"latency": .30, "capacity": .20, "net": .45, "carbon": .05}},
    "equal":    {"weights": {"latency": .25, "capacity": .25, "net": .25, "carbon": .25}},
    "green":    {"weights": {"latency": .05, "capacity": .20, "net": .10, "carbon": .65}},
    # extremes, to bound what weights alone can do
    "only-net": {"weights": {"latency": 0, "capacity": 0, "net": 1, "carbon": 0}},
    "only-car": {"weights": {"latency": 0, "capacity": 0, "net": 0, "carbon": 1}},
}

# ── Testbed model (fitted on the earlier-campaign timelines, see module doc) ────
NODE_IDLE_CORES = {"ms-cluster-01": 0.35, "ms-cluster-02": 0.30, "ms-cluster-03": 0.20,
                   "ms-cluster-04": 0.25, "ms-cluster-05": 0.21, "ms-cluster-06": 0.24}
NODE_CORES_PER_RPS = {"ms-cluster-01": 0.0138, "ms-cluster-02": 0.0177, "ms-cluster-03": 0.0120,
                      "ms-cluster-04": 0.0160, "ms-cluster-05": 0.0112, "ms-cluster-06": 0.0134}
CONTROL_PLANE_CORES = 2.0    # 'measured' slack: extra node cores DMOS counts as free
MEASURED_MEM_SLACK = 0.97    # memory share DMOS reads at the peak (campaign log)
L0_MS = 9.0                  # Hubble p95 of the frontend at low load (campaign log: Φ_lat≈0.991)
PEAK_WINDOW = (300, 600)     # k6 peak phase (s from the load start)
CYCLE_S = 30

# ── Latency model of one frontend replica (processing only, network apart) ──
# Calibrated on the peak phase of the archived runs (local serving, June
# 2026): k6 per-cluster latency minus the netem delay, against the frontend
# requests per ready replica of that cluster (medians per load bin; the spread
# is wide, so this is a model for orders of magnitude). Beyond ~55 req/s per
# replica the chain saturates: latency jumps to seconds and requests fail.
#            λ per replica: p50 ms, p95 ms, failure probability
PROC_CURVE = [(0, 45, 110, 0.0), (10, 48, 120, 0.0), (20, 56, 140, 0.0),
              (30, 62, 170, 0.0), (40, 70, 215, 0.0), (45, 80, 260, 0.001),
              (50, 95, 450, 0.005), (55, 130, 900, 0.03), (60, 160, 1300, 0.06),
              (80, 250, 1600, 0.12), (100, 600, 3000, 0.30),
              (130, 900, 4500, 0.45), (180, 1200, 6500, 0.55), (400, 1500, 8000, 0.70)]
FAIL_LATENCY_MS = 20.0       # a failed request returns fast (5xx); it counts as an SLO violation
SLO_MS = 2000.0              # violation = failed or > 2 s


def _proc_at(lam: float) -> Tuple[float, float, float]:
    """(p50, p95, failure probability) of a replica at lam req/s (linear interpolation)."""
    pts = PROC_CURVE
    if lam <= pts[0][0]:
        return pts[0][1:]
    for (x0, a0, b0, f0), (x1, a1, b1, f1) in zip(pts, pts[1:]):
        if lam <= x1:
            w = (lam - x0) / (x1 - x0)
            return a0 + w * (a1 - a0), b0 + w * (b1 - b0), f0 + w * (f1 - f0)
    return pts[-1][1:]


def latency_metrics(fleet: "Fleet", d: Dict[str, float], peak: float, reps: Dict[str, int],
                    dataplane: str, netem: bool = True, samples: int = 40000,
                    seed: int = 7) -> dict:
    """Client-side p95, SLO violations and errors for one placement (Monte Carlo).

    pool   (true distribution, frontend global, affinity none): a request
           entering cluster k is served by a replica drawn uniformly from all
           ready replicas; every replica carries peak / R req/s.
    local  (what the earlier runs did): served in the cluster it enters, by that
           cluster's replicas (peak·d_k / r_k req/s each).
    Network: the response leaves the ingress node (+d_k); a request served
    in i ≠ k pays one more round trip k↔i (+d_k + d_i). Backends are local
    to the serving cluster."""
    import numpy as np
    rng = np.random.default_rng(seed)
    names = fleet.names
    dk = np.array([fleet.delay[n] if netem else 0.0 for n in names])
    dvec = np.array([d[n] for n in names])
    r = np.array([max(0, reps[n]) for n in names], dtype=float)
    R = r.sum()
    k = rng.choice(len(names), size=samples, p=dvec / dvec.sum())
    if dataplane == "pool":
        i = rng.choice(len(names), size=samples, p=r / R)
        lam = np.full(len(names), peak / R)
    else:
        i = k.copy()
        with np.errstate(divide="ignore"):
            lam = np.where(r > 0, peak * dvec / np.maximum(r, 1e-9), np.inf)
    p50 = np.empty(len(names)); p95 = np.empty(len(names)); fail = np.empty(len(names))
    for j in range(len(names)):
        if not np.isfinite(lam[j]):          # no replica where the request must be served
            p50[j], p95[j], fail[j] = 1.0, 1.0, 1.0
        else:
            p50[j], p95[j], fail[j] = _proc_at(float(lam[j]))
    sigma = np.log(np.maximum(p95 / p50, 1.0001)) / 1.645
    proc = np.exp(np.log(p50[i]) + sigma[i] * rng.standard_normal(samples))
    failed = rng.random(samples) < fail[i]
    net = dk[k] + np.where(i != k, dk[k] + dk[i], 0.0)
    lat = net + np.where(failed, FAIL_LATENCY_MS, proc)
    viol = failed | (lat > SLO_MS)
    return {"p95_ms": float(np.percentile(lat, 95)), "p50_ms": float(np.percentile(lat, 50)),
            "slo_viol": float(viol.mean()), "err": float(failed.mean()),
            "over_500ms": float((failed | (lat > 500.0)).mean()),
            "net_mean_ms": float(net.mean()), "lam_per_replica_max": float(np.max(lam[np.isfinite(lam)]))}


@dataclasses.dataclass
class Fleet:
    cfg: ConfigLoader
    names: List[str]
    cores: Dict[str, float]
    mem: Dict[str, float]
    region: Dict[str, str]
    delay: Dict[str, float]
    shorts: List[str]


def load_fleet(config_dir: Path) -> Fleet:
    cfg = ConfigLoader(str(config_dir))
    cl = cfg.get_all_clusters()
    names = list(cl)
    return Fleet(
        cfg=cfg, names=names,
        cores={n: float(c.cpu_cores) for n, c in cl.items()},
        mem={n: float(c.memory_gb) for n, c in cl.items()},
        region={n: str((c.carbon or {}).get("region_code", c.region)) for n, c in cl.items()},
        delay={n: float((c.netem or {}).get("delay_ms", 0) or 0) for n, c in cl.items()},
        shorts=[rn.region_short(c.region) for c in cl.values()],
    )


def demand_vector(fleet: Fleet, regime: str) -> Tuple[Dict[str, float], int, str]:
    """(shares per cluster, peak req/s, label)."""
    if regime.startswith("w:"):
        body, _, peak = regime[2:].partition("@")
        w = [float(x) for x in body.split(",")]
        if len(w) != len(fleet.names):
            raise SystemExit(f"{regime}: needs {len(fleet.names)} weights")
        tot = sum(w)
        return {n: x / tot for n, x in zip(fleet.names, w)}, int(peak or rn.DEFAULT_PEAK), regime
    p = rn.parse_regime(regime)
    if p.get("to"):
        # steady-state model: one demand vector per regime
        raise SystemExit(f"{regime}: the hot spot moves during the peak; the simulator is "
                         f"steady-state: simulate the two halves (e.g. con and con-sg) separately")
    w = list(rn.DEMAND_WEIGHTS[str(p["demand"])])
    if p["hot"]:
        idx = fleet.shorts.index(str(p["hot"]))
        w[0], w[idx] = w[idx], w[0]
    tot = sum(w)
    return {n: x / tot for n, x in zip(fleet.names, w)}, int(p["peak"]), regime


_MIN_SUFFIX = re.compile(r"^(?P<base>.+)-min(?P<k>\d)$")


def split_min(name: str) -> Tuple[str, Optional[int]]:
    """'green_nu2-min0' -> ('green_nu2', 0): the '-min<k>' arm suffix of
    run_naming.py (entry service minimum k replicas per cluster)."""
    m = _MIN_SUFFIX.match(name)
    return (m.group("base"), int(m.group("k"))) if m else (name, None)


def build_model(name: str, weights_raw: dict) -> Optional[ScoreModel]:
    """ScoreModel for a profile name ('off'/'l2' → None).

    Name syntax: <profile>[@nu<x>][@g<y>] — @nu overrides ν (carbon
    sensitivity), @g the gate exponent g (gate mode only)."""
    parts = split_min(name)[0].split("@")
    base, nu, gamma = parts[0], None, None
    for p in parts[1:]:
        if p.startswith("nu"):
            nu = p[2:]
        elif p.startswith("g"):
            gamma = p[1:]
        else:
            raise SystemExit(f"profile '{name}': unknown suffix '@{p}' (use @nu<x>, @g<y>)")
    if base in ("off", "l2"):
        return None
    models = (weights_raw or {}).get("score_models") or {}
    if base in models:
        raw = dict(models[base])
    elif base in CANDIDATES:
        raw = {"demand_mode": "gate", "gate": GATE, "demand_floor": False,
               "cold_start": "structural", **CANDIDATES[base]}
    else:
        raise SystemExit(f"unknown profile '{base}' (score models: {sorted(models)}, "
                         f"candidates: {sorted(CANDIDATES)}, baselines: off, l2)")
    if nu:
        raw = {**raw, "parameters": {**(raw.get("parameters") or {}), "nu": float(nu)}}
    if gamma is not None:
        raw = {**raw, "gate": {**(raw.get("gate") or GATE), "gamma": float(gamma)}}
    raw.setdefault("weights", {})
    return ScoreModel.from_dict(name, raw)


def score_functions(fleet: Fleet, model: ScoreModel) -> ScoreFunctions:
    """As DMOSScheduler.__init__ builds them for a score model."""
    net = fleet.cfg.network_params
    lp = fleet.cfg.load_params
    params = ScoreParameters(rho=net.get("rho", 2.0), rtt_max_ms=net.get("rtt_max_ms", 500.0),
                             load_threshold=lp.get("threshold", 0.7),
                             load_steepness=lp.get("steepness", 10),
                             load_floor=lp.get("floor", 0.2))
    if model.parameters:
        params = dataclasses.replace(params, **model.parameters)
    return ScoreFunctions(weights={}, parameters=params, model=model)


def _req(value: str, unit_gb: bool = False) -> float:
    v = str(value)
    if unit_gb:
        if v.endswith("Mi"):
            return float(v[:-2]) / 1024.0
        if v.endswith("Gi"):
            return float(v[:-2])
        return float(v)
    return float(v[:-1]) / 1000.0 if v.endswith("m") else float(v)


def simulate(fleet: Fleet, profile: str, regime: str, carbon: CarbonClient,
             slack_mode: str = "worker", demand_factor: float = 1.0,
             min_replicas: Optional[int] = None, dataplane: str = "pool",
             netem: bool = True, budget: str = "global") -> dict:
    """Steady-state Level-1 placement over the peak window.

    dataplane 'pool': true distribution (served share = replica share);
    'local': each cluster serves the demand entering it (earlier runs)."""
    cfg = fleet.cfg
    entry = cfg.application.entry_service
    fe = cfg.get_service(entry)
    d, peak, label = demand_vector(fleet, regime)
    lam_in = {n: d[n] * peak for n in fleet.names}
    base_profile, suffix_min = split_min(profile)
    if min_replicas is None:
        min_replicas = suffix_min
    nmin = {n: (fe.get_min_replicas_for_cluster(n) if min_replicas is None else min_replicas)
            for n in fleet.names}
    nmax = {n: fe.get_max_replicas_for_cluster(n) for n in fleet.names}
    cap = {n: fe.get_capacity_for_cluster(n) for n in fleet.names}
    cpu_req, mem_req = _req(fe.cpu_request), _req(fe.memory_request, unit_gb=True)

    # budget N (dmos_main.schedule_service): 'cluster' clips each cluster's
    # need at its own maximum (current code); 'global' clips only the total
    # (the excess of a full cluster stays in the budget and goes elsewhere)
    need = {n: max(nmin[n], math.ceil(lam_in[n] * demand_factor / cap[n])) for n in fleet.names}
    if budget == "global":
        N = min(sum(nmax.values()), sum(need.values()))
    else:
        N = sum(min(nmax[n], need[n]) for n in fleet.names)

    model = build_model(profile, cfg.weights_raw)
    sf = score_functions(fleet, model) if model else None
    wd = WinnerDetermination()
    rtt_peer = {i: statistics.fmean(fleet.delay[i] + fleet.delay[j] for j in fleet.names if j != i)
                for i in fleet.names}

    def uniform(total: int) -> Dict[str, int]:
        K = len(fleet.names)
        base, rem = divmod(total, K)
        return {n: base + (1 if k < rem else 0) for k, n in enumerate(sorted(fleet.names))}

    reps = {n: 1 for n in fleet.names} if base_profile == "off" else uniform(N)
    if base_profile == "l2":
        reps = {n: max(nmin[n], min(nmax[n], r)) for n, r in uniform(N).items()}

    rows = []
    t = float(PEAK_WINDOW[0]) - 4 * CYCLE_S          # a few warm-up cycles before the window
    while t < PEAK_WINDOW[1]:
        R = sum(reps.values())
        share = ({n: reps[n] / R for n in fleet.names} if dataplane == "pool"
                 else dict(d))                       # share of the requests each cluster serves
        served = {n: peak * share[n] for n in fleet.names}
        busy = {n: NODE_IDLE_CORES.get(n, 0.3) + NODE_CORES_PER_RPS.get(n, 0.014) * served[n]
                for n in fleet.names}
        ci = {n: carbon.trace_value_at(fleet.region[n], t) or 300.0 for n in fleet.names}
        if t >= PEAK_WINDOW[0]:
            row = _cycle_metrics(fleet, d, reps, share, busy, ci, lam_in, dataplane == "local")
            row.update(latency_metrics(fleet, d, peak, reps, dataplane, netem,
                                       samples=20000, seed=int(t)))
            rows.append(row)
        if sf is not None:
            bids = []
            for n in fleet.names:
                cores = fleet.cores[n]
                u = min(0.95, busy[n] / cores)
                if slack_mode == "measured":
                    cpu_avail = cores + CONTROL_PLANE_CORES - busy[n]
                    mem_avail = fleet.mem[n] * MEASURED_MEM_SLACK
                else:
                    cpu_avail = max(0.0, cores - busy[n])
                    mem_avail = fleet.mem[n] * MEASURED_MEM_SLACK
                m = ClusterMetrics(
                    cpu_available_cores=cpu_avail, cpu_total_cores=cores,
                    memory_available_gb=mem_avail, memory_total_gb=fleet.mem[n],
                    request_rate_current=served[n], request_rate_max=cores * cap[n],
                    latency_mean_ms=L0_MS / (1.0 - u), latency_variance_ms2=0.0,
                    carbon_intensity_gco2_kwh=ci[n], cost_per_replica_hour=0.1,
                    network_rtt_ms=rtt_peer[n], ingress_rate_rps=lam_in[n],
                    ingress_demand_share=d[n])
                s = sf.compute_score_breakdown(m, predicted_load=peak)["total_score"]
                bid_cap = min(int(cpu_avail / cpu_req) if cpu_req > 0 else nmax[n],
                              int(mem_avail / mem_req) if mem_req > 0 else nmax[n], nmax[n])
                bid_cap = min(max(0, bid_cap), nmax[n])
                if s > 0 and bid_cap > 0:
                    bids.append(ClusterBid(cluster_name=n, score=s, capacity=bid_cap))
            allocs, ok = wd.allocate(bids, N)
            got = {a.cluster_name: a.replicas for a in allocs}
            reps = {n: max(nmin[n], min(nmax[n], got.get(n, 0))) for n in fleet.names}
        t += CYCLE_S

    out = {"profile": profile, "regime": label, "peak": peak, "N": N, "slack": slack_mode,
           "min_replicas": "config" if min_replicas is None else min_replicas,
           "dataplane": dataplane, "netem": netem, "budget": budget}
    keys = rows[0].keys()
    for k in keys:
        vals = [r[k] for r in rows]
        out[k] = statistics.fmean(vals) if isinstance(vals[0], (int, float)) else vals[-1]
    return out


def _cycle_metrics(fleet: Fleet, d, reps, share, busy, ci, lam_in, local: bool = False) -> dict:
    cfg = fleet.cfg
    entry = cfg.application.entry_service
    fe = cfg.get_service(entry)
    # backend replicas: the co-scaling rule of dmos_main._enforce_colocation
    floor = 0.3
    backends = 0
    for b in cfg.application.co_scaled_services:
        bc = cfg.get_service(b)
        if bc is None:
            continue
        for n in fleet.names:
            ratio = max(fe.get_capacity_for_cluster(n) / max(1e-9, bc.get_capacity_for_cluster(n))
                        * (bc.calls_per_request or 1.0), floor)
            backends += min(bc.get_max_replicas_for_cluster(n),
                            max(bc.get_min_replicas_for_cluster(n), math.ceil(reps[n] * ratio)))
    # requests entering k and served by i ≠ k pay one extra round trip d_k + d_i
    if local:                                    # served where it enters
        extra, remote = 0.0, 0.0
    else:
        extra = sum(d[k] * share[i] * (fleet.delay[k] + fleet.delay[i])
                    for k in fleet.names for i in fleet.names if i != k)
        remote = 1.0 - sum(d[k] * share[k] for k in fleet.names)
    row = {
        "R_frontend": sum(reps.values()),
        "R_backends": backends,
        "ci_served": sum(share[n] * ci[n] for n in fleet.names),
        "ci_if_local": sum(d[n] * ci[n] for n in fleet.names),
        "extra_rtt_ms": extra,
        "remote_share": remote,
        "node_cores_busy": sum(busy.values()),
        "max_node_util": max(busy[n] / fleet.cores[n] for n in fleet.names),
    }
    for n, s in zip(fleet.names, fleet.shorts):
        row[f"r_{s}"] = reps[n]
    return row


def default_config_dir(local_yaml: Optional[Path]) -> Path:
    """config/ of the repo, with local.yaml from --local-yaml when given."""
    src = REPO / "config"
    if local_yaml is None and (src / "local.yaml").is_file():
        return src
    tmp = Path(tempfile.mkdtemp(prefix="dmos_sim_cfg_"))
    for name in ("weights.yaml", "services.yaml", "carbon.yaml"):
        shutil.copy(src / name, tmp / name)
    shutil.copy(local_yaml or (src / "local.yaml.example"), tmp / "local.yaml")
    return tmp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--profiles", default="off l2 nca campaign latency_first carbon_first "
                                          "perf equal green perf@nu2 equal@nu2 green@nu2 "
                                          "only-net only-car@nu2")
    ap.add_argument("--regimes", default="uni bal con bal-sg con-sg bal-pl con-pl")
    ap.add_argument("--peaks", default="350", help='e.g. "350 500 700": overrides the peak of every regime')
    ap.add_argument("--slack", default="worker", choices=["measured", "worker"],
                    help="free resources as DMOS measures them: worker (default, "
                         "DMOS_FREE_RESOURCES=worker) or measured = cluster-wide (legacy)")
    ap.add_argument("--dataplane", default="pool", choices=["pool", "local"],
                    help="pool: true distribution (served share = replica share); "
                         "local: each cluster serves its own demand (the earlier runs)")
    ap.add_argument("--no-netem", action="store_true", help="no inter-cluster delay")
    ap.add_argument("--budget", default="global", choices=["cluster", "global"],
                    help="global (default, DMOS_ENTRY_BUDGET=total): only the total is capped; "
                         "cluster: each cluster's need capped at its maximum (legacy)")
    ap.add_argument("--min-replicas", type=int, default=None,
                    help="override the entry service's per-cluster minimum (e.g. 0: no replica "
                         "where there is no demand); backends keep their own minimum")
    ap.add_argument("--demand-factor", type=float, default=1.05,
                    help="measured ingress / offered load (Hubble counts slightly more than "
                         "k6 offers: with 1.05 the uniform regime gets the 12 frontend replicas "
                         "the earlier runs had); default 1.05")
    ap.add_argument("--local-yaml", type=Path, default=None)
    ap.add_argument("--trace-dir", type=Path, default=None)
    ap.add_argument("--window-start", default=None,
                    help="start of the carbon-trace replay (default: trace.window_start of "
                         "carbon.yaml), e.g. 2025-01-15T00:00:00 for a winter day")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true", help="only the CSV")
    args = ap.parse_args(argv)

    _loguru.remove()                                   # DMOS logs every allocation step
    fleet = load_fleet(default_config_dir(args.local_yaml))
    ccfg = dict((fleet.cfg.carbon_raw or {}).get("carbon_intensity", {}))
    ccfg["provider"] = "trace"
    trace = dict(ccfg.get("trace") or {})
    if args.trace_dir:
        trace["data_dir"] = str(args.trace_dir.resolve())
    if args.window_start:
        trace["window_start"] = args.window_start
    ccfg["trace"] = trace
    carbon = CarbonClient(ccfg)
    _loguru.remove()

    results = []
    for peak in args.peaks.split():
        for regime in args.regimes.split():
            if regime.startswith("w:"):
                reg = regime.split("@")[0] + f"@{peak}"
            else:                                    # same regime, peak of --peaks
                p = rn.parse_regime(regime)
                reg = str(p["demand"]) + (f"-{p['hot']}" if p["hot"] else "") + (
                    f"-p{peak}" if int(peak) != rn.DEFAULT_PEAK else "")
            for prof in args.profiles.split():
                results.append(simulate(fleet, prof, reg, carbon, args.slack, args.demand_factor,
                                        args.min_replicas, args.dataplane, not args.no_netem,
                                        args.budget))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    if not args.quiet:
        cols = [f"r_{x}" for x in fleet.shorts]
        last = None
        for r in results:
            key = (r["regime"], r["peak"])
            if key != last:
                print(f"\n{r['regime']}  (peak {r['peak']} req/s, N={r['N']}, data plane {r['dataplane']}, "
                      f"netem {'on' if r['netem'] else 'off'}, min={r['min_replicas']}; "
                      f"CI if served where it enters {r['ci_if_local']:.0f} g)")
                print(f"  {'profile':16s}" + "".join(f"{c[2:]:>5s}" for c in cols)
                      + f"{'R':>5s}{'pods':>6s}{'CI':>6s}{'net':>6s}{'p95':>7s}{'SLO%':>7s}{'err%':>6s}")
                last = key
            print(f"  {r['profile']:16s}" + "".join(f"{r[c]:5.1f}" for c in cols)
                  + f"{r['R_frontend']:5.1f}{r['R_frontend'] + r['R_backends']:6.0f}{r['ci_served']:6.0f}"
                  + f"{r['net_mean_ms']:6.0f}{r['p95_ms']:7.0f}{100 * r['slo_viol']:7.1f}{100 * r['err']:6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
