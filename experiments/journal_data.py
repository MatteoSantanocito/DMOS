"""journal_data.py — runs, per-run metrics and time series for the journal figures.

A dataset is a list of runs labelled (arm, regime). It comes either from one or
more campaign TSVs (experiments/results/_campaign_<name>.tsv, status OK rows
only) or from PROXY: pilot runs (29/10–1/11/2026) used to design the figures
before the definitive campaign ends. Proxy figures are watermarked.

Per run:
  metrics  scalars (k6 latency/errors, replica-minutes, CPU, CO2, served CI)
  ts       DataFrame on a 5 s grid, 0–900 s: per-region frontend ready/target
           replicas, served req/s, ingress->frontend p95 (Hubble), predictive
           scaler signals, CPU, carbon intensity CI_i(t)
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_sustainability as ps  # noqa: E402

CL = [f"ms-01-cluster-0{i}" for i in range(1, 7)]
REGION = dict(zip(CL, ["DE", "FR", "PL", "US", "SG", "NO"]))
CAP_RPS = 33.0          # frontend capacity per replica (config/local.yaml)
CAPS = np.array([4, 3, 2, 4, 2, 3])   # frontend max_replicas per cluster, order of CL (config/local.yaml)
PHASES = [("baseline", 0, 180), ("spike", 180, 300), ("peak", 300, 600),
          ("decay", 600, 720), ("cooldown", 720, 900)]
GRID = np.arange(0.0, 900.0 + 1e-9, 5.0)

# Pilot runs standing in for the campaign while it runs (v4 scaler for con;
# bal / con-tosg DMOS runs are v3, HPA has n = 1 there).
PROXY: Dict[tuple, List[str]] = {
    ("off", "con"): ["20261031_192926", "20261031_234808"],
    ("hpa-none", "con"): ["20261101_020245", "20261101_032450"],
    ("l2-pool-ps", "con"): ["20261101_145051", "20261101_155128"],
    ("locality-ps", "con"): ["20261031_204832", "20261031_214855", "20261031_224826"],
    ("nca-pool-ps", "con"): ["20261031_202847", "20261031_212908", "20261031_222835"],
    ("reach-pool-ps", "con"): ["20261031_194901", "20261031_220843", "20261031_230818"],
    ("reach-pool-ps-min0", "con"): ["20261031_200852", "20261031_210820", "20261031_232806",
                                    "20261101_153034"],
    ("reach_green-pool-ps-min0", "con"): ["20261101_151041", "20261101_161119"],
    ("off", "bal"): ["20261030_115510", "20261030_135235", "20261030_155043"],
    ("hpa-none", "bal"): ["20261101_022318"],
    ("reach-pool-ps", "bal"): ["20261030_200928", "20261030_204845", "20261030_212755"],
    ("off", "con-tosg"): ["20261029_182252", "20261030_174818"],
    ("hpa-none", "con-tosg"): ["20261101_030421"],
    ("reach-pool-ps", "con-tosg"): ["20261030_214736"],
}

NAME_RE = re.compile(r"^(\d{8}_\d{6})_(?:dmos-)?(.+?)__(.+)$")


def _f(x) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_profile(t, base: float, peak: float, ph=(180, 300, 600, 720)) -> np.ndarray:
    """Offered load (req/s) of the flash-crowd profile."""
    return np.interp(np.asarray(t, float), [0, ph[0], ph[1], ph[2], ph[3], 900],
                     [base, base, peak, peak, base, base])


def entry_weights(cfg: dict, t: float) -> np.ndarray:
    """Share of the load entering each region at time t (k6_flash_crowd.js pickIngress)."""
    p = cfg["phase_times_s"]
    if not (p[0] <= t < p[2]):
        w = np.asarray(cfg["baseline_weights"], float)
    else:
        w = np.asarray(cfg["spike_weights"], float)
        sh = cfg.get("shift") or {}
        if sh.get("enabled") and t >= p[1]:
            f = np.clip((t - sh["at_s"]) / max(1e-9, sh["ramp_s"]), 0, 1)
            w = (1 - f) * w + f * np.asarray(sh["weights"], float)
    return w / w.sum()


def min_fleet(arm: str) -> int:
    """Frontend replicas the arm keeps even without load: 1 per cluster, none with -min0."""
    return 1 if "min0" in arm else len(CL)


def best_split(n: int, rbar: np.ndarray, floor: int) -> np.ndarray:
    """Replicas per cluster minimising the expected RTT sum_i n_i rbar_i for a
    budget n, with the caps and a floor per cluster (greedy, exact: Prop. 1)."""
    x = np.minimum(np.full(len(CAPS), floor), CAPS)
    left = max(0, n - int(x.sum()))
    for i in np.argsort(rbar, kind="stable"):
        add = min(left, CAPS[i] - x[i])
        x[i] += add
        left -= add
    return x


def even_split(n: int, floor: int) -> np.ndarray:
    """The l2 arm's split: one replica at a time to the cluster with the fewest, within the caps."""
    x = np.minimum(np.full(len(CAPS), floor), CAPS)
    for _ in range(max(0, n - int(x.sum()))):
        free = np.where(x < CAPS)[0]
        if len(free) == 0:
            break
        x[free[np.argmin(x[free])]] += 1
    return x


@dataclass
class Run:
    path: Path
    arm: str
    regime: str
    block: str = ""

    @property
    def name(self) -> str:
        return self.path.name

    @cached_property
    def summary(self) -> dict:
        return json.loads((self.path / "flash_crowd_summary.json").read_text(encoding="utf-8"))

    @cached_property
    def manifest(self) -> dict:
        p = self.path / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    @cached_property
    def snaps(self) -> List[dict]:
        return ps.load_snapshots(self.path / "dmos_live.jsonl")

    @property
    def config_id(self) -> Optional[str]:
        return (self.manifest.get("dmos") or {}).get("config_id")

    @property
    def start(self) -> pd.Timestamp:
        return pd.to_datetime(self.name[:15], format="%Y%m%d_%H%M%S")


# ── dataset ─────────────────────────────────────────────────────────────────

def _resolve(results: Path, prefix: str) -> Path:
    hits = sorted(results.glob(prefix + "_*"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{prefix}: {len(hits)} matching runs in {results}")
    return hits[0]


def proxy_runs(results: Path) -> List[Run]:
    return [Run(_resolve(results, p), arm, reg, "proxy")
            for (arm, reg), prefixes in PROXY.items() for p in prefixes]


def campaign_runs(results: Path, tsvs: Iterable[Path]) -> List[Run]:
    """OK rows of campaign TSVs; arm, regime and block come from the TSV."""
    out = []
    for tsv in tsvs:
        with open(tsv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if (row.get("status") or "").strip() != "OK" or not (row.get("run") or "").strip():
                    continue
                d = results / row["run"].strip()
                if (d / "flash_crowd_summary.json").is_file():
                    out.append(Run(d, row["arm"].strip(), row["regime"].strip(), row["block"].strip()))
                else:
                    print(f"[journal] missing run folder {d}", file=sys.stderr)
    return out


# ── per-run metrics and time series ────────────────────────────────────────

class Context:
    """Cluster power/carbon model and CI replay.

    The cluster model is the one the runs used (config_snapshot/local.sanitized.yaml
    of the first run that has it), so the archived dataset is self-contained; the
    repo's config/local.yaml is the fallback. The Electricity Maps traces are not
    redistributable: they must be in <repo>/data/carbon_traces (see its README),
    and a missing trace is an error instead of a silent fallback profile."""

    def __init__(self, repo: Path, runs: Iterable[Run] = ()):
        snaps = (r.path / "config_snapshot" / "local.sanitized.yaml" for r in runs)
        local = next((p for p in snaps if p.is_file()), repo / "config" / "local.yaml")
        self.clusters = ps.load_clusters(local)
        self.replay = ps.CarbonReplay(ps.default_carbon_config(repo / "config" / "carbon.yaml"),
                                      trace_dir=repo / "data" / "carbon_traces")
        missing = sorted({c.region for c in self.clusters.values()
                          if not self.replay.source(c.region).startswith("trace:")})
        if missing:
            raise FileNotFoundError(
                f"no Electricity Maps trace for {missing} in {repo / 'data' / 'carbon_traces'}: "
                "download them as described in data/carbon_traces/README.md")

    def ci(self, run: Run, t: np.ndarray) -> Dict[str, np.ndarray]:
        p = run.path / "dmos_config.json"
        cfg = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        trace_cfg = ((cfg.get("carbon") or {}).get("trace")) or None
        return {c: np.array([_f(self.replay.ci(self.clusters[c].region, x, trace_cfg)) for x in t])
                for c in CL}


def rtt_matrix(run: Run) -> np.ndarray:
    """Median RTT (ms) between the clusters, from the pings recorded in the run;
    0 on the diagonal. A cluster does not ping itself: its address is the one
    missing from its own ping list."""
    rtt, ips = {c: {} for c in CL}, set()
    for s in run.snaps:
        for c, cc in (s.get("clusters") or {}).items():
            for ip, v in ((cc or {}).get("ping_rtt_ms") or {}).items():
                ips.add(ip)
                if v is not None and c in rtt:
                    rtt[c].setdefault(ip, []).append(v)
    own = {}
    for c in CL:
        miss = ips - set(rtt[c])
        if len(miss) == 1:
            own[miss.pop()] = c
    M = np.full((6, 6), np.nan)
    np.fill_diagonal(M, 0.0)
    for i, c in enumerate(CL):
        for ip, v in rtt[c].items():
            if ip in own:
                M[i, CL.index(own[ip])] = float(np.median(v))
    return M


def timeseries(run: Run, ctx: Context) -> pd.DataFrame:
    """Snapshot signals on GRID (5 s): replica counts as steps (last sample),
    rates and latencies by linear interpolation."""
    rows = []
    for s in run.snaps:
        r = {"t": _f(s.get("elapsed_s"))}
        svcs = s.get("services") or {}
        fe = (svcs.get("frontend") or {}).get("clusters") or {}
        cls = s.get("clusters") or {}
        psc = s.get("predictive_scaler") or {}
        r["pods_all"] = sum(_f(cc.get("ready_pods")) for sv in svcs.values()
                            for cc in (sv.get("clusters") or {}).values() if cc.get("ready_pods") is not None)
        for c in CL:
            g = REGION[c]
            fc, cc = fe.get(c) or {}, cls.get(c) or {}
            # the collector omits ready_pods when a cluster has no ready frontend (min0 arms)
            r[f"ready_{g}"] = _f(fc.get("ready_pods")) if fc.get("ready_pods") is not None else (0.0 if fc else np.nan)
            r[f"target_{g}"] = _f(fc.get("target_replicas"))
            r[f"served_{g}"] = _f(cc.get("frontend_net_bytes_s"))     # frontend req/s (historical name)
            r[f"p95in_{g}"] = _f((cc.get("l7_edge_p95_ms") or {}).get("ingress->frontend"))
            r[f"cpu_{g}"] = _f(cc.get("app_cpu_cores"))
            for k in ("fast_rps", "forecast_rps", "demand_rps", "need_replicas"):
                r[f"{k}_{g}"] = _f((psc.get(k) or {}).get(c))
        r["dmos_cpu"] = _f((s.get("dmos_process") or {}).get("cpu_cores"))
        rows.append(r)
    raw = pd.DataFrame(rows).dropna(subset=["t"]).sort_values("t").drop_duplicates("t")
    cols = {"t": GRID}
    for col in raw.columns:
        if col == "t":
            continue
        x, y = raw["t"].to_numpy(), raw[col].to_numpy(float)
        ok = ~np.isnan(y)
        if ok.sum() < 2:
            cols[col] = np.full(len(GRID), np.nan)
        elif col.startswith(("ready_", "target_", "need_replicas_")) or col == "pods_all":
            idx = np.searchsorted(x[ok], GRID, side="right") - 1
            cols[col] = y[ok][np.clip(idx, 0, None)]
        else:
            cols[col] = np.interp(GRID, x[ok], y[ok], left=np.nan, right=np.nan)
    out = pd.DataFrame(cols)
    cfg = run.summary["config"]
    out["load"] = load_profile(GRID, cfg["baseline_rps"], cfg["peak_rps"], tuple(cfg["phase_times_s"][:4]))
    # HTTP requests at the frontends: k6 follows the redirects of the POST endpoints, so the
    # HTTP rate is ~1.2x the nominal profile; the per-run factor comes from k6's own request count
    nominal = float(np.sum(out["load"].to_numpy()[:-1]) * 5.0)
    out["load_http"] = out["load"] * (run.summary["results"]["global"]["total_requests"] / nominal)
    out["need"] = np.maximum(min_fleet(run.arm), np.ceil(out["load_http"] / CAP_RPS))
    for k in ("fast_rps", "forecast_rps", "demand_rps", "target", "ready"):
        cs = [f"{k}_{REGION[c]}" for c in CL]
        out[f"{k}_total"] = out[cs].sum(axis=1, min_count=1)
    # Expected network RTT between a user's entry region and the frontend that
    # serves it (the model J of eq. J): with the pooled frontend (affinity: none) a
    # request goes to any ready replica, so the serving region follows the replica shares.
    R = rtt_matrix(run)
    off = ~np.eye(6, dtype=bool)
    R = np.where(np.isnan(R), np.nanmean(R[off]), R)
    ready = out[[f"ready_{REGION[c]}" for c in CL]].fillna(0).to_numpy()
    W = np.array([entry_weights(cfg, x) for x in GRID])
    rbar = W @ R                                    # demand-weighted RTT to each cluster
    floor = 0 if "min0" in run.arm else 1

    def j(x):
        return (x * rbar).sum(1) / np.maximum(x.sum(1), 1e-9)
    n = np.rint(ready.sum(1)).astype(int)
    out["exp_rtt"] = j(ready)
    out["exp_rtt_opt"] = j(np.array([best_split(k, r, floor) for k, r in zip(n, rbar)], float))
    out["exp_rtt_even"] = j(np.array([even_split(k, floor) for k in n], float))
    for c, v in ctx.ci(run, GRID).items():
        out[f"ci_{REGION[c]}"] = v
    return out


def pods_started(run: Run) -> float:
    """Pods that became ready during the run, all services and clusters (sum of the
    rises of each ready count; a missing count is a deployment with none ready)."""
    prev: Dict[tuple, float] = {}
    started = 0.0
    for s in run.snaps:
        for svc, sv in (s.get("services") or {}).items():
            for c, cc in (sv.get("clusters") or {}).items():
                v = _f(cc.get("ready_pods")) if cc.get("ready_pods") is not None else 0.0
                started += max(0.0, v - prev.get((svc, c), v))
                prev[(svc, c)] = v
    return started


def ci_bounds(ts: pd.DataFrame, served: np.ndarray, floor: int):
    """Cleanest and dirtiest carbon intensity of the served requests attainable with the
    same ready replicas at every step, the same caps and floor (pool routing: requests
    follow the replica shares). Weighted by the requests served at each step."""
    ci = ts[[f"ci_{r}" for r in REGION.values()]].to_numpy(float)
    n = np.rint(ts["ready_total"].fillna(0).to_numpy()).astype(int)
    w = served.sum(1)
    best = worst = tot = 0.0
    for k in range(len(ts)):
        if n[k] <= 0 or w[k] <= 0 or np.isnan(ci[k]).any():
            continue
        lo, hi = best_split(n[k], ci[k], floor), best_split(n[k], -ci[k], floor)   # cleanest / dirtiest first
        best += w[k] * (lo * ci[k]).sum() / max(lo.sum(), 1)
        worst += w[k] * (hi * ci[k]).sum() / max(hi.sum(), 1)
        tot += w[k]
    return (best / tot, worst / tot) if tot > 0 else (float("nan"), float("nan"))


def metrics(run: Run, ts: pd.DataFrame, ctx: Context) -> dict:
    s = run.summary["results"]
    g, ph, pc = s["global"], s["per_phase"], s["per_cluster"]
    req = g["total_requests"]
    m = {"run": run.name, "start": str(run.start), "config_id": run.config_id,
         "p50": g["p50_ms"], "p95": g["p95_ms"], "p99": g["p99_ms"], "mean_ms": g["avg_ms"],
         "err_pct": 100 * g["http_failure_rate"], "err_n": round(g["http_failure_rate"] * req),
         "requests": req}
    for name, *_ in PHASES:
        d = ph.get(name) or {}
        m[f"p50_{name}"], m[f"p95_{name}"], m[f"p99_{name}"] = d.get("p50"), d.get("p95"), d.get("p99")
        m[f"err_{name}"] = round((d.get("errorRate") or 0) * (d.get("requests") or 0))
    for c in CL:
        for q in ("p50", "p95", "p99"):
            m[f"{q}_{REGION[c]}"] = (pc.get(c) or {}).get(q)
    for w, d in (s.get("per_shift_window") or {}).items():
        m[f"p95_shift_{w}"] = d.get("p95")
    dt = 5.0 / 60.0
    m["peak_fe"] = float(np.nanmax(ts["ready_total"]))
    rt = ts.set_index("t")["ready_total"]
    m["decay_added"] = float(rt.loc[600:720].max() - rt.loc[600])   # replicas added while the load falls
    ts = ts.iloc[:-1]            # 180 intervals of 5 s = 0–900 s (left Riemann sum of steps)
    m["fe_rm"] = float(np.nansum(ts["ready_total"]) * dt)
    m["all_rm"] = float(np.nansum(ts["pods_all"]) * dt)
    m["excess_rm"] = float(np.nansum(np.maximum(0, ts["ready_total"] - ts["need"])) * dt)
    m["deficit_rm"] = float(np.nansum(np.maximum(0, ts["need"] - ts["ready_total"])) * dt)
    m["sg_zero_s"] = float((ts["ready_SG"] == 0).sum() * 5.0)
    # seconds after the spike starts until the frontend can carry the HTTP peak (ready x cap >= peak)
    peak = float(ts["load_http"].max())
    ok = ts[(ts.t >= 180) & (ts.ready_total * CAP_RPS >= peak)]
    m["t_capacity"] = float(ok.t.iloc[0] - 180) if len(ok) else float("nan")
    for k in ("exp_rtt", "exp_rtt_opt", "exp_rtt_even"):                   # per request
        m[f"{k}_ms"] = float(np.nansum(ts[k] * ts.load) / np.nansum(ts.load))
        pk = ts[(ts.t >= 300) & (ts.t < 600)]
        m[f"{k}_peak_ms"] = float(np.nanmean(pk[k]))
    served = ts[[f"served_{r}" for r in REGION.values()]].clip(lower=0).fillna(0).to_numpy()
    tot = served.sum()
    for j, r in enumerate(REGION.values()):
        m[f"served_share_{r}"] = float(served[:, j].sum() / tot) if tot > 0 else float("nan")
        m[f"rm_{r}"] = float(np.nansum(ts[f"ready_{r}"]) * dt)
    # Assumption 1 (pool routing): served share against the request-weighted replica share
    rs = ts[[f"ready_{r}" for r in REGION.values()]].fillna(0).to_numpy()
    rs = rs / np.maximum(rs.sum(1, keepdims=True), 1e-9)
    w = served.sum(1)
    rep_share = (rs * w[:, None]).sum(0) / max(w.sum(), 1e-9)
    for j, r in enumerate(REGION.values()):
        m[f"rep_share_{r}"] = float(rep_share[j])
    m["pool_dev_pp"] = float(100 * np.max(np.abs(served.sum(0) / max(tot, 1e-9) - rep_share))) if tot > 0 \
        else float("nan")
    m["pods_started"] = pods_started(run)
    m["ci_best_g_kwh"], m["ci_worst_g_kwh"] = ci_bounds(ts, served, 0 if "min0" in run.arm else 1)
    sm = ps.run_metrics(run.path, ctx.clusters, ctx.replay, "run") or {}
    for k in ("core_s", "core_s_per_1k_ok", "co2_dyn_g", "co2_dyn_mg_per_ok", "co2_node_g",
              "ci_served_g_kwh", "ci_compute_g_kwh", "energy_dyn_wh", "dmos_cpu_cores_mean",
              "dmos_cpu_cores_max", "dmos_rss_mb_max", "dmos_cycle_ms"):
        m[k] = sm.get(k)
    m["co2_ug_per_req"] = m["co2_dyn_mg_per_ok"] * 1000 if m["co2_dyn_mg_per_ok"] is not None else None
    return m


@dataclass
class Dataset:
    runs: List[Run]
    ctx: Context
    cache: Optional[Path] = None
    proxy: bool = False
    _ts: Dict[str, pd.DataFrame] = field(default_factory=dict)
    _m: Dict[str, dict] = field(default_factory=dict)

    def ts(self, run: Run) -> pd.DataFrame:
        if run.name not in self._ts:
            p = self.cache / f"{run.name}.ts.pkl" if self.cache else None
            if p and p.is_file():
                self._ts[run.name] = pd.read_pickle(p)
            else:
                self._ts[run.name] = timeseries(run, self.ctx)
                if p:
                    self._ts[run.name].to_pickle(p)
        return self._ts[run.name]

    def m(self, run: Run) -> dict:
        if run.name not in self._m:
            p = self.cache / f"{run.name}.m.json" if self.cache else None
            if p and p.is_file():
                self._m[run.name] = json.loads(p.read_text())
            else:
                self._m[run.name] = metrics(run, self.ts(run), self.ctx)
                if p:
                    p.write_text(json.dumps(self._m[run.name], default=str))
            self._m[run.name].update(arm=run.arm, regime=run.regime, block=run.block)
        return self._m[run.name]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([self.m(r) for r in self.runs])

    def group(self, arm: str, regime: str) -> List[Run]:
        return [r for r in self.runs if r.arm == arm and r.regime == regime]


def load(results: Path, tsvs: List[Path], proxy: bool, cache: Optional[Path]) -> Dataset:
    runs = proxy_runs(results) if proxy else campaign_runs(results, tsvs)
    if cache:
        # one cache per version of the metric code, so an edit never reuses stale numbers
        src = b"".join(Path(m.__file__).read_bytes() for m in (sys.modules[__name__], ps))
        cache = cache / hashlib.sha1(src).hexdigest()[:10]
        cache.mkdir(parents=True, exist_ok=True)
    return Dataset(runs, Context(HERE.parent, runs), cache, proxy)


# ── statistics ──────────────────────────────────────────────────────────────

def boot_mean_ci(x, n=10000, seed=0, level=0.95):
    """(mean, lo, hi) with a percentile bootstrap CI; no CI below n = 2."""
    x = np.asarray([v for v in x if v is not None and not math.isnan(v)], float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if len(x) == 1:
        return (x[0], np.nan, np.nan)
    bs = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    a = (1 - level) / 2
    return (x.mean(), *np.quantile(bs, [a, 1 - a]))


def boot_ratio_ci(x, y, n=10000, seed=0, level=0.95):
    """% change of mean(x) over mean(y), with a percentile bootstrap CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    est = 100 * (x.mean() / y.mean() - 1)
    if len(x) < 2 or len(y) < 2:
        return (est, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, (n, len(x))).mean(axis=1)
    by = rng.choice(y, (n, len(y))).mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(100 * (bx / by - 1), [a, 1 - a])
    return (est, lo, hi)


if __name__ == "__main__":       # self-check of the pure helpers
    cfg = {"phase_times_s": [180, 300, 600, 720, 900], "baseline_weights": [.3, .2, .15, .1, .1, .15],
           "spike_weights": [.85, .03, .03, .03, .03, .03],
           "shift": {"enabled": True, "at_s": 450, "ramp_s": 30, "weights": [.03, .03, .03, .03, .85, .03]}}
    assert np.allclose(entry_weights(cfg, 100), cfg["baseline_weights"])
    assert np.isclose(entry_weights(cfg, 400)[0], .85) and np.isclose(entry_weights(cfg, 500)[4], .85)
    assert np.isclose(entry_weights(cfg, 465)[0], .44)
    assert np.allclose(load_profile([0, 240, 450, 660, 800], 50, 350), [50, 200, 350, 200, 50])
    e, lo, hi = boot_ratio_ci([90, 92, 91], [100, 101, 99])
    assert -10.5 < lo < e < hi < -7.5, (e, lo, hi)
    W, R, S = np.array([[.5, .5]]), np.array([[0, 10], [10, 0]]), np.array([[1., 0.]])
    assert np.isclose(np.einsum("te,er,tr->t", W, R, S)[0], 5.0)   # half the users are 10 ms away
    assert NAME_RE.match("20261101_145051_dmos-l2-pool-ps__con").groups()[1:] == ("l2-pool-ps", "con")
    rb = np.array([12.8, 41.2, 59.2, 104.0, 202.1, 48.2])          # Section 3 example, N = 12, min 0
    assert best_split(12, rb, 0).tolist() == [4, 3, 2, 0, 0, 3]
    assert best_split(16, rb, 1).tolist() == [4, 3, 2, 3, 1, 3]
    assert even_split(12, 0).tolist() == [2, 2, 2, 2, 2, 2] and even_split(17, 1).sum() == 17
    assert min_fleet("reach-pool-ps-min0") == 1 and min_fleet("hpa-none") == 6
    tsb = pd.DataFrame({**{f"ci_{r}": [c] for r, c in zip(REGION.values(), [300, 20, 500, 350, 480, 25])},
                        "ready_total": [6]})
    b, w = ci_bounds(tsb, np.ones((1, 6)), 0)          # 6 replicas: cleanest FR 3 + NO 3; dirtiest PL 2 + SG 2 + US 2
    assert np.isclose(b, 22.5) and np.isclose(w, (2 * 500 + 2 * 480 + 2 * 350) / 6), (b, w)
    print("journal_data self-check OK")
