#!/usr/bin/env python3
"""
scaler_replay.py — offline replay of DMOS Level-2 scalers on a flash crowd.

Simulates, at 1 s resolution, the entering demand of a k6 flash-crowd run,
the way DMOS measures it and the time new replicas take to serve, and
compares two entry-service scalers on the SAME demand:

  reactive    the reactive Level 2 as it runs on the clusters (dmos_main.py,
              DMOS_SCALER=reactive): decisions every 30 s on rate[1m]; budget
              = sum ceil(demand_i / r); per-cluster trend predictor and PD +1
              outside the 0.2 deadband; +6 / -1 replica per cluster per cycle;
              scale-down cooldown 90 s and scale-up protection 90 s.
              A model of the decision rules, not the code itself
              (docs/PREDICTIVE-SCALER.md, §6, compares it with the cluster run).
  predictive  src.level2.predictive_scaler.PredictiveScaler: the class that
              runs on the clusters, driven by the simulated measurements.

Measurement model (Hubble metrics scraped every 15 s by Prometheus): a query
at time t sees the counter up to the last scrape s <= t; rate[w] is the mean
demand over [s - w, s]. Replicas requested at t serve from t + T_ready.
Level 1 is not simulated: the budget is spread over the clusters by
water-filling within the per-cluster caps, the same for both scalers (with
the frontend as a global service the total ready capacity serves the fleet).

Metrics (see Result.metrics): requests above the ready capacity at the target
CPU and at the CPU limit, time to the peak capacity, scaling decisions and
replicas added under load (each one changes the endpoints of the global
service on every cluster while traffic is high), replica-minutes.

  python experiments/scaler_replay.py                      # con @350, 6 clusters
  python experiments/scaler_replay.py --peak 500 --t-ready 35 --csv out.csv
  python experiments/scaler_replay.py --sweep              # sensitivity of the PS knobs
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.level2.predictive_scaler import PredictiveScaler, PredictiveScalerParams  # noqa: E402

# k6_flash_crowd.js phase ends (s): baseline | spike | peak | decay | cooldown
PHASES = (180, 300, 600, 720, 900)
REGIME_WEIGHTS = {                           # spike/peak weights of run_naming.py
    "uni": (1, 1, 1, 1, 1, 1), "bal": (40, 15, 12, 12, 11, 10), "s50": (50, 10, 10, 10, 10, 10),
    "s65": (65, 7, 7, 7, 7, 7), "s75": (75, 5, 5, 5, 5, 5), "con": (85, 3, 3, 3, 3, 3),
}
BASELINE_WEIGHTS = (30, 20, 15, 10, 10, 15)  # k6 baseline weights of the 6 regions
CAPS = (4, 3, 2, 4, 2, 3)                    # replica caps of the paper fleet (u = 0.8)
CAPACITY = 33.0                              # r, req/s per frontend (calibration 2026-10-28)
U_POD = 0.7                                  # r is measured at 70 % of the CPU limit


def offered(t: float, baseline: float, peak: float) -> float:
    """Total offered load of the k6 flash crowd at time t."""
    b, s, p, d, _ = PHASES
    if t < b:
        return baseline
    if t < s:
        return baseline + (peak - baseline) * (t - b) / (s - b)
    if t < p:
        return peak
    if t < d:
        return peak + (baseline - peak) * (t - p) / (d - p)
    return baseline


def weights_at(t: float, regime: str) -> Tuple[float, ...]:
    # k6_flash_crowd.js uses the regime weights in the spike and the peak only:
    # from the decay on, the demand is back to the baseline weights
    w = BASELINE_WEIGHTS if t < PHASES[0] or t >= PHASES[2] else REGIME_WEIGHTS[regime]
    tot = float(sum(w))
    return tuple(x / tot for x in w)


def demand_function(regime: str, baseline: float, peak: float, n: int):
    def fn(t: int) -> List[float]:
        w = weights_at(t, regime)
        tot = offered(t, baseline, peak)
        return [tot * w[i] for i in range(n)]
    return fn


@dataclass
class Fleet:
    n: int = 6
    caps: Tuple[int, ...] = CAPS
    capacity: float = CAPACITY
    min_replicas: int = 1
    t_ready: float = 30.0
    scrape_s: float = 15.0
    noise_seed: Optional[int] = None   # Poisson request counts (None = exact)
    budget: str = "total"              # entry budget: 'total' or 'pooled' (DMOS_ENTRY_BUDGET)


class Measurement:
    """Prometheus view of a 1 s demand series (per cluster).

    With noise_seed set, the requests of each second are Poisson counts of
    the offered rate (what Hubble actually counts), so the measured rates are
    as noisy as on the clusters (a cluster with 2 req/s varies by ~40 % over
    a 30 s window)."""

    def __init__(self, scrape_s: float, noise_seed: Optional[int] = None):
        self.scrape_s = scrape_s
        self.hist: List[List[float]] = []          # hist[t][cluster]
        self._rng = None
        if noise_seed is not None:
            import numpy as np
            self._rng = np.random.default_rng(noise_seed)

    def push(self, demand: List[float]) -> None:
        if self._rng is None:
            self.hist.append(list(demand))
        else:
            self.hist.append([float(x) for x in self._rng.poisson(demand)])

    def rate(self, t: int, window_s: float) -> List[float]:
        s = int(math.floor(t / self.scrape_s) * self.scrape_s)   # last scrape
        lo = max(0, int(s - window_s))
        rows = self.hist[lo:s] or [self.hist[0]]
        n = len(rows[0])
        return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def spread(budget: int, fleet: Fleet, demand: List[float]) -> List[int]:
    """Level-1 stand-in: the minimum everywhere, then the budget by
    water-filling within the caps, the clusters with more entering demand first."""
    out = [fleet.min_replicas] * fleet.n
    left = max(0, budget - sum(out))
    order = sorted(range(fleet.n), key=lambda i: -demand[i])
    while left > 0:
        moved = False
        for i in order:
            if left and out[i] < fleet.caps[i]:
                out[i] += 1
                left -= 1
                moved = True
        if not moved:
            break
    return out


def budget_of(demand: List[float], fleet: Fleet) -> int:
    """dmos_main.entry_budget: mode 'total' or 'pooled' (fleet.budget)."""
    if fleet.budget == "pooled":
        need = math.ceil(sum(demand) / fleet.capacity)
        return min(sum(fleet.caps), max(fleet.min_replicas * fleet.n, need))
    need = sum(max(fleet.min_replicas, math.ceil(d / fleet.capacity)) for d in demand)
    return min(need, sum(fleet.caps))


class Pods:
    """Spec and ready replicas per cluster, with the actuation lag."""

    def __init__(self, fleet: Fleet):
        self.fleet = fleet
        self.spec = [fleet.min_replicas] * fleet.n
        self.ready = list(self.spec)
        self.pending: List[Tuple[float, int, int]] = []   # (t_ready, cluster, +k)
        self.last_added = 0

    def set(self, t: float, target: List[int]) -> int:
        actions, self.last_added = 0, 0
        for i, tgt in enumerate(target):
            d = tgt - self.spec[i]
            if d == 0:
                continue
            actions += 1
            if d > 0:
                self.last_added += d
                self.pending.append((t + self.fleet.t_ready, i, d))
            else:
                self.ready[i] = max(0, self.ready[i] + d)       # removal is immediate
            self.spec[i] = tgt
        return actions

    def step(self, t: float) -> None:
        still = []
        for tr, i, d in self.pending:
            if tr <= t:
                self.ready[i] = min(self.spec[i], self.ready[i] + d)
            else:
                still.append((tr, i, d))
        self.pending = still


@dataclass
class Result:
    name: str
    series: List[Tuple[int, float, int, int]] = field(default_factory=list)  # t, demand, ready, spec
    decisions: List[int] = field(default_factory=list)                       # times with a change
    added: List[Tuple[int, int]] = field(default_factory=list)               # (t, replicas added)

    def record(self, t: int, pods: Pods, n_actions: int) -> None:
        if n_actions:
            self.decisions.append(t)
            if pods.last_added:
                self.added.append((t, pods.last_added))

    def metrics(self, fleet: Fleet) -> Dict[str, float]:
        """over_target_req: demand above the ready capacity at the target CPU
        (r per replica): requests served by pods hotter than the target.
        over_limit_req: above the CPU limit (r / 0.7): requests that queue.
        t_full_capacity_s: from the start of the spike to the peak capacity.
        decisions / pods added under load: while the demand is above twice the
        baseline. replica_min: replica-minutes (cost); excess_replica_min:
        ready replicas above the need of the demand (r per replica)."""
        r_max = fleet.capacity / U_POD
        base = self.series[0][1]
        loaded = [t for t, d, _, _ in self.series if d > 2 * base]
        t0, t1 = (loaded[0], loaded[-1]) if loaded else (PHASES[0], PHASES[3])
        peak_spec = max(s for t, _, _, s in self.series if PHASES[1] <= t < PHASES[2])
        t_full = next((t for t, _, r, _ in self.series if t >= PHASES[0] and r >= peak_spec), None)
        over_t = over_l = rep_s = excess = 0.0
        for _, dem, ready, _ in self.series:
            over_t += max(0.0, dem - ready * fleet.capacity)
            over_l += max(0.0, dem - ready * r_max)
            needed = max(fleet.n * fleet.min_replicas, math.ceil(dem / fleet.capacity))
            excess += max(0, ready - needed)
            rep_s += ready
        return {
            "over_target_req": round(over_t),
            "over_limit_req": round(over_l),
            "t_full_capacity_s": (t_full - PHASES[0]) if t_full is not None else float("nan"),
            "decisions_loaded": sum(1 for t in self.decisions if t0 <= t <= t1),
            "pods_added_loaded": sum(k for t, k in self.added if t0 <= t <= t1),
            "peak_replicas": peak_spec,
            "replica_min": round(rep_s / 60.0, 1),
            "excess_replica_min": round(excess / 60.0, 1),
        }


def run_reactive(fleet: Fleet, demand_fn, horizon: int = 900) -> Result:
    res = Result("reactive")
    meas, pods = Measurement(fleet.scrape_s, fleet.noise_seed), Pods(fleet)
    hist: List[Tuple[int, List[float]]] = []
    last_up, last_down = -1e9, -1e9
    for t in range(horizon):
        dem = demand_fn(t)
        meas.push(dem)
        pods.step(t)
        if t >= 120 and t % 30 == 0:                          # startup grace, 30 s polling
            m = meas.rate(t, 60)
            hist.append((t, m))
            past = next((v for (tt, v) in reversed(hist) if tt <= t - 60), hist[0][1])
            target = spread(budget_of(m, fleet), fleet, m)
            new = []
            for i, tgt in enumerate(target):
                # per-cluster trend predictor (window 60 s, horizon 120 s, gain
                # 0.5 up / 0.3 down, cap 2x, floor = measured), PD +1 out of 0.2
                x = m[i]
                deriv = (x - past[i]) / 60.0
                pred = max(x, min(2 * x, x + deriv * 120 * (0.5 if deriv > 0 else 0.3)))
                pd = 1 if pred > 0 and (pred - x) / pred > 0.2 else 0
                tgt = min(fleet.caps[i], tgt + pd)
                cur = pods.spec[i]
                if tgt > cur:
                    tgt = min(tgt, cur + 6)
                elif tgt < cur:
                    tgt = cur if (t - last_down < 90 or t - last_up < 90) else max(tgt, cur - 1)
                new.append(tgt)
            if any(n > c for n, c in zip(new, pods.spec)):
                last_up = t
            if any(n < c for n, c in zip(new, pods.spec)):
                last_down = t
            res.record(t, pods, pods.set(t, new))
        res.series.append((t, sum(dem), sum(pods.ready), sum(pods.spec)))
    return res


def _seconds(window: str) -> float:
    unit = window[-1]
    return float(window[:-1]) * {"s": 1, "m": 60, "h": 3600}[unit]


def run_predictive(fleet: Fleet, demand_fn, params: PredictiveScalerParams,
                   horizon: int = 900, scaler_cls=PredictiveScaler) -> Result:
    res = Result("predictive")
    meas, pods = Measurement(fleet.scrape_s, fleet.noise_seed), Pods(fleet)
    names = [f"c{i}" for i in range(fleet.n)]
    ps = scaler_cls(params, names, capacity={c: fleet.capacity for c in names},
                    min_replicas={c: fleet.min_replicas for c in names},
                    max_replicas={c: fleet.caps[i] for i, c in enumerate(names)},
                    budget_mode=fleet.budget)
    fast_w, stable_w = _seconds(params.fast_window), _seconds(params.stable_window)
    tick = max(1, int(params.tick_s))
    for t in range(horizon):
        dem = demand_fn(t)
        meas.push(dem)
        pods.step(t)
        if t % tick == 0:
            for i, c in enumerate(names):
                ps.observe_ready(c, pods.ready[i], t)
            f, s = meas.rate(t, fast_w), meas.rate(t, stable_w)
            cur = {c: pods.spec[i] for i, c in enumerate(names)}
            r = ps.tick(t, dict(zip(names, f)), dict(zip(names, s)), cur)
            if r.decide and t >= 120:
                d = ps.demand_for_budget()
                dv = [d[c] for c in names]
                budget = budget_of(dv, fleet)
                if hasattr(ps, "hold_budget"):                 # policy >= 4
                    budget = ps.hold_budget(t, budget)
                rec = dict(zip(names, spread(budget, fleet, dv)))
                out = ps.targets(t, rec, cur)
                new = [out[c] for c in names]
                for i, c in enumerate(names):
                    if new[i] > pods.spec[i]:
                        ps.record_scale_up(c, new[i], t)
                res.record(t, pods, pods.set(t, new))
        res.series.append((t, sum(dem), sum(pods.ready), sum(pods.spec)))
    return res


def _table(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    keys = list(rows[0][1].keys())
    head = f"{'scaler':<24}" + "".join(f"{k:>20}" for k in keys)
    lines = [head, "-" * len(head)]
    for name, m in rows:
        lines.append(f"{name:<24}" + "".join(f"{m[k]:>20}" for k in keys))
    return "\n".join(lines)


SWEEP = [("default", {}),
         ("headroom z=0", {"headroom_z": 0.0}), ("headroom z=2", {"headroom_z": 2.0}),
         ("alpha=0.3", {"alpha": 0.3}), ("alpha=0.8", {"alpha": 0.8}),
         ("beta=0.1", {"beta": 0.1}), ("beta=0.6", {"beta": 0.6}),
         ("no forecast (H~0)", {"horizon_s": 0.001}), ("fixed H=80 s", {"horizon_s": 80.0}),
         ("fast window 60s", {"fast_window": "60s"}), ("tick 30 s", {"tick_s": 30.0}),
         ("down window 60 s", {"down_window_s": 60.0}),
         ("down window 300 s", {"down_window_s": 300.0})]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--regime", default="con", choices=sorted(REGIME_WEIGHTS))
    ap.add_argument("--baseline", type=float, default=50.0)
    ap.add_argument("--peak", type=float, default=350.0)
    ap.add_argument("--t-ready", type=float, default=30.0, help="actuation lag of the pods (s)")
    ap.add_argument("--capacity", type=float, default=CAPACITY)
    ap.add_argument("--noise-seed", type=int, help="Poisson request counts with this seed")
    ap.add_argument("--budget", default="total", choices=("total", "pooled"),
                    help="entry budget of the predictive scaler (DMOS_ENTRY_BUDGET)")
    ap.add_argument("--csv", help="write the per-second timeline of both scalers")
    ap.add_argument("--sweep", action="store_true", help="sensitivity of the PS parameters")
    a = ap.parse_args(argv)

    fleet = Fleet(capacity=a.capacity, t_ready=a.t_ready, noise_seed=a.noise_seed, budget=a.budget)
    dem = demand_function(a.regime, a.baseline, a.peak, fleet.n)
    base = PredictiveScalerParams()
    re_ = run_reactive(fleet, dem)
    pr = run_predictive(fleet, dem, base)
    print(f"regime {a.regime}, baseline {a.baseline:.0f} -> peak {a.peak:.0f} req/s, "
          f"r = {fleet.capacity:.0f} req/s/replica, caps {fleet.caps}, T_ready {fleet.t_ready:.0f} s")
    print(_table([("reactive (legacy)", re_.metrics(fleet)),
                  ("predictive (default)", pr.metrics(fleet))]))
    if a.sweep:
        rows = []
        for name, kw in SWEEP:
            p = replace(base, **kw)
            p.validate()
            rows.append((name, run_predictive(fleet, dem, p).metrics(fleet)))
        print("\nsensitivity of the predictive scaler")
        print(_table(rows))
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "demand", "reactive_ready", "reactive_spec",
                        "predictive_ready", "predictive_spec"])
            for (t, d, rr, rs), (_, _, pr_, ps_) in zip(re_.series, pr.series):
                w.writerow([t, round(d, 2), rr, rs, pr_, ps_])
        print(f"timeline -> {a.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
