"""
Predictive, capacity-model scaler for the entry service (DMOS Level 2, "PS").

The reactive Level 2 (ReplicaScaler + PD, src/level2/scaler.py) sizes the
entry service on the demand measured over a 1-minute window, decides every
30 s and adds about one replica per cluster per decision, so a flash crowd is
served by new pods 60-70 s after it starts (docs/PREDICTIVE-SCALER.md, §2).
This module replaces that loop when DMOS_SCALER=predictive. It keeps the
Level-1 placement (the score decides WHERE the replicas go) and changes HOW
MANY replicas are requested and WHEN:

  1. Two measurement windows of the entering demand, as the Knative Pod
     Autoscaler (stable and panic windows): rate[1m] and a fast rate[30s],
     sampled every tick (10 s) instead of every 30 s.
  2. A damped-trend Holt forecaster per cluster on the fast signal
     (double exponential smoothing; Gardner & McKenzie 1985 for the damping),
     evaluated at the ACTUATION LAG: the time from a decision to the moment
     the new pods are ready. The lag is learned online from the replicas the
     scaler itself adds (time from the scale command to readyReplicas).
  3. An uncertainty headroom proportional to the forecaster's recent one-step
     error (z * 1.25 * MAD ~ z * sigma), in the spirit of the safety margin
     added to usage percentiles by Google Autopilot (EuroSys 2020).
  4. Capacity-model sizing: replicas = demand / r, where r is the calibrated
     req/s one entry replica serves at its target CPU (scripts/kubephos_fleet.py
     calibrate). The same model gives the per-cluster replica caps.
  5. Event-driven decisions: a decision is taken as soon as the forecast needs
     more replicas than the deployment has (min spacing 10 s), otherwise every
     30 s. A burst (need >= panic_ratio x current) also freezes scale-down for
     panic_hold_s, as the Knative panic mode.
  6. Asymmetric actuation: scale-up straight to the target in one decision;
     scale-down through a stabilization window (the target is the maximum
     recommendation of the last down_window_s, as the HPA scaleDown
     stabilization) and at most max_down_step replicas per cluster per
     decision (gRPC connections drain).
  7. Level 2 decides HOW MANY, Level 1 WHERE (policy 4): the hysteresis acts
     on the budget (hold_budget), which Level 1 places entirely; a replica
     that Level 1 moves between clusters is added where it is recommended
     only if the move lasts move_confirm_s, and removed where it was only
     once the new one is ready (make before break). With the pooled budget
     the total demand is forecast as one series.

Everything here is pure logic over measurements and replica counts: no
Prometheus or Kubernetes calls (the orchestrator does them), so the policy is
unit-tested and replayed offline (experiments/scaler_replay.py) with the same
code that runs on the clusters.
"""

from __future__ import annotations

import math
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

SCALER_MODES = ("reactive", "predictive")

# Revision of the policy logic, recorded in the provenance (it enters the
# config_id), so a change of the algorithm with the same parameters gives a
# different id. 1: first cluster run (2026-10-28 21:09, superseded);
# 2: need counts the overflow of full clusters + fleet guard on scale-down;
# 3: hysteresis, no scale-down of a small surplus while the demand is not falling;
# 4: hysteresis on the budget (Level 1 places all of it), confirmed moves made
#    before broken, pooled budget forecast on the total demand (2026-10-31).
POLICY_VERSION = 4


# ── Parameters ───────────────────────────────────────────────────────────────

@dataclass
class PredictiveScalerParams:
    """Every knob of the predictive scaler (recorded in the run provenance).

    Defaults are the values of the journal campaign; each one can be
    overridden with DMOS_PS_<NAME> (upper case) for sensitivity studies."""
    tick_s: float = 10.0               # measurement + forecast period
    decision_interval_s: float = 30.0  # periodic decision when nothing is urgent
    min_decision_spacing_s: float = 10.0
    fast_window: str = "30s"           # PromQL window of the fast demand signal
    stable_window: str = "1m"          # PromQL window of the stable signal
    alpha: float = 0.7                 # Holt level smoothing      } chosen on the offline
    beta: float = 0.6                  # Holt trend smoothing      } replay, 24 scenarios
    trend_damping_s: float = 120.0     # damped trend: H_eff = tau (1 - e^(-H/tau))
    horizon_s: float = 0.0             # > 0: fixed forecast horizon (no learning)
    t_ready_init_s: float = 30.0       # initial actuation lag (scale cmd -> ready)
    t_ready_ewma: float = 0.3          # weight of a new actuation-lag sample
    t_ready_min_s: float = 5.0
    t_ready_max_s: float = 180.0
    headroom_z: float = 1.5            # headroom = z * 1.25 * MAD of 1-step errors
    mad_ewma: float = 0.2
    max_forecast_multiple: float = 2.0  # forecast <= 2 x max(measured)
    min_demand_rps: float = 1.0        # below: noise (health checks), no trend
    panic_ratio: float = 1.5           # need >= 1.5 x current -> burst
    panic_hold_s: float = 60.0         # no scale-down for this long after a burst
    down_window_s: float = 120.0       # scale-down stabilization window
    down_tolerance_frac: float = 0.10  # budget drop absorbed while the demand is not falling
    falling_frac: float = 0.90         # falling = total demand < 0.9 x its window max
    move_confirm_s: float = 60.0       # a Level-1 move (same total) is applied after this long
    max_down_step: int = 1             # replicas removed per cluster per decision
    down_budget_per_decision: int = 3  # replicas removed per decision, all clusters
    up_budget_per_decision: int = 40   # sanity bound on replicas added per decision

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "PredictiveScalerParams":
        env = os.environ if env is None else env
        p = cls()
        for f in p.__dataclass_fields__.values():
            raw = env.get(f"DMOS_PS_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            try:
                value = type(getattr(p, f.name))(raw)
            except ValueError:
                raise ValueError(f"DMOS_PS_{f.name.upper()}={raw!r}: expected "
                                 f"{type(getattr(p, f.name)).__name__}")
            setattr(p, f.name, value)
        p.validate()
        return p

    def validate(self) -> None:
        if not (0 < self.alpha <= 1 and 0 <= self.beta <= 1):
            raise ValueError("alpha must be in (0, 1], beta in [0, 1]")
        for name in ("tick_s", "decision_interval_s", "min_decision_spacing_s",
                     "trend_damping_s", "t_ready_init_s", "down_window_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.move_confirm_s < 0:
            raise ValueError("move_confirm_s must be >= 0")
        if self.panic_ratio < 1:
            raise ValueError("panic_ratio must be >= 1")
        if self.max_down_step < 1 or self.down_budget_per_decision < 1:
            raise ValueError("scale-down steps must be >= 1")
        if self.headroom_z < 0 or self.max_forecast_multiple < 1:
            raise ValueError("headroom_z must be >= 0 and max_forecast_multiple >= 1")
        if not (0 <= self.down_tolerance_frac < 1 and 0 < self.falling_frac <= 1):
            raise ValueError("down_tolerance_frac must be in [0, 1), falling_frac in (0, 1]")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def scaler_mode(env: Optional[Dict[str, str]] = None) -> str:
    """DMOS_SCALER: 'reactive' (default, NCA 2026 Level 2) or 'predictive'."""
    env = os.environ if env is None else env
    mode = (env.get("DMOS_SCALER", "reactive") or "reactive").strip().lower()
    if mode not in SCALER_MODES:
        raise ValueError(f"DMOS_SCALER must be one of {SCALER_MODES}, got {mode!r}")
    return mode


# ── Forecaster ───────────────────────────────────────────────────────────────

class HoltForecaster:
    """Holt's linear method (double exponential smoothing) with irregular
    sampling and a damped trend.

    State: level l (req/s) and trend b (req/s per second). With a sample x
    taken dt seconds after the previous one:
        one-step forecast  f = l + b dt         (error e = x - f -> MAD)
        level              l' = a x + (1 - a) f
        trend              b' = g (l' - l) / dt + (1 - g) b
    Forecast at horizon H: l + b * tau (1 - exp(-H / tau)), i.e. the trend
    fades with time constant tau (damped trend), so a ramp seen in the last
    samples is not extrapolated linearly for minutes."""

    def __init__(self, alpha: float, beta: float, damping_s: float, mad_ewma: float):
        self.alpha, self.beta, self.damping_s, self.mad_ewma = alpha, beta, damping_s, mad_ewma
        self.level: Optional[float] = None
        self.trend = 0.0
        self.mad = 0.0
        self.t_last: Optional[float] = None
        self.samples = 0

    def update(self, t: float, x: float) -> None:
        x = max(0.0, float(x))
        if self.level is None or self.t_last is None:
            self.level, self.trend, self.t_last, self.samples = x, 0.0, t, 1
            return
        dt = t - self.t_last
        if dt <= 0:
            return
        one_step = max(0.0, self.level + self.trend * dt)
        err = abs(x - one_step)
        self.mad = err if self.samples < 2 else (self.mad_ewma * err + (1 - self.mad_ewma) * self.mad)
        new_level = self.alpha * x + (1 - self.alpha) * one_step
        self.trend = self.beta * (new_level - self.level) / dt + (1 - self.beta) * self.trend
        self.level, self.t_last = new_level, t
        self.samples += 1

    def effective_horizon(self, horizon_s: float) -> float:
        tau = self.damping_s
        return tau * (1.0 - math.exp(-max(0.0, horizon_s) / tau))

    def forecast(self, horizon_s: float, rising_only: bool = True) -> float:
        if self.level is None:
            return 0.0
        trend = max(0.0, self.trend) if rising_only else self.trend
        return max(0.0, self.level + trend * self.effective_horizon(horizon_s))

    @property
    def sigma(self) -> float:
        """Robust spread of the one-step error (1.25 MAD ~ sigma for a normal)."""
        return 1.25 * self.mad


# ── Policy ───────────────────────────────────────────────────────────────────

@dataclass
class ClusterSizing:
    """What the policy knows about one cluster after a tick."""
    fast_rps: float = 0.0
    stable_rps: float = 0.0
    forecast_rps: float = 0.0
    headroom_rps: float = 0.0
    demand_rps: float = 0.0     # used for sizing (budget of the entry service)
    need: int = 0               # ceil(demand / capacity), within [min, max]
    raw_need: int = 0           # ceil(demand / capacity), >= min, NOT capped: in the
                                # 'total' budget the excess of a full cluster is
                                # placed on the others (DMOSOrchestrator.entry_budget)


@dataclass
class TickResult:
    """Outcome of one tick: sizing per cluster and whether to decide now."""
    t: float
    clusters: Dict[str, ClusterSizing]
    need_total: int
    current_total: int
    burst: bool
    decide: bool
    reason: str
    horizon_s: float


class PredictiveScaler:
    """Per-cluster forecasting, sizing and decision gating for ONE service.

    The orchestrator calls, in order:
      tick(t, fast, stable, current)   every params.tick_s
      demand_for_budget()              when it takes a decision
      hold_budget(t, budget)           on the entry budget computed from it
      targets(t, recommended, current) to turn the Level-1 allocation into
                                       the replica counts to apply
      record_scale_up / observe_ready  to learn the actuation lag
    All methods are thread-safe (the orchestrator ticks in one thread and
    decides in a worker thread)."""

    def __init__(self, params: PredictiveScalerParams, clusters: Iterable[str],
                 capacity: Dict[str, float], min_replicas: Dict[str, int],
                 max_replicas: Dict[str, int], budget_mode: str = "total"):
        if budget_mode not in ("total", "per_cluster", "pooled"):
            raise ValueError(f"budget_mode must be 'total', 'per_cluster' or 'pooled', "
                             f"got {budget_mode!r}")
        self.p = params
        self.budget_mode = budget_mode
        self.clusters: List[str] = list(clusters)
        self.capacity = {c: float(capacity[c]) for c in self.clusters}
        self.min_replicas = {c: int(min_replicas[c]) for c in self.clusters}
        self.max_replicas = {c: int(max_replicas[c]) for c in self.clusters}
        self._lock = threading.RLock()
        self.forecasters = {c: HoltForecaster(params.alpha, params.beta,
                                              params.trend_damping_s, params.mad_ewma)
                            for c in self.clusters}
        self.sizing: Dict[str, ClusterSizing] = {c: ClusterSizing() for c in self.clusters}
        self.t_ready_s = params.t_ready_init_s
        self.t_ready_samples = 0
        self.last_decision_t: Optional[float] = None
        self.burst_until: float = -math.inf
        self._recs: Dict[str, Deque[Tuple[float, int]]] = {c: deque() for c in self.clusters}
        self._total_recs: Deque[Tuple[float, int]] = deque()
        self._total_demand: Deque[Tuple[float, float]] = deque()   # (t, sum of stable demand)
        self._pending_up: Dict[str, Tuple[int, float]] = {}
        # pooled budget: one forecaster on the total demand (the budget is its size)
        self.total_forecaster = HoltForecaster(params.alpha, params.beta,
                                               params.trend_damping_s, params.mad_ewma)
        self.total_sizing = ClusterSizing()
        self._held_peak: Optional[int] = None        # largest budget since the demand last fell
        self._up_since: Dict[str, float] = {}        # since when Level 1 asks more on a cluster

    # -- horizon ---------------------------------------------------------------
    @property
    def horizon_s(self) -> float:
        """Forecast horizon: fixed if configured, else the time until the
        replicas requested now are serving: one tick (next decision at most
        that far) + the learned actuation lag."""
        if self.p.horizon_s > 0:
            return self.p.horizon_s
        return self.p.tick_s + self.t_ready_s

    # -- measurement -----------------------------------------------------------
    def _raw_need(self, c: str, demand: float) -> int:
        cap = self.capacity[c]
        n = math.ceil(demand / cap) if cap > 0 else self.min_replicas[c]
        return max(self.min_replicas[c], n)

    def _need(self, c: str, demand: float) -> int:
        return min(self.max_replicas[c], self._raw_need(c, demand))

    def _need_total(self) -> int:
        """The entry budget the orchestrator will compute from demand_for_budget()
        (same rule as DMOSOrchestrator.entry_budget): 'total' caps only the sum,
        'per_cluster' caps every cluster at its maximum, 'pooled' sizes the
        total demand on the mean capacity of a replica."""
        if self.budget_mode == "per_cluster":
            return sum(z.need for z in self.sizing.values())
        if self.budget_mode == "pooled":
            cap_sum = sum(self.max_replicas.values())
            min_sum = sum(self.min_replicas.values())
            weighted = sum(self.max_replicas[c] * self.capacity[c] for c in self.clusters)
            r_mean = weighted / cap_sum if cap_sum > 0 else 0.0
            demand = sum(z.demand_rps for z in self.sizing.values())
            need = math.ceil(demand / r_mean) if r_mean > 0 else min_sum
            return min(cap_sum, max(min_sum, need))
        return min(sum(self.max_replicas.values()),
                   sum(z.raw_need for z in self.sizing.values()))

    def tick(self, t: float, fast: Dict[str, float], stable: Dict[str, float],
             current: Dict[str, int]) -> TickResult:
        """Update the forecasters with one measurement per cluster and say
        whether a decision is needed now."""
        with self._lock:
            h = self.horizon_s
            for c in self.clusters:
                x_f = max(0.0, float(fast.get(c, 0.0) or 0.0))
                x_s = max(0.0, float(stable.get(c, 0.0) or 0.0))
                fc = self.forecasters[c]
                fc.update(t, x_f)
                measured = max(x_f, x_s)
                if measured < self.p.min_demand_rps:
                    forecast, headroom = measured, 0.0
                else:
                    forecast = min(fc.forecast(h), self.p.max_forecast_multiple * measured)
                    headroom = self.p.headroom_z * fc.sigma
                demand = max(measured, forecast + headroom)
                self.sizing[c] = ClusterSizing(fast_rps=x_f, stable_rps=x_s,
                                               forecast_rps=forecast, headroom_rps=headroom,
                                               demand_rps=demand, need=self._need(c, demand),
                                               raw_need=self._raw_need(c, demand))
            if self.budget_mode == "pooled":
                self._size_on_total(t, h)
            need_total = self._need_total()
            self._total_demand.append((t, sum(z.stable_rps for z in self.sizing.values())))
            while self._total_demand and self._total_demand[0][0] < t - self.p.down_window_s:
                self._total_demand.popleft()
            current_total = sum(int(current.get(c, 0) or 0) for c in self.clusters)
            burst = need_total >= math.ceil(self.p.panic_ratio * max(1, current_total)) \
                and need_total > current_total
            if burst:
                self.burst_until = t + self.p.panic_hold_s
            since = math.inf if self.last_decision_t is None else t - self.last_decision_t
            if need_total > current_total and since >= self.p.min_decision_spacing_s:
                decide, reason = True, "burst" if burst else "need_up"
            elif since >= self.p.decision_interval_s:
                decide, reason = True, "periodic"
            else:
                decide, reason = False, ""
            return TickResult(t=t, clusters=dict(self.sizing), need_total=need_total,
                              current_total=current_total, burst=burst, decide=decide,
                              reason=reason, horizon_s=h)

    def _size_on_total(self, t: float, h: float) -> None:
        """Pooled budget: forecast the TOTAL demand and size on it. Summing the
        per-cluster forecasts counts the rise of some clusters and not the
        fall of the others (trend >= 0 per cluster): when the demand only moves
        between clusters (end of the peak of a flash crowd) the sum exceeds the
        total by up to 2 replicas. The total is split back over the clusters by
        their measured share (only the sum enters the pooled budget)."""
        z = self.sizing
        fast, stable = sum(s.fast_rps for s in z.values()), sum(s.stable_rps for s in z.values())
        fc = self.total_forecaster
        fc.update(t, fast)
        measured = max(fast, stable)
        if measured < self.p.min_demand_rps:
            forecast, headroom = measured, 0.0
        else:
            forecast = min(fc.forecast(h), self.p.max_forecast_multiple * measured)
            headroom = self.p.headroom_z * fc.sigma
        total = max(measured, forecast + headroom)
        self.total_sizing = ClusterSizing(fast_rps=fast, stable_rps=stable, forecast_rps=forecast,
                                          headroom_rps=headroom, demand_rps=total)
        meas = {c: max(s.fast_rps, s.stable_rps) for c, s in z.items()}
        msum = sum(meas.values())
        for c, s in z.items():
            d = total * (meas[c] / msum if msum > 0 else 1.0 / len(z))
            s.demand_rps, s.need, s.raw_need = d, self._need(c, d), self._raw_need(c, d)

    def hold_budget(self, t: float, budget: int) -> int:
        """Hysteresis on the number of entry replicas (Level 2): while the
        fleet demand is not falling, the budget drops by at most
        down_tolerance_frac (at least one replica) below the largest budget
        since the demand last fell, so the replicas added by the forecast
        during a ramp are not removed on the plateau that follows. Level 1
        places the held budget like any other, on the best clusters."""
        with self._lock:
            falling = self.demand_falling()
            if falling or self._held_peak is None:
                self._held_peak = budget
            else:
                self._held_peak = max(self._held_peak, budget)
            if falling or budget <= sum(self.min_replicas.values()):
                return budget
            tol = max(1, math.ceil(self.p.down_tolerance_frac * budget))
            return max(budget, min(self._held_peak, budget + tol))

    def demand_for_budget(self) -> Dict[str, float]:
        """Per-cluster demand (req/s) the entry-service budget is sized on."""
        with self._lock:
            return {c: s.demand_rps for c, s in self.sizing.items()}

    def demand_falling(self) -> bool:
        """The fleet demand is clearly falling: the total stable demand is below
        falling_frac of its maximum over the scale-down window."""
        with self._lock:
            if not self._total_demand:
                return False
            peak = max(d for _, d in self._total_demand)
            return peak > 0 and self._total_demand[-1][1] < self.p.falling_frac * peak

    def in_burst(self, t: float) -> bool:
        with self._lock:
            return t < self.burst_until

    # -- actuation -------------------------------------------------------------
    def targets(self, t: float, recommended: Dict[str, int],
                current: Dict[str, int]) -> Dict[str, int]:
        """Replica counts to apply, from the Level-1 recommendation (which
        places the whole budget, held by hold_budget).

        Up: when the total recommendation exceeds the current total (the fleet
        needs more replicas), every scale-up at once, largest first (bounded by
        the per-cluster maximum and up_budget_per_decision). When the total
        does not grow, a scale-up is a MOVE of Level 1 between clusters: it is
        applied only after the cluster has been recommended more for
        move_confirm_s, so a placement flipping between clusters with close
        scores does not start and stop pods (and their co-scaled backends).

        Down (make before break): never in a decision that scales up, nor
        while a scale-up is not ready yet, nor during a burst hold; a cluster
        goes down to the maximum recommendation of the last down_window_s
        (stabilization), at most max_down_step per cluster and
        down_budget_per_decision overall, and the fleet never goes below the
        total recommendation stabilized over the same window. The clusters
        furthest above their target go first, then those Level 1 recommends
        least (a cluster recommended 0 is emptied)."""
        with self._lock:
            self.last_decision_t = t
            hold = t < self.burst_until
            rec_all = {c: min(self.max_replicas[c], max(self.min_replicas[c],
                                                        int(recommended.get(c, self.min_replicas[c]))))
                       for c in self.clusters}
            cur_all = {c: int(current.get(c, 0) or 0) for c in self.clusters}
            total_rec, total_current = sum(rec_all.values()), sum(cur_all.values())
            self._total_recs.append((t, total_rec))
            while self._total_recs and self._total_recs[0][0] < t - self.p.down_window_s:
                self._total_recs.popleft()
            total_stabilized = max(r for _, r in self._total_recs)
            growth = total_rec > total_current
            stabilized: Dict[str, int] = {}
            for c in self.clusters:
                q = self._recs[c]
                q.append((t, rec_all[c]))
                while q and q[0][0] < t - self.p.down_window_s:
                    q.popleft()
                stabilized[c] = max(r for _, r in q)
                if rec_all[c] > cur_all[c]:
                    self._up_since.setdefault(c, t)
                else:
                    self._up_since.pop(c, None)
            out = dict(cur_all)
            up_left = self.p.up_budget_per_decision
            for c in sorted(self.clusters, key=lambda c: -(rec_all[c] - cur_all[c])):
                if rec_all[c] <= cur_all[c] or up_left <= 0:
                    continue
                if not growth and t - self._up_since[c] < self.p.move_confirm_s:
                    continue
                add = min(rec_all[c] - cur_all[c], up_left)
                up_left -= add
                out[c] = cur_all[c] + add
            scaling_up = any(out[c] > cur_all[c] for c in self.clusters)
            down_left = 0
            if not (hold or scaling_up or self._pending_up):
                down_left = min(self.p.down_budget_per_decision,
                                max(0, total_current - total_stabilized))
            for c in sorted(self.clusters,
                            key=lambda c: (-(cur_all[c] - stabilized[c]), rec_all[c])):
                if down_left <= 0:
                    break
                if stabilized[c] < cur_all[c]:
                    step = min(cur_all[c] - stabilized[c], self.p.max_down_step, down_left)
                    down_left -= step
                    out[c] = cur_all[c] - step
            return out

    # -- actuation-lag learning -------------------------------------------------
    def record_scale_up(self, cluster: str, target: int, t: float) -> None:
        with self._lock:
            self._pending_up[cluster] = (target, t)

    def pending_clusters(self) -> List[str]:
        with self._lock:
            return list(self._pending_up)

    def observe_ready(self, cluster: str, ready: int, t: float) -> Optional[float]:
        """Ready replicas of a cluster with a pending scale-up; returns the
        lag sample (s) once the requested replicas are all ready."""
        with self._lock:
            pending = self._pending_up.get(cluster)
            if pending is None:
                return None
            target, t_cmd = pending
            if ready < target:
                if t - t_cmd > self.p.t_ready_max_s:
                    self._pending_up.pop(cluster, None)   # never became ready: drop
                return None
            self._pending_up.pop(cluster, None)
            lag = min(self.p.t_ready_max_s, max(self.p.t_ready_min_s, t - t_cmd))
            w = self.p.t_ready_ewma if self.t_ready_samples else 1.0
            self.t_ready_s = w * lag + (1 - w) * self.t_ready_s
            self.t_ready_samples += 1
            return lag

    def snapshot(self) -> Dict[str, object]:
        """Provenance: parameters and the per-cluster model inputs."""
        return {
            "mode": "predictive",
            "policy_version": POLICY_VERSION,
            "budget_mode": self.budget_mode,
            "params": self.p.to_dict(),
            "capacity_req_per_sec": dict(self.capacity),
            "min_replicas": dict(self.min_replicas),
            "max_replicas": dict(self.max_replicas),
        }
