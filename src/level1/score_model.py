"""
Score models: HOW the observed demand share d_i enters the DMOS score.

A score model fixes, in one named and reviewable place (config/weights.yaml,
section `score_models`), everything Level 1 needs to rank clusters:

    demand_mode  where d_i enters the score
      gate       S = G(d)·[ω_lat Φ_lat + ω_cap Φ_cap + ω_net Φ_net + ω_car Φ_car]
                 G(d) = (ε + (1−ε)·d)^g        (paper NCA, eq. 1-2; g=0 → G≡1)
      additive   S = ω_lat Φ_lat + ω_cap Φ_cap + ω_net Φ_net + ω_car Φ_car + ω_dem·d
      locality   S = ω_lat Φ_lat + ω_cap Φ_cap + ω_net·(d·Φ_net) + ω_car Φ_car
    weights      latency, capacity, net, carbon (+ demand, additive mode only); Σ = 1
    gate         eps (ε), gamma (g)              — gate mode only
    demand_signal  what "d" is in the modes above:
      share      d_i = λ_i / Σ_j λ_j, the share of the demand ENTERING cluster i
                 (NCA 2026: right when each cluster serves its own entry traffic)
      reach      d_i = ψ_i = Σ_j share_j · exp(−RTT(j,i) / τ), the demand that can
                 reach cluster i within a few τ (reach_tau_ms, default 50 ms).
                 With a pooled entry service (Cilium global service, affinity
                 'none') a request entering j is served by any replica, so a
                 replica on i serves j's users at the cost RTT(j,i): ψ_i is the
                 demand cluster i serves cheaply. RTT(i,i) = 0, so ψ_i >= share_i,
                 and ψ → share as the RTTs grow (the NCA gate is the limit case).
    demand_floor guarantee ⌈demand/capacity⌉ replicas per cluster (orchestrator)
    cold_start   'structural': during the first COLD_START_SECONDS the
                 latency and capacity weights are 0 (their metrics are not
                 reliable yet) and the remaining weights are renormalised;
                 demand keeps its role. 'same': no phase distinction.
    parameters   optional overrides of ScoreParameters (eta, nu, ci_max,
                 rho, rtt_max_ms, ...)

Selected with DMOS_SCORE_MODEL=<name>. When a model is active the legacy
environment switches (DMOS_SCORE_GATE*, DMOS_WEIGHTS_COHERENT,
DMOS_LOCALITY_PURE_NET, DMOS_DEMAND_FLOOR, DMOS_SCORE_PROFILE) are IGNORED
(and reported), so a run is defined by its model alone. Without a model the
legacy behaviour is unchanged.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

DEMAND_MODES = ("gate", "additive", "locality")
DEMAND_SIGNALS = ("share", "reach")
DEFAULT_REACH_TAU_MS = 50.0
COLD_START_POLICIES = ("structural", "same")
WEIGHT_KEYS = ("latency", "capacity", "net", "carbon", "demand")
STRUCTURAL_ZEROED = ("latency", "capacity")

# Environment switches of the legacy (pre-model) configuration path.
LEGACY_ENV = ("DMOS_SCORE_GATE", "DMOS_SCORE_GATE_GAMMA", "DMOS_SCORE_GATE_EPS",
              "DMOS_WEIGHTS_COHERENT", "DMOS_LOCALITY_PURE_NET",
              "DMOS_DEMAND_FLOOR", "DMOS_SCORE_PROFILE")

PARAMETER_KEYS = ("eta", "sigma_squared", "kappa", "load_threshold", "load_steepness",
                  "load_floor", "horizon_seconds", "nu", "ci_max", "rho", "rtt_max_ms")


class ScoreModelError(ValueError):
    """Invalid score model definition or unknown model name."""


@dataclass(frozen=True)
class ScoreModel:
    name: str
    demand_mode: str
    weights: Dict[str, float]
    gate_eps: float = 0.05
    gate_gamma: float = 2.0
    demand_floor: bool = False
    cold_start: str = "structural"
    parameters: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    demand_signal: str = "share"     # 'share' | 'reach' (see module docstring)
    reach_tau_ms: float = DEFAULT_REACH_TAU_MS
    phase: str = "warm"              # 'warm' | 'cold' (set by cold_variant)
    cold_start_note: str = ""        # how the cold weights were obtained

    # ── construction / validation ───────────────────────────────────────
    @classmethod
    def from_dict(cls, name: str, raw: Dict[str, Any]) -> "ScoreModel":
        if not isinstance(raw, dict):
            raise ScoreModelError(f"score model '{name}': definition must be a mapping")
        mode = raw.get("demand_mode")
        if mode not in DEMAND_MODES:
            raise ScoreModelError(
                f"score model '{name}': demand_mode must be one of {DEMAND_MODES}, got {mode!r}")
        w_raw = raw.get("weights") or {}
        unknown = set(w_raw) - set(WEIGHT_KEYS)
        if unknown:
            raise ScoreModelError(f"score model '{name}': unknown weight keys {sorted(unknown)} "
                                  f"(allowed: {WEIGHT_KEYS})")
        weights = {k: float(w_raw.get(k, 0.0)) for k in WEIGHT_KEYS}
        if any(v < 0 for v in weights.values()):
            raise ScoreModelError(f"score model '{name}': weights must be >= 0")
        if mode != "additive" and weights["demand"] != 0.0:
            raise ScoreModelError(f"score model '{name}': a 'demand' weight is only meaningful "
                                  f"in additive mode (mode is {mode!r})")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ScoreModelError(f"score model '{name}': weights must sum to 1.0, got {total:.6f}")
        gate = raw.get("gate") or {}
        eps, gamma = float(gate.get("eps", 0.05)), float(gate.get("gamma", 2.0))
        if not 0.0 <= eps <= 1.0 or gamma < 0:
            raise ScoreModelError(f"score model '{name}': gate needs 0<=eps<=1 and gamma>=0")
        cold = raw.get("cold_start", "structural")
        if cold not in COLD_START_POLICIES:
            raise ScoreModelError(f"score model '{name}': cold_start must be one of "
                                  f"{COLD_START_POLICIES}, got {cold!r}")
        params = {k: float(v) for k, v in (raw.get("parameters") or {}).items()}
        bad = set(params) - set(PARAMETER_KEYS)
        if bad:
            raise ScoreModelError(f"score model '{name}': unknown parameters {sorted(bad)} "
                                  f"(allowed: {PARAMETER_KEYS})")
        signal = raw.get("demand_signal", "share")
        if signal not in DEMAND_SIGNALS:
            raise ScoreModelError(f"score model '{name}': demand_signal must be one of "
                                  f"{DEMAND_SIGNALS}, got {signal!r}")
        tau = float(raw.get("reach_tau_ms", DEFAULT_REACH_TAU_MS))
        if tau <= 0:
            raise ScoreModelError(f"score model '{name}': reach_tau_ms must be > 0")
        return cls(name=name, demand_mode=mode, weights=weights, gate_eps=eps,
                   gate_gamma=gamma, demand_floor=bool(raw.get("demand_floor", False)),
                   cold_start=cold, parameters=params,
                   demand_signal=signal, reach_tau_ms=tau,
                   description=str(raw.get("description", "")).strip())

    # ── derived views ───────────────────────────────────────────────────
    def cold_variant(self) -> "ScoreModel":
        """Weights used during the cold-start phase (see module docstring)."""
        if self.cold_start == "same":
            return replace(self, phase="cold", cold_start_note="same as warm (policy 'same')")
        kept = {k: (0.0 if k in STRUCTURAL_ZEROED else v) for k, v in self.weights.items()}
        rest = sum(kept.values())
        if rest <= 0:
            return replace(self, phase="cold", cold_start_note=(
                "same as warm: the model has no structural weight (net/carbon/demand) "
                "to keep during cold start"))
        cold = {k: v / rest for k, v in kept.items()}
        return replace(self, weights=cold, phase="cold",
                       cold_start_note="structural: latency and capacity weights set to 0, "
                                       "remaining weights renormalised")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "phase": self.phase, "demand_mode": self.demand_mode,
            "weights": dict(self.weights),
            "gate": {"eps": self.gate_eps, "gamma": self.gate_gamma}
            if self.demand_mode == "gate" else None,
            "demand_signal": self.demand_signal,
            "reach_tau_ms": self.reach_tau_ms if self.demand_signal == "reach" else None,
            "demand_floor": self.demand_floor, "cold_start": self.cold_start,
            "cold_start_note": self.cold_start_note or None,
            "parameters": dict(self.parameters), "description": self.description,
        }


# ── reach-weighted demand ───────────────────────────────────────────────────

def reach_shares(shares: Dict[str, float], rtt_ms: Dict[str, Dict[str, float]],
                 tau_ms: float = DEFAULT_REACH_TAU_MS) -> Dict[str, float]:
    """ψ_i = Σ_j share_j · exp(−RTT(j,i)/τ), with RTT(i,i) = 0.

    shares   demand share entering each cluster (sums to 1, or all 0)
    rtt_ms   rtt_ms[a][b] = RTT measured from cluster a to cluster b (ms).
             A missing pair uses the other direction; a pair missing in both
             directions counts as unreachable (no contribution).
    Each ψ_i is in [share_i, 1]."""
    if tau_ms <= 0:
        raise ValueError("tau_ms must be > 0")
    names = list(shares)

    def rtt(a: str, b: str) -> Optional[float]:
        if a == b:
            return 0.0
        for x, y in ((a, b), (b, a)):
            v = (rtt_ms.get(x) or {}).get(y)
            if v is not None and v >= 0 and math.isfinite(v):
                return float(v)
        return None

    out: Dict[str, float] = {}
    for i in names:
        psi = 0.0
        for j in names:
            r = rtt(j, i)
            if r is not None:
                psi += max(0.0, shares[j]) * math.exp(-r / tau_ms)
        out[i] = min(1.0, psi)
    return out


# ── registry helpers ────────────────────────────────────────────────────────

def parse_models(weights_raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Raw `score_models` section of weights.yaml (name → definition)."""
    models = (weights_raw or {}).get("score_models") or {}
    if not isinstance(models, dict):
        raise ScoreModelError("weights.yaml: 'score_models' must be a mapping")
    return models


def select_model(weights_raw: Dict[str, Any],
                 name: Optional[str] = None) -> Optional[ScoreModel]:
    """The model named by `name` (default: $DMOS_SCORE_MODEL), or None."""
    name = (name if name is not None else os.environ.get("DMOS_SCORE_MODEL", "")).strip()
    if not name:
        return None
    models = parse_models(weights_raw)
    if name not in models:
        raise ScoreModelError(f"score model '{name}' not found in config/weights.yaml "
                              f"(available: {', '.join(sorted(models)) or 'none'})")
    return ScoreModel.from_dict(name, models[name])


def ignored_legacy_env() -> List[str]:
    """Legacy switches present in the environment (ignored under a model)."""
    return [k for k in LEGACY_ENV if k in os.environ]
