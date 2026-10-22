"""
Provenance of a DMOS process: the configuration DMOS is ACTUALLY running with.

Why this exists
---------------
Environment overrides (DMOS_SCORE_GATE, DMOS_SCORE_GATE_GAMMA,
DMOS_WEIGHTS_COHERENT, DMOS_LOCALITY_PURE_NET, DMOS_DEMAND_FLOOR,
DMOS_SCORE_PROFILE, ...) change the algorithm without touching any file, so a
run label such as ``dmos-l1l2__con`` cannot tell which algorithm produced a
result. At startup DMOS serialises its *effective* configuration (after every
override is applied, read back from the live objects) and derives a stable
fingerprint, ``config_id``, from the algorithm-defining sections only.

The run harness stores this file in the run directory (``dmos_config.json``);
the analysis groups and cross-checks runs by ``config_id``.

The builder only READS attributes of already-constructed objects: it never
changes behaviour and never touches the network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]

# Sections that define the ALGORITHM. Only these enter the fingerprint:
# timestamps, PIDs, raw environment and git state are recorded but do not
# change config_id, so two runs of the same algorithm share the same id.
FINGERPRINT_SECTIONS = (
    "modes", "level1", "level2", "safeguards", "coscaling", "carbon", "clusters",
)


# ── Serialisation helpers ────────────────────────────────────────────────────

def _jsonable(obj: Any) -> Any:
    """json.dumps fallback for the few non-JSON types found in the config."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _clean_floats(obj: Any) -> Any:
    """Replace NaN/inf (not valid JSON) with None, recursively."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_floats(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text (sorted keys, no whitespace) used for hashing."""
    return json.dumps(_clean_floats(obj), sort_keys=True, separators=(",", ":"),
                      default=_jsonable, ensure_ascii=True)


# Descriptive keys inside the fingerprinted sections: WHERE a value came from
# (model/profile name), not WHAT the algorithm is. Two models with identical
# content under different names must share the config_id.
NON_ALGORITHMIC_KEYS = frozenset({"weights_source", "score_profile"})


def _strip_metadata(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_metadata(v) for k, v in obj.items() if k not in NON_ALGORITHMIC_KEYS}
    if isinstance(obj, list):
        return [_strip_metadata(v) for v in obj]
    return obj


def config_fingerprint(effective: Dict[str, Any]) -> str:
    """12-hex fingerprint of the algorithm-defining sections."""
    subset = _strip_metadata({k: effective.get(k) for k in FINGERPRINT_SECTIONS})
    return hashlib.sha256(canonical_json(subset).encode("ascii")).hexdigest()[:12]


def write_json_atomic(path: os.PathLike, data: Dict[str, Any]) -> Path:
    """Write JSON via a temp file + rename, so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_clean_floats(data), indent=2, sort_keys=False,
                              default=_jsonable, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


# ── Process / repository context ────────────────────────────────────────────

def dmos_env() -> Dict[str, str]:
    """Every DMOS_* environment variable visible to the process (sorted)."""
    return {k: os.environ[k] for k in sorted(os.environ) if k.startswith("DMOS_")}


def git_state(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Commit, branch and dirty files of the repository the code runs from."""
    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", str(repo_root), *args],
                                 capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.decode("utf-8", errors="replace").strip()

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    porcelain = _git("status", "--porcelain") or ""
    dirty = [line[3:] for line in porcelain.splitlines() if line.strip()]
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": dirty,
    }


# ── Component snapshots ──────────────────────────────────────────────────────

def demand_roles(sf) -> List[str]:
    """Where the observed demand share d_i enters a ScoreFunctions instance."""
    roles = []
    if getattr(sf, "gate_enabled", False) and getattr(sf, "gate_gamma", 0) != 0:
        roles.append("gate")              # G(d) multiplies the whole sum (g=0 → G≡1)
    if not getattr(sf, "locality_pure_net", False) and sf.omega_geo_load > 0:
        roles.append("locality_net")      # ω_net · (d · Φ_net)
    if getattr(sf, "omega_demand_share", 0) > 0:
        roles.append("additive")          # + ω_dem · d
    if sf.omega_demand > 0:
        roles.append("pressure")          # ω · Φ_pressure(util) — demand-derived
    return roles


def render_formula(sf) -> str:
    """Human-readable form of the score a ScoreFunctions instance computes."""
    locality = "net" if getattr(sf, "locality_pure_net", False) else "d*net"
    terms = [
        (sf.omega_latency, "lat"),
        (sf.omega_capacity, "cap"),
        (sf.omega_geo_load, locality),
        (sf.omega_carbon, "car"),
        (sf.omega_demand, "pressure"),
        (getattr(sf, "omega_demand_share", 0.0), "d"),
    ]
    inner = " + ".join(f"{w:g}*{name}" for w, name in terms if w) or "0"
    if getattr(sf, "gate_enabled", False):
        return (f"S = G(d; eps={sf.gate_eps:g}, g={sf.gate_gamma:g})"
                f" * ({inner})")
    return f"S = {inner}"


def score_function_snapshot(sf) -> Dict[str, Any]:
    """Effective weights, terms and parameters of one ScoreFunctions instance."""
    return {
        "formula": render_formula(sf),
        "demand_mode": getattr(sf, "demand_mode", "legacy"),
        "demand_roles": demand_roles(sf),
        "weights_source": getattr(sf, "weights_source", "profile"),
        "weights": {
            "latency": sf.omega_latency,
            "capacity": sf.omega_capacity,
            "locality": sf.omega_geo_load,     # net slot: Φ_net or d·Φ_net
            "carbon": sf.omega_carbon,
            "pressure": sf.omega_demand,
            "demand": getattr(sf, "omega_demand_share", 0.0),   # additive d term
        },
        "locality_term": "net" if getattr(sf, "locality_pure_net", False) else "demand_x_net",
        "gate": {
            "enabled": bool(getattr(sf, "gate_enabled", False)),
            "eps": getattr(sf, "gate_eps", None),
            "gamma": getattr(sf, "gate_gamma", None),
        },
        "parameters": dataclasses.asdict(sf.params),
    }


def scaler_snapshot(scaler) -> Dict[str, Any]:
    """Level-2 parameters of one ReplicaScaler (predictor + PD included)."""
    pred = scaler.predictor
    pd = scaler.controller
    return {
        "capacity_per_replica": scaler.capacity_per_replica,
        "min_replicas": scaler.min_replicas,
        "max_replicas": scaler.max_replicas,
        "safety_margin": scaler.safety_margin,
        "max_delta_up": scaler.max_delta_per_cycle,
        "max_delta_down": scaler.max_delta_down,
        "predictor": {
            "window_s": pred.window_seconds,
            "horizon_s": pred.horizon_seconds,
            "trend_factor_up": getattr(pred, "trend_factor_up", None),
            "trend_factor_down": getattr(pred, "trend_factor_down", None),
            "cap_multiple": getattr(pred, "cap_multiple", None),
        },
        "pd": {
            "kp": pd.kp,
            "kd": pd.kd,
            "deadband": pd.deadband,
            "output_min": pd.output_min,
            "output_max": pd.output_max,
        },
    }


# ── Effective configuration of a DMOSOrchestrator ───────────────────────────

def build_effective_config(orch) -> Dict[str, Any]:
    """
    Snapshot of the configuration a DMOSOrchestrator is running with.

    Values are read from the live objects (not from the YAML files), so every
    environment override and every hard-coded constant is captured as used.
    """
    # Imported here to avoid a circular import at module load.
    from src.level1 import dmos_scheduler as sched_mod

    cfg = orch.config
    sched = orch.scheduler

    # Level 2: controller parameters are normally identical everywhere; only
    # capacity / min / max differ per (service, cluster). Store them once and
    # flag the run if they are not homogeneous.
    controller_common: Optional[Dict[str, Any]] = None
    controller_heterogeneous = False
    replicas: Dict[str, Dict[str, Any]] = {}
    for svc, by_cluster in orch.scalers.items():
        replicas[svc] = {}
        for cl, scaler in by_cluster.items():
            snap = scaler_snapshot(scaler)
            ctrl = {k: snap[k] for k in
                    ("safety_margin", "max_delta_up", "max_delta_down", "predictor", "pd")}
            if controller_common is None:
                controller_common = ctrl
            elif ctrl != controller_common:
                controller_heterogeneous = True
            replicas[svc][cl] = {k: snap[k] for k in
                                 ("capacity_per_replica", "min_replicas", "max_replicas")}

    carbon_raw = (cfg.carbon_raw or {}).get("carbon_intensity", {}) or {}
    trace_cfg = carbon_raw.get("trace", {}) or {}

    effective: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "process": {
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        },
        "git": git_state(),
        "dmos_env": dmos_env(),
        # Named score model (DMOS_SCORE_MODEL), recorded for readability. Its
        # algorithmic content is captured by level1/modes, so the NAME does
        # not enter config_id.
        "score_model": ({**sched.score_model.to_dict(),
                         "cold_variant": sched.score_model.cold_variant().to_dict()}
                        if getattr(sched, "score_model", None) is not None else None),
        "ignored_legacy_env": list(getattr(sched, "ignored_legacy_env", [])),

        "modes": {
            "level1_scheduling": not orch.no_scheduling,
            "naive_backends": bool(orch.naive_backends),
            "demand_floor": bool(orch.demand_floor_enabled),
            "colocation": bool(orch.colocation_enabled),
            "monitored_services": list(orch.monitored_services),
        },
        "level1": {
            "score_profile": getattr(cfg, "active_score_profile", None),
            "cold_start_seconds": sched_mod.COLD_START_SECONDS,
            "warm": score_function_snapshot(sched.score_func_warm),
            "cold": score_function_snapshot(sched.score_func_cold),
            "eligibility": {
                "cpu_hard_limit_pct": sched_mod.CPU_HARD_LIMIT_PCT,
                "memory_hard_limit_pct": sched_mod.MEMORY_HARD_LIMIT_PCT,
                "min_cpu_cores_available": sched_mod.MIN_CPU_CORES_AVAILABLE,
                "min_memory_gb_available": sched_mod.MIN_MEMORY_GB_AVAILABLE,
            },
            "network_params": dict(cfg.network_params),
            "load_params": dict(cfg.load_params),
        },
        "level2": {
            "pd_applied_clamp": getattr(orch, "pd_correction_clamp", None),
            "controller": controller_common,
            "controller_heterogeneous": controller_heterogeneous,
            "replicas": replicas,
            "naive_target_rps": getattr(orch, "naive_target_rps", None),
        },
        "safeguards": {
            "polling_interval_s": orch.polling_interval,
            "debounce_s": orch.debounce_seconds,
            "scale_down_cooldown_s": orch.scale_down_cooldown,
            "scale_up_protection_s": orch.scale_up_protection_seconds,
            "entry_budget_up": orch.scale_up_budget_per_cycle,
            "entry_budget_down": orch.scale_down_budget_per_cycle,
            "backend_budget_up": orch.backend_scale_up_budget_per_cycle,
            "backend_budget_down": orch.backend_scale_down_budget_per_cycle,
            "dead_zone_pct": orch.dead_zone_pct,
            "startup_grace_s": orch.startup_grace_seconds,
            "high_threshold_rps": orch.high_threshold,
            "low_threshold_rps": orch.low_threshold,
        },
        "coscaling": {
            "dependencies": {k: list(v) for k, v in orch.service_dependencies.items()},
            "colocation_floor": orch.backend_colocation_floor,
            "calls_per_request": {n: s.calls_per_request for n, s in cfg.services.items()},
            "capacity_req_per_sec": {n: s.capacity_req_per_sec for n, s in cfg.services.items()},
        },
        "carbon": {
            "provider": carbon_raw.get("provider"),
            "trace": {
                "data_dir": trace_cfg.get("data_dir"),
                "window_start": trace_cfg.get("window_start"),
                "speedup": trace_cfg.get("speedup"),
                "zone_aliases": dict(trace_cfg.get("zone_aliases") or {}),
            },
            # Where each cluster's carbon intensity comes from: 'trace:<file>',
            # or a fallback ('realistic' = built-in profiles that depend on the
            # wall-clock hour, 'mock', 'default-300').
            "sources": carbon_sources(sched, cfg),
        },
        "clusters": {
            name: {
                "region": c.region,
                "cpu_cores": c.cpu_cores,
                "memory_gb": c.memory_gb,
                "prometheus_port": c.prometheus_port,
                "carbon_region": (c.carbon or {}).get("region_code"),
                "carbon_baseline_gco2_kwh": (c.carbon or {}).get("baseline_gco2_kwh"),
            }
            for name, c in cfg.clusters.items()
        },
    }
    effective["config_id"] = config_fingerprint(effective)
    effective["summary"] = summary_line(effective)
    return effective


def carbon_sources(sched, cfg) -> Dict[str, Optional[str]]:
    """Carbon-intensity source per cluster, resolved without loading data."""
    client = getattr(sched, "carbon_client", None)
    out: Dict[str, Optional[str]] = {}
    for name, c in cfg.clusters.items():
        region = (c.carbon or {}).get("region_code")
        try:
            out[name] = client.source_for(region) if client is not None else None
        except Exception:  # provenance must never break the scheduler
            out[name] = None
    return out


def summary_line(effective: Dict[str, Any]) -> str:
    """One-line description, printed by the harness before each run."""
    modes = effective.get("modes", {})
    l1 = effective.get("level1", {})
    warm = l1.get("warm", {})
    arm = "L1+L2" if modes.get("level1_scheduling") else "L2-only (uniform N/K)"
    floor = "on" if modes.get("demand_floor") else "off"
    src = warm.get("weights_source", "profile")
    if src == "profile":
        weights = f"profile={l1.get('score_profile')}"
    elif src.startswith("model:"):
        weights = f"model={src.split(':', 1)[1]} ({warm.get('demand_mode')})"
    else:
        weights = f"weights={src}"
    sources = (effective.get("carbon") or {}).get("sources") or {}
    carbon = ""
    if sources:
        n_trace = sum(1 for s in sources.values() if str(s).startswith("trace:"))
        carbon = f" | carbon trace {n_trace}/{len(sources)}"
    return (f"[{effective.get('config_id', '?')}] {arm} | {warm.get('formula', '?')}"
            f" | demand via {','.join(warm.get('demand_roles', [])) or 'none'}"
            f" | floor={floor} | {weights}{carbon}")
