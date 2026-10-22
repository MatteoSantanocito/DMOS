#!/usr/bin/env python3
"""
run_naming.py — the ONE place that turns a test name into a configuration.

Every orchestrator (run_one.sh, run_matrix_session.sh, run_skew_sweep.sh) and
the analysis (paper_lib.py) use these rules, so a name always means the same
configuration. A run folder is  experiments/results/<ts>_dmos-<arm>__<regime>.

ARM — what runs
  off              DMOS not running: 1 static replica per cluster
  l2               DMOS Level 2 only (src.dmos_main --no-scheduling): N/K per cluster
  l1l2             DMOS L1+L2 exactly as run in the NCA 2026 campaign (Figs. 2-3;
                   DMOS logs of 15-17 June 2026): demand gate g=2, Phi_loc =
                   Phi_net, Phi_pre off, weights lat/cap/net/car =
                   0.25/0.20/0.25/0.30, demand floor off (environment switches
                   on the legacy profile; formerly "l1l2coh")
  <model>          DMOS L1+L2 with a score model of config/weights.yaml
                   (score_models): nca, gate_g0, gate_g1, gate_g3, additive_dem50,
                   locality, latency_first, capacity_first, carbon_first, ...
  profile-<name>   DMOS L1+L2 with a legacy profile of score_weights
                   (DMOS_SCORE_PROFILE), e.g. profile-latonly
  legacy-default   code defaults, no switches: legacy profile, demand floor on
                   (old matrix/sweep "l1l2" without environment; runs of 7-10 June)
  legacy-gate1     legacy profile, floor off, gate g=1        (old run_one "l1l2")
  legacy-nofloor   legacy profile, floor off, gate off        (old "l1l2nofloor")
  reserved for E1, not implemented yet: hpa-none hpa-local karmada-fhpa karmada-budget

REGIME — what load:  <demand>[-<hot>][-p<peak>]
  demand  uni | bal | s50 | s65 | s75 | con     share of the hot cluster in the peak
  hot     cluster that receives that share, by region: de (default), fr, pl,
          us, sg, no — taken from .cache/test-env.json (cluster order)
  peak    peak rate in req/s (default 350); above 350 the k6 VU factor is 30
  The canonical form omits the defaults:  con, bal-sg, con-pl, s50-p500, con-p400

USAGE
  python experiments/run_naming.py list
  python experiments/run_naming.py show  ARM REGIME        # what a cell runs
  python experiments/run_naming.py plan  --arms "off l2 l1l2" --regimes "bal con"
  python experiments/run_naming.py shell ARM REGIME        # used by run_common.sh
  (all accept --env-json PATH, default .cache/test-env.json if present)
Exit code 2 = invalid name (the message says why and what to use instead).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent
WEIGHTS_YAML = REPO / "config" / "weights.yaml"
DEFAULT_ENV_JSON = REPO / ".cache" / "test-env.json"

# ── regimes ──────────────────────────────────────────────────────────────────
# Per-cluster k6 weights in the peak phase; position 1 is the hot cluster.
DEMAND_WEIGHTS: Dict[str, Tuple[int, ...]] = {
    "uni": (1, 1, 1, 1, 1, 1),
    "bal": (40, 15, 12, 12, 11, 10),
    "s50": (50, 10, 10, 10, 10, 10),
    "s65": (65, 7, 7, 7, 7, 7),
    "s75": (75, 5, 5, 5, 5, 5),
    "con": (85, 3, 3, 3, 3, 3),
}
DEMAND_ORDER = ["uni", "bal", "s50", "s65", "s75", "con"]
# Hot-cluster share (x axis of the skew plots).
DEMAND_SHARE: Dict[str, float] = {
    "uni": 0.167, "bal": 0.40, "s50": 0.50, "s65": 0.65, "s75": 0.75, "con": 0.85}
DEMAND_LABELS: Dict[str, str] = {
    "uni": "Uniform", "bal": "Balanced", "s50": "Skew 50%", "s65": "Skew 65%",
    "s75": "Skew 75%", "con": "Concentrated"}
DEFAULT_PEAK = 350
HIGH_PEAK_VU_FACTOR = 30          # k6 VU factor above DEFAULT_PEAK (default 20)
DEFAULT_REGIONS = ["DE", "FR", "PL", "US-VA", "SG", "NO"]   # paper Tab. III order

_REGIME_RE = re.compile(r"^(uni|bal|s50|s65|s75|con)(?:-([a-z]{2,3}))?(?:-p(\d{2,4}))?$")

# ── arms ─────────────────────────────────────────────────────────────────────
BASE_ARMS = {
    "off": "DMOS not running: 1 static replica per cluster",
    "l2": "DMOS Level 2 only (--no-scheduling): N/K per cluster",
    "l1l2": ("DMOS L1+L2 as run in the NCA 2026 campaign: gate g=2, Phi_loc=Phi_net, "
             "Phi_pre off, weights lat/cap/net/car 0.25/0.20/0.25/0.30, floor off"),
}
# Environment switches of the NCA 2026 campaign arm (read back from the DMOS
# logs of 15-17 June 2026: every DMOS start printed "DEMAND-GATE ON (eps=0.05,
# gamma=2.0) | COERENTE: Phi_loc=Phi_net puro, Phi_pre OFF" with weights
# lat=0.25 cap=0.2 geo_load=0.25 carbon=0.3, and the peak allocations show
# no demand floor).
L1L2_ENV = ("DMOS_DEMAND_FLOOR=0 DMOS_SCORE_GATE=1 DMOS_SCORE_GATE_GAMMA=2 "
            "DMOS_LOCALITY_PURE_NET=1 DMOS_WEIGHTS_COHERENT=0.25,0.20,0.25,0.30")
LEGACY_ARMS = {
    "legacy-default": ("",
                       "code defaults, no switches: legacy profile + demand floor on "
                       "(old matrix/sweep 'l1l2' without environment)"),
    "legacy-gate1": ("DMOS_DEMAND_FLOOR=0 DMOS_SCORE_GATE=1",
                     "legacy profile, floor off, gate g=1 (old run_one 'l1l2')"),
    "legacy-nofloor": ("DMOS_DEMAND_FLOOR=0",
                       "legacy profile, floor off, gate off: additive score (old 'l1l2nofloor')"),
}
RESERVED_ARMS = {
    "hpa-none": "E1: per-cluster HPA, frontend affinity none",
    "hpa-local": "E1: per-cluster HPA, frontend affinity local",
    "karmada-fhpa": "E1: Karmada with FederatedHPA",
    "karmada-budget": "E1: Karmada with the replica budget computed by DMOS",
}
# Old names -> new name. They are refused (not silently mapped) so that a
# label never changes meaning between campaigns.
RENAMED_ARMS = {
    "l1l2gate": "legacy-gate1",
    "l1l2coh": "l1l2",                # the NCA 2026 configuration
    "l1l2nofloor": "legacy-nofloor",
    "l1l2floor": "legacy-default",    # floor on is the code default
    "latonly": "profile-latonly",
    "netonly": "profile-netonly",
    "carbon": "profile-carbon",
}
RENAMED_REGIMES = {"con400": "con-p400"}

_ARM_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class NamingError(ValueError):
    """Invalid arm or regime name."""


@dataclass
class Cell:
    arm: str
    regime: str                       # canonical form
    label: str
    kind: str                         # off | dmos
    dmos_args: str = ""
    dmos_env: str = ""
    k6_env: str = ""
    run_args: str = ""
    arm_description: str = ""
    demand: str = ""
    hot_region: str = ""
    hot_cluster: str = ""
    peak_rps: int = DEFAULT_PEAK
    weights: Tuple[int, ...] = field(default_factory=tuple)


# ── configuration sources ────────────────────────────────────────────────────

def _weights_yaml() -> dict:
    try:
        import yaml
        return yaml.safe_load(WEIGHTS_YAML.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def score_models() -> Dict[str, str]:
    """Score models of config/weights.yaml -> description."""
    models = _weights_yaml().get("score_models") or {}
    return {str(k): str((v or {}).get("description", "")) for k, v in models.items()}


def legacy_profiles() -> List[str]:
    prof = _weights_yaml().get("score_weights") or {}
    return sorted(str(k) for k, v in prof.items() if k != "active" and isinstance(v, dict))


def fleet(env_json: Optional[Path] = None) -> List[Tuple[str, str]]:
    """[(cluster name, region)] in fleet order (= k6 weight order)."""
    path = Path(env_json) if env_json else DEFAULT_ENV_JSON
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
        out = [(str(c["name"]), str(c.get("region") or "")) for c in env.get("clusters", [])]
        if out:
            return out
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return [(f"ms-cluster-{i:02d}", r) for i, r in enumerate(DEFAULT_REGIONS, start=1)]


def region_short(region: str) -> str:
    """'US-VA' -> 'us', 'DE' -> 'de'."""
    return region.split("-")[0].lower()


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_regime(regime: str) -> Dict[str, object]:
    """'con-sg-p500' -> {'demand': 'con', 'hot': 'sg', 'peak': 500}.

    hot is None when omitted (= first cluster of the fleet), peak defaults to
    DEFAULT_PEAK. Legacy regimes of archived runs (con400) are accepted here,
    so the analysis can still read them; orchestrators use resolve()."""
    if regime in RENAMED_REGIMES:
        regime = RENAMED_REGIMES[regime]
    m = _REGIME_RE.match(regime or "")
    if not m:
        raise NamingError(
            f"unknown regime '{regime}': expected <demand>[-<hot>][-p<peak>] with "
            f"demand in {DEMAND_ORDER} (e.g. con, bal-sg, s50-p500)")
    demand, hot, peak = m.group(1), m.group(2), m.group(3)
    return {"demand": demand, "hot": hot, "peak": int(peak) if peak else DEFAULT_PEAK}


def regime_share(regime: str) -> Optional[float]:
    try:
        return DEMAND_SHARE[str(parse_regime(regime)["demand"])]
    except NamingError:
        return None


def regime_label(regime: str) -> str:
    try:
        p = parse_regime(regime)
    except NamingError:
        return regime
    text = DEMAND_LABELS[str(p["demand"])]
    if p["hot"]:
        text += f", hot {str(p['hot']).upper()}"
    if p["peak"] != DEFAULT_PEAK:
        text += f" @{p['peak']}rps"
    return text


def _resolve_arm(arm: str) -> Tuple[str, str, str, str]:
    """-> (kind, dmos_args, dmos_env, description)."""
    if arm in RENAMED_ARMS:
        raise NamingError(f"arm '{arm}' was renamed: use '{RENAMED_ARMS[arm]}' "
                          f"(a name keeps one meaning across campaigns)")
    if arm in RESERVED_ARMS:
        raise NamingError(f"arm '{arm}' ({RESERVED_ARMS[arm]}) is reserved: not implemented yet")
    if arm == "off":
        return "off", "", "", BASE_ARMS[arm]
    if arm == "l2":
        return "dmos", "--no-scheduling", "", BASE_ARMS[arm]
    if arm == "l1l2":
        return "dmos", "", L1L2_ENV, BASE_ARMS[arm]
    if arm in LEGACY_ARMS:
        env, desc = LEGACY_ARMS[arm]
        return "dmos", "", env, desc
    if arm.startswith("profile-"):
        name = arm[len("profile-"):]
        profiles = legacy_profiles()
        if profiles and name not in profiles:
            raise NamingError(f"unknown legacy profile '{name}': available {profiles}")
        return "dmos", "", f"DMOS_SCORE_PROFILE={name}", f"legacy profile '{name}' of score_weights"
    if not _ARM_RE.match(arm) or "__" in arm:
        raise NamingError(f"invalid arm name '{arm}'")
    models = score_models()
    if arm in models:
        return "dmos", "", f"DMOS_SCORE_MODEL={arm}", f"score model '{arm}': {models[arm]}"
    raise NamingError(
        f"unknown arm '{arm}'. Valid: {sorted(BASE_ARMS)} + score models {sorted(models)} "
        f"+ profile-<{'|'.join(legacy_profiles()) or 'name'}> + {sorted(LEGACY_ARMS)}")


def resolve(arm: str, regime: str, env_json: Optional[Path] = None) -> Cell:
    """Full configuration of one cell (raises NamingError on invalid names)."""
    if regime in RENAMED_REGIMES:
        raise NamingError(f"regime '{regime}' was renamed: use '{RENAMED_REGIMES[regime]}'")
    kind, dmos_args, dmos_env, desc = _resolve_arm(arm)
    p = parse_regime(regime)
    demand, hot, peak = str(p["demand"]), p["hot"], int(p["peak"])  # type: ignore[arg-type]

    clusters = fleet(env_json)
    weights = list(DEMAND_WEIGHTS[demand])
    if len(clusters) != len(weights):
        raise NamingError(f"the regimes are defined for a {len(weights)}-cluster fleet, "
                          f"the test environment has {len(clusters)} clusters")
    shorts = [region_short(r) for _, r in clusters]
    default_hot = shorts[0]
    if hot is not None and hot not in shorts:
        raise NamingError(f"unknown hot cluster '{hot}': use one of {shorts}")
    if hot is not None and demand == "uni":
        raise NamingError("regime 'uni' has no hot cluster")
    idx = shorts.index(hot) if hot else 0
    weights[0], weights[idx] = weights[idx], weights[0]
    if not 1 <= peak <= 5000:
        raise NamingError(f"peak {peak} req/s out of range")

    canon = demand
    if hot and hot != default_hot:
        canon += f"-{hot}"
    if peak != DEFAULT_PEAK:
        canon += f"-p{peak}"

    k6 = [f"DMOS_SPIKE_W{i}={w}" for i, w in enumerate(weights, start=1)]
    k6.append(f"DMOS_PEAK_RPS={peak}")
    k6.append(f"DMOS_SPIKE_REGION={clusters[idx][0]}")
    if peak > DEFAULT_PEAK:
        k6.append(f"DMOS_VU_FACTOR={HIGH_PEAK_VU_FACTOR}")

    return Cell(arm=arm, regime=canon, label=f"dmos-{arm}__{canon}", kind=kind,
                dmos_args=dmos_args, dmos_env=dmos_env, k6_env=" ".join(k6),
                arm_description=desc, demand=demand, hot_region=clusters[idx][1],
                hot_cluster=clusters[idx][0], peak_rps=peak, weights=tuple(weights))


def plan(arms: Sequence[str], regimes: Sequence[str],
         env_json: Optional[Path] = None) -> List[Cell]:
    """Cells in run order: for each regime, all arms one after the other."""
    cells, errors = [], []
    for reg in regimes:
        for arm in arms:
            try:
                cells.append(resolve(arm, reg, env_json))
            except NamingError as e:
                errors.append(f"{arm} x {reg}: {e}")
    if errors:
        raise NamingError("\n".join(dict.fromkeys(errors)))
    return cells


# ── CLI ──────────────────────────────────────────────────────────────────────

def _shell(cell: Cell) -> str:
    pairs = {"CELL_LABEL": cell.label, "CELL_KIND": cell.kind,
             "CELL_DMOS_ARGS": cell.dmos_args, "CELL_DMOS_ENV": cell.dmos_env,
             "CELL_K6_ENV": cell.k6_env, "CELL_RUN_ARGS": cell.run_args,
             "CELL_DESC": cell.arm_description}
    return "\n".join(f"{k}={shlex.quote(v)}" for k, v in pairs.items()) + "\n"


def _show(cell: Cell) -> str:
    return (f"{cell.label}\n"
            f"  arm     {cell.arm}: {cell.arm_description}\n"
            f"  regime  {cell.regime}: {regime_label(cell.regime)} — hot {cell.hot_cluster} "
            f"({cell.hot_region}), weights {'/'.join(map(str, cell.weights))}, peak {cell.peak_rps} req/s\n"
            f"  DMOS    {('python -m src.dmos_main ' + cell.dmos_args).strip() if cell.kind == 'dmos' else 'not running'}"
            f"{('  env: ' + cell.dmos_env) if cell.dmos_env else ''}\n"
            f"  k6 env  {cell.k6_env}\n")


def _list(env_json: Optional[Path]) -> str:
    lines = ["ARMS"]
    lines += [f"  {k:<16} {v}" for k, v in BASE_ARMS.items()]
    lines += [f"  {k:<16} score model: {v}" for k, v in score_models().items()]
    lines.append(f"  profile-<name>   legacy profile: {', '.join(legacy_profiles())}")
    lines += [f"  {k:<16} {d}" for k, (_, d) in LEGACY_ARMS.items()]
    lines += [f"  {k:<16} reserved — {v}" for k, v in RESERVED_ARMS.items()]
    lines += ["", "REGIMES  <demand>[-<hot>][-p<peak>]"]
    for d in DEMAND_ORDER:
        lines.append(f"  {d:<5} share {DEMAND_SHARE[d]:.2f}  weights {'/'.join(map(str, DEMAND_WEIGHTS[d]))}")
    shorts = [f"{region_short(r)} ({n})" for n, r in fleet(env_json)]
    lines.append(f"  hot   {', '.join(shorts)}   (default: the first)")
    lines.append(f"  peak  p<req/s>, default {DEFAULT_PEAK}")
    lines += ["", "RENAMED  " + ", ".join(f"{o} -> {n}" for o, n in {**RENAMED_ARMS, **RENAMED_REGIMES}.items())]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--env-json", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("show", "shell"):
        sp = sub.add_parser(name)
        sp.add_argument("arm")
        sp.add_argument("regime")
    sp = sub.add_parser("plan")
    sp.add_argument("--arms", required=True, help='space-separated, e.g. "off l2 l1l2"')
    sp.add_argument("--regimes", required=True, help='space-separated, e.g. "uni bal con"')
    sp.add_argument("--labels-only", action="store_true")
    args = ap.parse_args(argv)

    # UTF-8 and plain \n also on Windows: the shell output is eval'd by bash and
    # the model descriptions contain non-ASCII characters.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", newline="\n")
        except (AttributeError, ValueError):
            pass
    out = sys.stdout
    try:
        if args.cmd == "list":
            out.write(_list(args.env_json))
        elif args.cmd == "shell":
            out.write(_shell(resolve(args.arm, args.regime, args.env_json)))
        elif args.cmd == "show":
            out.write(_show(resolve(args.arm, args.regime, args.env_json)))
        elif args.cmd == "plan":
            cells = plan(args.arms.split(), args.regimes.split(), args.env_json)
            for c in cells:
                out.write(c.label + "\n" if args.labels_only else _show(c))
            if not args.labels_only:
                out.write(f"{len(cells)} cells\n")
    except NamingError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
