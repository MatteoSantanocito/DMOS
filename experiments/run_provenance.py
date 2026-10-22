#!/usr/bin/env python3
"""
run_provenance.py — inspect and audit the provenance of flash-crowd runs.

  show  RUN_DIR             what ran: DMOS configuration (config_id, formula,
                            gate, weights, L2), testbed state pre/post (netem,
                            affinity, Cilium, HPA), git, k6, provenance checks
  audit [RESULTS_DIR]       every run in RESULTS_DIR (default experiments/results):
                            config_id, validity, and labels that MIX different
                            configurations (same label, different algorithm)
  diff  RUN_A RUN_B         differences between the two DMOS configurations
                            and testbed states

Exit codes: show/diff 0; audit 0 if no invalid run and no mixed label, 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"
RUN_DIR_RE = re.compile(r"^(\d{8})_(\d{6})_(.+)$")
# Sections of dmos_config.json compared by `diff` (same as the fingerprint)
FINGERPRINT_SECTIONS = ("modes", "level1", "level2", "safeguards", "coscaling", "carbon", "clusters")


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_run(run_dir: Path) -> Dict[str, Any]:
    return {
        "dir": run_dir,
        "manifest": _read(run_dir / "manifest.json") or {},
        "dmos_config": _read(run_dir / "dmos_config.json"),
        "env_pre": _read(run_dir / "env_state_pre.json"),
        "env_post": _read(run_dir / "env_state_post.json"),
    }


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def run_status(run: Dict[str, Any]) -> Tuple[str, List[str]]:
    """'ok' | 'invalid' | 'aborted' | 'legacy' (+ reasons)."""
    prov = run["manifest"].get("provenance")
    if prov is None:
        violations = ((run["env_pre"] or {}).get("expectations") or {}).get("violations")
        if not run["manifest"] and violations:
            return "aborted", [f"not started, testbed gate: {v}" for v in violations]
        return "legacy", ["run without provenance (pre schema-2 harness)"]
    errs = [f"{c['code']}: {c['message']}" for c in prov.get("checks", []) if c["level"] == "error"]
    return ("ok" if prov.get("ok") else "invalid"), errs


# ── show ─────────────────────────────────────────────────────────────────────

def _env_lines(snap: Optional[Dict[str, Any]], title: str) -> List[str]:
    if not snap:
        return [f"  {title}: (not captured)"]
    v = snap.get("verdicts", {})
    lines = [f"  {title} ({snap.get('captured_at')}):"]
    lines.append("    netem       : " + ", ".join(f"{n}={x}" for n, x in v.get("netem", {}).items()))
    lines.append("    affinity    : " + ", ".join(f"{s}={a}" for s, a in v.get("affinity", {}).items()))
    lines.append(f"    endpoint-sync: {v.get('endpoint_sync')}")
    imgs = sorted(set(str(i) for i in (v.get("cilium_images") or {}).values()))
    lines.append(f"    cilium      : {imgs}")
    lines.append(f"    HPA         : {({n: h for n, h in (v.get('hpa') or {}).items() if h}) or 'none'}")
    if v.get("deployments_not_ready"):
        lines.append(f"    not ready   : {v['deployments_not_ready']}")
    if v.get("collection_errors"):
        lines.append(f"    errors      : {v['collection_errors']}")
    viol = (snap.get("expectations") or {}).get("violations") or []
    for x in viol:
        lines.append(f"    VIOLATION   : {x}")
    return lines


def cmd_show(run_dir: Path) -> int:
    run = load_run(run_dir)
    m, cfg = run["manifest"], run["dmos_config"]
    status, _ = run_status(run)
    print(f"RUN      {run_dir.name}   [{status.upper()}]")
    print(f"  label  {m.get('run_label')}  arm={m.get('arm')}  regime={m.get('regime')}")
    t = m.get("timing", {})
    print(f"  time   {t.get('started_at') or m.get('timestamp_iso')} -> {t.get('finished_at')}")
    g = m.get("git", {})
    dirty = g.get("dirty_paths") or []
    print(f"  git    {str(g.get('commit'))[:12]} ({g.get('branch')})"
          + (f"  DIRTY {len(dirty)} files -> {g.get('diff_file')}" if dirty else "  clean"))
    c = m.get("config", {})
    k6 = m.get("k6", {})
    print(f"  k6     peak={c.get('peak_rps')} baseline={c.get('baseline_rps')} phases={c.get('phase_times_s')}"
          f" weights={c.get('spike_weights')} mode={m.get('k6_mode')} {k6.get('version') or ''}")

    print("DMOS")
    if m.get("no_dmos"):
        print("  not running (--no-dmos arm)")
    elif not cfg:
        print("  configuration UNKNOWN (no dmos_config.json)")
    else:
        l1, l2, sg = cfg["level1"], cfg["level2"], cfg["safeguards"]
        print(f"  config_id   {cfg['config_id']}   source={m.get('dmos', {}).get('config_source')}")
        print(f"  L1          {'on' if cfg['modes']['level1_scheduling'] else 'OFF (uniform N/K)'}"
              f" | demand floor {'on' if cfg['modes']['demand_floor'] else 'off'}"
              f" | weights from {l1['warm'].get('weights_source', 'profile')}"
              f" (profile {l1.get('score_profile')})")
        print(f"  warm score  {l1['warm']['formula']}")
        print(f"  cold score  {l1['cold']['formula']}   (first {l1['cold_start_seconds']} s)")
        print(f"  demand via  {', '.join(l1['warm']['demand_roles']) or 'none'}")
        ctl = l2.get("controller") or {}
        pd, pr = ctl.get("pd", {}), ctl.get("predictor", {})
        print(f"  L2          PD Kp={pd.get('kp')} Kd={pd.get('kd')} deadband={pd.get('deadband')}"
              f" raw=[{pd.get('output_min')},{pd.get('output_max')}] applied=±{l2.get('pd_applied_clamp')}"
              f" | predictor W={pr.get('window_s')}s H={pr.get('horizon_s')}s up={pr.get('trend_factor_up')}"
              f" | rate +{ctl.get('max_delta_up')}/-{ctl.get('max_delta_down')}")
        print(f"  safeguards  cooldown={sg['scale_down_cooldown_s']}s protect={sg['scale_up_protection_s']}s"
              f" budget={sg['entry_budget_up']}/{sg['entry_budget_down']}"
              f" backend={sg['backend_budget_up']}/{sg['backend_budget_down']}"
              f" deadzone={sg['dead_zone_pct']} grace={sg['startup_grace_s']}s"
              f" | coscale floor={cfg['coscaling']['colocation_floor']}")
        print(f"  env         {cfg.get('dmos_env')}")
    seen = (m.get("dmos") or {}).get("config_ids_observed")
    if seen is not None:
        print(f"  observed on /metrics during the run: {seen or 'nothing'}")

    print("TESTBED")
    for line in _env_lines(run["env_pre"], "pre-run") + _env_lines(run["env_post"], "post-run"):
        print(line)

    prov = m.get("provenance")
    print("PROVENANCE")
    if prov is None:
        print("  legacy run: no provenance checks")
    else:
        print(f"  {'OK' if prov['ok'] else 'INVALID'} — {prov['errors']} errors, {prov['warnings']} warnings")
        for ch in prov.get("checks", []):
            print(f"  {ch['level'].upper():5s} {ch['code']}: {ch['message']}")
    return 0


# ── audit ────────────────────────────────────────────────────────────────────

def iter_runs(results: Path, include_archives: bool) -> Iterable[Path]:
    dirs = [results]
    if include_archives:
        dirs += [d for d in sorted(results.iterdir()) if d.is_dir() and d.name.startswith("_")]
    for base in dirs:
        for d in sorted(base.iterdir()):
            if d.is_dir() and RUN_DIR_RE.match(d.name):
                yield d


def cmd_audit(results: Path, include_archives: bool, as_json: bool) -> int:
    rows, by_label = [], {}
    for d in iter_runs(results, include_archives):
        run = load_run(d)
        status, reasons = run_status(run)
        m, cfg = run["manifest"], run["dmos_config"]
        label = m.get("run_label") or RUN_DIR_RE.match(d.name).group(3)
        cid = (cfg or {}).get("config_id") or ("off" if m.get("no_dmos") else None)
        pre = (run["env_pre"] or {}).get("verdicts", {})
        row = {
            "run": str(d.relative_to(results)), "label": label, "status": status,
            "config_id": cid, "netem_pre_ok": pre.get("netem_ok"),
            "affinity_frontend": (pre.get("affinity") or {}).get("frontend"),
            "dirty": bool((m.get("git") or {}).get("dirty_paths")), "reasons": reasons,
        }
        rows.append(row)
        by_label.setdefault(label, set()).add(cid)

    mixed = {lab: sorted(map(str, ids)) for lab, ids in by_label.items()
             if len({i for i in ids if i is not None}) > 1}
    if as_json:
        print(json.dumps({"runs": rows, "mixed_labels": mixed}, indent=2))
    else:
        print(f"{'run':55s} {'status':8s} {'config_id':13s} {'netem':6s} {'aff(fe)':8s} dirty")
        for r in rows:
            print(f"{r['run'][:55]:55s} {r['status']:8s} {str(r['config_id'] or '-'):13s} "
                  f"{str(r['netem_pre_ok'] if r['netem_pre_ok'] is not None else '-'):6s} "
                  f"{str(r['affinity_frontend'] or '-'):8s} {'yes' if r['dirty'] else ''}")
            for why in r["reasons"]:
                if r["status"] in ("invalid", "aborted"):
                    print(f"{'':57s}-> {why}")
        n = len(rows)
        cnt = {s: sum(r["status"] == s for r in rows) for s in ("ok", "invalid", "aborted", "legacy")}
        print(f"\n{n} runs: {cnt['ok']} ok, {cnt['invalid']} invalid, {cnt['aborted']} aborted "
              f"(testbed gate), {cnt['legacy']} legacy (no provenance)")
        if mixed:
            print("\nLABELS MIXING DIFFERENT CONFIGURATIONS (same label, different algorithm):")
            for lab, ids in sorted(mixed.items()):
                print(f"  {lab}: {ids}")
    return 2 if (mixed or any(r["status"] == "invalid" for r in rows)) else 0


# ── diff ─────────────────────────────────────────────────────────────────────

def cmd_diff(a_dir: Path, b_dir: Path) -> int:
    a, b = load_run(a_dir), load_run(b_dir)
    print(f"A = {a_dir.name}\nB = {b_dir.name}")
    ca, cb = a["dmos_config"] or {}, b["dmos_config"] or {}
    print(f"config_id: A={ca.get('config_id')}  B={cb.get('config_id')}")
    fa = flatten({k: ca.get(k) for k in FINGERPRINT_SECTIONS})
    fb = flatten({k: cb.get(k) for k in FINGERPRINT_SECTIONS})
    diffs = [(k, fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    print(f"\nDMOS configuration: {len(diffs)} differences")
    for k, va, vb in diffs:
        print(f"  {k}\n      A: {va}\n      B: {vb}")
    va = (a["env_pre"] or {}).get("verdicts", {})
    vb = (b["env_pre"] or {}).get("verdicts", {})
    keys = ("netem", "affinity", "endpoint_sync", "cilium_images", "hpa")
    tdiff = [(k, va.get(k), vb.get(k)) for k in keys if va.get(k) != vb.get(k)]
    print(f"\nTestbed (pre-run): {len(tdiff)} differences")
    for k, x, y in tdiff:
        print(f"  {k}\n      A: {x}\n      B: {y}")
    ka, kb = a["manifest"].get("config", {}), b["manifest"].get("config", {})
    kdiff = [(k, ka.get(k), kb.get(k)) for k in sorted(set(ka) | set(kb)) if ka.get(k) != kb.get(k)]
    print(f"\nk6 configuration: {len(kdiff)} differences")
    for k, x, y in kdiff:
        print(f"  {k}: A={x}  B={y}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("run_dir", type=Path)
    a = sub.add_parser("audit")
    a.add_argument("results_dir", type=Path, nargs="?", default=DEFAULT_RESULTS)
    a.add_argument("--include-archives", action="store_true",
                   help="also scan results/_*/ (archives, quarantine, aborted)")
    a.add_argument("--json", action="store_true")
    d = sub.add_parser("diff"); d.add_argument("run_a", type=Path); d.add_argument("run_b", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "show":
        return cmd_show(args.run_dir)
    if args.cmd == "audit":
        return cmd_audit(args.results_dir, args.include_archives, args.json)
    return cmd_diff(args.run_a, args.run_b)


if __name__ == "__main__":
    sys.exit(main())
