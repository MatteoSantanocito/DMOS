#!/usr/bin/env python3
"""
write_manifest.py — the provenance manifest of ONE flash-crowd run.

Called by run_flash_crowd.sh in two phases:

  start   right after the run directory is created (before k6):
          k6/test configuration, arm/regime, git state (+ diff when the repo
          is dirty), host, k6 version and script hash, copy of the config
          files actually used.
  finish  after k6 and the collector are done (and after the post-run
          testbed snapshot when the harness takes one): links the DMOS
          effective configuration, the testbed snapshots, the config_id the
          collector observed on /metrics, the k6 outcome, and computes the
          provenance CHECKS (errors invalidate the run, warnings are noted).

Backward compatible: every key of the former schema-1 manifest is kept with
the same meaning (paper_lib / paper_report read `config` and `git.commit`).

Run directory layout written by the harness (see docs/RUN-PROVENANCE.md):
  manifest.json          this file
  dmos_config.json       effective DMOS configuration (src/utils/provenance.py)
  dmos.log               DMOS stdout/stderr (when the harness started DMOS)
  env_state_pre.json     testbed snapshot before the run (capture_env_state.py)
  env_state_post.json    testbed snapshot after the run
  git_diff.patch         uncommitted code changes, if the repo was dirty
  config_snapshot/       copy of config/*.yaml used (local.yaml sanitised)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILES = ("weights.yaml", "services.yaml", "carbon.yaml")
# local.yaml keys never copied into the run directory (addresses, credentials)
SENSITIVE_CLUSTER_KEYS = ("ip", "kubeconfig_path", "server", "estimator_url")
SENSITIVE_TOP_KEYS = ("ssh", "alertmanager")


# ── helpers ──────────────────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git_bytes(*args: str) -> Optional[bytes]:
    """Raw git output (no decoding: diffs contain UTF-8 and binary hunks)."""
    try:
        p = subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                           capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def git(*args: str) -> Optional[str]:
    out = git_bytes(*args)
    return None if out is None else out.decode("utf-8", errors="replace")


def num(text: str):
    """argparse type: keep ints as ints (schema-1 wrote 350, not 350.0)."""
    try:
        return int(text)
    except ValueError:
        return float(text)


def parse_arm_regime(label: str) -> Dict[str, Optional[str]]:
    """Same convention as paper_lib.parse_arm_regime: '<arm>__<regime>'."""
    if "__" in label:
        arm, regime = label.split("__", 1)
        return {"arm": arm, "regime": regime}
    return {"arm": label, "regime": None}


# ── start ────────────────────────────────────────────────────────────────────

def git_block(run_dir: Path) -> Dict[str, Any]:
    commit = (git("rev-parse", "HEAD") or "unknown").strip()
    porcelain = git("status", "--porcelain") or ""
    lines = [l for l in porcelain.splitlines() if l.strip()]
    tracked = [l[3:] for l in lines if not l.startswith("??")]
    untracked = [l[3:] for l in lines if l.startswith("??")]
    diff_file = None
    if tracked:
        diff = git_bytes("diff", "HEAD", "--binary")
        if diff:
            # bytes as produced by git: `git apply git_diff.patch` restores the code
            (run_dir / "git_diff.patch").write_bytes(diff)
            diff_file = "git_diff.patch"
    # New files not yet committed are not in the diff: copy them too, so that
    # commit + git_diff.patch + untracked_snapshot/ rebuild the exact code.
    copied, skipped = copy_untracked(untracked, run_dir / "untracked_snapshot")
    return {
        "commit": commit,
        "dirty_files": len(lines),            # schema-1 meaning: a count
        "branch": (git("rev-parse", "--abbrev-ref", "HEAD") or "").strip() or None,
        "dirty_paths": tracked,
        "untracked_paths": untracked,
        "diff_file": diff_file,
        "untracked_snapshot": "untracked_snapshot" if copied else None,
        "untracked_skipped": skipped,
    }


UNTRACKED_MAX_FILE = 1_000_000     # bytes per file
UNTRACKED_MAX_TOTAL = 20_000_000   # bytes per run


def copy_untracked(paths: List[str], dst: Path):
    """Copy untracked (non-ignored) files, expanding directories, size-capped."""
    files: List[Path] = []
    for p in paths:
        src = REPO_ROOT / p.rstrip("/")
        if src.is_dir():
            files += [f for f in sorted(src.rglob("*")) if f.is_file()]
        elif src.is_file():
            files.append(src)
    copied, skipped, total = 0, [], 0
    for f in files:
        rel = f.relative_to(REPO_ROOT).as_posix()
        size = f.stat().st_size
        if size > UNTRACKED_MAX_FILE or total + size > UNTRACKED_MAX_TOTAL:
            skipped.append(rel)
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
        total += size
    return copied, skipped


def sanitized_local_yaml(src: Path) -> Optional[str]:
    """local.yaml without addresses/credentials (needs PyYAML; else skipped)."""
    try:
        import yaml  # noqa: WPS433 (optional dependency)
    except ImportError:
        return None
    try:
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    for k in SENSITIVE_TOP_KEYS:
        data.pop(k, None)
    clusters = data.get("clusters")
    items = clusters.values() if isinstance(clusters, dict) else (clusters or [])
    for c in items:
        if isinstance(c, dict):
            for k in SENSITIVE_CLUSTER_KEYS:
                c.pop(k, None)
    return "# sanitised copy (addresses/credentials removed) of config/local.yaml\n" + \
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def snapshot_config(run_dir: Path) -> Dict[str, Any]:
    dst = run_dir / "config_snapshot"
    dst.mkdir(exist_ok=True)
    files: Dict[str, Optional[str]] = {}
    for name in CONFIG_FILES:
        src = REPO_ROOT / "config" / name
        if src.is_file():
            shutil.copy2(src, dst / name)
            files[name] = sha256_file(src)
    local = REPO_ROOT / "config" / "local.yaml"
    if local.is_file():
        files["local.yaml"] = sha256_file(local)   # hash of the ORIGINAL file
        text = sanitized_local_yaml(local)
        if text is not None:
            (dst / "local.sanitized.yaml").write_text(text, encoding="utf-8")
    return {"dir": "config_snapshot", "sha256": files}


def cmd_start(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = run_dir.name.split("_", 2)
    parsed = parse_arm_regime(a.label)
    arm_env, regime_env = os.environ.get("DMOS_ARM"), os.environ.get("DMOS_REGIME")
    k6_script = REPO_ROOT / "experiments" / "k6_flash_crowd.js"
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        # ── schema-1 keys (unchanged) ────────────────────────────────────
        "run_label": a.label,
        "timestamp": "_".join(ts[:2]) if len(ts) >= 2 else None,
        "timestamp_iso": now_iso(),
        "k6_mode": a.k6_mode,
        "k6_remote": a.k6_remote,
        "no_dmos": 1 if a.no_dmos else 0,
        "config": {
            "peak_rps": a.peak_rps,
            "baseline_rps": a.baseline_rps,
            "spike_region": a.spike_region,
            "spike_weights": a.spike_weights,
            "phase_times_s": a.phase_times,
            "test_duration_s": a.test_duration,
            "vu_factor": a.vu_factor,
        },
        "git": git_block(run_dir),
        # ── schema-2 additions ───────────────────────────────────────────
        "arm": arm_env or parsed["arm"],
        "regime": regime_env or parsed["regime"],
        "arm_source": "env" if arm_env else "label",
        "k6": {
            "mode": a.k6_mode,
            "remote": a.k6_remote,
            "version": a.k6_version,
            "args": a.k6_args,
            "script": "experiments/k6_flash_crowd.js",
            "script_sha256": sha256_file(k6_script),
            "raw_events": bool(a.raw_out),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "config_snapshot": snapshot_config(run_dir),
        "timing": {"started_at": now_iso(), "finished_at": None},
    }
    write_json(run_dir / "manifest.json", manifest)
    if manifest["git"]["dirty_paths"]:
        print(f"[manifest] repo dirty ({len(manifest['git']['dirty_paths'])} tracked files): "
              f"diff saved to git_diff.patch")
    return 0


# ── finish ───────────────────────────────────────────────────────────────────

def observed_config_ids(run_dir: Path) -> Dict[str, int]:
    """config_id → number of collector snapshots that reported it."""
    seen: Dict[str, int] = {}
    for name in ("dmos_live.jsonl",):
        p = run_dir / name
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    cid = json.loads(line).get("dmos_config_id")
                except ValueError:
                    continue
                if cid:
                    seen[cid] = seen.get(cid, 0) + 1
    return seen


def env_summary(snap: Optional[Dict[str, Any]], fname: str) -> Optional[Dict[str, Any]]:
    if snap is None:
        return None
    v = snap.get("verdicts", {})
    return {
        "file": fname,
        "captured_at": snap.get("captured_at"),
        "netem_ok": v.get("netem_ok"),
        "netem_problems": v.get("netem_problems"),
        "affinity": v.get("affinity"),
        "endpoint_sync": v.get("endpoint_sync"),
        "cilium_images": v.get("cilium_images"),
        "any_hpa": v.get("any_hpa"),
        "deployments_not_ready": v.get("deployments_not_ready"),
        "loadgenerator_replicas": v.get("loadgenerator_replicas"),
        "loadgenerator_running": v.get("loadgenerator_running"),
        "collection_errors": v.get("collection_errors"),
        "expectations": snap.get("expectations"),
    }


def compute_checks(m: Dict[str, Any], run_dir: Path) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []

    def add(level: str, code: str, msg: str) -> None:
        checks.append({"level": level, "code": code, "message": msg})

    dmos = m.get("dmos", {})
    env = m.get("env_state", {})
    seen = dmos.get("config_ids_observed") or {}

    if not m.get("no_dmos"):
        if not dmos.get("config_file"):
            add("error", "dmos_config_missing",
                "DMOS arm without dmos_config.json: the algorithm of this run is unknown")
        if not seen:
            add("warn", "config_id_not_observed",
                "the collector never saw dmos_config_info on /metrics")
        elif len(seen) > 1:
            add("error", "config_changed_during_run",
                f"more than one DMOS configuration served this run: {seen}")
        elif dmos.get("config_id") and dmos["config_id"] not in seen:
            add("error", "config_mismatch",
                f"dmos_config.json says {dmos['config_id']}, /metrics said {list(seen)}")
        exp = dmos.get("expected_config_id")
        if exp and dmos.get("config_id") and exp != dmos["config_id"]:
            add("error", "config_precheck_mismatch",
                f"pre-run check built {exp}, the DMOS process ran {dmos['config_id']}")
        if not dmos.get("log_file"):
            add("warn", "dmos_log_missing", "DMOS log not stored in the run directory")
        fb = {c: s for c, s in (dmos.get("carbon_sources") or {}).items()
              if s and not str(s).startswith("trace:")}
        if fb:
            add("warn", "carbon_trace_fallback",
                f"no carbon trace for {sorted(fb)}: using {sorted(set(fb.values()))} "
                "(see config/carbon.yaml trace.zone_aliases)")
    elif seen:
        add("error", "dmos_running_in_off_arm",
            f"a DMOS process answered /metrics during a --no-dmos run: {seen}")

    pre, post = env.get("pre"), env.get("post")
    if pre is None:
        add("warn", "env_pre_missing", "no testbed snapshot before the run")
    elif pre.get("netem_ok") is False:
        add("error", "netem_invalid_pre", f"netem not in place before the run: {pre.get('netem_problems')}")
    if post is not None and post.get("netem_ok") is False:
        add("error", "netem_lost_during_run",
            f"netem not in place after the run: {post.get('netem_problems')}")
    if pre and post and pre.get("affinity") != post.get("affinity"):
        add("warn", "affinity_changed_during_run",
            f"Service affinity differs pre/post: {pre.get('affinity')} -> {post.get('affinity')}")
    lg = [(p, s) for p, s in (("pre", pre), ("post", post)) if s and s.get("loadgenerator_running")]
    if lg:
        add("error", "loadgenerator_running",
            "Online Boutique loadgenerator active (" + "/".join(p for p, _ in lg) + "): internal "
            f"traffic not from k6 — replicas {lg[0][1].get('loadgenerator_replicas')}")
    if pre and (pre.get("expectations") or {}).get("violations"):
        add("error", "env_expectation_violated",
            "; ".join(pre["expectations"]["violations"]))

    if m.get("git", {}).get("dirty_paths"):
        add("warn", "repo_dirty", "code not committed: see git_diff.patch")

    out = m.get("outcome", {})
    if not out.get("k6_summary"):
        add("error", "k6_summary_missing", "flash_crowd_summary.json missing or invalid")
    elif out.get("k6_run_valid") is False:
        add("error", "k6_vu_starved",
            f"k6 dropped iterations ({out.get('k6_dropped_pct')}): offered load not delivered")
    return checks


def cmd_finish(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir)
    mp = run_dir / "manifest.json"
    m = read_json(mp)
    if m is None:
        print(f"[manifest] {mp} missing or invalid — start phase not run?", file=sys.stderr)
        return 1

    # finish may run twice (run_flash_crowd.sh, then the orchestrator after the
    # post-run snapshot): values not passed again are kept from the first call.
    prev_dmos, prev_out = m.get("dmos", {}) or {}, m.get("outcome", {}) or {}
    if a.dmos_config_source is None:
        a.dmos_config_source = prev_dmos.get("config_source")
    if a.k6_exit is None:
        a.k6_exit = prev_out.get("k6_exit")

    cfg = read_json(run_dir / "dmos_config.json")
    expected = read_json(run_dir / "dmos_config_expected.json")
    m["dmos"] = {
        "enabled": not m.get("no_dmos"),
        "config_file": "dmos_config.json" if cfg else None,
        "config_id": (cfg or {}).get("config_id"),
        "expected_config_id": (expected or {}).get("config_id"),
        "score_model": ((cfg or {}).get("score_model") or {}).get("name"),
        "summary": (cfg or {}).get("summary"),
        "warm_formula": ((cfg or {}).get("level1", {}).get("warm", {}) or {}).get("formula"),
        "cold_formula": ((cfg or {}).get("level1", {}).get("cold", {}) or {}).get("formula"),
        "config_source": a.dmos_config_source,
        "log_file": "dmos.log" if (run_dir / "dmos.log").is_file() else None,
        "config_ids_observed": observed_config_ids(run_dir),
        "carbon_sources": ((cfg or {}).get("carbon") or {}).get("sources"),
    }
    m["env_state"] = {
        "pre": env_summary(read_json(run_dir / "env_state_pre.json"), "env_state_pre.json"),
        "post": env_summary(read_json(run_dir / "env_state_post.json"), "env_state_post.json"),
    }
    summ = read_json(run_dir / "flash_crowd_summary.json")
    validity = (summ or {}).get("validity", {})
    m["outcome"] = {
        "k6_exit": a.k6_exit,
        "k6_summary": bool(summ),
        "k6_run_valid": validity.get("run_valid"),
        "k6_dropped_pct": validity.get("dropped_pct"),
        "raw_events_file": "k6_events.json" if (run_dir / "k6_events.json").is_file() else None,
    }
    m.setdefault("timing", {})["finished_at"] = now_iso()
    checks = compute_checks(m, run_dir)
    m["provenance"] = {
        "checks": checks,
        "errors": sum(c["level"] == "error" for c in checks),
        "warnings": sum(c["level"] == "warn" for c in checks),
        "ok": not any(c["level"] == "error" for c in checks),
    }
    write_json(mp, m)

    p = m["provenance"]
    print(f"[manifest] provenance: {'OK' if p['ok'] else 'INVALID'} "
          f"({p['errors']} errors, {p['warnings']} warnings)"
          + (f" | {m['dmos']['summary']}" if m["dmos"]["summary"] else ""))
    for c in checks:
        print(f"[manifest]   {c['level'].upper():5s} {c['code']}: {c['message']}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Write/complete a run's manifest.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--no-dmos", action="store_true")
    s.add_argument("--k6-mode", default="")
    s.add_argument("--k6-remote", default="")
    s.add_argument("--k6-version", default="")
    s.add_argument("--k6-args", default="")
    s.add_argument("--raw-out", type=int, default=1)
    s.add_argument("--peak-rps", type=num)
    s.add_argument("--baseline-rps", type=num)
    s.add_argument("--spike-region", default="")
    s.add_argument("--spike-weights", default="")
    s.add_argument("--phase-times", default="")
    s.add_argument("--test-duration", type=num)
    s.add_argument("--vu-factor", type=num)

    f = sub.add_parser("finish")
    f.add_argument("--run-dir", required=True)
    f.add_argument("--k6-exit", type=int, default=None)
    f.add_argument("--dmos-config-source", default=None,
                   help="how dmos_config.json was obtained (dump|http|none)")

    a = ap.parse_args(argv)
    return cmd_start(a) if a.cmd == "start" else cmd_finish(a)


if __name__ == "__main__":
    sys.exit(main())
