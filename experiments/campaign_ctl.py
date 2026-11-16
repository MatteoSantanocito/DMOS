#!/usr/bin/env python3
"""Helpers of run_campaign.sh: wait out the Cilium re-apply windows, check a run.

  python experiments/campaign_ctl.py wait --minutes 24
      sleeps until a cell of 24 min can run without overlapping a window in
      which K3s re-applies the Cilium chart (every 10 h on the Kubephos fleet,
      ~77 min to go through the six clusters; docs/KUBEPHOS.md)
  python experiments/campaign_ctl.py check --arm A --regime R --since EPOCH --log F [--rc N] [--prestop]
      finds the run folder of that cell started after EPOCH and prints one line
      "<folder>\\t<OK|BAD>\\t<reasons>"; exit 0 only for OK

Windows: DMOS_REAPPLY_ANCHOR (start of one observed window, ISO UTC; empty =
no windows), DMOS_REAPPLY_PERIOD_H (10), DMOS_REAPPLY_LEN_MIN (80).
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
BEFORE_S = 3 * 60       # a cell must end this long before a window opens
AFTER_S = 8 * 60        # and start this long after it closes
FLEET_OK = "ClusterMesh, etichette Hubble, Gateway e pod ok"   # fleet_gate, run_common.sh


def windows_cfg():
    anchor = os.environ.get("DMOS_REAPPLY_ANCHOR", "2026-10-29T01:22:00+00:00")
    if not anchor:
        return None
    return (datetime.fromisoformat(anchor).timestamp(),
            float(os.environ.get("DMOS_REAPPLY_PERIOD_H", "10")) * 3600,
            float(os.environ.get("DMOS_REAPPLY_LEN_MIN", "80")) * 60)


def next_safe_start(now: float, run_s: float, cfg) -> float:
    """First t >= now such that [t, t + run_s] stays clear of every window."""
    if cfg is None:
        return now
    anchor, period, length = cfg
    t = now
    while True:
        k = int((t - anchor) // period)
        for w in (anchor + k * period, anchor + (k + 1) * period):
            if t < w + length + AFTER_S and t + run_s + BEFORE_S > w:
                t = w + length + AFTER_S
                break
        else:
            return t


def cmd_wait(a) -> int:
    now = time.time()
    t = next_safe_start(now, a.minutes * 60, windows_cfg())
    if t > now:
        print(f"[campaign] Cilium re-apply window: waiting until "
              f"{datetime.fromtimestamp(t).strftime('%d/%m %H:%M')} local", flush=True)
        time.sleep(t - now)
    return 0


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_run(results: Path, arm: str, regime: str, since: float):
    """Newest folder <ts>_dmos-<arm>__<regime> started at or after `since`."""
    label = f"dmos-{arm}__{regime}"
    found = []
    for d in results.iterdir():
        m = re.match(r"(\d{8}_\d{6})_(.+)$", d.name)
        if d.is_dir() and m and m.group(2) == label:
            if datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp() >= since - 5:
                found.append(d)
    return max(found) if found else None


def lifecycle_services(root: Path):
    import yaml
    app = (yaml.safe_load((root / "config" / "services.yaml").read_text(encoding="utf-8"))
           or {}).get("application", {})
    return [app["entry_service"]] + list(app.get("co_scaled_services") or [])


def check_run(run: Path, log_text: str, rc: int, prestop_services, n_clusters: int):
    """Reasons why the run is not valid ([] = valid)."""
    bad = []
    if rc != 0:
        bad.append(f"run_one rc={rc}")
    if (run / "INVALID_DATAPLANE.txt").exists():
        bad.append("INVALID_DATAPLANE (data plane changed during the run)")
    s = _json(run / "flash_crowd_summary.json")
    if not s:
        bad.append("no flash_crowd_summary.json")
    elif not (s.get("validity") or {}).get("run_valid"):
        bad.append(f"k6 run not valid: {s.get('validity')}")
    m = _json(run / "manifest.json")
    if not m:
        bad.append("no manifest.json")
    else:
        if (m.get("outcome") or {}).get("k6_exit") != 0:
            bad.append(f"k6 exit {(m.get('outcome') or {}).get('k6_exit')}")
        if not (m.get("provenance") or {}).get("ok", False):
            bad.append("provenance checks failed (manifest.json)")
        d = m.get("dmos") or {}
        if d.get("enabled") and (not d.get("config_id") or d.get("config_id") != d.get("expected_config_id")):
            bad.append(f"DMOS config_id {d.get('config_id')} != expected {d.get('expected_config_id')}")
    pre = _json(run / "env_state_pre.json")
    if not pre:
        bad.append("no env_state_pre.json")
    else:
        e = pre.get("expectations") or {}
        if not e.get("met"):
            bad.append("pre-run expectations not met: " + "; ".join(e.get("violations") or []))
        if not e.get("expect_reset"):
            bad.append("reset to 1 replica not verified (DMOS_EXPECT_RESET=0?)")
        cl = pre.get("clusters") or {}
        if len(cl) != n_clusters:
            bad.append(f"{len(cl)} clusters in env_state_pre, expected {n_clusters}")
        for svc in prestop_services or []:
            for n, c in cl.items():
                lc = ((c.get("deployments") or {}).get(svc) or {}).get("lifecycle") or {}
                if lc.get("pre_stop") != "sleep 15s" or lc.get("termination_grace_s") != 30:
                    bad.append(f"{svc}@{n}: preStop {lc.get('pre_stop')} grace {lc.get('termination_grace_s')}")
    if not (run / "env_state_post.json").exists():
        bad.append("no env_state_post.json")
    for phase in ("pre", "post"):
        if f"[fleet] {phase}-run: {FLEET_OK}" not in log_text:
            bad.append(f"fleet {phase}-run check (ClusterMesh, Hubble labels) not passed")
    return bad


def cmd_check(a) -> int:
    run = find_run(Path(a.results), a.arm, a.regime, a.since)
    try:
        log_text = Path(a.log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log_text = ""
    if run is None:
        why = [x.strip() for x in log_text.splitlines() if "ABORT" in x or "SKIP" in x][-1:]
        print(f"-\tBAD\tno run folder (cell stopped before the load) {' '.join(why)}")
        return 1
    env = _json(ROOT / ".cache" / "test-env.json") or {}
    bad = check_run(run, log_text, a.rc, lifecycle_services(ROOT) if a.prestop else None,
                    len(env.get("clusters") or []) or 6)
    print(f"{run}\t{'BAD' if bad else 'OK'}\t{' | '.join(bad)}")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wait")
    w.add_argument("--minutes", type=float, required=True)
    c = sub.add_parser("check")
    c.add_argument("--arm", required=True)
    c.add_argument("--regime", required=True)
    c.add_argument("--since", type=float, required=True)
    c.add_argument("--log", required=True)
    c.add_argument("--rc", type=int, default=0)
    c.add_argument("--prestop", action="store_true",
                   help="require preStop sleep 15s / grace 30s on the entry and co-scaled services")
    c.add_argument("--results", default=str(RESULTS))
    a = ap.parse_args(argv)
    return cmd_wait(a) if a.cmd == "wait" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
