#!/usr/bin/env python3
"""
check_dataplane.py — evidence, run by run, of how the frontend traffic was
served: by the global pool of all clusters (true distribution) or locally,
in the cluster each request entered.

Two independent signals over the peak phase (dmos_live.jsonl; with a moving
hot spot, regimes -to<cluster>, over the peak before the move):

1. Requests served (Hubble, server side: `frontend_net_bytes_s` per cluster).
   The share each cluster served is compared with
     - its share of ready frontend replicas  → equal with the true
       distribution (Cilium spreads connections uniformly over all ready
       endpoints of all clusters);
     - its share of the demand (k6 weights)  → equal with local serving.
   Distances are total-variation distances (0 = equal, 1 = disjoint):
     tv_replicas = ½ Σ |served_i − replicas_i|,  tv_demand = ½ Σ |served_i − demand_i|.

2. CPU per frontend replica (cAdvisor, on the node the pods run on).
   With the true distribution every replica carries the same load, so the
   CPU per replica is about the same on every cluster; with local serving
   the load per replica is ∝ demand_i / replicas_i, and the replicas of the
   hot cluster work much more than the others. (The CPU *share* is not used:
   frontend pods have a CPU limit, so saturated hot pods compress it.)
   The normalised CPU-per-replica vector is compared with both predictions.

Verdict of each signal:
  pool          closer to the true-distribution prediction (by MARGIN; the
                served shares must also be within TOL of the replica shares)
  local         closer to the local-serving prediction
  indistinct    the two predictions coincide (e.g. uniform demand and
                uniform replicas): the run cannot tell them apart, not an error
  inconsistent  neither matches (missing data, transients, a broken pool)

Run verdict (rule fixed before the campaign): the served-share signal is
primary; the CPU signal can only veto it. The run verdict is the served
verdict, unless the CPU verdict is the opposite definite one ('pool' vs
'local'), in which case it is 'disagree'. A CPU verdict 'inconsistent' or
'indistinct' does not change it (the CPU per replica is also moved by idle
overheads and by saturation at the pod CPU limit). If the served signal is
'indistinct', a definite CPU verdict is used.

The journal campaign assumes the true distribution. A 'local',
'inconsistent' or 'disagree' run is excluded from the dataset (criterion
declared before the campaign) and
re-run in a later session. 'indistinct' runs are kept: each session contains
discriminating cells (skewed demand with uniform replicas: off, l2) that
show the data plane of that session.

Usage:
  python experiments/check_dataplane.py                         # all runs
  python experiments/check_dataplane.py --results-dir <dir> --csv out.csv
Exit status 3 if any run is 'local', 'inconsistent' or 'disagree'.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
PEAK = (330.0, 600.0)        # peak phase, skipping its first 30 s (replicas still moving)
TOL = 0.10
MARGIN = 0.10
_RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}_.+$")


def _tv(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _norm(v: Dict[str, float]) -> Optional[Dict[str, float]]:
    tot = sum(max(0.0, x) for x in v.values())
    return {k: max(0.0, x) / tot for k, x in v.items()} if tot > 0 else None


def demand_shares(summary: dict) -> Optional[Dict[str, float]]:
    """Peak demand shares per cluster from flash_crowd_summary.json."""
    clusters = summary.get("clusters") or []
    w = {c.get("name"): float(c.get("weight_spike") or 0.0) for c in clusters if c.get("name")}
    if not w:
        cfg = summary.get("config") or {}
        ws = cfg.get("spike_weights") or []
        w = {f"ms-cluster-{i:02d}": float(x) for i, x in enumerate(ws, start=1)}
    return _norm(w)


def shift_config(summary: dict) -> Optional[dict]:
    """The moving hot spot of a run (k6 summary config.shift), None if the
    hot spot stays where it is."""
    sh = (summary.get("config") or {}).get("shift") or {}
    return sh if sh.get("enabled") else None


def check_window(summary: dict) -> tuple:
    """Window of the data-plane check. With a moving hot spot the demand
    changes inside the peak: the check uses the peak before the move, where
    the demand is the spike weights (the data plane is a property of the
    session and does not change when the demand moves)."""
    sh = shift_config(summary)
    if sh is None:
        return PEAK
    return (PEAK[0], min(PEAK[1], float(sh.get("at_s") or PEAK[1])))


def peak_data(jsonl: Path, entry: str = "frontend",
              window: Optional[tuple] = None) -> Optional[dict]:
    """Over the peak window (or `window`): mean served share and replica share
    per cluster, mean frontend CPU (cores) and mean ready replicas per cluster."""
    t0, t1 = window or PEAK
    served: Dict[str, float] = {}
    rep_share: Dict[str, float] = {}
    cpu: Dict[str, float] = {}
    reps: Dict[str, float] = {}
    n = 0
    with open(jsonl, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (t0 <= float(s.get("elapsed_s") or 0) < t1):
                continue
            cl = s.get("clusters") or {}
            fe = ((s.get("services") or {}).get(entry) or {}).get("clusters") or {}
            sv = _norm({c: float(v.get("frontend_net_bytes_s") or 0.0) for c, v in cl.items()})
            rp = _norm({c: float((fe.get(c) or {}).get("ready_pods") or 0.0) for c in cl})
            if sv is None or rp is None:
                continue
            n += 1
            for c in cl:
                served[c] = served.get(c, 0.0) + sv.get(c, 0.0)
                rep_share[c] = rep_share.get(c, 0.0) + rp.get(c, 0.0)
                d = fe.get(c) or {}
                cpu[c] = cpu.get(c, 0.0) + float(d.get("cpu_cores") or 0.0)
                reps[c] = reps.get(c, 0.0) + float(d.get("ready_pods") or 0.0)
    if n == 0:
        return None
    return {"served": {c: v / n for c, v in served.items()},
            "replicas": {c: v / n for c, v in rep_share.items()},
            "cpu": {c: v / n for c, v in cpu.items()},
            "reps": {c: v / n for c, v in reps.items()},
            "samples": n}


def verdict_served(tv_rep: float, tv_dem: float, tv_rep_dem: float) -> str:
    if tv_rep_dem <= TOL:
        return "indistinct"
    if tv_rep <= TOL and tv_dem > tv_rep + MARGIN:
        return "pool"
    if tv_dem <= TOL and tv_rep > tv_dem + MARGIN:
        return "local"
    return "inconsistent"


def cpu_per_replica_check(data: dict, dem: Dict[str, float]) -> Optional[dict]:
    """Normalised CPU per replica vs the pool (uniform) and local
    (∝ demand_i / replicas_i) predictions, over clusters with replicas."""
    live = [c for c, r in data["reps"].items() if r >= 0.5]
    if len(live) < 2 or sum(data["cpu"].get(c, 0.0) for c in live) <= 0:
        return None
    obs = _norm({c: data["cpu"].get(c, 0.0) / data["reps"][c] for c in live})
    pool = {c: 1.0 / len(live) for c in live}
    local = _norm({c: dem.get(c, 0.0) / data["reps"][c] for c in live})
    if obs is None or local is None:
        return None
    tv_pool, tv_local, tv_pred = _tv(obs, pool), _tv(obs, local), _tv(pool, local)
    if tv_pred <= TOL:
        v = "indistinct"
    elif tv_pool + MARGIN <= tv_local:
        v = "pool"
    elif tv_local + MARGIN <= tv_pool:
        v = "local"
    else:
        v = "inconsistent"
    return {"verdict": v, "tv_pool": tv_pool, "tv_local": tv_local}


def combine(served: str, cpu: Optional[str]) -> str:
    """Run verdict: the served-share verdict, vetoed only by an opposite
    definite CPU verdict; a definite CPU verdict decides an indistinct one."""
    if {served, cpu} == {"pool", "local"}:
        return "disagree"
    if served == "indistinct" and cpu in ("pool", "local"):
        return cpu
    return served


def check_run(run_dir: Path) -> Optional[dict]:
    jp, sp = run_dir / "dmos_live.jsonl", run_dir / "flash_crowd_summary.json"
    if not (jp.is_file() and sp.is_file()):
        return None
    summary = json.loads(sp.read_text(encoding="utf-8"))
    window = check_window(summary)
    data = peak_data(jp, window=window)
    dem = demand_shares(summary)
    if data is None or dem is None:
        return {"run": run_dir.name, "verdict": "inconsistent", "note": "no peak data"}
    tv_rep = _tv(data["served"], data["replicas"])
    tv_dem = _tv(data["served"], dem)
    tv_rd = _tv(data["replicas"], dem)
    v_served = verdict_served(tv_rep, tv_dem, tv_rd)
    cpu = cpu_per_replica_check(data, dem)
    hot = max(dem, key=dem.get)
    return {
        "run": run_dir.name,
        "verdict": combine(v_served, cpu["verdict"] if cpu else None),
        "verdict_served": v_served,
        "verdict_cpu_per_replica": cpu["verdict"] if cpu else None,
        "tv_served_vs_replicas": round(tv_rep, 3),
        "tv_served_vs_demand": round(tv_dem, 3),
        "tv_replicas_vs_demand": round(tv_rd, 3),
        "tv_cpu_vs_pool": None if cpu is None else round(cpu["tv_pool"], 3),
        "tv_cpu_vs_local": None if cpu is None else round(cpu["tv_local"], 3),
        "hot_cluster": hot,
        "hot_demand": round(dem[hot], 3),
        "hot_served": round(data["served"].get(hot, 0.0), 3),
        "hot_replicas": round(data["replicas"].get(hot, 0.0), 3),
        "samples": data["samples"],
        "window_s": list(window),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    rows: List[dict] = []
    for d in sorted(p for p in args.results_dir.iterdir() if p.is_dir() and _RUN_DIR_RE.match(p.name)):
        r = check_run(d)
        if r is not None:
            rows.append(r)
    if not rows:
        print("No runs with dmos_live.jsonl and flash_crowd_summary.json.")
        return 1
    print(f"{'run':52s} {'VERDICT':12s} {'served':12s} {'cpu/replica':12s}  hot: demand served replicas")
    for r in rows:
        if "tv_served_vs_replicas" not in r:
            print(f"{r['run']:52s} {r['verdict']:12s} ({r.get('note', '')})")
            continue
        print(f"{r['run']:52s} {r['verdict']:12s} {r['verdict_served']:12s} "
              f"{str(r['verdict_cpu_per_replica']):12s}  {r['hot_cluster'][-2:]}: "
              f"{r['hot_demand']:.2f} {r['hot_served']:.2f} {r['hot_replicas']:.2f}")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    if args.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k != "verdict", k))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return 3 if any(counts.get(k) for k in ("local", "inconsistent", "disagree")) else 0


if __name__ == "__main__":
    sys.exit(main())
