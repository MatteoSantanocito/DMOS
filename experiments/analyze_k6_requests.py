#!/usr/bin/env python3
"""
analyze_k6_requests.py — tail latency of one run, request by request.

Input: k6_requests.csv.gz (experiments/k6_events_to_csv.py) of a run with
DMOS_RAW_OUT=1, and optionally the run folder (dmos.log, dmos_live.jsonl,
carbon_t0) to line the slow requests up with the scaling actions of DMOS.

Prints:
  * p50/p95/p99 and share > SLOW ms per phase;
  * where the time of the slow requests goes (connecting / blocked / waiting):
    waiting ~ duration means the time is spent inside the fleet, not on the
    k6 -> ingress connection (no client-side TCP retransmission);
  * the 10 s windows, the "bursts" (windows with >= BURST slow requests), their
    share of all slow requests, and the p95/p99 of spike and peak without them;
  * the 503s by phase;
  * with --run-dir: every `Scaled <cluster>/<service> -> N` of dmos.log on the
    k6 time axis, and how many bursts start within LAG s after an action.

Usage:
  python experiments/analyze_k6_requests.py experiments/results/<run>/k6_requests.csv.gz \\
      [--run-dir experiments/results/<run>] [--slow 800] [--burst 40] [--lag 15]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

PHASES = ["baseline", "spike", "peak", "decay", "cooldown"]
SCALED_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*Scaled (\S+)/(\S+) (?:→|->|\\u2192) (\d+) replicas")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def load_requests(path: Path) -> pd.DataFrame:
    """One row per request. k6_events_to_csv writes the metrics of a request
    on consecutive rows (duration, blocked, connecting, waiting): rebuilt by
    position, which is exact, instead of grouping by (time, tags), which
    merges requests of the same millisecond."""
    df = pd.read_csv(path)
    metrics = sorted(df.metric.unique())
    k = len(metrics)
    n = len(df) // k
    df = df.iloc[:n * k]
    order = np.asarray(df.metric.astype(str)).reshape(-1, k)
    if not (order == order[0]).all():
        raise SystemExit("metrics of a request are not on consecutive rows: unexpected file")
    vals = df.value_ms.to_numpy().reshape(-1, k)
    w = df.iloc[::k][["t_ms", "status", "cluster", "phase", "endpoint", "method", "path"]].reset_index(drop=True)
    names = {"http_req_duration": "dur", "http_req_blocked": "blk",
             "http_req_connecting": "conn", "http_req_waiting": "wait"}
    for j, m in enumerate(order[0]):
        w[names.get(m, m)] = vals[:, j]
    return w


def scaling_actions(run_dir: Path) -> Tuple[List[Tuple[float, str, str, int]], Optional[float]]:
    """(epoch_s, cluster, service, replicas) of every scaling action in
    dmos.log; the log has local wall time, the offset comes from
    dmos_live.jsonl (timestamp local + epoch_s)."""
    offset = None
    live = run_dir / "dmos_live.jsonl"
    if live.is_file():
        with open(live, encoding="utf-8") as f:
            first = json.loads(f.readline())
        local = time.mktime(time.strptime(first["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
        offset = first["epoch_s"] - local - (float("0." + first["timestamp"][20:26]) if "." in first["timestamp"] else 0)
        offset = round(offset / 900) * 900        # whole quarter hours (time zones)
    acts = []
    log = run_dir / "dmos.log"
    if log.is_file() and offset is not None:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = SCALED_RE.search(ANSI.sub("", line))
            if m:
                t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")) + offset
                acts.append((t, m.group(2), m.group(3), int(m.group(4))))
    return acts, offset


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--slow", type=float, default=800.0, help="slow request threshold (ms)")
    ap.add_argument("--burst", type=int, default=40, help="slow requests in 10 s that make a burst")
    ap.add_argument("--lag", type=float, default=15.0, help="seconds after an action")
    a = ap.parse_args(argv)

    w = load_requests(a.csv)
    t0 = (float((a.run_dir / "carbon_t0").read_text().strip()) * 1000
          if a.run_dir and (a.run_dir / "carbon_t0").is_file() else w.t_ms.min())
    w["t"] = (w.t_ms - t0) / 1000.0
    w = w[(w.t >= 0) & (w.t < 900)].copy()
    slow = w.dur > a.slow
    print(f"{len(w)} requests, {int(slow.sum())} slower than {a.slow:.0f} ms ({100 * slow.mean():.2f}%)")
    for ph in PHASES:
        x = w[w.phase == ph].dur
        if len(x):
            print(f"  {ph:9s} n={len(x):6d} p50={x.quantile(.5):5.0f} p95={x.quantile(.95):5.0f} "
                  f"p99={x.quantile(.99):5.0f} >{a.slow:.0f}ms={100 * (x > a.slow).mean():.2f}%")
    s = w[slow]
    if len(s):
        print(f"slow requests: median connecting {s.conn.median():.1f} ms, blocked {s.blk.median():.1f} ms, "
              f"waiting {s.wait.median():.0f} ms of {s.dur.median():.0f} ms"
              f" | connecting > 200 ms: {int((s.conn > 200).sum())}")

    w["bin"] = (w.t // 10 * 10).astype(int)
    cnt = w[slow].groupby("bin").size()
    bursts = sorted(cnt[cnt >= a.burst].index)
    share = cnt[bursts].sum() / max(1, cnt.sum())
    print(f"bursts (>= {a.burst} slow in 10 s): {len(bursts)} of {w.bin.nunique()} windows, "
          f"{100 * share:.0f}% of the slow requests; at {bursts}")
    for ph in ("spike", "peak"):
        x = w[w.phase == ph]
        y = x[~x.bin.isin(bursts)]
        if len(x) and len(y):
            print(f"  {ph}: p95 {x.dur.quantile(.95):.0f} -> {y.dur.quantile(.95):.0f}, "
                  f"p99 {x.dur.quantile(.99):.0f} -> {y.dur.quantile(.99):.0f} without the bursts")
    e = w[w.status == 503]
    if len(e):
        print("503 by phase: " + ", ".join(f"{k} {v}" for k, v in e.groupby('phase').size().items()))

    if a.run_dir:
        acts, offset = scaling_actions(a.run_dir)
        if not acts:
            print("no scaling action found (dmos.log / dmos_live.jsonl missing?)")
            return 0
        rel = [((t * 1000 - t0) / 1000, c, svc, n) for t, c, svc, n in acts]
        print(f"scaling actions in dmos.log: {len(rel)} (log offset {offset / 3600:+.2f} h)")
        failed = (w.status >= 500) | (w.status == 0)
        for t, c, svc, n in rel:
            if 0 <= t < 900:
                win = (w.t >= t) & (w.t < t + a.lag)
                after = int((win & slow).sum())
                errs = int((win & failed).sum())
                print(f"  t={t:6.1f}s {c}/{svc} -> {n}   slow in the next {a.lag:.0f} s: {after}"
                      f"   failed: {errs}")
        # failed requests that follow an action (any direction) vs. the others
        near_any = np.zeros(len(w), dtype=bool)
        for t, *_ in rel:
            near_any |= ((w.t >= t) & (w.t < t + a.lag)).to_numpy()
        f_near, f_tot = int((failed.to_numpy() & near_any).sum()), int(failed.sum())
        print(f"failed requests within {a.lag:.0f} s after a scaling action: {f_near} of {f_tot}")
        starts = [b for b in bursts]
        near = [b for b in starts if any(0 <= b + 10 - t <= a.lag + 10 for t, *_ in rel)]
        print(f"bursts with a scaling action within {a.lag:.0f} s before: {len(near)} of {len(starts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
