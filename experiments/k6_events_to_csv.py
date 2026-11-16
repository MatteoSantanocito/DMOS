#!/usr/bin/env python3
"""
k6_events_to_csv.py — shrink k6_events.json (DMOS_RAW_OUT=1) to a small csv.gz.

The raw k6 JSON output of one 900 s run is 1-1.5 GB (one JSON line per metric
sample, with every tag). For the analysis of the tail latency we need, per HTTP
request: when, how long, its status, entry cluster, phase, endpoint, and the
connection / waiting times (a ~1 s connect time is a TCP SYN retransmission).

Output columns: t_ms (epoch ms), metric, value_ms, status, cluster, phase,
endpoint, method, path, shift.  metric is one of http_req_duration,
http_req_connecting, http_req_waiting, http_req_blocked (--metrics to change).

Usage:
  python experiments/k6_events_to_csv.py experiments/results/<run>/k6_events.json
  -> experiments/results/<run>/k6_requests.csv.gz  (a few MB)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Set
from urllib.parse import urlparse

DEFAULT_METRICS = ("http_req_duration", "http_req_connecting", "http_req_waiting",
                   "http_req_blocked")
COLUMNS = ["t_ms", "metric", "value_ms", "status", "cluster", "phase", "endpoint",
           "method", "path", "shift"]


def to_epoch_ms(ts: str) -> Optional[int]:
    """k6 time '2026-10-30T21:58:01.123456789+01:00' (nanoseconds) -> epoch ms."""
    if not ts:
        return None
    main, frac, tz = ts, "", ""
    if "." in ts:
        main, rest = ts.split(".", 1)
        i = 0
        while i < len(rest) and rest[i].isdigit():
            i += 1
        frac, tz = rest[:i], rest[i:]
    else:
        for sign in ("+", "-"):
            pos = ts.rfind(sign)
            if pos > 10:
                main, tz = ts[:pos], ts[pos:]
                break
        if ts.endswith("Z"):
            main, tz = ts[:-1], "Z"
    tz = "+00:00" if tz in ("Z", "") else tz
    try:
        dt = datetime.fromisoformat(main + tz)
    except ValueError:
        return None
    ms = int((frac + "000")[:3]) if frac else 0
    return int(dt.timestamp()) * 1000 + ms


def convert(lines: Iterable[str], out, metrics: Set[str]) -> int:
    w = csv.writer(out)
    w.writerow(COLUMNS)
    n = 0
    for line in lines:
        if '"Point"' not in line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if obj.get("type") != "Point" or obj.get("metric") not in metrics:
            continue
        data = obj.get("data") or {}
        tags = data.get("tags") or {}
        url = tags.get("url") or tags.get("name") or ""
        w.writerow([to_epoch_ms(data.get("time", "")), obj["metric"],
                    round(float(data.get("value", 0.0)), 2), tags.get("status", ""),
                    tags.get("cluster", ""), tags.get("phase", ""), tags.get("endpoint", ""),
                    tags.get("method", ""), urlparse(url).path if url else "",
                    tags.get("shift", "")])
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("events", type=Path, help="k6_events.json of a run")
    ap.add_argument("--out", type=Path, default=None, help="default: k6_requests.csv.gz next to it")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    a = ap.parse_args(argv)
    out = a.out or a.events.with_name("k6_requests.csv.gz")
    metrics = {m.strip() for m in a.metrics.split(",") if m.strip()}
    with open(a.events, "r", encoding="utf-8", errors="replace") as src, \
            gzip.open(out, "wt", encoding="utf-8", newline="") as dst:
        n = convert(src, dst, metrics)
    print(f"{n} samples -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
