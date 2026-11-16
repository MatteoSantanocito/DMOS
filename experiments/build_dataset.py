#!/usr/bin/env python3
"""build_dataset.py — the public dataset of the journal campaign, from the raw run folders.

  python experiments/build_dataset.py --results experiments/results \
      --tsv experiments/results/_campaign_2026-11-02-definitive.tsv --out ../dmos-2026-2-dataset

Copies the status-OK runs of the campaign TSV (every file of each run), the TSV,
the campaign log, the per-cell driver logs and the plans into --out, then scrubs
the copy (the originals are never modified):

  - LAN addresses 192.168.x.y become documentation addresses (198.51.100.0/24,
    then 203.0.113.0/24), with the same mapping in every file: the analysis
    identifies each cluster's own address in the ping lists, so the mapping must
    stay consistent; the mapping itself is not written;
  - the local home directory (C:\\Users\\<user>, any slash style, JSON-escaped
    too) becomes <HOME>;
  - manifest.json and dmos_config*.json: host.hostname and git.branch become "<redacted>".

It fails if any LAN address, home path or the given --forbid words survive.
Electricity Maps traces are not copied (not redistributable).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

LAN = re.compile(rb"\b192\.168\.\d{1,3}\.\d{1,3}\b")
HOME = re.compile(rb"(?i)C:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[A-Za-z0-9._-]+")
DOC_NETS = ["198.51.100.", "203.0.113."]


def ok_runs(tsv: Path) -> List[dict]:
    with open(tsv, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if (r.get("status") or "").strip() == "OK"]


def collect_ips(files: List[Path]) -> Dict[bytes, bytes]:
    seen = set()
    for p in files:
        seen.update(LAN.findall(p.read_bytes()))
    ordered = sorted(seen, key=lambda ip: tuple(int(x) for x in ip.split(b".")))
    if len(ordered) > 254 * len(DOC_NETS):
        raise SystemExit(f"too many distinct LAN addresses: {len(ordered)}")
    return {ip: f"{DOC_NETS[i // 254]}{i % 254 + 1}".encode() for i, ip in enumerate(ordered)}


def scrub_bytes(data: bytes, ipmap: Dict[bytes, bytes]) -> bytes:
    data = LAN.sub(lambda m: ipmap[m.group(0)], data)
    return HOME.sub(b"<HOME>", data)


def scrub_manifest(p: Path) -> None:
    m = json.loads(p.read_text(encoding="utf-8"))
    for section, key in (("host", "hostname"), ("git", "branch")):
        if isinstance(m.get(section), dict) and key in m[section]:
            m[section][key] = "<redacted>"
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check(files: List[Path], forbid: List[str]) -> List[str]:
    bad = []
    words = [re.compile(rb"\b" + re.escape(w).encode() + rb"\b", re.I) for w in forbid]
    for p in files:
        data = p.read_bytes()
        for rx in [LAN, HOME, *words]:
            if rx.search(data):
                bad.append(f"{p}: {rx.pattern!r}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--plans", type=Path, nargs="*", default=[], help="plan files to include")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--forbid", nargs="*", default=[], help="words that must not survive (e.g. user name)")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists: remove it first")
    rows = ok_runs(a.tsv)
    camp = a.out / "campaign"
    (a.out / "runs").mkdir(parents=True)
    camp.mkdir()
    for r in rows:
        shutil.copytree(a.results / r["run"].strip(), a.out / "runs" / r["run"].strip())
    stem = a.tsv.stem.lstrip("_").replace("campaign_", "")
    shutil.copy2(a.tsv, camp / a.tsv.name)
    log = a.tsv.with_suffix(".log")
    if log.is_file():
        shutil.copy2(log, camp / log.name)
    drv = camp / "driver_logs"
    drv.mkdir()
    for p in sorted(a.results.glob(f"_{stem}_*.log")):
        shutil.copy2(p, drv / p.name)
    for i, p in enumerate(a.plans):
        dst = camp / "plans" / (p.name if i == 0 else f"resume-{p.name}")
        dst.parent.mkdir(exist_ok=True)
        shutil.copy2(p, dst)

    files = sorted(p for p in a.out.rglob("*") if p.is_file())
    ipmap = collect_ips(files)
    for p in files:
        data = p.read_bytes()
        new = scrub_bytes(data, ipmap)
        if new != data:
            p.write_bytes(new)
    for name in ("manifest.json", "dmos_config.json", "dmos_config_expected.json"):
        for p in a.out.rglob(name):
            scrub_manifest(p)
    bad = check(files, a.forbid)
    if bad:
        print("\n".join(bad[:20]), file=sys.stderr)
        raise SystemExit(f"scrub incomplete: {len(bad)} files")
    with open(a.out / "CHECKSUMS.sha256", "w", encoding="utf-8", newline="\n") as f:
        for p in sorted(p for p in a.out.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256"):
            f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(a.out).as_posix()}\n")
    print(f"{len(rows)} runs, {len(files)} files, {len(ipmap)} LAN addresses pseudonymised -> {a.out}")
    return 0


if __name__ == "__main__":
    ip = {b"192.168.0.122": b"198.51.100.1"}
    assert scrub_bytes(b'{"192.168.0.122": 3.1}', ip) == b'{"198.51.100.1": 3.1}'
    assert scrub_bytes(b'"cwd": "C:\\\\Users\\\\bob\\\\x"', {}) == b'"cwd": "<HOME>\\\\x"'
    assert scrub_bytes(b"C:/Users/bob/Desktop", {}) == b"<HOME>/Desktop"
    assert scrub_bytes(b"10.42.0.7 192.168.0.1222", {}) == b"10.42.0.7 192.168.0.1222"
    sys.exit(main())
