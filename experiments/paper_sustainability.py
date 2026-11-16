#!/usr/bin/env python3
"""
paper_sustainability.py — carbon, energy, efficiency and control-plane cost of
each run (journal extension: "what does carbon awareness cost in latency?").

For every run in the results directory it rebuilds, snapshot by snapshot
(dmos_live.jsonl, every ~5 s), the carbon intensity CI_i(t) of each cluster
with the SAME function DMOS uses (CarbonClient.trace_value_at: hourly
Electricity Maps trace, window_start + speedup · elapsed), anchored to the
start of the load (<run>/carbon_t0, written by run_flash_crowd.sh), and
combines it with what the cluster served and consumed.

Metrics, per phase window (peak by default, and the whole run):

  Model-free (no power model):
    ci_served_g_kwh    mean CI of the requests served, weighted by the
                       frontend requests each cluster served (Hubble)
    ci_compute_g_kwh   mean CI of the CPU the application used
    core_s             CPU core-seconds of the whole application
    pod_s              ready pod-seconds of the whole application
    core_s_per_1k_ok   core-seconds per 1000 successful requests (k6)

  With a power model (estimates: relative comparison between arms only):
    co2_dyn_g          CO2 of the application's CPU: core-seconds x W per
                       busy core (from the tier model) x CI_i(t)
    co2_dyn_mg_per_ok  the same per successful request (mg)
    co2_node_g         CO2 of the worker nodes, linear power model
                       P = P_idle + (P_max - P_idle) x busy/cores, x CI_i(t)

  Performance (k6, for the trade-off): p95_ms, error_rate, ok_requests

  Control plane (DMOS arms): dmos_cpu_cores_mean/max, dmos_rss_mb_max,
  dmos_cycle_ms (mean duration of a scheduling cycle).

Runs collected before the whole-application fields existed (app_cpu_cores,
...) fall back to the five services the collector always recorded; the
column `accounting` says which ('app' or 'partial'). Runs without carbon_t0
are anchored to their first snapshot (`anchor` = 'first_snapshot'): for them
the CI timeline is the one of the load, not necessarily the one DMOS saw.

Usage:
  python experiments/paper_sustainability.py                       # all runs
  python experiments/paper_sustainability.py --phase peak --regimes "bal con"
  python experiments/paper_sustainability.py --arms "dmos-nca dmos-carbon_first dmos-latency_first"
Outputs (in --output-dir, default paper/figures/sustainability):
  sustainability_runs.csv, sustainability_summary.csv,
  fig_latency_carbon_tradeoff.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import paper_lib as pl  # noqa: E402

PHASE_WINDOWS: Dict[str, Tuple[float, float]] = {
    name: (float(t0), float(t1)) for t0, t1, name, _ in pl.PHASE_DEFS}
PHASE_WINDOWS['run'] = (float(pl.PHASE_DEFS[0][0]), float(pl.PHASE_DEFS[-1][1]))


# ── Context: clusters, power model, carbon replay ───────────────────────────

@dataclass
class ClusterInfo:
    name: str
    region: str
    cores: float
    p_idle: float
    p_max: float
    baseline_ci: float

    @property
    def w_per_busy_core(self) -> float:
        """Dynamic power of one busy core (W) in the linear tier model."""
        return (self.p_max - self.p_idle) / max(1.0, self.cores)


def load_clusters(local_yaml: Optional[Path] = None) -> Dict[str, ClusterInfo]:
    """Clusters of config/local.yaml with region code, cores and power tier
    (same tier model as paper_lib.TIER_POWER_MODEL)."""
    out = {}
    for name, c in pl.load_cluster_carbon_config(local_yaml).items():
        out[name] = ClusterInfo(name=name, region=str(c['region']), cores=float(c['cpu_cores']),
                                p_idle=float(c['p_idle']), p_max=float(c['p_max']),
                                baseline_ci=float(c['gco2_per_kwh']))
    return out


def default_carbon_config(carbon_yaml: Optional[Path] = None) -> dict:
    import yaml
    path = carbon_yaml or (REPO / 'config' / 'carbon.yaml')
    with open(path, encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}
    return raw.get('carbon_intensity', raw)


class CarbonReplay:
    """CI_i(elapsed since the anchor) through DMOS's own CarbonClient; one
    client per distinct trace configuration (window_start, speedup, ...)."""

    def __init__(self, base_config: dict, trace_dir: Optional[Path] = None):
        self.base = base_config
        self.trace_dir = str(Path(trace_dir).resolve()) if trace_dir else None
        self._clients: Dict[str, object] = {}

    def _client(self, trace_cfg: Optional[dict]):
        cfg = dict(self.base)
        cfg['provider'] = 'trace'
        trace = dict(cfg.get('trace') or {})
        # the run's own replay settings (dmos_config.json) win over carbon.yaml
        for k, v in (trace_cfg or {}).items():
            if v is not None:
                trace[k] = v
        if self.trace_dir:          # explicit --trace-dir wins over both
            trace['data_dir'] = self.trace_dir
        cfg['trace'] = trace
        key = json.dumps(trace, sort_keys=True, default=str)
        if key not in self._clients:
            from src.metrics.carbon_client import CarbonClient
            self._clients[key] = CarbonClient(cfg)
        return self._clients[key]

    def ci(self, region: str, elapsed_s: float, trace_cfg: Optional[dict] = None
           ) -> Optional[float]:
        return self._client(trace_cfg).trace_value_at(region, elapsed_s)

    def source(self, region: str, trace_cfg: Optional[dict] = None) -> str:
        return self._client(trace_cfg).source_for(region)


# ── One run ─────────────────────────────────────────────────────────────────

def _f(x) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _snapshot_epoch(snap: dict) -> Optional[float]:
    if snap.get('epoch_s') is not None:
        return float(snap['epoch_s'])
    try:   # legacy snapshots: ISO local time
        return datetime.fromisoformat(snap['timestamp']).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


def load_snapshots(path: Path) -> List[dict]:
    snaps = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{'):
                try:
                    snaps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    snaps.sort(key=lambda s: float(s.get('elapsed_s') or 0.0))
    return snaps


def read_anchor(run_dir: Path) -> Optional[float]:
    p = run_dir / 'carbon_t0'
    try:
        return float(p.read_text(encoding='utf-8').split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _cluster_usage(snap: dict, cl: str, info: ClusterInfo) -> Tuple[dict, bool]:
    """Per-cluster usage at one snapshot. Returns (values, whole_app)."""
    cblk = (snap.get('clusters') or {}).get(cl, {}) or {}
    whole = cblk.get('app_cpu_cores') is not None
    if whole:
        cpu = _f(cblk.get('app_cpu_cores')) or 0.0
        pods = _f(cblk.get('app_ready_pods')) or 0.0
    else:   # legacy: the five services the collector always recorded
        cpu, pods = 0.0, 0.0
        for svc in (snap.get('services') or {}).values():
            d = (svc.get('clusters') or {}).get(cl) or {}
            cpu += _f(d.get('cpu_cores')) or 0.0
            pods += _f(d.get('ready_pods')) or 0.0
    busy = _f(cblk.get('node_cpu_busy_cores'))
    cores = _f(cblk.get('node_cpu_cores')) or info.cores
    if busy is None:
        pct = _f(cblk.get('node_cpu_pct'))
        busy = None if pct is None else pct / 100.0 * cores
    return {
        'served_rps': max(0.0, _f(cblk.get('frontend_net_bytes_s')) or 0.0),  # frontend req/s (name is historical)
        'cpu': max(0.0, cpu),
        'pods': max(0.0, pods),
        'node_busy': busy,
        'node_cores': cores,
    }, whole


def _k6_phase(summary: dict, phase: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(requests, error_rate, p95_ms) of a phase, or of the whole run."""
    res = summary.get('results', {}) or {}
    if phase == 'run':
        g = res.get('global', {}) or {}
        return _f(g.get('total_requests')), _f(g.get('http_failure_rate')), _f(g.get('p95_ms'))
    d = (res.get('per_phase', {}) or {}).get(phase, {}) or {}
    return _f(d.get('requests')), _f(d.get('errorRate')), _f(d.get('p95'))


def run_metrics(run_dir: Path, clusters: Dict[str, ClusterInfo], replay: CarbonReplay,
                phase: str = 'peak', k6_summary: Optional[dict] = None,
                dmos_config: Optional[dict] = None) -> Optional[dict]:
    """All the metrics of one run over a phase window ('peak', ..., 'run')."""
    jp = run_dir / 'dmos_live.jsonl'
    if not jp.is_file():
        return None
    snaps = load_snapshots(jp)
    if len(snaps) < 2:
        return None
    if k6_summary is None:
        sp = run_dir / 'flash_crowd_summary.json'
        k6_summary = json.loads(sp.read_text(encoding='utf-8')) if sp.is_file() else {}
    if dmos_config is None:
        cp = run_dir / 'dmos_config.json'
        dmos_config = json.loads(cp.read_text(encoding='utf-8')) if cp.is_file() else {}
    trace_cfg = ((dmos_config or {}).get('carbon') or {}).get('trace') or None

    epochs = [_snapshot_epoch(s) for s in snaps]
    anchor = read_anchor(run_dir)
    anchor_mode = 'carbon_t0'
    if anchor is None:
        anchor = next((e for e in epochs if e is not None), None)
        anchor_mode = 'first_snapshot'
    w0, w1 = PHASE_WINDOWS[phase]

    el = [float(s.get('elapsed_s') or 0.0) for s in snaps]
    gaps = [b - a for a, b in zip(el, el[1:]) if b > a]
    typical = statistics.median(gaps) if gaps else 5.0

    acc = dict(served=0.0, served_ci=0.0, core_s=0.0, core_ci=0.0, pod_s=0.0,
               co2_dyn=0.0, e_dyn_kwh=0.0, co2_node=0.0, e_node_kwh=0.0, dur=0.0)
    whole_app = True
    sources = {}
    for k, snap in enumerate(snaps):
        t = el[k]
        dt = (el[k + 1] - t) if k + 1 < len(snaps) else typical
        dt = max(0.0, min(dt, 3 * typical))      # a stalled scrape does not count for minutes
        # part of [t, t+dt] inside the window
        lo, hi = max(t, w0), min(t + dt, w1)
        if hi <= lo:
            continue
        dtw = hi - lo
        acc['dur'] += dtw
        ep = epochs[k] if epochs[k] is not None else (anchor + t if anchor is not None else None)
        trace_elapsed = (ep - anchor) if (ep is not None and anchor is not None) else t
        for cl, info in clusters.items():
            u, whole = _cluster_usage(snap, cl, info)
            whole_app = whole_app and whole
            ci = replay.ci(info.region, trace_elapsed, trace_cfg)
            if ci is None:
                ci = info.baseline_ci
                sources[cl] = 'baseline'
            else:
                sources.setdefault(cl, replay.source(info.region, trace_cfg))
            acc['served'] += u['served_rps'] * dtw
            acc['served_ci'] += u['served_rps'] * dtw * ci
            acc['core_s'] += u['cpu'] * dtw
            acc['core_ci'] += u['cpu'] * dtw * ci
            acc['pod_s'] += u['pods'] * dtw
            e_dyn = u['cpu'] * info.w_per_busy_core * dtw / 3.6e6          # kWh
            acc['e_dyn_kwh'] += e_dyn
            acc['co2_dyn'] += e_dyn * ci
            if u['node_busy'] is not None:
                util = min(1.0, max(0.0, u['node_busy'] / max(1.0, u['node_cores'])))
                e_node = (info.p_idle + (info.p_max - info.p_idle) * util) * dtw / 3.6e6
                acc['e_node_kwh'] += e_node
                acc['co2_node'] += e_node * ci

    req, err, p95 = _k6_phase(k6_summary, phase)
    ok = req * (1.0 - (err or 0.0)) if req is not None else None

    # control plane (DMOS arms)
    dcpu = [_f((s.get('dmos_process') or {}).get('cpu_cores')) for s in snaps
            if w0 <= float(s.get('elapsed_s') or 0) < w1]
    dcpu = [x for x in dcpu if x is not None]
    drss = [_f((s.get('dmos_process') or {}).get('rss_mb')) for s in snaps]
    drss = [x for x in drss if x is not None]
    cyc = None
    for s in reversed(snaps):
        fe = (s.get('services') or {}).get('frontend') or {}
        if fe.get('scheduling_invocations'):
            cyc = _f(fe.get('scheduling_avg_ms'))
            break

    def per_1k(x):
        return x / ok * 1000.0 if (ok and x is not None) else None

    return {
        'phase': phase,
        'window_s': round(acc['dur'], 1),
        'anchor': anchor_mode,
        'accounting': 'app' if whole_app else 'partial',
        'ci_sources': ';'.join(f"{c}={sources.get(c, '?')}" for c in clusters),
        'ok_requests': ok,
        'error_rate': err,
        'p95_ms': p95,
        'served_requests_hubble': acc['served'],
        'ci_served_g_kwh': acc['served_ci'] / acc['served'] if acc['served'] > 0 else None,
        'ci_compute_g_kwh': acc['core_ci'] / acc['core_s'] if acc['core_s'] > 0 else None,
        'core_s': acc['core_s'],
        'pod_s': acc['pod_s'],
        'core_s_per_1k_ok': per_1k(acc['core_s']),
        'energy_dyn_wh': acc['e_dyn_kwh'] * 1000.0,
        'co2_dyn_g': acc['co2_dyn'],
        'co2_dyn_mg_per_ok': (acc['co2_dyn'] * 1000.0 / ok) if ok else None,
        'energy_node_wh': acc['e_node_kwh'] * 1000.0,
        'co2_node_g': acc['co2_node'],
        'dmos_cpu_cores_mean': statistics.fmean(dcpu) if dcpu else None,
        'dmos_cpu_cores_max': max(dcpu) if dcpu else None,
        'dmos_rss_mb_max': max(drss) if drss else None,
        'dmos_cycle_ms': cyc,
    }


# ── Aggregation, output, figure ─────────────────────────────────────────────

SUMMARY_COLS = ['p95_ms', 'error_rate', 'ok_requests', 'ci_served_g_kwh', 'ci_compute_g_kwh',
                'core_s_per_1k_ok', 'pod_s', 'co2_dyn_mg_per_ok', 'co2_dyn_g', 'co2_node_g',
                'dmos_cpu_cores_mean', 'dmos_rss_mb_max', 'dmos_cycle_ms']


def summarise(rows: List[dict]) -> List[dict]:
    by: Dict[str, List[dict]] = {}
    for r in rows:
        by.setdefault(r['scenario'], []).append(r)
    out = []
    for scen, rs in by.items():
        arm, regime = pl.parse_arm_regime(scen)
        s = {'scenario': scen, 'arm': arm, 'regime': regime or '', 'n': len(rs),
             'accounting': ','.join(sorted({r['accounting'] for r in rs})),
             'anchor': ','.join(sorted({r['anchor'] for r in rs}))}
        for c in SUMMARY_COLS:
            vals = [r[c] for r in rs if r.get(c) is not None]
            s[f'{c}_mean'] = statistics.fmean(vals) if vals else None
            s[f'{c}_std'] = statistics.stdev(vals) if len(vals) > 1 else None
        out.append(s)
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in keys})


def plot_tradeoff(summary: List[dict], out_dir: Path, phase: str) -> List[Path]:
    """One panel per regime: x = mean CI of the served requests, y = p95."""
    pl.setup_matplotlib()
    import matplotlib.pyplot as plt
    regimes = [r for r in pl.REGIME_ORDER if any(s['regime'] == r for s in summary)]
    regimes += sorted({s['regime'] for s in summary} - set(regimes))
    regimes = [r for r in regimes if any(s['ci_served_g_kwh_mean'] is not None and s['regime'] == r
                                         for s in summary)]
    if not regimes:
        return []
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.6 * len(regimes), 3.4), squeeze=False)
    for ax, reg in zip(axes[0], regimes):
        for s in summary:
            if s['regime'] != reg or s['ci_served_g_kwh_mean'] is None or s['p95_ms_mean'] is None:
                continue
            ax.errorbar(s['ci_served_g_kwh_mean'], s['p95_ms_mean'],
                        xerr=s['ci_served_g_kwh_std'] or 0, yerr=s['p95_ms_std'] or 0,
                        fmt='o', ms=7, capsize=3, color=pl.scenario_color(s['scenario']),
                        label=pl.ARM_LABELS.get(s['arm'], s['arm']))
        ax.set_title(pl.regime_label(reg) or reg)
        ax.set_xlabel('CI of served requests (gCO$_2$eq/kWh)')
        ax.set_ylabel(f'p95 latency, {phase} (ms)')
    h, l = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        hh, ll = ax.get_legend_handles_labels()
        for a, b in zip(hh, ll):
            if b not in l:
                h.append(a); l.append(b)
    fig.legend(h, l, loc='lower center', ncol=min(4, len(l)), bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return pl.save_figure(fig, out_dir, 'fig_latency_carbon_tradeoff')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--results-dir', type=Path, default=HERE / 'results')
    ap.add_argument('--output-dir', type=Path, default=REPO / 'paper' / 'figures' / 'sustainability')
    ap.add_argument('--phase', default='peak', choices=sorted(PHASE_WINDOWS))
    ap.add_argument('--arms', default='', help='space-separated arms (e.g. "dmos-nca dmos-carbon_first")')
    ap.add_argument('--regimes', default='', help='space-separated regimes (e.g. "bal con")')
    ap.add_argument('--local-yaml', type=Path, default=None)
    ap.add_argument('--trace-dir', type=Path, default=None,
                    help='folder of the carbon traces (default: trace.data_dir of config/carbon.yaml)')
    ap.add_argument('--no-figure', action='store_true')
    args = ap.parse_args(argv)

    arms, regimes = set(args.arms.split()), set(args.regimes.split())
    clusters = load_clusters(args.local_yaml)
    replay = CarbonReplay(default_carbon_config(), trace_dir=args.trace_dir)
    groups = pl.discover_runs(args.results_dir, load_timeline=False)

    rows = []
    for scen, group in groups.items():
        arm, regime = pl.parse_arm_regime(scen)
        if arms and arm not in arms:
            continue
        if regimes and (regime or '') not in regimes:
            continue
        for run in group.runs:
            m = run_metrics(run.run_dir, clusters, replay, args.phase,
                            k6_summary=run.k6_summary, dmos_config=run.dmos_config)
            if m is None:
                print(f"  [warn] no dmos_live.jsonl: {run.run_dir.name}")
                continue
            rows.append({'scenario': scen, 'run': run.run_dir.name, 'rep': run.repetition, **m})

    if not rows:
        print('No runs to analyse.')
        return 1
    summary = summarise(rows)
    write_csv(args.output_dir / 'sustainability_runs.csv', rows)
    write_csv(args.output_dir / 'sustainability_summary.csv', summary)

    print(f"\nSustainability ({args.phase}) — mean over repetitions")
    print(f"{'scenario':34s} {'n':>2s} {'p95 ms':>8s} {'CI srv':>7s} {'CI cpu':>7s} "
          f"{'core-s/1k':>9s} {'mgCO2/ok':>9s} {'DMOS cpu':>8s}  acc/anchor")

    def fmt(v, d=1):
        return '-' if v is None else f'{v:.{d}f}'
    for s in summary:
        print(f"{s['scenario']:34s} {s['n']:2d} {fmt(s['p95_ms_mean'], 0):>8s} "
              f"{fmt(s['ci_served_g_kwh_mean']):>7s} {fmt(s['ci_compute_g_kwh_mean']):>7s} "
              f"{fmt(s['core_s_per_1k_ok_mean'], 2):>9s} {fmt(s['co2_dyn_mg_per_ok_mean'], 3):>9s} "
              f"{fmt(s['dmos_cpu_cores_mean_mean'], 3):>8s}  {s['accounting']}/{s['anchor']}")
    if not args.no_figure:
        for p in plot_tradeoff(summary, args.output_dir, args.phase):
            print(f"  figure: {p}")
    print(f"  tables: {args.output_dir / 'sustainability_runs.csv'}, "
          f"{args.output_dir / 'sustainability_summary.csv'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
