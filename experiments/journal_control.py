#!/usr/bin/env python3
"""journal_control.py — how light the DMOS control plane is (journal paper, Sec. "Control plane overhead").

  python experiments/journal_control.py --tsv experiments/results/_campaign_2026-11-02-definitive.tsv

Needs runs.csv of journal_figures.py in --out (run that first). Three outputs in --out:
  table_control.tex     controller CPU/memory, application CPU saved against HPA, pods started
  control_timing.csv    one decision split into its steps, from the DMOS logs of the campaign
  fig17_control_scaling offline benchmark of the decision logic (the campaign's own code, unchanged)
                        on synthetic federations of 6 to 1000 clusters; also control_bench.csv
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import random
import re
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TS = "%Y-%m-%d %H:%M:%S"
DMOS_ARMS = ["l2-pool-ps", "reach-pool-ps", "nca-pool-ps", "locality-ps", "reach-pool-ps-min0",
             "reach_green-pool-ps-min0"]


# ── 1. one decision, step by step, from the logs ────────────────────────────

def decision_steps(log: str):
    """Per decision: (total, L1 metrics, L1 solve, frontend actuation, backend co-scaling) in s.
    The orchestrator logs the total at the end; Level 1 logs its collection and solve times;
    the actuation steps come from the 1-s timestamps between those lines."""
    lines = ANSI.sub("", log).splitlines()
    out, l1 = [], None
    first_coloc = last_fe = None
    for ln in lines:
        m = re.search(r"Scheduling completato in (\d+)ms \(raccolta=(\d+)ms, winner_det=(\d+)ms\)", ln)
        if m:
            l1 = (datetime.strptime(ln[:19], TS), int(m.group(2)) / 1000, int(m.group(3)) / 1000)
            first_coloc = last_fe = None
            continue
        if l1 and "_apply_predictive_targets" in ln or (l1 and re.search(r"Scaled \S+/frontend", ln)):
            last_fe = datetime.strptime(ln[:19], TS)
        if l1 and "_enforce_colocation" in ln and first_coloc is None:
            first_coloc = datetime.strptime(ln[:19], TS)
        m = re.search(r"__main__:schedule_service \| Scheduling completato in ([\d.]+)s", ln)
        if m and l1:
            end, total = datetime.strptime(ln[:19], TS), float(m.group(1))
            fe_end = last_fe or l1[0]
            coscale = (end - fe_end).total_seconds()
            frontend = max(0.0, (fe_end - l1[0]).total_seconds())
            out.append((total, l1[1], l1[2], frontend, max(0.0, coscale)))
            l1 = None
    return out


def timing(results: Path, tsv: Path) -> pd.DataFrame:
    rows = []
    with open(tsv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["status"].strip() != "OK" or r["arm"].strip() not in DMOS_ARMS:
                continue
            log = (results / r["run"].strip() / "dmos.log").read_text(encoding="utf-8", errors="replace")
            for s in decision_steps(log):
                rows.append(dict(arm=r["arm"].strip(), regime=r["regime"].strip(), total_s=s[0], l1_metrics_s=s[1],
                                 l1_solve_ms=1000 * s[2], frontend_s=s[3], coscale_s=s[4]))
    return pd.DataFrame(rows)


# ── 2. cost of the control plane against what it saves ─────────────────────

def control_table(runs: pd.DataFrame, tim: pd.DataFrame, out: Path) -> pd.DataFrame:
    regimes = [("bal", "Balanced"), ("con", "Concentrated"), ("con-tosg", "Shift")]
    rows, lines = [], []
    for reg, label in regimes:
        d = runs[(runs.regime == reg) & (runs.arm == "reach-pool-ps")]
        h = runs[(runs.regime == reg) & (runs.arm == "hpa-none")]
        if d.empty or h.empty:
            continue
        ctrl = d.dmos_cpu_cores_mean.mean() * 900.0          # controller core-s over the 15-min run
        saved = h.core_s.mean() - d.core_s.mean()            # application core-s saved against HPA
        t = tim[(tim.regime == reg) & (tim.arm == "reach-pool-ps")]
        rows.append(dict(regime=reg, cpu_mean=d.dmos_cpu_cores_mean.mean(), cpu_max=d.dmos_cpu_cores_max.mean(),
                         rss_mb=d.dmos_rss_mb_max.mean(), decision_s=t.total_s.median(), ctrl_core_s=ctrl,
                         saved_core_s=saved, ratio=saved / ctrl, pods_dmos=d.pods_started.mean(),
                         pods_hpa=h.pods_started.mean()))
        r = rows[-1]
        lines.append(f"{label} & {r['cpu_mean']:.3f} / {r['cpu_max']:.2f} & {r['rss_mb']:.0f} & "
                     f"{r['decision_s']:.1f} & {r['ctrl_core_s']:.0f} & {r['saved_core_s']:.0f} & "
                     f"{r['ratio']:.1f}$\\times$ & {r['pods_dmos']:.0f} / {r['pods_hpa']:.0f}" + r" \\")
    tex = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
           r"Regime & DMOS CPU & RSS & Decision & DMOS CPU & CPU saved & Ratio & Pods started \\",
           r" & mean / max (cores) & (MB) & (s, median) & (core$\cdot$s) & vs HPA (core$\cdot$s) & & DMOS / HPA \\",
           r"\midrule"] + lines + [r"\bottomrule", r"\end{tabular}"]
    (out / "table_control.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)


# ── 3. offline benchmark of the decision logic ─────────────────────────────

def bench(repo: Path, sizes, reps: int, seed: int = 7) -> pd.DataFrame:
    sys.path.insert(0, str(repo))
    import yaml
    from src.level1.score_functions import ClusterMetrics, ScoreFunctions, ScoreParameters
    from src.level1.score_model import reach_shares, select_model
    from src.level1.winner_determination import ClusterBid, WinnerDetermination
    from src.level2.predictive_scaler import PredictiveScaler, PredictiveScalerParams

    weights_raw = yaml.safe_load((repo / "config" / "weights.yaml").read_text(encoding="utf-8"))
    model = select_model(weights_raw, "reach")
    params = dataclasses.replace(ScoreParameters(), **(model.parameters or {}))
    score = ScoreFunctions(weights={}, parameters=params, model=model)
    rng = random.Random(seed)
    rows = []
    for k in sizes:
        names = [f"c{i:04d}" for i in range(k)]
        delay = {c: rng.uniform(0, 200) for c in names}                 # netem-like delay per cluster
        rtt = {a: {b: delay[a] + delay[b] for b in names if b != a} for a in names}
        caps = {c: rng.choice([2, 3, 4]) for c in names}
        demand = np.random.default_rng(seed + k).dirichlet(np.full(k, 0.3)) * 350.0 * k / 6
        shares = {c: float(x) / float(demand.sum()) for c, x in zip(names, demand)}
        cores = {c: rng.choice([4, 6, 8]) for c in names}
        ps = PredictiveScaler(PredictiveScalerParams(), names, {c: 33.0 for c in names}, {c: 1 for c in names},
                              caps, budget_mode="pooled")
        current = {c: 1 for c in names}
        t_l1, t_l2, mem = [], [], []
        for r in range(reps):
            tracemalloc.start()
            t0 = time.perf_counter()
            psi = reach_shares(shares, rtt, model.reach_tau_ms)
            bids = []
            for c in names:
                m = ClusterMetrics(cpu_available_cores=cores[c] * 0.6, cpu_total_cores=cores[c],
                                   memory_available_gb=4.0, memory_total_gb=8.0,
                                   request_rate_current=float(demand[names.index(c)]), request_rate_max=33.0 * caps[c],
                                   latency_mean_ms=rng.uniform(20, 120), latency_variance_ms2=0.0,
                                   carbon_intensity_gco2_kwh=rng.uniform(20, 600), cost_per_replica_hour=0.0,
                                   network_rtt_ms=float(np.mean(list(rtt[c].values()))),
                                   ingress_rate_rps=float(demand[names.index(c)]), ingress_demand_share=psi[c])
                bids.append(ClusterBid(cluster_name=c, score=score.compute_score_breakdown(m)["total_score"],
                                       capacity=caps[c]))
            bids.sort(key=lambda b: -b.score)
            budget = max(k, int(np.ceil(demand.sum() / 33.0)))
            alloc, _ = WinnerDetermination().allocate(bids, min(budget, sum(caps.values())))
            t1 = time.perf_counter()
            now = 1000.0 + 10.0 * r
            fast = {c: float(demand[i]) for i, c in enumerate(names)}
            ps.tick(now, fast, fast, current)
            ps.hold_budget(now, budget)
            ps.targets(now, {a.cluster_name: a.replicas for a in alloc}, current)
            t2 = time.perf_counter()
            mem.append(tracemalloc.get_traced_memory()[1] / 2**20)
            tracemalloc.stop()
            t_l1.append(1000 * (t1 - t0))
            t_l2.append(1000 * (t2 - t1))
        rows.append(dict(clusters=k, l1_ms=float(np.median(t_l1)), l1_ms_p95=float(np.percentile(t_l1, 95)),
                         l2_ms=float(np.median(t_l2)), l2_ms_p95=float(np.percentile(t_l2, 95)),
                         total_ms=float(np.median(np.add(t_l1, t_l2))), peak_mb=float(np.median(mem))))
        print(f"  K={k:5d}  L1 {rows[-1]['l1_ms']:9.2f} ms  L2 {rows[-1]['l2_ms']:8.2f} ms  "
              f"memory {rows[-1]['peak_mb']:6.1f} MB")
    return pd.DataFrame(rows)


def fig_scaling(b: pd.DataFrame, out: Path):
    sys.path.insert(0, str(HERE))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import journal_figures as jf
    jf.style()
    fig, ax = plt.subplots(figsize=(jf.COL, 2.3))
    ax.plot(b.clusters, b.l1_ms, "o-", color="#0072B2", ms=3.5, lw=1.2, label="Levels 0 and 1 ($\\psi$, score, distribution)")
    ax.plot(b.clusters, b.l2_ms, "s-", color="#D55E00", ms=3.5, lw=1.2, label="Level 2 (forecast, budget, actuation)")
    ax.axhline(30_000, color=jf.INK2, lw=0.7, ls=(0, (3, 1.5)))
    ax.text(b.clusters.max(), 30_000 * 0.55, "periodic decision (30 s)", fontsize=6, color=jf.INK2,
            ha="right", va="top")
    ax.axvline(6, color=jf.INK2, lw=0.6, ls=":", zorder=0)
    ax.text(6.5, 3000, "testbed\n(6 clusters)", fontsize=6, color=jf.INK2, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=80_000)
    ax.set_xlabel("Cluster nella federazione")
    ax.set_ylabel("Tempo di calcolo per decisione (ms)")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig17_control_scaling.{ext}")
    plt.close(fig)
    print("  fig17_control_scaling")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--tsv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "paper" / "figures" / "journal")
    ap.add_argument("--repo", type=Path, default=HERE.parent, help="DMOS checkout whose src/ is benchmarked")
    ap.add_argument("--sizes", default="6,12,25,50,100,250,500,1000")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args(argv)
    runs = pd.read_csv(a.out / "runs.csv")
    tim = timing(a.results, a.tsv)
    tim.to_csv(a.out / "control_timing.csv", index=False)
    print(tim.groupby("arm")[["total_s", "l1_metrics_s", "l1_solve_ms", "frontend_s", "coscale_s"]].median().round(3))
    print(control_table(runs, tim, a.out).round(3).to_string(index=False))
    b = bench(a.repo, [int(x) for x in a.sizes.split(",")], a.reps)
    b.to_csv(a.out / "control_bench.csv", index=False)
    fig_scaling(b, a.out)
    return 0


if __name__ == "__main__":
    assert decision_steps("2026-11-02 21:07:40 | x | Scheduling completato in 2005ms (raccolta=2004ms, winner_det=1ms)\n"
                          "2026-11-02 21:07:41 | __main__:_apply_predictive_targets | PS c1\n"
                          "2026-11-02 21:07:48 | __main__:schedule_service | Scheduling completato in 9.91s\n") \
        == [(9.91, 2.004, 0.001, 1.0, 7.0)]
    sys.exit(main())
