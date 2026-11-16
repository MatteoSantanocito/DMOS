#!/usr/bin/env python3
"""journal_figures.py — figures and tables of the journal paper.

  # definitive campaign (status OK rows of the TSV)
  python experiments/journal_figures.py --tsv experiments/results/_campaign_2026-11-02-definitive.tsv
  # pilot runs as stand-ins (watermarked), to design the figures
  python experiments/journal_figures.py --proxy

Options: --results <dir> (default experiments/results), --out <dir> (default
paper/figures/journal), --only fig02,fig05 (prefixes), --no-cache.
Output: figNN_<name>.pdf (vector) + .png (600 dpi), summary.csv (one row per
arm x regime: n, mean, 95% bootstrap CI), runs.csv, table_main.tex, overhead.csv.

Colours: arms and regions are fixed per entity in every figure (palettes
checked for colour-vision deficiency; arms also carry a distinct marker).
"""
from __future__ import annotations

import argparse
import re

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import journal_data as jd  # noqa: E402

COL, DCOL = 3.5, 7.16          # single / double column width (in)

ARMS = {   # key: (label, colour, marker); what each arm runs: experiments/run_naming.py show <arm> con
    "off": ("Static", "#8C8C8C", "s"),                       # 1 frontend per cluster, no scaling
    "hpa-none": ("Per-cluster HPA", "#D55E00", "v"),         # K8s HPA, CPU 70 %, each cluster alone
    "l2-pool-ps": ("Even split", "#E69F00", "D"),            # DMOS budget + scaler, no placement (N/K)
    "locality-ps": ("Locality term", "#CC79A7", "P"),        # demand on the net term only; per-cluster budget
    "nca-pool-ps": ("Local-demand gate", "#882255", "X"),    # gate on each cluster's own demand
    "reach-pool-ps": ("DMOS", "#0072B2", "o"),               # gate on reachable demand psi (50 ms)
    "reach-pool-ps-min0": ("DMOS (min 0)", "#56B4E9", "^"),  # + a cluster may hold no frontend
    "reach_green-pool-ps-min0": ("DMOS-Green (min 0)", "#009E73", "h"),   # + carbon-first weights
}
REGIMES = {"bal": "Balanced", "con": "Concentrated", "con-tosg": "Shifting (DE→SG)"}
REGION_COLOR = dict(zip(jd.REGION.values(), ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#e4e4e2"
CORE = ["off", "hpa-none", "l2-pool-ps", "reach-pool-ps"]
DMOS_FAMILY = ["l2-pool-ps", "reach-pool-ps", "reach-pool-ps-min0", "reach_green-pool-ps-min0"]
ABLATION = ["locality-ps", "nca-pool-ps", "reach-pool-ps", "reach-pool-ps-min0", "reach_green-pool-ps-min0"]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.titlelocation": "left", "lines.linewidth": 1.2, "legend.frameon": False,
        "legend.handlelength": 1.6, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# ── helpers ─────────────────────────────────────────────────────────────────

class Fig:
    def __init__(self, ds: jd.Dataset, out: Path):
        self.ds, self.out = ds, out
        self.df = ds.table()
        self.made = []

    def arms(self, regime, wanted=None):
        have = set(self.df.loc[self.df.regime == regime, "arm"])
        return [a for a in (wanted or ARMS) if a in have]

    def vals(self, arm, regime, col):
        d = self.df[(self.df.arm == arm) & (self.df.regime == regime)]
        return pd.to_numeric(d[col], errors="coerce").dropna().to_numpy()

    def ci(self, arm, regime, col):
        return jd.boot_mean_ci(self.vals(arm, regime, col))

    def regimes(self):
        return [r for r in REGIMES if r in set(self.df.regime)]

    def representative(self, arm, regime):
        """The run whose frontend replica-minutes are closest to the group median."""
        runs = self.ds.group(arm, regime)
        if not runs:
            return None
        v = np.array([self.ds.m(r)["fe_rm"] for r in runs])
        return runs[int(np.argmin(np.abs(v - np.median(v))))]

    def save(self, fig, name, caption):
        if self.ds.proxy:
            fig.text(0.995, 0.002, "PROXY: pilot runs, not the campaign", ha="right", va="bottom",
                     fontsize=5.5, color="#b3b3b3")
        for ext in ("pdf", "png"):
            fig.savefig(self.out / f"{name}.{ext}")
        plt.close(fig)
        self.made.append((name, caption))
        print(f"  {name}")


PHASE_LABEL = {"spike": "ramp"}   # display name only; the data columns keep the key "spike"


def shade_phases(ax, labels=True, y=0.98):
    for (name, a, b), c in zip(jd.PHASES, ["none", "#f2f2f0", "#e9e9e6", "#f2f2f0", "none"]):
        if c != "none":
            ax.axvspan(a, b, color=c, lw=0, zorder=0)
        if labels:
            ax.text((a + b) / 2, y, PHASE_LABEL.get(name, name), transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6, color=INK2)
    ax.set_xlim(0, 900)
    ax.set_xticks([0, 180, 300, 600, 720, 900])


def panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".strip(), loc="left", fontsize=8, color=INK, pad=3)


def arm_handles(arms, line=False):
    return [Line2D([], [], color=ARMS[a][1], marker=ARMS[a][2], ms=4.5, lw=1.2 if line else 0,
                   label=ARMS[a][0]) for a in arms]


def region_handles():
    return [Patch(facecolor=c, edgecolor="none", label=r) for r, c in REGION_COLOR.items()]


def mean_band(ds, runs, col):
    m = np.vstack([ds.ts(r)[col].to_numpy(float) for r in runs])
    return np.nanmean(m, 0), np.nanmin(m, 0), np.nanmax(m, 0)


def errpoint(ax, x, y, xerr=None, yerr=None, arm="reach-pool-ps", ms=5.5, z=4):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=ARMS[arm][2], color=ARMS[arm][1], ms=ms, mec="white",
                mew=0.6, elinewidth=0.9, capsize=0, zorder=z)


def asym(est, lo, hi):
    return None if np.isnan(lo) else [[est - lo], [hi - est]]


# ── figures ─────────────────────────────────────────────────────────────────

def fig01_testbed(F: Fig):
    """Testbed and workload: RTT matrix, carbon signal, offered load per entry region (one column)."""
    ds = F.ds
    fig = plt.figure(figsize=(COL, 4.45))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.45, 1.3, 0.9], hspace=0.68, wspace=0.28)
    regs = list(jd.REGION.values())
    # (a) RTT
    ax = fig.add_subplot(gs[0, 0:2])
    run = ds.runs[0]
    M = np.nanmedian(np.stack([jd.rtt_matrix(r) for r in ds.runs]), axis=0)   # median over all runs
    np.fill_diagonal(M, np.nan)
    ax.imshow(M, cmap="Blues", vmin=0, vmax=np.nanmax(M) * 1.15)
    ax.set_xticks(range(6), regs, fontsize=6)
    ax.set_yticks(range(6), regs, fontsize=6)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    for i in range(6):
        for j in range(6):
            txt = "–" if i == j else f"{M[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                    color="white" if (not np.isnan(M[i, j]) and M[i, j] > np.nanmax(M) * 0.6) else INK)
    panel(ax, "a", "Inter-cluster RTT (ms)")
    # region legend next to the matrix (shared by all panels)
    lax = fig.add_subplot(gs[0, 2])
    lax.axis("off")
    lax.legend(handles=region_handles(), loc="center left", fontsize=6.5, frameon=False,
               handlelength=1.2, title="Region", title_fontsize=6.5)
    # (b) carbon intensity seen during a run
    ax = fig.add_subplot(gs[1, :])
    ts = ds.ts(run)
    shade_phases(ax)
    for t_ in ax.texts:
        t_.set_fontsize(5)
    ends = []
    for r in regs:
        ax.plot(ts.t, ts[f"ci_{r}"], color=REGION_COLOR[r], lw=1.1)
        ends.append([ts[f"ci_{r}"].iloc[-1], r, float(ts[f"ci_{r}"].mean())])
    ends.sort()
    last = -1e9
    for v, r, mean in ends:               # direct labels with the run mean, nudged apart
        y = max(v, last + 70)
        ax.text(910, y, f"{r} {mean:.0f}", color=REGION_COLOR[r], fontsize=5.5, va="center", clip_on=False)
        last = y
    ax.set_ylabel("CI (gCO$_2$/kWh)")
    ax.set_xlabel("Run time (s)", labelpad=1)
    ax.tick_params(labelsize=6)
    # traces are UTC; the top axis shows Central European Summer Time (UTC+2) of the trace day
    top = ax.secondary_xaxis("top", functions=(lambda t: t * 96 / 3600 + 2, lambda h: (h - 2) * 3600 / 96))
    top.set_xticks([6, 12, 18, 24], ["06:00", "12:00", "18:00", "24:00"])
    top.tick_params(colors=INK2, labelsize=5.5)
    ax.set_ylim(0, 790)
    ax.set_title("(b) Carbon intensity in every run (top: CEST)", loc="left", fontsize=8, pad=11)
    # (c–e) offered load per entry region
    for k, reg in enumerate(["bal", "con", "con-tosg"]):
        ax = fig.add_subplot(gs[2, k])
        rr = [r for r in ds.runs if r.regime == reg]
        if not rr:
            ax.set_visible(False)
            continue
        cfg = rr[0].summary["config"]
        t = jd.GRID
        load = jd.load_profile(t, cfg["baseline_rps"], cfg["peak_rps"], tuple(cfg["phase_times_s"][:4]))
        W = np.array([jd.entry_weights(cfg, x) for x in t]) * load[:, None]
        shade_phases(ax, labels=False)
        ax.stackplot(t, W.T, colors=[REGION_COLOR[r] for r in regs], lw=0)
        ax.set_ylim(0, cfg["peak_rps"] * 1.05)
        ax.set_xticks([0, 300, 600, 900])
        ax.set_yticks([0, 200])
        ax.tick_params(labelsize=5.5)
        if k == 1:                                  # one shared x label for the row
            ax.set_xlabel("Run time (s)", labelpad=1)
        if k == 0:
            ax.set_ylabel("Load (req/s)")
        else:
            ax.set_yticklabels([])
        panel(ax, "cde"[k], {"bal": "Balanced", "con": "Concentrated", "con-tosg": "Shifting"}[reg])
    F.save(fig, "fig01_testbed_workload",
           "Testbed and workload. (a) RTT between the six clusters, median over all runs (ping exporter). "
           "(b) Carbon intensity of each region replayed by DMOS and by the analysis (one summer day "
           "of Electricity Maps data compressed into the 900-s run; label: mean over the run in gCO$_2$/kWh, "
           "FR and NO stay near 20 and lie close to the axis). (c–e) Offered load by entry region "
           "for the three demand regimes; shaded: spike, peak and decay phases.")


def fig02_tradeoff(F: Fig):
    """Latency against resources: every arm, every regime."""
    regs = F.regimes()
    fig, axs = plt.subplots(1, len(regs), figsize=(DCOL, 2.55), squeeze=False)
    used = []
    for ax, reg in zip(axs[0], regs):
        for a in F.arms(reg):
            x, y = F.vals(a, reg, "fe_rm"), F.vals(a, reg, "p95")
            ax.scatter(x, y, s=9, marker=ARMS[a][2], facecolor="none", edgecolor=ARMS[a][1], lw=0.6,
                       alpha=0.7, zorder=3)
            cx, cy = F.ci(a, reg, "fe_rm"), F.ci(a, reg, "p95")
            errpoint(ax, cx[0], cy[0], asym(*cx), asym(*cy), a)
            used.append(a)
        ax.set_xlabel("Frontend replica-minutes")
        panel(ax, "abc"[regs.index(reg)], REGIMES[reg])
        ax.margins(0.12)
    axs[0][0].set_ylabel("p95 latency (ms)")
    ax = axs[0][0]
    ax.annotate("better", xy=(0.05, 0.06), xytext=(0.24, 0.24), xycoords="axes fraction",
                fontsize=6.5, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
    order = [a for a in ARMS if a in used]
    fig.legend(handles=arm_handles(order), loc="upper center", ncol=len(order), bbox_to_anchor=(0.5, 1.07),
               columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(w_pad=1.2)
    F.save(fig, "fig02_latency_resource_tradeoff",
           "Latency–resource trade-off. Each arm: mean over repetitions with 95% bootstrap CI (filled), "
           "individual runs (hollow). Lower-left is better: DMOS reaches lower p95 than HPA with fewer "
           "frontend replica-minutes; the static fleet uses the fewest replicas but is the slowest.")


FOREST = [("p50", "Median latency"), ("p95", "p95 latency"), ("p99", "p99 latency"),
          ("p95_peak", "p95, peak phase"), ("exp_rtt_ms", "Expected RTT (model J)"),
          ("fe_rm", "Frontend replica-min"),
          ("all_rm", "Pod-min, all services"), ("core_s_per_1k_ok", "CPU core-s per 1k req"),
          ("co2_ug_per_req", "CO$_2$ per request"), ("ci_served_g_kwh", "CI of served requests")]


def fig03_forest(F: Fig, ref="hpa-none"):
    """Relative change against HPA, with bootstrap CIs."""
    regs = [r for r in F.regimes() if ref in F.arms(r)]
    fig, axs = plt.subplots(1, len(regs), figsize=(DCOL, 3.35), sharey=True, sharex=True, squeeze=False)
    used = []
    for ax, reg in zip(axs[0], regs):
        arms = [a for a in DMOS_FAMILY if a in F.arms(reg)]
        off = np.linspace(-0.27, 0.27, len(arms)) if len(arms) > 1 else [0]
        for i, (col, _) in enumerate(FOREST):
            yb = len(FOREST) - 1 - i
            if i % 2 == 0:
                ax.axhspan(yb - 0.5, yb + 0.5, color="#f6f6f4", lw=0, zorder=0)
            base = F.vals(ref, reg, col)
            for a, dy in zip(arms, off):
                x = F.vals(a, reg, col)
                if len(x) == 0 or len(base) == 0:
                    continue
                e, lo, hi = jd.boot_ratio_ci(x, base)
                errpoint(ax, e, yb + dy, xerr=asym(e, lo, hi), arm=a, ms=4.2)
                used.append(a)
        ax.axvline(0, color=INK2, lw=0.7, zorder=1)
        ax.set_xlabel(f"← better   change vs {ARMS[ref][0]} (%)")
        ax.grid(axis="y", visible=False)
        panel(ax, "abc"[regs.index(reg)], REGIMES[reg])
    axs[0][0].set_yticks(range(len(FOREST)), [lab for _, lab in FOREST][::-1])
    axs[0][0].set_ylim(-0.6, len(FOREST) - 0.4)
    order = [a for a in ARMS if a in used]
    fig.legend(handles=arm_handles(order), loc="upper center", ncol=len(order), bbox_to_anchor=(0.55, 1.05))
    fig.tight_layout(w_pad=0.8)
    F.save(fig, "fig03_effects_vs_hpa",
           "Effect sizes against HPA: percentage change of the mean with 95% bootstrap CI, per regime. "
           "For every metric lower is better. Expected RTT: the model J of Section 3 computed from the "
           "ready-replica shares (not a measured RTT). Missing CIs: fewer than two runs in a group.")


NEED_TXT = ("max(minimum fleet, ⌈HTTP load/33⌉): the HTTP load is the k6 profile scaled by the "
            "run's own request count (about 1.2x, k6 follows the redirects of the POST endpoints); "
            "minimum fleet 6 (1 per cluster), 1 for the min-0 arms.")


def fig04_trajectories(F: Fig, regime="con", arms=("hpa-none", "l2-pool-ps", "reach-pool-ps",
                                                     "reach_green-pool-ps-min0")):
    """Where and when the frontend replicas run: one representative run per arm."""
    arms = [a for a in arms if a in F.arms(regime)]
    if not arms:
        return
    n = len(arms)
    ncol = 2 if n == 4 else n          # 2x2 or one row: never an empty panel
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(COL, 1.45 * nrow + 0.75), sharex=True, sharey=True,
                            squeeze=False)                   # one column of the paper
    regs = list(jd.REGION.values())
    for k, a in enumerate(arms):
        ax = axs[k // ncol][k % ncol]
        run = F.representative(a, regime)
        ts = F.ds.ts(run)
        shade_phases(ax, labels=(k < ncol))
        for t_ in ax.texts:
            t_.set_fontsize(4.8)
        ax.stackplot(ts.t, np.nan_to_num(ts[[f"ready_{r}" for r in regs]].to_numpy().T),
                     colors=[REGION_COLOR[r] for r in regs], lw=0, step="post")
        ax.step(ts.t, ts.need, where="post", color=INK, lw=1.0, ls=(0, (3, 1.5)))
        ax.axhline(18, color=INK2, lw=0.6, ls=":")
        ax.set_ylim(0, 20.5)
        ax.set_xticks([0, 300, 600, 900])
        ax.tick_params(labelsize=6)
        panel(ax, "abcdefgh"[k], ARMS[a][0])
        ax.title.set_fontsize(7)
        if k % ncol == 0:
            ax.set_ylabel("Ready replicas")
        if k // ncol == nrow - 1:
            ax.set_xlabel("Run time (s)")
    for k in range(n, nrow * ncol):
        axs[k // ncol][k % ncol].set_visible(False)
    extra = [Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.5)), label="Required"),
             Line2D([], [], color=INK2, lw=0.6, ls=":", label="Fleet cap (18)")]
    fig.legend(handles=region_handles() + extra, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08),
               columnspacing=0.8, handletextpad=0.4, handlelength=1.3, fontsize=6.5, frameon=False)
    fig.tight_layout(h_pad=0.5, w_pad=0.5)
    F.save(fig, f"fig04_replica_placement_{regime}",
           f"Frontend replicas per region over time, {REGIMES[regime]} regime (run closest to the median "
           "replica-minutes of each arm). Dashed: replicas required, "
           + NEED_TXT + " The per-cluster caps (4/3/2/4/2/3) bind on DE in every arm.")


def fig05_dynamics(F: Fig, regime="con", arms=CORE):
    """Capacity tracking and latency at the hot entry, mean over repetitions."""
    arms = [a for a in arms if a in F.arms(regime)]
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.45), sharex=True)
    ax = axs[0]
    shade_phases(ax)
    for a in arms:
        m, lo, hi = mean_band(F.ds, F.ds.group(a, regime), "ready_total")
        ax.fill_between(jd.GRID, lo, hi, step="post", color=ARMS[a][1], alpha=0.15, lw=0)
        ax.step(jd.GRID, m, where="post", color=ARMS[a][1], lw=1.3)
    need = F.ds.ts(F.ds.group(arms[-1], regime)[0]).need
    ax.step(jd.GRID, need, where="post", color=INK, lw=1.0, ls=(0, (3, 1.5)), zorder=5)
    ax.set_ylim(0, 22)
    ax.set_ylabel("Ready frontend replicas (all regions)")
    ax.set_xlabel("Run time (s)")
    panel(ax, "a", "Provisioned capacity")
    ax = axs[1]
    shade_phases(ax)
    for a in arms:
        m, lo, hi = mean_band(F.ds, F.ds.group(a, regime), "exp_rtt")
        ax.fill_between(jd.GRID, lo, hi, color=ARMS[a][1], alpha=0.15, lw=0)
        ax.plot(jd.GRID, m, color=ARMS[a][1], lw=1.2)
    ax.set_ylabel("Expected RTT, model J (ms)")
    ax.set_xlabel("Run time (s)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    panel(ax, "b", "Network distance between users and their frontend")
    h = arm_handles(arms, line=True) + [Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 1.5)), label="Required")]
    fig.legend(handles=h, loc="upper center", ncol=len(h), bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.5)
    F.save(fig, f"fig05_dynamics_{regime}",
           f"Dynamics in the {REGIMES[regime]} regime, mean over repetitions (band: min–max over runs). "
           "(a) Ready frontend replicas against the replicas required, " + NEED_TXT +
           " (b) Expected network RTT between the entry region of a request and the region of the frontend "
           "serving it (model J): the pooled frontend spreads requests over all ready replicas, so the "
           "serving region follows the replica shares (RTT matrix measured in each run).")


def fig06_entry_latency(F: Fig, q="p95"):
    """p95 seen by the users of each entry region."""
    regs = F.regimes()
    cols = list(jd.REGION.values()) + ["All"]
    fig, axs = plt.subplots(1, len(regs), figsize=(DCOL, 2.5), squeeze=False,
                            gridspec_kw=dict(width_ratios=[len(F.arms(r)) + 1 for r in regs]))
    allv = pd.to_numeric(F.df[[f"{q}_{c}" for c in jd.REGION.values()] + [q]].stack(), errors="coerce")
    vmin, vmax = np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)
    for ax, reg in zip(axs[0], regs):
        arms = F.arms(reg)
        M = np.array([[np.nanmean(F.vals(a, reg, f"{q}_{c}" if c != "All" else q)) for a in arms]
                      for c in cols])
        ax.imshow(M, cmap="YlOrBr", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(cols)):
            for j in range(len(arms)):
                v = M[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v > vmin + 0.65 * (vmax - vmin) else INK)
        ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], rotation=35, ha="right")
        ax.set_yticks(range(len(cols)), cols)
        ax.axhline(len(cols) - 1.5, color="white", lw=2)
        ax.grid(False)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        panel(ax, "abc"[regs.index(reg)], REGIMES[reg])
    axs[0][0].set_ylabel("Entry region of the users")
    fig.tight_layout(w_pad=0.5)
    F.save(fig, f"fig06_entry_{q}",
           f"{q} latency (ms) seen by the users of each entry region, mean over repetitions; "
           "last row: all users. DMOS improves the users of every entry region except SG, the farthest "
           "one, whose frontends it keeps to the minimum.")


def fig07_phase_latency(F: Fig, regime="con"):
    """Latency per load phase."""
    arms = F.arms(regime, CORE + ["reach-pool-ps-min0", "reach_green-pool-ps-min0"])
    names = [p for p, *_ in jd.PHASES]
    fig, axs = plt.subplots(1, 3, figsize=(DCOL, 2.3), sharex=True)
    for ax, q, letter in zip(axs, ("p50", "p95", "p99"), "abc"):
        dx = np.linspace(-0.25, 0.25, len(arms))
        for a, d in zip(arms, dx):
            est = [F.ci(a, regime, f"{q}_{p}") for p in names]
            x = np.arange(len(names)) + d
            for xi, e in zip(x, est):
                errpoint(ax, xi, e[0], yerr=asym(*e), arm=a, ms=4)
        ax.set_xticks(range(len(names)), [PHASE_LABEL.get(p, p) for p in names], rotation=30, ha="right")
        for k in range(1, len(names)):
            ax.axvline(k - 0.5, color=GRIDC, lw=0.6, zorder=0)
        ax.set_ylabel(f"{q} latency (ms)")
        ax.grid(axis="x", visible=False)
        panel(ax, letter, f"{q} by phase, {REGIMES[regime]}")
    fig.legend(handles=arm_handles(arms, line=True), loc="upper center", ncol=len(arms),
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.5)
    F.save(fig, f"fig07_phase_latency_{regime}",
           f"Latency per load phase (k6), {REGIMES[regime]} regime, mean and 95% CI over repetitions. "
           "The gain of DMOS concentrates in the ramp and peak phases; its p99 in the ramp is above "
           "the static fleet and below HPA.")


def fig08_scaler(F: Fig, regime="con", arm="reach-pool-ps"):
    """Level 2 at work: predicted demand and the replica budget."""
    runs = F.ds.group(arm, regime)
    if not runs:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.4), sharex=True)
    ax = axs[0]
    shade_phases(ax)
    for col, c, ls, lab in [("load_http", INK2, "-", "Offered HTTP load (k6)"),
                            ("fast_rps_total", "#2a78d6", "-", "Measured by DMOS (30 s)"),
                            ("forecast_rps_total", "#eb6834", (0, (4, 1.5)), "Forecast (Holt)")]:
        m, lo, hi = mean_band(F.ds, runs, col)
        ax.fill_between(jd.GRID, lo, hi, color=c, alpha=0.12, lw=0)
        ax.plot(jd.GRID, m, color=c, lw=2.0 if col == "load_http" else 1.1, ls=ls,
                alpha=0.45 if col == "load_http" else 1, label=lab)
    m, _, _ = mean_band(F.ds, runs, "ready_total")
    ax.step(jd.GRID, m * jd.CAP_RPS, where="post", color=INK, lw=1.0, label="Ready capacity")
    ax.set_ylabel("Requests per second")
    ax.set_xlabel("Run time (s)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=6.3, ncol=2, columnspacing=0.8)
    panel(ax, "a", "Demand, forecast and capacity")
    ax = axs[1]
    shade_phases(ax)
    for col, c, ls, lab in [("need", INK2, "-", "Required"), ("target_total", "#eb6834", (0, (4, 1.5)),
                                                                "DMOS target"), ("ready_total", INK, "-", "Ready")]:
        m, lo, hi = mean_band(F.ds, runs, col)
        ax.fill_between(jd.GRID, lo, hi, step="post", color=c, alpha=0.12, lw=0)
        ax.step(jd.GRID, m, where="post", color=c, lw=2.0 if col == "need" else 1.1, ls=ls,
                alpha=0.45 if col == "need" else 1, label=lab)
    lags = [x for r in runs for x in actuation_lags(r)]
    if lags:
        ax.text(0.98, 0.04, f"learned actuation lag: median {np.median(lags):.0f} s "
                f"({len(lags)} samples)", transform=ax.transAxes, ha="right", fontsize=6, color=INK2)
    ax.set_ylabel("Frontend replicas")
    ax.set_xlabel("Run time (s)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=6.3, ncol=3, columnspacing=0.8)
    panel(ax, "b", "Replica budget and actuation")
    fig.tight_layout(w_pad=1.5)
    F.save(fig, f"fig08_predictive_scaler_{regime}",
           f"Level 2 (predictive scaler), {ARMS[arm][0]}, {REGIMES[regime]} regime, mean over {len(runs)} "
           "runs (band: min–max). (a) HTTP load offered by k6 (about 1.2x the nominal profile: k6 follows "
           "the redirects of the POST endpoints), demand measured by DMOS on the 30-s window, its Holt "
           "forecast at the learned horizon, and the capacity of the ready replicas. (b) Replicas "
           "required, DMOS target and ready replicas: the gap between target and ready is the actuation "
           "lag that DMOS learns online.")


LAG_RE = re.compile(r"actuation lag \S+: (\d+(?:\.\d+)?)s -> learned")


def actuation_lags(run) -> list:
    """Actuation-lag samples (scale command -> all replicas ready) logged by the scaler."""
    p = run.path / "dmos.log"
    if not p.is_file():
        return []
    return [float(x) for x in LAG_RE.findall(p.read_text(encoding="utf-8", errors="replace"))]


def served_share_panel(F: Fig, ax, regime, letter, arms=None, legend_y=-0.2, ncol=3):
    """Stacked bars: share of the requests served by each region, cleanest region first."""
    arms = arms or F.arms(regime)
    regs = list(jd.REGION.values())
    ref = F.ds.ts(F.ds.runs[0])
    ci_mean = {r: float(ref[f"ci_{r}"].mean()) for r in regs}
    order = sorted(regs, key=ci_mean.get)
    for j, a in enumerate(arms[::-1]):
        left = 0.0
        for r in order:
            v = np.nanmean(F.vals(a, regime, f"served_share_{r}")) * 100
            ax.barh(j, v, left=left, color=REGION_COLOR[r], height=0.7, edgecolor="white", lw=0.8)
            if v >= 3.5:
                ax.text(left + v / 2, j, f"{v:.0f}", ha="center", va="center", fontsize=5.2, color="white")
            left += v
    ax.set_yticks(range(len(arms)), [ARMS[a][0] for a in arms[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Requests served by region (%)")
    ax.grid(axis="y", visible=False)
    panel(ax, letter, f"Served requests, {REGIMES[regime]}")
    ax.legend(handles=[Patch(facecolor=REGION_COLOR[r], label=f"{r} ({ci_mean[r]:.0f})") for r in order],
              loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=ncol, fontsize=6, columnspacing=0.8,
              handletextpad=0.3, title="Region (mean gCO$_2$/kWh during the run), cleanest first",
              title_fontsize=6)


def fig09_sustainability(F: Fig, regime="con"):
    """Where requests are served and what it costs in carbon."""
    arms = F.arms(regime)
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    served_share_panel(F, axs[0], regime, "a", arms, ncol=6)
    ax = axs[1]
    for a in arms:
        x, y = F.vals(a, regime, "co2_ug_per_req"), F.vals(a, regime, "p95")
        ax.scatter(x, y, s=9, marker=ARMS[a][2], facecolor="none", edgecolor=ARMS[a][1], lw=0.6, alpha=0.7)
        cx, cy = F.ci(a, regime, "co2_ug_per_req"), F.ci(a, regime, "p95")
        errpoint(ax, cx[0], cy[0], asym(*cx), asym(*cy), a)
    ax.set_xlabel("CO$_2$ per request (µg, application CPU)")
    ax.set_ylabel("p95 latency (ms)")
    ax.margins(0.12)
    panel(ax, "b", "Carbon–latency trade-off")
    ax.legend(handles=arm_handles(arms), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6,
              ncol=1, handletextpad=0.3)
    fig.tight_layout(w_pad=1.5)
    F.save(fig, f"fig09_sustainability_{regime}",
           "Sustainability. (a) Share of the requests served by each region (frontend request rate), "
           "regions ordered by their mean carbon intensity. (b) CO$_2$ per request of the application CPU "
           "(power model, for comparison between arms) against p95; mean with 95% CI and single runs. "
           "The static fleet has the lowest CO$_2$ per request because it runs the fewest pods.")


def fig10_ablation(F: Fig, regime="con"):
    """What each design choice adds: locality term -> local-demand gate -> DMOS -> min 0 -> green."""
    arms = F.arms(regime, ABLATION)
    if len(arms) < 2:
        return
    mets = [("p95", "p95 latency (ms)"), ("fe_rm", "Frontend replica-min"), ("co2_ug_per_req", "CO$_2$ per request (µg)")]
    fig, axs = plt.subplots(1, 3, figsize=(DCOL, 2.5), sharex=True)
    rng = np.random.default_rng(1)
    x = np.arange(len(arms))
    for ax, (col, lab), letter in zip(axs, mets, "abc"):
        means = []
        for i, a in enumerate(arms):
            v = F.vals(a, regime, col)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=8, marker=ARMS[a][2], facecolor="none",
                       edgecolor=ARMS[a][1], lw=0.6, alpha=0.75, zorder=3)
            e = F.ci(a, regime, col)
            errpoint(ax, i + 0.22, e[0], yerr=asym(*e), arm=a, ms=5)
            means.append(e[0])
        split = arms.index("reach-pool-ps") + 1 if "reach-pool-ps" in arms else len(arms)
        ax.plot(x[:split] + 0.22, means[:split], color=INK2, lw=0.6, zorder=2)   # F3: demand signal
        ax.plot(x[split - 1:] + 0.22, means[split - 1:], color=INK2, lw=0.6, ls=":", zorder=2)  # F2
        if split < len(arms):
            ax.axvline(split - 0.5 + 0.11, color=GRIDC, lw=0.8, zorder=0)
        ax.margins(y=0.18)
        for i in range(1, len(arms)):
            d = 100 * (means[i] / means[i - 1] - 1)
            ax.text(i - 0.39, 1.0, f"{d:+.1f}%", transform=ax.get_xaxis_transform(), fontsize=5.5,
                    color=INK2, ha="center", va="top", zorder=5)
        if "hpa-none" in F.arms(regime):
            h = np.nanmean(F.vals("hpa-none", regime, col))
            ax.axhline(h, color=ARMS["hpa-none"][1], lw=0.8, ls=(0, (4, 2)))
            ax.text(len(arms) - 0.5, h, "HPA", color=ARMS["hpa-none"][1], fontsize=6, va="bottom", ha="right")
        ax.set_xticks(x, [ARMS[a][0] for a in arms], rotation=30, ha="right")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        panel(ax, letter, lab.split(" (")[0])
    fig.tight_layout(w_pad=1.2)
    F.save(fig, f"fig10_ablation_{regime}",
           f"Ablation, {REGIMES[regime]} regime. Left of the separator, the demand signal of the score "
           "(family F3): demand only in the network term, which also keeps the 'total' budget instead "
           "of the pooled one (Locality term); a gate on each cluster's own demand (Local-demand gate); "
           "the gate on reachable demand ψ (DMOS). Right, the sustainability steps (family F2): clusters "
           "allowed to hold no frontend (min 0) and carbon-first weights (DMOS-Green). Top labels: change "
           "from the previous arm; dashed line: Per-cluster HPA.")


def fig11_cost(F: Fig, regime="con"):
    """The price: failed requests by phase, over- and under-provisioning."""
    arms = F.arms(regime)
    names = [p for p, *_ in jd.PHASES]
    ramp = ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]       # sequential: phase order
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.5))
    ax = axs[0]
    for j, a in enumerate(arms):
        bottom = 0.0
        for p, c in zip(names, ramp):
            v = np.nanmean(F.vals(a, regime, f"err_{p}"))
            ax.bar(j, v, bottom=bottom, color=c, width=0.65, edgecolor="white", lw=0.6)
            bottom += v
        rate = np.nanmean(F.vals(a, regime, "err_pct"))
        ax.text(j, bottom, f"{rate:.3f}%", ha="center", va="bottom", fontsize=5.8, color=INK2)
    ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], rotation=30, ha="right")
    ax.set_ylabel("Failed requests per run")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(facecolor=c, label=p) for p, c in zip(names, ramp)], fontsize=6,
              loc="upper left", title="Phase", title_fontsize=6)
    panel(ax, "a", "Failed requests (label: error rate)")
    ax = axs[1]
    for j, a in enumerate(arms):
        ex, de = np.nanmean(F.vals(a, regime, "excess_rm")), np.nanmean(F.vals(a, regime, "deficit_rm"))
        ax.bar(j, ex, color="#3987e5", width=0.65)
        ax.bar(j, -de, color="#eb6834", width=0.65)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], rotation=30, ha="right")
    ax.set_ylabel("Replica-minutes")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(facecolor="#3987e5", label="Above need (over-provisioning)"),
                       Patch(facecolor="#eb6834", label="Below need (under-provisioning)")],
              fontsize=6, loc="lower right")
    panel(ax, "b", "Distance from the required replicas")
    fig.tight_layout(w_pad=1.5)
    F.save(fig, f"fig11_cost_{regime}",
           f"Costs in the {REGIMES[regime]} regime. (a) Failed requests per run by load phase (mean over "
           "repetitions): the elastic arms fail mostly in the cooldown, while they scale in. "
           "(b) Frontend replica-minutes above and below the replicas required, " + NEED_TXT)


def fig12_stability(F: Fig, arms=("off", "hpa-none", "reach-pool-ps")):
    """Repeatability: the same arm over the days and hours of the campaign."""
    regs = F.regimes()
    fig, axs = plt.subplots(len(regs), 2, figsize=(DCOL, 1.45 * len(regs) + 0.4), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.6, 1]))
    used = set()
    blocks = sorted(b for b in set(F.df.block) if b)
    for i, reg in enumerate(regs):
        ax = axs[i][0]
        for a in arms:
            d = F.df[(F.df.arm == a) & (F.df.regime == reg)]
            if d.empty:
                continue
            used.add(a)
            ax.plot(pd.to_datetime(d.start), d["p95"], ARMS[a][2], color=ARMS[a][1], ms=4, mec="white", mew=0.5)
        ax.set_ylabel("p95 (ms)")
        ax.text(-0.25, 0.5, REGIMES[reg], transform=ax.transAxes, rotation=90, va="center", ha="center",
                fontsize=7.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d/%m\n%H:%M"))
        # right: DMOS against HPA within each repetition (a drift of the testbed cancels out)
        ax = axs[i][1]
        for col, mk, lab in [("p95", "o", "p95"), ("fe_rm", "s", "replica-min")]:
            y = []
            for b in blocks:
                d = F.df[(F.df.regime == reg) & (F.df.block == b)]
                r_, h_ = d[d.arm == "reach-pool-ps"][col], d[d.arm == "hpa-none"][col]
                y.append(100 * (r_.mean() / h_.mean() - 1) if len(r_) and len(h_) else np.nan)
            ax.plot(range(len(blocks)), y, mk + "-", color=ARMS["reach-pool-ps"][1], ms=3.5, lw=0.7,
                    mfc="white" if col == "fe_rm" else ARMS["reach-pool-ps"][1], label=lab)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(range(len(blocks)), blocks)
        ax.set_ylabel("DMOS vs HPA (%)")
        if i == 0:
            ax.legend(fontsize=6, loc="lower right", ncol=2, handletextpad=0.3)
    axs[-1][0].set_xlabel("Run start")
    axs[-1][1].set_xlabel("Repetition")
    fig.legend(handles=arm_handles([a for a in arms if a in used]), loc="upper center", ncol=3,
               bbox_to_anchor=(0.4, 1.03))
    fig.tight_layout(h_pad=0.6)
    F.save(fig, "fig12_repeatability",
           "Repeatability over the campaign. Left: p95 of every run of the control arms (static, HPA) and "
           "of DMOS, interleaved over the days; the gap of about 7 h on 3/11 is the VM outage recorded in "
           "the plan. Right: DMOS against HPA within each repetition, for p95 and frontend replica-minutes.")


def fig13_pool(F: Fig):
    """Assumption 1: each cluster serves the share of requests of its share of ready replicas."""
    regs = list(jd.REGION.values())
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.5), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axs[0]
    for r in regs:
        ax.scatter(100 * F.df[f"rep_share_{r}"], 100 * F.df[f"served_share_{r}"], s=7, color=REGION_COLOR[r],
                   alpha=0.75, lw=0, label=r)
    lim = 100 * np.nanmax(F.df[[f"served_share_{r}" for r in regs]].to_numpy()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.6, ls=(0, (3, 1.5)))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Share of ready replicas (%)")
    ax.set_ylabel("Share of requests served (%)")
    ax.legend(fontsize=6, ncol=2, loc="upper left", handletextpad=0.2, columnspacing=0.6, markerscale=1.5)
    panel(ax, "a", "Every run and region")
    ax = axs[1]
    arms = [a for a in ARMS if a in set(F.df.arm)]
    rng = np.random.default_rng(2)
    for i, a in enumerate(arms):
        v = pd.to_numeric(F.df.loc[F.df.arm == a, "pool_dev_pp"], errors="coerce").dropna()
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=8, marker=ARMS[a][2], facecolor="none",
                   edgecolor=ARMS[a][1], lw=0.6)
    ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], rotation=35, ha="right")
    ax.set_ylabel("Largest gap, served − replica share (pp)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="x", visible=False)
    panel(ax, "b", "Per run, largest gap over the regions")
    fig.tight_layout(w_pad=1.5)
    F.save(fig, "fig13_pool_routing",
           "Validation of Assumption 1 (pool routing). (a) Share of the requests served by each region "
           "against its share of ready frontend replicas, weighted by the request rate, one point per run "
           "and region. (b) Largest deviation per run: below 2 percentage points in every run, so no run "
           "is excluded by the criterion of Section 3.")


def fig14_shift(F: Fig, regime="con-tosg", arms=CORE):
    """The hot spot moves from DE to SG at 450 s: latency before, during and after, and SG replicas."""
    arms = [a for a in arms if a in F.arms(regime)]
    if not arms:
        return
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.45), gridspec_kw=dict(width_ratios=[1, 1.4]))
    ax = axs[0]
    wins = [("p95_shift_pre", "pre\n300–450 s"), ("p95_shift_mix", "shift\n450–480 s"),
            ("p95_shift_post", "post\n480–600 s")]
    dx = np.linspace(-0.25, 0.25, len(arms))
    for a, d in zip(arms, dx):
        for k, (col, _) in enumerate(wins):
            e = F.ci(a, regime, col)
            errpoint(ax, k + d, e[0], yerr=asym(*e), arm=a, ms=4)
    ax.set_xticks(range(len(wins)), [w for _, w in wins])
    for k in (0.5, 1.5):
        ax.axvline(k, color=GRIDC, lw=0.6, zorder=0)
    ax.set_ylabel("p95 latency (ms)")
    ax.grid(axis="x", visible=False)
    panel(ax, "a", "p95 around the shift")
    ax = axs[1]
    shade_phases(ax, labels=False)
    ax.axvline(450, color=INK2, lw=0.6, ls=":")
    ax.text(452, 0.97, "DE→SG", transform=ax.get_xaxis_transform(), fontsize=6, color=INK2, va="top")
    for a in arms:
        for col, ls in (("ready_SG", "-"), ("ready_DE", (0, (3, 1.5)))):
            m, lo, hi = mean_band(F.ds, F.ds.group(a, regime), col)
            ax.step(jd.GRID, m, where="post", color=ARMS[a][1], lw=1.1, ls=ls)
    ax.set_xlim(150, 750)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("Run time (s)")
    ax.set_ylabel("Ready frontend replicas")
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.1, label="SG"),
                       Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 1.5)), label="DE")],
              fontsize=6, loc="lower left")
    panel(ax, "b", "Replicas on the old and the new hot region")
    fig.legend(handles=arm_handles(arms, line=True), loc="upper center", ncol=len(arms), bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=1.5)
    F.save(fig, "fig14_shift_" + regime,
           f"{REGIMES[regime]} regime: 85% of the demand enters DE until 450 s, then moves to SG within 30 s. "
           "(a) p95 before, during and after the shift (mean and 95% CI). (b) Ready frontend replicas on "
           "SG (solid) and DE (dashed), mean over repetitions. SG holds at most 2 frontends (cap), so no arm "
           "can follow the demand there and DMOS loses after the shift the advantage it has before it.")


def fig15_placement_model(F: Fig):
    """Measured placement against the model: expected RTT J of the replicas each arm deployed,
    the J-optimal split of the same number of replicas (caps, floors) and the even split."""
    regs = F.regimes()
    fig, axs = plt.subplots(1, len(regs), figsize=(DCOL, 2.5), sharey=True, squeeze=False)
    for ax, reg in zip(axs[0], regs):
        arms = [a for a in F.arms(reg) if a != "off"]
        for i, a in enumerate(arms):
            for col, mk, fc in (("exp_rtt_peak_ms", ARMS[a][2], ARMS[a][1]), ("exp_rtt_opt_peak_ms", "_", INK),
                                ("exp_rtt_even_peak_ms", "_", INK2)):
                e = F.ci(a, reg, col)
                if col == "exp_rtt_peak_ms":
                    errpoint(ax, i, e[0], yerr=asym(*e), arm=a, ms=5)
                else:
                    ax.plot([i - 0.3, i + 0.3], [e[0], e[0]], color=fc, lw=1.4 if "opt" in col else 0.9,
                            ls="-" if "opt" in col else (0, (2, 1.2)))
        ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], rotation=40, ha="right", fontsize=6)
        ax.grid(axis="x", visible=False)
        panel(ax, "abc"[regs.index(reg)], REGIMES[reg])
    axs[0][0].set_ylabel("Expected RTT J, peak phase (ms)")
    axs[0][0].set_ylim(bottom=0)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.4, label="Latency optimum, same replicas (caps, floors)"),
                        Line2D([], [], color=INK2, lw=0.9, ls=(0, (2, 1.2)), label="Even split, same replicas")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.8)
    F.save(fig, "fig15_placement_model",
           "Placement against the model of Section 3, peak phase (300–600 s). Marker: expected RTT J of "
           "the ready replicas of each arm (eq. J, measured RTT matrix and entry shares). Solid: latency "
           "optimum for the same number of replicas with the caps and the per-cluster floor (Section 3.5). "
           "Dashed: even split of the same replicas. DMOS is at the optimum in the balanced and concentrated "
           "regimes and about 3 ms above it in the shifting one; the remaining J is set by the budget and "
           "the floors, not by the placement.")


def fig16_carbon_bound(F: Fig, regs=("bal", "con")):
    """Carbon intensity served against the cleanest and dirtiest placement of the same replicas."""
    regs = [r for r in regs if r in F.regimes()]
    fig = plt.figure(figsize=(DCOL, 2.75))
    gs = fig.add_gridspec(1, len(regs) + 1, width_ratios=[1] * len(regs) + [1.15], wspace=0.75)
    axs = [[fig.add_subplot(gs[0, 0])]]
    axs[0] += [fig.add_subplot(gs[0, k], sharey=axs[0][0]) for k in range(1, len(regs))]
    for ax, reg in zip(axs[0], regs):
        arms = [a for a in F.arms(reg) if a != "off"]
        for i, a in enumerate(arms):
            best, worst = np.nanmean(F.vals(a, reg, "ci_best_g_kwh")), np.nanmean(F.vals(a, reg, "ci_worst_g_kwh"))
            obs = F.ci(a, reg, "ci_served_g_kwh")
            ax.plot([i, i], [best, worst], color=GRIDC, lw=5, solid_capstyle="butt", zorder=1)
            for y in (best, worst):
                ax.plot([i - 0.18, i + 0.18], [y, y], color=INK2, lw=0.8, zorder=2)
            errpoint(ax, i, obs[0], yerr=asym(*obs), arm=a, ms=5)
            share = 100 * (worst - obs[0]) / (worst - best) if worst > best else float("nan")
            ax.text(i, worst + 6, f"{share:.0f}%", ha="center", va="bottom", fontsize=6, color=INK2)
        ax.set_xticks(range(len(arms)), [ARMS[a][0] for a in arms], rotation=35, ha="right", fontsize=6.3)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        panel(ax, "ab"[regs.index(reg)], REGIMES[reg])
    axs[0][0].set_ylabel("CI of served requests (gCO$_2$/kWh)")
    axs[0][0].set_ylim(150, 340)
    for ax in axs[0][1:]:
        ax.tick_params(labelleft=False)
    reg_c = "con" if "con" in F.regimes() else regs[-1]
    ax_c = fig.add_subplot(gs[0, len(regs)])
    served_share_panel(F, ax_c, reg_c, "abc"[len(regs)], [a for a in F.arms(reg_c) if a != "off"],
                       legend_y=-0.24)
    ax_c.tick_params(axis="y", labelsize=6.3)
    F.save(fig, "fig16_carbon_bound",
           "Carbon intensity of the served requests against what the same replicas could attain. Bar: from "
           "the cleanest placement (bottom tick) to the dirtiest one (top tick) of the ready replicas at every "
           "step, with the same caps and per-cluster floor, requests following the replica shares. Marker: "
           "observed value (mean and 95% CI). Label: share of the attainable improvement reached, from "
           "the dirtiest placement. Last panel: share of the requests served by each region, cleanest first.")


# ── tables ──────────────────────────────────────────────────────────────────

SUMMARY_COLS = ["p50", "p95", "p99", "p95_peak", "exp_rtt_ms", "err_pct", "err_n", "fe_rm", "all_rm", "excess_rm",
                "deficit_rm", "peak_fe", "t_capacity", "core_s_per_1k_ok", "co2_ug_per_req",
                "ci_served_g_kwh", "dmos_cpu_cores_mean", "dmos_rss_mb_max", "dmos_cycle_ms",
                "exp_rtt_peak_ms", "exp_rtt_opt_peak_ms", "exp_rtt_even_peak_ms", "pool_dev_pp", "pods_started", "ci_best_g_kwh", "ci_worst_g_kwh",
                "decay_added", "sg_zero_s"] + [f"err_{p}" for p, *_ in jd.PHASES] + [f"p99_{p}" for p, *_ in jd.PHASES]
DESC_COLS = [("p99", "{:.0f}")] + [(f"err_{p}", "{:.1f}") for p, *_ in jd.PHASES] + \
            [("pods_started", "{:.1f}"), ("decay_added", "{:.1f}"), ("sg_zero_s", "{:.0f}"), ("pool_dev_pp", "{:.1f}")]


def tables(F: Fig):
    F.df.to_csv(F.out / "runs.csv", index=False)
    rows = []
    for reg in F.regimes():
        for a in F.arms(reg):
            row = {"regime": reg, "arm": a, "label": ARMS[a][0], "n": len(F.ds.group(a, reg))}
            for c in SUMMARY_COLS:
                e, lo, hi = F.ci(a, reg, c)
                row[c], row[c + "_lo"], row[c + "_hi"] = e, lo, hi
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(F.out / "summary.csv", index=False)
    S[["regime", "label", "n", "dmos_cpu_cores_mean", "dmos_rss_mb_max", "dmos_cycle_ms"]] \
        .dropna(subset=["dmos_rss_mb_max"]).to_csv(F.out / "overhead.csv", index=False)

    def cell(a, reg, c, fmt):
        v = F.vals(a, reg, c)
        if len(v) == 0:
            return "--"
        return (fmt.format(v.mean()) + (r"\,$\pm$\," + fmt.format(v.std(ddof=1)) if len(v) > 1 else ""))
    cols = [("p50", "{:.0f}"), ("p95", "{:.0f}"), ("p99", "{:.0f}"), ("err_pct", "{:.3f}"),
            ("fe_rm", "{:.0f}"), ("co2_ug_per_req", "{:.2f}"), ("ci_served_g_kwh", "{:.0f}")]
    lines = [r"\begin{tabular}{llr" + "r" * len(cols) + "}", r"\toprule",
             r"Regime & Arm & $n$ & p50 & p95 & p99 & Err.\,\% & Repl.-min & CO$_2$/req ($\mu$g) & CI served \\",
             r" & & & (ms) & (ms) & (ms) & & frontend & & (g/kWh) \\", r"\midrule"]
    for reg in F.regimes():
        for k, a in enumerate(F.arms(reg)):
            lines.append((REGIMES[reg] if k == 0 else "") + f" & {ARMS[a][0]} & {len(F.ds.group(a, reg))} & "
                         + " & ".join(cell(a, reg, c, f) for c, f in cols) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (F.out / "table_main.tex").write_text("\n".join(lines).replace("→", r"$\to$"), encoding="utf-8")

    # descriptives promised by the plan (no tests): errors per phase, p99, pods started,
    # replicas added in the decay, SG at 0, largest pool-routing gap
    phases = [p for p, *_ in jd.PHASES]
    lines = [r"\begin{tabular}{ll" + "r" * len(DESC_COLS) + "}", r"\toprule",
             r"Regime & Arm & p99 & \multicolumn{5}{c}{Failed requests per phase} & Pods & Added in & SG at 0 & Pool gap \\",
             r"\cmidrule(lr){4-8}",
             r" & & (ms) & " + " & ".join(phases) + r" & started & decay & (s) & (pp) \\", r"\midrule"]
    for reg in F.regimes():
        for k, a in enumerate(F.arms(reg)):
            row = []
            for c, f in DESC_COLS:
                v = F.vals(a, reg, c)
                row.append(f.format(v.mean()) if len(v) else "--")
            lines.append((REGIMES[reg] if k == 0 else "") + f" & {ARMS[a][0]} & " + " & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (F.out / "table_desc.tex").write_text("\n".join(lines).replace("→", r"$\to$"), encoding="utf-8")

    ov = S.dropna(subset=["dmos_rss_mb_max"])
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Regime & Arm & CPU (cores, mean) & RSS (MB, max) & Decision time (s, mean) \\", r"\midrule"]
    for _, r in ov.iterrows():
        lines.append(f"{REGIMES[r.regime]} & {r.label} & {r.dmos_cpu_cores_mean:.3f} & {r.dmos_rss_mb_max:.0f} & "
                     f"{r.dmos_cycle_ms / 1000:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (F.out / "table_overhead.tex").write_text("\n".join(lines).replace("→", r"$\to$"), encoding="utf-8")
    print("  summary.csv, runs.csv, overhead.csv, table_main.tex, table_desc.tex, table_overhead.tex")


FIGS = [fig01_testbed, fig02_tradeoff, fig03_forest, fig04_trajectories, fig05_dynamics,
        fig06_entry_latency, fig07_phase_latency, fig08_scaler, fig09_sustainability, fig10_ablation,
        fig11_cost, fig12_stability, fig13_pool, fig14_shift, fig15_placement_model, fig16_carbon_bound]
PER_REGIME = {fig04_trajectories, fig05_dynamics, fig07_phase_latency, fig11_cost}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--tsv", type=Path, action="append", default=[])
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE.parent / "paper" / "figures" / "journal")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    if not a.proxy and not a.tsv:
        ap.error("--tsv <campaign tsv> or --proxy")
    a.out.mkdir(parents=True, exist_ok=True)
    ds = jd.load(a.results, a.tsv, a.proxy, None if a.no_cache else a.out / "_cache")
    if not ds.runs:
        print("no runs", file=sys.stderr)
        return 1
    style()
    F = Fig(ds, a.out)
    print(f"{len(ds.runs)} runs -> {a.out}")
    only = [s for s in a.only.split(",") if s]
    for f in FIGS:
        if only and not any(f.__name__.startswith(o) for o in only):
            continue
        for args in ([(reg,) for reg in F.regimes()] if f in PER_REGIME else [()]):
            try:
                f(F, *args)
            except Exception as e:      # a partial campaign lacks some groups: skip that figure only
                plt.close("all")
                print(f"  {f.__name__}{args}: skipped ({type(e).__name__}: {e})", file=sys.stderr)
    tables(F)
    src = "PROXY (pilot runs)" if a.proxy else ", ".join(p.name for p in a.tsv)
    md = [f"# Journal figures — {len(ds.runs)} runs, {src}", ""]
    md += [f"## {name}\n\n![{name}]({name}.png)\n\n{cap}\n" for name, cap in F.made]
    (a.out / "index.md").write_text("\n".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
