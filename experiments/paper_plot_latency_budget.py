#!/usr/bin/env python3
"""
paper_plot_latency_budget.py — Figura "latency budget" (rete vs processing)
============================================================================

Argomenta la distinzione L1 vs L2 decomponendo la latenza al picco in due
strati MISURATI, per arm (off / l2 / l1l2), e per regime (uni/bal/con):

  - PROCESSING (leva di L2): tempo server-side, dall'edge L7 `ingress->frontend`
    di Hubble (include il fan-out del frontend verso i backend). Sotto carico
    cresce per saturazione; lo scaling L2 lo riduce.
  - NETWORK/GEO (leva di L1): ritardo di rete della zona che SERVE la domanda,
    pesato per la quota di traffico per-cluster. = Σ_c share_c · netem_c.
    L1 piazza le repliche vicino alla domanda → riduce il geo effettivo.

Dati: <run>/dmos_live.jsonl (campo clusters[].l7_edge_p95_ms, .traffic_pct) +
netem per-cluster da .cache/test-env.json. Si aggregano gli snapshot della
fase di picco (elapsed_s in [--peak-lo, --peak-hi]) e le ripetizioni (media).

USO
  python experiments/paper_plot_latency_budget.py                 # tutti i regimi
  python experiments/paper_plot_latency_budget.py --regimes con   # solo con
  python experiments/paper_plot_latency_budget.py --table-only    # niente figura
"""

from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

ARMS = ["dmos-off", "dmos-l2", "dmos-l1l2"]
ARM_LABEL = {"dmos-off": "off", "dmos-l2": "L2", "dmos-l1l2": "L1+L2"}
PROC_EDGE = "ingress->frontend"


def load_netem() -> Dict[str, float]:
    """region/cluster_name -> netem delay_ms, da .cache/test-env.json."""
    env = json.loads((REPO_ROOT / ".cache" / "test-env.json").read_text(encoding="utf-8"))
    out = {}
    for c in env["clusters"]:
        out[c["name"]] = float((c.get("netem") or {}).get("delay_ms", 0) or 0)
    return out


def run_components(run_dir: str, netem: Dict[str, float],
                   peak_lo: float, peak_hi: float) -> Optional[Tuple[float, float]]:
    """(network_ms, processing_ms) per un run, medie sulla fase di picco."""
    jl = os.path.join(run_dir, "dmos_live.jsonl")
    if not os.path.isfile(jl):
        return None
    nets, procs = [], []
    with open(jl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                snap = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not (peak_lo <= snap.get("elapsed_s", 0) <= peak_hi):
                continue
            clusters = snap.get("clusters", {})
            wsum = pnum = nnum = 0.0
            for cl, cd in clusters.items():
                w = cd.get("traffic_pct") or 0.0
                if w <= 0:
                    continue
                p = (cd.get("l7_edge_p95_ms") or {}).get(PROC_EDGE)
                ndelay = netem.get(cl, 0.0)
                if p is None:
                    continue
                wsum += w
                pnum += w * p
                nnum += w * ndelay
            if wsum > 0:
                procs.append(pnum / wsum)
                nets.append(nnum / wsum)
    if not procs:
        return None
    return float(np.mean(nets)), float(np.mean(procs))


def collect(results_dir: Path, netem, regimes, peak_lo, peak_hi) -> Dict:
    """{regime: {arm: {'net':[..reps], 'proc':[..reps]}}}"""
    data: Dict = {}
    for reg in regimes:
        data[reg] = {}
        for arm in ARMS:
            label = f"{arm}__{reg}"
            dirs = sorted(glob.glob(str(results_dir / f"*_{label}")))
            nets, procs = [], []
            for d in dirs:
                comp = run_components(d, netem, peak_lo, peak_hi)
                if comp:
                    nets.append(comp[0])
                    procs.append(comp[1])
            data[reg][arm] = {"net": nets, "proc": procs, "n": len(nets)}
    return data


def print_table(data: Dict):
    print(f"\n  {'regime':8} {'arm':7} {'n':>2}  {'network':>9} {'processing':>11} {'totale':>9}")
    print("  " + "-" * 52)
    for reg, arms in data.items():
        for arm in ARMS:
            d = arms[arm]
            if d["n"] == 0:
                print(f"  {reg:8} {ARM_LABEL[arm]:7} {0:>2}  {'--':>9} {'--':>11} {'--':>9}")
                continue
            net = np.mean(d["net"]); proc = np.mean(d["proc"])
            print(f"  {reg:8} {ARM_LABEL[arm]:7} {d['n']:>2}  {net:9.1f} {proc:11.1f} {net+proc:9.1f}")
        print()


def plot(data: Dict, regimes, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.4 * len(regimes), 3.6),
                             squeeze=False, sharey=True)
    C_NET, C_PROC = "#4C72B0", "#DD8452"
    any_data = False
    for ax, reg in zip(axes[0], regimes):
        arms_present = [a for a in ARMS if data[reg][a]["n"] > 0]
        x = np.arange(len(arms_present))
        nets = [np.mean(data[reg][a]["net"]) for a in arms_present]
        procs = [np.mean(data[reg][a]["proc"]) for a in arms_present]
        if arms_present:
            any_data = True
        ax.bar(x, nets, 0.6, label="network (geo, L1)", color=C_NET)
        ax.bar(x, procs, 0.6, bottom=nets, label="processing (L2)", color=C_PROC)
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[a] for a in arms_present])
        ax.set_title(f"regime: {reg}")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    axes[0][0].set_ylabel("latency budget al picco (ms)")
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=2, frameon=True,
                   bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    if not any_data:
        plt.close(fig)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"fig_latency_budget.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Figura latency budget (rete vs processing).")
    ap.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    ap.add_argument("--regimes", nargs="+", default=["uni", "bal", "con"])
    ap.add_argument("--peak-lo", type=float, default=300.0)
    ap.add_argument("--peak-hi", type=float, default=600.0)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    netem = load_netem()
    print(f"  netem per-cluster: {netem}")
    data = collect(args.results_dir, netem, args.regimes, args.peak_lo, args.peak_hi)
    print_table(data)
    (args.results_dir.parent / "latency_budget_data.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")

    if not args.table_only:
        out_dir = args.output_dir or (REPO_ROOT / "paper_artifacts" / "latency_budget")
        paths = plot(data, args.regimes, out_dir)
        if paths:
            for p in paths:
                print(f"  scritto: {p}")
        else:
            print("  [warn] nessun dato sufficiente per la figura.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
