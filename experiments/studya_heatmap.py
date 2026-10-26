#!/usr/bin/env python3
"""
studya_heatmap.py — Heatmap "% richieste SERVITE per cluster" × arm (Study A).
================================================================================

Mostra, per il regime concentrato (con), come ciascun arm distribuisce le
richieste EFFETTIVAMENTE SERVITE fra i cluster (contatori Envoy rq_total →
attribuzione per cluster proprietario dell'endpoint, via studyb_collect).

  righe  = arm   (off / l2 / l1l2coh)
  colonne= cluster (DE/FR/PL/c04/c05/c06)
  cella  = % delle richieste totali servite da quel cluster (la riga somma 100)

Storia attesa sotto domanda concentrata su un cluster (con):
  - off / l2 : la capacità NON segue la domanda → forte spillover cross-cluster
               (servito sparso / accodato altrove) o concentrazione che satura.
  - l1l2coh  : la capacità segue la domanda → servito allineato all'origine,
               minimo spillover.

INPUT: un JSON per arm, prodotto da `studyb_collect.py finalize`
       (campo `served_fraction`).

USO
  python experiments/studya_heatmap.py --regime con \
      --runs off=studya_off.json l2=studya_l2.json l1l2coh=studya_l1l2coh.json
  python experiments/studya_heatmap.py ... --output-dir paper_artifacts/studya
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paper_lib import setup_matplotlib, save_figure, ARM_LABELS  # noqa: E402

CLUSTER_LABEL = {"01": "DE\n(c01)", "02": "FR\n(c02)", "03": "PL\n(c03)",
                 "04": "c04", "05": "c05", "06": "c06"}
ARM_ORDER = ["off", "l2", "l1l2coh"]


def row_label(arm):
    return ARM_LABELS.get(f"dmos-{arm}", arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="arm=path.json (es. off=studya_off.json ...)")
    ap.add_argument("--regime", default="con")
    ap.add_argument("--output-dir", default="paper_artifacts/studya")
    ap.add_argument("--basename", default="fig_studya_served_heatmap")
    args = ap.parse_args()

    data = {}
    for kv in args.runs:
        arm, path = kv.split("=", 1)
        data[arm] = json.load(open(path))["served_fraction"]

    arms = [a for a in ARM_ORDER if a in data] + \
           [a for a in data if a not in ARM_ORDER]
    clusters = [c for c in ["01", "02", "03", "04", "05", "06"]
                if any(data[a].get(c, 0) > 0 for a in arms)]

    setup_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    M = np.array([[data[a].get(c, 0) * 100 for c in clusters] for a in arms])

    fig, ax = plt.subplots(figsize=(1.1 * len(clusters) + 1.8,
                                    0.9 * len(arms) + 1.6))
    im = ax.imshow(M, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=max(100.0, M.max()))

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([CLUSTER_LABEL.get(c, c) for c in clusters], fontsize=9)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([row_label(a) for a in arms], fontsize=9)
    ax.set_title(f"% richieste servite per cluster — regime '{args.regime}'")

    # annota ogni cella; testo bianco sulle celle scure
    thr = 0.55 * max(100.0, M.max())
    for i in range(len(arms)):
        for j in range(len(clusters)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v >= thr else "black")

    ax.set_xticks(np.arange(-.5, len(clusters), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(arms), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% servito", fontsize=9)

    fig.tight_layout()
    written = save_figure(fig, Path(args.output_dir), args.basename)
    print(f"\n  Heatmap richieste servite (regime {args.regime}):")
    for a in arms:
        row = "  ".join(f"c{c}={data[a].get(c,0)*100:.0f}%" for c in clusters)
        print(f"    {row_label(a):20s} {row}")
    for p in written:
        print(f"  -> {p}")


if __name__ == "__main__":
    sys.exit(main())
