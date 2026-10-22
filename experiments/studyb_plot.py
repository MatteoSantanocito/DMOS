#!/usr/bin/env python3
"""
studyb_plot.py — Figura dello Studio B: trade-off carbon vs P95.
================================================================================

COSA MOSTRA
-----------
Due run identiche tranne il profilo di pesi DMOS:
    - 'latency'  (latency_priority, ω_lat=0.40)  → traffico verso il cluster
                  veloce e centrale (DE, no netem): P95 basso, CI alta.
    - 'carbon'   (carbon_priority,  ω_car=0.40)  → la redistribuzione GSLB sposta
                  il traffico sul cluster verde (FR, nucleare ma +150ms netem):
                  CI_eff bassa, P95 piu' alta.

Lo stesso score multi-obiettivo pilota sia lo scaling L1/L2 sia i pesi Envoy del
2o livello: cambiando profilo cambia la ripartizione del traffico servito, e con
essa la carbon-intensity efficace. La figura rende visibile il trade-off:
"−X% di CO2 al costo di +Y ms di P95" → il multi-obiettivo e' azionabile.

Pannello (a): frazione di richieste servite per cluster, una barra impilata per
              run → mostra dove la redistribuzione manda il traffico.
Pannello (b): trade-off a 2 punti  CI_eff (x)  vs  P95 (y), con annotato il
              risparmio carbon e il costo in latenza.

INPUT  (prodotti da studyb_collect.py finalize)
  studyb_latency.json , studyb_carbon.json

USO
  python experiments/studyb_plot.py \
      --latency studyb_latency.json --carbon studyb_carbon.json
  python experiments/studyb_plot.py ... --output-dir paper_artifacts/studyb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from paper_lib import setup_matplotlib, save_figure  # noqa: E402

# Etichette regione per i cluster carbon (DE/FR/PL) — vedi config/clusters.yaml
CLUSTER_REGION = {"01": "DE", "02": "FR", "03": "PL",
                  "04": "c04", "05": "c05", "06": "c06"}
REGION_COLOR = {"DE": "#8c8c8c", "FR": "#2ca02c", "PL": "#7f3b08",
                "c04": "#c7c7c7", "c05": "#c7c7c7", "c06": "#c7c7c7"}
RUN_LABEL = {"latency": "latency_priority", "carbon": "carbon_priority"}
RUN_COLOR = {"latency": "#1f77b4", "carbon": "#2ca02c"}


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", required=True, help="studyb_latency.json")
    ap.add_argument("--carbon", required=True, help="studyb_carbon.json")
    ap.add_argument("--output-dir", default="paper_artifacts/studyb")
    ap.add_argument("--basename", default="fig_studyb_tradeoff")
    args = ap.parse_args()

    runs = {"latency": load(args.latency), "carbon": load(args.carbon)}

    setup_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # ── Pannello (a): frazione servita per cluster (barre impilate) ──────────
    order = ["latency", "carbon"]
    xpos = np.arange(len(order))
    # cluster presenti (con traffico in almeno una run)
    clusters = [c for c in ["01", "02", "03", "04", "05", "06"]
                if any(runs[r]["served_fraction"].get(c, 0) > 0 for r in order)]
    bottom = np.zeros(len(order))
    for c in clusters:
        vals = np.array([runs[r]["served_fraction"].get(c, 0) * 100 for r in order])
        reg = CLUSTER_REGION[c]
        axA.bar(xpos, vals, bottom=bottom, width=0.55,
                color=REGION_COLOR[reg], edgecolor="white", linewidth=0.6,
                label=f"{reg}")
        for i, v in enumerate(vals):
            if v >= 6:
                axA.text(xpos[i], bottom[i] + v / 2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=8,
                         color="white", fontweight="bold")
        bottom += vals
    axA.set_xticks(xpos)
    axA.set_xticklabels([RUN_LABEL[r] for r in order], fontsize=8)
    axA.set_ylabel("Traffico servito per cluster (%)")
    axA.set_ylim(0, 100)
    axA.set_title("(a) Redistribuzione del traffico")
    axA.legend(title="cluster", fontsize=8, title_fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, 1.13),
               ncol=len(clusters), framealpha=0.9, columnspacing=1.0)
    axA.grid(axis="x", visible=False)

    # ── Pannello (b): trade-off CI_eff vs P95 ────────────────────────────────
    for r in order:
        x = runs[r]["ci_eff_gco2_kwh"]
        y = runs[r]["p95_ms"]
        axB.scatter(x, y, s=90, color=RUN_COLOR[r], zorder=3,
                    edgecolor="black", linewidth=0.6, label=RUN_LABEL[r])
        axB.annotate(RUN_LABEL[r], (x, y), textcoords="offset points",
                     xytext=(8, 6), fontsize=8)
    # freccia trade-off latency -> carbon
    xl, yl = runs["latency"]["ci_eff_gco2_kwh"], runs["latency"]["p95_ms"]
    xc, yc = runs["carbon"]["ci_eff_gco2_kwh"], runs["carbon"]["p95_ms"]
    axB.annotate("", xy=(xc, yc), xytext=(xl, yl),
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=1.3,
                                 connectionstyle="arc3,rad=0.15"))
    savings = (xl - xc) / xl * 100 if xl else 0.0
    d_p95 = yc - yl
    axB.text(0.5, 0.06,
             f"−{savings:.0f}% CO$_2$   /   +{d_p95:.0f} ms P95",
             transform=axB.transAxes, ha="center", fontsize=9,
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff7e6", ec="0.6"))
    axB.set_xlabel("CI efficace, pesata sul traffico (gCO$_2$/kWh)")
    axB.set_ylabel("P95 end-to-end (ms)")
    axB.set_title("(b) Trade-off carbon vs latenza")
    # margini
    xs = [xl, xc]; ys = [yl, yc]
    dx = (max(xs) - min(xs)) or 1; dy = (max(ys) - min(ys)) or 1
    axB.set_xlim(min(xs) - 0.30 * dx, max(xs) + 0.45 * dx)
    axB.set_ylim(min(ys) - 0.30 * dy, max(ys) + 0.30 * dy)

    fig.tight_layout()
    written = save_figure(fig, Path(args.output_dir), args.basename)

    print(f"\n  Studio B — risparmio carbon: {savings:.1f}%  "
          f"(CI_eff {xl:.0f} → {xc:.0f} gCO2/kWh)")
    print(f"  Studio B — costo latenza:   +{d_p95:.0f} ms P95 "
          f"({yl:.0f} → {yc:.0f} ms)")
    for p in written:
        print(f"  -> {p}")


if __name__ == "__main__":
    sys.exit(main())
