"""
analyze_baseline.py — DMOS OFF / Baseline QoS Analysis
=======================================================
Analizza i test DMOS OFF (o qualsiasi baseline senza scaling).
Prende direttamente i CSV Locust — nessun JSONL richiesto.

Usage:
    python experiments/analyze_baseline.py \\
        results/multiingress/flash_crowd_timeseries_20260326_134354.csv \\
        results/multiingress/flash_crowd_cluster_latency_20260326_134354.csv \\
        --label "Flash Crowd OFF (CPU 150m)"

Output:
    results/<scenario>_off/<prefix>_baseline_qos.png
    results/<scenario>_off/<prefix>_baseline_report.json
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
KNOWN_CLUSTERS  = ["cluster1", "cluster2", "cluster3"]
CLUSTER_REGIONS = {"cluster1": "DE", "cluster2": "FR", "cluster3": "PL"}
COLORS = {
    "cluster1": "#1f77b4",
    "cluster2": "#ff7f0e",
    "cluster3": "#2ca02c",
}
SLO_MS = 1000
W = 78  # console width


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_timeseries(csv_path: str) -> dict:
    data = {cn: {"timestamps": [], "p95_ms": [], "slo_pct": [], "count": []}
            for cn in KNOWN_CLUSTERS}
    path = Path(csv_path)
    if not path.exists():
        print(f"  ❌ File not found: {csv_path}")
        sys.exit(1)

    day_offset = 0
    prev_t = None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp", "").strip()
            if not ts_str:
                continue
            try:
                t = datetime.strptime(ts_str, "%H:%M:%S")
                if prev_t and t < prev_t:
                    day_offset += 1
                t += timedelta(days=day_offset)
                prev_t = t
            except ValueError:
                continue
            for cn in KNOWN_CLUSTERS:
                try:
                    data[cn]["timestamps"].append(t)
                    data[cn]["p95_ms"].append(float(row.get(f"{cn}_p95_ms", 0) or 0))
                    data[cn]["slo_pct"].append(float(row.get(f"{cn}_slo_pct", 0) or 0))
                    data[cn]["count"].append(float(row.get(f"{cn}_count",   0) or 0))
                except (ValueError, KeyError):
                    pass
    return data


def load_cluster_latency(csv_path: str) -> dict:
    result = {}
    path = Path(csv_path)
    if not path.exists():
        print(f"  ⚠  cluster_latency CSV not found: {csv_path}")
        return result
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cn = row.get("cluster", "").strip()
            if cn not in KNOWN_CLUSTERS:
                continue
            try:
                result[cn] = {
                    "requests":  int(float(row.get("requests", 0) or 0)),
                    "failures":  int(float(row.get("failures", 0) or 0)),
                    "fail_pct":  float(row.get("fail_pct", 0) or 0),
                    "avg_ms":    float(row.get("avg_ms",   0) or 0),
                    "p50_ms":    float(row.get("p50_ms",   0) or 0),
                    "p90_ms":    float(row.get("p90_ms",   0) or 0),
                    "p95_ms":    float(row.get("p95_ms",   0) or 0),
                    "p99_ms":    float(row.get("p99_ms",   0) or 0),
                    "slo_pct":   float(row.get("slo_pct",  0) or 0),
                }
            except (ValueError, KeyError):
                pass
    return result


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_qos_metrics(ts_data: dict, window_s: int = 10) -> dict:
    result = {}
    for cn in KNOWN_CLUSTERS:
        p95  = ts_data[cn]["p95_ms"]
        slo  = ts_data[cn]["slo_pct"]
        cnts = ts_data[cn]["count"]
        active = [v for v in p95 if v > 0]
        if not active:
            result[cn] = {}
            continue

        peak_p95    = max(active)
        peak_idx    = p95.index(peak_p95)
        mean_p95    = sum(active) / len(active)
        slo_dur_s   = sum(1 for v in p95 if v > SLO_MS) * window_s
        mean_slo    = sum(slo) / len(slo) if slo else 0

        n_warm   = max(1, len(active) // 5)
        baseline = sum(active[:n_warm]) / n_warm

        recovery_s = None
        for i in range(peak_idx + 1, len(p95)):
            if 0 < p95[i] <= baseline * 2:
                recovery_s = (i - peak_idx) * window_s
                break

        # Throughput CV during spike
        cv = None
        if cnts and baseline:
            spike_cnts = [cnts[i] for i, v in enumerate(p95)
                          if v > baseline * 1.5 and i < len(cnts)]
            if len(spike_cnts) > 1:
                m = sum(spike_cnts) / len(spike_cnts)
                if m > 0:
                    var = sum((c - m)**2 for c in spike_cnts) / len(spike_cnts)
                    cv  = var**0.5 / m * 100

        result[cn] = {
            "mean_p95_ms":    mean_p95,
            "peak_p95_ms":    peak_p95,
            "baseline_p95_ms": baseline,
            "slo_duration_s": slo_dur_s,
            "mean_slo_pct":   mean_slo,
            "recovery_s":     recovery_s,
            "throughput_cv":  cv,
        }

    global_slo = sum(r.get("slo_duration_s", 0) for r in result.values() if r)
    return {"per_cluster": result, "global_slo_duration_s": global_slo}


def compute_global_p95_flat(cluster_latency: dict) -> float | None:
    total_req = sum(cl["requests"] for cl in cluster_latency.values())
    if total_req == 0:
        return None
    return sum(cl["p95_ms"] * cl["requests"] for cl in cluster_latency.values()) / total_req


# ── Console Report ────────────────────────────────────────────────────────────
def print_report(ts_data: dict, cluster_latency: dict,
                 qos: dict, label: str, ts_csv: str, lat_csv: str):
    def header(s): print(f"\n{'═'*W}\n  {s}\n{'═'*W}")
    def kv(k, v): print(f"    {k:<42} {v}")

    print("=" * W)
    print("  DMOS BASELINE QoS REPORT")
    print("=" * W)
    print(f"  Scenario:  {label}")
    print(f"  Timeseries CSV:     {Path(ts_csv).name}")
    print(f"  Cluster latency CSV: {Path(lat_csv).name}")

    # ── Global flat p95 ──
    header("1. GLOBAL P95 (flat / k6-style)")
    kv("SLO threshold:", "1000 ms")
    if cluster_latency:
        flat = compute_global_p95_flat(cluster_latency)
        if flat is not None:
            status = "✅" if flat < SLO_MS else ("⚠️" if flat < 2000 else "❌")
            kv("p95 globale flat (pesato):", f"{flat:.0f} ms  {status}")
        print()
        print(f"    {'Cluster':<14} {'Requests':>9}  {'Avg':>7}  {'p50':>7}  "
              f"{'p90':>7}  {'p95':>7}  {'p99':>7}  {'Fail%':>6}  {'SLO%':>6}")
        print(f"    {'─'*80}")
        for cn in KNOWN_CLUSTERS:
            cl  = cluster_latency.get(cn, {})
            reg = CLUSTER_REGIONS.get(cn, "")
            if not cl:
                print(f"    {cn+' ('+reg+')':<18}  N/A")
                continue
            print(f"    {cn+' ('+reg+')':<18} {cl['requests']:>7}  "
                  f"{cl['avg_ms']:>6.0f}ms  {cl['p50_ms']:>6.0f}ms  "
                  f"{cl['p90_ms']:>6.0f}ms  {cl['p95_ms']:>6.0f}ms  "
                  f"{cl['p99_ms']:>6.0f}ms  {cl['fail_pct']:>5.1f}%  "
                  f"{cl['slo_pct']:>5.1f}%")

    # ── Advanced QoS ──
    header("2. ADVANCED QoS METRICS")
    per_cl       = qos["per_cluster"]
    global_slo_d = qos["global_slo_duration_s"]
    kv("Global SLO violation duration:", f"{global_slo_d:.0f} s  ({global_slo_d/60:.1f} min)")
    print()
    print(f"    {'Cluster':<12} {'SLO dur(s)':>10}  {'Mean p95':>9}  "
          f"{'Peak p95':>9}  {'Baseline':>9}  {'Recovery':>9}  {'Throughput CV':>13}")
    print(f"    {'─'*85}")
    for cn in KNOWN_CLUSTERS:
        m   = per_cl.get(cn, {})
        reg = CLUSTER_REGIONS.get(cn, "")
        if not m:
            print(f"    {cn+' ('+reg+')':<18}  N/A")
            continue
        slo_d = f"{m['slo_duration_s']:.0f}s"         if m.get("slo_duration_s") is not None else "N/A"
        mp95  = f"{m['mean_p95_ms']:.0f}ms"           if m.get("mean_p95_ms")    is not None else "N/A"
        pk95  = f"{m['peak_p95_ms']:.0f}ms"           if m.get("peak_p95_ms")    is not None else "N/A"
        base  = f"{m['baseline_p95_ms']:.0f}ms"       if m.get("baseline_p95_ms") is not None else "N/A"
        rec   = f"{m['recovery_s']:.0f}s"             if m.get("recovery_s")     is not None else "N/A"
        cv    = f"{m['throughput_cv']:.1f}%"           if m.get("throughput_cv")  is not None else "N/A"
        print(f"    {cn+' ('+reg+')':<18} {slo_d:>10}  {mp95:>9}  {pk95:>9}  "
              f"{base:>9}  {rec:>9}  {cv:>13}")

    print(f"\n{'═'*W}")
    print("  ✅ Baseline report complete")
    print(f"{'═'*W}\n")


# ── Plots ─────────────────────────────────────────────────────────────────────
def generate_plots(ts_data: dict, cluster_latency: dict,
                   qos: dict, label: str, output_dir: Path, prefix: str):
    """
    2-page QoS plot for baseline/OFF tests:
      Page 1: p95 over time | SLO% over time | Throughput over time | KPI table
      Page 2: Bar charts (p95, peak, SLO duration) per cluster
    """
    _fmt = lambda ax: (
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M")),
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5)),
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    )

    per_cl = qos["per_cluster"]

    # ── Page 1: timeseries ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Baseline QoS Analysis — {label}", fontsize=14, fontweight="bold")

    # [0,0] p95 over time
    ax = axes[0, 0]
    for cn in KNOWN_CLUSTERS:
        ts  = ts_data[cn]["timestamps"]
        p95 = ts_data[cn]["p95_ms"]
        if ts and any(v > 0 for v in p95):
            ax.plot(ts, p95, lw=2, color=COLORS[cn],
                    label=f"{cn} ({CLUSTER_REGIONS.get(cn,'')})")
    ax.axhline(SLO_MS, color="red", ls=":", lw=1.5, alpha=0.7, label="SLO 1000ms")
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("p95 Latency per Cluster Over Time")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    _fmt(ax)

    # [0,1] SLO% over time
    ax = axes[0, 1]
    for cn in KNOWN_CLUSTERS:
        ts  = ts_data[cn]["timestamps"]
        slo = ts_data[cn]["slo_pct"]
        if ts and any(v > 0 for v in slo):
            ax.plot(ts, slo, lw=2, color=COLORS[cn],
                    label=f"{cn} ({CLUSTER_REGIONS.get(cn,'')})")
    ax.axhline(5, color="orange", ls="--", lw=1.3, alpha=0.7, label="5% threshold")
    ax.set_ylabel("SLO Violation Rate (%)")
    ax.set_title("SLO Violations (>1000ms) per Cluster")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    _fmt(ax)

    # [1,0] Throughput over time
    ax = axes[1, 0]
    for cn in KNOWN_CLUSTERS:
        ts  = ts_data[cn]["timestamps"]
        cnt = ts_data[cn]["count"]
        if ts and any(v > 0 for v in cnt):
            ax.plot(ts, cnt, lw=2, color=COLORS[cn],
                    label=f"{cn} ({CLUSTER_REGIONS.get(cn,'')})")
    ax.set_ylabel("Requests per 10s window")
    ax.set_title("Throughput per Cluster")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    _fmt(ax)

    # [1,1] KPI table
    ax = axes[1, 1]
    ax.axis("off")

    flat = compute_global_p95_flat(cluster_latency) if cluster_latency else None
    global_slo_d = qos["global_slo_duration_s"]

    # Aggregate metrics
    all_means = [per_cl[cn]["mean_p95_ms"]   for cn in KNOWN_CLUSTERS if per_cl.get(cn) and per_cl[cn].get("mean_p95_ms")]
    all_peaks = [per_cl[cn]["peak_p95_ms"]   for cn in KNOWN_CLUSTERS if per_cl.get(cn) and per_cl[cn].get("peak_p95_ms")]
    all_recs  = [per_cl[cn]["recovery_s"]    for cn in KNOWN_CLUSTERS if per_cl.get(cn) and per_cl[cn].get("recovery_s")]
    worst_peak = max(all_peaks) if all_peaks else None
    max_rec    = max(all_recs)  if all_recs  else None

    total_req  = sum(cl["requests"] for cl in cluster_latency.values()) if cluster_latency else 0
    total_fail = sum(cl["failures"] for cl in cluster_latency.values()) if cluster_latency else 0
    fail_pct   = total_fail / total_req * 100 if total_req > 0 else 0

    def _s(v): return "✅" if v else "—"

    kpis = [
        ("Metric",                         "Value",                               "✓"),
        ("p95 flat (k6/Romano-style)",
         f"{flat:.0f} ms" if flat else "N/A",
         "✅" if flat and flat < SLO_MS else ("⚠️" if flat else "—")),
        ("Global SLO violation duration",
         f"{global_slo_d:.0f} s  ({global_slo_d/60:.1f} min)",
         "✅" if global_slo_d == 0 else ("⚠️" if global_slo_d < 120 else "❌")),
        ("Peak p95 (worst cluster)",
         f"{worst_peak:.0f} ms" if worst_peak else "N/A",
         "✅" if worst_peak and worst_peak < SLO_MS else ("⚠️" if worst_peak and worst_peak < 2000 else "❌" if worst_peak else "—")),
        ("Recovery time (worst cluster)",
         f"{max_rec:.0f} s" if max_rec else "No spike detected",
         "✅" if max_rec and max_rec < 120 else ("⚠️" if max_rec else "—")),
        ("Total requests",
         f"{total_req:,}",
         "✅" if total_req > 0 else "—"),
        ("Failure rate",
         f"{fail_pct:.2f}%",
         "✅" if fail_pct < 1 else ("⚠️" if fail_pct < 5 else "❌")),
        ("Scaling events",
         "0  (DMOS OFF — fixed replicas)",
         "—"),
    ]

    table = ax.table(cellText=kpis, cellLoc="center", loc="center",
                     colWidths=[0.46, 0.38, 0.10])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for j in range(3):
        c = table[0, j]
        c.set_facecolor("#c0392b")
        c.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(kpis)):
        for j in range(3):
            table[i, j].set_facecolor("#fdf2f8" if i % 2 == 0 else "white")
    ax.set_title("KPI Summary — Baseline (DMOS OFF)", fontsize=12, fontweight="bold", pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = output_dir / f"{prefix}_baseline_qos.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {p1.name}")

    # ── Page 2: bar charts ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Baseline Metrics per Cluster — {label}", fontsize=13, fontweight="bold")

    clusters = KNOWN_CLUSTERS
    x = np.arange(len(clusters))
    xlabels = [f"{cn}\n({CLUSTER_REGIONS.get(cn,'')})" for cn in clusters]

    bar_metrics = [
        ("p95_ms",         "p95 Latency (ms)",          "Aggregate p95 per Cluster",         cluster_latency, True),
        ("peak_p95_ms",    "Peak p95 Latency (ms)",     "Spike Peak p95 per Cluster",         None,            False),
        ("slo_duration_s", "SLO Violation Duration (s)", "SLO Violation Duration per Cluster", None,            False),
    ]

    for ax, (key, ylabel, title, lat_src, from_lat) in zip(axes, bar_metrics):
        if from_lat and lat_src:
            vals = [lat_src.get(cn, {}).get(key, 0) or 0 for cn in clusters]
        else:
            vals = [per_cl.get(cn, {}).get(key, 0) or 0 for cn in clusters]

        bars = ax.bar(x, vals, color=[COLORS[cn] for cn in clusters], alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        if key == "p95_ms":
            ax.axhline(SLO_MS, color="red", ls=":", lw=1.5, alpha=0.7, label="SLO 1000ms")
            ax.legend(fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    p2 = output_dir / f"{prefix}_baseline_bars.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 {p2.name}")

    return p1, p2


# ── JSON Export ───────────────────────────────────────────────────────────────
def export_json(ts_data: dict, cluster_latency: dict,
                qos: dict, label: str, output_file: Path):
    per_cl = qos["per_cluster"]
    out = {
        "label":    label,
        "mode":     "baseline_off",
        "slo_threshold_ms": SLO_MS,
        "global": {
            "p95_flat_ms":       compute_global_p95_flat(cluster_latency) if cluster_latency else None,
            "slo_duration_s":    qos["global_slo_duration_s"],
            "total_requests":    sum(cl["requests"] for cl in cluster_latency.values()) if cluster_latency else 0,
            "total_failures":    sum(cl["failures"] for cl in cluster_latency.values()) if cluster_latency else 0,
        },
        "per_cluster": {},
    }
    for cn in KNOWN_CLUSTERS:
        cl  = cluster_latency.get(cn, {}) if cluster_latency else {}
        m   = per_cl.get(cn, {})
        out["per_cluster"][cn] = {
            "region":            CLUSTER_REGIONS.get(cn, ""),
            "requests":          cl.get("requests"),
            "fail_pct":          cl.get("fail_pct"),
            "avg_ms":            cl.get("avg_ms"),
            "p50_ms":            cl.get("p50_ms"),
            "p90_ms":            cl.get("p90_ms"),
            "p95_ms":            cl.get("p95_ms"),
            "p99_ms":            cl.get("p99_ms"),
            "slo_pct":           cl.get("slo_pct"),
            "mean_p95_ms":       m.get("mean_p95_ms"),
            "peak_p95_ms":       m.get("peak_p95_ms"),
            "baseline_p95_ms":   m.get("baseline_p95_ms"),
            "slo_duration_s":    m.get("slo_duration_s"),
            "recovery_s":        m.get("recovery_s"),
            "throughput_cv_pct": m.get("throughput_cv"),
        }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  📄 {output_file.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Analyze DMOS OFF / baseline test from Locust CSVs"
    )
    parser.add_argument("timeseries_csv",    help="Locust per-cluster timeseries CSV")
    parser.add_argument("cluster_latency_csv", nargs="?", default=None,
                        help="Locust cluster_latency CSV (optional, auto-detected if omitted)")
    parser.add_argument("--label",    default="Baseline (DMOS OFF)",
                        help="Human-readable label for plots/report")
    parser.add_argument("--scenario", default=None,
                        help="Scenario slug for output dir (e.g. flash_crowd_off)")
    args = parser.parse_args()

    ts_path  = Path(args.timeseries_csv)
    prefix   = ts_path.stem.replace("_timeseries", "").replace("timeseries_", "")

    # Auto-detect cluster_latency CSV
    if args.cluster_latency_csv:
        lat_path = Path(args.cluster_latency_csv)
    else:
        # Replace "timeseries" with "cluster_latency" in filename
        lat_name = ts_path.name.replace("_timeseries_", "_cluster_latency_")
        lat_path = ts_path.parent / lat_name
        if not lat_path.exists():
            lat_path = None
            print("  ⚠  cluster_latency CSV not found — skipping aggregate stats")

    # Output dir
    scenario_slug = args.scenario or (
        ts_path.stem.split("_timeseries")[0].split("timeseries_")[-1]
    )
    output_dir = Path("results") / f"{scenario_slug}_off"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*W}")
    print(f"  DMOS BASELINE ANALYZER")
    print(f"{'='*W}")
    print(f"  Timeseries:  {ts_path.name}")
    print(f"  Latency CSV: {lat_path.name if lat_path else 'N/A'}")
    print(f"  Output:      {output_dir}/\n")

    # Load
    print("  Loading data...")
    ts_data = load_timeseries(str(ts_path))
    cluster_latency = load_cluster_latency(str(lat_path)) if lat_path else {}

    # Compute
    print("  Computing QoS metrics...")
    qos = compute_qos_metrics(ts_data)

    # Report
    print_report(ts_data, cluster_latency, qos, args.label, str(ts_path), str(lat_path or ""))

    # Plots
    print("  Generating plots...")
    generate_plots(ts_data, cluster_latency, qos, args.label, output_dir, prefix)

    # JSON
    json_out = output_dir / f"{prefix}_baseline_report.json"
    export_json(ts_data, cluster_latency, qos, args.label, json_out)

    print(f"\n{'='*W}")
    print(f"✅ Done! Output: {output_dir}/")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
