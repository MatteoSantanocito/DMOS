# Metrics and outputs — what a run produces and how results are computed

This document describes every file a flash-crowd run writes, every metric in
those files, how the analysis scripts turn them into the numbers, figures and
tables of the papers, and the definitions behind each reported metric
(phases, p95, error rate, SLO violation, goodput, carbon).

| Related | |
|---|---|
| How DMOS computes its decisions (and the `dmos_*` metrics it exports) | [dmos-architecture.md](dmos-architecture.md) |
| Provenance files and checks (`config_id`, `manifest.json`) | [RUN-PROVENANCE.md](RUN-PROVENANCE.md) |
| Running the experiments | [../experiments/RUNBOOK.md](../experiments/RUNBOOK.md) |

## Contents

1. [The workload and its phases](#1-the-workload-and-its-phases)
2. [The run directory](#2-the-run-directory)
3. [Client-side metrics (k6)](#3-client-side-metrics-k6)
4. [Scheduler and cluster timeline (`dmos_live.jsonl`)](#4-scheduler-and-cluster-timeline-dmos_livejsonl)
5. [Metric definitions used in the papers](#5-metric-definitions-used-in-the-papers)
6. [Analysis pipeline](#6-analysis-pipeline)
7. [Figures and tables](#7-figures-and-tables)
8. [Caveats](#8-caveats)

---

## 1. The workload and its phases

`experiments/k6_flash_crowd.js` drives a `ramping-arrival-rate` scenario
(open model: the request rate is imposed, independent of response time)
against the Cilium Ingress of every cluster. Phase boundaries are
`DMOS_PHASE_TIMES` (default `180,300,600,720,900` s):

| Phase | Time (s) | Offered rate | Ingress weights |
|---|---|---|---|
| `baseline` | 0–180 | `DMOS_BASELINE_RPS` (50) | baseline weights |
| `spike` | 180–300 | linear ramp 50 → `DMOS_PEAK_RPS` | spike weights (regime) |
| `peak` | 300–600 | `DMOS_PEAK_RPS` (350 in the harness) | spike weights (regime) |
| `decay` | 600–720 | linear ramp back to 50 | baseline weights |
| `cooldown` | 720–900 | 50 | baseline weights |

- **Ingress weights.** Each request picks a cluster at random with these
  weights. Baseline weights are by region: DE 0.30, FR 0.20, PL 0.15,
  NO 0.15, US-VA 0.10, SG 0.10 (override `DMOS_INGRESS_W1..6`). Spike weights
  come from `DMOS_SPIKE_W1..6` (normalised), set by the regime: `uni`
  1/1/1/1/1/1, `bal` 40/15/12/12/11/10, `s50` 50/10×5, `s65` 65/7×5,
  `s75` 75/5×5, `con` 85/3×5 (hottest cluster ms-cluster-01; its share is the
  "skew" of the papers). Without `DMOS_SPIKE_W*`, `DMOS_SPIKE_REGION` gets
  `DMOS_SPIKE_WEIGHT` (0.60) and the others share the rest.
- **Endpoint mix.** `/` 40 %, two product pages 15 % + 10 %, `GET /cart`
  10 %, `POST /cart` 15 %, `POST /setCurrency` 5 %, `POST /cart/checkout` 5 %.
- **Requests.** Timeout `DMOS_REQ_TIMEOUT_S` = 14 s; per-VU random seed from
  `DMOS_RNG_SEED` (20260521) for reproducible choices; VU pool
  `max(200, 2·peak)` pre-allocated, up to `peak × DMOS_VU_FACTOR` (20).
- **Tags** on every request: `ingress` (e.g. `ms-cluster-01-DE`), `cluster`,
  `endpoint`, `phase`, `method`, plus the global `run_label`.

The phase of a request is decided by k6's own clock
(`exec.instance.currentTestRunDuration`), so phases and load stay in sync.

---

## 2. The run directory

A run lives in `experiments/results/<YYYYMMDD>_<HHMMSS>_<label>/`, with
`<label> = dmos-<arm>__<regime>` (e.g. `dmos-l1l2__con`, `dmos-nca__con-sg`,
`dmos-off__s50-p500`). Arm and regime names are defined once in
`experiments/run_naming.py` (`python experiments/run_naming.py list`): the
regime is `<demand>[-<hot cluster>][-p<peak>]`.
Runs that fail a pre-run gate go to `experiments/results/_aborted/` instead.

| File | Written by | When | Content |
|---|---|---|---|
| `manifest.json` | `write_manifest.py start/finish` | start, end | run identity, k6 configuration, git state, host, DMOS `config_id`, testbed checks, outcome ([RUN-PROVENANCE.md](RUN-PROVENANCE.md)) |
| `dmos_config.json` | DMOS (`--config-dump`) or `GET :8081/config` | DMOS start | effective configuration and `config_id`, carbon source per cluster (`carbon.sources`) |
| `dmos_config_expected.json` | `dmos_main --dump-config-only` | before the reset | configuration the arm should produce (pre-check) |
| `dmos.log` | DMOS stdout/stderr | whole run | every cycle: traffic, scores and their components, allocations, PD, scaling, co-scaling |
| `env_state_pre.json`, `env_state_post.json` | `capture_env_state.py` | before / after | Service annotations, Deployments (incl. `loadgenerator` replicas), HPAs, Cilium config, netem qdiscs, RTTs |
| `config_snapshot/` | `write_manifest.py start` | start | copies of `weights.yaml`, `services.yaml`, `carbon.yaml` and `local.sanitized.yaml` (no IPs, keys or users) |
| `git_diff.patch`, `untracked_snapshot/` | `write_manifest.py start` | start | uncommitted changes of the code that ran |
| `flash_crowd_summary.json` | k6 `handleSummary` | end of k6 | per-phase and per-cluster aggregates, validity (§3.1) |
| `k6_stdout.txt` | k6 | whole run | console output and the text summary tables |
| `k6_events.json` | k6 `--out json=` (`DMOS_RAW_OUT=1`, default) | whole run | one JSON line per metric sample: the raw data of every request (§3.2); by far the largest file of a run |
| `dmos_live.jsonl` | `collect_dmos_live.py` | every 5 s | scheduler and cluster timeline (§4) |
| `collector_stdout.txt` | collector | whole run | console output of the collector |
| `slo_compliance.json` | `paper_slo_compliance.py` | analysis | cache of SLO counts (§5.4) |
| `cluster_phase_stats.json` | `paper_lib.compute_cluster_phase_stats` | analysis | cache of per-cluster × per-phase latency stats |

The two cache files are derived; deleting them forces a rescan of
`k6_events.json`.

---

## 3. Client-side metrics (k6)

### 3.1 `flash_crowd_summary.json`

```
run_label, timestamp_iso
validity:  run_valid, dropped_iterations, dropped_pct
config:    baseline_rps, peak_rps, spike_region, spike_weight, phase_times_s,
           num_clusters, baseline_weights[], spike_weights[]
clusters[]: name, region, ingress_url, weight_baseline, weight_spike
results:
  per_phase.<phase>:   requests, rps, p50, p95, p99, errorRate
  per_cluster.<name>:  requests, p50, p95, p99, errorRate        (whole run)
  global:              total_requests, avg_rps, p50_ms, p95_ms, p99_ms, avg_ms,
                       http_failure_rate
```

| Field | Definition |
|---|---|
| `per_phase.*.requests` | requests whose `phase` tag is that phase |
| `per_phase.*.rps` | `requests / phase duration` (achieved, not offered) |
| `per_phase.*.p50/p95/p99` | percentiles (ms) of `http_req_duration` of the phase (k6 Trend) |
| `per_phase.*.errorRate` | fraction of requests with HTTP status ≥ 500 **or** status 0 (connection error or 14 s timeout); 4xx are not errors |
| `per_cluster.*` | same, by entry cluster (`cluster` tag), over the whole run |
| `global.http_failure_rate` | k6 `http_req_failed`: status outside 200–399 (4xx included) |
| `global.avg_rps` | total requests / 900 s |
| `validity.dropped_pct` | `dropped_iterations / (dropped + executed)`; the run is **valid** if < 0.5 % (k6 could deliver the offered rate) |

Two error notions coexist: `errorRate` (5xx + status 0) and
`http_failure_rate` / the k6 `expected_response` tag (anything outside
200–399). The SLO metric of §5.4 uses the latter.

### 3.2 `k6_events.json`

k6's JSON output: one object per line, either a metric definition
(`"type":"Metric"`) or a sample (`"type":"Point"`) with
`data.time`, `data.value` and `data.tags`. The analysis reads the
`http_req_duration` points (value in ms) and their tags `phase`, `cluster`
and `expected_response`. Custom metrics also appear:
`phase_<phase>_latency|requests|errors` and
`cluster_<name>_latency|errors|requests` (dashes replaced by `_`).

---

## 4. Scheduler and cluster timeline (`dmos_live.jsonl`)

`experiments/collect_dmos_live.py` runs on the operator host for the length
of the test plus 90 s, one snapshot every 5 s (`--interval 5`; slower if the
scrape takes longer). Each line is one snapshot:

```
timestamp, elapsed_s, dmos_enabled, dmos_config_id
services.<svc>:
  actual_traffic_rps                     dmos_actual_traffic
  predicted_traffic_total_rps            Σ clusters of predicted_traffic
  total_current_replicas, total_target_replicas
  scheduling_avg_ms, scheduling_invocations   from dmos_scheduling_duration_seconds
  clusters.<cluster>:
    current_replicas, target_replicas, score, predicted_traffic   (DMOS gauges)
    scale_up_events, scale_down_events                            (DMOS counter, cumulative)
    cpu_cores, memory_mb, ready_pods, restarts                    (cluster Prometheus)
clusters.<cluster>:
  hubble_http_rps        all L7 requests served in the namespace (server-side)
  frontend_net_bytes_s   frontend requests from the ingress, server-side (req/s; historical name)
  traffic_pct            share of the previous field across clusters (%)
  node_cpu_pct           1 − avg idle CPU of the cluster's nodes (%)
  l7_edge_p95_ms         p95 per source→destination workload ("ingress->frontend", "frontend->cartservice", …)
  ping_rtt_ms            mean RTT per peer IP (ping_exporter)
```

Notes:

- `<svc>` covers frontend, cartservice, productcatalogservice,
  checkoutservice and recommendationservice. **currencyservice, adservice and
  shippingservice are not collected**, although DMOS co-scales them.
- DMOS fields are 0 when DMOS is not running (`off` arm): use `ready_pods`,
  read from Kubernetes, for replica counts valid in every arm.
- `scale_up_events`/`scale_down_events` count only the frontend actions
  `scale_up`/`scale_down`; co-scaling actions (`scale_*_colocation`) are not
  in the snapshot.
- `traffic_pct` is the share of frontend requests **served** by each
  cluster, not the demand entering it (that is DMOS's λⁱⁿ, visible in
  `dmos.log`).
- `elapsed_s` counts from the collector start, which precedes the k6 start by
  a few seconds (see §8).

`paper_lib._load_dmos_live_jsonl` flattens a run into a long table with one
row per (snapshot, cluster): `replicas` (sum of `ready_pods` over the five
services), `cpu_rate` (`node_cpu_pct`), `hubble_reqs_rate`, `memory_bytes`,
`phase`. `paper_lib.load_per_service_per_cluster` keeps the per-service
detail.

---

## 5. Metric definitions used in the papers

### 5.1 Phase of interest

Peak-phase metrics (the papers' tables) use the `peak` phase, 300–600 s, as
tagged by k6.

### 5.2 Latency percentiles

`results.per_phase.peak.p95` (and p50, p99) from
`flash_crowd_summary.json`: k6 percentiles of `http_req_duration` over all
requests of the phase, all clusters together. Per-cluster × per-phase
percentiles (`cluster_phase_stats.json`) are computed with
`numpy.percentile` on the raw events.

These are **client-side** latencies (end to end, as seen by k6, including
the netem delay of the responses). The latency DMOS uses for Φ_lat is
**server-side** (Hubble, measured at Envoy inside the cluster): the right
signal for a scheduling decision, but not comparable with the evaluation
numbers.

### 5.3 Error rate

`results.per_phase.peak.errorRate`: 5xx or status 0, §3.1.

### 5.4 SLO violation

`experiments/paper_slo_compliance.py` scans `k6_events.json`:

```
violation = (http_req_duration > T)  OR  (expected_response != "true")
SLO violation % = 100 · violations / requests        (per phase and global)
```

with T = 1000 and 2000 ms (both cached); the papers use **T = 2000 ms**
(`SLO_THRESHOLD_MS` in `paper_stat_compare.py`). A failed request counts as a
violation even when it is fast, so an arm that sheds load quickly is not
rewarded.

Not every output uses this definition:

| Definition | Used by |
|---|---|
| slow **or** failed (above) | `slo_compliance.json` → Table 1, Tab. IV of the NCA paper, skew-robustness figure; `paper_plot_detailed.py` |
| slow only (`duration > T`) | Fig. 3, 3b, 5, 7, 8 and `cluster_phase_stats.json` |

A request that times out (14 s) is slow under both. A fast failure (e.g. a
503 in a few ms) counts only under the first.

### 5.5 Goodput

```
goodput = peak.rps · (1 − peak.errorRate)          per run
```

(`paper_plot_skew_robustness.py`): successful requests per second during the
peak.

### 5.6 Aggregation over repetitions

Repetitions are the runs with the same label (and, with provenance, the same
`config_id`). For each metric the scripts compute the mean and a 95 %
bootstrap confidence interval of the mean (`paper_lib.bootstrap_ci`, 1000
resamples, seed 42). Pairwise comparisons use Welch's t-test and Cohen's d
(`paper_stat_compare.py`).

### 5.7 Run inclusion

`paper_lib.discover_runs` keeps a run only if:

1. its directory name matches `<date>_<time>_<label>`;
2. it has `manifest.json` and `flash_crowd_summary.json`;
3. `validity.dropped_pct` < 0.5 % (k6 delivered the offered load);
4. its provenance has no error (runs from before provenance tracking are kept).

It warns when one label mixes different `config_id`s: those runs are not
repetitions of the same algorithm.

### 5.8 Carbon (analysis side)

`paper_lib.compute_carbon_per_run` estimates emissions from the timeline:

```
P_cluster(t) = P_idle + (P_max − P_idle) · node_cpu_pct / 100
               HIGH 15–100 W, MEDIUM 12–75 W, LOW 10–50 W (tier from cpu_cores)
CO2 = Σ_t P · Δt / 3.6·10⁶ · baseline_gco2_kwh(cluster)
```

This is a relative comparison between arms: it uses each cluster's **static**
`baseline_gco2_kwh` from `local.yaml`, not the time-varying intensity that
DMOS saw during the run.

---

## 6. Analysis pipeline

```
python experiments/paper_generate_all.py [--tag NAME] [--scenarios …] [--skip-supplementary] [--no-report]
```

writes `paper_artifacts/<timestamp or tag>/{figures,tables}/` and runs, in
order (`--results-dir`, default `experiments/results`):

| Step | Script | Produces | Failure |
|---|---|---|---|
| 0 | `paper_slo_compliance.py` | `slo_compliance.json` in every run, summary table | best-effort |
| 1 | `paper_plot_trajectories.py` | Fig. 1–4 | fatal |
| 2 | `paper_plot_supplementary.py` | Fig. 5–8 | fatal |
| 2b | `paper_plot_carbon.py` | Fig. 9 | fatal |
| 2f | `paper_plot_detailed.py` (skip with `--skip-detailed`) | per-regime detail plots | best-effort |
| 2c | `paper_plot_metastable.py` | Fig. 11 (needs a `--naive-backends` run) | best-effort |
| 2d | `paper_plot_skew_robustness.py` | skew-robustness figure and data | best-effort |
| 2e | `paper_ablation_compare.py` | objective-ablation figure and tables | best-effort |
| 3 | `paper_stat_compare.py` | Tables 1–2 (LaTeX, Markdown, JSON) | fatal |
| 4 | `paper_report.py` | `paper_report.html`, `hypothesis_checks.json` | fatal |

The arm lists of the figure scripts default to the core of the NCA paper,
`dmos-off`, `dmos-l2`, `dmos-l1l2`. Other arms are passed explicitly
(`--arms dmos-off dmos-l2 dmos-nca`, `--scenarios …`, `--dmos-arm dmos-nca`
for the detailed and hot-share figures, `DMOS_H_ARM` for the hypotheses);
`paper_lib` has labels and colours for every score model and legacy arm.
Regime variants (`con-sg`, `s50-p500`) take the hot-cluster share of their
base regime on the skew axis (`paper_lib.regime_skew`).

---

## 7. Figures and tables

`paper_artifacts/` is generated output and is not in the repository.

### 7.1 Pipeline outputs (`paper_artifacts/<ts>/`)

| Output | Script | Content |
|---|---|---|
| `figures/fig1_replicas_heatmap` | `paper_plot_trajectories.py` | mean replicas (ready pods) per service × cluster during the peak, one panel per arm |
| `figures/fig2_p95_latency_overlay` | same | p95 latency over time (5 s bins from `k6_events.json`), arms overlaid |
| `figures/fig3_slo_violation_overlay` | same | % requests slower than 2 s over time (10 s bins) |
| `figures/fig3b_slo_per_phase_bars` | same | % requests slower than 2 s per phase and arm, CI95 |
| `figures/fig4_latency_cdf_per_phase` | same | latency CDF per phase |
| `figures/fig5_slo_heatmap` | `paper_plot_supplementary.py` | % requests slower than 2 s, cluster × phase, one panel per arm |
| `figures/fig6_throughput_achieved` | same | achieved request rate over time vs the offered one |
| `figures/fig7_boxplot_p95_per_cluster` | same | peak-phase p95 per entry cluster across repetitions |
| `figures/fig8_geo_fairness` | same | variance across clusters of the peak % requests slower than 2 s (lower = fairer) |
| `figures/fig9_carbon_emissions` | `paper_plot_carbon.py` | CO₂ per arm (§5.8) |
| `figures/fig11_cooldown_adaptation` | `paper_plot_metastable.py` | naive vs co-scaled backends (metastability study) |
| `figures/fig_skew_robustness` + `skew_robustness_data.json` | `paper_plot_skew_robustness.py` | peak p95, % SLO and goodput vs hottest-cluster share, mean ± CI95 |
| `figures/fig_ablation_tradeoff`, `ablation_*.tex`, `ablation.md`, `ablation_data.json` | `paper_ablation_compare.py` | weight profiles of E3 (`latency_first`, `capacity_first`, `carbon_first` vs `nca`, regime `bal` by default; `--arms` for others) |
| `tables/table1_metrics.tex` | `paper_stat_compare.py` | peak p50/p95/p99, SLO > 2 s, throughput: mean [CI95] per arm |
| `tables/table2_stat_tests.tex` | same | each arm vs the baseline (`--baseline`, default `dmos-off`): Δ%, Welch p, Cohen's d |
| `tables/tables.md`, `tables/tables_data.json` | same | the same numbers in Markdown and JSON |
| `paper_report.html` | `paper_report.py` | self-contained report (figures embedded) |
| `hypothesis_checks.json` | same | pass/fail of four fixed checks on the stress regime, for `dmos-off` and the full-DMOS arm (`DMOS_H_ARM`, default `dmos-l1l2`) on the regime `DMOS_H_REGIME` (default `con`): H1 peak p95 < 1 s for L1L2 and > 3 s for OFF; H2 SLO > 1 s violations < 2 % for L1L2 and > 10 % for OFF; H3 variance of the error rate across clusters < 0.0002; H4 CO₂ at least 10 % below OFF |
| `manifest.json` | `paper_generate_all.py` | input runs, git commit, options |
| `figures/detailed/<regime>/*.png`, `figures/detailed/slo_data.json` | `paper_plot_detailed.py` | per regime: key metrics, latency/error/throughput time series of a representative repetition, per-cluster p95 and errors, heatmaps, CDF; across regimes: % SLO violations |

Each figure is saved as PDF and PNG (`--formats`), 600 dpi.

### 7.2 Scripts outside the pipeline

| Script | Output | Notes |
|---|---|---|
| `paper_plot_hotshare.py` | `paper/figures/figpaper_hotshare_skew.{pdf,png}` | **Fig. 3 of the NCA paper**: % of frontend replicas (DMOS `target_replicas`) on the hot cluster (ms-cluster-01 by default) during 360–600 s vs skew; arms `dmos-off/l2/l1l2` (`--dmos-arm` replaces `dmos-l1l2`, `--hot sg` reads the `…-sg` regimes and the SG share); run from the repository root |
| `paper_plot_latency_budget.py` (+ `latency_budget_probe.py`) | `paper_artifacts/latency_budget/`, `latency_budget_data.json` | network vs processing share of latency, from `l7_edge_p95_ms` and `ping_rtt_ms` |
| `studya_heatmap.py`, `studyb_plot.py` | `paper_artifacts/studya`, `paper_artifacts/studyb` | in-cluster load studies (outside the papers) |

### 7.3 Where the NCA 2026 numbers come from

| Paper item | Source |
|---|---|
| Tab. IV p95, error, SLO > 2 s, goodput | `flash_crowd_summary.json` (peak) + `slo_compliance.json`, aggregated by `paper_plot_skew_robustness.py` / `paper_stat_compare.py` |
| Fig. 2 (robustness to skew) | `fig_skew_robustness` |
| Fig. 3 (hottest-cluster replica share) | `paper_plot_hotshare.py` |

See [publications/2026-nca-dmos.md](publications/2026-nca-dmos.md).

---

## 8. Caveats

- **Two clocks.** k6 phases come from k6's clock; the timeline's
  `elapsed_s` starts when the collector starts, a few seconds before k6
  (longer with k6 on the remote VM). `paper_lib.assign_phase` applies the k6
  boundaries to `elapsed_s`, so timeline phases are shifted by that offset.
- **DMOS gauges hold the last cycle.** `dmos_target_replicas` and
  `dmos_cluster_score` change only when a cycle runs (about every 30 s under
  load, never while the total traffic is between 10 and 30 req/s); the
  collector samples them every 5 s.
- **Stale comments in the analysis scripts.** Several docstrings
  (`paper_stat_compare.py`, `paper_report.py`, `paper_plot_supplementary.py`)
  mention a 60 s k6 timeout and older figure names; the workload uses 14 s and
  the file names are those of §7.
- **Raw data size.** `k6_events.json` dominates disk use; keep
  `slo_compliance.json` and `cluster_phase_stats.json` if the raw events are
  pruned, since several figures need either the events or these caches.
