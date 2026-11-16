# Metrics and outputs — what a run produces and how results are computed

This document describes every file a flash-crowd run writes, every metric in
those files, how the analysis scripts turn them into the numbers, figures and
tables of the paper, and the definitions behind each reported metric
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
5. [Metric definitions used in the paper](#5-metric-definitions-used-in-the-paper)
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
  "skew"). Without `DMOS_SPIKE_W*`, `DMOS_SPIKE_REGION` gets
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
`<label> = dmos-<arm>__<regime>` (e.g. `dmos-reach-pool-ps__con`, `dmos-hpa-none__con-tosg`,
`dmos-off__s50-p500`). Arm and regime names are defined once in
`experiments/run_naming.py` (`python experiments/run_naming.py list`): the
regime is `<demand>[-<hot cluster>][-to<cluster>][-p<peak>]` (`-to<cluster>`: the hot
spot moves to that cluster at 450 s, inside the peak).
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
| `hpa_baseline.json` | `hpa_baseline.py` | HPA arm: before / after the run | the HPAs created (per cluster: service, min, max, CPU target), whether each read its metric during the run, their removal |
| `carbon_t0` | `run_flash_crowd.sh` | load start | epoch (s) that anchors the carbon-trace replay: DMOS reads it (`DMOS_CARBON_T0_FILE`) and so does the analysis (§5.9) |
| `collector_stdout.txt` | collector | whole run | console output of the collector |
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
           num_clusters, baseline_weights[], spike_weights[],
           shift: {enabled, at_s, ramp_s, end_s, region, weights[]}   (enabled false = no move)
clusters[]: name, region, ingress_url, weight_baseline, weight_spike, weight_shift
results:
  per_phase.<phase>:   requests, rps, p50, p95, p99, errorRate
  per_shift_window.<pre|mix|post>: start_s, end_s, requests, rps, p50, p95, p99, errorRate
                       (only when the hot spot moves; null otherwise)
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
| `per_shift_window.*` | same, over the parts of the peak before the move (`pre`, from 300 s), during it (`mix`) and after it (`post`, to 600 s); requests of the peak carry the tag `shift` |
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
timestamp, elapsed_s, epoch_s, dmos_enabled, dmos_config_id
dmos_process            cpu_cores, cpu_seconds_total, rss_mb, threads, processes
                        of the local DMOS process (psutil; null in the off arm)
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
  app_cpu_cores          CPU of every container of the namespace (cores)
  app_memory_mb          working set of every container of the namespace
  app_ready_pods         ready pods of every service of the namespace
  node_cpu_busy_cores    non-idle CPU of the cluster's nodes (cores)
  node_cpu_cores         cores of the cluster's nodes
  services_all.<svc>     cpu_cores, ready_pods for EVERY service (pod name → Deployment)
```

Notes:

- `services.<svc>` covers frontend, cartservice, productcatalogservice,
  checkoutservice and recommendationservice only. The whole application
  (currencyservice, adservice, shippingservice, …) is in `app_*` and
  `services_all`, recorded from October 2026 on; older runs do not have them.
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
detail. The journal pipeline (`journal_data.timeseries`) resamples the
timeline on a 5 s grid, 0–900 s: per-region frontend ready and target
replicas, served req/s, ingress→frontend p95 (Hubble), predictive-scaler
signals, CPU and carbon intensity CI_i(t).

---

## 5. Metric definitions used in the paper

### 5.1 Phase of interest

The paper's main metrics are computed over the whole run (0–900 s,
`results.global` of `flash_crowd_summary.json`); per-phase values (e.g. the
p95 of the `peak` phase, 300–600 s, as tagged by k6, and the failed requests
per phase) are reported alongside.

### 5.2 Latency percentiles

`results.global.p95_ms` (and p50, p99) and `results.per_phase.<phase>.p95`
from `flash_crowd_summary.json`: k6 percentiles of `http_req_duration` over
all requests of the run or of the phase, all clusters together. Per-cluster × per-phase
percentiles (`cluster_phase_stats.json`) are computed with
`numpy.percentile` on the raw events.

These are **client-side** latencies (end to end, as seen by k6, including
the netem delay of the responses). The latency DMOS uses for Φ_lat is
**server-side** (Hubble, measured at Envoy inside the cluster): the right
signal for a scheduling decision, but not comparable with the evaluation
numbers.

### 5.3 Error rate

`results.per_phase.peak.errorRate`: 5xx or status 0, §3.1. The journal
pipeline reports `err_pct` = 100 · `global.http_failure_rate` (status outside
200–399, 4xx included) and the number of failed requests per phase.

### 5.4 SLO violation

The paper reports latency percentiles and failed requests, not an SLO
violation rate. `paper_lib.compute_cluster_phase_stats` scans `k6_events.json`
and caches in `cluster_phase_stats.json`, per cluster × phase, the share of
requests slower than a threshold T (default 2000 ms):

```
SLO violation % = 100 · (requests with http_req_duration > T) / requests
```

A request that times out (14 s) counts as slow; a fast failure (e.g. a 503 in
a few ms) does not.

### 5.5 Goodput

```
goodput = peak.rps · (1 − peak.errorRate)          per run
```

successful requests per second during the peak.

### 5.6 Aggregation over repetitions

Repetitions are the runs with the same label (and, with provenance, the same
`config_id`). `journal_figures.py` reports for each arm × regime the mean and
a 95 % percentile bootstrap confidence interval of the mean
(`journal_data.boot_mean_ci`, 10 000 resamples). `journal_tests.py` runs the
tests fixed in the header of the campaign plan before the campaign started:
an exact permutation test on the difference of means (all relabellings, 252
for 5 vs 5), one-sided in the stated direction unless marked two-sided, Holm
correction within each family (F1 primary: DMOS vs HPA and vs the even split;
F2 sustainability: carbon-aware weights and scale to zero; F3 ablation:
demand signal of the placement score), α = 0.05; effects are the % change of
the means with a 95 % bootstrap CI, plus Cliff's δ.

### 5.7 Run inclusion

The journal pipeline uses only the runs with status `OK` in the campaign TSV
(`experiments/results/_campaign_<name>.tsv`, written by `run_campaign.sh`; a
run that fails `campaign_ctl.py check` is retried). `paper_lib.discover_runs`
(used by `paper_sustainability.py` on a whole results directory) keeps a run
only if:

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
DMOS saw during the run. §5.9 replaces it in the paper's analysis.

### 5.9 Sustainability and efficiency (`paper_sustainability.py`)

```
python experiments/paper_sustainability.py [--phase peak|run|...] [--arms "..."] [--regimes "..."] \
       [--trace-dir data/carbon_traces] [--output-dir paper/figures/sustainability]
```

**Carbon timeline.** For every snapshot the script recomputes CI_i(t) of each
cluster with DMOS's own function (`CarbonClient.trace_value_at`: hourly trace,
`window_start + speedup · elapsed`, zone aliases), with the replay settings of
the run's `dmos_config.json` (else `carbon.yaml`) and `elapsed` measured from
`carbon_t0`. Since DMOS is anchored to the same file, the analysis and DMOS
see the same timeline, and all arms see the same intensity at the same point
of the load. Without the file (older runs) the anchor is the first snapshot
(`anchor = first_snapshot`): the timeline of the load, not necessarily the one
DMOS saw (its anchor was its own first cycle). A cluster without a trace uses
its `baseline_gco2_kwh` (`ci_sources = …=baseline`).

| Metric | Definition | Needs a power model |
|---|---|---|
| `ci_served_g_kwh` | Σ served_i·Δt·CI_i / Σ served_i·Δt, served = frontend req/s of the cluster (`frontend_net_bytes_s`) | no |
| `ci_compute_g_kwh` | Σ cpu_i·Δt·CI_i / Σ cpu_i·Δt, cpu = `app_cpu_cores` | no |
| `core_s`, `pod_s` | Σ cpu_i·Δt, Σ pods_i·Δt (whole application) | no |
| `core_s_per_1k_ok` | `core_s` per 1000 successful k6 requests of the window | no |
| `co2_dyn_g`, `co2_dyn_mg_per_ok` | Σ cpu_i·Δt·W_i·CI_i / 3.6·10⁶, W_i = (P_max − P_idle)/cores of the tier (about 10 W per busy core) | yes |
| `co2_node_g`, `energy_node_wh` | linear node model of §5.8 on `node_cpu_busy_cores`, with CI_i(t) | yes |
| `dmos_cpu_cores_mean/max`, `dmos_rss_mb_max`, `dmos_cycle_ms` | control-plane cost: DMOS process (`dmos_process`) and mean cycle duration | no |

The model-free metrics carry the result; the power-model ones are estimates
(VMs, no power meter) for relative comparisons only. Older runs fall back to
the five services of `services.<svc>` (`accounting = partial`).

Outputs: `sustainability_runs.csv` (one row per run), `sustainability_summary.csv`
(mean and standard deviation per scenario) and `fig_latency_carbon_tradeoff`
(one panel per regime: CI of the served requests vs peak p95, one point per arm).
`journal_data` takes the carbon, efficiency and control-plane metrics of each
run from `paper_sustainability.run_metrics`.

---

## 6. Analysis pipeline

The paper's numbers come from the campaign TSV of `run_campaign.sh`
(status `OK` rows only) and three scripts that share the library
`experiments/journal_data.py` (runs, per-run metrics, 5 s time series; cache in
`--out/_cache`):

```
TSV=experiments/results/_campaign_2026-11-02-definitive.tsv
python experiments/journal_figures.py --tsv $TSV     # figures and tables (run first)
python experiments/journal_tests.py   --tsv $TSV     # pre-specified tests
python experiments/journal_control.py --tsv $TSV     # control-plane cost (needs runs.csv)
```

Common options: `--results <dir>` (default `experiments/results`) and
`--out <dir>` (default `paper/figures/journal`); `journal_figures.py` also
takes `--only fig02,fig05` (prefixes) and `--no-cache`. `--proxy` uses pilot
runs as stand-ins to check the pipeline (figures watermarked). The public
dataset of the campaign is built from the raw run folders by
`python experiments/build_dataset.py --results experiments/results --tsv $TSV --out <DATASET_DIR>`
(status-OK runs, the TSV, campaign and driver logs, plans; LAN addresses, home
paths and host names scrubbed; carbon traces not copied). The steps are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[../experiments/RUNBOOK.md](../experiments/RUNBOOK.md).

---

## 7. Figures and tables

`paper/` is generated output and is not in the repository. Each figure is
saved as PDF (vector) and PNG (600 dpi).

### 7.1 Pipeline outputs (`--out`, default `paper/figures/journal/`)

| Output | Script | Content |
|---|---|---|
| `fig01_testbed_workload` | `journal_figures.py` | testbed and workload: RTT matrix, carbon signal, offered load per entry region |
| `fig02_latency_resource_tradeoff` | same | latency against resources, every arm and regime (mean, 95 % bootstrap CI) |
| `fig03_effects_vs_hpa` | same | % change against `hpa-none` with bootstrap CIs, per regime |
| `fig04_replica_placement_<regime>` | same | where and when the frontend replicas run, one representative run per arm |
| `fig05_dynamics_<regime>` | same | capacity tracking and latency at the hot entry, mean over repetitions |
| `fig06_entry_p95` | same | p95 seen by the users of each entry region |
| `fig07_phase_latency_<regime>` | same | latency per load phase |
| `fig08_predictive_scaler_<regime>` | same | Level 2 at work: predicted demand and replica budget (`reach-pool-ps`) |
| `fig09_sustainability_<regime>` | same | where requests are served and their carbon cost |
| `fig10_ablation_<regime>` | same | ablation: locality term → local-demand gate → DMOS → min 0 → carbon-first |
| `fig11_cost_<regime>` | same | failed requests per phase, over- and under-provisioning |
| `fig12_repeatability` | same | the same arm over the days and hours of the campaign |
| `fig13_pool_routing` | same | pool-routing check: served share against ready-replica share per cluster |
| `fig14_shift_<regime>` | same | hot spot moving from DE to SG (`con-tosg`): latency before, during and after, SG replicas |
| `fig15_placement_model` | same | measured placement against the expected-RTT model, its optimal split and the even split |
| `fig16_carbon_bound` | same | carbon intensity served against the cleanest and dirtiest placement of the same replicas |
| `runs.csv`, `summary.csv` | same | per-run metrics; per arm × regime n, mean and 95 % bootstrap CI |
| `overhead.csv` | same | DMOS process CPU, memory and decision time per arm × regime |
| `table_main.tex`, `table_desc.tex`, `table_overhead.tex` | same | main results (p50/p95/p99, errors, frontend replica-minutes, CO₂ per request, CI served), descriptives (p99, failures per phase, pods started, replicas added in the decay, SG at 0, pool gap), control-plane overhead |
| `tests.md`, `tests.csv`, `table_tests.tex` | `journal_tests.py` | the pre-specified tests (§5.6): exact permutation test, Holm per family F1/F2/F3, effects with bootstrap CI and Cliff's δ |
| `table_control.tex` | `journal_control.py` | controller CPU/memory, application CPU saved against HPA, pods started |
| `control_timing.csv` | same | one decision split into its steps, from the DMOS logs of the campaign |
| `fig17_control_scaling`, `control_bench.csv` | same | offline benchmark of the decision logic (the campaign's code, unchanged) on synthetic federations of 6 to 1000 clusters |

### 7.2 Scripts outside the pipeline

| Script | Output | Notes |
|---|---|---|
| `karmada_baseline.py` | `<run>/karmada_baseline.json`, `<run>/karmada_trace.jsonl` | Karmada arms: adoption (component versions and flags, initial placement), verification after the load, release; trace every 15 s of the binding placement per cluster and of the FederatedHPA replicas ([KARMADA-BASELINE.md](KARMADA-BASELINE.md)) |
| `paper_sustainability.py` | `paper/figures/sustainability/` | carbon, efficiency and control-plane cost per run and per scenario (§5.9) |
| `sim_profiles.py` | console, `--csv` | offline Level-1 study, no testbed: DMOS's own score, budget and allocation code on a steady peak, per weight profile (`<model>[@nu<x>][@g<y>]`), regime and peak; replicas per cluster, CI of the served requests, extra RTT. Use it to check that a profile can change the placement before running it on the testbed (model and limits in the script docstring) |
| `check_dataplane.py` | console, `--csv` | per run: was the frontend served by the global pool (served share = replica share) or locally (= demand share)? Two signals (Hubble served share, CPU per replica); verdict `pool` / `local` / `indistinct` / `inconsistent` / `disagree`; exclusion criterion of the journal campaign |
| `scaler_replay.py` | console | offline replay of the reactive and predictive Level 2 ([PREDICTIVE-SCALER.md](PREDICTIVE-SCALER.md)) |

---

## 8. Caveats

- **Two clocks.** k6 phases come from k6's clock; the timeline's
  `elapsed_s` starts when the collector starts, a few seconds before k6
  (longer with k6 on the remote VM). `paper_lib.assign_phase` applies the k6
  boundaries to `elapsed_s`, so timeline phases are shifted by that offset.
- **DMOS gauges hold the last cycle.** `dmos_target_replicas` and
  `dmos_cluster_score` change only when a decision runs (with the reactive
  Level 2 about every 30 s under load, never while the total traffic is
  between 10 and 30 req/s; with the predictive scaler when a decision is due
  on its 10 s tick); the collector samples them every 5 s.
- **Raw data size.** `k6_events.json` dominates disk use (`DMOS_RAW_OUT=0`
  keeps it on the k6 VM). The journal pipeline reads `flash_crowd_summary.json`,
  `dmos_live.jsonl`, `dmos_config.json`, `manifest.json` and `dmos.log`; keep
  `cluster_phase_stats.json` if the raw events are pruned.
