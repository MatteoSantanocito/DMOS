# DMOS — Architecture and design

This is the reference description of **what the code does**: every value
below is taken from the source and the configuration files, with the place
where it is defined. The paper-level view (equations, claims, results) is in
[`publications/2026-nca-dmos.md`](publications/2026-nca-dmos.md), which maps
each equation of the NCA 2026 paper to this document and to the code.

| Topic | Where else |
|---|---|
| Every configuration key and environment variable | [configuration-reference.md](configuration-reference.md) |
| What a run produces, metric definitions, figures | [METRICS-AND-OUTPUTS.md](METRICS-AND-OUTPUTS.md) |
| Run traceability (`config_id`, manifests, checks) | [RUN-PROVENANCE.md](RUN-PROVENANCE.md) |
| Running experiments | [../experiments/RUNBOOK.md](../experiments/RUNBOOK.md) |
| Cluster setup | [setup-guide.md](setup-guide.md) |

## Contents

1. [Overview](#1-overview)
2. [Testbed](#2-testbed)
3. [Application and service model](#3-application-and-service-model)
4. [Data plane](#4-data-plane)
5. [Observed state: metrics and queries](#5-observed-state-metrics-and-queries)
6. [Control loop](#6-control-loop)
7. [Level 1 — placement](#7-level-1--placement)
8. [Level 2 — predictor and PD correction](#8-level-2--predictor-and-pd-correction)
9. [Backend co-scaling](#9-backend-co-scaling)
10. [Safeguards and timing constants](#10-safeguards-and-timing-constants)
11. [Interfaces: ports, endpoints, exported metrics](#11-interfaces-ports-endpoints-exported-metrics)
12. [Code map](#12-code-map)
13. [Known limitations and paper ↔ code differences](#13-known-limitations-and-paper--code-differences)

---

## 1. Overview

DMOS (Demand-gated Multi-Objective Orchestration System) is a
**centralized multi-cluster scheduler**. A single Python process
(`python -m src.dmos_main`) runs on an operator host outside the clusters and,
every 30 s, decides how many replicas of each Online Boutique service run on
each of K Kubernetes clusters. It reads each cluster's own Prometheus and
writes replica counts through each cluster's Kubernetes API (`scale` of the
Deployment). It never routes requests: the data plane (Cilium) does.

```
                  ┌──────────────────────── DMOS host ────────────────────────┐
                  │  python -m src.dmos_main                                   │
                  │   periodic check (30 s) ─► event queue ─► 3 workers        │
                  │   Level 1: score + proportional allocation (budget N)      │
                  │   Level 2: trend predictor + PD correction (±1)            │
                  │   co-scaling of the 7 backends from the frontend count     │
                  │   :9090 /metrics      :8081 /health /config /webhook ...   │
                  └──────┬─────────────────────────────────────┬──────────────┘
           PromQL (HTTP :32090)                       kube-API (scale)
                         │                                     │
      ┌──────────────────┼─────────── K clusters (K3s + Cilium ClusterMesh) ──┼─────┐
      │  ms-cluster-01 … ms-cluster-06                                              │
      │  kube-prometheus-stack (ns observability) ◄── Hubble L7, cAdvisor, KSM,     │
      │                                               ping_exporter                 │
      │  Cilium Ingress :30080 ──► frontend ──gRPC──► 7 backends (+ redis-cart)     │
      │  tc netem on each node: emulated inter-region delay                         │
      └─────────────────────────────────────────────────────────────────────────────┘
                         ▲
                k6 load generator (dedicated VM): per-cluster ingress weights
```

DMOS controls only replica counts. Everything it decides is derived from:

- the **ingress demand** entering each cluster (Hubble L7, client-reported);
- the clusters' **resources** (CPU, memory), **latency** (Hubble p95),
  **network position** (RTT to peers via ping_exporter) and **carbon
  intensity** of the region (replayed traces);
- a static **service model** (`config/services.yaml`): capacity per replica,
  per-cluster replica caps, call fan-out.

The decision has three layers (the paper's L0/L1/L2):

| Layer | Question | Code |
|---|---|---|
| L0 eligibility | which clusters may receive replicas | `DMOSScheduler._compute_cluster_score` |
| L1 placement | how the replica budget N is split across clusters | `ScoreFunctions` + `WinnerDetermination.allocate` |
| L2 control | timing (predictor) and fine correction (PD, ±1) per cluster | `TrafficPredictor`, `PDController`, `DMOSOrchestrator.schedule_service` |

Backends are not scheduled on their own: they follow the frontend on each
cluster by **co-scaling** (§9).

---

## 2. Testbed

The evaluation fleet has six single-region K3s clusters on Proxmox VMs,
created with [unict-cclab/proxmox-k3s](https://github.com/unict-cclab/proxmox-k3s)
v0.3.0 (K3s, Cilium with kube-proxy replacement, Hubble, ClusterMesh,
kube-prometheus-stack). The topology lives in `config/local.yaml`
(gitignored; template `config/local.yaml.example`), from which
`scripts/render-runtime-files.sh` produces `.cache/test-env.json` for the
harness.

| Cluster | HW tier (worker) | `cpu_cores` / `memory_gb` | Region code | Emulated delay (netem, egress) | `baseline_gco2_kwh` | Frontend cap |
|---|---|---|---|---|---|---|
| ms-cluster-01 | HIGH | 8 / 12 | `DE` (Frankfurt) | 0 ms | 350 | 6 |
| ms-cluster-02 | MEDIUM | 6 / 8 | `FR` (Paris) | 30 ± 5 ms | 80 | 5 |
| ms-cluster-03 | LOW | 4 / 8 | `PL` (Warsaw) | 50 ± 10 ms | 650 | 3 |
| ms-cluster-04 | HIGH | 8 / 12 | `US-VA` (Ashburn) | 100 ± 15 ms | 380 | 6 |
| ms-cluster-05 | LOW | 4 / 8 | `SG` (Singapore) | 200 ± 25 ms | 480 | 3 |
| ms-cluster-06 | MEDIUM | 6 / 8 | `NO` (Oslo) | 40 ± 8 ms | 25 | 5 |

- `cpu_cores` and `memory_gb` are the values DMOS uses as the cluster's total
  resources (`ClusterMetrics.cpu_total_cores` / `memory_total_gb`, Φ_cap's
  hardware size, L0 utilization).
- The netem delay is applied by `scripts/setup-netem-and-ping.sh netem-only`
  as a root qdisc on the worker's interface
  (`tc qdisc replace dev <iface> root netem delay <D>ms <J>ms distribution normal`).
  It delays *all* traffic leaving the worker: the round trip between
  clusters *i* and *j* is roughly `delay_i + delay_j`, and the responses to
  the load generator pay `delay_i`. `experiments/capture_env_state.py` checks
  the qdisc before and after every run (see [RUN-PROVENANCE.md](RUN-PROVENANCE.md)).
- `baseline_gco2_kwh` is a static fallback; the carbon intensity actually used
  is described in §5.3.
- "Frontend cap" is `max_replicas_per_cluster` of the frontend in
  `config/services.yaml`, set per hardware tier and leaving room for the
  co-scaled backends on the same node.

DMOS accepts any K ≥ 2: clusters are read from `local.yaml`, and the
per-cluster caps in `services.yaml` fall back to `autoscaling.max_replicas`
for clusters they do not list.

---

## 3. Application and service model

The workload is Google's Online Boutique (namespace `online-boutique`). The
Kubernetes manifests of the application come from upstream; this repository
holds DMOS's **model** of each service (`config/services.yaml`) and the
patches applied to the testbed (`experiments/apply_resource_hardening.sh`,
`experiments/apply_pod_lifecycle.sh`).

| Service | Capacity / replica (req/s) | Fan-out (calls per frontend request) | CPU request | Max replicas per cluster | How it is sized |
|---|---:|---:|---|---|---|
| frontend | 60 | 1.0 | 100m | 6/5/3/6/3/5 | L1 + L2 |
| cartservice | 400 | 0.65 | 200m | 2 | co-scaling |
| productcatalogservice | 1200 | 2.25 | 150m | 3 | co-scaling |
| checkoutservice | 160 | 0.05 | 300m | 2 | co-scaling |
| recommendationservice | 40 | 0.40 | 200m | 6/4/2/6/2/4 | co-scaling |
| currencyservice | 250 (estimate) | 3.4 | 100m | 6/4/3/6/3/4 | co-scaling |
| adservice | 400 (estimate) | 0.43 | 200m | 3 | co-scaling |
| shippingservice | 500 (estimate) | 0.14 | 100m | 2 | co-scaling |

Per-cluster lists follow the order ms-cluster-01…06. `min_replicas` is 1 for
every service and cluster. Capacities were measured with k6 (frontend: HTTP,
`experiments/k6_capacity_test.js`; backends: gRPC from inside the cluster,
`experiments/k6_capacity_grpc.js`, see
[`../experiments/CAPACITY-GRPC.md`](../experiments/CAPACITY-GRPC.md)) and set to
80 % of the knee; `recommendationservice` uses an effective capacity measured
under the flash crowd (40 instead of 140 req/s). Fan-outs were measured with
Hubble as the ratio of backend to frontend request rates. `redis-cart`
(stateful), `emailservice` and `paymentservice` are not managed.

Only the frontend is "monitored": its replica counts come from L1/L2, and
the seven backends above are sized from them (§9). Which service is the
entry service, which services are co-scaled (and in which order) and how the
entering demand is measured come from `services.yaml → application`
([configuration-reference §4.1](configuration-reference.md#41-application--the-application-dmos-manages)),
so another application needs a different `services.yaml`, not different
code.

---

## 4. Data plane

| Element | Manifest / setup | Role for DMOS |
|---|---|---|
| Cilium (kube-proxy replacement, Hubble, ClusterMesh) | installed by proxmox-k3s; DMOS-specific Helm upgrade (Hubble metrics, Ingress controller NodePort) by `scripts/setup-dmos-on-clusters.ps1` with `config/cilium-hubble-values.yaml` | service load balancing, L7 metrics |
| Cilium Ingress, NodePort **30080** | `deployments/ingress-frontend-cilium.yaml` (`ingressClassName: cilium`) | entry point of the load per cluster; the k6 weights decide how much load enters each cluster |
| L7 visibility policy on the frontend | `deployments/cnp-frontend-l7.yaml` (`l7-visibility-frontend`) | `rules: http: [{}]` sends the frontend's HTTP traffic through Envoy, so Hubble counts every request (`hubble_http_requests_total`) |
| L7 visibility policy on the backends | `deployments/cnp-backends-l7.yaml` | same for the gRPC backends (used for fan-out measurements and diagnostics) |
| Global service: frontend | `deployments/global-service-frontend.yaml` | `service.cilium.io/global: "true"`, `service.cilium.io/shared: "true"`, `service.cilium.io/affinity: "none"` — the data plane of the NCA paper |
| Global services: backends | `deployments/global-service-backends.yaml` | cart, productcatalog, checkout, recommendation: global with `affinity: "local"` (currency, ad, shipping are plain Services) |

**Frontend policy.** Port 8080 of the frontend accepts traffic only from the
`ingress`, `cluster`, `host` and `remote-node` entities; anything else
(`world`, e.g. the frontend's own NodePort) is dropped at L3. `remote-node`
is what lets a frontend pod receive requests that entered another cluster.

**Cross-cluster serving (paper data plane).** The frontend is a Cilium
ClusterMesh global service with `affinity: none`: Cilium merges the ready
frontend pods of all clusters into one backend pool and balances each
connection uniformly per endpoint across that pool, wherever the request
entered (Cilium docs: a global service is shared by default; `none` is the
default affinity, `local` would prefer healthy local endpoints). A cluster's
serving share is therefore its share of the frontend replicas, which is the
lever DMOS acts on. EndpointSlice synchronisation
(`clustermesh-enable-endpoint-sync`) is not needed for this: it only exports
EndpointSlices for other consumers. Every run checks the frontend's affinity
before starting (§13, [RUN-PROVENANCE.md](RUN-PROVENANCE.md)); the
measurements of the earlier "v1" setup, and the opt-in overflow pilot
(`deployments/clustermesh-overflow/`), are in
[CILIUM-CLUSTERMESH-NOTES.md](CILIUM-CLUSTERMESH-NOTES.md).

**Why ingress demand is measured client-side.** If a request entering
cluster *i* were served by a frontend on cluster *j*, a server-side count
would attribute it to *j*, and more replicas on *j* would look like more
demand on *j* (a feedback loop). DMOS therefore counts the requests that the
ingress of each cluster *sends* to the frontend (`reporter="client"`,
`traffic_direction="egress"`), wherever the pod that serves them lives
(§5.1).

---

## 5. Observed state: metrics and queries

Each cluster has its own Prometheus (kube-prometheus-stack, namespace
`observability`, NodePort `prometheus_port`, default **32090**). DMOS keeps one
`PrometheusClient` per cluster (`prom_map`), so every query returns the value
of that cluster only. Hubble metrics are scraped every **15 s** (ServiceMonitor
`interval: "15s"` in `config/cilium-hubble-values.yaml`); all Hubble rates use
a `[1m]` window (about four samples).

### 5.1 Queries

| Quantity | Symbol | Method (`src/metrics/prometheus_client.py`) | PromQL (per cluster) | Used by |
|---|---|---|---|---|
| Ingress demand entering the cluster | λᵢⁱⁿ | `get_ingress_rate` | `sum(rate(hubble_http_requests_total{destination_workload="frontend", destination_namespace="online-boutique", reporter="client", traffic_direction="egress"}[1m]))`; fallback (only if 0): server-side `reporter="server", traffic_direction="ingress", source_workload=""`. Entry workload, namespace, an optional filter on the clients (`source_workload=~"<sources>"`) and the fallback are set in `services.yaml → application` (the values shown are the defaults) | budget N, L2, demand share dᵢ, trigger |
| Request rate of the service | λᵢ | `get_request_rate` | `sum(rate(hubble_http_requests_total{destination_workload="<svc>", destination_namespace="online-boutique"}[1m]))`; fallbacks in order: Istio, `http_requests_total`, `container_network_receive_bytes_total / 4000` | Φ_pressure (legacy path only) |
| Latency, 95th percentile | Lᵢ | `get_latency_p95` | `histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{destination_namespace="online-boutique"}[1m])) by (le)) * 1000` | Φ_lat |
| Free CPU (cores) | Cᵢ | `get_cpu_available` | `sum(kube_node_status_capacity{resource="cpu"}) − sum(rate(container_cpu_usage_seconds_total{image!=""}[2m]))` | L0, Φ_cap, bid capacity |
| Free memory (GB) | Mᵢ | `get_memory_available_gb` | `sum(kube_node_status_capacity{resource="memory"}) − sum(container_memory_working_set_bytes{image!=""})` | L0, Φ_cap, bid capacity |
| Mean RTT to the other clusters (ms) | RTTᵢ | `get_network_rtt_ms` | `avg(ping_rtt_mean_seconds{target=~"<peer IPs>"}) * 1000`; 5 ms if unavailable | Φ_net |

Notes:

- **Latency is cluster-wide.** The p95 is computed over *all* HTTP/gRPC flows
  that Hubble sees in the namespace on that cluster (no workload filter), not
  over the frontend only. If no value is available (or the quantile is
  `+Inf`), the cluster's `latency.baseline_ms` from `local.yaml` (2 ms) is
  used.
- **Free resources are summed over all nodes** reported by
  kube-state-metrics, while `cpu_cores`/`memory_gb` in `local.yaml` describe
  the worker (see §13).
- If the total ingress rate over all clusters is 0, the demand shares are set
  to 1/K.

### 5.2 Demand share

```
dᵢ = λᵢⁱⁿ / Σⱼ λⱼⁱⁿ          (1/K for every cluster when Σ = 0)
```

computed once per cycle in `DMOSScheduler.collect_scores` (Step 0) before the
per-cluster scores.

### 5.3 Carbon intensity

`src/metrics/carbon_client.py`, configured by `config/carbon.yaml`. The
active provider is **`trace`**: hourly life-cycle carbon intensity from
Electricity Maps (2025), stored in `data/carbon_traces/<ZONE>_2025_hourly.csv`
(downloaded once by each user with their own Electricity Maps access, e.g.
with `scripts/fetch_carbon_traces.py`; the files are not distributed with the
repository, see [`../data/carbon_traces/README.md`](../data/carbon_traces/README.md))
and replayed as

```
trace_time = window_start + speedup · (now − t0)          (wraps at the end of the trace)
CI(t)      = linear interpolation of the two adjacent hourly samples
```

with `window_start = 2025-06-02T00:00:00` and `speedup = 96` (one 15-min run ≈
one day of trace). `t0` is fixed at the first carbon query of the process and
logged ("Trace replay anchored: …"). The trace provider is never cached.

The CSV is looked up by the Electricity Maps **zone** of the cluster:
`trace.zone_aliases[region_code]`, or the region code itself when there is
no alias. The aliases in `carbon.yaml` map `NO` → `NO-NO1` (Oslo bidding
zone) and `US-VA` → `US-MIDA-PJM` (PJM Interconnection), so the reference
fleet needs the files `DE`, `FR`, `PL`, `US-MIDA-PJM`, `SG` and `NO-NO1`:

```bash
python scripts/fetch_carbon_traces.py --zones DE FR PL US-MIDA-PJM SG NO-NO1 --year 2025   # key in ELECTRICITYMAPS_TOKEN
```

(the key needs historical data: an Electricity Maps API trial does not
include it, the academic access does). A region without a file falls back
to the **`realistic`** provider: a built-in hourly profile
per region (`REALISTIC_PROFILES`), multiplied by a seasonal factor of the
current month and by a ±5 % deterministic jitter derived from the current
minute, so that value depends on the wall-clock time of the run, not on the
trace replay. If a region has neither a trace nor a profile, the mock values
of `carbon.yaml` are used, and if those are missing the scheduler uses
300 gCO₂/kWh.

Every run records the source of each cluster in `dmos_config.json →
carbon.sources` (`trace:<file>`, `realistic`, `mock`, `default-300`; part of
the `config_id`), the start-up summary prints `carbon trace k/K`, and the
manifest raises the warning `carbon_trace_fallback` when a cluster is not
on a trace. In the NCA 2026 campaign the trace was used for DE, FR and PL
only: NO (no alias at the time), US-VA and SG ran on the synthetic profiles
(DMOS logs: "Trace NO/SG/US-VA: assente → fallback 'realistic'").

---

## 6. Control loop

`DMOSOrchestrator.run()` starts three kinds of threads:

| Thread | What it does |
|---|---|
| `periodic_check_thread` | every `polling_interval` = **30 s** (hard-coded): resets the per-cycle churn budgets, reads λᵢⁱⁿ of the frontend on all clusters, runs the per-cluster predictors, and enqueues a scheduling event when the traffic is above or below the thresholds |
| 3 × `worker_thread` | take events from a priority queue and run `schedule_service` (one at a time per service, see debounce) |
| `webhook_server` (Flask, :8081) | `/health`, `/config`, `/webhook/alert` (Alertmanager), `/api/schedule` (manual trigger) |

### 6.1 When a scheduling cycle is triggered

In `periodic_check_thread`, for the frontend:

```
λ        = Σᵢ λᵢⁱⁿ                                   (observed total)
λ̂        = max(λ, maxᵢ predictorᵢ(λᵢⁱⁿ))             (largest per-cluster prediction)
λ_eff    = λ           if 0 < λ < 2 req/s            (traffic floor: predictor bypassed)
         = max(λ, λ̂)   otherwise
λ_eff > 30 req/s  → event "scale_up"  (priority 0 if only the prediction is above 30, else 1)
λ_eff < 10 req/s  → event "scale_down" (priority 2)
10 ≤ λ_eff ≤ 30   → no event
```

Every event runs the **same** full cycle (`schedule_service` ignores the
action). Events closer than `debounce_seconds` = **30 s** for the same service
are dropped (`should_process`, timestamp set before running so that two
workers cannot run the same service concurrently). Under load the cycle
therefore runs about every 30 s. Between 10 and 30 req/s of total traffic
no event is raised and the placement stays as it is: a hysteresis band
between "scale down" and "scale up". The experiments never sit in it — the
k6 baseline alone is 50 req/s, the peak 200–500 req/s — so in every
measured phase DMOS runs a full cycle every 30 s.

### 6.2 One cycle (`DMOSOrchestrator.schedule_service`)

1. **Traffic.** λᵢⁱⁿ of every cluster (in parallel). If the total is 0, a
   fallback of `capacity × min_replicas × 0.5` = 30 req/s spread uniformly
   is used. Gauge `dmos_actual_traffic`.
2. **Start-up grace.** For the first `startup_grace_seconds` = **120 s** after
   start the cycle stops here: traffic is observed (and the predictors fill
   their history) but nothing is scaled.
3. **Budget N** (paper Eq. 7). For each cluster
   `needᵢ = clip(⌈λᵢⁱⁿ / cᵢ⌉, c_minᵢ, c_maxᵢ)` with the frontend capacity
   cᵢ = 60 req/s, `c_min` = 1 and the per-cluster cap; `N = Σᵢ needᵢ`.
   The `needᵢ` are also the **demand floors** (passed to L1 only when the
   demand floor is enabled).
4. **Placement.** L1 (§7) splits N across the clusters, or, with
   `--no-scheduling` (arm `l2`), N is split uniformly: ⌊N/K⌋ each, the
   remainder to the first clusters in name order. If L1 cannot place the
   whole budget (`success = false`, §7.6) the cycle **ends here** — no
   cluster and no backend is scaled in that cycle.
5. **Per allocated cluster** (in score order):
   1. strategic target `T = clip(allocationᵢ, c_min, c_max)`;
   2. L2 correction `δ = clip(PD(λ̂ᵢ, λᵢⁱⁿ), −1, +1)` (§8), target
      `clip(T + δ, c_min, c_max)`;
   3. rate limit: at most **+6 / −1** replicas from the current count;
   4. **dead zone**: a change of −1 or 0 is skipped when the total traffic
      changed by less than 10 % since the previous cycle;
   5. churn budgets for the cycle: at most **8** frontend pods added and
      **2** removed (shared by all clusters);
   6. scale-down only if the last scale-down of the service is older than
      **90 s** *and* the last scale-up is older than **90 s**;
   7. `kubectl scale` (Kubernetes API) and gauges
      `dmos_current_replicas`, `dmos_target_replicas`,
      `dmos_predicted_traffic`, `dmos_cluster_score`,
      `dmos_scaling_events_total`.
6. **Clusters with no allocation** (score 0, not eligible, or allocated 0)
   are scaled down to `c_min` (one step, subject to the 90 s guards). Since
   `c_min = 1`, every cluster always keeps at least one frontend: the replicas
   running can exceed N by the number of clusters allocated 0.
7. **Co-scaling** of the backends from the frontend counts now in
   Kubernetes (§9), or the reactive `--naive-backends` alternative.

---

## 7. Level 1 — placement

### 7.1 Per-cluster observed state

`DMOSScheduler._collect_cluster_metrics` builds a `ClusterMetrics` for each
cluster with parallel queries (§5.1):

| Field | Warm phase | Cold phase (first 120 s) |
|---|---|---|
| `cpu_available_cores`, `memory_available_gb` | Prometheus | Prometheus |
| `cpu_total_cores`, `memory_total_gb` | `local.yaml` | `local.yaml` |
| `request_rate_current` | `get_request_rate("frontend")` (fallback: CPU usage × capacity) | λᵢⁱⁿ |
| `request_rate_max` | `cpu_cores × capacity_req_per_sec` (8 × 60 = 480 req/s on HIGH) | same |
| `latency_mean_ms` | Hubble p95 (variance 0) | `latency.baseline_ms` |
| `network_rtt_ms` | ping_exporter | ping_exporter |
| `carbon_intensity_gco2_kwh` | carbon client (300 if None) | same |
| `ingress_rate_rps`, `ingress_demand_share` | λᵢⁱⁿ, dᵢ | same |

Because of the start-up grace (§6.2 step 2), no allocation is ever computed
during the cold phase; see §7.4.

### 7.2 L0 — eligibility

A cluster is excluded (score 0, no bid) if any of these holds
(`dmos_scheduler.py`, module constants):

| Condition | Constant |
|---|---|
| CPU utilization > 95 % | `CPU_HARD_LIMIT_PCT` |
| memory utilization > 90 % | `MEMORY_HARD_LIMIT_PCT` |
| free CPU < 0.2 cores | `MIN_CPU_CORES_AVAILABLE` |
| free memory < 0.2 GB | `MIN_MEMORY_GB_AVAILABLE` |

Utilization is `(total − free) / total`, with *total* from `local.yaml` and
*free* from Prometheus.

### 7.3 Objective components

All components are in [0, 1] (`src/level1/score_functions.py`). Parameter
values are the defaults of `ScoreParameters` unless noted.

| Component | Formula | Parameters | Notes |
|---|---|---|---|
| latency Φ_lat | `1 / (1 + η·Lᵢ) · exp(−var/σ²)` | η = 0.001 /ms, σ² = 10⁶ ms² | the variance is always 0 in the warm phase, so Φ_lat = 1/(1 + η·p95) |
| capacity Φ_cap | `hw · (β + (1−β)·slack)` | hw = min(1, cpu_cores/8), slack = min(free CPU fraction, free memory fraction), β = 0.3 | clipped to [0, 1] |
| network proximity Φ_net | `exp(−ρ · min(RTTᵢ, RTT_max) / RTT_max)` | **ρ = 3.0** (`weights.yaml → network_parameters.rho`), RTT_max = 500 ms | falls back to RTT = 5 ms |
| carbon Φ_car | `exp(−ν · CIᵢ / CI_max)` | ν = 0.5, CI_max = 500 gCO₂/kWh | CI above CI_max is not clipped |
| demand share d | `clip(dᵢ, 0, 1)` | — | §5.2 |
| pressure Φ_pressure | `1 / (1 + exp(−k·(u − τ)))`, damped by `max(0.4, 1 − 0.6·(u − 1.5))` for u > 1.5 | u = λᵢ / λᵢᵐᵃˣ, τ = 0.7, k = 10 | legacy path only (weight `omega_demand`); not part of any score model |

`weights.yaml → carbon_parameters` (which says `ci_max: 800`) is **not read**;
the values above are the ones in effect, as recorded in each run's
`dmos_config.json`.

### 7.4 Score models (the role of demand)

A **score model** fixes in one named place how the demand share enters the
score, the four objective weights, the gate, the demand floor and the cold
start. It is selected with `DMOS_SCORE_MODEL=<name>` (the run harness does it:
`bash experiments/run_one.sh <model> <regime>`) and defined in
`config/weights.yaml → score_models`; validation in
`src/level1/score_model.py`.

| `demand_mode` | Score |
|---|---|
| `gate` | `S = G(d) · [ω_lat Φ_lat + ω_cap Φ_cap + ω_net Φ_net + ω_car Φ_car]`, with `G(d) = (ε + (1−ε)·d)^g` (g = 0 → G ≡ 1) |
| `additive` | `S = ω_lat Φ_lat + ω_cap Φ_cap + ω_net Φ_net + ω_car Φ_car + ω_dem · d` |
| `locality` | `S = ω_lat Φ_lat + ω_cap Φ_cap + ω_net · (d · Φ_net) + ω_car Φ_car` |

Weights must be ≥ 0 and sum to 1; a `demand` weight is accepted only in
`additive` mode. Under a model the legacy environment switches
(`DMOS_SCORE_GATE*`, `DMOS_WEIGHTS_COHERENT`, `DMOS_LOCALITY_PURE_NET`,
`DMOS_DEMAND_FLOOR`, `DMOS_SCORE_PROFILE`) are ignored and reported in the log
and in the run provenance.

Models defined today:

| Model | Mode | Gate (ε, g) | ω lat / cap / net / car (/ dem) | Purpose |
|---|---|---|---|---|
| `nca` | gate | 0.05, 2 | 0.25 / 0.35 / 0.30 / 0.10 | configuration *described* in the NCA 2026 paper (Eq. 1–2, §V-B); the campaign ran arm `l1l2` (§7.5, §13 item 2) |
| `gate_g0` | gate | 0.05, 0 | same as `nca` | E2: no demand in the score (pure weighted sum) |
| `gate_g1` | gate | 0.05, 1 | same as `nca` | E2: linear gate |
| `gate_g3` | gate | 0.05, 3 | same as `nca` | E2: stronger gate |
| `additive_dem50` | additive | — | 0.125 / 0.175 / 0.15 / 0.05 / 0.50 | E2: demand as a fifth weighted term |
| `locality` | locality | — | same as `nca` | demand multiplies only the proximity term |
| `latency_first` | gate | 0.05, 2 | 0.55 / 0.15 / 0.15 / 0.15 | E3 profile (proposal) |
| `capacity_first` | gate | 0.05, 2 | 0.15 / 0.55 / 0.15 / 0.15 | E3 profile (proposal) |
| `carbon_first` | gate | 0.05, 2 | 0.15 / 0.15 / 0.15 / 0.55 | E3 profile (proposal) |

All models have `demand_floor: false` and `cold_start: structural`.

**Gate strength, example** (ε = 0.05, g = 2): d = 0.85 → G = 0.735;
d = 1/6 → G = 0.043; d = 0.03 → G = 0.006. A cluster without demand keeps a
small positive score (G(0) = ε^g), so it still bids but receives almost
nothing.

**Cold start.** The first 120 s after DMOS starts are, by design (paper
§IV-E), an observation window: the live metrics (latency, load) are not yet
reliable, so DMOS only listens to the system and performs no scaling
operation — the start-up grace of the orchestrator (`startup_grace_s` =
120 s, §6.2) stops every cycle before the score is computed. The model's
`cold_start` field defines the score of that phase, recorded in the
provenance: `structural` (ω_lat = ω_cap = 0, the other weights
renormalised, demand keeps its role — the paper's description) or `same`
(the warm weights). The first allocation happens after the window, with the
warm score.

### 7.5 Legacy configuration path (no score model)

Without `DMOS_SCORE_MODEL` the score is configured by the active profile of
`weights.yaml → score_weights` (`active: "flash_crowd_aware"`, or
`DMOS_SCORE_PROFILE=<profile>`) plus environment switches. This path is kept
unchanged so that the historical arms remain reproducible.

```
S = ω_latency·Φ_lat + ω_capacity·Φ_cap + ω_geo_load·Φ_loc + ω_carbon·Φ_car + ω_demand·Φ_pressure
Φ_loc = d·Φ_net                    (Φ_net alone if DMOS_LOCALITY_PURE_NET=1)
S ← G(d)·S                         if DMOS_SCORE_GATE=1  (ε = DMOS_SCORE_GATE_EPS, default 0.05;
                                                          g = DMOS_SCORE_GATE_GAMMA, default 1)
DMOS_WEIGHTS_COHERENT="lat,cap,net,car" replaces the four weights and sets ω_demand = 0
```

`flash_crowd_aware` = 0.10 / 0.20 / 0.35 / 0.10 / 0.25. The YAML key names are
historical: `omega_geo_load` weighs Φ_loc and `omega_demand` weighs
Φ_pressure. The cold phase uses the `cold_start` profile
(0 / 0 / 0.35 / 0.30 / 0.35), but the environment switches apply to both
phases, so with `DMOS_WEIGHTS_COHERENT` the cold phase has the warm weights.

Arms of the run harness that use this path (names defined in
`experiments/run_naming.py`, the same in every script; old names in
brackets):

| Arm | Environment | Resulting score |
|---|---|---|
| `l1l2` (was `l1l2coh`) | `DMOS_DEMAND_FLOOR=0 DMOS_SCORE_GATE=1 DMOS_SCORE_GATE_GAMMA=2 DMOS_LOCALITY_PURE_NET=1 DMOS_WEIGHTS_COHERENT=0.25,0.20,0.25,0.30` | `S = G(d; 0.05, 2)·(0.25 Φ_lat + 0.20 Φ_cap + 0.25 Φ_net + 0.30 Φ_car)`, no floor — **the configuration of the NCA 2026 campaign**, read back from its DMOS logs (15–17 June 2026) |
| `legacy-gate1` (was `l1l2` of `run_one.sh`) | `DMOS_DEMAND_FLOOR=0 DMOS_SCORE_GATE=1` | `flash_crowd_aware`, Φ_loc = d·Φ_net, gate g = 1 |
| `legacy-nofloor` (was `l1l2nofloor`) | `DMOS_DEMAND_FLOOR=0` | additive `flash_crowd_aware`, no gate, no floor (negative control) |
| `legacy-default` (was `l1l2floor`, and `l1l2` of the old matrix/sweep scripts) | none | code default: `flash_crowd_aware` with demand floors |
| `profile-<name>` (was `latonly`, `netonly`, `carbon`) | `DMOS_SCORE_PROFILE=<name>` | that profile, with demand floors |
| `l2` | `--no-scheduling` | no L1: uniform N/K |
| `off` | DMOS not started | one replica per service per cluster |

The demand floor (`DMOS_DEMAND_FLOOR`, default on in the code) pre-assigns
`needᵢ` replicas to each cluster and lets the score distribute only the rest.
Since the budget is itself `N = Σ needᵢ` (§6.2), with the floor on **nothing
is left for the score**: the allocation equals the per-cluster need, whatever
the weights (the score matters only if a cluster is excluded by L0, does not
bid, or has a bid capacity below its need). This applies to `legacy-default`
and to every `profile-*` arm, which therefore produce the same placement.

### 7.6 Bids and allocation

**Bid.** Every eligible cluster with score > 0 bids `(score, capacity)` with

```
capacity = min(⌊free CPU / CPU request⌋, ⌊free memory / memory request⌋, c_max)
```

(frontend requests: 100m CPU, 64Mi memory). The code comment next to this
line says the CPU/memory terms are "only for the log", but they are part of
the minimum; on an idle cluster `c_max` is the binding term.

**Allocation** (`WinnerDetermination.allocate`, paper Eq. 8):

1. Floors (when enabled): each cluster starts at `min(needᵢ, capacityᵢ)`; the
   remaining budget is `N − Σ floors`.
2. Multi-pass capacity check: the ideal share of each active cluster is
   `remaining · Sᵢ / Σ_active S`; clusters whose ideal share exceeds their
   spare capacity are set to their capacity and removed; repeat.
3. Largest remainder (Hamilton): ⌊share⌋ to each, then one replica each to
   the largest fractional parts until the budget is used.
4. **Geo-boost**: if the highest-scored active cluster has no more replicas
   than the lowest-scored one, their scores differ and the lowest has at least
   2, one replica moves from the lowest to the highest.
5. Result: clusters with > 0 replicas; `success` is true only if the whole
   budget was placed. Otherwise the orchestrator abandons the cycle (§6.2).

The Jain fairness index of the allocation is logged.

---

## 8. Level 2 — predictor and PD correction

Each (service, cluster) pair has a `ReplicaScaler`, used as a container for a
`TrafficPredictor` and a `PDController` and for the rate limits.
(`ReplicaScaler.compute_target_replicas` itself is not called.)

### 8.1 Trend predictor (`src/level2/predictor.py`, paper Eq. 9)

```
derivative = (λ(t) − λ(t − τ)) / τ          τ ≈ window W = 60 s (closest stored sample)
λ̂ = λ + derivative · H · f                   H = 120 s, f = 0.5 if derivative > 0, 0.3 otherwise
λ̂ = min(λ̂, 2·λ)                             cap
λ̂ = max(λ̂, λ)                               floor: never predicts a decrease
```

The floor makes the negative-trend factor (0.3) inert. With fewer than two
samples the prediction is λ. W and H can be overridden with
`DMOS_PREDICTOR_WINDOW_S` / `DMOS_PREDICTOR_HORIZON_S`. Each call of
`predict` stores a sample: the periodic check and the scheduling cycle both
call it, so there are about two samples per 30 s per cluster. The prediction
is used (a) to trigger cycles early (§6.1) and (b) as the reference of the PD
correction.

### 8.2 PD correction (`src/level2/pd_controller.py`, paper Eq. 10)

```
e  = (λ̂ − λ) / λ̂                              (0 if λ̂ = 0, state reset)
|e| < 0.20  → output 0 (deadband; state updated)
u  = Kp·e + Kd·(e − e_prev)/Δt                 Kp = 2.0, Kd = 20.0
u  = clip(u, −3, +3), rounded to an integer
δ  = clip(u, −1, +1)                           applied correction (pd_correction_clamp)
```

Because λ̂ ≥ λ, e ≥ 0 and the proportional term only adds replicas while the
traffic rises; the derivative term can be negative when the error shrinks.
With Kp = 2 the proportional term alone reaches ±0.5 (rounded to a replica)
only for e ≥ 0.25.

---

## 9. Backend co-scaling

At the end of every cycle (`DMOSOrchestrator._enforce_colocation`, paper
Eq. 11), for each backend *b* of `services.yaml → application.co_scaled_services`
(in that order) and each cluster *i*:

```
ratio_nominal = (c_frontend / c_b) · fanout_b
ratio         = max(ratio_nominal, 0.3)                 backend_colocation_floor
targetᵢ       = clip(⌈frontend_replicasᵢ · ratio⌉, c_min_b, c_max_b,i)
```

`frontend_replicasᵢ` is read from Kubernetes (a fact, not a metric), so the
backends are immune to the retry-inflated request rates seen under
saturation. Resulting ratios with the current `services.yaml`:

| Backend | Nominal | Effective |
|---|---:|---:|
| cartservice | 60/400 × 0.65 = 0.10 | 0.30 (floor) |
| productcatalogservice | 60/1200 × 2.25 = 0.11 | 0.30 (floor) |
| checkoutservice | 60/160 × 0.05 = 0.02 | 0.30 (floor) |
| recommendationservice | 60/40 × 0.40 = 0.60 | 0.60 |
| currencyservice | 60/250 × 3.4 = 0.82 | 0.82 |
| adservice | 60/400 × 0.43 = 0.06 | 0.30 (floor) |
| shippingservice | 60/500 × 0.14 = 0.02 | 0.30 (floor) |

Co-scaling is bidirectional (it follows the frontend up and down), applies
the change in one step (no ±rate limit) and has its own churn budget per
cycle: **12** pods added, **6** removed. It records a scale-up/down time for
the backend service (used by nothing else).

`--naive-backends` replaces co-scaling with reactive scaling on each
backend's observed L7 rate (`ceil(rps / 50)`, frontend budgets). It was built
for a metastability study and no current harness uses it.

---

## 10. Safeguards and timing constants

All in `src/dmos_main.py` unless noted; the effective values of a run are in
its `dmos_config.json`.

| Constant | Value | Effect |
|---|---|---|
| `polling_interval` | 30 s (hard-coded; `global.scheduling_interval_seconds` in `local.yaml` is not read) | period of the periodic check |
| `high_threshold` / `low_threshold` | 30 / 10 req/s | event triggers (§6.1) |
| traffic floor | 2 req/s | below it the predictor is bypassed in the trigger |
| `debounce_seconds` | 30 s | minimum time between two cycles of the same service |
| `startup_grace_seconds` | 120 s | no Kubernetes operation after start |
| `COLD_START_SECONDS` (`dmos_scheduler.py`) | 120 s | cold-phase score (never used, §7.4) |
| `max_delta_up` / `max_delta_down` | +6 / −1 | per cluster per cycle (frontend) |
| `pd_correction_clamp` | 1 | applied PD correction |
| `dead_zone_pct` | 10 % | skip −1/0 changes when traffic is stable |
| `scale_down_cooldown` | 90 s | since the last scale-down of the service |
| `scale_up_protection_seconds` | 90 s | no scale-down within 90 s of a scale-up |
| `scale_up_budget_per_cycle` / `scale_down_budget_per_cycle` | 8 / 2 pods | frontend, all clusters together |
| `backend_scale_up_budget_per_cycle` / `…_down_…` | 12 / 6 pods | co-scaling |
| `backend_colocation_floor` | 0.3 | §9 |
| `scaler_safety_margin` | 0.15 | passed to `ReplicaScaler`, no effect (method not called) |

The cooldown and protection timestamps are per service, not per cluster: a
scale-up anywhere blocks scale-downs of that service everywhere for 90 s.

---

## 11. Interfaces: ports, endpoints, exported metrics

### Per cluster

| Service | Port |
|---|---|
| Cilium Ingress (HTTP, load entry) | NodePort 30080 |
| Prometheus (kube-prometheus-stack) | NodePort 32090 (`prometheus_port`) |
| Kubernetes API (K3s, control-plane node) | 6443, from the kubeconfig |

### DMOS host

| Endpoint | Content |
|---|---|
| `:9090/metrics` | Prometheus metrics below |
| `GET :8081/health` | `status`, queue size, workers, `config_id` |
| `GET :8081/config` | effective configuration of the process (JSON, see [RUN-PROVENANCE.md](RUN-PROVENANCE.md)) |
| `POST :8081/webhook/alert` | Alertmanager webhook: alerts with a `service` label become events (priority from `severity`) |
| `POST :8081/api/schedule` | manual trigger, body `{"service": "frontend"}` |

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `dmos_actual_traffic` | gauge | service | total λⁱⁿ of the last cycle |
| `dmos_predicted_traffic` | gauge | cluster, service | λ̂ᵢ of the last cycle |
| `dmos_current_replicas` | gauge | cluster, service | replicas read from Kubernetes before the change |
| `dmos_target_replicas` | gauge | cluster, service | target after rate limit |
| `dmos_cluster_score` | gauge | cluster, service | Sᵢ (set for clusters that received replicas) |
| `dmos_scaling_events_total` | counter | cluster, service, action | `scale_up`, `scale_down`, `scale_up_colocation`, `scale_down_colocation`, `*_naive` |
| `dmos_scheduling_duration_seconds` | histogram | service | duration of a cycle |
| `dmos_config_info` | gauge (= 1) | config_id, level1, demand_roles, gate_gamma, score_profile | provenance of the process |

At start-up the replica gauges are initialised from Kubernetes. How these
metrics are collected during a run and turned into results is described in
[METRICS-AND-OUTPUTS.md](METRICS-AND-OUTPUTS.md).

### Command line

```
python -m src.dmos_main [--no-scheduling] [--naive-backends]
                        [--config-dump PATH] [--dump-config-only PATH]
```

`--config-dump` (default `$DMOS_CONFIG_DUMP`) writes the effective
configuration at start; `--dump-config-only` builds it without opening ports
or contacting any cluster and exits (used by the harness to check an arm
before a run).

---

## 12. Code map

| File | Content |
|---|---|
| `src/dmos_main.py` | `DMOSOrchestrator`: threads, trigger, cycle, L2 application, co-scaling, safeguards, HTTP endpoints, CLI |
| `src/level1/dmos_scheduler.py` | `DMOSScheduler`: metric collection, L0, two score functions (warm/cold), bids, call to the allocator |
| `src/level1/score_functions.py` | `ClusterMetrics`, `ScoreParameters`, `ScoreFunctions` (components, gate, legacy path) |
| `src/level1/score_model.py` | `ScoreModel`: named score models, validation, cold variant |
| `src/level1/winner_determination.py` | `WinnerDetermination.allocate` (floors, capping, LRM, geo-boost) |
| `src/level2/predictor.py` | `TrafficPredictor` |
| `src/level2/pd_controller.py` | `PDController` |
| `src/level2/scaler.py` | `ReplicaScaler` (container of predictor, PD and limits) |
| `src/metrics/prometheus_client.py` | per-cluster PromQL queries |
| `src/metrics/carbon_client.py` | carbon providers (`trace`, `realistic`, `mock`, `electricitymaps`) |
| `src/k8s/client.py` | per-cluster Kubernetes clients (`get_deployment_replicas`, `scale_deployment`) |
| `src/utils/config_loader.py` | loading of `config/*.yaml` |
| `src/utils/provenance.py` | effective configuration and `config_id` |

`deployments/dmos-scheduler.yaml` and `deployments/dmos-config.yaml` describe
an in-cluster deployment of DMOS that is not supported (no image is built or
published); DMOS runs on the operator host.

---

## 13. Known limitations and paper ↔ code differences

Items marked **verify** need a check on the running testbed (phase G0 of the
test plan); items marked **decide** need a choice for the journal version.

| # | Topic | What the code does | Consequence |
|---|---|---|---|
| 1 | Network sensitivity ρ | 3.0 (`network_parameters.rho`), also in the NCA campaign | Tab. I of the NCA paper says 2.0 — **decide** |
| 2 | Weights of the NCA campaign | the campaign ran gate g = 2 with lat/cap/net/car = 0.25/0.20/0.25/0.30 and Φ_loc = Φ_net, floor off (arm `l1l2`, §7.5; DMOS logs of 15–17 June 2026) | §V-B of the paper states (0.25, 0.35, 0.30, 0.10), which is the `nca` model: the published results come from `l1l2` — **decide** which is "DMOS" for the journal |
| 3 | Carbon traces | not distributed with the repository (Electricity Maps terms); a cluster without its file uses the synthetic `realistic` profile, which depends on the wall-clock hour and month (§5.3); NO uses its trace via `zone_aliases` | download the six zones with an access that includes historical data (academic access); in the NCA campaign NO, US-VA and SG were all synthetic. Each run records the source per cluster |
| 4 | Cycle aborted when the budget does not fit | `success = false` → nothing scaled in that cycle, backends included (§6.2) | only when a cluster is excluded by L0 or its CPU/memory-based bid capacity is below its need — **verify** in `dmos.log` ("Scheduling fallito") |
| 5 | Cluster excluded by L0 | it receives no allocation and is scaled down to `c_min` in one step when the 90 s guards allow (§6.2 step 6) | under saturation an overloaded cluster can lose replicas at once — **verify** ("Cluster esclusi" in `dmos.log`) |
| 6 | Latency signal | p95 over all L7 flows of the namespace on the cluster; 2 ms baseline when missing | a cluster without L7 data looks fast (Φ_lat ≈ 1) |
| 7 | Free resources | summed over all nodes in kube-state-metrics, totals from `local.yaml` (worker) — **verify** whether the control-plane node is included | free fraction can exceed 1 (Φ_cap clipped), L0 thresholds less strict than they read |
| 8 | Data plane | frontend global, shared, `affinity: none` as in the paper (§4), checked before every run; the four backends stay `affinity: local` | the paper says "the services are exposed as global services with affinity: none" — **decide** whether that includes the backends |
| 9 | Geo-boost | one replica moved to the best-scored cluster in a tie (§7.6) | not described in the paper |
| 10 | PD clamp, co-scaling floor | ±1 applied, floor 0.3 | Tab. II reports the raw ±3; the floor is not in Eq. 11 |
| 11 | Unused configuration | `scheduling_interval_seconds`, `carbon_parameters`, `global_autoscaling` (services.yaml), `objectives` | values there have no effect; see [configuration-reference.md](configuration-reference.md) |
| 12 | Single process | DMOS is not replicated; state (predictor history, cooldowns) is in memory | a restart resets history and grace |
| 13 | Demand floor on (code default) | budget = Σ floors, so the score distributes nothing (§7.5) | `legacy-default` and `profile-*` arms place replicas by demand only |
| 14 | Online Boutique `loadgenerator` | the reset scripts scale it to 0; every run refuses to start if it has replicas and the manifest flags it (error `loadgenerator_running`) | in the June 2026 runs it was running (the reset scaled it to 1): the archived runs show loadgenerator → frontend flows in Hubble on five of the six clusters; its rate was not recorded |
