# [ARCHIVED] DMOS — Metrics and Measurement

> **This document is archived (October 2026)** and kept for history. Its
> content is now covered, and corrected, by:
>
> - [`../dmos-architecture.md`](../dmos-architecture.md) §5 and §11 — the
>   queries DMOS runs and the metrics it exports;
> - [`../METRICS-AND-OUTPUTS.md`](../METRICS-AND-OUTPUTS.md) — run outputs,
>   metric definitions (p95, error rate, SLO, goodput) and figures;
> - [`../traffic-metrics-architecture.md`](../traffic-metrics-architecture.md)
>   — the ingress-demand signal.
>
> Statements below that no longer hold: the backends do have an L7 policy
> (`cnp-backends-l7.yaml`), so the namespace-wide p95 includes gRPC flows;
> the carbon chain is trace → built-in profiles (not Electricity Maps → static
> table); the scaler's safety margin and the predictor's "EMA" have no effect
> (there is no EMA); the collector samples every 5 s; the evaluation metrics
> of §9 (MAPE, provisioning ratio, proactive %, flapping) belong to the
> Locust-era analysis and are not computed by the current pipeline.

> Canonical reference for what DMOS measures, where the data comes from, the
> PromQL queries it runs, and how the scheduler consumes the results.
>
> This document supersedes and consolidates the earlier Italian-language docs
> `dmos-metrics-analysis.md` and `dmos-traffic-measurement.md`. See the
> "Migration / history" footer for context.

---

## Table of contents

1. [What DMOS measures](#1-what-dmos-measures)
2. [Where the data comes from](#2-where-the-data-comes-from)
3. [Per-cluster Prometheus topology](#3-per-cluster-prometheus-topology)
4. [Canonical PromQL queries](#4-canonical-promql-queries)
5. [How DMOS consumes the metrics](#5-how-dmos-consumes-the-metrics)
6. [The 5-component cluster score](#6-the-5-component-cluster-score)
7. [Fallback chains and graceful degradation](#7-fallback-chains-and-graceful-degradation)
8. [DMOS-exposed Prometheus metrics](#8-dmos-exposed-prometheus-metrics)
9. [Evaluation metrics (offline / post-hoc)](#9-evaluation-metrics-offline--post-hoc)
10. [Methodological notes and known limitations](#10-methodological-notes-and-known-limitations)
11. [Migration / history](#11-migration--history)

---

## 1. What DMOS measures

DMOS — the Distributed Multi-Objective Scheduler — places replicas
of a service across six Kubernetes clusters in different geographic regions.
To do that it needs an honest picture of, for every cluster:

| Signal | Why DMOS needs it |
|---|---|
| **Per-cluster request rate** (req/s reaching each frontend) | Splits demand across clusters; feeds the per-cluster `TrafficPredictor`. |
| **External ingress rate** (req/s entering the system from outside) | Drives Φ_demand — the share of geographic demand each cluster sees. |
| **Per-service request rate** (req/s for downstream services) | Diagnostics only — backends are sized by co-scaling from frontend replicas, not from their measured rate (see §4.2). |
| **HTTP P95 latency** (server-side, per workload) | Feeds Φ_resp — penalises clusters whose response time is degrading. |
| **CPU and memory headroom per node** | Feeds Φ_cap — the spare-capacity score. |
| **Replica state** (current vs desired, per service per cluster) | Closes the loop between scheduler intent and actual cluster state. |
| **Inter-cluster network RTT** | Feeds Φ_net — used inside the locality term Φ_demand × Φ_net. |
| **Carbon intensity per region** | Feeds Φ_carbon — biases placement toward greener clusters. |

Latency is consumed at two distinct vantage points and the distinction is
load-bearing — see [§10.4](#104-server-side-vs-client-side-latency).

---

## 2. Where the data comes from

DMOS reads exclusively from per-cluster Prometheus instances. It does **not**
talk to the Kubernetes API for metrics, and it does **not** scrape exporters
directly.

### 2.1 Hubble L7 (Cilium httpV2 exporter)

Cilium's Hubble component, with the `httpV2` metrics profile and an L7
`CiliumNetworkPolicy` carrying `rules: http`, instruments Envoy proxies inside
each cluster and emits two HTTP-level metrics:

| Metric | Type | Purpose |
|---|---|---|
| `hubble_http_requests_total` | Counter | Per-request count, with rich labels |
| `hubble_http_request_duration_seconds_bucket` | Histogram | Per-bucket latency observations for `histogram_quantile()` |

Labels carried by both metrics:

| Label | Meaning |
|---|---|
| `destination_workload` | Kubernetes Deployment/StatefulSet of the receiving pod |
| `destination_namespace` | Namespace of the receiving pod |
| `source_workload` | Originating workload — empty string `""` for traffic entering from outside the cluster mesh |
| `traffic_direction` | `ingress` or `egress` from Hubble's perspective |
| `reporter` | `client` or `server` — which side of the proxy emitted the sample |
| `method` | HTTP method |
| `status_code` | HTTP response code |
| `le` | Histogram bucket upper bound (latency metric only) |

Hubble L7 is the **primary** source for both request rate and latency on
services that have an L7 CNP attached. For Online Boutique that is the
`frontend` deployment.

### 2.2 kube-state-metrics (via kube-prometheus-stack)

Standard Kubernetes object state — pod counts, resource requests, restart
counts. DMOS uses it as auxiliary input (e.g.
`kube_pod_container_resource_requests` as the denominator of the CPU-usage
query). Replica state (current vs desired) is deliberately **not** read from
these series: the scheduler reads it fresh from the Kubernetes API
(`get_deployment_replicas` in `src/k8s/client.py`) so scaling and co-scaling
decisions never act on scrape-delayed counts.

### 2.3 node-exporter

Per-node CPU and memory utilization. Used to compute headroom for Φ_cap.

### 2.4 ping_exporter (czerwonk)

A small exporter that performs ICMP RTT measurements between configured peer
IPs. The metric DMOS reads is:

```
ping_rtt_mean_seconds{target="<peer-ip>"}
```

These are inter-cluster RTTs, used to compute Φ_net.

### 2.5 Electricity Maps API (with static fallback)

Real-time grid carbon intensity per region (gCO₂eq/kWh). The carbon client
queries the Electricity Maps zone for each cluster's geographic region. If
the API is unreachable or returns an error, the client falls back to a
static lookup table compiled from regional yearly averages. See
[§7.1](#71-carbon-intensity-fallback) for the fallback chain.

---

## 3. Per-cluster Prometheus topology

Each of the six clusters runs its **own** Prometheus instance
(kube-prometheus-stack, namespace `observability`). There is no
federation, no Thanos, no remote-write into a global store.

| Cluster | Region (HW tier) | Prometheus URL |
|---|---|---|
| `ms-cluster-01` | EU-DE (HIGH) | `http://<CLUSTER_01_IP>:32090` |
| `ms-cluster-02` | EU-FR (MEDIUM) | `http://<CLUSTER_02_IP>:32090` |
| `ms-cluster-03` | EU-PL (LOW) | `http://<CLUSTER_03_IP>:32090` |
| `ms-cluster-04` | US-VA (HIGH) | `http://<CLUSTER_04_IP>:32090` |
| `ms-cluster-05` | APAC-SG (LOW) | `http://<CLUSTER_05_IP>:32090` |
| `ms-cluster-06` | EU-NO (MEDIUM) | `http://<CLUSTER_06_IP>:32090` |

Each Prometheus is exposed via NodePort `32090` — the default
`prometheus_port` in `src/utils/config_loader.py` (override per cluster in
`local.yaml` if the topology is not homogeneous). DMOS instantiates one
`PrometheusClient` per cluster directly — there is no `build_prom_map()`
helper. The mapping `cluster_name -> PrometheusClient` is built in the
scheduler bootstrap and held on the DMOS instance for the lifetime of the
process.

Because each Prometheus only sees pods scheduled on its own cluster, no
`cluster=` label filter is needed: the *origin* of the data already disambiguates.

```
DMOS scheduler
    ├─ PrometheusClient("http://<CLUSTER_01_IP>:32090")  → ms-cluster-01
    ├─ PrometheusClient("http://<CLUSTER_02_IP>:32090")  → ms-cluster-02
    ├─ ...
    └─ PrometheusClient("http://<CLUSTER_06_IP>:32090")  → ms-cluster-06
```

---

## 4. Canonical PromQL queries

These three queries are the contractual surface between DMOS and the
underlying observability stack. Any change to label semantics or selectors
must be reflected here.

### 4.1 External ingress rate (Φ_demand input)

The rate at which **external** clients enter the system on each cluster.
`get_ingress_rate()` measures it at the **origin** side: the local cluster's
Cilium Ingress Envoy is the HTTP *client* that forwards each request to the
frontend, so counting client-reported samples attributes the request to the
cluster where it **entered** — regardless of which cluster's pod ends up
serving it. This keeps Φ_demand loop-free by construction, even if
cross-cluster service sharing were enabled.

```promql
sum(rate(hubble_http_requests_total{
    destination_workload="frontend",
    destination_namespace="online-boutique",
    reporter="client",
    traffic_direction="egress"
}[1m]))
```

Why each label matters:

| Label | Why |
|---|---|
| `destination_workload="frontend"` | Restrict to the user-facing service. |
| `destination_namespace="online-boutique"` | Multi-tenant isolation. |
| `reporter="client"` | Sample recorded on the forwarding (origin) node — attributes the request to the cluster where it entered. |
| `traffic_direction="egress"` | The Ingress Envoy *emits* the request toward the frontend; that egress flow is the entry event. |

If the client-side query returns no samples, `get_ingress_rate()` falls back
to the server-side variant:

```promql
sum(rate(hubble_http_requests_total{
    destination_workload="frontend",
    destination_namespace="online-boutique",
    reporter="server",
    traffic_direction="ingress",
    source_workload=""
}[1m]))
```

Here `reporter="server"` counts once on the receiving Envoy,
`traffic_direction="ingress"` drops egress samples, and `source_workload=""`
keeps only **external** traffic: cross-cluster mesh traffic carries the
source pod's workload name and is excluded — including it would create a
positive feedback loop in Φ_demand for any cluster that starts redirecting
to peers (see §10.3). The fallback is valid as long as Services are not
shared across clusters (the current baseline); with
`io.cilium/shared-service: "true"` it would attribute requests to the
*serving* cluster instead of the entry cluster, which is why it is
loop-prone and logged as a warning.

Window choice: `[1m]` is the current canonical window, used together with
the **15 s** Hubble scrape interval set in `config/prometheus-helm-values.yaml`
(4 samples per window). If your environment runs Hubble at a slower scrape
interval (e.g. legacy 60 s), widen the window proportionally. See
[§10.2](#102-rate-window-vs-scrape-interval).

### 4.2 Per-service request rate (diagnostics)

`PrometheusClient` can also report the request rate to any named workload:

```promql
sum(rate(hubble_http_requests_total{
    destination_workload="<svc>",
    destination_namespace="online-boutique"
}[1m]))
```

This query is intentionally permissive on `source_workload`, so it counts
internal mesh traffic — which is exactly why it is **not** a scaling signal.
Under saturation, internal gRPC retries inflate the counter (no label
distinguishes a retry from an original request). The scheduler therefore
monitors only the frontend (`monitored_services = ['frontend']` in
`src/dmos_main.py`): backends have no Level-2 scaler of their own and are
sized exclusively by co-scaling from the frontend replica count — see
`docs/BACKEND-CO-SCALING.md`. The per-service rate remains available for
diagnostics.

### 4.3 P95 latency

Tail latency at the server side, computed from the Hubble latency histogram:

```promql
histogram_quantile(0.95, sum by(le)(
    rate(hubble_http_request_duration_seconds_bucket{
        destination_namespace="online-boutique"
    }[1m])
)) * 1000
```

Note the selector filters on `destination_namespace`, not on a specific
workload: in practice the only L7-instrumented workload in the namespace is
the frontend (backends have no L7 CNP — see §7.2), so the histogram is
frontend latency. The `* 1000` converts seconds to milliseconds.

The `sum by(le)` is required: `histogram_quantile` only operates on a vector
indexed by `le`. Aggregating away every other label first is the standard
Prometheus pattern.

---

## 5. How DMOS consumes the metrics

The scheduler talks to Prometheus through a thin client and a couple of
specialised helpers. The relevant entry points are:

### 5.1 `src/metrics/prometheus_client.py`

The single class `PrometheusClient` wraps the three queries above. The methods
the scheduler calls on each per-cluster client are:

- **`get_ingress_rate(namespace="online-boutique") -> float`** — runs the
  query in §4.1 and returns the external ingress rate for that cluster. The
  result is the per-cluster Φ_demand numerator.
- **`get_request_rate(service, namespace="default") -> float`** —
  runs the query in §4.2 for an arbitrary service (callers pass
  `"online-boutique"` explicitly). Returns 0.0 if no source has samples
  for the service.
- **`get_latency_p95(service, namespace="online-boutique") -> Optional[float]`** —
  runs the query in §4.3 and returns latency in milliseconds, or `0.0` if
  the histogram is empty (with a `+Inf` guard when every sample falls in the
  top bucket).

The client also exposes `get_cpu_available()`, `get_memory_available_gb()`
and `get_network_rtt_ms()` for the cluster-headroom and network-RTT score
inputs; see the source for their PromQL.

### 5.2 `src/metrics/hubble_client.py`

Not currently present in the tree — the system functions on
`prometheus_client.py` alone. When it exists, this module hosts
Hubble-specific helpers that go beyond plain PromQL — for example, decoded
label-set queries used during diagnostic sweeps. The scheduler hot path does
**not** depend on it: it is optional tooling.

### 5.3 `src/level1/score_functions.py`

The Level-1 scoring module is where every Φ component is consumed. For each
cluster, on each scheduling cycle, the scheduler builds a `ClusterMetrics`
record from the Prometheus clients and feeds it to:

- `compute_latency_score(metrics) -> Φ_resp` — uses P95 latency.
- `compute_capacity_score(metrics) -> Φ_cap` — uses CPU and memory headroom.
- `compute_load_score(metrics) -> Φ_pressure` — inverted sigmoid of
  utilization `1/(1+exp(-k·(util-τ)))`: HIGH for clusters that need more
  replicas.
- `compute_network_score(metrics) -> Φ_net` — uses inter-cluster RTT.
- `compute_geo_load_score(...) -> Φ_locality` — computes `Φ_demand × Φ_net`.
- `compute_carbon_score(metrics) -> Φ_carbon` — uses carbon intensity.
- `compute_demand_score(metrics) -> Φ_demand` — uses the per-cluster external
  ingress rate from §4.1, normalised by the total across clusters.
- `compute_total_score(...)` — the weighted sum (see §6).

### 5.4 Scheduling loop call sequence

For each scheduling cycle (default cadence ~30 s), per cluster:

```
1. PrometheusClient.get_ingress_rate()           → Φ_demand input
2. PrometheusClient.get_request_rate(service)    → per-service load
3. PrometheusClient.get_latency_p95(service)     → Φ_resp input
4. PrometheusClient.get_cpu_available()          ┐
   PrometheusClient.get_memory_available_gb()    ├─ Φ_cap input
5. PrometheusClient.get_network_rtt_ms(peers)    → Φ_net input
6. CarbonClient.get_carbon_intensity(region)     → Φ_carbon input
7. score_functions.compute_total_score(...)      → Φ_i
8. winner_determination → per-cluster replica targets
9. k8s.scale_deployment(cluster, target)         → close the loop
```

Per-cluster `ReplicaScaler` instances retain their own `TrafficPredictor`
state, so a spike on `ms-cluster-02` does not contaminate the predictor on
`ms-cluster-01`.

---

## 6. The 5-component cluster score

The composite score Φ_i for cluster *i* is a weighted sum of five
independently normalised sub-scores:

```
Φ_i = ω_lat·Φ_resp(i)
    + ω_cap·Φ_cap(i)
    + ω_geo_load·(Φ_demand(i)·Φ_net(i))      # Φ_locality
    + ω_carbon·Φ_carbon(i)
    + ω_demand·Φ_pressure(i)
```

Weight names exactly as configured (see the `balanced` profile in
`config/weights.yaml`):

| Weight | Default | Drives |
|---|---|---|
| `omega_latency` | 0.25 | Pulls replicas toward clusters with low P95 latency. |
| `omega_capacity` | 0.20 | Pulls replicas toward clusters with CPU/memory headroom. |
| `omega_geo_load` | 0.25 | Φ_locality = Φ_demand × Φ_net — anchors replicas to clusters that are the natural home of the demand, weighted by network proximity. |
| `omega_carbon` | 0.20 | Pulls replicas toward greener regions. |
| `omega_demand` | 0.10 | Φ_pressure — utilization pressure (inverted sigmoid): prioritises clusters approaching their target utilization. |

In the coherent configuration (`DMOS_WEIGHTS_COHERENT`, optionally with
`DMOS_LOCALITY_PURE_NET=1`), `omega_demand` is zeroed and Φ_locality reduces
to Φ_net: demand then enters the score only through the multiplicative
demand gate (`DMOS_SCORE_GATE`).

`score_functions.py` sanity-checks that the weights sum to 1.0 at
construction time and logs the active weight vector so post-hoc analyses can
recover the configuration of any given run from the log.

---

## 7. Fallback chains and graceful degradation

DMOS is designed so that no single observability gap takes the scheduler
offline.

### 7.1 Carbon intensity fallback

```
1. CarbonClient.get_carbon_intensity(region)
   ├─ Try: Electricity Maps API  ──── ✅ live grid intensity
   └─ Fall back to: static table  ──── ⚠️ regional yearly average
```

The static fallback values are checked in to source so the scheduler always
returns a usable number. Φ_carbon never returns `None`.

### 7.2 Hubble unavailable / L7 metrics empty

For services without an L7 CNP (every backend in Online Boutique:
`cartservice`, `productcatalog`, `checkoutservice`, `recommendationservice`,
…), `hubble_http_requests_total` carries no samples. `PrometheusClient`
falls back to a coarse rate estimate based on container network byte
counters:

```promql
sum(rate(container_network_receive_bytes_total{
    namespace="online-boutique",
    pod=~"<svc>.*"
}[1m]))
```

The byte rate is converted to a request rate by dividing by an empirical
average request size (≈4 kB for HTTP, calibrated against the frontend
profile). The conversion is approximate (±20–30 % for gRPC) and is logged
distinctively so it is obvious when the system is on the fallback path:

```
✅ Traffic from Hubble (ms-cluster-01):   7.2 req/s          # primary
⚠️ Traffic from network (ms-cluster-01): 3840 B/s → 0.96 req/s   # fallback
```

The 15 % safety margin baked into the replica scaler absorbs the systematic
bias of the byte-based estimate.

### 7.3 Prometheus unavailable

If a per-cluster Prometheus is unreachable, the corresponding
`PrometheusClient` returns `0.0` (or `None` for queries with optional return
types) and emits a warning. The scheduler treats a zero rate as "no demand
right now" and naturally drives the cluster toward the configured
`min_replicas` floor — i.e. it degrades to a high-availability minimum
rather than to zero capacity.

### 7.4 Startup grace period

For the first 120 s after DMOS boots, the scheduler **observes but does not
act**: no `kubectl scale` calls are issued, and `dmos_predicted_traffic` is
not published. This gives the per-cluster `TrafficPredictor` time to
accumulate enough history that its derivative term is meaningful, and
prevents large false-positive scale-ups driven by 1–2-point gradients. The
constant lives at `self.startup_grace_seconds = 120` in `src/dmos_main.py`,
aligned with the two-phase `COLD_START_SECONDS = 120`.

### 7.5 Traffic floor

After grace, the scheduler normally uses `effective = max(actual, predicted)`
so it scales pre-emptively. To avoid the predictor's EMA holding replicas
hot for tens of minutes after a test ends, a traffic floor of `2.0 req/s`
inverts that rule: when `0 < actual < 2.0`, the scheduler ignores the
predictor and uses `actual` directly, allowing scale-down to proceed.

---

## 8. DMOS-exposed Prometheus metrics

DMOS exposes its own Prometheus endpoint at `http://localhost:9090/metrics`
on the scheduler host. These are the metrics the offline analysis pipeline
(`experiments/collect_dmos_live.py` → `experiments/paper_lib.py` +
`paper_generate_all.py`) consumes.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `dmos_actual_traffic` | Gauge | `service` | Currently measured rate, summed across clusters. |
| `dmos_predicted_traffic` | Gauge | `cluster`, `service` | Next-cycle prediction from `TrafficPredictor`. |
| `dmos_current_replicas` | Gauge | `cluster`, `service` | Replicas currently running. |
| `dmos_target_replicas` | Gauge | `cluster`, `service` | Replicas targeted by the most recent decision. |
| `dmos_cluster_score` | Gauge | `cluster`, `service` | Most recently computed Φ_i. |
| `dmos_scaling_events_total` | Counter | `cluster`, `service`, `action` | Cumulative scale-up / scale-down operations. |
| `dmos_scheduling_duration_seconds` | Histogram | `service` | Wall-clock duration of one scheduling pass. |

These are the source for every "DMOS-side" plot in the evaluation pipeline.
Client-side experimental metrics (P95 per cluster, SLO violation rate) come
from the load generator, not DMOS.

---

## 9. Evaluation metrics (offline / post-hoc)

The evaluation pipeline complements the live scheduler signals with several
derived metrics. They are **not** consumed by the scheduler — they exist to
characterise behaviour after a run.

### 9.1 P95 latency per cluster (client-side)

Empirical P95 over 10-second windows, computed by the load generator from
its own `response.elapsed` timestamps. Distinct from the histogram-based
Hubble P95 used in Φ_resp — see [§10.4](#104-server-side-vs-client-side-latency).

### 9.2 SLO violation rate

Percentage of requests with response time above 1000 ms, per 10 s window
per cluster. Threshold of 1 s is conservative relative to comparable
literature.

### 9.3 Jain fairness index

Standard Jain index over per-cluster replica counts:

```
J(x) = (Σ xᵢ)² / (N · Σ xᵢ²),  with N fixed at 6.
```

`N` is held at 6 (the total cluster count) — **not** the count of clusters
with non-zero replicas — so concentrating all replicas on one cluster
correctly yields `J = 1/6 ≈ 0.167` instead of a misleading `J = 1.0`.

### 9.4 Provisioning ratio

```
ratio_t = (R_t · C_pod) / max(λ_t, λ_min)
```

with `C_pod = 60 req/s` (80 % of the measured capacity-test knee point) and
`λ_min = R_min · C_pod` representing the high-availability floor. Values
near 1.15× indicate the scheduler is honouring its 15 % safety margin
without over-provisioning.

### 9.5 Proactive %

Fraction of scale-up events for which `predicted > current × 1.03` at the
moment of the event — i.e. the predictor's trend term, not the current
load, drove the decision. The 3 % threshold filters out EMA noise around
steady-state.

### 9.6 Prediction accuracy: MAPE / RMSE / R² / directional accuracy

Standard regression metrics computed on the (`actual`, `predicted`) time
series. `MAPE active` (filtered to snapshots with non-trivial traffic) is
the primary headline number — `MAPE overall` is dominated by the
post-test EMA tail and is reported only as a reference.

### 9.7 Scaling oscillation (flapping)

A sliding window of 20 samples (~5 minutes at a 15 s collector cadence)
flags as "flapping" any window with ≥3 direction reversals
(+1 → -1 or -1 → +1) in the per-step replica-count delta.

---

## 10. Methodological notes and known limitations

### 10.1 Hubble counts more requests than the load generator

For `frontend`, `hubble_http_requests_total` is consistently ~1.4× the
transaction rate reported by the load generator. The reason is that Hubble
counts every HTTP request reaching the Envoy proxy — including assets
(CSS, JS, images) and downstream API calls fan-out from a single user
transaction. The ratio is **stable** because it depends on the page
profile, not on load. `config/services.yaml` calibrates
`capacity_req_per_sec` against the Hubble figure, not against load-generator
transactions, so the math is internally consistent.

### 10.2 `rate()` window vs scrape interval

`rate()` requires at least two samples in its window. Empirically:

```
rule of thumb: rate_window ≥ 2 × scrape_interval
```

With the **canonical 15 s scrape** configured in
`config/prometheus-helm-values.yaml`, `[1m]` gives 4 samples per window —
comfortably above the 2-sample floor. If you ever step the scrape back to
60 s (e.g. for a high-cardinality production cluster where the scrape cost
dominates), widen the rate window to `[5m]` or higher; otherwise the
network-bytes fallback will engage intermittently and give the appearance
of an outage on perfectly healthy services.

### 10.3 Keeping Φ_demand loop-free

Attributing a request to the cluster that *serves* it (rather than the one
where it *enters*) would inflate the receiving cluster's `ingress_rate`,
push that cluster's Φ_demand up, draw more replicas toward it, and attract
more cross-cluster redirections — a positive feedback loop. Two mechanisms
guard against this: the primary query in §4.1 measures at the **origin**
(client-reported samples on the entry cluster's Ingress Envoy), and the
server-side fallback filters on `source_workload=""` so only **external**
traffic is counted. The same reasoning is why frontend and backend Services
are not annotated `io.cilium/shared-service: "true"` — see
`docs/global-services-clustermesh.md`.

### 10.4 Server-side vs client-side latency

Two latency vantage points coexist and they are not interchangeable:

| Source | Where it is measured | Consumed by |
|---|---|---|
| `histogram_quantile(0.95, hubble_http_request_duration_seconds_bucket{…})` | Inside the cluster, at Envoy (server-side) | DMOS Φ_resp |
| Load-generator `response.elapsed` | Outside the cluster, end-to-end (client-side) | Offline evaluation only |

Server-side latency is the right scheduling signal — it isolates intra-cluster
processing time, which is what a scaling decision can actually move.
Client-side latency is the right evaluation signal — it represents the user
experience and is comparable across baselines and prior work.

### 10.5 Granularity

The collector samples at 15 s, fine enough to capture scaling events
(K8s scale operations take 30–60 s end to end) but too coarse for sub-15 s
latency spikes. The load-generator CSV uses 10 s windows and is the higher-
resolution source for client-side metrics.

### 10.6 EMA tail post-test

When load drops abruptly to zero, the `TrafficPredictor`'s EMA decays over
10–15 minutes. Without the traffic floor (§7.5) this would keep replicas
hot well past the end of a test. With the floor enabled, the post-test
behaviour is an artefact only of the offline `predicted_traffic` series,
not of actual replica state — but it does inflate `MAPE overall`, which is
why `MAPE active` is the reported primary statistic.

---

## 11. Migration / history

This document consolidates and supersedes two earlier Italian-language
references — `docs/dmos-metrics-analysis.md` (a metric-by-metric catalogue
of evaluation KPIs and their formulas) and `docs/dmos-traffic-measurement.md`
(a deep dive on `_get_per_cluster_traffic`, the Hubble-vs-network fallback
chain, and the startup grace period). The two documents had grown
substantially overlapping coverage of Hubble PromQL and the
client-vs-server latency distinction; they have been merged here, retranslated
into English for public release, refactored around the canonical PromQL
contract in §4, and updated to remove references to legacy artefacts
(`build_prom_map()`, `get_pod_count()`, `source="reserved:ingress"`, NGINX
as the active ingress) that no longer reflect the system. The original two
files have since been removed from the repository.
