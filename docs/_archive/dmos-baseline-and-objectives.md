# [ARCHIVED] DMOS — Starting Point and Project Objectives (thesis era)

> **This document is archived (October 2026).** It records where DMOS came
> from — an extension of G. Romano's latency-driven metascheduler (2025) —
> and the objectives set during the thesis work (March 2026, three-cluster
> testbed, Locust). Sections 1–4 (Romano's design and limitations) remain an
> accurate account of the starting point.
>
> The DMOS design described in sections 5–9 has since changed. For the
> current system see [`../dmos-architecture.md`](../dmos-architecture.md) and,
> for the published design, [`../publications/2026-nca-dmos.md`](../publications/2026-nca-dmos.md).
> In particular: the replica budget is computed from the ingress demand
> (not by Level 2); the score has four objectives with demand as a gate
> (score model `nca`); Φ_cap is hardware size × slack; latency is the p95
> itself (not p95/1.65); co-scaling covers seven backends, is bidirectional
> and fan-out-aware; the safeguards are 10 % dead zone, 90 s cooldowns, +6/−1
> and churn budgets; the predictor has no EMA; the testbed has six clusters
> and the load generator is k6.

## Baseline: Romano's Metascheduler (2025)

> Reference: G. Romano, *Microservice orchestration in multi-cluster environments*, MSc Thesis, 2025.

---

### 1. Original context and motivation

Romano's work originates in the context of the **Cloud Continuum** (or *Compute Continuum*), a paradigm that aims to unify computational resources distributed along an edge–fog–cloud continuum, allowing applications to be deployed transparently across heterogeneous infrastructures [Romano §1].

The central problem is the management of **microservice applications** on **multi-cluster Kubernetes** architectures: in environments where multiple geographically distributed clusters must cooperate, the native Kubernetes scheduling system is inadequate, because it operates exclusively within a single cluster and has no visibility into the global behavior of the system.

Romano proposes the construction of an **external metascheduler** that, operating above Kubernetes, dynamically redistributes microservice replicas among clusters as a function of observed latency.

---

### 2. Reference infrastructure

The infrastructure on which Romano builds and tests his system (3 physical
servers on a private LAN) is the starting point of DMOS; the current DMOS
testbed has since evolved into a **6-cluster fleet of Proxmox VMs on three
hardware tiers**:

| Component | Romano (2025) | DMOS (current) |
|-----------|---------------|----------------|
| **Machines** | 3 physical servers (ms01, ms02, ms03) on a private LAN | 6 K3s VM clusters on Proxmox (`ms-cluster-01`..`06`), 3 HW tiers: HIGH 8 vCPU/12 GB, MEDIUM 6 vCPU/8 GB, LOW 4 vCPU/8 GB; inter-region RTT emulated with tc netem (0–200 ms) |
| **Kubernetes** | K3s (lightweight distribution) on each machine | K3s on each VM |
| **CNI & Service Mesh** | Cilium + ClusterMesh (L3/L4/L7 inter-cluster interconnection) | Cilium ClusterMesh v1.19 (endpoint-sync disabled: services are de facto cluster-local, affinity=local baseline) |
| **Load Balancing** | Cilium Global Services: incoming requests on a cluster are automatically balanced across all global pods, regardless of the cluster | Cluster-local serving; cross-cluster overflow is an experimental east-west pilot (`deployments/clustermesh-overflow/`) |
| **Test application** | Online Boutique (Google microservices demo, ~11 microservices) | Online Boutique (same) |
| **Metrics** | Per-cluster Prometheus (NodePort 30090, standalone) + ping_exporter (inter-cluster latency) + Hubble L7 (HTTP traffic) | Per-cluster kube-prometheus-stack (namespace `observability`, NodePort 32090) + ping_exporter + Hubble L7 |
| **Load generator** | K6 | Locust in intermediate iterations (legacy); the current evaluation uses **k6** again (`experiments/k6_flash_crowd.js`) |

Romano's key architectural idea is **PROM_MAP**: each cluster has a local Prometheus instance that sees exclusively its own pods. The metascheduler does not embed static endpoint references in the code — the configuration is provided entirely through environment variables (e.g. `PROM_RO1`, `PROM_RO2`, `PROM_RO3`), allowing dynamic discovery of clusters without modifying the source code (p. 48-50, §5.2). The main loop sequentially calls the collection functions (`verificatraffic`, `verificalatencyintercluster`) for all active clusters and aggregates the data to build a global view of the system (Figure 5-3, p. 51).

This PROM_MAP pattern has been **inherited entirely by DMOS** (`src/level1/dmos_scheduler.py`), with the addition of parallelization via `ThreadPoolExecutor` — Romano queries clusters sequentially, DMOS queries them in parallel to reduce the latency of the scheduling cycle as the number of clusters grows.

---

### 3. Romano's metascheduler algorithm

Romano's metascheduler operates with a periodic loop (`CHECK_INTERVAL`):

```
every CHECK_INTERVAL seconds:
  1. Detect active clusters via cilium_ClusterMesh_remote_cluster_nodes
  2. For each cluster i:
     a. Query Prometheus_i → check traffic via hubble_http_requests_total
     b. Query Prometheus_i → get inter-cluster latency via ping_rtt_mean_seconds
  3. Compute global_replicas (autoscaling):
       total_sum = sum(traffic per cluster)
       global_replicas = int(MIN_REPLICAS + (MAX_REPLICAS - MIN_REPLICAS)
                             * (total_sum / TRAFFIC_THRESHOLD))
       global_replicas = clamp(global_replicas, MIN_REPLICAS, MAX_REPLICAS)
  4. Distribute global_replicas among clusters:
       score[c] = WT * normalized_local[c] + WL * normalized_remote[c]
       replicas[c] = max(MIN_REPLICAS_PER_CLUSTER,
                         round(global_replicas * score[c] / sum(scores)))
  5. Apply assignments via Kubernetes API (scale deployment):
       api.patch_namespaced_deployment_scale(
           name="frontend", namespace="default", body=body
       )
       → Only the "frontend" deployment is scaled.
         Backends (cartservice, productcatalogservice, etc.) remain static.
```

**Metrics used in the algorithm (Romano):**

| Metric | Source | Use |
|---------|-------|-----|
| `hubble_http_requests_total{source="reserved:ingress"}` | Per-cluster Prometheus (PROM_MAP) | Incoming traffic to the cluster through its own ingress controller — primary input both for global autoscaling and for the local component of the score |
| `ping_rtt_mean_seconds` | ping_exporter | Inter-cluster latency — input for the remote component of the score |

**Score function (Romano):** two-dimensional, combines local traffic and remote traffic weighted by latency:

```python
scores[c] = WT * normalized_local[c] + WL * normalized_remote[c]
```

where:
- `normalized_local[c]` = local Hubble traffic of cluster `c`, normalized over the maximum
- `normalized_remote[c]` = sum over all remote clusters `r` of `(traffic_to_r × latency(c,r))`, normalized over the maximum — represents the "cost" of serving traffic that arrives from high-latency remote clusters

The cluster with more local traffic and favorable latency to remote clusters receives more replicas. The actual balancing is then delegated to **Cilium Global Services**: with more active pods in a cluster, Cilium increases the probability that requests will be served locally.

**Autoscaling in Romano:** the linear formula `global_replicas = MIN + (MAX-MIN) * traffic/threshold` constitutes a form of **reactive** autoscaling — it scales the total number of replicas in direct proportion to the currently measured traffic, without future prediction.

**Romano's results:** scenario 1 (regularly increasing load, 10 min): global latency reduced from p95=3.88s to p95=1.23s with the metascheduler active. Scenario 2 (rapid traffic increase): analogous reduction documented in tables 5-1 and 5-2. The tests are conducted with K6 on two scenarios: uniform load and unbalanced load.

---

### 4. Limitations explicitly acknowledged by Romano

Romano in the *Limitations and future developments* section (p. 60) identifies two classes of open problems:

#### 4.1 Lack of awareness of hardware resources

> *«The metascheduler, in its current version, presents some limitations: it does not consider the hardware resources of the clusters, such as CPU and memory consumption. This is effectively a problem for the scaling decision since it can lead to increasing replicas on clusters that, despite being low-latency, do not have the resources necessary to handle new pods.»*

**The concrete problem:** a score based only on latency can assign replicas to a cluster that has CPU >90% or exhausted memory. K8s would accept the scale request but the pods would remain `Pending` indefinitely, degrading the service without the metascheduler noticing.

#### 4.2 Reactive autoscaling without prediction

> *«This also represents a possible future development, namely, extending the algorithm to consider hardware resources, and furthermore predictive autoscaling logic based on machine learning could also be considered.»*

**The concrete problem:** Romano implements autoscaling of the total replica count via a linear formula proportional to current traffic. This is a **purely reactive** approach: the system scales in response to already observed traffic, it does not anticipate the future peak. If traffic grows rapidly (flash crowd, sharp ramp-up), the system always scales late with respect to actual demand — the time required to create new pods (image pull, readiness probe, etc.) adds to the observation delay.

#### 4.3 Implicit limitations (not explicitly stated by Romano)

From the analysis of the thesis and the structure of the algorithm, further unnamed limitations emerge:

| Limitation | Description |
|------------|-------------|
| **Two-dimensional score** | Local traffic + latency × remote traffic — no hardware capacity, no carbon |
| **No carbon-awareness** | The carbon intensity of the energy mix per cluster is not considered |
| **Linear reactive autoscaling** | `global_replicas ∝ traffic_now` — no prediction of future trend |
| **No oscillation control** | Without hysteresis mechanisms, the system can oscillate (flapping) between consecutive decisions |
| **Inter-cluster vs intra-cluster latency** | Romano measures latency *between* clusters (ICMP RTT from A to B) to decide which cluster is "close" to the others. DMOS measures latency *within* each cluster (`hubble_http_request_duration_seconds_bucket`, p95 HTTP L7) to detect application saturation. They are two signals with completely different semantics. |
| **No fairness measure** | There is no fairness index to evaluate the distribution of replicas across clusters |
| **Static backends** | Only the `frontend` deployment (namespace `default`) is scaled. Backends (cartservice, productcatalogservice, checkoutservice, recommendationservice) always remain at the statically configured number of replicas, regardless of load. If the frontend scales to 6 replicas on cluster1 but the backend has 1 replica on cluster2, every gRPC call frontend→backend becomes cross-cluster, increasing end-to-end latency. |

---

## DMOS: objectives and design choices

DMOS (*Distributed Microservices Orchestration System*) is born as a **systematic extension** of Romano's work, with the objective of transforming the metascheduler from a system of **reactive redistribution** to a system of **proactive multi-objective orchestration**.

---

### 5. Primary objectives of DMOS

#### O1 — Proactive autoscaling based on traffic prediction

**Problem it solves:** Romano's metascheduler never scales the total number of replicas. DMOS must decide *how many* total replicas are needed before the peak even manifests.

**Approach adopted (Level 2):**

The `TrafficPredictor` module (`src/level2/predictor.py`) implements a trend-based prediction inspired by the PD model:

```
Λ^pred(t) = Λ(t) + (dΛ/dt) · Δt_horizon · damping_factor
```

where:
- `Λ(t)` = current traffic measured by Hubble L7 (req/s)
- `dΛ/dt` = approximated derivative over time window τ (default 5 min)
- `Δt_horizon` = prediction horizon (default 120s)
- `damping_factor` = 0.5 for scale-up (proactive), 0.3 for scale-down (conservative)

The total number of replicas is computed by `ReplicaScaler` (`src/level2/scaler.py`):

```
base_replicas = ceil(Λ^pred / capacity_per_replica × (1 + safety_margin))
target_replicas = clamp(base_replicas + PD_correction, min_replicas, max_replicas)
```

A PD controller (`PDController`) corrects the residual error between predicted and actual traffic, preventing both under-provisioning (positive error → additional scale-up) and excessive over-provisioning (negative error → brake on scale-down).

**Expected result:** DMOS must scale *before* demand exceeds current capacity (average Time-to-Scale < 0s, i.e. proactive).

---

#### O2 — Multi-objective scheduling with carbon-awareness and geo-awareness

**Problem it solves:** Romano optimizes only latency. DMOS must balance five objectives in tension with each other: application response time, available capacity, geographic position/load (combined), environmental impact, and geographic demand.

**Approach adopted (Level 1):**

The score function `ScoreFunctions` (`src/level1/score_functions.py`) implements a composite score with **5 terms** in which `Φ_load` and `Φ_net` appear **multiplied** under a single weight `ω_geo`:

```
score_i = ω_resp   · Φ_resp(i)
        + ω_cap    · Φ_cap(i)
        + ω_geo    · ( Φ_load(i) · Φ_net(i) )      ← multiplicative term
        + ω_carbon · Φ_carbon(i)
        + ω_demand · Φ_demand(i)

with  ω_resp + ω_cap + ω_geo + ω_carbon + ω_demand = 1.0
```

**Why multiplicative.** `Φ_locality = Φ_demand · Φ_net` reflects the constraint "good cluster ⇔ close *and* where demand originates". An additive formulation with two separate weights (`ω_demand · Φ_demand + ω_net · Φ_net`) would allow a cluster with no incoming demand but very central to win thanks only to high `Φ_net`, or a demand-heavy but distant cluster to win on `Φ_demand` alone. With the product, it suffices that one of the two signals be low to collapse the entire locality component. (Before the score refactor this term was called `Φ_geo = Φ_load · Φ_net`.)

Each component has a precise semantics:

| Component | Formula | Objective |
|-----------|---------|-----------|
| **Φ_resp(i)** | `(1/(1+η·E[L_i])) · exp(-Var(L_i)/σ²)` | Prefer clusters with low p95 application response time. `E[L_i]` and `Var(L_i)` derive from `hubble_http_request_duration_seconds_bucket` (mean=p95/1.65). Fallback: `baseline_latency_ms` from config. |
| **Φ_cap(i)** | `(R_avail/R_tot)^κ · (1 - λ_i/λ_max)` | Prefer clusters with free resources and low load |
| **Φ_load(i)** | `1 / (1 + exp(k·(λ/λ_max − τ)))` (sigmoid with threshold τ) | Φ_load ≈ 1 below τ, drops rapidly above τ. Penalizes only *truly saturated* clusters, preventing a distant but unloaded cluster from canceling the RTT advantage of a close cluster. Parameters: τ=0.7, k=10. |
| **Φ_net(i)** | `exp(-ρ · RTT_avg(i) / RTT_max)` | Prefer clusters with low average RTT to peers. RTT from `ping_rtt_mean_seconds` (ping_exporter, czerwonk/ping_exporter). Fallback 5ms if unavailable. Parameters: ρ=3.0, RTT_max=500ms. |
| **Φ_carbon(i)** | `exp(-ν · CI_i / CI_max)` | Prefer clusters powered by low-carbon-intensity energy. ν=0.5, CI_max=800 gCO₂/kWh. |
| **Φ_demand(i)** | `ingress_rate_i / Σ_j ingress_rate_j` | Fraction of ingress traffic on cluster i (Cilium Ingress + Hubble L7 with filter `source_workload=""` to isolate external traffic). Quantifies the actual **geographic demand**. |

The weights `ω` are configurable per profile (`cold_start`, `carbon_agnostic`, `balanced`, `carbon_priority`, `latency_priority`, `geo_aware`, `romano_like`) in `config/weights.yaml`. The `geo_aware` profile (ω_geo=0.45, ω_demand=0.20) is the one active in the experimental tests of the thesis; the `cold_start` profile (ω_resp=ω_cap=0) is applied by the scheduler in the first 120 s of execution, until Hubble and CPU have statistically significant samples.

**Carbon-awareness:** the carbon intensity `CI_i` (gCO₂/kWh) per geographic region is provided by `CarbonClient` (`src/metrics/carbon_client.py`). The exponential formula ensures that clusters with very high `CI` (e.g. PL: 650 gCO₂/kWh) systematically receive fewer replicas compared to low-emission clusters (e.g. FR: 80 gCO₂/kWh), at parity of latency and capacity.

---

#### O3 — Hardware resource awareness (hard constraints)

**Problem it solves:** Romano's limitation #1 — assigning replicas to clusters without resources.

**Approach adopted:** `DMOSScheduler._compute_cluster_score()` implements **hard constraints** verified *before* the score computation:

```python
CPU_HARD_LIMIT_PCT    = 95.0  # Cluster excluded if CPU > 95%
MEMORY_HARD_LIMIT_PCT = 90.0  # Cluster excluded if RAM > 90%
MIN_CPU_CORES_FREE    = 0.2   # At least 0.2 free cores
MIN_MEMORY_GB_FREE    = 0.2   # At least 0.2 GB free
```

A cluster that violates any of these constraints is excluded from allocation and receives score=0, regardless of its latency. This ensures that replicas are assigned exclusively to clusters physically capable of hosting them.

CPU and memory metrics are read from each cluster's local Prometheus via PROM_MAP — the same pattern as Romano, but extended to include `container_cpu_usage_seconds_total` and `node_memory_MemAvailable_bytes`.

---

#### O4 — Equitable distribution of replicas (Jain Fairness Index)

**Problem it solves:** a purely greedy scheduler could concentrate all replicas on the cluster with the highest score, violating the multi-cluster high-availability requirement.

**Approach adopted:** `WinnerDetermination.allocate()` (`src/level1/winner_determination.py`) implements a **score-proportional allocation**:

```
quota_i = score_i / Σ(score_j)
replicas_i = round(total_replicas × quota_i)
```

with the guarantee that every cluster with `capacity > 0` receives at least 1 replica.

**Fairness** is measured with the Jain index:

```
J(X) = (Σ r_i)² / (N · Σ r_i²)   ∈ [1/N, 1]
```

where `r_i` is the share of replicas assigned to cluster `i`. A value close to 1.0 indicates equitable distribution; a value close to 1/N indicates concentration on a single cluster.

Romano's metascheduler did not have any fairness measure in the distribution.

---

#### O5 — Oscillation prevention (anti-flapping)

**Problem it solves:** a purely reactive scaling system tends to oscillate — scale up, then immediately down, creating instability and resource waste.

**Mechanisms implemented in DMOS:**

| Mechanism | Implementation | Purpose |
|-----------|----------------|-------|
| **Dead zone** | If `|Δtraffic| < 15%`, no scaling | Ignores noisy variations |
| **Scale-down cooldown** | 60s between two consecutive scale-downs | Avoids premature scale-down |
| **Scale-up protection** | 120s lock after a scale-up | Allows replicas to stabilize before evaluating a scale-down |
| **max_delta_per_cycle** | Max ±4 replicas per cycle (30s) | Rate limiting on scaling decisions |
| **Traffic floor** | If `0 < FE < 2.0 rps`, bypasses the predictor and uses min_replicas | Avoids scaling to 0 for residual probe traffic |

None of these mechanisms existed in Romano's baseline.

---

#### O6 — Semantic change of the latency metric: from inter-cluster to intra-cluster

**Problem it solves:** Romano uses `ping_rtt_mean_seconds` as the only latency measure to drive replica distribution. This metric only captures the network cost between clusters, but is blind to application saturation — a cluster can have low ping and HTTP p95 of 2 seconds if its CPU is saturated.

**Comparison between the two metrics:**

| | `ping_rtt_mean_seconds` (Romano) | `hubble_http_request_duration_seconds_bucket` (DMOS) |
|---|---|---|
| **Layer** | L3 — network | L7 — HTTP application |
| **What it captures** | Packet transit time between nodes | Full HTTP request duration: network + queue + processing |
| **Depends on application load** | No — stable even with CPU at 100% | Yes — grows with cluster saturation |
| **Available without traffic** | Always (ping is active independently) | No — requires active HTTP requests; in their absence: fallback to `baseline_latency_ms` from config |
| **Noise** | Low, stable signal | Higher, varies instant by instant with load |
| **Correlation with user experience** | Indirect — only the network component | Direct — it is exactly what the user perceives |
| **Semantics for the scheduler** | "How topologically close is this cluster to the others?" | "How stressed is this cluster at this moment?" |

**Concrete example:** suppose cluster1 has ping=0.3ms toward the other clusters and cluster2 has ping=0.8ms. Romano assigns more replicas to cluster1. But if cluster1 has CPU at 92% and HTTP p95 = 1.8s, while cluster2 has p95 = 95ms, DMOS instead assigns more replicas to cluster2 — the less stressed cluster, not the topologically closest.

**Trade-off of DMOS's choice:** the metric is more precise for detecting cluster health, but introduces a dependency on active traffic. During startup or in idle, `hubble_http_request_duration_seconds_bucket` produces no data → DMOS falls back to `baseline_latency_ms` configured statically (see `dmos_scheduler.py:154`). In these cases, the two approaches converge to a fixed estimate.

**The DMOS fallback is not comparable to Romano's ping.** It might seem that both use a fixed latency value when there is no traffic, but the difference is substantial:

| | Romano (`ping_rtt_mean_seconds`) | DMOS fallback (`baseline_latency_ms`) |
|---|---|---|
| **Type** | Real-time measurement | Manually configured static value |
| **Per-cluster** | Yes — different values for each pair of clusters | No — identical for all clusters (`2.0ms` for cluster1, cluster2, cluster3) |
| **Differentiation** | Provides information on the actual network topology | DMOS reintroduces differentiation through `Φ_net` (ping_exporter) inside the multiplicative term `Φ_locality = Φ_demand · Φ_net`, in addition to `Φ_resp` from Hubble L7 |
| **Effect on scheduling** | The cluster with lowest ping receives more replicas | Inter-cluster RTT enters via `Φ_net`; in cold start (`cold_start` profile, first 120 s) `ω_geo_load=0.35` with `Φ_demand≈1` makes Φ_net dominant, replicating Romano's behavior |

<!-- TODO(review): possibly stale — cold_start profile weights (ω_geo=0.35, ω_carbon=0.30, ω_demand=0.35) referenced here may not match the active config in config/weights.yaml -->

In practice, when DMOS uses the fallback (absence of HTTP traffic), the `Φ_resp` component produces the same value for all three clusters. The problem is solved by the **`cold_start` profile** (first 120 s of execution) which zeroes `ω_resp` and `ω_cap` and redistributes the weights over `ω_geo` (0.35), `ω_carbon` (0.30) and `ω_demand` (0.35). With `Φ_load ≈ 1` at cold start, the multiplicative term `Φ_load·Φ_net` effectively collapses onto `Φ_net`, replicating exactly Romano's topology signal.

This is an **open limitation** of DMOS in the lab infrastructure: the integration of ping as a differentiated fallback (or as a fourth source in the graceful degradation chain) would make the system more robust during cold start phases.

**Implementation in DMOS** (`prometheus_client.py:get_latency_p95()`):
```promql
histogram_quantile(0.95,
  sum(rate(hubble_http_request_duration_seconds_bucket{
    destination_namespace="online-boutique"
  }[1m])) by (le)
) * 1000
```
The obtained p95 is converted into: `mean = p95 / 1.65`, `variance = (p95 - mean)²` — both used in the formula `Φ_resp(i)`.

---

#### O7 — Evolution of the Hubble traffic filter: from `source="reserved:ingress"` to `destination_workload`

**How Romano uses Hubble for traffic:** Romano queries `hubble_http_requests_total` with the filter `source="reserved:ingress"` — counting requests entering the cluster through its own ingress controller. This is a measure of the **incoming traffic to the cluster**, which reflects how many requests Cilium has routed to that cluster.

**How DMOS uses Hubble:** DMOS queries the same metric with a different filter — `destination_workload="frontend"` — which counts the HTTP requests that actually reach the **frontend microservice pods**, regardless of where they originate. The query:

```promql
sum(rate(hubble_http_requests_total{
    destination_workload="frontend",
    destination_namespace="online-boutique"
}[1m]))
```

**Why this filter is more precise:** Romano's filter `source="reserved:ingress"` measures traffic at the moment of entry into the cluster, but does not capture sub-requests internally generated by the microservice chain (e.g. frontend → productcatalog, frontend → recommendation). The filter `destination_workload="frontend"` measures the actual application traffic that the frontend microservice must process, including all requests directed to the pod (HTTP sub-requests, static assets, API calls). This makes the metric more representative of the **actual load** on the service.

**Note on the systematic multiplier:** the difference in filter also explains why Hubble/DMOS counts ~1.43× more requests compared to Locust: Locust counts user transactions (1 GET per task), while Hubble with `destination_workload="frontend"` counts every single HTTP request to the pod, including sub-requests generated by each loaded page.

---

#### O8 — Backend co-location: scaling proportional to frontend distribution

**Problem it solves:** Romano scales only the `frontend` deployment — backends remain static. With Cilium Global Services, if the frontend on cluster1 calls cartservice but on cluster1 there are no replicas of cartservice, the gRPC request is forwarded to a backend on another cluster (cross-cluster call), adding intra-cluster→inter-cluster network latency to every single internal call within the microservice chain.

**Approach adopted:** after each `frontend` scheduling cycle, `DMOSOrchestrator._enforce_colocation()` (`dmos_main.py:293`) distributes the backends proportionally to the current frontend distribution:

```
1. Read the current frontend distribution per cluster:
   cluster_parent_reps = {c: k8s.get_deployment_replicas("frontend", c)}
   total_parent_reps   = sum(cluster_parent_reps.values())

2. For each dependent backend (cartservice, productcatalogservice,
   checkoutservice, recommendationservice):

   frontend_share[c]       = cluster_parent_reps[c] / total_parent_reps
   proportional_target[c]  = max(min_replicas,
                                 round(total_backend_reps × frontend_share[c]))

   if current_backend[c] < proportional_target[c]:
       k8s.scale_deployment(backend, cluster=c, replicas=proportional_target[c])
```

Co-location scales **only up** — it never scales down a backend that already has more replicas than the proportional target. Backend scale-down is handled separately by a *backend reset* logic in the `periodic_check_thread` (`dmos_main.py:755`):

- If `frontend_total ≤ min_replicas` AND `backend_total > min_replicas × N_clusters` → immediate reset to minimum per cluster
- If `backend_total > frontend_total × 2` → gradual scale-down via event queue

**Why co-location is correct with Cilium Global Services:** Cilium does not expose a per-cluster endpoint to microservices — it exposes a global ClusterIP and balances connections across globally available pods. The probability that a call `frontend@cluster1 → cartservice` goes to `cartservice@cluster1` is proportional to the fraction `pods_cluster1 / pods_total`. With co-location active, this fraction follows that of the frontend → most internal calls remain local to the cluster.

**Configured dependent services** (`dmos_main.py:188`):
```python
service_dependencies = {
    'frontend': [
        'cartservice',
        'productcatalogservice',
        'checkoutservice',
        'recommendationservice',
    ]
}
```

---

### 6. Two-level architecture

DMOS introduces an explicit separation between two responsibilities that in Romano's baseline were indistinct:

```
┌─────────────────────────────────────────────────────────────┐
│                        DMOS                                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LEVEL 2 — Global scaling                            │   │
│  │  TrafficPredictor + PDController + ReplicaScaler     │   │
│  │  → "How many total replicas are needed?"             │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │ total_replicas                 │
│  ┌──────────────────────────▼───────────────────────────┐   │
│  │  LEVEL 1 — Distribution among clusters               │   │
│  │  DMOSScheduler + ScoreFunctions + WinnerDetermination│   │
│  │  → "How to distribute replicas among clusters?"      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Level 2** (new with respect to Romano): determines the total number of replicas needed for the entire multi-cluster system. Operates on aggregated traffic (sum over all clusters) and produces an integer `total_replicas`.

**Level 1** (evolution of Romano): given `total_replicas`, distributes these replicas among the clusters based on the multi-objective score. Operates separately per-cluster and produces the assignments `{cluster_i: n_i replicas}`.

The separation allows the scaling algorithm (Level 2) to be modified independently of the distribution algorithm (Level 1), and vice versa.

---

### 7. Comparative table: Romano vs DMOS

| Dimension | Romano (baseline) | DMOS (current state) |
|-----------|-------------------|---------------------|
| **Total scaling** | Reactive linear: `MIN + (MAX-MIN) × traffic/threshold` | Proactive: EMA + derivative prediction + PD controller |
| **Cluster score** | Two-dimensional: local traffic + latency × remote traffic | 5-dimensional: Φ_resp + Φ_cap + Φ_locality (Φ_demand·Φ_net) + Φ_carbon + Φ_pressure |
| **Carbon-awareness** | Absent | Φ_carbon = exp(-ν·CI/CI_max) |
| **Hardware constraints** | Not verified | Hard limits on CPU/RAM before the score |
| **Traffic metric** | `hubble_http_requests_total{source="reserved:ingress"}` | `hubble_http_requests_total{destination_workload="frontend"}` |
| **Latency metric** | `ping_rtt_mean_seconds` (inter-cluster RTT, ICMP L3) | `hubble_http_request_duration_seconds_bucket` (intra-cluster p95, HTTP L7) |
| **Traffic prediction** | None — reactive to current traffic | Trend-based (EMA + derivative), 120s horizon |
| **Anti-flapping** | Absent | Dead zone + cooldown + scale-up protection |
| **Fairness** | Not measured | Jain Fairness Index |
| **Vickrey auction** | Absent | Implemented (for future extensions) |
| **Load generator** | K6 | Locust (programmable scenarios) |
| **Prometheus pattern** | PROM_MAP (per-cluster) | PROM_MAP (inherited, same pattern) |
| **Test application** | Online Boutique | Online Boutique (same deployment) |
| **Backend scaling** | Only `frontend` scaled — static backends | Proportional co-location: 4 backends scaled as a function of frontend distribution |
| **Namespace** | `default` | `online-boutique` |
| **Infrastructure** | K3s + Cilium + 3 machines | K3s + Cilium + 3 machines (same) |

---

### 8. Open limitations in DMOS (current state)

Despite the extensions with respect to Romano, DMOS in its current version presents limitations that define the boundaries of the present work:

| Limitation | Description | Impact |
|------------|-------------|---------|
| **EMA tail at end of test** | The EMA predictor remains elevated for some minutes after traffic stops | Slight over-provisioning during cooldown |
| **Prediction without ML** | The predictor is linear trend-based, does not learn periodic patterns (e.g. daily peaks) | Higher MAPE on non-linear traffic |
| **No workload-aware for backends** | Backend services (cartservice, productcatalog, etc.) use network bytes as a proxy for gRPC load | Accuracy ±20-30% for backends |
| **Lab environment** | 3 LAN machines, single-node per cluster | Results not directly transferable to real cloud infrastructures |
| **Static carbon intensity** | CI per cluster is configured manually, not updated in real time | The environmental benefit is approximated |
| **Φ_resp neutral at cold start** | The `baseline_ms` fallback being identical for all cancels the discrimination by response time in the absence of HTTP traffic | Resolved at scheduler level by the `cold_start` profile (ω_resp=ω_cap=0) active in the first 120 s, where the decision is based on Φ_geo (Φ_load·Φ_net), Φ_carbon and Φ_demand |

---

### 9. Future developments

From the analysis of the limitations, concrete extension directions emerge, some directly connected to Romano's baseline:

#### SF1 — Geo-awareness via Φ_net (implemented)

> **Status:** delivered in the current system. Originally identified as future development, it was subsequently realized as part of the thesis work and today constitutes the geographic term of the score.

Romano's `ping_rtt_mean_seconds` has been integrated into DMOS via `czerwonk/ping_exporter` (one per cluster, target = node IPs of the other clusters). The resulting score component is:

```
Φ_net(i) = exp(-ρ · RTT_avg(i) / RTT_max)         with ρ=3.0, RTT_max=500 ms
```

`Φ_net` does not appear, however, as a fifth separate additive term (the original "Option B" hypothesis): it has instead been **multiplicatively fused** with `Φ_load` under a single weight `ω_geo`, obtaining

```
score_i = … + ω_geo · ( Φ_load(i) · Φ_net(i) ) + …
```

This choice — more stringent than additive aggregation — reflects the constraint "good cluster ⇔ close *and* unloaded" and prevents a single high signal from compensating for the other being low. See `dmos-architecture.md` for the numerical detail and `geo-awareness-implementation.md` for the implementation history.

**Geographic simulation in the lab.** Since the nodes are on the same LAN (`<LAN_SUBNET>`, native RTT ~0.2 ms), the differentiation is introduced with `tc netem` on `ens18` (egress):
- cluster1 (Frankfurt): no delay
- cluster2 (Paris): `delay 150 ms 20 ms distribution normal`
- cluster3 (Warsaw): `delay 350 ms 20 ms distribution normal`

In a real WAN deployment, `tc netem` would not be necessary: the delays arise spontaneously from the topology.

#### SF2 — ML prediction for periodic traffic

Replace the linear trend-based TrafficPredictor with a model that learns periodic patterns (LSTM, Prophet). Romano himself suggests *"predictive autoscaling logic based on machine learning"* as future development (§6.2). The DMOS infrastructure is already ready: it would suffice to replace `TrafficPredictor.predict()` with an offline-trained model.

#### SF3 — Real-time carbon intensity

Integrate the Electricity Maps APIs to update `CI_i(t)` in real time instead of using static values from config. It would require an asynchronous polling thread in `CarbonClient` and would handle daily variations (e.g. FR nuclear greener at night, DE solar greener at noon).

#### SF4 — L7 metrics for gRPC backends

Extend the L7 CNP to backends as well (cartservice, productcatalog, etc.) to obtain `hubble_http_requests_total` also for gRPC services. Currently DMOS uses `container_network_receive_bytes_total / 4000` as an empirical estimate (±20-30%). With L7 CNP on the backends, the measurement would become exact.

---

### 10. Scientific positioning

DMOS is placed in the line of research on **proactive autoscaling** for multi-cluster environments, integrating:

1. **Multi-objective orchestration** (scheduling): extends Romano's work by adding capacity, load and carbon to the score function, approaching the approaches of Chiaro et al. [5] on latency-aware scheduling in the Cloud-Edge Continuum.

2. **Predictive autoscaling** (scaling): implements a PD controller with trend-based prediction, a category documented in the literature as *proactive scaling* or *predictive HPA*, an alternative to native Kubernetes HPA (which is purely reactive and based on CPU/memory).

3. **Green computing**: the integration of `Φ_carbon` into the score function places DMOS in the category of **carbon-aware schedulers**, a growing field with the increase in environmental sensitivity of cloud providers (e.g. Google Cloud Carbon Footprint, Azure Sustainability).

4. **Cilium infrastructure**: the use of Hubble L7 to measure traffic is an original choice with respect to the literature, which typically uses native Kubernetes metrics (CPU, memory) or Istio. Cilium/Hubble does not require additional sidecars (it is part of the CNI) and is freely available in K3s environments.

---

*Document drafted on the basis of: G. Romano (2025), DMOS source code (src/), `double_wave_hubble` experiment (March 1, 2026).*
