# Backend Co-Scaling in DMOS

Technical documentation of how DMOS sizes the backend services. It describes
**what**, **why** and **how**: the exact algorithm, its properties, its
limits and how it relates to the literature. In compact form the rule is

```
n_{i,b} = clip( ceil(r_b · n_i), 1, max_b ),    r_b = max( 0.3, f_b · μ / μ_b )
```

where `n_i` is the number of frontend replicas on cluster *i*, `f_b` the calls
backend *b* receives per frontend request (measured with Hubble), `μ` and
`μ_b` the per-replica capacities of the frontend and of the backend, and
`max_b` the replica cap of the backend on that cluster. The overall picture is
in [`dmos-architecture.md` §9](dmos-architecture.md#9-backend-co-scaling).

---

## 1. Context and problem

Online Boutique is a microservice application: a user-facing **frontend**
service that, for every HTTP request, **fans out** gRPC calls to several
**backend** services (`cartservice`, `productcatalogservice`,
`checkoutservice`, `recommendationservice`, `currencyservice`, `adservice`,
`shippingservice`).

For every service and every cluster, DMOS must decide **how many replicas** to
allocate. The key question: **which signal should a backend be sized on?**

### 1.1 Why NOT on the backend's request count

The obvious approach is to measure the rate of requests reaching the backend
(`hubble_http_requests_total{destination_workload=<backend>}` via Cilium
Hubble) and size accordingly. **This signal is unreliable under saturation.**

Hubble counts **every HTTP/gRPC request on the wire**. When the system is in
distress, internal gRPC calls fail and are **retried** (gRPC / Envoy retries).
Every retry is a new request that Hubble counts, and **there is no label that
distinguishes a retry from an original request**. As a result the counter is
inflated by a factor of `fan-out × retry amplification`.

Real measurement (flash-crowd run, cooldown phase, offered external load =
50 req/s):

| Metric | Reading | Ground truth |
|---|---|---|
| `hubble_http_requests_total` (productcatalogservice) | ~1040 req/s | ~50 external req/s |
| Inflation factor | ~20× | — |

A scaler that trusts this number **scales on a lie**. Worse, scaling causes
Cilium/Envoy endpoint churn → more broken gRPC connections → more retries →
an even more inflated counter → a **positive feedback loop**. This is the
signature of a **metastable failure** (Bronson et al., HotOS 2021): a failure
that persists after the trigger is removed, sustained by the retries.

**Conclusion:** the volume of internal traffic is a **corrupted** demand
signal under saturation. It must be kept out of the control loop.

---

## 2. Design principle

Two rules:

1. **Measure demand at the edge, not inside the mesh.** The only demand
   signal immune to retries is the **external ingress rate**, the traffic
   entering through the Cilium Ingress, measured by `get_ingress_rate()`
   (client-side query at the source; server-side fallback with
   `source_workload=""`, see
   [`traffic-metrics-architecture.md`](traffic-metrics-architecture.md)). The
   load generator does not retry, and the internal fan-out does not go through
   the Ingress, so this count is the real demand.

2. **Propagate demand along the dependency graph.** A backend's load **is not
   an independent variable**: it is *caused* by the frontend's fan-out. The
   backend is therefore sized **by deriving it from the frontend**, not by
   measuring it. This is **dependency-aware autoscaling**.

Diagram:

```
   ingress rate (external, clean)
        │
        ▼
   budget N + L1 placement + L2 correction (frontend)
        │
        ▼
   frontend replicas per cluster                             (Kubernetes fact)
        │
        ▼  co-location (this document)
   N backend replicas per cluster   (cart / pcatalog / checkout / reco /
                                     currency / ad / shipping)
```

The corrupted Hubble count enters **neither** step: the frontend uses the
ingress rate, the backends use a Kubernetes fact (the number of frontend
replicas).

---

## 3. The co-scaling algorithm

Implemented in `_enforce_colocation()` (`src/dmos_main.py`), called at the end
of `schedule_service('frontend')` (on both the reactive and the predictive
path), i.e. at every scheduling cycle of the frontend (~30 s).

### 3.1 Dependency graph

The entry service and the co-scaled services, in the order in which they are
processed (they share the per-cycle churn budget), are read from
`config/services.yaml → application` (see
[configuration-reference.md §4.1](configuration-reference.md#41-application--the-application-dmos-manages)).
For Online Boutique:

```yaml
application:
  entry_service: frontend
  co_scaled_services: [cartservice, productcatalogservice, checkoutservice,
                       recommendationservice, currencyservice, adservice,
                       shippingservice]   # stateless on the hot path; redis-cart excluded (stateful)
```

DMOS derives `service_dependencies = {entry_service: co_scaled_services}` from it.

A star graph of depth 1: the `frontend` is the parent, the seven backends are
the dependents.

### 3.2 Formula

For every backend `dep` and every cluster `c`:

```
parent_reps(c)   = current frontend replicas on c             [k8s, fresh read]
nominal_ratio    = ( frontend_capacity(c) / backend_capacity(c) ) × fanout(dep)
effective_ratio  = max( nominal_ratio, r_floor )              [r_floor = 0.3]
target(dep,c)    = clamp( min_replicas(dep,c),
                          max_replicas(dep,c),
                          ceil( parent_reps(c) × effective_ratio ) )
current(dep,c)   = current backend replicas on c              [k8s, fresh read]
delta            = target(dep,c) − current(dep,c)
```

- `*_capacity` = `capacity_req_per_sec` from `config/services.yaml`, with
  per-cluster overrides (`get_capacity_for_cluster`). These are the values
  **measured** by the capacity tests. A per-cluster calibration in
  `config/local.yaml` (`capacity_req_per_sec: {frontend: r}`, see
  [KUBEPHOS.md](KUBEPHOS.md)) overrides the frontend value: on the reference
  fleet it is μ = 33 req/s instead of the 60 req/s of `services.yaml`.
- `fanout(dep)` = `calls_per_request` from `config/services.yaml`: how many
  calls the backend receives per external request, **measured with Hubble**.
  A service called N times per request receives N× the traffic and must be
  scaled N times more (e.g. `currencyservice` ~3.4×: without the fan-out term
  it would stay under-provisioned).
- `r_floor` = `backend_colocation_floor = 0.3` (`src/dmos_main.py`): a floor on
  the ratio. Isolated-replica benchmarks **overestimate** the effective
  capacity under gRPC fan-out (connection-pool exhaustion, tail-latency
  cascade; Dean & Barroso, "The Tail at Scale"); with very low nominal ratios
  (e.g. cart: 60/400 × 0.65 ≈ 0.10) the backends would stay at 1 replica even
  with the frontend scaled out → saturation cascade. The value was calibrated
  empirically: 0.5 over-provisioned at moderate load, 0.3 is the balance point.
- `clamp` honours each backend's `min_replicas` (1 for every backend) and
  `max_replicas_per_cluster` (tier-aware).
- The k8s reads are **fresh** (right before the decision) to avoid decisions
  based on stale cached values.

### 3.3 Bidirectional + churn budget

`delta` is applied in **both directions**: when the frontend grows the
backends grow, when the frontend shrinks the backends shrink. Co-location is
the **sole sizer** of the backends. `delta` is applied in a single step (the
+6/−1 rate limit applies to the frontend, not to co-scaling); co-scaling is
subject to neither the cooldown nor the dead zone.

Every operation honours the **per-cycle churn budgets** (anti-thundering-herd).
Co-scaling has **dedicated** budgets, separate from the frontend's reactive
ones (8 up / 2 down): without that separation the frontend ramp used up the
budget and starved co-scaling, leaving the backends at 1 replica.

| Budget | Value | Meaning |
|---|---|---|
| `backend_scale_up_budget_per_cycle` | 12 | max backend pods added per cycle (30 s), whole federation |
| `backend_scale_down_budget_per_cycle` | 6 | max backend pods removed per cycle, whole federation |

If the budget is used up, the operation is **deferred** to the next cycle
(logged as `deferred — backend budget esaurito`). Scaling is thus spread over
several cycles, giving the Cilium/Envoy data plane time to settle.

### 3.4 Numerical example

Per-replica capacity and fan-out (from `services.yaml`; frontend = 60 req/s):

| Backend | capacity (req/s) | fanout | nominal ratio = 60/cap × fanout | effective ratio = max(nominal, 0.3) |
|---|---|---|---|---|
| cartservice | 400 | 0.65 | 0.098 | 0.30 (floor) |
| productcatalogservice | 1200 | 2.25 | 0.113 | 0.30 (floor) |
| checkoutservice | 160 | 0.05 | 0.019 | 0.30 (floor) |
| recommendationservice | 40 | 0.40 | 0.60 | 0.60 (capacity) |
| currencyservice | 250 | 3.4 | 0.816 | 0.82 (capacity) |
| adservice | 400 | 0.43 | 0.065 | 0.30 (floor) |
| shippingservice | 500 | 0.14 | 0.017 | 0.30 (floor) |

If at the peak the frontend has **6 replicas** on `ms-cluster-01` (HIGH tier
cap):

| Backend | `ceil(6 × effective_ratio)` | target (after clamp) |
|---|---|---|
| cartservice | ceil(1.80) = 2 | 2 (max 2) |
| productcatalogservice | ceil(1.80) = 2 | 2 |
| checkoutservice | ceil(1.80) = 2 | 2 (max 2) |
| recommendationservice | ceil(3.60) = 4 | 4 |
| currencyservice | ceil(4.90) = 5 | 5 |
| adservice | ceil(1.80) = 2 | 2 |
| shippingservice | ceil(1.80) = 2 | 2 (max 2) |

"Weak" backends (low effective capacity per replica: recommendation) and
high-fan-out ones (currency, ~3.4 calls per request) scale on the
capacity × fan-out term and get more replicas; for the others the 0.3 floor
governs, which still makes them grow with the frontend instead of leaving
them at 1 replica.

With the calibrated μ = 33 req/s of the reference fleet the nominal ratios
are lower: only recommendationservice (33/40 × 0.40 = 0.33) and
currencyservice (33/250 × 3.4 = 0.45) stay above the floor, and with 6
frontends they get ceil(1.98) = 2 and ceil(2.69) = 3 replicas; every other
backend runs on the floor (2 replicas).

---

## 4. Per cluster: why it matters

The co-scaling ratio is computed **per cluster**, not globally, for two
reasons:

1. **Heterogeneous fleet.** The clusters are on three hardware tiers (HIGH
   8 vCPU, MEDIUM 6 vCPU, LOW 4 vCPU). `get_capacity_for_cluster` and
   `get_max_replicas_for_cluster` can reflect the tier: in `services.yaml`
   today only the caps (`max_replicas_per_cluster`) vary by tier, while the
   per-replica capacities are the same on every cluster.

2. **gRPC locality.** Co-locating the backend **on the same cluster** as the
   frontend that calls it keeps the gRPC fan-out **intra-cluster**. A backend
   placed elsewhere would force every call through an inter-cluster hop (tens
   to hundreds of ms via ClusterMesh). Per-cluster co-location preserves
   locality: wherever there is a frontend, its backends are there too. The
   frontend itself is a global service with `affinity: none`, so a request
   may be served by a frontend on another cluster than the one it entered;
   that frontend then calls the backends of its own cluster (the backends are
   local or `affinity: local`, see
   [CILIUM-CLUSTERMESH-NOTES.md](CILIUM-CLUSTERMESH-NOTES.md)). This is why the
   backends follow the frontend replicas of their cluster rather than the
   demand entering it.

---

## 5. Properties

| Property | How it is guaranteed |
|---|---|
| **Immune to retry corruption** | Uses only Kubernetes facts (frontend replicas) + measured capacities. No corruptible volume metric in the loop. |
| **gRPC locality preserved** | Per-cluster co-scaling: the backends live where the calling frontend lives. |
| **Convergence** | Bidirectional: the backends track the frontend up and down; no drift is possible. |
| **Bounded churn** | Every operation honours the per-cycle budgets; scaling is spread over time. |
| **Consistent with L1** | In the L1+L2 arm, Level 1 places the frontend geo-aware; the backends inherit that distribution → they stay close to the caller. |

---

## 6. Implementation

Changes in `src/dmos_main.py`:

1. **`_get_per_cluster_traffic`**: for the `frontend` it uses
   `get_ingress_rate()` instead of `get_request_rate()` (clean demand source).
2. **`monitored_services = [entry_service]`** (the `frontend`): only the entry
   service is scheduled on traffic. The per-backend L2 scaler (which read the
   corrupted Hubble count) **is no longer invoked**.
3. **`_enforce_colocation`**: rewritten as the sole, bidirectional,
   per-cluster sizer (this document).
4. **Removed** the *Backend reset* and *gradual backend scale-down* blocks
   from `periodic_check_thread`: with co-location as the only bidirectional
   sizer the backends cannot drift, so those safety nets became obsolete.

### Interaction with Level 1 and Level 2

| Arm | Frontend | Backend |
|---|---|---|
| `off` | 1 fixed replica (DMOS off) | 1 fixed replica |
| `l2` (`--no-scheduling`) | budget N from the ingress rate, uniform distribution N/K, L2 correction | co-location |
| `l1l2` (an earlier campaign), the score models of `config/weights.yaml`, `profile-*`, `legacy-*` (names from `experiments/run_naming.py`) | L1 (score) places the budget N; L2 corrects | co-location |

In every arm with DMOS, co-location is invoked identically at the end of
`schedule_service('frontend')`. The **Level 1 score function is not
modified**: it keeps deciding the geo-distribution. It decides it for the
frontend, and the backends inherit it.

**Experimental alternative.** With `--naive-backends`, co-location is replaced
by reactive scaling of the backends on their observed L7 rate
(`ceil(rps / 50)`, frontend budgets): it is the "metastable" baseline used to
show the problem of §1; no current harness uses it.

**Observability.** The collector (`dmos_live.jsonl`) records replicas and CPU
of cartservice, productcatalogservice, checkoutservice and
recommendationservice, but not of currencyservice, adservice and
shippingservice; their decisions can be read in `dmos.log` (lines
`Co-location: <cluster> — <service>: a → b`).

---

## 7. Limits and assumptions

To be explicit: co-location is a **feed-forward** (open-loop) control:

- **It assumes a stable fan-out.** The fixed ratio (constant capacities and
  `calls_per_request`) is correct as long as the ratio of frontend calls to
  backend calls does not change. This holds while the endpoint mix of the
  workload is fixed (k6: 40 % `/`, 25 % product pages, 25 % cart, 5 %
  currency change, 5 % checkout); with a mix that changes over time the
  fan-out would vary.
- **Depth-1 graph.** It models `frontend → backend`. It does not capture
  `backend → backend` dependencies (marginal in Online Boutique).
- **No feedback from the actual backend state.** If the capacity model is
  inaccurate for a backend, nothing corrects it afterwards.

Possible mitigation/extension: a **closed-loop CPU adjustment** of the backend
(a bounded signal in [0, 100 %], which cannot be inflated like the request
count) on top of the feed-forward co-location floor.

---

## 8. Relation to the literature

Co-location is the **white-box, interpretable** version of an established
approach:

- **Dependency-aware / call-graph-aware autoscaling** is a mainstream
  direction (survey 2025, arXiv:2507.17128): incorporating the analysis of the
  invocation path into scaling decisions enables coordinated actions and
  prevents failure propagation.
- **Propagating demand from the ingress** along the graph to size the
  components is a known method (scaling driven by user traffic with
  trace-level SLOs).
- **Learned** versions replace the fixed ratio with a model: Graph-PHPA (LSTM
  for the workload + GNN for the workload→resources relation), STaleX
  (spatiotemporal-aware). Useful for heterogeneous workloads or deep graphs,
  at the cost of interpretability and a training burden.
- The problem that co-location **avoids**, a scaler feeding a feedback loop
  through retry-inflated metrics, is a **metastable failure** (Bronson et
  al., HotOS 2021).

**Summary:** DMOS adopts an interpretable feed-forward propagation, anchored
to measured capacities, suited to the homogeneous workload of the
experiments; a learned graph-based model is the natural extension, left as
future work.

### References

- N. Bronson et al., *Metastable Failures in Distributed Systems*, HotOS 2021.
- *Auto-scaling Approaches for Microservice Applications: A Survey and
  Taxonomy*, arXiv:2507.17128, 2025.
- *Graph-PHPA: Graph-based Proactive Horizontal Pod Autoscaling using
  LSTM-GNN*, arXiv:2209.02551.
- *STaleX: A Spatiotemporal-Aware Adaptive Auto-scaling Framework for
  Microservices*, arXiv:2501.18734.
