# Backend gRPC Capacity Test — Documentation

> Manual for `experiments/run_capacity_grpc.sh` + `experiments/k6_capacity_grpc.js`:
> what it measures, how it works, how to read the result.
>
> **In one sentence**: it measures empirically how many requests per second
> **one replica** of each gRPC backend microservice (cartservice,
> productcatalogservice, recommendationservice, checkoutservice) can sustain,
> running k6 **inside the cluster**, to replace the rough guesses of
> `capacity_req_per_sec` in `config/services.yaml` with **measured** values.
>
> **Cross-reference**:
> - `experiments/run_capacity_grpc.sh` — the orchestrator
> - `experiments/k6_capacity_grpc.js` — the gRPC load client (runs in-cluster)
> - `experiments/demo.proto` — the Online Boutique proto (hipstershop)
> - `experiments/run_capacity.sh` — the HTTP equivalent for the **frontend**
> - `config/services.yaml` — where the measured values end up
> - `experiments/THROTTLE-CHECK.md` — the companion test on the CPU/cluster side
> - `docs/BACKEND-CO-SCALING.md` — how DMOS uses the backend capacities
>
> **Coverage:** the four backends above have measured capacities. DMOS also
> co-scales `currencyservice`, `adservice` and `shippingservice`, whose
> capacities in `services.yaml` are still **estimates**: the script has no
> mapping for them.

---

## Contents

1. [What it is and why it is needed](#1-what-it-is-and-why-it-is-needed)
2. [Why a test separate from the HTTP one](#2-why-a-test-separate-from-the-http-one)
3. [Why IN-CLUSTER and not kubectl port-forward](#3-why-in-cluster-and-not-kubectl-port-forward)
4. [How the script works](#4-how-the-script-works)
5. [The isolation problem (per-backend caveats)](#5-the-isolation-problem-per-backend-caveats)
6. [Methodology: knee point and SLA](#6-methodology-knee-point-and-sla)
7. [Usage: commands and env variables](#7-usage-commands-and-env-variables)
8. [Example output and interpretation](#8-example-output-and-interpretation)
9. [How to update config/services.yaml](#9-how-to-update-configservicesyaml)
10. [gRPC capacity vs HTTP capacity vs throttle check](#10-grpc-capacity-vs-http-capacity-vs-throttle-check)
11. [Known limits and when to rerun it](#11-known-limits-and-when-to-rerun-it)

---

## 1. What it is and why it is needed

DMOS computes the capacity of a cluster for each service as:

```
cluster capacity = capacity_req_per_sec x max_replicas_per_cluster
```

For the **frontend**, `capacity_req_per_sec = 60` in `config/services.yaml` is
measured (`run_capacity.sh`); a per-cluster calibration in `config/local.yaml`
can override it. For the **4 gRPC backends** the values in
`config/services.yaml` were originally **rough guesses**:

| Backend | estimated `capacity_req_per_sec` | Origin |
|---|---|---|
| cartservice | 50 | guess ("1 gRPC replica easily handles 50+ rps") |
| productcatalogservice | 80 | guess ("read-only Go, 100+ rps easily") |
| recommendationservice | 40 | guess ("Python is slower, 30+ rps") |
| checkoutservice | 20 | guess ("slow but little traffic, 20+ rps") |

A guess is not publishable data: the backend capacities enter the co-scaling
ratio `r_b = max(0.3, f_b · μ / μ_b)` (see `docs/BACKEND-CO-SCALING.md`), so
sizing the backends on invented numbers would rest on nothing. This tool
replaces the 4 guesses with 4 **measurements**.

> Output of the script: one `capacity_req_per_sec: <N>` line per backend, to
> paste into `config/services.yaml`.

---

## 2. Why a test separate from the HTTP one

`run_capacity.sh` tests the **frontend** by sending **HTTP** requests to the
Cilium Ingress. It cannot test the backends because:

1. **The backends speak gRPC, not HTTP.** They only expose a gRPC Service on a
   ClusterIP. No HTTP Ingress reaches them.
2. **They are not exposed externally.** Only the frontend has an Ingress.

So a client that speaks gRPC with `demo.proto` is needed:
`k6_capacity_grpc.js` uses the built-in `k6/net/grpc` module.

---

## 3. Why IN-CLUSTER and not kubectl port-forward

> This section documents an important methodological choice.

The **first version** of the tool ran k6 on the operator's machine and
reached the backend through `kubectl port-forward`. **That was wrong.** A full
run on the 4 backends proved it with data:

**Evidence 1 — a fixed latency tax.** The `p50` (median) at the lowest load,
on all 4 backends:

| Backend | Minimum load | measured p50 | expected p50 (service) |
|---|---|---|---|
| cartservice | 25 rps | 47 ms | ~1-3 ms |
| productcatalogservice | 100 rps | 47 ms | ~3-8 ms |
| recommendationservice | 10 rps | 49 ms | ~10-30 ms |
| checkoutservice | 5 rps | 54 ms | variable |

Four services of different complexity, at almost zero load, **pinned at
~47-54 ms**: this is not the microservice, it is the round trip of `kubectl
port-forward` (local machine -> API server -> pod), a fixed tax on *every*
request. cartservice and productcatalog have an SLA of p95 = 50 ms: the
~47 ms floor ate it up -> `knee n/a` with the pod completely idle.

**Evidence 2 — port-forward collapses under load.** At high rates the k6 logs
were full of:

```
dial tcp 127.0.0.1:50555: connectex: No connection could be made
because the target machine actively refused it
```

`127.0.0.1:50555` is the **local** listener of the port-forward. When k6 ramps
up the rate it creates many VUs -> many new TCP connections -> `kubectl
port-forward` cannot keep up and refuses them. The observed "knees"
(recommendation at 150 rps, checkout at 50 rps) were the limit of the
**port-forward**, not of the microservice.

**Conclusion**: `kubectl port-forward` is not a valid tool for benchmarking a
microservice. The measurement must be taken **in-cluster**.

**The solution**: k6 runs as a **Job inside the cluster**, in the
`online-boutique` namespace, and hits the backend through its ClusterIP (e.g.
`cartservice:7070`). Advantages:

- **Pod-to-pod** gRPC: sub-millisecond latency, no fixed tax.
- No fragile listener: the cluster network handles thousands of connections.
- The traffic goes through the same **Cilium L7 proxy** as the real
  frontend->backend traffic (the `l7-visibility-*` `CiliumNetworkPolicy`
  objects parse those ports as HTTP). We measure the **production path**.
- It is also the method a reviewer expects.

> The `l7-visibility-backends` CNPs admit ingress `fromEndpoints: [{}]`, i.e.
> any pod **in the same namespace**. Since the k6 Job runs in
> `online-boutique`, it is allowed to call the backends without changing the
> policies.

---

## 4. How the script works

For each backend, `run_capacity_grpc.sh` runs, in order:

```
 0. Pre-flight: kubectl, jq, k6 script, demo.proto, target kubeconfig
 1. loadgenerator -> 0 on ALL clusters   (no background traffic)
 2. ConfigMap dmos-k6-cap-scripts  (k6_capacity_grpc.js + demo.proto)
 3. target backend -> 1 replica on the target cluster
 4. backend dependencies -> 4 replicas on the target cluster   (boost)
 5. wait for Ready: backend + dependencies (rollout status)
 6. create a k6 Job (grafana/k6 image) that mounts the ConfigMap on /scripts
    and hits <backend>:<port> via ClusterIP
 7. follow `kubectl logs -f` of the k6 pod  (clean output, appended lines)
 8. extract the summary JSON from the logs (between markers) -> results/
 9. delete the Job, restore the dependencies -> 1 replica
10. print the recommended capacity_req_per_sec (80% of the knee point)
```

With `--all` it repeats steps 3-10 for all 4 backends and prints a summary
table + the snippet ready for `services.yaml`.

### Backend -> gRPC method mapping

| Backend | gRPC port | Method tested |
|---|---|---|
| cartservice | 7070 | `hipstershop.CartService/GetCart` |
| productcatalogservice | 3550 | `hipstershop.ProductCatalogService/ListProducts` |
| recommendationservice | 8080 | `hipstershop.RecommendationService/ListRecommendations` |
| checkoutservice | 5050 | `hipstershop.CheckoutService/PlaceOrder` |

### The k6 Job

- Image: `grafana/k6:latest` (standard k6, built-in gRPC module).
  Can be overridden with `DMOS_K6_IMAGE` if an internal registry is needed.
- ConfigMap `dmos-k6-cap-scripts` mounted on `/scripts`: it contains
  `k6_capacity_grpc.js` and `demo.proto`. k6 loads the proto with
  `load(['.'], 'demo.proto')` (next to the script).
- k6 pod resources: requests `500m`/`512Mi`, limits `4`/`4Gi` (the load
  generator needs CPU). This is why the test is best run on a cluster with
  headroom: the default `ms-cluster-01` (HIGH tier, 8 vCPU).
- The machine-readable JSON is printed between the markers
  `===DMOS_CAPACITY_JSON_BEGIN===` / `===DMOS_CAPACITY_JSON_END===` on stdout
  and retrieved with `kubectl logs`.

### Why the dependencies are boosted (step 4)

If I measure recommendationservice but productcatalogservice (1 replica)
saturates first, I am measuring productcatalog. To avoid this, the backend's
dependencies are scaled to 4 replicas **on the target cluster**; with
`service.cilium.io/affinity: local` the backend calls the local replicas.
They go back to 1 at the end of the test.

### Why loadgenerator -> 0 (step 1)

If the loadgenerator is running, the frontend keeps calling the backends and
that traffic contaminates the measurement. The script turns it off on **all**
clusters and **leaves it at 0**: it must be turned back on by hand before the
traffic phase:

```bash
for c in 01 02 03 04 05 06; do
  kubectl --kubeconfig "${KUBECONFIG_DIR:-$HOME/dmos-iac}"/kubeconfig-ms-cluster-$c \
    -n online-boutique scale deploy loadgenerator --replicas=1
done
```

---

## 5. The isolation problem (per-backend caveats)

Not all backends can be isolated in the same way. What the test really
measures must be stated openly:

| Backend | Isolation | What it really measures |
|---|---|---|
| **cartservice** | Clean | cartservice + 1 GET on redis-cart (redis is very fast). A real measurement of the pod. |
| **productcatalogservice** | Clean | In-memory Go, no dependencies. A clean measurement. |
| **recommendationservice** | Partial | `ListRecommendations` makes 1 call to `productcatalog.ListProducts`. We boost productcatalog to 4 replicas; the latency of the hop still enters the measurement. |
| **checkoutservice** | Fan-out | `PlaceOrder` orchestrates 6 services (cart, catalog, currency, payment, shipping, email). All boosted to 4. Tested with random users and an **empty cart**. |

### Note on checkoutservice's empty cart

`k6_capacity_grpc.js` sends `PlaceOrder` with random `user_id`s, hence empty
carts. In the Online Boutique demo `PlaceOrder` **works anyway** (an order
with shipping only). Implications:

- **Advantage**: stateless test, no pre-seeding (and `PlaceOrder` empties the
  cart at the end of the flow anyway: pre-populating would not hold up at a
  high rate).
- **Limit**: with 0 items there are fewer calls to `GetProduct` and `Convert`
  (one per item). The measured fan-out is somewhat lighter than a real order.
- **Why it is acceptable**: checkoutservice's CPU cost is dominated by the
  orchestration (7 gRPC calls, serialisation, error handling), which is
  roughly constant and is captured. The value is a reasonable proxy.

---

## 6. Methodology: knee point and SLA

1. **Fixed-rate steps.** For each value of `DMOS_STEPS`, k6 generates traffic
   at *exactly* that rate with the `constant-arrival-rate` executor.
2. **Settle + Measure.** Each step has a `settle` phase (default 12 s,
   discarded) and a `measure` phase (default 60 s, recorded).
3. **Knee point** = the last step for which **all** of these hold:
   - `p95 < DMOS_SLA_P95_MS`
   - `p99 < 4 x DMOS_SLA_P95_MS` (`DMOS_P99_SLA_FACTOR`, default 4: a step
     with an exploding tail, e.g. from CFS throttling, is not a valid
     operating point)
   - `error_rate < DMOS_SLA_ERR_RATE` (default 2%)
   The analysis stops after 2 consecutive steps that violate the SLA.
4. **Recommended capacity = 80% of the knee** (safety margin).
   `services.yaml` gets `round(knee_rps x 0.80)`.

### Default SLA and steps (per backend)

In-cluster the latency is low (~1-30 ms), so the absolute SLAs have ample
margin. All values can be **overridden** via env.

| Backend | Default p95 SLA | Default steps (rps) |
|---|---|---|
| cartservice | 50 ms | 300,400,500,600,700,800,1000,1300 |
| productcatalogservice | 50 ms | 500,1000,1500,2000,2500,3000,4000,5000 |
| recommendationservice | 120 ms | 100,110,120,130,140,150,170,200 |
| checkoutservice | 500 ms | 100,150,200,250,300,350,450,600 |

If the test ends with "knee N/A" (SLA already violated at the first step) or
never reaches saturation (all steps OK), rerun it with suitable `DMOS_STEPS`.

---

## 7. Usage: commands and env variables

Run from **Git Bash** on Windows (or any bash), from the repository root.

```bash
# A single backend (default: cartservice on ms-cluster-01)
bash experiments/run_capacity_grpc.sh

# A specific backend
bash experiments/run_capacity_grpc.sh --backend productcatalogservice
bash experiments/run_capacity_grpc.sh --backend recommendationservice
bash experiments/run_capacity_grpc.sh --backend checkoutservice

# All 4 in sequence (recommended for a full calibration)
bash experiments/run_capacity_grpc.sh --all

# On a different cluster
bash experiments/run_capacity_grpc.sh --backend cartservice --cluster ms-cluster-04

# Help
bash experiments/run_capacity_grpc.sh --help
```

### Env variables (optional overrides)

| Variable | Default | Effect |
|---|---|---|
| `DMOS_BACKEND` | cartservice | target backend (same as `--backend`) |
| `DMOS_CLUSTER` | ms-cluster-01 | target cluster (same as `--cluster`) |
| `DMOS_STEPS` | per backend | list of step rates, e.g. `"50,100,150,200"` |
| `DMOS_STEP_DURATION_S` | 60 | duration of the measure phase per step |
| `DMOS_SETTLE_S` | 12 | duration of the settle phase per step |
| `DMOS_SLA_P95_MS` | per backend | p95 SLA for the knee |
| `DMOS_SLA_ERR_RATE` | 0.02 | error-rate SLA for the knee |
| `DMOS_K6_IMAGE` | grafana/k6:latest | image of the k6 Job |
| `DMOS_NO_RESTORE` | 0 | `1` = do not restore the dependencies (debug) |
| `KUBECONFIG_DIR` | from `.cache/test-env.json`, else `$HOME/dmos-iac` | directory of the kubeconfigs |

Example — recommendationservice with a custom range and a looser SLA:

```bash
DMOS_STEPS="20,40,60,80,100,120" DMOS_SLA_P95_MS=150 \
  bash experiments/run_capacity_grpc.sh --backend recommendationservice
```

### Duration

~9-10 min per backend (8 steps x 72 s), plus the first pull of the k6 image.
`--all` takes ~40 min.

### netem makes no difference

`tc netem` emulates geographic latencies on **inter-cluster** traffic. The
k6->backend traffic is **intra-cluster** (k6 pod and backend on the same
cluster): netem does not touch it. netem can be left on or off.

---

## 8. Example output and interpretation

```
================================================================================
  CAPACITY TEST gRPC — cartservice
  Metodo: hipstershop.CartService/GetCart
  Target: cartservice:7070
  SLA: p95 < 50ms, err < 2.0%
================================================================================

  Step | rps tgt | rps obt | rps/rep |   p50  |   p95  |   p99  | Err%  | SLA
  ----------------------------------------------------------------------------
     1 |    25r/s |    25.0 |    25.0 |     2ms |     4ms |     7ms |  0.0% |  OK
     2 |    50r/s |    50.0 |    50.0 |     2ms |     5ms |     9ms |  0.0% |  OK
     3 |    75r/s |    75.0 |    75.0 |     3ms |     7ms |    12ms |  0.0% |  OK
     4 |   100r/s |   100.0 |   100.0 |     4ms |    11ms |    20ms |  0.0% |  OK
     5 |   150r/s |   150.0 |   150.0 |     7ms |    26ms |    48ms |  0.0% |  OK
     6 |   200r/s |   199.3 |   199.3 |    19ms |    71ms |   140ms |  0.3% | FAIL
     7 |   300r/s |   268.0 |   268.0 |    55ms |   210ms |   390ms |  2.8% | FAIL

  -> SLA violato per 2 step consecutivi: saturazione confermata

================================================================================
  Knee-point:             150.0 rps/replica  (a 150 rps target)
  Capacita raccomandata:  120 rps/replica  (80% del knee)
  -> config/services.yaml, sezione cartservice:
     capacity_req_per_sec: 120
================================================================================
```

How to read it:

- **Low, rising p50** (2 -> 19 ms): in-cluster there is no 47 ms port-forward
  tax; latency starts low and rises **gradually** towards the knee, the sign
  of **genuine** CPU saturation.
- **rps tgt vs rps obt**: as long as they match, the pod keeps up. When
  `rps obt < rps tgt` (step 7) the pod can no longer drain the rate ->
  saturated.
- **knee point = 150**: the last step that meets the SLA. **recommended = 120**.

With `--all`, the final table:

```
=== Riepilogo capacity backend gRPC ===
  Backend                  | knee (rps/rep)   | capacity_req_per_sec
  --------------------------------------------------------------------------
  cartservice              | 150.0            | 120
  productcatalogservice    | 400.0            | 320
  recommendationservice    | 220.0            | 176
  checkoutservice          | 60.0             | 48
```

> The numbers above are **illustrative**. The real values come out of running
> the test.

### Files produced

- `experiments/results/capacity_grpc_<backend>_<timestamp>.json` —
  machine-readable summary (extracted from the Job logs)
- `experiments/results/capacity_grpc_<backend>_<timestamp>.k6log.txt` — full
  log of the k6 Job

---

## 9. How to update config/services.yaml

At the end the script prints the ready-made snippet. For each backend, in
`config/services.yaml`, under `services: <backend>:`, replace the value:

```yaml
  cartservice:
    ...
    # Capacity per replica (req/s) — MEASURED with in-cluster k6 gRPC.
    # Test: experiments/run_capacity_grpc.sh --backend cartservice
    #   method hipstershop.CartService/GetCart, 1 isolated replica, k6 Job
    #   inside the cluster, knee = 150 rps/replica, recommended 80% = 120.
    capacity_req_per_sec: 120     # was 50 (guess)
```

### Checking the CPU budget after the update

`capacity_req_per_sec` must be reviewed together with `max_replicas`. The DMOS
model stays linear **only if** the worker has CPU for all the replicas without
throttling:

```
max_replicas x CPU_limit(backend) + other_pods  <=  CPU_worker
```

If the measured capacity is much higher than the guess, it may make sense to
**lower** `max_replicas` (fewer replicas for the same cluster capacity).

---

## 10. gRPC capacity vs HTTP capacity vs throttle check

| | `run_capacity.sh` (HTTP) | `run_capacity_grpc.sh` (gRPC) | `throttle_check.sh` |
|---|---|---|---|
| **Question** | how many rps can 1 frontend replica sustain? | how many rps can 1 backend replica sustain? | do N replicas scale linearly? |
| **Service** | frontend | the 4 gRPC backends | frontend (extensible) |
| **Where k6 runs** | on the operator's machine, via Ingress | **inside the cluster** (Job) | on the operator's machine, via Ingress |
| **Key metric** | client-side p95 latency | client-side p95 latency (in-cluster) | CFS throttled periods |
| **Calibrates** | frontend `capacity_req_per_sec` | backend `capacity_req_per_sec` | confirms `max_replicas` |
| **netem** | must be off (it skews the p95) | no difference (intra-cluster traffic) | no difference (measures CPU) |
| **Duration** | ~17 min | ~10 min/backend | ~4 min/cluster |

Together they validate the **two factors** of the DMOS capacity model:
`run_capacity*` measures `capacity_req_per_sec`, `throttle_check` validates
`max_replicas`. For the frontend, their product is the per-cluster capacity
used by Level 2; the backend capacities enter the co-scaling ratio.

---

## 11. Known limits and when to rerun it

### Known limits

- **recommendation/checkout are not perfectly isolated**: see sec. 5. The
  measurement includes the latency of the downstream hops (by design: it is
  what the pod also does in production).
- **checkout with an empty cart**: see sec. 5.
- **The k6 pod shares the cluster with the backend**: the load generator runs
  on the same cluster as the backend under test. On a cluster with headroom
  (HIGH tier, 8 vCPU) this is not a problem: k6 is limited to 4 vCPU and the
  backend has its requests guaranteed. This is why the default is
  `ms-cluster-01`.
- **Uniform capacity across tiers**: the backend pod has a fixed CPU limit
  (cartservice 300m, productcatalog 250m, checkout 500m, recommendation 500m),
  so `capacity_req_per_sec` does not depend on the worker's tier, by the same
  reasoning as for the frontend. Measuring **once**, on any cluster, is
  enough.
- **The Cilium L7 proxy is in the path**: the k6->backend traffic goes through
  Envoy (the `l7-visibility-*` CNPs parse those ports as HTTP). This is the
  real production path, so it is correct; at the test rates (hundreds of rps)
  Envoy is not a bottleneck.

### When to rerun it

- After every **re-provisioning** of the fleet.
- After changing a backend's **CPU limit** in `config/services.yaml`.
- After changing a microservice's **image version**.
- When the **representative gRPC method** being considered changes.

### Position in the pipeline

A **calibration** step, run once after re-provisioning, before the traffic
phase:

```
re-provision -> DMOS setup on the clusters -> verify-cilium -> netem + ping
   -> run_capacity.sh (frontend, netem OFF)
   -> run_capacity_grpc.sh --all (backends)        <-- THIS TOOL
   -> update config/services.yaml
   -> throttle_check.sh
   -> turn loadgenerator back on / traffic phase -> start DMOS
```
