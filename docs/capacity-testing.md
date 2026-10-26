# Capacity Testing — Determining `lambda_max` for DMOS

How to empirically measure the per-replica saturation request rate
(`lambda_max`) for each Online Boutique microservice, and how that value
feeds into the DMOS load-pressure signal.

> Status: this document consolidates `capacity-test.md` (frontend, HTTP)
> and `backend-capacity-test.md` (gRPC backends) into a single English
> reference. See "Migration / history" at the end of this file.

---

## Table of contents

1. [Goal: what `lambda_max` is and why it matters](#1-goal)
2. [How DMOS uses the capacity](#2-phi-load)
3. [Experimental setup](#3-setup)
4. [The three measurement layers (HTTP, observer, gRPC)](#4-three-layers)
5. [Frontend procedure (k6, HTTP)](#5-frontend)
6. [Backend procedure (k6 gRPC Job in-cluster, per service)](#6-backend)
7. [System-level observation (Hubble L7 + Prometheus)](#7-observer)
8. [Data analysis: knee-finding and saturation criterion](#8-analysis)
9. [Writing results into `config/services.yaml`](#9-config)
10. [Caveats and known limitations](#10-caveats)
11. [Testbed-specific resource tuning (prerequisite)](#11-testbed)
12. [Migration / history](#12-history)

---

## 1. Goal: what `lambda_max` is and why it matters <a name="1-goal"></a>

**`lambda_max`** is the **empirical saturation request rate per replica**
of a microservice — the maximum sustained requests-per-second that a
single pod can serve while keeping p95 latency and error rate within an
acceptable SLA band. It is determined experimentally with stepped load
against an isolated single-replica deployment. In `config/services.yaml`
the stored value is called **`capacity_req_per_sec`** (per-cluster override
`capacity_per_cluster`) and is **80 % of the measured knee** (§8.3).

Getting it wrong distorts every sizing decision:

| capacity too **low** | capacity too **high** |
|---|---|
| more replicas than needed → over-provisioning, CPU contention on the node | too few replicas → saturation, latency and errors |

---

## 2. How DMOS uses the capacity <a name="2-phi-load"></a>

| Use | Formula | Code |
|---|---|---|
| Frontend replica budget N (paper Eq. 7) and per-cluster need | `needᵢ = clip(⌈λᵢⁱⁿ / c_frontend⌉, c_min, c_max)`, `N = Σ needᵢ` | `DMOSOrchestrator.schedule_service` |
| Backend co-scaling ratio (Eq. 11) | `max((c_frontend / c_backend) · fanout, 0.3)` | `_enforce_colocation` |
| `request_rate_max` of a cluster | `cpu_cores × c_frontend` (e.g. 8 × 60 = 480 req/s) | `_collect_cluster_metrics` |
| Utilization pressure Φ_pressure (legacy score path only) | `1 / (1 + exp(−10 · (λᵢ / request_rate_max − 0.7)))` | `ScoreFunctions.compute_load_score` |

With the score models (e.g. `nca`) Φ_pressure is not part of the score: the
capacity matters through the budget and the co-scaling. Since
`needᵢ = ⌈λᵢⁱⁿ / c⌉`, each frontend replica is planned to receive at most
`c` = 80 % of its knee, the margin that absorbs measurement noise and the
burstiness of real traffic. See
[dmos-architecture.md §6.2 and §9](dmos-architecture.md#62-one-cycle-dmosorchestratorschedule_service).

---

## 3. Experimental setup <a name="3-setup"></a>

### 3.1 Single-replica isolation

Capacity is a per-pod property. Multiple replicas mask saturation behind
load balancing, so the test deployment **must** run with exactly one
replica per cluster of the service under test. All other DMOS-managed
services are also pinned to one replica each, so the call graph is
deterministic.

```powershell
# Pin every DMOS-managed service to one replica per cluster
# (kubeconfig files: <KUBECONFIG_DIR>/kubeconfig-ms-cluster-NN)
$services = @("frontend","cartservice","productcatalogservice",
              "checkoutservice","recommendationservice",
              "currencyservice","adservice","shippingservice")
$clusters = @("ms-cluster-01","ms-cluster-02","ms-cluster-03",
              "ms-cluster-04","ms-cluster-05","ms-cluster-06")

foreach ($cluster in $clusters) {
    foreach ($svc in $services) {
        kubectl scale deployment $svc -n online-boutique `
                --replicas=1 --kubeconfig "$env:KUBECONFIG_DIR\kubeconfig-$cluster"
    }
}
```

The services DMOS does not manage (`redis-cart`, `paymentservice`,
`emailservice`) run at one replica from the manifest.

### 3.2 Disabling background traffic

The bundled Online Boutique `loadgenerator` and any DMOS instance must
be stopped during a capacity test, otherwise their traffic is summed
into the rate observed at the service under test:

```powershell
# 1. stop DMOS in its terminal (Ctrl+C)
# 2. scale loadgenerator to zero on every cluster
foreach ($cluster in $clusters) {
    kubectl --kubeconfig "$env:KUBECONFIG_DIR\kubeconfig-$cluster" `
        scale deployment loadgenerator -n online-boutique --replicas=0
}
# 3. wait ~60 s for the system to settle
```

### 3.3 Stepped load (staircase)

Three families of load patterns exist; only one is appropriate for
capacity calibration:

* **Spike test** (instant burst): no time for the system to warm up;
  measures *worst-case* capacity (cold connection pools, JIT not run,
  cold caches). Useful for separate flash-crowd experiments, not for
  `lambda_max`.
* **Ramp test** (continuous ramp): the system is in permanent
  transition; hard to attribute metrics to a specific load level.
* **Step test (staircase)**: each load level is held long enough to
  reach steady state, then stepped up. This is what the capacity test
  uses.

```
Users
 300 |                                          ____
 280 |                                     ____/
 260 |                                ____/
 240 |                           ____/
  ...
  60 |               ____/
  40 |          ____/
  20 | _________/        <- instant spawn at 10 user/s
     +--+--+--+--+--+--+--+--+--+--+--+--+
        S1 S2 S3 S4 S5  ...   (each step = 90 s)
```

For each step:

1. **Spawn** — the load generator adds N more virtual users at a fixed
   spawn rate (typically a couple of seconds of transition).
2. **Warm-up** — the first ~30 s of the step are noisy: HTTP/gRPC
   connection pools resize, GC fires, JIT warms up.
3. **Steady state** — the last ~60 s are what the analyzer aggregates.
4. **Decision** — at end of step, p95 / fail-rate are checked against
   the SLA; if violated, the previous step is the knee point and the
   test stops.

90 s per step is the recommended floor; shorter steps produce noisy
metrics because connection pools have not settled.

### 3.4 Tools used

| Tool | Used for | Source path |
|---|---|---|
| **k6 (HTTP)** | Frontend HTTP capacity (stepped arrival-rate) | `experiments/k6_capacity_test.js` (+ `experiments/run_capacity.sh`) |
| **k6 (gRPC, in-cluster Job)** | Per-service gRPC capacity (one backend at a time) | `experiments/k6_capacity_grpc.js` + `experiments/run_capacity_grpc.sh` + `experiments/demo.proto` |
| `backend_capacity_observer.py` | System-level Prometheus/Hubble observer (parallel to k6; from the ghz era, see §7) | `experiments/backend_capacity_observer.py` |

DMOS uses **k6** for both HTTP and gRPC load. The gRPC client runs as a
**Job inside the cluster** (image `grafana/k6`), so no local gRPC tool
is needed; the historical `ghz`-over-port-forward method is superseded
(see Section 6 and `experiments/CAPACITY-GRPC.md`). Locust is present in
the repo only for the historical multi-ingress workload; it is **not**
used for capacity calibration.

To install (frontend test only — the gRPC Job pulls its own image):

```powershell
# k6 — https://k6.io/docs/get-started/installation/
choco install k6                       # Windows (Chocolatey)
brew install k6                        # macOS
sudo apt install k6                    # Debian/Ubuntu via k6 apt repo
```

---

## 4. The three measurement layers (HTTP, observer, gRPC) <a name="4-three-layers"></a>

Capacity can be observed at three levels of detail, each answering a
different question:

### Level 1 — Client-side HTTP (k6)

```
k6 ---> HTTP ---> frontend ---> [hidden backend chain]
                  measure here: rps, p95, error rate
```

Sees end-to-end HTTP latency and rps from the *client* perspective.
Does **not** see which downstream service is responsible for slowdowns.
Produces `lambda_max` for the frontend only.

### Level 2 — System-level observer (Hubble L7 + Prometheus)

```
k6 ---> HTTP ---> frontend ---> cartservice ---> redis
                      |                        ^
                      v                        |
                  productcatalog          Prometheus + Hubble
                                          measure each service
```

Runs a normal frontend load test, but in parallel scrapes Hubble L7
metrics for *every* service. Identifies which service saturates first
under realistic call-graph traffic. Cannot impose a controlled rate on
any single backend.

### Level 3 — Per-service gRPC isolation (k6 Job in-cluster)

```
k6 Job (in-cluster) ---> gRPC ---> cartservice ---> redis
                         measure here: rps, p95, error rate
                         (frontend bypassed)
```

Drives gRPC directly against a single backend at a controlled rate,
from a k6 Job running *inside* the target cluster.
This is the only way to obtain a rigorously calibrated `lambda_max` for
a specific backend service.

| Aspect | k6 HTTP | k6 HTTP + observer | k6 gRPC Job |
|---|---|---|---|
| Protocol | HTTP | HTTP + Prometheus | gRPC |
| Target | frontend | frontend (+ observe backends) | one backend |
| Rate controlled on | frontend | frontend | backend under test |
| Sees backends? | no (black-box) | yes (via Hubble) | yes (direct) |
| Isolates a service? | no | no | yes (with real downstream deps) |
| Produces | `lambda_max` for frontend | bottleneck identification | `lambda_max` per backend |

The three layers are complementary: k6 HTTP calibrates the frontend, the
in-cluster k6 gRPC Job calibrates each backend, and the observer
cross-checks both under realistic traffic.

---

## 5. Frontend procedure (k6, HTTP) <a name="5-frontend"></a>

### 5.1 Entry point

The frontend is reached through **Cilium Ingress** (NodePort) at:

```
http://<node_ip>:30080
```

Cilium Ingress, **not** NGINX, is the project's ingress controller. The
k6 script therefore points at port 30080 on the cluster's node IP. The
cluster → ingress-URL map is read at runtime from `.cache/test-env.json`
(gitignored; rendered by `scripts/render-runtime-files.sh` from
`config/local.yaml`), and the target cluster is selected with
`-e DMOS_CLUSTER=ms-cluster-NN`.

### 5.2 Workload mix

The k6 capacity script uses a 7-endpoint mix designed to exercise every
backend service in the call graph at the realistic ratios of the
multi-ingress experiment:

| Endpoint | Method | Weight | Backend reach |
|---|---|---|---|
| `/` | GET | 0.40 | productcatalog + recommendation + currency × N |
| `/product/OLJCESPC7Z` | GET | 0.15 | productcatalog + currency |
| `/product/66VCHSJNUP` | GET | 0.10 | productcatalog + currency |
| `/cart` | GET | 0.10 | cartservice + currency |
| `/cart` | POST | 0.15 | cartservice (add item) |
| `/setCurrency` | POST | 0.05 | currencyservice |
| `/cart/checkout` | POST | 0.05 | checkout fan-out (6 services) |

Source: `experiments/k6_capacity_test.js` (`ENDPOINTS` array).

The mix matters: a heavier checkout share would lower the measured knee
because each request touches more services. The 5% checkout weight is
chosen to drive the heavy path enough to keep it warm, without dominating
the average per-request cost.

### 5.3 Arrival-rate executor

The script builds one k6 `constant-arrival-rate` scenario per step (not
a single `ramping-arrival-rate` ramp): each step runs at a **fixed
target rps** for its whole duration, split into a `settle` phase
(discarded) and a `measure` phase (aggregated):

```javascript
scenarios[measureName] = {
  executor: 'constant-arrival-rate',
  rate: rate,                        // fixed rps for this step
  timeUnit: '1s',
  duration: `${STEP_DURATION_S}s`,
  startTime: `${cursor}s`,
  tags: { step: ..., phase: 'measure' },
  exec: 'doRequest',
};
```

Arrival-rate executors control **target rps directly** (instead of
indirectly through virtual users), which eliminates the "users → rps"
conversion that Locust requires and keeps each measurement free of the
in-stage transitions a ramping executor would introduce. There is no
per-VU `sleep`: the rate is enforced entirely by the executor.

### 5.4 Step parameters (defaults)

The defaults baked into `k6_capacity_test.js` (all overridable with
`-e VAR=value`):

| Env var | Default | Meaning |
|---|---|---|
| `DMOS_CLUSTER` | `ms-cluster-01` | target cluster (key into `.cache/test-env.json`) |
| `DMOS_STEPS` | `5,10,20,30,40,50,60,70,80,100` | comma-separated rps of the steps |
| `DMOS_STEP_DURATION_S` | 90 s | measured duration of each step |
| `DMOS_SETTLE_S` | 15 s | discarded settle phase before each measure |
| `DMOS_SLA_P95_MS` | 200 ms | per-step SLA p95 threshold |
| `DMOS_SLA_ERR_RATE` | 0.02 | per-step SLA error-rate threshold |
| `DMOS_REPLICAS` | 1 | replicas behind the test (for rps/replica) |

A run lasts `steps × (SETTLE + STEP_DURATION)`, i.e. with the defaults:
10 × 105 s ≈ 17.5 minutes per cluster. The orchestrator
(`experiments/run_capacity.sh`) overrides the step range with
`25,50,75,100,125,150,175,200,250,300` rps, suited to the measured
frontend knee.

### 5.5 Running the test

The orchestrator `experiments/run_capacity.sh` automates the whole
procedure: it pins the service to 1 replica on the target cluster and
0 on the other five (to avoid ClusterMesh contamination), waits for the
rollout, snapshots Prometheus before/after, runs k6, and restores the
fleet:

```bash
# 0. netem must be OFF: the verdict is client-side p95 latency, and
#    netem inflates it (false knee). Restore it afterwards.
bash scripts/setup-netem-and-ping.sh remove-netem

# 1. Default: frontend on ms-cluster-01
bash experiments/run_capacity.sh

# 2. Another cluster
bash experiments/run_capacity.sh --cluster ms-cluster-05

# 3. Raw k6 with overrides (no orchestration — pin replicas yourself)
k6 run -e DMOS_CLUSTER=ms-cluster-02 -e DMOS_STEPS="10,20,30,40,50" \
       -e DMOS_SLA_P95_MS=200 experiments/k6_capacity_test.js

# 4. Restore the geographic simulation
bash scripts/setup-netem-and-ping.sh netem-only
```

Also stop DMOS and scale the bundled `loadgenerator` to 0 first, so they
don't add background traffic (Section 3.2).

Expected end-of-run output (illustrative — your numbers will differ):

```
===========================================================================
  CAPACITY TEST — ms-cluster-01
  URL: http://<CLUSTER_IP>:30080
  Replicas: 1 | SLA: p95 < 200ms, err < 2.0%
  Steps: 10 | Settle 15s + measure 90s per step | Tot 17.5 min
===========================================================================

  Step | rps tgt | rps obt | rps/rep |   p50  |   p95  |   p99  | Err%  | SLA
  -------------------------------------------------------------------------
     1 |     5r/s |     4.9 |     4.9 |   65ms |   88ms |  120ms |  0.0% |  OK
     2 |    10r/s |     9.9 |     9.9 |   66ms |   92ms |  130ms |  0.0% |  OK
     ...
     6 |    50r/s |    49.6 |    49.6 |   80ms |  150ms |  260ms |  0.2% |  OK
     7 |    75r/s |    74.1 |    74.1 |   95ms |  190ms |  380ms |  0.6% |  OK
     8 |   100r/s |    88.3 |    88.3 |  180ms |  720ms | 1100ms |  2.8% | FAIL

  -> SLA violato per 2 step consecutivi: saturazione confermata
===========================================================================
  Knee-point:                75.0 rps/replica  (a 75 rps target, 1 replica/e)
  Capacita raccomandata:     60 rps/replica  (80% del knee, margine per spike)

  -> Aggiorna config/services.yaml:
     capacity_req_per_sec: 60
===========================================================================
```

The script auto-emits the YAML snippet to paste into `services.yaml`.

### 5.6 Per-cluster differences (frontend)

In the original 3-cluster DMOS thesis the frontend capacity was split
per cluster, because the test was run with `tc netem` active and each
cluster's netem delay lowered the measurable rps. With the current
methodology (netem off during the test, backends isolated with
`service.cilium.io/affinity: local`), per-replica frontend capacity is
**uniform across the fleet**: the pod has a fixed CPU limit, so the rps
one replica sustains does not depend on the worker size. Hardware
heterogeneity (3 tiers: HIGH 8 vCPU, MED 6, LOW 4) is expressed through
`max_replicas_per_cluster` — how many replicas a cluster can host — not
through `capacity_req_per_sec`:

```yaml
frontend:
  capacity_req_per_sec: 60          # global, uniform (knee 75 × 0.80)
  # capacity_per_cluster: NOT set → every cluster uses the global value
```

`config_loader.get_capacity_for_cluster(name)` returns the per-cluster
value when present and falls back to `capacity_req_per_sec` otherwise —
the hook exists for future per-cluster splits (e.g. per-tier CPU
limits), but none is currently configured.

Backends do **not** need a per-cluster split either: their gRPC traffic
is intra-cluster and is not affected by inter-cluster netem.

---

## 6. Backend procedure (k6 gRPC Job in-cluster, per service) <a name="6-backend"></a>

> Full manual: `experiments/CAPACITY-GRPC.md`. This section is a summary.

### 6.1 Why an in-cluster k6 Job (and not ghz over port-forward)

DMOS sizes seven gRPC backends. Four of them — `cartservice`,
`productcatalogservice`, `checkoutservice`, `recommendationservice` — have a
measured capacity; `currencyservice`, `adservice` and `shippingservice` use
estimates (see `config/services.yaml`). Backends cannot be hit with HTTP;
they need a gRPC load generator. The current tool is **k6 with its built-in `k6/net/grpc`
module** (`experiments/k6_capacity_grpc.js`), running as a **Job inside
the target cluster** and orchestrated by
`experiments/run_capacity_grpc.sh`.

The historical method — `ghz` on the developer machine reaching the pod
through `kubectl port-forward` — is **superseded**. A full run
demonstrated two artifacts: a fixed ~47–54 ms latency tax on every
request (the PC → API server → pod round trip, which alone ate the
50 ms SLA of cartservice/productcatalog), and connection-refused storms
at high rates (the local port-forward listener collapses long before
the microservice does). The knees it reported were the port-forward's,
not the service's. See `experiments/CAPACITY-GRPC.md` Section 3 for the
full evidence.

### 6.2 Reaching the pod

The k6 Job (image `grafana/k6:latest`) runs in the `online-boutique`
namespace of the target cluster and hits the backend Service **via
ClusterIP** — pod-to-pod gRPC, sub-millisecond base latency, through
the same Cilium L7 proxy as real frontend→backend traffic, i.e. the
production data path. The script and `experiments/demo.proto` are
shipped to the cluster in a ConfigMap mounted at `/scripts`.

| Backend | gRPC port | RPC method tested |
|---|---|---|
| `cartservice` | 7070 | `hipstershop.CartService/GetCart` |
| `productcatalogservice` | 3550 | `hipstershop.ProductCatalogService/ListProducts` |
| `recommendationservice` | 8080 | `hipstershop.RecommendationService/ListRecommendations` |
| `checkoutservice` | 5050 | `hipstershop.CheckoutService/PlaceOrder` |

### 6.3 Dependencies are NOT mocked

This is the most important conceptual point of backend testing.

**The k6 Job does not isolate a service from its downstream
dependencies.** When it calls `checkoutservice/PlaceOrder`, the
checkoutservice internally calls cartservice, productcatalogservice,
currencyservice, shippingservice, paymentservice, and emailservice —
exactly as it would in production. The only thing the Job bypasses is
the *frontend*: k6 replaces the frontend as the request source.

| Service | RPC tested | Downstream calls | Latency profile |
|---|---|---|---|
| `productcatalogservice` | `ListProducts` | none (in-memory catalog) | pure |
| `cartservice` | `GetCart` | `redis-cart` | quasi-pure (+1 ms Redis) |
| `recommendationservice` | `ListRecommendations` | `productcatalogservice` | + 1 gRPC hop |
| `checkoutservice` | `PlaceOrder` | cart, catalog, currency, shipping, payment, email | full fan-out (heaviest path) |

This is the *correct* behavior for `lambda_max` calibration. The value
in `services.yaml` represents how many rps a replica can serve **in the
real system**; in the real system the dependencies are present and
their latency is part of the response time. Mocking the dependencies
would yield an inflated number that the service can never reach in
production.

Note: the checkoutservice test sends `PlaceOrder` with random user ids,
i.e. empty carts — a slightly lighter fan-out than a real order, but a
reasonable proxy because the CPU cost is dominated by the orchestration
itself (see `CAPACITY-GRPC.md` Section 5).

### 6.4 Test order and dependency boosting

Test in dependency order: services with the fewest downstream
dependencies first, so that when a service-under-test reaches its knee
you know its dependencies were not the bottleneck.

```
1. productcatalogservice  (0 dependencies)
2. cartservice            (1 dep: redis-cart, non-gRPC)
3. recommendationservice  (1 dep: productcatalog, validated in step 1)
4. checkoutservice        (6 deps: validated in steps 1–3 + non-DMOS)
```

To keep the dependencies out of the bottleneck, `run_capacity_grpc.sh`
scales the service-under-test's dependencies to **4 replicas on the
target cluster** for the duration of the test (with
`service.cilium.io/affinity: local` the backend calls the local
replicas) and restores them to 1 afterwards. It also scales the bundled
`loadgenerator` to 0 on **all** clusters before the test — remember to
turn it back on afterwards.

### 6.5 Per-service SLA thresholds (defaults)

In-cluster latency is low (~1–30 ms), so the absolute SLAs have wide
margins. All values are overridable via `DMOS_SLA_P95_MS` /
`DMOS_STEPS`:

| Service | SLA p95 default | Default steps (rps) |
|---|---|---|
| frontend (HTTP, Section 5) | 200 ms | 25…300 |
| cartservice | 50 ms | 300,400,500,600,700,800,1000,1300 |
| productcatalogservice | 50 ms | 500,1000,1500,2000,2500,3000,4000,5000 |
| recommendationservice | 120 ms | 100,110,120,130,140,150,170,200 |
| checkoutservice | 500 ms | 100,150,200,250,300,350,450,600 |

Source: the per-backend `B_STEPS` / `B_SLA_P95` tables in
`experiments/run_capacity_grpc.sh`. The gRPC knee additionally requires
`p99 < 4 × SLA p95` (`DMOS_P99_SLA_FACTOR`, default 4); see Section 8.1.

### 6.6 Running

Run from Git Bash, from the repo root:

```bash
# One backend (default: cartservice on ms-cluster-01, tier HIGH)
bash experiments/run_capacity_grpc.sh --backend cartservice

# All four backends in sequence (~40 min, recommended for calibration)
bash experiments/run_capacity_grpc.sh --all

# Another cluster
bash experiments/run_capacity_grpc.sh --backend cartservice --cluster ms-cluster-04

# Custom steps / SLA via environment variables
DMOS_STEPS="20,40,60,80,100,120" DMOS_SLA_P95_MS=150 \
  bash experiments/run_capacity_grpc.sh --backend recommendationservice
```

For each backend the script writes:

```
experiments/results/
  capacity_grpc_<backend>_<timestamp>.json        # summary + knee + per-step
  capacity_grpc_<backend>_<timestamp>.k6log.txt   # full k6 Job log
```

The JSON summary includes the recommended `capacity_req_per_sec` value
(80 % of the knee) and the script prints the YAML snippet to paste into
`services.yaml` directly.

---

## 7. System-level observation (Hubble L7 + Prometheus) <a name="7-observer"></a>

While k6 hits the frontend, a parallel observer scrapes Hubble L7 and
container metrics for *every* service every 15 s and identifies which
service hits its knee first.

| Metric | Prometheus query (sketch) |
|---|---|
| Served rps | `sum(rate(hubble_http_requests_total{destination_workload="<svc>", destination_namespace="online-boutique"}[1m]))` |
| p95 latency | `histogram_quantile(0.95, sum by (le) (rate(hubble_http_request_duration_seconds_bucket{destination_workload="<svc>"}[1m]))) * 1000` |
| Error rate | `rate(hubble_http_requests_total{...,http_status_code=~"5.."}[1m]) / rate(hubble_http_requests_total{...}[1m])` |
| CPU | `sum(rate(container_cpu_usage_seconds_total{pod=~"<svc>.*"}[1m]))` |
| Replicas | `count(kube_pod_info{created_by_name=~"<svc>-.*"})` |

The rps of a backend counts every call it receives (from the frontend and
from other backends), which is what loads it; under saturation retries
inflate it, which is why DMOS does not size backends from it
([BACKEND-CO-SCALING.md](BACKEND-CO-SCALING.md)). Exact queries:
`experiments/backend_capacity_observer.py`.

Two terminals are needed:

```powershell
# Terminal 1 — drive load
k6 run -e DMOS_CLUSTER=ms-cluster-01 experiments/k6_capacity_test.js

# Terminal 2 — observe (start immediately after k6)
python experiments/backend_capacity_observer.py --cluster ms-cluster-01
```

End-of-run report (illustrative):

```
Service                   | Knee rps/rep | Recommended | Current | Delta   | Violations
frontend                  |        39.2  |          39 |      39 |   +0.5% |        2
cartservice               |        35.8  |          36 |      50 |  -28.4% |        1   <<
productcatalogservice     |        78.5  |          79 |      80 |   -1.9% |        0
checkoutservice           |         6.8  |           7 |      20 |  -66.0% |        3   <<
recommendationservice     |        31.2  |          31 |      40 |  -22.0% |        1   <<

Bottleneck: checkoutservice (knee=6.8 rps/replica)
```

The observer is for *cross-checking* — it does not replace the
per-backend k6 gRPC Job, because the rps it sees on each backend is
whatever the call graph delivers, not a controlled rate.

---

## 8. Data analysis: knee-finding and saturation criterion <a name="8-analysis"></a>

### 8.1 Definition of the knee point

The knee point is the **last step where p95 < SLA AND error_rate <
2 %**. Beyond it, latency rises super-linearly and small additional
load causes large latency increases.

```
p95 (ms)
 400 |                                /
 300 |                           ____/  <- SLA breached above this
 200 |                     ____/
 150 |               ____/  <- knee point (last step OK)
 100 |        ______/
  70 |_______/
        +--+--+--+--+--+--+--+--+--+-- rps/replica
           5 10 15 20 25 30 35 40 45
```

Algorithmically:

```python
if step["p95_ms"] < sla_p95_ms and step["fail_rate"] < 0.02:
    knee_point_rps = step["rps_per_replica"]   # accept this step
else:
    # 2 consecutive violations confirm saturation -> stop
    break
```

Two consecutive violations are required (instead of one) so that a
single noisy step does not falsely stop the test.

The gRPC script (`k6_capacity_grpc.js`) adds a third condition to the
step acceptance: `p99 < 4 × SLA p95` (`DMOS_P99_SLA_FACTOR`, default 4),
so a step whose p95 is nominally in-SLA but whose tail latency has
already exploded is not accepted as the knee.

### 8.2 Why p95 (and not mean or p50)

* The **mean** is dominated by a few outliers (a single 20 s timeout
  drops the mean only slightly, but indicates real saturation in the
  tail).
* The **median (p50)** hides the slowest 50 % of requests entirely.
* **p95** guarantees that 95 % of users see acceptable latency.

A pathological example: 1000 requests with mean 80 ms, p50 65 ms, p95
340 ms, p99 800 ms. Mean and p50 both look fine; the p95 is the SLA
violator and the right signal for `lambda_max`.

### 8.3 From knee point to `capacity_req_per_sec`

```
capacity_req_per_sec  =  round(knee_point_rps_per_replica × 0.80)
```

Both k6 capacity scripts (HTTP and gRPC) apply a **0.80 factor** to the
raw knee in their summary output. Because DMOS plans
`⌈λ / capacity_req_per_sec⌉` replicas, the factor is the headroom each
replica keeps below its knee in steady state (§2).

If you prefer the raw knee, divide the recommended value by 0.80 before
pasting it into `services.yaml`: DMOS will then plan fewer replicas for the
same load. Keep the factor to stay comparable with the published runs.

### 8.4 Burst vs gradual: known sensitivity

Capacity depends on the load *pattern*. Empirically (sessions of
27/02/2026):

| Scenario | Pattern | Knee point | Why |
|---|---|---|---|
| flash_crowd (burst) | instant spike | ~25 rps/pod | cold connection pools, GC fires during spike |
| gradual (stepped) | 90 s steps | ~43–47 rps/pod | warmed pools, JIT compiled, stable GC |

The gradual value is roughly 2× the burst value. `lambda_max` is
calibrated against the **gradual** case because that is the regime in
which a steady-state autoscaler operates; flash-crowd resilience is
verified separately by the flash-crowd experiment, not by `lambda_max`.

---

## 9. Writing results into `config/services.yaml` <a name="9-config"></a>

After a successful run, transcribe the recommended values (the script's
auto-emitted YAML snippet) into `config/services.yaml`. The fields:

* `capacity_req_per_sec` — global default, used when no per-cluster
  override is set.
* `capacity_per_cluster` — per-cluster overrides; currently **unset**
  for every service (per-replica capacity is uniform, Section 5.6).

The currently recorded values (measured, with the methodology commented
per-service in `services.yaml` itself):

```yaml
services:
  frontend:
    capacity_req_per_sec: 60       # k6 HTTP: knee 75 × 0.80

  cartservice:
    capacity_req_per_sec: 400      # k6 gRPC in-cluster; conservative
                                   # (noisy knee band 500–700 under CFS throttling)

  productcatalogservice:
    capacity_req_per_sec: 1200     # knee 1500 × 0.80

  checkoutservice:
    capacity_req_per_sec: 160      # knee 200 × 0.80 (heavy fan-out path)

  recommendationservice:
    capacity_req_per_sec: 40       # isolated knee 140, derated ≈3×:
                                   # Python GIL queueing under real bursty load
```

`currencyservice` (250), `adservice` (400) and `shippingservice` (500) carry
**estimates**, not measurements; they are candidates for the same gRPC
procedure.

The recommendationservice case is the documented exception to the plain
`knee × 0.80` rule: the isolated gRPC test overstates concurrency-bound
Python services, so the stored value is calibrated on the observed
degradation under the real flash-crowd workload (see the comment block
in `services.yaml`).

The loader helper that resolves the per-cluster value at runtime is
already in place:

```python
# src/utils/config_loader.py
def get_capacity_for_cluster(self, cluster_name: str) -> float:
    if (self.capacity_per_cluster
            and cluster_name in self.capacity_per_cluster):
        return float(self.capacity_per_cluster[cluster_name])
    return float(self.capacity_req_per_sec)
```

For the four gRPC backends, the per-cluster split is normally
unnecessary because they communicate intra-cluster and are not affected
by the inter-cluster netem. If a measurement *does* show a backend
behaving differently across clusters, the most likely cause is CPU
contention with co-located workloads on that K3s node — investigate
with `kubectl top nodes` before resorting to a per-cluster override.

### 9.1 When to re-run

The capacity test must be re-run whenever any of the following
changes:

* Resource limits/requests of the service under test (CPU especially).
* The endpoint mix of the k6 workload (`ENDPOINTS` in
  `experiments/k6_capacity_test.js`, Section 5.2) or the gRPC
  methods/payloads in `experiments/k6_capacity_grpc.js`.
* The container image of the service (new code path, new dependencies).
* The hardware of the K3s nodes (different CPU/RAM).
* Any change that affects the steady-state baseline observed in §11.

---

## 10. Caveats and known limitations <a name="10-caveats"></a>

### 10.1 Cold caches and connection pools

The first ~30 s of every step are dominated by connection-pool
resizing, DNS resolution, and HTTP/2 stream setup. The test ignores
this region by aggregating the *full* step (90 s); the warm-up is
amortized away. For a flash-crowd test, this assumption is wrong —
which is why flash-crowd is a separate experiment.

### 10.2 JIT and GC warm-up (Java, Python, Node.js)

* **adservice (Java/JVM)**: the JIT compiles hot methods during the
  first few hundred requests; before that, throughput is depressed
  and tail latency is bursty. A 90 s step generally crosses this
  transition, but the very first step of every test should be
  considered noisy.
* **recommendationservice (Python)**: the GIL serializes Python
  bytecode; effective concurrency is one core regardless of how many
  vCPU cores the limit grants. A higher CPU limit helps reduce CFS
  throttling but does not lift the GIL ceiling.
* **currencyservice / paymentservice (Node.js)**: V8 heap grows
  generationally; the GC eventually stabilizes around the working set,
  but early-step measurements may show elevated p95 from GC pauses
  that disappear once the heap finds its size.

### 10.3 Sidecar / dataplane overhead

Every pod runs the Cilium dataplane in eBPF; gRPC and HTTP both pay
that small per-packet cost. The test measures end-to-end latency as
seen by the driver, which already includes this overhead — i.e.
`lambda_max` is correct *for the production data plane*. If the data
plane is replaced (e.g. NGINX Ingress instead of Cilium Ingress, or
sidecar Envoy in mesh mode), the values must be re-measured.

### 10.4 Load generator co-located with the target (gRPC Job)

The historical `ghz`-over-`kubectl port-forward` method was retired
precisely because the forwarded stream added a fixed ~47–54 ms tax per
request and collapsed at high rates (Section 6.1). The in-cluster k6
Job removes that artifact, but the load generator now shares the
cluster with the backend under test. Mitigation: the Job's CPU is
capped (request 500m, limit 4 vCPU) and the default target is `ms-cluster-01`
(tier HIGH, 8 vCPU), so the backend's CPU requests stay guaranteed.

### 10.5 Workload sensitivity

`lambda_max` is valid **only for the workload mix used to measure it**
(Section 5.2). A different mix — heavier on `/cart/checkout`, say —
would touch more services per request and lower the measured knee.
Industrial capacity testing (SPEC, TPC-W) calls this the "workload
profile" and reports capacity as "valid for workload W".

### 10.6 Small-VM testbed

Each of the 6 clusters is a small K3s cluster on Proxmox VMs, in three
hardware tiers (HIGH 8 vCPU / 12 GB, MED 6 / 8, LOW 4 / 8). All 11
microservices, kube-system, monitoring (Prometheus), and Cilium share
the same few vCPUs. Capacity numbers measured here are **not portable**
to multi-node clusters with one pod per node — those would have less
CPU contention and a substantially higher knee. They *are* portable to
other K3s testbeds with the same resource limits, which is also why the
per-replica value does not need re-measuring per tier (fixed pod CPU
limits, see Section 5.6).

### 10.7 Shared dependencies

When testing checkoutservice, the downstream services (currency,
shipping, payment, email) only receive traffic *from checkoutservice*
during the test. In production they also receive traffic directly from
the frontend (e.g. `/setCurrency`). The measured `lambda_max` for
checkoutservice can therefore be slightly optimistic in the worst case
where the non-DMOS dependencies are themselves under load. The
system-level observer (Section 7) is the cross-check for this.

---

## 11. Testbed-specific resource tuning (prerequisite) <a name="11-testbed"></a>

A valid capacity test requires a **stable** testbed: no OOMKills, no
CFS throttling pathologies. Three corrections are applied to the
default Online Boutique manifests before any capacity test is run.
They are documented here because re-running the capacity test after
forgetting any of them silently produces nonsensical numbers.

### 11.1 The K3s single-node context

Online Boutique is designed for GKE with 2–4 vCPU nodes and 1–2 pods
per node. The DMOS testbed is the opposite extreme:

| Aspect | GKE (production) | K3s single-node (testbed) |
|---|---|---|
| Nodes per cluster | 3–10 | 1 |
| Pods per node | 1–3 | 11+ (all services) |
| CPUs per node | 2–4 | 4 (shared by everything) |
| Default CPU requests | ~400m / node | ~2400m / 4000m → 60% |
| CPU contention | low | high |

Default Kubernetes CFS scheduling at `200m` CPU gives a container
**20 ms of CPU per 100 ms window**; if the process needs more CPU
inside that window, it is suspended for the remaining 80 ms. This
manifests as a latency spike independent of overall node load.

### 11.2 Correction 1 — OOMKill on Node.js services

`currencyservice` and `paymentservice` are Node.js. The V8 heap grows
under concurrent load:

| State | V8 heap |
|---|---|
| idle | 80–120 MB |
| moderate (5–10 concurrent req) | 150–250 MB |
| sustained (20+ concurrent req) | 350–500 MB |

The default 256 Mi memory limit is below the sustained-load heap, so
the kernel OOM killer terminates the pod (`Reason: OOMKilled, Exit
Code: 137`). Fix:

```bash
kubectl set resources deployment currencyservice \
  -n online-boutique \
  --limits=cpu=500m,memory=512Mi \
  --requests=cpu=100m,memory=256Mi \
  --kubeconfig "$KUBECONFIG_DIR/kubeconfig-ms-cluster-01"
# repeat for paymentservice and for every cluster (ms-cluster-01..06)
```

### 11.3 Correction 2 — CPU throttling on currencyservice

The frontend calls `currencyservice` once per displayed product price.
The home page lists ~10 products → ~10 sequential gRPC calls per
request. Each call needs ~3–5 ms of CPU. With `200m` (20 ms / 100 ms
budget), after ~5 calls the pod is throttled for 80 ms before the
next call can run. Result: a single isolated request to `/` takes
533 ms — *with no concurrent load*.

Raising `currencyservice` to `500m` reduces this to 187 ms (-65 %).
The improvement appears with one user, which proves the bottleneck
was structural (CFS throttling) rather than concurrent contention.

### 11.4 Correction 3 — CPU throttling under concurrency

Once correction 2 lifts the per-request floor, concurrent load
revealed a second issue: every backend service was at the same
`200m` default, and throttling compounded across hops. Languages with
burst-CPU profiles suffered most:

* **adservice (Java/JVM)** — JIT compilation and stop-the-world GC
  need short bursts of much more than 20 ms; each one was throttled
  into a long pause.
* **recommendationservice (Python)** — the GIL already serializes
  execution; on top of that, throttling dropped useful CPU time on
  the only thread.

Final per-service settings (as tuned on the original 3-cluster thesis
testbed; the values the current fleet assumes are tracked per-service
in the `resources_per_replica` blocks of `config/services.yaml`):

| Service | Lang | CPU before | CPU after |
|---|---|---|---|
| `frontend` | Go | 200m | **1000m** |
| `adservice` | Java | 200m | **1000m** |
| `recommendationservice` | Python | 200m | **1000m** |
| `currencyservice` | Node.js | 200m | **500m** |
| `paymentservice` | Node.js | 200m | **500m** |
| `cartservice` | Go | 200m | **500m** |
| `shippingservice` | Go | 200m | **500m** |
| `productcatalogservice` | Go | 200m | **500m** |
| `checkoutservice` | Go | 200m | **500m** |
| `emailservice` | Go | 200m | **500m** |

Sum of *requests* (not limits) after the changes is ~1.3 of 4 cores
per node, i.e. the node is **not over-committed on requests**. Limits
sum to more than the node, but that is the expected pattern: pods are
allowed to burst into idle CPU.

### 11.5 Effect of all three corrections

| Metric | Before | After |
|---|---|---|
| Single isolated request, p95 | 533 ms | 187 ms |
| OOMKilled restarts (currencyservice) | 26 | 0 |
| Concurrent latency (40 vusers) | 2.7–10 s | < 100 ms |
| Capacity test reaches saturation? | no (failed at step 1) | yes |

The tracked source of truth for the testbed resource/probe patches is
now `experiments/apply_resource_hardening.sh` (idempotent, applies to
all six clusters); the per-service CPU/memory settings DMOS assumes are
recorded in `config/services.yaml`. The hardening script must be re-run
after re-applying the upstream Online Boutique manifests, since those
overwrite the patched defaults.

---

## 12. Migration / history <a name="12-history"></a>

This document consolidates two prior Italian-language documents:

* `docs/_archive/capacity-test.md` — covered the **frontend** capacity
  test (HTTP). Source of Sections 5, 8, and 11. Last revised March 13,
  2026.
* `docs/_archive/backend-capacity-test.md` — covered the **backend**
  capacity test (gRPC via ghz) and the system-level observer. Source of
  Sections 6 and 7. Last revised April 16, 2026.

Where the two source documents covered the same step (knee-point
definition, stepped-load rationale, p95 motivation), the more detailed
of the two narratives was kept and the other was dropped. Where they
diverged (frontend HTTP entry point via Cilium Ingress NodePort vs gRPC
port-forward; per-cluster splitting only for the frontend), both are
preserved as separate sub-sections.

**Tooling change.** The original frontend procedure used Locust. As of
this consolidation, **k6 is the active tool** (`experiments/k6_capacity_test.js`),
chosen for its arrival-rate executors — one `constant-arrival-rate`
scenario per step (Section 5.3), which controls target rps directly
instead of through virtual-user math — and for its single
JS file with no Python runtime dependency. Locust is still present in
the repo as `experiments/locustfile_multiingress.py` but only for the
multi-ingress experiment, not for capacity calibration.

**Terminology.** Throughout this document `λ_max` is the formula
symbol; the on-disk field in `services.yaml` is `capacity_req_per_sec`
(global) and `capacity_per_cluster` (per-cluster overrides). The two
refer to the same quantity. Both k6 capacity scripts apply a 0.80
factor on top of the raw knee in their auto-emitted YAML snippet; see
Section 8.3.

The two source files are retained under `docs/_archive/` for
traceability and are not deleted by this consolidation.
