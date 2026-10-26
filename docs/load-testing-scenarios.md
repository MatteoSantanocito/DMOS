# Load Testing Scenarios — DMOS

Documentation of the canonical load-test scenario used to evaluate DMOS:
the **flash-crowd** scenario implemented in `experiments/k6_flash_crowd.js`
with the Grafana k6 `ramping-arrival-rate` executor.

---

## Table of contents

1. [Overview](#1-overview)
2. [Why k6 (and why an open-arrival model)](#2-why-k6)
3. [Test architecture](#3-test-architecture)
4. [The `flash_crowd` scenario](#4-flash_crowd-scenario)
5. [Geographic distribution and ingress selection](#5-geographic-distribution)
6. [Endpoint mix](#6-endpoint-mix)
7. [Compared configurations](#7-three-comparison-scenarios)
8. [Run command and outputs](#8-run-command)
9. [Test prerequisites](#9-prerequisites)
10. [Per-cluster post-hoc analysis](#10-post-hoc)
11. [Legacy Locust workload](#11-legacy-locust)
12. [References](#12-references)

---

## 1. Overview <a name="1-overview"></a>

The DMOS evaluation uses a single canonical scenario: a **flash crowd**
that ramps the request rate from a 50 req/s baseline to a **350 req/s
peak**, holds it, then declines — five phases over 15 minutes, driven
simultaneously against the six cluster ingresses with a phase-dependent
geographic distribution. The scenario stresses the system in the regime
where DMOS is supposed to add the most value — fast traffic transients
across geographically distributed ingresses with heterogeneous
inter-cluster latency (tc netem 0–200 ms).

The canonical evaluation sweeps **six traffic-concentration regimes**
(hottest-cluster share 0.167 → 0.85, see §5) against three system arms
(§7), with 3 repetitions per cell.

The active load-generation tool is **Grafana k6**. The script lives at
`experiments/k6_flash_crowd.js` and is orchestrated per-run by
`experiments/run_flash_crowd.sh`. Locust is not used in the canonical
evaluation; see §11 for the legacy multi-ingress workload.

---

## 2. Why k6 (and why an open-arrival model) <a name="2-why-k6"></a>

k6 was chosen over Locust for three reasons:

1. **Open-arrival executor** (`ramping-arrival-rate`): k6 controls the target
   request rate `λ(t)` directly, independently of how slowly the server
   responds. This is what real users do during a flash crowd — they continue
   to arrive even if the service has slowed down.
2. **Low overhead**: a single k6 binary written in Go drives thousands of
   requests per second from one machine without becoming a client-side
   bottleneck.
3. **Per-request JSON output** with the `--out json` flag, including
   timestamps, latency, status code, and custom tags (`ingress`,
   `cluster`, `endpoint`, `phase`) — enabling per-cluster, per-phase
   post-hoc analysis at the granularity of individual requests.

### Closed model vs. open model

Locust uses a **closed model**: a fixed pool of N virtual users (VUs) each
runs `request → wait_time → request` sequentially. Effective throughput is

```
λ_closed = N_VU / (R + Z)
```

where R is the average response time and Z is the think time. When the
server slows (R↑), throughput **automatically drops** — masking the real
breaking point of the system.

k6's `ramping-arrival-rate` uses an **open model**:

```
λ_open(t) = target_rate(t)
```

Arrivals are independent of response time. If the server is too slow to
keep up, k6 dynamically allocates more VUs (up to `maxVUs`) to keep the
target rate. This faithfully reproduces the dynamics of a real flash crowd
where users keep arriving and overload the system if the autoscaler does
not react.

---

## 3. Test architecture <a name="3-test-architecture"></a>

```
        ┌──────────── k6 (dedicated load-gen VM or operator machine) ────────────┐
        │       ramping-arrival-rate, 50 → 350 → 50 req/s (5 phases, 15 min)     │
        └──┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
           │ w1       │ w2       │ w3       │ w4       │ w5       │ w6   (phase-dependent
           ▼          ▼          ▼          ▼          ▼          ▼      geo weights, §5)
        ms-cluster-01 … ms-cluster-06   — one Cilium Ingress :30080 per cluster,
                                          tc netem 0–200 ms outbound delay per cluster
           │
           ▼
        frontend (per cluster; Cilium global service)
           │
           ▼
        backends (cartservice, productcatalog, recommendation,
        checkout, currency, shipping, ad, payment, email, redis)
           │
           ▼
        Prometheus per-cluster (kube-prometheus-stack, ns observability,
        NodePort 32090)
           │
           ▼
        DMOS Scheduler (operator machine)
        /metrics → :9090
```

Each of the six clusters runs an independent K3s + Cilium + Hubble +
Prometheus stack, joined by ClusterMesh. The frontend and four backends are
Cilium global services. As in the NCA paper, the frontend has `affinity:
none`: each cluster's ingress spreads the requests that enter it uniformly
over the ready frontend pods of all clusters, so a cluster serves a share of
the traffic equal to its share of the frontend replicas
([CILIUM-CLUSTERMESH-NOTES.md](CILIUM-CLUSTERMESH-NOTES.md)). The backends
keep `affinity: local`.
Cluster names, IPs and ingress URLs are read by the k6 script from
`.cache/test-env.json` (rendered by `scripts/render-runtime-files.sh` from
`config/local.yaml`).

---

## 4. The `flash_crowd` scenario <a name="4-flash_crowd-scenario"></a>

Defined in `experiments/k6_flash_crowd.js`:

```javascript
export const options = {
    scenarios: {
        flash_crowd: {
            executor: 'ramping-arrival-rate',
            startRate: BASELINE_RPS,                      // default 50
            timeUnit: '1s',
            preAllocatedVUs: Math.max(200, PEAK_RPS * 2),
            maxVUs: Math.ceil(PEAK_RPS * VU_FACTOR),      // default factor 20
            stages: [
                { duration: '180s', target: BASELINE_RPS },  // baseline
                { duration: '120s', target: PEAK_RPS },      // spike (linear ramp)
                { duration: '300s', target: PEAK_RPS },      // sustained peak
                { duration: '120s', target: BASELINE_RPS },  // decay
                { duration: '180s', target: BASELINE_RPS },  // cooldown
            ],
            gracefulStop: '30s',
        },
    },
    thresholds: { dropped_iterations: ['count<2000'] },
    summaryTrendStats: ['avg', 'min', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
};
```

Every parameter is env-tunable (`DMOS_BASELINE_RPS`, `DMOS_PEAK_RPS`,
`DMOS_PHASE_TIMES`, `DMOS_VU_FACTOR`, …). The canonical evaluation pins
`DMOS_PEAK_RPS=350`. Five phases, total **900 s = 15 min**:

| Phase          | Start | End   | Target rate (rps) | Purpose                                                                             |
|----------------|-------|-------|-------------------|--------------------------------------------------------------------------------------|
| Baseline       | 0 s   | 180 s | 50                | Covers DMOS' 120 s startup grace period and the predictor's baseline window          |
| Spike          | 180 s | 300 s | 50 → 350 (linear) | Sudden surge over 120 s; geo distribution shifts to the spike weights (§5)           |
| Sustained peak | 300 s | 600 s | 350               | 5 minutes of steady peak — at a 30 s scheduling interval, ~10 cycles to converge     |
| Decay          | 600 s | 720 s | 350 → 50          | Tests scale-down reactivity; geo distribution returns to baseline                    |
| Cooldown       | 720 s | 900 s | 50                | Verifies the system returns to a clean baseline state                                |

The VU pool is sized for the worst case (`maxVUs = PEAK_RPS × VU_FACTOR`):
under saturation (dmos-off arm) latencies can approach the 14 s request
timeout, and an undersized pool would silently deliver less load than
offered. `handleSummary` enforces a **validity gate**: if more than 0.5 %
of offered iterations are dropped, the run is marked `RUN NON VALIDO` and
must be repeated with a higher `DMOS_VU_FACTOR`.

---

## 5. Geographic distribution and ingress selection <a name="5-geographic-distribution"></a>

The script targets the six Cilium Ingresses with **phase-dependent
weights**. During **baseline / decay / cooldown** each cluster receives a
region-based default share (override via `DMOS_INGRESS_W1..WN`):

| Region | Baseline weight |
|--------|-----------------|
| DE     | 0.30            |
| FR     | 0.20            |
| PL     | 0.15            |
| NO     | 0.15            |
| US-VA  | 0.10            |
| SG     | 0.10            |

Each cluster also has an emulated outbound delay (`tc netem`, 0–200 ms in
the reference testbed, declared per cluster in `config/local.yaml`).

During **spike / sustained peak** the distribution shifts: either the
legacy single-spike form — `DMOS_SPIKE_REGION` (default `ms-cluster-01`)
receives `DMOS_SPIKE_WEIGHT` (default 0.60) and the others split the
remainder — or an explicit per-cluster override `DMOS_SPIKE_W1..W6`
(weights are normalized by their sum). The canonical evaluation sweeps
**six traffic-concentration regimes**, characterised by the
hottest-cluster share during the peak:

| Regime | Spike weights (W1…W6) | Hottest-cluster share |
|--------|------------------------|-----------------------|
| uni    | 1/1/1/1/1/1            | 0.167                 |
| bal    | 40/15/12/12/11/10      | 0.40                  |
| s50    | 50/10/10/10/10/10      | 0.50                  |
| s65    | 65/7/7/7/7/7           | 0.65                  |
| s75    | 75/5/5/5/5/5           | 0.75                  |
| con    | 85/3/3/3/3/3           | 0.85                  |

> **Note (per-VU vs per-iteration ingress selection).** An earlier variant
> assigned each k6 VU one ingress at initialization (per-VU geographic
> affinity). The current `experiments/k6_flash_crowd.js` picks the ingress
> **per iteration** (`pickIngress` inside the `default` function): per-VU
> affinity caused cascading stalls when one slow cluster saturated its VUs
> and k6 hit `maxVUs`, collapsing the global request rate across all
> ingresses. The per-iteration variant preserves the global weight
> distribution and avoids the cascade; the two are observationally
> equivalent at the cluster level outside saturation.

---

## 6. Endpoint mix <a name="6-endpoint-mix"></a>

Each iteration selects an Online Boutique endpoint via `pickWeighted`.

| Endpoint              | Method | Weight | Type  | Backend services hit                                          |
|-----------------------|--------|--------|-------|---------------------------------------------------------------|
| `/`                   | GET    | 0.40   | Read  | frontend → productcatalog + currency + ad                     |
| `/product/OLJCESPC7Z` | GET    | 0.15   | Read  | frontend → productcatalog + currency + recommendation         |
| `/product/66VCHSJNUP` | GET    | 0.10   | Read  | frontend → productcatalog + currency + recommendation         |
| `/cart`               | GET    | 0.10   | Read  | frontend → cart + currency                                    |
| `/cart`               | POST   | 0.15   | Write | frontend → cart + productcatalog                              |
| `/setCurrency`        | POST   | 0.05   | Write | frontend → currency                                           |
| `/cart/checkout`      | POST   | 0.05   | Write | frontend → cart + checkout + shipping + payment + email + currency + productcatalog (8-service fan-out) |

Distribution: **75 % reads, 20 % cart writes, 5 % checkout**. The 5 %
checkout share is small but disproportionately impactful: each checkout
amplifies the inter-cluster latency penalty across 8 microservices.

---

## 7. Compared configurations <a name="7-three-comparison-scenarios"></a>

The evaluation runs the same `flash_crowd` scenario against system arms
that isolate each DMOS component's contribution:

| Arm (run label `dmos-<arm>__<regime>`) | L1 (placement) | L2 (budget + predictor + PD) + co-scaling | What it isolates |
|---|---|---|---|
| **`l1l2`** | ON | ON | full system as run in the NCA 2026 campaign (gate g = 2, weights 0.25/0.20/0.25/0.30) |
| a score model, e.g. **`nca`** | ON | ON | full system with a named score (`nca` = the configuration described in the paper) |
| **`l2`** | OFF (uniform N/K) | ON | contribution of the placement |
| **`off`** | OFF | OFF | static baseline: 1 replica per cluster per service |

Every name — arm and regime — is defined in `experiments/run_naming.py` and
means the same configuration in every script and in the analysis
(`python experiments/run_naming.py list`). The score models (`gate_g0`,
`gate_g1`, `gate_g3`, `additive_dem50`, `locality`, `*_first`) are the
variants of the journal extension; `legacy-*` and `profile-*` arms keep the
historical configurations reachable
([dmos-architecture.md §7.4–7.5](dmos-architecture.md#74-score-models-the-role-of-demand)).

A regime name can also move the hot share to another cluster and change the
peak: `<demand>[-<hot>][-p<peak>]`, e.g. `con-sg` (85 % on Singapore),
`s50-p500` (50 % on DE at 500 req/s). With another hot cluster the weights
of §5 are swapped between ms-cluster-01 and that cluster.

All arms use the same flash-crowd profile and `tc netem` settings and are
swept over the concentration regimes of §5, with repeated sessions per (arm,
regime) cell. Results are produced by the analysis pipeline (§10).

---

## 8. Run command and outputs <a name="8-run-command"></a>

The canonical way is one cell at a time through the run harness, which
resets the testbed, checks it, starts DMOS with the arm's configuration,
runs k6 on the load-generation VM and records the provenance
([../experiments/RUNBOOK.md](../experiments/RUNBOOK.md)):

```bash
export DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>
bash experiments/run_one.sh l1l2 con       # peak 350 req/s, regime con
bash experiments/run_one.sh l2   con
bash experiments/run_one.sh off  con
bash experiments/run_one.sh nca  con-sg    # 85 % of the demand on Singapore
```

`run_matrix_session.sh --arms "…" --regimes "…"` runs one repetition of a
matrix (default: off / l2 / l1l2 × the six regimes), `run_skew_sweep.sh` the
s50/s65/s75 part; `--dry-run` prints the plan.
`experiments/run_flash_crowd.sh --label <label>` runs only the k6 part (with
DMOS started by hand, or `--no-dmos`); `k6 run experiments/k6_flash_crowd.js`
runs the script alone.

Each run produces a folder `experiments/results/<date>_<time>_<label>/`; its
files (manifest, effective DMOS configuration, k6 summary and raw events,
scheduler/cluster timeline, testbed snapshots) are described in
[METRICS-AND-OUTPUTS.md §2](METRICS-AND-OUTPUTS.md#2-the-run-directory). Every
k6 event carries the tags `ingress`, `cluster`, `endpoint`, `phase`,
`method`, plus the global `run_label`.

---

## 9. Test prerequisites <a name="9-prerequisites"></a>

Before launching the test:

1. **Apply geographic netem** (for the canonical scenario):
   ```bash
   bash scripts/setup-netem-and-ping.sh
   ```
   Applies the per-cluster `netem.delay_ms` declared in
   `config/local.yaml` (0–200 ms in the reference testbed). Use
   `bash scripts/setup-netem-and-ping.sh verify` to check the rules.

2. **Keep the bundled `loadgenerator` off** so it does not contaminate the
   measured rates and latencies:
   ```bash
   for kc in "${KUBECONFIG_DIR:-$HOME/dmos-iac}"/kubeconfig-ms-cluster-*; do
       kubectl --kubeconfig "$kc" scale deployment loadgenerator \
               -n online-boutique --replicas=0
   done
   ```
   The harness's reset (`scripts/reset_test.ps1`, `clean_reset.sh`) scales
   every Deployment to 1 replica and then the `loadgenerator` back to 0, and
   every cell refuses to start while it has replicas (recorded in
   `env_state_pre.json`; override with `DMOS_ALLOW_LOADGENERATOR=1`).

3. **Verify the `kube-prometheus-stack` is up** on each cluster (NodePort
   32090, namespace `observability`) and Hubble metrics are populated
   (`hubble_http_requests_total`).

4. **DMOS before k6.** During its first 120 s (cold start) DMOS only
   observes the system — no scale operation — and the predictor accumulates
   its history.
   The harness starts k6 a few seconds after DMOS is ready, so the grace
   falls inside the 180 s baseline phase and the spike (t = 180 s) finds
   DMOS active. When running by hand, keep the same order.

5. **Verify time sync between operator and clusters** (`chronyd` or
   `systemd-timesyncd`). The k6 events are correlated with the collector
   timeline and Prometheus snapshots in post-processing — drift > 1 s
   breaks per-phase alignment.

---

## 10. Per-cluster post-hoc analysis <a name="10-post-hoc"></a>

Beyond the raw JSON events (sliceable by the `ingress` / `cluster` /
`phase` tags), the script itself maintains per-phase custom metrics
(`phase_<name>_latency`, `phase_<name>_requests`, `phase_<name>_errors`)
and per-cluster metrics (`cluster_<name>_latency`, `cluster_<name>_errors`,
`cluster_<name>_requests`), printed as tables in the k6 summary and
archived in `flash_crowd_summary.json`.

The canonical analysis pipeline aggregates the archived runs:

- `experiments/paper_lib.py` — shared library: run discovery by
  `<arm>__<regime>` label, per-phase/per-cluster aggregation, statistics
  across repetitions;
- `experiments/paper_generate_all.py` — regenerates all figures and
  tables from the archived run folders;
- `experiments/collect_dmos_live.py` — the scheduler-side collector whose
  JSONL timeline (Φ scores, replica trajectories, scaling events) is
  correlated with the k6 events for cause-effect analysis.

---

## 11. Legacy Locust workload <a name="11-legacy-locust"></a>

The repository still contains `experiments/locustfile_multiingress.py`
from earlier work. It is **not** used for the canonical evaluation
and is preserved only for reproducing pre-migration runs. New work should
use the k6 scenario described above. There is no plan to keep the Locust
workload in sync with future changes to the test methodology.

---

## 12. References <a name="12-references"></a>

- Active script: [`experiments/k6_flash_crowd.js`](../experiments/k6_flash_crowd.js)
- Per-run orchestrator: [`experiments/run_flash_crowd.sh`](../experiments/run_flash_crowd.sh)
- Scheduler-side collector: [`experiments/collect_dmos_live.py`](../experiments/collect_dmos_live.py)
- Analysis pipeline: [`experiments/paper_lib.py`](../experiments/paper_lib.py) + [`experiments/paper_generate_all.py`](../experiments/paper_generate_all.py)
- Capacity calibration with k6 (different scenario): [`docs/capacity-testing.md`](capacity-testing.md)
- Geographic netem setup: [`scripts/setup-netem-and-ping.sh`](../scripts/setup-netem-and-ping.sh)
- DMOS architecture overview: [`docs/dmos-architecture.md`](dmos-architecture.md)
- Open vs. closed load models, Schroeder/Wierman/Harchol-Balter (NSDI'06)
