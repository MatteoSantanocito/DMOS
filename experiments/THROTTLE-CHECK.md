# Throttle Check — Documentation

> Manual for the `experiments/throttle_check.sh` tool: what it checks, how it
> works, how to read the result.
>
> **In one sentence**: it checks that, when the frontend is scaled to
> `max_replicas` on a cluster, the replicas do **not** hit CFS throttling,
> i.e. that the DMOS model "cluster capacity = capacity_req_per_sec x
> max_replicas" is **linear and exact**.
>
> **Cross-reference**:
> - `experiments/throttle_check.sh` — the script
> - `experiments/k6_throttle_load.js` — the load generator used by the script
> - `experiments/CAPACITY-GRPC.md` — the companion capacity tests

---

## Contents

1. [What it is and why it is needed](#1-what-it-is-and-why-it-is-needed)
2. [The problem: CFS throttling and the linear model](#2-the-problem-cfs-throttling-and-the-linear-model)
3. [How the script works](#3-how-the-script-works)
4. [The key metric: CFS throttled periods](#4-the-key-metric-cfs-throttled-periods)
5. [Interpreting the verdict](#5-interpreting-the-verdict)
6. [Usage: commands and env variables](#6-usage-commands-and-env-variables)
7. [Example output](#7-example-output)
8. [What to do if the verdict is FAIL](#8-what-to-do-if-the-verdict-is-fail)
9. [Throttle check vs capacity test](#9-throttle-check-vs-capacity-test)
10. [When to rerun it](#10-when-to-rerun-it)

---

## 1. What it is and why it is needed

The DMOS fleet is heterogeneous, with 3 tiers:

| Tier | Worker | frontend `max_replicas` |
|---|---|---|
| HIGH | 8 vCPU | 6 |
| MEDIUM | 6 vCPU | 5 |
| LOW | 4 vCPU | 3 |

(HIGH: physically the worker could hold 8 frontend replicas; the cap is 6 to
leave room for the co-located backends — see the comment in
`config/services.yaml`.)

DMOS models the capacity of a cluster as:

```
cluster capacity = capacity_req_per_sec (60) x max_replicas_per_cluster
```

This model is **linear**: 8 replicas give 8x the throughput of 1 replica. But
it is linear **only if** the worker has enough CPU for every replica to run
without being choked. If `max_replicas` is set too high for that worker, the
replicas hit **CFS throttling** and the throughput becomes **sublinear** (N
replicas give less than Nx).

`throttle_check.sh` verifies empirically that the chosen `max_replicas`
(6/5/3) are below the throttling threshold. It is a **lightweight** sanity
check (~4 min per cluster), to be run after every re-provisioning.

---

## 2. The problem: CFS throttling and the linear model

### 2.1 What CFS throttling is

The Linux kernel uses **CFS** (Completely Fair Scheduler) to share the CPU.
When a container has a **CPU limit** (the `frontend` pod has a 500m limit),
the kernel grants it a **quota** of CPU time for every **period** (default
100 ms):

- limit 500m = 0.5 core = 50 ms of CPU every 100 ms period
- if the container uses up all 50 ms before the end of the period, it is
  **throttled**: suspended until the start of the next period

Throttling is NOT an error: it is the CPU limit doing its job. But if a pod
is throttled often, it means it **wants more CPU than it is granted** -> its
latency rises and its throughput drops.

### 2.2 Why it affects the linear model

`capacity_req_per_sec = 60` was measured with the **capacity test** on 1
**isolated** replica (idle worker). In that regime the pod always gets the
CPU it needs.

When DMOS scales to N replicas, there are two regimes:

| Regime | Condition | Capacity per replica |
|---|---|---|
| Linear | `N x 500m + other pods <= worker CPU` | ~60 rps (uniform) |
| Sublinear | the worker saturates, the replicas compete | < 60 rps (degrades) |

The `throttle_check` tells the two regimes apart: if at `max_replicas` + 80%
load the throttle ratio is ~0, we are in the **linear** regime -> the DMOS
model is exact. If the ratio is high, we are in the **sublinear** regime ->
`max_replicas` is too high and must be reduced.

### 2.3 CPU budget (why we expect throttling ~0)

Budget for the 3 tiers (frontend pod CPU limit = 500m):

| Tier | Worker | frontend (max_repl x 500m) | other pods | Total | Utilisation |
|---|---|---|---|---|---|
| HIGH | 8000m | 6 x 500 = 3000m | ~2200m | 5200m | 65% |
| MEDIUM | 6000m | 5 x 500 = 2500m | ~2200m | 4700m | 78% |
| LOW | 4000m | 3 x 500 = 1500m | ~2200m | 3700m | 93% |

Every tier stays below the worker's CPU (worst-case budget at full limit:
actual usage is well below the limits) -> we expect a throttle ratio of ~0 on
all of them. The `throttle_check` confirms it empirically.

---

## 3. How the script works

For the target cluster, `throttle_check.sh` runs 8 steps:

1. **Pre-flight** — checks `k6`, `kubectl`, `jq`, `curl`, `python`; finds the
   kubeconfig and the Prometheus URL of the cluster.
2. **Reads `max_replicas`** of the cluster from `config/services.yaml`
   (`frontend.max_replicas_per_cluster`).
3. **Scales the frontend** to `max_replicas` replicas on the target cluster
   and waits for the rollout to complete.
4. **Prometheus PRE snapshot** — reads the cumulative CFS counters.
5. **Generates load** with `k6_throttle_load.js`: a constant rate at 80% of
   capacity (= 0.8 x 60 x max_replicas) for ~3 min.
6. **Prometheus POST snapshot** — reads the CFS counters + CPU used again.
7. **Computes the throttle ratio** and gives the verdict.
8. **Restore** — scales the frontend back to 1 replica (also on Ctrl+C, via a
   trap).

Load generated per tier (80% of capacity):

| Tier | max_replicas | Capacity (60 x N) | 80% load |
|---|---|---|---|
| HIGH | 6 | 360 rps | 288 rps |
| MEDIUM | 5 | 300 rps | 240 rps |
| LOW | 3 | 180 rps | 144 rps |

---

## 4. The key metric: CFS throttled periods

The script queries two cAdvisor counters (exposed by the kubelet, scraped by
Prometheus):

| Metric | Meaning |
|---|---|
| `container_cpu_cfs_periods_total` | total number of CFS periods elapsed for the container |
| `container_cpu_cfs_throttled_periods_total` | number of periods in which the container was throttled (hit its CPU limit) |

The **throttle ratio** is:

```
                sum(rate(container_cpu_cfs_throttled_periods_total{pod=~"frontend-.*"}[2m]))
throttle_ratio = ----------------------------------------------------------------------------
                sum(rate(container_cpu_cfs_periods_total{pod=~"frontend-.*"}[2m]))
```

Interpretation:
- **ratio = 0.00** -> the frontend pods NEVER hit their CPU limit. They have
  all the CPU they need. Perfectly linear model.
- **ratio = 0.10** -> 10% of the CFS periods are throttled. The pods
  sometimes hit the limit. Mild throttling.
- **ratio = 0.50** -> half of the periods are throttled. The pods are
  CPU-starved. Their effective capacity is far below 60 rps. Sublinear model.

The script uses a `[2m]` window (the last 2 minutes = the steady state of the
load generated for 3 min).

---

## 5. Interpreting the verdict

The script compares the throttle ratio with the `DMOS_THROTTLE_WARN`
threshold (default 0.05 = 5%) and produces 3 verdicts (the FAIL threshold is
3 x `DMOS_THROTTLE_WARN`):

| Verdict | Condition | Meaning | Action |
|---|---|---|---|
| **OK** | ratio < 5% | linear model confirmed | none, proceed |
| **WARN** | 5% <= ratio < 15% | mild throttling, worker close to saturation | consider lowering `max_replicas` by 1, or accept it (Φ_cap partly compensates at runtime) |
| **FAIL** | ratio >= 15% | significant throttling, sublinear model | **lower `max_replicas_per_cluster`** for that cluster |

Script exit code: `0` = OK/WARN, `1` = FAIL, `2` = missing prerequisites or
throttle ratio not available.

---

## 6. Usage: commands and env variables

### Basic command

```bash
cd <REPO_ROOT>
bash experiments/throttle_check.sh --cluster ms-cluster-01
```

### Validating all 3 tiers (after a re-provisioning)

```bash
bash experiments/throttle_check.sh --cluster ms-cluster-01   # HIGH
bash experiments/throttle_check.sh --cluster ms-cluster-02   # MEDIUM
bash experiments/throttle_check.sh --cluster ms-cluster-03   # LOW
```

One representative cluster per tier is enough: the others of the same tier
have identical hardware.

### Env variables / flags

| Flag / env | Default | Description |
|---|---|---|
| `--cluster <name>` / `DMOS_CLUSTER` | ms-cluster-01 | cluster to test |
| `--load-pct <n>` / `DMOS_LOAD_PCT` | 80 | % of capacity to generate as load |
| `--duration <s>` / `DMOS_LOAD_DURATION_S` | 180 | duration of the load in seconds |
| `DMOS_CAP_PER_REPLICA` | 60 | capacity per replica (from the capacity test) |
| `DMOS_THROTTLE_WARN` | 0.05 | ratio threshold for the WARN verdict |

The script reads `max_replicas` from `config/services.yaml` only; on a fleet
calibrated through `config/local.yaml`, set `DMOS_CAP_PER_REPLICA` to the
calibrated frontend capacity.

Example: a more aggressive test (95% load, 4 min):
```bash
bash experiments/throttle_check.sh --cluster ms-cluster-03 --load-pct 95 --duration 240
```

---

## 7. Example output

```
=== Pre-flight - throttle check su ms-cluster-03 ===
  [OK]   Tool presenti
  [OK]   Cluster target: ms-cluster-03
  [OK]   IP=<CLUSTER_IP>  Prometheus=http://<CLUSTER_IP>:32090

=== Lettura max_replicas per ms-cluster-03 ===
  [OK]   max_replicas=3  capacita=180 rps  carico target (80%)=144 rps

=== Scale frontend -> 3 repliche su ms-cluster-03 ===
  [OK]   frontend Ready con 3 repliche

=== Snapshot Prometheus PRE-carico ===
  CFS throttled periods (cumul): 1240
  CFS total periods (cumul):     58300

=== Genero carico 144 rps per 180s ===
  ... (k6 constant-arrival-rate) ...

=== Snapshot Prometheus POST-carico ===
  Throttle ratio (frontend, 2m): 0.018
  CPU usata da frontend:         0.62 cores
  CPU usata da tutti i pod:      2.74 cores

=== Verdetto - ms-cluster-03 (max_replicas=3) ===
  [OK]   Throttle ratio 1.80% < soglia 5.0%
  [OK]   MODELLO LINEARE CONFERMATO: capacita = 60 x 3 = 180 rps

  Le 3 repliche frontend non sono CPU-throttled a 80% di carico.
  max_replicas_per_cluster=3 e' sotto la soglia fisica del worker. OK.
```

---

## 8. What to do if the verdict is FAIL

If a tier comes out FAIL (throttle ratio >= 15%):

1. **Lower `max_replicas_per_cluster`** for that cluster in
   `config/services.yaml`. For example, if LOW is FAIL at max_replicas=3:
   ```yaml
   max_replicas_per_cluster:
     ms-cluster-03: 2     # was 3
     ms-cluster-05: 2     # was 3 (same tier)
   ```

2. **Rerun the check** on that cluster:
   ```bash
   bash experiments/throttle_check.sh --cluster ms-cluster-03
   ```

3. Repeat until the verdict is OK.

Possible causes of a FAIL:
- `max_replicas` simply too high for that worker
- The Online Boutique backends consume more CPU than expected
- Prometheus / kube-prometheus-stack spiking on CPU under load
- A worker with less CPU than expected (check with `kubectl get nodes`)

Note: with a small residual WARN, DMOS partly compensates at runtime through
Φ_cap: its slack term (the `cpu_available_fraction` and memory fraction read
live from Prometheus) lowers the Level 1 placement score of a saturated
cluster, in the score models that give Φ_cap a non-zero weight. It does not
stop scaling on that cluster. The `throttle_check` is there to calibrate the
static model properly; the dynamic component is only a safety net.

---

## 9. Throttle check vs capacity test

They are two different tools, do not confuse them:

| | Capacity test (`run_capacity.sh`) | Throttle check (`throttle_check.sh`) |
|---|---|---|
| **Question** | how much can **1 replica** sustain? | do the **N replicas** get throttled? |
| **Scales the frontend to** | 1 replica (isolated) | `max_replicas` |
| **Verdict based on** | **client-side** p95 latency (k6) | **server-side** CFS throttle ratio (Prometheus) |
| **Output** | `capacity_req_per_sec` for services.yaml | OK/WARN/FAIL verdict |
| **Duration** | ~17 min | ~4 min |
| **netem** | must be run **WITHOUT** netem (see below) | **no difference**, can be left ON |
| **When** | once (pod capacity) | after every re-provisioning |
| **Must be redone if** | the pod's CPU limit / the workload changes | the workers' hardware or `max_replicas` changes |

The capacity test establishes **the number `60`**. The throttle check
verifies that **multiplying it by `max_replicas`** gives a genuinely linear
result.

### 9.1 netem: why the throttle check is NOT affected

The `throttle_check` bases its verdict on the **CFS throttle ratio**, a
server-side CPU metric (`container_cpu_cfs_throttled_periods_total`, scraped
by Prometheus). tc netem acts on the **network interface** (egress of
`eth0`): it delays packets on the wire, but **does not make the CPU work
harder**. The work of "processing N requests/s" is identical with or without
netem.

So the throttle check can be run with netem active. In fact, since the flash
crowd test runs WITH netem, running the throttle check in the same state is
**more representative** of the real conditions.

The **capacity test**, by contrast, bases its verdict on the **client-side
p95 latency**: netem adds milliseconds to every response -> the p95 is
inflated -> the measured knee point is wrong. This is why the capacity test
must be run **without netem** (`bash scripts/setup-netem-and-ping.sh
remove-netem` before, then `netem-only` after).

Rule of thumb:
- verdict from **client latency** -> netem skews it -> **netem OFF**
- verdict from a **server CPU metric** -> netem does not touch it -> **netem ON is fine**

---

## 10. When to rerun it

Run `throttle_check.sh` (3 tiers):

- **After every re-provisioning** of the fleet
- **After changing `max_replicas_per_cluster`** in `services.yaml`
- **After changing the hardware specs** of the workers
- It does **NOT** need to be rerun before every flash crowd test: once the
  `max_replicas` are validated, they stay valid until the hardware changes

---

*Update this document if the logic of `throttle_check.sh` or the DMOS capacity
model changes.*
