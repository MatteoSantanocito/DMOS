# Reach-weighted gate, pooled budget, k6 close to every ingress

Three changes made after the campaign of 29/10/2026 (24 runs: `off`,
`locality`, `locality-ps` on `con`, `bal`, `con-p500`, `con-tosg`). They
target the three findings of that campaign that kept Level 1 from deciding
*where* the replicas go. All three are opt-in: existing arms keep their
`config_id`.

| Change | Switch | Arms |
|---|---|---|
| k6 client exempt from netem | `DMOS_NETEM_CLIENT_LOCAL` (default 1) | all (testbed) |
| pooled entry budget | `DMOS_ENTRY_BUDGET=pooled` | `<arm>-pool` |
| reach-weighted demand in the gate | score model `reach` (`demand_signal: reach`) | `reach`, `reach-pool`, `reach-pool-ps` (the configuration evaluated in the paper) |

## 1. What the 29/10 runs showed

* **The data plane is a pool.** The frontend CPU per replica was the same on
  every cluster in every run (e.g. `off con`: 0.18 core on each of the six),
  whatever the demand entering each cluster: with the Cilium global service and
  affinity `none` a request entering cluster *j* is served by any frontend
  replica, with probability proportional to the replicas of each cluster.
* **The budget always filled every cap.** All 15 DMOS runs ended the peak with
  4/3/2/4/2/3 = 18 frontend replicas (the caps), in every regime. The `total`
  budget sums the per-cluster round-ups: DE ⌈370/33⌉ = 12 plus one replica for
  each of the five clusters with ~15 req/s, 17–18, while the pool needs
  ⌈420/33⌉ = 13. The capacity of 33 req/s per replica is right: at 500 req/s
  each replica served 33 req/s at 77 % of its CPU limit.
* **Replicas went far from the users.** With caps full, 6 of 18 replicas sat on
  US-VA (+100 ms) and SG (+200 ms): 33 %, the same share as `off` (2 of 6). The
  gain over `off` (p95 −11 %) came from more replicas, none from placement.
  The `locality` score could not do better: d·Φ_net is ~0.01 for every cluster
  but the hot one, so the other five get almost equal scores.
* **k6 was far from its own ingress.** netem delayed everything leaving a
  cluster except its own nodes, so the responses to the k6 VM too: a request
  entering SG waited 200 ms on its own ingress (p50 ~490 ms from SG with every
  arm, and the same ~480–516 ms after the hot spot moved to SG).

## 2. Changes

### k6 close to every ingress (`scripts/setup-netem-and-ping.sh`)

The k6 VM (host of `DMOS_K6_REMOTE` or `.k6.remote`; or
`DMOS_NETEM_CLIENT_IPS`) is added to the destinations exempt from the delay,
like the cluster's nodes and storage. The users of each region are then close
to the ingress they enter, and the emulated geography is only between the
clusters, where the placement decides. Every cell checks it
(`capture_env_state.py --expect-client-local`, from `run_common.sh`) and does
not start if a shaped cluster delays the k6 client. `DMOS_NETEM_CLIENT_LOCAL=0`
restores the geography of the 29/10 runs.

**The netem has to be applied again once** (`bash scripts/setup-netem-and-ping.sh netem-only`).
Runs before and after this change are not comparable.

### Pooled budget (`DMOS_ENTRY_BUDGET=pooled`)

```
N = min(Σ max_i, max(Σ min_i, ⌈Σ_i D_i / r̄⌉))      r̄ = Σ max_i·r_i / Σ max_i
```

The same rule in the orchestrator (`DMOSOrchestrator.entry_budget`) and in the
predictive scaler (`PredictiveScaler._need_total`), so a decision is triggered
by the budget it will apply. With the 29/10 peak of `con`: 13 instead of 16–18.

### Reach-weighted demand in the gate (score model `reach`)

The local-demand gate of the model `nca`, `G(d) = (ε + (1−ε)·d)^g`, uses the share of the demand *entering*
each cluster, which is the right signal when each cluster serves its own entry
traffic. With a pool, a replica on cluster *i* serves the users entering *j* at
the cost RTT(j,i). The `reach` signal replaces d_i with

```
ψ_i = Σ_j d_j · exp(−RTT(j,i) / τ)        RTT(i,i) = 0, τ = 50 ms
```

measured with the ping_exporter RTTs between every pair of clusters
(`PrometheusClient.get_peer_rtts_ms`, `DMOSScheduler.refresh_rtt_matrix`).
ψ_i ≥ d_i, and ψ → d when the clusters are far apart: the local-demand gate is the limit
case. A cluster near the hot spot (FR, NO for DE) can now attract replicas, a
far one (SG) cannot. Without any RTT measurement the scheduler falls back to
d (warning in the log). Weights and gate are those of `nca`.

Each decision logs `Reach-weighted demand (τ=50 ms): <cluster>: d=… → ψ=…`.

## 3. Expected effect (simulation with the 29/10 data)

Level-1 allocation (`WinnerDetermination.allocate`) on the score components
logged at the peak of `locality-ps con`, the ping RTTs of the fleet and the
caps; extra network delay = mean RTT between the entry cluster and the serving
replica under pool routing.

| `con` (85 % DE) | replicas DE FR PL US SG NO | total | mean extra RTT | replicas on US+SG |
|---|---|---|---|---|
| `off` | 1 1 1 1 1 1 | 6 | 78 ms | 33 % |
| 29/10 (`locality`, total) | 4 3 2 4 2 3 | 18 | 70 ms | 33 % |
| `locality` + pooled | 3 2 2 3 2 2 | 14 | 75 ms | 36 % |
| `reach` + pooled | 4 3 2 2 1 3 | 15 | 56 ms | 20 % |
| `reach` + pooled, min 0 | 4 3 2 2 0 3 | 14 | 46 ms | 14 % |

`bal`: 95 → 84 ms (reach + pooled). The allocation is the same for τ between
50 and 200 ms; τ = 20–30 ms leaves one replica fewer on US-VA (14 in total). The
pooled budget alone makes things worse: it is the reach gate that decides
where the replicas go. Tests: `tests/test_reach_and_pooled.py`.

## 4. Limits

* Near capacity is small on this fleet: DE + FR + NO + PL hold 12 replicas, so
  at 350 req/s US-VA still gets 1–2. The gain grows when the near clusters can
  hold the whole pool.
* ψ depends on the ping RTTs, which are symmetric only on average; a pair
  measured in one direction only uses that value for both.
* The spike tail of the scaling transients (29/10: p99 +300–700 ms vs `off`
  during the ramp) is not addressed here; it needs a run with
  `DMOS_RAW_OUT=1` to be diagnosed.
