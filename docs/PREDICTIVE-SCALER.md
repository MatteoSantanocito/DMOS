# Predictive Level-2 scaler (DMOS-PS)

DMOS has two levels. **Level 1** decides *where* the replicas of the entry
service go (the multi-objective score: latency, capacity, network proximity ×
demand, carbon). **Level 2** decides *how many* replicas are needed and *when*
to change them. This document describes the predictive Level 2 introduced for
the journal extension (`DMOS_SCALER=predictive`, arms `<arm>-ps`), why it was
needed, how it works, how its parameters were chosen and how it is evaluated.
The earlier reactive Level 2 stays available unchanged
(`DMOS_SCALER=reactive`, the default), so every legacy arm kept for comparison
keeps its meaning and its `config_id`. The configuration evaluated in the paper
is the arm `reach-pool-ps`: score model `reach`, pooled budget, predictive
scaler.

Code: `src/level2/predictive_scaler.py` (policy), `src/dmos_main.py`
(`_predictive_tick`, `_apply_predictive_targets`), `experiments/scaler_replay.py`
(offline replay), `tests/test_predictive_scaler.py`.

---

## 1. Problem: the reactive Level 2 serves a flash crowd 60–70 s late

On the 6-cluster Kubephos fleet (journal testbed, netem geography, Cilium 1.20
global service with affinity `none`), the arm `locality` with the reactive
scaler improved the peak of a flash crowd (p95 −12 % vs. no DMOS) but made the
spike worse (spike p99 1506 ms vs. ~700 ms without DMOS). The DMOS log of the
run (`dmos-locality__con`) gives the timeline from the start of the
spike:

| t from spike start | event | cause |
|---|---|---|
| +3 s | DMOS still reads 60 req/s | the demand is `rate(...[1m])`: the spike is not in the window yet |
| **+34 s** | DMOS reads 66 req/s, the offered load is ~130 | a 1-minute mean on a ramp lags by ~30 s; the next decision waits for the 30 s polling cycle |
| +36 s | first scale-up (DE 1→3, US-VA 1→2) | budget from the *measured* demand; the trend prediction (297 req/s at +64 s) is computed but not used for the budget |
| **+68 s** | the new pods are ready | pod start + readiness probe every 15 s |
| +66, +96, +126 s | about +1 replica per cluster per cycle until the caps | PD correction clamped to ±1 on top of the L1 target, 30 s cycle |

During the 120 s of the spike the reactive scaler issued 11 frontend scaling
actions and co-scaled 16 backend pods (among them `adservice`, JVM, and
`cartservice`, .NET, whose start burns CPU on the node the frontend needs).
Each action also changes the endpoints of the global service on every cluster
of the mesh. The arm without DMOS, which never scales, had the better spike
tail: **the cost is not only being late, it is scaling in many small steps
while the load is high.**

The measurement delays caused by netem on DMOS's own Prometheus queries are
not the cause: a full decision cycle (6 clusters, queries in parallel) took
~3 s, of which netem accounts for 1–2 s.

---

## 2. Design

The predictive scaler keeps the Level-1 placement and replaces the Level-2
loop of the entry service. Each element follows an established technique.

### 2.1 Two demand windows (stable and fast)

Every tick (`tick_s` = 10 s) DMOS reads the demand entering each cluster
(client-side Hubble L7 metric, `PrometheusClient.get_ingress_rate`) over two
windows: the historical `rate[1m]` (stable) and `rate[30s]` (fast). With the
15 s scrape the fast window has 2–3 samples: it sees a ramp ~15–20 s earlier
than the stable one, at the price of more noise, which the forecaster filters.
This is the stable/panic window pair of the Knative Pod Autoscaler (60 s / 6 s
by default) adapted to a 15 s scrape.

### 2.2 Damped-trend Holt forecaster per cluster

On the fast signal `x_t` of each cluster (irregular sampling, `dt` seconds):

```
one-step forecast   f = l + b·dt                    (error e = x - f)
level               l' = α x + (1 - α) f
trend               b' = β (l' - l)/dt + (1 - β) b      (req/s per second)
forecast at H       λ̂(t+H) = l + max(0, b) · τ (1 - e^(-H/τ))
```

Holt's linear method (double exponential smoothing) outperforms single
exponential smoothing for proactive Kubernetes up-scaling (Gutman & Syrota,
2023). The trend is **damped** with time constant τ = `trend_damping_s` = 120 s
(Gardner & McKenzie, 1985): a ramp seen in the last samples is not extrapolated
linearly for minutes. Only a positive trend is used for sizing; a falling
demand is handled by the stabilized scale-down (§2.6).

### 2.3 Horizon = actuation lag, learned online

The forecast is evaluated at the time when the replicas requested now will
serve:

```
H = tick_s + T_ready
```

where `T_ready` is the **actuation lag** (scale command → `readyReplicas`).
It is learned online from the scaler's own scale-ups (EWMA, weight 0.3; prior
30 s; bounded to 5–180 s): after each scale-up the tick reads the ready
replicas of that cluster until the requested number is ready. A fixed horizon
can be forced with `DMOS_PS_HORIZON_S` (sensitivity studies).

### 2.4 Uncertainty headroom

The demand used for sizing adds a headroom proportional to the forecaster's
recent one-step error:

```
D_i = max( measured_i , min(λ̂_i, 2 · measured_i) + z · 1.25 · MAD_i )
```

with `measured_i = max(fast_i, stable_i)`, `MAD_i` an EWMA of |e| (1.25·MAD ≈ σ
for a normal error) and z = 1.5. The headroom is large when the demand moves
and the forecast is wrong, and vanishes on a steady demand. This is the safety
margin Google Autopilot adds to usage percentiles (Rzadca et al., EuroSys
2020), made adaptive. The cap at twice the measured demand bounds the
overshoot of a very steep ramp. Below 1 req/s (health checks) no trend is used.

### 2.5 Capacity-model sizing (the same model as the replica caps)

Replicas needed per cluster: `need_i = ceil(D_i / r_i)`, within
`[min_i, max_i]`, where `r_i` is the calibrated req/s of one entry replica at
its target CPU (`capacity_req_per_sec` in `config/local.yaml`, measured by
`scripts/kubephos_fleet.py calibrate`) and `max_i` the per-cluster replica cap
computed from the same calibration (`docs/KUBEPHOS.md`, "Replica cap"). The
entry budget is `N = min(Σ max_i, Σ need_i)` (`DMOSOrchestrator.entry_budget`,
mode `total`) and Level 1 distributes N over the clusters by score. One
calibrated model therefore gives both the upper bound and the requirement.

### 2.6 Event-driven decisions, burst hold, asymmetric actuation

* **When.** A decision is taken as soon as the forecast needs more replicas
  than the deployments have (`need > current`, at most every
  `min_decision_spacing_s` = 10 s), otherwise every `decision_interval_s` = 30 s.
  `need` is the budget the decision will compute (`entry_budget`): with the
  default `total` budget the demand of a cluster above its cap counts too,
  because Level 1 places that excess on the other clusters. (Counting each
  cluster only up to its cap delayed the step from 10 to 15 replicas by 60 s
  in the first cluster run.)
* **Burst.** If `need ≥ panic_ratio · current` (1.5) the scaler enters a burst
  hold for `panic_hold_s` = 60 s: no scale-down during it (as the Knative panic
  mode, which does not scale down while panicking).
* **Up.** Straight to the Level-1 target in one decision (no ±1 clamp, no
  per-cycle step), bounded by the per-cluster cap and by
  `up_budget_per_decision` = 40 replicas overall.
* **Down.** Through a stabilization window: the applied target is the maximum
  recommendation of the last `down_window_s` = 120 s (the HPA `scaleDown`
  stabilization; its default is 300 s), then at most `max_down_step` = 1
  replica per cluster and `down_budget_per_decision` = 3 overall per decision,
  so the gRPC connections of the removed pods drain.
* **Level 2 says how many, Level 1 where (policy 4).** The hysteresis acts on
  the *budget*: while the fleet demand is not falling (total stable demand
  ≥ `falling_frac` = 90 % of its maximum over the window) and the budget is
  above the fleet minimum, the budget drops by at most `down_tolerance_frac`
  = 10 % (at least one replica) below the largest budget since the demand last
  fell (`hold_budget`). The forecast and the headroom add a replica or two
  during a ramp; without hysteresis that surplus was removed on the plateau
  that follows (second cluster run: US-VA 4→3 at t ≈ 483 s, with its
  backends), i.e. exactly while the load is highest. Level 1 places the held
  budget like any other, so the surplus sits on the best clusters.
* **Moves of Level 1, made before broken.** When the total recommendation does
  not grow, a scale-up is Level 1 moving a replica between clusters: it is
  applied only once the cluster has been recommended more for
  `move_confirm_s` = 60 s, so a placement that flips between two clusters with
  close scores (FR 3→2→3 every 40 s in a pilot run) starts and stops no pod. The
  replica where it was is removed only after the new one is ready: no
  scale-down in a decision that scales up or while a scale-up is not ready,
  and the fleet never goes below the total recommendation stabilized over the
  window (fleet guard). In the first cluster run, without the guard, US-VA lost
  a frontend in the middle of the peak and got it back in the decay,
  restarting six co-scaled backends each time.
* **Order of the scale-downs.** The clusters furthest above their stabilized
  target first, then those Level 1 recommends least: a cluster recommended 0
  (arm `-min0`) is emptied.
* **Pooled budget: forecast of the total.** With `DMOS_ENTRY_BUDGET=pooled` the
  budget is the size of the total demand, so the total is forecast as one
  series (`total_forecaster`) and split back over the clusters by their
  measured share. Summing per-cluster forecasts counted the rise of the
  clusters that gain demand and not the fall of the one that loses it (the
  trend of a cluster is used only when positive): when the demand moves
  between clusters at constant total (end of the peak of a flash crowd), the
  sum asked up to 2 replicas more on the clusters (+2 in all 3 `con` pilot
  runs) and 40 % more in the unit test of the same situation.

Policy 3 (superseded by policy 4) kept a surplus of 10 % of the stabilized
total per cluster: the replica left where Level 1 moved it away was never
removed, so the fleet stayed 2–3 replicas above the budget at the peak, on the
clusters Level 1 had left (US and SG in `reach-pool-ps-min0 con`: L1 14, 16
running for 4 minutes; SG never emptied).

The policy logic has a revision number (`POLICY_VERSION`, recorded in the
provenance as `level2.predictive_scaler.policy_version`, part of the
`config_id`): 1 = first cluster run (superseded), 2 = overflow-aware need and
fleet guard, 3 = hysteresis per cluster, 4 = hysteresis on the budget,
confirmed moves made before broken, pooled forecast of the total.

This replaces the reactive safeguards (dead zone, scale-down cooldown,
scale-up protection, per-cycle ±1 PD clamp, per-cycle budgets) with two
mechanisms that have a precise meaning: immediate scale-up to the target,
stabilized and bounded scale-down.

### 2.7 Readiness probe (testbed, all arms)

The testbed hardening had relaxed liveness and readiness probes to 15 s
periods to avoid killing CPU-throttled pods (`experiments/apply_resource_hardening.sh`).
A new pod therefore became Ready up to 15 s after it could serve. The
readiness period is now 5 s with 18 allowed failures (the same 90 s of
tolerance); the liveness stays at 15 s × 6. This applies to every arm that
scales (DMOS reactive and predictive, HPA, Karmada); the arm without DMOS does
not create pods during a run.

---

## 3. Parameters

All parameters are fields of `PredictiveScalerParams` and can be overridden
with `DMOS_PS_<NAME>` (upper case), e.g. `DMOS_PS_DOWN_WINDOW_S=300`. They are
written in the run provenance (`dmos_config.json` →
`level2.predictive_scaler.params`) and enter the `config_id`.

| parameter | default | meaning |
|---|---|---|
| `tick_s` | 10 | measurement + forecast period (s) |
| `decision_interval_s` | 30 | periodic decision when nothing is urgent (s) |
| `min_decision_spacing_s` | 10 | minimum time between two decisions (s) |
| `fast_window` / `stable_window` | 30s / 1m | PromQL windows of the demand |
| `alpha`, `beta` | 0.7, 0.6 | Holt level and trend smoothing (chosen offline, §5.3) |
| `trend_damping_s` | 120 | τ of the damped trend (s) |
| `horizon_s` | 0 | 0 = learned (`tick_s + T_ready`); > 0 = fixed horizon (s) |
| `t_ready_init_s`, `t_ready_ewma` | 30, 0.3 | prior and EWMA weight of the actuation lag |
| `t_ready_min_s`, `t_ready_max_s` | 5, 180 | bounds of one lag sample; a scale-up not ready after 180 s is dropped |
| `headroom_z`, `mad_ewma` | 1.5, 0.2 | headroom = z·1.25·MAD; EWMA weight of the error |
| `max_forecast_multiple` | 2.0 | forecast ≤ 2 × measured |
| `min_demand_rps` | 1.0 | below: no trend (noise) |
| `panic_ratio`, `panic_hold_s` | 1.5, 60 | burst: need ≥ 1.5 × current; no scale-down for 60 s |
| `down_window_s` | 120 | scale-down stabilization window (s) |
| `down_tolerance_frac`, `falling_frac` | 0.10, 0.90 | hysteresis on the budget: drop of at most 10 % while the demand is ≥ 90 % of its window maximum |
| `move_confirm_s` | 60 | a Level-1 move (same total) is applied after this long |
| `max_down_step`, `down_budget_per_decision` | 1, 3 | replicas removed per cluster / overall per decision |
| `up_budget_per_decision` | 40 | sanity bound on replicas added per decision |

---

## 4. Implementation

| piece | where |
|---|---|
| policy: forecaster, sizing, decision gating, actuation, lag learning | `src/level2/predictive_scaler.py` (`HoltForecaster`, `PredictiveScaler`, `PredictiveScalerParams`) |
| tick (every 10 s): fast/stable demand, current and ready replicas, decision | `DMOSOrchestrator._predictive_tick` |
| budget on the sized demand, Level 1, actuation, co-scaling of the backends | `DMOSOrchestrator.schedule_service` → `_apply_predictive_targets` → `_enforce_colocation` |
| fast PromQL window | `PrometheusClient.ingress_queries(window=...)` |
| ready replicas | `KubernetesClient.get_deployment_ready_replicas` |
| provenance (`modes.scaler`, `level2.predictive_scaler`, summary `scaler=predictive`) | `src/utils/provenance.py` (recorded only when selected) |
| arm suffix `-ps` | `experiments/run_naming.py` |
| timeline (`predictive_scaler` per snapshot: fast, forecast, demand, need, horizon, lag, burst) | `experiments/collect_dmos_live.py` |

The policy object is pure: it receives measurements and replica counts and
returns targets, and holds a lock because the orchestrator ticks in one thread
and decides in a worker thread. The same class runs on the clusters and in the
offline replay. The co-scaled backends follow the frontend replicas exactly as
with the reactive scaler; their churn budget is reset per decision.

Prometheus metrics exported by DMOS (`:9090/metrics`): `dmos_ps_fast_rps`,
`dmos_ps_forecast_rps`, `dmos_ps_demand_rps`, `dmos_ps_need_replicas`
(per cluster), `dmos_ps_horizon_seconds`, `dmos_ps_actuation_lag_seconds`,
`dmos_ps_burst`; `dmos_predicted_traffic` carries the sized demand.

---

## 5. Offline evaluation (`experiments/scaler_replay.py`)

### 5.1 Model

The replay simulates at 1 s resolution the demand of a k6 flash crowd per
cluster (phases 180/300/600/720/900 s, regime weights of `run_naming.py`), the
way DMOS measures it (Hubble counters scraped every 15 s: a query at t sees the
counter up to the last scrape; `rate[w]` is the mean over `[s−w, s]`; with
`--noise-seed` the requests of each second are Poisson counts, as Hubble
counts them) and the actuation lag of new replicas. Both scalers see the same
demand and the same Level-1 stand-in (water-filling of the budget within the
caps). The **predictive** scaler is the production class; the **reactive** one
is a model of the decision rules of `dmos_main.py` (30 s cycle on `rate[1m]`,
per-cluster trend predictor and PD +1 outside the 0.2 deadband, +6/−1 per
cycle, 90 s cooldown and protection).

Validation of the reactive model against the cluster run `dmos-locality__con`
(spike from t = 180 s; total frontend replicas requested):

| decision | replay | cluster run |
|---|---|---|
| 1st | t = 210 s → 8 | t ≈ 216 s → 9 |
| 2nd | 240 s → 10 | ≈ 246 s → 11 |
| 3rd | 270 s → 13 | ≈ 276 s → 16 |
| 4th | 300 s → 17 | ≈ 306 s → 18 |

### 5.2 Results (6 clusters, caps 4/3/2/4/2/3, r = 33 req/s, T_ready = 30 s, Poisson noise)

Metrics: `over_target_req` = requests above the ready capacity at the target
CPU (r per replica: pods running hotter than the target); `over_limit_req` =
above the CPU limit (r/0.7: requests that queue); decisions and replicas added
*while the demand is above twice the baseline*; replica-minutes (cost).

| regime | scaler | over target | over limit | decisions under load | replicas added under load | replica-min |
|---|---|---|---|---|---|---|
| con @350 | reactive | 89 | 0 | 8 | 13 | 167.0 |
| con @350 | **predictive** | **0** | 0 | **5** | **9** | 184.8 |
| con @500 | reactive | 5230 | 9 | 5 | 12 | 188.0 |
| con @500 | **predictive** | **200** | 0 | 4 | 12 | 205.3 |

At 350 req/s the fleet is never short of capacity with either scaler; the
predictive one scales *earlier* (during the ramp, at low load), so fewer
decisions and fewer new pods fall while the demand is high. At 500 req/s it
removes 96 % of the requests above the target capacity. The cost is +10 %
replica-minutes (forecast headroom and the 120 s stabilization window on the
way down).

### 5.2b Policy 4 against policy 3 (pooled budget, T_ready = 20 s, 5 noise seeds)

The replay now switches to the baseline weights at the start of the decay
(600 s), as `k6_flash_crowd.js` does; before policy 4 it kept the regime weights
until 720 s and could not show the redistribution of the decay. The tables
above were computed with the old model.

| regime | policy | over target | over limit | time to peak capacity (s) | replicas added under load | replica-min | excess replica-min | replicas added in the decay |
|---|---|---|---|---|---|---|---|---|
| con @350 | 3 | 240 | 0 | 144 | 12.2 | 170.6 | 50.2 | +5.0 |
| con @350 | **4** | 310 | 0 | 144 | **7.2** | **151.2** | **30.9** | **0** |
| bal @350 | 3 | 310 | 0 | 148 | 7.6 | 151.3 | 31.1 | 0 |
| bal @350 | 4 | 310 | 0 | 152 | 7.6 | 152.4 | 32.1 | 0 |
| con @250 | 3 | 1 | 0 | 246 | 10.4 | 139.5 | 38.1 | +2.6 |
| con @250 | **4** | 1 | 0 | **182** | **5.2** | **122.7** | **21.3** | **0** |
| bal @250 | 3 | 1 | 0 | 258 | 5.8 | 122.5 | 21.1 | +0.6 |
| bal @250 | **4** | 1 | 0 | **222** | **4.2** | **119.6** | **18.2** | **0** |

Policy 4 adds no replica in the decay, starts up to 40 % fewer pods under
load and uses 11–12 % fewer replica-minutes in `con`; it reaches the peak
capacity 36–64 s earlier at 250 req/s. The cost: at `con` @350 70 more
requests (of ~250 000) run above the target CPU during the spike, none above
the limit (the headroom of one total forecaster is smaller than the sum of
six). The replay has no Level-1 flipping and no `-min0`: those are covered by
the unit tests (`tests/test_predictive_scaler.py`) and must be checked on the
clusters.

### 5.3 Parameter choice and ablation

α, β and z were chosen on a grid (α ∈ {0.5…0.8}, β ∈ {0.3…0.6},
z ∈ {1, 1.5, 2}) over 24 replays (regimes `con`/`bal` × peaks 350/500 ×
T_ready 25/40 s × 3 noise seeds), minimizing the requests above target with
the cost as tie-break. The best region is α 0.7–0.8, β 0.5–0.6, z 1.5–2; the
default (0.7, 0.6, 1.5) sits in its middle. Mean over the 24 replays:

| scaler | over target | excess replica-min | decisions per run |
|---|---|---|---|
| reactive | 2301 | 38.2 | 9.3 |
| predictive (0.7, 0.6, 1.5) | 688 (−70 %) | 54.6 | 10.3 |

Noise does not make the predictive scaler oscillate (decisions per run close
to the reactive ones). Ablation at con @500 (one component changed at a time):

| variant | over target | note |
|---|---|---|
| predictive (default) | 200 | |
| no forecast (H ≈ 0) | 3010 | the forecast is the main contribution |
| fast window 60 s instead of 30 s | 1643 | the fast signal matters |
| tick 30 s instead of 10 s | 1166 | event-driven decisions matter |
| no headroom (z = 0) | 269 | small effect on this profile |
| down window 60 s / 300 s | 200 / 200 | cost 193.8 / 218.3 replica-min |

Reproduce: `python experiments/scaler_replay.py --peak 500 --noise-seed 1 --sweep`.

---

## 6. Evaluation on the clusters

Arms (`bash experiments/run_one.sh <arm> <regime>`; the paper's campaign runs
them with `bash experiments/run_campaign.sh experiments/plans/2026-11-02-definitive.plan`,
k6 on a VM next to the clusters, `DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>`):

* `off` — no DMOS; `hpa-none` — Kubernetes HPA (baselines);
* `reach-pool-ps` — the paper's DMOS; `l2-pool-ps` (same budget and scaler,
  uniform placement), `nca-pool-ps`, `locality-ps`, `reach-pool-ps-min0`,
  `reach_green-pool-ps-min0` — the other arms of the campaign;
* `locality`, `nca`, `gate_g0` (reactive Level 2) and `locality-ps`, `nca-ps`,
  `gate_g0-ps` (the same Level 1 with the predictive scaler) — legacy
  configurations kept for comparison.

Paired comparison `<arm>` vs `<arm>-ps` isolates the Level-2 contribution.
Metrics per phase from `flash_crowd_summary.json` (p50/p95/p99, errors) and,
from `dmos_live.jsonl`, the time from the start of the spike until the
frontend replicas reach their peak, the replica-minutes, the scaling actions
during the spike and the learned actuation lag (`predictive_scaler.actuation_lag_s`).
Five repetitions per cell in the campaign.

### 6.1 First cluster runs (`con` @350 req/s)

Behaviour of the scaler (from `dmos.log` and `dmos_live.jsonl`; spike from t = 180 s):

| | reactive (`locality`) | predictive v3 (`locality-ps`) |
|---|---|---|
| first scale-up after the start of the spike | +36 s | +15 s (v2/v3) |
| learned actuation lag (scale command → ready) | — | 19 s (readiness probe 5 s) |
| 18 frontends (all caps) reached | t ≈ 341 s | t = 354 s |
| replicas removed during peak and decay | none (dead zone) | none (v3; v1/v2 removed and re-added one, see §2.6) |
| scale-down after the run | from t ≈ 700 s | from t = 838 s, when the demand falls |

Latency: the only pair measured on a clean client path is reactive
vs. predictive v2: spike p99 1506 → 1002 ms, peak p99
950 → 1319 ms (v2 still removed a replica in the middle of the peak, fixed in
v3). In the later runs of that series the load generator ran on a remote
client over a VPN, and its path dominated the latency under load: the same
reactive configuration went from peak p99 950 ms to 5254 ms with unchanged
app-node CPU and throttling, and k6 spent up to 2.5 s receiving response
bodies. Those runs are kept for the scaler behaviour only. **The latency
evaluation needs the load generator next to the clusters** (k6 VM,
`DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>`).

---

## 7. Limits and threats to validity

* The parameters were chosen on a simulated demand (k6 profile + Poisson
  noise) and the reactive scaler of the replay is a model of its rules; the
  cluster runs are the evaluation, the replay only explains and tunes.
* The forecaster sees only the ramp it is in: a step change of the demand
  (instead of a ramp) is caught by the fast window and the burst trigger, not
  by the trend.
* `r` is calibrated once per fleet; with hosts shared by two app VMs the
  marginal CPU per request differs per cluster (29–62 mcore/req on the
  journal fleet), so the capacity model is an average.
* The scale-down is deliberately slow (120 s window, 1 replica per cluster per
  decision): it trades replica-minutes for stability at the end of a peak.
* The readiness-probe change (§2.7) also helps every scaling baseline; it is
  part of the testbed, not of DMOS.

---

## References

* Knative Serving, *Configuring the Knative Pod Autoscaler* (stable and panic
  windows, panic threshold, no scale-down while panicking).
  https://knative.dev/docs/serving/autoscaling/kpa-specific/
* Kubernetes, *Horizontal Pod Autoscaler — configurable scaling behavior*
  (KEP-853; scale-down stabilization window, default 300 s).
  https://www.kubernetes.dev/resources/keps/853/
* K. Rzadca et al., *Autopilot: workload autoscaling at Google*, EuroSys 2020
  (usage percentile plus safety margin; slack vs. risk).
  https://research.google/pubs/autopilot-workload-autoscaling-at-google-scale/
* E. S. Gardner Jr., E. McKenzie, *Forecasting trends in time series*,
  Management Science 31(10), 1985 (damped-trend exponential smoothing).
* C. C. Holt, *Forecasting seasonals and trends by exponentially weighted
  moving averages*, 1957 (repr. Int. J. Forecasting 20(1), 2004).
* D. Gutman, S. Syrota, *Proactive automatic up-scaling for Kubernetes*,
  Adaptive Systems of Automatic Control, 2023 (double vs. single exponential
  smoothing for proactive scaling).
* A. Fatahi Baarzi, G. Kesidis, *SHOWAR: right-sizing and efficient
  scheduling of microservices*, SoCC 2021 (control-theoretic horizontal scaling).
* A. U. Gias, G. Casale, M. Woodside, *ATOM: model-driven autoscaling for
  microservices*, ICDCS 2019 (model-based sizing).
