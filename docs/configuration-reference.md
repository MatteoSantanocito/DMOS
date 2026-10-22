# Configuration reference

Every input DMOS and its tooling read: the YAML files in `config/`, the files
generated from them, and the environment variables. For each key: what reads
it and what it changes. Keys that are present in the files but **not read by
any code** are marked *not read*; their values have no effect.

How the values are used is explained in
[dmos-architecture.md](dmos-architecture.md). The values a given run actually
used are recorded in its `dmos_config.json`
([RUN-PROVENANCE.md](RUN-PROVENANCE.md)).

## Contents

1. [Files](#1-files)
2. [`local.yaml` — testbed](#2-localyaml--testbed)
3. [`weights.yaml` — score](#3-weightsyaml--score)
4. [`services.yaml` — service model](#4-servicesyaml--service-model)
5. [`carbon.yaml` — carbon intensity](#5-carbonyaml--carbon-intensity)
6. [Helm values and generated files](#6-helm-values-and-generated-files)
7. [Environment variables](#7-environment-variables)
8. [Hard-coded parameters](#8-hard-coded-parameters)

---

## 1. Files

| File | In git | Read by | Content |
|---|---|---|---|
| `config/local.yaml` | no (template `local.yaml.example`) | DMOS (`ConfigLoader`), `render-runtime-files.sh`, `bootstrap-assisted.sh`, `paper_lib` (carbon accounting) | clusters, SSH, netem, Alertmanager |
| `config/weights.yaml` | yes | DMOS | score models, legacy weight profiles, score parameters |
| `config/services.yaml` | yes | DMOS | capacity, fan-out, replica bounds per service |
| `config/carbon.yaml` | yes | DMOS (`CarbonClient`) | carbon provider and its options |
| `config/cilium-hubble-values.yaml` | yes | none of the scripts (reference) | Hubble metrics values |
| `config/prometheus-helm-values.yaml` | yes | `bootstrap-assisted.sh`, `upgrade-prometheus.ps1` | standalone Prometheus (NodePort 30090), alternative to kube-prometheus-stack |
| `config/proxmox-k3s-fleet.yaml.example` | yes (template; the filled-in copy holds the Proxmox token, keep it out of git) | `proxmox-k3s cluster create/mesh` (external tool) | the 6-cluster reference fleet: VM sizes and tiers, K3s/Cilium versions, CIDRs, ClusterMesh ids ([setup-guide §2.1](setup-guide.md#21-clusters)) |
| `data/carbon_traces/<ZONE>_2025_hourly.csv` | yes | `CarbonClient` (provider `trace`) | hourly carbon intensity ([attribution](../data/carbon_traces/README.md)) |
| `.cache/test-env.json`, `.cache/test-env.sh`, `.cache/*-values.yaml` | no | harness, k6, collector, setup scripts | generated from `local.yaml` (§6) |

`ConfigLoader` (`src/utils/config_loader.py`) reads `local.yaml`,
`weights.yaml`, `services.yaml` and `carbon.yaml` once at start from the
directory `config/` (relative to the working directory: run DMOS from the
repository root). A missing file stops the start-up; for `local.yaml` the error
explains how to create it. There is no overlay or merge between files.

Validation at start-up:

- the active legacy weight profile must sum to 1 (`ScoreWeightsConfig`);
- the selected score model is validated by `ScoreModel.from_dict` (§3.1);
- `objectives.balanced` must sum to 1 (parsed although not used).

---

## 2. `local.yaml` — testbed

Created by `bash scripts/bootstrap-assisted.sh` (interactive) or by copying
`config/local.yaml.example`. After every change run
`bash scripts/render-runtime-files.sh` to regenerate `.cache/`.

### 2.1 `clusters[]`

A list of cluster entries (a mapping name → entry is also accepted). The
order matters: it is the order of `DMOS_SPIKE_W1..N` / `DMOS_INGRESS_W1..N`
in k6 and of the clusters in the logs.

| Key | Default | Read by | Effect |
|---|---|---|---|
| `name` | required | everywhere | cluster id (e.g. `ms-cluster-01`); key of the per-cluster maps in `services.yaml` |
| `display_name` | `name` | `test-env.json` | label only |
| `region` | required | k6 (baseline weights by region), `test-env` | label used by k6 to pick the baseline ingress weight (§7.3) |
| `location` | required | — | label only |
| `ip` | required | DMOS, render, harness | IP of the node that exposes the NodePorts (worker): Prometheus `http://<ip>:<prometheus_port>`, Ingress `http://<ip>:30080`, ping targets, SSH for netem |
| `kubeconfig_path` | required | DMOS (`KubernetesClient`), harness, `capture_env_state.py` | kubeconfig of the cluster; its `server` (control-plane node) is used as is |
| `prometheus_port` | 32090 | DMOS, render | NodePort of the cluster's Prometheus |
| `k8s_api_port` | 6443 | — | not used (the kubeconfig decides) |
| `server`, `estimator_url` | `""` | — | not used |
| `resources.cpu_cores` | 4 | DMOS | total CPU of the cluster for Φ_cap (hardware size = cores/8), L0 utilization, `request_rate_max = cores × capacity`; tier in the carbon accounting of the analysis |
| `resources.memory_gb` | 8 | DMOS | total memory for L0 and Φ_cap |
| `pricing.cost_per_replica_hour` | 0.1 | DMOS (carried in `ClusterMetrics`) | no effect on the score |
| `carbon.region_code` | `DE` | DMOS | zone of the carbon intensity: name of the trace file `<code>_*.csv`, key of the built-in profiles |
| `carbon.baseline_gco2_kwh` | — | analysis only (`paper_lib`) | static intensity used by the carbon accounting of the figures (§5.8 of [METRICS-AND-OUTPUTS.md](METRICS-AND-OUTPUTS.md)); DMOS itself never reads it |
| `latency.baseline_ms` | 50 | DMOS | latency assumed when Hubble has no p95 (and in the cold phase); 2.0 on the testbed |
| `latency.variance_ms` | 15 | DMOS | cold-phase variance only (no effect, σ² = 10⁶) |
| `netem.delay_ms`, `netem.jitter_ms`, `netem.interface` | 0, 0, — | `setup-netem-and-ping.sh`, `capture_env_state.py` | emulated one-way delay on the node's interface; also the expected value checked before each run (tolerance 3 ms) |

### 2.2 Other sections

| Key | Read by | Effect |
|---|---|---|
| `ssh.user`, `ssh.key` | render → `test-env.*`, netem setup, `capture_env_state.py`, `run_flash_crowd.sh` (key for the k6 VM by default) | SSH to the nodes and the k6 VM |
| `alertmanager.webhook_url` | `render-runtime-files.sh` | renders `deployments/alertmanager-config.yaml` from the template, pointing Alertmanager to DMOS's `/webhook/alert` |
| `global.scheduling_interval_seconds` | *not read* | the cycle period is hard-coded (30 s) |
| `global.score_request_timeout_seconds` | *not read* | Prometheus timeouts are fixed in the code (5 s) |

---

## 3. `weights.yaml` — score

### 3.1 `score_models.<name>`

Selected with `DMOS_SCORE_MODEL=<name>`; when set, it alone defines the score
and the legacy switches of §3.2 and §7.1 are ignored. Validation errors stop
the start-up (and the harness's pre-check aborts the run).

| Key | Values | Meaning |
|---|---|---|
| `description` | text | free text, recorded in the provenance |
| `demand_mode` | `gate` \| `additive` \| `locality` | where the demand share d enters the score ([architecture §7.4](dmos-architecture.md#74-score-models-the-role-of-demand)) |
| `weights` | `latency`, `capacity`, `net`, `carbon`, `demand` ≥ 0, sum = 1 | `demand` only in `additive` mode; missing keys are 0 |
| `gate.eps`, `gate.gamma` | 0 ≤ ε ≤ 1 (default 0.05), g ≥ 0 (default 2) | gate `(ε + (1−ε)d)^g`; `gate` mode only |
| `demand_floor` | bool (default false) | pre-assign ⌈λᵢⁱⁿ/c⌉ replicas per cluster before the score |
| `cold_start` | `structural` (default) \| `same` | weights of the first 120 s: latency and capacity set to 0 and the rest renormalised, or the warm weights |
| `parameters` | any of `eta`, `sigma_squared`, `kappa`, `load_threshold`, `load_steepness`, `load_floor`, `horizon_seconds`, `nu`, `ci_max`, `rho`, `rtt_max_ms` | overrides of `ScoreParameters` for this model (e.g. `{rho: 2.0}`) |

Defined models: `nca`, `gate_g0`, `gate_g1`, `gate_g3`, `additive_dem50`,
`locality`, `latency_first`, `capacity_first`, `carbon_first` (table in
[architecture §7.4](dmos-architecture.md#74-score-models-the-role-of-demand)).

**Adding a model.** Add an entry under `score_models`, check it offline with

```
DMOS_SCORE_MODEL=<name> python -m src.dmos_main --dump-config-only /tmp/cfg.json
```

and run it with `bash experiments/run_one.sh <name> <regime>`: every model
is automatically an arm name (`dmos-<name>__<regime>`) of
`experiments/run_naming.py`. The NCA 2026 campaign did not use a model: it
ran the legacy switches of the arm `l1l2` (§3.2, §7.1), whose weights
(0.25/0.20/0.25/0.30) differ from those of `nca`.

### 3.2 `score_weights` (legacy path)

Used only when `DMOS_SCORE_MODEL` is not set.

| Key | Effect |
|---|---|
| `active` | profile of the warm phase (default `balanced` if missing; currently `flash_crowd_aware`); `DMOS_SCORE_PROFILE` overrides it. An unknown name falls back to 0.25/0.20/0.25/0.20/0.10 with a warning |
| `<profile>.omega_latency` | weight of Φ_lat |
| `<profile>.omega_capacity` | weight of Φ_cap |
| `<profile>.omega_geo_load` | weight of Φ_loc = d·Φ_net (Φ_net alone with `DMOS_LOCALITY_PURE_NET=1`) |
| `<profile>.omega_carbon` | weight of Φ_car |
| `<profile>.omega_demand` | weight of **Φ_pressure** (utilization sigmoid), not of d |
| `cold_start` | profile of the first 120 s (if missing: the active profile) |

Profiles in the file: `cold_start`, `carbon_agnostic`, `balanced`,
`carbon_priority`, `latency_priority`, `geo_aware`, `romano_like`,
`geo_balanced`, `flash_crowd_aware`, and the single-objective `latonly`,
`netonly`, `carbon`. The five weights of a profile must sum to 1. In the run
harness a profile is the arm `profile-<name>`.

### 3.3 Score parameters

| Key | Read | Value in effect |
|---|---|---|
| `network_parameters.rho` | yes | **3.0** (code default 2.0; NCA paper Tab. I: 2.0) |
| `network_parameters.rtt_max_ms` | yes | 500 |
| `network_parameters.fallback_rtt_ms` | read, not used | the client returns 5 ms when ping data is missing (hard-coded) |
| `load_parameters.threshold`, `.steepness` | yes | τ = 0.7, k = 10 (Φ_pressure, legacy path only) |
| `load_parameters.floor` | read, not used | — |
| `carbon_parameters.nu`, `.ci_max` | *not read* | ν = 0.5 and CI_max = **500** come from `ScoreParameters`; the file says 800 |
| `objectives.*` | parsed, *not used* | — |

η (0.001), σ² (10⁶) and the Φ_cap constants (β = 0.3, reference 8 cores)
come from the code; a score model can override η, σ², ν, CI_max, ρ and
RTT_max through `parameters`.

---

## 4. `services.yaml` — service model

`services.<name>` (a list with `name:` is also accepted):

| Key | Read by | Effect |
|---|---|---|
| `namespace`, `deployment_name` | DMOS | the Deployment that is read and scaled |
| `capacity_req_per_sec` | DMOS | c: budget N = Σ⌈λᵢ/c⌉ (frontend), co-scaling ratio, `request_rate_max` |
| `capacity_per_cluster.<cluster>` | DMOS | per-cluster override of c (not set today) |
| `calls_per_request` | DMOS | fan-out f of the co-scaling ratio (c_front/c_b)·f |
| `autoscaling.min_replicas`, `autoscaling.max_replicas` | DMOS | bounds, unless overridden per cluster |
| `min_replicas_per_cluster.<cluster>`, `max_replicas_per_cluster.<cluster>` | DMOS | per-cluster bounds (`c_min`, `c_max`); `c_max` is also the cap of L1 |
| `resources_per_replica.requests.cpu` / `.memory` | DMOS | bid capacity ⌊free / request⌋ ([architecture §7.6](dmos-architecture.md#76-bids-and-allocation)); they must match the live Deployments |
| `resources_per_replica.limits.*` | — | documentation of the live Deployments |
| `autoscaling.target_cpu_percent` | — | not used |
| `sla.*`, `power_watts` | — | not used |
| `global_autoscaling.*` (safety margin, PD gains 5/300, ±5, prediction 300/600 s) | *not read* | the values in effect are in the code: Kp 2, Kd 20, ±3, W 60 s, H 120 s, +6/−1 ([architecture §8](dmos-architecture.md#8-level-2--predictor-and-pd-correction)) |

Only `frontend` is scheduled; the services listed in
`DMOSOrchestrator.service_dependencies` (cart, productcatalog, checkout,
recommendation, currency, ad, shipping) are co-scaled; any other entry is
loaded but never scaled. The resources of the live Deployments are set by
`experiments/apply_resource_hardening.sh` and
`experiments/apply_pod_lifecycle.sh`, not by this file.

---

## 5. `carbon.yaml` — carbon intensity

All keys are under `carbon_intensity:`.

| Key | Effect |
|---|---|
| `provider` | `trace` (in use), `realistic`, `mock`, `electricitymaps`, `watttime` (not implemented → `realistic`) |
| `trace.data_dir` | directory of the CSV traces (relative to the repository root) |
| `trace.window_start` | instant of the trace at which the replay starts (`2025-06-02T00:00:00`) |
| `trace.speedup` | time compression of the replay (96: a 15-min run spans a day) |
| `trace.zone_aliases` | region code → Electricity Maps zone of the trace file, when they differ (`"NO": NO-NO1`, `US-VA: US-MIDA-PJM`); quote `"NO"`, a bare NO is a YAML boolean |
| `realistic.noise_enabled`, `realistic.noise_percent` | ±5 % deterministic jitter of the built-in profiles |
| `mock.enabled`, `mock.values.<code>`, `mock.hourly_variation.*` | static values per region (used by `mock`, and as the last fallback of `realistic`) |
| `electricitymaps.api_url`, `.api_key` | live API (placeholder key → `realistic`) |
| `cache.ttl_seconds` | 300 s cache of the non-trace providers |
| `cache.enabled` | not read (the cache is always on) |
| `fallback.use_baseline` | not read |

Fallback chain: `trace` without a CSV for the zone → `realistic` (built-in
hourly profile × seasonal factor of the current month × jitter) → `mock`
values → 300 gCO₂/kWh in the scheduler. The zone of a cluster is
`trace.zone_aliases[region_code]` (or the region code), and its file is
found by the pattern `<zone>_*hourly*.csv`. The repository has traces for
`DE`, `FR`, `PL`, `NO-NO1` (and `IT-NO`, `ES`); `US-MIDA-PJM` and `SG` are
still to be downloaded, so US-VA and SG fall back
([architecture §5.3](dmos-architecture.md#53-carbon-intensity)):
`python scripts/fetch_carbon_traces.py --zones US-MIDA-PJM SG --year 2025`
(token in `ELECTRICITYMAPS_TOKEN`). The source used by each cluster is
recorded in every run (`dmos_config.json → carbon.sources`).

---

## 6. Helm values and generated files

`scripts/render-runtime-files.sh` writes, from `local.yaml`:

| File | Content | Used by |
|---|---|---|
| `.cache/test-env.json` | `ssh`, and for each cluster `name`, `ip`, `region`, `display_name`, `ingress_url` (`:30080`), `prometheus_url`, `kubeconfig_path`, `netem` | k6 (`DMOS_TEST_ENV`), collector, harness, `capture_env_state.py` |
| `.cache/test-env.sh` | the same as shell variables (`DMOS_CLUSTER<i>_*`, arrays) | shell scripts |
| `.cache/cilium-<cluster>-values.yaml` | Cilium values with ClusterMesh peers | `bootstrap-assisted.sh` installation path |
| `.cache/ping-exporter-<cluster>-values.yaml` | ping targets = the other clusters' IPs | `setup-netem-and-ping.sh` |
| `deployments/alertmanager-config.yaml` | only when `alertmanager.webhook_url` is set | Alertmanager |

The harness reads the k6 VM from `DMOS_K6_REMOTE`, or from a `k6.remote` key
of `test-env.json`; the renderer does not write that key, so set the
variable (or re-add the key after each render).

On the proxmox-k3s testbed the Hubble metrics are enabled by
`scripts/setup-dmos-on-clusters.ps1`, which writes its own Helm values
(httpV2 with the labels DMOS filters on, Cilium Ingress on NodePort 30080),
and the 15 s scrape comes from `deployments/hubble-servicemonitor-monitoring.yaml`.

---

## 7. Environment variables

### 7.1 DMOS process (`python -m src.dmos_main`)

| Variable | Default | Effect |
|---|---|---|
| `DMOS_SCORE_MODEL` | unset | named score model (§3.1) |
| `DMOS_CONFIG_DUMP` | unset | path where the effective configuration is written at start (same as `--config-dump`) |
| `DMOS_PREDICTOR_WINDOW_S`, `DMOS_PREDICTOR_HORIZON_S` | 60, 120 | predictor window and horizon |
| `DMOS_SCORE_PROFILE` | `score_weights.active` | legacy profile |
| `DMOS_SCORE_GATE` | 0 | legacy: 1 enables the gate |
| `DMOS_SCORE_GATE_EPS`, `DMOS_SCORE_GATE_GAMMA` | 0.05, 1.0 | legacy gate parameters |
| `DMOS_LOCALITY_PURE_NET` | 0 | legacy: 1 makes the proximity term Φ_net instead of d·Φ_net |
| `DMOS_WEIGHTS_COHERENT` | unset | legacy: `"lat,cap,net,car"` replaces the four weights and drops Φ_pressure |
| `DMOS_DEMAND_FLOOR` | 1 | legacy: 0 disables the demand floor |
| `DMOS_LOCUST_URL` | `http://localhost:8089` | read, not used |

The six legacy switches (`DMOS_SCORE_PROFILE` … `DMOS_DEMAND_FLOOR`) are
ignored when `DMOS_SCORE_MODEL` is set. Every `DMOS_*` variable of the process
is recorded in `dmos_config.json`.

Command-line flags: `--no-scheduling` (no L1, uniform N/K), `--naive-backends`,
`--config-dump PATH`, `--dump-config-only PATH`.

### 7.2 Run harness (`run_one.sh`, `run_matrix_session.sh`, `run_skew_sweep.sh`, `run_flash_crowd.sh`)

The configuration of a cell — DMOS switches, k6 weights, peak, hot cluster —
comes from its **name** (`dmos-<arm>__<regime>`, `experiments/run_naming.py`),
not from the environment: do not export the scoring variables of §7.1 in the
shell that launches a session. The variables below tune the harness itself.

| Variable | Default | Effect |
|---|---|---|
| `DMOS_K6_REMOTE` | `k6.remote` of `test-env.json` | `user@host` of the k6 VM (required by `run_cell`; empty → k6 runs locally in `run_flash_crowd.sh`) |
| `DMOS_K6_SSH_KEY` | `ssh.key` of `test-env.json` | key for the k6 VM |
| `DMOS_REQUIRE_NETEM` | 1 | 0 lets a run start without the expected netem |
| `DMOS_ENV_CAPTURE` | 1 | 0 skips the testbed snapshots |
| `DMOS_EXPECT_FRONTEND_AFFINITY` | `none` | affinity the frontend Service must have on every cluster for a cell to start (empty: no check) |
| `DMOS_ALLOW_LOADGENERATOR` | 0 | 1 lets a cell start while the Online Boutique `loadgenerator` has replicas |
| `DMOS_PIDFILE` | `.cache/dmos.pid` | PID of the DMOS process started by the harness (used by `kill_dmos` on Linux/macOS) |
| `SETTLE` | 75 | seconds of settling after the reset |
| `DMOS_PEAK_RPS` | 250 (the regime sets it: 350, or `-p<rps>`) | peak rate |
| `DMOS_BASELINE_RPS` | 50 | baseline rate |
| `DMOS_SPIKE_W1..N`, `DMOS_SPIKE_REGION` | set by the regime (demand and hot cluster) | forwarded to k6 |
| `DMOS_INGRESS_W1..N`, `DMOS_SPIKE_WEIGHT` | unset | forwarded to k6 |
| `DMOS_PHASE_TIMES` | `180,300,600,720,900` | forwarded to k6 |
| `DMOS_VU_FACTOR` | 20 (30 when the regime's peak is above 350) | forwarded to k6 |
| `DMOS_RAW_OUT` | 1 | 0 skips `k6_events.json` (the SLO metrics then cannot be computed) |
| `DMOS_ARM`, `DMOS_REGIME` | from the label | recorded in the manifest |
| `DMOS_KUBECTL`, `DMOS_SSH` | `kubectl`, `ssh` | command used by `capture_env_state.py` (JSON list or shell string) |
| `KUBECONFIG_DIR`, `DMOS_KUBECONFIG_DIR`, `DMOS_IAC`, `KCBASE` | from `test-env.json` or `~/dmos-iac` | kubeconfig location for the reset and capacity scripts |

### 7.3 k6 script (`k6_flash_crowd.js`, passed with `-e`)

`DMOS_RUN_LABEL`, `DMOS_BASELINE_RPS` (50), `DMOS_PEAK_RPS` (400 if run
directly), `DMOS_SPIKE_REGION` (`ms-cluster-01`), `DMOS_SPIKE_WEIGHT` (0.60),
`DMOS_SPIKE_W<i>`, `DMOS_INGRESS_W<i>`, `DMOS_PHASE_TIMES`, `DMOS_VU_FACTOR`
(20), `DMOS_REQ_TIMEOUT_S` (14), `DMOS_RNG_SEED` (20260521), `DMOS_TEST_ENV`
(`../.cache/test-env.json`). Their meaning is in
[METRICS-AND-OUTPUTS.md §1](METRICS-AND-OUTPUTS.md#1-the-workload-and-its-phases).
The capacity scripts have their own variables, documented in
[capacity-testing.md](capacity-testing.md) and
[../experiments/CAPACITY-GRPC.md](../experiments/CAPACITY-GRPC.md).

### 7.4 Analysis

| Variable | Used by | Effect |
|---|---|---|
| `DMOS_RESULTS_DIR`, `DMOS_PAPER_TAG`, `DMOS_PAPER_BASELINE`, `DMOS_TARGET_REPS` | `paper_generate_all.py` | input directory, output tag, baseline arm of the tests, expected repetitions |
| `DMOS_SLO_MS` | `paper_plot_trajectories.py` | SLO threshold of Fig. 3/3b (default 2000) |
| `DMOS_H_REGIME`, `DMOS_H_ARM` | `paper_report.py` | regime (default `con`) and full-DMOS arm (default `dmos-l1l2`) on which the hypotheses are checked |

---

## 8. Hard-coded parameters

The control-loop constants (cycle period, thresholds, debounce, grace,
cooldowns, rate limits, churn budgets, PD gains, predictor shape, co-scaling
floor, L0 limits) are not configurable from files. They are listed with their
location in [architecture §10](dmos-architecture.md#10-safeguards-and-timing-constants)
and recorded in each run's `dmos_config.json`.
