# Run provenance — knowing exactly what produced every result

A flash-crowd run is only interpretable if three things are known: **which
algorithm** DMOS executed, **which state the testbed was in**, and **which
code** ran. None of them can be inferred from a run label: environment
variables (`DMOS_SCORE_GATE`, `DMOS_SCORE_GATE_GAMMA`, `DMOS_WEIGHTS_COHERENT`,
`DMOS_LOCALITY_PURE_NET`, `DMOS_DEMAND_FLOOR`, `DMOS_SCORE_PROFILE`, …) change
the algorithm without touching a file, netem disappears at every VM reboot,
and Service annotations can be edited live. The legacy arm `l1l2` is the
example: its early runs are labelled `dmos-l1l2__*`, but the configuration they
ran (gate g = 2, weights 0.25/0.20/0.25/0.30) could only be recovered from the
DMOS log printed at each start. Every run directory therefore records all
three, and the analysis refuses runs whose provenance is broken.

Runs are started by **name** — `dmos-<arm>__<regime>`, defined once in
`experiments/run_naming.py` and identical in every script — so a label
always means the same configuration. New arms should select the algorithm
with a **score model** (`DMOS_SCORE_MODEL=<name>`, or `bash
experiments/run_one.sh <name> <regime>`): the model defines in one named,
versioned place the role of demand, the weights, the gate, the demand floor
and the cold start
([architecture §7.4](dmos-architecture.md#74-score-models-the-role-of-demand)),
and under a model the legacy switches are ignored. The arm `l1l2` and the
`legacy-*` arms (legacy configurations kept for comparison) set the switches themselves. Either way,
the effective configuration below is what identifies a run; do not export
scoring variables in the shell that launches a session.

For the meaning of the result files (`flash_crowd_summary.json`,
`dmos_live.jsonl`, …) see [METRICS-AND-OUTPUTS.md](METRICS-AND-OUTPUTS.md).

## What a run directory contains

`experiments/results/<YYYYmmdd_HHMMSS>_<label>/`

| File | Written by | Content |
|---|---|---|
| `manifest.json` | `write_manifest.py` (start + finish) | k6/test configuration, arm/regime, git state, host, k6 version and script hash, links to everything below, **provenance checks** |
| `dmos_config.json` | the DMOS process itself | **effective** DMOS configuration after every override, with its fingerprint `config_id` |
| `dmos_config_expected.json` | `dmos_main --dump-config-only`, run by `run_cell` before the reset | configuration the arm is expected to produce; compared with `dmos_config.json` at the end |
| `dmos.log` | DMOS stdout/stderr | full scheduler log (also the carbon-trace replay anchor) |
| `env_state_pre.json` | `capture_env_state.py --phase pre` | testbed snapshot before the run |
| `env_state_post.json` | `capture_env_state.py --phase post` | testbed snapshot after the run |
| `config_snapshot/` | `write_manifest.py` | copy of `config/weights.yaml`, `services.yaml`, `carbon.yaml` and a sanitised `local.yaml` (no addresses or credentials) |
| `git_diff.patch`, `untracked_snapshot/` | `write_manifest.py` | only when the repo was not clean: `commit + git_diff.patch + untracked_snapshot/` rebuild the exact code |
| `dmos_live.jsonl` | collector | timeline every 5 s; each snapshot carries `dmos_config_id`, the configuration that was serving at that moment |
| `flash_crowd_summary.json`, `k6_stdout.txt`, `k6_events.json` | k6 | client-side results (unchanged) |

### `dmos_config.json` — the effective configuration

Built by `src/utils/provenance.py` from the **live objects**, so it shows what
DMOS really uses, not what a YAML file says:

- `modes` — Level 1 on/off (`--no-scheduling`), demand floor, co-location, monitored services
- `score_model` — the selected model (warm definition and its
  `cold_variant`), or `null` on the legacy path; `ignored_legacy_env` lists
  the legacy switches that were set but ignored because of the model (both
  recorded for readability; the algorithm they define is in `level1`)
- `level1.warm` / `level1.cold` — for both score phases: `formula`
  (e.g. `S = G(d; eps=0.05, g=2) * (0.25*lat + 0.35*cap + 0.3*net + 0.1*car)`),
  `demand_mode`, `demand_roles` (where the demand share enters: `gate`,
  `locality_net`, `additive`, `pressure`), weights (including `demand` for the
  additive term) and their source (score model, YAML profile or
  `DMOS_WEIGHTS_COHERENT`), gate ε/g, score parameters (η, β, ν, CI_max, ρ,
  R_max, …)
- `level1.eligibility` — L0 thresholds
- `level2` — PD gains, deadband, raw output range **and the clamp actually applied**,
  predictor window/horizon/damping/cap, rate limits, per-(service, cluster) capacity/min/max
- `safeguards` — cooldowns, churn budgets, dead zone, startup grace
- `coscaling` — dependencies, co-location floor, fan-out, capacities
- `carbon` — provider, trace replay window and the source of each cluster
- `clusters` — region, cores, memory, carbon zone (no addresses)
- `application` — entry service, co-scaled services, namespaces, and how the
  entering demand is measured (`demand`: entry workload, `source_workloads`,
  `server_side_fallback`), from `services.yaml → application`
- `dmos_env` (all `DMOS_*` variables), `git`, `process` — recorded, **not** part of the fingerprint

`config_id` is a hash of the algorithm-defining sections only: two runs share
it **if and only if** DMOS ran the same algorithm with the same parameters.
Names are not part of it (two models with identical definitions share a
`config_id`), and neither are the application namespaces (KubePhos, for
instance, uses a new namespace for every run); the demand measurement is,
because it changes what DMOS reacts to. The configuration path is part of it (`demand_mode` is `legacy` for
the environment switches), so a legacy arm and a score model never share one.
DMOS also exports it on
`/metrics` (`dmos_config_info{config_id=…}`) and on `GET :8081/config`, so a
DMOS started by hand is captured too.

Check an arm's configuration **before** a run, offline, without contacting any cluster:

```bash
DMOS_SCORE_MODEL=reach DMOS_ENTRY_BUDGET=pooled DMOS_SCALER=predictive python -m src.dmos_main --dump-config-only /tmp/cfg.json   # arm reach-pool-ps
DMOS_SCORE_GATE=1 DMOS_SCORE_GATE_GAMMA=2 python -m src.dmos_main --dump-config-only /tmp/cfg.json   # legacy switches
```

`run_cell` does this automatically for every DMOS cell: an unknown model or an
invalid configuration stops the cell before the testbed is reset, and the
dump becomes `dmos_config_expected.json`.

### `env_state_*.json` — the testbed state

Per cluster: Cilium annotations of every Service (`global`, `affinity`,
`shared`), `cilium-config` (incl. `clustermesh-enable-endpoint-sync`) and the
Cilium image, clustermesh-apiserver readiness, HPAs, Deployments (replicas,
readiness, image, resources), Nodes, **netem measured on the worker**
(`tc qdisc`, same criteria as `preflight_runs.sh`) and ping RTT from
Prometheus. `verdicts` summarises it (netem per cluster, affinity per Service
across clusters, endpoint-sync, HPA presence, not-ready deployments, replicas
of the Online Boutique `loadgenerator`).

`dmos_config.json` also records, under `carbon.sources`, where each cluster's
carbon intensity comes from (`trace:<file>`, `realistic`, `mock`,
`default-300`), resolved through `trace.zone_aliases`.

## Gates and checks

**Before** the run (`experiments/lib/run_common.sh`) the cell does **not**
start, and its directory moves to `experiments/results/_aborted/`, when:

| Gate | Default | Disable / change |
|---|---|---|
| netem missing (or not verifiable) on a cluster that should shape traffic | on | `DMOS_REQUIRE_NETEM=0` |
| frontend affinity differs from the expected one on some cluster | expected `none` (paper data plane) | `DMOS_EXPECT_FRONTEND_AFFINITY=local` (or empty: no check) |
| Online Boutique `loadgenerator` with replicas > 0 | on | `DMOS_ALLOW_LOADGENERATOR=1` |
| an HPA in the namespace before the run (the HPA baseline creates its own afterwards) | on | `DMOS_EXPECT_NO_HPA=0` |

Names are validated earlier still: an unknown arm or regime, a renamed one
(e.g. `l1l2coh`, `con400`) or an invalid DMOS configuration stops the cell —
or the whole session — before the testbed is reset.
`capture_env_state.py` also offers `--expect-no-hpa` for arms that require it.

**After** the run, `manifest.json → provenance.checks`:

| Code | Level | Meaning |
|---|---|---|
| `dmos_config_missing` | error | DMOS arm without effective configuration: algorithm unknown |
| `config_changed_during_run` | error | more than one `config_id` served the run (DMOS restarted with other settings) |
| `config_mismatch` | error | `dmos_config.json` and `/metrics` disagree |
| `config_precheck_mismatch` | error | the configuration that ran differs from the one expected by the pre-check (`dmos_config_expected.json`) |
| `dmos_running_in_off_arm` | error | a DMOS process answered during a `--no-dmos` run |
| `netem_invalid_pre` | error | netem not in place before the run |
| `netem_lost_during_run` | error | netem gone after the run (VM rebooted mid-run) |
| `env_expectation_violated` | error | a requested gate (affinity, HPA, loadgenerator) failed |
| `loadgenerator_running` | error | the Online Boutique `loadgenerator` had replicas before or after the run: internal traffic that k6 does not count |
| `hpa_present` | error | an HPA was in the namespace (before or after) during a non-HPA arm: it fights the arm for the replicas |
| `hpa_baseline_missing` | error | HPA arm without `hpa_baseline.json` |
| `hpa_not_applied` | error | HPA arm: the baseline HPAs were not all created |
| `hpa_not_active` | error | HPA arm: a baseline HPA did not read its CPU metric during the run (`ScalingActive` not true) |
| `hpa_not_removed` | warn | HPA arm: the HPAs were not all removed after the run (the next cell removes them) |
| `hpa_not_seen_post` | warn | HPA arm: no HPA in the post-run snapshot |
| `karmada_present` | error | a Deployment was managed by Karmada (label `karmada.io/managed`) during a non-Karmada arm: Karmada reverts the replicas the arm sets |
| `karmada_baseline_missing` | error | Karmada arm without `karmada_baseline.json` |
| `karmada_not_adopted` | error | Karmada arm: the adoption did not complete cleanly (`karmada_baseline.json → apply.errors`) |
| `karmada_pod_restarted` | error | Karmada arm: adopting changed a Deployment's revision or uid (pods rolled out or recreated) |
| `karmada_not_verified` | error | Karmada arm: no verification after the load |
| `karmada_cluster_dropped` | error | Karmada arm: a cluster left a binding during the run (its Deployment deleted or left unmanaged: scheduler without `--enable-empty-workload-propagation`) |
| `fhpa_not_active` | error | Karmada arm: a FederatedHPA did not read its CPU metric (`ScalingActive` not true) |
| `karmada_lost_management` | error | Karmada arm: Deployments no longer managed by Karmada at the end of the load |
| `karmada_cluster_not_ready`, `fhpa_never_scaled`, `karmada_host_unchecked`, `karmada_version_unexpected`, `karmada_not_released` | warn | a member cluster not Ready in Karmada during the run; no FederatedHPA changed its replicas; scheduler flags / estimators / versions not verified; images not at v1.19.0; the release after the run did not complete (the next prep retries) |
| `k6_summary_missing` / `k6_vu_starved` | error | client results missing / offered load not delivered |
| `shift_not_applied` | error | the regime moves the hot spot (`-to<cluster>`) but k6 reports no move, or a move at another time (old k6 script on the VM) |
| `shift_unexpected` | error | k6 moved the hot spot in a regime without a move (stray `DMOS_SHIFT_*` variables in the shell) |
| `carbon_trace_fallback` | warn | some cluster had no carbon trace and used a synthetic source (listed) |
| `config_id_not_observed`, `dmos_log_missing`, `env_pre_missing`, `affinity_changed_during_run`, `repo_dirty` | warn | noted, run kept |

`paper_lib.discover_runs()` **skips** runs with provenance errors (like the
k6 VU-starvation filter) and **warns** when one label mixes different
`config_id`s — those runs are not repetitions of the same algorithm. The
journal analysis (`experiments/journal_*.py`) uses only the runs with status
`OK` in the campaign TSV written by `run_campaign.sh` (a run that fails
`campaign_ctl.py check` is retried). Legacy
runs (before this harness) carry no provenance and are kept as they are.

## Inspecting runs

```bash
python experiments/run_provenance.py show  experiments/results/<run>      # one run, everything
python experiments/run_provenance.py audit                                # all runs: validity, mixed labels
python experiments/run_provenance.py audit --include-archives             # also results/_*/
python experiments/run_provenance.py diff  <run_A> <run_B>                # what differs
```

Each run gets a status: `ok`, `invalid` (provenance errors), `aborted`
(stopped by the pre-run testbed gate, in `results/_aborted/`) or `legacy`
(no provenance). `audit` exits with code 2 when a run is invalid or a label
mixes configurations.

## Using the harness

`run_one.sh` (and `run_campaign.sh`, which runs `run_one.sh` for every cell
of a plan) resolves the names with `run_naming.py` and goes through `run_cell` in
`experiments/lib/run_common.sh`, so every run is traceable with no extra
flag. DMOS is stopped between cells by `kill_dmos`, which stops only the
processes started as `python -m src.dmos_main`. `run_flash_crowd.sh` used on its own (DMOS started by hand)
still writes the manifest and fetches the effective configuration from
`:8081/config`; only `dmos.log` and the testbed snapshots are missing, and the
manifest says so.
