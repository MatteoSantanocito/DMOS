# DMOS experiment runbook

How to run the experiments of DMOS on a testbed: one run, a whole campaign
from a plan file, and the analysis. To regenerate the results of the paper
from the archived data, without a testbed, see
[docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).

All commands run from **Git Bash (or any bash), from the repository root**.
The scripts read the topology and the credentials from `.cache/test-env.json`,
generated from `config/local.yaml` by `bash scripts/render-runtime-files.sh`;
no address is hard-coded.

## Tools

```
experiments/
  run_campaign.sh           unattended campaign: the cells of a plan file, one at a time
  campaign_ctl.py           its helpers: wait for safe windows, check each run
  run_one.sh                one cell (arm x regime): reset, checks, DMOS or baseline, load, provenance
  lib/run_common.sh         the logic shared by the two scripts above
  lib/fleet.sh              fleet helpers (ClusterMesh, Hubble labels, pod lifecycle)
  run_naming.py             test names: arm x regime -> configuration (the only place)
  run_flash_crowd.sh        one load run: k6 on the load VM + collector
  k6_flash_crowd.js         the k6 workload (5 phases in 900 s, 6 entry points)
  collect_dmos_live.py      collector: DMOS /metrics and every cluster's Prometheus, every 5 s
  capture_env_state.py      testbed state before and after a run (netem, affinity, Cilium, HPA)
  write_manifest.py         provenance manifest of a run
  run_provenance.py         show / audit / diff the provenance of runs
  hpa_baseline.py           the HPA baseline (one HPA per service and cluster)
  preflight_runs.sh         GO / NO-GO gate before a session
  validate_run.sh           performance verdict of a run
  journal_*.py              analysis of the campaign (figures, tests, control-plane cost)
  build_dataset.py          the public dataset from the run folders
  plans/                    plan files (the paper's: 2026-11-02-definitive.plan)
  results/                  one folder per run (gitignored)
```

## Prerequisites

- `bash`, `jq`, `curl`, `kubectl`, `helm`, `ssh`/`scp`, Python with the
  packages of `requirements-lock.txt`. On Windows the scripts use the
  virtual environment in `venv/Scripts` and PowerShell (`pwsh`) for the reset
  and to keep the machine awake during a campaign.
- The kubeconfig of every cluster, with its path in `config/local.yaml`.
- A dedicated load VM reachable over SSH with k6 installed (the paper used
  k6 v2.3.0): `export DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>`; the SSH key is
  `ssh.key` of `config/local.yaml` (override: `DMOS_K6_SSH_KEY`).
- The pooled data plane: Service `frontend` global, shared, with
  `affinity: none` (`deployments/global-service-frontend.yaml`). Every cell
  checks it before starting.
- No scoring variable exported in the shell (`DMOS_SCORE_*`,
  `DMOS_WEIGHTS_COHERENT`, `DMOS_LOCALITY_PURE_NET`, `DMOS_DEMAND_FLOOR`):
  the configuration of each arm is decided by its name.
  Check with `env | grep '^DMOS_'`.

## 0. Pre-flight

A reboot or maintenance of the clusters resets volatile state, netem first,
and invalidates geo-aware runs without visible errors. Before every session:

```bash
bash experiments/preflight_runs.sh            # GO / NO-GO
bash experiments/preflight_runs.sh --watch    # wait until the six clusters are back
bash experiments/preflight_runs.sh --fix      # re-apply netem if it was reset
```

It checks the reachability of every entry point and Prometheus, node
readiness, the emulated RTTs, the Cilium annotations of the managed Services,
the pods and the DMOS add-ons. Exit 0 = GO, 1 = NO-GO, 2 = missing
prerequisites. `bash experiments/clean_reset.sh [--deep]` resets the fleet.

## 1. Test names

A run is named `dmos-<arm>__<regime>` and its folder is
`experiments/results/<timestamp>_dmos-<arm>__<regime>/`. `run_naming.py`
turns a name into the DMOS and k6 configuration:

```bash
python experiments/run_naming.py list                   # arms and regimes
python experiments/run_naming.py show reach-pool-ps con # what a cell runs
```

The arms of the paper are `off`, `hpa-none`, `l2-pool-ps`, `reach-pool-ps`,
`nca-pool-ps`, `locality-ps`, `reach-pool-ps-min0` and
`reach_green-pool-ps-min0`; the regimes are `bal` (40% of the peak demand from
DE), `con` (85% from DE) and `con-tosg` (85% from DE, moving to SG at 450 s
within 30 s), all with a peak of 350 req/s. The other arms and regimes listed
by `run_naming.py` (legacy configurations, other skews, other hot clusters,
other peaks) are kept for further experiments and were not used in the paper.

## 2. One run

```bash
bash experiments/apply_pod_lifecycle.sh          # once: preStop and grace period on the scaled services
bash experiments/run_one.sh reach-pool-ps con    # one cell
bash experiments/run_one.sh reach-pool-ps con --dry-run
```

For each cell `run_one.sh`:

1. stops any DMOS or baseline left over, resets every Deployment to one
   replica, waits 75 s and checks that every entry point answers, that
   ClusterMesh, the Gateways and the Hubble labels are in order, and that
   delays, global-service affinity and replicas are as expected (with
   `DMOS_FLEET_DIR` set, through `scripts/kubephos_fleet.py check`); the cell
   does not start if a check fails;
2. records the testbed state (`env_state_pre.json`), starts the arm (DMOS with
   its configuration, or the HPAs) and writes the effective configuration
   (`dmos_config.json`, with its `config_id`);
3. runs `run_flash_crowd.sh`: the collector samples DMOS and the clusters
   every 5 s while k6 applies the 15-minute load from the load VM;
4. stops the arm, records the state again (`env_state_post.json`), repeats the
   data-plane checks (a failure writes `INVALID_DATAPLANE.txt`) and closes
   `manifest.json`.

The files of a run are described in
[docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md#files-of-a-run) and
[docs/METRICS-AND-OUTPUTS.md](../docs/METRICS-AND-OUTPUTS.md).

## 3. A campaign

A campaign is a plan file, one cell per line:

```
# comment
@prestop                                   apply preStop/grace, then require it in every run
<block> <arm> <regime> [VAR=value ...]     e.g.  D1 reach-pool-ps con
```

The campaign of the paper:

```bash
export DMOS_FLEET_DIR=<fleet directory of scripts/kubephos_fleet.py>
export DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>
bash experiments/run_campaign.sh experiments/plans/2026-11-02-definitive.plan
```

For every cell, `run_campaign.sh` waits until the cell cannot overlap a window
in which K3s re-applies the Cilium chart (`campaign_ctl.py wait`; on another
fleet set `DMOS_REAPPLY_ANCHOR=` to disable the windows), runs `run_one.sh`,
and checks the run (`campaign_ctl.py check`). A run that is not valid is moved
to `results/_retried/` and the cell is repeated, up to
`DMOS_CAMPAIGN_TRIES` (3) attempts; two cells in a row without a valid run
stop the campaign. Outputs:

- `results/_campaign_<plan>.tsv`: one row per attempt (`time`, `block`,
  `arm`, `regime`, `try`, `status` OK/BAD, `run`, `reasons`); the analysis
  reads only the OK rows;
- `results/_campaign_<plan>.log`: progress;
- `results/_<plan>_<block>_<arm>_<regime>_t<n>.log`: one log per attempt.

**Resuming.** If a campaign stops, copy the plan into `plans/resume/` with the
same file name, remove the cells already OK in the TSV and run the copy: the
results go to the same TSV, because its name comes from the plan's file name.
Record every deviation at the end of the original plan, with date and reason,
without changing its cells.

The paper's plan takes about 35 hours (18 cells x 5 blocks, about 23 minutes
per cell on average, including the reset and the re-apply windows).

## 4. Checking a run

```bash
python experiments/run_provenance.py show experiments/results/<run>   # everything about a run
python experiments/run_provenance.py audit                            # all runs: validity, mixed configurations
bash experiments/validate_run.sh experiments/results/<run>            # performance verdict
```

`validate_run.sh` judges the performance of an arm (errors at the peak), not
the health of the testbed: in the concentrated regime the static fleet
exceeds 10% errors by construction and is reported as degraded although the
run is valid. The validity of the testbed is in the provenance checks of the
manifest, which `campaign_ctl.py check` uses.

## 5. Analysis

```bash
T=experiments/results/_campaign_2026-11-02-definitive.tsv
python experiments/journal_figures.py --tsv $T --no-cache    # figures, runs.csv, summary.csv, tables
python experiments/journal_tests.py   --tsv $T               # the 18 pre-specified tests
python experiments/journal_control.py --tsv $T               # control-plane cost (after journal_figures)
```

The default output folder is `paper/figures/journal` (`--out` to change it).
The mapping from outputs to the paper is in
[docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md#outputs-and-the-paper).

## Operational notes

- **Starting DMOS by hand**: `python -m src.dmos_main` from the repository
  root, with the environment printed by `run_naming.py show`; metrics on
  `http://localhost:9090/metrics`, configuration on `:8081`. Both listen on
  all interfaces without authentication: run DMOS on a trusted network.
- **Stopping DMOS from the scripts**: `kill_dmos` (in `lib/run_common.sh`)
  stops only the Python processes whose command line contains
  `-m src.dmos_main`.
- **Online Boutique load generator**: it generates in-cluster traffic that k6
  does not count but DMOS sees as demand. The reset scales it to 0, and every
  cell refuses to start if it is running (`DMOS_ALLOW_LOADGENERATOR=1`
  disables the check).
- **netem**: if the pre-flight reports flat RTTs, `--fix` re-applies the
  delays (`scripts/setup-netem-and-ping.sh`).
- **Capacity calibration** (`capacity_req_per_sec`): see
  [docs/capacity-testing.md](../docs/capacity-testing.md) and
  [CAPACITY-GRPC.md](CAPACITY-GRPC.md).
