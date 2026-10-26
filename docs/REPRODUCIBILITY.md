# Reproducing the DMOS evaluation from a fresh `git clone`

The end-to-end path from an empty machine to the figures and tables of the
papers. Each step links to the detailed guide; this page fixes the order,
the inputs and the expected outputs.

**What "reproduce" means.** The NCA 2026 evaluation is a skew-robustness
study: 3 configurations (OFF static, L2 uniform scaling, L1L2 full DMOS) × 6
demand regimes (hottest-cluster share `uni 0.167 / bal 0.40 / s50 / s65 / s75
/ con 0.85`), at a constant 350 req/s peak (the paper says three
repetitions; the repetitions actually behind each cell are listed in
[publications/2026-nca-dmos.md](publications/2026-nca-dmos.md)). Absolute
latencies depend on the hardware; the reproducible objects are the **trends**
(SLO violations vs. skew, the crossover between a 0.5 and a 0.65
hottest-cluster share, the equal-budget ablation OFF → L2 → L1L2). The exact
configuration of the paper and its known differences from the code are in
[publications/2026-nca-dmos.md](publications/2026-nca-dmos.md).

---

## 0. What ships in the repo vs. what you provide

| In the repository | You provide |
|---|---|
| scheduler (`src/`), configuration (`config/*.yaml`, `local.yaml.example`) | K ≥ 2 Kubernetes clusters (reference: 6 K3s clusters on Proxmox) |
| the proxmox-k3s file of the reference fleet (`config/proxmox-k3s-fleet.yaml.example`) | 3 Proxmox VE hosts (32 GB each in the reference), a VM template, IP addresses, an API token |
| manifests (`deployments/`), setup and validation scripts (`scripts/`) | `config/local.yaml` (gitignored) |
| load generator and run harness (`experiments/`) | a VM for k6 |
| analysis pipeline (`experiments/paper_*.py`, `paper_lib.py`) | — |
| the script that downloads the carbon traces (`scripts/fetch_carbon_traces.py`) | *(optional)* Electricity Maps access with historical data (academic access) to download the traces into `data/carbon_traces/` |

The carbon traces are Electricity Maps data and are **not distributed** with
the repository: download them with your own access (researchers: the
Electricity Maps academic access) as described in
[data/carbon_traces/README.md](../data/carbon_traces/README.md). The
reference clusters map to the zones `DE`, `FR`, `PL`, `NO-NO1` (alias of
`NO`), `US-MIDA-PJM` (alias of `US-VA`) and `SG`
(`config/carbon.yaml → trace.zone_aliases`); a cluster without its file
falls back to built-in hourly profiles
([dmos-architecture.md §5.3](dmos-architecture.md#53-carbon-intensity)):

```bash
export ELECTRICITYMAPS_TOKEN=<your key>
python scripts/fetch_carbon_traces.py --zones DE FR PL US-MIDA-PJM SG NO-NO1 --year 2025
```

Each run records which source every cluster used (`dmos_config.json →
carbon.sources`) and warns (`carbon_trace_fallback`) when one is not a
trace.

## 1. Operator machine and testbed

```bash
git clone https://github.com/MatteoSantanocito/dmos.git && cd dmos
python -m venv venv && source venv/Scripts/activate   # Linux/macOS: venv/bin/activate
pip install -r requirements.txt
```

Build the clusters and the DMOS add-ons, write `config/local.yaml`, render
`.cache/`, apply netem and ping_exporter, harden the workload:
[setup-guide.md](setup-guide.md). Every experiment script reads the topology
from `.cache/test-env.json` — no IP is hard-coded in the repository.

## 2. Validate

```bash
bash scripts/validate-setup.sh          # kubeconfigs, ClusterMesh, Prometheus, Hubble, app Ready
python scripts/dmos-ready-check.py      # end-to-end readiness (exit 0 = GO)
bash experiments/preflight_runs.sh      # per-session GO/NO-GO (netem, pods, ClusterMesh, DMOS layers)
```

The Online Boutique `loadgenerator` must stay at 0 replicas: the harness's
reset scales it to 0 and every cell refuses to start otherwise (see
[setup-guide.md §6](setup-guide.md#6-before-the-first-run)). The frontend
Service must be global with `affinity: none` (`kubectl apply -f
deployments/global-service-frontend.yaml` on each cluster); every cell
checks it too.

## 3. (Optional) Re-calibrate capacities

`config/services.yaml` ships the per-replica capacities used in the paper. On
different hardware re-measure them:

```bash
bash experiments/run_capacity.sh --cluster ms-cluster-01   # frontend, HTTP
bash experiments/run_capacity_grpc.sh --all                # backends, k6 Job in-cluster
```

Method: [capacity-testing.md](capacity-testing.md), [../experiments/CAPACITY-GRPC.md](../experiments/CAPACITY-GRPC.md).

## 4. Run the evaluation

Test names are `dmos-<arm>__<regime>`, defined once in
`experiments/run_naming.py` (`python experiments/run_naming.py list`). The
L1L2 configuration that produced the NCA 2026 results is the arm **`l1l2`**
(gate g = 2, weights lat/cap/net/car 0.25/0.20/0.25/0.30, Φ_loc = Φ_net,
floor off — read back from the campaign's DMOS logs); the score model `nca`
is the configuration as *described* in the paper (weights
0.25/0.35/0.30/0.10). One repetition of the whole core (18 cells, the three
arms of a regime consecutive so that they share the testbed state):

```bash
export DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>
bash experiments/run_matrix_session.sh                       # off l2 l1l2 × uni bal s50 s65 s75 con
bash experiments/run_matrix_session.sh --arms "off l2 nca"   # the same with the paper's stated weights
```

Run it once per session, as many sessions as repetitions. One cell:
`bash experiments/run_one.sh l1l2 con`. `--dry-run` on either script prints
what would run without touching the testbed; the `config_id` of each run
tells which configuration ran.

**Every run is self-describing**: its directory stores the effective DMOS
configuration (`dmos_config.json`, `config_id`, carbon source per cluster),
the DMOS log, the testbed state before and after (netem measured on the
workers, Service annotations, `cilium-config`, Deployments, loadgenerator)
and a manifest with provenance checks; a cell does not start if netem is
missing, if the frontend's affinity is not `none` or if the loadgenerator
runs.

```bash
python experiments/run_provenance.py show  experiments/results/<run_dir>
python experiments/run_provenance.py audit
bash experiments/validate_run.sh <run_dir>     # heuristic CLEAN/SUSPECT/DEGRADED verdict (see RUNBOOK)
```

See [RUN-PROVENANCE.md](RUN-PROVENANCE.md) and [../experiments/RUNBOOK.md](../experiments/RUNBOOK.md).

**Footprint**: a cell takes about 15 min of load plus reset and settling
(~18–20 min); the raw k6 events are the bulk of the disk use under
`experiments/results/` (gitignored). `bash experiments/archive_run.sh` prunes
them after a run; compute `slo_compliance.json` first if you prune
([METRICS-AND-OUTPUTS.md §8](METRICS-AND-OUTPUTS.md#8-caveats)).

## 5. Figures and tables

```bash
python experiments/paper_generate_all.py
python experiments/paper_plot_skew_robustness.py                    # Fig. 2 (off / l2 / l1l2)
python experiments/paper_plot_hotshare.py                           # Fig. 3
python experiments/paper_plot_skew_robustness.py --arms dmos-off dmos-l2 dmos-nca   # another DMOS arm
python experiments/paper_plot_hotshare.py --dmos-arm dmos-nca --hot sg
```

What each script produces and how every metric is defined:
[METRICS-AND-OUTPUTS.md](METRICS-AND-OUTPUTS.md). `results/` and
`paper_artifacts/` stay local by design.

## Known limits

- Absolute p95/goodput values shift with host hardware and VM contention;
  compare arms within one testbed, not against the paper's absolute numbers.
- The data plane is the paper's: frontend global with `affinity: none`, so
  requests are served by the frontend pods of all clusters in proportion to
  their replicas; the backends stay `affinity: local`
  ([CILIUM-CLUSTERMESH-NOTES.md](CILIUM-CLUSTERMESH-NOTES.md)).
- Carbon intensity is replayed from 2025 traces for the zones that have a
  file, and from built-in hourly profiles (wall-clock dependent) for the
  others; in the NCA campaign NO, US-VA and SG used the profiles.
- The paper's stated parameters differ from the campaign's in two places:
  the weights (§V-B) and ρ = 2.0 (Tab. I; the code and the campaign use
  3.0). See [publications/2026-nca-dmos.md](publications/2026-nca-dmos.md).
- The Online Boutique manifests are taken from the upstream `main` branch at
  install time; the image versions of a run are in its `env_state_pre.json`.
