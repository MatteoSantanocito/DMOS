# DMOS — Demand-gated Multi-Objective Orchestration System

DMOS is a control plane for a geo-distributed federation of Kubernetes
clusters. It decides how many replicas of the entry service of a microservice
application are needed and in which clusters to place them, weighing latency,
capacity, network proximity and carbon intensity, and it sizes the backends of
each cluster from the frontend replicas placed there.

This is release `DMOS-2026.2`, the artifact of the article that evaluates
DMOS on a federation of six clusters with a pre-specified campaign of 90 runs
(see [Citation](#citation)). The version presented at IEEE NCA 2026 is
available at the git tag `nca-2026.1`.

---

## Contents

1. [What DMOS does](#what-dmos-does)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Reproducing the paper](#reproducing-the-paper)
5. [Data availability](#data-availability)
6. [Repository layout](#repository-layout)
7. [Documentation](#documentation)
8. [Testing](#testing)
9. [Citation](#citation)
10. [License](#license)

---

## What DMOS does

The frontend of the application is exposed as a Cilium ClusterMesh **global
service with `affinity: none`**: every ready frontend replica, in any cluster,
can serve the users of any region, and the share of traffic a cluster serves
follows its share of ready replicas (the *pooled data plane*). On this data
plane the number of replicas depends only on the total demand, while their
placement determines how much network delay the requests pay, so DMOS takes
the two decisions separately. A central process, running outside the clusters,
reads the Prometheus instance of every cluster (Hubble L7 request rates, node
resources, `ping_exporter` round-trip times) and a carbon-intensity source, and:

1. **Level 2 — how many and when.** Every 10 s it forecasts the total demand
   with a damped Holt method at the horizon at which new replicas will be ready
   (10 s plus an actuation delay learned at run time), adds an error-driven
   margin, and sets the federation-wide budget `N`. Brakes on scale-down (hold
   at the peak, gradual removal, hold after a burst) avoid removing replicas
   that would be needed again shortly afterwards.
2. **Level 0 — eligibility.** It excludes the clusters whose CPU or memory
   cannot host new replicas.
3. **Level 1 — where.** It distributes `N` among the eligible clusters in
   proportion to the score

   ```
   S_i = G(ψ_i) · B_i        B_i = Σ_k ω_k Φ_i^k   (latency, capacity, network, carbon)
   ψ_i = Σ_e d_e · exp(−RTT_ei / τ),  τ = 50 ms    (reachable demand)
   G(ψ) = (ε + (1 − ε) ψ)^g,  ε = 0.05, g = 2       (demand gate)
   ```

   by waterfilling within the per-cluster capacity, then rounding with the
   largest-remainder method and the per-cluster minimum. Demand is a
   precondition of the score, not one more weight: the weights `ω` express the
   operator's policy among the clusters that can serve the demand.
4. **Backend co-scaling.** The backends of each cluster follow its frontend
   replicas, `n_{i,b} = clip(⌈r_b n_i⌉, 1, n̄_b)` with
   `r_b = max(0.3, f_b μ / μ_b)`, rather than their observed traffic.

Each behaviour is selected by environment variables (`DMOS_SCORE_MODEL`,
`DMOS_ENTRY_BUDGET`, `DMOS_SCALER`, ...), and every run records its effective
configuration and a `config_id`
([docs/RUN-PROVENANCE.md](docs/RUN-PROVENANCE.md)). The design is described in
[docs/dmos-architecture.md](docs/dmos-architecture.md),
[docs/REACH-AND-POOLED-BUDGET.md](docs/REACH-AND-POOLED-BUDGET.md),
[docs/PREDICTIVE-SCALER.md](docs/PREDICTIVE-SCALER.md) and
[docs/BACKEND-CO-SCALING.md](docs/BACKEND-CO-SCALING.md).

---

## Architecture

```
┌──────────────────────────── operator machine ─────────────────────────────┐
│  python -m src.dmos_main                                                   │
│   L2 forecast → budget N ─► L0 eligibility ─► L1 score + distribution      │
│   ─► backend co-scaling                   :9090 /metrics   :8081 /config   │
└───────────┬──────────────────────────────────┬────────────────────────────┘
       PromQL│                           kube-API (scale)
             ▼                                  ▼
┌──────────── 6 × K3s clusters, Cilium ClusterMesh (global frontend) ──────────┐
│  cluster-01 DE   cluster-02 FR   cluster-03 PL                                │
│  cluster-04 US   cluster-05 SG   cluster-06 NO                                │
│  each: Cilium (Hubble L7, Gateway API) + kube-prometheus-stack                │
│        + ping_exporter + Online Boutique;  tc netem 0–200 ms per cluster      │
└───────────────────────────────────────────────────────────────────────────────┘
          ▲ k6 load generator (dedicated VM), per-region entry weights
```

The testbed of the paper (six K3s clusters of three nodes each, with 4 to 8
vCPUs per application node) is described in [docs/KUBEPHOS.md](docs/KUBEPHOS.md)
and [docs/setup-guide.md](docs/setup-guide.md).

---

## Quick start

```bash
git clone https://github.com/MatteoSantanocito/DMOS.git dmos && cd dmos
python -m venv venv && source venv/bin/activate      # Windows (Git Bash): source venv/Scripts/activate
pip install -r requirements-lock.txt                 # exact versions of the paper (Python 3.12)
```

1. **Testbed.** Build the clusters and the DMOS add-ons, write
   `config/local.yaml` from `config/local.yaml.example` (it is gitignored),
   render `.cache/` and apply netem: [docs/setup-guide.md](docs/setup-guide.md).
2. **Check.** `bash scripts/validate-setup.sh` and
   `bash experiments/preflight_runs.sh`.
3. **Run DMOS** as in the paper (from the repository root):

   ```bash
   DMOS_SCORE_MODEL=reach DMOS_ENTRY_BUDGET=pooled DMOS_SCALER=predictive python -m src.dmos_main
   ```

   The first 120 s are an observation window in which nothing is scaled.
   `python experiments/run_naming.py show <arm> <regime>` prints the exact
   configuration of every arm of the campaign.

Carbon intensity is replayed from the hourly Electricity Maps traces of 2025,
which are not distributed with the repository: download them into
`data/carbon_traces/` with your own access, as described in
[data/carbon_traces/README.md](data/carbon_traces/README.md).

---

## Reproducing the paper

The evaluation is a pre-specified campaign of 90 runs: 18 cells (arm × demand
regime), each repeated five times, in the order fixed by
[experiments/plans/2026-11-02-definitive.plan](experiments/plans/2026-11-02-definitive.plan).
The plan also lists the 18 statistical tests (families F1, F2, F3) and the
deviations from it. The campaign ran on the code of this release: `src/` and
`config/` are those recorded in every run (`git.commit` 41aa0252 in each
`manifest.json`).

| Arm (identifier) | Name in the paper | Regimes |
|---|---|---|
| `off` | Static | bal, con, con-tosg |
| `hpa-none` | Per-cluster HPA | bal, con, con-tosg |
| `l2-pool-ps` | Even split | bal, con, con-tosg |
| `reach-pool-ps` | DMOS | bal, con, con-tosg |
| `nca-pool-ps` | Local-demand gate | con |
| `locality-ps` | Locality term | con |
| `reach-pool-ps-min0` | DMOS (min 0) | bal, con |
| `reach_green-pool-ps-min0` | DMOS-Green | bal, con |

**From the archived dataset (no testbed needed).** All figures, tables and
tests of the paper can be regenerated from the raw run data:

```bash
pip install -r requirements-lock.txt
# 1. download and unpack the dataset (see Data availability), e.g. into ../dmos-2026-2-dataset
# 2. download the Electricity Maps traces into data/carbon_traces/ (README there)
D=../dmos-2026-2-dataset
T=$D/campaign/_campaign_2026-11-02-definitive.tsv
python experiments/journal_figures.py --results $D/runs --tsv $T --out out --no-cache
python experiments/journal_tests.py   --results $D/runs --tsv $T --out out
python experiments/journal_control.py --results $D/runs --tsv $T --out out
```

`journal_figures.py` writes the figures, `runs.csv`, `summary.csv` and the
descriptive tables; `journal_tests.py` the 18 pre-specified tests (exact
permutation test, Holm correction within each family) in `tests.md`,
`tests.csv` and `table_tests.tex`; `journal_control.py` the cost of the control
plane (`table_control.tex`, `control_timing.csv`) and an offline benchmark of
the decision logic. The step-by-step guide, with the mapping from each figure
and table of the paper to its output, is
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

**Re-running the campaign** requires an equivalent testbed and takes about
35 hours: [experiments/RUNBOOK.md](experiments/RUNBOOK.md).

---

## Data availability

The raw data of the 90 runs of the paper (time series sampled every 5 s,
DMOS decisions and logs, k6 summaries, testbed state before and after each run,
manifests with the effective configuration), the campaign index and logs and
the plans are archived on Zenodo:

> **Dataset:** DOI [10.5281/zenodo.24282190](https://doi.org/10.5281/zenodo.24282190) (CC BY 4.0)
>
> **Code:** every GitHub release of DMOS is archived on Zenodo; this one is
> `DMOS-2026.2` (the DOI is on the Zenodo record linked from the GitHub release)

In the dataset, the private addresses of the testbed are replaced by
documentation addresses (consistently across all files) and local paths are
redacted; [experiments/build_dataset.py](experiments/build_dataset.py) is the
script that produced it from the raw run folders. The Electricity Maps carbon
intensity traces are not redistributable and are not included.

---

## Repository layout

```
src/
  dmos_main.py            orchestrator: control loop, Level 2, co-scaling, HTTP endpoints
  level1/                 score functions and models, scheduler (Level 0 + scores), allocation
  level2/                 predictive scaler (damped Holt), legacy reactive scaler
  metrics/                per-cluster Prometheus client, carbon-intensity client
  k8s/                    multi-cluster Kubernetes client
  utils/                  configuration loader, provenance (effective config, config_id), logger
config/                   weights.yaml (score models), services.yaml, carbon.yaml,
                          local.yaml.example, Helm values
data/carbon_traces/       where the Electricity Maps 2025 traces go (not distributed; README)
deployments/              manifests DMOS expects in each cluster (global services, CNPs, HPA, ...)
scripts/                  testbed setup, rendering of .cache/, validation, netem, trace download
experiments/              k6 workload, campaign harness (run_campaign.sh, run_one.sh, lib/),
                          collector and provenance tools, analysis (journal_*.py),
                          plans/ (the pre-specified plan of the paper), build_dataset.py
tests/                    pytest suite (no cluster needed)
docs/                     documentation (index below)
```

---

## Documentation

**Design**

- [Architecture and design](docs/dmos-architecture.md)
- [Reachable demand and pooled budget](docs/REACH-AND-POOLED-BUDGET.md)
- [Predictive Level-2 scaler](docs/PREDICTIVE-SCALER.md)
- [Backend co-scaling](docs/BACKEND-CO-SCALING.md)
- [Start-up: observation window and two-phase score](docs/two-phase-scheduling.md)
- [Measuring demand: the ingress rate](docs/traffic-metrics-architecture.md)
- [Network proximity and the emulated geography](docs/geo-awareness-implementation.md)

**Testbed and configuration**

- [Setup guide](docs/setup-guide.md), [the six-cluster fleet](docs/KUBEPHOS.md)
- [Configuration reference](docs/configuration-reference.md): every key and environment variable
- [Ingress architecture](docs/ingress-architecture.md), [Hubble L7 visibility](docs/cilium-hubble-l7-visibility.md), [Cilium network policies](docs/cilium-network-policies.md), [ClusterMesh notes](docs/CILIUM-CLUSTERMESH-NOTES.md)
- [Troubleshooting](docs/troubleshooting.md), [Integration guide](docs/integration-guide.md)

**Experiments**

- [Reproducibility guide](docs/REPRODUCIBILITY.md): from the dataset or the testbed to the figures
- [Experiment runbook](experiments/RUNBOOK.md): running the campaign
- [Plans](experiments/plans/README.md): the pre-specified plan and its deviations
- [Run provenance](docs/RUN-PROVENANCE.md), [Metrics and outputs](docs/METRICS-AND-OUTPUTS.md)
- [Load-testing scenarios](docs/load-testing-scenarios.md)
- [Capacity testing: frontend](docs/capacity-testing.md), [gRPC backends](experiments/CAPACITY-GRPC.md), [throttle check](experiments/THROTTLE-CHECK.md)
- Optional tools, not used in the paper's campaign: [Karmada baseline](docs/KARMADA-BASELINE.md), [live flow map](docs/LIVE-FLOW-MAP.md)

Comments in some configuration files and shell scripts are partly in Italian;
the configuration files are kept byte-identical to those recorded in the runs.

---

## Testing

```bash
pytest          # from the repository root; pytest.ini limits collection to tests/
```

The suite covers the score components and models, the predictive scaler, the
budget and allocation, configuration loading, the carbon and Prometheus
clients, the test naming (`run_naming.py`), the provenance tools and the
statistics of the paper (`journal_tests.py`). It needs no cluster; the
cluster-side checks are `scripts/validate-setup.sh` and
`experiments/preflight_runs.sh`.

---

## Citation

If you use DMOS, please cite the paper that introduced it:

> A. Marchese, M. Santanocito, O. Tomarchio. *DMOS: Demand-Gated Multi-Objective
> Replica Placement for Geo-Distributed Multi-Cluster Systems.* 2026 24th IEEE
> International Symposium on Network Computing and Applications (NCA), 2026,
> pp. 80–87. DOI: [10.1109/NCA71379.2026.00021](https://doi.org/10.1109/NCA71379.2026.00021).
> Code: release `nca-2026.1`.

```bibtex
@inproceedings{Marchese2026DMOS,
  author    = {Marchese, Angelo and Santanocito, Matteo and Tomarchio, Orazio},
  title     = {{DMOS}: Demand-Gated Multi-Objective Replica Placement for
               Geo-Distributed Multi-Cluster Systems},
  booktitle = {2026 24th IEEE International Symposium on Network Computing and Applications (NCA)},
  year      = {2026},
  pages     = {80--87},
  doi       = {10.1109/NCA71379.2026.00021}
}
```

This release (`DMOS-2026.2`) extends that version with the predictive Level 2,
the reachable-demand score and a new evaluation campaign. If you use it or its
data, please also cite the archived artifacts (the article describing this
release is under review; its reference will be added here):

> [software] M. Santanocito, A. Marchese, O. Tomarchio. *DMOS* (version
> `DMOS-2026.2`). Zenodo, 2026.

> [dataset] M. Santanocito, A. Marchese, O. Tomarchio. *DMOS 2026.2 evaluation
> campaign: raw data of 90 runs on a six-cluster federation*. Zenodo, 2026.
> DOI: [10.5281/zenodo.24282190](https://doi.org/10.5281/zenodo.24282190).

DMOS originates from the MSc thesis of M. Santanocito, *Distributed
Multi-Objective Scheduler for Kubernetes Multi-Cluster (DMOS)*, University of
Catania, 2026.

## License

The code is released under the MIT license ([`LICENSE`](LICENSE)); the
dataset under CC BY 4.0. The carbon intensity data used by the experiments
come from Electricity Maps and are not distributed; publications that use them
must cite Electricity Maps (see
[data/carbon_traces/README.md](data/carbon_traces/README.md)).
