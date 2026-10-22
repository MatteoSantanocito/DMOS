# DMOS — Demand-gated Multi-Objective Orchestration System

> A demand-gated, multi-objective scheduler that decides, every 30 seconds,
> how many replicas of each microservice run on each cluster of a
> geo-distributed Kubernetes federation — weighing latency, capacity, network
> proximity and carbon intensity, and placing replicas only where the demand
> is.

DMOS started as an MSc thesis at the University of Catania and is the
subject of the IEEE NCA 2026 paper *DMOS: Demand-Gated Multi-Objective
Replica Placement for Geo-Distributed Multi-Cluster Systems*
([publications](docs/publications/README.md)).

---

## Contents

1. [What DMOS does](#what-dmos-does)
2. [Architecture](#architecture)
3. [Quick start](#quick-start)
4. [Repository layout](#repository-layout)
5. [Documentation](#documentation)
6. [Testing](#testing)
7. [Citation](#citation)
8. [License](#license)

---

## What DMOS does

A central control loop, running on an operator machine, reads each
cluster's Prometheus (Cilium Hubble L7 metrics, node resources,
`ping_exporter` RTTs) and a carbon-intensity source, and every cycle:

1. **Budget** — sizes the frontend from the demand entering each cluster:
   `N = Σᵢ ⌈λᵢⁱⁿ / capacity⌉`, within per-cluster caps.
2. **Placement (Level 1)** — scores every eligible cluster and splits N in
   proportion to the score (largest remainder, capacity-aware). In the
   published design demand is a **multiplicative gate**, not one more weight:

   ```
   S_i = G(d_i) · [ ω_lat·Φ_lat + ω_cap·Φ_cap + ω_net·Φ_net + ω_car·Φ_car ]
   G(d) = (ε + (1−ε)·d)^g          d_i = cluster's share of the entering demand
   ```

   A cluster without demand cannot attract replicas whatever its other
   qualities; among those with demand, the weights decide.
3. **Control (Level 2)** — a trend predictor anticipates the scale-up and a
   PD controller adds a ±1 correction per cluster; rate limits, cooldowns,
   a dead zone and churn budgets keep the system stable.
4. **Co-scaling** — the seven stateless backends follow the frontend on each
   cluster in proportion to capacity and call fan-out.

How demand enters the score, the weights, the gate and the start-up policy
are a named **score model** in `config/weights.yaml` (`nca` is the
configuration described in the NCA 2026 paper; `gate_g0/g1/g3`,
`additive_dem50`, `locality`, `*_first` are the variants of the journal
extension), selected with `DMOS_SCORE_MODEL`. The NCA campaign itself ran
the arm `l1l2` (gate g = 2, weights 0.25/0.20/0.25/0.30; see
[the publication page](docs/publications/2026-nca-dmos.md)). Every run
records its effective configuration and a `config_id`.

---

## Architecture

```
┌──────────────────────── operator machine ─────────────────────────┐
│  python -m src.dmos_main                                           │
│   budget N ─► L1 score + proportional allocation ─► L2 predictor/PD │
│   ─► co-scaling of the backends        :9090 /metrics  :8081 /config│
└──────────┬──────────────────────────────┬─────────────────────────┘
     PromQL│                        kube-API (scale)
           ▼                              ▼
┌──────────────── 6 × K3s clusters on Proxmox, Cilium ClusterMesh ────────────────┐
│  ms-cluster-01 EU-DE (HIGH)   ms-cluster-02 EU-FR (MED)   ms-cluster-03 EU-PL (LOW)│
│  ms-cluster-04 US-VA (HIGH)   ms-cluster-05 SG    (LOW)   ms-cluster-06 NO    (MED)│
│  each: Cilium (Hubble L7, Ingress :30080) + kube-prometheus-stack (:32090)        │
│        + ping_exporter + Online Boutique;  tc netem 0–200 ms per cluster          │
└──────────────────────────────────────────────────────────────────────────────────┘
          ▲ k6 load generator (dedicated VM), per-cluster ingress weights
```

Details: [docs/dmos-architecture.md](docs/dmos-architecture.md).

---

## Quick start

```bash
git clone <repo-url> dmos && cd dmos
python -m venv venv && source venv/Scripts/activate   # Linux/macOS: venv/bin/activate
pip install -r requirements.txt
```

1. **Testbed** — build the clusters and the DMOS add-ons, write
   `config/local.yaml` (gitignored), render `.cache/`, apply netem:
   [docs/setup-guide.md](docs/setup-guide.md).
2. **Check** — `bash scripts/validate-setup.sh` and
   `bash experiments/preflight_runs.sh`.
3. **Run DMOS** — `DMOS_SCORE_MODEL=nca python -m src.dmos_main` (from the
   repository root). One cycle every 30 s; the first 120 s are an
   observation window (nothing is scaled).
4. **Run experiments** — every test is named `dmos-<arm>__<regime>`
   (`python experiments/run_naming.py list`):
   `bash experiments/run_one.sh l1l2 con` runs one cell,
   `bash experiments/run_matrix_session.sh` one repetition of the NCA core
   (off / l2 / l1l2 × 6 regimes). See [experiments/RUNBOOK.md](experiments/RUNBOOK.md).
5. **Analyse** — `python experiments/paper_generate_all.py`
   ([docs/METRICS-AND-OUTPUTS.md](docs/METRICS-AND-OUTPUTS.md)).

No Electricity Maps key is needed at runtime: the `trace` provider replays
the hourly traces committed in `data/carbon_traces/`.

---

## Repository layout

```
src/
  dmos_main.py            orchestrator: control loop, L2 application, co-scaling, HTTP endpoints
  level1/                 score functions, score models, scheduler (L0 + scores), allocation
  level2/                 trend predictor, PD controller
  metrics/                per-cluster Prometheus client, carbon client
  k8s/                    multi-cluster Kubernetes client
  utils/                  configuration loader, provenance (effective config, config_id), logger
config/                   weights.yaml (score models, legacy profiles), services.yaml,
                          carbon.yaml, local.yaml.example, Helm values
data/carbon_traces/       Electricity Maps 2025 hourly traces (ODbL)
deployments/              manifests DMOS expects in each cluster (+ clustermesh-overflow/ pilot)
scripts/                  setup, rendering of .cache/, validation, netem, carbon trace download
experiments/              k6 workloads, run harness (run_one.sh, lib/run_common.sh),
                          provenance tools, collector, analysis pipeline (paper_*.py)
tests/                    pytest suite (no cluster needed)
docs/                     documentation (index below), publications/
```

---

## Documentation

**Design**

- [Architecture and design](docs/dmos-architecture.md) — the reference: every mechanism and constant, with the code that implements it
- [Start-up: observation window and two-phase score](docs/two-phase-scheduling.md)
- [Measuring demand: the ingress rate](docs/traffic-metrics-architecture.md)
- [Network proximity (Φ_net) and the emulated geography](docs/geo-awareness-implementation.md)
- [Backend co-scaling](docs/BACKEND-CO-SCALING.md)

**Configuration and setup**

- [Setup guide](docs/setup-guide.md)
- [Configuration reference](docs/configuration-reference.md) — every key and environment variable
- [Ingress architecture](docs/ingress-architecture.md), [Hubble L7 visibility](docs/cilium-hubble-l7-visibility.md), [Cilium network policies](docs/cilium-network-policies.md), [ClusterMesh notes](docs/CILIUM-CLUSTERMESH-NOTES.md)
- [Troubleshooting](docs/troubleshooting.md), [Integration guide](docs/integration-guide.md)

**Experiments and results**

- [Reproducibility guide](docs/REPRODUCIBILITY.md) — from `git clone` to the figures
- [Experiment runbook](experiments/RUNBOOK.md)
- [Run provenance](docs/RUN-PROVENANCE.md) — what each run records and how to audit it
- [Metrics and outputs](docs/METRICS-AND-OUTPUTS.md) — run files, metric definitions, figures and tables
- [Load-testing scenarios](docs/load-testing-scenarios.md)
- [Capacity testing — frontend](docs/capacity-testing.md), [gRPC backends](experiments/CAPACITY-GRPC.md), [throttle check](experiments/THROTTLE-CHECK.md)

**Publications** — [docs/publications/](docs/publications/README.md): one page
per paper with the code version, configuration, figure → script mapping and
citation.

Historical documents (earlier testbeds, superseded designs) are kept in
[docs/_archive/](docs/_archive/).

---

## Testing

```bash
pytest tests/ -v        # from the repository root
```

The suite covers the score components and score models, configuration
loading, the carbon and Prometheus clients (trace lookup per region), the
allocation, the test naming (`run_naming.py`) and the provenance tools
(effective configuration, testbed capture, run manifests).
It needs no cluster; cluster-side checks are in `scripts/validate-setup.sh`
and `experiments/preflight_runs.sh`.

---

## Citation

If you build on DMOS, please cite the NCA 2026 paper
([BibTeX](docs/publications/2026-nca-dmos.md#how-to-cite)):

> A. Marchese, M. Santanocito, O. Tomarchio. *DMOS: Demand-Gated
> Multi-Objective Replica Placement for Geo-Distributed Multi-Cluster
> Systems.* IEEE International Symposium on Network Computing and
> Applications (NCA), 2026.

DMOS originates from the MSc thesis: M. Santanocito, *Distributed
Multi-Objective Scheduler for Kubernetes Multi-Cluster (DMOS)*, University
of Catania, 2026.

## License

The code is released under the license in [`LICENSE`](LICENSE). The carbon
traces in `data/carbon_traces/` are Electricity Maps data under the Open
Database License (ODbL); see [their README](data/carbon_traces/README.md).
