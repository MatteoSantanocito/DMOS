# Reproducing the results of the paper

This guide regenerates every figure, table and statistical test of the
evaluation of the article (release `DMOS-2026.2`) from the raw data of its campaign. Two paths
are possible:

- **A. From the archived dataset**, on any machine with Python 3.12: the 90 runs
  are already recorded, and the analysis is deterministic.
- **B. From a testbed**, re-running the campaign: this requires a federation
  equivalent to the one described in [KUBEPHOS.md](KUBEPHOS.md) and about 35
  hours ([experiments/RUNBOOK.md](../experiments/RUNBOOK.md)).

## What was run

The campaign is fixed by the plan
[experiments/plans/2026-11-02-definitive.plan](../experiments/plans/2026-11-02-definitive.plan),
written and committed before the first run: 8 arms, 3 demand regimes, 18 cells,
5 repetitions (blocks D1 to D5), 90 runs. Within each block the regimes, and
within each regime the arms, run in a random order fixed by a seed. The plan
header also lists the primary metrics and the 18 tests:

| Family | Tests | Question |
|---|---|---|
| F1 | 9 | DMOS (`reach-pool-ps`) against the per-cluster HPA (p95 and frontend replica-minutes) and against the even split (p95), in each regime |
| F2 | 6 | sustainability levers, in the balanced and concentrated regimes: carbon-first weights (served carbon intensity, CO2 per request) and minimum 0 (replica-minutes) |
| F3 | 3 | demand signal of the placement score, concentrated regime (one test two-sided) |

Each test is an exact permutation test on the difference of the means of the
five runs of two arms (252 relabellings), one-sided in the stated direction
unless marked otherwise, with Holm's correction within each family at
α = 0.05.

**Deviations from the plan**, recorded at the end of the plan file: the
shutdown of two virtual machines interrupted one run (D5, `nca-pool-ps`,
concentrated); it was cancelled and the remaining cells were run with
[plans/resume/2026-11-02-definitive.plan](../experiments/plans/resume/2026-11-02-definitive.plan),
without changing the cells or their order. Two runs of `hpa-none` in the
balanced regime failed the post-run data-plane check (`INVALID_DATAPLANE`) and
were repeated, as the harness does automatically; the campaign index keeps
them as `BAD` rows, and the analysis reads only the `OK` rows.

**Code.** The campaign ran on commit 41aa0252 (`git.commit` in every
`manifest.json`). `src/` and `config/` of this release are identical to that
commit; the release adds the analysis scripts `experiments/journal_*.py`, the
plans, `experiments/build_dataset.py`, tests and documentation.

## A. From the archived dataset

1. **Code and environment.**

   ```bash
   git clone https://github.com/MatteoSantanocito/DMOS.git dmos && cd dmos
   git checkout DMOS-2026.2
   python -m venv venv && source venv/bin/activate
   pip install -r requirements-lock.txt
   ```

2. **Dataset.** Download the archive from Zenodo (DOI in the
   [README](../README.md#data-availability)) and unpack it, for example next to
   the repository as `../dmos-2026-2-dataset`. Its layout:

   ```
   dmos-2026-2-dataset/
     README.md  LICENSE  CHECKSUMS.sha256
     campaign/  _campaign_2026-11-02-definitive.tsv   index: one row per attempt, status OK/BAD
                _campaign_2026-11-02-definitive.log   campaign driver log
                driver_logs/                          one log per attempt
                plans/                                the plan and the resume plan
     runs/<timestamp>_dmos-<arm>__<regime>/           the 90 runs (files below)
   ```

   `sha256sum -c CHECKSUMS.sha256` (inside the folder) verifies the files.

3. **Carbon intensity traces.** Download the Electricity Maps hourly traces of
   2025 for the six zones into `data/carbon_traces/`, as described in
   [its README](../data/carbon_traces/README.md). The analysis stops with an
   error if a trace is missing, instead of falling back to built-in profiles.

4. **Analysis.**

   ```bash
   D=../dmos-2026-2-dataset
   T=$D/campaign/_campaign_2026-11-02-definitive.tsv
   python experiments/journal_figures.py --results $D/runs --tsv $T --out out --no-cache
   python experiments/journal_tests.py   --results $D/runs --tsv $T --out out
   python experiments/journal_control.py --results $D/runs --tsv $T --out out
   ```

   `journal_control.py` reads `out/runs.csv`, so it runs after
   `journal_figures.py`. The per-run metrics are cached in `out/_cache/`
   (one cache per version of the metric code); `--no-cache` recomputes them.
   On a laptop the three steps take a few minutes.

### Outputs and the paper

| In the paper | Output in `out/` | Script |
|---|---|---|
| Testbed and workload figure | `fig01_testbed_workload.pdf` | `journal_figures.py` |
| Latency against resources | `fig02_latency_resource_tradeoff.pdf` | `journal_figures.py` |
| Ready frontend replicas per region (concentrated) | `fig04_replica_placement_con.pdf` | `journal_figures.py` |
| Latency per load phase (concentrated) | `fig07_phase_latency_con.pdf` | `journal_figures.py` |
| Level 2 in the concentrated regime | `fig08_predictive_scaler_con.pdf` | `journal_figures.py` |
| Ablation | `fig10_ablation_con.pdf` | `journal_figures.py` |
| Shift regime | `fig14_shift_con-tosg.pdf` | `journal_figures.py` |
| Expected latency cost against the model | `fig15_placement_model.pdf` | `journal_figures.py` |
| Carbon intensity against its bounds | `fig16_carbon_bound.pdf` | `journal_figures.py` |
| Table of the pre-specified tests | `table_tests.tex`, `tests.md`, `tests.csv` | `journal_tests.py` |
| Table of the control-plane cost | `table_control.tex`, `control_timing.csv` | `journal_control.py` |
| Numbers quoted in the text | `runs.csv` (one row per run), `summary.csv` (per cell), `overhead.csv` | `journal_figures.py` |

`journal_figures.py` also writes figures not used in the article (the other
`figNN_*` files) and `index.md`, which describes every output. The tables of
the clusters, the arms and the control constants, the architecture figure and
the run life-cycle figure are written by hand in the article.

## B. From a testbed

1. Build the federation and install the add-ons: [setup-guide.md](setup-guide.md)
   and [KUBEPHOS.md](KUBEPHOS.md). The paper's parameters (netem delays
   0/30/50/100/200/40 ms, frontend caps 4/3/2/4/2/3, 33 req/s per frontend
   replica) are in `config/local.yaml`, written from `config/local.yaml.example`;
   the values actually used are in `config_snapshot/local.sanitized.yaml` of
   every run of the dataset.
2. Run the campaign with the plan:

   ```bash
   bash experiments/run_campaign.sh experiments/plans/2026-11-02-definitive.plan
   ```

   See [experiments/RUNBOOK.md](../experiments/RUNBOOK.md) for the
   environment variables, the per-run checks and how to resume an interrupted
   campaign.
3. Analyse as in A.4, with `--results experiments/results` and
   `--tsv experiments/results/_campaign_2026-11-02-definitive.tsv`.

Absolute latencies depend on the hardware and on the network of the testbed;
the comparisons between arms are what the campaign is designed to measure.

## Files of a run

| File | Content | Used by the analysis |
|---|---|---|
| `flash_crowd_summary.json` | k6 results: latency percentiles per phase, region and shift window; errors; load configuration | yes |
| `dmos_live.jsonl` | one snapshot every 5 s: per cluster, replicas (target and ready), demand, forecast, CPU, served requests, RTTs | yes |
| `dmos_config.json` | effective DMOS configuration and `config_id` (DMOS arms only) | yes (carbon replay settings) |
| `carbon_t0` | anchor of the carbon replay (epoch seconds) | yes |
| `dmos.log` | DMOS log (DMOS arms only) | yes (actuation delays, decision timing) |
| `manifest.json` | provenance: commit, configuration hashes, checks before and after the run | yes (`config_id`) |
| `config_snapshot/` | copies of `weights.yaml`, `services.yaml`, `carbon.yaml` and the sanitised `local.yaml` | yes (cluster model) |
| `env_state_pre.json`, `env_state_post.json` | testbed state before and after the run (netem, affinity, Cilium, HPA) | no (audit) |
| `collector_stdout.txt`, `k6_stdout.txt` | console output of the collector and of k6 | no |
| `dmos_config_expected.json`, `hpa_baseline.json`, `untracked_snapshot/`, `git_diff.patch` | configuration check, HPA objects, files outside git at run time | no (audit) |

Every `collector_stdout.txt` ends with a `UnicodeEncodeError` printed after
the data were written (a console-encoding issue, fixed in this release); it
does not affect the data.
