# Publications

Papers built on DMOS, newest first. Each entry has its own page with the
bibliographic data, a summary, the figures and the measured results, the
tests performed (with the repetitions actually behind each cell), the exact
code version and configuration that produced the results, and the mapping
from every figure and table to the script that generates it — so any result
can be traced back to the code that produced it. The paper PDFs are not in
the repository.

| Year | Venue | Title | Code | DMOS arm | Page |
|---|---|---|---|---|---|
| 2026 | IEEE NCA 2026 | DMOS: Demand-Gated Multi-Objective Replica Placement for Geo-Distributed Multi-Cluster Systems | `6f11cbd9` (suggested tag `nca-2026`) | `l1l2` (gate g = 2, weights 0.25/0.20/0.25/0.30) | [2026-nca-dmos.md](2026-nca-dmos.md) |

**Origin.** DMOS started as the MSc thesis *Distributed Multi-Objective
Scheduler for Kubernetes Multi-Cluster (DMOS)* (University of Catania,
2026), on an earlier three-cluster testbed; its design documents are in
[`../_archive/`](../_archive/).

**In preparation.** A journal extension of the NCA 2026 paper (external
baselines, role of the demand gate, weight profiles and carbon, generalisation
to other hot clusters, load shapes and a second application). Its runs use the
provenance-tracked harness ([`../RUN-PROVENANCE.md`](../RUN-PROVENANCE.md)),
the test names of `experiments/run_naming.py` and the named score models
(`nca`, `gate_g0`, `gate_g1`, `gate_g3`, `additive_dem50`, `locality`,
`*_first`) defined in `config/weights.yaml`; commands per phase in
[`../../experiments/RUNBOOK.md`](../../experiments/RUNBOOK.md).

## Adding a publication

1. Tag the code used for the camera-ready version, e.g.
   `git tag -a <venue>-<year> <commit> -m "<title>"`.
2. Create `<year>-<venue>-<short-name>.md` from the NCA page: bibliographic
   data, summary, contributions, figures (PNG/SVG copies in a folder of the
   same name), results with the repetitions per cell, configuration (arms
   and their `config_id`s from `run_provenance.py audit`), figure → script
   mapping, how to cite.
3. Add a row to the table above.

Paper sources (`paper/`), run directories and generated artifacts
(`paper_artifacts/`) are kept out of the repository by design (see
`.gitignore`); each page keeps the published figures and the data behind
them, and records where the numbers come from.
