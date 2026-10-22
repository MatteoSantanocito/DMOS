# Start-up behaviour: observation window and two-phase score

DMOS treats the first two minutes after start differently from steady
operation. As in the NCA 2026 paper (§IV-E, "two-phase startup"), the first
120 s are a **cold start**: the live metrics (latency, load) are not yet
reliable, so DMOS **only listens to the system** — it reads the traffic and
fills the predictors' history, but performs no scaling operation. The first
placement decision is taken after that window. This page explains the two
mechanisms that implement it. The rest of the design is in
[dmos-architecture.md](dmos-architecture.md).

## The two mechanisms

| Mechanism | Where | Duration | What it does |
|---|---|---|---|
| **Start-up grace** | `DMOSOrchestrator.startup_grace_seconds` (`src/dmos_main.py`) | 120 s from the orchestrator's start | every scheduling cycle returns right after reading the traffic: no score is computed and no Kubernetes operation is issued; the predictors record their history |
| **Cold-phase score** | `COLD_START_SECONDS` (`src/level1/dmos_scheduler.py`) | 120 s from the scheduler's start | defines the score of the cold phase (`score_func_cold`) and collects the per-cluster metrics in "cold mode" |

### Why the cold phase relies on structural signals

The rationale (as in FIRM, Sinan or Cilantro: first structural signals, then
observed ones) is that some observed metrics are unreliable right after a
deploy: p95 latency is dominated by warm-up (JIT/GC of `adservice`, Python
and Node.js services), CPU readings spike during start-up, and the
request-rate history is short.

In cold mode `_collect_cluster_metrics`:

- does not query Hubble for latency: `latency_mean_ms` is the cluster's
  `latency.baseline_ms` from `local.yaml` (2 ms on the testbed);
- uses the ingress demand λᵢⁱⁿ as `request_rate_current` instead of the
  frontend's Hubble rate (this only matters for Φ_pressure, legacy path);
- collects CPU, memory, RTT and carbon as usual.

The cold score function is:

| Configuration | Cold weights |
|---|---|
| score model with `cold_start: structural` (all current models) | ω_lat = ω_cap = 0, the other weights renormalised; the demand keeps its role (e.g. `nca`: gate g = 2 on net 0.75 / carbon 0.25) — the paper's description |
| score model with `cold_start: same` | the warm weights |
| legacy path (e.g. arm `l1l2`) | the `cold_start` profile of `weights.yaml` (0 / 0 / 0.35 / 0.30 / 0.35), modified by the same environment switches as the warm phase (with `DMOS_WEIGHTS_COHERENT` the cold weights equal the warm ones) |

The Level-2 loop is the same in both phases: the predictor and the PD
correction always work on the per-cluster ingress demand λᵢⁱⁿ.

## How the two windows line up

The scheduler is created inside the orchestrator's constructor, a fraction
of a second (or a few seconds, if a Prometheus probe times out) before the
orchestrator records its own start time. Both windows last 120 s, so the
grace period ends right after the cold phase:

```
t_s  scheduler start           cold phase   [t_s, t_s + 120 s)
t_o  orchestrator start (≥ t_s) grace period [t_o, t_o + 120 s)

first cycle that passes the grace check: t ≥ t_o + 120 s ≥ t_s + 120 s → warm score
```

Consequences, all intended:

- the first 120 s are observation only; the first placement decision (the
  first cycle after the grace, between 120 and 150 s) uses the warm score;
- the cold score function is built and recorded in the provenance
  (`level1.cold` in `dmos_config.json`) as the definition of the cold phase,
  but no allocation is computed during that phase;
- runs that differ only in their cold-start policy therefore behave
  identically.

In the flash-crowd workload the load starts with a 180 s baseline at 50
req/s, so the observation window falls in the baseline phase of k6 and the
spike starts at t = 180 s of k6 (DMOS is started before k6 by the harness).

The grace was introduced because aggressive early scale-ups on a short
predictor history disrupted connections on the NodePorts while the load
generator was opening sessions. Making the cold score drive allocations
would require a shorter grace (cycles between its end and 120 s would use
the cold score); it is not done, so the code matches the paper.

## Logs

The phase banner printed by `collect_scores` appears only from the first
cycle after the grace, so normally only the warm banner is seen:

```
[Phase 2 — DYNAMIC RESCHEDULING] t=1xx s | ...
```

During the grace each cycle logs instead:

```
⏳ Grace period: <elapsed>/120s (<remaining>s remaining) — traffic observed, no k8s ops
```

The Phase 2 banner text still names the legacy components
(`Φ_resp + Φ_cap + Φ_geo(load×net) + Φ_carbon + Φ_demand`) whatever the score
in use; the formula actually used is printed at start-up ("Score model …" or
"Score weights …") and recorded in `dmos_config.json`.

## References

- Qiu et al., *FIRM*, OSDI 2020; Zhang et al., *Sinan*, ASPLOS 2021;
  Bhardwaj et al., *Cilantro*, OSDI 2023 — placement without performance
  data followed by feedback-driven rescheduling.
