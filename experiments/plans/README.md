# Campaign plans

A plan file lists the cells of a campaign, one per line, in the order in which
`experiments/run_campaign.sh` runs them (format in
[../RUNBOOK.md](../RUNBOOK.md#3-a-campaign)).

| File | What it is |
|---|---|
| `2026-11-02-definitive.plan` | the campaign of the article (release `DMOS-2026.2`): 90 runs, written and committed before the first run |
| `resume/2026-11-02-definitive.plan` | the 7 cells left after the interruption of 3 November 2026, unchanged; it writes to the same campaign index |

The plan is kept exactly as it was fixed, in its original language (Italian),
with the deviations appended at the end as the plan itself prescribes. Its
header, in English:

> Definitive campaign of the journal paper, plan v2. PRE-SPECIFIED: not to be
> modified after the start.
>
> **Code:** scaler v4 (`POLICY_VERSION 4`), even split N/K within the caps for
> `l2`, `reach_green` model, all committed before the start. Analysis only on
> the runs with status OK in the campaign `.tsv`.
>
> **Design:** 5 repetitions (D1 to D5). In every repetition the regimes in
> random order and, within each regime, the arms in random order
> (`random.Random(20261102 + repetition)`). `off` and `hpa-none` in every block
> as a control of drifts. 18 cells per repetition, 90 runs, about 35 h with
> the Cilium windows.
>
> - core: `off`, `hpa-none`, `l2-pool-ps`, `reach-pool-ps` x `bal`, `con`, `con-tosg`
> - ablation: `locality-ps`, `nca-pool-ps` (`reach-pool-ps` from the core) x `con`
> - sustainability: `reach-pool-ps-min0`, `reach_green-pool-ps-min0` x `bal`, `con`
>
> **Primary metrics:** global k6 p95; replica-minutes of the ready frontends;
> dynamic CO2 per successful request; served carbon intensity.
>
> **Tests:** exact permutation test, 5 against 5, one-sided in the stated
> direction (minimum p 1/252 = 0.004), Holm within each family. Effects
> reported as % difference with 95% bootstrap CI.
>
> - F1 (primary, 9 tests): `reach-pool-ps` < `hpa-none` on p95 and on
>   replica-minutes; `reach-pool-ps` < `l2-pool-ps` on p95; in each of the 3 regimes.
> - F2 (sustainability, 6 tests, `bal` and `con`): `reach_green-pool-ps-min0` <
>   `reach-pool-ps-min0` on served CI and CO2 per request; `reach-pool-ps-min0` <
>   `reach-pool-ps` on replica-minutes.
> - F3 (ablation, `con`, 3 tests on p95): `reach-pool-ps` < `nca-pool-ps`;
>   `reach-pool-ps` < `locality-ps`; `nca-pool-ps` vs `locality-ps` two-sided.
> - Descriptive, without tests: errors per phase, p99, pods started, replicas
>   during the decline, SG at 0, comparison with `off`.
>
> **Deviations from the plan:** noted at the end of this file with date and
> reason, without touching the cells.

The deviation recorded at the end of the plan: on 3 November 2026 two virtual
machines of the fleet shut down during the first attempt of cell D5
`nca-pool-ps con`; that run was cancelled and the campaign resumed with the 7
remaining cells in `resume/`, unchanged.

`nca-pool-ps` is the arm called *Local-demand gate* in the article.
