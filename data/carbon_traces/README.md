# Carbon-intensity traces

The `trace` provider of DMOS (`config/carbon.yaml`, `src/metrics/carbon_client.py`)
replays hourly life-cycle carbon intensity (gCO₂eq/kWh) of 2025, one CSV per
Electricity Maps zone, from this directory.

**The CSV files are not distributed with the repository.** They are
Electricity Maps data, available under Electricity Maps' own terms (for
researchers: the academic access, which does not allow redistribution).
Download them yourself, with your own access, into this directory; git
ignores them (`.gitignore`). Without a file DMOS still runs: that cluster
falls back to the built-in hourly profiles, and every run records the source
of each cluster (`dmos_config.json → carbon.sources`, warning
`carbon_trace_fallback`).

## Files used by the reference fleet

| File | Zone | Cluster |
|---|---|---|
| `DE_2025_hourly.csv` | Germany | ms-cluster-01 |
| `FR_2025_hourly.csv` | France | ms-cluster-02 |
| `PL_2025_hourly.csv` | Poland | ms-cluster-03 |
| `US-MIDA-PJM_2025_hourly.csv` | United States, PJM (Virginia) | ms-cluster-04 (region code `US-VA`, via `trace.zone_aliases`) |
| `SG_2025_hourly.csv` | Singapore | ms-cluster-05 |
| `NO-NO1_2025_hourly.csv` | Norway, Oslo (NO1) | ms-cluster-06 (region code `NO`, via `trace.zone_aliases`) |

The NCA 2026 campaign used traces for DE, FR and PL only (NO, US-VA and SG
ran on the built-in profiles).

**Versions used for the journal experiments** (record yours here when you
download):

| Zones | Source | Version | Hours estimated by Electricity Maps |
|---|---|---|---|
| DE, FR, PL, US-MIDA-PJM, SG, NO-NO1 | portal datasets, Carbon Intensity, hourly, 2025 | 2026-02-10 | DE 12, FR 5, PL 0, US-MIDA-PJM 12, SG 135, NO-NO1 0 (of 8760) |

The NCA 2026 runs used earlier files of DE, FR and PL downloaded through the
API on 2026-05-29: same annual means (within 0.2 gCO₂eq/kWh), different hourly values (DE: correlation 0.93)
(revisions between data versions).

## How to get them

1. **Access.** Electricity Maps no longer offers dataset downloads to general
   users; researchers can request the academic access by signing up in the
   Electricity Maps portal with an institutional email address (see
   *Academic data access and availability* in the Electricity Maps Help
   Center). Check on the portal's Coverage page that the zones have hourly
   data for 2025.
2. **Download**, either:
   - through the API, with the script of this repository:
     ```bash
     export ELECTRICITYMAPS_TOKEN=<your key>      # PowerShell: $env:ELECTRICITYMAPS_TOKEN = "<your key>"
     python scripts/fetch_carbon_traces.py --zones DE FR PL US-MIDA-PJM SG NO-NO1 --year 2025
     ```
     (the key needs access to historical data, `past-range`: a plain API
     trial gets HTTP 401 "This endpoint is not included in your API trial");
   - or as CSV datasets from the portal: signal Carbon Intensity, hourly
     granularity, year 2025. The portal names the file like
     `snapshots_<version>_<ZONE>-2025-hourly.csv`: rename it
     `<ZONE>_2025_hourly.csv`. Pick the zone of the cluster, not its country
     (Virginia is the PJM zone `US-MIDA-PJM`, not `US`).
3. **Check** that DMOS finds them:
   ```bash
   python -m src.dmos_main --dump-config-only cfg_check.json
   ```
   The summary line ends with `carbon trace 6/6`.

Format read by DMOS: a column whose name contains `Datetime` and one whose
name contains `Life cycle` (else `direct`), one row per hour, UTC. Both the
files written by the script (`Datetime (UTC)`,
`Carbon Intensity gCO2eq/kWh (Life cycle)`, `2025-01-01 00:00:00`) and the
portal datasets (more columns, `2025-01-01T00:00:00.000000`) are read as
they are.

## Attribution

Publications that use these data must cite Electricity Maps as the data
source (<https://www.electricitymaps.com>).
