# Carbon-intensity traces

Hourly life-cycle carbon intensity (gCO₂eq/kWh) for 2025, one file per
Electricity Maps zone, replayed by the `trace` provider of DMOS
(`config/carbon.yaml`, `src/metrics/carbon_client.py`).

| File | Zone | Used by the reference fleet |
|---|---|---|
| `DE_2025_hourly.csv` | Germany | ms-cluster-01 |
| `FR_2025_hourly.csv` | France | ms-cluster-02 |
| `PL_2025_hourly.csv` | Poland | ms-cluster-03 |
| `NO-NO1_2025_hourly.csv` | Norway, Oslo (NO1) | ms-cluster-06 (region code `NO`, via `trace.zone_aliases`) |
| `ES_2025_hourly.csv`, `IT-NO_2025_hourly.csv` | Spain, Italy North | not used |

US-VA (zone `US-MIDA-PJM`) and SG are not included; fetch them with

```bash
python scripts/fetch_carbon_traces.py --zones US-MIDA-PJM SG --year 2025   # token in ELECTRICITYMAPS_TOKEN
```

Format: `Datetime (UTC)`, `Carbon Intensity gCO2eq/kWh (Life cycle)`.

## Source and license

Data: **Electricity Maps** (<https://www.electricitymaps.com>), hourly
carbon-intensity datasets for 2025, retrieved from the Electricity Maps data
portal / API and reduced to the two columns above. The datasets are made
available under the **Open Database License (ODbL)**: if you redistribute
these files or a database derived from them, keep this attribution and
distribute the derived database under the same license. The license of the
DMOS code does not apply to this directory.
