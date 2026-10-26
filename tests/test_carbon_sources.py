"""
Tests for the carbon trace lookup of src/metrics/carbon_client.py:
zone aliases (region code -> Electricity Maps zone), trace file discovery and
the per-region source recorded in each run's provenance.
"""
from pathlib import Path

import yaml

from src.metrics.carbon_client import CarbonClient

REPO = Path(__file__).resolve().parent.parent

CSV = ("Datetime (UTC),Zone id,Carbon Intensity gCO₂eq/kWh (Life cycle)\n"
       "2025-06-02 00:00:00,{z},{a}\n"
       "2025-06-02 01:00:00,{z},{b}\n")


def _client(tmp_path, aliases=None, zones=("DE", "NO-NO1")):
    for z in zones:
        (tmp_path / f"{z}_2025_hourly.csv").write_text(CSV.format(z=z, a=100, b=200), encoding="utf-8")
    return CarbonClient({
        "provider": "trace",
        "trace": {"data_dir": str(tmp_path), "window_start": "2025-06-02T00:00:00",
                  "speedup": 1.0, "zone_aliases": aliases or {}},
        "mock": {"enabled": True, "values": {"XX": 123}},
    })


def test_zone_alias_and_trace_file(tmp_path):
    c = _client(tmp_path, aliases={"NO": "NO-NO1", "US-VA": "US-MIDA-PJM"})
    assert c.trace_zone("NO") == "NO-NO1"
    assert c.trace_zone("DE") == "DE"                      # no alias: the code itself
    assert c.trace_file("NO").endswith("NO-NO1_2025_hourly.csv")
    assert c.trace_file("US-VA") is None                    # alias target not downloaded


def test_source_for_each_fallback(tmp_path):
    c = _client(tmp_path, aliases={"NO": "NO-NO1", "US-VA": "US-MIDA-PJM"})
    assert c.source_for("DE") == "trace:DE_2025_hourly.csv"
    assert c.source_for("NO") == "trace:NO-NO1_2025_hourly.csv"
    assert c.source_for("SG") == "realistic"                # built-in profile exists
    assert c.source_for("XX") == "mock"
    assert c.source_for("ZZ") == "default-300"
    assert CarbonClient({"provider": "realistic"}).source_for("DE") == "realistic"


def test_alias_reads_the_trace_value(tmp_path):
    without = _client(tmp_path)
    with_alias = _client(tmp_path, aliases={"NO": "NO-NO1"})
    assert with_alias.get_carbon_intensity("NO") == 100.0   # first sample of the trace
    # without the alias NO has no file and falls back to the synthetic profile
    assert without.source_for("NO") == "realistic"


def test_repo_config_maps_the_fleet_to_its_trace_zones(tmp_path):
    # read carbon.yaml directly: no config/local.yaml needed (fresh clone).
    # The traces are not distributed with the repository: the repo aliases are
    # checked against stand-in files named like the real ones.
    cfg = yaml.safe_load((REPO / "config" / "carbon.yaml").read_text(encoding="utf-8"))["carbon_intensity"]
    aliases = cfg["trace"]["zone_aliases"]
    # a bare NO key would be parsed by YAML as the boolean False
    assert "NO" in aliases and False not in aliases
    for zone in ("DE", "FR", "PL", "US-MIDA-PJM", "SG", "NO-NO1"):
        (tmp_path / f"{zone}_2025_hourly.csv").write_text(CSV.format(z=zone, a=1, b=2), encoding="utf-8")
    cfg["trace"]["data_dir"] = str(tmp_path)
    c = CarbonClient(cfg)
    expected = {"DE": "DE", "FR": "FR", "PL": "PL", "US-VA": "US-MIDA-PJM", "SG": "SG", "NO": "NO-NO1"}
    for region, zone in expected.items():
        assert c.source_for(region) == f"trace:{zone}_2025_hourly.csv"


def test_trace_files_are_not_tracked_by_git():
    # Electricity Maps data are not redistributed with the repository.
    ignore = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "data/carbon_traces/*.csv" in ignore.splitlines()


def test_yaml_bare_no_is_boolean():
    # documents why region "NO" must be quoted in config files
    assert yaml.safe_load("NO: 1") == {False: 1}
    assert yaml.safe_load('"NO": 1') == {"NO": 1}


PORTAL_CSV = (
    "Datetime (UTC),Country,Zone name,Zone id,Carbon intensity gCO₂eq/kWh (direct),"
    "Carbon intensity gCO₂eq/kWh (Life cycle),Carbon-free energy percentage (CFE%),"
    "Renewable energy percentage (RE%),Data source,Data estimated,Data estimation method\n"
    "2025-01-01T00:00:00.000000,Singapore,Singapore,SG,373.58,493.14,3.24,3.24,emcsg.com,false,\"\"\n"
    "2025-01-01T01:00:00.000000,Singapore,Singapore,SG,356.11,472.39,7.58,7.58,emcsg.com,false,\"\"\n"
)


def test_portal_dataset_format_is_read(tmp_path):
    """CSV datasets of the Electricity Maps portal: ISO datetimes with
    fractional seconds and both direct and life-cycle columns."""
    (tmp_path / "SG_2025_hourly.csv").write_text(PORTAL_CSV, encoding="utf-8")
    c = CarbonClient({"provider": "trace",
                      "trace": {"data_dir": str(tmp_path), "window_start": "2025-01-01T00:00:00",
                                "speedup": 1.0}})
    series = c._load_trace_csv("SG")
    assert [v for _, v in series] == [493.14, 472.39]          # life cycle, not direct
    assert series[1][0] - series[0][0] == 3600


def test_datetime_formats():
    parse = CarbonClient._parse_dt
    expected = parse("2025-06-02 13:00:00")
    for raw in ("2025-06-02T13:00:00.000000", "2025-06-02T13:00:00Z",
                "2025-06-02T13:00:00.000Z", "2025-06-02T13:00:00+00:00", "2025-06-02 13:00"):
        assert parse(raw) == expected, raw
