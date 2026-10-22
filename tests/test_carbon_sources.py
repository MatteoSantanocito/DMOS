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


def test_repo_config_maps_norway_to_its_trace():
    # read carbon.yaml directly: no config/local.yaml needed (fresh clone)
    cfg = yaml.safe_load((REPO / "config" / "carbon.yaml").read_text(encoding="utf-8"))["carbon_intensity"]
    aliases = cfg["trace"]["zone_aliases"]
    # a bare NO key would be parsed by YAML as the boolean False
    assert "NO" in aliases and False not in aliases
    c = CarbonClient(cfg)
    assert c.source_for("NO").startswith("trace:NO-NO1")
    assert c.source_for("DE").startswith("trace:DE")


def test_yaml_bare_no_is_boolean():
    # documents why region "NO" must be quoted in config files
    assert yaml.safe_load("NO: 1") == {False: 1}
    assert yaml.safe_load('"NO": 1') == {"NO": 1}
