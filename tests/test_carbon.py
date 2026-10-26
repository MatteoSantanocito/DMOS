"""
Test Carbon client
"""

from pathlib import Path

import yaml

from src.metrics.carbon_client import CarbonClient

REPO = Path(__file__).resolve().parent.parent

def test_carbon_client():
    """Test carbon intensity queries"""
    # carbon.yaml only: the test must also run on a fresh clone without local.yaml
    raw = yaml.safe_load((REPO / "config" / "carbon.yaml").read_text(encoding="utf-8"))
    client = CarbonClient(raw['carbon_intensity'])
    
    # Test mock values
    regions = ['DE', 'US-VA', 'SG']
    
    for region in regions:
        ci = client.get_carbon_intensity(region)
        if ci:
            print(f"✅ Carbon intensity {region}: {ci:.1f} gCO2/kWh")
        else:
            print(f"❌ Failed to get CI for {region}")
    
    # Test cache
    ci_cached = client.get_carbon_intensity('DE')
    print(f"✅ Cache test: {ci_cached:.1f} gCO2/kWh (should be same)")

if __name__ == "__main__":
    test_carbon_client()