"""
Test Carbon client
"""

from src.utils.config_loader import ConfigLoader
from src.metrics.carbon_client import CarbonClient

def test_carbon_client():
    """Test carbon intensity queries"""
    config = ConfigLoader()
    
    client = CarbonClient(config.carbon_raw['carbon_intensity'])
    
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