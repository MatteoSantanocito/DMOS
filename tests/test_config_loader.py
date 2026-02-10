"""
Test ConfigLoader
"""

import pytest
from src.utils.config_loader import ConfigLoader

def test_config_loader():
    """Test loading configuration"""
    config = ConfigLoader(config_dir="config")
    
    # Test clusters loaded
    assert len(config.clusters) == 3
    assert 'cluster1' in config.clusters
    assert 'cluster2' in config.clusters
    assert 'cluster3' in config.clusters
    
    # Test cluster properties
    cluster1 = config.get_cluster('cluster1')
    assert cluster1.region == 'EU'
    assert cluster1.cpu_cores == 4
    assert cluster1.memory_gb == 7.75
    
    # Test weights
    assert abs(config.weights.alpha + config.weights.beta + 
               config.weights.gamma + config.weights.delta - 1.0) < 1e-6
    
    # Test services
    assert len(config.services) > 0
    frontend = config.get_service('frontend')
    assert frontend is not None
    assert frontend.namespace == 'online-boutique'
    
    print("✅ ConfigLoader test passed")

if __name__ == "__main__":
    test_config_loader()