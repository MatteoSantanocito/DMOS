"""
Integration test for Karmada-style architecture
Resource Estimators + Coordinator
"""

import asyncio
from src.utils.config_loader import ConfigLoader
from src.level1.cluster_estimator import ResourceEstimator
from src.level1.dmos_scheduler import Coordinator


async def test_resource_estimator():
    """Test Resource Estimator standalone"""
    print("\n=== Test Resource Estimator ===")
    
    config = ConfigLoader()
    
    estimator = ResourceEstimator(
        cluster_name='cluster1',
        config=config,
        port=8081
    )
    
    await estimator.start()
    
    print(f"✅ Resource Estimator started for cluster1")
    print(f"   GET  http://localhost:8081/metrics")
    print(f"   POST http://localhost:8081/estimate")
    
    await asyncio.sleep(30)


async def test_coordinator():
    """Test Coordinator (requires estimators running)"""
    print("\n=== Test Coordinator (Karmada-style) ===")
    
    config = ConfigLoader()
    coordinator = Coordinator(config=config)
    
    result = await coordinator.schedule_service_detailed(
        service_name='frontend',
        demand=10
    )
    
    print(f"\n✅ Scheduling completed:")
    print(f"   Service: {result['service']}")
    print(f"   Demand: {result['demand']}")
    print(f"   Success: {result['success']}")
    print(f"   Total time: {result['timing']['total_ms']:.1f}ms")
    print(f"\n   Allocations:")
    for alloc in result['allocations']:
        print(f"      {alloc['cluster']}: {alloc['replicas']} replicas ({alloc['quota']:.1%})")
    
    print(f"\n   Scores:")
    for cluster, data in result['scores'].items():
        print(f"      {cluster}: score={data['score']:.3f}, capacity={data['capacity']}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'estimator':
        asyncio.run(test_resource_estimator())
    else:
        asyncio.run(test_coordinator())