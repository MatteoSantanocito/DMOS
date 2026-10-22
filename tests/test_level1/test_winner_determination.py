"""
Test winner determination algorithm
"""

from src.level1.winner_determination import (
    WinnerDetermination,
    ClusterBid,
    Allocation
)

def test_greedy_allocation():
    """Test greedy allocation algorithm"""
    
    wd = WinnerDetermination()
    
    # Create bids
    bids = [
        ClusterBid(cluster_name="cluster1", score=0.85, capacity=10),
        ClusterBid(cluster_name="cluster2", score=0.70, capacity=8),
        ClusterBid(cluster_name="cluster3", score=0.45, capacity=5),
    ]
    
    demand = 15
    
    allocations, success = wd.allocate(bids, demand)
    
    print("\n✅ Greedy allocation test:")
    print(f"   Demand: {demand} replicas")
    print(f"   Success: {success}")
    
    for alloc in allocations:
        print(f"   {alloc}")
    
    # Verify
    assert success == True, "Should satisfy demand"
    # Score-proportional LRM allocator distributes across all clusters
    # in proportion to score, then assigns remainders by largest fractional part.
    # With scores (0.85, 0.70, 0.45) → quotas (0.425, 0.350, 0.225) → 15 replicas
    # → (6.375, 5.250, 3.375) → integer floors (6, 5, 3) + 1 LRM remainder → (7, 5, 3).
    assert len(allocations) == 3, "Should distribute across all 3 clusters"

    # cluster1 has the highest score → most replicas
    assert allocations[0].cluster_name == "cluster1"
    assert allocations[0].replicas == 7
    assert allocations[1].cluster_name == "cluster2"
    assert allocations[1].replicas == 5
    assert allocations[2].cluster_name == "cluster3"
    assert allocations[2].replicas == 3

    # Total must equal demand
    assert sum(a.replicas for a in allocations) == demand

    # Test quotas
    assert abs(allocations[0].quota - 7/15) < 1e-6
    assert abs(allocations[1].quota - 5/15) < 1e-6
    assert abs(allocations[2].quota - 3/15) < 1e-6


def test_insufficient_capacity():
    """Test allocation with insufficient capacity"""
    
    wd = WinnerDetermination()
    
    bids = [
        ClusterBid(cluster_name="cluster1", score=0.85, capacity=5),
        ClusterBid(cluster_name="cluster2", score=0.70, capacity=3),
    ]
    
    demand = 20  # More than total capacity (8)
    
    allocations, success = wd.allocate(bids, demand)
    
    print("\n⚠️  Insufficient capacity test:")
    print(f"   Demand: {demand}, Total capacity: 8")
    print(f"   Success: {success}")
    
    assert success == False, "Should fail due to insufficient capacity"
    
    # Should still allocate what's possible
    total_allocated = sum(a.replicas for a in allocations)
    assert total_allocated == 8, "Should allocate all available"


def test_jain_fairness():
    """Test Jain fairness index computation"""
    
    wd = WinnerDetermination()
    
    # Perfect fairness: equal quotas
    allocations_fair = [
        Allocation("c1", replicas=5, quota=0.333, score=0.8),
        Allocation("c2", replicas=5, quota=0.333, score=0.7),
        Allocation("c3", replicas=5, quota=0.334, score=0.6),
    ]
    
    jain_fair = wd.compute_fairness_jain_index(allocations_fair)
    
    # Unfair: one cluster gets most
    allocations_unfair = [
        Allocation("c1", replicas=14, quota=0.70, score=0.9),
        Allocation("c2", replicas=4, quota=0.20, score=0.5),
        Allocation("c3", replicas=2, quota=0.10, score=0.3),
    ]
    
    jain_unfair = wd.compute_fairness_jain_index(allocations_unfair)
    
    print("\n✅ Jain fairness test:")
    print(f"   Fair allocation: {jain_fair:.3f}")
    print(f"   Unfair allocation: {jain_unfair:.3f}")
    
    assert jain_fair > jain_unfair, "Fair should have higher Jain index"
    assert jain_fair > 0.99, "Equal quotas should give ~1.0"


if __name__ == "__main__":
    test_greedy_allocation()
    test_insufficient_capacity()
    test_jain_fairness()
    print("\n✅ All winner determination tests passed!")