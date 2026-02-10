"""
Test score functions
"""

import pytest
from src.level1.score_functions import (
    ScoreFunctions, 
    ClusterMetrics, 
    ScoreParameters
)

def test_latency_score():
    """Test latency score computation"""
    
    weights = {
        'omega_latency': 0.4,
        'omega_capacity': 0.3,
        'omega_load': 0.1,
        'omega_carbon': 0.2
    }
    
    score_func = ScoreFunctions(weights)
    
    # Good cluster: low latency, low variance
    metrics_good = ClusterMetrics(
        cpu_available_cores=2.0,
        cpu_total_cores=4.0,
        memory_available_gb=4.0,
        memory_total_gb=8.0,
        request_rate_current=100.0,
        request_rate_max=500.0,
        latency_mean_ms=50.0,
        latency_variance_ms2=15.0,
        carbon_intensity_gco2_kwh=300.0,
        cost_per_replica_hour=0.12
    )
    
    # Bad cluster: high latency, high variance
    metrics_bad = ClusterMetrics(
        cpu_available_cores=2.0,
        cpu_total_cores=4.0,
        memory_available_gb=4.0,
        memory_total_gb=8.0,
        request_rate_current=100.0,
        request_rate_max=500.0,
        latency_mean_ms=200.0,
        latency_variance_ms2=400.0,
        carbon_intensity_gco2_kwh=300.0,
        cost_per_replica_hour=0.12
    )
    
    score_good = score_func.compute_latency_score(metrics_good)
    score_bad = score_func.compute_latency_score(metrics_bad)
    
    print(f"\n✅ Latency score test:")
    print(f"   Good cluster (50ms, var=15): {score_good:.3f}")
    print(f"   Bad cluster (200ms, var=400): {score_bad:.3f}")
    
    assert score_good > score_bad, "Good latency should score higher"
    assert 0 <= score_good <= 1, "Score should be in [0,1]"
    assert 0 <= score_bad <= 1, "Score should be in [0,1]"


def test_capacity_score():
    """Test capacity score computation"""
    
    weights = {
        'omega_latency': 0.4,
        'omega_capacity': 0.3,
        'omega_load': 0.1,
        'omega_carbon': 0.2
    }
    
    score_func = ScoreFunctions(weights)
    
    # High capacity available
    metrics_high_cap = ClusterMetrics(
        cpu_available_cores=3.5,  # 87.5% available
        cpu_total_cores=4.0,
        memory_available_gb=7.0,
        memory_total_gb=8.0,
        request_rate_current=50.0,  # Low load
        request_rate_max=500.0,
        latency_mean_ms=50.0,
        latency_variance_ms2=15.0,
        carbon_intensity_gco2_kwh=300.0,
        cost_per_replica_hour=0.12
    )
    
    # Low capacity (saturated)
    metrics_low_cap = ClusterMetrics(
        cpu_available_cores=0.4,  # Only 10% available
        cpu_total_cores=4.0,
        memory_available_gb=0.8,
        memory_total_gb=8.0,
        request_rate_current=450.0,  # High load
        request_rate_max=500.0,
        latency_mean_ms=50.0,
        latency_variance_ms2=15.0,
        carbon_intensity_gco2_kwh=300.0,
        cost_per_replica_hour=0.12
    )
    
    score_high = score_func.compute_capacity_score(metrics_high_cap)
    score_low = score_func.compute_capacity_score(metrics_low_cap)
    
    print(f"\n✅ Capacity score test:")
    print(f"   High capacity (87% free, low load): {score_high:.3f}")
    print(f"   Low capacity (10% free, high load): {score_low:.3f}")
    
    assert score_high > score_low, "High capacity should score higher"


def test_carbon_score():
    """Test carbon score computation"""
    
    weights = {
        'omega_latency': 0.4,
        'omega_capacity': 0.3,
        'omega_load': 0.1,
        'omega_carbon': 0.2
    }
    
    score_func = ScoreFunctions(weights)
    
    # Green cluster (low carbon)
    metrics_green = ClusterMetrics(
        cpu_available_cores=2.0,
        cpu_total_cores=4.0,
        memory_available_gb=4.0,
        memory_total_gb=8.0,
        request_rate_current=100.0,
        request_rate_max=500.0,
        latency_mean_ms=50.0,
        latency_variance_ms2=15.0,
        carbon_intensity_gco2_kwh=100.0,  # Low CI (renewable)
        cost_per_replica_hour=0.12
    )
    
    # Dirty cluster (high carbon)
    metrics_dirty = ClusterMetrics(
        cpu_available_cores=2.0,
        cpu_total_cores=4.0,
        memory_available_gb=4.0,
        memory_total_gb=8.0,
        request_rate_current=100.0,
        request_rate_max=500.0,
        latency_mean_ms=50.0,
        latency_variance_ms2=15.0,
        carbon_intensity_gco2_kwh=450.0,  # High CI (coal)
        cost_per_replica_hour=0.12
    )
    
    score_green = score_func.compute_carbon_score(metrics_green)
    score_dirty = score_func.compute_carbon_score(metrics_dirty)
    
    print(f"\n✅ Carbon score test:")
    print(f"   Green cluster (100 gCO2/kWh): {score_green:.3f}")
    print(f"   Dirty cluster (450 gCO2/kWh): {score_dirty:.3f}")
    
    assert score_green > score_dirty, "Green cluster should score higher"


def test_total_score():
    """Test total multi-dimensional score"""
    
    weights = {
        'omega_latency': 0.4,
        'omega_capacity': 0.3,
        'omega_load': 0.1,
        'omega_carbon': 0.2
    }
    
    score_func = ScoreFunctions(weights)
    
    # Excellent cluster
    metrics_excellent = ClusterMetrics(
        cpu_available_cores=3.5,
        cpu_total_cores=4.0,
        memory_available_gb=7.0,
        memory_total_gb=8.0,
        request_rate_current=50.0,
        request_rate_max=500.0,
        latency_mean_ms=50.0,
        latency_variance_ms2=15.0,
        carbon_intensity_gco2_kwh=100.0,
        cost_per_replica_hour=0.10
    )
    
    # Poor cluster
    metrics_poor = ClusterMetrics(
        cpu_available_cores=0.5,
        cpu_total_cores=4.0,
        memory_available_gb=1.0,
        memory_total_gb=8.0,
        request_rate_current=450.0,
        request_rate_max=500.0,
        latency_mean_ms=200.0,
        latency_variance_ms2=400.0,
        carbon_intensity_gco2_kwh=450.0,
        cost_per_replica_hour=0.20
    )
    
    breakdown_excellent = score_func.compute_score_breakdown(metrics_excellent)
    breakdown_poor = score_func.compute_score_breakdown(metrics_poor)
    
    print(f"\n✅ Total score test:")
    print(f"\n   Excellent cluster:")
    print(f"      Φ_lat: {breakdown_excellent['phi_latency']:.3f}")
    print(f"      Φ_cap: {breakdown_excellent['phi_capacity']:.3f}")
    print(f"      Φ_load: {breakdown_excellent['phi_load']:.3f}")
    print(f"      Φ_carbon: {breakdown_excellent['phi_carbon']:.3f}")
    print(f"      TOTAL: {breakdown_excellent['total_score']:.3f}")
    
    print(f"\n   Poor cluster:")
    print(f"      Φ_lat: {breakdown_poor['phi_latency']:.3f}")
    print(f"      Φ_cap: {breakdown_poor['phi_capacity']:.3f}")
    print(f"      Φ_load: {breakdown_poor['phi_load']:.3f}")
    print(f"      Φ_carbon: {breakdown_poor['phi_carbon']:.3f}")
    print(f"      TOTAL: {breakdown_poor['total_score']:.3f}")
    
    assert breakdown_excellent['total_score'] > breakdown_poor['total_score']


if __name__ == "__main__":
    test_latency_score()
    test_capacity_score()
    test_carbon_score()
    test_total_score()
    print("\n✅ All score function tests passed!")