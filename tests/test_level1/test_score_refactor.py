"""
Smoke test for the post-refactor score function semantics
(src/level1/score_functions.py).

Validates that the new Φ_pressure (inverted sigmoid), Φ_cap (HW size +
slack with floor), and Φ_locality (Φ_demand × Φ_net) combine to give the
HIGHEST total score to clusters with HIGH demand on adequate hardware
during a multi-modal flash crowd workload.

Reference: paper §X refactor, see module-level docstring of
src/level1/score_functions.py for the literature references.
"""

import sys
from pathlib import Path

# Add repo root to path so `src.level1.*` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.level1.score_functions import (
    ScoreFunctions, ScoreParameters, ClusterMetrics
)


def lat_for_util(u: float) -> float:
    """Realistic latency model: low if under-target, ramps up to ~1500ms
    when over-saturated. Matches the observation in the previous pilots."""
    if u < 0.85:
        return 80 + 200 * u             # 80-250ms (healthy)
    return 200 + 1400 * min(2.0, (u - 0.85) / 0.3)   # → ~1500ms (saturated)


# ── Test fixture ─────────────────────────────────────────────────────────
# 6 clusters matching our testbed (config/local.yaml). Distribution:
#   25% + 12% + 10% + 28% + 8% + 17% = 100% (multimodal P1, peak=400rps).
# Two HIGH-demand clusters (01, 04), two MED (02, 06), two LOW (03, 05).
CLUSTERS = [
    # (name, cores, mem, rps_now, rps_max, cpu_af, mem_af, carbon, rtt, demand)
    ('cluster-01', 8, 12, 100, 100, 0.10, 0.50, 350, 85,  0.25),   # tier-HIGH, SAT
    ('cluster-02', 6,  8,  48,  60, 0.40, 0.60,  80, 110, 0.12),   # tier-MED, healthy
    ('cluster-03', 4,  8,  40,  40, 0.15, 0.50, 650, 124, 0.10),   # tier-LOW, SAT
    ('cluster-04', 8, 12, 112, 100, 0.15, 0.50, 380, 164, 0.28),   # tier-HIGH, SAT
    ('cluster-05', 4,  8,  32,  40, 0.30, 0.60, 480, 245, 0.08),   # tier-LOW, healthy
    ('cluster-06', 6,  8,  68,  60, 0.15, 0.50,  25, 117, 0.17),   # tier-MED, SAT
]

WEIGHTS = {
    'omega_latency':  0.10,
    'omega_capacity': 0.20,
    'omega_geo_load': 0.35,    # → weighs Φ_locality (Φ_demand × Φ_net)
    'omega_carbon':   0.10,
    'omega_demand':   0.25,    # → weighs Φ_pressure (inverted sigmoid)
}


def build_metrics(cluster_spec):
    """Build ClusterMetrics for the smoke test."""
    name, cores, mem, rps_now, rps_max, cpu_af, mem_af, carbon, rtt, demand = cluster_spec
    util = rps_now / rps_max
    lat = lat_for_util(util)
    return ClusterMetrics(
        cpu_available_cores=cores * cpu_af,
        cpu_total_cores=cores,
        memory_available_gb=mem * mem_af,
        memory_total_gb=mem,
        request_rate_current=rps_now,
        request_rate_max=rps_max,
        latency_mean_ms=lat,
        latency_variance_ms2=lat * 10,
        carbon_intensity_gco2_kwh=carbon,
        cost_per_replica_hour=0.10,
        network_rtt_ms=rtt,
        ingress_rate_rps=rps_now,
        ingress_demand_share=demand,
    )


def test_hot_clusters_rank_top_2():
    """Top-2 ranked clusters by score must be the 2 HIGH-demand clusters."""
    sf = ScoreFunctions(WEIGHTS, ScoreParameters())
    scores = []
    for spec in CLUSTERS:
        m = build_metrics(spec)
        bd = sf.compute_score_breakdown(m)
        scores.append((spec[0], bd['total_score'], spec[-1]))  # name, score, demand

    ranked = sorted(scores, key=lambda x: -x[1])
    top2_names = {n for n, _, _ in ranked[:2]}
    hot_names = {n for n, _, d in scores if d >= 0.20}

    assert top2_names == hot_names, (
        f"Hot clusters (demand>=20%) should rank top-2.\n"
        f"  ranking: {[(n, f'{s:.3f}') for n,s,_ in ranked]}\n"
        f"  top-2 : {top2_names}\n"
        f"  hot   : {hot_names}"
    )


def test_phi_pressure_inverted_sigmoid():
    """Φ_pressure should INCREASE with utilization (inverted from legacy)."""
    sf = ScoreFunctions(WEIGHTS, ScoreParameters())
    # Build 3 versions of the same cluster at different util levels
    def metrics_at_util(u):
        return ClusterMetrics(
            cpu_available_cores=8 * 0.5, cpu_total_cores=8,
            memory_available_gb=12 * 0.5, memory_total_gb=12,
            request_rate_current=100 * u, request_rate_max=100,
            latency_mean_ms=lat_for_util(u), latency_variance_ms2=100,
            carbon_intensity_gco2_kwh=300, cost_per_replica_hour=0.1,
            network_rtt_ms=100, ingress_rate_rps=100 * u,
            ingress_demand_share=0.2,
        )

    s_low = sf.compute_load_score(metrics_at_util(0.2))
    s_mid = sf.compute_load_score(metrics_at_util(0.7))
    s_high = sf.compute_load_score(metrics_at_util(1.0))

    # Inverted sigmoid: should be monotonically INCREASING up to ~util=1.5
    assert s_low < s_mid < s_high, (
        f"Φ_pressure should increase with util: "
        f"got util=0.2 → {s_low:.3f}, util=0.7 → {s_mid:.3f}, util=1.0 → {s_high:.3f}"
    )
    # At util=0.7 (target), Φ_pressure should be ≈ 0.5
    assert abs(s_mid - 0.5) < 0.05, f"At util=tau=0.7, Φ_pressure should be ≈0.5, got {s_mid:.3f}"


def test_phi_cap_separates_hardware_from_slack():
    """Saturated tier-HIGH must not be zeroed-out by Φ_cap."""
    sf = ScoreFunctions(WEIGHTS, ScoreParameters())

    # Saturated tier-HIGH (8c, ~5% slack):
    m_high_sat = ClusterMetrics(
        cpu_available_cores=8 * 0.05, cpu_total_cores=8,
        memory_available_gb=12 * 0.50, memory_total_gb=12,
        request_rate_current=200, request_rate_max=100,
        latency_mean_ms=1500, latency_variance_ms2=10000,
        carbon_intensity_gco2_kwh=350, cost_per_replica_hour=0.1,
        network_rtt_ms=85, ingress_rate_rps=200, ingress_demand_share=0.4,
    )
    # Idle tier-LOW (4c, plenty of slack):
    m_low_idle = ClusterMetrics(
        cpu_available_cores=4 * 0.9, cpu_total_cores=4,
        memory_available_gb=8 * 0.9, memory_total_gb=8,
        request_rate_current=5, request_rate_max=40,
        latency_mean_ms=100, latency_variance_ms2=500,
        carbon_intensity_gco2_kwh=350, cost_per_replica_hour=0.1,
        network_rtt_ms=85, ingress_rate_rps=5, ingress_demand_share=0.02,
    )

    cap_high_sat = sf.compute_capacity_score(m_high_sat)
    cap_low_idle = sf.compute_capacity_score(m_low_idle)

    # Both should be > 0 (no zero-collapse), and saturated-tier-HIGH must
    # not be CATASTROPHICALLY lower than idle-tier-LOW. Old formula:
    # high_sat ≈ 0, low_idle ≈ 0.5 → diff 0.5. New formula: floor protects.
    assert cap_high_sat > 0.20, (
        f"Saturated tier-HIGH Φ_cap should not collapse to ~0. Got {cap_high_sat:.3f}"
    )
    assert cap_low_idle > 0, f"Idle tier-LOW Φ_cap > 0 expected. Got {cap_low_idle:.3f}"


def test_phi_locality_is_demand_times_net():
    """Φ_locality (via compute_geo_load_score) = Φ_demand × Φ_net."""
    sf = ScoreFunctions(WEIGHTS, ScoreParameters())
    m = build_metrics(CLUSTERS[0])   # cluster-01: demand=0.25, RTT=85
    locality = sf.compute_geo_load_score(m)
    phi_demand = sf.compute_demand_score(m)
    phi_net = sf.compute_network_score(m)
    expected = phi_demand * phi_net
    assert abs(locality - expected) < 1e-9, (
        f"Φ_locality should equal Φ_demand × Φ_net: "
        f"got {locality:.6f}, expected {expected:.6f}"
    )


if __name__ == '__main__':
    print('Running smoke tests for score function refactor...')
    print('-' * 60)
    test_hot_clusters_rank_top_2()
    print('  [PASS] test_hot_clusters_rank_top_2')
    test_phi_pressure_inverted_sigmoid()
    print('  [PASS] test_phi_pressure_inverted_sigmoid')
    test_phi_cap_separates_hardware_from_slack()
    print('  [PASS] test_phi_cap_separates_hardware_from_slack')
    test_phi_locality_is_demand_times_net()
    print('  [PASS] test_phi_locality_is_demand_times_net')
    print('-' * 60)
    print('All smoke tests passed.')
