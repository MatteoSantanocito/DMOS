"""
Multi-dimensional score functions for cluster selection
Implements equations from DMOS paper
"""

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..utils.logger import get_logger

logger = get_logger("ScoreFunctions")


@dataclass
class ClusterMetrics:
    """
    Metrics for a single cluster at time t
    """
    # Resources
    cpu_available_cores: float
    cpu_total_cores: float
    memory_available_gb: float
    memory_total_gb: float
    
    # Traffic
    request_rate_current: float  # λ_i(t) in req/s
    request_rate_max: float      # λ_i^max capacity
    
    # Latency
    latency_mean_ms: float       # E[L_i]
    latency_variance_ms2: float  # var(L_i)
    
    # Carbon
    carbon_intensity_gco2_kwh: float  # CI_i(t)
    
    # Cost
    cost_per_replica_hour: float  # Π_i
    
    @property
    def cpu_available_fraction(self) -> float:
        """Fraction of CPU available"""
        if self.cpu_total_cores == 0:
            return 0.0
        return self.cpu_available_cores / self.cpu_total_cores
    
    @property
    def memory_available_fraction(self) -> float:
        """Fraction of memory available"""
        if self.memory_total_gb == 0:
            return 0.0
        return self.memory_available_gb / self.memory_total_gb
    
    @property
    def load_fraction(self) -> float:
        """Current load as fraction of max"""
        if self.request_rate_max == 0:
            return 0.0
        return self.request_rate_current / self.request_rate_max


@dataclass
class ScoreParameters:
    """
    Parameters for score computation (from config)
    """
    # Latency component
    eta: float = 0.01           # Sensitivity parameter
    sigma_squared: float = 100  # Variance threshold
    
    # Capacity component
    kappa: float = 2.0          # Quadratic penalty exponent
    
    # Load prediction component
    mu: float = 1.0             # Load penalty coefficient
    horizon_seconds: int = 600  # Prediction horizon (10 min)
    
    # Carbon component
    nu: float = 0.5             # Carbon penalty coefficient
    ci_max: float = 500.0       # Max carbon intensity for normalization


class ScoreFunctions:
    """
    Compute multi-dimensional score for cluster selection
    
    Implements equation from paper:
    score_i = ω_1 * Φ_lat(i) + ω_2 * Φ_cap(i) + ω_3 * Φ_load(i) + ω_4 * Φ_carbon(i)
    """
    
    def __init__(
        self, 
        weights: Dict[str, float],
        parameters: Optional[ScoreParameters] = None
    ):
        """
        Initialize score functions
        
        Args:
            weights: Dict with keys omega_latency, omega_capacity, omega_load, omega_carbon
            parameters: Optional ScoreParameters (uses defaults if None)
        """
        self.omega_latency = weights.get('omega_latency', 0.4)
        self.omega_capacity = weights.get('omega_capacity', 0.3)
        self.omega_load = weights.get('omega_load', 0.1)
        self.omega_carbon = weights.get('omega_carbon', 0.2)
        
        # Validate weights sum to 1
        total = self.omega_latency + self.omega_capacity + self.omega_load + self.omega_carbon
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Score weights must sum to 1.0, got {total}")
        
        self.params = parameters or ScoreParameters()
        
        logger.info(f"Score weights: lat={self.omega_latency}, cap={self.omega_capacity}, "
                   f"load={self.omega_load}, carbon={self.omega_carbon}")
    
    def compute_latency_score(self, metrics: ClusterMetrics) -> float:
        """
        Compute latency component Φ_lat(i)
        
        From paper:
        Φ_lat(i) = (1 / (1 + η * E[L_i])) * exp(-var(L_i) / σ²)
        
        Args:
            metrics: Cluster metrics
        
        Returns:
            Latency score in [0, 1] (higher is better)
        """
        L_mean = metrics.latency_mean_ms
        L_var = metrics.latency_variance_ms2
        
        # First term: soft threshold on mean latency
        term1 = 1.0 / (1.0 + self.params.eta * L_mean)
        
        # Second term: exponential penalty on variance
        term2 = math.exp(-L_var / self.params.sigma_squared)
        
        score = term1 * term2
        
        logger.debug(f"Φ_lat: L_mean={L_mean:.1f}ms, L_var={L_var:.1f}, "
                    f"term1={term1:.3f}, term2={term2:.3f}, score={score:.3f}")
        
        return score
    
    def compute_capacity_score(self, metrics: ClusterMetrics) -> float:
        """
        Compute capacity component Φ_cap(i)
        
        From paper:
        Φ_cap(i) = (R_i^avail / R_i^tot)^κ * (1 - λ_i / λ_i^max)
        
        Args:
            metrics: Cluster metrics
        
        Returns:
            Capacity score in [0, 1] (higher is better)
        """
        # Use minimum of CPU and memory fractions (conservative)
        resource_fraction = min(
            metrics.cpu_available_fraction,
            metrics.memory_available_fraction
        )
        
        # Apply quadratic penalty (κ = 2)
        term1 = resource_fraction ** self.params.kappa
        
        # Traffic-based term
        term2 = 1.0 - metrics.load_fraction
        
        score = term1 * term2
        
        logger.debug(f"Φ_cap: cpu_frac={metrics.cpu_available_fraction:.2f}, "
                    f"mem_frac={metrics.memory_available_fraction:.2f}, "
                    f"load_frac={metrics.load_fraction:.2f}, "
                    f"term1={term1:.3f}, term2={term2:.3f}, score={score:.3f}")
        
        return max(0.0, score)  # Ensure non-negative
    
    def compute_load_score(
        self, 
        metrics: ClusterMetrics, 
        predicted_load: Optional[float] = None
    ) -> float:
        """
        Compute load prediction component Φ_load(i)
        
        From paper:
        Φ_load(i) = exp(-μ * λ_i^pred / λ_i^max)
        
        Args:
            metrics: Cluster metrics
            predicted_load: Optional predicted load (if None, uses current load)
        
        Returns:
            Load score in [0, 1] (higher is better)
        """
        # Use predicted load if provided, else current load
        load = predicted_load if predicted_load is not None else metrics.request_rate_current
        
        if metrics.request_rate_max == 0:
            logger.warning("Max request rate is 0, returning score 0")
            return 0.0
        
        load_fraction_pred = load / metrics.request_rate_max
        
        score = math.exp(-self.params.mu * load_fraction_pred)
        
        logger.debug(f"Φ_load: load={load:.1f}, max={metrics.request_rate_max:.1f}, "
                    f"frac={load_fraction_pred:.3f}, score={score:.3f}")
        
        return score
    
    def compute_carbon_score(self, metrics: ClusterMetrics) -> float:
        """
        Compute carbon component Φ_carbon(i)
        
        From paper:
        Φ_carbon(i) = exp(-ν * CI_i(t) / CI_max)
        
        Args:
            metrics: Cluster metrics
        
        Returns:
            Carbon score in [0, 1] (higher is better, lower carbon is better)
        """
        ci_normalized = metrics.carbon_intensity_gco2_kwh / self.params.ci_max
        
        score = math.exp(-self.params.nu * ci_normalized)
        
        logger.debug(f"Φ_carbon: CI={metrics.carbon_intensity_gco2_kwh:.1f} gCO2/kWh, "
                    f"normalized={ci_normalized:.3f}, score={score:.3f}")
        
        return score
    
    def compute_total_score(
        self, 
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> float:
        """
        Compute total multi-dimensional score
        
        From paper:
        score_i = ω_1 * Φ_lat + ω_2 * Φ_cap + ω_3 * Φ_load + ω_4 * Φ_carbon
        
        Args:
            metrics: Cluster metrics
            predicted_load: Optional predicted load for Φ_load
        
        Returns:
            Total score in [0, 1] (higher is better)
        """
        phi_lat = self.compute_latency_score(metrics)
        phi_cap = self.compute_capacity_score(metrics)
        phi_load = self.compute_load_score(metrics, predicted_load)
        phi_carbon = self.compute_carbon_score(metrics)
        
        total_score = (
            self.omega_latency * phi_lat +
            self.omega_capacity * phi_cap +
            self.omega_load * phi_load +
            self.omega_carbon * phi_carbon
        )
        
        logger.info(f"Total score: {total_score:.3f} = "
                   f"{self.omega_latency}*{phi_lat:.3f} + "
                   f"{self.omega_capacity}*{phi_cap:.3f} + "
                   f"{self.omega_load}*{phi_load:.3f} + "
                   f"{self.omega_carbon}*{phi_carbon:.3f}")
        
        return total_score
    
    def compute_score_breakdown(
        self, 
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute score with detailed breakdown
        
        Args:
            metrics: Cluster metrics
            predicted_load: Optional predicted load
        
        Returns:
            Dict with individual scores and total
        """
        phi_lat = self.compute_latency_score(metrics)
        phi_cap = self.compute_capacity_score(metrics)
        phi_load = self.compute_load_score(metrics, predicted_load)
        phi_carbon = self.compute_carbon_score(metrics)
        
        total = (
            self.omega_latency * phi_lat +
            self.omega_capacity * phi_cap +
            self.omega_load * phi_load +
            self.omega_carbon * phi_carbon
        )
        
        return {
            'phi_latency': phi_lat,
            'phi_capacity': phi_cap,
            'phi_load': phi_load,
            'phi_carbon': phi_carbon,
            'total_score': total,
            'weights': {
                'omega_latency': self.omega_latency,
                'omega_capacity': self.omega_capacity,
                'omega_load': self.omega_load,
                'omega_carbon': self.omega_carbon
            }
        }