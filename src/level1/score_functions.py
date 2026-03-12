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
    Metrics per un cluster all'istante t
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
        """frazione di CPU disponibile"""
        if self.cpu_total_cores == 0:
            return 0.0
        return self.cpu_available_cores / self.cpu_total_cores
    
    @property
    def memory_available_fraction(self) -> float:
        """frazione di memoria disponibile"""
        if self.memory_total_gb == 0:
            return 0.0
        return self.memory_available_gb / self.memory_total_gb
    
    @property
    def load_fraction(self) -> float:
        """frazione di carico corrente rispetto al massimo"""
        if self.request_rate_max == 0:
            return 0.0
        return self.request_rate_current / self.request_rate_max


@dataclass
class ScoreParameters:
    """
    Parameters per il calcolo dello score (from config)
    """
    # Latency component
    eta: float = 0.01           # Parametro di sensibilità alla latenza
    sigma_squared: float = 100  # threshold per la penalità sulla varianza (ms^2)
    
    # Capacity component
    kappa: float = 2.0          # espontenziale per la penalità sulla capacità
    
    # Load prediction component
    mu: float = 1.0             # penalità per il carico predetto
    horizon_seconds: int = 600  # orizzonte di predizione per il carico (10 minuti)
    
    # Carbon component
    nu: float = 0.5             # coefficiente per la penalità sulla carbon intensity
    ci_max: float = 500.0       # massimo CI atteso (gCO2/kWh) per normalizzazione


class ScoreFunctions:
    """
    Calcola il multi-dimensional score per la cluster selection:
    score_i = ω_1 * Φ_lat(i) + ω_2 * Φ_cap(i) + ω_3 * Φ_load(i) + ω_4 * Φ_carbon(i)
    """
    
    def __init__(
        self, 
        weights: Dict[str, float],
        parameters: Optional[ScoreParameters] = None
    ):

        self.omega_latency = weights.get('omega_latency', 0.4)
        self.omega_capacity = weights.get('omega_capacity', 0.3)
        self.omega_load = weights.get('omega_load', 0.1)
        self.omega_carbon = weights.get('omega_carbon', 0.2)
        
        # Valida che le pesature devono sommare a 1
        total = self.omega_latency + self.omega_capacity + self.omega_load + self.omega_carbon
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"La somma delle pesature deve essere 1.0, ho {total}")
        
        self.params = parameters or ScoreParameters()
        
        logger.info(f"Score weights: lat={self.omega_latency}, cap={self.omega_capacity}, "
                   f"load={self.omega_load}, carbon={self.omega_carbon}")
    
    def compute_latency_score(self, metrics: ClusterMetrics) -> float:
        """
        Φ_lat(i) = (1 / (1 + η * E[L_i])) * exp(-var(L_i) / σ²)
        
        """
        L_mean = metrics.latency_mean_ms
        L_var = metrics.latency_variance_ms2
        
        # Primo termine: penalità lineare sulla latenza media
        term1 = 1.0 / (1.0 + self.params.eta * L_mean)
        
        # Secondo termine: penalità esponenziale sulla varianza (stabilità)
        term2 = math.exp(-L_var / self.params.sigma_squared)
        
        score = term1 * term2
        
        logger.debug(f"Φ_lat: L_mean={L_mean:.1f}ms, L_var={L_var:.1f}, "
                    f"term1={term1:.3f}, term2={term2:.3f}, score={score:.3f}")
        
        return score
    
    def compute_capacity_score(self, metrics: ClusterMetrics) -> float:
        """
        Φ_cap(i) = (R_i^avail / R_i^tot)^κ * (1 - λ_i / λ_i^max)
        """
        # Usa la frazione di risorse disponibili (CPU e memoria) e prendi il minimo
        resource_fraction = min(
            metrics.cpu_available_fraction,
            metrics.memory_available_fraction
        )
        
        # Applica la penalità esponenziale sulla capacità disponibile
        term1 = resource_fraction ** self.params.kappa
        
        # Termine di penalità sul carico attuale (più è vicino al massimo, più penalizza)
        term2 = 1.0 - metrics.load_fraction
        
        score = term1 * term2
        
        logger.debug(f"Φ_cap: cpu_frac={metrics.cpu_available_fraction:.2f}, "
                    f"mem_frac={metrics.memory_available_fraction:.2f}, "
                    f"load_frac={metrics.load_fraction:.2f}, "
                    f"term1={term1:.3f}, term2={term2:.3f}, score={score:.3f}")
        
        return max(0.0, score)  
    
    def compute_load_score(
        self, 
        metrics: ClusterMetrics, 
        predicted_load: Optional[float] = None
    ) -> float:
        """
        Φ_load(i) = exp(-μ * λ_i^pred / λ_i^max)
        
        """
        # Usa il carico predetto se disponibile, altrimenti quello attuale
        load = predicted_load if predicted_load is not None else metrics.request_rate_current
        
        if metrics.request_rate_max == 0:
            logger.warning("request rate max è 0, non posso calcolare, restituisco 0")
            return 0.0
        
        load_fraction_pred = load / metrics.request_rate_max
        
        score = math.exp(-self.params.mu * load_fraction_pred)
        
        logger.debug(f"Φ_load: load={load:.1f}, max={metrics.request_rate_max:.1f}, "
                    f"frac={load_fraction_pred:.3f}, score={score:.3f}")
        
        return score
    
    def compute_carbon_score(self, metrics: ClusterMetrics) -> float:
        """
        Φ_carbon(i) = exp(-ν * CI_i(t) / CI_max)
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
        score_i = ω_1 * Φ_lat + ω_2 * Φ_cap + ω_3 * Φ_load + ω_4 * Φ_carbon
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
        
        logger.info(f"Score totale: {total_score:.3f} = "
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
        Calcola score con breakdown dettagliato
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