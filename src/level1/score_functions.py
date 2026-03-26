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

    # Network / Geo-awareness
    # RTT media dal cluster i verso tutti i peer cluster (ms).
    # Misurata da ping_exporter (czerwonk/ping_exporter) con target = IP nodi peer.
    # Con tc netem attivo: cluster1≈250ms, cluster2≈325ms, cluster3≈425ms (avg verso peer).
    # Senza netem (pura LAN): ≈ 5ms per tutti i cluster.
    network_rtt_ms: float = 5.0

    # Demand / Ingress-level traffic
    # λ_ingress(i): rate delle richieste che entrano nel cluster via Cilium Ingress (req/s).
    # Fonte: hubble_http_requests_total{destination_workload="frontend",
    #           destination_namespace="online-boutique", reporter="server",
    #           traffic_direction="ingress"}
    # Cilium Ingress usa Envoy come proxy → Hubble L7 misura traffico inbound al frontend.
    # Nota: source="reserved:ingress" non è esposto (labelsContext non include source_identity);
    # il filtro reporter="server"+traffic_direction="ingress" è equivalente funzionale.
    # Hubble L7 abilitato dalla CNP con rules: http su fromEntities: ingress.
    ingress_rate_rps: float = 0.0

    # Φ_demand pre-normalizzata: ingress_rate_i / Σ_j ingress_rate_j
    # Pre-calcolata in collect_scores() dove si ha il totale su tutti i cluster.
    # In [0,1], con Σ_i ingress_demand_share = 1.
    # Fallback: 1/N (distribuzione uniforme) se Hubble L7 non disponibile.
    ingress_demand_share: float = 0.0
    
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
    eta: float = 0.001                  # Parametro di sensibilità alla latenza
    # FIX: era 0.01 — con P95 reali in range 3000–5000ms (Online Boutique, catena gRPC),
    # phi_lat = 1/(1+0.01×4000) ≈ 0.024 per tutti → differenziazione quasi nulla (Δ≈0.01).
    # Con η=0.001: phi_lat(3000ms)=0.250, phi_lat(4500ms)=0.182 → Δ=0.068 significativo.
    sigma_squared: float = 1_000_000    # threshold per la penalità sulla varianza (ms^2)
    # FIX: era 100 ms^2 — con quel valore anche P95=100ms dà variance≈573 →
    # exp(-573/100)≈0 → phi_lat≈0 per tutti i cluster. Con 1_000_000 ms^2,
    # term2≈1 per P95 < 3000ms e si penalizza solo varianza estrema (>3s).
    # Differenziazione ora affidata esclusivamente a term1 = 1/(1+η×mean).
    
    # Capacity component
    kappa: float = 2.0          # espontenziale per la penalità sulla capacità
    
    # Load prediction component
    mu: float = 1.0             # penalità per il carico predetto
    horizon_seconds: int = 600  # orizzonte di predizione per il carico (10 minuti)
    
    # Carbon component
    nu: float = 0.5             # coefficiente per la penalità sulla carbon intensity
    ci_max: float = 500.0       # massimo CI atteso (gCO2/kWh) per normalizzazione

    # Network / Geo-awareness component
    # Φ_net(i) = exp(-ρ × RTT_avg_i / RTT_max)
    # ρ: sensibilità — più alto = penalità più rapida per alta RTT
    # RTT_max: RTT massima attesa — normalizza la RTT per mantenerla in [0,1]
    rho: float = 2.0            # parametro di sensibilità alla RTT
    rtt_max_ms: float = 500.0   # RTT massima (ms) per normalizzazione


class ScoreFunctions:
    """
    Calcola il multi-dimensional score per la cluster selection:
    score_i = ω_lat·Φ_lat(i) + ω_cap·Φ_cap(i) + ω_load·Φ_load(i)
            + ω_carbon·Φ_carbon(i) + ω_net·Φ_net(i) + ω_demand·Φ_demand(i)
    """

    def __init__(
        self,
        weights: Dict[str, float],
        parameters: Optional[ScoreParameters] = None
    ):
        self.omega_latency = weights.get('omega_latency', 0.30)
        self.omega_capacity = weights.get('omega_capacity', 0.20)
        self.omega_load = weights.get('omega_load', 0.15)
        self.omega_carbon = weights.get('omega_carbon', 0.20)
        self.omega_network = weights.get('omega_network', 0.05)
        self.omega_demand = weights.get('omega_demand', 0.10)

        # Valida che le pesature sommano a 1.0
        total = (self.omega_latency + self.omega_capacity +
                 self.omega_load + self.omega_carbon +
                 self.omega_network + self.omega_demand)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"La somma delle pesature deve essere 1.0, ho {total}")

        self.params = parameters or ScoreParameters()

        logger.info(
            f"Score weights: lat={self.omega_latency}, cap={self.omega_capacity}, "
            f"load={self.omega_load}, carbon={self.omega_carbon}, "
            f"network={self.omega_network}, demand={self.omega_demand}"
        )
    
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
    
    def compute_network_score(self, metrics: ClusterMetrics) -> float:
        """
        Φ_net(i) = exp(-ρ × min(RTT_i, RTT_max) / RTT_max)

        Misura la "vicinanza geografica" del cluster rispetto agli utenti,
        usando la RTT media verso i peer cluster come proxy della distanza.

        - Alta RTT verso i peer → cluster geograficamente periferico → score basso
        - Bassa RTT verso i peer → cluster centrale/vicino agli utenti → score alto

        Con tc netem attivo (simulazione geografica):
          cluster1 (Frankfurt): RTT_avg ≈ 250ms → Φ_net ≈ 0.368
          cluster2 (Paris):     RTT_avg ≈ 325ms → Φ_net ≈ 0.272
          cluster3 (Warsaw):    RTT_avg ≈ 425ms → Φ_net ≈ 0.183

        Senza netem (pura LAN, tutti ≈ 5ms):
          Φ_net ≈ exp(-2×5/500) = exp(-0.02) ≈ 0.980 per tutti i cluster
          → differenziazione nulla, omega_network ha impatto trascurabile.
        """
        rtt = min(metrics.network_rtt_ms, self.params.rtt_max_ms)
        score = math.exp(-self.params.rho * rtt / self.params.rtt_max_ms)

        logger.debug(
            f"Φ_net: RTT={metrics.network_rtt_ms:.1f}ms "
            f"(capped={rtt:.1f}ms), ρ={self.params.rho}, "
            f"RTT_max={self.params.rtt_max_ms:.0f}ms, score={score:.3f}"
        )
        return score

    def compute_demand_score(self, metrics: ClusterMetrics) -> float:
        """
        Φ_demand(i) = λ_ingress(i) / Σ_j λ_ingress(j)

        Misura la domanda geografica: la frazione di traffico totale che arriva
        in ingresso su questo cluster via Nginx Ingress Controller.

        Il valore è pre-normalizzato in [0,1] e salvato in
        metrics.ingress_demand_share da collect_scores() in dmos_scheduler.py.

        Con Cilium Ingress disponibile:
          Φ_demand(cluster1)=0.60 → 60% degli utenti entrano da cluster1
          → DMOS alloca più repliche su cluster1 per servire gli utenti localmente

        Senza Hubble L7 (fallback):
          Φ_demand(i) = 1/N per tutti i cluster → nessun effetto differenziale

        Nota: Φ_demand è lineare (non esponenziale come le altre componenti)
        perché la frazione di traffico è già una quantità normalizzata in [0,1]
        con semantica diretta: 1.0 = tutto il traffico, 0.0 = nessun traffico.
        """
        score = min(1.0, max(0.0, metrics.ingress_demand_share))

        logger.debug(
            f"Φ_demand: ingress={metrics.ingress_rate_rps:.1f} req/s, "
            f"share={metrics.ingress_demand_share:.3f}, score={score:.3f}"
        )
        return score

    def compute_total_score(
        self,
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> float:
        """
        score_i = ω_lat·Φ_lat + ω_cap·Φ_cap + ω_load·Φ_load
                + ω_carbon·Φ_carbon + ω_net·Φ_net + ω_demand·Φ_demand
        """
        phi_lat = self.compute_latency_score(metrics)
        phi_cap = self.compute_capacity_score(metrics)
        phi_load = self.compute_load_score(metrics, predicted_load)
        phi_carbon = self.compute_carbon_score(metrics)
        phi_net = self.compute_network_score(metrics)
        phi_demand = self.compute_demand_score(metrics)

        total_score = (
            self.omega_latency * phi_lat +
            self.omega_capacity * phi_cap +
            self.omega_load * phi_load +
            self.omega_carbon * phi_carbon +
            self.omega_network * phi_net +
            self.omega_demand * phi_demand
        )

        logger.info(
            f"Score totale: {total_score:.3f} = "
            f"{self.omega_latency}×{phi_lat:.3f}(lat) + "
            f"{self.omega_capacity}×{phi_cap:.3f}(cap) + "
            f"{self.omega_load}×{phi_load:.3f}(load) + "
            f"{self.omega_carbon}×{phi_carbon:.3f}(carbon) + "
            f"{self.omega_network}×{phi_net:.3f}(net) + "
            f"{self.omega_demand}×{phi_demand:.3f}(demand)"
        )

        return total_score

    def compute_score_breakdown(
        self,
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calcola score con breakdown dettagliato per ogni componente.
        Include phi_network (geo-awareness via RTT) e phi_demand (ingress traffic).
        """
        phi_lat = self.compute_latency_score(metrics)
        phi_cap = self.compute_capacity_score(metrics)
        phi_load = self.compute_load_score(metrics, predicted_load)
        phi_carbon = self.compute_carbon_score(metrics)
        phi_net = self.compute_network_score(metrics)
        phi_demand = self.compute_demand_score(metrics)

        total = (
            self.omega_latency * phi_lat +
            self.omega_capacity * phi_cap +
            self.omega_load * phi_load +
            self.omega_carbon * phi_carbon +
            self.omega_network * phi_net +
            self.omega_demand * phi_demand
        )

        return {
            'phi_latency': phi_lat,
            'phi_capacity': phi_cap,
            'phi_load': phi_load,
            'phi_carbon': phi_carbon,
            'phi_network': phi_net,
            'phi_demand': phi_demand,
            'total_score': total,
            'network_rtt_ms': metrics.network_rtt_ms,
            'ingress_rate_rps': metrics.ingress_rate_rps,
            'ingress_demand_share': metrics.ingress_demand_share,
            'weights': {
                'omega_latency': self.omega_latency,
                'omega_capacity': self.omega_capacity,
                'omega_load': self.omega_load,
                'omega_carbon': self.omega_carbon,
                'omega_network': self.omega_network,
                'omega_demand': self.omega_demand,
            }
        }