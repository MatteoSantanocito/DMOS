"""
Multi-dimensional score functions for cluster selection (DMOS Level 1)
======================================================================

Implements the multi-objective cluster scoring used by Level 1 winner
determination. Each score component Φ_i ∈ [0,1] captures a distinct
objective; the total score is a weighted sum:

    S_i = ω_lat·Φ_lat  +  ω_cap·Φ_cap  +  ω_geo·(Φ_demand·Φ_net)
        + ω_carbon·Φ_carbon  +  ω_pressure·Φ_pressure

This formula differs from traditional load-balancer scoring in two key
ways, motivated by the application domain (REPLICA placement, not
REQUEST routing):

  1. Φ_pressure (formerly Φ_load) uses an INVERTED-sigmoid semantic:
     the cluster's score INCREASES with utilization above the target
     (τ ≈ 0.7), so that saturated clusters become PRIMARY allocation
     targets rather than being avoided. This mirrors the Kubernetes
     Horizontal Pod Autoscaler target-utilization model and is also
     used by CarbonScaler (Hanafy et al., HotCarbon'23) and CASPER
     (Souza et al., E-Energy'23).

  2. Φ_geo_load now expresses LOCALITY (Φ_demand × Φ_net): clusters
     where both demand and proximity are high. Previously it expressed
     "unloaded × close" which conflicts with replica placement under
     concentrated load (the saturated cluster has Φ_load ≈ 0 → penalty).

DESIGN-RATIONALE NOTES — references for the paper
─────────────────────────────────────────────────
  • Load-balancer style scoring (Φ_load HIGH when idle):
      Ousterhout et al. "Sparrow: Distributed, Low Latency Scheduling"
      (SOSP 2013); Kubernetes LeastRequestedPriority. ← CONTRAST citation:
      these route REQUESTS away from busy nodes. Inappropriate for our
      problem of placing REPLICAS where they are needed.

  • Replica-placement style scoring (Φ_pressure HIGH when busy):
      Kubernetes HPA target-utilization model
        (kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/);
      Hanafy et al. "CarbonScaler" (HotCarbon 2023);
      Souza et al. "CASPER" (E-Energy 2023). ← PRIMARY references for
      our inverted-sigmoid choice.

  • Hardware-aware capacity scoring (Φ_cap as HW size, not slack):
      Verma et al. "Borg" (EuroSys 2015) — "fit" combines resource size
      with task profile;
      Grandl et al. "Tetris: Multi-resource packing for cluster
      schedulers" (NSDI 2014) — dot-product alignment of demand and
      hardware.

  • Multi-objective additive sums (vs. multiplicative AND):
      Borg, Apollo (Boutin et al., OSDI 2014), most production schedulers.
      The single multiplicative term in our formula is reserved for
      LOCALITY (demand × net), where the AND semantic is conceptually
      meaningful: replicas should be near users AND in clusters that
      receive their traffic.

  • Adler & Wolf, "Effective Resource Management for Geographically
      Distributed Cloud Services" (2018) — the "demand-topology
      alignment" concept that underpins our Φ_locality term.
"""

import math
import os
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
    
    # Load prediction component (sigmoid saturation threshold)
    # Φ_load = 1 / (1 + exp(k × (λ/λ_max - τ)))
    # Sotto τ: Φ_load ≈ 1 (non premia cluster scarichi).
    # Sopra τ: Φ_load crolla (penalizza solo saturazione reale).
    load_threshold: float = 0.7   # τ: soglia di saturazione (70% del massimo)
    load_steepness: float = 10.0  # k: pendenza della sigmoid
    load_floor: float = 0.2       # floor: Φ_load minimo (impedisce azzeramento Φ_geo)
    horizon_seconds: int = 600    # orizzonte di predizione per il carico (10 minuti)
    
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
    score_i = ω_lat·Φ_lat(i) + ω_cap·Φ_cap(i) + ω_geo·(Φ_load(i)·Φ_net(i))
            + ω_carbon·Φ_carbon(i) + ω_demand·Φ_demand(i)

    Il termine moltiplicativo Φ_geo_load = Φ_load × Φ_net garantisce che un cluster
    sia favorito SOLO se è sia scarico (Φ_load alto) sia geograficamente vicino
    (Φ_net alto). Se uno dei due è basso, il prodotto crolla → penalizzazione.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        parameters: Optional[ScoreParameters] = None,
        model=None,
    ):
        """
        Two configuration paths:

        * model (a src.level1.score_model.ScoreModel) — the demand role and
          every weight come from a named score model (DMOS_SCORE_MODEL);
          `weights` is ignored and so are the legacy env switches.
        * no model — LEGACY path, unchanged: weights from the YAML profile,
          modified by DMOS_WEIGHTS_COHERENT / DMOS_LOCALITY_PURE_NET /
          DMOS_SCORE_GATE*.
        """
        self.model = model
        # Weight of the ADDITIVE demand term ω_dem·d (additive model only).
        self.omega_demand_share = 0.0
        if model is not None:
            self._init_from_model(model, parameters)
            return
        self.demand_mode = "legacy"

        self.omega_latency = weights.get('omega_latency', 0.25)
        self.omega_capacity = weights.get('omega_capacity', 0.20)
        self.omega_geo_load = weights.get('omega_geo_load', 0.25)
        self.omega_carbon = weights.get('omega_carbon', 0.20)
        self.omega_demand = weights.get('omega_demand', 0.10)

        # ── Design COERENTE (toggle) — la domanda compare SOLO nel gate ────────
        # Rende vero il claim "pesi riconfigurabili (latency-only, carbon-only)":
        # oggi la domanda entra 3 volte (gate G, località Φ_dem·Φ_net, e Φ_pre
        # che è util=carico/capacità → demand-derivato), quindi i pesi NON sono
        # liberi. I due flag sotto isolano la domanda nel solo gate:
        #   DMOS_LOCALITY_PURE_NET=1 → Φ_loc = Φ_net (prossimità pura): si toglie
        #     Φ_dem dal prodotto località → la domanda non è più contata qui.
        #   DMOS_WEIGHTS_COHERENT="lat,cap,net,car" → profilo a 4 obiettivi
        #     demand-free; omega_demand (peso di Φ_pre) è azzerato → Φ_pre esce
        #     dalla somma. Il ruolo anti-idle di Φ_pre è sussunto dal gate (che
        #     sopprime i cluster senza domanda); il dimensionamento "serve aiuto"
        #     resta nel budget N + nel PD di L2.
        self.locality_pure_net = os.environ.get("DMOS_LOCALITY_PURE_NET", "0") == "1"
        w_coh = os.environ.get("DMOS_WEIGHTS_COHERENT", "").strip()
        # Where the weights come from (recorded in the run provenance).
        self.weights_source = "env:DMOS_WEIGHTS_COHERENT" if w_coh else "profile"
        if w_coh:
            parts = [float(x) for x in w_coh.split(",")]
            if len(parts) != 4:
                raise ValueError(
                    "DMOS_WEIGHTS_COHERENT richiede 4 valori 'lat,cap,net,car', "
                    f"ricevuti {len(parts)}: {w_coh!r}"
                )
            (self.omega_latency, self.omega_capacity,
             self.omega_geo_load, self.omega_carbon) = parts
            self.omega_demand = 0.0  # Φ_pre fuori dalla somma (design coerente)

        # Valida che le pesature sommano a 1.0
        total = (self.omega_latency + self.omega_capacity +
                 self.omega_geo_load + self.omega_carbon +
                 self.omega_demand)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"La somma delle pesature deve essere 1.0, ho {total}")

        self.params = parameters or ScoreParameters()

        # ── Demand-GATE moltiplicativo (toggle DMOS_SCORE_GATE=1) ──────────────
        # Alternativa al demand-floor: invece di garantire ⌈domanda/cap⌉ con un
        # vincolo duro, MOLTIPLICA lo score per la quota di domanda → i cluster
        # senza domanda (Φ_dem≈0) collassano a S≈0 (niente repliche), e lo score
        # multi-obiettivo modula SOLO tra i cluster con domanda. Così la
        # concentrazione demand-aware emerge dallo score (i pesi tornano
        # decisionali) anziché da un floor esterno.
        #   S_i = (eps + (1-eps)·Φ_dem_i)^gamma · Σ_k ω_k Φ_k
        # gamma→∞ ≈ floor (allocazione pura per domanda); gamma=1 = gate lineare.
        self.gate_enabled = os.environ.get("DMOS_SCORE_GATE", "0") == "1"
        self.gate_eps = float(os.environ.get("DMOS_SCORE_GATE_EPS", "0.05"))
        self.gate_gamma = float(os.environ.get("DMOS_SCORE_GATE_GAMMA", "1.0"))

        logger.info(
            f"Score weights: lat={self.omega_latency}, cap={self.omega_capacity}, "
            f"geo_load={self.omega_geo_load}, carbon={self.omega_carbon}, "
            f"demand={self.omega_demand}"
            + (f" | DEMAND-GATE ON (eps={self.gate_eps}, gamma={self.gate_gamma})"
               if self.gate_enabled else "")
            + (" | COERENTE: Φ_loc=Φ_net puro, Φ_pre OFF (domanda solo nel gate)"
               if self.locality_pure_net else "")
        )

    def _init_from_model(self, model, parameters: Optional[ScoreParameters]) -> None:
        """
        Configure from a ScoreModel. The four objectives map to the existing
        slots; demand enters according to model.demand_mode:
          gate      gate on (ε, g), net slot = Φ_net, no additive demand term
          additive  gate off, net slot = Φ_net, + ω_dem·d
          locality  gate off, net slot = d·Φ_net
        Φ_pressure is not part of any model (weight 0).
        """
        w = model.weights
        self.omega_latency = w["latency"]
        self.omega_capacity = w["capacity"]
        self.omega_geo_load = w["net"]          # Φ_net, or d·Φ_net in locality mode
        self.omega_carbon = w["carbon"]
        self.omega_demand = 0.0                 # Φ_pressure: not used by models
        self.omega_demand_share = w["demand"]   # > 0 only in additive mode
        self.demand_mode = model.demand_mode
        self.locality_pure_net = model.demand_mode != "locality"
        self.gate_enabled = model.demand_mode == "gate"
        self.gate_eps = model.gate_eps
        self.gate_gamma = model.gate_gamma
        self.weights_source = (f"model:{model.name}"
                               + (" (cold-start variant)" if model.phase == "cold" else ""))
        self.params = parameters or ScoreParameters()
        logger.info(
            f"Score model '{model.name}' [{model.phase}] demand_mode={model.demand_mode}: "
            f"lat={self.omega_latency:g}, cap={self.omega_capacity:g}, "
            f"net={self.omega_geo_load:g}, car={self.omega_carbon:g}"
            + (f", dem={self.omega_demand_share:g}" if self.omega_demand_share else "")
            + (f" | gate eps={self.gate_eps:g} g={self.gate_gamma:g}" if self.gate_enabled else "")
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
        Φ_cap(i) — capacity score per replica placement.

        SEMANTIC (post-refactor, paper §X):
            HIGH for clusters that CAN HOST more replicas, decomposed as:
              (a) hardware tier — physical capacity (CPU cores)
              (b) slack — resource headroom remaining

        FORMULA:
            Φ_cap = hw_size · (β + (1-β)·slack)

            where:
              hw_size = min(1.0, cpu_total / REFERENCE_CORES)
              slack   = min(cpu_available_fraction, memory_available_fraction)
              β       = 0.3   (minimum contribution from HW size alone, even
                               when slack=0 — ensures a saturated tier-HIGH
                               cluster is still preferable to a tier-LOW one)

        DESIGN-RATIONALE — why this differs from the original form
        ───────────────────────────────────────────────────────────
        Original:  Φ_cap = (resource_fraction)^κ × (1 - load_fraction)

        Both terms collapsed to ≈ 0 under saturation, leaving the saturated
        cluster (which is precisely where we need more replicas) with the
        WORST capacity score. This is the canonical anti-pattern of using
        load-balancer scoring for replica placement (see module-level
        docstring for refs).

        Our reformulation [Borg EuroSys'15 "fit" concept; Tetris NSDI'14
        dot-product alignment] separates the two signals:
          • hw_size captures the cluster's PHYSICAL ability to host pods;
            it does not depend on current load.
          • slack captures whether the K8s scheduler has resources free
            for new pods; β=0.3 ensures slack cannot zero out the score.

        EXAMPLES with REFERENCE_CORES=8 (tier-HIGH), β=0.3:
          tier-HIGH (8c) idle:        1.00 · (0.3 + 0.7·1.00) = 1.00
          tier-HIGH (8c) saturated:   1.00 · (0.3 + 0.7·0.05) = 0.335
          tier-MED (6c) idle:         0.75 · (0.3 + 0.7·1.00) = 0.75
          tier-LOW (4c) idle:         0.50 · (0.3 + 0.7·1.00) = 0.50
          tier-LOW (4c) saturated:    0.50 · (0.3 + 0.7·0.05) = 0.168

        Note: a SATURATED tier-HIGH (0.335) beats an IDLE tier-LOW (0.50)?
        No — but the gap is reduced. This is intentional: Φ_pressure
        captures the "needs help" signal independently, so Φ_cap can focus
        purely on HW-fitness for replica hosting.
        """
        # Reference HW size: tier-HIGH = 8 cores in our 3-tier testbed
        # (see config/local.yaml). Cluster with fewer cores → fractional score.
        REFERENCE_CORES = 8.0
        hw_size = min(1.0, metrics.cpu_total_cores / REFERENCE_CORES)

        # Slack: how much HW headroom remains for new pods.
        # Use the bottleneck resource (min of CPU/memory available fractions).
        slack = min(
            metrics.cpu_available_fraction,
            metrics.memory_available_fraction
        )

        # Combine: hw_size always contributes (floor β=0.3), slack modulates
        # the remaining 70% — this guarantees that a saturated tier-HIGH
        # cluster is not zeroed-out (was the bug in the original formula).
        beta = 0.3
        score = hw_size * (beta + (1.0 - beta) * slack)

        logger.debug(
            f"Φ_cap: cpu_total={metrics.cpu_total_cores}, "
            f"hw_size={hw_size:.3f}, slack={slack:.3f}, score={score:.3f}"
        )

        return max(0.0, min(1.0, score))
    
    def compute_load_score(
        self,
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> float:
        """
        Φ_pressure(i) — replica-placement pressure score.

        SEMANTIC (post-refactor, paper §X):
            HIGH for clusters that NEED more replicas.
            Aligned with the K8s HPA target-utilization model:
            once utilization crosses the target (τ ≈ 0.7), the cluster
            is signaling "I need help" and should receive priority in
            replica allocation.

        FORMULA — INVERTED sigmoid:
            Φ_pressure(util) = 1 / (1 + exp(-k · (util - τ)))

            where:
              util = current_load / capacity
              τ    = 0.7   (HPA-style target utilization)
              k    = 10    (steepness; sharp transition around τ)

        Soft saturation cap at util > 1.5:
            For grossly over-saturated clusters (util > 1.5), the score
            is multiplicatively dampened. This avoids "starvation" of
            other clusters: an infinite-utilization cluster cannot keep
            absorbing 100% of the replica budget.

        DESIGN-RATIONALE — paper references for the inversion
        ──────────────────────────────────────────────────────
        The ORIGINAL formula was:
            Φ_load = 1 / (1 + exp(+k·(util - τ)))   ← DECREASING in util
        This is the canonical load-balancer score (Sparrow, K8s default),
        designed for ROUTING REQUESTS away from busy nodes.

        For our problem (REPLICA placement), we INVERT the sign of the
        exponent argument, making the score INCREASING in util — the
        cluster gains priority as it approaches/exceeds its target.
        This mirrors:
          • K8s HPA: when util > target, the controller requests more
            replicas (essentially Φ_pressure ≈ 1.0).
          • CarbonScaler (Hanafy et al., HotCarbon'23): treats "high
            utilization" as a positive signal for the elasticity policy.
          • CASPER (Souza et al., E-Energy'23): combines latency and
            carbon objectives where saturation triggers reallocation.

        The PREVIOUS implementation produced the catastrophic failure
        documented in paper §X.X: at high load (>250 rps spike), the
        most-demanded cluster (cluster-01, 28% of traffic) saw its
        Φ_load → floor (0.2) while idle clusters had Φ_load ≈ 0.9. The
        weighted sum gave the IDLE clusters higher total scores, so
        L1 allocated replicas AWAY from the saturated cluster. Errors
        on cluster-01 spiked to 23%.

        EXAMPLES with τ=0.7, k=10:
          util = 0.0 (idle):                Φ_pressure = 1/(1+exp(7))  ≈ 0.001
          util = 0.5 (half-loaded):         Φ_pressure = 1/(1+exp(2))  ≈ 0.119
          util = 0.7 (at target):           Φ_pressure = 1/(1+exp(0))  = 0.500
          util = 1.0 (capacity reached):    Φ_pressure = 1/(1+exp(-3)) ≈ 0.953
          util = 1.5 (severely loaded):     Φ_pressure ≈ 1.000
          util = 2.0 (catastrophic):        Φ_pressure ≈ 0.500 (damped)
        """
        # IMPORTANT — Per-cluster pressure (paper §X bugfix):
        # Always use the cluster's LOCAL load (metrics.request_rate_current).
        # The legacy `predicted_load` parameter was set GLOBALLY by callers
        # (dmos_main passes current_traffic = sum across clusters), which
        # made Φ_pressure identical for all clusters when not None — the
        # very bug that was crashing replica placement (see paper §X.X).
        #
        # The parameter is kept in the signature for backward-compat but
        # only honored when EXPLICITLY interpreted as per-cluster (e.g.,
        # if the scheduler ever wires a per-cluster L2 predictor output).
        # In the current code path, predicted_load is ignored.
        load = metrics.request_rate_current

        if metrics.request_rate_max == 0:
            logger.warning("Φ_pressure: request_rate_max=0 → fallback 0.5 (neutral)")
            return 0.5  # neutral fallback (was 0.0; that biased against unknown clusters)

        util = load / metrics.request_rate_max
        k = self.params.load_steepness     # 10 (steep transition around τ)
        tau = self.params.load_threshold    # 0.7 (HPA-style target)

        # Inverted sigmoid: score INCREASES with utilization.
        # Note the sign flip: -k·(util-τ) instead of +k·(util-τ).
        raw_score = 1.0 / (1.0 + math.exp(-k * (util - tau)))

        # Soft saturation cap: above util=1.5 (50% over capacity), start
        # dampening to prevent starvation of other clusters. Without this,
        # a perpetually-saturated cluster would absorb 100% of the budget.
        OVERSAT_THRESHOLD = 1.5
        if util > OVERSAT_THRESHOLD:
            damping = max(0.4, 1.0 - 0.6 * (util - OVERSAT_THRESHOLD))
            raw_score *= damping

        # Bounded to [0, 1]
        score = max(0.0, min(1.0, raw_score))

        logger.debug(
            f"Φ_pressure: load={load:.1f}, max={metrics.request_rate_max:.1f}, "
            f"util={util:.3f}, τ={tau}, k={k}, score={score:.3f} "
            f"(inverted sigmoid: HIGH when util > τ)"
        )

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
          → differenziazione nulla, Φ_net≈1 per tutti → Φ_geo_load≈Φ_load.
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

    def compute_geo_load_score(
        self,
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> float:
        """
        Φ_locality(i) = Φ_demand(i) × Φ_net(i)

        DESIGN-RATIONALE — semantic rewrite (paper §X)
        ───────────────────────────────────────────────
        Previously this method computed Φ_geo_load = Φ_load × Φ_net,
        i.e. "unloaded AND close". For replica scheduling this was
        counterproductive (saturated cluster → Φ_load floor → product
        collapses → cluster avoided, even though it needs replicas).

        We now compute Φ_locality = Φ_demand × Φ_net, i.e.
        "users-are-here AND topologically-close". This is the
        multiplicative "AND" version of the demand-topology alignment
        concept from Adler & Wolf (2018) and aligns with the locality
        primitive used in geo-distributed serving research.

        The method name remains `compute_geo_load_score` for
        backward-compat with the caller (dmos_scheduler.py:_compute_cluster_score),
        but the SEMANTIC has been intentionally rewritten.

        Properties of the multiplicative form:
          • Cluster with demand AND close → product high → strong
            preference for placing replicas there.
          • Cluster with demand but far → Φ_net suppresses → moderate
            preference (still has demand, so not zero).
          • Cluster close but with no demand → Φ_demand low → product
            low (we don't put replicas where there's no traffic).
          • Cluster with neither → near-zero contribution.

        Note: Φ_pressure (formerly Φ_load) is now a STANDALONE term in
        the score sum, not part of this multiplication. The two signals
        are conceptually distinct:
          - Φ_pressure: "this cluster is/will-be saturated, send help"
          - Φ_locality: "this cluster is the natural home of the demand"
        Adding to the score additively (instead of multiplying with Φ_net)
        lets each express its information without one zeroing out the other.
        """
        phi_demand = self.compute_demand_score(metrics)
        phi_net = self.compute_network_score(metrics)
        locality = phi_demand * phi_net

        logger.debug(
            f"Φ_locality: Φ_demand={phi_demand:.3f} × Φ_net={phi_net:.3f} "
            f"= {locality:.3f} (RTT={metrics.network_rtt_ms:.0f}ms, "
            f"demand_share={metrics.ingress_demand_share:.3f})"
        )
        return locality

    def _apply_demand_gate(self, total: float, phi_demand: float) -> float:
        """
        Gate moltiplicativo demand-aware: S_i ← (ε + (1-ε)·Φ_dem)^γ · S_i.

        Quando abilitato (DMOS_SCORE_GATE=1) la domanda diventa un FATTORE che
        annulla lo score dei cluster senza traffico, recuperando la
        concentrazione del demand-floor come proprietà emergente dello score
        (i pesi tornano decisionali sul residuo). γ→∞ ≈ floor puro.
        Applicato in ENTRAMBI i path (compute_total_score e
        compute_score_breakdown) perché lo scheduler usa il secondo.
        """
        if not self.gate_enabled:
            return total
        gate = (self.gate_eps + (1.0 - self.gate_eps) * phi_demand) ** self.gate_gamma
        return gate * total

    def compute_total_score(
        self,
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> float:
        """
        score_i = ω_lat·Φ_lat                       (latency penalty)
                + ω_cap·Φ_cap                        (HW size + slack)
                + ω_geo·(Φ_demand·Φ_net)             (LOCALITY: demand AND close)
                + ω_carbon·Φ_carbon                  (grid intensity)
                + ω_demand·Φ_pressure                (saturation = NEEDS replicas)

        NOTE — SEMANTIC RENAMING (post-refactor):
          • The weight `omega_geo_load` now scales the LOCALITY term
            (Φ_demand × Φ_net), not the legacy `Φ_load × Φ_net`.
          • The weight `omega_demand` now scales Φ_PRESSURE (the
            inverted-sigmoid utilization signal), not the standalone
            Φ_demand. The standalone demand signal is captured INSIDE
            the locality multiplicative term.
          • Field names in `weights.yaml` are KEPT unchanged for
            backward compatibility, but the documentation above each
            profile explains the post-refactor semantics.

        This way, all 5 score components carry independent information
        and there is no double-counting of Φ_net (which previously
        appeared in both `Φ_geo_load` and could be re-introduced via
        a separate `Φ_locality` term — see paper §X for the discussion
        on this design choice).
        """
        phi_lat = self.compute_latency_score(metrics)
        phi_cap = self.compute_capacity_score(metrics)
        phi_pressure = self.compute_load_score(metrics, predicted_load)
        phi_net = self.compute_network_score(metrics)
        phi_demand = self.compute_demand_score(metrics)
        phi_carbon = self.compute_carbon_score(metrics)

        # Locality: demand × proximity (legacy) oppure prossimità PURA (coerente).
        # Con DMOS_LOCALITY_PURE_NET=1 la domanda esce dalla località (resta solo
        # Φ_net) → la domanda è contata solo nel gate (design coerente).
        phi_locality = phi_net if self.locality_pure_net else (phi_demand * phi_net)

        total_score = (
            self.omega_latency * phi_lat +
            self.omega_capacity * phi_cap +
            self.omega_geo_load * phi_locality +
            self.omega_carbon * phi_carbon +
            self.omega_demand * phi_pressure +
            self.omega_demand_share * phi_demand    # additive model only (else 0)
        )

        # Demand-GATE moltiplicativo (se abilitato): la domanda diventa un
        # fattore che cancella i cluster senza traffico, recuperando la
        # concentrazione del floor come proprietà emergente dello score.
        total_score = self._apply_demand_gate(total_score, phi_demand)

        # Logging: explicit naming for diagnostics. Note the renaming:
        # 'geo' now shows Φ_demand × Φ_net; the 'demand' weight slot
        # now drives Φ_pressure (legacy compatibility with weights.yaml).
        logger.info(
            f"Score totale: {total_score:.3f} = "
            f"{self.omega_latency}×{phi_lat:.3f}(lat) + "
            f"{self.omega_capacity}×{phi_cap:.3f}(cap) + "
            f"{self.omega_geo_load}×{phi_locality:.3f}(locality[Φ_dem={phi_demand:.3f}*Φ_net={phi_net:.3f}]) + "
            f"{self.omega_carbon}×{phi_carbon:.3f}(carbon) + "
            f"{self.omega_demand}×{phi_pressure:.3f}(pressure[util-based])"
        )

        return total_score

    def compute_score_breakdown(
        self,
        metrics: ClusterMetrics,
        predicted_load: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Compute total score with detailed per-component breakdown
        (used by paper analytics scripts to plot score composition).

        SEMANTIC NOTE (post-refactor):
          • `phi_geo_load` now stores Φ_LOCALITY (Φ_demand × Φ_net).
            The legacy key name is kept for backward-compat with existing
            paper plots; the value reflects the new semantic.
          • `phi_pressure` is a NEW key: the inverted-sigmoid load score
            ("cluster needs replicas"). `phi_load` is kept as an alias
            for the same value, again for backward-compat.
        """
        phi_lat = self.compute_latency_score(metrics)
        phi_cap = self.compute_capacity_score(metrics)
        phi_pressure = self.compute_load_score(metrics, predicted_load)
        phi_net = self.compute_network_score(metrics)
        phi_demand = self.compute_demand_score(metrics)
        phi_carbon = self.compute_carbon_score(metrics)
        phi_locality = phi_net if self.locality_pure_net else (phi_demand * phi_net)  # paper § Φ_locality (coerente: Φ_net puro)

        total = (
            self.omega_latency * phi_lat +
            self.omega_capacity * phi_cap +
            self.omega_geo_load * phi_locality +
            self.omega_carbon * phi_carbon +
            self.omega_demand * phi_pressure +
            self.omega_demand_share * phi_demand    # additive model only (else 0)
        )
        # Stesso gate del placement path (compute_total_score). Lo scheduler usa
        # QUESTO metodo (dmos_scheduler.py:395) → senza questa riga il gate
        # non avrebbe alcun effetto sull'allocazione.
        total = self._apply_demand_gate(total, phi_demand)

        return {
            # ── Primary components (after semantic refactor) ───────────
            'phi_latency':  phi_lat,
            'phi_capacity': phi_cap,
            'phi_locality': phi_locality,        # = Φ_demand × Φ_net
            'phi_pressure': phi_pressure,         # inverted-sigmoid util score
            'phi_carbon':   phi_carbon,
            # ── Backward-compat aliases (deprecated, kept for old plots) ──
            'phi_geo_load': phi_locality,         # alias of phi_locality
            'phi_load':     phi_pressure,         # alias of phi_pressure
            # ── Individual building blocks (for debug/plotting) ────────
            'phi_network':           phi_net,
            'phi_demand':            phi_demand,
            'network_rtt_ms':        metrics.network_rtt_ms,
            'ingress_rate_rps':      metrics.ingress_rate_rps,
            'ingress_demand_share':  metrics.ingress_demand_share,
            'total_score':           total,
            # ── Weights snapshot ───────────────────────────────────────
            'weights': {
                'omega_latency':  self.omega_latency,
                'omega_capacity': self.omega_capacity,
                'omega_geo_load': self.omega_geo_load,    # → weighs phi_locality
                'omega_carbon':   self.omega_carbon,
                'omega_demand':   self.omega_demand,      # → weighs phi_pressure
                'omega_demand_share': self.omega_demand_share,  # additive d term
            },
            'demand_mode': self.demand_mode,
        }