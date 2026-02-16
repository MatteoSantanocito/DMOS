"""
DMOS Cluster Estimator
Deployed IN each member cluster (like Karmada Resource Estimator)
Computes multi-objective score locally and responds to scheduler queries
"""

from typing import Dict, Optional
from dataclasses import asdict
from flask import Flask, request, jsonify
import threading
import time

# Assicurati che i percorsi di import siano corretti per il tuo progetto
# Se dmos_src è la root sulla VM, questi import relativi vanno bene
from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader
from ..metrics.prometheus_client import PrometheusClient
from ..metrics.carbon_client import CarbonClient
from ..metrics.latency_calculator import LatencyCalculator
from .score_functions import ScoreFunctions, ClusterMetrics

logger = get_logger("ClusterEstimator")


class ClusterEstimator:
    """
    Cluster Estimator (deployed in each cluster)
    """
    CPU_HARD_LIMIT_PCT = 90.0
    MEMORY_HARD_LIMIT_PCT = 90.0
    MIN_CPU_CORES_AVAILABLE = 0.2
    MIN_MEMORY_GB_AVAILABLE = 0.2
    def __init__(self, cluster_name: str, config_path: str = "config"):
        """
        Initialize cluster estimator
        """
        self.cluster_name = cluster_name
        
        # Load configuration
        self.config = ConfigLoader(config_path)
        
        # --- FIX DI EMERGENZA PER IL BUG 'list object has no attribute get' ---
        # Se ConfigLoader ha caricato i servizi come lista, li convertiamo in Dict
        if isinstance(self.config.services, list):
            logger.warning("⚠️ Rilevata lista servizi invece di dizionario. Applicazione patch automatica...")
            services_dict = {}
            for s in self.config.services:
                # Gestiamo sia oggetti ServiceConfig che dizionari raw
                s_name = s.name if hasattr(s, 'name') else s.get('name')
                services_dict[s_name] = s
            self.config.services = services_dict
            logger.info("✅ Patch applicata: servizi convertiti in dizionario.")
        # ---------------------------------------------------------------------

        self.cluster_config = self.config.get_cluster(cluster_name)
        
        if not self.cluster_config:
            raise ValueError(f"Cluster {cluster_name} not found in config")
        
        logger.info(f"Initializing Cluster Estimator for {cluster_name} "
                    f"({self.cluster_config.region} - {self.cluster_config.location})")
        
        # Initialize clients
        self.prometheus = PrometheusClient(
            url=self.config.prometheus.url,
            timeout=5
        )
        
        self.carbon_client = CarbonClient(
            self.config.carbon_raw.get('carbon_intensity', {})
        )
        
        self.latency_calc = LatencyCalculator()
        
        # Initialize score functions
        self.score_func = ScoreFunctions(
            weights={
                'omega_latency': self.config.score_weights.omega_latency,
                'omega_capacity': self.config.score_weights.omega_capacity,
                'omega_load': self.config.score_weights.omega_load,
                'omega_carbon': self.config.score_weights.omega_carbon
            }
        )
        
        # Cached metrics (updated periodically)
        self.cached_metrics: Optional[ClusterMetrics] = None
        self.cache_timestamp = 0
        self.cache_ttl = 30  # seconds
        
        # Start background metrics updater
        self._start_metrics_updater()
    
    def _start_metrics_updater(self):
        """Start background thread to update metrics periodically"""
        def updater():
            while True:
                try:
                    self.cached_metrics = self._collect_metrics()
                    self.cache_timestamp = time.time()
                    logger.debug(f"Metrics updated for {self.cluster_name}")
                except Exception as e:
                    logger.error(f"Error updating metrics: {e}")
                
                time.sleep(self.cache_ttl)
        
        thread = threading.Thread(target=updater, daemon=True)
        thread.start()
        logger.info(f"Started background metrics updater (interval={self.cache_ttl}s)")
    
    def _collect_metrics(self) -> ClusterMetrics:
        """
        Collect current cluster metrics
        """
        # CPU metrics
        cpu_available = self.prometheus.get_cpu_available() or 0.0
        cpu_total = self.cluster_config.cpu_cores
        
        # Memory metrics
        memory_available = self.prometheus.get_memory_available_gb() or 0.0
        memory_total = self.cluster_config.memory_gb
        
        # Traffic metrics (for now, use placeholder)
        request_rate = self.prometheus.get_request_rate(
        service="frontend",
        namespace="online-boutique"
        )
        if request_rate is None:
            # Fallback: stima dal CPU usage se disponibile
            cpu_usage_pct = self.prometheus.get_cpu_usage_percent(
                deployment="frontend",
                namespace="online-boutique"
            )
            if cpu_usage_pct is not None and cpu_usage_pct > 0:
                # Stima: a CPU 100% → capacity_req_per_sec del servizio
                service_config = self.config.get_service("frontend")
                capacity = service_config.capacity_req_per_sec if service_config else 50
                request_rate = (cpu_usage_pct / 100.0) * capacity
                logger.debug(f"Traffic estimated from CPU usage: {cpu_usage_pct:.1f}% "
                            f"→ ~{request_rate:.1f} req/s")
            else:
                # Ultimo fallback: usa 0 (nessun traffico rilevato)
                request_rate = 0.0
                logger.warning("No traffic metrics available, defaulting to 0 req/s")
        
        
        service_config = self.config.get_service("frontend")
        capacity_per_core = service_config.capacity_req_per_sec if service_config else 50
        request_rate_max = cpu_total * capacity_per_core
       
        # Latency metrics
        latency_mean = None
        latency_variance = None
        
        # Prova p95 latency da Prometheus (Istio metrics)
        latency_p95 = self.prometheus.get_latency_p95(
            service="frontend",
            namespace="online-boutique"
        )
        if latency_p95 is not None and latency_p95 > 0:
            # Stima mean e variance dal p95
            # Per distribuzione log-normale tipica: mean ≈ p95 / 1.65
            latency_mean = latency_p95 / 1.65
            # Variance stimata: (p95 - mean)^2
            latency_variance = (latency_p95 - latency_mean) ** 2
            logger.debug(f"Latency from Prometheus: p95={latency_p95:.1f}ms "
                        f"→ mean≈{latency_mean:.1f}ms, var≈{latency_variance:.1f}")
        else:
            # Fallback: baseline dalla configurazione del cluster
            latency_mean = self.cluster_config.baseline_latency_ms
            latency_variance = self.cluster_config.latency.get('variance_ms', 15.0) ** 2
            logger.debug(f"Latency from config baseline: mean={latency_mean:.1f}ms")
        
        
        # Carbon intensity
        carbon_region = self.cluster_config.carbon.get('region_code', 'DE')
        carbon_intensity = self.carbon_client.get_carbon_intensity(carbon_region) or 300.0
        
        # Cost
        cost_per_replica = self.cluster_config.cost_per_replica_hour
        
        # ── Compute CPU/Memory utilization percentages for hard constraints ──
        cpu_utilization_pct = 0.0
        if cpu_total > 0:
            cpu_used = cpu_total - cpu_available
            cpu_utilization_pct = (cpu_used / cpu_total) * 100.0
        
        mem_utilization_pct = 0.0
        if memory_total > 0:
            mem_used = memory_total - memory_available
            mem_utilization_pct = (mem_used / memory_total) * 100.0
        
        metrics = ClusterMetrics(
            cpu_available_cores=cpu_available,
            cpu_total_cores=cpu_total,
            memory_available_gb=memory_available,
            memory_total_gb=memory_total,
            request_rate_current=request_rate,
            request_rate_max=request_rate_max,
            latency_mean_ms=latency_mean,
            latency_variance_ms2=latency_variance,
            carbon_intensity_gco2_kwh=carbon_intensity,
            cost_per_replica_hour=cost_per_replica
        )
        
        logger.debug(f"Collected metrics: "
                    f"cpu={cpu_available:.1f}/{cpu_total} ({cpu_utilization_pct:.0f}%), "
                    f"mem={memory_available:.1f}/{memory_total}GB ({mem_utilization_pct:.0f}%), "
                    f"traffic={request_rate:.1f} req/s, "
                    f"latency={latency_mean:.1f}ms, "
                    f"CI={carbon_intensity:.0f}gCO2/kWh")
        
        return metrics
    
    def get_metrics(self, use_cache: bool = True) -> ClusterMetrics:
        if use_cache and self.cached_metrics:
            age = time.time() - self.cache_timestamp
            if age < self.cache_ttl:
                return self.cached_metrics
        return self._collect_metrics()
    
    def compute_score(
    self, 
    service_name=None,
    predicted_load=None
    ):
        """
        Compute multi-objective score for this cluster.
        
        Now includes HARD CONSTRAINTS: if CPU or memory utilization exceeds
        the threshold, the cluster is marked as ineligible (capacity=0, score=0).
        This prevents scheduling to saturated clusters regardless of their
        scores on other dimensions.
        
        Args:
            service_name: Service to score for (optional)
            predicted_load: Optional predicted load for scoring
        
        Returns:
            Dict with score breakdown, capacity, and eligibility status
        """
        metrics = self.get_metrics(use_cache=True)
        
        # ── Hard Constraint Check ─────────────────────────────────────────
        # Compute utilization percentages
        cpu_util_pct = 0.0
        if metrics.cpu_total_cores > 0:
            cpu_util_pct = ((metrics.cpu_total_cores - metrics.cpu_available_cores) 
                        / metrics.cpu_total_cores) * 100.0
        
        mem_util_pct = 0.0
        if metrics.memory_total_gb > 0:
            mem_util_pct = ((metrics.memory_total_gb - metrics.memory_available_gb) 
                        / metrics.memory_total_gb) * 100.0
        
        # Check hard limits
        is_eligible = True
        exclusion_reason = None
        
        if cpu_util_pct > self.CPU_HARD_LIMIT_PCT:
            is_eligible = False
            exclusion_reason = f"CPU saturated ({cpu_util_pct:.0f}% > {self.CPU_HARD_LIMIT_PCT}%)"
            
        elif mem_util_pct > self.MEMORY_HARD_LIMIT_PCT:
            is_eligible = False
            exclusion_reason = f"Memory saturated ({mem_util_pct:.0f}% > {self.MEMORY_HARD_LIMIT_PCT}%)"
            
        elif metrics.cpu_available_cores < self.MIN_CPU_CORES_AVAILABLE:
            is_eligible = False
            exclusion_reason = f"Insufficient CPU ({metrics.cpu_available_cores:.2f} < {self.MIN_CPU_CORES_AVAILABLE} cores)"
            
        elif metrics.memory_available_gb < self.MIN_MEMORY_GB_AVAILABLE:
            is_eligible = False
            exclusion_reason = f"Insufficient memory ({metrics.memory_available_gb:.2f} < {self.MIN_MEMORY_GB_AVAILABLE} GB)"
        
        if not is_eligible:
            logger.warning(f"⛔ {self.cluster_name} EXCLUDED: {exclusion_reason}")
            return {
                'cluster_name': self.cluster_name,
                'score': 0.0,
                'score_breakdown': {
                    'phi_latency': 0.0,
                    'phi_capacity': 0.0,
                    'phi_load': 0.0,
                    'phi_carbon': 0.0
                },
                'capacity': 0,
                'eligible': False,
                'exclusion_reason': exclusion_reason,
                'metrics': {
                    'cpu_available_cores': metrics.cpu_available_cores,
                    'cpu_total_cores': metrics.cpu_total_cores,
                    'cpu_utilization_pct': cpu_util_pct,
                    'memory_available_gb': metrics.memory_available_gb,
                    'memory_total_gb': metrics.memory_total_gb,
                    'memory_utilization_pct': mem_util_pct,
                    'carbon_intensity_gco2_kwh': metrics.carbon_intensity_gco2_kwh,
                    'latency_mean_ms': metrics.latency_mean_ms
                }
            }
        
        # ── Normal scoring (cluster is eligible) ──────────────────────────
        breakdown = self.score_func.compute_score_breakdown(
            metrics,
            predicted_load=predicted_load
        )
        
        # Calculate available capacity (in replicas)
        service_config = self.config.get_service(service_name) if service_name else None
        
        if service_config:
            # Parse CPU request (e.g., "100m" -> 0.1 cores)
            cpu_req_str = service_config.cpu_request
            if cpu_req_str.endswith('m'):
                cpu_req = float(cpu_req_str[:-1]) / 1000.0
            else:
                cpu_req = float(cpu_req_str)
            
            # Parse memory request (e.g., "64Mi" -> 0.0625 GB)
            mem_req_str = service_config.memory_request
            if mem_req_str.endswith('Mi'):
                mem_req_gb = float(mem_req_str[:-2]) / 1024.0
            elif mem_req_str.endswith('Gi'):
                mem_req_gb = float(mem_req_str[:-2])
            else:
                mem_req_gb = float(mem_req_str)
            
            capacity_cpu = int(metrics.cpu_available_cores / cpu_req) if cpu_req > 0 else 0
            capacity_mem = int(metrics.memory_available_gb / mem_req_gb) if mem_req_gb > 0 else 0
            capacity = min(capacity_cpu, capacity_mem)
        else:
            capacity = int(metrics.cpu_available_cores * 2)
        
        # Apply max replicas constraint
        max_replicas = service_config.max_replicas if service_config else 20
        capacity = min(capacity, max_replicas)
        
        result = {
            'cluster_name': self.cluster_name,
            'score': breakdown['total_score'],
            'score_breakdown': {
                'phi_latency': breakdown['phi_latency'],
                'phi_capacity': breakdown['phi_capacity'],
                'phi_load': breakdown['phi_load'],
                'phi_carbon': breakdown['phi_carbon']
            },
            'capacity': max(0, capacity),
            'eligible': True,
            'exclusion_reason': None,
            'metrics': {
                'cpu_available_cores': metrics.cpu_available_cores,
                'cpu_total_cores': metrics.cpu_total_cores,
                'cpu_utilization_pct': cpu_util_pct,
                'memory_available_gb': metrics.memory_available_gb,
                'memory_total_gb': metrics.memory_total_gb,
                'memory_utilization_pct': mem_util_pct,
                'carbon_intensity_gco2_kwh': metrics.carbon_intensity_gco2_kwh,
                'latency_mean_ms': metrics.latency_mean_ms
            }
        }
        
        logger.info(f"Score computed: {breakdown['total_score']:.3f}, "
                f"capacity={capacity}, "
                f"cpu={cpu_util_pct:.0f}%, mem={mem_util_pct:.0f}%")
        
        return result

def create_estimator_api(cluster_name: str, config_path: str = "config") -> Flask:
    app = Flask(f"cluster-estimator-{cluster_name}")
    estimator = ClusterEstimator(cluster_name, config_path)
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'cluster': cluster_name}), 200
    
    @app.route('/estimate', methods=['POST'])
    def estimate():
        data = request.json or {}
        service_name = data.get('service_name')
        predicted_load = data.get('predicted_load')
        
        try:
            result = estimator.compute_score(
                service_name=service_name,
                predicted_load=predicted_load
            )
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error computing score: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        try:
            metrics = estimator.get_metrics(use_cache=False)
            return jsonify(asdict(metrics)), 200
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app


def run_estimator(cluster_name: str, port: int = 8080, config_path: str = "config"):
    app = create_estimator_api(cluster_name, config_path)
    logger.info(f"Starting Cluster Estimator for {cluster_name} on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.level1.cluster_estimator <cluster_name> [port]")
        sys.exit(1)
    
    cluster_name = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    run_estimator(cluster_name, port)