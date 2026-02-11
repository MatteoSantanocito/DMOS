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
        request_rate = 100.0  # Placeholder
        request_rate_max = cpu_total * 100.0  # Estimate: 100 req/s per core
        
        # Latency metrics
        latency_mean = self.cluster_config.baseline_latency_ms
        latency_variance = self.cluster_config.latency.get('variance_ms', 15.0) ** 2
        
        # Carbon intensity
        carbon_region = self.cluster_config.carbon.get('region_code', 'DE')
        carbon_intensity = self.carbon_client.get_carbon_intensity(carbon_region) or 300.0
        
        # Cost
        cost_per_replica = self.cluster_config.cost_per_replica_hour
        
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
        
        logger.debug(f"Collected metrics: cpu={cpu_available:.1f}/{cpu_total}, "
                     f"mem={memory_available:.1f}/{memory_total}GB, "
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
        service_name: Optional[str] = None,
        predicted_load: Optional[float] = None
    ) -> Dict:
        metrics = self.get_metrics(use_cache=True)
        
        # Compute score breakdown
        breakdown = self.score_func.compute_score_breakdown(
            metrics,
            predicted_load=predicted_load
        )
        
        # Calculate available capacity
        service_config = self.config.get_service(service_name) if service_name else None
        
        if service_config:
            # Parse CPU request
            cpu_req_str = str(service_config.resources_per_replica.get('requests', {}).get('cpu', '100m'))
            if cpu_req_str.endswith('m'):
                cpu_req = float(cpu_req_str[:-1]) / 1000.0
            else:
                cpu_req = float(cpu_req_str)
            
            # Parse Memory request
            mem_req_str = str(service_config.resources_per_replica.get('requests', {}).get('memory', '64Mi'))
            if mem_req_str.endswith('Mi'):
                mem_req_gb = float(mem_req_str[:-2]) / 1024.0
            elif mem_req_str.endswith('Gi'):
                mem_req_gb = float(mem_req_str[:-2])
            else:
                try:
                    mem_req_gb = float(mem_req_str)
                except ValueError:
                    mem_req_gb = 0.064 # Default fallback

            capacity_cpu = int(metrics.cpu_available_cores / cpu_req) if cpu_req > 0 else 100
            capacity_mem = int(metrics.memory_available_gb / mem_req_gb) if mem_req_gb > 0 else 100
            capacity = min(capacity_cpu, capacity_mem)
            
            # Max replicas constraint
            max_replicas = service_config.autoscaling.get('max_replicas', 20)
            capacity = min(capacity, max_replicas)
        else:
            capacity = int(metrics.cpu_available_cores * 2)
        
        result = {
            'cluster_name': self.cluster_name,
            'score': breakdown['total_score'],
            'score_breakdown': breakdown,
            'capacity': max(0, capacity),
            'metrics': asdict(metrics)
        }
        
        logger.info(f"Score computed: {breakdown['total_score']:.3f}, capacity={capacity}")
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