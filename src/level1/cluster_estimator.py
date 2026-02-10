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
    
    Responsibilities:
    1. Monitor local cluster metrics (Prometheus, K8s API)
    2. Query carbon intensity for cluster region
    3. Compute multi-objective score when requested
    4. Return score + capacity to scheduler
    
    Similar to Karmada Resource Estimator but with multi-objective scoring
    """
    
    def __init__(self, cluster_name: str, config_path: str = "config"):
        """
        Initialize cluster estimator
        
        Args:
            cluster_name: Name of this cluster (e.g., "cluster1")
            config_path: Path to config directory
        """
        self.cluster_name = cluster_name
        
        # Load configuration
        self.config = ConfigLoader(config_path)
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
            self.config.carbon_raw['carbon_intensity']
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
        
        Returns:
            ClusterMetrics object
        """
        # CPU metrics
        cpu_available = self.prometheus.get_cpu_available() or 0.0
        cpu_total = self.cluster_config.cpu_cores
        
        # Memory metrics
        memory_available = self.prometheus.get_memory_available_gb() or 0.0
        memory_total = self.cluster_config.memory_gb
        
        # Traffic metrics (for now, use placeholder)
        # TODO: Implement per-service traffic aggregation
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
        """
        Get cluster metrics
        
        Args:
            use_cache: If True, return cached metrics if fresh enough
        
        Returns:
            ClusterMetrics
        """
        if use_cache and self.cached_metrics:
            age = time.time() - self.cache_timestamp
            if age < self.cache_ttl:
                logger.debug(f"Using cached metrics (age={age:.1f}s)")
                return self.cached_metrics
        
        # Collect fresh metrics
        return self._collect_metrics()
    
    def compute_score(
        self, 
        service_name: Optional[str] = None,
        predicted_load: Optional[float] = None
    ) -> Dict:
        """
        Compute multi-objective score for this cluster
        
        Args:
            service_name: Service to score for (optional)
            predicted_load: Optional predicted load for scoring
        
        Returns:
            Dict with score breakdown and capacity
        """
        metrics = self.get_metrics(use_cache=True)
        
        # Compute score breakdown
        breakdown = self.score_func.compute_score_breakdown(
            metrics,
            predicted_load=predicted_load
        )
        
        # Calculate available capacity (in replicas)
        # Conservative estimate: use minimum of CPU and memory constraints
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
            
            capacity_cpu = int(metrics.cpu_available_cores / cpu_req)
            capacity_mem = int(metrics.memory_available_gb / mem_req_gb)
            capacity = min(capacity_cpu, capacity_mem)
        else:
            # Generic estimate
            capacity = int(metrics.cpu_available_cores * 2)  # 2 replicas per core
        
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
            'metrics': {
                'cpu_available_cores': metrics.cpu_available_cores,
                'cpu_total_cores': metrics.cpu_total_cores,
                'memory_available_gb': metrics.memory_available_gb,
                'memory_total_gb': metrics.memory_total_gb,
                'carbon_intensity_gco2_kwh': metrics.carbon_intensity_gco2_kwh,
                'latency_mean_ms': metrics.latency_mean_ms
            }
        }
        
        logger.info(f"Score computed: {breakdown['total_score']:.3f}, capacity={capacity}")
        
        return result


# ============================================================================
# HTTP API (Flask) - Karmada Estimator exposes gRPC, we use REST for simplicity
# ============================================================================

def create_estimator_api(cluster_name: str, config_path: str = "config") -> Flask:
    """
    Create Flask API for cluster estimator
    
    Args:
        cluster_name: This cluster's name
        config_path: Path to config
    
    Returns:
        Flask app
    """
    app = Flask(f"cluster-estimator-{cluster_name}")
    estimator = ClusterEstimator(cluster_name, config_path)
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({'status': 'healthy', 'cluster': cluster_name}), 200
    
    @app.route('/estimate', methods=['POST'])
    def estimate():
        """
        Score estimation endpoint
        
        Request body:
        {
            "service_name": "frontend",
            "predicted_load": 150.0  // optional
        }
        
        Response:
        {
            "cluster_name": "cluster1",
            "score": 0.708,
            "capacity": 12,
            "score_breakdown": {...},
            "metrics": {...}
        }
        """
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
            return jsonify({'error': str(e)}), 500
    
    @app.route('/metrics', methods=['GET'])
    def get_metrics():
        """Get current cluster metrics"""
        try:
            metrics = estimator.get_metrics(use_cache=False)
            return jsonify(asdict(metrics)), 200
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app


def run_estimator(cluster_name: str, port: int = 8080, config_path: str = "config"):
    """
    Run cluster estimator as HTTP server
    
    Args:
        cluster_name: This cluster's name
        port: HTTP port
        config_path: Path to config
    """
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