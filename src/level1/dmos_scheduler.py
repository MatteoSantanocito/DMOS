"""
DMOS Scheduler (Centralized with Local Estimation)
Computes cluster scores locally using centralized Prometheus,
then performs winner determination.

Architecture:
- Central Prometheus (192.168.1.245:30090) aggregates metrics from all clusters
- Scheduler creates one ClusterEstimator per cluster locally
- No need for remote HTTP estimator processes
- Score computation happens in-process (faster, no network overhead)
"""

import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader
from .score_functions import ScoreFunctions, ClusterMetrics, ScoreParameters
from .winner_determination import WinnerDetermination, ClusterBid, Allocation
from ..metrics.prometheus_client import PrometheusClient
from ..metrics.carbon_client import CarbonClient

logger = get_logger("DMOSScheduler")


# Hard constraint thresholds
CPU_HARD_LIMIT_PCT = 90.0       # Exclude cluster if CPU utilization > 90%
MEMORY_HARD_LIMIT_PCT = 90.0    # Exclude cluster if memory utilization > 90%
MIN_CPU_CORES_AVAILABLE = 0.2   # At least 0.2 cores free
MIN_MEMORY_GB_AVAILABLE = 0.2   # At least 0.2 GB free


class DMOSScheduler:
    """
    DMOS Centralized Scheduler with Local Score Computation
    
    Instead of querying remote estimator HTTP endpoints, this version
    computes all scores locally using the centralized Prometheus instance.
    This eliminates the need to deploy and manage estimator processes
    on each cluster node.
    """
    
    def __init__(self, config_path: str = "config"):
        self.config = ConfigLoader(config_path)
        
        # Winner determination algorithm
        self.winner_det = WinnerDetermination()
        
        # Centralized Prometheus client
        self.prometheus = PrometheusClient(
            url=self.config.prometheus.url,
            timeout=5
        )
        
        # Carbon client (shared across all clusters)
        self.carbon_client = CarbonClient(
            self.config.carbon_raw['carbon_intensity']
        )
        
        # Score functions (shared, same weights for all clusters)
        self.score_func = ScoreFunctions(
            weights={
                'omega_latency': self.config.score_weights.omega_latency,
                'omega_capacity': self.config.score_weights.omega_capacity,
                'omega_load': self.config.score_weights.omega_load,
                'omega_carbon': self.config.score_weights.omega_carbon
            }
        )
        
        # Cluster configurations
        self.cluster_configs = self.config.get_all_clusters()
        
        logger.info(f"DMOS Scheduler initialized (local mode) with "
                    f"{len(self.cluster_configs)} clusters")
        for name, cfg in self.cluster_configs.items():
            logger.info(f"  {name}: {cfg.region} ({cfg.ip})")
    
    def _collect_cluster_metrics(
        self, 
        cluster_name: str,
        service_name: str = "frontend"
    ) -> Optional[ClusterMetrics]:
        """
        Collect metrics for a specific cluster from centralized Prometheus
        
        Args:
            cluster_name: Cluster to collect metrics for
            service_name: Service name for traffic/latency queries
            
        Returns:
            ClusterMetrics or None if collection failed
        """
        cluster_cfg = self.cluster_configs.get(cluster_name)
        if not cluster_cfg:
            logger.error(f"Unknown cluster: {cluster_name}")
            return None
        
        try:
            # ── CPU metrics ───────────────────────────────────────────
            cpu_available = self.prometheus.get_cpu_available() or 0.0
            cpu_total = cluster_cfg.cpu_cores
            
            # ── Memory metrics ────────────────────────────────────────
            memory_available = self.prometheus.get_memory_available_gb() or 0.0
            memory_total = cluster_cfg.memory_gb
            
            # ── Traffic metrics (real from Prometheus) ────────────────
            svc_cfg = self.config.get_service(service_name)
            namespace = svc_cfg.namespace if svc_cfg else "online-boutique"
            
            request_rate = self.prometheus.get_request_rate(
                service=service_name,
                namespace=namespace
            )
            
            if request_rate is None:
                # Fallback: estimate from CPU usage
                cpu_usage_pct = self.prometheus.get_cpu_usage_percent(
                    deployment=service_name,
                    namespace=namespace
                )
                if cpu_usage_pct is not None and cpu_usage_pct > 0:
                    capacity = svc_cfg.capacity_req_per_sec if svc_cfg else 50
                    request_rate = (cpu_usage_pct / 100.0) * capacity
                else:
                    request_rate = 0.0
                    logger.warning(f"No traffic metrics for {cluster_name}, defaulting to 0")
            
            # Max request rate
            capacity_per_core = svc_cfg.capacity_req_per_sec if svc_cfg else 50
            request_rate_max = cpu_total * capacity_per_core
            
            # ── Latency metrics (real from Prometheus) ────────────────
            latency_p95 = self.prometheus.get_latency_p95(
                service=service_name,
                namespace=namespace
            )
            
            if latency_p95 is not None and latency_p95 > 0:
                latency_mean = latency_p95 / 1.65
                latency_variance = (latency_p95 - latency_mean) ** 2
            else:
                latency_mean = cluster_cfg.baseline_latency_ms
                latency_variance = cluster_cfg.latency.get('variance_ms', 15.0) ** 2
            
            # ── Carbon intensity ──────────────────────────────────────
            carbon_region = cluster_cfg.carbon.get('region_code', 'DE')
            carbon_intensity = self.carbon_client.get_carbon_intensity(carbon_region) or 300.0
            
            # ── Cost ──────────────────────────────────────────────────
            cost_per_replica = cluster_cfg.cost_per_replica_hour
            
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
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics for {cluster_name}: {e}")
            return None
    
    def _compute_cluster_score(
        self,
        cluster_name: str,
        service_name: str = "frontend",
        predicted_load: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Compute score for a single cluster (locally, no HTTP)
        
        Includes hard constraint checking.
        
        Args:
            cluster_name: Cluster to score
            service_name: Service name
            predicted_load: Optional predicted load
            
        Returns:
            Score result dict or None
        """
        metrics = self._collect_cluster_metrics(cluster_name, service_name)
        if metrics is None:
            return None
        
        # ── Hard Constraint Check ─────────────────────────────────────
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
        
        if cpu_util_pct > CPU_HARD_LIMIT_PCT:
            is_eligible = False
            exclusion_reason = f"CPU {cpu_util_pct:.0f}% > {CPU_HARD_LIMIT_PCT}%"
        elif mem_util_pct > MEMORY_HARD_LIMIT_PCT:
            is_eligible = False
            exclusion_reason = f"Memory {mem_util_pct:.0f}% > {MEMORY_HARD_LIMIT_PCT}%"
        elif metrics.cpu_available_cores < MIN_CPU_CORES_AVAILABLE:
            is_eligible = False
            exclusion_reason = f"CPU avail {metrics.cpu_available_cores:.2f} < {MIN_CPU_CORES_AVAILABLE}"
        elif metrics.memory_available_gb < MIN_MEMORY_GB_AVAILABLE:
            is_eligible = False
            exclusion_reason = f"Mem avail {metrics.memory_available_gb:.2f}GB < {MIN_MEMORY_GB_AVAILABLE}GB"
        
        if not is_eligible:
            logger.warning(f"⛔ {cluster_name} EXCLUDED: {exclusion_reason}")
            return {
                'cluster_name': cluster_name,
                'score': 0.0,
                'score_breakdown': {
                    'phi_latency': 0.0, 'phi_capacity': 0.0,
                    'phi_load': 0.0, 'phi_carbon': 0.0
                },
                'capacity': 0,
                'eligible': False,
                'exclusion_reason': exclusion_reason,
                'metrics': {
                    'cpu_utilization_pct': cpu_util_pct,
                    'mem_utilization_pct': mem_util_pct,
                    'carbon_intensity_gco2_kwh': metrics.carbon_intensity_gco2_kwh,
                    'latency_mean_ms': metrics.latency_mean_ms
                }
            }
        
        # ── Score computation ─────────────────────────────────────────
        breakdown = self.score_func.compute_score_breakdown(
            metrics, predicted_load=predicted_load
        )
        
        # ── Capacity calculation ──────────────────────────────────────
        service_config = self.config.get_service(service_name)
        
        if service_config:
            cpu_req_str = service_config.cpu_request
            if cpu_req_str.endswith('m'):
                cpu_req = float(cpu_req_str[:-1]) / 1000.0
            else:
                cpu_req = float(cpu_req_str)
            
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
            max_replicas = service_config.max_replicas
        else:
            capacity = int(metrics.cpu_available_cores * 2)
            max_replicas = 20
        
        capacity = min(max(0, capacity), max_replicas)
        
        logger.info(f"Score {cluster_name}: {breakdown['total_score']:.3f} "
                    f"(lat={breakdown['phi_latency']:.3f}, cap={breakdown['phi_capacity']:.3f}, "
                    f"load={breakdown['phi_load']:.3f}, carbon={breakdown['phi_carbon']:.3f}) "
                    f"capacity={capacity}, cpu={cpu_util_pct:.0f}%, "
                    f"CI={metrics.carbon_intensity_gco2_kwh:.0f}gCO2")
        
        return {
            'cluster_name': cluster_name,
            'score': breakdown['total_score'],
            'score_breakdown': {
                'phi_latency': breakdown['phi_latency'],
                'phi_capacity': breakdown['phi_capacity'],
                'phi_load': breakdown['phi_load'],
                'phi_carbon': breakdown['phi_carbon']
            },
            'capacity': capacity,
            'eligible': True,
            'exclusion_reason': None,
            'metrics': {
                'cpu_available_cores': metrics.cpu_available_cores,
                'cpu_total_cores': metrics.cpu_total_cores,
                'cpu_utilization_pct': cpu_util_pct,
                'memory_available_gb': metrics.memory_available_gb,
                'memory_total_gb': metrics.memory_total_gb,
                'mem_utilization_pct': mem_util_pct,
                'carbon_intensity_gco2_kwh': metrics.carbon_intensity_gco2_kwh,
                'latency_mean_ms': metrics.latency_mean_ms
            }
        }
    
    def collect_scores(
        self,
        service_name: str,
        predicted_load: Optional[float] = None
    ) -> List[ClusterBid]:
        """
        Collect scores from all clusters (locally, no HTTP)
        """
        logger.info(f"Computing scores for '{service_name}' across all clusters...")
        
        bids = []
        excluded = []
        
        # Compute scores for each cluster
        for cluster_name in self.cluster_configs.keys():
            result = self._compute_cluster_score(
                cluster_name, service_name, predicted_load
            )
            
            if result is None:
                logger.warning(f"No result for {cluster_name}")
                continue
            
            if not result.get('eligible', True):
                excluded.append(cluster_name)
                continue
            
            if result['score'] > 0 and result['capacity'] > 0:
                bids.append(ClusterBid(
                    cluster_name=result['cluster_name'],
                    score=result['score'],
                    capacity=result['capacity']
                ))
        
        if excluded:
            logger.warning(f"⛔ Excluded clusters: {excluded}")
        
        logger.info(f"Collected {len(bids)}/{len(self.cluster_configs)} bids "
                    f"({len(excluded)} excluded)")
        
        return bids
    
    def schedule_service(
        self,
        service_name: str,
        total_replicas: int,
        predicted_load: Optional[float] = None
    ) -> Tuple[List[Allocation], bool]:
        """
        Schedule a service across clusters
        """
        logger.info(f"Scheduling '{service_name}' with {total_replicas} replicas")
        
        # Step 1: Collect scores
        start_time = time.time()
        bids = self.collect_scores(service_name, predicted_load)
        collection_time = (time.time() - start_time) * 1000
        
        if not bids:
            logger.error("No eligible clusters available!")
            return [], False
        
        # Step 2: Winner determination
        allocations, success = self.winner_det.allocate(bids, total_replicas)
        
        # Step 3: Log results
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Scheduling completed in {total_time:.0f}ms "
                    f"(collection={collection_time:.0f}ms, "
                    f"winner_det={(total_time - collection_time):.0f}ms)")
        
        if success:
            logger.info(f"✅ Allocated {total_replicas} replicas across "
                       f"{len(allocations)} clusters:")
            for alloc in allocations:
                logger.info(f"   {alloc}")
            
            jain = self.winner_det.compute_fairness_jain_index(allocations)
            logger.info(f"   Jain fairness index: {jain:.3f}")
        else:
            logger.error(f"❌ Failed to satisfy demand ({total_replicas} replicas)")
        
        return allocations, success