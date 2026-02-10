"""
Configuration loader for DMOS
Loads YAML configuration files with validation
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from .logger import get_logger

logger = get_logger("ConfigLoader")


@dataclass
class ClusterConfig:
    """Configuration for a single cluster"""
    name: str
    region: str
    location: str
    ip: str
    kubeconfig_path: str
    resources: Dict[str, Any]
    pricing: Dict[str, float]
    carbon: Dict[str, Any]
    latency: Dict[str, float]
    
    server: str = ""  # Kubernetes API server URL
    estimator_url: str = ""  # Cluster estimator endpoint
    display_name: Optional[str] = None  # Human-readable name
    
    @property
    def cpu_cores(self) -> int:
        return self.resources.get('cpu_cores', 4)
    
    @property
    def memory_gb(self) -> float:
        return self.resources.get('memory_gb', 8.0)
    
    @property
    def cost_per_replica_hour(self) -> float:
        return self.pricing.get('cost_per_replica_hour', 0.1)
    
    @property
    def baseline_latency_ms(self) -> float:
        return self.latency.get('baseline_ms', 50.0)


@dataclass
class WeightsConfig:
    """Multi-objective weights configuration"""
    alpha: float  # Latency weight
    beta: float   # Cost weight
    gamma: float  # Fairness weight
    delta: float  # Carbon weight
    
    def __post_init__(self):
        total = self.alpha + self.beta + self.gamma + self.delta
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class ScoreWeightsConfig:
    """Score function weights configuration"""
    omega_latency: float
    omega_capacity: float
    omega_load: float
    omega_carbon: float
    
    def __post_init__(self):
        total = self.omega_latency + self.omega_capacity + self.omega_load + self.omega_carbon
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Score weights must sum to 1.0, got {total}")


@dataclass
class ServiceConfig:
    """Configuration for a single service"""
    name: str
    namespace: str
    deployment_name: str
    resources_per_replica: Dict[str, Any]
    capacity_req_per_sec: int
    sla: Dict[str, Any]
    power_watts: int
    autoscaling: Dict[str, Any]
    
    @property
    def cpu_request(self) -> str:
        return self.resources_per_replica.get('requests', {}).get('cpu', '100m')
    
    @property
    def memory_request(self) -> str:
        return self.resources_per_replica.get('requests', {}).get('memory', '64Mi')
    
    @property
    def min_replicas(self) -> int:
        return self.autoscaling.get('min_replicas', 1)
    
    @property
    def max_replicas(self) -> int:
        return self.autoscaling.get('max_replicas', 10)


@dataclass
class PrometheusConfig:
    """Prometheus configuration"""
    url: str
    timeout_seconds: int
    queries: Dict[str, str]


@dataclass
class CarbonConfig:
    """Carbon intensity configuration"""
    provider: str
    mock: Dict[str, Any] = field(default_factory=dict)
    electricitymaps: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    """
    Loads and manages DMOS configuration files
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigLoader
        
        Args:
            config_dir: Directory containing YAML config files
        """
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        logger.info(f"Loading configuration from: {self.config_dir}")
        
        # Load all configs
        self.clusters_raw = self._load_yaml("clusters.yaml")
        self.weights_raw = self._load_yaml("weights.yaml")
        self.services_raw = self._load_yaml("services.yaml")
        self.monitoring_raw = self._load_yaml("monitoring.yaml")
        self.carbon_raw = self._load_yaml("carbon.yaml")
        
        # Parse into structured objects
        self._parse_configs()
        
        logger.info(f"Loaded {len(self.clusters)} clusters, {len(self.services)} services")
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file"""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        logger.debug(f"Loaded {filename}")
        return data
    
    def _parse_configs(self):
        """Parse raw YAML data into structured config objects"""
        
        # Parse clusters (FIX: gestisci formato dict)
        self.clusters: Dict[str, ClusterConfig] = {}
        
        clusters_data = self.clusters_raw.get('clusters', {})
        
        if isinstance(clusters_data, dict):
            # Formato dict (nuovo) ✅
            for cluster_name, cluster_data in clusters_data.items():
                if 'name' not in cluster_data:
                    cluster_data['name'] = cluster_name
                cluster = ClusterConfig(**cluster_data)
                self.clusters[cluster.name] = cluster
        else:
            # Formato lista (legacy)
            for cluster_data in clusters_data:
                cluster = ClusterConfig(**cluster_data)
                self.clusters[cluster.name] = cluster
        
        # Parse weights (get default scenario: balanced)
        weights_scenario = self.weights_raw.get('objectives', {}).get('balanced', {})
        self.weights = WeightsConfig(
            alpha=weights_scenario.get('alpha', 0.35),
            beta=weights_scenario.get('beta', 0.25),
            gamma=weights_scenario.get('gamma', 0.15),
            delta=weights_scenario.get('delta', 0.25)
        )
        
        # Parse score weights
        score_weights_data = self.weights_raw.get('score_weights', {})
        self.score_weights = ScoreWeightsConfig(
            omega_latency=score_weights_data.get('omega_latency', 0.4),
            omega_capacity=score_weights_data.get('omega_capacity', 0.3),
            omega_load=score_weights_data.get('omega_load', 0.1),
            omega_carbon=score_weights_data.get('omega_carbon', 0.2)
        )
        
        # Parse score parameters
        self.score_params = self.weights_raw.get('parameters', {})
        
        # Parse services (FIX: stessa logica)
        self.services: Dict[str, ServiceConfig] = {}
        
        services_data = self.services_raw.get('services', {})
        
        if isinstance(services_data, dict):
            # Formato dict
            for service_name, service_data in services_data.items():
                if 'name' not in service_data:
                    service_data['name'] = service_name
                service = ServiceConfig(**service_data)
                self.services[service.name] = service
        else:
            # Formato lista
            for service_data in services_data:
                service = ServiceConfig(**service_data)
                self.services[service.name] = service
        
        # Parse autoscaling global params
        self.autoscaling_params = self.services_raw.get('global_autoscaling', {})
        
        # Parse Prometheus config
        prom_data = self.monitoring_raw.get('prometheus', {})
        self.prometheus = PrometheusConfig(
            url=prom_data.get('url', 'http://localhost:9090'),
            timeout_seconds=prom_data.get('timeout_seconds', 5),
            queries=prom_data.get('queries', {})
        )
        
        # Parse Carbon config
        carbon_data = self.carbon_raw.get('carbon_intensity', {})
        self.carbon = CarbonConfig(
            provider=carbon_data.get('provider', 'mock'),
            mock=carbon_data.get('mock', {}),
            electricitymaps=carbon_data.get('electricitymaps', {}),
            cache=carbon_data.get('cache', {})
        )
        
        # Global settings
        self.global_config = self.clusters_raw.get('global', {})
    
    def get_cluster(self, name: str) -> Optional[ClusterConfig]:
        """Get cluster config by name"""
        return self.clusters.get(name)
    
    def get_service(self, name: str) -> Optional[ServiceConfig]:
        """Get service config by name"""
        return self.services.get(name)
    
    def get_all_clusters(self) -> Dict[str, ClusterConfig]:
        """Get all cluster configs"""
        return self.clusters
    
    def get_all_services(self) -> Dict[str, ServiceConfig]:
        """Get all service configs"""
        return self.services
    
    def get_prometheus_url(self) -> str:
        """Get Prometheus URL"""
        return self.prometheus.url
    
    def get_scheduling_interval(self) -> int:
        """Get scheduling interval in seconds"""
        return self.global_config.get('scheduling_interval_seconds', 30)
    
    def get_pd_params(self) -> Dict[str, float]:
        """Get PD controller parameters"""
        pd_config = self.autoscaling_params.get('pd_controller', {})
        return {
            'kp': pd_config.get('kp', 5.0),
            'kd': pd_config.get('kd', 300.0)
        }
    
    def reload(self):
        """Reload all configuration files"""
        logger.info("Reloading configuration...")
        self.__init__(config_dir=str(self.config_dir))