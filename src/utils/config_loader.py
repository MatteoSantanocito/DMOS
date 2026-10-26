"""
Configuration loader for DMOS
Loads YAML configuration files with validation
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from .logger import get_logger

logger = get_logger("ConfigLoader")

# Kubernetes workload / namespace names accepted in the `application` section
# (DNS-1123 label). The names end up inside PromQL label matchers, so nothing
# else is allowed.
K8S_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")

# Entry service assumed when services.yaml has no `application` section.
DEFAULT_ENTRY_SERVICE = "frontend"


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

    # NodePort della Prometheus instance del cluster.
    # Default 32090 = porta scelta da unict-cclab/proxmox-k3s (kube-prometheus-stack).
    # 30090 = porta storica della testbed originale della tesi (standalone prometheus).
    # Override per cluster in local.yaml se la topologia non è omogenea.
    prometheus_port: int = 32090

    # Porta dell'API server di Kubernetes. 6443 è il default di K3s standalone
    # (sia testbed originale sia setup proxmox-k3s). Override solo se gira
    # K8s vanilla con configurazione non-default.
    k8s_api_port: int = 6443

    # Optional netem configuration (consumed by scripts/setup-netem-and-ping.sh
    # and experiments/capture_env_state.py, not by the runtime scheduler).
    # delay_ms = 0 or interface = ""
    # → netem skipped on this cluster.
    netem: Dict[str, Any] = field(default_factory=dict)
    
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
    """Score function weights configuration (5 componenti con Φ_geo_load moltiplicativo)"""
    omega_latency: float
    omega_capacity: float
    omega_geo_load: float    # Termine moltiplicativo: ω_geo · (Φ_load × Φ_net)
    omega_carbon: float
    omega_demand: float = 0.0    # Domanda geografica via Cilium Ingress (Φ_demand)

    def __post_init__(self):
        total = (self.omega_latency + self.omega_capacity +
                 self.omega_geo_load + self.omega_carbon +
                 self.omega_demand)
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
    capacity_per_cluster: Dict[str, float] = field(default_factory=dict)
    max_replicas_per_cluster: Dict[str, int] = field(default_factory=dict)
    min_replicas_per_cluster: Dict[str, int] = field(default_factory=dict)
    # [FANOUT] chiamate gRPC verso questo servizio per ogni richiesta al
    # servizio d'ingresso (frontend). Misurato da Hubble (rps_dep/rps_frontend).
    # Usato dal co-scaling fanout-aware: un servizio chiamato N volte per
    # richiesta va dimensionato per N×traffico. Default 1.0 (1:1).
    calls_per_request: float = 1.0

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

    def get_min_replicas_for_cluster(self, cluster_name: str) -> int:
        """Returns min replicas for a specific cluster.

        Uses min_replicas_per_cluster if configured, otherwise falls back
        to the global autoscaling.min_replicas.
        """
        if self.min_replicas_per_cluster and cluster_name in self.min_replicas_per_cluster:
            return int(self.min_replicas_per_cluster[cluster_name])
        return self.min_replicas

    def get_max_replicas_for_cluster(self, cluster_name: str) -> int:
        """Returns max replicas for a specific cluster.

        Uses max_replicas_per_cluster if configured, otherwise falls back
        to the global autoscaling.max_replicas. This prevents node overcommit
        on high-latency clusters where co-located backends + frontend exceed
        the node's allocatable CPU.
        """
        if self.max_replicas_per_cluster and cluster_name in self.max_replicas_per_cluster:
            return int(self.max_replicas_per_cluster[cluster_name])
        return self.max_replicas

    def get_capacity_for_cluster(self, cluster_name: str) -> float:
        """Returns capacity (rps) for a specific cluster.

        Uses capacity_per_cluster if configured, otherwise falls back
        to the global capacity_req_per_sec. This allows services affected
        by tc netem (e.g. frontend) to have accurate per-cluster capacity
        while backend services (intra-cluster gRPC, no netem) use a single value.
        """
        if self.capacity_per_cluster and cluster_name in self.capacity_per_cluster:
            return float(self.capacity_per_cluster[cluster_name])
        return float(self.capacity_req_per_sec)


@dataclass
class DemandConfig:
    """How the demand entering each cluster is measured.

    The demand of cluster i is the rate of HTTP requests sent to the entry
    service by clients running in cluster i, counted on the client side
    (Hubble `reporter="client"`), so a request stays attributed to the cluster
    it entered even when a frontend of another cluster serves it.
    """
    # Workloads whose requests to the entry service count as demand (e.g. the
    # ingress proxy). Empty = every client in the cluster (default).
    source_workloads: List[str] = field(default_factory=list)
    # When the client-side count is 0 or unavailable, fall back to a
    # server-side count (attributes requests to the cluster that SERVES them:
    # only correct when traffic is not shared across clusters).
    server_side_fallback: bool = True


@dataclass
class ApplicationConfig:
    """The application DMOS manages (services.yaml → `application`)."""
    entry_service: str                       # sized on the demand entering each cluster
    co_scaled_services: List[str]            # sized from the entry service, in this order
    namespace: Optional[str] = None          # default namespace of the services
    demand: DemandConfig = field(default_factory=DemandConfig)


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

        # `local.yaml` is the single environment-specific source of truth
        # (cluster topology, ssh credentials, netem, alertmanager webhook).
        # It is gitignored — generated by scripts/bootstrap-assisted.sh or
        # copied manually from config/local.yaml.example.
        self.local_raw = self._load_yaml("local.yaml")

        self.weights_raw = self._load_yaml("weights.yaml")
        self.services_raw = self._load_yaml("services.yaml")
        self.carbon_raw = self._load_yaml("carbon.yaml")

        # Backward-compat aliases for downstream code that still references
        # `clusters_raw` and `monitoring_raw` separately. The single
        # local.yaml supersedes both. monitoring_raw is now an empty stub
        # (the Prometheus URL is derived per-cluster from `clusters[].ip`).
        self.clusters_raw = self.local_raw
        self.monitoring_raw = {}

        # Parse into structured objects
        self._parse_configs()

        logger.info(f"Loaded {len(self.clusters)} clusters, {len(self.services)} services")

    # Files that are generated per-deployment (gitignored, contain real IPs).
    # When missing, point the user to the bootstrap script or to the template.
    _GENERATED_FILES = {"local.yaml"}

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file"""
        filepath = self.config_dir / filename

        if not filepath.exists():
            if filename in self._GENERATED_FILES:
                template_path = self.config_dir / f"{filename}.example"
                hint = (
                    f"\n\nThis file is environment-specific and gitignored. "
                    f"It contains your cluster topology, SSH credentials and "
                    f"optional alerting webhook — the single source of truth "
                    f"for everything DMOS and its tooling need to know about "
                    f"your deployment.\n\n"
                    f"Two ways to create it:\n"
                    f"  (1) Run: bash scripts/bootstrap-assisted.sh\n"
                    f"      (interactive — also renders the Helm values for "
                    f"Cilium and ping_exporter into .cache/)\n"
                    f"  (2) Copy the template by hand:\n"
                    f"      cp {template_path} {filepath}\n"
                    f"      then edit {filepath} and replace every "
                    f"<PLACEHOLDER>, and run\n"
                    f"      `bash scripts/render-runtime-files.sh` to "
                    f"produce the Helm values."
                )
                raise FileNotFoundError(
                    f"Config file not found: {filepath}{hint}"
                )
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
        
        # Parse score weights — legge il profilo attivo da score_weights.active
        # Override da env per gli esperimenti di ablation (Study B): permette di
        # cambiare profilo senza editare il YAML, es:
        #   DMOS_SCORE_PROFILE=latonly python -m src.dmos_main
        score_weights_raw = self.weights_raw.get('score_weights', {})
        active_profile = os.environ.get(
            'DMOS_SCORE_PROFILE',
            score_weights_raw.get('active', 'balanced')
        )
        score_weights_data = score_weights_raw.get(active_profile, {})
        if not score_weights_data:
            logger.warning(
                f"Profilo score '{active_profile}' non trovato in weights.yaml, "
                f"uso defaults hardcoded"
            )
            score_weights_data = {}
        else:
            logger.info(f"Score profile attivo: '{active_profile}'")
        # Name of the profile actually in use (recorded in the run provenance).
        # Suffix marks the hard-coded fallback when the profile does not exist.
        self.active_score_profile = (
            active_profile if score_weights_raw.get(active_profile)
            else f"{active_profile} (missing -> hardcoded defaults)"
        )

        self.score_weights = ScoreWeightsConfig(
            omega_latency=score_weights_data.get('omega_latency', 0.25),
            omega_capacity=score_weights_data.get('omega_capacity', 0.20),
            omega_geo_load=score_weights_data.get('omega_geo_load', 0.25),
            omega_carbon=score_weights_data.get('omega_carbon', 0.20),
            omega_demand=score_weights_data.get('omega_demand', 0.10),
        )

        # Parse cold_start weights (Phase 1 — blind allocation)
        # Usato dallo scheduler nei primi COLD_START_SECONDS secondi,
        # quando le metriche osservate (latenza p95, CPU, load) non sono affidabili.
        cold_start_data = score_weights_raw.get('cold_start', {})
        if cold_start_data:
            self.cold_start_weights = ScoreWeightsConfig(
                omega_latency=cold_start_data.get('omega_latency', 0.00),
                omega_capacity=cold_start_data.get('omega_capacity', 0.00),
                omega_geo_load=cold_start_data.get('omega_geo_load', 0.35),
                omega_carbon=cold_start_data.get('omega_carbon', 0.30),
                omega_demand=cold_start_data.get('omega_demand', 0.35),
            )
            logger.info("Cold-start weights caricati (Phase 1 blind allocation)")
        else:
            # Fallback: stessi pesi del profilo attivo (nessuna distinzione di fase)
            self.cold_start_weights = self.score_weights
            logger.warning("Profilo 'cold_start' non trovato in weights.yaml, "
                           "Phase 1 userà gli stessi pesi di Phase 2")

        # Parse network parameters (geo-awareness via ping_exporter)
        net_params = self.weights_raw.get('network_parameters', {})
        self.network_params = {
            'rho': net_params.get('rho', 2.0),
            'rtt_max_ms': net_params.get('rtt_max_ms', 500.0),
            'fallback_rtt_ms': net_params.get('fallback_rtt_ms', 5.0),
        }

        # Parse load parameters (sigmoid saturation threshold)
        load_params = self.weights_raw.get('load_parameters', {})
        self.load_params = {
            'threshold': load_params.get('threshold', 0.7),
            'steepness': load_params.get('steepness', 10),
            'floor': load_params.get('floor', 0.2),
        }

        # Parse score parameters (legacy key)
        self.score_params = self.weights_raw.get('parameters', {})
        
        # Parse services — accetta sia list che dict in services.yaml,
        # normalizza sempre a dict {service_name: ServiceConfig}.
        self.services: Dict[str, ServiceConfig] = {}

        app_raw = self.services_raw.get('application') or {}
        if not isinstance(app_raw, dict):
            raise ValueError("services.yaml: `application` must be a mapping")
        default_namespace = app_raw.get('namespace')
        if default_namespace is not None and (
                not isinstance(default_namespace, str) or not K8S_NAME_RE.match(default_namespace)):
            raise ValueError(f"services.yaml: application.namespace {default_namespace!r} "
                             f"is not a valid Kubernetes namespace name")

        services_data = self.services_raw.get('services', {})

        if isinstance(services_data, list):
            logger.debug("Converting services list to dictionary format")
            services_data = {
                s['name']: s
                for s in services_data
                if isinstance(s, dict) and 'name' in s
            }

        for service_name, service_data in services_data.items():
            if 'name' not in service_data:
                service_data['name'] = service_name
            # A service without its own namespace lives in application.namespace.
            if not service_data.get('namespace'):
                if default_namespace is None:
                    raise ValueError(
                        f"services.yaml: service '{service_name}' has no namespace and "
                        f"application.namespace is not set")
                service_data['namespace'] = default_namespace
            service = ServiceConfig(**service_data)
            self.services[service.name] = service

        self.application = self._parse_application(app_raw, default_namespace)
        
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

        # SSH credentials (used by tooling on the operator machine — not by
        # the runtime scheduler). Stored verbatim from local.yaml so scripts
        # that import ConfigLoader can use them without re-parsing the file.
        self.ssh = self.local_raw.get('ssh', {})

        # Optional Alertmanager webhook URL (consumed by render-runtime-files
        # when templating deployments/alertmanager-config.yaml).
        self.alertmanager_webhook_url = (
            self.local_raw.get('alertmanager', {}).get('webhook_url') or ''
        )

        # Per-cluster netem configuration (consumed by scripts/setup-netem-and-ping.sh
        # via .cache/test-env.json). Each entry is a dict with keys:
        # delay_ms, jitter_ms, interface.
        self.netem_per_cluster = {
            c.name: c.netem or {} for c in self.clusters.values()
        }

    _APPLICATION_KEYS = {'namespace', 'entry_service', 'co_scaled_services', 'demand'}
    _DEMAND_KEYS = {'source_workloads', 'server_side_fallback'}

    def _parse_application(self, app_raw: Dict[str, Any],
                           default_namespace: Optional[str]) -> ApplicationConfig:
        """Validate services.yaml → `application`.

        Without the section the defaults reproduce the historical behaviour:
        entry service `frontend`, every other service co-scaled in file order,
        demand counted from any client, server-side fallback on.
        """
        unknown = set(app_raw) - self._APPLICATION_KEYS
        if unknown:
            raise ValueError(f"services.yaml: unknown key(s) in `application`: {sorted(unknown)} "
                             f"(allowed: {sorted(self._APPLICATION_KEYS)})")
        names = list(self.services)

        entry = app_raw.get('entry_service', DEFAULT_ENTRY_SERVICE)
        if entry not in self.services:
            raise ValueError(f"services.yaml: application.entry_service '{entry}' is not a "
                             f"service of `services` ({names})")

        co_scaled = app_raw.get('co_scaled_services')
        if co_scaled is None:
            co_scaled = [n for n in names if n != entry]
        if not isinstance(co_scaled, list) or not all(isinstance(n, str) for n in co_scaled):
            raise ValueError("services.yaml: application.co_scaled_services must be a list of service names")
        for n in co_scaled:
            if n not in self.services:
                raise ValueError(f"services.yaml: co-scaled service '{n}' is not a service of "
                                 f"`services` ({names})")
            if n == entry:
                raise ValueError(f"services.yaml: the entry service '{entry}' cannot be co-scaled")
        if len(set(co_scaled)) != len(co_scaled):
            raise ValueError("services.yaml: application.co_scaled_services contains duplicates")

        demand_raw = app_raw.get('demand') or {}
        if not isinstance(demand_raw, dict):
            raise ValueError("services.yaml: application.demand must be a mapping")
        unknown = set(demand_raw) - self._DEMAND_KEYS
        if unknown:
            raise ValueError(f"services.yaml: unknown key(s) in `application.demand`: {sorted(unknown)} "
                             f"(allowed: {sorted(self._DEMAND_KEYS)})")
        sources = demand_raw.get('source_workloads') or []
        if not isinstance(sources, list) or not all(
                isinstance(s, str) and K8S_NAME_RE.match(s) for s in sources):
            raise ValueError("services.yaml: application.demand.source_workloads must be a list of "
                             "Kubernetes workload names (lowercase letters, digits and '-')")
        if len(set(sources)) != len(sources):
            raise ValueError("services.yaml: application.demand.source_workloads contains duplicates")
        fallback = demand_raw.get('server_side_fallback', True)
        if not isinstance(fallback, bool):
            raise ValueError("services.yaml: application.demand.server_side_fallback must be true or false")

        for n in [entry] + co_scaled:
            svc = self.services[n]
            if not K8S_NAME_RE.match(svc.deployment_name) or not K8S_NAME_RE.match(svc.namespace):
                raise ValueError(f"services.yaml: service '{n}' has an invalid deployment_name or "
                                 f"namespace ({svc.deployment_name!r}, {svc.namespace!r})")

        return ApplicationConfig(
            entry_service=entry,
            co_scaled_services=list(co_scaled),
            namespace=default_namespace,
            demand=DemandConfig(source_workloads=list(sources), server_side_fallback=fallback),
        )

    def entry_service_config(self) -> ServiceConfig:
        """ServiceConfig of the entry service (application.entry_service)."""
        return self.services[self.application.entry_service]

    def ingress_query_kwargs(self) -> Dict[str, Any]:
        """Arguments of PrometheusClient.get_ingress_rate for this application."""
        entry = self.entry_service_config()
        demand = self.application.demand
        return {
            'namespace': entry.namespace,
            'entry_workload': entry.deployment_name,
            'source_workloads': list(demand.source_workloads),
            'server_side_fallback': demand.server_side_fallback,
        }

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