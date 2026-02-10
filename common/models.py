"""
Modelli dati condivisi per VODA-MS
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

@dataclass
class ClusterResources:
    """Risorse di un cluster"""
    cpu_total: float
    cpu_available: float
    cpu_reserved: float
    memory_total: float  # GB
    memory_available: float
    memory_reserved: float
    
    @property
    def cpu_usable(self) -> float:
        """CPU effettivamente usabile (totale - riservato)"""
        return self.cpu_total - self.cpu_reserved
    
    @property
    def memory_usable(self) -> float:
        """Memory effettivamente usabile"""
        return self.memory_total - self.memory_reserved
    
    @property
    def cpu_utilization(self) -> float:
        """Percentuale utilizzo CPU [0, 1]"""
        usable = self.cpu_usable
        if usable == 0:
            return 1.0
        return (usable - self.cpu_available) / usable
    
    @property
    def memory_utilization(self) -> float:
        """Percentuale utilizzo memoria [0, 1]"""
        usable = self.memory_usable
        if usable == 0:
            return 1.0
        return (usable - self.memory_available) / usable

@dataclass
class LatencyMetrics:
    """Metriche di latenza"""
    mean: float  # ms
    p50: float
    p95: float
    p99: float
    variance: float
    
    @classmethod
    def from_prometheus(cls, mean: float, p50: float, p95: float, p99: float):
        """Calcola varianza da percentili"""
        # Approssimazione: var ≈ (p95 - p50)^2
        variance = (p95 - p50) ** 2
        return cls(
            mean=mean,
            p50=p50,
            p95=p95,
            p99=p99,
            variance=variance
        )

@dataclass
class TrafficMetrics:
    """Metriche di traffico"""
    current_rps: float  # Richieste/secondo correnti
    max_rps: float      # Capacità massima
    
    @property
    def utilization(self) -> float:
        """Percentuale utilizzo traffico [0, 1]"""
        if self.max_rps == 0:
            return 1.0
        return min(1.0, self.current_rps / self.max_rps)

@dataclass
class ClusterMetrics:
    """Metriche correnti di un cluster"""
    cluster_id: int
    timestamp: datetime
    resources: ClusterResources
    traffic: TrafficMetrics
    latency: LatencyMetrics
    pod_count: Dict[str, int] = field(default_factory=dict)  # service -> replicas
    
    def to_dict(self) -> dict:
        """Serializza per JSON"""
        return {
            'cluster_id': self.cluster_id,
            'timestamp': self.timestamp.isoformat(),
            'resources': asdict(self.resources),
            'traffic': asdict(self.traffic),
            'latency': asdict(self.latency),
            'pod_count': self.pod_count
        }

@dataclass
class ClusterInfo:
    """Informazioni statiche cluster"""
    name: str
    id: int
    location: str
    ip: str
    kubeconfig_path: str
    cost_per_replica_hour: float
    max_rps: float
    latency_matrix: Dict[int, float]  # cluster_id -> latency_ms
    agent_host: str
    agent_port: int
    
    # Risorse totali
    cpu_total: float
    cpu_reserved: float
    memory_total: float
    memory_reserved: float
    
    @property
    def agent_url(self) -> str:
        """URL dell'agent REST API"""
        return f"http://{self.agent_host}:{self.agent_port}"

@dataclass
class ScoreComponents:
    """Componenti dello score secondo la tesi"""
    phi_lat: float  # Φ_lat [0, 1]
    phi_cap: float  # Φ_cap [0, 1]
    phi_load: float # Φ_load [0, 1]
    
    def weighted_sum(self, omega_lat: float, omega_cap: float, omega_load: float) -> float:
        """Score totale pesato"""
        return (omega_lat * self.phi_lat + 
                omega_cap * self.phi_cap + 
                omega_load * self.phi_load)

@dataclass
class Bid:
    """Bid di un cluster per un servizio (Asta Vickrey)"""
    cluster_id: int
    cluster_name: str
    service_name: str
    score: float  # Score totale [0, 1]
    score_components: ScoreComponents
    capacity: int  # Max repliche ospitabili
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'service_name': self.service_name,
            'score': self.score,
            'score_components': asdict(self.score_components),
            'capacity': self.capacity,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class AllocationQuota:
    """Quota assegnata da Livello 1 (Asta Vickrey)"""
    cluster_id: int
    cluster_name: str
    service_name: str
    quota: float  # [0, 1] percentuale
    total_demand: int  # Repliche totali richieste
    
    @property
    def estimated_replicas(self) -> int:
        """Stima repliche da questa quota"""
        return max(1, int(self.quota * self.total_demand))
    
    def to_dict(self) -> dict:
        return {
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'service_name': self.service_name,
            'quota': self.quota,
            'total_demand': self.total_demand,
            'estimated_replicas': self.estimated_replicas
        }

@dataclass
class PredictionResult:
    """Risultato predizione traffico (Livello 2)"""
    service_name: str
    current_rps: float
    predicted_rps: float
    trend: float  # derivata (rps/sec)
    delta_t: int  # orizzonte predizione (secondi)
    timestamp: datetime
    
    @property
    def growth_rate(self) -> float:
        """Tasso di crescita percentuale"""
        if self.current_rps == 0:
            return 0.0
        return (self.predicted_rps - self.current_rps) / self.current_rps

@dataclass
class PDControllerState:
    """Stato PD Controller (Livello 2)"""
    service_name: str
    error: float              # Errore corrente
    error_prev: float         # Errore precedente
    error_derivative: float   # Derivata errore
    correction: int           # Repliche da aggiungere/rimuovere
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'service_name': self.service_name,
            'error': self.error,
            'error_derivative': self.error_derivative,
            'correction': self.correction,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ServiceConfig:
    """Configurazione di un servizio"""
    name: str
    namespace: str
    capacity_per_replica: Dict[str, float]  # rps, cpu, memory
    sla: Dict[str, float]  # max_latency_ms, target_availability
    dependencies: List[str] = field(default_factory=list)
    
@dataclass
class ScalingDecision:
    """Decisione finale di scaling"""
    service_name: str
    cluster_id: int
    current_replicas: int
    target_replicas: int
    delta: int
    reason: str  # "prediction", "pd_correction", "quota_change"
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            'service_name': self.service_name,
            'cluster_id': self.cluster_id,
            'current_replicas': self.current_replicas,
            'target_replicas': self.target_replicas,
            'delta': self.delta,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat()
        }