"""
Prometheus client for querying metrics — Multi-Cluster Edition

Approccio PROM_MAP (come tesi Romano):
  Ogni cluster ha il proprio Prometheus locale che vede solo i suoi pod.
  DMOS interroga ciascun Prometheus separatamente per ottenere metriche
  accurate per cluster (CPU, memoria, traffico, latenza).

Configurazione via environment variables:
  PROM_CLUSTER1=http://192.168.1.245:30090
  PROM_CLUSTER2=http://192.168.1.246:30090
  PROM_CLUSTER3=http://192.168.1.247:30090

Oppure automaticamente da clusters.yaml (ip:30090 per ogni cluster).
"""

import os
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger("PrometheusClient")


class PrometheusClient:
    """
    Client for querying Prometheus metrics.
    
    Ogni istanza è associata a UN Prometheus (di un cluster specifico).
    Per multi-cluster, creare un'istanza per cluster tramite build_prom_map().
    """
    
    def __init__(self, url: str, timeout: int = 5, cluster_name: str = None):
        """
        Initialize Prometheus client
        
        Args:
            url: Prometheus server URL (e.g., http://192.168.1.245:30090)
            timeout: Request timeout in seconds
            cluster_name: Nome del cluster associato (per logging)
        """
        self.url = url.rstrip('/')
        self.timeout = timeout
        self.cluster_name = cluster_name or "unknown"
        
        # Locust URL per traffico reale (durante i test)
        self.locust_url = os.environ.get("DMOS_LOCUST_URL", "http://localhost:8089")
        self._locust_available = None  # None=non testato, True/False
        
        # Test connection
        if not self._test_connection():
            logger.warning(f"Prometheus not reachable at {self.url} (cluster: {self.cluster_name})")
        else:
            logger.info(f"✅ Prometheus connected: {self.url} (cluster: {self.cluster_name})")
    
    def _test_connection(self) -> bool:
        """Test connection to Prometheus"""
        try:
            r = requests.get(f"{self.url}/-/healthy", timeout=self.timeout)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Prometheus connection failed ({self.cluster_name}): {e}")
            return False
    
    def query(self, query_str: str) -> Optional[Dict]:
        """
        Execute Prometheus query
        
        Args:
            query_str: PromQL query
        
        Returns:
            Dictionary containing 'result' list or None
        """
        try:
            response = requests.get(
                f"{self.url}/api/v1/query",
                params={'query': query_str},
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Prometheus HTTP error: {response.status_code}")
                return None
            
            data = response.json()
            
            if data.get('status') != 'success':
                logger.error(f"Prometheus query error: {data.get('error', 'unknown')}")
                return None
            
            # --- FIX CRITICO: Normalizzazione dell'output ---
            result_data = data.get('data', {})
            
            if isinstance(result_data, list):
                return {'result': result_data}
            
            return result_data
            
        except Exception as e:
            logger.error(f"Prometheus query exception ({self.cluster_name}): {e}")
            return None

    def query_range(
        self, 
        query: str, 
        start: datetime, 
        end: datetime, 
        step: str = '30s'
    ) -> Optional[Dict[str, Any]]:
        """Execute range query"""
        params = {
            'query': query,
            'start': start.timestamp(),
            'end': end.timestamp(),
            'step': step
        }
        
        try:
            r = requests.get(
                f"{self.url}/api/v1/query_range",
                params=params,
                timeout=self.timeout
            )
            r.raise_for_status()
            
            data = r.json()
            if data['status'] == 'success':
                return data['data']
            else:
                logger.error(f"Range query failed: {data.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"Range query error: {e}")
            return None
    
    def get_cpu_available(self, cluster_label: Optional[str] = None) -> Optional[float]:
        """
        Get available CPU cores.
        
        Con PROM_MAP: ogni client interroga il suo Prometheus locale,
        quindi non serve filtrare per cluster label.
        """
        capacity_query = 'sum(kube_node_status_capacity{resource="cpu"})'
        usage_query = 'sum(rate(container_cpu_usage_seconds_total{image!=""}[2m]))'
        
        capacity_result = self.query(capacity_query)
        usage_result = self.query(usage_query)
        
        cpu_capacity = 0.0
        cpu_usage = 0.0
        
        if capacity_result and capacity_result.get('result'):
            cpu_capacity = float(capacity_result['result'][0]['value'][1])
        
        if usage_result and usage_result.get('result'):
            cpu_usage = float(usage_result['result'][0]['value'][1])
        
        available = cpu_capacity - cpu_usage
        logger.debug(f"CPU {self.cluster_name}: capacità={cpu_capacity:.1f}, "
                     f"USATA={cpu_usage:.2f}, disponibile={available:.2f}")
        
        return max(0.0, available)
    
    def get_memory_available_gb(self, cluster_label: Optional[str] = None) -> Optional[float]:
        """Get available memory in GB"""
        capacity_query = 'sum(kube_node_status_capacity{resource="memory"})'
        usage_query = 'sum(container_memory_working_set_bytes{image!=""})'
        
        capacity_result = self.query(capacity_query)
        usage_result = self.query(usage_query)
        
        mem_capacity = 0.0
        mem_usage = 0.0
        
        if capacity_result and capacity_result.get('result'):
            mem_capacity = float(capacity_result['result'][0]['value'][1])
        
        if usage_result and usage_result.get('result'):
            mem_usage = float(usage_result['result'][0]['value'][1])
        
        available_bytes = mem_capacity - mem_usage
        return max(0.0, available_bytes / (1024**3))
    
    def get_request_rate(self, service: str, namespace: str = "default") -> float:
        """
        Get request rate for service using multi-source fallback chain.
        
        Priority:
          Try 0: Locust API (real measured throughput during load tests)
          Try 1: Hubble HTTP metrics (Cilium eBPF, per-cluster)
          Try 2: Istio metrics
          Try 3: HTTP generic metrics
          Try 4: Container network bytes (per-cluster, calibrated)
        
        Returns:
          Request rate in req/s, or 0.0 if no source available
        """
        
        # ── Try 0: Locust API (traffico reale misurato) ─────────────────
        # REGOLA: se Locust è raggiungibile, SEMPRE usare il suo dato.
        if self._locust_available is not False:
            try:
                resp = requests.get(
                    f"{self.locust_url}/stats/requests",
                    timeout=2
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._locust_available = True
                    
                    total_rps = data.get("total_rps", 0)
                    
                    if not total_rps:
                        for stat in data.get("stats", []):
                            if stat.get("name") == "Aggregated":
                                total_rps = stat.get("current_rps", 0)
                                break
                    
                    logger.info(f"✅ Traffic from Locust ({self.cluster_name}): {total_rps:.1f} req/s")
                    return float(total_rps)
                        
            except requests.exceptions.ConnectionError:
                self._locust_available = False
                logger.info(f"ℹ️ Locust non raggiungibile, uso fallback Prometheus ({self.cluster_name})")
            except Exception as e:
                logger.debug(f"Locust query failed: {e}")
        
        # ── Try 1: Hubble HTTP metrics (Cilium eBPF) ───────────────────
        query = f'''
        sum(rate(hubble_http_requests_total{{
            source="reserved:ingress"
        }}[1m]))
        '''
        
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                rps = float(result['result'][0]['value'][1])
                if rps > 0:
                    logger.info(f"✅ Traffic from Hubble ({self.cluster_name}): {rps:.1f} req/s")
                    return rps
        except Exception:
            pass
        
        # ── Try 2: Istio metrics ────────────────────────────────────────
        query = f'''
        sum(rate(istio_requests_total{{
            destination_service_name="{service}",
            destination_service_namespace="{namespace}"
        }}[1m]))
        '''
        
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                rps = float(result['result'][0]['value'][1])
                if rps > 0:
                    logger.info(f"✅ Traffic from Istio ({self.cluster_name}): {rps:.1f} req/s")
                    return rps
        except Exception:
            pass
        
        # ── Try 3: HTTP generic metrics ─────────────────────────────────
        query = f'sum(rate(http_requests_total{{service="{service}",namespace="{namespace}"}}[1m]))'
        
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                rps = float(result['result'][0]['value'][1])
                if rps > 0:
                    logger.info(f"✅ Traffic from HTTP ({self.cluster_name}): {rps:.1f} req/s")
                    return rps
        except Exception:
            pass

        # ── Try 4: Container network bytes (per-cluster) ────────────────
        # Con PROM_MAP ogni Prometheus vede solo i pod del suo cluster,
        # quindi il network bytes riflette il traffico locale.
        query = f'''
        sum(rate(container_network_receive_bytes_total{{
            namespace="{namespace}",
            pod=~"{service}.*"
        }}[1m]))
        '''
        
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                bytes_per_sec = float(result['result'][0]['value'][1])
                if bytes_per_sec > 0:
                    estimated_rps = bytes_per_sec / 4000
                    logger.info(
                        f"⚠️ Traffic from network ({self.cluster_name}): "
                        f"{bytes_per_sec:.0f} B/s → {estimated_rps:.1f} req/s"
                    )
                    return max(0, estimated_rps)
        except Exception as e:
            logger.error(f"Network query exception: {e}")
        
        return 0.0

    def get_latency_p95(self, service: str, namespace: str = "online-boutique") -> Optional[float]:
        """Get p95 latency for a service in milliseconds"""
        # Try Hubble first
        query = f'''
        histogram_quantile(0.95,
          sum(rate(hubble_http_request_duration_seconds_bucket{{
            destination=~"{service}.*"
          }}[5m])) by (le)
        ) * 1000
        '''
        
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                val = float(result['result'][0]['value'][1])
                if val > 0:
                    return val
        except Exception:
            pass
        
        # Fallback: Istio
        query = f'''
        histogram_quantile(0.95, 
          sum(rate(istio_request_duration_milliseconds_bucket{{
            destination_service_name="{service}",
            destination_service_namespace="{namespace}"
          }}[5m])) by (le)
        )
        '''
        
        result = self.query(query)
        if result and result.get('result'):
            return float(result['result'][0]['value'][1])
        
        return 0.0
    
    def get_pod_count(self, deployment: str, namespace: str = "online-boutique") -> int:
        """Get current number of running pods on THIS cluster"""
        query = f'count(kube_pod_info{{namespace="{namespace}", created_by_name=~"{deployment}-.*"}})'
        
        result = self.query(query)
        if result and result.get('result'):
            return int(float(result['result'][0]['value'][1]))
        return 0
    
    def get_cpu_usage_percent(self, deployment: str, namespace: str = "online-boutique") -> Optional[float]:
        """Get CPU usage percentage on THIS cluster"""
        query = f'''
        100 * sum(rate(container_cpu_usage_seconds_total{{
          namespace="{namespace}",
          pod=~"{deployment}-.*"
        }}[5m])) 
        / 
        sum(kube_pod_container_resource_requests{{
          namespace="{namespace}",
          pod=~"{deployment}-.*",
          resource="cpu"
        }})
        '''
        
        result = self.query(query)
        if result and result.get('result'):
            return float(result['result'][0]['value'][1])
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PROM_MAP: Factory per creare client per-cluster
# ═══════════════════════════════════════════════════════════════════════════════

def build_prom_map(clusters_config: list = None) -> Dict[str, 'PrometheusClient']:
    """
    Build a map of PrometheusClient instances, one per cluster.
    
    Approccio Romano: ogni cluster ha il suo Prometheus locale.
    
    Configurazione:
      1. Da variabili d'ambiente: PROM_CLUSTER1=http://..., PROM_CLUSTER2=...
      2. Da config clusters (fallback): usa ip:30090 per ogni cluster
    
    Returns:
        Dict[cluster_name, PrometheusClient]
    """
    prom_map = {}
    
    # Metodo 1: Variabili d'ambiente (come Romano)
    for key, value in os.environ.items():
        if key.startswith("PROM_"):
            cluster_name = key[5:].lower()  # PROM_CLUSTER1 → cluster1
            prom_map[cluster_name] = PrometheusClient(
                url=value,
                cluster_name=cluster_name
            )
            logger.info(f"PROM_MAP: {cluster_name} → {value}")
    
    # Metodo 2: Da configurazione clusters (fallback)
    if not prom_map and clusters_config:
        for cluster in clusters_config:
            name = cluster.get('name', cluster.get('cluster_name', 'unknown'))
            ip = cluster.get('ip', 'localhost')
            url = f"http://{ip}:30090"
            prom_map[name] = PrometheusClient(
                url=url,
                cluster_name=name
            )
            logger.info(f"PROM_MAP (from config): {name} → {url}")
    
    if not prom_map:
        logger.warning("PROM_MAP vuota! Nessun Prometheus configurato.")
    else:
        logger.info(f"PROM_MAP inizializzata con {len(prom_map)} cluster: {list(prom_map.keys())}")
    
    return prom_map