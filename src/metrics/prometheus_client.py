"""
Prometheus client for querying metrics
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger("PrometheusClient")


class PrometheusClient:
    """
    Client for querying Prometheus metrics
    """
    
    def __init__(self, url: str, timeout: int = 5):
        """
        Initialize Prometheus client
        
        Args:
            url: Prometheus server URL (e.g., http://192.168.1.245:30090)
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip('/')
        self.timeout = timeout
        
        # Test connection
        if not self._test_connection():
            logger.warning(f"Prometheus not reachable at {self.url}")
    
    def _test_connection(self) -> bool:
        """Test connection to Prometheus"""
        try:
            r = requests.get(f"{self.url}/-/healthy", timeout=self.timeout)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"Prometheus connection failed: {e}")
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
            # Restituiamo un dizionario che ha SEMPRE la chiave 'result'
            # Così chi chiama questa funzione può fare .get('result') senza crashare
            result_data = data.get('data', {})
            
            # Se result_data è già una lista (caso strano di alcune API), lo avvolgiamo
            if isinstance(result_data, list):
                return {'result': result_data}
            
            # Se è un dict (standard), ritorniamo quello
            return result_data
            
        except Exception as e:
            logger.error(f"Prometheus query exception: {e}")
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
        """Get available CPU cores"""
        query = 'sum(kube_node_status_capacity{resource="cpu"})'
        if cluster_label:
            query = f'sum(kube_node_status_capacity{{resource="cpu", cluster="{cluster_label}"}})'
        
        query += ' - sum(kube_pod_container_resource_requests{resource="cpu"})'
        
        result = self.query(query)
        # Ora result è sempre un dizionario grazie al fix in query()
        if result and result.get('result'):
            return float(result['result'][0]['value'][1])
        return 0.0 # Default a 0 se fallisce
    
    def get_memory_available_gb(self, cluster_label: Optional[str] = None) -> Optional[float]:
        """Get available memory in GB"""
        query = 'sum(kube_node_status_capacity{resource="memory"})'
        if cluster_label:
            query = f'sum(kube_node_status_capacity{{resource="memory", cluster="{cluster_label}"}})'
        
        query += ' - sum(kube_pod_container_resource_requests{resource="memory"})'
        
        result = self.query(query)
        if result and result.get('result'):
            memory_bytes = float(result['result'][0]['value'][1])
            return memory_bytes / (1024**3)  # Convert to GB
        return 0.0
    
    def get_request_rate(self, service: str, namespace: str = "default") -> Optional[float]:
        """Get request rate for service"""
        
        # Try 1: Istio metrics
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
                logger.info(f"✅ Traffic from Istio: {rps:.1f} req/s")
                return rps
        except Exception:
            pass # Silent fail
        
        # Try 2: HTTP metrics
        query = f'sum(rate(http_requests_total{{service="{service}",namespace="{namespace}"}}[1m]))'
        
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                rps = float(result['result'][0]['value'][1])
                logger.info(f"✅ Traffic from HTTP: {rps:.1f} req/s")
                return rps
        except Exception:
            pass

        # Try 3: Container network bytes (Fallback)
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
                estimated_rps = bytes_per_sec / 4000
                logger.info(f"✅ Traffic from network: {bytes_per_sec:.0f} bytes/s → {estimated_rps:.1f} req/s")
                return max(0, estimated_rps)
        except Exception as e:
            logger.error(f"Network query exception: {e}")
        
        return 0.0

    def get_latency_p95(self, service: str, namespace: str = "online-boutique") -> Optional[float]:
        """Get p95 latency for a service in milliseconds"""
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
        
        return 0.0 # Default
    
    def get_pod_count(self, deployment: str, namespace: str = "online-boutique") -> int:
        """Get current number of running pods"""
        query = f'count(kube_pod_info{{namespace="{namespace}", created_by_name=~"{deployment}-.*"}})'
        
        result = self.query(query)
        if result and result.get('result'):
            return int(float(result['result'][0]['value'][1]))
        return 0
    
    def get_cpu_usage_percent(self, deployment: str, namespace: str = "online-boutique") -> Optional[float]:
        """Get CPU usage percentage"""
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