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
import math
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
                timeout=self.timeout
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
        Ogni istanza di PrometheusClient parla con il Prometheus del suo cluster
        (:30090), quindi tutte le metriche restituite sono per-cluster.

        Priority:
          Try 1: Hubble HTTP metrics (Cilium L7, richiede Nginx Ingress + CNP L7)
                 → metrica più precisa: conta richieste HTTP reali
          Try 2: Istio metrics (se Istio installato)
          Try 3: HTTP generic metrics (se l'app le espone)
          Try 4: Container network bytes (fallback sempre disponibile, stima empirica)

        Returns:
          Request rate in req/s per questo cluster, o 0.0 se nessuna fonte disponibile.
        """

        # ── Try 1: Hubble HTTP metrics (Cilium L7 via Cilium Ingress) ────
        # Disponibile dopo: Cilium Ingress Controller attivo + CNP frontend con rules: http.
        # Il traffico arriva da Cilium Ingress Envoy → Hubble conta le richieste HTTP.
        #
        # NOTA [1m]: Hubble-metrics scrape_interval = 15s (configurato in
        # prometheus-helm-values.yaml extraScrapeConfigs job 'hubble').
        # Con 15s interval, rate()[1m] ha ~4 campioni → stabile.
        # Rispetto a [5m] (usato quando scrape era 60s), il lag durante flash crowd
        # scende da ~5min a ~60s → DMOS scala ~4× più veloce al picco.
        #
        # NOTA destination_workload: filtro per servizio specifico, altrimenti
        # tutti i servizi restituirebbero il totale del namespace.
        query = (
            f'sum(rate(hubble_http_requests_total{{'
            f'destination_workload="{service}",'
            f'destination_namespace="{namespace}"'
            f'}}[1m]))'
        )

        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                rps = float(result['result'][0]['value'][1])
                if rps > 0:
                    logger.info(f"✅ Traffic from Hubble ({self.cluster_name}): {rps:.1f} req/s")
                    return rps
                else:
                    logger.debug(
                        f"Hubble: query OK ma rate=0 ({self.cluster_name}, svc={service}) "
                        f"— possibile gap di scrape, fallback a network"
                    )
            else:
                logger.debug(f"Hubble: query vuota ({self.cluster_name}, svc={service})")
        except Exception as e:
            logger.debug(f"Hubble query exception ({self.cluster_name}): {e}")
        
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
        """
        Get p95 latency for a service in milliseconds.
        Filtra per destination_namespace per coerenza con get_request_rate().
        """
        # Try Hubble first (disponibile con Cilium Ingress + CNP L7)
        # [1m] con scrape_interval=15s → ~4 campioni per finestra → stabile
        query = f'''
        histogram_quantile(0.95,
          sum(rate(hubble_http_request_duration_seconds_bucket{{
            destination_namespace="{namespace}"
          }}[1m])) by (le)
        ) * 1000
        '''
        
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                val = float(result['result'][0]['value'][1])
                # Guard contro +Inf: histogram_quantile ritorna +Inf quando tutto il
                # traffico cade nell'ultimo bucket (es. latenza > upper bound del bucket).
                # float('+Inf') > 0 è True → passerebbe il check, poi 1/(1+η×∞)=0.
                if val > 0 and math.isfinite(val):
                    logger.debug(
                        f"Hubble p95 latency ({self.cluster_name}): {val:.1f}ms"
                    )
                    return val
                elif math.isinf(val):
                    logger.warning(
                        f"Hubble p95 latency ({self.cluster_name}): +Inf — "
                        f"tutti i sample nel bucket superiore, fallback a baseline"
                    )
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
    
    def get_network_rtt_ms(self, peer_ips: List[str]) -> float:
        """
        Restituisce la RTT media (ms) verso i peer cluster via ping_exporter.

        ping_exporter (czerwonk/ping_exporter, deploy Helm in namespace observability)
        misura la RTT con ICMP ping verso target configurati (IP nodi degli altri cluster).
        La metrica esposta è `ping_rtt_mean_seconds` con label `target=<IP>`.

        La RTT inter-cluster è usata come proxy della distanza geografica utente-cluster:
        - Cluster con bassa RTT verso i peer → geograficamente centrale → Φ_net alto
        - Cluster con alta RTT verso i peer → geograficamente periferico → Φ_net basso

        Con tc netem attivo:
          cluster1 → cluster2 RTT ≈ 150ms, cluster1 → cluster3 RTT ≈ 350ms
          → cluster1 RTT_avg = (150+350)/2 = 250ms → Φ_net ≈ 0.368

        Args:
            peer_ips: IP nodi degli altri cluster
                      (es. ["192.168.1.246", "192.168.1.247"] per cluster1)

        Returns:
            RTT media in ms, o fallback_rtt (5.0 ms) se ping_exporter non disponibile.
        """
        if not peer_ips:
            return 5.0  # nessun peer → fallback LAN

        # Costruisce regex OR per matchare tutti i target peer in una query
        targets_regex = "|".join(peer_ips)
        query = f'avg(ping_rtt_mean_seconds{{target=~"{targets_regex}"}}) * 1000'

        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                val = float(result['result'][0]['value'][1])
                if val > 0:
                    logger.info(
                        f"✅ Network RTT ({self.cluster_name}): {val:.1f} ms "
                        f"[targets: {targets_regex}]"
                    )
                    return val
                else:
                    logger.debug(
                        f"ping_exporter: RTT=0 ({self.cluster_name}) — "
                        f"ping_exporter non ancora scrappato o target non raggiungibili"
                    )
        except Exception as e:
            logger.debug(f"ping_exporter query exception ({self.cluster_name}): {e}")

        logger.debug(
            f"ping_exporter non disponibile ({self.cluster_name}), "
            f"RTT fallback a 5.0 ms (LAN baseline senza netem)"
        )
        return 5.0  # fallback: latenza LAN senza delay simulato

    def get_ingress_rate(self, namespace: str = "online-boutique") -> float:
        """
        Misura il traffico in ingresso nel cluster via Cilium Ingress Controller (Envoy).

        Con Cilium Ingress:
          Locust → Cilium Ingress Envoy (NodePort 30080) → frontend pod
                            ↑
          Hubble L7 (abilitato dalla CNP con rules: http su fromEntities: ingress)
          esporta hubble_http_requests_total con traffic_direction="ingress".

        Label strategy:
          La configurazione Hubble usa labelsContext=source_namespace,source_workload,...
          Per il traffico da reserved:ingress, source_namespace e source_workload sono
          vuoti (identità riservata). Il filtro corretto è:
            reporter="server"          → il frontend pod è il destinatario
            traffic_direction="ingress"→ traffico in arrivo al frontend
          Questo cattura tutto il traffico inbound al frontend (utenti via ingress +
          health check kubelet), ma la componente esterna è dominante durante il test.
          Poiché Φ_demand usa quote relative (rate_i / Σ_j rate_j), il rumore
          costante dei health check si cancella nella normalizzazione.

        NOTA [5m]: stesso rationale di get_request_rate() — Hubble scrape interval
          15s (scrape_interval configurato nel job 'hubble' di prometheus-helm-values.yaml).
          Con 15s interval, rate()[1m] ha ~4 campioni → stabile e reattivo.
          Lag ridotto da ~5min ([5m] con 60s scrape) a ~60s ([1m] con 15s scrape).

        Uso:
          Level 1 (WHERE): Φ_demand(i) = ingress_rate_i / Σ_j ingress_rate_j
          → cluster con più traffico ingresso = più utenti → più repliche.

        Returns:
            req/s in ingresso per questo cluster, o 0.0 se Hubble L7 non disponibile.
        """
        query = (
            f'sum(rate(hubble_http_requests_total{{'
            f'destination_workload="frontend",'
            f'destination_namespace="{namespace}",'
            f'reporter="server",'
            f'traffic_direction="ingress"'
            f'}}[1m]))'
        )
        try:
            result = self.query(query)
            if result and result.get('result') and len(result['result']) > 0:
                rps = float(result['result'][0]['value'][1])
                if rps > 0:
                    logger.info(
                        f"✅ Ingress rate ({self.cluster_name}): {rps:.1f} req/s "
                        f"[hubble L7 ingress traffic_direction]"
                    )
                    return rps
        except Exception as e:
            logger.debug(f"Hubble ingress query exception ({self.cluster_name}): {e}")

        logger.debug(
            f"hubble L7 ingress non disponibile ({self.cluster_name}), "
            f"Φ_demand userà il fallback uniforme"
        )
        return 0.0

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