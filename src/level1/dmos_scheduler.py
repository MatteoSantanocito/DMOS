"""
DMOS Scheduler (Centralized with Per-Cluster Prometheus)
Computes cluster scores locally using per-cluster Prometheus instances,
then performs winner determination.

Architecture (PROM_MAP, approccio Romano):
- Each cluster has its own Prometheus (ip:30090)
- Scheduler creates one PrometheusClient per cluster
- Each client queries only its cluster's metrics → accurate CPU/memory
- Score computation happens in-process (faster, no network overhead)
"""

import time
import math
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
CPU_HARD_LIMIT_PCT = 95.0       # Esclude cluster if CPU utilization > 95%
MEMORY_HARD_LIMIT_PCT = 90.0    # Esclude cluster if memory utilization > 90%
MIN_CPU_CORES_AVAILABLE = 0.2   # Almeno 0.2 core liberi (per evitare cluster sovraccarichi)
MIN_MEMORY_GB_AVAILABLE = 0.2   # Almeno 0.2 GB liberi (per evitare cluster sovraccarichi)

# ── Two-phase scheduling ──────────────────────────────────────────────────────
# FASE 1 — "Blind allocation" (0 → COLD_START_SECONDS):
#   Usa solo segnali strutturali/freddi: Φ_demand, Φ_net, Φ_carbon.
#   Φ_response_time, Φ_cap, Φ_load disabilitati (ω=0): metriche non affidabili
#   a freddo (pod appena avviati, pochi campioni Hubble nella finestra [5m]).
#   Level 2: usa ingress rate Cilium Ingress come proxy del traffico (Hubble non ha dati).
#
# FASE 2 — "Dynamic rescheduling" (COLD_START_SECONDS → ∞):
#   Score completo con tutti i segnali osservati.
#   Level 2: usa Hubble destination_workload (carico effettivo sui pod).
#
# Riferimenti: FIRM (OSDI'20), Sinan (ASPLOS'21), Cilantro (OSDI'23).
COLD_START_SECONDS = 120  # durata Phase 1 in secondi

class DMOSScheduler:
    
    def __init__(self, config_path: str = "config"):
        self.config = ConfigLoader(config_path)
        
        # Winner determination algorithm
        self.winner_det = WinnerDetermination()
        
        # ── PROM_MAP: per-cluster Prometheus (approccio Romano) ──────
        self.prom_map = {}
        self.cluster_configs = self.config.get_all_clusters()
        
        for name, cfg in self.cluster_configs.items():
            prom_url = f"http://{cfg.ip}:30090"
            self.prom_map[name] = PrometheusClient(
                url=prom_url,
                timeout=5,
                cluster_name=name
            )
        for prom in self.prom_map.values():
            prom._locust_available = False
        
        # Carbon client (condiviso, non cluster-specifico)
        self.carbon_client = CarbonClient(
            self.config.carbon_raw['carbon_intensity']
        )
        
        # Network parameters (geo-awareness via ping_exporter)
        net_params = self.config.network_params
        self._network_rho = net_params.get('rho', 2.0)
        self._network_rtt_max_ms = net_params.get('rtt_max_ms', 500.0)
        self._network_fallback_rtt_ms = net_params.get('fallback_rtt_ms', 5.0)

        # Mappa cluster → IP nodi peer (per get_network_rtt_ms())
        # cluster1 interroga [cluster2_ip, cluster3_ip], ecc.
        all_ips = {n: cfg.ip for n, cfg in self.cluster_configs.items()}
        self._peer_ips: Dict[str, list] = {
            name: [ip for n, ip in all_ips.items() if n != name]
            for name in self.cluster_configs
        }

        # ── Two-phase scheduling ──────────────────────────────────────
        # Timestamp di avvio per determinare Phase 1 vs Phase 2.
        self.scheduler_start_time = time.time()

        score_params = ScoreParameters(
            rho=self._network_rho,
            rtt_max_ms=self._network_rtt_max_ms,
        )

        # Phase 2: score completo (pesi del profilo attivo in weights.yaml)
        self.score_func_warm = ScoreFunctions(
            weights={
                'omega_latency': self.config.score_weights.omega_latency,
                'omega_capacity': self.config.score_weights.omega_capacity,
                'omega_load': self.config.score_weights.omega_load,
                'omega_carbon': self.config.score_weights.omega_carbon,
                'omega_network': self.config.score_weights.omega_network,
                'omega_demand': self.config.score_weights.omega_demand,
            },
            parameters=score_params,
        )

        # Phase 1: solo segnali strutturali/freddi (profilo cold_start)
        cs = self.config.cold_start_weights
        self.score_func_cold = ScoreFunctions(
            weights={
                'omega_latency': cs.omega_latency,   # 0.00
                'omega_capacity': cs.omega_capacity,  # 0.00
                'omega_load': cs.omega_load,          # 0.00
                'omega_carbon': cs.omega_carbon,      # 0.30
                'omega_network': cs.omega_network,    # 0.35
                'omega_demand': cs.omega_demand,      # 0.35
            },
            parameters=score_params,
        )

        # Alias per compatibilità con codice esistente che usa self.score_func
        self.score_func = self.score_func_warm

        logger.info(f"DMOS Scheduler inizializato con:"
                    f"{len(self.cluster_configs)} clusters")
        for name, cfg in self.cluster_configs.items():
            logger.info(f"  {name}: {cfg.region} ({cfg.ip})")
        logger.info(f"Two-phase scheduling attivo: "
                    f"Phase 1 (blind) = {COLD_START_SECONDS}s, "
                    f"Phase 2 (dynamic rescheduling) dopo")
    
    # ── Two-phase helpers ─────────────────────────────────────────────────────

    def _is_cold_start(self) -> bool:
        """Restituisce True se siamo ancora in Phase 1 (blind allocation)."""
        return (time.time() - self.scheduler_start_time) < COLD_START_SECONDS

    def _get_active_score_func(self) -> ScoreFunctions:
        """
        Restituisce la ScoreFunctions corretta in base alla fase corrente:
          Phase 1 → score_func_cold (ω_lat=ω_cap=ω_load=0, solo strutturali)
          Phase 2 → score_func_warm (score completo)
        """
        return self.score_func_cold if self._is_cold_start() else self.score_func_warm

    def _collect_cluster_metrics(
        self,
        cluster_name: str,
        service_name: str = "frontend",
        ingress_rate_rps: float = 0.0,
        ingress_demand_share: float = 0.0,
        cold_start_mode: bool = False,
    ) -> Optional[ClusterMetrics]:
        """
        Raccoglie le metriche del cluster.

        cold_start_mode=True (Phase 1):
          - request_rate_current ← ingress_rate_rps (proxy Nginx, Hubble non ha dati)
          - latency_mean/variance ← valori baseline da config (Hubble p95 non affidabile)

        cold_start_mode=False (Phase 2):
          - request_rate_current ← Hubble destination_workload (carico effettivo pod)
          - latency_mean/variance ← Hubble p95 histogram [5m] (campioni sufficienti)
        """
        
        cluster_cfg = self.cluster_configs.get(cluster_name)
        if not cluster_cfg:
            logger.error(f"Unknown cluster: {cluster_name}")
            return None
        
        # ── Ottieni il client Prometheus specifico per il cluster ────────────────
        prom = self.prom_map.get(cluster_name)
        if not prom:
            logger.error(f"Non riesco a connettermi al prometheus per il cluster: {cluster_name}")
            return None
        
        try:
            svc_cfg = self.config.get_service(service_name)
            namespace = svc_cfg.namespace if svc_cfg else "online-boutique"
            capacity_per_core = svc_cfg.capacity_req_per_sec if svc_cfg else 50
            peer_ips = self._peer_ips.get(cluster_name, [])
            carbon_region = cluster_cfg.carbon.get('region_code', 'DE')

            # ── Parallel Prometheus queries ───────────────────────────
            # Le query CPU, memoria, traffico, latenza e RTT vengono lanciate
            # in parallelo sullo stesso Prometheus per ridurre il tempo di
            # raccolta da ~5s (sequenziale) a ~1s (bottleneck = query più lenta).
            with ThreadPoolExecutor(max_workers=5) as pool:
                fut_cpu    = pool.submit(prom.get_cpu_available)
                fut_mem    = pool.submit(prom.get_memory_available_gb)
                fut_rtt    = pool.submit(prom.get_network_rtt_ms, peer_ips)
                fut_carbon = pool.submit(
                    self.carbon_client.get_carbon_intensity, carbon_region
                )
                # In Phase 1 request_rate e latency vengono dai valori di config,
                # non da Hubble → non serve fare query.
                if cold_start_mode:
                    fut_rate    = None
                    fut_latency = None
                else:
                    fut_rate    = pool.submit(
                        prom.get_request_rate, service_name, namespace
                    )
                    fut_latency = pool.submit(
                        prom.get_latency_p95, service_name, namespace
                    )

            # ── Raccolta risultati ────────────────────────────────────
            cpu_available  = fut_cpu.result()    or 0.0
            cpu_total      = cluster_cfg.cpu_cores
            memory_available = fut_mem.result()  or 0.0
            memory_total   = cluster_cfg.memory_gb
            network_rtt_ms = fut_rtt.result()
            carbon_intensity = fut_carbon.result() or 300.0
            request_rate_max = cpu_total * capacity_per_core

            # ── Traffic metrics ───────────────────────────────────────
            if cold_start_mode:
                # Phase 1: ingress rate Cilium Ingress come proxy del traffico sui pod.
                # Hubble non ha campioni sufficienti in questa fase.
                request_rate = ingress_rate_rps
                logger.debug(
                    f"[Phase 1] {cluster_name}: request_rate={request_rate:.1f} req/s "
                    f"(proxy Cilium Ingress, Hubble non disponibile)"
                )
            else:
                # Phase 2: Hubble destination_workload → carico effettivo sui pod.
                request_rate = fut_rate.result()
                if request_rate is None:
                    # Fallback: stima da CPU usage
                    cpu_usage_pct = prom.get_cpu_usage_percent(
                        deployment=service_name,
                        namespace=namespace
                    )
                    if cpu_usage_pct is not None and cpu_usage_pct > 0:
                        request_rate = (cpu_usage_pct / 100.0) * capacity_per_core
                    else:
                        request_rate = 0.0
                        logger.warning(
                            f"Nessuna metrica di traffico per {cluster_name}, setto default a 0"
                        )

            # ── Latency metrics ───────────────────────────────────────
            if cold_start_mode:
                # Phase 1: usa valori baseline da config.
                # Φ_response_time ha ω=0 in cold_start, ma popolare i campi
                # con valori plausibili evita divisioni per zero e log fuorvianti.
                latency_mean = cluster_cfg.baseline_latency_ms
                latency_variance = cluster_cfg.latency.get('variance_ms', 15.0) ** 2
                logger.debug(
                    f"[Phase 1] {cluster_name}: latency baseline "
                    f"{latency_mean:.1f}ms (Hubble p95 non affidabile)"
                )
            else:
                # Phase 2: Hubble p95 histogram su [1m] → abbastanza campioni.
                latency_p95 = fut_latency.result()
                # Guard contro +Inf (histogram_quantile lo ritorna quando tutto il
                # traffico cade nell'ultimo bucket) e NaN.
                # math.isfinite() cattura entrambi i casi.
                if latency_p95 is not None and latency_p95 > 0 and math.isfinite(latency_p95):
                    # Usa P95 direttamente come stima della latenza media.
                    # Il vecchio approccio (latency_mean = p95/1.65) derivava mean e
                    # variance da un singolo quantile assumendo distribuzione normale
                    # centrata sullo 0, producendo variance ≈ (0.239×p95)^2.
                    # Con sigma_squared=100 anche P95=100ms → variance=573 →
                    # exp(-573/100)≈0 → phi_lat≈0 per tutti i cluster.
                    # Fix: usa P95 come proxy della latenza media (stima conservativa)
                    # e imposta variance=0 → term2=1 sempre → differenziazione
                    # affidata interamente a term1 = 1/(1+η×P95).
                    latency_mean = latency_p95
                    latency_variance = 0.0
                    logger.debug(
                        f"[Phase 2] {cluster_name}: latency_p95={latency_p95:.1f}ms "
                        f"→ latency_mean={latency_mean:.1f}ms (Hubble)"
                    )
                else:
                    latency_mean = cluster_cfg.baseline_latency_ms
                    latency_variance = 0.0
                    logger.debug(
                        f"[Phase 2] {cluster_name}: Hubble p95 non disponibile, "
                        f"fallback baseline={latency_mean:.1f}ms"
                    )

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
                cost_per_replica_hour=cost_per_replica,
                network_rtt_ms=network_rtt_ms,
                ingress_rate_rps=ingress_rate_rps,
                ingress_demand_share=ingress_demand_share,
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Errore durante la raccolta delle metriche per {cluster_name}: {e}")
            return None
    
    def _compute_cluster_score(
        self,
        cluster_name: str,
        service_name: str = "frontend",
        predicted_load: Optional[float] = None,
        ingress_rate_rps: float = 0.0,
        ingress_demand_share: float = 0.0,
        cold_start_mode: bool = False,
        active_score_func: Optional[ScoreFunctions] = None,
    ) -> Optional[Dict]:

        metrics = self._collect_cluster_metrics(
            cluster_name, service_name,
            ingress_rate_rps=ingress_rate_rps,
            ingress_demand_share=ingress_demand_share,
            cold_start_mode=cold_start_mode,
        )
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
            logger.warning(f" {cluster_name} Escluso perché: {exclusion_reason}")
            return {
                'cluster_name': cluster_name,
                'score': 0.0,
                'score_breakdown': {
                    'phi_latency': 0.0, 'phi_capacity': 0.0,
                    'phi_load': 0.0, 'phi_carbon': 0.0,
                    'phi_network': 0.0, 'phi_demand': 0.0,
                },
                'capacity': 0,
                'eligible': False,
                'exclusion_reason': exclusion_reason,
                'metrics': {
                    'cpu_utilization_pct': cpu_util_pct,
                    'mem_utilization_pct': mem_util_pct,
                    'carbon_intensity_gco2_kwh': metrics.carbon_intensity_gco2_kwh,
                    'latency_mean_ms': metrics.latency_mean_ms,
                    'network_rtt_ms': metrics.network_rtt_ms,
                    'ingress_rate_rps': metrics.ingress_rate_rps,
                }
            }
        
        # ── Score computation ─────────────────────────────────────────
        score_func = active_score_func or self.score_func_warm
        breakdown = score_func.compute_score_breakdown(
            metrics, predicted_load=predicted_load
        )
        
        # ── Capacity calculation ──────────────────────────────────────
        # Usa max_replicas dal config come limite superiore della bid.capacity.
        # Il check di esclusione hard (CPU_HARD_LIMIT_PCT/MEMORY_HARD_LIMIT_PCT)
        # sopra rimuove già i cluster davvero saturi prima di arrivare qui.
        # Usare capacity_cpu/capacity_mem come hard cap causa concentrazione
        # eccessiva su un singolo cluster quando i nodi hanno poca CPU libera
        # (es. 0.2 core → capacity=2) anche se il cluster è ben sotto il 95%.
        # La penalizzazione del carico avviene già tramite Φ_cap e Φ_load nello
        # score: un cluster carico ottiene score più basso → meno repliche.
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

            # Calcola capacity_cpu/mem solo per il log (debug), non come hard cap
            capacity_cpu = int(metrics.cpu_available_cores / cpu_req) if cpu_req > 0 else 0
            capacity_mem = int(metrics.memory_available_gb / mem_req_gb) if mem_req_gb > 0 else 0
            capacity = min(capacity_cpu, capacity_mem)
            max_replicas = service_config.max_replicas
        else: 
            capacity = int(metrics.cpu_available_cores * 2)
            max_replicas = 20
            
        capacity = min(max(0,capacity), max_replicas)
        
        logger.info(
            f"Score {cluster_name}: {breakdown['total_score']:.3f} "
            f"(lat={breakdown['phi_latency']:.3f}, cap={breakdown['phi_capacity']:.3f}, "
            f"load={breakdown['phi_load']:.3f}, carbon={breakdown['phi_carbon']:.3f}, "
            f"net={breakdown['phi_network']:.3f}|RTT={metrics.network_rtt_ms:.0f}ms, "
            f"demand={breakdown['phi_demand']:.3f}|{metrics.ingress_rate_rps:.1f}rps) "
            f"capacity={capacity}, cpu={cpu_util_pct:.0f}%, "
            f"CI={metrics.carbon_intensity_gco2_kwh:.0f}gCO2"
        )

        return {
            'cluster_name': cluster_name,
            'score': breakdown['total_score'],
            'score_breakdown': {
                'phi_latency': breakdown['phi_latency'],
                'phi_capacity': breakdown['phi_capacity'],
                'phi_load': breakdown['phi_load'],
                'phi_carbon': breakdown['phi_carbon'],
                'phi_network': breakdown['phi_network'],
                'phi_demand': breakdown['phi_demand'],
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
                'latency_mean_ms': metrics.latency_mean_ms,
                'network_rtt_ms': metrics.network_rtt_ms,
                'ingress_rate_rps': metrics.ingress_rate_rps,
                'ingress_demand_share': metrics.ingress_demand_share,
            }
        }
    
    def collect_scores(
        self,
        service_name: str,
        predicted_load: Optional[float] = None
    ) -> List[ClusterBid]:

        # ── Determinazione fase corrente ───────────────────────────────────────
        cold_start = self._is_cold_start()
        elapsed = time.time() - self.scheduler_start_time
        active_score_func = self._get_active_score_func()

        if cold_start:
            logger.info(
                f"[Phase 1 — BLIND ALLOCATION] t={elapsed:.0f}s < {COLD_START_SECONDS}s | "
                f"score: Φ_demand + Φ_net + Φ_carbon | "
                f"Level 2: ingress rate Cilium Ingress come proxy traffico"
            )
        else:
            logger.info(
                f"[Phase 2 — DYNAMIC RESCHEDULING] t={elapsed:.0f}s | "
                f"score completo: Φ_resp + Φ_cap + Φ_load + Φ_carbon + Φ_net + Φ_demand | "
                f"Level 2: Hubble destination_workload"
            )

        logger.info(f"Calcolo score per '{service_name}' su tutti i cluster...")

        # ── Step 0: Pre-raccolta ingress rates per Φ_demand ────────────────────
        # Φ_demand(i) = ingress_rate_i / Σ_j ingress_rate_j
        # Richiede il totale su tutti i cluster → raccogliamo prima del loop score.
        # In Phase 1 questa metrica serve anche come proxy per Level 2 (request_rate).
        svc_cfg = self.config.get_service(service_name)
        namespace = svc_cfg.namespace if svc_cfg else "online-boutique"
        ingress_rates: Dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=len(self.prom_map)) as executor:
            futures_ingress = {
                executor.submit(prom.get_ingress_rate, namespace): cluster_name
                for cluster_name, prom in self.prom_map.items()
            }
            for future in as_completed(futures_ingress):
                ingress_rates[futures_ingress[future]] = future.result()

        total_ingress = sum(ingress_rates.values())
        n = len(self.prom_map) or 1
        if total_ingress > 0:
            demand_shares = {
                name: rate / total_ingress
                for name, rate in ingress_rates.items()
            }
            logger.info(
                "Ingress rates: "
                + ", ".join(f"{c}={r:.1f} req/s" for c, r in ingress_rates.items())
                + f" | total={total_ingress:.1f} req/s"
            )
        else:
            demand_shares = {name: 1.0 / n for name in ingress_rates}
            logger.debug(
                "hubble_http_requests_total[ingress] non disponibile, "
                "Φ_demand uniforme (1/N per cluster)"
            )

        bids = []
        excluded = []

        # Calcola score per ogni cluster in parallelo
        with ThreadPoolExecutor(max_workers=len(self.cluster_configs)) as executor:
            futures_scores = {
                executor.submit(
                    self._compute_cluster_score,
                    cluster_name, service_name, predicted_load,
                    ingress_rates.get(cluster_name, 0.0),
                    demand_shares.get(cluster_name, 1.0 / n),
                    cold_start,
                    active_score_func,
                ): cluster_name
                for cluster_name in self.cluster_configs.keys()
            }
            for future in as_completed(futures_scores):
                cluster_name = futures_scores[future]
                result = future.result()

                if result is None:
                    logger.warning(f"Nessun risultato per  {cluster_name}")
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
            logger.warning(f"Cluster esclusi: {excluded}")
        
        logger.info(f"Raccolte {len(bids)}/{len(self.cluster_configs)} offerte "
                    f"({len(excluded)} esclusi)")
        
        return bids
    
    def schedule_service(
        self,
        service_name: str,
        total_replicas: int,
        predicted_load: Optional[float] = None
    ) -> Tuple[List[Allocation], bool]:

        logger.info(f"Scheduling '{service_name}' con {total_replicas} repliche")
        
        # Step 1: Raccolta offerte (scores + capacity) da tutti i cluster
        start_time = time.time()
        bids = self.collect_scores(service_name, predicted_load)
        collection_time = (time.time() - start_time) * 1000
        
        if not bids:
            logger.error("Nessun cluster idoneo disponibile!")
            return [], False
        
        # Step 2: Winner determination
        allocations, success = self.winner_det.allocate(bids, total_replicas)
        
        # Step 3: Log results
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Scheduling completato in {total_time:.0f}ms "
                    f"(raccolta={collection_time:.0f}ms, "
                    f"winner_det={(total_time - collection_time):.0f}ms)")
        
        if success:
            logger.info(f"Allocate {total_replicas} repliche su "
                       f"{len(allocations)} cluster:")
            for alloc in allocations:
                logger.info(f"   {alloc}")
            
            jain = self.winner_det.compute_fairness_jain_index(allocations)
            logger.info(f"   Indice di fairness di Jain: {jain:.3f}")
        else:
            logger.error(f"  Impossibile soddisfare la domanda ({total_replicas} repliche)")
        
        return allocations, success