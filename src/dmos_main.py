"""
DMOS Event-Driven + Periodic Polling Orchestrator

Changes vs previous version:
  1. Clusters NOT in allocations are now scaled down to min_replicas
  2. All configured services are now monitored and scheduled
  3. [NEW] Anti-oscillation fixes:
     A. debounce_seconds: 15 → 30
     B. max_delta_per_cycle: 3 → 2
     C. Dead zone: skip scaling if |Δ| ≤ 1 and traffic change < 15%
     D. Asymmetric cooldown: scale-down waits 60s, scale-up waits 30s
  4. [NEW] Co-location: proportional backend distribution across clusters
  5. [NEW] Proactive scaling: uses predictor to trigger before threshold crossing
  6. [NEW] PROM_MAP: per-cluster Prometheus (approccio Romano)
     - Locust API called ONCE for global traffic (not per-cluster)
     - Fallback: sum network bytes from each cluster's Prometheus
     - CPU/memory: each cluster_estimator queries its own Prometheus
"""

import os
import time
import threading
import requests as http_requests
from queue import PriorityQueue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Set
from flask import Flask, request, jsonify
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.utils.logger import setup_logging
from src.utils.config_loader import ConfigLoader
from src.level1.dmos_scheduler import DMOSScheduler
from src.level2.scaler import ReplicaScaler
from src.k8s.client import KubernetesClient
from src.metrics.prometheus_client import PrometheusClient

logger = setup_logging("DMOSMain", log_dir="logs", level="INFO")


@dataclass(order=True)
class SchedulingEvent:
    priority: int
    timestamp: datetime = field(compare=False)
    service: str = field(compare=False)
    action: str = field(compare=False)
    reason: str = field(compare=False)


class DMOSOrchestrator:
    """DMOS with Event-Driven + Periodic Polling + Anti-Oscillation"""
    
    # Prometheus metrics
    scaling_events = Counter(
        'dmos_scaling_events_total',
        'Total scaling events',
        ['cluster', 'service', 'action']
    )
    
    current_replicas = Gauge(
        'dmos_current_replicas',
        'Current replica count per cluster',
        ['cluster', 'service']
    )
    
    target_replicas = Gauge(
        'dmos_target_replicas',
        'Target replica count',
        ['cluster', 'service']
    )
    
    cluster_score = Gauge(
        'dmos_cluster_score',
        'Multi-objective score',
        ['cluster', 'service']
    )
    
    prediction_traffic = Gauge(
        'dmos_predicted_traffic',
        'Predicted traffic',
        ['cluster', 'service']
    )
    
    actual_traffic = Gauge(
        'dmos_actual_traffic',
        'Actual traffic',
        ['service']
    )
    
    scheduling_duration = Histogram(
        'dmos_scheduling_duration_seconds',
        'Scheduling duration',
        ['service']
    )
    
    def __init__(self, config_path: str = "config", num_workers: int = 3):
        logger.info("Initializing DMOS Orchestrator...")
        
        self.config = ConfigLoader(config_path)
        self.scheduler = DMOSScheduler(config_path)
        
        # ── PROM_MAP: un client Prometheus per cluster (approccio Romano) ──
        self.prom_map = {}
        for name, cluster in self.config.clusters.items():
            prom_url = f"http://{cluster.ip}:30090"
            self.prom_map[name] = PrometheusClient(
                url=prom_url,
                cluster_name=name
            )
        
        # ── Locust config (traffico globale, non per-cluster) ──
        self.locust_url = os.environ.get("DMOS_LOCUST_URL", "http://localhost:8089")
        self._locust_available = None  # None=not tested, True/False
        
        k8s_configs = {}
        for name, cluster in self.config.clusters.items():
            k8s_configs[name] = {
                'kubeconfig_path': cluster.kubeconfig_path,
                'server': f'https://{cluster.ip}:6443'
            }
        self.k8s = KubernetesClient(k8s_configs)
        
        # Scalers per-cluster: ogni cluster ha il suo ReplicaScaler e il suo
        # TrafficPredictor. In questo modo la history del traffico è separata
        # per cluster — il predictor vede la serie storica di un solo cluster
        # e può calcolare trend corretti.
        # Struttura: self.scalers[service_name][cluster_name] = ReplicaScaler
        self.scalers: Dict[str, Dict[str, ReplicaScaler]] = {}
        for svc_name, svc_cfg in self.config.services.items():
            self.scalers[svc_name] = {}
            for cluster_name in self.config.clusters:
                self.scalers[svc_name][cluster_name] = ReplicaScaler(
                    capacity_per_replica=svc_cfg.capacity_req_per_sec,
                    min_replicas=svc_cfg.min_replicas,
                    max_replicas=svc_cfg.max_replicas,
                    safety_margin=0.15,
                    max_delta_per_cycle=4
                )
        
        # All cluster names for scale-down of non-allocated clusters
        self.all_cluster_names: list = list(self.config.clusters.keys())
        
        self.event_queue: PriorityQueue[SchedulingEvent] = PriorityQueue()
        self.num_workers = num_workers
        self.running = False
        
        self.last_processed: Dict[str, datetime] = {}
        self.debounce_seconds = 30  # [FIX A] Was 15
        
        # Polling config
        self.polling_interval = 30  # secondi
        self.high_threshold = 30    # req/s
        self.low_threshold = 10     # req/s
        
        # [FIX D] Asymmetric cooldown tracking
        self.last_scale_down: Dict[str, datetime] = {}
        self.scale_down_cooldown = 60  # seconds — scale-down slower

        # [FIX E] Scale-up protection: block scale-down for N seconds after a scale-up.
        # Root cause: when traffic ramps up, the predictor triggers a scale-up.
        # But at the *same* polling cycle the actual reading may still be below
        # low_threshold (Hubble [5m] lags), so a scale-down event is queued
        # immediately after. Adding a protection window avoids this oscillation.
        self.last_scale_up: Dict[str, datetime] = {}
        self.scale_up_protection_seconds = 120  # seconds
        
        # [FIX C] Dead zone: track previous traffic for stability check
        self.previous_traffic: Dict[str, float] = {}
        self.dead_zone_pct = 0.15  # 15% traffic change threshold
        
        # List of services to monitor
        self.monitored_services = list(self.config.services.keys())

        # Startup grace period: DMOS monitors but does NOT issue any k8s scale
        # calls for the first N seconds after start. This prevents Cilium BPF
        # map updates (triggered by pod additions) from disrupting incoming TCP
        # connections while the test framework is establishing sessions.
        # Root cause: if DMOS starts at the same time as Locust, the predictor
        # only has ~22s of history → derivative is large → aggressive scale-up
        # triggers parallel Cilium BPF updates → new SYNs on NodePorts are
        # dropped → Windows TCP timeout = 21s → 100% Locust failure.
        # After 90s the predictor history is long enough that derivatives are
        # gentle and scale decisions are stable.
        self.startup_grace_seconds = 90  # seconds
        self.startup_time = datetime.now()
        
        # ── Co-location: service dependency map ──────────────────────────
        self.service_dependencies: Dict[str, list] = {
            'frontend': [
                'cartservice',
                'productcatalogservice',
                'checkoutservice',
                'recommendationservice',
            ],
        }
        self.colocation_enabled = True
        
        start_http_server(9090)
        logger.info("📊 Metrics server started on :9090/metrics")
        logger.info(f"✅ Orchestrator initialized ({num_workers} workers)")
        logger.info(f"📋 Monitored services: {self.monitored_services}")
        logger.info(f"🌐 Clusters: {self.all_cluster_names}")
        logger.info(f"🔧 Anti-oscillation: debounce={self.debounce_seconds}s, "
                     f"max_delta=4, dead_zone={self.dead_zone_pct*100:.0f}%, "
                     f"scale_down_cooldown={self.scale_down_cooldown}s, "
                     f"scale_up_protection={self.scale_up_protection_seconds}s")
        if self.colocation_enabled:
            logger.info(f"🔗 Co-location: enabled — dependencies: {self.service_dependencies}")
        logger.info(f"⏳ Startup grace: {self.startup_grace_seconds}s "
                    f"(no k8s ops until {self.startup_grace_seconds}s after boot)")
    
    # ── Traffic Helper ─────────────────────────────────────────────────────
    
    def _get_per_cluster_traffic(self, service_name: str) -> Dict[str, float]:
        """
        Misura il traffico reale per-cluster da Prometheus/Hubble locale.

        Ogni istanza di PrometheusClient in prom_map parla con il Prometheus
        del suo cluster (:30090), che vede solo i pod di quel cluster.
        Il risultato è il traffico effettivamente servito da ciascun cluster —
        non una stima proporzionale.

        Con Nginx Ingress + CNP L7 attiva: usa Hubble (hubble_http_requests_total)
        Fallback: network bytes (container_network_receive_bytes_total / 4000)

        Returns:
            Dict {cluster_name: rps} — traffico reale per-cluster
        """
        per_cluster: Dict[str, float] = {}
        for cname, prom in self.prom_map.items():
            rps = prom.get_request_rate(
                service=service_name,
                namespace="online-boutique"
            )
            per_cluster[cname] = max(0.0, rps or 0.0)

        total = sum(per_cluster.values())
        if total > 0:
            breakdown = ", ".join(f"{c}={v:.1f}" for c, v in per_cluster.items())
            logger.info(
                f"✅ Traffic per-cluster [{service_name}]: {breakdown} "
                f"| totale={total:.1f} req/s"
            )
        return per_cluster
    
    # ── Helpers ────────────────────────────────────────────────────────────
    
    def should_process(self, service: str) -> bool:
        if service not in self.last_processed:
            return True
        elapsed = (datetime.now() - self.last_processed[service]).total_seconds()
        return elapsed >= self.debounce_seconds
    
    def _is_traffic_stable(self, service_name: str, current_traffic: float) -> bool:
        """[FIX C] Returns True if traffic changed < dead_zone_pct since last check."""
        prev = self.previous_traffic.get(service_name)
        if prev is None or prev <= 0:
            return False
        return abs(current_traffic - prev) / prev < self.dead_zone_pct
    
    def _can_scale_down(self, service_name: str) -> bool:
        """[FIX D+E] Returns True if scale-down cooldown AND scale-up protection have elapsed.

        Two independent guards:
          1. Cooldown from last scale-DOWN:  scale_down_cooldown  (60s)
             Prevents rapid back-to-back scale-downs (oscillation).
          2. Protection from last scale-UP:  scale_up_protection_seconds (120s)
             Prevents premature scale-down right after a scale-up event.
             This is the main fix for Issue 1 (wrong scale-downs during ramp-up).
        """
        now = datetime.now()

        # Guard 1: cooldown from last scale-down
        last_down = self.last_scale_down.get(service_name)
        if last_down is not None:
            if (now - last_down).total_seconds() < self.scale_down_cooldown:
                return False

        # Guard 2: protection from last scale-up
        last_up = self.last_scale_up.get(service_name)
        if last_up is not None:
            elapsed_since_up = (now - last_up).total_seconds()
            if elapsed_since_up < self.scale_up_protection_seconds:
                remaining = self.scale_up_protection_seconds - elapsed_since_up
                logger.debug(
                    f"  🛡 {service_name}: scale-up protection active "
                    f"({elapsed_since_up:.0f}s since last up, {remaining:.0f}s left)"
                )
                return False

        return True
    
    def _enforce_colocation(self, parent_service: str):
        """
        Proportional co-location: for every cluster that has replicas of
        parent_service, scale dependent backend services proportionally.
        """
        if not self.colocation_enabled:
            return
        
        deps = self.service_dependencies.get(parent_service, [])
        if not deps:
            return
        
        parent_cfg = self.config.get_service(parent_service)
        
        # Step 1: Get frontend distribution across clusters
        cluster_parent_reps = {}
        total_parent_reps = 0
        for cluster_name in self.all_cluster_names:
            reps = self.k8s.get_deployment_replicas(
                cluster=cluster_name,
                deployment=parent_cfg.deployment_name,
                namespace=parent_cfg.namespace
            ) or 0
            cluster_parent_reps[cluster_name] = reps
            total_parent_reps += reps
        
        if total_parent_reps == 0:
            return
        
        active_clusters = [c for c, r in cluster_parent_reps.items() if r > 0]
        
        # Step 2: For each dependent service, distribute proportionally
        for dep_svc_name in deps:
            dep_cfg = self.config.get_service(dep_svc_name)
            if dep_cfg is None:
                continue
            
            total_dep_reps = 0
            cluster_dep_current = {}
            for cluster_name in self.all_cluster_names:
                reps = self.k8s.get_deployment_replicas(
                    cluster=cluster_name,
                    deployment=dep_cfg.deployment_name,
                    namespace=dep_cfg.namespace
                ) or 0
                cluster_dep_current[cluster_name] = reps
                total_dep_reps += reps
            
            target_total = max(total_dep_reps, len(active_clusters))
            
            for cluster_name in self.all_cluster_names:
                parent_reps = cluster_parent_reps[cluster_name]
                current_dep = cluster_dep_current[cluster_name]
                
                if parent_reps == 0:
                    continue
                
                frontend_share = parent_reps / total_parent_reps
                proportional_target = max(
                    dep_cfg.min_replicas,
                    int(round(target_total * frontend_share))
                )
                
                if current_dep < proportional_target:
                    logger.info(
                        f"Co-location: {cluster_name} — {dep_svc_name}: "
                        f"{current_dep} → {proportional_target} "
                        f"(frontend share: {frontend_share:.0%})"
                    )
                    
                    self.k8s.scale_deployment(
                        cluster=cluster_name,
                        deployment=dep_cfg.deployment_name,
                        replicas=proportional_target,
                        namespace=dep_cfg.namespace
                    )
                    
                    self.scaling_events.labels(
                        cluster=cluster_name,
                        service=dep_svc_name,
                        action='scale_up_colocation'
                    ).inc()
                    
                    self.current_replicas.labels(
                        cluster=cluster_name,
                        service=dep_svc_name
                    ).set(current_dep)
                    
                    self.target_replicas.labels(
                        cluster=cluster_name,
                        service=dep_svc_name
                    ).set(proportional_target)
    
    # ── Core Scheduling ───────────────────────────────────────────────────
    
    def schedule_service(self, service_name: str, reason: str = "event"):
        start_time = time.time()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Scheduling di {service_name} (motivo: {reason})")
        logger.info(f"{'='*70}")
        
        # ── Traffic: per-cluster da Prometheus/Hubble ──────────────────
        per_cluster_traffic = self._get_per_cluster_traffic(service_name)
        current_traffic = sum(per_cluster_traffic.values())

        if current_traffic == 0:
            svc_cfg_temp = self.config.get_service(service_name)
            current_traffic = svc_cfg_temp.capacity_req_per_sec * svc_cfg_temp.min_replicas * 0.5
            # Distribuzione fallback uniforme tra cluster
            n = len(self.prom_map) or 1
            per_cluster_traffic = {c: current_traffic / n for c in self.prom_map}
            logger.warning(f"No metrics for {service_name}, using conservative fallback: "
                          f"{current_traffic:.1f} req/s "
                          f"(min_replicas={svc_cfg_temp.min_replicas} × "
                          f"capacity={svc_cfg_temp.capacity_req_per_sec} × 0.5)")

        self.actual_traffic.labels(service=service_name).set(current_traffic)

        traffic_stable = self._is_traffic_stable(service_name, current_traffic)
        prev_t = self.previous_traffic.get(service_name, 0)
        logger.info(f"Current traffic: {current_traffic:.1f} req/s "
                     f"(prev={prev_t:.1f}, stable={traffic_stable})")
        
        self.previous_traffic[service_name] = current_traffic

        # ── Startup grace period ────────────────────────────────────────────
        # Block ALL k8s scale operations for the first startup_grace_seconds.
        # Traffic is still measured above so the predictor history accumulates.
        # Once the grace window expires, DMOS will have a stable derivative
        # baseline and scale decisions will be much more conservative.
        elapsed_startup = (datetime.now() - self.startup_time).total_seconds()
        if elapsed_startup < self.startup_grace_seconds:
            remaining = self.startup_grace_seconds - elapsed_startup
            logger.info(
                f"⏳ Grace period: {elapsed_startup:.0f}/{self.startup_grace_seconds}s "
                f"({remaining:.0f}s remaining) — traffic observed, no k8s ops"
            )
            return

        svc_cfg = self.config.get_service(service_name)
        total_replicas = max(
            svc_cfg.min_replicas,
            int(current_traffic / svc_cfg.capacity_req_per_sec)
        )
        
        logger.info(f"Total replicas needed: {total_replicas}")
        
        allocations, success = self.scheduler.schedule_service(
            service_name=service_name,
            total_replicas=total_replicas,
            predicted_load=current_traffic
        )
        
        if not success:
            logger.error(f"❌ Scheduling fallito")
            return
        
        allocated_cluster_names: Set[str] = set()
        any_scale_down = False
        
        for allocation in allocations:
            cluster_name = allocation.cluster_name
            allocated_cluster_names.add(cluster_name)
            
            self.cluster_score.labels(
                cluster=cluster_name,
                service=service_name
            ).set(allocation.score)
            
            current_reps = self.k8s.get_deployment_replicas(
                cluster=cluster_name,
                deployment=svc_cfg.deployment_name,
                namespace=svc_cfg.namespace
            ) or 0
            
            self.current_replicas.labels(
                cluster=cluster_name,
                service=service_name
            ).set(current_reps)
            
            # Traffico misurato per questo cluster (non più stima proporzionale)
            cluster_traffic = per_cluster_traffic.get(cluster_name, 0.0)

            decision = self.scalers[service_name][cluster_name].compute_target_replicas(
                current_replicas=current_reps,
                current_traffic=cluster_traffic
            )
            
            self.target_replicas.labels(
                cluster=cluster_name,
                service=service_name
            ).set(decision.target_replicas)
            
            self.prediction_traffic.labels(
                cluster=cluster_name,
                service=service_name
            ).set(decision.predicted_traffic)
            
            delta = decision.delta_replicas
            
            # [FIX C] Dead zone: skip ±1 changes when traffic is stable
            if (abs(delta) <= 1
                    and traffic_stable
                    and current_reps >= svc_cfg.min_replicas):
                logger.info(f" {cluster_name}: Δ={delta:+d} in dead zone — skipping")
                continue
            
            # [FIX D] Asymmetric cooldown: block rapid scale-downs
            if delta < 0 and not self._can_scale_down(service_name):
                elapsed = (datetime.now() - self.last_scale_down.get(
                    service_name, datetime.now())).total_seconds()
                remaining = self.scale_down_cooldown - elapsed
                logger.info(f"  ⏳ {cluster_name}: scale-down Δ={delta:+d} "
                            f"blocked by cooldown ({remaining:.0f}s left)")
                continue
            
            logger.info(f"{cluster_name}: {current_reps} → {decision.target_replicas} "
                        f"(Δ={delta:+d})")
            
            if delta != 0:
                action = 'scale_up' if delta > 0 else 'scale_down'
                self.scaling_events.labels(
                    cluster=cluster_name,
                    service=service_name,
                    action=action
                ).inc()
                
                self.k8s.scale_deployment(
                    cluster=cluster_name,
                    deployment=svc_cfg.deployment_name,
                    replicas=decision.target_replicas,
                    namespace=svc_cfg.namespace
                )
                
                if delta < 0:
                    any_scale_down = True
                elif delta > 0:
                    # [FIX E] Record scale-up timestamp to activate protection window
                    self.last_scale_up[service_name] = datetime.now()

        if any_scale_down:
            self.last_scale_down[service_name] = datetime.now()
        
        # Scale down clusters NOT in allocations
        for cluster_name in self.all_cluster_names:
            if cluster_name not in allocated_cluster_names:
                current_reps = self.k8s.get_deployment_replicas(
                    cluster=cluster_name,
                    deployment=svc_cfg.deployment_name,
                    namespace=svc_cfg.namespace
                ) or 0
                
                target = svc_cfg.min_replicas
                
                if current_reps > target:
                    if self._can_scale_down(service_name):
                        logger.info(
                            f"📉 {cluster_name}: not in allocation → "
                            f"scaling {current_reps} → {target} (min_replicas)"
                        )
                        
                        self.scaling_events.labels(
                            cluster=cluster_name,
                            service=service_name,
                            action='scale_down'
                        ).inc()
                        
                        self.k8s.scale_deployment(
                            cluster=cluster_name,
                            deployment=svc_cfg.deployment_name,
                            replicas=target,
                            namespace=svc_cfg.namespace
                        )
                        
                        self.last_scale_down[service_name] = datetime.now()
                    else:
                        logger.debug(f"  ⏳ {cluster_name}: non-alloc scale-down blocked by cooldown")
                else:
                    logger.debug(
                        f"✓ {cluster_name}: not in allocation, "
                        f"already at {current_reps} ≤ {target}"
                    )
                
                # Always update gauges so collector sees real values
                self.current_replicas.labels(
                    cluster=cluster_name,
                    service=service_name
                ).set(current_reps)
                
                self.target_replicas.labels(
                    cluster=cluster_name,
                    service=service_name
                ).set(min(current_reps, target))
        
        # ── Co-location enforcement ──────────────────────────────────────
        self._enforce_colocation(service_name)
        
        duration = time.time() - start_time
        self.scheduling_duration.labels(service=service_name).observe(duration)
        logger.info(f"Scheduling completato in {duration:.2f}s")
    
    # ── Event Processing ──────────────────────────────────────────────────
    
    def process_event(self, event: SchedulingEvent):
        if not self.should_process(event.service):
            logger.debug(f"Skipping {event.service} (debounced)")
            return
        
        logger.info(f"Processing: {event.action} {event.service}")
        
        try:
            self.schedule_service(event.service, reason=event.reason)
            self.last_processed[event.service] = datetime.now()
        except Exception as e:
            logger.error(f"Errore nell'elaborazione di {event.service}: {e}")
    
    # ── Periodic Traffic Monitor ──────────────────────────────────────────
    
    def periodic_check_thread(self):
        """Polling periodico per traffico con scaling proattivo"""
        logger.info(f"   Avvio controllo periodico traffico ({self.polling_interval}s interval)...")
        logger.info(f"   High threshold: {self.high_threshold} req/s")
        logger.info(f"   Low threshold: {self.low_threshold} req/s")
        logger.info(f"   Servizi monitorati: {self.monitored_services}")
        logger.info(f"   Scaling proattivo: abilitato (usa predittore per attivazione anticipata)")
        
        while self.running:
            try:
                for service_name in self.monitored_services:
                    
                    # ── Traffic: per-cluster da Prometheus/Hubble ──────────
                    per_cluster_traffic = self._get_per_cluster_traffic(service_name)
                    current_traffic = sum(per_cluster_traffic.values())

                    svc_cfg = self.config.get_service(service_name)
                    current_replicas = self.k8s.get_deployment_replicas(
                        cluster='cluster1',
                        deployment=svc_cfg.deployment_name,
                        namespace=svc_cfg.namespace
                    )

                    if current_replicas is None:
                        current_replicas = 1

                    if current_traffic == 0:
                        logger.warning(f"Nessuna metrica Prometheus per {service_name}, uso stima di fallback")

                        fe_reps = 0
                        fe_cfg = self.config.get_service('frontend')
                        for cn in self.config.clusters:
                            r = self.k8s.get_deployment_replicas(
                                cluster=cn,
                                deployment=fe_cfg.deployment_name,
                                namespace=fe_cfg.namespace
                            ) or 0
                            fe_reps += r

                        if fe_reps <= fe_cfg.min_replicas:
                            current_traffic = 5.0
                            logger.info(f"  Fallback: frontend al minimo ({fe_reps}) → backend BASSO ({current_traffic} req/s)")
                        elif current_replicas > fe_reps:
                            current_traffic = 8.0
                            logger.info(f"  Fallback: backend({current_replicas}) > frontend({fe_reps}) → BASSO ({current_traffic} req/s)")
                        else:
                            current_traffic = 25.0
                            logger.info(f"  Fallback: backend OK ({current_replicas} vs fe={fe_reps}) → MEDIO ({current_traffic} req/s)")

                    # ── Proactive check: predictor per-cluster ─────────────
                    # Usa il predictor del cluster con più traffico come proxy
                    # per stimare se il sistema nel complesso sta crescendo.
                    predicted_traffic = current_traffic
                    if service_name in self.scalers:
                        now = datetime.now()
                        best_pred = current_traffic
                        for cn in self.all_cluster_names:
                            if cn in self.scalers.get(service_name, {}):
                                pred_result, _ = self.scalers[service_name][cn].predictor.predict(
                                    current_rate=per_cluster_traffic.get(cn, 0.0),
                                    timestamp=now
                                )
                                best_pred = max(best_pred, pred_result)
                        predicted_traffic = best_pred
                    
                    logger.info(f"📊 {service_name}: {current_traffic:.1f} req/s "
                               f"(Predetto: {predicted_traffic:.1f}) | "
                               f"Repliche: {current_replicas}")

                    # [FIX F] Traffic floor: if actual traffic is near-zero, bypass
                    # the predictor and use the real value as effective traffic.
                    # Without this fix, after Locust stops the EMA decays slowly
                    # (e.g. pred=21.6 for 15+ min) keeping replicas=3 needlessly.
                    # Threshold: 2 rps ≈ background noise (Kubernetes health-checks).
                    TRAFFIC_FLOOR_RPS = 2.0
                    if 0 < current_traffic < TRAFFIC_FLOOR_RPS:
                        effective_traffic = current_traffic
                        logger.info(
                            f"⬇ {service_name}: traffic floor active "
                            f"({current_traffic:.2f} < {TRAFFIC_FLOOR_RPS} rps) "
                            f"— predictor bypassed (was {predicted_traffic:.1f})"
                        )
                    else:
                        effective_traffic = max(current_traffic, predicted_traffic)
                    
                    if effective_traffic > self.high_threshold:
                        reason_suffix = ""
                        if predicted_traffic > current_traffic and current_traffic <= self.high_threshold:
                            reason_suffix = "_proactive"
                            logger.info(f"{service_name} PROACTIVE: current={current_traffic:.1f} "
                                       f"< threshold={self.high_threshold}, ma predetto="
                                       f"{predicted_traffic:.1f} > threshold → pre-scaling!")
                        else:
                            logger.info(f" {service_name} HIGH traffic ({effective_traffic:.1f} > {self.high_threshold})")
                        
                        event = SchedulingEvent(
                            priority=0 if reason_suffix == "_proactive" else 1,
                            timestamp=datetime.now(),
                            service=service_name,
                            action='scale_up',
                            reason=f'traffic_high_{effective_traffic:.0f}rps{reason_suffix}'
                        )
                        self.event_queue.put(event)
                    
                    elif effective_traffic < self.low_threshold:
                        logger.info(f"ℹ{service_name} LOW traffic ({effective_traffic:.1f} < {self.low_threshold})")
                        
                        event = SchedulingEvent(
                            priority=2,
                            timestamp=datetime.now(),
                            service=service_name,
                            action='scale_down',
                            reason=f'traffic_low_{effective_traffic:.0f}rps'
                        )
                        self.event_queue.put(event)
                    else:
                        logger.debug(f"{service_name} traffic OK ({effective_traffic:.1f} req/s)")
                        
                # === BACKEND RESET ===
                # Skip backend reset during startup grace period
                _elapsed_gr = (datetime.now() - self.startup_time).total_seconds()
                if _elapsed_gr < self.startup_grace_seconds:
                    logger.debug(
                        f"⏳ Backend reset skipped (grace: {_elapsed_gr:.0f}/"
                        f"{self.startup_grace_seconds}s)"
                    )
                    time.sleep(self.polling_interval)
                    continue

                fe_cfg = self.config.get_service('frontend')
                fe_total = 0
                for cluster_name in self.config.clusters:
                    reps = self.k8s.get_deployment_replicas(
                        cluster=cluster_name,
                        deployment=fe_cfg.deployment_name,
                        namespace=fe_cfg.namespace
                    ) or 0
                    fe_total += reps
                
                fe_at_minimum = (fe_total <= fe_cfg.min_replicas)
                
                backend_services = ['cartservice', 'productcatalogservice', 
                                    'checkoutservice', 'recommendationservice']
                
                for backend_name in backend_services:
                    be_cfg = self.config.get_service(backend_name)
                    be_total = 0
                    for cluster_name in self.config.clusters:
                        be_reps = self.k8s.get_deployment_replicas(
                            cluster=cluster_name,
                            deployment=be_cfg.deployment_name,
                            namespace=be_cfg.namespace
                        ) or 0
                        be_total += be_reps
                    
                    if fe_at_minimum and be_total > be_cfg.min_replicas * len(self.config.clusters):
                        logger.info(f"Backend reset: {backend_name} ha {be_total} repliche "
                                   f"ma frontend al minimo  ({fe_total}) → reset al minimo")
                        
                        min_per_cluster = max(1, be_cfg.min_replicas)
                        for cluster_name in self.config.clusters:
                            self.k8s.scale_deployment(
                                cluster=cluster_name,
                                deployment=be_cfg.deployment_name,
                                replicas=min_per_cluster,
                                namespace=be_cfg.namespace
                            )
                        logger.info(f"  {backend_name} resettato a {min_per_cluster} per cluster")
                    
                    elif not fe_at_minimum and be_total > fe_total * 2:
                        target_total = max(be_cfg.min_replicas * len(self.config.clusters),
                                          fe_total)
                        if be_total > target_total + 2:
                            logger.info(f"Scale-down graduale backend: {backend_name} "
                                       f"{be_total} → target ~{target_total}")
                            event = SchedulingEvent(
                                priority=2,
                                timestamp=datetime.now(),
                                service=backend_name,
                                action='scale_down',
                                reason=f'backend_overscaled_{be_total}_vs_fe_{fe_total}'
                            )
                            self.event_queue.put(event)
                            
                time.sleep(self.polling_interval)
                
            except Exception as e:
                import traceback
                logger.error(f"Errore nel controllo periodico: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.polling_interval)
    
    # ── Webhook Server ────────────────────────────────────────────────────
                    
    def webhook_server(self):
        app = Flask("dmos-webhook")
        
        @app.route('/webhook/alert', methods=['POST'])
        def alert_webhook():
            data = request.json
            logger.info(f"Received webhook: {len(data.get('alerts', []))} alerts")
            
            for alert in data.get('alerts', []):
                labels = alert.get('labels', {})
                service = labels.get('service')
                action = labels.get('action', 'schedule')
                severity = labels.get('severity', 'info')
                alertname = labels.get('alertname', 'Unknown')
                
                if not service:
                    continue
                
                priority_map = {'critical': 0, 'warning': 1, 'info': 2}
                priority = priority_map.get(severity, 2)
                
                logger.info(f"Alert: {alertname} - {service} ({severity})")
                
                event = SchedulingEvent(
                    priority=priority,
                    timestamp=datetime.now(),
                    service=service,
                    action=action,
                    reason=f'prometheus_{alertname}'
                )
                self.event_queue.put(event)
            
            return jsonify({'status': 'received'}), 200
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'queue_size': self.event_queue.qsize(),
                'workers': self.num_workers
            }), 200
        
        @app.route('/api/schedule', methods=['POST'])
        def manual_schedule():
            data = request.json
            service = data.get('service')
            action = data.get('action', 'schedule')
            
            if not service:
                return jsonify({'error': 'service required'}), 400
            
            logger.info(f"Manual trigger: {service} - {action}")
            
            event = SchedulingEvent(
                priority=1,
                timestamp=datetime.now(),
                service=service,
                action=action,
                reason='manual'
            )
            self.event_queue.put(event)
            
            return jsonify({'status': 'queued'}), 200
        
        logger.info("🌐 Starting webhook server on :8081...")
        app.run(host='0.0.0.0', port=8081, debug=False)
    
    def worker_thread(self, worker_id: int):
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                logger.info(f"Worker {worker_id}: {event.service}")
                self.process_event(event)
                self.event_queue.task_done()
            except:
                continue
    
    def run(self):
        self.running = True
        
        threads = []
        
        t = threading.Thread(target=self.webhook_server, daemon=True)
        t.start()
        threads.append(t)
        logger.info("Server webhook avviato")
        
        t = threading.Thread(target=self.periodic_check_thread, daemon=True)
        t.start()
        threads.append(t)
        logger.info("Controllo periodico del traffico avviato")
        
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker_thread, args=(i,), daemon=True)
            t.start()
            threads.append(t)
        logger.info(f"{self.num_workers} worker avviati")
        
        logger.info("\n" + "="*70)
        logger.info("DMOS Orchestrator in esecuzione (Event-Driven + Polling)")
        logger.info("Webhook: http://localhost:8081/webhook/alert")
        logger.info("Metrics: http://localhost:9090/metrics")
        logger.info(f"Polling: Every {self.polling_interval}s")
        logger.info(f"Services: {self.monitored_services}")
        logger.info(f"Anti-oscillation: ON")
        logger.info("="*70 + "\n")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n\nArresto richiesto")
            self.running = False
            for t in threads:
                t.join(timeout=5)
            logger.info("Arrestato")


def main():
    orchestrator = DMOSOrchestrator(config_path="config", num_workers=3)
    orchestrator.run()


if __name__ == "__main__":
    main()