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
"""

import time
import threading
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
        self.prometheus = PrometheusClient(url=self.config.prometheus.url)
        
        k8s_configs = {}
        for name, cluster in self.config.clusters.items():
            k8s_configs[name] = {
                'kubeconfig_path': cluster.kubeconfig_path,
                'server': f'https://{cluster.ip}:6443'
            }
        self.k8s = KubernetesClient(k8s_configs)
        
        # [ANTI-OSC FIX B] Create scalers with reduced max_delta
        self.scalers: Dict[str, ReplicaScaler] = {}
        for svc_name, svc_cfg in self.config.services.items():
            self.scalers[svc_name] = ReplicaScaler(
                capacity_per_replica=svc_cfg.capacity_req_per_sec,
                min_replicas=svc_cfg.min_replicas,
                max_replicas=svc_cfg.max_replicas,
                safety_margin=0.15,
                max_delta_per_cycle=2  # [FIX B] Was 3, now 2
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
        
        # [FIX C] Dead zone: track previous traffic for stability check
        self.previous_traffic: Dict[str, float] = {}
        self.dead_zone_pct = 0.15  # 15% traffic change threshold
        
        # List of services to monitor
        self.monitored_services = list(self.config.services.keys())
        
        # ── Co-location: service dependency map ──────────────────────────
        # If a cluster has replicas of a "parent" service, it must also have
        # at least min_replicas of each dependent "child" service.
        # This avoids cross-cluster gRPC calls that add latency.
        #
        # Online Boutique dependency tree:
        #   frontend → cartservice, productcatalogservice, currencyservice,
        #              recommendationservice, checkoutservice, shippingservice, adservice
        #   checkoutservice → cartservice, currencyservice, shippingservice,
        #                     paymentservice, emailservice
        #
        # We only enforce co-location for services DMOS manages (in services.yaml).
        # Services not in services.yaml (currencyservice, shippingservice, etc.)
        # are assumed to already have 1 replica per cluster from initial deploy.
        self.service_dependencies: Dict[str, list] = {
            'frontend': [
                'cartservice',
                'productcatalogservice',
                'checkoutservice',
                'recommendationservice',
            ],
            # checkoutservice depends on cartservice, but cart is already
            # a frontend dependency — no need to duplicate
        }
        self.colocation_enabled = True
        
        start_http_server(9090)
        logger.info("📊 Metrics server started on :9090/metrics")
        logger.info(f"✅ Orchestrator initialized ({num_workers} workers)")
        logger.info(f"📋 Monitored services: {self.monitored_services}")
        logger.info(f"🌐 Clusters: {self.all_cluster_names}")
        logger.info(f"🔧 Anti-oscillation: debounce={self.debounce_seconds}s, "
                     f"max_delta=2, dead_zone={self.dead_zone_pct*100:.0f}%, "
                     f"scale_down_cooldown={self.scale_down_cooldown}s")
        if self.colocation_enabled:
            logger.info(f"🔗 Co-location: enabled — dependencies: {self.service_dependencies}")
    
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
        """[FIX D] Returns True if scale-down cooldown has elapsed."""
        last = self.last_scale_down.get(service_name)
        if last is None:
            return True
        return (datetime.now() - last).total_seconds() >= self.scale_down_cooldown
    
    def _enforce_colocation(self, parent_service: str):
        """
        Proportional co-location: for every cluster that has replicas of
        parent_service, scale dependent backend services proportionally.
        
        Strategy:
        - Collect frontend distribution across clusters (e.g. 3/3/3 = 33%/33%/33%)
        - For each backend, determine its total replicas needed (from schedule_service)
        - Distribute those replicas across clusters following the same proportion
        - At minimum, every cluster with frontend > 0 gets min_replicas of each backend
        
        This ensures backend services are co-located and proportionally distributed,
        avoiding cross-cluster gRPC calls.
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
            
            # Get current total backend replicas across all clusters
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
            
            # Target: at least 1 per active cluster, total = max(current_total, len(active))
            target_total = max(total_dep_reps, len(active_clusters))
            
            for cluster_name in self.all_cluster_names:
                parent_reps = cluster_parent_reps[cluster_name]
                current_dep = cluster_dep_current[cluster_name]
                
                if parent_reps == 0:
                    # No frontend → backend at min_replicas (handled by schedule_service)
                    continue
                
                # Proportional target: at least min_replicas, proportional to frontend share
                frontend_share = parent_reps / total_parent_reps
                proportional_target = max(
                    dep_cfg.min_replicas,
                    int(round(target_total * frontend_share))
                )
                
                if current_dep < proportional_target:
                    logger.info(
                        f"🔗 Co-location: {cluster_name} — {dep_svc_name}: "
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
        logger.info(f"Scheduling {service_name} (reason: {reason})")
        logger.info(f"{'='*70}")
        
        current_traffic = self.prometheus.get_request_rate(
            service=service_name,
            namespace="online-boutique"
        )
        
        if current_traffic is None:
            # No metrics available — use conservative estimate based on capacity
            # This avoids phantom scaling: assume each existing replica handles
            # half its capacity (light load, not zero)
            svc_cfg_temp = self.config.get_service(service_name)
            current_traffic = svc_cfg_temp.capacity_req_per_sec * svc_cfg_temp.min_replicas * 0.5
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
        
        svc_cfg = self.config.get_service(service_name)
        total_replicas = max(
            svc_cfg.min_replicas,
            int(current_traffic / svc_cfg.capacity_req_per_sec * 1.2)
        )
        
        logger.info(f"Total replicas needed: {total_replicas}")
        
        allocations, success = self.scheduler.schedule_service(
            service_name=service_name,
            total_replicas=total_replicas,
            predicted_load=current_traffic
        )
        
        if not success:
            logger.error(f"❌ Scheduling failed")
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
            
            cluster_traffic = current_traffic * allocation.quota
            
            decision = self.scalers[service_name].compute_target_replicas(
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
                logger.info(f"  🔇 {cluster_name}: Δ={delta:+d} in dead zone — skipping")
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
        # After scheduling parent service, ensure dependent backends exist
        # on every cluster that has the parent.
        self._enforce_colocation(service_name)
        
        duration = time.time() - start_time
        self.scheduling_duration.labels(service=service_name).observe(duration)
        logger.info(f"Scheduling completed in {duration:.2f}s")
    
    # ── Event Processing ──────────────────────────────────────────────────
    
    def process_event(self, event: SchedulingEvent):
        if not self.should_process(event.service):
            logger.debug(f"⏭️  Skipping {event.service} (debounced)")
            return
        
        logger.info(f"🔄 Processing: {event.action} {event.service}")
        
        try:
            self.schedule_service(event.service, reason=event.reason)
            self.last_processed[event.service] = datetime.now()
        except Exception as e:
            logger.error(f"Error processing {event.service}: {e}")
    
    # ── Periodic Traffic Monitor ──────────────────────────────────────────
    
    def periodic_check_thread(self):
        """Polling periodico per traffico con scaling proattivo"""
        logger.info(f"🔄 Starting periodic traffic checker ({self.polling_interval}s interval)...")
        logger.info(f"   High threshold: {self.high_threshold} req/s")
        logger.info(f"   Low threshold: {self.low_threshold} req/s")
        logger.info(f"   Monitored services: {self.monitored_services}")
        logger.info(f"   Proactive scaling: enabled (uses predictor for early trigger)")
        
        while self.running:
            try:
                for service_name in self.monitored_services:
                    
                    current_traffic = self.prometheus.get_request_rate(
                        service=service_name,
                        namespace="online-boutique"
                    )
                    
                    svc_cfg = self.config.get_service(service_name)
                    current_replicas = self.k8s.get_deployment_replicas(
                        cluster='cluster1',
                        deployment=svc_cfg.deployment_name,
                        namespace=svc_cfg.namespace
                    )
                    
                    if current_replicas is None:
                        current_replicas = 1
                    
                    if current_traffic is None or current_traffic == 0:
                        logger.warning(f"No Prometheus metrics for {service_name}, using fallback estimation")
                        
                        # Fallback values must stay BELOW high_threshold (30 rps)
                        # to avoid phantom scaling without real traffic
                        if current_replicas == 1:
                            current_traffic = 15.0  # Conservative: under threshold
                            logger.info(f"  Fallback: {current_replicas} replica → assume MODERATE load ({current_traffic} req/s)")
                        elif current_replicas > 3:
                            current_traffic = 5.0   # Many replicas but no metrics → scale down
                            logger.info(f"  Fallback: {current_replicas} replicas → assume LOW load ({current_traffic} req/s)")
                        else:
                            current_traffic = 15.0  # Under threshold, maintain
                            logger.info(f"  Fallback: {current_replicas} replicas → assume MODERATE load ({current_traffic} req/s)")
                    
                    # ── Proactive check: use predictor to look ahead ──────────
                    # Feed current traffic to the predictor and get prediction
                    predicted_traffic = current_traffic  # default: same as current
                    if service_name in self.scalers:
                        pred_result, _ = self.scalers[service_name].predictor.predict(
                            current_rate=current_traffic,
                            timestamp=datetime.now()
                        )
                        predicted_traffic = pred_result
                    
                    logger.info(f"📊 {service_name}: {current_traffic:.1f} req/s "
                               f"(predicted: {predicted_traffic:.1f}) | "
                               f"Replicas: {current_replicas}")
                    
                    # Decision: trigger based on EITHER current OR predicted traffic
                    # This is the key to proactive scaling
                    effective_traffic = max(current_traffic, predicted_traffic)
                    
                    if effective_traffic > self.high_threshold:
                        reason_suffix = ""
                        if predicted_traffic > current_traffic and current_traffic <= self.high_threshold:
                            reason_suffix = "_proactive"
                            logger.info(f"🔮 {service_name} PROACTIVE: current={current_traffic:.1f} "
                                       f"< threshold={self.high_threshold}, but predicted="
                                       f"{predicted_traffic:.1f} > threshold → pre-scaling!")
                        else:
                            logger.info(f"⚠️  {service_name} HIGH traffic ({effective_traffic:.1f} > {self.high_threshold})")
                        
                        event = SchedulingEvent(
                            priority=0 if reason_suffix == "_proactive" else 1,
                            timestamp=datetime.now(),
                            service=service_name,
                            action='scale_up',
                            reason=f'traffic_high_{effective_traffic:.0f}rps{reason_suffix}'
                        )
                        self.event_queue.put(event)
                    
                    elif effective_traffic < self.low_threshold:
                        logger.info(f"ℹ️  {service_name} LOW traffic ({effective_traffic:.1f} < {self.low_threshold})")
                        
                        event = SchedulingEvent(
                            priority=2,
                            timestamp=datetime.now(),
                            service=service_name,
                            action='scale_down',
                            reason=f'traffic_low_{effective_traffic:.0f}rps'
                        )
                        self.event_queue.put(event)
                    else:
                        logger.debug(f"✓ {service_name} traffic OK ({effective_traffic:.1f} req/s)")
                
                time.sleep(self.polling_interval)
                
            except Exception as e:
                import traceback
                logger.error(f"Error in periodic check: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.polling_interval)
    
    # ── Webhook Server ────────────────────────────────────────────────────
                    
    def webhook_server(self):
        app = Flask("dmos-webhook")
        
        @app.route('/webhook/alert', methods=['POST'])
        def alert_webhook():
            data = request.json
            logger.info(f"📨 Received webhook: {len(data.get('alerts', []))} alerts")
            
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
                
                logger.info(f"🚨 Alert: {alertname} - {service} ({severity})")
                
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
            
            logger.info(f"🔧 Manual trigger: {service} - {action}")
            
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
        logger.info(f"👷 Worker {worker_id} started")
        
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                logger.info(f"👷 Worker {worker_id}: {event.service}")
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
        logger.info("✅ Webhook server started")
        
        t = threading.Thread(target=self.periodic_check_thread, daemon=True)
        t.start()
        threads.append(t)
        logger.info("✅ Periodic traffic checker started")
        
        for i in range(self.num_workers):
            t = threading.Thread(target=self.worker_thread, args=(i,), daemon=True)
            t.start()
            threads.append(t)
        logger.info(f"✅ {self.num_workers} workers started")
        
        logger.info("\n" + "="*70)
        logger.info("DMOS Orchestrator Running (Event-Driven + Polling)")
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
            logger.info("\n\nShutdown requested")
            self.running = False
            for t in threads:
                t.join(timeout=5)
            logger.info("Stopped")


def main():
    orchestrator = DMOSOrchestrator(config_path="config", num_workers=3)
    orchestrator.run()


if __name__ == "__main__":
    main()