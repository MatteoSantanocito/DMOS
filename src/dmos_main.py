"""
DMOS Event-Driven + Periodic Polling Orchestrator
"""

import time
import threading
from queue import PriorityQueue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
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
    """DMOS with Event-Driven + Periodic Polling"""
    
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
        
        self.scalers: Dict[str, ReplicaScaler] = {}
        for svc_name, svc_cfg in self.config.services.items():
            self.scalers[svc_name] = ReplicaScaler(
                capacity_per_replica=svc_cfg.capacity_req_per_sec,
                min_replicas=svc_cfg.min_replicas,
                max_replicas=svc_cfg.max_replicas,
                safety_margin=0.15,
                max_delta_per_cycle=3
            )
        
        self.event_queue: PriorityQueue[SchedulingEvent] = PriorityQueue()
        self.num_workers = num_workers
        self.running = False
        
        self.last_processed: Dict[str, datetime] = {}
        self.debounce_seconds = 15
        
        # Polling config
        self.polling_interval = 30  # secondi
        self.high_threshold = 30    # req/s
        self.low_threshold = 10     # req/s
        
        start_http_server(9090)
        logger.info("📊 Metrics server started on :9090/metrics")
        logger.info("✅ Orchestrator initialized (3 workers)")
    
    def should_process(self, service: str) -> bool:
        if service not in self.last_processed:
            return True
        elapsed = (datetime.now() - self.last_processed[service]).total_seconds()
        return elapsed >= self.debounce_seconds
    
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
            logger.warning(f"No metrics for {service_name}, using default 100 req/s")
            current_traffic = 100.0
        
        self.actual_traffic.labels(service=service_name).set(current_traffic)
        logger.info(f"Current traffic: {current_traffic:.1f} req/s")
        
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
        
        for allocation in allocations:
            cluster_name = allocation.cluster_name
            
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
            
            logger.info(f"{cluster_name}: {current_reps} → {decision.target_replicas} (Δ={decision.delta_replicas:+d})")
            
            if decision.delta_replicas != 0:
                action = 'scale_up' if decision.delta_replicas > 0 else 'scale_down'
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
        
        duration = time.time() - start_time
        self.scheduling_duration.labels(service=service_name).observe(duration)
        logger.info(f"Scheduling completed in {duration:.2f}s")
    
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
    
    
    def periodic_check_thread(self):
        """Polling periodico per traffico"""
        logger.info(f"🔄 Starting periodic traffic checker ({self.polling_interval}s interval)...")
        logger.info(f"   High threshold: {self.high_threshold} req/s")
        logger.info(f"   Low threshold: {self.low_threshold} req/s")
        
        while self.running:
            try:
                for service_name in ['frontend']:
                    
                    # Leggi traffico da Prometheus
                    current_traffic = self.prometheus.get_request_rate(
                        service=service_name,
                        namespace="online-boutique"
                    )
                    
                    # Leggi repliche correnti (sempre, per fallback se serve)
                    svc_cfg = self.config.get_service(service_name)
                    current_replicas = self.k8s.get_deployment_replicas(
                        cluster='cluster1',
                        deployment=svc_cfg.deployment_name,
                        namespace=svc_cfg.namespace
                    )
                    
                    if current_replicas is None:
                        current_replicas = 1  # Fallback safe
                    
                    # Se metriche non disponibili, usa fallback
                    if current_traffic is None or current_traffic == 0:
                        logger.warning(f"No Prometheus metrics for {service_name}, using fallback estimation")
                        
                        # Stima euristica basata su repliche
                        if current_replicas == 1:
                            current_traffic = 50.0  # Forza scaling up per test
                            logger.info(f"  Fallback: {current_replicas} replica → assume HIGH load ({current_traffic} req/s)")
                        elif current_replicas > 3:
                            current_traffic = 8.0  # Forza scaling down
                            logger.info(f"  Fallback: {current_replicas} replicas → assume LOW load ({current_traffic} req/s)")
                        else:
                            current_traffic = 25.0  # Mantieni
                            logger.info(f"  Fallback: {current_replicas} replicas → assume MEDIUM load ({current_traffic} req/s)")
                    
                    logger.info(f"📊 {service_name}: {current_traffic:.1f} req/s | Replicas: {current_replicas}")
                    
                    # Check threshold
                    if current_traffic > self.high_threshold:
                        logger.info(f"⚠️  {service_name} HIGH traffic ({current_traffic:.1f} > {self.high_threshold})")
                        
                        event = SchedulingEvent(
                            priority=1,
                            timestamp=datetime.now(),
                            service=service_name,
                            action='scale_up',
                            reason=f'traffic_high_{current_traffic:.0f}rps'
                        )
                        self.event_queue.put(event)
                    
                    elif current_traffic < self.low_threshold:
                        logger.info(f"ℹ️  {service_name} LOW traffic ({current_traffic:.1f} < {self.low_threshold})")
                        
                        event = SchedulingEvent(
                            priority=2,
                            timestamp=datetime.now(),
                            service=service_name,
                            action='scale_down',
                            reason=f'traffic_low_{current_traffic:.0f}rps'
                        )
                        self.event_queue.put(event)
                    else:
                        logger.debug(f"✓ {service_name} traffic OK ({current_traffic:.1f} req/s)")
                
                # Sleep tra iterazioni
                time.sleep(self.polling_interval)
                
            except Exception as e:
                import traceback
                logger.error(f"Error in periodic check: {e}")
                logger.error(traceback.format_exc())
                time.sleep(self.polling_interval)
                    
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
        
        # 🆕 Periodic checker
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