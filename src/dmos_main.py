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

import math
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import PriorityQueue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Set
from flask import Flask, request, jsonify
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from src.utils.logger import setup_logging
from src.utils.config_loader import ConfigLoader
from src.utils.provenance import build_effective_config, write_json_atomic
from src.level1.dmos_scheduler import DMOSScheduler
from src.level1.winner_determination import Allocation
from src.level2.scaler import ReplicaScaler
from src.level2.predictive_scaler import (PredictiveScaler, PredictiveScalerParams,
                                          scaler_mode)
from src.k8s.client import KubernetesClient
from src.metrics.prometheus_client import PrometheusClient

logger = setup_logging("DMOSMain", log_dir="logs", level="INFO")

# DMOS_ENTRY_BUDGET values (DMOSOrchestrator.entry_budget)
ENTRY_BUDGET_MODES = ("total", "per_cluster", "pooled")


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

    # Predictive scaler (DMOS_SCALER=predictive, docs/PREDICTIVE-SCALER.md)
    ps_fast_rps = Gauge('dmos_ps_fast_rps', 'Entering demand, fast window', ['cluster'])
    ps_forecast_rps = Gauge('dmos_ps_forecast_rps',
                            'Holt forecast of the entering demand at the horizon', ['cluster'])
    ps_demand_rps = Gauge('dmos_ps_demand_rps',
                          'Demand the entry service is sized on (forecast + headroom)', ['cluster'])
    ps_need = Gauge('dmos_ps_need_replicas', 'Replicas needed for the sized demand', ['cluster'])
    ps_horizon_s = Gauge('dmos_ps_horizon_seconds', 'Forecast horizon of the predictive scaler')
    ps_t_ready_s = Gauge('dmos_ps_actuation_lag_seconds',
                         'Learned actuation lag (scale command -> replicas ready)')
    ps_burst = Gauge('dmos_ps_burst', '1 while the predictive scaler holds a burst')

    # Provenance: one series per process, value always 1. The collector reads
    # config_id from here, proving which configuration served every snapshot.
    config_info = Gauge(
        'dmos_config_info',
        'Effective DMOS configuration of this process (value always 1)',
        ['config_id', 'level1', 'demand_roles', 'gate_gamma', 'score_profile']
    )
    
    def __init__(self, config_path: str = "config", num_workers: int = 3,
                 no_scheduling: bool = False, naive_backends: bool = False,
                 dry_run: bool = False, config_dump_path: Optional[str] = None):
        # dry_run: build every component and the effective configuration, but
        # open no port, create no Kubernetes client and probe no Prometheus.
        # Used by `--dump-config-only` (offline check of an arm's config).
        self.dry_run = dry_run
        self.no_scheduling = no_scheduling
        # Baseline 'naive' (paper §metastable): scala i backend reattivamente sul
        # loro rps L7 osservato (hubble_http_requests_total), invece del co-scaling
        # feed-forward. Sotto saturazione i retry gonfiano la metrica -> over-scaling
        # e churn -> fallimento metastabile. L1+L2 (frontend) restano attivi:
        # cosi' l'unica differenza vs l1l2 e' il metodo di sizing dei backend.
        self.naive_backends = naive_backends
        self.naive_target_rps = 50.0   # rps per replica backend (stile HPA)
        if naive_backends:
            logger.info("  MODALITA' NAIVE-BACKENDS — backend reattivi su "
                        "hubble_http_requests_total (baseline metastabile)")
        if no_scheduling:
            logger.info("=" * 60)
            logger.info("  MODALITÀ NO-SCHEDULING — Level 1 DISABILITATO")
            logger.info("  Level 2 (scaling) attivo: calcola repliche con PD controller")
            logger.info("  Level 1 (scheduling) OFF: distribuzione uniforme N/K per cluster")
            logger.info("  Nessun punteggio, nessun Hamilton, nessuna riallocazione geo-aware")
            logger.info("=" * 60)
        logger.info("Initializing DMOS Orchestrator...")
        
        self.config = ConfigLoader(config_path)
        self.scheduler = DMOSScheduler(config_path, probe_prometheus=not dry_run)

        # Minimum replicas of the entry service per cluster, overriding
        # services.yaml (DMOS_ENTRY_MIN_REPLICAS, set by the '-min<k>' arms).
        # 0 lets a cluster without demand hold no entry replica: with the
        # frontend exposed as a Cilium global service its users are served by
        # the replicas of the other clusters. Applied before the scalers are
        # built, so the budget, the targets and the scale-down all use it; the
        # backends keep their own minimum. Recorded in the run provenance.
        self.entry_min_replicas = self._entry_min_replicas_override()

        # How the entry service's budget N is bounded (DMOS_ENTRY_BUDGET):
        #   total        each cluster needs ceil(demand_i / capacity); only the
        #                SUM is capped (Σ max): a full cluster's excess stays in
        #                the budget and the allocation places it on the others
        #                (right with the frontend served by the global pool).
        #                Default from the journal campaign on.
        #   per_cluster  each cluster's need is capped at its own maximum before
        #                summing: the excess of a full hot cluster is lost (NCA
        #                2026 campaign, local serving). Kept to reproduce it.
        self.entry_budget_mode = self._entry_budget_mode()
        if self.entry_min_replicas is not None:
            entry_cfg = self.config.entry_service_config()
            entry_cfg.min_replicas_per_cluster = {
                c: self.entry_min_replicas for c in self.config.clusters}
            logger.info(f"Entry service '{self.config.application.entry_service}': "
                        f"minimum {self.entry_min_replicas} replica(s) per cluster "
                        f"(DMOS_ENTRY_MIN_REPLICAS)")

        # ── PROM_MAP: un client Prometheus per cluster (approccio Romano) ──
        # Le porte sono lette per cluster da local.yaml (default 32090 per
        # kube-prometheus-stack via proxmox-k3s; 30090 per testbed legacy).
        self.prom_map = {}
        for name, cluster in self.config.clusters.items():
            prom_url = f"http://{cluster.ip}:{cluster.prometheus_port}"
            self.prom_map[name] = PrometheusClient(
                url=prom_url,
                cluster_name=name,
                test_connection=not dry_run,
                ingress_demand_match=cluster.ingress_demand_match,
            )

        # ── Locust config (traffico globale, non per-cluster) ──
        self.locust_url = os.environ.get("DMOS_LOCUST_URL", "http://localhost:8089")
        self._locust_available = None  # None=not tested, True/False

        k8s_configs = {}
        for name, cluster in self.config.clusters.items():
            # NON sovrascrivere il server dell'API: i kubeconfig di proxmox-k3s
            # puntano gia' al nodo control-plane corretto (es. <control-plane-ip>).
            # cluster.ip e' l'IP del WORKER (usato per i NodePort Prometheus) e
            # su :6443 NON risponde — l'API server K3s gira sul CP. Omettendo
            # 'server', KubernetesClient tiene il server del kubeconfig.
            k8s_configs[name] = {
                'kubeconfig_path': cluster.kubeconfig_path,
            }
        self.k8s = None if dry_run else KubernetesClient(k8s_configs)
        
        # Scalers per-cluster: ogni cluster ha il suo ReplicaScaler e il suo
        # TrafficPredictor. In questo modo la history del traffico è separata
        # per cluster — il predictor vede la serie storica di un solo cluster
        # e può calcolare trend corretti.
        # Struttura: self.scalers[service_name][cluster_name] = ReplicaScaler
        #
        # [FIX Q] Capacity per-cluster: con netem attivo la capacità reale del
        # frontend varia 5× tra cluster (16/6/3 rps). Usando un valore globale
        # (30 o 16) lo scaler sottostimava le repliche necessarie sui cluster
        # lenti, producendo 5xx identici con/senza DMOS. Ora ogni scaler usa
        # svc_cfg.get_capacity_for_cluster(cluster_name) che legge
        # capacity_per_cluster da services.yaml se presente.
        # Per-cluster rate limits and margin (recorded in the run provenance).
        self.scaler_safety_margin = 0.15
        self.max_delta_up = 6     # [FIX G] scale-UP aggressivo
        self.max_delta_down = 1   # [FIX I+T] scale-DOWN ultra-conservativo
        self.scalers: Dict[str, Dict[str, ReplicaScaler]] = {}
        for svc_name, svc_cfg in self.config.services.items():
            self.scalers[svc_name] = {}
            for cluster_name in self.config.clusters:
                cluster_capacity = svc_cfg.get_capacity_for_cluster(cluster_name)
                cluster_min_replicas = svc_cfg.get_min_replicas_for_cluster(cluster_name)
                cluster_max_replicas = svc_cfg.get_max_replicas_for_cluster(cluster_name)
                self.scalers[svc_name][cluster_name] = ReplicaScaler(
                    capacity_per_replica=cluster_capacity,
                    min_replicas=cluster_min_replicas,
                    max_replicas=cluster_max_replicas,
                    safety_margin=self.scaler_safety_margin,
                    max_delta_per_cycle=self.max_delta_up,
                    max_delta_down=self.max_delta_down
                                            # [FIX I+T] scale-DOWN ultra-conservativo:
                                            # max 1 pod rimosso per servizio/cluster/ciclo.
                                            # Era 2: pcatalog C3 passava 3→1 di colpo,
                                            # rompendo le connessioni gRPC Envoy L7 e
                                            # causando cascata di timeout. Con 1: 3→2→1
                                            # su 2 cicli (60s), Envoy ha tempo di riconfigurarsi.
                )
        
        # All cluster names for scale-down of non-allocated clusters
        self.all_cluster_names: list = list(self.config.clusters.keys())
        
        self.event_queue: PriorityQueue[SchedulingEvent] = PriorityQueue()
        self.num_workers = num_workers
        self.running = False
        
        self.last_processed: Dict[str, datetime] = {}
        self.debounce_seconds = 30  # [FIX A] Was 15

        # L2 PD correction actually APPLIED on top of the L1 strategic target:
        # the controller output is clipped to [-3, +3] inside PDController, then
        # clamped here to ±pd_correction_clamp (see schedule_service, Step 2b).
        self.pd_correction_clamp = 1
        
        # Polling config
        self.polling_interval = 30  # secondi
        self.high_threshold = 30    # req/s
        self.low_threshold = 10     # req/s
        
        # [FIX D] Asymmetric cooldown tracking
        self.last_scale_down: Dict[str, datetime] = {}
        self.scale_down_cooldown = 90  # seconds — (was 180: troppo lungo, accumulava
        # scale-down pendenti durante dead zone → burst al rilascio → decline crash.
        # Con 90s: scale-down più frequente ma meno aggressivo)

        # [FIX E] Scale-up protection: block scale-down for N seconds after a scale-up.
        # Root cause: when traffic ramps up, the predictor triggers a scale-up.
        # But at the *same* polling cycle the actual reading may still be below
        # low_threshold (Hubble [5m] lags), so a scale-down event is queued
        # immediately after. Adding a protection window avoids this oscillation.
        #
        # NOTA: questa finestra blocca solo lo scale-DOWN, NON blocca scale-up
        # multipli consecutivi (quelli sono limitati solo da max_delta + polling_interval).
        # Ridotto 120 → 45s: con Hubble scrape interval=15s e finestra [1m],
        # la lettura del traffico è stabile già dopo ~45s. 120s era eccessivo e
        # causava scale-down prematuri durante la fase di cooldown del flash crowd.
        self.last_scale_up: Dict[str, datetime] = {}
        self.scale_up_protection_seconds = 90  # seconds (era 120)
        
        # [FIX S] Global scale-down budget per polling cycle.
        # Limita il numero TOTALE di pod terminati in un ciclo di 30s, condiviso
        # tra tutti i servizi e cluster. Evita il thundering herd: senza budget,
        # DMOS terminava 15+ pod simultaneamente durante il decline del traffico,
        # rompendo le connessioni gRPC persistenti (Envoy L7) e causando cascata
        # di timeout irrecuperabile. Con budget=3, lo scale-down si distribuisce
        # su ~5 cicli (2.5 min) dando tempo a Envoy di riconfigurarsi.
        self.scale_down_budget_per_cycle = 2  # (was 3: troppi pod rimossi al rilascio dead zone)
        self._cycle_pods_terminated = 0

        # [FIX A2] Global scale-UP budget per polling cycle.
        # Simmetrico allo scale-down budget. Senza, lo scale-up era illimitato:
        # max_delta_up=6 PER servizio/cluster -> fino a ~180 pod aggiunti in un
        # ciclo -> churn incontrollato degli endpoint Cilium/Envoy. Con budget=8
        # uno spike reale (~29 pod su 4-5 cicli) viene comunque servito, ma si
        # taglia il thundering herd e si smorza il loop di feedback in cooldown.
        self.scale_up_budget_per_cycle = 8
        self._cycle_pods_added = 0

        # [FIX COSCALE] Budget di churn SEPARATO per il co-scaling backend.
        # Il co-scaling e' feed-forward dependency-driven: insegue il frontend e
        # NON deve competere con lo scaling reattivo del frontend per il budget
        # anti-thundering-herd condiviso. Con il budget condiviso, durante la
        # rampa i 6 frontend lo esaurivano e il co-scaling (che gira per ultimo)
        # veniva sempre deferito -> i backend restavano a 1 replica anche col
        # target=2 (collo recommendation, Python a 0.2 core) -> mascherava il
        # placement. Budget dedicato e generoso: i target di co-scaling sono
        # gia' limitati da c_max e dal rapporto di capacita', quindi non c'e'
        # rischio di thundering herd.
        self.backend_scale_up_budget_per_cycle = 12
        self.backend_scale_down_budget_per_cycle = 6
        self._cycle_backend_pods_added = 0
        self._cycle_backend_pods_terminated = 0

        # [DEMAND-FLOOR] toggle: con l'overflow data-plane (LB L7 futuro) le
        # repliche tornano fungibili tra cluster → si disattiva il floor e lo
        # score torna a governare tutta la collocazione. Default ON (no overflow).
        #   DMOS_DEMAND_FLOOR=0  → disattiva (per la modalità overflow/LB)
        self.demand_floor_enabled = os.environ.get("DMOS_DEMAND_FLOOR", "1") != "0"
        # A score model (DMOS_SCORE_MODEL) defines the floor itself: the env
        # switch is ignored (and reported by the scheduler) in that case.
        if self.scheduler.score_model is not None:
            self.demand_floor_enabled = self.scheduler.score_model.demand_floor

        # [FIX C] Dead zone: track previous traffic for stability check
        self.previous_traffic: Dict[str, float] = {}
        self.dead_zone_pct = 0.10  # 10% traffic change threshold
        # Era 25%: bloccava scale-down anche durante il decline (variazione <25% tra cicli)
        # → frontend restava a 9 repliche per sempre.
        # Con 10%: durante il picco sostenuto (variazione ~2-5%) → blocca scale-down (protezione).
        # Durante il decline (variazione >10% per ciclo) → permette scale-down.
        
        # The application DMOS manages (services.yaml → `application`):
        # entry service, co-scaled services, how the entering demand is measured.
        self.application = self.config.application

        # List of services to monitor.
        # [FIX option-1] Solo il servizio d'ingresso (frontend) e' schedulato a
        # traffico: si dimensiona sull'ingress rate (pulito). I backend NON sono
        # piu' schedulati dal loro scaler L2 (leggeva hubble_http_requests_total,
        # corrotto dai retry sotto saturazione): sono dimensionati esclusivamente
        # dalla co-location, invocata al termine di schedule_service(<entry>).
        self.monitored_services = [self.application.entry_service]

        # ── Level-2 scaler of the entry service (DMOS_SCALER) ────────────
        # 'reactive' (default): ReplicaScaler + PD per cluster, the NCA 2026
        # Level 2. 'predictive': PredictiveScaler (two demand windows, Holt
        # forecast at the learned actuation lag, uncertainty headroom,
        # capacity-model sizing, event-driven decisions, stabilized
        # scale-down); docs/PREDICTIVE-SCALER.md. Level 1 is the same in both.
        self.scaler_mode = scaler_mode()
        self.ps: Optional[PredictiveScaler] = None
        if self.scaler_mode == "predictive":
            entry_cfg = self.config.entry_service_config()
            ps_params = PredictiveScalerParams.from_env()
            self.ps = PredictiveScaler(
                ps_params, self.all_cluster_names,
                capacity={c: entry_cfg.get_capacity_for_cluster(c) for c in self.all_cluster_names},
                min_replicas={c: entry_cfg.get_min_replicas_for_cluster(c)
                              for c in self.all_cluster_names},
                max_replicas={c: entry_cfg.get_max_replicas_for_cluster(c)
                              for c in self.all_cluster_names},
                budget_mode=self.entry_budget_mode,
            )
            # decisions may follow each other every min_decision_spacing_s
            self.debounce_seconds = ps_params.min_decision_spacing_s
            logger.info(f"Level 2: PREDICTIVE scaler (tick {ps_params.tick_s:.0f}s, "
                        f"windows {ps_params.fast_window}/{ps_params.stable_window}, "
                        f"horizon {'learned' if ps_params.horizon_s <= 0 else ps_params.horizon_s}, "
                        f"down window {ps_params.down_window_s:.0f}s)")

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
        self.startup_grace_seconds = 120  # seconds — allineato a COLD_START_SECONDS in dmos_scheduler.py
        # 0-120s: nessun k8s op (grace) + Phase 1 weights (solo Φ_demand, Φ_net, Φ_carbon)
        # 120s+:  Phase 2 completo con Φ_response_time, Φ_cap, Φ_load
        self.startup_time = datetime.now()
        
        # ── Co-location: service dependency map ──────────────────────────
        # Services sized from the entry service, in this order (the order
        # matters: they share the per-cycle backend churn budget). From
        # services.yaml → application.co_scaled_services; for Online Boutique:
        # cart, productcatalog, checkout, recommendation, currency, ad,
        # shipping — the stateless services on the hot path (redis-cart is
        # stateful and excluded), co-scaled fan-out-aware (calls_per_request).
        self.service_dependencies: Dict[str, list] = {
            self.application.entry_service: list(self.application.co_scaled_services),
        }
        self.colocation_enabled = True

        # ── Backend co-location floor ratio ───────────────────────────────
        # Effective co-location ratio: r = max(C_front/C_back, r_floor)
        # Isolated-replica capacity benchmarks under-estimate effective
        # service capacity under microservice gRPC fanout: connection-pool
        # exhaustion and tail-latency cascade [Dean & Barroso, CACM 2013
        # "The Tail at Scale"] reduce real throughput well below the bench
        # figure. With nominal C_back >> C_front (e.g., cartservice ~1200
        # rps vs frontend ~60 rps → ratio=0.05), `ceil(3·0.05)=1` keeps
        # backends at 1 replica even when frontend is scaled, producing
        # the saturation cascade documented in paper §X.X.
        #
        # We introduce a floor on the co-location ratio so backend
        # provisioning grows with frontend regardless of the (overly
        # optimistic) capacity headroom in services.yaml.
        #
        # FLOOR VALUE TUNING (empirically calibrated):
        # - 0.5 (initial): backends ≈ ceil(frontend × 0.5) → with frontend=3,
        #     backends=2 each. Sufficient under heavy 400rps single-spike
        #     but OVER-PROVISIONS at moderate 300rps multimodal (10 pods on
        #     an 8-core VM → CPU contention, observed degradation).
        # - 0.3 (current): backends ≈ ceil(frontend × 0.3) → with frontend=3,
        #     backends=1; with frontend=4, backends=2; with frontend=6,
        #     backends=2. Avoids over-provisioning while still scaling at
        #     higher frontend counts. Paper §X.X tuning section discusses
        #     why 0.3 balances cascade-protection vs HW-saturation.
        self.backend_colocation_floor = 0.3

        if not dry_run:
            start_http_server(9090)
            logger.info("📊 Metrics server started on :9090/metrics")

        # ── Provenance: effective configuration of THIS process ──────────
        # Built from the live objects (after every env override), written
        # BEFORE the Kubernetes calls below so it is available within seconds
        # of the start, and exported as dmos_config_info{config_id=...}.
        self.effective_config = build_effective_config(self)
        self.config_id = self.effective_config["config_id"]
        logger.info(f"🧾 Effective config {self.effective_config['summary']}")
        if config_dump_path:
            try:
                write_json_atomic(config_dump_path, self.effective_config)
                logger.info(f"🧾 Effective config written to {config_dump_path}")
            except Exception as e:  # never let provenance break the scheduler
                logger.error(f"❌ Cannot write effective config to {config_dump_path}: {e}")

        if dry_run:
            logger.info("Dry run: no metrics server, no Kubernetes calls")
            return

        warm = self.effective_config["level1"]["warm"]
        self.config_info.labels(
            config_id=self.config_id,
            level1="on" if not self.no_scheduling else "off",
            demand_roles=",".join(warm["demand_roles"]) or "none",
            gate_gamma=str(warm["gate"]["gamma"] if warm["gate"]["enabled"] else "off"),
            score_profile=str(self.effective_config["level1"]["score_profile"]),
        ).set(1)

        self._initialize_replica_gauges()
        logger.info(f"✅ Orchestrator initialized ({num_workers} workers)")
        logger.info(f"📋 Monitored services: {self.monitored_services}")
        logger.info(f"🌐 Clusters: {self.all_cluster_names}")
        logger.info(f"🔧 Anti-oscillation: debounce={self.debounce_seconds}s, "
                     f"max_delta_up={self.max_delta_up}, max_delta_down={self.max_delta_down}, "
                     f"dead_zone={self.dead_zone_pct*100:.0f}%, "
                     f"scale_down_cooldown={self.scale_down_cooldown}s, "
                     f"scale_up_protection={self.scale_up_protection_seconds}s")
        if self.colocation_enabled:
            logger.info(f"🔗 Co-location: enabled — dependencies: {self.service_dependencies}")
        logger.info(f"⏳ Startup grace: {self.startup_grace_seconds}s "
                    f"(no k8s ops until {self.startup_grace_seconds}s after boot)")
    
    # ── Startup Initialization ──────────────────────────────────────────────

    @staticmethod
    def _entry_budget_mode() -> str:
        mode = os.environ.get("DMOS_ENTRY_BUDGET", "total").strip().lower() or "total"
        if mode not in ENTRY_BUDGET_MODES:
            raise ValueError(f"DMOS_ENTRY_BUDGET must be one of {ENTRY_BUDGET_MODES}, got {mode!r}")
        return mode

    @staticmethod
    def entry_budget(per_cluster_traffic: Dict[str, float], clusters, svc_cfg,
                     mode: str = "total"):
        """Budget N of the entry service and the per-cluster floors.

        need_i = max(min_i, ceil(traffic_i / capacity_i)). 'per_cluster':
        N = Σ min(max_i, need_i) (NCA 2026). 'total': N = min(Σ max_i, Σ need_i).
        'pooled': the replicas form one pool (Cilium global service, affinity
        'none': every replica serves every cluster's users), so the need is
        sized on the total demand, N = ceil(Σ traffic_i / r̄) within
        [Σ min_i, Σ max_i], r̄ = capacity per replica averaged with the
        per-cluster caps as weights. 'total' rounds up every cluster and counts
        the minimum of a cluster with little demand as if it served only that
        demand: on 29/10/2026 it asked 17-18 replicas for ~13 of need.
        Floors (used by the demand floor only) are always min(max_i, need_i).
        At least 1 replica in the federation."""
        if mode == "pooled":
            return DMOSOrchestrator._pooled_budget(per_cluster_traffic, clusters, svc_cfg)
        total, floors, cap_sum = 0, {}, 0
        for c in clusters:
            traffic = per_cluster_traffic.get(c, 0.0)
            capacity = svc_cfg.get_capacity_for_cluster(c)
            c_min = svc_cfg.get_min_replicas_for_cluster(c)
            c_max = svc_cfg.get_max_replicas_for_cluster(c)
            need = max(c_min, math.ceil(traffic / capacity) if capacity > 0 else c_min)
            floors[c] = min(c_max, need)
            total += floors[c] if mode == "per_cluster" else need
            cap_sum += c_max
        if mode != "per_cluster":
            total = min(total, cap_sum)
        # Only possible with a 0 minimum and no measured demand anywhere:
        # keep one entry replica in the federation.
        return max(1, total), floors

    @staticmethod
    def _pooled_budget(per_cluster_traffic: Dict[str, float], clusters, svc_cfg):
        floors, min_sum, cap_sum, weighted_cap, demand = {}, 0, 0, 0.0, 0.0
        for c in clusters:
            traffic = max(0.0, float(per_cluster_traffic.get(c, 0.0) or 0.0))
            capacity = svc_cfg.get_capacity_for_cluster(c)
            c_min = svc_cfg.get_min_replicas_for_cluster(c)
            c_max = svc_cfg.get_max_replicas_for_cluster(c)
            need = max(c_min, math.ceil(traffic / capacity) if capacity > 0 else c_min)
            floors[c] = min(c_max, need)
            min_sum += c_min
            cap_sum += c_max
            weighted_cap += c_max * capacity
            demand += traffic
        r_mean = weighted_cap / cap_sum if cap_sum > 0 else 0.0
        need = math.ceil(demand / r_mean) if r_mean > 0 else min_sum
        total = min(cap_sum, max(min_sum, need))
        return max(1, total), floors

    @staticmethod
    def uniform_split(total: int, clusters, svc_cfg) -> Dict[str, int]:
        """N/K per cluster within [min_i, max_i]: one replica at a time to the
        cluster with the fewest, ties in name order. Without binding caps this is
        base + remainder in name order (the NCA 2026 l2 arm); with them, a
        replica a full cluster cannot take goes to the others instead of being
        cut by the cap."""
        names = sorted(clusters)
        caps = {c: svc_cfg.get_max_replicas_for_cluster(c) for c in names}
        out = {c: min(caps[c], svc_cfg.get_min_replicas_for_cluster(c)) for c in names}
        for _ in range(max(0, min(total, sum(caps.values())) - sum(out.values()))):
            c = min((c for c in names if out[c] < caps[c]), key=out.get)
            out[c] += 1
        return out

    @staticmethod
    def _entry_min_replicas_override() -> Optional[int]:
        """DMOS_ENTRY_MIN_REPLICAS as a non-negative int, or None when unset."""
        raw = os.environ.get("DMOS_ENTRY_MIN_REPLICAS", "").strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"DMOS_ENTRY_MIN_REPLICAS must be a non-negative integer, got {raw!r}")
        if value < 0:
            raise ValueError(f"DMOS_ENTRY_MIN_REPLICAS must be >= 0, got {value}")
        return value

    def _initialize_replica_gauges(self):
        """
        Legge il numero reale di repliche da Kubernetes all'avvio e inizializza
        i gauge dmos_current_replicas / dmos_target_replicas.

        Senza questo, i gauge partono a 0 (default Prometheus) anche se i pod
        esistono già, causando:
          - Falso primo scaling event "0→N" nell'analisi
          - avg_tts / provisioning ratio distorti nei primi snapshot
          - Classificazione REACTIVE errata sul primo ciclo
        """
        logger.info("⚙️  Initializing replica gauges from Kubernetes state...")
        for svc_name, svc_cfg in self.config.services.items():
            for cluster_name in self.all_cluster_names:
                reps = self.k8s.get_deployment_replicas(
                    cluster=cluster_name,
                    deployment=svc_cfg.deployment_name,
                    namespace=svc_cfg.namespace
                )
                if reps is None:
                    reps = svc_cfg.get_min_replicas_for_cluster(cluster_name)
                    logger.warning(f"  {cluster_name}/{svc_name}: k8s read failed, "
                                   f"defaulting to min_replicas={reps}")
                self.current_replicas.labels(cluster=cluster_name, service=svc_name).set(reps)
                self.target_replicas.labels(cluster=cluster_name, service=svc_name).set(reps)
                logger.info(f"  {cluster_name}/{svc_name}: {reps} replicas")

    # ── Traffic Helper ─────────────────────────────────────────────────────

    def _get_per_cluster_traffic(self, service_name: str) -> Dict[str, float]:
        """
        Misura il traffico reale per-cluster da Prometheus/Hubble locale.

        Ogni istanza di PrometheusClient in prom_map parla con il Prometheus
        del suo cluster (porta da local.yaml, default 32090), che vede solo
        i pod di quel cluster.
        Il risultato è il traffico effettivamente servito da ciascun cluster —
        non una stima proporzionale.

        Con Cilium Ingress + CNP L7 attiva: usa Hubble (hubble_http_requests_total)
        Fallback: network bytes (container_network_receive_bytes_total / 4000)

        Returns:
            Dict {cluster_name: rps} — traffico reale per-cluster
        """
        per_cluster: Dict[str, float] = {}
        # [FIX option-1] Il servizio d'ingresso si dimensiona sulla domanda che
        # ENTRA in ogni cluster (PrometheusClient.get_ingress_rate, configurata
        # da services.yaml → application.demand). E' immune all'inflazione da
        # retry/fan-out interni che corrompe hubble_http_requests_total sotto
        # saturazione. I backend NON passano piu' di qui: sono dimensionati
        # dalla co-location.
        with ThreadPoolExecutor(max_workers=len(self.prom_map)) as pool:
            if service_name == self.application.entry_service:
                ingress_kwargs = self.config.ingress_query_kwargs()
                futures = {
                    pool.submit(prom.get_ingress_rate, **ingress_kwargs): cname
                    for cname, prom in self.prom_map.items()
                }
            else:
                svc_cfg = self.config.get_service(service_name)
                workload = svc_cfg.deployment_name if svc_cfg else service_name
                namespace = (svc_cfg.namespace if svc_cfg
                             else self.config.entry_service_config().namespace)
                futures = {
                    pool.submit(prom.get_request_rate, workload, namespace): cname
                    for cname, prom in self.prom_map.items()
                }
            for fut in as_completed(futures):
                cname = futures[fut]
                per_cluster[cname] = max(0.0, fut.result() or 0.0)

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
        Co-location: i servizi backend dipendenti sono dimensionati interamente
        in proporzione alle repliche del parent (frontend), su base per-cluster.

        [FIX option-1] Unica via di dimensionamento dei backend: lo scaler L2
        per-backend e' stato rimosso dal flusso (leggeva hubble_http_requests_total,
        corrotto dai retry sotto saturazione). La co-location usa solo fatti
        Kubernetes (repliche del frontend) + il modello di capacita' misurato
        -> immune alla corruzione della metrica. Bidirezionale: il backend
        insegue il frontend sia in su sia in giu'; rispetta i budget di churn.
        """
        if not self.colocation_enabled:
            return
        
        deps = self.service_dependencies.get(parent_service, [])
        if not deps:
            return

        parent_cfg = self.config.get_service(parent_service)

        # Repliche del frontend per cluster (fatto Kubernetes, non una metrica)
        cluster_parent_reps = {}
        for cluster_name in self.all_cluster_names:
            cluster_parent_reps[cluster_name] = self.k8s.get_deployment_replicas(
                cluster=cluster_name,
                deployment=parent_cfg.deployment_name,
                namespace=parent_cfg.namespace
            ) or 0

        if sum(cluster_parent_reps.values()) == 0:
            return

        for dep_svc_name in deps:
            dep_cfg = self.config.get_service(dep_svc_name)
            if dep_cfg is None:
                continue

            for cluster_name in self.all_cluster_names:
                parent_reps = cluster_parent_reps[cluster_name]

                # ── Capacity-based ratio (nominal) ───────────────────────
                # C_front / C_back from services.yaml. Under-estimates real
                # capacity demand under gRPC fanout (see floor argument below).
                parent_cap = parent_cfg.get_capacity_for_cluster(cluster_name)
                dep_cap = dep_cfg.get_capacity_for_cluster(cluster_name)
                if dep_cap <= 0:
                    dep_cap = 50.0
                # [FANOUT] un servizio chiamato N volte per richiesta riceve
                # N×traffico → va scalato N volte di più. fanout misurato da
                # Hubble (calls_per_request in services.yaml). Senza questo, i
                # servizi ad alto fanout (es. currency ~3.4×) restano sotto-
                # provisionati anche se gestiti → collo (vedi audit currency).
                fanout = getattr(dep_cfg, "calls_per_request", 1.0) or 1.0
                capacity_ratio_nominal = (parent_cap / dep_cap) * fanout

                # ── Effective ratio: max(nominal, floor) ─────────────────
                # Floor protects against capacity over-estimation in
                # benchmarks (Dean & Barroso 2013 "tail at scale" cascade).
                # See `self.backend_colocation_floor` init for rationale.
                effective_ratio = max(capacity_ratio_nominal,
                                      self.backend_colocation_floor)
                ratio_source = ('floor'
                                if effective_ratio == self.backend_colocation_floor
                                else 'capacity')

                c_min = dep_cfg.get_min_replicas_for_cluster(cluster_name)
                c_max = dep_cfg.get_max_replicas_for_cluster(cluster_name)
                target = min(c_max, max(c_min,
                                        math.ceil(parent_reps * effective_ratio)))

                # Lettura fresca delle repliche correnti del backend
                current = self.k8s.get_deployment_replicas(
                    cluster=cluster_name,
                    deployment=dep_cfg.deployment_name,
                    namespace=dep_cfg.namespace
                ) or 0

                delta = target - current
                if delta == 0:
                    continue

                # [FIX COSCALE] Budget di churn DEDICATO al co-scaling (separato
                # dal budget reattivo del frontend) — vedi init. Evita che la
                # rampa del frontend affami il co-scaling lasciando i backend a 1.
                if delta > 0 and self._cycle_backend_pods_added >= self.backend_scale_up_budget_per_cycle:
                    logger.info(f"Co-location {cluster_name}/{dep_svc_name}: "
                                f"scale-up Δ={delta:+d} deferred — backend budget esaurito")
                    continue
                if delta < 0 and self._cycle_backend_pods_terminated >= self.backend_scale_down_budget_per_cycle:
                    logger.info(f"Co-location {cluster_name}/{dep_svc_name}: "
                                f"scale-down Δ={delta:+d} deferred — backend budget esaurito")
                    continue

                logger.info(
                    f"Co-location: {cluster_name} — {dep_svc_name}: "
                    f"{current} → {target} (Δ={delta:+d}, frontend={parent_reps}, "
                    f"ratio={effective_ratio:.2f} [{ratio_source}], "
                    f"nominal={capacity_ratio_nominal:.2f}, "
                    f"floor={self.backend_colocation_floor:.2f})"
                )

                self.k8s.scale_deployment(
                    cluster=cluster_name,
                    deployment=dep_cfg.deployment_name,
                    replicas=target,
                    namespace=dep_cfg.namespace
                )

                if delta > 0:
                    self._cycle_backend_pods_added += delta
                    self.last_scale_up[dep_svc_name] = datetime.now()
                    action = 'scale_up_colocation'
                else:
                    self._cycle_backend_pods_terminated += abs(delta)
                    self.last_scale_down[dep_svc_name] = datetime.now()
                    action = 'scale_down_colocation'

                self.scaling_events.labels(
                    cluster=cluster_name, service=dep_svc_name, action=action
                ).inc()
                self.current_replicas.labels(
                    cluster=cluster_name, service=dep_svc_name
                ).set(current)
                self.target_replicas.labels(
                    cluster=cluster_name, service=dep_svc_name
                ).set(target)
    
    def _naive_backend_scaling(self, parent_service: str):
        """
        BASELINE 'naive' (--naive-backends): scala i backend reattivamente sul
        loro rps L7 OSSERVATO via get_request_rate() = hubble_http_requests_total,
        stile HPA: target = ceil(rps / naive_target_rps). Sotto saturazione i retry
        applicativi gonfiano quella metrica -> over-scaling e churn -> e' il
        fallimento metastabile che il co-scaling feed-forward (eq. co-scale) evita.

        Stessi safeguard di _enforce_colocation (churn budget, c_min/c_max) cosi'
        l'UNICA differenza vs l1l2 e' la FONTE del segnale (metrica osservata e
        corrotta, anziche' repliche del frontend).
        """
        if not self.colocation_enabled:
            return
        deps = self.service_dependencies.get(parent_service, [])
        if not deps:
            return
        target_rps = self.naive_target_rps
        for dep_svc_name in deps:
            dep_cfg = self.config.get_service(dep_svc_name)
            if dep_cfg is None:
                continue
            for cluster_name in self.all_cluster_names:
                prom = self.prom_map.get(cluster_name)
                if prom is None:
                    continue
                # rps L7 OSSERVATO del backend su questo cluster, server-side
                # (= hubble_http_requests_total verso il workload del backend).
                # È la metrica gonfiata dai retry sotto saturazione: il cuore del
                # baseline metastabile. Query diretta (reporter="server", workload
                # esatto) perché get_request_rate cadeva a 0 sui backend.
                q = (f'sum(rate(hubble_http_requests_total{{'
                     f'destination_workload="{dep_cfg.deployment_name}",'
                     f'destination_namespace="{dep_cfg.namespace}",'
                     f'reporter="server"}}[1m]))')
                rps = 0.0
                try:
                    res = prom.query(q)
                    if res and res.get('result'):
                        rps = float(res['result'][0]['value'][1])
                except (ValueError, KeyError, IndexError, TypeError):
                    rps = 0.0
                logger.info(f"[NAIVE] {cluster_name}/{dep_svc_name}: "
                            f"rps_L7_osservato={rps:.1f}")
                c_min = dep_cfg.get_min_replicas_for_cluster(cluster_name)
                c_max = dep_cfg.get_max_replicas_for_cluster(cluster_name)
                target = min(c_max, max(c_min, math.ceil(rps / target_rps)))

                current = self.k8s.get_deployment_replicas(
                    cluster=cluster_name,
                    deployment=dep_cfg.deployment_name,
                    namespace=dep_cfg.namespace
                ) or 0
                delta = target - current
                if delta == 0:
                    continue
                if delta > 0 and self._cycle_pods_added >= self.scale_up_budget_per_cycle:
                    continue
                if delta < 0 and self._cycle_pods_terminated >= self.scale_down_budget_per_cycle:
                    continue

                logger.info(
                    f"[NAIVE] {cluster_name} — {dep_svc_name}: {current} → {target} "
                    f"(Δ={delta:+d}, rps_osservato={rps:.1f}, target_rps={target_rps:.0f})"
                )
                self.k8s.scale_deployment(
                    cluster=cluster_name,
                    deployment=dep_cfg.deployment_name,
                    replicas=target,
                    namespace=dep_cfg.namespace
                )
                if delta > 0:
                    self._cycle_pods_added += delta
                    self.last_scale_up[dep_svc_name] = datetime.now()
                    action = 'scale_up_naive'
                else:
                    self._cycle_pods_terminated += abs(delta)
                    self.last_scale_down[dep_svc_name] = datetime.now()
                    action = 'scale_down_naive'
                self.scaling_events.labels(
                    cluster=cluster_name, service=dep_svc_name, action=action).inc()
                self.current_replicas.labels(
                    cluster=cluster_name, service=dep_svc_name).set(current)
                self.target_replicas.labels(
                    cluster=cluster_name, service=dep_svc_name).set(target)

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

        # [FIX R] total_replicas = somma delle necessità per-cluster.
        # Prima: total = max(min, int(global_traffic / global_capacity))
        #   → 75 req/s / 30 = 2 repliche totali → winner determination inutile.
        # Ora: somma ceil(cluster_traffic / cluster_capacity) per ogni cluster,
        # usando la capacità per-cluster (netem-aware). Questo dà al winner
        # determination un budget realistico da distribuire secondo i punteggi
        # multi-obiettivo (latenza, carbon, network, demand). Il tetto per
        # cluster si applica al totale ('total', default) o a ogni cluster
        # ('per_cluster', NCA 2026): vedi entry_budget / DMOS_ENTRY_BUDGET.
        # Predictive scaler: the budget is sized on the demand expected when the
        # replicas requested now are ready (forecast + headroom, >= measured).
        predictive = self.ps is not None and service_name == self.application.entry_service
        budget_traffic = self.ps.demand_for_budget() if predictive else per_cluster_traffic
        if predictive:
            # churn budgets are per decision (the reactive loop resets them per cycle)
            self._cycle_backend_pods_added = 0
            self._cycle_backend_pods_terminated = 0
        total_replicas, cluster_floors = self.entry_budget(
            budget_traffic, self.all_cluster_names, svc_cfg, self.entry_budget_mode)
        if predictive:
            # Level-2 hysteresis on HOW MANY; Level 1 places all of them
            held = self.ps.hold_budget(time.time(), total_replicas)
            if held != total_replicas:
                logger.info(f"PS budget held at {held} (sized {total_replicas}): "
                            f"the demand is not falling")
            total_replicas = held
        logger.info(f"Total replicas needed: {total_replicas} [budget {self.entry_budget_mode}"
                    f"{', predictive' if predictive else ''}] "
                    f"(sum of per-cluster needs from {budget_traffic}); "
                    f"floors={cluster_floors}")

        if self.no_scheduling:
            # ── NO-SCHEDULING: distribuzione uniforme N/K ─────────────────
            # Level 1 disabilitato: nessun punteggio, nessun Hamilton.
            # Le repliche vengono distribuite equamente tra i K cluster.
            # Level 2 (PD controller) rimane attivo per il dimensionamento.
            K = len(self.all_cluster_names)
            split = self.uniform_split(total_replicas, self.all_cluster_names, svc_cfg)
            allocations = [Allocation(cluster_name=c, replicas=r, quota=1.0 / K,
                                      score=1.0 / K)  # score fittizio uniforme
                           for c, r in split.items()]
            success = True
            logger.info(f"[NO-SCHEDULING] Distribuzione uniforme: "
                        f"{', '.join(f'{a.cluster_name}={a.replicas}' for a in allocations)}")
        else:
            allocations, success = self.scheduler.schedule_service(
                service_name=service_name,
                total_replicas=total_replicas,
                predicted_load=sum(budget_traffic.values()) if predictive else current_traffic,
                floors=cluster_floors if self.demand_floor_enabled else None
            )

        if not success:
            logger.error(f"❌ Scheduling fallito")
            return
        
        allocated_cluster_names: Set[str] = set()
        any_scale_down = False

        # ── Step 1: leggi repliche correnti su tutti i cluster in parallelo ──
        # Solo la LETTURA è parallelizzata (operazione idempotente e sicura).
        # Le SCRITTURE (scale_deployment) restano sequenziali per garantire che
        # last_scale_up venga aggiornato prima che _can_scale_down venga valutato
        # per i cluster non allocati nello stesso ciclo: se cluster2 viene
        # scalato UP prima, last_scale_up[service] viene settato e blocca lo
        # scale-down di cluster3 per i successivi scale_up_protection_seconds.
        all_relevant_clusters = list({a.cluster_name for a in allocations} |
                                     set(self.all_cluster_names))
        with ThreadPoolExecutor(max_workers=len(all_relevant_clusters)) as pool:
            rep_futures = {
                pool.submit(
                    self.k8s.get_deployment_replicas,
                    cname, svc_cfg.deployment_name, svc_cfg.namespace
                ): cname
                for cname in all_relevant_clusters
            }
            current_reps_map: Dict[str, int] = {
                rep_futures[f]: (f.result() or 0)
                for f in as_completed(rep_futures)
            }

        if predictive:
            self._apply_predictive_targets(service_name, svc_cfg, allocations, current_reps_map)
            self._enforce_colocation(service_name)
            duration = time.time() - start_time
            self.scheduling_duration.labels(service=service_name).observe(duration)
            logger.info(f"Scheduling completato in {duration:.2f}s")
            return

        # ── Step 2: decisioni e scale-ops sequenziali (SCALE-UP prima) ───────
        for allocation in allocations:
            cluster_name = allocation.cluster_name
            allocated_cluster_names.add(cluster_name)

            self.cluster_score.labels(
                cluster=cluster_name,
                service=service_name
            ).set(allocation.score)

            current_reps = current_reps_map.get(cluster_name, 0)
            self.current_replicas.labels(
                cluster=cluster_name,
                service=service_name
            ).set(current_reps)

            cluster_traffic = per_cluster_traffic.get(cluster_name, 0.0)

            # ── [PATCH-L1L2-HYBRID] Two-tier scaling: L1 strategic + L2 PD fine ──
            # Architettura ibrida feedforward+feedback:
            #
            #  - L1 (strategic, feedforward): il winner determination con score
            #    multi-obiettivo decide il target per-cluster basandosi su
            #    domanda, latenza, capacità, carbon, geo-prossimità. È il
            #    driver primario, riflette il claim del paper.
            #
            #  - L2 (tactical, closed-loop feedback): il PD controller misura
            #    l'errore tra traffico predetto e osservato sul cluster e
            #    produce una correzione fine (Δ ∈ [-3, +3]) basata su
            #    proporzionale + derivativo. Qui la clampiamo a ±1 per
            #    garantire che L1 rimanga il decisore strategico: il PD
            #    può aggiungere o togliere 1 replica per cluster se vede
            #    un errore di forecast significativo (oltre il deadband
            #    ±0.20), ma non può MAI stravolgere l'allocazione L1.
            #
            # Effetto netto: L1 colloca le repliche dove il multi-obiettivo
            # dice che servono; L2 corregge piccoli scostamenti dovuti a
            # imperfezioni della predizione locale di traffico. È la stessa
            # logica feedforward+feedback dei controlli di processo
            # industriali (model-based + correzione integrale-derivativa).
            scaler = self.scalers[service_name][cluster_name]
            c_min = svc_cfg.get_min_replicas_for_cluster(cluster_name)
            c_max = svc_cfg.get_max_replicas_for_cluster(cluster_name)

            # ── Step 2a: L1 STRATEGIC TARGET (feedforward) ────────────────
            strategic_target = max(c_min, min(c_max, allocation.replicas))

            # ── Step 2b: L2 PD FINE CORRECTION (closed-loop, ±1 cap) ─────
            # Il predictor.predict() viene chiamato anche dal periodic_check
            # per il trigger proattivo: qui usiamo lo stesso meccanismo per
            # alimentare il PD controller con (predicted, actual) sul cluster.
            now = datetime.now()
            pred_result, _ = scaler.predictor.predict(
                current_rate=cluster_traffic, timestamp=now
            )
            # PD può restituire delta in [-3, +3] (con deadband ±0.20 interno)
            pd_raw_delta = scaler.controller.compute(
                predicted_traffic=pred_result,
                actual_traffic=cluster_traffic,
                timestamp=now,
            )
            # Clamp a ±1: L1 resta primario, PD aggiunge solo correzione fine
            pd_correction = max(-self.pd_correction_clamp,
                                min(self.pd_correction_clamp, pd_raw_delta))

            target_replicas_raw = max(
                c_min,
                min(c_max, strategic_target + pd_correction)
            )

            # ── Step 2c: Rate-limit anti-thundering-herd (FIX G + I) ──────
            delta_raw = target_replicas_raw - current_reps
            if delta_raw > scaler.max_delta_per_cycle:
                target_replicas = current_reps + scaler.max_delta_per_cycle
            elif delta_raw < -scaler.max_delta_down:
                target_replicas = current_reps - scaler.max_delta_down
            else:
                target_replicas = target_replicas_raw

            # Gauge updates
            self.target_replicas.labels(
                cluster=cluster_name,
                service=service_name
            ).set(target_replicas)
            self.prediction_traffic.labels(
                cluster=cluster_name,
                service=service_name
            ).set(pred_result)

            delta = target_replicas - current_reps
            logger.info(
                f"  L1+L2 placement: {cluster_name} | "
                f"L1 strategic={allocation.replicas} (score={allocation.score:.3f}) | "
                f"L2 PD raw={pd_raw_delta:+d}→clamped={pd_correction:+d} "
                f"(pred={pred_result:.1f}, actual={cluster_traffic:.1f}) | "
                f"→target={target_replicas_raw} | "
                f"rate-lim={target_replicas} (Δ={delta:+d}, current={current_reps})"
            )

            # [FIX C] Dead zone — blocca scale-down quando il traffico è stabile.
            # Con dead_zone_pct=10%:
            #   Picco sostenuto (variazione ~2-5% tra cicli): traffic_stable=True
            #     → blocca delta=-1 → backend restano sovradimensionati (protezione)
            #   Decline reale (variazione >10% tra cicli): traffic_stable=False
            #     → permette delta=-1 → scale-down graduale (max_delta_down=1 + budget)
            # Scale-UP (delta > 0) non viene mai bloccato dalla dead zone.
            if (delta <= 0
                    and abs(delta) <= 1
                    and traffic_stable
                    and current_reps >= svc_cfg.get_min_replicas_for_cluster(cluster_name)):
                logger.info(f" {cluster_name}: Δ={delta:+d} in dead zone — skipping")
                continue

            # [FIX S] Global scale-down budget: max N pod terminati per ciclo.
            # Previene il thundering herd: 15+ pod uccisi in 5s rompevano le
            # connessioni gRPC Envoy L7 e il sistema non recuperava.
            if delta < 0 and self._cycle_pods_terminated >= self.scale_down_budget_per_cycle:
                logger.info(
                    f"  🛡️ {cluster_name}: scale-down Δ={delta:+d} deferred — "
                    f"budget esaurito ({self._cycle_pods_terminated}/"
                    f"{self.scale_down_budget_per_cycle} pod già terminati questo ciclo)"
                )
                continue

            # [FIX A2] Global scale-up budget: max N pod aggiunti per ciclo.
            if delta > 0 and self._cycle_pods_added >= self.scale_up_budget_per_cycle:
                logger.info(
                    f"  🛡️ {cluster_name}: scale-up Δ={delta:+d} deferred — "
                    f"budget esaurito ({self._cycle_pods_added}/"
                    f"{self.scale_up_budget_per_cycle} pod già aggiunti questo ciclo)"
                )
                continue

            # [FIX D] Asymmetric cooldown
            if delta < 0 and not self._can_scale_down(service_name):
                elapsed = (datetime.now() - self.last_scale_down.get(
                    service_name, datetime.now())).total_seconds()
                remaining = self.scale_down_cooldown - elapsed
                logger.info(f"  ⏳ {cluster_name}: scale-down Δ={delta:+d} "
                            f"blocked by cooldown ({remaining:.0f}s left)")
                continue

            logger.info(f"{cluster_name}: {current_reps} → {target_replicas} "
                        f"(Δ={delta:+d})")

            if delta != 0:
                action = 'scale_up' if delta > 0 else 'scale_down'
                self.scaling_events.labels(
                    cluster=cluster_name,
                    service=service_name,
                    action=action
                ).inc()

                # [FIX S/A2] Aggiorna i contatori di budget del ciclo
                if delta < 0:
                    self._cycle_pods_terminated += abs(delta)
                elif delta > 0:
                    self._cycle_pods_added += delta

                self.k8s.scale_deployment(
                    cluster=cluster_name,
                    deployment=svc_cfg.deployment_name,
                    replicas=target_replicas,
                    namespace=svc_cfg.namespace
                )

                if delta < 0:
                    any_scale_down = True
                elif delta > 0:
                    # [FIX E] Aggiorna SUBITO last_scale_up: il loop sui cluster
                    # non-allocati (qui sotto) chiama _can_scale_down che legge
                    # questo valore — deve trovarlo già aggiornato per bloccare
                    # eventuali scale-down nello stesso ciclo di scheduling.
                    self.last_scale_up[service_name] = datetime.now()

        # ── Step 3: scale-down cluster non allocati ───────────────────────────
        # Eseguito DOPO il loop degli allocated, così last_scale_up è già
        # aggiornato dagli scale-up precedenti e _can_scale_down funziona correttamente.
        for cluster_name in self.all_cluster_names:
            if cluster_name not in allocated_cluster_names:
                current_reps = current_reps_map.get(cluster_name, 0)
                target = svc_cfg.get_min_replicas_for_cluster(cluster_name)

                self.current_replicas.labels(
                    cluster=cluster_name,
                    service=service_name
                ).set(current_reps)
                self.target_replicas.labels(
                    cluster=cluster_name,
                    service=service_name
                ).set(min(current_reps, target))

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
                        any_scale_down = True
                    else:
                        logger.debug(f"  ⏳ {cluster_name}: non-alloc scale-down blocked by cooldown")
                else:
                    logger.debug(
                        f"✓ {cluster_name}: not in allocation, "
                        f"already at {current_reps} ≤ {target}"
                    )

        if any_scale_down:
            self.last_scale_down[service_name] = datetime.now()
        
        # ── Sizing backend: co-location feed-forward (default) o naive reattivo ──
        if self.naive_backends:
            self._naive_backend_scaling(service_name)
        else:
            self._enforce_colocation(service_name)
        
        duration = time.time() - start_time
        self.scheduling_duration.labels(service=service_name).observe(duration)
        logger.info(f"Scheduling completato in {duration:.2f}s")
    
    # ── Predictive scaler (DMOS_SCALER=predictive) ────────────────────────

    def _apply_predictive_targets(self, service_name: str, svc_cfg, allocations,
                                  current_reps_map: Dict[str, int]) -> None:
        """Turn the Level-1 allocation into replica counts with the predictive
        policy (immediate scale-up, stabilized and bounded scale-down) and
        apply them. Clusters without an allocation are recommended their
        minimum, through the same stabilization."""
        recommended = {c: svc_cfg.get_min_replicas_for_cluster(c) for c in self.all_cluster_names}
        scores = {}
        for a in allocations:
            recommended[a.cluster_name] = a.replicas
            scores[a.cluster_name] = a.score
            self.cluster_score.labels(cluster=a.cluster_name, service=service_name).set(a.score)
        current = {c: int(current_reps_map.get(c, 0) or 0) for c in self.all_cluster_names}
        now = time.time()
        hold = self.ps.in_burst(now)
        targets = self.ps.targets(now, recommended, current)
        for c in self.all_cluster_names:
            cur, tgt, rec = current[c], targets[c], recommended[c]
            self.current_replicas.labels(cluster=c, service=service_name).set(cur)
            self.target_replicas.labels(cluster=c, service=service_name).set(tgt)
            logger.info(f"  PS {c}: L1={rec} (score={scores.get(c, 0):.3f}) current={cur} "
                        f"-> {tgt}{' [burst hold]' if hold and tgt >= cur and rec < cur else ''}")
            if tgt == cur:
                continue
            action = 'scale_up' if tgt > cur else 'scale_down'
            self.scaling_events.labels(cluster=c, service=service_name, action=action).inc()
            if self.k8s.scale_deployment(cluster=c, deployment=svc_cfg.deployment_name,
                                         replicas=tgt, namespace=svc_cfg.namespace):
                if tgt > cur:
                    self.ps.record_scale_up(c, tgt, now)
                    self.last_scale_up[service_name] = datetime.now()
                else:
                    self.last_scale_down[service_name] = datetime.now()

    def _predictive_tick(self) -> None:
        """One measurement of the predictive scaler: fast and stable demand
        per cluster, current and (for pending scale-ups) ready replicas;
        updates the forecasters and enqueues a decision when one is due."""
        service = self.application.entry_service
        svc_cfg = self.config.get_service(service)
        p = self.ps.p
        kwargs = self.config.ingress_query_kwargs()
        pending = self.ps.pending_clusters()
        fast: Dict[str, float] = {}
        stable: Dict[str, float] = {}
        current: Dict[str, int] = {}
        ready: Dict[str, int] = {}
        n_workers = 3 * len(self.all_cluster_names) + len(pending)
        with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
            futs = {}
            for c, prom in self.prom_map.items():
                futs[pool.submit(prom.get_ingress_rate, window=p.fast_window, **kwargs)] = ("f", c)
                futs[pool.submit(prom.get_ingress_rate, window=p.stable_window, **kwargs)] = ("s", c)
            for c in self.all_cluster_names:
                futs[pool.submit(self.k8s.get_deployment_replicas, c,
                                 svc_cfg.deployment_name, svc_cfg.namespace)] = ("r", c)
            for c in pending:
                futs[pool.submit(self.k8s.get_deployment_ready_replicas, c,
                                 svc_cfg.deployment_name, svc_cfg.namespace)] = ("ready", c)
            for fut in as_completed(futs):
                kind, c = futs[fut]
                try:
                    value = fut.result()
                except Exception as e:
                    logger.warning(f"PS tick: {kind} read failed on {c}: {e}")
                    value = None
                if kind == "f":
                    fast[c] = max(0.0, value or 0.0)
                elif kind == "s":
                    stable[c] = max(0.0, value or 0.0)
                elif kind == "r":
                    current[c] = int(value or 0)
                elif value is not None:
                    ready[c] = int(value)
        now = time.time()
        for c, r in ready.items():
            lag = self.ps.observe_ready(c, r, now)
            if lag is not None:
                logger.info(f"PS actuation lag {c}: {lag:.0f}s -> learned {self.ps.t_ready_s:.0f}s")
        res = self.ps.tick(now, fast, stable, current)
        for c, z in res.clusters.items():
            self.ps_fast_rps.labels(cluster=c).set(z.fast_rps)
            self.ps_forecast_rps.labels(cluster=c).set(z.forecast_rps)
            self.ps_demand_rps.labels(cluster=c).set(z.demand_rps)
            self.ps_need.labels(cluster=c).set(z.need)
            self.prediction_traffic.labels(cluster=c, service=service).set(z.demand_rps)
        self.ps_horizon_s.set(res.horizon_s)
        self.ps_t_ready_s.set(self.ps.t_ready_s)
        self.ps_burst.set(1 if self.ps.in_burst(now) else 0)
        detail = ", ".join(f"{c.rsplit('-', 1)[-1]}={z.fast_rps:.0f}/{z.stable_rps:.0f}"
                           f"->{z.demand_rps:.0f}:{z.raw_need}" for c, z in res.clusters.items())
        if self.ps.budget_mode == "pooled":
            z = self.ps.total_sizing
            detail += (f" | total {z.fast_rps:.0f}/{z.stable_rps:.0f} -> forecast "
                       f"{z.forecast_rps:.0f} + {z.headroom_rps:.0f} = {z.demand_rps:.0f}")
        logger.info(f"PS tick: need {res.need_total} vs current {res.current_total} | "
                    f"H={res.horizon_s:.0f}s (lag {self.ps.t_ready_s:.0f}s) | "
                    f"fast/stable->demand:need {detail}"
                    f"{' | BURST' if res.burst else ''}{' | decide: ' + res.reason if res.decide else ''}")
        in_grace = (datetime.now() - self.startup_time).total_seconds() < self.startup_grace_seconds
        if res.decide and not in_grace:
            self.event_queue.put(SchedulingEvent(
                priority=0 if res.reason in ("burst", "need_up") else 1,
                timestamp=datetime.now(), service=service,
                action='scale_up' if res.reason in ("burst", "need_up") else 'schedule',
                reason=f'ps_{res.reason}'))

    # ── Event Processing ──────────────────────────────────────────────────
    
    def process_event(self, event: SchedulingEvent):
        if not self.should_process(event.service):
            logger.debug(f"Skipping {event.service} (debounced)")
            return

        logger.info(f"Processing: {event.action} {event.service}")

        # [FIX L] Aggiorna last_processed PRIMA di chiamare schedule_service,
        # non dopo. Con più worker in parallelo, se due worker dequeue-ano eventi
        # per lo stesso servizio quasi contemporaneamente, entrambi passano il
        # controllo should_process() prima che il primo abbia finito (last_processed
        # veniva aggiornato solo al completamento). Il risultato era una doppia
        # esecuzione concorrente di schedule_service sullo stesso servizio:
        # le due k8s calls scrivevano target diversi e l'ultima sovrascriveva
        # la prima (es. cluster2/productcatalogservice: 8 → 3 a 13:20:14).
        # Settando last_processed subito, il secondo worker vedrà che il debounce
        # è già scattato e salterà il processing.
        self.last_processed[event.service] = datetime.now()

        try:
            self.schedule_service(event.service, reason=event.reason)
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
            if self.ps is not None:
                try:
                    self._predictive_tick()
                except Exception as e:
                    import traceback
                    logger.error(f"Errore nel tick del predictive scaler: {e}")
                    logger.error(traceback.format_exc())
                time.sleep(self.ps.p.tick_s)
                continue
            try:
                # [FIX S/A2] Reset budget scale-down/up ad ogni ciclo di polling
                self._cycle_pods_terminated = 0
                self._cycle_pods_added = 0
                # [FIX COSCALE] reset del budget dedicato al co-scaling backend
                self._cycle_backend_pods_added = 0
                self._cycle_backend_pods_terminated = 0

                for service_name in self.monitored_services:
                    per_cluster_traffic = self._get_per_cluster_traffic(service_name)
                    current_traffic = sum(per_cluster_traffic.values())

                    svc_cfg = self.config.get_service(service_name)
                    # Usa il primo cluster definito in local.yaml come riferimento
                    # per la lettura delle repliche correnti (era hardcoded 'cluster1').
                    # Il primo cluster è quello con netem 0ms nel testbed thesis, ma
                    # in produzione N-cluster qualsiasi cluster va bene perché DMOS
                    # bilancia tutto attraverso lo scheduling.
                    first_cluster = self.all_cluster_names[0]
                    current_replicas = self.k8s.get_deployment_replicas(
                        cluster=first_cluster,
                        deployment=svc_cfg.deployment_name,
                        namespace=svc_cfg.namespace
                    )

                    if current_replicas is None:
                        current_replicas = 1

                    if current_traffic == 0:
                        logger.warning(f"Nessuna metrica Prometheus per {service_name}, uso stima di fallback")

                        fe_reps = 0
                        fe_cfg = self.config.entry_service_config()
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
                        
                # [FIX option-1] Rimossi i blocchi "Backend reset" e
                # "scale-down graduale backend": con la co-location come unico
                # dimensionatore bidirezionale i backend inseguono sempre il
                # frontend e non possono andare alla deriva -> quelle reti di
                # sicurezza sono obsolete (e farebbero a pugni con la co-location).
                            
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
                'workers': self.num_workers,
                'config_id': self.config_id,
            }), 200

        @app.route('/config', methods=['GET'])
        def effective_config():
            # Provenance: the run harness fetches this into the run directory
            # whatever way DMOS was started (also standalone / by hand).
            return jsonify(self.effective_config), 200

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
    import argparse
    parser = argparse.ArgumentParser(description="DMOS Orchestrator")
    parser.add_argument(
        "--no-scheduling", action="store_true",
        help="Disabilita Level 1 (scheduling geo-aware). "
             "Le repliche vengono distribuite uniformemente N/K. "
             "Level 2 (scaling PD controller) resta attivo."
    )
    parser.add_argument(
        "--naive-backends", action="store_true",
        help="BASELINE metastabile: scala i backend reattivamente su "
             "hubble_http_requests_total (stile HPA) invece del co-scaling "
             "feed-forward. L1+L2 (frontend) restano attivi."
    )
    parser.add_argument(
        "--config-dump", metavar="PATH",
        default=os.environ.get("DMOS_CONFIG_DUMP") or None,
        help="Write the effective configuration (JSON, with config_id) to PATH "
             "at startup. Default: $DMOS_CONFIG_DUMP. The run harness points "
             "this into the run directory."
    )
    parser.add_argument(
        "--dump-config-only", metavar="PATH",
        help="Build the effective configuration, write it to PATH, print the "
             "summary and exit. Opens no port and contacts no cluster: used to "
             "check an arm's configuration offline before a run."
    )
    args = parser.parse_args()

    if args.dump_config_only:
        orchestrator = DMOSOrchestrator(
            config_path="config",
            num_workers=3,
            no_scheduling=args.no_scheduling,
            naive_backends=args.naive_backends,
            dry_run=True,
            config_dump_path=args.dump_config_only,
        )
        print(orchestrator.effective_config["summary"])
        return

    orchestrator = DMOSOrchestrator(
        config_path="config",
        num_workers=3,
        no_scheduling=args.no_scheduling,
        naive_backends=args.naive_backends,
        config_dump_path=args.config_dump,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()