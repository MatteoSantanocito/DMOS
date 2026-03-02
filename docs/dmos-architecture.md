# DMOS — Architettura del Sistema
## Guida completa all'infrastruttura e ai componenti

> Documento aggiornato il 01/03/2026.
> Descrive l'architettura fisica, la rete, i componenti DMOS e il flusso
> end-to-end dell'orchestrazione multi-cluster carbon-aware.

---

## Indice

1. [Overview](#1-overview)
2. [Setup fisico](#2-setup-fisico)
3. [Applicazione: Online Boutique](#3-applicazione-online-boutique)
4. [Infrastruttura di rete](#4-infrastruttura-di-rete)
5. [Osservabilità](#5-osservabilità)
6. [DMOS — componenti](#6-dmos--componenti)
7. [Flusso end-to-end](#7-flusso-end-to-end)
8. [Porte e URL di riferimento](#8-porte-e-url-di-riferimento)
9. [Esperimenti e profili carbon](#9-esperimenti-e-profili-carbon)
10. [Limitazioni note](#10-limitazioni-note)

---

## 1. Overview

DMOS (Dynamic Multi-cluster Orchestration System) è un orchestratore centralizzato
che gestisce il deployment di Online Boutique su tre cluster Kubernetes simulando
un ambiente multi-regione europeo con diversa intensità di carbonio.

**Obiettivi**:
- Scaling reattivo/proattivo basato sul traffico reale per-cluster
- Carbon-aware scheduling: preferisce cluster a bassa intensità di CO₂
- Multi-objective: bilancia latenza, capacità, carico, costo e carbonio

**Approccio "PROM_MAP"**: ogni cluster ha il suo Prometheus locale (port 30090).
DMOS interroga ciascun Prometheus separatamente per metriche accurate per-cluster.

**Sorgente metriche primaria**: **Hubble L7** (Cilium observability layer) via
`hubble_http_requests_total` — contatore HTTP esatto per pod, abilitato dalla
CiliumNetworkPolicy con `rules: http` sul pod frontend.

```
        ┌─────────── LAN 192.168.1.0/24 ──────────────┐
        │                                               │
   [ms01:245]       [ms02:246]       [ms03:247]        │
   k3s cluster1    k3s cluster2    k3s cluster3        │
   EU-DE (350 gCO₂) EU-FR (80 gCO₂) EU-PL (650 gCO₂) │
        │                │                │             │
        └────────────────┴────────────────┘             │
                         │                              │
               [DMOS host — Windows]                    │
               localhost (192.168.1.x)                 │
               ├─ dmos_main.py → :9090                 │
               ├─ locust → :8089                       │
               └─ kubectl, kubeconfig per cluster      │
        └────────────────────────────────────────────┘
```

---

## 2. Setup fisico

### Nodi

| Nodo | IP | Cluster k3s | Regione simulata | Carbon Intensity |
|------|-----|-------------|-----------------|-----------------|
| ms01 | 192.168.1.245 | cluster1 | EU-DE (Germania) | ~350 gCO₂/kWh |
| ms02 | 192.168.1.246 | cluster2 | EU-FR (Francia) | ~80 gCO₂/kWh |
| ms03 | 192.168.1.247 | cluster3 | EU-PL (Polonia) | ~650 gCO₂/kWh |

Ogni nodo è un **k3s single-node cluster**: un solo nodo fa sia da control
plane che da worker. Tutti i pod vengono schedulati sullo stesso nodo.

### Risorse per nodo

```yaml
# config/clusters.yaml
resources:
  cpu_cores: 4
  memory_gb: 8
```

### CNI: Cilium

I cluster usano **Cilium** come CNI (Container Network Interface):
- Pod CIDR: `10.42.0.0/16` (k3s default)
- Service CIDR: `10.43.0.0/16`
- kube-proxy replacement: abilitato (Cilium gestisce il routing dei Service)
- Hubble: abilitato (observability layer Cilium per metriche HTTP L7)

---

## 3. Applicazione: Online Boutique

Online Boutique è un'applicazione e-commerce microservizi di Google, usata
come workload di test. Tutti i microservizi girano nel namespace `online-boutique`.

### Microservizi gestiti da DMOS

| Servizio | Deployment | Capacità/replica | Min | Max |
|----------|------------|-----------------|-----|-----|
| frontend | frontend | 30 rps | 1 | 20 |
| cartservice | cartservice | 10 rps | 1 | 20 |
| productcatalogservice | productcatalogservice | 15 rps | 1 | 20 |
| checkoutservice | checkoutservice | 5 rps | 1 | 20 |
| recommendationservice | recommendationservice | 10 rps | 1 | 20 |

### Dipendenze dei servizi (co-location)

```
frontend → cartservice
         → productcatalogservice
         → checkoutservice
         → recommendationservice
```

DMOS implementa **co-location proporzionale**: se il frontend ha X repliche su
un cluster, i backend vengono distribuiti proporzionalmente tra i cluster
attivi con frontend.

### Traffico esterno

Il traffico utente entra tramite **Nginx Ingress Controller** (namespace
`ingress-nginx`) su porta NodePort **30080**, viene instradato al frontend.

```
Locust → 192.168.1.X:30080 (Nginx NodePort)
             ↓
         Nginx pod (ingress-nginx namespace)
             ↓  [ClusterIP, via service-upstream: true]
         frontend Service (10.43.x.x:80)
             ↓  [Cilium endpoint routing]
         frontend pod (10.42.0.x:8080)
             ↓  [Envoy L7 proxy intercetta per Hubble]
         Risposta HTTP 200
```

---

## 4. Infrastruttura di rete

### Nginx Ingress Controller

Deploy: `kubectl apply -f https://...ingress-nginx/.../baremetal/deploy.yaml`

```yaml
# deployments/ingress-frontend.yaml
metadata:
  annotations:
    nginx.ingress.kubernetes.io/service-upstream: "true"
    # ↑ CRITICO: usa ClusterIP invece di pod IP diretto
    #   Necessario con Cilium per evitare problemi same-node e con CNP
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "10"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
```

**Porta NodePort**: 30080 (HTTP), fissata con `kubectl patch svc`.

### CiliumNetworkPolicy (CNP) — stato attuale

La CNP `l7-visibility-frontend` (namespace `online-boutique`) protegge il pod
frontend **e abilita la visibilità L7 Hubble** tramite il blocco `rules: http`:

```yaml
# deployments/cnp-l7-frontend.yaml — versione finale con Hubble L7
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: l7-visibility-frontend
  namespace: online-boutique
spec:
  endpointSelector:
    matchLabels:
      app: frontend

  # Nessuna sezione egress → default allow all egress
  # (evita i bug con ClusterIP scope descritti in cilium-hubble-l7-visibility.md)

  ingress:
  - fromEntities:
    - cluster   # tutti i pod del cluster (incluso ingress-nginx)
    - host      # nodo locale (kubelet health check)
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - {}   # ← CHIAVE: regola HTTP vuota → attiva Envoy L7 proxy
               #           → Hubble inizia a contare hubble_http_requests_total
               #           → DMOS usa questa metrica come sorgente primaria
```

**Effetti della CNP**:
- ✅ `192.168.1.X:30080` (via Nginx, entity=cluster) → 200
- ✅ `hubble_http_requests_total` popolata da Envoy per ogni richiesta
- ❌ `192.168.1.X:30007` (NodePort diretto, entity=world) → timeout/bloccato
- ✅ Frontend → gRPC backend → funziona (no egress restrictions)

**Perché `world` non va in timeout con L7 attivo**: il traffico `world` viene
bloccato a **L3** dalla CNP (non è in `fromEntities`) prima di raggiungere Envoy.
Envoy riceve solo traffico `cluster` (Nginx) e `host` (kubelet), che gestisce
correttamente anche in configurazione same-node.

Vedi `docs/cilium-hubble-l7-visibility.md` per la trattazione tecnica completa.

---

## 5. Osservabilità

### Prometheus (per-cluster)

Ogni cluster ha un Prometheus locale che scrapa i pod del suo cluster:

| Cluster | URL | Scraping |
|---------|-----|---------|
| cluster1 | `http://192.168.1.245:30090` | Pod in `online-boutique`, node metrics, Hubble-metrics |
| cluster2 | `http://192.168.1.246:30090` | Pod in `online-boutique`, node metrics, Hubble-metrics |
| cluster3 | `http://192.168.1.247:30090` | Pod in `online-boutique`, node metrics, Hubble-metrics |

**Metriche chiave usate da DMOS**:

| Metrica | Sorgente | Usata in |
|---------|----------|---------|
| `hubble_http_requests_total` | Cilium/Envoy | `get_request_rate()` Try 1 (primaria) |
| `hubble_http_request_duration_seconds_bucket` | Cilium/Envoy | `get_latency_p95()` Try 1 |
| `container_network_receive_bytes_total` | cAdvisor | `get_request_rate()` Try 4 (fallback) |
| `container_cpu_usage_seconds_total` | cAdvisor | `get_cpu_available()` |
| `kube_node_status_capacity` | kube-state-metrics | `get_cpu_available()`, `get_memory_available_gb()` |
| `container_memory_working_set_bytes` | cAdvisor | `get_memory_available_gb()` |

### Hubble (Cilium observability)

Hubble è l'observability layer di Cilium. **Stato attuale**: **attivo e funzionante**
per il traffico Nginx → Frontend grazie alla CNP con `rules: http`.

```bash
# Verifica traffico L7 in tempo reale (tutti i cluster)
kubectl exec -n kube-system <cilium-pod> --context cluster1 \
  -- hubble observe --type l7 --namespace online-boutique --last 20

# Output atteso (con Locust attivo):
# controller (ingress-nginx) → frontend:8080 HTTP/1.1 200 15ms GET /
# controller (ingress-nginx) → frontend:8080 HTTP/1.1 200 18ms GET /product/ID
```

**Metriche Prometheus da Hubble** (scraping automatico su ogni cluster):
```promql
# Request rate per frontend (usato da DMOS)
sum(rate(hubble_http_requests_total{
    destination_workload="frontend",
    destination_namespace="online-boutique"
}[5m]))

# Latency p95
histogram_quantile(0.95,
  sum(rate(hubble_http_request_duration_seconds_bucket{
      destination_namespace="online-boutique"
  }[5m])) by (le)
) * 1000
```

### DMOS metrics server

DMOS espone le sue metriche su `localhost:9090/metrics`:

| Metrica | Tipo | Descrizione |
|---------|------|-------------|
| `dmos_actual_traffic{service}` | Gauge | Traffico totale corrente (req/s) |
| `dmos_predicted_traffic{cluster,service}` | Gauge | Traffico previsto dal predictor |
| `dmos_current_replicas{cluster,service}` | Gauge | Repliche attuali su k8s |
| `dmos_target_replicas{cluster,service}` | Gauge | Repliche target calcolate |
| `dmos_cluster_score{cluster,service}` | Gauge | Score multi-obiettivo del cluster |
| `dmos_scaling_events_total{cluster,service,action}` | Counter | Scaling events totali |
| `dmos_scheduling_duration_seconds{service}` | Histogram | Durata ciclo scheduling |

> **Nota**: durante la startup grace period (primi 90s), `dmos_current_replicas`
> viene aggiornata con le repliche k8s reali (min=1 per cluster, visibile nel monitor).
> `dmos_predicted_traffic` rimane a 0 (nessuna predizione durante grace period).

---

## 6. DMOS — componenti

### Architettura interna

```
DMOSOrchestrator (dmos_main.py)
    │
    ├─ DMOSScheduler (Level 1) ──────────────────────────────────────
    │    │  Seleziona cluster e distribuisce le repliche
    │    ├─ ScoreFunctions     → Φ = ω₁·lat + ω₂·cap + ω₃·load + ω₄·carbon
    │    ├─ WinnerDetermination → allocations = distribute(total_replicas, scores)
    │    └─ CarbonClient       → recupera CI(t) da Electricity Maps API
    │
    ├─ ReplicaScaler[svc][cluster] (Level 2) ────────────────────────
    │    │  Calcola target_replicas per ogni cluster separatamente
    │    ├─ TrafficPredictor   → trend-based prediction (EMA + derivata)
    │    └─ PDController       → correzione PD: Kp=5.0, Kd=300.0
    │
    ├─ prom_map[cluster] ────────────────────────────────────────────
    │    │  Un PrometheusClient per cluster (PROM_MAP approach)
    │    └─ get_request_rate() → fallback chain:
    │         Try 1: Hubble L7 → hubble_http_requests_total [5m] ✅
    │         Try 2: Istio     → istio_requests_total         ❌
    │         Try 3: HTTP      → http_requests_total           ❌
    │         Try 4: Network   → container_network_bytes/4000  ✅ (fallback)
    │
    ├─ KubernetesClient ─────────────────────────────────────────────
    │    │  Esegue scale_deployment() su tutti i cluster
    │    └─ get_deployment_replicas()
    │
    ├─ Webhook Server (Flask :8081) ─────────────────────────────────
    │    └─ POST /webhook/alert → Prometheus AlertManager trigger
    │
    └─ Periodic Polling Thread (ogni 30s) ───────────────────────────
         └─ Controlla traffico → emette eventi in PriorityQueue
```

### Level 1 — Cluster Selection (DMOSScheduler)

Sceglie **quali cluster** ricevono repliche e quante, usando una funzione
di score multi-obiettivo:

```
score_i = ω_lat · Φ_latency(i)
        + ω_cap · Φ_capacity(i)
        + ω_load · Φ_load(i)
        + ω_carbon · Φ_carbon(i)
```

Profili di peso configurati in `config/weights.yaml`:

| Profilo | ω_lat | ω_cap | ω_load | ω_carbon |
|---------|-------|-------|--------|---------|
| carbon_agnostic | 0.45 | 0.35 | 0.20 | 0.00 |
| **balanced** (default) | 0.35 | 0.25 | 0.15 | 0.25 |
| carbon_priority | 0.25 | 0.20 | 0.15 | 0.40 |

**Φ_carbon**: calcolato con CI reale da Electricity Maps API:
```
Φ_carbon(i) = exp(-ν · CI_i / CI_max)
```
Con ν=0.5 e CI_max=800, il cluster FR (80 gCO₂) ha score molto più alto di PL (650 gCO₂).

### Level 2 — Autoscaling (ReplicaScaler)

Calcola le repliche target per ogni cluster:

```python
# Per ogni cluster, scaler indipendente:
decision = scalers["frontend"]["cluster1"].compute_target_replicas(
    current_replicas=current,
    current_traffic=cluster1_traffic   # dato Hubble, non stima
)
```

**Pipeline interna**:
1. `TrafficPredictor.predict()` → stima traffico tra 10 minuti (EMA + derivata)
2. `base_replicas = predicted / capacity_per_replica` (30 rps/pod per frontend)
3. `safety_replicas = ceil(base * 1.15)` (15% safety margin)
4. `pd_adj = PDController.compute(error, derivative)` (Kp=5, Kd=300)
5. `target = clamp(safety + pd_adj, min=1, max=20)`

**Anti-oscillation**:
- Dead zone: salta ±1 Δ se traffico stabile (<15% variazione)
- Debounce: 30s tra scheduling dello stesso servizio
- Scale-down cooldown: 60s — dopo un scale-down, nessun altro scale-down per 60s
- **Scale-up protection: 120s** — dopo ogni scale-up, nessun scale-down per 120s
  *(previene il pattern "scala su → scala giù subito" durante il ramp-up)*
- max_delta_per_cycle: 4 repliche per ciclo

### Startup Grace Period

Nei primi **90 secondi** dall'avvio, DMOS monitora il traffico ma blocca
tutti gli ordini di scaling k8s:

```python
self.startup_grace_seconds = 90  # seconds

# In schedule_service():
elapsed_startup = (datetime.now() - self.startup_time).total_seconds()
if elapsed_startup < self.startup_grace_seconds:
    # Traffic still measured → predictor history accumulates
    # k8s ops blocked → no premature scaling
    return
```

**Motivazione**: evita scaling prematuro basato su derivate calcolate su 1–2 punti
(molto rumorose). Dopo 90s il predittore ha storia sufficiente per decisioni stabili.

### Event-driven + Polling

```
            ┌─ Prometheus Alert ──→ webhook :8081
            │
PriorityQueue ─┼─ Polling ogni 30s ──→ traffic check
(events)    │
            └─ Manual API ──────→ POST /api/schedule

            ↓  3 worker threads
        process_event() → schedule_service()
```

Priorità eventi: 0=critico (proattivo), 1=warning, 2=info (scale-down).

---

## 7. Flusso end-to-end

### Flusso del traffico utente

```
Locust (localhost)
    │
    ▼
192.168.1.X:30080   ← NodePort Nginx Ingress
    │
    ▼
Nginx pod (ingress-nginx ns)
    │  [service-upstream: true → usa ClusterIP]
    ▼
frontend Service (10.43.x.x:80)
    │  [Cilium endpoint routing]
    ▼
frontend pod (10.42.0.x:8080)
    │  [Envoy L7 proxy — CNP rules: http → intercetta]
    │  → hubble_http_requests_total++ per ogni richiesta
    │
    ├─ gRPC → cartservice ClusterIP
    ├─ gRPC → productcatalogservice ClusterIP
    ├─ gRPC → checkoutservice ClusterIP
    └─ gRPC → recommendationservice ClusterIP
```

### Flusso del ciclo DMOS

```
ogni 30s (polling):
    1. Per ogni cluster: query Hubble su Prometheus → rps esatto per-servizio
    2. Predictor per-cluster: stima rps tra 10 min (EMA + derivata)
    3. effective_traffic = max(current, predicted)
       ECCEZIONE — traffic floor: se 0 < current < 2 rps → effective_traffic = current
       (bypassa il predictor: evita che EMA decadente tenga pod attivi per 15+ min dopo
        la fine del test)
    4. Se > high_threshold (30 rps) → emette evento scale_up
    5. Se < low_threshold (10 rps)  → emette evento scale_down

startup grace period (primi 90s):
    - I passi 1-2 vengono eseguiti normalmente (accumulo storia)
    - I passi 4-5 vengono bloccati (nessun evento emesso / nessuna azione k8s)

worker thread (event, dopo 90s):
    1. schedule_service(frontend):
       a. _get_per_cluster_traffic() → {c1: r1, c2: r2, c3: r3} via Hubble
       b. total = r1 + r2 + r3
       c. total_replicas = ceil(total / 30)
       d. DMOSScheduler.schedule_service(total_replicas) → allocations
          - Calcola score(c1), score(c2), score(c3)
          - WinnerDetermination → distribuisce repliche per score
       e. Per ogni cluster allocato:
          - ReplicaScaler[c].compute_target_replicas(r_i)
          - k8s.scale_deployment(c, target)
       f. _enforce_colocation(frontend) → backend proporzionali
```

---

## 8. Porte e URL di riferimento

### Per cluster (stessa struttura su ms01/02/03)

| Servizio | Porta | URL esempio |
|----------|-------|-------------|
| Nginx Ingress (HTTP) | 30080 | `http://192.168.1.245:30080/` |
| Prometheus | 30090 | `http://192.168.1.245:30090/` |
| Nginx metrics | 10254 | `http://192.168.1.245:10254/metrics` |
| frontend NodePort (bloccato da CNP) | 30007 | `http://192.168.1.245:30007/` → timeout |

### DMOS host (localhost)

| Servizio | Porta | URL |
|----------|-------|-----|
| DMOS metrics (Prometheus) | 9090 | `http://localhost:9090/metrics` |
| DMOS webhook | 8081 | `http://localhost:8081/webhook/alert` |
| DMOS health | 8081 | `http://localhost:8081/health` |
| Locust web UI | 8089 | `http://localhost:8089` |
| Locust stats API | 8089 | `http://localhost:8089/stats/requests` |

---

## 9. Esperimenti e profili carbon

### Ordine di avvio consigliato

**IMPORTANTE**: avviare prima DMOS, poi Locust dopo 90s.

```powershell
# Terminale 1 — DMOS (grace period con traffico idle ≈ 0)
python dmos_main.py

# Terminale 2 — collect_metrics (avviare subito dopo DMOS, cattura tutto)
python experiments/collect_metrics_simple.py 1560 --scenario double_wave_hubble

# Aspettare ~90 secondi che la grace period termini, poi:

# Terminale 3 — Locust (--autostart: avvia test E mantiene web server :8089)
locust -f experiments/locustfile_multiingress.py `
  --autostart --users 300 --spawn-rate 10 `
  --web-host 0.0.0.0 --web-port 8089 `
  --run-time 26m
```

> **Perché DMOS prima di Locust**: durante la grace period (90s) DMOS accumula
> storia del predictor con traffico idle (~0 rps). Quando Locust parte la derivata
> sale subito → proactive scaling immediato. Se Locust parte prima, i primi 90s
> di carico reale trovano il sistema a min_replicas=1 per cluster (non scala) →
> p95 alto e test sprecato.

> **Nota**: usare `--autostart` invece di `--headless`. Con `--headless` il web server
> Locust su :8089 non parte e `collect_metrics` non può leggere le stats Locust.

### Locustfile disponibili

| File | Scopo |
|------|-------|
| `locustfile_multiingress.py` | Multi-cluster con TaskSet, scenari variabili |
| `locustfile_capacity.py` | Capacity test graduale per misurare rps/pod |
| `locustfile_scenarios.py` | Flash crowd, gradual ramp, wave |
| `locustfile_variable.py` | Pattern variabili per LSTM |

### Scenari di carico (locustfile_multiingress.py)

```powershell
$env:DMOS_SCENARIO="flash_crowd"     # burst improvviso
$env:DMOS_SCENARIO="gradual_ramp"    # rampa graduale
$env:DMOS_SCENARIO="steady_state"    # carico costante
$env:DMOS_SCENARIO="double_wave"     # doppia onda (26 min)
locust -f experiments/locustfile_multiingress.py --autostart ...
```

### Profili carbon (config/weights.yaml)

```powershell
# Per cambiare profilo: modifica config/weights.yaml → active: "..."
# carbon_agnostic: ignora CO₂, massimizza capacità+latenza
# balanced:        equilibrio (default per tesi)
# carbon_priority: massimizza green, accetta più latenza
```

### Collector metriche

```powershell
python experiments/collect_metrics_simple.py [durata_sec] --scenario [nome]
# Legge ogni 15s: DMOS metrics (:9090), Locust stats (:8089), scrive JSONL + TXT

# Durata consigliata: 90s (grace) + durata scenario
# Per double_wave (26 min): 90 + 1560 = 1650s
python experiments/collect_metrics_simple.py 1650 --scenario double_wave_hubble

# I primi 90s avranno replicas=0 e pred=0.0 (grace period, escludibili in post)
# ma Locust non è ancora partito → rps Locust = 0 → nessun dato sprecato
```

---

## 10. Limitazioni note

### Misura traffico backend approssimata

I servizi backend (cartservice, productcatalog, checkoutservice, recommendationservice)
non hanno CNP L7 → `hubble_http_requests_total` non disponibile per gRPC.
DMOS usa `container_network_receive_bytes / 4000` come stima.

**Impatto**: DMOS può sotto/sovra-stimare il carico backend del 20-30%.
**Mitigazione**: la safety margin del 15% compensa le sottostime sistematiche.
**Fix futuro**: aggiungere CNP L7 sulle porte gRPC dei backend (es. 7000 cartservice).

### Carbon intensity simulata

I cluster sono fisicamente nella stessa LAN.
Le regioni (DE/FR/PL) sono simulate via `config/clusters.yaml`.
Se Electricity Maps API non è disponibile, usa `baseline_gco2_kwh` da config.

### Anti-oscillation conservativo

Debounce 30s + scale-down cooldown 60s + dead zone 15% rendono DMOS
più lento a reagire ai cambi di carico. Calibrato per evitare oscillazioni
nei test di capacity, ma può essere troppo conservativo per flash crowd rapidi.

### Scrape interval Hubble e finestra rate()

**Configurazione attuale**: Hubble-metrics viene scraping ogni **~60s**.
Con la query `rate([5m])` ci sono 4-5 sample → stabile.

**Regola generale**: `window_rate ≥ 2 × scrape_interval`

| scrape_interval | window minima | window usata | ritardo ramp-up |
|---|---|---|---|
| 60s (attuale) | 120s | `[5m]` (300s) | ~5 minuti |
| 15s (ottimizzato) | 30s | `[1m]` (60s) | ~1 minuto |
| 2m (conservativo) | 4m | `[10m]` | ~10 minuti |

#### Cambiare scrape_interval a 15s — pro e contro

**✅ Pro:**
- Ritardo ramp-up FE: da 5 minuti a ~1 minuto (flash crowd rilevabile in ~2 cicli DMOS)
- `rate([1m])` riflette il carico istantaneo, non una media di 5 minuti
- Nessuna modifica architetturale: 1 riga Prometheus config + 1 riga `prometheus_client.py`
- Il costo su k3s è trascurabile: ~3 scrape extra/min × ~10ms/scrape ≈ +0.1% CPU

**⚠️ Contro:**
- `rate([1m])` ha meno smoothing: cattura transitori brevi (burst probe, spike 15-30s).
  In pratica è gestito dai guard esistenti (dead zone 15%, scale-up protection 120s)
- 4x più campioni Hubble in TSDB: +~600 KB/giorno (trascurabile con ~100 serie attive)
- Uno scrape fallito pesa 25% del rate con [1m] vs 5-10% con [5m].
  Su nodi sotto pressione CPU, questo può causare anomalie brevi

**Non applicare in produzione** con alta cardinality (migliaia di pod): il costo
di 4x scraping del metrics endpoint Cilium diventa significativo.

**Come applicare la modifica** (su ogni cluster):
```bash
# 1. Trova il job di scraping Hubble
kubectl get configmap -n monitoring --context cluster1

# 2a. Con ConfigMap standalone — modifica il blocco job_name: 'hubble':
#   scrape_interval: 15s   (invece di 60s o del global default)

# 2b. Con kube-prometheus-stack — patch del PodMonitor:
kubectl patch podmonitor hubble-metrics -n monitoring --context cluster1 \
  --type='json' \
  -p='[{"op":"replace","path":"/spec/podMetricsEndpoints/0/interval","value":"15s"}]'
```

```python
# 3. Aggiorna la query in src/metrics/prometheus_client.py
# Cambia [5m] → [1m] nel Try 1 Hubble
query = f'sum(rate(hubble_http_requests_total{{...}}[1m]))'
```

Vedi §4 di `dmos-traffic-measurement.md` per la guida completa con verifica.

### Ritardo ramp-up "FE" all'inizio del test (~5 minuti)

La metrica `dmos_actual_traffic` (indicata come **FE** nel monitor) usa
`rate(hubble_http_requests_total[5m])`. Nei primi **~5 minuti** dopo l'avvio
di Locust, FE mostra un valore molto più basso del traffico reale:

```
t=+1min:  FE=0.6 rps,  Locust rps=9.7   ← finestra [5m] ancora "diluita" da idle
t=+5min:  FE≈14 rps,   Locust rps=24    ← finestra [5m] quasi interamente a piena carico
t=+7min:  FE≈34 rps,   Locust rps=24    ← FE sopra Locust (moltiplicatore 1.43×, §6 traffic-measurement.md)
```

**Causa**: la finestra di 5 minuti include campioni sia del periodo idle pre-Locust
(~0.6 rps da kubelet probe) sia del periodo di carico. Man mano che passa il tempo,
i campioni idle escono dalla finestra e FE converge al valore reale.

**Perché non è un bug critico**: il `TrafficPredictor` usa la **derivata** del
traffico, non solo il valore assoluto. Con FE che sale velocemente (derivata alta),
il predictor anticipa il picco e triggera lo scaling. La scale-up protection (120s)
previene l'oscillazione nella fase di ramp-up.

**Impatto sull'analisi**: `mape_active` (vedi `analyze_test_complete.py`) usa una
soglia `max(5, peak × 10%)` per escludere i primi campioni dove FE è ancora basso.
Questo evita che il ritardo ramp-up distorca il MAPE dell'intera sessione.

**Fix futuro**: Prometheus scrape_interval Hubble a 15s → `rate([1m])` con 4 sample
garantiti → ritardo ridotto da 5 minuti a ~1 minuto.
