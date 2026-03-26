# Two-Phase Scheduling in DMOS

## Motivazione

Il sistema DMOS deve prendere decisioni di scheduling al momento del deploy, quando le
metriche osservate (latenza p95, utilizzo CPU, request rate) non sono ancora
statisticamente significative. Usare queste metriche a freddo produce stime rumorose
che portano a decisioni di scheduling errate nei primi cicli di controllo.

La soluzione è separare il ciclo di vita dello scheduler in due fasi distinte, ciascuna
con segnali e obiettivi propri. Questo approccio è ispirato alla distinzione
**"offline phase + online phase"** presente in letteratura (FIRM OSDI'20, Sinan
ASPLOS'21, Cilantro OSDI'23), adattato al contesto multi-cluster con Nginx Ingress e
Cilium Hubble.

---

## Φ_response_time vs Φ_lat di Romano

Prima di descrivere le fasi è necessario chiarire una distinzione terminologica
importante rispetto alla tesi di Romano.

Romano chiama **Φ_lat** la componente di latenza del suo score. La misura tramite
**ping RTT** tra cluster: è una stima della distanza geografica, non della latenza
effettiva del servizio.

In DMOS questa componente si chiama **Φ_response_time** e misura qualcosa di
fondamentalmente diverso:

| | Romano (Φ_lat) | DMOS (Φ_response_time) |
|-|----------------|------------------------|
| **Fonte** | ping ICMP inter-cluster | Hubble `hubble_http_request_duration_seconds_bucket` |
| **Cosa misura** | RTT di rete tra nodi | Latenza HTTP p95 osservata sui pod reali |
| **Include** | Solo propagation delay | Coda, processing time, dipendenze via mesh |
| **Finestra** | Strutturale (non cambia con il traffico) | `[5m]` su Prometheus, aggiornata ogni ciclo |
| **Disponibile a freddo** | ✅ Sì | ❌ No (richiede campioni sufficienti) |

In DMOS il concetto analogo al "ping RTT" di Romano è **Φ_net** (non Φ_response_time),
calcolato tramite ping_exporter. Φ_response_time è un segnale aggiuntivo e più ricco,
disponibile solo dopo che Hubble ha accumulato abbastanza campioni.

Questa distinzione è il motivo principale per cui ha senso una Phase 1 separata:
Φ_response_time non può essere usata al momento del deploy.

---

## Architettura a due fasi

```
t=0s          t=120s                        t=∞
│             │
│── FASE 1 ───│──────────── FASE 2 ──────────────────────────────────→
│   Blind     │   Dynamic rescheduling
│   allocation│
│             │
│  Φ_demand   │  Φ_demand       ← invariato (Nginx ingress rate)
│  Φ_net      │  Φ_net          ← invariato (ping RTT inter-cluster)
│  Φ_carbon   │  Φ_carbon       ← invariato (carbon intensity statica)
│  ─────────  │  + Φ_response_time  ← Hubble p95 HTTP [5m], ora affidabile
│  (ω=0)      │  + Φ_cap            ← CPU/RAM stabilizzati
│  (ω=0)      │  + Φ_load           ← campioni Hubble sufficienti
│  (ω=0)      │
│             │
│  Level 2:   │  Level 2:
│  Nginx rate │  Hubble destination_workload
│  come proxy │  (carico effettivo sui pod)
```

---

## Fase 1 — Blind Allocation (0 → 120s)

### Segnali utilizzati

| Segnale | Fonte | Perché affidabile a freddo |
|---------|-------|---------------------------|
| **Φ_demand** | `nginx_ingress_controller_requests` | Disponibile al primo request, non richiede storia |
| **Φ_net** | ping_exporter RTT inter-cluster | Strutturale, non dipende dal traffico applicativo |
| **Φ_carbon** | carbon intensity per regione (config) | Statico, non richiede osservazione |

### Segnali disabilitati (ω = 0)

| Segnale | Fonte | Perché non affidabile a freddo |
|---------|-------|-------------------------------|
| **Φ_response_time** | Hubble histogram `_bucket` p95 su `[5m]` | Con < 120s di campioni la stima è statisticamente rumorosa |
| **Φ_cap** | CPU e RAM da Prometheus per-cluster | Pod appena avviati: warm-up JVM/runtime → letture CPU instabili |
| **Φ_load** | `hubble_http_requests_total` rate su `[1m]` | Pochi campioni nella finestra → stime non rappresentative |

### Motivazione statistica per 120 secondi

Il threshold di 120s non è arbitrario. È derivato dalla necessità di avere un p95
statisticamente stabile dalla finestra Hubble `[5m]`:

```
Al momento dello switch (t=120s):
  finestra [5m] contiene 120s di dati su 300s disponibili

  Con carico minimo (20 utenti, 0.5 req/s per utente):
    20 × 0.5 = 10 req/s per cluster × 120s = 1200 campioni
    p95 = campione in posizione #1140 → stima stabile ✓

  Con carico peak (300 utenti):
    100 × 0.5 = 50 req/s × 120s = 6000 campioni → ottimo ✓

Con solo 90s (alternativa scartata):
    10 req/s × 90s = 900 campioni → al limite per p95 stabile
    Preferito 120s per margine statistico sufficiente anche a basso carico.
```

### Pesi profilo `cold_start` in `weights.yaml`

```yaml
cold_start:
  omega_latency:  0.00   # Φ_response_time disabilitato: nessun campione Hubble affidabile
  omega_capacity: 0.00   # Φ_cap disabilitato: CPU instabile durante warm-up pod
  omega_load:     0.00   # Φ_load disabilitato: pochi campioni nella finestra [1m]
  omega_carbon:   0.30   # statico per regione, sempre affidabile
  omega_network:  0.35   # RTT inter-cluster stabile, non dipende dal traffico
  omega_demand:   0.35   # Nginx ingress rate, disponibile al primo request
```

I pesi disabilitati non vengono semplicemente ignorati: ScoreFunctions li moltiplica
per 0, quindi le componenti vengono comunque calcolate (per completezza dei log e
per essere pronte al cambio di fase), ma non contribuiscono allo score.

### Gestione della latenza in Phase 1

Con `ω_response_time = 0`, Φ_response_time non influenza lo score. Tuttavia
`_collect_cluster_metrics()` deve comunque popolare `latency_mean` e
`latency_variance` per evitare divisioni per zero nelle formule e log fuorvianti.

In Phase 1, questi campi vengono popolati con i **valori baseline da config**
(`cluster_cfg.baseline_latency_ms`, `cluster_cfg.latency['variance_ms']`):
valori stimati dalla configurazione del cluster, non osservati a runtime.

```python
# Phase 1: usa valori baseline da config (ω=0, quindi non influenza score)
latency_mean = cluster_cfg.baseline_latency_ms
latency_variance = cluster_cfg.latency.get('variance_ms', 15.0) ** 2

# Phase 2: usa Hubble p95 histogram [5m] (campioni sufficienti)
latency_p95 = prom.get_latency_p95(service, namespace)
latency_mean = latency_p95 / 1.65
latency_variance = ((latency_p95 - latency_mean) / 1.65) ** 2
```

### Level 2 in Phase 1: doppio ruolo di Nginx

In Phase 1, la metrica Nginx ingress rate serve **due scopi simultanei**:

```
nginx_ingress_controller_requests
  │
  ├─→ Level 1: Φ_demand(i) = rate_i / Σ rate_j  (WHERE — quota di repliche)
  │
  └─→ Level 2: target_replicas = rate_i / cap_per_pod  (HOW MANY — numero repliche)
```

Questo è coerente perché in Phase 1 non c'è ancora redistribuzione via cluster mesh
(i pod sono appena avviati): il traffico che entra dal Nginx va direttamente ai pod
locali, quindi `ingress_rate ≈ pod_request_rate`.

In Phase 2 i due segnali si separano: Φ_demand continua a usare Nginx (domanda
geografica), ma Level 2 passa a Hubble `destination_workload` che misura il carico
**effettivo** sui pod, incluso il traffico ricevuto via cluster mesh da altri cluster.

---

## Fase 2 — Dynamic Rescheduling (120s → ∞)

### Segnali aggiuntivi attivati

| Segnale | Fonte | Finestra | Motivazione |
|---------|-------|---------|-------------|
| **Φ_response_time** | `hubble_http_request_duration_seconds_bucket` | `[5m]` | Smoothing alto: decisioni strutturali WHERE |
| **Φ_cap** | CPU/RAM da Prometheus per-cluster | `[1m]` | Reattivo: stato risorse attuale |
| **Φ_load** | `hubble_http_requests_total` rate | `[1m]` | Reattivo: carico attuale vs capacità |

### Level 2 in Phase 2

Il ReplicaScaler passa a **Hubble destination_workload**:

```promql
rate(hubble_http_requests_total{destination_workload="frontend"}[1m])
```

Con cluster mesh attivo, i pod possono ricevere traffico da altri cluster. Hubble vede
il carico **effettivo sui pod**, indipendentemente dall'ingress di origine.
Questo può divergere dall'ingress rate Nginx: un pod di cluster1 può ricevere richieste
entrate dal Nginx di cluster2 via mesh → Hubble conta questo carico, Nginx no.

### Finestre temporali: control theory

```
Level 1 (WHERE) — loop esterno, lento:
  Φ_response_time usa [5m] → alto smoothing → no thrashing tra cluster
  Spostare repliche tra cluster è costoso (pod termination + startup + routing)

Level 2 (HOW MANY) — loop interno, veloce:
  request_rate usa [1m] → reattivo → segue il traffico nel ciclo successivo
  Aggiungere/rimuovere pod nello stesso cluster è economico (~15-30s)
```

Il loop esterno (Level 1) è sempre più lento del loop interno (Level 2): principio
classico di control theory per evitare oscillazioni tra i livelli.

---

## Timeline completa

```
t=0s    Deploy. Nginx subito attivo. Hubble inizia a campionare (0 campioni).
        Phase 1 attiva.
        Score: f(Φ_demand, Φ_net, Φ_carbon)
        Level 2: ingress_rate Nginx → target_replicas

t=30s   Cycle 1 (polling ogni 30s):
          Log: "[Phase 1 — BLIND ALLOCATION] t=30s < 120s | ..."
          cluster1=75 req/s (Nginx) → Φ_demand=0.50
          Hubble: ~30s/300s campioni nella finestra [5m] → p95 non usato (ω=0)
          Azione: 2 pod cluster1, 2 pod cluster2, 2 pod cluster3

t=60s   Cycle 2:
          Log: "[Phase 1 — BLIND ALLOCATION] t=60s < 120s | ..."
          Hubble: ~60s/300s campioni → ancora Phase 1

t=90s   Cycle 3:
          Log: "[Phase 1 — BLIND ALLOCATION] t=90s < 120s | ..."
          Hubble: ~90s/300s campioni → ancora Phase 1 (< 120s)

t=120s  *** SWITCH Phase 1 → Phase 2 ***
          Log: "[Phase 2 — DYNAMIC RESCHEDULING] t=120s | score completo: ..."
          Score: Φ_resp + Φ_cap + Φ_load + Φ_carbon + Φ_net + Φ_demand
          Level 2: passa a Hubble destination_workload
          Hubble: 120s/300s campioni → p95 statisticamente utile (≥1200 campioni)

t=150s  Cycle 5: primo rescheduling completo con metriche reali.
          Φ_response_time attivo: possibile riallocazione basata su latenza osservata.
          Se cluster1 ha p95 più alta del previsto → score scende → meno repliche future.

t=300s  Hubble [5m] window completamente piena: p95 al massimo della stabilità.
```

---

## Implementazione

### Costante globale e commento architetturale

```python
# dmos_scheduler.py

# ── Two-phase scheduling ──────────────────────────────────────────────────────
# FASE 1 — "Blind allocation" (0 → COLD_START_SECONDS):
#   Usa solo segnali strutturali/freddi: Φ_demand, Φ_net, Φ_carbon.
#   Φ_response_time, Φ_cap, Φ_load disabilitati (ω=0): metriche non affidabili
#   a freddo (pod appena avviati, pochi campioni Hubble nella finestra [5m]).
#   Level 2: usa ingress rate Nginx come proxy traffico (Hubble non ha dati).
#
# FASE 2 — "Dynamic rescheduling" (COLD_START_SECONDS → ∞):
#   Score completo con tutti i segnali osservati.
#   Level 2: usa Hubble destination_workload (carico effettivo sui pod).
#
# Riferimenti: FIRM (OSDI'20), Sinan (ASPLOS'21), Cilantro (OSDI'23).
COLD_START_SECONDS = 120
```

### Stato e due istanze ScoreFunctions

```python
class DMOSScheduler:
    def __init__(self, ...):
        # Timestamp di avvio: base per calcolo elapsed time nelle fasi
        self.scheduler_start_time = time.time()

        score_params = ScoreParameters(rho=..., rtt_max_ms=...)

        # Phase 2: score completo (pesi del profilo attivo in weights.yaml)
        self.score_func_warm = ScoreFunctions(
            weights={active_profile_weights},
            parameters=score_params,
        )

        # Phase 1: solo segnali strutturali (profilo cold_start da weights.yaml)
        # ω_lat=ω_cap=ω_load=0 → Φ_response_time, Φ_cap, Φ_load non contribuiscono
        self.score_func_cold = ScoreFunctions(
            weights={cold_start_weights},   # caricato da config_loader
            parameters=score_params,
        )
```

### Helpers di fase

```python
def _is_cold_start(self) -> bool:
    """True se siamo ancora in Phase 1 (blind allocation)."""
    return (time.time() - self.scheduler_start_time) < COLD_START_SECONDS

def _get_active_score_func(self) -> ScoreFunctions:
    """Restituisce score_func_cold (Phase 1) o score_func_warm (Phase 2)."""
    return self.score_func_cold if self._is_cold_start() else self.score_func_warm
```

### Raccolta metriche phase-aware

```python
def _collect_cluster_metrics(self, ..., cold_start_mode: bool = False):
    # ── Traffic metrics ───────────────────────────────────────────────
    if cold_start_mode:
        # Phase 1: ingress rate Nginx come proxy del carico sui pod
        # (Hubble destination_workload non ha campioni sufficienti)
        request_rate = ingress_rate_rps
    else:
        # Phase 2: Hubble destination_workload (carico effettivo pod)
        request_rate = prom.get_request_rate(service, namespace)

    # ── Latency metrics ───────────────────────────────────────────────
    if cold_start_mode:
        # Phase 1: valori baseline da config (ω_response_time=0 → non influenza score)
        # Popolati per evitare NaN nelle formule e log fuorvianti
        latency_mean = cluster_cfg.baseline_latency_ms
        latency_variance = cluster_cfg.latency.get('variance_ms', 15.0) ** 2
    else:
        # Phase 2: Hubble p95 histogram [5m] (campioni statisticamente sufficienti)
        latency_p95 = prom.get_latency_p95(service, namespace)
        latency_mean = latency_p95 / 1.65
        latency_variance = ((latency_p95 - latency_mean) / 1.65) ** 2
```

### Orchestrazione in `collect_scores()`

```python
def collect_scores(self, service_name, predicted_load=None):
    # Determina fase corrente
    cold_start = self._is_cold_start()
    elapsed = time.time() - self.scheduler_start_time
    active_score_func = self._get_active_score_func()

    if cold_start:
        logger.info(
            f"[Phase 1 — BLIND ALLOCATION] t={elapsed:.0f}s < {COLD_START_SECONDS}s | "
            f"score: Φ_demand + Φ_net + Φ_carbon | "
            f"Level 2: ingress rate Nginx come proxy traffico"
        )
    else:
        logger.info(
            f"[Phase 2 — DYNAMIC RESCHEDULING] t={elapsed:.0f}s | "
            f"score completo: Φ_resp + Φ_cap + Φ_load + Φ_carbon + Φ_net + Φ_demand | "
            f"Level 2: Hubble destination_workload"
        )

    # Pre-raccolta ingress rates (doppio ruolo in Phase 1: Φ_demand + Level 2 proxy)
    ingress_rates = {c: prom.get_ingress_rate(ns) for c, prom in self.prom_map.items()}
    ...

    # Calcolo score per ogni cluster con fase e score_func corretti
    result = self._compute_cluster_score(
        ...,
        cold_start_mode=cold_start,
        active_score_func=active_score_func,
    )
```

### Modifiche a `config_loader.py`

`ConfigLoader` carica il profilo `cold_start` da `weights.yaml` e lo espone come
`self.cold_start_weights`:

```python
# In ConfigLoader._parse():
cold_start_data = score_weights_raw.get('cold_start', {})
if cold_start_data:
    self.cold_start_weights = ScoreWeightsConfig(
        omega_latency=cold_start_data.get('omega_latency', 0.00),
        omega_capacity=cold_start_data.get('omega_capacity', 0.00),
        omega_load=cold_start_data.get('omega_load', 0.00),
        omega_carbon=cold_start_data.get('omega_carbon', 0.30),
        omega_network=cold_start_data.get('omega_network', 0.35),
        omega_demand=cold_start_data.get('omega_demand', 0.35),
    )
else:
    # Fallback: usa stessi pesi del profilo attivo (log di warning)
    # Comportamento: Phase 1 == Phase 2 → nessuna distinzione di fase
    self.cold_start_weights = self.score_weights
    logger.warning("Profilo 'cold_start' non trovato in weights.yaml, "
                   "Phase 1 userà gli stessi pesi di Phase 2")
```

Il fallback garantisce che il sistema funzioni anche senza il profilo `cold_start`
in `weights.yaml`, semplicemente senza la distinzione di fase.

---

## Log di runtime

### Phase 1 (ogni ciclo fino a t=120s)

```
INFO  [DMOSScheduler] [Phase 1 — BLIND ALLOCATION] t=30s < 120s |
      score: Φ_demand + Φ_net + Φ_carbon |
      Level 2: ingress rate Nginx come proxy traffico
INFO  [DMOSScheduler] Ingress rates: cluster1=75.0 req/s, cluster2=50.0 req/s,
      cluster3=25.0 req/s | total=150.0 req/s
INFO  [DMOSScheduler] Score cluster1: 0.584
      (lat=0.000, cap=0.000, load=0.000, carbon=0.803, net=0.368, demand=0.500)
DEBUG [DMOSScheduler] [Phase 1] cluster1: request_rate=75.0 req/s
      (proxy Nginx ingress, Hubble non disponibile)
DEBUG [DMOSScheduler] [Phase 1] cluster1: latency baseline 60.0ms
      (Hubble p95 non affidabile)
```

### Switch Phase 1 → Phase 2 (t=120s)

```
INFO  [DMOSScheduler] [Phase 2 — DYNAMIC RESCHEDULING] t=120s |
      score completo: Φ_resp + Φ_cap + Φ_load + Φ_carbon + Φ_net + Φ_demand |
      Level 2: Hubble destination_workload
INFO  [DMOSScheduler] Score cluster1: 0.631
      (lat=0.712, cap=0.654, load=0.789, carbon=0.803, net=0.368, demand=0.500)
```

La differenza nel breakdown dello score tra Phase 1 e Phase 2 (lat/cap/load da 0
a valori reali) è visibile nei log e nei grafici Prometheus, permettendo di
verificare sperimentalmente il cambio di comportamento.

---

## Confronto con Romano

Romano non implementa una distinzione di fase: il suo scheduler applica sempre la
stessa score function, incluse le componenti che richiedono metriche osservate. Al
momento del deploy, quando quelle metriche non sono disponibili, Romano usa valori
di fallback configurati staticamente senza segnalare esplicitamente lo stato.

| Aspetto | Romano | DMOS two-phase |
|---------|--------|----------------|
| Componente latenza | Φ_lat (ping RTT) | Φ_response_time (Hubble p95 HTTP) |
| Fase iniziale | Score completo con fallback statici | Score ridotto: solo segnali strutturali |
| Segnale Level 2 a freddo | Fallback CPU-based o 0 | Ingress rate Nginx (proxy affidabile e coerente) |
| Transizione | Nessuna (score invariato nel tempo) | Esplicita a t=120s, loggata con elapsed time |
| Fallback se metrica assente | Silenzioso | Fallback documentato con warning in log |
| Motivazione teorica | Implicita | FIRM (OSDI'20), Sinan (ASPLOS'21), Cilantro (OSDI'23) |

---

## Riferimenti

- **FIRM**: Qiu et al., "FIRM: An Intelligent Fine-grained Resource Management
  Framework for SLO-Oriented Microservices", OSDI 2020.
  Initial placement senza latency data → latency-SLO triggered rescheduling.

- **Sinan**: Zhang et al., "Sinan: ML-Based and QoS-Aware Resource Management
  for Cloud Microservices", ASPLOS 2021.
  Due fasi esplicite: initial allocation → tail latency observation → reschedule.

- **Cilantro**: Bhardwaj et al., "Cilantro: Performance-Aware Resource Allocation
  for General Objectives via Online Feedback", OSDI 2023.
  Cold-start allocation senza performance data → feedback loop iterativo basato
  su confidence delle metriche osservate.

---

*Vedi anche: [`docs/traffic-metrics-architecture.md`](traffic-metrics-architecture.md)
per la spiegazione dettagliata di Φ_demand e del ruolo di Nginx Ingress.*
