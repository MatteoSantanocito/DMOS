# DMOS — Punto di Partenza e Obiettivi di Progetto

## Baseline: il Metascheduler di Romano (2025)

> Riferimento: G. Romano, *Orchestrazione di microservizi in ambienti multicluster*, Tesi di Laurea Magistrale, 2025.

---

### 1. Contesto e motivazione originale

Il lavoro di Romano nasce nel contesto del **Cloud Continuum** (o *Compute Continuum*), un paradigma che mira a unificare le risorse computazionali distribuite lungo un continuum edge–fog–cloud, consentendo alle applicazioni di essere distribuite su infrastrutture eterogenee in modo trasparente [Romano §1].

Il problema centrale è la gestione di **applicazioni a microservizi** su architetture **multicluster Kubernetes**: in ambienti dove più cluster geograficamente distribuiti devono cooperare, il sistema di scheduling nativo di Kubernetes non è adeguato, perché opera esclusivamente all'interno del singolo cluster e non ha visibilità sul comportamento globale del sistema.

Romano propone la costruzione di un **metascheduler esterno** che, operando al di sopra di Kubernetes, ridistribuisce dinamicamente le repliche dei microservizi tra i cluster in funzione della latenza osservata.

---

### 2. Infrastruttura di riferimento

L'infrastruttura su cui Romano costruisce e testa il suo sistema è la stessa su cui DMOS è sviluppato:

| Componente | Dettaglio |
|-----------|-----------|
| **Macchine** | 3 server fisici (ms01, ms02, ms03) su LAN 192.168.1.0/24 |
| **Kubernetes** | K3s (lightweight distribution) su ogni macchina |
| **CNI & Service Mesh** | Cilium + ClusterMesh (interconnessione L3/L4/L7 tra cluster) |
| **Load Balancing** | Cilium Global Services: le richieste in ingresso su un cluster sono bilanciate automaticamente tra tutti i pod globali, indipendentemente dal cluster |
| **Applicazione di test** | Online Boutique (Google microservices demo, ~11 microservizi) |
| **Metriche** | Prometheus per-cluster (NodePort 30090) + ping_exporter (latenza inter-cluster) + Hubble L7 (traffico HTTP) |
| **Generatore di carico** | K6 (Romano), poi sostituito da Locust in DMOS |

L'idea architetturale chiave di Romano è il **PROM_MAP**: ogni cluster ha un'istanza Prometheus locale che vede esclusivamente i propri pod. Il metascheduler interroga tutte e tre le istanze in parallelo (`ThreadPoolExecutor`) e aggrega i dati per costruire una vista globale del sistema. Questo pattern è stato **integralmente ereditato da DMOS** (vedi `src/level1/dmos_scheduler.py`).

---

### 3. Algoritmo del metascheduler di Romano

Il metascheduler di Romano opera con un loop periodico:

```
ogni T secondi:
  1. Per ogni cluster i:
     a. Interroga Prometheus_i → ottieni latenza inter-cluster (ping_exporter)
     b. Calcola score_i basato esclusivamente sulla latenza
  2. Algoritmo greedy di winner determination:
     a. Ordina cluster per score decrescente
     b. Assegna repliche proporzionalmente agli score
  3. Applica le assegnazioni via API Kubernetes (scale deployment)
```

**Funzione di score (Romano):** basata unicamente sulla latenza media osservata dal ping_exporter:

```
score_i = f(latenza_inter-cluster_i)
```

Il cluster con latenza minore riceve più repliche. Il bilanciamento del carico (quale cluster serve effettivamente le richieste) è demandato a **Cilium Global Services**: manipolando il numero di repliche per cluster, Romano influenza indirettamente le probabilità di routing di Cilium, che bilancia round-robin in proporzione al numero di endpoint disponibili.

**Risultati di Romano:** i test mostrano una riduzione della latenza p95 rispetto al baseline (deployment uniforme senza scheduler), in particolare nei casi in cui la distribuzione del traffico era sbilanciata geograficamente. I test sono condotti con K6 su due scenari: carico uniforme e carico sbilanciato.

---

### 4. Limitazioni esplicitamente riconosciute da Romano

Romano nella sezione *Limitazioni e sviluppi futuri* (p. 60) identifica due classi di problemi aperti:

#### 4.1 Mancanza di awareness delle risorse hardware

> *«Il metascheduler, nella versione attuale, presenta alcune limitazioni: non considera le risorse hardware dei cluster, come il consumo di CPU e memoria. Questo a tutti gli effetti risulta un problema per la scelta di scaling in quanto può portare ad incrementare le repliche su cluster che, pur essendo a bassa latenza, non dispongono delle risorse necessarie a gestire nuovi pod.»*

**Il problema concreto:** uno score basato solo sulla latenza può assegnare repliche a un cluster che ha CPU >90% o memoria esaurita. K8s accetterebbe la richiesta di scale ma i pod rimarrebbero in `Pending` indefinitamente, degradando il servizio senza che il metascheduler se ne accorga.

#### 4.2 Assenza di autoscaling predittivo

> *«Questo rappresenta anche un possibile sviluppo futuro, cioè, estendere l'algoritmo per considerare risorse hardware ed inoltre si potrebbero considerare anche logiche di autoscaling predittivo basate su machine learning.»*

**Il problema concreto:** il metascheduler di Romano **redistribuisce** le repliche esistenti tra i cluster, ma non cambia mai il numero totale di repliche. Se il carico cresce, nessun nuovo pod viene creato: il sistema può solo spostare i pod già esistenti, senza aumentare la capacità complessiva. Questo è il limite architetturale più profondo del baseline.

#### 4.3 Limitazioni implicite (non esplicitamente dette da Romano)

Dall'analisi del codice e dei risultati emergono ulteriori limitazioni non nominate:

| Limitazione | Descrizione |
|------------|-------------|
| **Score monodimensionale** | Solo latenza, nessuna ponderazione multi-obiettivo |
| **Nessuna carbon-awareness** | L'intensità carbonica del mix energetico per cluster non è considerata |
| **Nessuna predizione del traffico** | Il metascheduler reagisce al traffico corrente, non anticipa i picchi |
| **Nessun controllo dell'oscillazione** | Senza meccanismi di hysteresis, il sistema può oscillare (flapping) |
| **Latenza misurata con ping** | Il ping misura la latenza di rete pura (ICMP), non la latenza applicativa end-to-end percepita dall'utente finale |
| **Traffico non misurato per-cluster** | Romano usa il traffico globale, non la quota di traffico realmente servita da ciascun cluster |

---

## DMOS: obiettivi e scelte progettuali

DMOS (*Distributed Microservices Orchestration System*) nasce come **estensione sistematica** del lavoro di Romano, con l'obiettivo di trasformare il metascheduler da un sistema di **redistribuzione reattiva** a un sistema di **orchestrazione proattiva multi-obiettivo**.

---

### 5. Obiettivi primari di DMOS

#### O1 — Autoscaling proattivo basato sulla predizione del traffico

**Problema che risolve:** il metascheduler di Romano non scala mai il numero totale di repliche. DMOS deve decidere *quante* repliche totali servono prima ancora che il picco si manifesti.

**Approccio adottato (Level 2):**

Il modulo `TrafficPredictor` (`src/level2/predictor.py`) implementa una predizione trend-based ispirata al modello PD:

```
Λ^pred(t) = Λ(t) + (dΛ/dt) · Δt_horizon · damping_factor
```

dove:
- `Λ(t)` = traffico corrente misurato da Hubble L7 (req/s)
- `dΛ/dt` = derivata approssimata su finestra temporale τ (default 5 min)
- `Δt_horizon` = orizzonte di predizione (default 120s)
- `damping_factor` = 0.5 per scale-up (proattivo), 0.3 per scale-down (conservativo)

Il numero totale di repliche è calcolato da `ReplicaScaler` (`src/level2/scaler.py`):

```
base_replicas = ceil(Λ^pred / capacity_per_replica × (1 + safety_margin))
target_replicas = clamp(base_replicas + PD_correction, min_replicas, max_replicas)
```

Un controllore PD (`PDController`) corregge l'errore residuo tra traffico predetto e traffico reale, prevenendo sia l'under-provisioning (errore positivo → scale-up aggiuntivo) che l'over-provisioning eccessivo (errore negativo → freno al scale-down).

**Risultato atteso:** DMOS deve scalare *prima* che la domanda superi la capacità corrente (Time-to-Scale medio < 0s, cioè proattivo).

---

#### O2 — Scheduling multi-obiettivo con carbon-awareness

**Problema che risolve:** Romano ottimizza solo la latenza. DMOS deve bilanciare quattro obiettivi in tensione tra loro: latenza, capacità disponibile, carico previsto, impatto ambientale.

**Approccio adottato (Level 1):**

La funzione di score `ScoreFunctions` (`src/level1/score_functions.py`) implementa uno score composito:

```
Φ_i = ω_lat · Φ_lat(i) + ω_cap · Φ_cap(i) + ω_load · Φ_load(i) + ω_carbon · Φ_carbon(i)
```

Ogni componente ha una semantica precisa:

| Componente | Formula | Obiettivo |
|-----------|---------|-----------|
| **Φ_lat(i)** | `(1/(1+η·E[L_i])) · exp(-Var(L_i)/σ²)` | Preferire cluster a bassa latenza media e bassa varianza |
| **Φ_cap(i)** | `(R_avail/R_tot)^κ · (1 - λ_i/λ_max)` | Preferire cluster con risorse libere e carico basso |
| **Φ_load(i)** | `exp(-μ · λ^pred_i / λ_max_i)` | Penalizzare cluster già carichi secondo la predizione |
| **Φ_carbon(i)** | `exp(-ν · CI_i / CI_max)` | Preferire cluster alimentati da energia a bassa intensità carbonica |

I pesi `ω` sono configurabili per profilo (`balanced`, `latency_first`, `green_first`, ecc.), consentendo di variare il trade-off tra obiettivi senza modificare il codice.

**Carbon-awareness:** l'intensità carbonica `CI_i` (gCO₂/kWh) per regione geografica è fornita da `CarbonClient` (`src/metrics/carbon_client.py`). La formula esponenziale garantisce che cluster con `CI` molto alto (es. PL: 650 gCO₂/kWh) ricevano sistematicamente meno repliche rispetto a cluster a bassa emissione (es. FR: 80 gCO₂/kWh), a parità di latenza e capacità.

---

#### O3 — Awareness delle risorse hardware (hard constraints)

**Problema che risolve:** la limitazione #1 di Romano — assegnare repliche a cluster senza risorse.

**Approccio adottato:** `DMOSScheduler._compute_cluster_score()` implementa **hard constraints** verificati *prima* del calcolo dello score:

```python
CPU_HARD_LIMIT_PCT    = 95.0  # Cluster escluso se CPU > 95%
MEMORY_HARD_LIMIT_PCT = 90.0  # Cluster escluso se RAM > 90%
MIN_CPU_CORES_FREE    = 0.2   # Almeno 0.2 core liberi
MIN_MEMORY_GB_FREE    = 0.2   # Almeno 0.2 GB liberi
```

Un cluster che viola uno qualsiasi di questi vincoli viene escluso dall'allocazione e riceve score=0, indipendentemente dalla sua latenza. Questo garantisce che le repliche vengano assegnate esclusivamente a cluster fisicamente in grado di ospitarle.

Le metriche di CPU e memoria sono lette dal Prometheus locale di ciascun cluster via PROM_MAP — lo stesso pattern di Romano, ma esteso per includere `container_cpu_usage_seconds_total` e `node_memory_MemAvailable_bytes`.

---

#### O4 — Distribuzione equa delle repliche (Jain Fairness Index)

**Problema che risolve:** uno scheduler puramente greedy potrebbe concentrare tutte le repliche sul cluster con lo score più alto, violando il requisito di alta disponibilità multi-cluster.

**Approccio adottato:** `WinnerDetermination.allocate()` (`src/level1/winner_determination.py`) implementa una **allocazione proporzionale agli score**:

```
quota_i = score_i / Σ(score_j)
replicas_i = round(total_replicas × quota_i)
```

con garanzia che ogni cluster con `capacity > 0` riceva almeno 1 replica.

La **fairness** è misurata con l'indice di Jain:

```
J(X) = (Σ r_i)² / (N · Σ r_i²)   ∈ [1/N, 1]
```

dove `r_i` è la quota di repliche assegnata al cluster `i`. Un valore prossimo a 1.0 indica distribuzione equa; un valore prossimo a 1/N indica concentrazione su un singolo cluster.

Il metascheduler di Romano non disponeva di alcuna misura di equità nella distribuzione.

---

#### O5 — Prevenzione dell'oscillazione (anti-flapping)

**Problema che risolve:** un sistema di scaling reattivo puro tende ad oscillare — scala up, poi immediatamente down, creando instabilità e sprechi di risorse.

**Meccanismi implementati in DMOS:**

| Meccanismo | Implementazione | Scopo |
|-----------|----------------|-------|
| **Dead zone** | Se `|Δtraffic| < 15%`, nessuno scaling | Ignora variazioni rumorose |
| **Scale-down cooldown** | 60s tra due scale-down consecutivi | Evita scale-down prematuro |
| **Scale-up protection** | 120s di lock dopo uno scale-up | Permette alle repliche di stabilizzarsi prima di valutare un scale-down |
| **max_delta_per_cycle** | Max ±4 repliche per ciclo (30s) | Rate limiting sulle decisioni di scaling |
| **Traffic floor** | Se `0 < FE < 2.0 rps`, bypassa il predictor e usa min_replicas | Evita scala a 0 per traffico residuo da probe |

Nessuno di questi meccanismi esisteva nel baseline di Romano.

---

#### O6 — Misurazione accurata del traffico applicativo (Hubble L7)

**Problema che risolve:** Romano misura la latenza con ping ICMP (livello rete) e non misura il traffico applicativo per-cluster in modo diretto.

**Approccio adottato:** DMOS usa `hubble_http_requests_total`, un contatore HTTP incrementato da Envoy (Cilium L7 sidecar) ogni volta che una richiesta HTTP attraversa la chain Nginx→microservizio. La query PromQL:

```promql
sum(rate(hubble_http_requests_total{
    destination_workload="frontend",
    destination_namespace="online-boutique"
}[5m]))
```

viene eseguita separatamente su ogni Prometheus per-cluster, producendo una misura del traffico realmente servito da ciascun cluster — non del traffico globale. Questo è fondamentale per il calcolo corretto di `Φ_load(i)` nella funzione di score.

**Nota sul moltiplicatore sistematico:** Hubble conta ogni sotto-richiesta HTTP (immagini, API calls, assets statici), mentre Locust conta solo le transazioni utente. Il rapporto stabile è `FE/Locust ≈ 1.43×`. Questo disallineamento è documentato e non costituisce un errore: il parametro `capacity_req_per_sec` è calibrato sulle unità Hubble.

---

### 6. Architettura a due livelli

DMOS introduce una separazione esplicita tra due responsabilità che nel baseline di Romano erano indistinte:

```
┌─────────────────────────────────────────────────────────────┐
│                        DMOS                                  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LEVEL 2 — Scaling globale                           │   │
│  │  TrafficPredictor + PDController + ReplicaScaler     │   │
│  │  → "Quante repliche totali servono?"                 │   │
│  └──────────────────────────┬───────────────────────────┘   │
│                             │ total_replicas                 │
│  ┌──────────────────────────▼───────────────────────────┐   │
│  │  LEVEL 1 — Distribuzione tra cluster                 │   │
│  │  DMOSScheduler + ScoreFunctions + WinnerDetermination│   │
│  │  → "Come distribuire le repliche tra i cluster?"     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Level 2** (nuovo rispetto a Romano): determina il numero totale di repliche necessarie per l'intero sistema multicluster. Opera sul traffico aggregato (somma di tutti i cluster) e produce un intero `total_replicas`.

**Level 1** (evoluzione di Romano): ricevuto `total_replicas`, distribuisce queste repliche tra i cluster in base allo score multi-obiettivo. Opera separatamente per-cluster e produce le assegnazioni `{cluster_i: n_i replicas}`.

La separazione consente di modificare l'algoritmo di scaling (Level 2) indipendentemente dall'algoritmo di distribuzione (Level 1), e viceversa.

---

### 7. Tabella comparativa: Romano vs DMOS

| Dimensione | Romano (baseline) | DMOS (stato attuale) |
|-----------|-------------------|---------------------|
| **Scaling totale** | ❌ Nessuno — il totale è fisso | ✅ Proattivo: predizione + PD controller |
| **Score cluster** | Latenza ICMP (1 dimensione) | Latenza + Capacità + Carico + Carbon (4 dimensioni) |
| **Carbon-awareness** | ❌ Assente | ✅ Φ_carbon = exp(-ν·CI/CI_max) |
| **Hardware constraints** | ❌ Non verificati | ✅ Hard limits su CPU/RAM prima dello score |
| **Misura del traffico** | ❌ Assente (solo latenza ping) | ✅ Hubble L7 per-cluster (req/s) |
| **Predizione** | ❌ Reattivo puro | ✅ Trend-based (EMA + derivata) + PD |
| **Anti-flapping** | ❌ Assente | ✅ Dead zone + cooldown + protection |
| **Fairness** | Non misurata | ✅ Jain Fairness Index |
| **Vickrey auction** | ❌ Assente | ✅ Implementato (per future estensioni) |
| **Generatore di carico** | K6 | Locust (più flessibile, scenari programmabili) |
| **Pattern Prometheus** | PROM_MAP (per-cluster) | PROM_MAP (ereditato, stesso pattern) |
| **Applicazione test** | Online Boutique | Online Boutique (stesso deployment) |
| **Infrastruttura** | K3s + Cilium + 3 macchine | K3s + Cilium + 3 macchine (stessa) |

---

### 8. Limitazioni aperte in DMOS (stato attuale)

Nonostante le estensioni rispetto a Romano, DMOS nella versione attuale presenta limitazioni che definiscono i confini del presente lavoro:

| Limitazione | Descrizione | Impatto |
|------------|-------------|---------|
| **Finestra [5m] di Hubble** | `rate([5m])` con scrape 60s → ~5 min di ramp-up delay all'avvio | Scaling lento nei primi 5 min di ogni test; critico per flash crowd |
| **EMA lenta al termine del test** | Il predictor EMA rimane elevato per ~15 min dopo lo stop del traffico | Over-provisioning nella fase post-test |
| **Predizione senza ML** | Il predictor è trend-based lineare, non impara pattern periodici (es. picchi giornalieri) | MAPE più alta su traffico non lineare |
| **No workload-aware per backend** | I servizi backend (cartservice, productcatalog, ecc.) usano network bytes come proxy del carico gRPC | Accuratezza ±20-30% per i backend |
| **Ambiente lab** | 3 macchine LAN, single-node per cluster | Risultati non direttamente trasferibili a infrastrutture cloud reali |
| **Carbon intensity statica** | CI per cluster è configurata manualmente, non aggiornata in tempo reale | Il beneficio ambientale è approssimato |

---

### 9. Posizionamento scientifico

DMOS si colloca nel filone della ricerca su **proactive autoscaling** per ambienti multicluster, integrando:

1. **Orchestrazione multi-obiettivo** (scheduling): estende il lavoro di Romano aggiungendo capacità, carico e carbon alla funzione di score, avvicinandosi agli approcci di Chiaro et al. [5] su latency-aware scheduling nel Cloud-Edge Continuum.

2. **Autoscaling predittivo** (scaling): implementa un controllore PD con predizione trend-based, categoria documentata nella letteratura come *proactive scaling* o *predictive HPA*, alternativa all'HPA nativo di Kubernetes (che è puramente reattivo e basato su CPU/memoria).

3. **Green computing**: l'integrazione di `Φ_carbon` nella funzione di score pone DMOS nella categoria dei **carbon-aware schedulers**, campo in crescita con l'aumento della sensibilità ambientale dei provider cloud (es. Google Cloud Carbon Footprint, Azure Sustainability).

4. **Infrastruttura Cilium**: l'uso di Hubble L7 per la misurazione del traffico è una scelta originale rispetto alla letteratura, che tipicamente usa metriche Kubernetes native (CPU, memory) o Istio. Cilium/Hubble non richiede sidecar aggiuntivi (è parte del CNI) ed è disponibile gratuitamente in ambienti K3s.

---

*Documento redatto sulla base di: G. Romano (2025), codice sorgente DMOS (src/), esperimento `double_wave_hubble` (2026-03-01).*
