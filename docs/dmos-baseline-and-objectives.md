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

Il metascheduler di Romano opera con un loop periodico (`CHECK_INTERVAL`):

```
ogni CHECK_INTERVAL secondi:
  1. Rileva i cluster attivi via cilium_ClusterMesh_remote_cluster_nodes
  2. Per ogni cluster i:
     a. Interroga Prometheus_i → verifica traffico via hubble_http_requests_total
     b. Interroga Prometheus_i → ottieni latenza inter-cluster via ping_rtt_mean_seconds
  3. Calcola global_replicas (autoscaling):
       somma_totale = sum(traffico per cluster)
       global_replicas = int(MIN_REPLICAS + (MAX_REPLICAS - MIN_REPLICAS)
                             * (somma_totale / TRAFFIC_THRESHOLD))
       global_replicas = clamp(global_replicas, MIN_REPLICAS, MAX_REPLICAS)
  4. Distribuisce global_replicas tra i cluster:
       score[c] = WT * normalized_local[c] + WL * normalized_remote[c]
       replicas[c] = max(MIN_REPLICAS_PER_CLUSTER,
                         round(global_replicas * score[c] / sum(scores)))
  5. Applica le assegnazioni via API Kubernetes (scale deployment):
       api.patch_namespaced_deployment_scale(
           name="frontend", namespace="default", body=body
       )
       → Solo il deployment "frontend" viene scalato.
         I backend (cartservice, productcatalogservice, ecc.) rimangono statici.
```

**Metriche usate nell'algoritmo (Romano):**

| Metrica | Fonte | Uso |
|---------|-------|-----|
| `hubble_http_requests_total{source="reserved:ingress"}` | Prometheus per-cluster (PROM_MAP) | Traffico in ingresso al cluster attraverso il proprio ingress controller — input primario sia per l'autoscaling globale che per la componente locale dello score |
| `ping_rtt_mean_seconds` | ping_exporter | Latenza inter-cluster — input per la componente remota dello score |

**Funzione di score (Romano):** bi-dimensionale, combina traffico locale e traffico remoto pesato dalla latenza:

```python
scores[c] = WT * normalized_local[c] + WL * normalized_remote[c]
```

dove:
- `normalized_local[c]` = traffico Hubble locale del cluster `c`, normalizzato sul massimo
- `normalized_remote[c]` = somma su tutti i cluster remoti `r` di `(traffico_verso_r × latenza(c,r))`, normalizzata sul massimo — rappresenta il "costo" di servire traffico che arriva da cluster remoti ad alta latenza

Il cluster con più traffico locale e latenza favorevole verso i cluster remoti riceve più repliche. Il bilanciamento effettivo è poi demandato a **Cilium Global Services**: con più pod attivi in un cluster, Cilium aumenta la probabilità che le richieste vengano servite localmente.

**Autoscaling in Romano:** la formula lineare `global_replicas = MIN + (MAX-MIN) * traffic/threshold` costituisce una forma di autoscaling **reattivo** — scala il numero totale di repliche in proporzione diretta al traffico corrente misurato, senza predizione futura.

**Risultati di Romano:** scenario 1 (carico crescente regolare, 10 min): latenza globale ridotta da p95=3.88s a p95=1.23s con metascheduler attivo. Scenario 2 (incremento rapido del traffico): riduzione analoga documentata nelle tabelle 5-1 e 5-2. I test sono condotti con K6 su due scenari: carico uniforme e carico sbilanciato.

---

### 4. Limitazioni esplicitamente riconosciute da Romano

Romano nella sezione *Limitazioni e sviluppi futuri* (p. 60) identifica due classi di problemi aperti:

#### 4.1 Mancanza di awareness delle risorse hardware

> *«Il metascheduler, nella versione attuale, presenta alcune limitazioni: non considera le risorse hardware dei cluster, come il consumo di CPU e memoria. Questo a tutti gli effetti risulta un problema per la scelta di scaling in quanto può portare ad incrementare le repliche su cluster che, pur essendo a bassa latenza, non dispongono delle risorse necessarie a gestire nuovi pod.»*

**Il problema concreto:** uno score basato solo sulla latenza può assegnare repliche a un cluster che ha CPU >90% o memoria esaurita. K8s accetterebbe la richiesta di scale ma i pod rimarrebbero in `Pending` indefinitamente, degradando il servizio senza che il metascheduler se ne accorga.

#### 4.2 Autoscaling reattivo senza predizione

> *«Questo rappresenta anche un possibile sviluppo futuro, cioè, estendere l'algoritmo per considerare risorse hardware ed inoltre si potrebbero considerare anche logiche di autoscaling predittivo basate su machine learning.»*

**Il problema concreto:** Romano implementa un autoscaling del totale repliche tramite una formula lineare proporzionale al traffico corrente. Questo è un approccio **reattivo puro**: il sistema scala in risposta al traffico già osservato, non anticipa il picco futuro. Se il traffico cresce rapidamente (flash crowd, ramp-up brusco), il sistema scala sempre in ritardo rispetto alla domanda effettiva — il tempo necessario a creare nuovi pod (pull immagine, readiness probe, ecc.) si somma al ritardo di osservazione.

#### 4.3 Limitazioni implicite (non esplicitamente dette da Romano)

Dall'analisi della tesi e della struttura dell'algoritmo emergono ulteriori limitazioni non nominate:

| Limitazione | Descrizione |
|------------|-------------|
| **Score bi-dimensionale** | Traffico locale + latenza × traffico remoto — no capacità hardware, no carbon |
| **Nessuna carbon-awareness** | L'intensità carbonica del mix energetico per cluster non è considerata |
| **Autoscaling reattivo lineare** | `global_replicas ∝ traffic_now` — nessuna predizione del trend futuro |
| **Nessun controllo dell'oscillazione** | Senza meccanismi di hysteresis, il sistema può oscillare (flapping) tra decisioni consecutive |
| **Latenza inter-cluster vs intra-cluster** | Romano misura la latenza *tra* i cluster (RTT ICMP da A a B) per decidere quale cluster è "vicino" agli altri. DMOS misura la latenza *dentro* ciascun cluster (`hubble_http_request_duration_seconds_bucket`, p95 HTTP L7) per rilevare saturazione applicativa. Sono due segnali con semantica completamente diversa. |
| **Nessuna misura di equità** | Non esiste un indice di fairness per valutare la distribuzione delle repliche tra cluster |
| **Backend statici** | Solo il deployment `frontend` (namespace `default`) viene scalato. I backend (cartservice, productcatalogservice, checkoutservice, recommendationservice) rimangono sempre al numero di repliche configurato staticamente, indipendentemente dal carico. Se il frontend scala a 6 repliche su cluster1 ma il backend ha 1 replica su cluster2, ogni chiamata gRPC frontend→backend diventa cross-cluster, aumentando la latenza end-to-end. |

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
| **Φ_lat(i)** | `(1/(1+η·E[L_i])) · exp(-Var(L_i)/σ²)` | Preferire cluster a bassa latenza media e bassa varianza. `E[L_i]` e `Var(L_i)` derivano da `hubble_http_request_duration_seconds_bucket` (p95 intra-cluster → mean=p95/1.65). Fallback: `baseline_latency_ms` da config. |
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

#### O6 — Cambio semantico della metrica di latenza: da inter-cluster a intra-cluster

**Problema che risolve:** Romano usa `ping_rtt_mean_seconds` come unica misura di latenza per guidare la distribuzione delle repliche. Questa metrica cattura solo il costo di rete tra cluster, ma è cieca alla saturazione applicativa — un cluster può avere ping basso e p95 HTTP di 2 secondi se la sua CPU è satura.

**Confronto tra le due metriche:**

| | `ping_rtt_mean_seconds` (Romano) | `hubble_http_request_duration_seconds_bucket` (DMOS) |
|---|---|---|
| **Layer** | L3 — rete | L7 — applicazione HTTP |
| **Cosa cattura** | Tempo di transito dei pacchetti tra nodi | Durata completa della richiesta HTTP: rete + coda + processing |
| **Dipende dal carico applicativo** | No — stabile anche con CPU al 100% | Sì — cresce con la saturazione del cluster |
| **Disponibile senza traffico** | Sempre (ping è attivo indipendentemente) | No — richiede richieste HTTP attive; in assenza: fallback a `baseline_latency_ms` da config |
| **Rumore** | Basso, segnale stabile | Più alto, varia istante per istante con il carico |
| **Correlazione con esperienza utente** | Indiretta — solo componente di rete | Diretta — è esattamente ciò che l'utente percepisce |
| **Semantica per lo scheduler** | "Quanto è vicino topologicamente questo cluster agli altri?" | "Quanto è sotto stress questo cluster in questo momento?" |

**Esempio concreto:** supponiamo che cluster1 abbia ping=0.3ms verso gli altri cluster e cluster2 abbia ping=0.8ms. Romano assegna più repliche a cluster1. Ma se cluster1 ha CPU al 92% e p95 HTTP = 1.8s, mentre cluster2 ha p95 = 95ms, DMOS assegna invece più repliche a cluster2 — il cluster meno stressato, non il più vicino topologicamente.

**Trade-off della scelta di DMOS:** la metrica è più precisa per rilevare la salute del cluster, ma introduce una dipendenza dal traffico attivo. In fase di avvio o in idle, `hubble_http_request_duration_seconds_bucket` non produce dati → DMOS ricade su `baseline_latency_ms` configurato staticamente (vedi `dmos_scheduler.py:154`). In questi casi i due approcci convergono a una stima fissa.

**Il fallback di DMOS non è paragonabile al ping di Romano.** Potrebbe sembrare che entrambi usino un valore fisso di latenza quando non c'è traffico, ma la differenza è sostanziale:

| | Romano (`ping_rtt_mean_seconds`) | DMOS fallback (`baseline_latency_ms`) |
|---|---|---|
| **Tipo** | Misurazione in tempo reale | Valore statico configurato a mano |
| **Per-cluster** | Sì — valori diversi per ogni coppia di cluster | No — identico per tutti i cluster (`2.0ms` per cluster1, cluster2, cluster3) |
| **Differenziazione** | Fornisce informazioni sulla topologia di rete reale | Annulla la dimensione latenza: Φ_lat uguale per tutti i cluster |
| **Effetto sullo scheduling** | Il cluster con ping più basso riceve più repliche | La latenza non influenza la decisione — il peso Φ_lat diventa neutro |

In pratica, quando DMOS usa il fallback (assenza di traffico HTTP), la componente `Φ_lat` produce lo stesso valore per tutti e tre i cluster. La decisione di scheduling in quella fase dipende esclusivamente da `Φ_cap`, `Φ_load` e `Φ_carbon`. Romano invece, anche senza traffico, continua a differenziare i cluster tramite il ping.

Questo è un **limite aperto** di DMOS nell'infrastruttura lab: l'integrazione del ping come fallback differenziato (o come quarta sorgente nel graceful degradation chain) renderebbe il sistema più robusto nelle fasi di cold start.

**Implementazione in DMOS** (`prometheus_client.py:get_latency_p95()`):
```promql
histogram_quantile(0.95,
  sum(rate(hubble_http_request_duration_seconds_bucket{
    destination_namespace="online-boutique"
  }[5m])) by (le)
) * 1000
```
Il p95 ottenuto viene convertito in: `mean = p95 / 1.65`, `variance = (p95 - mean)²` — entrambi usati nella formula `Φ_lat(i)`.

---

#### O7 — Evoluzione del filtro Hubble per il traffico: da `source="reserved:ingress"` a `destination_workload`

**Come Romano usa Hubble per il traffico:** Romano interroga `hubble_http_requests_total` con il filtro `source="reserved:ingress"` — conta le richieste che entrano nel cluster attraverso il proprio ingress controller. Questa è una misura del **traffico in ingresso al cluster**, che riflette quante richieste Cilium ha instradato verso quel cluster.

**Come DMOS usa Hubble:** DMOS interroga la stessa metrica con un filtro diverso — `destination_workload="frontend"` — che conta le richieste HTTP che raggiungono effettivamente i **pod del microservizio frontend**, indipendentemente da dove originano. La query:

```promql
sum(rate(hubble_http_requests_total{
    destination_workload="frontend",
    destination_namespace="online-boutique"
}[5m]))
```

**Perché questo filtro è più preciso:** il filtro `source="reserved:ingress"` di Romano misura il traffico al momento dell'ingresso nel cluster, ma non cattura le sub-richieste generate internamente dalla catena di microservizi (es. frontend → productcatalog, frontend → recommendation). Il filtro `destination_workload="frontend"` misura il traffico applicativo reale che il microservizio frontend deve processare, includendo tutte le richieste dirette al pod (sub-richieste HTTP, assets statici, API calls). Questo rende la metrica più rappresentativa del **carico effettivo** sul servizio.

**Nota sul moltiplicatore sistematico:** la differenza di filtro spiega anche perché Hubble/DMOS conta ~1.43× più richieste rispetto a Locust: Locust conta le transazioni utente (1 GET per task), mentre Hubble con `destination_workload="frontend"` conta ogni singola richiesta HTTP al pod, incluse le sotto-richieste generate da ogni pagina caricata.

---

#### O8 — Co-location dei backend: scaling proporzionale alla distribuzione del frontend

**Problema che risolve:** Romano scala solo il deployment `frontend` — i backend rimangono statici. Con Cilium Global Services, se il frontend di cluster1 chiama cartservice ma su cluster1 non esistono repliche di cartservice, la richiesta gRPC viene inoltrata a un backend su un altro cluster (cross-cluster call), aggiungendo latenza di rete intra-cluster→inter-cluster ad ogni singola chiamata interna alla catena di microservizi.

**Approccio adottato:** dopo ogni ciclo di scheduling del `frontend`, `DMOSOrchestrator._enforce_colocation()` (`dmos_main.py:293`) distribuisce i backend proporzionalmente alla distribuzione corrente del frontend:

```
1. Legge la distribuzione frontend attuale per cluster:
   cluster_parent_reps = {c: k8s.get_deployment_replicas("frontend", c)}
   total_parent_reps   = sum(cluster_parent_reps.values())

2. Per ogni backend dipendente (cartservice, productcatalogservice,
   checkoutservice, recommendationservice):

   frontend_share[c]       = cluster_parent_reps[c] / total_parent_reps
   proportional_target[c]  = max(min_replicas,
                                 round(total_backend_reps × frontend_share[c]))

   if current_backend[c] < proportional_target[c]:
       k8s.scale_deployment(backend, cluster=c, replicas=proportional_target[c])
```

La co-location scala **solo in su** — non scala mai giù un backend che ha già più repliche del target proporzionale. Il scale-down dei backend è gestito separatamente da una logica di *backend reset* nel `periodic_check_thread` (`dmos_main.py:755`):

- Se `frontend_total ≤ min_replicas` AND `backend_total > min_replicas × N_clusters` → reset immediato al minimo per cluster
- Se `backend_total > frontend_total × 2` → scale-down graduale via event queue

**Perché la co-location è corretta con Cilium Global Services:** Cilium non espone un endpoint per-cluster ai microservizi — espone un ClusterIP globale e bilancia le connessioni sui pod disponibili globalmente. La probabilità che una chiamata `frontend@cluster1 → cartservice` vada su `cartservice@cluster1` è proporzionale alla frazione `pods_cluster1 / pods_total`. Con co-location attiva, questa frazione segue quella del frontend → la maggior parte delle chiamate interne rimane locale al cluster.

**Servizi dipendenti configurati** (`dmos_main.py:188`):
```python
service_dependencies = {
    'frontend': [
        'cartservice',
        'productcatalogservice',
        'checkoutservice',
        'recommendationservice',
    ]
}
```

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
| **Scaling totale** | ✅ Reattivo lineare: `MIN + (MAX-MIN) × traffic/threshold` | ✅ Proattivo: predizione EMA + derivata + PD controller |
| **Score cluster** | Bi-dimensionale: traffico locale + latenza×traffico remoto | Quadri-dimensionale: Φ_lat + Φ_cap + Φ_load + Φ_carbon |
| **Carbon-awareness** | ❌ Assente | ✅ Φ_carbon = exp(-ν·CI/CI_max) |
| **Hardware constraints** | ❌ Non verificati | ✅ Hard limits su CPU/RAM prima dello score |
| **Metrica traffico** | `hubble_http_requests_total{source="reserved:ingress"}` | `hubble_http_requests_total{destination_workload="frontend"}` |
| **Metrica latenza** | `ping_rtt_mean_seconds` (RTT inter-cluster, ICMP L3) | `hubble_http_request_duration_seconds_bucket` (p95 intra-cluster, HTTP L7) |
| **Predizione del traffico** | ❌ Nessuna — reattivo al traffico corrente | ✅ Trend-based (EMA + derivata), orizzonte 120s |
| **Anti-flapping** | ❌ Assente | ✅ Dead zone + cooldown + scale-up protection |
| **Fairness** | Non misurata | ✅ Jain Fairness Index |
| **Vickrey auction** | ❌ Assente | ✅ Implementato (per future estensioni) |
| **Generatore di carico** | K6 | Locust (scenari programmabili) |
| **Pattern Prometheus** | PROM_MAP (per-cluster) | PROM_MAP (ereditato, stesso pattern) |
| **Applicazione test** | Online Boutique | Online Boutique (stesso deployment) |
| **Scaling backend** | ❌ Solo `frontend` scalato — backend statici | ✅ Co-location proporzionale: 4 backend scalati in funzione della distribuzione frontend |
| **Namespace** | `default` | `online-boutique` |
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
| **Φ_lat neutro in cold start** | Il fallback `baseline_ms=2.0ms` identico per tutti annulla la discriminazione per latenza in assenza di traffico HTTP | La decisione in cold start dipende solo da Φ_cap, Φ_load, Φ_carbon |

---

### 9. Sviluppi futuri

Dall'analisi delle limitazioni emergono direzioni di estensione concrete, alcune direttamente collegate al baseline di Romano:

#### SF1 — Φ_network: latenza di rete come quinta dimensione dello score

La limitazione del fallback neutro di Φ_lat apre la strada all'introduzione di un componente dedicato alla topologia di rete, che erediterebbe direttamente il `ping_rtt_mean_seconds` di Romano:

```
Φ_i = ω_lat·Φ_lat + ω_cap·Φ_cap + ω_load·Φ_load + ω_carbon·Φ_carbon + ω_net·Φ_network
```

dove:

```
Φ_network(i) = exp(-γ · ping_rtt_mean_ms(i) / ping_max)
```

I due segnali sono **ortogonali e complementari**: Φ_lat misura lo stress applicativo attuale del cluster (segnale dinamico, dipende dal carico), Φ_network misura la vicinanza topologica del cluster agli altri (segnale stabile, dipende dalla rete). Entrambi sono necessari per uno scheduler completo.

**Limite specifico del lab:** nell'infrastruttura attuale tutti e tre i cluster sono sulla stessa LAN (`192.168.1.x`) con ping inter-cluster di ~0.1-0.3ms — valori praticamente indistinguibili. Φ_network non fornirebbe discriminazione utile in questo ambiente. Il suo valore emerge in deployment **realmente geo-distribuiti**:

| Rotta | RTT LAN (lab attuale) | RTT WAN (deployment reale) |
|-------|----------------------|---------------------------|
| DE → FR | ~0.2ms | ~15ms |
| DE → PL | ~0.2ms | ~25ms |
| FR → PL | ~0.2ms | ~20ms |

In un deployment reale, Φ_network guiderebbe il routing verso cluster geograficamente vicini all'utente finale, mentre Φ_lat correggerebbe in tempo reale quando un cluster vicino è saturo.

**Tre opzioni di integrazione:**

| Opzione | Descrizione | Complessità |
|---------|-------------|-------------|
| **A — Fallback dinamico** | Usare `ping_rtt_mean_seconds` al posto di `baseline_ms=2.0` quando Hubble non è disponibile | Minima — solo modifica al `get_latency_p95()` |
| **B — Quinta dimensione** | Aggiungere `Φ_network` come componente esplicita con peso `ω_net` | Media — modifica a `ScoreFunctions` e config |
| **C — Fusione in Φ_lat** | `E[L_i] = α·L_http(i) + (1-α)·L_ping(i)` con α=1 se Hubble disponibile | Media — modifica alla formula di Φ_lat |

L'opzione B è la più pulita architetturalmente perché mantiene la separazione tra i due segnali e consente di pesarli indipendentemente tramite config.

#### SF2 — Predizione ML per traffico periodico

Sostituire il TrafficPredictor trend-based lineare con un modello che apprende pattern periodici (LSTM, Prophet). Romano stesso suggerisce *"logiche di autoscaling predittivo basate su machine learning"* come sviluppo futuro (§6.2). L'infrastruttura DMOS è già pronta: basterebbe sostituire `TrafficPredictor.predict()` con un modello addestrato offline.

#### SF3 — Carbon intensity in tempo reale

Integrare le API di Electricity Maps per aggiornare `CI_i(t)` in tempo reale invece di usare valori statici da config. Richiederebbe un thread di polling asincrono in `CarbonClient` e gestirebbe variazioni giornaliere (es. FR nuclear più verde di notte, DE solar più verde a mezzogiorno).

#### SF4 — Metriche L7 per i backend gRPC

Estendere la CNP L7 anche ai backend (cartservice, productcatalog, ecc.) per ottenere `hubble_http_requests_total` anche per i servizi gRPC. Attualmente DMOS usa `container_network_receive_bytes_total / 4000` come stima empirica (±20-30%). Con CNP L7 sui backend la misurazione diventerebbe esatta.

---

### 10. Posizionamento scientifico

DMOS si colloca nel filone della ricerca su **proactive autoscaling** per ambienti multicluster, integrando:

1. **Orchestrazione multi-obiettivo** (scheduling): estende il lavoro di Romano aggiungendo capacità, carico e carbon alla funzione di score, avvicinandosi agli approcci di Chiaro et al. [5] su latency-aware scheduling nel Cloud-Edge Continuum.

2. **Autoscaling predittivo** (scaling): implementa un controllore PD con predizione trend-based, categoria documentata nella letteratura come *proactive scaling* o *predictive HPA*, alternativa all'HPA nativo di Kubernetes (che è puramente reattivo e basato su CPU/memoria).

3. **Green computing**: l'integrazione di `Φ_carbon` nella funzione di score pone DMOS nella categoria dei **carbon-aware schedulers**, campo in crescita con l'aumento della sensibilità ambientale dei provider cloud (es. Google Cloud Carbon Footprint, Azure Sustainability).

4. **Infrastruttura Cilium**: l'uso di Hubble L7 per la misurazione del traffico è una scelta originale rispetto alla letteratura, che tipicamente usa metriche Kubernetes native (CPU, memory) o Istio. Cilium/Hubble non richiede sidecar aggiuntivi (è parte del CNI) ed è disponibile gratuitamente in ambienti K3s.

---

*Documento redatto sulla base di: G. Romano (2025), codice sorgente DMOS (src/), esperimento `double_wave_hubble` (2026-03-01).*
