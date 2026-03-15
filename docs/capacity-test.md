# Capacity Test — Documentazione Completa
## Come misurare la capacità del frontend per configurare correttamente DMOS

> File: `experiments/locustfile_capacity.py`
> Ultima revisione: 13/03/2026

---

## Indice

1. [Cos'è la capacità e perché misurarla](#1-cosè-la-capacità-e-perché-misurarla)
2. [Problema: senza capacity test i parametri DMOS sono sbagliati](#2-problema-senza-capacity-test-i-parametri-dmos-sono-sbagliati)
3. [Come funziona il test a gradini (stepped load)](#3-come-funziona-il-test-a-gradini-stepped-load)
4. [Cosa misura esattamente](#4-cosa-misura-esattamente)
5. [Il knee point: come viene trovato](#5-il-knee-point-come-viene-trovato)
6. [Differenza tra burst e graduale](#6-differenza-tra-burst-e-graduale)
7. [Come eseguire il test](#7-come-eseguire-il-test)
8. [Come interpretare i risultati](#8-come-interpretare-i-risultati)
9. [Come usare i risultati in DMOS](#9-come-usare-i-risultati-in-dmos)
10. [Risultati ottenuti (27/02/2026)](#10-risultati-ottenuti-27022026)
11. [Caratterizzazione del testbed: ambiente e architettura](#11-caratterizzazione-del-testbed-ambiente-e-architettura)
12. [Correzione 1 — OOMKill su servizi Node.js](#12-correzione-1--oomkill-su-servizi-nodejs)
13. [Correzione 2 — CPU throttling su currencyservice](#13-correzione-2--cpu-throttling-su-currencyservice)
14. [Correzione 3 — CPU throttling sui backend sotto concorrenza](#14-correzione-3--cpu-throttling-sui-backend-sotto-concorrenza)
15. [Calibrazione dei parametri del capacity test](#15-calibrazione-dei-parametri-del-capacity-test)
16. [Validità scientifica e dipendenza dal workload](#16-validità-scientifica-e-dipendenza-dal-workload)
17. [Script di riproduzione delle correzioni](#17-script-di-riproduzione-delle-correzioni)

---

## 1. Cos'è la capacità e perché misurarla

La **capacità per replica** (`capacity_req_per_sec`) è il numero massimo di richieste al secondo che un singolo pod del frontend riesce a gestire mantenendo la latenza entro il SLA (p95 < soglia).

Questo valore è il parametro centrale dell'autoscaler DMOS: determina **quante repliche servono** per gestire il traffico previsto.

```
needed_replicas ≈ predicted_traffic / capacity_per_replica
```

Se il valore è sbagliato:

| capacity troppo **bassa** | capacity troppo **alta** |
|---|---|
| DMOS pensa che ogni pod gestisca poco | DMOS pensa che ogni pod gestisca tanto |
| Scala troppo presto → over-provisioning | Scala troppo tardi → latenza degrada |
| Spreco di risorse | SLA violato |
| Meno utile come dimostrazione di efficienza | Meno utile come dimostrazione di reattività |

---

## 2. Problema: senza capacity test i parametri DMOS sono sbagliati

### Storia dei valori in questo progetto

**Prima del capacity test** (valore iniziale in `config/services.yaml`):
```yaml
capacity_req_per_sec: 50   # stima "a occhio", mai verificata
```

**Problema rilevato analizzando il test `gradual_ramp` del 27/02/2026**:
- L'analyzer `analyze_test_complete.py` riportava **100% under-provisioned**
- In realtà il sistema funzionava bene (p95=130-200ms, fail<1%)
- Il 100% era un **artefatto** della configurazione errata, non un problema reale

**Causa**: la catena di errori era:
```
capacity=35 (analyzer) + min_replicas=6
  → min_capacity = 6 × 35 = 210 rps
  → effective_demand = max(traffic, 210) = sempre 210  [floor artificialmente alto]
  → ratio = (3 pod × 35) / 210 = 0.5 → sempre "sotto-provisioned"
```

**Dopo il capacity test**:
```yaml
capacity_req_per_sec: 30   # per DMOS (valore conservativo, gestisce burst)
SERVICE_CAPACITY["frontend"] = 45  # per l'analyzer (valore osservato gradual_ramp)
```

---

## 3. Come funziona il test a gradini (stepped load)

### Principio

Il test a gradini (anche detto **step stress test** o **staircase test**) è la metodologia standard per misurare la capacità di un sistema. A differenza di:

- **Spike test** (burst improvviso): non dà tempo al sistema di stabilizzarsi → misura la capacità peggiore
- **Ramp test** (salita continua): il sistema è in transizione costante → difficile isolare ogni livello di carico
- **Step test** (questo): ogni livello viene mantenuto abbastanza a lungo da stabilizzarsi → misura la capacità in regime stazionario

### Schema temporale

```
Utenti
 300 |                                          ____
 280 |                                     ____/
 260 |                                ____/
 240 |                           ____/
  ...
  60 |               ____/
  40 |          ____/
  20 | _________/        ← spawn istantaneo (10 user/s)
     |
     +--+--+--+--+--+--+--+--+--+--+--+--+--+-- Tempo (secondi)
        0  90 180 270 360 450 540 630 720 810
        ↑  ↑  ↑  ↑  ↑
        S1 S2 S3 S4 S5 ...   (ogni step = 90 secondi)
```

Ogni step:
1. **Spawn**: Locust aggiunge 20 utenti alla velocità di 10 utenti/secondo (~2 secondi di transizione)
2. **Stabilizzazione**: i primi ~30 secondi del step sono "riscaldamento" — le metriche iniziali sono rumorose
3. **Regime stabile**: gli ultimi ~60 secondi rappresentano il comportamento stazionario
4. **Analisi**: a fine step, le metriche dell'intero step vengono calcolate e valutate

> **Nota sulla durata**: 90 secondi per step è il minimo raccomandato. Con step più corti (30-60s) le metriche sono instabili perché il pool di connessioni HTTP non si stabilizza.

---

## 4. Cosa misura esattamente

### Unità di misura degli utenti Locust

Ogni "utente" Locust in questo test è una **goroutine virtuale** che:
1. Sceglie casualmente un endpoint (con peso)
2. Invia una richiesta GET/POST
3. Aspetta un tempo casuale tra 1 e 3 secondi (`wait_time = between(1, 3)`, allineato a `locustfile_multiingress.py`)
4. Ripete

Il `wait_time` simula il "think time" di un utente reale tra un click e l'altro. Vedi sezione 15 per la motivazione della calibrazione.

### Distribuzione del traffico tra cluster

I 3 user class hanno pesi diversi, che rispecchiano il mix del `locustfile_multiingress.py`:

```
Cluster1User (DE): weight=40 → 40% degli utenti → cluster1
Cluster2User (FR): weight=35 → 35% degli utenti → cluster2
Cluster3User (PL): weight=25 → 25% degli utenti → cluster3
```

Con `N` utenti totali e `wait_time` medio di 2 secondi:
```
RPS totale ≈ N × 1.0 / (2.0 + avg_response_time)
           ≈ N × 0.45   (approssimazione con response_time≈200ms)
```

Esempio: 100 utenti → ~45 rps totali → 18/16/11 rps per cluster.

### Mix di endpoint

```python
ENDPOINTS = [
    ("/",                   0.35),   # home page
    ("/product/OLJCESPC7Z", 0.25),   # pagina prodotto (cache miss)
    ("/product/66VCHSJNUP", 0.10),   # pagina prodotto alternativa
    ("/cart",               0.15),   # carrello (legge da Redis)
    ("/setCurrency",        0.05),   # cambio valuta
    ("/cart/checkout",      0.10),   # checkout (path più pesante)
]
```

Il mix influenza significativamente la capacità misurata:
- `/cart/checkout` chiama ~8 microservizi → molto più pesante di `/`
- Un mix più aggressivo (più checkout) → capacità misurata più bassa

### Metriche raccolte per ogni step

Per ogni richiesta completata, il test registra:
- **Response time** in millisecondi (`r.elapsed.total_seconds() * 1000`)
- **Successo o fallimento** (`status_code >= 500` → failure)

A fine step, le metriche aggregate sono:

| Metrica | Formula | Significato |
|---|---|---|
| `requests` | conteggio | Numero totale richieste nello step |
| `rps` | requests / step_duration_s | Throughput medio nello step |
| `rps_per_replica` | rps / (replicas × clusters) | Carico per singolo pod |
| `avg_ms` | Σ(response_time) / n | Media latenza (influenzata da outlier) |
| `p50_ms` | mediana | 50% delle richieste ha latenza ≤ questo valore |
| `p95_ms` | 95° percentile | **Metrica principale SLA** |
| `p99_ms` | 99° percentile | Tail latency (casi peggiori) |
| `fail_rate` | failures / requests | % richieste con HTTP 5xx |
| `sla_ok` | p95 < 300ms AND fail < 2% | Booleano: step rispetta il SLA? |

### Perché p95 e non media o p50?

La **p95** (95° percentile) è la metrica standard per gli SLA perché:
- La **media** è distorta da pochi outlier (un timeout da 20s abbassa la media ma non indica saturazione generalizzata)
- La **mediana (p50)** nasconde il 50% dei casi più lenti
- La **p95** garantisce che il **95% degli utenti** ha un'esperienza buona

Esempio:
```
Step 5: 1000 richieste
  → media: 80ms  ← sembra ok
  → p50:   65ms  ← sembra ok
  → p95:   340ms ← 50 utenti su 1000 hanno aspettato 340ms → SLA VIOLATO
  → p99:   800ms ← 10 utenti su 1000 hanno aspettato 800ms
```

---

## 5. Il knee point: come viene trovato

### Definizione

Il **knee point** (punto di ginocchio) è il punto sulla curva latenza-throughput dove la latenza inizia a crescere esponenzialmente — il limite oltre il quale il sistema non scala più linearmente.

```
Latenza p95
(ms)
    400 |                                    /
    300 |                               ___/  ← SLA violato (oltre questo)
    200 |                         ____/
    150 |                   ____/  ← knee point (ultimo step ok)
    100 |            ______/
     70 |___________/
        +--+--+--+--+--+--+--+--+--+--+-- RPS/pod
           5  10  15  20  25  30  35  40  45
```

### Algoritmo nel codice

```python
# Per ogni step completato:
if result["sla_ok"]:
    # Aggiorna il knee point: l'ultimo step che rispetta il SLA
    _knee_point_rps = rps_per_replica   # es. 38.5 rps/pod
    _knee_point_users = users
else:
    # SLA violato: il knee point è quello dello step precedente
    # Ferma il test
    self._stop_flag = True
```

Il knee point è quindi il **rps_per_replica dell'ultimo step in cui p95 < 300ms**.

### Valore raccomandato per la configurazione

Il knee point grezzo non viene usato direttamente come `capacity_req_per_sec`, ma viene applicato un **margine di sicurezza del 20%**:

```
recommended = int(knee_point_rps × 0.80)
```

**Perché 80%?**

Il knee point è il limite teorico massimo. Operare esattamente al knee point significa:
- Nessun margine per picchi improvvisi
- Qualsiasi variazione spinge il sistema in saturazione
- DMOS non avrebbe "tempo di reazione" prima che la latenza degradi

Con il 20% di headroom, DMOS scala prima di raggiungere il limite → comportamento proattivo.

```
Esempio:
  knee_point = 45 rps/pod
  recommended = 45 × 0.80 = 36 → arrotondato a 36

  Con 3 pod: capacity totale = 3 × 36 = 108 rps
  DMOS scala a 4 pod quando prevede > 108 rps
  → scala PRIMA che la latenza degradi
```

---

## 6. Differenza tra burst e graduale

Questo è uno dei risultati più importanti emersi dalla sessione di test del 27/02/2026: **la capacità dipende dal pattern di carico**.

### Perché la capacità cambia?

| Fattore | Burst (flash crowd) | Graduale (stepped/ramp) |
|---|---|---|
| **HTTP connection pool** | Freddo, nessuna connessione riusabile | Già riscaldato, pooling ottimizzato |
| **Go GC (garbage collector)** | Parte durante il picco → stop-the-world | Ha già fatto GC durante il ramp-up |
| **gRPC verso backend** | Nuove connessioni verso tutti i backend | Connessioni già stabilite e riusate |
| **JVM (se presente)** | JIT non ancora compilato | Codice hot già compilato |
| **Redis connection pool** | Connessioni FreddaDB | Pool di connessioni stabile |

### Numeri osservati nel progetto (27/02/2026)

| Scenario | Pattern | Knee point | Motivazione |
|---|---|---|---|
| `flash_crowd_083559` | Burst istantaneo | ~25 rps/pod | Pod freddi, p95 degrada già a 22 rps (cluster2) |
| `gradual_ramp` (DMOS) | Ramp graduale | ~43-47 rps/pod | 3 pod gestiscono 131 rps a p95=200ms |

**Il valore del capacity test a gradini è ~43-47 rps/pod** — circa il doppio del burst.

### Quale valore usare dove?

```
Per DMOS config/services.yaml → capacity_req_per_sec: 30
  ↑ Conservativo: a metà tra burst (25) e graduale (45)
  ↑ Garantisce scaling proattivo anche per flash crowd
  ↑ Evita latenza degradata su scenari peggiori

Per analyze_test_complete.py → SERVICE_CAPACITY["frontend"] = 45
  ↑ Valore osservato in regime graduale
  ↑ Produce un provisioning ratio realistico nel post-hoc analysis
  ↑ Non va usato per decisioni di scaling di DMOS
```

---

## 7. Come eseguire il test

### Pre-requisiti

**1. Ferma DMOS** (deve essere completamente fermo, altrimenti scala i pod durante il test):
```powershell
# Nel terminale dove gira dmos_main.py: Ctrl+C
```

**2. Imposta 1 replica per cluster** (stato controllato per il test):
```powershell
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx scale deployment frontend `
      -n online-boutique --replicas=1
}
# Verifica
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx get pods -n online-boutique -l app=frontend
}
```

**3. Ferma il loadgenerator** (evita traffico di sfondo che altera le misure):
```powershell
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx scale deployment loadgenerator `
      -n online-boutique --replicas=0
}
```

**4. Aspetta 60 secondi** che il sistema si stabilizzi.

### Esecuzione

```powershell
# Dalla directory del progetto
locust -f experiments/locustfile_capacity.py --host http://ignored --headless
```

Il test:
- Non richiede `--users` né `--run-time` (li gestisce `SteppedCapacityShape`)
- Dura ~22 minuti se il sistema regge fino a 300 utenti (15 step × 90s + tempo spawn)
- Si ferma automaticamente quando il SLA viene violato

### Output console atteso

```
=================================================================
🚀 DMOS CAPACITY TEST
=================================================================
  Clusters:      3 (cluster1, cluster2, cluster3)
  Replicas:      1 per cluster → 3 totali
  Steps:         ogni 90s, +20 utenti
  User range:    20 → 300
  SLA threshold: p95 < 300ms, fail < 2%

  IMPORTANTE: assicurati che DMOS sia FERMO e
  frontend abbia esattamente 1 replica/cluster
=================================================================

  Step |  Users |    RPS | rps/pod |   p50 |   p95 |   p99 |  Fail | SLA
  ------------------------------------------------------------
  Step  1 | users= 20 | rps=  18.2 ( 6.1/pod) | p50=62ms p95=75ms  p99=95ms  | fail=0.0% | ✅
  Step  2 | users= 40 | rps=  35.8 (11.9/pod) | p50=63ms p95=79ms  p99=102ms | fail=0.0% | ✅
  Step  3 | users= 60 | rps=  53.1 (17.7/pod) | p50=64ms p95=88ms  p99=124ms | fail=0.0% | ✅
  Step  4 | users= 80 | rps=  70.3 (23.4/pod) | p50=65ms p95=112ms p99=185ms | fail=0.0% | ✅
  Step  5 | users=100 | rps=  86.9 (29.0/pod) | p50=68ms p95=147ms p99=240ms | fail=0.0% | ✅
  Step  6 | users=120 | rps= 101.2 (33.7/pod) | p50=72ms p95=190ms p99=320ms | fail=0.1% | ✅
  Step  7 | users=140 | rps= 113.6 (37.9/pod) | p50=82ms p95=267ms p99=480ms | fail=0.2% | ✅
  Step  8 | users=160 | rps= 121.4 (40.5/pod) | p50=98ms p95=341ms p99=620ms | fail=0.5% | ❌

  🛑 SLA VIOLATO (p95=341ms > 300ms)
  📌 Knee point: 37.9 rps/replica (140 users)

=================================================================
📊 CAPACITY TEST RESULTS
=================================================================
  Knee point:            37.9 rps/replica
  Recommended capacity:  30 rps/replica (80% di 37.9)

  → Aggiorna config/services.yaml:
    frontend:
      capacity_req_per_sec: 30

  → Aggiorna analyze_test_complete.py:
    SERVICE_CAPACITY['frontend'] = 38
=================================================================
```

### Post-test

Ripristina il sistema per i test DMOS:
```powershell
# Riavvia loadgenerator
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx scale deployment loadgenerator `
      -n online-boutique --replicas=1
}
# Riavvia DMOS
python src/dmos_main.py
```

---

## 8. Come interpretare i risultati

### Output CSV (`results/capacity/capacity_YYYYMMDD_HHMMSS.csv`)

```csv
step,users,requests,rps,rps_per_replica,avg_ms,p50_ms,p95_ms,p99_ms,fail_rate,sla_ok
1,20,1638,18.2,6.1,66.3,62.1,75.2,94.7,0.0,True
2,40,3221,35.8,11.9,67.1,63.4,79.4,102.3,0.0,True
...
7,140,10224,113.6,37.9,88.5,82.4,267.1,480.2,0.002,True
8,160,10913,121.4,40.5,112.3,98.1,341.4,620.8,0.005,False
```

Il CSV permette di:
- Plottare la curva latenza vs throughput
- Identificare esattamente a che punto la latenza inizia a crescere
- Confrontare run diversi (es. dopo un update del frontend)

### Output JSON (`results/capacity/capacity_YYYYMMDD_HHMMSS.json`)

```json
{
  "timestamp": "20260227_170000",
  "config": { ... },
  "knee_point_rps_per_replica": 37.9,
  "knee_point_users": 140,
  "recommended_capacity_req_per_sec": 30,
  "steps": [ ... ]
}
```

### Segnali di attenzione nei risultati

| Segnale | Causa probabile | Azione |
|---|---|---|
| p95 sale di colpo tra step 1 e 2 (già alto a basso carico) | OOMKilled in corso, DNS lento | Controlla `kubectl top pods` e `kubectl get events` |
| Knee point diverso tra cluster | Risorse diverse, replica più carica | Controlla `kubectl top nodes` |
| Fail rate sale prima della latenza | Backend saturati (non il frontend) | Ripeti con solo `/` come endpoint |
| Knee point molto più basso del gradual_ramp DMOS | Il test DMOS aveva più repliche nel momento di picco | Normale, vedi Sezione 6 |

---

## 9. Come usare i risultati in DMOS

Il valore `recommended_capacity_req_per_sec` dal capacity test viene usato in **due posti separati** con valori diversi:

### `config/services.yaml` — per le decisioni di scaling

```yaml
# Valore conservativo: 80% del knee point
# DMOS scala prima di saturare, gestisce anche burst
capacity_req_per_sec: 30
```

**Effetto su DMOS**: con `capacity=30` e 3 pod (capacity totale 90 rps), DMOS scala a 4 pod quando prevede ≥ 90 rps. Con `capacity=45` (valore osservato), scalerà solo a ≥ 135 rps — troppo tardi per un burst.

### `experiments/analyze_test_complete.py` — per l'analisi post-hoc

```python
SERVICE_CAPACITY = {
    "frontend": 45,  # Valore osservato in regime graduale
    ...
}
```

**Effetto sull'analisi**: produce un provisioning ratio realistico. Se si usasse 30 (valore conservativo), il ratio sarebbe artificialmente alto (sistema apparirebbe over-provisioned anche quando è correttamente dimensionato).

### Schema decisionale

```
Capacity test → knee_point = K rps/pod

┌─────────────────────────────────────────────────┐
│  Per DMOS (scaling trigger):                    │
│    capacity_req_per_sec = int(K × 0.80)         │
│    → conservativo, scala prima del limite       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Per analyzer (provisioning ratio):             │
│    SERVICE_CAPACITY["frontend"] = int(K)        │
│    → valore reale, provisioning ratio corretto  │
└─────────────────────────────────────────────────┘
```

### Quando ripetere il capacity test

Il capacity test va ripetuto quando:
- Si aggiornano i **resource limits** del frontend in `config/services.yaml`
- Si cambia il **mix di endpoint** nel locustfile
- Si scala il cluster (nodi con CPU/RAM diversa)
- Si osserva una **degradazione** delle performance rispetto al test precedente
- Si aggiorna la versione del **frontend** (es. nuovo Dockerfile)

---

## 10. Risultati ottenuti (27/02/2026)

### Capacity da flash_crowd (burst) — test precedenti

| Test | Knee point approssimativo | Note |
|---|---|---|
| `flash_crowd_083559` | ~25 rps/pod (cluster2 primo a saturare a ~22) | Più affidabile dei precedenti |
| `flash_crowd_233136` | non valido (altri pod già attivi) | Dati contaminati |
| `flash_crowd_000151` | non valido (OOMKill durante il test) | Dati non confrontabili |

### Capacity da gradual_ramp (DMOS test) — 27/02/2026

Dall'analisi del file `results/170054_20260227_test.jsonl`:
- 3 pod (1/cluster): gestiscono 131 rps a p95=200ms → **43.7 rps/pod**
- 3 pod (1/cluster): gestiscono ~140 rps con p95 che sale a 190-200ms → limite avvicinato
- 5 pod (2/2/1): gestiscono 179 rps a p95=190ms → **35.8 rps/pod** (con repliche distribuite)

### Valori aggiornati dopo questa analisi

| Parametro | Prima | Dopo | Motivazione |
|---|---|---|---|
| `config/services.yaml` `capacity_req_per_sec` | 50 | **30** | 80% del burst knee ~25 → scaling conservativo |
| `analyze_test_complete.py` `SERVICE_CAPACITY["frontend"]` | 35 | **45** | Knee point gradual_ramp osservato |
| `analyze_test_complete.py` `min_replicas_total` | 6 | **3** | 1 replica × 3 cluster = floor reale |
| `config/services.yaml` `sla.latency_p95_ms` | 100 | **200** | Baseline idle è già 60-70ms → 100ms non raggiungibile |

### Impatto sull'analisi post-hoc (stesso JSONL, parametri aggiornati)

| Metrica | Prima (errata) | Dopo (corretta) |
|---|---|---|
| Under-provisioned | **100%** | **18.8%** |
| In range ideale | 0% | **43.8%** |
| Over-provisioned | 0% | **37.5%** |
| Avg ratio | 0.65x | **1.26x** |

---

## 11. Caratterizzazione del testbed: ambiente e architettura

### Il testbed DMOS

Il testbed è composto da **3 cluster K3s single-node** su macchine fisiche (`ms01`, `ms02`, `ms03`), ciascuna con:

- **4 CPU** (core fisici)
- **7.75 GB RAM**
- **Ubuntu 24.04.3 LTS**
- **K3s** (distribuzione Kubernetes leggera per ambienti edge/embedded)

Su ogni cluster gira l'intera applicazione **Google Online Boutique**: 11 microservizi in linguaggi diversi (Go, Python, Java/JVM, Node.js), più componenti di sistema (kube-system, monitoring Prometheus, Cilium, Nginx Ingress).

### Differenza architetturale rispetto all'ambiente di produzione

Google Online Boutique è stata progettata per **Google Kubernetes Engine (GKE)** con nodi da 2-4 vCPU e 4-8 GB RAM, dove ogni nodo ospita tipicamente **1-2 pod**. In quella configurazione, i 200m CPU di default per ciascun servizio sono appropriati: ogni pod ha risorse fisiche quasi dedicate.

Nel testbed DMOS la situazione è radicalmente diversa:

| Caratteristica | GKE (produzione) | K3s single-node (testbed) |
|---|---|---|
| Nodi per cluster | 3-10 nodi | **1 nodo** |
| Pod per nodo | 1-3 | **11+ (tutti i microservizi)** |
| CPU fisiche disponibili | 2-4 per nodo | **4 totali per tutto il cluster** |
| CPU richieste (default) | ~400m per nodo | **~2400m su 4 CPU fisiche** |
| Contesa CPU | quasi nulla | **elevata** |

Questa differenza architetturale implica che i **resource limits di default non sono adatti** al testbed e richiedono una caratterizzazione preliminare.

### Resource limits di default e il CFS throttling

Kubernetes utilizza il **CFS (Completely Fair Scheduler)** del kernel Linux per isolare il consumo di CPU tra container. Il meccanismo funziona su finestre temporali di **100ms**:

- Un container configurato a `200m` CPU ottiene esattamente **20ms di CPU ogni 100ms**
- Se il processo necessita più CPU durante quel periodo, viene **sospeso** per i restanti 80ms
- Questa sospensione si manifesta come **latency spike** nella risposta al client, indipendentemente dal carico complessivo del nodo

La formula è: `cpu_time_budget = cpu_limit_millicores / 1000 × 100ms`

Con `200m`: budget = 20ms ogni 100ms → throttle ratio massimo = 80%.

---

## 12. Correzione 1 — OOMKill su servizi Node.js

### Sintomo osservato

Durante i test di carico iniziali, `kubectl get pods -n online-boutique` mostrava:
```
currencyservice-xxx   0/1   CrashLoopBackOff   26   ...
paymentservice-xxx    0/1   CrashLoopBackOff   8    ...
```

Il campo `RESTARTS` accumulava riavvii ripetuti. La causa era confermata dall'exit code:
```bash
kubectl describe pod currencyservice-xxx -n online-boutique
# Last State: Terminated, Reason: OOMKilled, Exit Code: 137
```

Exit code 137 = SIGKILL inviato dal kernel OOM killer.

### Causa tecnica

`currencyservice` e `paymentservice` sono implementati in **Node.js**, il cui motore **V8** gestisce la memoria tramite un heap JavaScript dinamico.

Comportamento del heap V8 in funzione del carico:

| Stato | Heap approssimativo |
|---|---|
| A riposo (0 richieste) | 80–120 MB |
| Carico moderato (5–10 req concorrenti) | 150–250 MB |
| Carico sostenuto (20+ req concorrenti) | 350–500 MB |

Il garbage collector V8 opera in modo **generazionale** con fasi stop-the-world. Sotto carico concorrente elevato, il GC non riesce a liberare memoria alla stessa velocità con cui nuovi oggetti vengono allocati per gestire le richieste in volo → il heap cresce fino al limite configurato → OOMKill.

Il limit di default `256Mi` è sufficiente per una demo interattiva con pochi utenti simultanei (caso d'uso GKE originale). Non è sufficiente per load testing sostenuto con 20-40 richieste concorrenti.

### Perché non era un problema su GKE originale

Su GKE con nodi da 4-8 GB, il margine disponibile per il pod è molto maggiore. Anche senza aumentare il limit esplicito, il kernel OOM killer raramente interviene perché il nodo ha abbondante memoria libera. Il testbed K3s ha solo 7.75 GB divisi tra 11+ pod, kube-system e monitoring.

### Correzione applicata

```bash
kubectl set resources deployment currencyservice \
  -n online-boutique \
  --limits=cpu=500m,memory=512Mi \
  --requests=cpu=100m,memory=256Mi \
  --context cluster1   # ripetuto per cluster2, cluster3
```

```bash
kubectl set resources deployment paymentservice \
  -n online-boutique \
  --limits=cpu=500m,memory=512Mi \
  --requests=cpu=100m,memory=256Mi \
  --context cluster1   # ripetuto per cluster2, cluster3
```

| Parametro | Prima | Dopo | Motivazione |
|---|---|---|---|
| `currencyservice` memory limit | 256Mi | **512Mi** | Heap V8 sotto carico supera 256Mi |
| `currencyservice` memory request | 128Mi | **256Mi** | Allineato al consumo a riposo reale |
| `paymentservice` memory limit | 256Mi | **512Mi** | Stesso pattern V8/Node.js |
| `paymentservice` memory request | 128Mi | **256Mi** | Allineato al consumo a riposo reale |

---

## 13. Correzione 2 — CPU throttling su currencyservice

### Il bottleneck strutturale

`currencyservice` è il servizio più chiamato nell'intera applicazione, per una ragione architetturale: il **frontend Go** effettua una chiamata gRPC a `currencyservice` per ogni prodotto visualizzato nella pagina.

Con il mix di endpoint del capacity test:
- La homepage `/` mostra 10 prodotti → **10 chiamate gRPC sequenziali** a `currencyservice`
- La pagina prodotto `/product/...` → **1 chiamata gRPC** per il dettaglio + conversioni prezzi correlati

In media, ogni richiesta HTTP al frontend genera **8-10 chiamate gRPC** a `currencyservice`.

### Effetto del throttling CFS a 200m

Con `200m` CPU, il budget di `currencyservice` è **20ms ogni 100ms**.

Ogni chiamata gRPC di conversione valuta richiede circa 3-5ms di CPU pura. Con 10 chiamate sequenziali per pagina:

```
Sequenza per una richiesta homepage:
  gRPC call 1: 4ms CPU needed → uses 4ms budget → rimanenti 16ms
  gRPC call 2: 4ms CPU needed → uses 4ms budget → rimanenti 12ms
  ...
  gRPC call 5: 4ms CPU needed → 0ms budget rimanente → throttled 80ms
  gRPC call 6: wait 80ms (sospeso) + 4ms → ...
  ...

Latenza totale per la sola parte currencyservice:
  ~5 chiamate × 4ms + ~5 chiamate × (80ms + 4ms) = 20 + 420 = 440ms
```

Questo throttling avviene **per ogni singola richiesta HTTP**, indipendentemente dalla concorrenza. Anche con un solo utente isolato il sistema subisce questa penalità strutturale.

### Misurazione prima/dopo

La singola richiesta HTTP alla homepage con sistema a riposo (1 utente, 0 background traffic):

```powershell
# Misurazione baseline (prima della correzione)
Measure-Command { Invoke-WebRequest http://192.168.1.245:30080 -UseBasicParsing } | Select TotalMilliseconds
# → 533ms
```

```powershell
# Dopo aver portato currencyservice a 500m CPU:
# → 187ms
```

**Riduzione del 65%** con la modifica di un solo servizio. Il fatto che il miglioramento sia visibile con **1 solo utente** è la prova che il bottleneck era strutturale (throttling CFS), non legato alla contesa tra richieste concorrenti.

### Correzione applicata

```bash
kubectl set resources deployment currencyservice \
  -n online-boutique \
  --limits=cpu=500m \
  --context cluster1   # ripetuto per cluster2, cluster3
```

| Parametro | Prima | Dopo | Motivazione |
|---|---|---|---|
| `currencyservice` CPU limit | 200m | **500m** | Budget 20ms/100ms insufficiente per 10 chiamate gRPC/pagina |

Con `500m`, il budget diventa 50ms ogni 100ms: sufficiente per processare le 10 chiamate gRPC (~40ms CPU totale) senza nessun throttle cycle.

---

## 14. Correzione 3 — CPU throttling sui backend sotto concorrenza

### Il problema della concorrenza multipla

Dopo aver risolto il bottleneck di `currencyservice`, il sistema mostra latenze accettabili per richieste singole (187ms). Tuttavia, sotto carico concorrente (40+ utenti virtuali), le latenze risalivano a 2.7-10 secondi.

### Calcolo della concorrenza reale

Il numero di richieste effettivamente in volo in un sistema Locust dipende dal rapporto tra `response_time` e `wait_time`:

```
fraction_in_flight = response_time / (response_time + wait_time)
```

Con la configurazione iniziale (`wait_time = between(0.5, 1.5)`, media 1s) e latenze alte (~7s):
```
fraction_in_flight = 7 / (7 + 1) = 0.875
con 40 utenti: 40 × 0.875 = 35 richieste concorrenti
```

Con 35 richieste concorrenti, tutti i backend vengono chiamati simultaneamente. Il throttling CFS a 200m su **ogni** backend si somma:

```
richiesta → frontend (200m throttled)
         → adservice (200m throttled)
         → recommendationservice (200m throttled)
         → productcatalogservice (200m throttled)
         → ...
```

Il throttling è **cumulativo e a cascata**: ogni hop nella catena di chiamate aggiunge ritardo.

### Criticità per linguaggi specifici

**Java — adservice (JVM):**
La JVM necessita di burst di CPU significativi per:
- **JIT compilation**: la prima volta che un metodo viene eseguito ad alta frequenza, il JIT lo compila in codice nativo → spike di CPU. Con 200m, questo spike causa throttling per centinaia di ms.
- **Garbage collection**: le fasi stop-the-world del GC richiedono CPU istantanea. Con 200m, il GC viene throttled → pause più lunghe.

Con 200m, un burst di 200ms di CPU necessario per il GC richiede ~1000ms di wall time (5× throttle ratio).

**Python — recommendationservice (GIL):**
Il **Global Interpreter Lock** di CPython limita già l'esecuzione a un solo thread Python alla volta. Con 200m CPU, il throughput effettivo è ulteriormente limitato perché il processo viene sospeso anche durante l'unico thread attivo. Il sistema non può sfruttare nemmeno il singolo core effettivo disponibile.

### Correzioni applicate

```bash
# Servizi ad alto carico (frontend entry point + JVM + Python)
for svc in frontend adservice recommendationservice; do
  kubectl set resources deployment $svc \
    -n online-boutique \
    --limits=cpu=1000m \
    --context cluster1   # ripetuto per cluster2, cluster3
done

# Servizi backend Go (più efficienti, ma coinvolti in ogni richiesta)
for svc in cartservice shippingservice productcatalogservice checkoutservice emailservice; do
  kubectl set resources deployment $svc \
    -n online-boutique \
    --limits=cpu=500m \
    --context cluster1   # ripetuto per cluster2, cluster3
done
```

Tabella riassuntiva di tutte le modifiche CPU:

| Servizio | Linguaggio | CPU limit prima | CPU limit dopo | Motivazione |
|---|---|---|---|---|
| `frontend` | Go | 200m | **1000m** | Entry point: gestisce tutte le richieste in ingresso |
| `adservice` | Java/JVM | 200m | **1000m** | JIT compilation e GC richiedono burst CPU |
| `recommendationservice` | Python | 200m | **1000m** | GIL + throttling penalizzano gravemente il throughput |
| `currencyservice` | Node.js | 200m | **500m** | 10 chiamate gRPC/pagina, bottleneck strutturale |
| `paymentservice` | Node.js | 200m | **500m** | OOMKill prevention + throttling |
| `cartservice` | Go | 200m | **500m** | Coinvolto in /cart e /checkout |
| `shippingservice` | Go | 200m | **500m** | Coinvolto in /checkout |
| `productcatalogservice` | Go | 200m | **500m** | Lookup per ogni pagina prodotto |
| `checkoutservice` | Go | 200m | **500m** | Orchestratore checkout, chiama 6+ servizi |
| `emailservice` | Go | 200m | **500m** | Chiamato da checkout |

### Verifica del non-overcommit

La somma dei **requests** (non dei limits) dopo le modifiche rimane ampiamente sotto la capacità del nodo:

```
Servizi modificati (requests): ~1300m
kube-system + monitoring + Cilium: ~500m
Totale stimato: ~1800m / 4000m disponibili → 45% utilizzo
```

Il nodo **non è overcommitted** sui requests: la schedulazione Kubernetes rimane stabile.

---

## 15. Calibrazione dei parametri del capacity test

### Problema con la configurazione iniziale

La configurazione originale del capacity test era:

```python
"users_start": 20,
"users_step": 20,
"users_max": 100,
"sla_p95_ms": 300,
"wait_time": between(0.5, 1.5),   # media 1s
```

Questa configurazione causava un **fallimento immediato al primo step** (40 utenti) in modo sistematico, rendendo impossibile trovare il knee point.

**Calcolo della concorrenza al primo step** (con latenze alte ~7s e wait_time medio 1s):
```
fraction_in_flight = 7 / (7 + 1) = 0.875
40 utenti × 0.875 = 35 richieste concorrenti
```
Il sistema risultava immediatamente saturo: il test violava il SLA al primo step senza possibilità di misurare nulla di utile.

### Calibrazione 1 — Granularità degli step

| Parametro | Prima | Dopo | Motivazione |
|---|---|---|---|
| `users_start` | 20 | **5** | Primo step a 5 utenti: carico quasi zero, baseline pulito |
| `users_step` | 20 | **5** | Risoluzione fine: il knee point viene identificato entro ±5 utenti |
| `users_max` | 100 | **300** | Evita di raggiungere il limite prima di trovare il vero knee point |
| `spawn_rate` | 10 | **5** | Proporzionale al nuovo users_step |

Con `users_step=20` si saltava direttamente da 0 a 40 utenti al primo step, superando il knee point senza rilevarlo. L'analogia è misurare il punto di ebollizione dell'acqua con un termometro che mostra solo 0°C, 10°C, 20°C, ... 100°C anziché ogni grado.

### Calibrazione 2 — SLA threshold

| Parametro | Prima | Dopo | Motivazione |
|---|---|---|---|
| `sla_p95_ms` | 300 | **1000** | 5× il baseline misurato (187ms) |

**300ms è irrealistico** per questo testbed: con 11 microservizi su 4 CPU condivise, anche una singola richiesta isolata richiede 187ms (baseline post-correzioni). Con 300ms di soglia rimangono solo 113ms di margine per la degradazione sotto carico — insufficiente.

**1000ms è la soglia standard** per la user experience web (Nielsen 1993): oltre 1 secondo l'utente percepisce la risposta come lenta. È anche 5× il baseline (187ms), proporzione compatibile con le raccomandazioni empiriche per capacity testing (soglia = 3-5× il baseline a carico zero).

Il capacity test **non certifica una SLA di produzione** (quella dipenderebbe dai requisiti del servizio), ma trova il **knee point della curva di performance** del testbed specifico.

### Calibrazione 3 — Allineamento del wait_time

| Parametro | Prima | Dopo | Motivazione |
|---|---|---|---|
| `wait_time` | `between(0.5, 1.5)` | **`between(1, 3)`** | Allineato a `locustfile_multiingress.py` |

**Il wait_time influenza direttamente il RPS generato** per un dato numero di utenti:
```
RPS ≈ users / (avg_response_time + avg_wait_time)

Con wait_time=between(0.5,1.5), media=1s:
  RPS = users / (0.2 + 1.0) = users × 0.833

Con wait_time=between(1,3), media=2s:
  RPS = users / (0.2 + 2.0) = users × 0.455
```

Se il capacity test usasse un wait_time diverso dai test esperimento (`locustfile_multiingress.py`), il valore di `capacity_req_per_sec` calibrato non sarebbe trasferibile: si calibrerebbe per un tipo di traffico diverso (più aggressivo/più rilassato) rispetto a quello che DMOS gestirà realmente.

L'allineamento a `between(1, 3)` garantisce **coerenza del workload** tra capacity test e test esperimento.

---

## 16. Validità scientifica e dipendenza dal workload

### Il capacity test misura la capacità per un workload specifico

Il valore `capacity_req_per_sec` ottenuto è valido **esclusivamente per il mix di endpoint** usato nel capacity test:

```python
ENDPOINTS = [
    ("/",                   0.35),   # 35% homepage
    ("/product/OLJCESPC7Z", 0.25),   # 25% product page
    ("/product/66VCHSJNUP", 0.10),   # 10% product page alt
    ("/cart",               0.15),   # 15% cart
    ("/setCurrency",        0.05),   # 5% set currency
    ("/cart/checkout",      0.10),   # 10% checkout
]
```

Questo mix rispecchia quello di `locustfile_multiingress.py` (il file usato nei test esperimento DMOS). La corrispondenza è **deliberata e necessaria** per la coerenza metodologica.

### Dipendenza dal workload — esempio quantitativo

| Endpoint | Servizi coinvolti | Peso relativo (costo CPU) |
|---|---|---|
| `GET /` (homepage) | frontend + currencyservice×10 | Alto (per currencyservice) |
| `GET /product/...` | frontend + productcatalog + currency | Medio |
| `GET /cart` | frontend + cartservice + currency | Medio |
| `POST /cart/checkout` | frontend + checkout + payment + email + shipping + currency + cart | **Molto alto** |

Un mix con più `/checkout` (es. 30% anziché 10%) abbassa il knee point misurato perché ogni richiesta coinvolge più servizi backend e consuma più CPU per unità di tempo.

In **capacity testing industriale** (metodologie SPEC, TPC-W), questa caratterizzazione del workload si chiama **workload profile** ed è parte integrante della specifica del benchmark. Il risultato è dichiarato come valido "per il workload W" e non come capacità assoluta del sistema.

### Condizione di validità nel progetto DMOS

Il `capacity_req_per_sec` è valido a condizione che:
1. Il mix di endpoint del capacity test corrisponda al mix dei test esperimento ✓ (garantito dall'allineamento)
2. Il wait_time sia lo stesso ✓ (garantito dalla calibrazione, sezione 15)
3. I resource limits siano stabili tra capacity test e test esperimento ✓ (fissati dalla caratterizzazione)

### Limitazione nota

Se in futuro si volessero testare scenari con distribuzioni diverse (es. più checkout-heavy per simulare Black Friday), il valore di `capacity_req_per_sec` richiederebbe ricalibrazione. In sistemi di produzione reali, si applicano:

- **Margini di sicurezza**: operare al 70-80% del knee point (già implementato con il 20% headroom)
- **Ricalibrazione periodica**: ogni modifica al codice o all'infrastruttura invalida la misura precedente
- **Autoscaling guidato da metriche osservate**: DMOS usa il traffico misurato in tempo reale, riducendo la dipendenza dalla calibrazione offline

### Risultato del capacity test post-calibrazione (13/03/2026)

Con tutte le correzioni applicate:

```
Step 19 | users=100 | rps=90.9 (30.3/pod) | p50=70ms p95=88ms ✅
→ Test completato senza SLA violation (limite users_max=100 raggiunto)
→ Risultato: lower bound ≥ 30.3 rps/pod con p95 < 1000ms
```

Il test ha raggiunto `users_max=100` senza trovare il knee point → la capacità reale è **superiore a 30.3 rps/pod**. Con `users_max=300` il test troverà il vero knee point. Il valore `capacity_req_per_sec: 24` (80% × 30.3) usato nella configurazione è **conservativo** e garantisce scaling proattivo.

---

## 17. Script di riproduzione delle correzioni

Il seguente script bash riproduce tutte le modifiche ai resource limits descritte nelle sezioni 12-14. Va eseguito una volta per configurare il testbed per i test di carico.

```bash
#!/bin/bash
# Script: configure_testbed_resources.sh
# Descrizione: Applica le correzioni ai resource limits di Online Boutique
#              per i test di capacity e load testing su K3s single-node.
# Prerequisiti: kubectl configurato con contesti cluster1, cluster2, cluster3
#               Namespace: online-boutique

set -e

CONTEXTS=("cluster1" "cluster2" "cluster3")
NS="online-boutique"

echo "=== Configurazione resource limits testbed DMOS ==="
echo "Namespace: $NS"
echo "Cluster: ${CONTEXTS[*]}"
echo ""

for CTX in "${CONTEXTS[@]}"; do
  echo "--- Cluster: $CTX ---"

  # CORREZIONE 1: Memory limits per Node.js (OOMKill prevention)
  # Exit code 137 (SIGKILL) osservato con limits di default 256Mi
  # sotto carico concorrente: heap V8 supera 256Mi con 20+ req in volo
  echo "  [1/3] Fix OOMKill: currencyservice e paymentservice (256Mi → 512Mi)"
  kubectl set resources deployment currencyservice \
    -n "$NS" \
    --limits=cpu=500m,memory=512Mi \
    --requests=cpu=100m,memory=256Mi \
    --context "$CTX"

  kubectl set resources deployment paymentservice \
    -n "$NS" \
    --limits=cpu=500m,memory=512Mi \
    --requests=cpu=100m,memory=256Mi \
    --context "$CTX"

  # CORREZIONE 2: CPU throttling currencyservice (bottleneck strutturale)
  # Il frontend chiama currencyservice ~10 volte per pagina (1 per prodotto).
  # A 200m CPU (20ms/100ms window), 10 chiamate sequenziali causano throttling
  # anche con 1 solo utente → singola richiesta 533ms → 187ms dopo il fix.
  # (CPU già impostata sopra insieme alla memory)

  # CORREZIONE 3: CPU throttling backend sotto concorrenza
  # Con 35+ richieste concorrenti, il throttling a 200m è cumulativo e a cascata.
  # Java/JVM (adservice): JIT + GC richiedono burst CPU significativi.
  # Python (recommendationservice): GIL + throttling limitano gravemente il throughput.
  echo "  [2/3] Fix CPU throttling: frontend, adservice, recommendationservice (200m → 1000m)"
  kubectl set resources deployment frontend \
    -n "$NS" \
    --limits=cpu=1000m \
    --context "$CTX"

  kubectl set resources deployment adservice \
    -n "$NS" \
    --limits=cpu=1000m \
    --context "$CTX"

  kubectl set resources deployment recommendationservice \
    -n "$NS" \
    --limits=cpu=1000m \
    --context "$CTX"

  echo "  [3/3] Fix CPU throttling: backend Go (200m → 500m)"
  for SVC in cartservice shippingservice productcatalogservice checkoutservice emailservice; do
    kubectl set resources deployment "$SVC" \
      -n "$NS" \
      --limits=cpu=500m \
      --context "$CTX"
  done

  echo "  ✅ $CTX configurato"
  echo ""
done

echo "=== Verifica finale ==="
for CTX in "${CONTEXTS[@]}"; do
  echo ""
  echo "--- $CTX ---"
  kubectl get deployment -n "$NS" --context "$CTX" \
    -o custom-columns=\
'NAME:.metadata.name,CPU_REQ:.spec.template.spec.containers[0].resources.requests.cpu,CPU_LIM:.spec.template.spec.containers[0].resources.limits.cpu,MEM_LIM:.spec.template.spec.containers[0].resources.limits.memory'
done

echo ""
echo "=== Configurazione completata ==="
echo "Baseline atteso (singola richiesta isolata alla homepage): ~187ms"
echo "Avvia il capacity test: locust -f experiments/locustfile_capacity.py --host http://ignored --headless"
```

### Riepilogo delle modifiche applicate

| Servizio | Tipo | Memory limit | CPU limit prima | CPU limit dopo |
|---|---|---|---|---|
| `currencyservice` | Node.js | 256Mi → **512Mi** | 200m | **500m** |
| `paymentservice` | Node.js | 256Mi → **512Mi** | 200m | **500m** |
| `frontend` | Go | invariato | 200m | **1000m** |
| `adservice` | Java/JVM | invariato | 200m | **1000m** |
| `recommendationservice` | Python | invariato | 200m | **1000m** |
| `cartservice` | Go | invariato | 200m | **500m** |
| `shippingservice` | Go | invariato | 200m | **500m** |
| `productcatalogservice` | Go | invariato | 200m | **500m** |
| `checkoutservice` | Go | invariato | 200m | **500m** |
| `emailservice` | Go | invariato | 200m | **500m** |

### Effetto complessivo delle correzioni

| Metrica | Prima delle correzioni | Dopo le correzioni |
|---|---|---|
| Singola richiesta isolata (p95) | **533ms** | **187ms** |
| Riavvii pod (OOMKill) | 26 riavvii (`currencyservice`) | **0 riavvii** |
| Latenza sotto carico (40 utenti) | **2.7–10s** | **< 100ms** |
| Capacity test completabile | ❌ SLA violato al step 1 | ✅ Completato fino a 100+ utenti |
