# DMOS — Analisi delle Metriche di Valutazione

> Documento tecnico che descrive le metriche raccolte, le loro sorgenti, le formule di calcolo
> e l'interpretazione dei risultati nell'ambito della valutazione sperimentale di DMOS.

---

## 1. Architettura del sistema di raccolta

Il sistema di valutazione di DMOS si compone di tre strati distinti che producono dati
complementari, ognuno con una granularità e una semantica propria.

```
┌─────────────────────────────────────────────────────────────────────┐
│  LOCUST (client-side, esterno ai cluster)                           │
│  locustfile_multiingress.py                                         │
│  → p95 latency per cluster, SLO violation rate, conteggio richieste │
│  → CSV per-cluster: results/multiingress/{scenario}_timeseries_*.csv│
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP (192.168.1.245/246/247:30080)
┌────────────────────────────▼────────────────────────────────────────┐
│  NGINX INGRESS (per-cluster, NodePort 30080)                        │
│  Proxy L4 tra Locust e i pod frontend                               │
│  → Consente a Hubble L7 di vedere traffico pod-to-pod               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  KUBERNETES + CILIUM/HUBBLE (server-side, per-cluster)              │
│  Prometheus per-cluster (NodePort 30090)                            │
│  hubble_http_requests_total, hubble_http_request_duration_seconds   │
│  dmos_current_replicas, dmos_predicted_traffic, dmos_cluster_score  │
└────────────────────────────┬────────────────────────────────────────┘
                             │ scrape ogni 15s
┌────────────────────────────▼────────────────────────────────────────┐
│  COLLECT_METRICS_SIMPLE.PY (aggregatore)                            │
│  → DMOS metrics endpoint: http://localhost:9090/metrics             │
│  → Locust web API: http://localhost:8089                            │
│  → Output: results/{HHMMSS}_{YYYYMMDD}_{scenario}.jsonl            │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  ANALYZE_TEST_COMPLETE.PY (analisi post-hoc)                        │
│  → Page 1: Scaling & Resource Allocation                            │
│  → Page 2: Quality of Service & Fairness                            │
│  → JSON con tutti i KPI aggregati                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 Separazione client-side vs server-side

Una distinzione fondamentale attraversa tutte le metriche di questo sistema:

| Punto di misura | Strumento | Semantica | Uso |
|----------------|-----------|-----------|-----|
| **Client-side** | Locust (`r.elapsed`) | Latenza totale osservata dall'utente: rete + coda + processing + risposta | Valutazione tesi — comparabile con Romano (k6) e Cilantro (wrk2) |
| **Server-side** | Hubble L7 (`hubble_http_request_duration_seconds_bucket`) | Latenza HTTP misurata al layer CNI, all'interno del cluster | Input allo scheduler DMOS (Φ_lat) — non direttamente comparabile con latenza client |

Questa separazione riflette una scelta architettonica precisa: le metriche di valutazione
(tesi) usano il punto di vista dell'utente finale; le metriche di scheduling (DMOS) usano il
punto di vista del cluster. Romano non distingue tra le due — usa k6 client-side per la
valutazione e Hubble server-side per lo scheduling, ma non esplicita questa dualità.

---

## 2. Sorgenti dati

### 2.1 DMOS Prometheus endpoint (`localhost:9090/metrics`)

DMOS espone un endpoint Prometheus standard, scrappato da `collect_metrics_simple.py`
ogni 15 secondi. Le metriche esposte seguono la naming convention Prometheus e includono
label `{cluster=..., service=...}` per la disambiguazione.

| Metrica Prometheus | Tipo | Label | Significato |
|-------------------|------|-------|-------------|
| `dmos_actual_traffic` | Gauge | `service` | Traffico misurato corrente (req/s), aggregato per servizio su tutti i cluster |
| `dmos_predicted_traffic` | Gauge | `cluster`, `service` | Traffico predetto dal `TrafficPredictor` per il prossimo ciclo di scheduling |
| `dmos_current_replicas` | Gauge | `cluster`, `service` | Repliche attualmente attive su quel cluster per quel servizio |
| `dmos_target_replicas` | Gauge | `cluster`, `service` | Repliche target calcolate dall'ultimo ciclo di scheduling |
| `dmos_cluster_score` | Gauge | `cluster`, `service` | Score multi-obiettivo Φ_i calcolato nell'ultimo ciclo |
| `dmos_scaling_events_total` | Counter | `cluster`, `service`, `action` | Numero cumulativo di scale-up/scale-down eseguiti |
| `dmos_scheduling_duration_seconds` | Histogram | `service` | Durata in secondi di ogni invocazione del ciclo di scheduling |

**Come viene usato `collect_metrics_simple.py`:** per ogni snapshot (ogni 15s), il collector
scrappa l'endpoint DMOS, deserializza il testo Prometheus con regex, e scrive una riga JSON
nel file `.jsonl`. Ogni riga rappresenta lo stato dell'intero sistema in un dato istante e
include traffico effettivo, repliche per cluster, traffico predetto, score, e — se disponibile —
le metriche Locust globali dell'istante corrente.

### 2.2 Locust Web API (`localhost:8089`)

L'interfaccia REST di Locust espone le statistiche aggregate di tutti gli utenti virtuali.
Il collector le recupera via `GET /stats/requests` ad ogni scrape e le salva nel `.jsonl`.

| Campo API Locust | Metrica estratta | Unità |
|-----------------|-----------------|-------|
| `response_times.50` | p50 (mediana) latency globale | ms |
| `response_times.95` | p95 latency globale | ms |
| `response_times.99` | p99 latency globale | ms |
| `current_rps` | Throughput attuale | req/s |
| `num_failures` / `num_requests` | Failure ratio globale | % |
| `user_count` | Utenti virtuali attivi | — |

⚠️ Questi valori sono **globali** — aggregano le richieste verso tutti e tre i cluster.
Non è possibile distinguere cluster1 da cluster2 da questo endpoint.

### 2.3 Locust CSV per-cluster (`results/multiingress/`)

`locustfile_multiingress.py` produce autonomamente due file CSV alla fine di ogni test:

- `{scenario}_cluster_latency_{timestamp}.csv` — riepilogo aggregato per cluster
- `{scenario}_timeseries_{timestamp}.csv` — serie temporale per finestre di 10 secondi

Il CSV timeseries ha una colonna per ogni cluster per ogni metrica:

```
timestamp, cluster1_count, cluster1_avg_ms, cluster1_p95_ms, cluster1_slo_pct,
           cluster2_count, cluster2_avg_ms, cluster2_p95_ms, cluster2_slo_pct,
           cluster3_count, cluster3_avg_ms, cluster3_p95_ms, cluster3_slo_pct
```

Questo CSV è la sorgente primaria per le metriche p95 per-cluster e SLO violation rate.
Viene scritto interamente dentro `locustfile_multiingress.py` tramite il listener
`@events.quit.add_listener` — nessun processo esterno è necessario.

**Come viene misurata la latenza in Locust:** ogni richiesta HTTP usa
`requests.Session.get()` / `.post()` eseguiti in un thread OS reale
(via `ThreadPoolExecutor`, non gevent). Il tempo è misurato come
`response.elapsed.total_seconds() * 1000` — il campo `elapsed` di `requests`
usa `datetime.utcnow()` prima e dopo il trasferimento, escludendo il tempo di
setup della connessione TCP ma includendo il tempo di trasmissione dei dati.
Questa misura è **immune al gevent polling delay** che su Windows azzerava
artificialmente le latenze nella versione precedente del locustfile.

---

## 3. Metriche di tesi — dettaglio completo

### 3.1 p95 Latency per Cluster

**Cosa misura:** il 95° percentile della distribuzione delle latenze di risposta HTTP,
calcolato separatamente per ogni cluster, su finestre temporali di 10 secondi.

**Sorgente:** `locustfile_multiingress.py` — tracking interno per-cluster.

**Perché p95 e non media o p50:**
La media è fortemente influenzata dagli outlier e non rappresenta l'esperienza dell'utente
tipico in condizioni di carico. Il p95 è la metrica standard nell'industria per SLA HTTP
(cf. Romano [Tabella 5-1]: "p95 response time"; Cilantro [OSDI'23]: "tail latency at p99").
Il p95 cattura le code di distribuzione — esattamente la situazione critica in cui il cluster
è sotto stress e alcune richieste iniziano a fare timeout o a rallentare significativamente.

**Come viene calcolato:**
All'interno di `track_request()`, ogni risposta HTTP viene aggiunta a una lista per finestra
temporale (`_windowed_stats[window][cluster_name]["response_times"]`). Alla fine di ogni
finestra di 10s, il p95 viene calcolato con:

```python
sorted_rts = sorted(response_times)
idx = max(0, int(len(sorted_rts) * 0.95) - 1)
p95 = sorted_rts[idx]
```

Questo è il **p95 empirico** (ordinamento e taglio al 95° quantile), distinto dal p95
stimato da istogrammi Prometheus (che usa interpolazione lineare tra i bucket).

**Valori attesi e interpretazione:**

| Valore p95 | Interpretazione |
|-----------|----------------|
| < 500 ms | Eccellente — carico basso o cluster ben dimensionato |
| 500–1000 ms | Accettabile — avvicinamento alla soglia SLO |
| ≈ 1000 ms | Limite SLO — DMOS dovrebbe avere già scalato |
| > 1000 ms | Violazione SLO — sotto-provisioning o spike non gestito |
| > 3000 ms | Sovraccarico grave — comparabile con Romano senza scheduler (p95=3.88s) |

**Cosa dimostra nella tesi:** con DMOS attivo, il p95 per cluster dovrebbe rimanere
sistematicamente sotto la soglia SLO di 1000 ms, anche durante i picchi di traffico.
Il confronto diretto è con Romano [§5.3]: p95 globale senza scheduler ≈ 3.88s,
con scheduler ≈ 1.23s. DMOS dovrebbe ottenere risultati comparabili o migliori,
grazie al pre-scaling proattivo che anticipa il picco prima che il p95 degrade.

---

### 3.2 SLO Violation Rate

**Cosa misura:** la percentuale di richieste HTTP con latenza superiore alla soglia SLO
(1000 ms), calcolata per finestre temporali di 10 secondi e per ogni cluster.

**Sorgente:** `locustfile_multiingress.py` — calcolata inline in `track_request()`.

**Soglia SLO = 1000 ms — motivazione:**
- Romano [Tabella 5-1]: "good" p95 ≈ 1.0–1.5s con scheduler attivo
- Cilantro [OSDI'23]: usa soglia 2s per latenza p99; DMOS usa 1s per p95 (più conservativa)
- Standard de facto per applicazioni web interattive (Google: <200ms percepito come istantaneo,
  <1000ms come accettabile, >1000ms come lento)

**Come viene calcolata:**

```python
# In track_request():
if response_time_ms > SLO_THRESHOLD_MS:        # SLO_THRESHOLD_MS = 1000
    _cluster_stats[cluster_name]["slo_violations"] += 1
    _windowed_stats[window][cluster_name]["slo_violations"] += 1

# Nel CSV timeseries, per ogni finestra:
slo_pct = round(slo_violations / total_requests * 100, 1)
```

Viene scritto nel CSV come `{cluster}_slo_pct` (percentuale, 0–100).

**Metriche derivate calcolate in `analyze_test_complete.py`:**
- **SLO violation mean per cluster:** media di `slo_pct` su tutte le finestre temporali
  del test per quel cluster.
- **Global SLO mean:** media su tutti i cluster e tutte le finestre, usata nel KPI summary.

**Valori attesi e interpretazione:**

| SLO violation rate | Interpretazione |
|-------------------|----------------|
| < 1% | Ottimo — quasi tutte le richieste sotto SLO |
| 1–5% | Accettabile — soglia di tolleranza tipica per SLA commerciali |
| 5–15% | Degradato — DMOS sotto-provisioning o reattivo |
| > 15% | Critico — sistema in overload |

**Cosa dimostra nella tesi:** l'SLO violation rate è la metrica più diretta per valutare
se DMOS garantisce l'esperienza utente. Un valore < 5% durante i picchi dimostra che
il pre-scaling proattivo ha mantenuto la latenza sotto controllo prima che le code
si formassero. L'SLO violation rate è analogamente usato in Cilantro [§5.1] come
"fraction of time in violation" — sebbene Cilantro usi p99 e soglia 2s.

---

### 3.3 Jain Fairness Index

**Cosa misura:** la distribuzione equa delle repliche tra i cluster. Un valore vicino a 1.0
indica che le repliche sono distribuite uniformemente; un valore vicino a 1/N (dove N è
il numero di cluster) indica che quasi tutte le repliche sono concentrate su un singolo cluster.

**Sorgente:** `dmos_current_replicas{cluster=..., service="frontend"}` — letto dal
`.jsonl` prodotto dal collector, calcolato su ogni snapshot.

**Formula (Jain's Fairness Index):**

$$J(\mathbf{x}) = \frac{\left(\sum_{i=1}^{N} x_i\right)^2}{N \cdot \sum_{i=1}^{N} x_i^2}$$

dove:
- $x_i$ = numero di repliche del servizio `frontend` sul cluster $i$
- $N$ = numero totale di cluster = **3** (fisso, indipendentemente da quanti cluster
  hanno repliche attive)
- $J \in [1/N, 1] = [0.333, 1.0]$

**Perché N è fisso a 3:**
Una scelta comune ma errata è calcolare N come il numero di cluster che hanno repliche
attive. Questo produce J=1.0 ogni volta che un solo cluster ha repliche (divisione
perfettamente "equa" su 1 cluster) — che è il caso peggiore, non il migliore.
Fissare N=3 garantisce che J=1.0 significhi distribuzione uniforme tra tutti i cluster
previsti dal sistema, mentre J=0.333 significhi concentrazione totale su un cluster.

**Come viene calcolato** (in `compute_jain_fairness()`):

```python
reps = [replicas_cluster1, replicas_cluster2, replicas_cluster3]
total = sum(reps)
if total > 0:
    sum_sq = sum(x**2 for x in reps)
    n = 3  # sempre 3, non len(reps con repliche > 0)
    jain = (total**2) / (n * sum_sq) if sum_sq > 0 else 0.0
```

**Esempi numerici:**

| Distribuzione | reps | J |
|-------------|------|---|
| Perfettamente equa | [3, 3, 3] | 1.000 |
| Leggermente sbilanciata | [4, 3, 2] | 0.972 |
| Moderatamente sbilanciata | [5, 2, 2] | 0.900 |
| Molto sbilanciata | [7, 1, 1] | 0.694 |
| Concentrata su 1 cluster | [9, 0, 0] | 0.333 |

**Valori attesi e interpretazione:**

| Jain Index | Interpretazione | Stato |
|-----------|----------------|-------|
| > 0.90 | Distribuzione sostanzialmente equa | ✅ Ottimo |
| 0.75–0.90 | Leggero sbilanciamento | ⚠️ Accettabile |
| 0.50–0.75 | Sbilanciamento moderato | ⚠️ Richiede attenzione |
| < 0.50 | Concentrazione su pochi cluster | ❌ Non accettabile |

**Cosa dimostra nella tesi:** il Jain Index misura la qualità della distribuzione
proporzionale di DMOS — l'obiettivo O4 del sistema. Romano non disponeva di questa
metrica. Cilantro [§4.3] usa "NJC Fairness" (Nash-Justain Criterion), una variante del
Jain Index per sistemi multi-tenant. Un Jain medio > 0.85 durante il test dimostra che
DMOS non concentra eccessivamente le repliche sui cluster con score più alto, garantendo
alta disponibilità multi-cluster.

**Interpretazione nel contesto DMOS:** un Jain basso non è necessariamente "sbagliato" —
se cluster3 (PL) ha carbon intensity molto alta, DMOS può legittimamente assegnargli
meno repliche. Il Jain Index deve essere letto insieme ai cluster scores (Φ_i) per
distinguere sbilanciamento intenzionale (guidato dallo score) da sbilanciamento
accidentale (bug di allocazione).

---

### 3.4 Provisioning Ratio (Resource Utilization)

**Cosa misura:** il rapporto tra la capacità di elaborazione effettivamente provisonata
(repliche × capacity_per_replica) e la domanda effettiva (traffico corrente).
Indica se il sistema è sovra-dimensionato, sotto-dimensionato o nel range ideale.

**Sorgente:** `dmos_actual_traffic{service="frontend"}` e
`dmos_current_replicas{cluster=..., service="frontend"}` — dal `.jsonl`.

**Formula:**

$$\text{ratio}_t = \frac{R_t \cdot C_{\text{pod}}}{\max(\lambda_t, \lambda_{\text{min}})}$$

dove:
- $R_t$ = numero totale di repliche attive all'istante $t$
- $C_{\text{pod}}$ = capacità per replica = **45 req/s** (misurata sperimentalmente
  nel capacity test del 27/02/2026: knee point a gradual ramp ≈ 43.7 req/s/pod)
- $\lambda_t$ = traffico corrente in req/s
- $\lambda_{\text{min}} = R_{\text{min}} \cdot C_{\text{pod}}$ = capacità minima HA
  (floor di alta disponibilità: min_replicas_per_cluster × N_clusters × capacity)

**Perché il denominatore usa max(λ, λ_min):**
Durante le fasi a basso traffico (warm-up, cooldown), il sistema mantiene comunque
le repliche minime per alta disponibilità (vincolo architetturale, non spreco).
Senza questa correzione, anche 1 req/s con 3 repliche attive darebbe ratio=135 —
un "over-provisioning" che in realtà è solo il floor HA. Il `max()` elimina questo
artefatto: se il traffico è inferiore alla capacità minima garantita, il denominatore
usa la capacità minima (ratio ≈ 1.0 = sistema al livello minimo corretto).

**Metriche derivate:**

```
over_provisioned   (ratio > 1.5x)  →  capacità > 150% della domanda
ideal range        (1.0x ≤ ratio ≤ 1.5x)  →  safety margin tra 0% e 50%
under_provisioned  (ratio < 1.0x)  →  capacità insufficiente per la domanda
```

**Soglia "ideale" a 1.15x:**
Un ratio di 1.15x corrisponde al 15% di safety margin — esattamente il parametro
`safety_margin=0.15` configurato in `ReplicaScaler`. Il sistema è progettato per
mantenere sempre il 15% di capacità libera rispetto alla domanda stimata.

**Le statistiche (mean, median, over_pct, under_pct, ideal_pct) escludono il warm-up:**
Gli snapshot con `traffic < max(10%, HIGH_THRESHOLD)` vengono esclusi dal calcolo
delle statistiche aggregate ma mantenuti nella serie temporale per il plot.
Questo evita che la fase iniziale (bassa attività) distorca le statistiche.

**Valori attesi e interpretazione:**

| Ratio medio | Interpretazione |
|------------|----------------|
| 1.1–1.3x | Ottimo — sistema ben calibrato con safety margin |
| 1.3–1.8x | Over-provisioning moderato — conservativo ma accettabile |
| > 2.0x | Over-provisioning eccessivo — predictor troppo aggressivo o EMA lenta |
| < 1.0x | Under-provisioning — sistema in deficit di capacità → SLO a rischio |

**Cosa dimostra nella tesi:** il provisioning ratio è la metrica quantitativa
dell'efficienza delle risorse. Cilantro [§5.2] usa "useful resource usage" (frazione
del tempo in cui le risorse allocate sono effettivamente usate) — concettualmente
equivalente al reciproco del provisioning ratio. Un sistema con ratio medio ≈ 1.15–1.3x
durante il picco dimostra che DMOS scala di quanto necessario (non troppo, non troppo poco).

---

### 3.5 Replica Distribution % per Cluster

**Cosa misura:** la quota percentuale di repliche del `frontend` assegnate a ciascun cluster
rispetto al totale, nel tempo. Visualizza come DMOS ridistribuisce dinamicamente il carico.

**Sorgente:** `dmos_current_replicas{cluster=..., service="frontend"}` — dal `.jsonl`.

**Come viene calcolata:**

```python
totale = replicas_cluster1 + replicas_cluster2 + replicas_cluster3
pct_cluster_i = replicas_cluster_i / totale * 100  # se totale > 0
```

Questa percentuale è normalizzata a 100% (area plot stacked) — mostra la composizione
relativa, non il numero assoluto.

**Cosa dimostra nella tesi:**
- Nelle fasi di carico equilibrato (weights Locust = [40, 35, 25]), la distribuzione
  dovrebbe riflettere i pesi del traffico — cluster1 ≈ 40%, cluster2 ≈ 35%, cluster3 ≈ 25%.
- Nelle fasi a basso traffico (tutti al minimo: 1 replica × 3 cluster = 33%/33%/33%),
  la distribuzione converge all'equità.
- L'evoluzione temporale mostra come DMOS risponde a variazioni di traffico
  e di score per cluster (carbon intensity, latenza, capacità hardware).

**Relazione con Jain Index:** la distribuzione % è la rappresentazione visiva di ciò
che il Jain Index quantifica numericamente. Una curva % stabile e prossima ai valori
attesi per ogni cluster (quota × 100%) corrisponde a un Jain Index elevato.

---

### 3.6 Proactive % (Time to Scale)

**Cosa misura:** la percentuale di eventi di scale-up guidati dalla *predizione* del
traffico futuro, piuttosto che dal traffico corrente già in eccesso.

**Sorgente:** `dmos_current_replicas` e `dmos_predicted_traffic` — dal `.jsonl`.
Il calcolo è interamente post-hoc in `compute_time_to_scale()`.

**Metodo di classificazione (demand-driven, primario):**

Un evento di scale-up (incremento di `total_replicas` tra uno snapshot e il successivo)
è classificato come **proattivo** se, al momento dell'evento:

```
predicted_traffic > current_traffic × 1.03
```

ovvero: il predictor vede una crescita del traffico di almeno il 3% rispetto al
corrente — il che significa che la componente di tendenza (`dΛ/dt`) è positiva e
ha spinto il sistema a scalare *prima* che la domanda aumentasse.

È **reattivo** se la predizione non anticipa crescita (predicted ≈ current o predicted < current)
e il traffico corrente da solo ha già giustificato lo scale-up.

**Perché la soglia è 3% e non 0%:**
Il predictor DMOS usa un EMA (Exponential Moving Average) con componente di tendenza.
Per costruzione, in assenza di variazioni, `predicted ≈ current` — ma con piccole
oscillazioni numeriche. La soglia del 3% elimina i falsi positivi: solo quando la
componente trend è significativamente positiva si classifica il comportamento come
proattivo.

**Time to Scale (TtS) per eventi proattivi:**
Viene stimato quanto in anticipo rispetto al picco effettivo è avvenuto lo scale-up:

```python
# Cerca il momento in cui il traffico raggiunge il livello predetto
for j in range(i+1, i+20):
    if traffic[j] >= predicted[i] * 0.90:   # entro 10% della previsione
        tts = -(timestamps[j] - timestamps[i]).total_seconds()
        # negativo = scale-up avvenuto prima che il traffico raggiungesse quel livello
        break
else:
    tts = -60.0   # il traffico non ha mai raggiunto la previsione → molto proattivo
```

Il TtS negativo indica *quanti secondi prima del picco* è avvenuto lo scale-up.
Un TtS = -90s significa che DMOS ha scalato 90 secondi prima che la domanda
raggiungesse il livello predetto.

**Valori attesi e interpretazione:**

| Proactive % | Interpretazione | Obiettivo tesi |
|------------|----------------|---------------|
| > 70% | Sistema prevalentemente proattivo | ✅ Eccellente |
| 50–70% | Maggioranza proattiva | ✅ Sufficiente |
| 30–50% | Mix proattivo/reattivo | ⚠️ Accettabile |
| < 30% | Prevalentemente reattivo | ❌ Predictor insufficiente |

**Cosa dimostra nella tesi:** questa è la metrica chiave per dimostrare il contributo
principale di DMOS rispetto al baseline di Romano. Romano [§4.2] è esplicitamente reattivo
(Proactive % = 0% per costruzione). DMOS deve dimostrare Proactive % > 50% per
giustificare l'introduzione del `TrafficPredictor` come miglioramento architetturale.

**Nota metodologica:** il Proactive % viene calcolato solo sugli eventi di scale-up
(`delta > 0`), non sugli scale-down. Gli scale-down sono per design conservativi
(cooldown 60s, max_delta limitato) e non sono la dimensione rilevante per dimostrare
il pre-scaling.

---

## 4. Metriche di supporto

### 4.1 Prediction Accuracy (MAPE, RMSE, R², Directional)

**Scopo:** valutare la qualità del `TrafficPredictor` — prerequisito per giustificare
la fiducia del sistema nella predizione per guidare lo scaling proattivo.

**Sorgente:** coppia `(dmos_actual_traffic, dmos_predicted_traffic)` per ogni snapshot.

#### 4.1.1 MAPE — Mean Absolute Percentage Error

$$\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{\hat{\lambda}_i - \lambda_i}{\lambda_i} \right| \times 100$$

dove $\lambda_i$ = traffico reale, $\hat{\lambda}_i$ = traffico predetto.

**Varianti calcolate:**
- **MAPE active** (primaria): calcolata solo sugli snapshot dove
  `traffic ≥ max(5, peak × 10%)` — esclude il tail post-test in cui l'EMA
  decade lentamente mentre il traffico reale è già a 0. Il tail inflaziona
  artificialmente la MAPE (es. predicted=80, actual=2 → errore=3900%).
- **MAPE overall** (riferimento): calcolata su tutti gli snapshot con traffico > 1 req/s.

**Interpretazione:**

| MAPE active | Qualità predizione |
|------------|-------------------|
| < 15% | Eccellente |
| 15–25% | Buona |
| 25–40% | Accettabile |
| > 40% | Scadente — tuning necessario |

#### 4.1.2 RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{\lambda}_i - \lambda_i)^2}$$

Espresso in req/s. A differenza della MAPE (che è relativa), l'RMSE è assoluto:
un RMSE=10 req/s su un picco di 150 req/s è eccellente, ma su un picco di 20 req/s
è critico. Va sempre interpretato in relazione al range di traffico del test.

#### 4.1.3 R² — Coefficiente di Determinazione

$$R^2 = 1 - \frac{\sum_i (\lambda_i - \hat{\lambda}_i)^2}{\sum_i (\lambda_i - \bar{\lambda})^2}$$

Misura la frazione della varianza del traffico spiegata dalla predizione.
R² vicino a 1.0 indica che il predictor cattura fedelmente la forma della curva.
R² negativo indica che il predictor è peggiore di una previsione con la media costante.

#### 4.1.4 Directional Accuracy

La percentuale di snapshot consecutivi in cui la direzione del cambiamento
(crescita vs calo) è predetta correttamente:

```python
# Corretto se entrambi aumentano o entrambi diminuiscono
correct = (actual[i] - actual[i-1]) e (predicted[i] - predicted[i-1]) hanno stesso segno
```

Per lo scaling proattivo, la Directional Accuracy è più importante della MAPE assoluta:
importa che il predictor sappia *quando il traffico cresce*, non che predica esattamente
quanti req/s.

---

### 4.2 Scaling Oscillation (Flapping)

**Cosa misura:** l'instabilità del ciclo di scaling — la tendenza del sistema a scalare
su e poi immediatamente giù in cicli rapidi.

**Sorgente:** serie temporale di `total_replicas` dal `.jsonl`.

**Come viene calcolata:**
Per ogni coppia di snapshot consecutivi si calcola la direzione del cambiamento
(-1 = scale-down, 0 = invariato, +1 = scale-up). Un "direction change" è una reversione:
da +1 a -1 o da -1 a +1.

Il flapping viene rilevato tramite sliding window di 20 campioni (≈ 5 minuti con scrape
ogni 15s): se in quella finestra ci sono ≥ 3 direction changes, la finestra è classificata
come "flapping window".

**Perché questo è importante per la tesi:** il baseline di Romano non ha meccanismi
anti-flapping (nessun cooldown, nessuna dead zone). DMOS implementa scale-down cooldown
(60s) e dead zone (±15% di variazione del traffico non genera scaling). Zero flapping
windows nel test dimostra l'efficacia di questi meccanismi.

---

## 5. Pipeline di calcolo completa

```
JSONL (ogni 15s)
    │
    ├─► extract_service_ts("frontend")
    │       → timestamps, traffic[], predicted[], total_replicas[]
    │         per_cluster_replicas{cluster: []}
    │         per_cluster_scores{cluster: []}
    │
    ├─► extract_locust_ts()
    │       → timestamps_locust[], p50_rt[], p95_rt[], p99_rt[], rps[], fail_ratio[]
    │
    ├─► compute_prediction_accuracy(traffic, predicted)
    │       → MAPE (active, overall), RMSE, R², directional_accuracy
    │
    ├─► compute_provisioning_ratio(traffic, total_replicas, capacity=45, min_floor=3)
    │       → ratios[], mean, median, over_pct, under_pct, ideal_pct
    │
    ├─► compute_time_to_scale(timestamps, traffic, total_replicas, predicted, capacity=45)
    │       → events[], proactive_count, reactive_count, proactive_pct, avg_tts
    │
    ├─► compute_scaling_oscillation(total_replicas, timestamps, window=20)
    │       → direction_changes, flapping_windows, window_changes_timeline[]
    │
    ├─► compute_jain_fairness(data, service="frontend")
    │       → jain_timestamps[], jain_values[]  (J per ogni snapshot)
    │
    └─► load_locust_cluster_csv(scenario)   ← da results/multiingress/
            → per cluster: timestamps[], p95_ms[], slo_pct[], count[]
```

---

## 6. Output grafici

### Page 1 — Scaling & Resource Allocation (`{prefix}_page1_scaling.png`)

| Plot | Dati | Obiettivo visivo |
|------|------|-----------------|
| [0,0] Traffic & Prediction | `traffic[]`, `predicted[]` | Verificare che il predictor anticipi i picchi |
| [0,1] Replica Distribution % | `per_cluster_replicas` normalizzato | Vedere la redistribuzione dinamica tra cluster |
| [1,0] Provisioning Ratio | `ratios[]` con fill over/under | Identificare periodi di over/under-provisioning |
| [1,1] Time to Scale | eventi proattivi (verde) / reattivi (rosso) | Visualizzare il comportamento proattivo vs reattivo |

### Page 2 — Quality of Service & Fairness (`{prefix}_page2_qos.png`)

| Plot | Dati | Obiettivo visivo |
|------|------|-----------------|
| [0,0] p95 per Cluster | `locust_cluster[cn]["p95_ms"]` | Confrontare latenza tra cluster nel tempo |
| [0,1] SLO Violation Rate | `locust_cluster[cn]["slo_pct"]` | Vedere quando e quanto DMOS fallisce il SLO |
| [1,0] Jain Index | `jain_timestamps[]`, `jain_values[]` | Valutare l'equità della distribuzione nel tempo |
| [1,1] KPI Summary | aggregati di tutte le metriche | Tabella riassuntiva con stato (✅/⚠️/❌) |

---

## 7. Tabella di confronto con letteratura

| Metrica | DMOS | Romano [2025] | Cilantro [OSDI'23] |
|---------|------|--------------|-------------------|
| **Latenza end-user** | p95 per-cluster (Locust) | p95 globale (k6) | p99 (wrk2) |
| **Soglia SLO** | 1000 ms | Non definita esplicitamente | 2000 ms |
| **SLO violation rate** | % req > 1000ms per cluster | Non calcolata | Fraction of time in violation |
| **Fairness** | Jain Index (N=3 cluster) | Non misurata | NJC (Nash-Justain Criterion) |
| **Resource efficiency** | Provisioning ratio (capacity/demand) | Non misurata | Useful resource usage |
| **Proactive scaling** | Proactive % (demand-driven) | 0% per costruzione | Non applicabile (diverso paradigma) |
| **Durata esperimento** | 20–60 min per scenario | 10 min per scenario | 6 ore |
| **Numero scenari** | 4 (flash_crowd, gradual_ramp, double_wave, sinusoidal) | 2 (uniforme, sbilanciato) | Multipli con trace reali |

---

## 8. Note metodologiche e limitazioni

### 8.1 Granularità temporale

Il collector scrappa ogni 15s — sufficientemente fino per catturare eventi di scaling
(che richiedono almeno 30-60s per completarsi in K3s) ma non abbastanza fine per
catturare burst di latenza brevi (< 15s). Il CSV Locust usa finestre di 10s — più fine,
ma limitato alle metriche client-side.

### 8.2 Effetto EMA post-test

Il `TrafficPredictor` usa un EMA che decade lentamente. Quando Locust smette di generare
traffico, il traffico reale crolla a 0 ma il predetto rimane elevato per 10-15 minuti.
Questo produce ratio di over-provisioning artificialmente alti nella fase post-test e
MAPE overall inflazionata. Per questo motivo la MAPE active (solo fase a traffico attivo)
è la metrica primaria, e le statistiche di provisioning escludono il warm-up con traffico < 10% del picco.

### 8.3 Moltiplicatore sistematico Hubble vs Locust

Hubble conta ≈ 1.43× più richieste di Locust per lo stesso traffico.
Motivo: Locust conta le transazioni utente (1 GET per task di browsing), mentre Hubble
con `destination_workload="frontend"` conta ogni singola richiesta HTTP, incluse le
sotto-richieste generate dal caricamento di ogni pagina (CSS, immagini, API calls interne).
Questo moltiplicatore è sistematico e stabile — non inficia le metriche relative
(MAPE, ratio, Jain) ma implica che il traffico mostrato nel `.jsonl` (da DMOS/Hubble)
sia più alto di quanto mostrato da Locust per lo stesso carico effettivo.

### 8.4 p95 Locust vs p95 Hubble

Sono due metriche diverse con sorgenti e semantiche distinte:

| | p95 Locust (client-side) | p95 Hubble (server-side) |
|---|---|---|
| **Misura** | `response.elapsed` dall'invio alla ricezione completa | `histogram_quantile(0.95, hubble_http_request_duration...)` |
| **Include** | Rete client→cluster + Nginx + coda + processing + risposta | Solo rete Nginx→pod + processing + risposta |
| **Uso** | Valutazione tesi, comparazione con Romano | Input allo scheduler Φ_lat |
| **Disponibile** | Solo quando Locust è attivo | Solo quando c'è traffico HTTP nel cluster |

Per la valutazione nella tesi si usa **sempre il p95 Locust** — è il valore comparabile
con i risultati di Romano e rappresenta l'esperienza effettiva dell'utente finale.

---

*Documento aggiornato: 2026-03-21. Riferimenti: G. Romano (2025), A. Bhardwaj et al.
"Cilantro" (OSDI'23), codice sorgente DMOS (src/), locustfile_multiingress.py v3,
analyze_test_complete.py v2.*
