# [ARCHIVED] Backend Capacity Test — Documentazione Completa

> **Documento storico**: metodo `ghz` + `kubectl port-forward`, superato dal
> Job k6 in-cluster. Riferimenti attuali:
> [`../capacity-testing.md`](../capacity-testing.md) e
> [`../../experiments/CAPACITY-GRPC.md`](../../experiments/CAPACITY-GRPC.md).
## Come misurare la capacita dei servizi gRPC per calibrare DMOS

> File principali:
>   - `experiments/ghz_backend_capacity.py` — Test isolato per-servizio
>   - `experiments/backend_capacity_observer.py` — Osservazione system-level
>   - `experiments/k6_capacity_test.js` — Capacity test frontend (esistente)
>   - `experiments/locustfile_multiingress.py` — Capacity test frontend con ingress 40/30/30
>
> Ultima revisione: 16/04/2026

> **⚠️ Stato di validazione sperimentale.** Solo la capacità del **frontend** è stata misurata empiricamente (k6 stepped load + knee detection + 80% safety margin), con valori per-cluster `39/35/30 req/s` (DE/FR/PL). Le capacità dei 4 backend (`cartservice=50`, `productcatalogservice=80`, `checkoutservice=20`, `recommendationservice=40`) configurate in `config/services.yaml` sono **stime conservative** non ancora verificate con la metodologia descritta in questo documento (ghz + observer). Il documento descrive la procedura corretta per validarle; il presente lavoro di tesi le considera limitazione aperta. Vedi §1 ("Stato attuale nel progetto") per il dettaglio e §14 per i risultati frontend.

---

## Indice

1. [Perche serve un capacity test sui backend](#1-perche-serve-un-capacity-test-sui-backend)
2. [Architettura dei servizi e call graph](#2-architettura-dei-servizi-e-call-graph)
3. [I 3 livelli di misurazione della capacity](#3-i-3-livelli-di-misurazione-della-capacity)
4. [Cos'e ghz e come funziona](#4-cose-ghz-e-come-funziona)
5. [Dipendenze: cosa succede realmente quando ghz chiama un backend](#5-dipendenze-cosa-succede-realmente-quando-ghz-chiama-un-backend)
6. [Phase 1 — Test isolato con ghz](#6-phase-1--test-isolato-con-ghz)
7. [Phase 2 — Osservazione system-level con backend_capacity_observer](#7-phase-2--osservazione-system-level-con-backend_capacity_observer)
8. [Il knee point e il valore raccomandato](#8-il-knee-point-e-il-valore-raccomandato)
9. [Come i risultati si collegano a DMOS](#9-come-i-risultati-si-collegano-a-dmos)
10. [Guida operativa passo-passo](#10-guida-operativa-passo-passo)
11. [Interpretazione dei risultati e casi limite](#11-interpretazione-dei-risultati-e-casi-limite)
12. [Confronto tra i 3 strumenti di capacity testing](#12-confronto-tra-i-3-strumenti-di-capacity-testing)
13. [Validita scientifica e riferimenti](#13-validita-scientifica-e-riferimenti)
14. [Risultati backend](#14-risultati-backend)

---

## 1. Perche serve un capacity test sui backend

DMOS scala **tutti e 5** i servizi dell'applicazione, non solo il frontend. Per ogni servizio la formula di scaling e':

```
needed_replicas = predicted_traffic / capacity_per_replica
```

Il valore `capacity_per_replica` (configurato in `config/services.yaml` come `capacity_req_per_sec`) e' il parametro piu critico dell'intero sistema di autoscaling. Se e' sbagliato:

| capacity troppo **alta** | capacity troppo **bassa** |
|---|---|
| DMOS pensa che 1 replica basti per piu traffico | DMOS pensa che 1 replica gestisca poco |
| Scala troppo **tardi** | Scala troppo **presto** |
| SLA violato sotto carico | Over-provisioning, spreco di risorse |
| Il servizio satura prima che DMOS reagisca | Repliche inutili su nodi gia carichi |

### Stato attuale nel progetto

| Servizio | `capacity_req_per_sec` | Come e' stato determinato |
|---|---|---|
| **frontend** | 39 (per-cluster: 39/35/30) | Misurato con k6_capacity_test.js (knee=49, 80%=39) |
| **cartservice** | 50 | Stimato: "1 replica gRPC gestisce facilmente 50+ rps" |
| **productcatalogservice** | 80 | Stimato: "1 replica Go read-only gestisce 100+ rps" |
| **checkoutservice** | 20 | Stimato: "1 replica gestisce 20+ rps, lento ma poco traffico" |
| **recommendationservice** | 40 | Stimato: "Python piu lento di Go, 1 replica 30+ rps" |

**Il problema**: 4 valori su 5 sono stime a occhio, mai verificate sperimentalmente. L'unico valore misurato e' quello del frontend, perche' era l'unico testabile direttamente con tool HTTP (k6, Locust). I backend comunicano via gRPC e richiedono strumenti diversi.

---

## 2. Architettura dei servizi e call graph

Online Boutique e' un'applicazione a microservizi dove il frontend HTTP funge da gateway e chiama i backend via gRPC. Il call graph completo e':

```
                     +---> productcatalogservice  (Go, in-memory, 0 dipendenze)
                     |         port: 3550
                     |
 k6/Locust           |
   HTTP     ------>  frontend  +---> cartservice ---> redis-cart
  :30080             (Go)      |       port: 7070      (Redis, non scalato)
                     |
                     |     +---> recommendationservice ---> productcatalogservice
                     |     |       port: 8080 (Python)
                     |     |
                     +-----+---> checkoutservice ------+---> cartservice
                                   port: 5050 (Go)     |---> productcatalogservice
                                                        |---> currencyservice  (non scalato)
                                                        |---> shippingservice  (non scalato)
                                                        |---> paymentservice   (non scalato)
                                                        +---> emailservice     (non scalato)
```

### Servizi scalati da DMOS (5)

| Servizio | Linguaggio | Porta gRPC | Ruolo |
|---|---|---|---|
| frontend | Go | — (HTTP :8080) | Gateway, riceve traffico HTTP da k6/Locust |
| cartservice | Go (C#) | 7070 | Lettura/scrittura carrello su Redis |
| productcatalogservice | Go | 3550 | Catalogo prodotti in-memory |
| checkoutservice | Go | 5050 | Orchestratore checkout (fan-out a 6 servizi) |
| recommendationservice | Python | 8080 | Raccomandazioni basate su catalogo |

### Servizi NON scalati da DMOS (6)

redis-cart, currencyservice, shippingservice, paymentservice, emailservice, adservice — hanno repliche fisse, non partecipano all'autoscaling.

### Traffico relativo (dal mix di endpoint del frontend)

A 150 rps frontend, il traffico atteso sui backend e':

```
Endpoint mix del frontend:
  GET  /                   40%  --> productcatalog + recommendation
  GET  /product/OLJCESPC7Z 15%  --> productcatalog
  GET  /product/66VCHSJNUP 10%  --> productcatalog
  GET  /cart               10%  --> cartservice
  POST /cart               15%  --> cartservice
  POST /setCurrency         5%  --> currencyservice
  POST /cart/checkout       5%  --> checkoutservice (fan-out)

Traffico derivato a 150 rps frontend:
  productcatalogservice:  ~97 rps  (65% delle richieste lo coinvolgono)
  cartservice:            ~37 rps  (25% delle richieste)
  recommendationservice:  ~48 rps  (32% — home + product pages)
  checkoutservice:        ~7.5 rps (5% delle richieste)
```

---

## 3. I 3 livelli di misurazione della capacity

Ci sono 3 modi complementari per misurare la capacity, ciascuno con un livello di dettaglio diverso:

### Livello 1: k6 / Locust (frontend-only, client-side)

```
k6 ---> HTTP ---> frontend ---> [backend chain nascosta]
         ^
         |
     Misura qui: latenza HTTP, rps, error rate
```

**Cosa vede**: la latenza e il throughput dal punto di vista del client HTTP.

**Cosa NON vede**: quale servizio nella catena sta causando il rallentamento. Se cartservice diventa lento, k6 vede solo che il frontend risponde piu lentamente.

**Strumenti**: `k6_capacity_test.js` (corrente). Le varianti Locust storiche (`locustfile_capacity.py`, `capacity_test_async.py`) sono state archiviate in `experiments/_archive/`.

**Produce**: `capacity_per_cluster` per il frontend (39/35/30 rps — gia misurato).

### Livello 2: k6 + Backend Observer (system-level, server-side)

```
k6 ---> HTTP ---> frontend ---> cartservice ---> redis
                      |                           ^
                      |                           |
                      v                     Prometheus vede
                  productcatalog            il traffico Hubble
                      ^                     su OGNI servizio
                      |
                  recommendation
```

**Cosa vede**: il throughput/latenza/CPU di **ogni singolo servizio** tramite query Prometheus su metriche Hubble (`hubble_http_requests_total`, `hubble_http_request_duration_seconds_bucket`). Identifica il collo di bottiglia.

**Cosa NON fa**: non puo isolare la capacity di un singolo servizio. Il traffico arriva dal frontend, quindi ogni servizio riceve il traffico determinato dal call graph, non un rate controllato.

**Strumenti**: `backend_capacity_observer.py` (da lanciare in parallelo a k6/Locust)

**Produce**: knee point system-level per servizio, identificazione del bottleneck.

### Livello 3: ghz (per-service isolation, gRPC diretto)

```
ghz ---> gRPC ---> cartservice ---> redis-cart
                       ^
                       |
                   Misura qui: latenza gRPC, rps, error rate
                   (il frontend e' bypassato)
```

**Cosa fa**: manda richieste gRPC **direttamente** al singolo servizio, bypassando il frontend. Equivalente di k6 ma per gRPC.

**Cosa vede**: la capacity grezza del servizio singolo sotto un rate controllato. Il rate e' esattamente quello che imposti (10 rps, 20 rps, 30 rps...), non quello che arriva dal frontend.

**Strumenti**: `ghz_backend_capacity.py`

**Produce**: `capacity_req_per_sec` per ogni backend (i valori che oggi sono stimati).

### Tabella riassuntiva

| Aspetto | k6 / Locust | k6 + Observer | ghz |
|---|---|---|---|
| **Protocollo** | HTTP | HTTP + Prometheus | gRPC |
| **Target** | Frontend | Frontend (+ osserva backend) | Singolo backend |
| **Rate controllato?** | Si (sul frontend) | Si (sul frontend) | Si (sul backend) |
| **Vede i backend?** | No (black box) | Si (via Hubble/Prometheus) | Si (diretto) |
| **Isola un servizio?** | No | No | Si |
| **Serve per** | `capacity_per_cluster` frontend | Bottleneck analysis | `capacity_req_per_sec` backend |
| **Dipendenze** | Non rilevante | Non rilevante | Colpite realmente |

---

## 4. Cos'e ghz e come funziona

### Descrizione

[ghz](https://ghz.sh/) e' un tool open source di benchmarking per servizi gRPC, scritto in Go. E' l'equivalente di k6/Locust/wrk per il mondo gRPC:

- Carica una definizione `.proto` del servizio
- Chiama un metodo RPC specifico con un payload JSON
- Genera traffico a un rate controllato (rps)
- Misura latenza (p50, p90, p95, p99), throughput, errori
- Output in JSON per parsing automatizzato

### Installazione

**Windows** (binario pre-compilato):
```
1. Vai su https://github.com/bojand/ghz/releases
2. Scarica ghz-windows-x86_64.zip
3. Estrai ghz.exe in una cartella nel PATH
4. Verifica: ghz --version
```

**Linux/Mac** (via Go):
```bash
go install github.com/bojand/ghz/cmd/ghz@latest
```

### Proto file

ghz ha bisogno della definizione `.proto` del servizio per sapere come serializzare le richieste. Per Online Boutique il file e' `demo.proto`, disponibile nel repository Google:

```
https://github.com/GoogleCloudPlatform/microservices-demo/blob/main/protos/demo.proto
```

Lo script `ghz_backend_capacity.py` lo scarica automaticamente in `experiments/protos/demo.proto`.

### Esempio di invocazione manuale

```bash
# 1. Port-forward del servizio
kubectl port-forward svc/cartservice 7070:7070 -n online-boutique --context cluster1

# 2. Singola invocazione (verifica che funzioni)
ghz --proto experiments/protos/demo.proto \
    --call hipstershop.CartService/GetCart \
    --data '{"user_id":"test"}' \
    --total 1 \
    localhost:7070

# 3. Stepped load (10 rps per 60 secondi)
ghz --proto experiments/protos/demo.proto \
    --call hipstershop.CartService/GetCart \
    --data '{"user_id":"test"}' \
    --rps 10 \
    --duration 60s \
    --format json \
    localhost:7070
```

### Come arriva ghz al servizio

ghz gira sulla macchina locale (Windows). Per raggiungere i pod gRPC dentro il cluster si usa `kubectl port-forward`:

```
ghz (localhost:7070) ---> port-forward ---> svc/cartservice:7070 (dentro il cluster)
```

Lo script `ghz_backend_capacity.py` gestisce automaticamente il port-forward: lo apre prima del test e lo chiude dopo.

---

## 5. Dipendenze: cosa succede realmente quando ghz chiama un backend

Questo e' il punto piu importante da capire.

**ghz NON isola il servizio dalle sue dipendenze downstream.** Quando ghz chiama `checkoutservice/PlaceOrder`, il checkoutservice internamente chiama cartservice, productcatalogservice, currencyservice, shippingservice, paymentservice ed emailservice. Queste chiamate avvengono normalmente, come se la richiesta fosse arrivata dal frontend.

L'unico servizio che ghz bypassa e' il **frontend** — ghz lo sostituisce come generatore di richieste.

### Mappa delle dipendenze per servizio

| Servizio | ghz chiama | Il servizio chiama | Dipendenze totali | Tipo di misura |
|---|---|---|---|---|
| **productcatalogservice** | `GetProduct` | Nessuno (catalogo in-memory) | 0 | **Pura**: misuri solo il servizio Go |
| **cartservice** | `GetCart` | redis-cart | 1 | **Quasi pura**: +1ms Redis locale |
| **recommendationservice** | `ListRecommendations` | productcatalogservice | 1 | **Con 1 hop gRPC**: +latenza catalog |
| **checkoutservice** | `PlaceOrder` | cart, catalog, currency, shipping, payment, email | 6 | **Fan-out completo**: il path piu pesante |

### Perche questo e' il comportamento corretto

Il valore `capacity_req_per_sec` in services.yaml rappresenta quante richieste/secondo **una replica puo gestire nel sistema reale**. Nel sistema reale, le dipendenze ci sono e il tempo della risposta le include.

Se misurassi il checkoutservice con mock delle dipendenze, otterresti un numero piu alto ma **inutilizzabile** — perche' nel deployment reale quelle dipendenze aggiungono latenza e il servizio non raggiungera mai quel throughput.

### L'unica attenzione: dipendenze non saturate

Quando testi un servizio, le dipendenze devono essere **attive e non saturate**. Se testi checkoutservice a 50 rps e currencyservice (1 replica, non scalato) puo gestire solo 30 rps, il bottleneck e' currencyservice, non il checkoutservice.

Regola pratica:
- Il servizio target: **1 replica** (quello che misuri)
- Le dipendenze scalate da DMOS (productcatalog, cartservice): **1 replica** (fissata)
- Le dipendenze non scalate da DMOS (redis, currency, shipping, payment, email): **gia 1 replica fissa** — tipicamente sufficienti perche' il rate diretto e' basso

### Ordine di test consigliato

L'ordine segue le dipendenze — prima i servizi senza dipendenze, poi quelli che dipendono da servizi gia verificati:

```
1. productcatalogservice  → 0 dipendenze (misura piu pulita)
2. cartservice            → 1 dipendenza (redis-cart, non gRPC)
3. recommendationservice  → 1 dipendenza (productcatalog, gia testato al punto 1)
4. checkoutservice        → 6 dipendenze (testare per ultimo, quando tutto e' verificato)
```

---

## 6. Phase 1 — Test isolato con ghz

### Scopo

Misurare la capacity grezza di ciascun backend con la stessa metodologia del capacity test frontend: **stepped load + knee detection + 80% safety margin**.

### Come funziona

```
Rate (rps)
 100 |                                          ____
  90 |                                     ____/
  80 |                                ____/
  70 |                           ____/
  60 |                      ____/
  50 |                 ____/  ← SLA violato → STOP
  40 |            ____/  ← knee point (ultimo step OK)
  30 |       ____/
  20 |  ____/
  10 | /
     +--+--+--+--+--+--+--+--+--+-- Tempo
        1  2  3  4  5  6  7  8  9   Step (60s ciascuno)
```

Per ogni step:
1. ghz manda richieste gRPC a un rate costante per `step_duration` secondi
2. Raccoglie latenza p50/p95/p99, throughput effettivo, errori
3. Valuta il SLA: `p95 < sla_p95_ms AND error_rate < 2%`
4. Se SLA rispettato: aggiorna il knee point
5. Se SLA violato per 2 step consecutivi: saturazione confermata, test terminato

### Configurazione RPC per servizio

| Servizio | RPC | Payload | SLA p95 |
|---|---|---|---|
| cartservice | `hipstershop.CartService/GetCart` | `{"user_id":"capacity-test-user"}` | 50ms |
| productcatalogservice | `hipstershop.ProductCatalogService/GetProduct` | `{"id":"OLJCESPC7Z"}` | 75ms |
| checkoutservice | `hipstershop.CheckoutService/PlaceOrder` | (oggetto completo con indirizzo + carta) | 200ms |
| recommendationservice | `hipstershop.RecommendationService/ListRecommendations` | `{"user_id":"test","product_ids":["OLJCESPC7Z"]}` | 100ms |

Le RPC scelte sono quelle piu chiamate dal frontend nel workload mix reale — non sono chiamate sintetiche ma riproducono il pattern d'uso reale.

### Esecuzione

```powershell
# Tutti i backend su cluster1 (ordine automatico: catalog → cart → reco → checkout)
python experiments/ghz_backend_capacity.py --cluster cluster1

# Singolo servizio
python experiments/ghz_backend_capacity.py --cluster cluster1 --service cartservice

# Parametri personalizzati
python experiments/ghz_backend_capacity.py --cluster cluster1 --step-duration 90 --step-increment 5 --max-rate 200
```

### Output

Per ogni servizio lo script produce 3 file:

```
results/capacity/
  ghz_cartservice_cluster1_20260416_143022.json   ← summary + knee point
  ghz_cartservice_cluster1_20260416_143022.csv    ← serie temporale step-by-step
  ghz_cartservice_cluster1_20260416_143022.png    ← grafico latenza + throughput
```

E a console:

```
================================================================
  CAPACITY TEST: cartservice @ cluster1
================================================================
  RPC:           hipstershop.CartService/GetCart
  Porta gRPC:    7070
  SLA p95:       50ms
  Dipendenze:    redis-cart
================================================================

  Step |   Rate |   Actual |     p50 |     p95 |     p99 |  Err% | SLA
  --------------------------------------------------------------------
     1 |  10r/s |   10.0r/s |     3ms |     8ms |    12ms | 0.0% |  OK
     2 |  20r/s |   20.0r/s |     3ms |     9ms |    14ms | 0.0% |  OK
     3 |  30r/s |   30.0r/s |     4ms |    11ms |    18ms | 0.0% |  OK
     4 |  40r/s |   40.0r/s |     4ms |    14ms |    22ms | 0.0% |  OK
     5 |  50r/s |   49.8r/s |     5ms |    19ms |    30ms | 0.0% |  OK
     6 |  60r/s |   59.2r/s |     7ms |    28ms |    45ms | 0.0% |  OK
     7 |  70r/s |   68.1r/s |    10ms |    42ms |    68ms | 0.0% | WARN
     8 |  80r/s |   72.3r/s |    15ms |    65ms |   102ms | 0.5% | FAIL

  SLA violato per 2 step consecutivi → saturazione confermata

================================================================
  Knee point:            59.2 rps/replica (a 60 rps target)
  Capacita raccomandata: 47 rps/replica (80% del knee)
  Stima corrente:        50 rps/replica (delta: +18.4%)
================================================================
```

---

## 7. Phase 2 — Osservazione system-level con backend_capacity_observer

### Scopo

Monitorare **tutti e 5 i servizi** via Prometheus/Hubble **durante** un test di carico sul frontend (k6 o Locust). Identifica quale servizio satura per primo sotto carico reale.

### Differenza rispetto a ghz

| Aspetto | ghz (Phase 1) | Observer (Phase 2) |
|---|---|---|
| Chi genera il traffico | ghz (gRPC diretto al backend) | k6/Locust (HTTP al frontend) |
| Rate controllato su backend? | Si (lo imposti esplicitamente) | No (dipende dal call graph) |
| Cosa misura | Capacity isolata del singolo servizio | Comportamento sotto carico reale |
| Domanda a cui risponde | "Quante rps puo gestire 1 replica?" | "Sotto carico, quale servizio satura primo?" |

### Come funziona

L'observer non genera traffico — lo **osserva**. Ogni 15 secondi interroga Prometheus per ogni servizio:

| Metrica | Query Prometheus | Cosa misura |
|---|---|---|
| Traffico (rps) | `sum(rate(hubble_http_requests_total{destination_workload="<svc>"}[1m]))` | Richieste/s in arrivo al servizio |
| Latenza p95 (ms) | `histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{destination_workload="<svc>"}[1m])) by (le)) * 1000` | Tempo di risposta |
| Error rate | `rate(hubble_http_requests_total{..., http_status_code=~"5.."}[1m]) / rate(total)` | Percentuale errori 5xx |
| CPU (cores) | `sum(rate(container_cpu_usage_seconds_total{pod=~"<svc>.*"}[1m]))` | Consumo CPU totale |
| Pod count | `count(kube_pod_info{created_by_name=~"<svc>-.*"})` | Numero di repliche attive |

### Esecuzione

Servono **2 terminali** in parallelo:

```powershell
# Terminale 1 — genera traffico
k6 run -e CLUSTER=cluster1 experiments/k6_capacity_test.js

# Terminale 2 — osserva (lanciare subito dopo k6)
python experiments/backend_capacity_observer.py --cluster cluster1
```

Alla fine (Ctrl+C sull'observer), produce il report:

```
================================================================
  CAPACITY ANALYSIS — Per-Service Knee Points
================================================================

  [cluster1]
  Service                   | Knee rps/rep | Max rps/rep | Recommended | Current | Delta  | Violations
  ----------------------------------------------------------------------------------------------------
  frontend                  |        39.2  |       45.1  |          31 |      39 |  +0.5% |          2
  cartservice               |        35.8  |       37.2  |          29 |      50 | -28.4% |          1  <<
  productcatalogservice     |        78.5  |       82.1  |          63 |      80 |  -1.9% |          0
  checkoutservice           |         6.8  |        7.2  |           5 |      20 | -66.0% |          3  <<
  recommendationservice     |        31.2  |       35.8  |          25 |      40 | -22.0% |          1  <<

  Bottleneck: checkoutservice (knee=6.8 rps/replica)
```

### Output

```
results/capacity/
  backend_observer_20260416_150000.csv    ← serie temporale per servizio (ogni 15s)
  backend_observer_20260416_150000.json   ← summary con knee points e raccomandazioni
```

---

## 8. Il knee point e il valore raccomandato

### Definizione

Il **knee point** e' l'ultimo step in cui la latenza p95 e' sotto il SLA e l'error rate e' sotto il 2%. E' il punto sulla curva latenza-throughput dove il sistema passa da scaling lineare a saturazione.

```
Latenza p95
(ms)
    200 |                                    /
    150 |                               ___/  ← SLA violato (oltre questo)
    100 |                         ____/
     50 |                   ____/  ← knee point
     20 |            ______/
     10 |___________/
        +--+--+--+--+--+--+--+--+-- RPS/replica
          10  20  30  40  50  60  70  80
```

### Formula del valore raccomandato

```
recommended = int(knee_point_rps * 0.80)
```

Il 20% di headroom serve perche':
- Il knee point e' il **limite teorico** — operare li non lascia margine per picchi
- DMOS deve scalare **prima** che la latenza degradi (proattivo)
- Con 20% di buffer, DMOS ha tempo di reagire

### SLA per servizio

I SLA sono definiti in `config/services.yaml` e riflettono il ruolo del servizio:

| Servizio | SLA p95 | Motivazione |
|---|---|---|
| frontend | 200ms | Percepito dall'utente, baseline idle 60-70ms |
| cartservice | 50ms | gRPC locale + Redis, deve essere velocissimo |
| productcatalogservice | 75ms | Read-only in-memory Go, veloce |
| checkoutservice | 200ms | Path lungo (fan-out), tolleranza piu alta |
| recommendationservice | 100ms | Python piu lento di Go, ma non critico |

---

## 9. Come i risultati si collegano a DMOS

### Il flusso capacity → config → scaling

```
                    capacity test
                         |
                         v
              +---------------------+
              | capacity_req_per_sec |  <-- config/services.yaml
              +---------------------+
                         |
                         v
              +---------------------+
              | ReplicaScaler       |  <-- src/level2/scaler.py
              | base_replicas =     |
              |   predicted_traffic |
              |   / capacity        |
              +---------------------+
                         |
                         v
              +---------------------+
              | target_replicas =   |
              |   base + safety     |
              |   + PD correction   |
              |   + constraints     |
              +---------------------+
                         |
                         v
              kubectl scale deployment <svc> --replicas=N
```

### Dove finiscono i valori misurati

**Per servizi con capacita uniforme tra cluster** (backend gRPC):

```yaml
# config/services.yaml
cartservice:
  capacity_req_per_sec: 47   # ← valore da ghz (80% di knee=59)
```

I backend gRPC comunicano intra-cluster (cartservice-c1 → redis-cart-c1), quindi il netem inter-cluster **non li impatta**. Un valore unico `capacity_req_per_sec` e' sufficiente.

**Per servizi con capacita variabile per cluster** (frontend HTTP):

```yaml
# config/services.yaml
frontend:
  capacity_req_per_sec: 39     # ← valore globale (cluster migliore)
  capacity_per_cluster:
    cluster1: 39               # DE, 0ms netem
    cluster2: 35               # FR, 150ms netem
    cluster3: 30               # PL, 350ms netem
```

Il frontend riceve traffico HTTP dall'esterno, quindi il netem tra il client e il cluster impatta la latenza e potenzialmente il throughput.

### Il metodo `get_capacity_for_cluster()` in config_loader.py

```python
def get_capacity_for_cluster(self, cluster_name: str) -> float:
    if self.capacity_per_cluster and cluster_name in self.capacity_per_cluster:
        return float(self.capacity_per_cluster[cluster_name])
    return float(self.capacity_req_per_sec)
```

Se `capacity_per_cluster` e' definito, usa il valore per-cluster. Altrimenti usa `capacity_req_per_sec`. Per i backend gRPC tipicamente basta il valore globale.

---

## 10. Guida operativa passo-passo

### Prerequisiti

```powershell
# 1. ghz installato
ghz --version

# 2. OpenVPN attiva
ping <CLUSTER1_IP>

# 3. kubectl funzionante
kubectl get nodes --context cluster1
```

### Preparazione cluster

```powershell
# Ferma DMOS (Ctrl+C se in esecuzione)

# Scala tutto a 1 replica su tutti i cluster
$services = @("frontend","cartservice","productcatalogservice","checkoutservice","recommendationservice")
$clusters = @("cluster1","cluster2","cluster3")

foreach ($cluster in $clusters) {
    foreach ($svc in $services) {
        kubectl scale deployment $svc -n online-boutique --replicas=1 --context $cluster
    }
}

# Verifica (aspetta ~30s)
Start-Sleep -Seconds 30
foreach ($cluster in $clusters) {
    Write-Host "=== $cluster ==="
    kubectl get pods -n online-boutique --context $cluster
}

# Verifica servizi non scalati attivi (redis, currency, ecc.)
kubectl get pods -n online-boutique --context cluster1 | Select-String "redis|currency|shipping|payment|email"
```

### Phase 1 — ghz su tutti i cluster

```powershell
cd $DMOS_REPO   # path della repo DMOS sul tuo sistema

# Cluster 1 (DE, 0ms netem) — ~15-20 minuti
python experiments/ghz_backend_capacity.py --cluster cluster1

# Cluster 2 (FR, 150ms netem) — ~15-20 minuti
python experiments/ghz_backend_capacity.py --cluster cluster2

# Cluster 3 (PL, 350ms netem) — ~15-20 minuti
python experiments/ghz_backend_capacity.py --cluster cluster3
```

### Phase 2 — k6 + observer (2 terminali)

```powershell
# Terminale 1:
k6 run -e CLUSTER=cluster1 experiments/k6_capacity_test.js

# Terminale 2 (lanciare subito dopo):
python experiments/backend_capacity_observer.py --cluster cluster1

# Alla fine: Ctrl+C sull'observer → genera report
```

### Ripristino

```powershell
# Repliche al minimo (DMOS scalera da solo)
foreach ($cluster in $clusters) {
    foreach ($svc in $services) {
        kubectl scale deployment $svc -n online-boutique --replicas=1 --context $cluster
    }
}

# Riavvia DMOS
python -m src.dmos_main
```

---

## 11. Interpretazione dei risultati e casi limite

### Risultati attesi per i backend gRPC

I backend gRPC comunicano intra-cluster (stessa macchina), quindi il netem inter-cluster **non dovrebbe impattarli**. I risultati tra cluster1, cluster2, cluster3 dovrebbero essere quasi identici per i backend. Se lo sono, conferma che il netem e' configurato correttamente (impatta solo il traffico inter-nodo, non quello intra-pod).

### Segnali di attenzione

| Segnale | Causa probabile | Azione |
|---|---|---|
| Backend identico tra cluster | Corretto: gRPC intra-cluster non impattato da netem | Usa un valore unico `capacity_req_per_sec` |
| Backend molto diverso tra cluster | Contesa CPU con altri workload, risorse diverse tra nodi | Verifica `kubectl top nodes`, considera `capacity_per_cluster` |
| Knee point molto piu basso della stima | Stima era ottimistica | Aggiorna services.yaml con il valore misurato |
| Knee point molto piu alto della stima | Stima era conservativa | Aggiorna — permette meno repliche, risparmio risorse |
| checkoutservice satura prima degli altri | Fan-out a 6 servizi, piu lento per design | Normale — verifica che il rate atteso (5% del frontend) non superi il knee |
| Errori al primo step (SLA violato subito) | Servizio non raggiungibile, port-forward fallito, pod non running | Verifica `kubectl get pods`, riprova |
| p95 altissimo ma throughput normale | Una dipendenza e' lenta (es. redis sovraccarico) | Testa le dipendenze isolatamente |

### Quando ripetere il test

- Cambio dei resource limits in `config/services.yaml`
- Aggiornamento delle immagini dei microservizi
- Cambio hardware dei nodi cluster
- Aggiunta/rimozione di servizi non scalati
- Dopo qualsiasi modifica che cambia il profilo di risorse

---

## 12. Confronto tra i 3 strumenti di capacity testing

| | k6 / Locust | k6 + Observer | ghz |
|---|---|---|---|
| **Cosa testa** | Frontend HTTP (end-to-end) | Frontend + osserva tutti i backend | Singolo backend gRPC |
| **Protocollo** | HTTP | HTTP + query Prometheus | gRPC |
| **Rate controllato?** | Sul frontend | Sul frontend | Sul backend specifico |
| **Isola il servizio?** | No | No | Si (ma con dipendenze reali) |
| **Vede i backend?** | No (latenza aggregata) | Si (per servizio via Hubble) | Si (quello testato) |
| **Dipendenze** | Implicite nel frontend | Osservate via Prometheus | Colpite realmente |
| **Produce** | `capacity_per_cluster` frontend | Bottleneck analysis | `capacity_req_per_sec` backend |
| **Durata** | ~25 min / cluster | Come k6 (run in parallelo) | ~15-20 min / cluster |
| **Serve** | Gia fatto (39/35/30) | Validazione system-level | Calibrazione backend |
| **Tool** | k6_capacity_test.js | backend_capacity_observer.py | ghz_backend_capacity.py |

### Quando usare quale

```
Devo calibrare i backend?
  → ghz_backend_capacity.py (Phase 1)

Devo verificare che il sistema regga sotto carico reale?
  → k6_capacity_test.js + backend_capacity_observer.py (Phase 2)

Devo ri-calibrare il frontend dopo una modifica?
  → k6_capacity_test.js (gia esistente)

Devo capire perche il p95 e' alto sotto carico?
  → k6 + backend_capacity_observer.py (identifica il bottleneck)
```

---

## 13. Validita scientifica e riferimenti

### Metodologia

La metodologia adottata (stepped load + knee detection + safety margin) e' consistente con la letteratura sulla caratterizzazione delle performance di microservizi:

- **Gan et al. (ASPLOS 2019)** — *An Open-Source Benchmark Suite for Microservices and Their Hardware-Software Implications for Cloud & Edge Systems*: propongono il profiling per-servizio sotto carico controllato per identificare i colli di bottiglia nelle catene di microservizi. La nostra Phase 1 (ghz per servizio) segue questa metodologia.

- **Sriraman & Dhanotia (ASPLOS 2020)** — *Accelerometer: Understanding Acceleration Opportunities for Data Center Overheads at Hyperscale*: analizzano la tail latency (p95/p99) nei microservizi e mostrano come la latenza end-to-end sia dominata dal servizio piu lento nella catena. La nostra Phase 2 (observer) identifica questo servizio.

- **Universal Scalability Law (Gunther)** — modella il throughput come funzione della concorrenza, identificando il punto di saturazione (knee point) oltre il quale aggiungere carico riduce il throughput anziche aumentarlo. Il nostro stepped load test misura esattamente questa curva.

- **Zhou et al. (SoCC 2018)** — *Overload Control for Scaling WeChat Microservices*: mostrano l'importanza di capacity limits accurati per il controllo dell'overload nei microservizi. Stime approssimative portano a cascading failures.

### Limiti della misurazione

1. **ghz via port-forward**: il port-forward aggiunge un hop di rete rispetto al path reale (frontend → backend intra-cluster). La latenza misurata e' leggermente piu alta del reale.

2. **Dipendenze condivise**: quando si testa checkoutservice, i servizi downstream (currencyservice, ecc.) ricevono traffico solo dal checkout. Nel sistema reale ricevono traffico anche dal frontend direttamente. La capacity misurata per il checkout potrebbe essere ottimistica se le dipendenze sono gia cariche.

3. **Workload sintetico**: il payload di test (es. `GetCart` con un `user_id` fisso) non replica esattamente la distribuzione reale dei dati. La capacity misurata e' rappresentativa ma non identica al workload di produzione.

4. **Single-node**: il testbed K3s single-node per cluster implica contesa CPU tra tutti i servizi. La capacity misurata e' specifica per questo ambiente e non trasferibile a cluster multi-nodo.

---

## 14. Risultati backend

> Questa sezione va compilata dopo l'esecuzione dei test.
> Inserire qui i risultati di ghz per ogni servizio e cluster.

### Phase 1 — ghz (da compilare)

| Servizio | Cluster | Knee (rps/rep) | p95 al knee | Raccomandato (80%) | Stima precedente | Delta |
|---|---|---|---|---|---|---|
| productcatalogservice | cluster1 | — | — | — | 80 | — |
| productcatalogservice | cluster2 | — | — | — | 80 | — |
| productcatalogservice | cluster3 | — | — | — | 80 | — |
| cartservice | cluster1 | — | — | — | 50 | — |
| cartservice | cluster2 | — | — | — | 50 | — |
| cartservice | cluster3 | — | — | — | 50 | — |
| recommendationservice | cluster1 | — | — | — | 40 | — |
| recommendationservice | cluster2 | — | — | — | 40 | — |
| recommendationservice | cluster3 | — | — | — | 40 | — |
| checkoutservice | cluster1 | — | — | — | 20 | — |
| checkoutservice | cluster2 | — | — | — | 20 | — |
| checkoutservice | cluster3 | — | — | — | 20 | — |

### Phase 2 — System-level observer (da compilare)

| Servizio | Cluster | Max rps/rep osservato | p95 max osservato | Bottleneck? |
|---|---|---|---|---|
| frontend | cluster1 | — | — | — |
| cartservice | cluster1 | — | — | — |
| productcatalogservice | cluster1 | — | — | — |
| checkoutservice | cluster1 | — | — | — |
| recommendationservice | cluster1 | — | — | — |

### Valori aggiornati in services.yaml (da compilare)

```yaml
# Dopo i test, aggiornare config/services.yaml:
cartservice:
  capacity_req_per_sec: __   # era 50, misurato: __
productcatalogservice:
  capacity_req_per_sec: __   # era 80, misurato: __
checkoutservice:
  capacity_req_per_sec: __   # era 20, misurato: __
recommendationservice:
  capacity_req_per_sec: __   # era 40, misurato: __
```
