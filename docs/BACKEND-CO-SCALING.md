# Backend Co-Scaling in DMOS

Documentazione tecnica del dimensionamento dei servizi backend in DMOS.
Descrive **cosa**, **perché** e **come** — con l'algoritmo preciso, le
proprietà, i limiti e il posizionamento rispetto alla letteratura. Nel paper
NCA 2026 corrisponde all'Eq. 11; il quadro complessivo è in
[`dmos-architecture.md` §9](dmos-architecture.md#9-backend-co-scaling).

---

## 1. Contesto e problema

Online Boutique è un'applicazione a microservizi: un servizio **frontend**
rivolto all'utente che, a ogni richiesta HTTP, effettua un **fan-out** di
chiamate gRPC verso più servizi **backend** (`cartservice`,
`productcatalogservice`, `checkoutservice`, `recommendationservice`,
`currencyservice`, `adservice`, `shippingservice`).

DMOS deve decidere, per ogni servizio e ogni cluster, **quante repliche**
allocare. La domanda chiave: **su quale segnale si dimensiona un backend?**

### 1.1 Perché NON sul conteggio di richieste del backend

L'istinto è: misurare il tasso di richieste che arrivano al backend
(`hubble_http_requests_total{destination_workload=<backend>}` via Cilium
Hubble) e
dimensionare di conseguenza. **Questo segnale è inaffidabile sotto saturazione.**

Hubble conta **ogni richiesta HTTP/gRPC sul filo**. Quando il sistema è in
distress, le chiamate gRPC interne falliscono e vengono **ritentate** (retry di
gRPC / Envoy). Ogni retry è una nuova richiesta che Hubble conta — e **non
esiste un'etichetta che distingua un retry da un originale**. Risultato: il
contatore si gonfia di un fattore `fan-out × amplificazione-retry`.

Misurazione reale (run flash-crowd, fase di cooldown, carico esterno offerto =
50 req/s):

| Metrica | Lettura | Verità |
|---|---|---|
| `hubble_http_requests_total` (productcatalogservice) | ~1040 req/s | ~50 req/s esterni |
| Fattore di gonfiaggio | ~20× | — |

Uno scaler che si fida di questo numero **scala su una bugia**. Peggio: lo
scaling provoca churn degli endpoint Cilium/Envoy → più connessioni gRPC rotte
→ più retry → contatore ancora più gonfio → **loop di feedback positivo**.
È la firma di un **metastable failure** (Bronson et al., HotOS 2021): un guasto
che persiste anche dopo la rimozione del trigger, sostenuto dai retry.

**Conclusione:** il volume di traffico interno è un segnale di domanda
**corrotto** sotto saturazione. Va escluso dall'anello di controllo.

---

## 2. Principio di design

Due regole:

1. **Misurare la domanda al confine, non dentro il mesh.** L'unico segnale di
   domanda immune ai retry è il **tasso di ingress esterno** — il traffico che
   entra dalla Cilium Ingress, misurato da `get_ingress_rate()` (query
   client-side all'origine; fallback server-side con `source_workload=""` —
   vedi [`traffic-metrics-architecture.md`](traffic-metrics-architecture.md)). Il generatore di carico non
   ritenta, e il fan-out interno non passa dall'Ingress → questo conteggio =
   domanda reale.

2. **Propagare la domanda lungo il grafo delle dipendenze.** Il carico di un
   backend **non è una variabile indipendente**: è *causato* dal fan-out del
   frontend. Quindi il backend si dimensiona **derivandolo dal frontend**, non
   misurandolo. Questo è **dependency-aware autoscaling**.

Schema:

```
   ingress rate (esterno, pulito)
        │
        ▼
   budget N + placement L1 + correzione L2 (frontend)
        │
        ▼
   repliche frontend per cluster                             (fatto Kubernetes)
        │
        ▼  co-location (questo documento)
   N repliche backend per cluster   (cart / pcatalog / checkout / reco /
                                     currency / ad / shipping)
```

Il conteggio Hubble corrotto **non entra** in nessuno dei due passi: il
frontend usa l'ingress rate, i backend usano un fatto Kubernetes (il numero di
repliche del frontend).

---

## 3. L'algoritmo di co-scaling

Implementato in `_enforce_colocation()` (`src/dmos_main.py`), invocato al
termine di `schedule_service('frontend')` — quindi a ogni ciclo di scheduling
del frontend (~30 s).

### 3.1 Grafo delle dipendenze

Il servizio d'ingresso e i servizi co-scalati, nell'ordine in cui vengono
processati (condividono il budget di churn del ciclo), si leggono da
`config/services.yaml → application` (vedi
[configuration-reference.md §4.1](configuration-reference.md#41-application--the-application-dmos-manages)).
Per Online Boutique:

```yaml
application:
  entry_service: frontend
  co_scaled_services: [cartservice, productcatalogservice, checkoutservice,
                       recommendationservice, currencyservice, adservice,
                       shippingservice]   # stateless sul path caldo; redis-cart escluso (stateful)
```

DMOS ne ricava `service_dependencies = {entry_service: co_scaled_services}`.

Un grafo a una stella di profondità 1: il `frontend` è il parent, i sette
backend sono i dipendenti.

### 3.2 Formula

Per ogni backend `dep` e ogni cluster `c`:

```
parent_reps(c)   = repliche correnti del frontend su c        [k8s, lettura fresca]
ratio_nominale   = ( capacita_frontend(c) / capacita_backend(c) ) × fanout(dep)
ratio_effettivo  = max( ratio_nominale, r_floor )             [r_floor = 0.3]
target(dep,c)    = clamp( min_repliche(dep,c),
                          max_repliche(dep,c),
                          ceil( parent_reps(c) × ratio_effettivo ) )
current(dep,c)   = repliche correnti del backend su c         [k8s, lettura fresca]
delta            = target(dep,c) − current(dep,c)
```

- `capacita_*` = `capacity_req_per_sec` da `config/services.yaml`, con override
  per-cluster (`get_capacity_for_cluster`). Sono i valori **misurati** dal
  capacity test gRPC.
- `fanout(dep)` = `calls_per_request` da `config/services.yaml`: quante chiamate
  riceve il backend per richiesta esterna, **misurato con Hubble**. Un servizio
  chiamato N volte per richiesta riceve N× traffico e va scalato N volte di più
  (es. `currencyservice` ~3.4×: senza fanout resterebbe sotto-provisionato).
- `r_floor` = `backend_colocation_floor = 0.3` (`src/dmos_main.py`): pavimento
  sul rapporto. I benchmark a replica isolata **sovrastimano** la capacità
  effettiva sotto fan-out gRPC (esaurimento connection-pool, tail-latency
  cascade — Dean & Barroso, "The Tail at Scale"); con rapporti nominali molto
  bassi (es. cart: 60/400×0.65 ≈ 0.10) i backend resterebbero a 1 replica anche
  col frontend scalato → cascata di saturazione. Valore calibrato
  empiricamente: 0.5 sovra-provisionava a carichi moderati, 0.3 è l'equilibrio.
- `clamp` rispetta `min/max_replicas_per_cluster` di ogni backend (tier-aware).
- Le letture k8s sono **fresche** (subito prima della decisione) per evitare
  decisioni su valori cachati obsoleti.

### 3.3 Bidirezionale + budget di churn

`delta` viene applicato in **entrambe le direzioni**: quando il frontend sale i
backend salgono, quando il frontend scende i backend scendono. La co-location è
il **dimensionatore unico** dei backend. Il `delta` è applicato in un solo
passo (il rate limit +6/−1 vale per il frontend, non per il co-scaling); il
co-scaling non è soggetto a cooldown né a dead zone.

Ogni operazione rispetta i **budget di churn per ciclo** (anti-thundering-herd).
Il co-scaling ha budget **dedicati**, separati da quelli reattivi del frontend
(8 up / 2 down): senza separazione la rampa del frontend esauriva il budget e
affamava il co-scaling, lasciando i backend a 1 replica.

| Budget | Valore | Significato |
|---|---|---|
| `backend_scale_up_budget_per_cycle` | 12 | max pod backend aggiunti per ciclo (30 s), totale fleet |
| `backend_scale_down_budget_per_cycle` | 6 | max pod backend terminati per ciclo, totale fleet |

Se il budget è esaurito, l'operazione viene **rimandata** al ciclo successivo
(loggata come `deferred — backend budget esaurito`). Lo scaling si distribuisce
così su più cicli, dando al data-plane Cilium/Envoy il tempo di stabilizzarsi.

### 3.4 Esempio numerico

Capacità per replica e fanout (da `services.yaml`; frontend = 60 req/s):

| Backend | capacità (req/s) | fanout | ratio nominale = 60/cap × fanout | ratio effettivo = max(nominale, 0.3) |
|---|---|---|---|---|
| cartservice | 400 | 0.65 | 0.098 | 0.30 (floor) |
| productcatalogservice | 1200 | 2.25 | 0.113 | 0.30 (floor) |
| checkoutservice | 160 | 0.05 | 0.019 | 0.30 (floor) |
| recommendationservice | 40 | 0.40 | 0.60 | 0.60 (capacity) |
| currencyservice | 250 | 3.4 | 0.816 | 0.82 (capacity) |
| adservice | 400 | 0.43 | 0.065 | 0.30 (floor) |
| shippingservice | 500 | 0.14 | 0.017 | 0.30 (floor) |

Se al picco il frontend ha **6 repliche** su `ms-cluster-01` (cap tier HIGH):

| Backend | `ceil(6 × ratio_eff)` | target (post-clamp) |
|---|---|---|
| cartservice | ceil(1.80) = 2 | 2 (max 2) |
| productcatalogservice | ceil(1.80) = 2 | 2 |
| checkoutservice | ceil(1.80) = 2 | 2 (max 2) |
| recommendationservice | ceil(3.60) = 4 | 4 |
| currencyservice | ceil(4.90) = 5 | 5 |
| adservice | ceil(1.80) = 2 | 2 |
| shippingservice | ceil(1.80) = 2 | 2 (max 2) |

I backend "deboli" (capacità effettiva per replica bassa: recommendation) o ad
alto fanout (currency, ~3.4 chiamate per richiesta) scalano sul termine
capacità×fanout e ricevono più repliche; per gli altri governa il floor 0.3,
che li fa comunque crescere col frontend invece di lasciarli a 1 replica.

---

## 4. Per-cluster: perché conta

Il rapporto di co-scaling è calcolato **per cluster**, non globale, per due
motivi:

1. **Fleet eterogeneo.** I cluster sono su tre tier hardware (HIGH 8 vCPU,
   MEDIUM 6 vCPU, LOW 4 vCPU). `get_capacity_for_cluster` e
   `get_max_replicas_for_cluster` possono riflettere il tier: oggi in
   `services.yaml` variano per tier solo i tetti (`max_replicas_per_cluster`),
   mentre le capacità per replica sono uguali su tutti i cluster.

2. **Località gRPC.** Co-locare il backend **sullo stesso cluster** del
   frontend che lo chiama tiene il fan-out gRPC **intra-cluster**. Un backend
   piazzato altrove costringerebbe ogni chiamata a un hop inter-cluster
   (decine/centinaia di ms via ClusterMesh). La co-location per-cluster
   preserva la località: dove c'è frontend, lì ci sono i suoi backend.

---

## 5. Proprietà

| Proprietà | Come è garantita |
|---|---|
| **Immune alla corruzione da retry** | Usa solo fatti Kubernetes (repliche del frontend) + capacità misurate. Nessuna metrica di volume corruttibile nell'anello. |
| **Località gRPC preservata** | Co-scaling per-cluster: i backend stanno dove sta il frontend chiamante. |
| **Convergenza** | Bidirezionale: i backend inseguono il frontend in su e in giù; nessuna deriva possibile. |
| **Churn limitato** | Ogni operazione rispetta i budget per-ciclo; lo scaling si spalma nel tempo. |
| **Coerente con L1** | Nell'arm L1+L2, Level 1 piazza il frontend geo-aware; i backend ereditano quella distribuzione → restano vicini al chiamante. |

---

## 6. Implementazione

Modifiche in `src/dmos_main.py` (insieme di patch "option-1"):

1. **`_get_per_cluster_traffic`** — per il `frontend` usa `get_ingress_rate()`
   invece di `get_request_rate()` (sorgente di domanda pulita).
2. **`monitored_services = [entry_service]`** (il `frontend`) — solo il
   servizio d'ingresso è schedulato a traffico. Lo scaler L2 per-backend (che leggeva il conteggio Hubble
   corrotto) **non viene più invocato**.
3. **`_enforce_colocation`** — riscritta come dimensionatore unico,
   bidirezionale, per-cluster (questo documento).
4. **Rimossi** i blocchi *Backend reset* e *scale-down graduale backend* da
   `periodic_check_thread`: con la co-location come unico sizer bidirezionale i
   backend non possono andare alla deriva → reti di sicurezza obsolete.

### Interazione con Level 1 e Level 2

| Arm | Frontend | Backend |
|---|---|---|
| `off` | 1 replica fissa (DMOS spento) | 1 replica fissa |
| `l2` (`--no-scheduling`) | budget N dall'ingress rate, distribuzione uniforme N/K, correzione L2 | co-location |
| `l1l2` (campagna NCA), `nca` e gli altri modelli di score, `profile-*`, `legacy-*` (nomi di `experiments/run_naming.py`) | L1 (score) piazza il budget N; L2 corregge | co-location |

In tutti gli arm con DMOS la co-location è invocata identica al termine di
`schedule_service('frontend')`. La **score function di Level 1 non è
modificata**: continua a decidere la geo-distribuzione — la decide per il
frontend, e i backend la ereditano.

**Alternativa sperimentale.** Con `--naive-backends` la co-location è
sostituita da uno scaling reattivo dei backend sul loro rate L7 osservato
(`ceil(rps / 50)`, budget del frontend): è il baseline "metastabile" usato per
mostrare il problema del §1; nessun harness attuale lo usa.

**Osservabilità.** Il collector (`dmos_live.jsonl`) registra repliche e CPU di
cartservice, productcatalogservice, checkoutservice e recommendationservice,
ma non di currencyservice, adservice e shippingservice; le loro decisioni si
leggono in `dmos.log` (righe `Co-location: <cluster> — <servizio>: a → b`).

---

## 7. Limiti e assunzioni

Onestà intellettuale — la co-location è un controllo **feed-forward**
(anello aperto):

- **Assume un fan-out stabile.** Il rapporto fisso (capacità e
  `calls_per_request` costanti) è corretto se il
  rapporto chiamate-frontend / chiamate-backend non varia. Vale finché il mix
  di endpoint del workload è fisso (k6: 40 % `/`, 25 % pagine prodotto, 25 %
  carrello, 5 % cambio valuta, 5 % checkout); con un mix che cambia nel tempo
  il fan-out varierebbe.
- **Grafo a profondità 1.** Modella `frontend → backend`. Non cattura
  dipendenze `backend → backend` (in Online Boutique sono marginali).
- **Nessun feedback dallo stato reale del backend.** Se il modello di capacità
  è impreciso per un backend, non c'è correzione a posteriori.

Mitigazione/estensione possibile (fase 2): un **ritocco closed-loop sulla CPU**
del backend (segnale limitato [0,100%], non gonfiabile come il conteggio) sopra
il floor feed-forward della co-location.

---

## 8. Posizionamento nella letteratura

La co-location è la versione **white-box e interpretabile** di un approccio
riconosciuto:

- **Dependency-aware / call-graph-aware autoscaling** è una direzione mainstream
  (survey 2025, arXiv:2507.17128): "incorporare l'analisi del percorso di
  invocazione nelle decisioni di scaling permette azioni coordinate ed evita la
  propagazione dei guasti".
- **Propagazione della domanda dall'ingress** lungo il grafo per dimensionare i
  componenti è un metodo noto (scaling guidato dal traffico utente con SLO
  trace-level).
- Le versioni **apprese** sostituiscono il rapporto fisso con un modello:
  Graph-PHPA (LSTM per il workload + GNN per la relazione workload→risorse),
  STaleX (spatiotemporal-aware). Utili per workload eterogenei o grafi profondi,
  al costo di interpretabilità e di un onere di training.
- Il problema che la co-location **evita** — lo scaler che alimenta un loop di
  feedback via metriche gonfiate dai retry — è un **metastable failure**
  (Bronson et al., HotOS 2021).

**Sintesi per la tesi:** DMOS adotta una propagazione feed-forward
interpretabile, ancorata a capacità misurate, adeguata al workload omogeneo
dell'esperimento; un modello graph-based appreso è l'estensione naturale,
lasciata come lavoro futuro.

### Riferimenti

- N. Bronson et al., *Metastable Failures in Distributed Systems*, HotOS 2021.
- *Auto-scaling Approaches for Microservice Applications: A Survey and
  Taxonomy*, arXiv:2507.17128, 2025.
- *Graph-PHPA: Graph-based Proactive Horizontal Pod Autoscaling using
  LSTM-GNN*, arXiv:2209.02551.
- *STaleX: A Spatiotemporal-Aware Adaptive Auto-scaling Framework for
  Microservices*, arXiv:2501.18734.
