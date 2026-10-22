# Capacity Test gRPC dei Backend — Documentazione

> Manuale di `experiments/run_capacity_grpc.sh` + `experiments/k6_capacity_grpc.js`:
> cosa misura, come funziona, come si interpreta il risultato.
>
> **In una frase**: misura empiricamente quante richieste/secondo regge **una
> replica** di ciascun microservizio backend gRPC (cartservice,
> productcatalogservice, recommendationservice, checkoutservice), eseguendo k6
> **dentro il cluster**, per sostituire le stime a occhio di
> `capacity_req_per_sec` in `config/services.yaml` con valori **misurati**.
>
> **Cross-reference**:
> - `experiments/run_capacity_grpc.sh` — l'orchestratore
> - `experiments/k6_capacity_grpc.js` — il client di carico gRPC (gira in-cluster)
> - `experiments/demo.proto` — il proto di Online Boutique (hipstershop)
> - `experiments/run_capacity.sh` — l'equivalente HTTP per il **frontend**
> - `config/services.yaml` — dove finiscono i valori misurati
> - `experiments/THROTTLE-CHECK.md` — il test gemello lato CPU/cluster
> - `dmos-iac/HETEROGENEOUS-CLUSTER-SETUP.md` sez. 4.4 — il modello di capacita'
>   (`dmos-iac` è il repository esterno di provisioning del fleet, non incluso qui)
>
> **Copertura:** i quattro backend sopra hanno capacità misurate. DMOS co-scala
> anche `currencyservice`, `adservice` e `shippingservice`, le cui capacità in
> `services.yaml` sono ancora **stime**: lo script non ha un mapping per loro.

---

## Indice

1. [Cos'e' e perche' serve](#1-cose-e-perche-serve)
2. [Perche' un test separato da quello HTTP](#2-perche-un-test-separato-da-quello-http)
3. [Perche' IN-CLUSTER e non kubectl port-forward](#3-perche-in-cluster-e-non-kubectl-port-forward)
4. [Come funziona lo script](#4-come-funziona-lo-script)
5. [Il problema dell'isolamento (caveat per backend)](#5-il-problema-dellisolamento-caveat-per-backend)
6. [Metodologia: knee-point e SLA](#6-metodologia-knee-point-e-sla)
7. [Uso: comandi e variabili env](#7-uso-comandi-e-variabili-env)
8. [Output di esempio e interpretazione](#8-output-di-esempio-e-interpretazione)
9. [Come aggiornare config/services.yaml](#9-come-aggiornare-configservicesyaml)
10. [Capacity gRPC vs Capacity HTTP vs Throttle check](#10-capacity-grpc-vs-capacity-http-vs-throttle-check)
11. [Limiti noti e quando rieseguirlo](#11-limiti-noti-e-quando-rieseguirlo)

---

## 1. Cos'e' e perche' serve

DMOS calcola la capacita' di un cluster per ogni servizio come:

```
capacita' cluster = capacity_req_per_sec x max_replicas_per_cluster
```

Per il **frontend** `capacity_req_per_sec = 60` e' misurato (`run_capacity.sh`).
Per i **4 backend gRPC** i valori in `config/services.yaml` erano **stime a
occhio**:

| Backend | `capacity_req_per_sec` stimato | Origine |
|---|---|---|
| cartservice | 50 | stima ("1 replica gRPC gestisce facilmente 50+ rps") |
| productcatalogservice | 80 | stima ("Go read-only, 100+ rps facilmente") |
| recommendationservice | 40 | stima ("Python piu' lento, 30+ rps") |
| checkoutservice | 20 | stima ("lento ma poco traffico, 20+ rps") |

Una stima non e' un dato pubblicabile: se DMOS decide lo scheduling su numeri
inventati, il componente **Phi_cap** della score function poggia sul nulla.
Questo strumento sostituisce le 4 stime con 4 **misure**.

> Risultato dello script: una riga `capacity_req_per_sec: <N>` per ogni backend,
> da incollare in `config/services.yaml`.

---

## 2. Perche' un test separato da quello HTTP

`run_capacity.sh` testa il **frontend** mandando richieste **HTTP** all'Ingress
Cilium. Non puo' testare i backend perche':

1. **I backend parlano gRPC, non HTTP.** Espongono solo un Service gRPC su
   ClusterIP. Non c'e' un Ingress HTTP che li raggiunge.
2. **Non sono esposti all'esterno.** Solo il frontend ha un Ingress.

Quindi serve un client che parli gRPC con `demo.proto`: `k6_capacity_grpc.js`
usa il modulo integrato `k6/net/grpc`.

---

## 3. Perche' IN-CLUSTER e non kubectl port-forward

> Questa sezione documenta una scelta di metodologia importante (utile anche
> per il capitolo "metodi" della tesi).

La **prima versione** dello strumento lanciava k6 sul PC e raggiungeva il
backend con `kubectl port-forward`. **Era sbagliato.** Un run completo sui 4
backend lo ha dimostrato con i dati:

**Prova 1 — tassa fissa di latenza.** Il `p50` (mediana) al carico piu' basso,
su tutti e 4 i backend:

| Backend | Carico minimo | p50 misurato | p50 atteso (servizio) |
|---|---|---|---|
| cartservice | 25 rps | 47 ms | ~1-3 ms |
| productcatalogservice | 100 rps | 47 ms | ~3-8 ms |
| recommendationservice | 10 rps | 49 ms | ~10-30 ms |
| checkoutservice | 5 rps | 54 ms | variabile |

Quattro servizi di complessita' diversa, a carico quasi nullo, **inchiodati a
~47-54 ms**: non e' il microservizio, e' il round-trip del `kubectl
port-forward` (PC -> API server -> pod). Una tassa fissa su *ogni* richiesta.
cartservice e productcatalog hanno SLA p95 = 50 ms: il pavimento da ~47 ms se
lo mangiava -> `knee n/a` con il pod completamente scarico.

**Prova 2 — il port-forward crolla sotto carico.** Ai rate alti i log k6 erano
pieni di:

```
dial tcp 127.0.0.1:50555: connectex: No connection could be made
because the target machine actively refused it
```

`127.0.0.1:50555` e' il listener **locale** del port-forward. Quando k6 sale di
rate crea molte VU -> molte connessioni TCP nuove -> `kubectl port-forward` non
regge e le rifiuta. I "knee" osservati (recommendation a 150 rps, checkout a 50
rps) erano il limite del **port-forward**, non del microservizio.

**Conclusione**: `kubectl port-forward` non e' uno strumento valido per
benchmarkare un microservizio. La misura va fatta **in-cluster**.

**La soluzione**: k6 gira come **Job dentro il cluster**, nel namespace
`online-boutique`, e colpisce il backend via ClusterIP (es. `cartservice:7070`).
Vantaggi:

- gRPC **pod-to-pod**: latenza sub-millisecondo, nessuna tassa fissa.
- Nessun listener fragile: la rete del cluster regge migliaia di connessioni.
- Il traffico attraversa lo stesso **proxy L7 Cilium** del traffico reale
  frontend->backend (le `CiliumNetworkPolicy` `l7-visibility-*` parsano quelle
  porte come HTTP). Misuriamo il **percorso di produzione**.
- E' anche il metodo che un reviewer si aspetta.

> Le CNP `l7-visibility-backends` ammettono ingress `fromEndpoints: [{}]` —
> qualsiasi pod **dello stesso namespace**. Il Job k6, essendo in
> `online-boutique`, e' autorizzato a chiamare i backend senza modifiche alle
> policy.

---

## 4. Come funziona lo script

`run_capacity_grpc.sh`, per ogni backend, esegue in ordine:

```
 0. Pre-flight: kubectl, jq, k6 script, demo.proto, kubeconfig target
 1. loadgenerator -> 0 su TUTTI i cluster   (niente traffico di fondo)
 2. ConfigMap dmos-k6-cap-scripts  (k6_capacity_grpc.js + demo.proto)
 3. backend target -> 1 replica sul cluster target
 4. dipendenze del backend -> 4 repliche sul cluster target   (boost)
 5. attende Ready: backend + dipendenze (rollout status)
 6. crea un Job k6 (immagine grafana/k6) che monta la ConfigMap su /scripts
    e colpisce <backend>:<porta> via ClusterIP
 7. segue `kubectl logs -f` del pod k6  (output pulito, righe appese)
 8. estrae il JSON di summary dai log (fra marcatori) -> results/
 9. cancella il Job, ripristina le dipendenze -> 1 replica
10. stampa il capacity_req_per_sec consigliato (80% del knee-point)
```

Con `--all` ripete i passi 3-10 per tutti e 4 i backend e stampa una tabella
riepilogativa + lo snippet pronto per `services.yaml`.

### Mappatura backend -> metodo gRPC

| Backend | Porta gRPC | Metodo testato |
|---|---|---|
| cartservice | 7070 | `hipstershop.CartService/GetCart` |
| productcatalogservice | 3550 | `hipstershop.ProductCatalogService/ListProducts` |
| recommendationservice | 8080 | `hipstershop.RecommendationService/ListRecommendations` |
| checkoutservice | 5050 | `hipstershop.CheckoutService/PlaceOrder` |

### Il Job k6

- Immagine: `grafana/k6:latest` (k6 standard, modulo gRPC integrato).
  Sovrascrivibile con `DMOS_K6_IMAGE` se serve un registry interno.
- ConfigMap `dmos-k6-cap-scripts` montata su `/scripts`: contiene
  `k6_capacity_grpc.js` e `demo.proto`. k6 carica il proto con
  `load(['.'], 'demo.proto')` (accanto allo script).
- Risorse del pod k6: requests `500m`/`512Mi`, limits `3`/`2Gi` (il generatore
  di carico ha bisogno di CPU). Per questo conviene eseguire il test su un
  cluster con margine — il default `ms-cluster-01` (tier HIGH, 8 vCPU).
- Il JSON machine-readable arriva fra i marcatori
  `===DMOS_CAPACITY_JSON_BEGIN===` / `===DMOS_CAPACITY_JSON_END===` su stdout,
  recuperato da `kubectl logs`.

### Perche' si potenziano le dipendenze (passo 4)

Se misuro recommendationservice ma productcatalogservice (1 replica) satura per
primo, sto misurando productcatalog. Per evitarlo le dipendenze del backend
vengono scalate a 4 repliche **sul cluster target**; con
`service.cilium.io/affinity: local` il backend chiama le repliche locali.
A fine test tornano a 1.

### Perche' loadgenerator -> 0 (passo 1)

Se il loadgenerator gira, il frontend chiama in continuazione i backend e quel
traffico contamina la misura. Lo script lo spegne su **tutti** i cluster e lo
**lascia a 0** — va riacceso a mano prima della fase di traffico:

```bash
for c in 01 02 03 04 05 06; do
  kubectl --kubeconfig "${KUBECONFIG_DIR:-$HOME/dmos-iac}"/kubeconfig-ms-cluster-$c \
    -n online-boutique scale deploy loadgenerator --replicas=1
done
```

---

## 5. Il problema dell'isolamento (caveat per backend)

Non tutti i backend si isolano allo stesso modo. Va dichiarato apertamente
(anche nella tesi) cosa misura davvero il test:

| Backend | Isolamento | Cosa misura davvero |
|---|---|---|
| **cartservice** | Pulito | cartservice + 1 GET su redis-cart (redis e' velocissimo). Misura reale del pod. |
| **productcatalogservice** | Pulito | Go in-memory, nessuna dipendenza. Misura pulita. |
| **recommendationservice** | Parziale | `ListRecommendations` fa 1 chiamata a `productcatalog.ListProducts`. Potenziamo productcatalog a 4 repliche; la latenza dell'hop entra comunque nella misura. |
| **checkoutservice** | Fan-out | `PlaceOrder` orchestra 6 servizi (cart, catalog, currency, payment, shipping, email). Tutti potenziati a 4. Test con utenti casuali e **carrello vuoto**. |

### Nota sul carrello vuoto di checkoutservice

`k6_capacity_grpc.js` invia `PlaceOrder` con `user_id` casuali, quindi carrelli
vuoti. Nel demo Online Boutique `PlaceOrder` **funziona lo stesso** (ordine con
sola spedizione). Implicazioni:

- **Vantaggio**: test stateless, niente pre-seeding (e comunque `PlaceOrder`
  svuota il carrello a fine flusso: pre-popolare non reggerebbe un rate alto).
- **Limite**: con 0 articoli ci sono meno chiamate a `GetProduct` e `Convert`
  (una per articolo). Il fan-out misurato e' un po' piu' leggero di un ordine
  reale.
- **Perche' va bene**: il costo CPU di checkoutservice e' dominato
  dall'orchestrazione (7 chiamate gRPC, serializzazione, gestione errori), che
  e' circa costante e viene catturata. Valore = proxy ragionevole.

---

## 6. Metodologia: knee-point e SLA

1. **Step a rate fisso.** Per ogni valore di `DMOS_STEPS` k6 genera traffico a
   *quel* rate esatto con l'executor `constant-arrival-rate`.
2. **Settle + Measure.** Ogni step ha una fase `settle` (default 12s, scartata)
   e una `measure` (default 60s, registrata).
3. **Knee-point** = ultimo step in cui valgono **entrambe**:
   - `p95 < DMOS_SLA_P95_MS`
   - `error_rate < DMOS_SLA_ERR_RATE` (default 2%)
   Il test si ferma dopo 2 step consecutivi che violano lo SLA.
4. **Capacita' raccomandata = 80% del knee** (margine di sicurezza).
   `services.yaml` riceve `round(knee_rps x 0.80)`.

### SLA e step di default (per backend)

In-cluster la latenza e' bassa (~1-30 ms), quindi gli SLA assoluti hanno ampio
margine. Tutti i valori sono **sovrascrivibili** via env.

| Backend | SLA p95 default | Step default (rps) |
|---|---|---|
| cartservice | 50 ms | 25,50,75,100,150,200,300,400 |
| productcatalogservice | 50 ms | 50,100,200,300,400,500,600,800 |
| recommendationservice | 120 ms | 10,25,50,75,100,150,200,300 |
| checkoutservice | 500 ms | 5,10,20,30,40,50,75,100 |

Se il test esce con "knee N/A" (SLA gia' violato al primo step) o non raggiunge
mai la saturazione (tutti gli step OK), si rilancia con `DMOS_STEPS` adatti.

---

## 7. Uso: comandi e variabili env

Eseguire da **Git Bash** su Windows, dalla root del repo.

```bash
# Un solo backend (default: cartservice su ms-cluster-01)
bash experiments/run_capacity_grpc.sh

# Un backend specifico
bash experiments/run_capacity_grpc.sh --backend productcatalogservice
bash experiments/run_capacity_grpc.sh --backend recommendationservice
bash experiments/run_capacity_grpc.sh --backend checkoutservice

# Tutti e 4 in sequenza (consigliato per la calibrazione completa)
bash experiments/run_capacity_grpc.sh --all

# Su un cluster diverso
bash experiments/run_capacity_grpc.sh --backend cartservice --cluster ms-cluster-04

# Aiuto
bash experiments/run_capacity_grpc.sh --help
```

### Variabili env (override opzionali)

| Variabile | Default | Effetto |
|---|---|---|
| `DMOS_BACKEND` | cartservice | backend target (come `--backend`) |
| `DMOS_CLUSTER` | ms-cluster-01 | cluster target (come `--cluster`) |
| `DMOS_STEPS` | per-backend | lista rps degli step, es. `"50,100,150,200"` |
| `DMOS_STEP_DURATION_S` | 60 | durata fase measure per step |
| `DMOS_SETTLE_S` | 12 | durata fase settle per step |
| `DMOS_SLA_P95_MS` | per-backend | SLA p95 per il knee |
| `DMOS_SLA_ERR_RATE` | 0.02 | SLA error rate per il knee |
| `DMOS_K6_IMAGE` | grafana/k6:latest | immagine del Job k6 |
| `DMOS_NO_RESTORE` | 0 | `1` = non ripristina le dipendenze (debug) |
| `KUBECONFIG_DIR` | `$HOME/dmos-iac` | cartella dei kubeconfig |

Esempio — recommendationservice con range custom e SLA piu' largo:

```bash
DMOS_STEPS="20,40,60,80,100,120" DMOS_SLA_P95_MS=150 \
  bash experiments/run_capacity_grpc.sh --backend recommendationservice
```

### Durata

~9-10 min per backend (8 step x 72s), piu' il primo pull dell'immagine k6.
`--all` ~ 40 min.

### netem e' indifferente

Il `tc netem` simula latenze geografiche sul traffico **inter-cluster**. Il
traffico k6->backend e' **intra-cluster** (pod k6 e backend nello stesso
cluster): netem non lo tocca. Si puo' lasciare netem acceso o spento.

---

## 8. Output di esempio e interpretazione

```
================================================================================
  CAPACITY TEST gRPC — cartservice
  Metodo: hipstershop.CartService/GetCart
  Target: cartservice:7070
  SLA: p95 < 50ms, err < 2.0%
================================================================================

  Step | rps tgt | rps obt | rps/rep |   p50  |   p95  |   p99  | Err%  | SLA
  ----------------------------------------------------------------------------
     1 |    25r/s |    25.0 |    25.0 |     2ms |     4ms |     7ms |  0.0% |  OK
     2 |    50r/s |    50.0 |    50.0 |     2ms |     5ms |     9ms |  0.0% |  OK
     3 |    75r/s |    75.0 |    75.0 |     3ms |     7ms |    12ms |  0.0% |  OK
     4 |   100r/s |   100.0 |   100.0 |     4ms |    11ms |    20ms |  0.0% |  OK
     5 |   150r/s |   150.0 |   150.0 |     7ms |    26ms |    48ms |  0.0% |  OK
     6 |   200r/s |   199.3 |   199.3 |    19ms |    71ms |   140ms |  0.3% | FAIL
     7 |   300r/s |   268.0 |   268.0 |    55ms |   210ms |   390ms |  2.8% | FAIL

  -> SLA violato per 2 step consecutivi: saturazione confermata

================================================================================
  Knee-point:             150.0 rps/replica  (a 150 rps target)
  Capacita raccomandata:  120 rps/replica  (80% del knee)
  -> config/services.yaml, sezione cartservice:
     capacity_req_per_sec: 120
================================================================================
```

Come si legge:

- **p50 basso e crescente** (2 -> 19 ms): in-cluster non c'e' la tassa da 47 ms
  del port-forward; la latenza parte bassa e sale **gradualmente** verso il
  knee — il segno di una saturazione CPU **vera**.
- **rps tgt vs rps obt**: finche' coincidono il pod sta al passo. Quando
  `rps obt < rps tgt` (step 7) il pod non smaltisce piu' il rate -> saturo.
- **knee-point = 150**: ultimo step con SLA rispettato. **raccomandato = 120**.

Con `--all`, la tabella finale:

```
=== Riepilogo capacity backend gRPC ===
  Backend                  | knee (rps/rep)   | capacity_req_per_sec
  --------------------------------------------------------------------------
  cartservice              | 150.0            | 120
  productcatalogservice    | 400.0            | 320
  recommendationservice    | 220.0            | 176
  checkoutservice          | 60.0             | 48
```

> I numeri sopra sono **illustrativi**. I valori veri escono eseguendo il test.

### File prodotti

- `experiments/results/capacity_grpc_<backend>_<timestamp>.json` — summary
  machine-readable (estratto dai log del Job)
- `experiments/results/capacity_grpc_<backend>_<timestamp>.k6log.txt` — log
  completo del Job k6

---

## 9. Come aggiornare config/services.yaml

Lo script stampa, alla fine, lo snippet pronto. Per ogni backend, in
`config/services.yaml`, sotto `services: <backend>:`, sostituire il valore:

```yaml
  cartservice:
    ...
    # Capacita' per replica (req/s) — MISURATA con k6 gRPC in-cluster.
    # Test: experiments/run_capacity_grpc.sh --backend cartservice
    #   metodo hipstershop.CartService/GetCart, 1 replica isolata, Job k6
    #   dentro il cluster, knee = 150 rps/replica, raccomandato 80% = 120.
    capacity_req_per_sec: 120     # era 50 (stima)
```

### Verifica del budget CPU dopo l'aggiornamento

`capacity_req_per_sec` va riguardato insieme a `max_replicas`. Il modello DMOS
resta lineare **solo se** il worker ha CPU per tutte le repliche senza
throttling:

```
max_replicas x CPU_limit(backend) + altri_pod  <=  CPU_worker
```

Se la capacita' misurata e' molto piu' alta della stima, puo' avere senso
**abbassare** `max_replicas` (meno repliche per la stessa capacita' di
cluster). Vedi `dmos-iac/HETEROGENEOUS-CLUSTER-SETUP.md` sez. 4.4.

---

## 10. Capacity gRPC vs Capacity HTTP vs Throttle check

| | `run_capacity.sh` (HTTP) | `run_capacity_grpc.sh` (gRPC) | `throttle_check.sh` |
|---|---|---|---|
| **Domanda** | quante rps regge 1 replica frontend? | quante rps regge 1 replica di un backend? | N repliche scalano linearmente? |
| **Servizio** | frontend | i 4 backend gRPC | frontend (estendibile) |
| **Dove gira k6** | sul PC, via Ingress | **dentro il cluster** (Job) | sul PC, via Ingress |
| **Metrica chiave** | p95 latency lato client | p95 latency lato client (in-cluster) | CFS throttled periods |
| **Calibra** | `capacity_req_per_sec` frontend | `capacity_req_per_sec` backend | conferma `max_replicas` |
| **netem** | va spento (falsa la p95) | indifferente (traffico intra-cluster) | indifferente (misura CPU) |
| **Durata** | ~17 min | ~10 min/backend | ~4 min/cluster |

Insieme verificano i **due fattori** del modello di capacita' DMOS:
`run_capacity*` misura `capacity_req_per_sec`, `throttle_check` valida
`max_replicas`. Il prodotto e' la capacita' di cluster usata in `Phi_cap`.

---

## 11. Limiti noti e quando rieseguirlo

### Limiti noti

- **recommendation/checkout non perfettamente isolati**: vedi sez. 5. La misura
  include la latenza degli hop a valle (per design: e' cio' che il pod fa anche
  in produzione).
- **checkout a carrello vuoto**: vedi sez. 5.
- **Il pod k6 condivide il cluster col backend**: il generatore di carico gira
  sullo stesso cluster del backend sotto test. Su un cluster con margine (tier
  HIGH, 8 vCPU) non e' un problema: k6 e' limitato a 3 vCPU e il backend ha le
  sue requests garantite. Per questo il default e' `ms-cluster-01`.
- **Capacita' uniforme tra i tier**: il pod backend ha un CPU limit fisso
  (cartservice 300m, productcatalog 250m, checkout 500m, recommendation 500m),
  quindi `capacity_req_per_sec` non dipende dal tier del worker — stesso
  ragionamento del frontend. Basta misurare **una volta**, su un cluster
  qualsiasi.
- **Il proxy L7 Cilium e' nel percorso**: il traffico k6->backend passa da
  Envoy (le CNP `l7-visibility-*` parsano quelle porte come HTTP). E' il
  percorso reale di produzione, quindi corretto; ai rate del test (centinaia di
  rps) Envoy non e' un collo di bottiglia.

### Quando rieseguirlo

- Dopo ogni **re-provisioning** del fleet.
- Dopo aver cambiato il **CPU limit** di un backend in `config/services.yaml`.
- Dopo aver cambiato la **versione dell'immagine** di un microservizio.
- Quando cambia il **metodo gRPC rappresentativo** considerato.

### Posizione nella pipeline

Passo di **calibrazione**, una tantum dopo il re-provisioning, prima della fase
di traffico:

```
re-provision -> setup DMOS sui cluster -> verify-cilium -> netem + ping
   -> run_capacity.sh (frontend, netem OFF)
   -> run_capacity_grpc.sh --all (backend)        <-- QUESTO STRUMENTO
   -> aggiorna config/services.yaml
   -> throttle_check.sh
   -> riaccendi loadgenerator / fase di traffico -> lancio DMOS
```
