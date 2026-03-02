# DMOS — Come Misura il Traffico
## Guida completa alle sorgenti di misurazione

> Documento aggiornato il 01/03/2026.
> Riflette lo stato attuale dopo l'abilitazione di Hubble L7 via CNP con `rules: http`.
> Basato su `src/dmos_main.py` e `src/metrics/prometheus_client.py`.

---

## Indice

1. [Schema generale](#1-schema-generale)
2. [_get_per_cluster_traffic()](#2-_get_per_cluster_traffic)
3. [Catena di fallback per-cluster](#3-catena-di-fallback-per-cluster)
4. [Try 1 — Hubble L7 (sorgente primaria)](#4-try-1--hubble-l7-sorgente-primaria)
5. [Try 4 — Container network bytes (fallback)](#5-try-4--container-network-bytes-fallback)
6. [Perché DMOS mostra un valore diverso da Locust](#6-perché-dmos-mostra-un-valore-diverso-da-locust)
7. [Startup grace period](#7-startup-grace-period)
8. [Come viene usato il valore finale](#8-come-viene-usato-il-valore-finale)
9. [Cosa misura DMOS in pratica](#9-cosa-misura-dmos-in-pratica)
10. [Locust e DMOS — relazione attuale](#10-locust-e-dmos--relazione-attuale)
11. [Diagramma di flusso completo](#11-diagramma-di-flusso-completo)

---

## 1. Schema generale

DMOS misura quante **richieste al secondo** riceve il frontend in ogni cluster,
per decidere quante repliche allocare su ciascuno.

Dal refactoring (febbraio–marzo 2026), DMOS usa **Prometheus per-cluster** con
**Hubble L7 come sorgente primaria** e container network bytes come fallback:

```
DMOS._get_per_cluster_traffic("frontend")
    │
    ├─ cluster1: PrometheusClient(192.168.1.245:30090).get_request_rate()
    │    └─ Try 1: hubble_http_requests_total{destination_workload="frontend"} ✅
    ├─ cluster2: PrometheusClient(192.168.1.246:30090).get_request_rate()
    │    └─ Try 1: hubble_http_requests_total{destination_workload="frontend"} ✅
    └─ cluster3: PrometheusClient(192.168.1.247:30090).get_request_rate()
         └─ Try 1: hubble_http_requests_total{destination_workload="frontend"} ✅
              → Misura HTTP reale per-cluster (contatore esatto)
```

La metrica risultante viene esposta come:
```
dmos_actual_traffic{service="frontend"}             # su localhost:9090/metrics
dmos_predicted_traffic{cluster="clusterN", service} # predizione traffico futuro
```

---

## 2. _get_per_cluster_traffic()

```python
# src/dmos_main.py
def _get_per_cluster_traffic(self, service_name: str) -> Dict[str, float]:
    per_cluster: Dict[str, float] = {}
    for cname, prom in self.prom_map.items():
        rps = prom.get_request_rate(
            service=service_name,
            namespace="online-boutique"
        )
        per_cluster[cname] = max(0.0, rps or 0.0)

    total = sum(per_cluster.values())
    if total > 0:
        breakdown = ", ".join(f"{c}={v:.1f}" for c, v in per_cluster.items())
        logger.info(
            f"✅ Traffic per-cluster [{service_name}]: {breakdown} "
            f"| totale={total:.1f} req/s"
        )
    return per_cluster
```

**PROM_MAP** — un PrometheusClient per cluster:

| Cluster | Prometheus URL | Node |
|---------|---------------|------|
| cluster1 | `http://192.168.1.245:30090` | ms01 (EU-DE) |
| cluster2 | `http://192.168.1.246:30090` | ms02 (EU-FR) |
| cluster3 | `http://192.168.1.247:30090` | ms03 (EU-PL) |

Ogni Prometheus vede **solo i pod del suo cluster** → le metriche sono
automaticamente per-cluster, senza bisogno di label filter aggiuntivi.

---

## 3. Catena di fallback per-cluster

Ogni chiamata `prom.get_request_rate(service, namespace)` tenta 4 sorgenti nell'ordine:

```
get_request_rate(service="frontend", namespace="online-boutique")
    │
    ├─ Try 1: Hubble HTTP metrics        ✅ FUNZIONA — sorgente primaria per frontend
    ├─ Try 2: Istio metrics              ❌ Istio non installato (sempre vuoto)
    ├─ Try 3: HTTP generic metrics       ❌ Online Boutique non le espone (sempre vuoto)
    └─ Try 4: Container network bytes   ✅ Fallback finale per backend (cartservice, ecc.)
```

**Frontend**: usa Try 1 (Hubble L7) grazie alla CNP con `rules: http`.
**Backend services** (cartservice, productcatalog, checkoutservice, recommendationservice):
non hanno CNP L7, quindi cadono su Try 4 (network bytes). Questo è accettabile —
i backend ricevono gRPC interno dal frontend, la stima da bytes è sufficiente.

Log DMOS attesi:
```
✅ Traffic from Hubble (cluster1): 7.2 req/s    ← frontend (L7 esatto)
✅ Traffic from Hubble (cluster2): 6.3 req/s
✅ Traffic from Hubble (cluster3): 4.7 req/s
⚠️ Traffic from network (cluster1): 3840 B/s → 0.96 req/s   ← cartservice (stima)
```

---

## 4. Try 1 — Hubble L7 (sorgente primaria)

### Query PromQL

```python
# src/metrics/prometheus_client.py — get_request_rate() Try 1
query = (
    f'sum(rate(hubble_http_requests_total{{'
    f'destination_workload="{service}",'
    f'destination_namespace="{namespace}"'
    f'}}[5m]))'
)
```

### Label chiave

| Label | Valore | Descrizione |
|-------|--------|-------------|
| `destination_workload` | `"frontend"` | Nome del deployment Kubernetes di destinazione |
| `destination_namespace` | `"online-boutique"` | Namespace del pod destinazione |
| `reporter` | `"server"` | Il dato viene catturato lato Envoy del server |
| `source_workload` | `"controller"` (Nginx) | Pod sorgente della richiesta |

### Perché la finestra è `[5m]` e non `[1m]`

Hubble-metrics viene scraping da Prometheus ogni **~60 secondi**. La funzione `rate()`
di Prometheus richiede almeno **2 data point** nella finestra per calcolare la derivata.

Con `[1m]`:
- In finestra ci può essere solo 1 sample se il timing di scrape è sfavorevole
- `rate()` con 1 sample → `NaN` o `0` → fallback intermittente a network
- Questo causava inconsistenza tra cluster (cluster1 usava network, cluster3 Hubble)

Con `[5m]`:
- Ci sono sempre 4–5 sample nella finestra → `rate()` stabile e monotona
- Coerente su tutti e 3 i cluster, indipendentemente dal timing di scrape

```
Regola generale: window_rate ≥ 2 × scrape_interval
Con scrape_interval=60s: window ≥ 120s → [5m] (300s) è abbondante
```

### Ottimizzazione possibile: scrape_interval 15s → rate([1m])

Riducendo lo scrape interval da 60s a **15s**, la finestra `[1m]` garantisce sempre
4 sample → `rate([1m])` stabile. Il ritardo ramp-up scende da ~5 minuti a ~1 minuto.

#### Come fare la modifica

**Passo 1 — trovare il ConfigMap Prometheus su ogni cluster:**

```bash
# Verifica quale ConfigMap gestisce lo scraping
kubectl get configmap -n monitoring --context cluster1

# Con kube-prometheus-stack (Helm), il job Hubble è dentro:
# "prometheus-kube-prometheus-prometheus" (scrape_config generato da PodMonitor)
# Con Prometheus standalone, è nel ConfigMap "prometheus-server"
```

**Passo 2 — aggiornare il scrape_interval per il job hubble:**

```yaml
# Cerca il blocco job_name: 'hubble' o 'cilium-agent' e cambia:
scrape_configs:
  - job_name: 'hubble'
    scrape_interval: 15s    # ← cambia da 60s (default) a 15s
    static_configs:
      - targets: ['<cilium-pod-ip>:9965']
```

Con kube-prometheus-stack, la modifica va nel `PodMonitor` o nel `values.yaml` Helm:
```yaml
# Se esiste un PodMonitor per Hubble:
kubectl patch podmonitor hubble-metrics -n monitoring \
  --type='json' \
  -p='[{"op":"replace","path":"/spec/podMetricsEndpoints/0/interval","value":"15s"}]'
```

**Passo 3 — aggiornare la query in DMOS:**

```python
# src/metrics/prometheus_client.py — Try 1 Hubble
query = (
    f'sum(rate(hubble_http_requests_total{{'
    f'destination_workload="{service}",'
    f'destination_namespace="{namespace}"'
    f'}}[1m]))'   # ← da [5m] a [1m]
)
```

**Verifica** (dopo il restart di Prometheus e DMOS):
```bash
# Deve restituire un rate con sample ogni 15s invece di 60s
curl "http://192.168.1.245:30090/api/v1/query_range?query=hubble_http_requests_total&step=15s" | jq .
```

#### Pro e contro

| | Configurazione attuale | Con scrape 15s + rate([1m]) |
|---|---|---|
| **Ritardo ramp-up** | ~5 minuti | ~1 minuto |
| **Flash crowd detection** | ❌ 5 min di lag (critico) | ✅ ~1 min di lag (accettabile) |
| **Stabilità rate()** | ⭐⭐⭐ Alta (4-5 sample su 5 min) | ⭐⭐ Media (4 sample su 1 min, più jitter) |
| **CPU nodo k3s** | Baseline | +~0.1% (3 scrape extra/min, ciascuno ~10ms) |
| **Storage TSDB** | Baseline | +4x campioni Hubble (~600 KB/giorno) |
| **Modifiche codice** | — | 1 riga in `prometheus_client.py` |
| **Modifiche cluster** | — | 1 riga per ConfigMap/PodMonitor su 3 cluster |

**Pro principali:**
- Il ritardo di 5 minuti è il principale limite per testare scenari a carico variabile rapido
- Nessuna modifica architetturale — solo parametri di configurazione
- Il costo in CPU e storage è trascurabile sul lab k3s (bassa cardinality, ~100 serie Hubble attive)

**Contro principali:**
- `rate([1m])` cattura transitori brevi (burst di probe, spike momentanei di 15-30s) che `rate([5m])` avrebbe assorbito. In pratica, il jitter aggiuntivo è gestito dai guard esistenti: dead zone 15%, scale-up protection 120s, max_delta_per_cycle=4
- Se un nodo è sotto pressione CPU, uno scrape fallito pesa 25% del rate con [1m] vs 5-10% con [5m]. Con il nostro setup a bassa cardinality, questo è improbabile

**Raccomandazione per il lab:**
Applicare la modifica prima di eseguire il test `flash_crowd`. Per `double_wave` (rampa lenta)
il beneficio è minore ma la modifica non causa problemi — `rate([1m])` è comunque stabile
con 15s scrape.

### Perché funziona (Hubble L7 same-node)

La precedente documentazione indicava una "limitazione same-node" di Cilium L7.
Questa si è rivelata un'errata diagnosi. Il vero problema era:

1. **Il traffico `world`** (Locust → NodePort 30007) passava per Envoy e andava in timeout
   perché Envoy non gestisce correttamente connessioni da indirizzi `world` su NodePort
2. **`fromEndpoints: {}` è namespace-scoped**: bloccava Nginx (in `ingress-nginx`)
   già a L3, prima che arrivasse a L7

Con la CNP corretta che include `rules: http:` (vedi §4 di `cilium-hubble-l7-visibility.md`):
- Il traffico `world` è bloccato a L3 dalla CNP (non arriva a Envoy) ✅
- Il traffico `cluster` (Nginx → Frontend) arriva a Envoy → L7 parsing → Hubble conta ✅
- Il traffico same-node pod-to-pod con sorgente `cluster` entity funziona con Envoy ✅

**Verifica**: `hubble observe --type l7 -n online-boutique` mostra un flusso continuo di
eventi come `controller (ingress-nginx) → frontend:8080 HTTP/1.1 200 GET /`.

### Effetto ramp-up: ritardo ~5 minuti all'inizio del test

**Comportamento osservato**: nei primi ~5 minuti dopo l'avvio di Locust, il valore
`dmos_actual_traffic` (chiamato **FE** nel monitor) è significativamente più basso del
traffico reale misurato da Locust.

**Causa**: la finestra `rate([5m])` include campioni sia del periodo idle pre-Locust
che del periodo a piena carico. I due periodi si "mescolano" nella media:

```
Prima di Locust (idle):   ~0.6 rps  ← kubelet liveness/readiness probes
Con Locust (carico):      ~34 rps   ← traffico reale
```

Esempio osservato al minuto 1 del test `double_wave_hubble`:
```
FE (dmos_actual_traffic): 0.6 rps   ← finestra [5m] quasi tutta idle
Locust rps (reale):       9.7 rps   ← già a piena velocità (10s rolling avg)
```

Dopo circa 5 minuti di Locust attivo, la finestra [5m] è interamente occupata da
campioni di carico → FE converge al valore reale × 1.43 (vedi §6 per il moltiplicatore).

**Da dove viene 0.6 rps in idle**: Kubernetes esegue liveness e readiness probe
verso il pod frontend ogni ~10s. Con 3 cluster (1 pod/cluster in idle), questo genera
~3 probe / 10s = ~0.3–0.6 rps che Hubble/Envoy conta come `hubble_http_requests_total`.
Questo è normale e atteso — è il "rumore di fondo" del sistema.

**Impatto sullo scaling DMOS durante il ramp-up**:
- Nei primi ~5 min: FE basso → il predictor vede tuttavia una derivata crescente (positivo)
- La **scale-up protection (120s)** è già attiva → impedisce oscillazioni precoci
- Lo scaling avviene comunque, ma potenzialmente 1–2 cicli da 30s più tardi

**Perché non è un bug critico**: il `TrafficPredictor` usa la derivata del traffico
oltre al valore assoluto. Anche con FE=0.6 rps, se la derivata è alta
(traffico che sale velocemente), il predictor anticipa il picco e triggera lo scaling.
La scale-up protection garantisce che il pod scalato a t+30s non venga rimosso a t+60s.

**Fix futuro**: ridurre `scrape_interval` di Prometheus per Hubble a 15s →
la query può usare `rate([1m])` (4 sample garantiti) → ritardo ramp-up ridotto a ~1 minuto.

### Condizioni necessarie

Per avere `✅ Traffic from Hubble`:
1. **CNP con `rules: http`** applicata sul pod frontend (porta 8080)
2. **Nginx Ingress** attivo su porta 30080 (tutto il traffico esterno passa da Nginx)
3. **Hubble** attivo nel cluster (installato automaticamente con Cilium)
4. **Hubble-metrics** esportato su Prometheus (scraping configurato)
5. **Finestra `[5m]`** nella query → almeno 5 minuti di traffico attivo per avere sample stabili

---

## 5. Try 4 — Container network bytes (fallback)

### Query PromQL

```promql
sum(rate(container_network_receive_bytes_total{
    namespace="online-boutique",
    pod=~"{service}.*"
}[1m]))
```

### Conversione bytes → req/s

```python
estimated_rps = bytes_per_sec / 4000
```

La costante `4000` è una calibrazione: una richiesta HTTP media verso il
frontend pesa circa 4 KB di traffico in ricezione sul pod (headers + body).

### Precisione

| Causa di imprecisione | Effetto |
|----------------------|---------|
| Dimensione richieste variabile (600 B GET / – 3 KB POST) | ±30% errore relativo |
| Traffico gRPC in ingresso dai backend | Sovrastima leggera |
| Finestra temporale `[1m]` con scrape variable | ±5–10% jitter |

La safety margin del 15% nel ReplicaScaler compensa le sottostime sistematiche.

---

## 6. Perché DMOS mostra un valore diverso da Locust

### Con Hubble L7 attivo (frontend) — moltiplicatore ~1.4×

Con Hubble, DMOS conta **tutte le richieste HTTP** intercettate da Envoy, incluse le
**sub-request** generate da ogni "transazione" Locust.

| Fonte | Misura | Perché differiscono |
|-------|--------|---------------------|
| Locust `current_rps` | Transazioni al secondo (es. `GET /`) | Una sola richiesta per task |
| Hubble `hubble_http_requests_total` | Richieste HTTP al pod Envoy | Include assets: JS, CSS, immagini, product calls, recommend calls... |

**Esempio osservato nel primo esperimento `double_wave_hubble`**:
```
Locust:  24.1 rps (300 utenti × 0.08 task/s)
Hubble:  34.4 rps (3 cluster × ~11.5 rps/cluster)
Ratio:   34.4 / 24.1 ≈ 1.43×
```

**Questo comportamento è corretto e atteso**: ogni visita a una pagina genera
~1.4 richieste HTTP al pod frontend (homepage carica anche immagini, API calls
per recommendations, product listings, ecc.).

Il moltiplicatore è **stabile** nel tempo (dipende dal profilo navigazione Locust,
non dalla load). Non è un errore di misura — Hubble misura più accuratamente il
carico reale sul pod frontend rispetto alle transazioni Locust.

**Nel monitor**:
```
FE: 11.5rps→12.3pred  replicas=3(c1/c2/c3=1/1/1) | p95=85ms rps=24.1 users=300 fail=0.0%
    ↑ DMOS Hubble cluster1 (1 dei 3)                 ↑ Locust totale (tutte le transazioni)
```
Totale DMOS: 11.5 × 3 ≈ 34.5 vs Locust 24.1 → ratio 1.43× ✅ (non un bug)

**Implicazione per la capacity**: `config/services.yaml` usa `capacity_req_per_sec: 30`
calibrato su carico Hubble, non su transazioni Locust. Corretto così.

### Con fallback network bytes (backend services)

Il valore DMOS è sistematicamente più basso rispetto al carico reale perché:
- La costante `/4000` stima la dimensione media HTTP, non la dimensione reale gRPC
- Le richieste gRPC interne hanno dimensioni variabili (tipicamente 200–2000 byte)

---

## 7. Startup grace period

### Comportamento

Nei primi **90 secondi** dall'avvio di DMOS, il sistema entra in una **grace period**
durante la quale:

| Fase | Comportamento |
|------|---------------|
| **Lettura traffico** | ✅ Normale — `get_request_rate()` interroga Prometheus ogni ciclo |
| **Accumulo predittore** | ✅ Normale — TrafficPredictor accumula storia per cluster |
| **Scaling k8s** | ❌ Bloccato — nessuna chiamata a `kubectl scale` |
| **`dmos_current_replicas`** | ✅ Pubblicata — riflette le repliche reali k8s (min=1 per cluster) |
| **`dmos_predicted_traffic`** | ❌ Non pubblicata — nessuna predizione durante grace period |
| **Backend reset** | ❌ Saltato — nessun reset dei backend ai minimi |

### Motivazione tecnica

```python
# src/dmos_main.py
self.startup_grace_seconds = 90

# In schedule_service():
elapsed_startup = (datetime.now() - self.startup_time).total_seconds()
if elapsed_startup < self.startup_grace_seconds:
    logger.info(
        f"⏳ Grace period: {elapsed_startup:.0f}/{self.startup_grace_seconds}s "
        f"— traffic observed, no k8s ops"
    )
    return
```

La grace period evita che DMOS inizi a scalare prima che il predittore abbia
abbastanza storia. Senza di essa:
- Il predittore userebbe derivate calcolate su 1–2 punti → molto rumorose
- DMOS potrebbe fare scale-up aggressivi nei primi cicli (falsi positivi)
- Possibile conflitto con Locust che sta ancora inizializzando le sessioni

### Implicazioni per collect_metrics

Durante la grace period, `collect_metrics_simple.py` registra:
- `replicas = 1 per cluster` (gauge pubblicata: riflette min_replicas k8s reali)
- `predicted_traffic = 0.0` (gauge non pubblicata — nessuna predizione)
- `actual_traffic` è presente (DMOS continua a misurarlo, ~0.6 rps da kubelet probe)

**Per la valutazione**: i primi 90s sono il **periodo di warm-up**. In post-processing
puoi escluderli filtrando i record dove `predicted_traffic == 0.0`.

### Ordine di avvio consigliato

Avviare **sempre DMOS prima di Locust** per non sprecare test durante la grace period.

```
t=0s    → Avvia DMOS main (grace period con traffico idle ≈ 0)
t=2s    → Avvia collect_metrics (registra dall'inizio)
t=90s   → Grace period termina — DMOS pronto a scalare
t=90s   → Avvia Locust --autostart (test inizia ora, nessun secondo sprecato)
```

**Perché questo ordine**:
- ✅ DMOS accumula 90s di storia su traffico idle → baseline predictor pulita
- ✅ Quando Locust parte, la derivata sale subito → proactive scaling immediato
- ✅ Il test (es. 26 min double_wave) è tutto in fase "attiva"
- ❌ Se Locust parte prima: i primi 90s a min_replicas=1 → p95 alto, falluires, test sprecato

**Ordine sbagliato da evitare**:
```
t=0s    → Locust --autostart   ← SBAGLIATO: carica il sistema prima che DMOS possa scalare
t=5s    → DMOS (grace 90s)     ← DMOS non scala → 90s di sistema sovraccarico
```

---

## 8. Come viene usato il valore finale

```python
# src/dmos_main.py — periodic_check_thread()
per_cluster_traffic = self._get_per_cluster_traffic(service_name)
current_traffic = sum(per_cluster_traffic.values())

# [Traffic floor] Se il traffico è quasi zero, bypassare il predictor.
# Il predictor EMA decade lentamente: dopo la fine del test mantiene
# un'EMA elevata (es. 21.6 rps) per 15+ minuti, tenendo le repliche attive.
# Con il traffic floor: se actual < 2 rps, si usa actual invece di max(actual, pred).
TRAFFIC_FLOOR_RPS = 2.0
if 0 < current_traffic < TRAFFIC_FLOOR_RPS:
    effective_traffic = current_traffic      # bypass predictor → < low_threshold → scale-down
else:
    effective_traffic = max(current_traffic, predicted_traffic)  # usa il massimo (proattivo)

# Aggiorna la metrica DMOS
self.actual_traffic.labels(service=service_name).set(current_traffic)

# In schedule_service(): per ogni cluster usa il traffico reale (non stima proporzionale)
cluster_traffic = per_cluster_traffic.get(cluster_name, 0.0)

# Calcola repliche con predictor + PD controller per quel cluster
decision = self.scalers[service_name][cluster_name].compute_target_replicas(
    current_replicas=current_reps,
    current_traffic=cluster_traffic    # dato reale per-cluster da Hubble
)
```

### Formula di scaling (semplificata)

```
base_replicas    = predicted_traffic / capacity_per_replica
safety_replicas  = ceil(base_replicas × 1.15)          # safety margin 15%
pd_adjustment    = Kp × error + Kd × derivative         # Kp=5.0, Kd=300.0
target_replicas  = clamp(safety_replicas + pd_adj, min=1, max=20)
```

`capacity_per_replica` da `config/services.yaml`:
```yaml
frontend:
  capacity_req_per_sec: 30   # ogni replica gestisce max 30 rps
```

### Scalers per-cluster indipendenti

DMOS ha un `ReplicaScaler` separato per ogni coppia `(service, cluster)`:
- TrafficPredictor con storia separata per cluster
- PD controller con stato indipendente
- Se cluster2 (FR) riceve più traffico per il carbon-aware scheduling, il suo
  predictor si adatta senza influenzare cluster1 e cluster3

---

## 9. Cosa misura DMOS in pratica

### Scenario A — Frontend con Hubble L7 (stato attuale) ✅

**Condizioni**: CNP con `rules: http` + Nginx Ingress su 30080.

**Cosa misura**: `hubble_http_requests_total` — contatore incrementato da Envoy
ogni volta che una richiesta HTTP attraversa la proxy chain Nginx→Frontend.

**Precisione**: ⭐⭐⭐ Alta — contatore HTTP esatto, uguale a Locust ±5%.

**Log DMOS**:
```
✅ Traffic from Hubble (cluster1): 7.2 req/s
✅ Traffic from Hubble (cluster2): 6.3 req/s
✅ Traffic from Hubble (cluster3): 4.7 req/s
✅ Traffic per-cluster [frontend]: cluster1=7.2, cluster2=6.3, cluster3=4.7 | totale=18.2 req/s
```

### Scenario B — Backend services (cartservice, productcatalog, ecc.)

**Condizioni**: no CNP L7 sui backend → nessun `hubble_http_requests_total` per gRPC.

**Cosa misura**: `container_network_receive_bytes_total / 4000` — stima da bytes
di rete ricevuti dal pod backend.

**Precisione**: ⭐⭐ Media — gRPC ha dimensioni variabili, stima ±20-30%.

**Log DMOS**:
```
⚠️ Traffic from network (cluster1): 3840 B/s → 0.96 req/s
```

### Scenario C — IDLE (nessun test, solo health check)

**Cosa misura**: traffico quasi zero (~0.1 rps da kubelet probe).
**Conseguenza**: DMOS mantiene min_replicas (1 per cluster).

---

## 10. Locust e DMOS — relazione attuale

Locust e DMOS sono **indipendenti**. DMOS non chiama la Locust API per misurare il traffico.

| Componente | Porta | Ruolo |
|------------|-------|-------|
| Locust (web UI + stats) | `localhost:8089` | Genera traffico HTTP sui cluster |
| DMOS (metrics server) | `localhost:9090` | Espone metriche di scaling |
| DMOS (webhook server) | `localhost:8081` | Riceve alert da Prometheus |

Locust genera carico su `http://192.168.1.X:30080` (Nginx).
DMOS misura quel carico via Hubble su Prometheus (`:30090`).

**`collect_metrics_simple.py`** legge entrambi:
- Stats Locust da `:8089/stats/requests` → `rps`, `users`, `p95` Locust-side
- Metriche DMOS da `:9090/metrics` → `dmos_actual_traffic`, repliche per cluster

```
FE: 7.2rps→8.1pred  replicas=3(c1/c2/c3=1/1/1) | p95=85ms rps=21.0 users=100 fail=0.0%
    ↑ DMOS Hubble cluster1                           ↑ Locust totale tutti i cluster
```

I due valori **non coincidono**: DMOS mostra il valore per-cluster (cluster1),
Locust mostra il totale. Per confrontarli: DMOS_totale ≈ Σ cluster ≈ Locust_total.

---

## 11. Diagramma di flusso completo

```
┌─────────────────────────────────────────────────────────────────────┐
│              DMOS — ogni ciclo (~30s)                               │
└─────────────────────────────────────────────────────────────────────┘
                          │
              [t < 90s → grace period: solo osserva]
                          │
                          ▼
            _get_per_cluster_traffic("frontend")
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   cluster1:30090  cluster2:30090  cluster3:30090
   get_request_rate("frontend", "online-boutique")
          │
     ┌────▼──────────────────────────────────────────────────────────┐
     │ Try 1: Hubble L7 ✅ (ATTIVO per frontend)                     │
     │  sum(rate(hubble_http_requests_total{                         │
     │      destination_workload="frontend",                         │
     │      destination_namespace="online-boutique"                  │
     │  }[5m]))                                                      │
     │  → rps esatto (Envoy intercetta ogni richiesta HTTP)          │
     │  → log: ✅ Traffic from Hubble (clusterN): X.X req/s          │
     ├───────────────────────────────────────────────────────────────┤
     │ Try 2: Istio ❌ (non installato)                              │
     ├───────────────────────────────────────────────────────────────┤
     │ Try 3: http_requests_total ❌ (app non espone)                │
     ├───────────────────────────────────────────────────────────────┤
     │ Try 4: Network bytes ✅ (fallback per backend)                │
     │  sum(rate(container_network_receive_bytes_total{              │
     │      namespace="online-boutique", pod=~"frontend.*"           │
     │  }[1m])) / 4000                                              │
     │  → log: ⚠️ Traffic from network (clusterN): X B/s → Y req/s  │
     └──────────────────────────┬────────────────────────────────────┘
                                │
           per_cluster = {c1: rps1, c2: rps2, c3: rps3}
                                │
           current_traffic = rps1 + rps2 + rps3
                                │
     ┌──────────────────────────▼──────────────────┐
     │  dmos_actual_traffic{service="frontend"}     │
     └──────────────────────────┬──────────────────┘
                                │
           ┌────────────────────▼──────────────────────────────┐
           │ Traffic floor check (periodic_check_thread)        │
           │ if 0 < current < 2.0 rps:                          │
           │     effective = current  ← bypass predictor        │
           │ else:                                              │
           │     effective = max(current, predicted)            │
           │     (proactive: usa il valore più alto)            │
           └────────────────────┬──────────────────────────────┘
                                │
     Per ogni cluster (scheduling post grace period):
     ┌──────────────────────────▼──────────────────┐
     │  ReplicaScaler[cluster_N]                    │
     │  predict(cluster_traffic_N) → TrafficPredictor│
     │  PD_controller.compute(error, derivative)    │
     │  → target_replicas                           │
     └──────────────────────────┬──────────────────┘
                                │
     ┌──────────────────────────▼──────────────────┐
     │  DMOSScheduler (Level 1)                     │
     │  score(lat, cap, load, carbon) per cluster   │
     │  WinnerDetermination → allocations           │
     └──────────────────────────┬──────────────────┘
                                │
     ┌──────────────────────────▼──────────────────┐
     │  Anti-oscillation guards (_can_scale_down)   │
     │  Guard 1: scale_down_cooldown=60s            │
     │      → no scale-down per 60s dopo scale-down │
     │  Guard 2: scale_up_protection=120s           │
     │      → no scale-down per 120s dopo scale-up  │
     │  (previene: scala su → scala giù subito)     │
     └──────────────────────────┬──────────────────┘
                                │
     k8s.scale_deployment(cluster_N, target_N)
```

---

## Riepilogo rapido

| Servizio | Sorgente usata | Precisione | Note |
|----------|---------------|------------|------|
| `frontend` | **Hubble L7** (Try 1) | ⭐⭐⭐ Alta | Richiede CNP con `rules: http` + Nginx 30080 |
| `cartservice` | Network bytes (Try 4) | ⭐⭐ Media | No CNP L7 per gRPC |
| `productcatalogservice` | Network bytes (Try 4) | ⭐⭐ Media | Idem |
| `checkoutservice` | Network bytes (Try 4) | ⭐⭐ Media | Idem |
| `recommendationservice` | Network bytes (Try 4) | ⭐⭐ Media | Idem |

**Hotpath della query Hubble**:
```
rate([5m]) → stabilità vs scrape_interval 60s (4-5 campioni garantiti)
destination_workload → filtro per-servizio (evita di sommare tutto il namespace)
destination_namespace → filtro per namespace (isolamento multi-tenant)
```
