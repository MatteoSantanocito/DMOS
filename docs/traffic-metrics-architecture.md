# Architettura delle Metriche di Traffico: Φ_demand e il Ruolo di Nginx Ingress

## Contesto e Motivazione

Il sistema DMOS misura il traffico per prendere due tipi di decisioni ortogonali:

| Livello | Domanda | Metrica chiave | Uso |
|---------|---------|----------------|-----|
| Level 1 (WHERE) | Su quale cluster mettere le repliche? | `nginx_ingress_controller_requests` | Score Φ_demand → cluster selection |
| Level 2 (HOW MANY) | Quante repliche servono in quel cluster? | `hubble_http_requests_total{destination_workload="frontend"}` | Autoscaling PD controller |

Queste due metriche misurano aspetti complementari dello stesso traffico e non sono intercambiabili.

---

## Separazione dei Ruoli: Routing vs Scaling

Una distinzione fondamentale per comprendere il sistema è che **DMOS non controlla il routing del traffico**. Il traffico viene instradato da entità esterne, e DMOS osserva questo routing come dato di fatto per regolare le repliche.

### Chi controlla cosa

```
┌─ ROUTING (esterno a DMOS) ─────────────────────────────────────────┐
│  Esperimento: Locust weights (ClusterNUser con host fisso per IP)   │
│  Produzione:  GeoDNS / Anycast BGP / Global Load Balancer           │
│                                                                     │
│  Decisione: quale cluster IP raggiunge ogni utente                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ traffico già instradato
┌─ SCALING (DMOS) ───────────────────────────────────────────────────┐
│  Level 1: su quale cluster allocare le repliche (WHERE)             │
│  Level 2: quante repliche servono in quel cluster (HOW MANY)        │
│                                                                     │
│  Decisione: numero di pod per cluster per servire il traffico       │
└─────────────────────────────────────────────────────────────────────┘
```

### Il ruolo di Cilium Cluster Mesh

Cilium Cluster Mesh non è un router per il traffico degli utenti. Gestisce la **comunicazione pod-to-pod cross-cluster**: un microservizio su cluster1 che deve chiamare un microservizio su cluster2. Questa è comunicazione interna all'applicazione (es. frontend → cartservice → checkoutservice), non l'ingresso degli utenti.

```
Utente → Nginx cluster1 → frontend pod cluster1
                               │
                               └─(Cilium mesh)─→ cartservice cluster2
                                                        │
                               ┌─(Cilium mesh)─→ checkoutservice cluster1
```

Hubble osserva tutto questo traffico interno e genera le metriche `destination_workload`. Ma la decisione iniziale di quale Nginx raggiunge l'utente è presa da Locust (o GeoDNS in produzione), non da Cilium.

---

## Le Tre Metriche di Traffico

### 1. `hubble_http_requests_total{destination_workload="frontend"}` (Level 2)

Questa metrica misura le **richieste che arrivano ai pod frontend** nel cluster locale.

```
Locust → Nginx (NodePort 30080) → [Envoy L7] → frontend pod
                                                    ↑
                                          Hubble conta qui
```

Con Cilium cluster mesh, un pod frontend può ricevere traffico anche da un altro cluster (via mesh). Quindi questa metrica misura il **carico effettivo dei pod**, indipendentemente da dove sono arrivati gli utenti originariamente.

**Uso in DMOS:** Level 2 — stima il numero di repliche necessarie in quel cluster per soddisfare la domanda corrente (`get_request_rate()` in `prometheus_client.py`).

### 2. `hubble_http_requests_total{source="reserved:ingress"}` (approccio Romano)

Nella tesi di Romano (senza Nginx Ingress), il traffico arriva direttamente dall'esterno al pod frontend. Hubble vede il source come `reserved:ingress` (traffico dal world, identità Cilium per il traffico esterno non-pod).

```
Locust → [world] → frontend pod (port 30007 NodePort)
          ↑
    source="reserved:ingress"
    Hubble conta qui
```

Questa metrica misura quanti utenti **bussano alla porta del cluster**, indipendentemente da dove vengono poi serviti. È una misura della **domanda geografica** — quanti utenti sono stati instradati verso quel cluster.

**Limitazione con Nginx:** quando si introduce Nginx Ingress come proxy, il traffico esterno colpisce il pod Nginx (non il frontend direttamente). Hubble vede il source verso il frontend come il pod Nginx (un'identità interna, non `reserved:ingress`). La metrica `source="reserved:ingress"` non misura più il traffico frontend.

### 3. `nginx_ingress_controller_requests{ingress="frontend-ingress"}` (approccio DMOS con Nginx)

Questa è l'**equivalente funzionale** di `source="reserved:ingress"` nell'architettura con Nginx Ingress.

```
Locust → Nginx pod (NodePort 30080) → frontend pod
              ↑
  nginx_ingress_controller_requests conta qui
```

Nginx Ingress Controller espone questa metrica per ogni risorsa Ingress configurata. Conta esattamente le richieste HTTP che entrano nel cluster dall'esterno e vengono gestite dall'Ingress `frontend-ingress`.

**Proprietà fondamentale:** questa metrica è **per-cluster** per costruzione — ogni cluster ha il proprio Nginx Ingress Controller che conta solo il traffico che lo attraversa. Non risente del cluster mesh: anche se poi il traffico viene servito da un pod in un altro cluster, questa metrica ha già contato la richiesta come "arrivata in questo cluster".

---

## Come Nginx Cambia il Flow di Traffico

### Architettura senza Nginx (Romano)

```
Utente (da Francoforte)
    │
    ├─→ cluster1:30007 ──→ frontend pod (diritto, NodePort)
    │                       source=reserved:ingress ✓ Hubble vede
    │
    └─→ cluster2:30007 ──→ frontend pod
                            source=reserved:ingress ✓
```

Hubble L7 non funziona su questa path perché Cilium Envoy non riesce a gestire DNAT per NodePort esterno: il traffico bypassa la CNP L7, Envoy non viene ingaggiato, Hubble non vede i dettagli HTTP.

### Architettura con Nginx Ingress (DMOS)

```
Utente (da Francoforte)
    │
    ├─→ cluster1:30080 ──→ [Nginx pod] ──→ [Envoy L7 ✓] ──→ frontend pod
    │                          ↑                                    ↑
    │              nginx_ingress_controller_requests        hubble_http_requests_total
    │              (conta qui: domanda geografica)          (conta qui: carico pod)
    │
    └─→ cluster2:30080 ──→ [Nginx pod] ──→ [Envoy L7 ✓] ──→ frontend pod
```

Con Nginx:
- Locust colpisce Nginx via NodePort (L4, nessun problema DNAT)
- Nginx fa proxy interno al frontend (pod-to-pod, Hubble L7 funziona perché è traffico intra-cluster)
- La CNP con `fromEndpoints: {}` + `rules: http: [{}]` abilita Envoy solo per il traffico pod-to-pod
- `nginx_ingress_controller_requests` cattura la domanda geografica (quante richieste entrano in questo cluster)
- `hubble_http_requests_total{destination_workload="frontend"}` cattura il carico effettivo dei pod

---

## Φ_demand: La Componente di Domanda Geografica

### Motivazione

Il sistema di scoring Level 1 deve rispondere alla domanda: **dove sono gli utenti?**

Le componenti esistenti Φ_net (RTT inter-cluster) e Φ_lat (latenza osservata) catturano indirettamente la posizione geografica. Φ_demand la misura direttamente: il cluster che riceve più traffico in ingresso è quello più vicino agli utenti (o più preferito dal load balancer), e deve ricevere più repliche.

Questo è particolarmente utile quando:
- Il traffico non è distribuito uniformemente tra i cluster (flash crowd su un cluster specifico)
- La distribuzione del traffico cambia nel tempo (peak hours geografiche)
- Il routing degli utenti non corrisponde perfettamente alla RTT (CDN, anycast routing)

### Formula

```
Φ_demand(i) = λ_ingress(i) / Σ_j λ_ingress(j)
```

dove:
- `λ_ingress(i)`: rate delle richieste in ingresso nel cluster i (da `nginx_ingress_controller_requests`)
- `Σ_j λ_ingress(j)`: traffico totale su tutti i cluster
- Il risultato è in `[0, 1]`, con `Σ_i Φ_demand(i) = 1`

**Caso degenerato:** se `nginx_ingress_controller_requests` non è disponibile (Nginx non deployato), Φ_demand = `1/N` per tutti i cluster (distribuzione uniforme, nessun effetto sullo score).

### Proprietà

| Scenario | Φ_demand(cluster1) | Φ_demand(cluster2) | Φ_demand(cluster3) |
|----------|---------------------|---------------------|---------------------|
| Traffico uniforme (33% ciascuno) | 0.33 | 0.33 | 0.33 |
| Flash crowd su cluster1 (60-20-20) | 0.60 | 0.20 | 0.20 |
| Tutto su cluster2 | 0.00 | 1.00 | 0.00 |

**Conseguenza sull'allocazione:** se cluster1 riceve il 60% del traffico, Φ_demand(1)=0.60. Con `ω_demand=0.10`, il suo score cresce di 0.06 rispetto al caso uniforme → più repliche allocate su cluster1 → riduzione della latenza per gli utenti già lì.

### Implementazione

**Query PromQL** (per-cluster, interrogata sul Prometheus locale di ogni cluster):

```promql
sum(rate(nginx_ingress_controller_requests{
    ingress="frontend-ingress",
    namespace="online-boutique"
}[1m]))
```

**Normalizzazione** (in `dmos_scheduler.py`, `collect_scores()`):

```python
# Raccolta pre-score: tasso ingress per ogni cluster
ingress_rates = {
    name: prom.get_ingress_rate(namespace)
    for name, prom in self.prom_map.items()
}
total_ingress = sum(ingress_rates.values())

# Normalizzazione → Φ_demand
n = len(ingress_rates)
demand_shares = {
    name: (rate / total_ingress if total_ingress > 0 else 1.0 / n)
    for name, rate in ingress_rates.items()
}
```

La normalizzazione avviene **prima** del calcolo degli score individuali, perché richiede il totale su tutti i cluster.

### Perché la pre-raccolta è necessaria solo per Φ_demand

Tutte le altre componenti di score normalizzano contro **costanti fisse definite in config**, note a priori:

| Score | Denominatore della normalizzazione | Noto prima del loop? |
|-------|-----------------------------------|----------------------|
| Φ_lat | η, σ² (parametri config) | ✅ Sì |
| Φ_cap | `cpu_total`, `mem_total` (config per-cluster) | ✅ Sì |
| Φ_load | `request_rate_max = cpu_total × capacity_rps` (config) | ✅ Sì |
| Φ_carbon | `CI_max = 800 gCO₂/kWh` (costante config) | ✅ Sì |
| Φ_net | `RTT_max = 500ms` (costante config) | ✅ Sì |
| **Φ_demand** | `Σ_j λ_ingress(j)` — **somma runtime di tutti i cluster** | ❌ No |

Per calcolare Φ_lat di cluster1, basta interrogare Prometheus di cluster1. Il denominatore è scritto nel config e non dipende dagli altri cluster.

Per calcolare Φ_demand di cluster1, **bisogna prima aver interrogato cluster2 e cluster3**, perché il denominatore emerge solo dopo aver aggregato tutti:

```python
# Φ_carbon: calcolabile dentro il loop, cluster per cluster
phi_carbon = exp(-ν * CI_cluster1 / 800)   # 800 è fisso in config

# Φ_demand: NON calcolabile dentro il loop senza un passaggio precedente
phi_demand = ingress_cluster1 / (ingress_c1 + ingress_c2 + ingress_c3)
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                questo aggregato è noto solo dopo il giro su tutti
```

Questo giustifica la struttura a due passaggi in `collect_scores()`: prima si raccolgono tutti gli ingress rates, poi si calcola lo score per ogni cluster.

---

## Il Loop di Controllo: Traffico Passato, Repliche Future

DMOS è un **controller reattivo** (con componente predittiva): osserva il traffico già avvenuto e aggiusta le repliche per il traffico futuro. Le metriche misurate sono sempre riferite al passato recente (finestre `[1m]`, `[5m]`).

```
t=0s    Locust inizia il gradual_ramp.
        Ogni cluster ha min_replicas=1 pod.
        cluster1 riceve 75 req/s → 1 pod saturo → latenza alta.

t=30s   DMOS polling cycle:
        ┌─ Fase 1: raccoglie ingress rates (Nginx metrics)
        │   cluster1=75 req/s, cluster2=50, cluster3=25 → total=150
        │   demand_shares = {c1:0.50, c2:0.33, c3:0.17}
        │
        ├─ Fase 2: raccoglie metriche per-cluster
        │   latency_p95, cpu_avail, carbon_intensity, RTT...
        │
        ├─ Fase 3: calcola score → cluster1=0.567, c2=0.644, c3=0.619
        │
        └─ Fase 4: Winner Determination → kubectl scale → 3+2+1 pod

t=60s   I nuovi pod sono Running e ricevono traffico.
        cluster1: 75 req/s su 3 pod → ~25 req/s per pod → latenza scende.

t=90s   Nuovo polling cycle — DMOS ricalcola con le metriche aggiornate.
```

Il traffico osservato in `[t-30s, t]` determina le repliche attive in `[t, t+30s]`. Questo **control lag di 30 secondi** è il principale limite del sistema reattivo. La componente predittiva di Level 2 (finestra `[5m]`) tenta di anticipare i picchi prima che la latenza degradi.

---

## Integrazione nella Score Function

### Formula completa aggiornata

```
score_i = ω_resp·Φ_response_time(i) + ω_cap·Φ_cap(i) + ω_load·Φ_load(i)
        + ω_carbon·Φ_carbon(i) + ω_net·Φ_net(i) + ω_demand·Φ_demand(i)
```

**Nota terminologica:** la componente di latenza è chiamata **Φ_response_time**
(nel codice: `omega_latency`) per distinguerla esplicitamente dal Φ_lat di Romano.
Romano misura la latenza tramite ping RTT (distanza geografica stimata), che in DMOS
è catturata da **Φ_net**. Φ_response_time misura invece la latenza HTTP p95 osservata
da Hubble sui pod reali sotto carico: include coda, processing time e dipendenze
inter-servizio via cluster mesh.

### Pesi default (profilo "balanced", Phase 2)

| Componente | Peso | Cosa misura | Disponibile a freddo |
|------------|------|-------------|---------------------|
| ω_resp (Φ_response_time) | 0.25 | Latenza HTTP p95 osservata da Hubble | ❌ No (Phase 2 only) |
| ω_cap (Φ_cap) | 0.20 | Risorse disponibili (CPU, RAM) | ❌ No (Phase 2 only) |
| ω_load (Φ_load) | 0.15 | Carico predetto vs capacità massima | ❌ No (Phase 2 only) |
| ω_carbon (Φ_carbon) | 0.20 | Carbon intensity della regione | ✅ Sì |
| ω_net (Φ_net) | 0.10 | RTT inter-cluster (proxy distanza geografica) | ✅ Sì |
| ω_demand (Φ_demand) | 0.10 | Traffico ingresso per-cluster (domanda diretta) | ✅ Sì |

Le componenti con "❌ No" nella colonna "Disponibile a freddo" sono disabilitate
(ω=0) durante la **Phase 1** dello scheduler (primi 120s dal deploy). Le componenti
con "✅ Sì" sono usate in entrambe le fasi.

Vedi [`docs/two-phase-scheduling.md`](two-phase-scheduling.md) per la descrizione
completa dell'approccio a due fasi e la motivazione statistica del threshold di 120s.

### Confronto Φ_net vs Φ_demand

| | Φ_net | Φ_demand |
|-|-------|----------|
| **Fonte** | ping_exporter (ICMP RTT inter-cluster) | nginx_ingress_controller_requests |
| **Cosa misura** | Distanza geografica stimata (RTT) | Domanda utente osservata (req/s) |
| **Variabilità** | Lenta (dipende da netem/topologia) | Veloce (segue il traffico in tempo reale) |
| **Robustezza** | Alta (stabile, non dipende dal traffico) | Bassa se Nginx non disponibile (fallback uniforme) |
| **Complementarietà** | Strutturale (dove *potrebbe* essere meglio) | Operativa (dove *effettivamente* arrivano gli utenti) |

I due segnali sono complementari:
- Φ_net dice "cluster1 è geograficamente centrale, potrebbe servire bene gli utenti europei"
- Φ_demand dice "cluster1 sta ricevendo il 60% del traffico, gli utenti ci sono già lì"

Usarli insieme aumenta la robustezza: se la RTT è simile tra cluster (netem spento), Φ_demand può comunque differenziare in base al traffico effettivo.

---

## Differenza rispetto alla Tesi di Romano

Romano non implementa Φ_demand. Il suo sistema usa `source="reserved:ingress"` per misurare il traffico ingress, ma questa metrica è usata **solo** per il rate del traffico globale (poi distribuito artificialmente con `cluster_traffic = total * quota`), non come componente dello score Level 1.

Contributo originale di questa tesi:
1. **Nginx Ingress come enabler di Hubble L7**: risolve il problema DNAT/Envoy che impedisce a Romano di usare le metriche Hubble sul NodePort esterno.
2. **Φ_demand come componente di score Level 1**: per la prima volta, il traffico ingresso per-cluster entra direttamente nella funzione di score, non solo come input per lo scaling.
3. **Traffico reale per-cluster**: grazie a `get_ingress_rate()` + normalizzazione, DMOS conosce la distribuzione geografica reale degli utenti, non la stima artificialmente.

---

## Verifica Sperimentale

### Query di verifica disponibilità metrica

```bash
# Verifica che Nginx esponga la metrica su ogni cluster
curl "http://192.168.1.245:30090/api/v1/query?query=nginx_ingress_controller_requests" | jq '.data.result'
curl "http://192.168.1.246:30090/api/v1/query?query=nginx_ingress_controller_requests" | jq '.data.result'
curl "http://192.168.1.247:30090/api/v1/query?query=nginx_ingress_controller_requests" | jq '.data.result'
```

### Cosa cercare nei log DMOS

Con Nginx disponibile e Φ_demand attivo:
```
INFO  [DMOSScheduler] Ingress rates: cluster1=12.3 req/s, cluster2=8.1 req/s, cluster3=4.6 req/s | total=25.0 req/s
INFO  [DMOSScheduler] Score cluster1: 0.712 (lat=0.823, cap=0.654, load=0.789, carbon=0.498, net=0.368, demand=0.492)
```

Senza Nginx (Φ_demand uniforme, fallback silenzioso):
```
DEBUG [DMOSScheduler] Nginx ingress metrics non disponibili, Φ_demand uniforme
INFO  [DMOSScheduler] Score cluster1: 0.698 (lat=0.823, cap=0.654, load=0.789, carbon=0.498, net=0.368, demand=0.333)
```

### Scenario flash crowd (verifica comportamento atteso)

Con un flash crowd che colpisce cluster1 (es. 70% del traffico su cluster1):
- **Senza Φ_demand**: DMOS può allocare repliche basandosi solo su RTT e capacità → potenziale mismatch
- **Con Φ_demand**: cluster1 ottiene `Φ_demand(1)=0.70` → score significativamente più alto → più repliche → latenza utente ridotta

Il test `gradual_ramp` in `experiments/locustfile_multiingress.py` è progettato per osservare questo comportamento su un arco di 22 minuti con ramp-up graduale da 20 a 300 utenti.

---

## Ruolo di Nginx nei Primi 120s (Phase 1)

Durante la Phase 1 dello scheduler (blind allocation), la metrica Nginx ingress rate
serve un **doppio ruolo**:

```
nginx_ingress_controller_requests
  │
  ├─→ Level 1 (WHERE):  Φ_demand(i) = rate_i / Σ rate_j
  │                     determina la quota di repliche per cluster
  │
  └─→ Level 2 (HOW MANY): target_replicas = rate_i / capacity_per_pod
                           proxy del carico sui pod (Hubble non ha dati)
```

Dopo 120s, Level 2 passa a Hubble `destination_workload` mentre Φ_demand continua
a usare Nginx. La metrica Nginx rimane quindi sempre attiva, indipendentemente dalla
fase.

*Vedi [`docs/two-phase-scheduling.md`](two-phase-scheduling.md) per dettagli
sull'architettura a due fasi.*
