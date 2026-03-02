# Scelta dell'architettura Ingress: Nginx + CNP L7 vs Cilium Ingress Controller

> Documento tecnico sulla motivazione della scelta architetturale dell'ingress in DMOS,
> confrontata con il baseline di Romano (2025).

---

## 1. Il punto di partenza: Romano usa Cilium Ingress Controller

Romano abilita il **Cilium Ingress Controller** su ogni cluster come punto di accesso unico
all'applicazione Online Boutique. Cilium Ingress usa Envoy come proxy HTTP nativo — lo stesso
processo che gestisce il dataplane L4/L7 di Cilium. Quando una richiesta entra dal load
generator (K6), attraversa Envoy e Cilium la registra automaticamente con il label
`source="reserved:ingress"` nei flow L7 di Hubble.

```
K6 → Cilium Ingress (Envoy) → frontend pod
          ↑
  Hubble: source="reserved:ingress" generato automaticamente
  → hubble_http_requests_total{source="reserved:ingress"}
```

Questo è il segnale che Romano usa come input primario per il calcolo dello score e
dell'autoscaling.

---

## 2. La scelta di DMOS: Nginx Ingress Controller + CiliumNetworkPolicy L7

DMOS usa invece il **Nginx Ingress Controller** (bare-metal, NodePort 30080), affiancato da
una **CiliumNetworkPolicy** esplicita che abilita la visibilità L7 di Hubble sui pod
`frontend`.

```
Locust → Nginx Ingress (:30080) → ClusterIP "frontend" → frontend pod
                                                               ↑
                               CNP l7-visibility-frontend intercetta qui
                               → Hubble vede HTTP L7 flows
                               → hubble_http_request_duration_seconds_bucket
                               → hubble_http_requests_total{destination_workload="frontend"}
```

I due file che implementano questa architettura:

- `deployments/ingress-frontend.yaml` — Nginx Ingress resource (namespace `online-boutique`)
- `deployments/cnp-frontend-l7.yaml` — CiliumNetworkPolicy L7 sul pod `frontend`

---

## 3. Perché non abbiamo usato Cilium Ingress Controller

### 3.1 Problema con K3s e pod CIDR differenti tra cluster

L'infrastruttura ha tre cluster K3s con **pod CIDR e service CIDR distinti** per cluster
(necessario per il ClusterMesh). In questa configurazione, il Cilium Ingress Controller ha un
comportamento problematico: quando Envoy (il proxy Cilium) riceve una connessione e la
inoltra al pod backend, le connessioni dirette pod→pod sullo stesso nodo **bypassano il
ClusterIP**. Il dataplane Cilium in questi casi non riesce a tracciare correttamente il flusso
L7 e le metriche Hubble diventano inaffidabili o assenti.

Nginx Ingress, con l'annotazione:

```yaml
nginx.ingress.kubernetes.io/service-upstream: "true"
```

forza il routing sempre attraverso il **ClusterIP** del service (non direttamente al pod IP).
Questo garantisce che ogni richiesta passi dal eBPF dataplane di Cilium, che può quindi
ispezionare il traffico L7 e produrre le metriche Hubble correttamente.

### 3.2 Visibilità L7 separabile dal routing

Con Cilium Ingress, la visibilità L7 è un effetto collaterale del fatto che Cilium è il
proxy — non è possibile controllarla indipendentemente dal routing. Con Nginx + CNP, le due
responsabilità sono separate:

- **Nginx**: gestisce il routing HTTP, i timeout, gli header, il load balancing verso i pod
- **CNP**: controlla la visibilità L7 e il perimetro di sicurezza del pod

Questo consente, ad esempio, di rimuovere la CNP per un test senza toccare il routing, o
di modificare le policy di sicurezza senza sostituire il controller.

### 3.3 Maturità e debugging

Nginx Ingress è lo standard de-facto per K3s bare-metal. La sua documentazione, i suoi log e
i suoi strumenti di debug sono molto più consolidati rispetto al Cilium Ingress Controller,
che è ancora in evoluzione attiva. In un ambiente di ricerca dove i test di carico vengono
eseguiti ripetutamente con configurazioni diverse, la debuggability del componente di ingress
è un fattore rilevante.

---

## 4. Come la CNP abilita la visibilità L7 di Hubble

Hubble registra metriche L7 (HTTP, gRPC, DNS) **solo quando una CiliumNetworkPolicy L7 è
attiva** su un endpoint. Senza policy L7, Cilium opera in modalità L3/L4 pura e Hubble vede
solo connessioni TCP (source IP, destination IP, porta) — nessuna informazione su URL,
metodi HTTP, status code, o durata delle richieste.

La CNP `l7-visibility-frontend` (`cnp-frontend-l7.yaml`) configura:

```yaml
ingress:
- fromEntities:
  - cluster   # pod di qualsiasi namespace (incluso ingress-nginx)
  - host       # kubelet health check
  toPorts:
  - ports:
    - port: "8080"
      protocol: TCP
```

Applicando questa policy, Cilium attiva il proxy Envoy **in-process** sul path
`ingress-nginx → frontend:8080`. Da questo momento Hubble registra ogni richiesta HTTP con:
- metodo, URL, status code
- durata della richiesta (da cui deriva `hubble_http_request_duration_seconds_bucket`)
- workload sorgente e destinazione (da cui deriva il filtro `destination_workload="frontend"`)

La policy blocca anche l'accesso diretto `world → frontend:8080` — tutto il traffico esterno
è forzato a passare da Nginx, che è l'unico mittente `cluster`-identity autorizzato.

**Nota:** la sezione `egress` è deliberatamente assente dalla CNP. Cilium per K3s con pod/service
CIDR multipli non riesce a risolvere correttamente `toEntities:cluster` per i ClusterIP
(10.43.0.0/16 non è un pod CIDR). Restringere l'egress romperebbe DNS e tutte le chiamate
gRPC verso i backend. Il default Cilium (`allow all egress`) è la scelta corretta qui.

---

## 5. Impatto sulla metrica di traffico: cambio di filtro Hubble

La scelta di Nginx + CNP cambia il filtro necessario per leggere il traffico da Hubble:

| | Romano (Cilium Ingress) | DMOS (Nginx + CNP) |
|---|---|---|
| **Filtro Hubble per traffico** | `source="reserved:ingress"` | `destination_workload="frontend"` |
| **Cosa conta** | Richieste che entrano dal Cilium Ingress | Richieste che raggiungono i pod `frontend` |
| **Include sub-richieste interne** | No — solo ingress esterno | Sì — include tutti i flussi verso `frontend` |
| **Moltiplicatore vs Locust** | ~1.0× (corrispondenza diretta) | ~1.43× (include sotto-richieste HTTP per pagina) |

Il filtro `destination_workload="frontend"` è **più preciso come misura del carico sul pod**
perché include ogni richiesta HTTP che il frontend deve processare, non solo quelle
che entrano dall'esterno. Tuttavia introduce il moltiplicatore sistematico ~1.43× rispetto
al conteggio Locust, che va tenuto presente nell'analisi dei risultati.

---

## 6. Pro e contro della scelta DMOS

### Pro

| Vantaggio | Dettaglio |
|-----------|-----------|
| **Metriche L7 affidabili su K3s** | Nginx + `service-upstream: true` garantisce che ogni richiesta attraversi il dataplane eBPF di Cilium → Hubble vede tutti i flussi HTTP |
| **Doppia metrica disponibile** | Con CNP attiva, DMOS può leggere sia `hubble_http_requests_total` (rate) sia `hubble_http_request_duration_seconds_bucket` (latenza p95) — Romano ha solo il rate |
| **Separazione responsabilità** | Routing (Nginx) e visibilità/sicurezza (CNP) sono componenti indipendenti e modificabili separatamente |
| **Debug più facile** | Log Nginx dettagliati per ogni richiesta; Hubble UI mostra i flow L7 con URL e status code |
| **Standard per K3s bare-metal** | Nginx Ingress è il controller raccomandato per K3s su hardware fisico; Cilium Ingress è pensato per ambienti cloud managed |

### Contro

| Svantaggio | Dettaglio |
|------------|-----------|
| **Hop aggiuntivo** | La richiesta attraversa Nginx (processo separato) prima di arrivare al pod. Con Cilium Ingress, Envoy è in-process nel dataplane Cilium — latenza marginalmente inferiore (~1-2ms in LAN) |
| **Due componenti da mantenere** | Nginx Ingress + CNP invece di un solo Cilium Ingress. Il rollback richiede di gestire entrambi |
| **Filtro diverso da Romano** | `destination_workload="frontend"` invece di `source="reserved:ingress"` — introduce il moltiplicatore 1.43×, che rende il confronto diretto dei numeri di traffico non immediato |
| **CNP richiesta esplicita** | Senza la CNP, Hubble non produce metriche L7 → se la policy viene rimossa per errore, DMOS perde la misurazione di latenza e deve ricadere sul fallback `baseline_ms=2.0` |
| **Non trasferibile 1:1 a cloud managed** | In cloud (GKE, EKS, AKS), il Cilium Ingress Controller è nativamente supportato e la CNP manuale potrebbe essere superflua o incompatibile |

---

## 7. Architettura completa di test

Entrambe le architetture (Romano e DMOS) hanno la stessa topologia di test a 3 ingress —
il load generator invia traffico a tutti e tre i cluster simultaneamente:

```
                    ┌─────────────────────────────────────────┐
                    │             ROMANO                       │
  K6 Load Test ─── Ingress 1 (Cilium/Envoy) ─┐               │
               ─── Ingress 2 (Cilium/Envoy) ──┤→ Service      │
               ─── Ingress 3 (Cilium/Envoy) ─┘   Frontend     │
                    │             (Global)                     │
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │             DMOS                         │
  Locust ─── :245:30080 (Nginx) ─┐                           │
         ─── :246:30080 (Nginx) ──┤→ Service Frontend         │
         ─── :247:30080 (Nginx) ─┘   (Global, ClusterIP)     │
                    │  ↑ CNP L7 su ogni cluster               │
                    └─────────────────────────────────────────┘
```

La topologia è identica: 3 ingress, un Global Service, load generator distribuito.
La differenza è solo nel tipo di ingress controller e nel meccanismo di visibilità L7.

---

*Documento redatto sulla base di: G. Romano (2025) §4.4,
`deployments/ingress-frontend.yaml`, `deployments/cnp-frontend-l7.yaml`,
`src/metrics/prometheus_client.py`.*
