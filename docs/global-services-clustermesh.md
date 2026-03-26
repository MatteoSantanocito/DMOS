# Global Services con Cilium ClusterMesh — Configurazione e Verifica

**Data**: 20 marzo 2026
**Contesto**: Estensione del sistema DMOS per supportare routing cross-cluster del servizio frontend
**Motivazione**: Suggerimento del relatore — introduzione della geo-awareness come criterio di scheduling

---

## 1. Contesto e Motivazione

### 1.1 Osservazioni del relatore

Durante un incontro con il relatore sono emerse le seguenti osservazioni sull'architettura del sistema:

- **Chiarire l'obiettivo di ottimizzazione**: il sistema deve dichiarare esplicitamente cosa ottimizza (response time p95 < 200ms sotto vincolo di capacità)
- **Geo-awareness**: prima di decidere su quale cluster schedulare, considerare la distribuzione geografica degli utenti e la latenza di rete cluster→utente
- **Latenza di rete con ping**: usare la RTT misurata via ping come proxy per la prossimità geografica utente-cluster
- **Placement dei microservizi**: in prospettiva futura, valutare la latenza inter-cluster per decidere dove schedulare i microservizi nella catena applicativa

### 1.2 Confronto con la tesi di Romano (2025)

La tesi di Romano Giuseppe (2025) implementa un sistema simile con le seguenti differenze architetturali:

| Aspetto | Romano | Questo sistema (prima) |
|---|---|---|
| Ingress controller | Cilium Ingress | Nginx Ingress |
| Global Services | ✅ abilitati | ❌ non configurati |
| Routing cross-cluster | Via Cilium ClusterMesh + Global Services | Ogni cluster indipendente |
| Geo-awareness | `ping_exporter` + `ping_rtt_mean_seconds` | Non presente |
| Metrica traffico | `hubble_http_requests_total{source="reserved:ingress"}` | `hubble_http_requests_total{destination_workload="frontend"}` |

Romano misura la **latenza inter-cluster** (cluster→cluster) tramite `ping_exporter` deployato via Helm, con target configurati sugli IP dei nodi degli altri cluster. Utilizza `tc netem` per iniettare artificialmente delay di rete e simulare distanze geografiche realistiche, poiché tutti i cluster sono fisicamente sulla stessa LAN.

### 1.3 Perché i Global Services sono prerequisito alla geo-awareness

Senza Global Services ogni cluster serve le proprie richieste in modo completamente indipendente:

```
Utente → Nginx cluster1 → frontend pod cluster1  (sempre)
Utente → Nginx cluster2 → frontend pod cluster2  (sempre)
```

In questo scenario la latenza inter-cluster non ha impatto sull'utente: ogni richiesta viene servita interamente dal cluster che la riceve. Il ping inter-cluster non aggiunge informazione utile allo scheduling.

Con Global Services attivi, Cilium ClusterMesh forma un **pool unico** di pod frontend tra tutti i cluster:

```
Utente → Nginx cluster1 → frontend pod cluster1  ←
                        → frontend pod cluster2  ← Cilium sceglie
                        → frontend pod cluster3  ←
```

In questo scenario la latenza inter-cluster impatta effettivamente l'esperienza utente: una richiesta che entra su cluster1 può essere servita da un pod su cluster2, aggiungendo la latenza del tunnel vxlan ClusterMesh. Il ping inter-cluster diventa quindi un input rilevante per lo scheduling.

---

## 2. Stato pre-configurazione

### 2.1 ClusterMesh già attivo

Il sistema disponeva già di Cilium ClusterMesh configurato e operativo su tutti e 3 i cluster, con connettività vxlan tra i nodi:

| Cluster | IP nodo | Regione simulata | Carbon Intensity |
|---|---|---|---|
| cluster1 | 192.168.1.245 | Frankfurt (DE) | 350 gCO2/kWh |
| cluster2 | 192.168.1.246 | Paris (FR) | 80 gCO2/kWh |
| cluster3 | 192.168.1.247 | Warsaw (PL) | 650 gCO2/kWh |

Verifica connettività ClusterMesh (eseguita su tutti i cluster):
```
✅ Service "clustermesh-apiserver" of type "NodePort" found
✅ All 1 nodes are connected to all clusters [min:2 / avg:2.0 / max:2]
🔌 cluster2: 1/1 configured, 1/1 connected - KVStoreMesh: 1/1 configured, 1/1 connected
🔌 cluster3: 1/1 configured, 1/1 connected - KVStoreMesh: 1/1 configured, 1/1 connected
```

### 2.2 Servizi senza annotazioni Global Service

Prima della configurazione, il servizio `frontend` nel namespace `online-boutique` era un semplice ClusterIP senza annotazioni Cilium. Ogni cluster operava in isolamento completo: nessun endpoint cross-cluster era sincronizzato tramite ClusterMesh.

```bash
kubectl get svc frontend -n online-boutique --context cluster1 --show-labels
# Nessuna annotazione io.cilium/global-service
```

### 2.3 CiliumNetworkPolicy originale

La CNP `l7-visibility-frontend` permetteva ingress al frontend solo da:
- `fromEntities: cluster` — pod del cluster locale (qualsiasi namespace, incluso `ingress-nginx`)
- `fromEntities: host` — processo kubelet sul nodo locale

Il traffico proveniente da nodi di cluster remoti (classificato da Cilium come `remote-node`) veniva **bloccato** dalla policy, rendendo impossibile il routing cross-cluster anche se il ClusterMesh fosse stato configurato per farlo.

---

## 3. Modifiche effettuate

### 3.1 Nuovo file: `deployments/global-service-frontend.yaml`

Creato un manifest Service con le annotazioni necessarie per abilitare i Global Services:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: online-boutique
  annotations:
    io.cilium/global-service: "true"
    io.cilium/shared-service: "true"
spec:
  selector:
    app: frontend
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
```

**Significato delle annotazioni:**

- `io.cilium/global-service: "true"` — rende il Service visibile e raggiungibile da tutti i cluster connessi alla ClusterMesh. Cilium sincronizza gli endpoint del frontend tra i cluster tramite il KVStore distribuito.
- `io.cilium/shared-service: "true"` — abilita il routing bidirezionale. Senza questa annotazione il Global Service è in modalità "read-only": gli altri cluster vedono gli endpoint ma non li usano per il bilanciamento. Con `shared-service: true` ogni cluster può instradare richieste verso i pod frontend degli altri cluster.

**Applicazione:**
```bash
kubectl apply -f deployments/global-service-frontend.yaml --context cluster1
kubectl apply -f deployments/global-service-frontend.yaml --context cluster2
kubectl apply -f deployments/global-service-frontend.yaml --context cluster3
```

Output: `service/frontend configured` su tutti e 3 i cluster.

### 3.2 Modifica: `deployments/cnp-frontend-l7.yaml`

Aggiunta una seconda regola ingress per permettere traffico dai nodi dei cluster remoti:

**Prima (ingress originale):**
```yaml
ingress:
- fromEntities:
  - cluster   # pod del cluster locale
  - host      # kubelet locale
  toPorts:
  - ports:
    - port: "8080"
      protocol: TCP
```

**Dopo (con regola cross-cluster):**
```yaml
ingress:
# Regola 1: traffico intra-cluster
- fromEntities:
  - cluster      # pod di qualsiasi namespace nel cluster locale
  - host         # processo sul nodo (kubelet, health check)
  toPorts:
  - ports:
    - port: "8080"
      protocol: TCP

# Regola 2: traffico cross-cluster via Cilium ClusterMesh (Global Services)
- fromEntities:
  - remote-node  # nodi/pod dei cluster remoti connessi via ClusterMesh
  toPorts:
  - ports:
    - port: "8080"
      protocol: TCP
```

**Motivazione:** Con Global Services attivi, il Nginx di cluster2 o cluster3 può instradare richieste verso pod frontend di cluster1 attraverso il tunnel vxlan ClusterMesh. In Cilium, questo traffico proveniente da nodi remoti viene classificato come entità `remote-node`. Senza questa regola, tutto il traffico cross-cluster veniva silenziosamente bloccato dalla policy a livello L3, rendendo i Global Services inoperativi nonostante la corretta configurazione delle annotazioni.

**Applicazione:**
```bash
kubectl apply -f deployments/cnp-frontend-l7.yaml --context cluster1
kubectl apply -f deployments/cnp-frontend-l7.yaml --context cluster2
kubectl apply -f deployments/cnp-frontend-l7.yaml --context cluster3
```

Output: `ciliumnetworkpolicy.cilium.io/l7-visibility-frontend configured` su tutti e 3 i cluster.

---

## 4. Verifica e risultati

### 4.1 Verifica annotazioni

```bash
kubectl get svc frontend -n online-boutique --context cluster1 \
  -o jsonpath='{.metadata.annotations.io\.cilium/global-service}'
# Output: true
```

Verificato su tutti e 3 i cluster con output `true`.

### 4.2 Verifica ClusterMesh status

```
cilium clustermesh status --context cluster1

✅ Service "clustermesh-apiserver" of type "NodePort" found
✅ Cluster access information is available: 192.168.1.245:32379
✅ Deployment clustermesh-apiserver is ready
ℹ️  KVStoreMesh is enabled
✅ All 1 nodes are connected to all clusters [min:2 / avg:2.0 / max:2]
✅ All 1 KVStoreMesh replicas are connected to all clusters [min:2 / avg:2.0 / max:2]
🔌 cluster2: 1/1 configured, 1/1 connected
🔌 cluster3: 1/1 configured, 1/1 connected
```

### 4.3 Verifica service table Cilium — endpoint cross-cluster

Eseguendo `cilium service list` dentro il pod agente di cluster1, la riga del frontend mostra:

```
ID   Frontend              Service Type   Backend
17   10.43.87.229:80       ClusterIP      1 => 10.42.0.219:8080 (active)
                                          2 => 10.44.0.209:8080 (active)
```

**Analisi:**
- `10.42.0.x` — pod CIDR di cluster1. Questo è il pod frontend locale.
- `10.44.0.x` — pod CIDR di un cluster remoto (cluster2 o cluster3). Questo endpoint è stato sincronizzato via ClusterMesh dopo l'applicazione delle annotazioni Global Service.

Cilium sta già bilanciando il traffico tra i due backend. Questo è il segnale diretto che il ClusterMesh ha sincronizzato gli endpoint e che il dataplane Cilium è configurato per il routing cross-cluster.

**Nota importante:** `kubectl get endpoints` continua a mostrare solo un endpoint (il pod locale). Questo è comportamento atteso: Kubernetes non è a conoscenza degli endpoint cross-cluster. La sincronizzazione avviene esclusivamente a livello del dataplane Cilium, invisibile alle API Kubernetes standard.

### 4.4 Verifica routing cross-cluster con Hubble

Con traffico attivo (generato tramite Locust con scenario `flash_crowd`, 50 utenti iniziali distribuiti sui 3 cluster con pesi 40/35/25), l'output di `hubble observe --last 50` su cluster1 mostra **due identità frontend distinte** che ricevono traffico da Nginx:

```
# Pod frontend LOCALE di cluster1 (identity 91770 — presente in cilium endpoint list)
ingress-nginx/ingress-nginx-controller-644c9c6b99-mc77d → online-boutique/frontend-76dbb9564c-dcg4b:8080   FORWARDED

# Pod frontend di CLUSTER REMOTO (identity 138068 — non presente in cilium endpoint list di cluster1)
ingress-nginx/ingress-nginx-controller-644c9c6b99-wrgws → online-boutique/frontend-fc6cd7cb8-jgmt8:8080   FORWARDED
```

Il pod `frontend-fc6cd7cb8-jgmt8` con identity Cilium `138068` non è presente nell'endpoint list locale di cluster1 — è un pod che risiede fisicamente su cluster2 o cluster3, raggiunto attraverso il tunnel vxlan ClusterMesh.

Ulteriore conferma dalla presenza di traffico ClusterMesh attivo verso il clustermesh-apiserver:
```
10.44.0.76:60860 (host) -> kube-system/clustermesh-apiserver-56746c559d-vwbqf:9881   FORWARDED
```

`10.44.0.76` è il nodo di un cluster remoto che mantiene la connessione verso il clustermesh-apiserver di cluster1 per la sincronizzazione continua degli endpoint.

### 4.5 Verifica integrità applicativa

Durante il test Locust, i log del checkoutservice su cluster1 mostrano ordini completati correttamente:

```json
{"message":"payment went through (transaction_id: d7ee9347-...)","severity":"info"}
{"message":"order confirmation email sent to \"test@example.com\"","severity":"info"}
```

Le failures registrate da Locust su `/cart/checkout` (~20%) sono **timeout lato client** — Locust registra come failure le richieste che superano il timeout configurato, ma gli ordini vengono comunque completati lato server. Questo comportamento era già presente prima dell'introduzione dei Global Services ed è dovuto alla catena di chiamate gRPC del checkout (frontend → checkoutservice → paymentservice → shippingservice → emailservice) che sotto carico può superare il timeout del client HTTP.

---

## 5. Impatto sull'architettura DMOS

### 5.1 Cambiamento nel significato delle metriche Hubble

Con Global Services attivi, la metrica `hubble_http_requests_total{destination_workload="frontend"}` su cluster1 misura le richieste **servite dai pod frontend di cluster1**, incluse quelle inoltrate da Nginx di cluster2 e cluster3. Non misura più solo le richieste "entrate" da cluster1.

Questo cambiamento è coerente con l'obiettivo del sistema: DMOS deve sapere quanto lavoro sta facendo ciascun cluster, non solo quante richieste sono arrivate al suo ingress.

### 5.2 Ruolo di DMOS con Global Services

Il sistema DMOS mantiene il proprio ruolo invariato: decide il **numero di repliche** per cluster e su **quale cluster** scalare, in base allo score multi-dimensionale (latenza, capacità, carico, carbon). Il routing delle singole richieste è delegato al dataplane Cilium ClusterMesh, che distribuisce automaticamente il traffico tra tutti i pod frontend disponibili.

La combinazione è quindi:
- **DMOS** → decide la distribuzione delle repliche tra cluster (scheduling ad alto livello)
- **Cilium ClusterMesh** → esegue il routing effettivo delle singole richieste tra i pod (forwarding a basso livello)

### 5.3 Prerequisito per la geo-awareness (prossimo step)

Con i Global Services attivi, il ping inter-cluster diventa un input rilevante per lo scheduling. Se DMOS sa che cluster2 ha latenza media di 150ms verso gli utenti e cluster1 ha latenza di 8ms, può:
1. Allocare più repliche su cluster1 (cluster più vicino agli utenti)
2. Pesare il routing di Cilium indirettamente tramite il numero di repliche disponibili per cluster

Il passo successivo prevede:
1. Deploy di `ping_exporter` (progetto `czerwonk/ping_exporter`) via Helm su ogni cluster
2. Simulazione delle distanze geografiche con `tc netem` sulle interfacce dei nodi
3. Aggiunta della funzione score `Φ_net` in `src/level1/score_functions.py`
4. Aggiornamento dei pesi in `config/weights.yaml` con `omega_network`

---

## 6. Riepilogo file modificati

| File | Tipo | Descrizione modifica |
|---|---|---|
| `deployments/global-service-frontend.yaml` | Nuovo | Manifest Service con annotazioni `io.cilium/global-service` e `io.cilium/shared-service` |
| `deployments/cnp-frontend-l7.yaml` | Modificato | Aggiunta regola ingress `fromEntities: remote-node` per traffico cross-cluster |

---

## 7. Comandi utili per manutenzione

```bash
# Verifica annotazioni Global Service su tutti i cluster
for ctx in cluster1 cluster2 cluster3; do
  echo -n "$ctx: "
  kubectl get svc frontend -n online-boutique --context $ctx \
    -o jsonpath='{.metadata.annotations.io\.cilium/global-service}'
  echo
done

# Verifica backend cross-cluster nel dataplane Cilium
kubectl exec -n kube-system ds/cilium --context cluster1 -- \
  cilium service list | grep "10.43.87.229"

# Osserva flussi cross-cluster in tempo reale
cilium hubble port-forward --context cluster1 &
hubble observe --follow --namespace online-boutique \
  --to-label app=frontend --server 127.0.0.1:4245

# Rollback completo (rimuove Global Services, torna a isolamento per cluster)
kubectl annotate svc frontend -n online-boutique \
  io.cilium/global-service- io.cilium/shared-service- \
  --context cluster1 --context cluster2 --context cluster3
```
