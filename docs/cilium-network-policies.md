# Cilium Network Policies — Documentazione Completa
## Contesto: DMOS Multi-Cluster con Cilium ClusterMesh

> Documento creato il 27/02/2026 a seguito della sessione di debugging
> che ha risolto il problema dei timeout al 100% nei test Locust multiingress.

---

## Indice

1. [Architettura del cluster](#1-architettura-del-cluster)
2. [Cos'è una CiliumNetworkPolicy](#2-cosè-una-ciliumnetworkpolicy)
3. [Identità e Entità in Cilium](#3-identità-e-entità-in-cilium)
4. [L7 Rules e il proxy Envoy](#4-l7-rules-e-il-proxy-envoy)
5. [Le policy trovate nel cluster](#5-le-policy-trovate-nel-cluster)
6. [Il problema: perché Locust aveva 100% timeout](#6-il-problema-perché-locust-aveva-100-timeout)
7. [Diagnosi passo per passo](#7-diagnosi-passo-per-passo)
8. [La soluzione finale](#8-la-soluzione-finale)
9. [Comportamento Cilium con NodePort](#9-comportamento-cilium-con-nodeport)
10. [OOMKilled: currencyservice e paymentservice](#10-oomkilled-currencyservice-e-paymentservice)
11. [Checklist pre-test](#11-checklist-pre-test)
12. [Comandi utili di riferimento](#12-comandi-utili-di-riferimento)

---

## 1. Architettura del cluster

Il setup DMOS usa **3 cluster Kubernetes** collegati via **Cilium ClusterMesh**:

```
Windows (192.168.1.x)  ←── test traffic ──→  cluster1 (192.168.1.245)
                                              cluster2 (192.168.1.246)
                                              cluster3 (192.168.1.247)

Ogni cluster:
  - NodePort 30007  → frontend (Online Boutique)
  - NodePort 30090  → Prometheus
  - NodePort 30091  → Grafana

Cilium sostituisce kube-proxy (modalità eBPF full replacement).
Cilium ClusterMesh connette i 3 cluster per service discovery e policy cross-cluster.
```

**Servizi esposti:**

| Servizio | Tipo | Porta interna | NodePort |
|---|---|---|---|
| `frontend` | ClusterIP | 80 → pod:8080 | — |
| `frontend-external` | NodePort | 80 → pod:8080 | **30007** |
| `prometheus` | NodePort | 9090 | 30090 |
| `grafana` | NodePort | 80 | 30091 |

> **Nota**: `frontend` (ClusterIP) e `frontend-external` (NodePort) sono due Service
> distinti che puntano agli stessi pod (`selector: app=frontend`).

---

## 2. Cos'è una CiliumNetworkPolicy

Una `CiliumNetworkPolicy` (CNP) è una risorsa Kubernetes custom (CRD) che estende le
standard `NetworkPolicy` di Kubernetes con funzionalità avanzate di Cilium:

- **L3/L4**: filtraggio per IP, CIDR, porta, protocollo
- **L7**: ispezione HTTP, gRPC, Kafka (tramite proxy Envoy integrato)
- **Identità-based**: filtraggio per label dei pod, non per IP (più robusto in ambienti dinamici)
- **Entità speciali**: `world`, `host`, `cluster`, `remote-node`, ecc.

### Semantica fondamentale

```
Nessuna policy → Default ALLOW (tutto permesso)
Almeno una policy seleziona un endpoint → Default DENY (tutto bloccato tranne l'esplicito)
```

Questo è il punto più critico: **appena applichi una CNP a un pod, tutto il traffico
non esplicitamente permesso viene droppato silenziosamente** (senza RST, senza ICMP
unreachable — il mittente vede solo timeout).

### Struttura base

```yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: esempio
  namespace: online-boutique
spec:
  # A quali pod si applica questa policy
  endpointSelector:
    matchLabels:
      app: frontend

  # Traffico IN ENTRATA permesso
  ingress:
  - fromEndpoints:        # sorgenti permesse (altri pod)
    - matchLabels:
        app: checkoutservice
    toPorts:              # porte/protocolli permessi
    - ports:
      - port: "8080"
        protocol: TCP

  # Traffico IN USCITA permesso
  egress:
  - toEndpoints:
    - {}                  # tutti i pod del cluster
```

### Operatori logici nelle regole

All'interno di una singola regola `ingress`, **i diversi campi source sono OR tra loro**
SOLO se sono dello stesso tipo. Tra tipi diversi (`fromEndpoints` + `fromEntities`)
nello stesso blocco regola, il comportamento è **AND** in Cilium.

**SBAGLIATO** (AND — condizione impossibile per traffico esterno):
```yaml
ingress:
- fromEndpoints:      # deve essere un pod
  - {}
  fromEntities:       # AND deve anche essere world/host
  - world
  - host
```

**CORRETTO** (OR — due regole separate):
```yaml
ingress:
- fromEndpoints:      # regola 1: traffico da pod
  - {}
  toPorts: [...]
- fromEntities:       # regola 2: traffico esterno
  - world
  - host
  toPorts: [...]
```

---

## 3. Identità e Entità in Cilium

Cilium non usa gli IP per identificare le sorgenti del traffico (gli IP cambiano
con i restart dei pod). Usa invece le **identità**, derivate dai label Kubernetes.

### Entità speciali (non-pod)

| Entità | Cosa rappresenta |
|---|---|
| `world` | Tutto il traffico da fuori il cluster (internet, LAN esterna) |
| `host` | Il nodo Kubernetes stesso (network namespace del nodo) |
| `cluster` | Tutti gli endpoint all'interno del cluster |
| `remote-node` | Nodi di altri cluster (ClusterMesh) |
| `init` | Pod in fase di inizializzazione (senza identity ancora) |

### Come Cilium classifica il traffico NodePort

Quando un pacchetto arriva da Windows (192.168.1.x) sulla NodePort 30007:

```
1. Pacchetto arriva al nodo: src=192.168.1.x, dst=192.168.1.245:30007
2. Cilium eBPF hook (XDP/tc) intercetta
3. DNAT: dst diventa 10.42.0.94:8080 (pod IP)
4. Il sorgente (192.168.1.x) non è un pod Cilium → identità = "world"
5. La network policy viene valutata DOPO il DNAT
   → porta valutata: 8080 ✓
   → sorgente valutata: "world" → richiede fromEntities: ["world"]
```

---

## 4. L7 Rules e il proxy Envoy

### Cosa sono le L7 Rules

Le L7 rules permettono a Cilium di ispezionare il contenuto delle richieste HTTP/gRPC,
non solo IP e porte. Utili per:
- Visibilità in **Hubble** (tracing HTTP, metodi, URL, status code)
- Policy basate su path HTTP (`/api/v1/*`)
- Rate limiting per endpoint specifici

```yaml
toPorts:
- ports:
  - port: "8080"
    protocol: TCP
  rules:
    http:           # ← L7 rule: abilita ispezione HTTP
    - {}            # {} = permetti tutto l'HTTP (nessun filtro su path/metodo)
```

### Il proxy Envoy in Cilium

Quando una L7 rule è attiva su una porta, Cilium inserisce un **proxy Envoy** nel
percorso del traffico:

```
Pod A ──► Envoy (Cilium L7 proxy) ──► Pod B (porta 8080)
```

Envoy decodifica l'HTTP, applica le regole L7, raccoglie le metriche, e poi
forwarda la richiesta al pod destinazione.

### Il comportamento critico: L7 contamina tutto il traffico sulla porta

**Questo è il punto che ha causato il problema nel nostro cluster.**

Se **qualsiasi** regola ingress per una determinata porta usa `rules: http`,
Cilium forza **tutto il traffico** su quella porta attraverso Envoy —
**indipendentemente da quale regola ha il match**.

```yaml
ingress:
- fromEndpoints:          # regola 1: traffico da pod → Envoy
  - {}
  toPorts:
  - ports:
    - port: "8080"
    rules:
      http:               # ← L7 rule qui...
      - {}

- fromEntities:           # regola 2: traffico esterno
  - world
  toPorts:
  - ports:
    - port: "8080"        # ← ...forza Envoy ANCHE qui, anche senza rules: http
```

**Risultato**: il traffico da `world` (Locust su Windows) viene instradato attraverso
Envoy, ma Envoy non riesce a gestire correttamente il traffico NodePort con sorgente
esterna → **drop silenzioso → timeout**.

### Come evitarlo

**Opzione A — Regole separate senza L7 per il traffico esterno** *(quella adottata)*:

```yaml
ingress:
- fromEndpoints:          # traffico interno pod: con L7 (per Hubble)
  - {}
  toPorts:
  - ports:
    - port: "8080"
    rules:
      http:
      - {}

- fromEntities:           # traffico esterno: senza L7 (L4 puro)
  - world
  - host
  toPorts:
  - ports:
    - port: "8080"
    # nessun rules: http → bypass Envoy per questo traffico
```

**Opzione B — Nessuna L7 rule** (perde visibilità Hubble, ma massima compatibilità):

```yaml
ingress:
- fromEndpoints:
  - {}
  toPorts:
  - ports:
    - port: "8080"
- fromEntities:
  - world
  - host
  toPorts:
  - ports:
    - port: "8080"
```

**Opzione C — L7 su entrambe le regole** (Envoy anche per esterno — richiede
configurazione Envoy corretta per NodePort):

```yaml
ingress:
- fromEndpoints:
  - {}
  toPorts:
  - ports:
    - port: "8080"
    rules:
      http:
      - {}
- fromEntities:
  - world
  - host
  toPorts:
  - ports:
    - port: "8080"
    rules:
      http:             # ← Envoy per esterno, ma potrebbe avere problemi
      - {}
```

---

## 5. Le policy trovate nel cluster

### `l7-visibility-frontend`

**Scopo**: proteggere il pod `frontend` e abilitare visibilità L7 Hubble per DMOS.

**Applicata a**: pod con label `app=frontend` in namespace `online-boutique`.

**Stato originale (problematico — v1)**:
```yaml
spec:
  endpointSelector:
    matchLabels:
      app: frontend
  egress:
  - toEndpoints:
    - {}                  # egress verso tutti i pod (namespace-scoped! bug)
  - toEntities:
    - world               # egress verso internet
  ingress:
  - fromEndpoints:
    - {}                  # SOLO da pod dello stesso namespace → blocca Nginx!
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - {}              # L7 → Envoy per TUTTO il traffico su 8080
                          # Con fromEntities:world → timeout Locust via NodePort
```

**Problemi**: (1) `fromEndpoints:{}` è namespace-scoped → Nginx (in `ingress-nginx`)
bloccato. (2) Se `world` venisse aggiunto, L7 causerebbe timeout Envoy per traffico
esterno. (3) Sezione egress → blocca DNS (kube-system) e gRPC backend (ClusterIP).

**Stato intermedio (senza L7)**:
```yaml
ingress:
- fromEntities: [cluster, host]  # fix Bug #1: copre tutti i namespace
  toPorts:
  - ports: [{port: "8080", protocol: TCP}]
  # Nessun rules:http → nessun Envoy → nessun timeout
  # MA: hubble_http_requests_total congelato (no Envoy = no L7 counting)
```

**Stato finale (con Hubble L7)** — versione attuale in produzione:
```yaml
spec:
  endpointSelector:
    matchLabels:
      app: frontend
  # Nessun egress → allow all (fix Bug #2/#3: ClusterIP + DNS funzionano)
  ingress:
  - fromEntities:
    - cluster   # tutti i pod del cluster, qualsiasi namespace (incluso ingress-nginx)
    - host      # nodo locale (kubelet health check)
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - {}   # ← CHIAVE: attiva Envoy L7 → hubble_http_requests_total++
               #   Sicuro perché world è assente → bloccato a L3 prima di Envoy
```

**Perché ora L7 funziona senza timeout**: `world` (Locust esterno) è bloccato
a L3 dalla CNP. Envoy riceve solo traffico `cluster` (Nginx) e `host` (kubelet)
— per questi, il proxy same-node funziona correttamente.

### `l7-visibility-backends`

**Scopo**: visibilità L7 per tutti i servizi backend.

**Applicata a**: tutti i pod tranne `frontend`, `loadgenerator`, `redis-cart`.

```yaml
spec:
  endpointSelector:
    matchExpressions:
    - key: app
      operator: NotIn
      values:
      - frontend
      - loadgenerator
      - redis-cart
  egress:
  - toEndpoints:
    - {}                  # egress solo verso altri pod del cluster
  ingress:
  - fromEndpoints:
    - {}                  # SOLO da pod interni
    toPorts:
    - ports:
      - port: "3550"      # adservice
      - port: "5050"      # emailservice
      - port: "7000"      # cartservice
      - port: "7070"      # frontend→backend gRPC
      - port: "8080"      # vari servizi HTTP
      - port: "9555"      # adservice
      - port: "50051"     # gRPC generico
      rules:
        http:
        - {}
```

**Nota**: questa policy non causa problemi per il traffico Locust perché Locust
parla direttamente col `frontend` (porta 8080), non con i backend. I backend
ricevono solo traffico gRPC interno dal frontend.

---

## 6. Il problema: perché Locust aveva 100% timeout

### Catena causale completa

```
Locust (Windows 192.168.1.x)
    │
    │  HTTP GET http://192.168.1.245:30007/
    ▼
Node eth0 (cluster1)
    │
    │  Cilium eBPF NodePort handler
    │  DNAT: :30007 → pod 10.42.0.94:8080
    ▼
Cilium policy engine (valuta DOPO DNAT)
    │
    │  Verifica: src=192.168.1.x → identità = "world"
    │  Regola attiva: fromEndpoints=[{}] + rules:http → Envoy obbligatorio
    │  "world" non è un endpoint Cilium
    │
    │  ┌─ Se rules:http presente per la porta → TUTTO passa da Envoy
    │  │  Envoy non gestisce src="world" su NodePort → DROP
    │  └─ (o) "world" non matcha fromEndpoints → DROP
    │
    ▼
TIMEOUT (nessun RST, nessuna risposta — pacchetto droppato silenziosamente)
```

### Perché il monitor DMOS vedeva ~7 rps comunque

Il controller DMOS misura il traffico da **Prometheus interno** (`dmos_actual_traffic`),
che legge le metriche dai Prometheus per-cluster (`:30090`). Quella ~7 rps era
**traffico di background** (health checks, readiness probes di Kubernetes, traffico
inter-pod interno) — visibile a Prometheus ma distinto dal traffico Locust esterno.

Locust invece genera traffico **esterno** attraverso la NodePort, che:
1. Non viene mai consegnato al pod (droppato da Cilium)
2. Non compare nelle metriche Prometheus del cluster
3. Il monitor mostra sempre `locust: waiting...` perché Locust era avviato con
   `--headless` (nessuna web UI su porta 8089)

---

## 7. Diagnosi passo per passo

### Step 1 — Verifica connettività base
```powershell
curl --max-time 5 http://192.168.1.245:30007/
# Risultato: timeout → conferma che il problema è a livello di rete/policy
```

### Step 2 — Verifica tipo del Service
```powershell
kubectl --context cluster1 get svc frontend -n online-boutique
# Risultato: TYPE=ClusterIP → il frontend "main" non è esposto
# (ma frontend-external è NodePort su 30007 — trovato dopo)
```

### Step 3 — Trova quale servizio usa la porta 30007
```powershell
kubectl --context cluster1 get svc -A | Select-String "30007"
# Risultato: frontend-external NodePort 80:30007/TCP → il servizio esiste
```

### Step 4 — Verifica che Prometheus (altra NodePort) funzioni
```powershell
curl --max-time 5 http://192.168.1.245:30090
# Risultato: risponde → la rete da Windows ai nodi funziona
# Conclusione: il problema è SPECIFICO del frontend, non della rete
```

### Step 5 — Verifica gli endpoint del servizio
```powershell
kubectl --context cluster1 get endpoints frontend-external -n online-boutique
# Risultato: 10.42.0.94:8080 → il servizio punta correttamente al pod
```

### Step 6 — Identifica le Cilium NetworkPolicy
```powershell
kubectl --context cluster1 get ciliumnetworkpolicy -n online-boutique
# Risultato: l7-visibility-backends, l7-visibility-frontend
```

### Step 7 — Test definitivo: elimina la policy
```powershell
kubectl --context cluster1 delete ciliumnetworkpolicy l7-visibility-frontend -n online-boutique
curl --max-time 5 http://192.168.1.245:30007/
# Risultato: risponde con HTML! → confermato, era la CNP
```

### Step 8 — Applica la policy corretta
*(vedi sezione 8)*

---

## 8. La soluzione finale

### Policy applicata (tutti e 3 i cluster) — con Hubble L7

```yaml
# deployments/cnp-l7-frontend.yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: l7-visibility-frontend
  namespace: online-boutique
spec:
  endpointSelector:
    matchLabels:
      app: frontend

  # Nessun egress → allow all (DNS, gRPC backend, internet funzionano)
  # CRITICO: non aggiungere egress → Bug #2/#3 (ClusterIP non ha identità Cilium)

  ingress:
  # fromEntities:cluster → copre pod di qualsiasi namespace (incluso ingress-nginx)
  # fromEntities:host → kubelet health check
  # world NON presente → traffico esterno bloccato a L3 (prima di Envoy)
  # rules:http → attiva Envoy L7 → hubble_http_requests_total++ per ogni richiesta
  - fromEntities:
    - cluster
    - host
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - {}
```

### Applica su tutti i cluster

```powershell
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl apply -f deployments/cnp-l7-frontend.yaml --context $ctx
    kubectl rollout restart deployment/frontend -n online-boutique --context $ctx
}
Start-Sleep 30
```

### Verifica

```powershell
# Nginx (porta 30080) → risponde 200 ✅
curl.exe -o NUL -w "%{http_code}" http://192.168.1.245:30080/
curl.exe -o NUL -w "%{http_code}" http://192.168.1.246:30080/
curl.exe -o NUL -w "%{http_code}" http://192.168.1.247:30080/

# NodePort diretto (30007) → timeout (world bloccato) ✅
curl.exe --max-time 5 -o NUL -w "%{http_code}" http://192.168.1.245:30007/

# Hubble L7 attivo (con Locust in esecuzione)
kubectl exec -n kube-system `
  (kubectl get pod -n kube-system -l k8s-app=cilium --context cluster1 -o jsonpath='{.items[0].metadata.name}') `
  --context cluster1 -- hubble observe --type l7 --namespace online-boutique --last 5
# → controller (ingress-nginx) → frontend:8080 HTTP/200 GET / FORWARDED
```

---

## 9. Comportamento Cilium con NodePort

### Come funziona il path del pacchetto (kube-proxy replacement mode)

```
Client (Windows)                    Node (cluster1)
192.168.1.x:XXXXX  ──────────────► 192.168.1.245:30007
                                         │
                                    [Cilium eBPF - XDP/tc hook]
                                         │  DNAT
                                         ▼
                                    Pod 10.42.0.94:8080
                                         │
                                    [Cilium policy engine]
                                    src: 192.168.1.x → "world"
                                    dst port: 8080
                                    → valuta ingress rules
```

### Punti critici

1. **La policy viene valutata DOPO il DNAT** → la porta rilevante è 8080 (non 30007)
2. **La sorgente è classificata come `world`** → richiede `fromEntities: ["world"]`
3. **SNAT e `host`**: in alcune configurazioni Cilium fa SNAT del traffico NodePort
   usando l'IP del nodo → la sorgente diventa il nodo stesso → classificata come `host`.
   Per sicurezza, includere entrambi: `fromEntities: ["world", "host"]`

### Differenza tra `world` e `host`

| Entità | Quando usarla |
|---|---|
| `world` | Traffico da IP esterni al cluster (internet, LAN esterna, Windows) |
| `host` | Traffico originato dal nodo stesso (kubelet, NodePort con SNAT, health checks) |

In ambienti bare-metal con NodePort, è buona pratica includere **entrambi**.

---

## 10. OOMKilled: currencyservice e paymentservice

### Problema rilevato

```
currencyservice: 81 restarts, Reason: OOMKilled, Limit: memory: 128Mi
paymentservice:  75 restarts, Reason: OOMKilled, Limit: memory: 128Mi
```

### Causa

Entrambi i servizi sono scritti in **Node.js**. Il runtime V8 di Node.js alloca
memoria in modo aggressivo sotto carico (heap, buffer, compilazione JIT).
128Mi è sufficiente a riposo ma insufficiente con traffico sostenuto da Locust
**e** dal `loadgenerator` (che gira in parallelo).

### Fix applicato

```powershell
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    # currencyservice
    kubectl --context $ctx patch deployment currencyservice -n online-boutique `
      -p '{"spec":{"template":{"spec":{"containers":[{"name":"server","resources":
      {"limits":{"memory":"256Mi"},"requests":{"memory":"128Mi"}}}]}}}}'

    # paymentservice
    kubectl --context $ctx patch deployment paymentservice -n online-boutique `
      -p '{"spec":{"template":{"spec":{"containers":[{"name":"server","resources":
      {"limits":{"memory":"256Mi"},"requests":{"memory":"128Mi"}}}]}}}}'
}
```

### Perché fermare il loadgenerator durante i test

Il pod `loadgenerator` è il generatore di traffico built-in di Online Boutique.
Se gira mentre esegui Locust:

| Problema | Spiegazione |
|---|---|
| Metriche impure | DMOS vede traffico misto (loadgenerator + Locust) e non può distinguerli |
| OOM accelerato | Doppio carico sui servizi Node.js → OOMKilled più frequente |
| Scaling spurio | DMOS potrebbe scalare in risposta al loadgenerator, non a Locust |
| Risultati non riproducibili | Il baseline non è zero, varia con il comportamento del loadgenerator |

```powershell
# Prima di ogni test Locust: ferma il loadgenerator
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx scale deployment loadgenerator `
      -n online-boutique --replicas=0
}

# Dopo il test: riavvialo se vuoi
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx scale deployment loadgenerator `
      -n online-boutique --replicas=1
}
```

---

## 11. Checklist pre-test

Prima di lanciare un test Locust multiingress, verifica:

```powershell
# 1. Nginx (porta 30080) raggiungibile da Windows
curl.exe -o NUL -w "%{http_code}" http://192.168.1.245:30080/   # → 200
curl.exe -o NUL -w "%{http_code}" http://192.168.1.246:30080/   # → 200
curl.exe -o NUL -w "%{http_code}" http://192.168.1.247:30080/   # → 200

# 2. NodePort diretto (30007) bloccato → timeout (comportamento corretto)
# curl.exe --max-time 5 http://192.168.1.245:30007/   # → timeout ✅

# 3. Frontend pods Running su tutti i cluster
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx get pods -n online-boutique -l app=frontend
}

# 4. Nessun pod in crash-loop
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx get pods -n online-boutique
    # Controlla RESTARTS — valori alti indicano OOMKill o crash
}

# 5. LoadGenerator spento (evita traffico misto)
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl --context $ctx get deployment loadgenerator -n online-boutique
    # DESIRED deve essere 0
}

# 6. CNP con Hubble L7 applicata (verifica rules: http presente)
kubectl --context cluster1 get cnp l7-visibility-frontend `
  -n online-boutique -o jsonpath='{.spec.ingress[0].toPorts[0].rules}'
# → {"http":[{}]}

# 7. Hubble L7 attivo — verifica flusso eventi (richiede piccolo traffico di test)
curl.exe http://192.168.1.245:30080/ > $null
kubectl exec -n kube-system `
  (kubectl get pod -n kube-system -l k8s-app=cilium --context cluster1 -o jsonpath='{.items[0].metadata.name}') `
  --context cluster1 -- hubble observe --type l7 --namespace online-boutique --last 5
# → deve mostrare eventi HTTP, non essere vuoto

# 8. Locust: usare --autostart (NON --headless) per mantenere web server :8089
# locust -f experiments/locustfile_multiingress.py `
#   --autostart --users 300 --spawn-rate 10 `
#   --web-host 0.0.0.0 --web-port 8089 --run-time 35m
# Con --headless il web server è spento → collect_metrics mostra locust: N/A
```

---

## 12. Comandi utili di riferimento

### Gestione CiliumNetworkPolicy

```powershell
# Lista tutte le CNP in un namespace
kubectl --context cluster1 get ciliumnetworkpolicy -n online-boutique

# Vedi il contenuto di una CNP
kubectl --context cluster1 get ciliumnetworkpolicy l7-visibility-frontend `
  -n online-boutique -o yaml

# Vedi solo la sezione ingress (formattata)
kubectl --context cluster1 get ciliumnetworkpolicy l7-visibility-frontend `
  -n online-boutique -o jsonpath='{.spec.ingress}' | python -m json.tool

# Applica una CNP da stringa PowerShell
$yaml = @' ... '@
$yaml | kubectl --context cluster1 apply -f -

# Applica su tutti i cluster
foreach ($ctx in @("cluster1","cluster2","cluster3")) { $yaml | kubectl --context $ctx apply -f - }

# Elimina una CNP (attenzione: torna default ALLOW per quel pod)
kubectl --context cluster1 delete ciliumnetworkpolicy l7-visibility-frontend -n online-boutique
```

### Diagnostica connettività

```powershell
# Test NodePort dall'esterno
curl -v --max-time 5 http://192.168.1.245:30007/

# Lista tutti i NodePort
kubectl --context cluster1 get svc -A --field-selector spec.type=NodePort

# Verifica endpoint di un servizio
kubectl --context cluster1 get endpoints frontend-external -n online-boutique

# Test dall'interno del cluster (bypassa firewall host, non bypassa CNP)
kubectl --context cluster1 run nettest --image=nicolaka/netshoot --rm -it `
  --restart=Never -n online-boutique -- curl -v http://10.43.195.100:80/

# Log del pod precedente (--previous = ultimo crash)
kubectl --context cluster1 logs -n online-boutique <pod-name> --previous --tail=50

# Stato dettagliato pod (cerca OOMKill)
kubectl --context cluster1 describe pod -n online-boutique <pod-name> `
  | Select-String -Pattern "State|Reason|Exit|OOM|Limit" -Context 2
```

### Gestione repliche e risorse

```powershell
# Scala un deployment
kubectl --context cluster1 scale deployment frontend -n online-boutique --replicas=2

# Modifica memory limit
kubectl --context cluster1 patch deployment currencyservice -n online-boutique `
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"server","resources":
  {"limits":{"memory":"256Mi"},"requests":{"memory":"128Mi"}}}]}}}}'

# Ferma/riavvia loadgenerator
kubectl --context cluster1 scale deployment loadgenerator -n online-boutique --replicas=0
kubectl --context cluster1 scale deployment loadgenerator -n online-boutique --replicas=1
```

---

## Glossario

| Termine | Definizione |
|---|---|
| **CNP** | CiliumNetworkPolicy — risorsa Kubernetes custom di Cilium |
| **eBPF** | Extended Berkeley Packet Filter — tecnologia kernel per processing pacchetti ad alte prestazioni usata da Cilium |
| **Envoy** | Proxy L7 open-source (usato da Istio, Cilium) per ispezione e routing HTTP/gRPC |
| **L4** | Layer 4 (Transport) — filtraggio per IP e porta |
| **L7** | Layer 7 (Application) — filtraggio per contenuto HTTP (path, metodo, header) |
| **NodePort** | Tipo di Service Kubernetes che espone una porta su ogni nodo del cluster |
| **ClusterIP** | Tipo di Service Kubernetes raggiungibile solo dall'interno del cluster |
| **DNAT** | Destination NAT — modifica l'IP/porta di destinazione di un pacchetto |
| **SNAT** | Source NAT — modifica l'IP sorgente di un pacchetto |
| **OOMKilled** | Out Of Memory Killed — il kernel Linux termina un processo che supera il memory limit |
| **ClusterMesh** | Feature di Cilium per connettere più cluster Kubernetes con service discovery e policy condivise |
| **Hubble** | Strumento di observability di Cilium per visualizzare il traffico di rete in tempo reale |
| **default deny** | Comportamento di Cilium: se una policy seleziona un pod, tutto il traffico non esplicitamente permesso è bloccato |
