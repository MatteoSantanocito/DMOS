# Nginx Ingress + CiliumNetworkPolicy + Hubble L7 — Architettura e Problemi Risolti

> Documento aggiornato il 01/03/2026.
> Descrive il percorso completo di debug per portare Nginx Ingress a funzionare
> con Cilium CNP su cluster k3s single-node, e come Hubble L7 è stato abilitato
> come sorgente primaria di metriche HTTP per DMOS.

---

## Indice

1. [Obiettivo: perché Nginx Ingress](#1-obiettivo-perché-nginx-ingress)
2. [Il problema originale: NodePort + Envoy = timeout](#2-il-problema-originale-nodeport--envoy--timeout)
3. [Evoluzione della soluzione: da no-L7 a Hubble L7](#3-evoluzione-della-soluzione-da-no-l7-a-hubble-l7)
4. [Tre bug nei CiliumNetworkPolicy scoperti durante il debug](#4-tre-bug-nei-ciliumnetworkpolicy-scoperti-durante-il-debug)
5. [La CNP finale con Hubble L7](#5-la-cnp-finale-con-hubble-l7)
6. [Nginx: annotation service-upstream obbligatoria](#6-nginx-annotation-service-upstream-obbligatoria)
7. [Hubble L7 come sorgente primaria per DMOS](#7-hubble-l7-come-sorgente-primaria-per-dmos)
8. [Stack di rete finale](#8-stack-di-rete-finale)
9. [Cosa cambia per Locust e per DMOS](#9-cosa-cambia-per-locust-e-per-dmos)
10. [Riepilogo comandi di deploy e verifica](#10-riepilogo-comandi-di-deploy-e-verifica)

---

## 1. Obiettivo: perché Nginx Ingress

Il setup DMOS originale esponeva il frontend tramite **NodePort diretto** (porta 30007).
Volevamo abilitare metriche HTTP L7 (`hubble_http_requests_total`) per DMOS.

In Cilium, L7 richiede l'attivazione del proxy **Envoy** tramite `rules: http` nella CNP.
Con NodePort esterno + Envoy attivo, il traffico da `world` (Locust) andava in timeout
perché Envoy non gestisce correttamente connessioni da indirizzi IP esterni su NodePort.

**Nginx Ingress** risolve il problema strutturalmente:
- Locust parla con Nginx (porta 30080) — Nginx è `world` entity, CNP lo lascia passare
- Nginx parla con frontend — Nginx diventa `cluster` entity, Envoy lo gestisce ✅
- Il frontend NodePort diretto (30007) viene bloccato dalla CNP

**Risultato finale**: Nginx funziona, Hubble L7 funziona, metriche HTTP esatte in DMOS.

---

## 2. Il problema originale: NodePort + Envoy = timeout

### Come funziona NodePort senza L7

```
Locust (192.168.1.x) → TCP :30007 → eBPF DNAT → frontend-pod:8080
```

Funziona perché Cilium gestisce il DNAT in kernel space (eBPF) senza interazione
user-space. Nessun proxy intermedio.

### Cosa succede con `rules: http:` (Hubble L7) su porta 8080

Quando la CNP ha `rules: http: [{}]` su porta 8080, Cilium attiva il proxy **Envoy**
per quella porta. Cilium lo attiva per **tutto il traffico sulla porta** — non solo
per la regola specifica che contiene il blocco `http:`.

```
Locust → NodePort:30007 → DNAT → frontend:8080
                                      ↓
            Cilium: "porta 8080 ha L7 rule → redirect a Envoy"
                                      ↓
            Envoy riceve: src=192.168.1.x (IP esterno, entity="world")
            Envoy non gestisce correttamente traffico "world" su socket
            progettati per traffico cluster-interno (pod-to-pod)
                                      ↓
                          TCP timeout (no RST, drop silenzioso)
```

### Perché non si può "isolare" L7 per regola

Un tentativo intuitivo: avere L7 solo su una regola e TCP puro su un'altra:

```yaml
# ❌ NON funziona — Envoy viene attivato su TUTTA la porta 8080
ingress:
- fromEndpoints: [{}]
  toPorts:
  - ports: [{port: "8080"}]
    rules: {http: [{}]}    # ← questa riga attiva Envoy globalmente su :8080

- fromEntities: [world]
  toPorts:
  - ports: [{port: "8080"}]
    # anche questo traffico passa per Envoy → timeout
```

Questa è una **limitazione di implementazione di Cilium**: una regola L7 su una porta
attiva Envoy per tutti i pacchetti su quella porta, indipendentemente dalla regola
che fa match.

**Soluzione**: non mettere `world` in `fromEntities`. Il traffico esterno viene
bloccato a L3 prima di raggiungere Envoy.

---

## 3. Evoluzione della soluzione: da no-L7 a Hubble L7

### Fase 1 — Nginx senza L7 (soluzione temporanea)

Prima implementazione: CNP senza `rules: http` per evitare i timeout Envoy.

```yaml
# CNP intermedia (solo TCP, no Envoy)
ingress:
- fromEntities: [cluster, host]
  toPorts:
  - ports: [{port: "8080", protocol: TCP}]
  # Nessun rules:http → nessun Envoy → no timeout
  # MA: hubble_http_requests_total = 0 (Envoy non attivo)
```

**Problema residuo**: `hubble_http_requests_total` restava congelato a 0.
DMOS ricadeva sul fallback network bytes per misurare il traffico.

### Fase 2 — Debug del contatore congelato

Con Locust a ~33 req/s, il contatore Hubble mostrava lo stesso valore (71,038)
per 5+ minuti. `hubble observe --type l7` → zero eventi L7.

**Diagnosi**: senza `rules: http`, Cilium non attiva Envoy → Hubble non intercetta
nulla → contatore non si incrementa.

### Fase 3 — CNP con L7 (soluzione finale)

Aggiungendo `rules: http: - {}` e rimuovendo `world` da `fromEntities`:

```yaml
ingress:
- fromEntities: [cluster, host]   # world assente → bloccato a L3
  toPorts:
  - ports: [{port: "8080"}]
    rules:
      http:
      - {}   # ← attiva Envoy → Hubble L7 → metriche HTTP esatte
```

**Risultato**: flusso massiccio di eventi L7 in Hubble:
```
hubble observe --type l7:
ingress-nginx/controller → online-boutique/frontend:8080 HTTP/200 18ms GET /
ingress-nginx/controller → online-boutique/frontend:8080 HTTP/200 22ms GET /product/ID
...
```

`hubble_http_requests_total` si aggiorna in tempo reale → DMOS usa questa sorgente ✅

---

## 4. Tre bug nei CiliumNetworkPolicy scoperti durante il debug

### Bug 1: `fromEndpoints: {}` è namespace-scoped

In una `CiliumNetworkPolicy` (namespaced), il selettore `fromEndpoints: [{}]`
matcha solo pod nello **stesso namespace** del policy (`online-boutique`).
Nginx è nel namespace `ingress-nginx` → bloccato.

```yaml
# ❌ NON permette Nginx (namespace: ingress-nginx)
fromEndpoints:
- {}   # matcha solo pod in online-boutique

# ✅ Permette pod di QUALSIASI namespace del cluster
fromEntities:
- cluster
```

**Effetto**: con `fromEndpoints: {}`, Nginx era bloccato a L3 → 504.
La mancanza di eventi L7 in Hubble era conseguenza di questo blocco L3,
non di una limitazione same-node di Envoy.

### Bug 2: `toEndpoints: {}` è namespace-scoped (stesso problema lato egress)

La CNP egress con `toEndpoints: [{}]` bloccava kube-dns (in `kube-system`):

```
Frontend → DNS lookup "cartservice" → 10.43.0.10:53 (kube-dns ClusterIP)
                                           ↓
         CNP egress: "toEndpoints:{}" = solo pod in online-boutique
         kube-dns è in kube-system → BLOCCATO
         → i/o timeout su DNS → HTTP 500 dal frontend
```

### Bug 3: `toEntities: cluster` non copre i Service ClusterIP

Anche dopo aver cambiato egress a `toEntities: cluster`, DNS e gRPC backend
continuavano a fallire:

```
Frontend → kube-dns ClusterIP (10.43.0.10:53) → BLOCKED
                    ↑
    ClusterIP = IP virtuale senza identità Cilium
    toEntities:cluster copre solo pod IP (Cilium-managed endpoints)
    I ClusterIP (10.43.0.0/16) NON hanno identità Cilium → non matchano
```

**Soluzione**: rimuovere completamente la sezione egress. In Cilium, senza una sezione
egress, il default è **allow-all egress** — tutto il traffico in uscita è permesso,
inclusi DNS e gRPC verso ClusterIP.

---

## 5. La CNP finale con Hubble L7

```yaml
# deployments/cnp-l7-frontend.yaml (versione finale)
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: l7-visibility-frontend
  namespace: online-boutique
spec:
  endpointSelector:
    matchLabels:
      app: frontend

  # Nessuna sezione egress → default allow all egress
  # (evita i bug con ClusterIP + namespace scope descritti sopra)

  ingress:
  # fromEntities:cluster copre pod di QUALSIASI namespace (incluso ingress-nginx)
  # fromEntities:host copre il nodo (kubelet health check)
  # world (esterno) non incluso → frontend non raggiungibile direttamente
  # rules: http: [{}] → attiva Envoy L7 → Hubble conta HTTP
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

### Comportamento risultante

| Traffico | Permesso? | Envoy? | Hubble L7? | Note |
|---------|-----------|--------|-----------|------|
| Nginx → Frontend | ✅ | ✅ | ✅ | `cluster` entity |
| Kubelet health check → Frontend | ✅ | ✅ | ✅ | `host` entity |
| Locust → Frontend (porta 30007 diretta) | ❌ | ❌ | ❌ | `world` bloccato a L3 |
| Frontend → kube-dns (DNS) | ✅ | N/A | N/A | No egress → allow all |
| Frontend → backend gRPC (ClusterIP) | ✅ | N/A | N/A | No egress → allow all |

---

## 6. Nginx: annotation service-upstream obbligatoria

Nginx Ingress Controller per default bypassa il Service ClusterIP e si connette
**direttamente ai pod IP** (dai Kubernetes Endpoints). In questa configurazione:

```
Nginx pod (10.42.0.44) → TCP a frontend pod IP (10.42.0.14:8080) direttamente
```

Questo può causare problemi con Cilium CNP in certi scenari. La soluzione:

```yaml
# deployments/ingress-frontend.yaml
annotations:
  nginx.ingress.kubernetes.io/service-upstream: "true"
  # Forza Nginx a usare il ClusterIP del Service invece del pod IP diretto.
  # Il traffico via ClusterIP è gestito correttamente da Cilium:
  # Nginx (cluster entity) → ClusterIP → DNAT → frontend pod → Envoy L7 check
```

**Importante**: con `service-upstream: true`, il path è:
```
Nginx → frontend.online-boutique.svc.cluster.local:80 (ClusterIP)
          ↓ Cilium DNAT
        frontend pod :8080
          ↓ CNP: fromEntities cluster → ALLOWED → rules:http → Envoy
        Hubble L7 intercepts → hubble_http_requests_total++
```

---

## 7. Hubble L7 come sorgente primaria per DMOS

### Metrica usata

```python
# src/metrics/prometheus_client.py — get_request_rate() Try 1
query = (
    f'sum(rate(hubble_http_requests_total{{'
    f'destination_workload="{service}",'
    f'destination_namespace="{namespace}"'
    f'}}[5m]))'
)
```

### Label disponibili in hubble_http_requests_total

| Label | Valori tipici | Descrizione |
|-------|--------------|-------------|
| `destination_workload` | `"frontend"` | Deployment k8s di destinazione |
| `destination_namespace` | `"online-boutique"` | Namespace destinazione |
| `source_workload` | `"controller"` | Deployment sorgente (Nginx) |
| `source_namespace` | `"ingress-nginx"` | Namespace sorgente |
| `reporter` | `"server"` | Catturato lato Envoy del destinatario |
| `http_method` | `"GET"`, `"POST"` | Metodo HTTP |
| `http_status_code` | `"200"`, `"404"`, ... | Status code risposta |

### Perché `[5m]` e non `[1m]`

| Finestra | Scrape samples in window | Comportamento |
|---------|------------------------|---------------|
| `[1m]` | 0–1 sample (scrape ogni ~60s) | Intermittente: a volte 0 → rate()=0 → fallback |
| `[2m]` | 1–2 sample | Ancora instabile |
| `[5m]` | 4–5 sample | ✅ Stabile, coerente su tutti i cluster |

### Verifica diretta

```powershell
# Rate attuale per frontend (cluster1)
(Invoke-WebRequest `
  'http://192.168.1.245:30090/api/v1/query?query=sum(rate(hubble_http_requests_total{destination_workload="frontend",destination_namespace="online-boutique"}[5m]))' `
  -UseBasicParsing).Content | ConvertFrom-Json | Select-Object -ExpandProperty data

# Latenza p95
(Invoke-WebRequest `
  'http://192.168.1.245:30090/api/v1/query?query=histogram_quantile(0.95,sum(rate(hubble_http_request_duration_seconds_bucket{destination_namespace="online-boutique"}[5m]))by(le))*1000' `
  -UseBasicParsing).Content | ConvertFrom-Json | Select-Object -ExpandProperty data
```

---

## 8. Stack di rete finale

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Windows (192.168.1.x) — Locust                                           │
│   HTTP GET http://192.168.1.245:30080/    (porta Nginx)                  │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │ TCP :30080 (entity=world)
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Nodo ms01 (cluster1, 192.168.1.245)                                      │
│                                                                          │
│  eth0:30080 (Nginx NodePort)                                             │
│    │ Cilium eBPF DNAT → ingress-nginx-pod:80 (10.42.0.44)               │
│    ▼                                                                     │
│  nginx-ingress pod (entity=cluster dopo ingresso nel cluster)            │
│    │ service-upstream=true → DNS lookup frontend.online-boutique:80      │
│    │ → ClusterIP 10.43.87.229:80                                         │
│    ▼                                                                     │
│  Cilium eBPF DNAT ClusterIP → frontend-pod:8080 (10.42.0.14)            │
│    │                                                                     │
│    ▼ CNP evaluation:                                                     │
│      src entity: cluster (Nginx) → ALLOWED                              │
│      port 8080: rules:http → redirect a Envoy L7 proxy                  │
│    │                                                                     │
│    ▼ Envoy L7 proxy (stesso nodo, same-node ok per cluster entity)       │
│      → decodifica HTTP request/response                                  │
│      → notifica Hubble: hubble_http_requests_total{...}++                │
│    │                                                                     │
│    ▼ frontend-pod:8080                                                   │
│      → gRPC → cartservice, productcatalog, ecc. (no egress CNP → OK)    │
│      → risposta HTTP 200                                                 │
│                                                                          │
│  Prometheus (:30090) scrapa Hubble-metrics ogni ~60s                     │
│  DMOS query: rate(hubble_http_requests_total{...}[5m]) → 7.2 req/s      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Cosa cambia per Locust e per DMOS

### Per Locust

L'URL dei cluster usa la porta Nginx (30080) invece del NodePort frontend (30007):

```python
# locustfile_multiingress.py
clusters = {
    "cluster1": "http://192.168.1.245:30080",  # Nginx Ingress
    "cluster2": "http://192.168.1.246:30080",
    "cluster3": "http://192.168.1.247:30080",
}
```

Il comportamento di Locust è identico — Nginx è trasparente per le richieste HTTP.

### Per DMOS

DMOS usa Hubble L7 come sorgente primaria invece del fallback network bytes:

| Aspetto | Prima (NodePort 30007) | Dopo (Nginx 30080 + Hubble L7) |
|---------|------------------------|--------------------------------|
| Fonte metriche HTTP | Network bytes / 4000 (stima) | `hubble_http_requests_total` (esatto) |
| Precisione | ⭐⭐ ±20-30% | ⭐⭐⭐ ±5% |
| Porta frontend | 30007 (NodePort diretto) | 30080 (Nginx NodePort) |
| Hubble L7 | ❌ Timeout con Envoy | ✅ Funzionante via `cluster` entity |
| Sicurezza | Frontend raggiungibile direttamente | Frontend protetto da CNP |
| Log DMOS | `⚠️ Traffic from network (clusterN): X B/s` | `✅ Traffic from Hubble (clusterN): X.X req/s` |

---

## 10. Riepilogo comandi di deploy e verifica

### Deploy completo su tutti i cluster (da zero)

```powershell
cd C:\Users\matte\Desktop\Voda

# 1. Nginx Ingress Controller (se non già installato)
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl apply -f "https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.10.0/deploy/static/provider/baremetal/deploy.yaml" --context $ctx
}
Start-Sleep 30

# 2. Fissa porta NodePort a 30080
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl patch svc ingress-nginx-controller -n ingress-nginx --context $ctx `
        --type='json' `
        -p='[{"op":"replace","path":"/spec/ports/0/nodePort","value":30080}]'
}

# 3. Ingress resource per il frontend (con service-upstream: true)
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl apply -f deployments/ingress-frontend.yaml --context $ctx
}

# 4. CiliumNetworkPolicy con Hubble L7
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl apply -f deployments/cnp-l7-frontend.yaml --context $ctx
}

# 5. Rollout restart frontend (connessioni stantie post-CNP)
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl rollout restart deployment/frontend -n online-boutique --context $ctx
}
Start-Sleep 30
```

### Verifica funzionamento completo

```powershell
# HTTP 200 via Nginx su tutti i cluster
curl.exe -o NUL -w "%{http_code}" http://192.168.1.245:30080/
curl.exe -o NUL -w "%{http_code}" http://192.168.1.246:30080/
curl.exe -o NUL -w "%{http_code}" http://192.168.1.247:30080/

# Flusso L7 in Hubble (richiede Locust attivo)
kubectl exec -n kube-system (kubectl get pod -n kube-system -l k8s-app=cilium -o jsonpath='{.items[0].metadata.name}' --context cluster1) `
  --context cluster1 -- hubble observe --type l7 --namespace online-boutique --last 10

# Metrica Hubble su Prometheus
(Invoke-WebRequest `
  'http://192.168.1.245:30090/api/v1/query?query=sum(rate(hubble_http_requests_total{destination_workload="frontend",destination_namespace="online-boutique"}[5m]))' `
  -UseBasicParsing).Content
```

### Se il frontend torna a dare 500 dopo restart

Le connessioni gRPC diventano stantie dopo modifiche CNP:

```powershell
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl rollout restart deployment/frontend -n online-boutique --context $ctx
}
Start-Sleep 20
curl.exe -o NUL -w "%{http_code}" http://192.168.1.245:30080/
```

### Se Hubble torna a mostrare il contatore congelato

```powershell
# 1. Verifica CNP ha rules: http
kubectl get cnp l7-visibility-frontend -n online-boutique --context cluster1 -o yaml | Select-String "http"

# 2. Se manca: riapplica la CNP
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl apply -f deployments/cnp-l7-frontend.yaml --context $ctx
    kubectl rollout restart deployment/frontend -n online-boutique --context $ctx
}
```
