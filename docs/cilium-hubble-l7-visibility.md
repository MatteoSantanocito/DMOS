# Cilium CNP + Hubble L7 — Analisi tecnica e soluzione

> Documento aggiornato il 01/03/2026.
> Documenta il percorso di debug per abilitare Hubble L7 su cluster k3s single-node,
> i bug Cilium CNP incontrati, la diagnosi corretta del problema "same-node", e la
> soluzione finale che ha reso `hubble_http_requests_total` funzionante su tutti e
> 3 i cluster.

---

## Indice

1. [Obiettivo e risultato finale](#1-obiettivo-e-risultato-finale)
2. [Come funziona Hubble L7 con Nginx Ingress](#2-come-funziona-hubble-l7-con-nginx-ingress)
3. [Diagnosi del problema: contatore congelato](#3-diagnosi-del-problema-contatore-congelato)
4. [Bug #1: fromEndpoints è namespace-scoped](#4-bug-1-fromendpoints-è-namespace-scoped)
5. [Bug #2: toEntities:cluster non copre i Service ClusterIP](#5-bug-2-toentitiescluster-non-copre-i-service-clusterip)
6. [Bug #3: ANY regola L7 su una porta attiva Envoy globalmente](#6-bug-3-any-regola-l7-su-una-porta-attiva-envoy-globalmente)
7. [La CNP finale funzionante](#7-la-cnp-finale-funzionante)
8. [Perché same-node L7 funziona (diagnosi corretta)](#8-perché-same-node-l7-funziona-diagnosi-corretta)
9. [Verifica stato Hubble e CNP](#9-verifica-stato-hubble-e-cnp)
10. [Riepilogo errori e fix](#10-riepilogo-errori-e-fix)

---

## 1. Obiettivo e risultato finale

### Obiettivo
Abilitare `hubble_http_requests_total` per il frontend di ogni cluster, in modo che
DMOS possa misurare il traffico HTTP reale per-cluster (contatore esatto invece di
stima da network bytes).

### Risultato finale: ✅ Hubble L7 funziona

```
✅ Traffic from Hubble (cluster1): 7.2 req/s
✅ Traffic from Hubble (cluster2): 6.3 req/s
✅ Traffic from Hubble (cluster3): 4.7 req/s
```

**Verifica `hubble observe`** (con Locust attivo):
```
controller (ingress-nginx/controller) → online-boutique/frontend:8080
    HTTP/1.1 200 18ms GET /
controller (ingress-nginx/controller) → online-boutique/frontend:8080
    HTTP/1.1 200 22ms GET /product/OLJCESPC7Z0
controller (ingress-nginx/controller) → online-boutique/frontend:8080
    HTTP/1.1 200 31ms POST /cart
```

### Perché è stato possibile

La "limitazione same-node" documentata in precedenza era un'**errata diagnosi**.
Il vero problema era diverso (vedi §8). Con la CNP corretta:
- `rules: http: - {}` attiva Envoy L7 sul pod frontend
- Il traffico `world` (Locust esterno) è bloccato a **L3** prima di raggiungere Envoy
- Il traffico `cluster` (Nginx → Frontend, anche same-node) funziona con Envoy ✅

---

## 2. Come funziona Hubble L7 con Nginx Ingress

### Path completo del traffico

```
Locust → 192.168.1.X:30080 (Nginx NodePort, entity=world)
               ↓
         Nginx pod (ingress-nginx namespace, entity=cluster)
               ↓ service-upstream=true → usa ClusterIP
         frontend Service ClusterIP (10.43.x.x:80)
               ↓ Cilium DNAT
         frontend pod :8080
               ↓
         CNP: fromEntities[cluster,host] → ALLOWED
         CNP: rules: http: [{}] → ATTIVA ENVOY L7
               ↓
         Envoy L7 proxy (intercetta e ispeziona HTTP)
               ↓
         Hubble osserva → hubble_http_requests_total++
               ↓
         Prometheus scrapa Hubble-metrics ogni ~60s
               ↓
         DMOS query: rate(hubble_http_requests_total{...}[5m])
```

### Cosa genera Envoy per Hubble

Per ogni richiesta HTTP, Hubble registra:
- `source_workload`: "controller" (Nginx)
- `destination_workload`: "frontend"
- `destination_namespace`: "online-boutique"
- `reporter`: "server" (catturato lato Envoy del destinatario)
- `http_method`, `http_status_code`, `http_protocol`
- `hubble_http_request_duration_seconds_bucket` (histogramma latenza)

---

## 3. Diagnosi del problema: contatore congelato

### Sintomo iniziale

Con Locust attivo a ~33 req/s, la query Prometheus mostrava:

```promql
sum(rate(hubble_http_requests_total{destination_namespace="online-boutique"}[5m]))
→ 0  oppure  NaN
```

E `query_range` su 5 minuti mostrava lo **stesso valore** (71,038) in tutti
gli 11 data point → il contatore era **congelato** nonostante il traffico.

### Causa radice

`hubble observe --type l7 -n online-boutique --last 20` restituiva **zero eventi L7**.
Solo eventi di tipo `[TCP SYN, FORWARDED]` — nessun evento HTTP.

La CNP attiva aveva solo regole **L3/L4** (TCP):
```yaml
ingress:
- fromEntities: [cluster, host]
  toPorts:
  - ports: [{port: "8080", protocol: TCP}]
  # ← NESSUN rules: http → nessun Envoy → Hubble non vede HTTP
```

**Senza `rules: http`**, Cilium NON attiva il proxy Envoy. Senza Envoy, Hubble
non può intercettare e contare le richieste HTTP. Il contatore rimane congelato.

### Fix

Aggiungere `rules: http: - {}` alla CNP (vedi §7).

---

## 4. Bug #1: fromEndpoints è namespace-scoped

### Comportamento

In una `CiliumNetworkPolicy` (tipo namespaced, non `CiliumClusterwideNetworkPolicy`),
i selettori `fromEndpoints` e `toEndpoints` sono **scoped al namespace del policy**.

```yaml
# CiliumNetworkPolicy nel namespace "online-boutique"

# ❌ Questo selector matcha SOLO pod in "online-boutique"
fromEndpoints:
- {}   # selettore vuoto = qualsiasi pod... MA solo nel proprio namespace!

# ✅ Questo copre pod di QUALSIASI namespace del cluster
fromEntities:
- cluster
```

### Perché causa problemi

**Ingress**: Nginx è nel namespace `ingress-nginx`. Con `fromEndpoints: {}` sulla CNP
del frontend (namespace `online-boutique`), Nginx era bloccato a L3 → 504.

**Conseguenza per Hubble**: Nginx bloccato a L3 → non arriva a Envoy → nessun
evento L7 → `hubble_http_requests_total` non si incrementa.

### Regola

| Selettore | Scope | Usa quando |
|-----------|-------|------------|
| `fromEndpoints: [{}]` | Solo namespace corrente | Vuoi permettere solo pod dello stesso namespace |
| `fromEntities: cluster` | Tutti i pod del cluster | Vuoi permettere Nginx, DMOS, altri namespace |
| `fromEntities: host` | Il nodo stesso | Kubelet health check, processi di sistema |

---

## 5. Bug #2: toEntities:cluster non copre i Service ClusterIP

### Comportamento

`toEntities: cluster` in Cilium copre i **pod IP** con identità Cilium managed.
NON copre i **Kubernetes Service ClusterIP** (IP virtuali nel range 10.43.0.0/16).

```
Frontend → DNS → 10.43.0.10:53 (kube-dns ClusterIP)
                      ↑
         ClusterIP = IP virtuale, nessuna identità Cilium
         toEntities:cluster non matcha → BLOCCATO → i/o timeout

Frontend → gRPC → 10.43.234.98:3550 (productcatalog ClusterIP)
                       ↑
              Stesso problema → BLOCCATO → HTTP 500
```

### Soluzione

**Rimuovere completamente la sezione egress** dalla CNP. Senza `egress` esplicito,
Cilium usa il default **allow-all egress** — tutto il traffico in uscita è permesso.

```yaml
spec:
  endpointSelector:
    matchLabels:
      app: frontend

  # ← NON mettere la sezione egress
  # Il default è: allow all egress (DNS, gRPC, internet → tutti funzionano)

  ingress:
  - fromEntities: [cluster, host]
    toPorts:
    - ports: [{port: "8080", protocol: TCP}]
      rules:
        http:
        - {}
```

---

## 6. Bug #3: ANY regola L7 su una porta attiva Envoy globalmente

### Comportamento

Quando una CNP ha `rules: http: [{}]` su porta 8080 per **qualsiasi** sorgente,
Cilium attiva il proxy Envoy per **TUTTO** il traffico su porta 8080 — non solo
per la sorgente specifica della regola.

```yaml
ingress:
# Regola A: cluster con L7
- fromEntities: [cluster]
  toPorts:
  - ports: [{port: "8080"}]
    rules: {http: [{}]}     # ← attiva Envoy globalmente su :8080

# Regola B: world senza L7 (ILLUSIONE — passa comunque per Envoy)
- fromEntities: [world]
  toPorts:
  - ports: [{port: "8080"}] # ← anche questo traffico passa per Envoy
```

### Conseguenze

- Non è possibile avere "L7 per alcuni sorgenti, TCP puro per altri" sulla stessa porta
- Qualsiasi `rules: http:` su porta 8080 → Envoy attivato per tutto il traffico su :8080

### Come viene gestito nella CNP corretta

Nella CNP finale, il problema è risolto alla radice: **il traffico `world` non
è presente in `fromEntities`**, quindi viene bloccato a L3 prima di raggiungere
la fase di valutazione Envoy. Envoy riceve solo `cluster` e `host`.

```yaml
ingress:
- fromEntities: [cluster, host]   # ← world NON è qui → bloccato a L3
  toPorts:
  - ports: [{port: "8080"}]
    rules: {http: [{}]}            # ← Envoy attivo solo per cluster/host
```

---

## 7. La CNP finale funzionante

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

  # Nessun egress → allow all (DNS, gRPC, internet funzionano)
  # Rimuovere egress è la soluzione al Bug #2 (ClusterIP non ha identità Cilium)

  ingress:
  - fromEntities:
    - cluster   # tutti i pod del cluster (incluso ingress-nginx, qualsiasi namespace)
    - host      # nodo locale (kubelet health check, processi di sistema)
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - {}   # regola HTTP vuota = allow all HTTP
               # EFFETTO: attiva il proxy Envoy L7 sul pod frontend
               # RISULTATO: Hubble intercetta e conta ogni richiesta HTTP
               # SICUREZZA: world è assente → traffico esterno bloccato a L3
```

### Applica su tutti i cluster

```powershell
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    kubectl apply -f deployments/cnp-l7-frontend.yaml --context $ctx
    kubectl rollout restart deployment/frontend -n online-boutique --context $ctx
}
# Aspetta ~30s, poi verifica
Start-Sleep 30
```

### Verifica

```powershell
# 1. HTTP 200 via Nginx (cluster entity → allowed → Envoy)
curl.exe -o NUL -w "%{http_code}" http://192.168.1.245:30080/   # → 200

# 2. Timeout via NodePort diretto (world entity → bloccato a L3)
curl.exe --max-time 5 -o NUL -w "%{http_code}" http://192.168.1.245:30007/  # → timeout

# 3. Flusso L7 in Hubble
kubectl exec -n kube-system <cilium-pod> --context cluster1 `
  -- hubble observe --type l7 --namespace online-boutique --last 10
# → controller (ingress-nginx) → frontend:8080 HTTP/1.1 200 GET /

# 4. Metrica Prometheus
(Invoke-WebRequest "http://192.168.1.245:30090/api/v1/query?query=sum(rate(hubble_http_requests_total{destination_workload=`"frontend`"}[5m]))" -UseBasicParsing).Content
# → {"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[...,"7.2"]}]}}
```

---

## 8. Perché same-node L7 funziona (diagnosi corretta)

### La precedente diagnosi errata

La documentazione precedente affermava che Hubble L7 non funzionasse in cluster
k3s single-node a causa di una "limitazione same-node" di Cilium:

> *"Il redirect eBPF che Cilium usa per instradare le connessioni verso Envoy non
> funziona correttamente per il percorso loopback same-node."*

Questa diagnosi era **errata**. La prova empirica citata (Nginx→Frontend senza
eventi L7 in Hubble) era conseguenza del Bug #1: Nginx era bloccato a L3
da `fromEndpoints: {}` namespace-scoped, e non raggiungeva mai Envoy.

### La diagnosi corretta

Il vero problema era **doppio**:

1. **Bug #1** (`fromEndpoints: {}` namespace-scoped): Nginx (in `ingress-nginx`)
   veniva bloccato a L3 dalla CNP del frontend (in `online-boutique`). Nginx
   non arrivava mai a Envoy → nessun evento L7.

2. **Bug #3** (L7 su porta 8080 attiva Envoy per tutto): in una versione
   intermedia della CNP, era presente `fromEntities: world` sulla stessa porta
   con L7 → `world` traffic (Locust via NodePort 30007) passava per Envoy →
   timeout perché Envoy non gestisce correttamente connessioni `world` su NodePort.

### Perché il traffico same-node pod-to-pod funziona con Envoy

Il redirect eBPF di Cilium per Envoy L7 gestisce correttamente il traffico tra
pod sullo stesso nodo **quando la sorgente è un'entità Cilium-managed** (`cluster`,
`host`, o endpoint con identità).

Il problema originale era specifico del traffico `world` (IP esterno che arriva
via NodePort con DNAT) attraverso Envoy — questa combinazione causa timeout.

Con la CNP corretta:
- `world` → bloccato a L3 (mai raggiunge Envoy)
- `cluster` (Nginx pod, anche same-node) → Envoy gestisce correttamente → L7 ✅
- `host` (kubelet) → Envoy gestisce correttamente → L7 ✅

### Prova empirica (stato attuale)

```
hubble observe --type l7 -n online-boutique:

TIMESTAMP           SOURCE                    DESTINATION             TYPE      VERDICT
28/02/2026 22:15:04 ingress-nginx/controller  online-boutique/frontend HTTP/200  FORWARDED
28/02/2026 22:15:04 ingress-nginx/controller  online-boutique/frontend HTTP/200  FORWARDED
28/02/2026 22:15:05 ingress-nginx/controller  online-boutique/frontend HTTP/200  FORWARDED
...
(centinaia di eventi al secondo — Hubble L7 funziona ✅)
```

---

## 9. Verifica stato Hubble e CNP

### Controlla CNP attive

```powershell
kubectl get cnp -A --context cluster1
# Expected:
# NAMESPACE         NAME                     AGE     VALID
# online-boutique   l7-visibility-frontend   10m     True
```

### Verifica che la CNP abbia rules: http

```powershell
kubectl get cnp l7-visibility-frontend -n online-boutique `
  --context cluster1 -o jsonpath='{.spec.ingress[0].toPorts[0].rules}'
# → {"http":[{}]}
```

### Controlla contatore Hubble in tempo reale

```powershell
# Valore attuale del contatore (deve essere > 0 e in aumento con Locust attivo)
(Invoke-WebRequest `
  "http://192.168.1.245:30090/api/v1/query?query=sum(hubble_http_requests_total{destination_workload=`"frontend`"})" `
  -UseBasicParsing).Content

# Rate negli ultimi 5 minuti (deve essere > 0 con Locust attivo)
(Invoke-WebRequest `
  "http://192.168.1.245:30090/api/v1/query?query=sum(rate(hubble_http_requests_total{destination_workload=`"frontend`",destination_namespace=`"online-boutique`"}[5m]))" `
  -UseBasicParsing).Content
```

### Diagnostica se il contatore è congelato

```powershell
# 1. Verifica che la CNP abbia rules: http
kubectl get cnp l7-visibility-frontend -n online-boutique --context cluster1 -o yaml | Select-String "http"
# Deve mostrare: "- {}" sotto "http:"

# 2. Verifica Envoy è attivo (L7 proxy presente)
kubectl exec -n kube-system <cilium-pod> --context cluster1 `
  -- cilium-dbg endpoint list | Select-String "frontend"
# Cerca "Policy Enabled: Ingress" e verifica che L7 sia indicato

# 3. Verifica flusso L7 in Hubble
kubectl exec -n kube-system <cilium-pod> --context cluster1 `
  -- hubble observe --type l7 --namespace online-boutique --last 5
# Se vuoto: Envoy non attivo → verifica CNP

# 4. Se il contatore non cresce dopo aver verificato la CNP
kubectl rollout restart deployment/frontend -n online-boutique --context cluster1
# Aspetta 30s poi ricontrolla
```

---

## 10. Riepilogo errori e fix

| Errore osservato | Causa | Fix |
|-----------------|-------|-----|
| `hubble_http_requests_total` congelato | CNP senza `rules: http` → no Envoy → no L7 | Aggiungi `rules: http: - {}` alla CNP |
| 504 Nginx → Frontend | L7 rule con `fromEntities: world` → Envoy su NodePort → timeout | Rimuovi `world` da `fromEntities`, usa solo `cluster, host` |
| 504 con `fromEndpoints: {}` | Namespace-scoped → Nginx (ingress-nginx) bloccato a L3 | Usa `fromEntities: cluster` |
| 500 dopo CNP applicata | Sezione egress con `toEndpoints: {}` → DNS bloccato (kube-system) | Rimuovi sezione egress (allow all) |
| 500 con `toEntities: cluster` | ClusterIP non ha identità Cilium → gRPC backend bloccato | Rimuovi sezione egress (allow all) |
| Connessione reset gRPC | Connessioni stantie dopo cambio CNP | `rollout restart deployment/frontend` |
| Nginx usa pod IP diretto | Default Nginx Ingress → potenziali problemi con CNP | Annotation `service-upstream: "true"` |
| rate() = 0 intermittente | Finestra `[1m]` con scrape ~60s → 0-1 sample → rate instabile | Usa `[5m]` (4-5 sample garantiti) |
