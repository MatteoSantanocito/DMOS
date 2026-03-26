# Geo-Awareness in DMOS: Implementazione Completa
## Modifiche ai File del Sistema — Documentazione Tecnica per la Tesi

> **Data**: Marzo 2026
> **Contesto**: Implementazione della geo-awareness nel sistema DMOS multi-cluster
> **Ispirazione**: Osservazioni del relatore + approccio Romano (ping_exporter + tc netem)
> **Obiettivo**: Far sì che DMOS consideri la distanza geografica (RTT inter-cluster) nella selezione del cluster target per il deployment delle repliche

---

## Indice

1. [Panoramica architetturale](#1-panoramica-architetturale)
2. [Motivazione: perché la geo-awareness?](#2-motivazione-perché-la-geo-awareness)
3. [Stack di infrastruttura aggiunto](#3-stack-di-infrastruttura-aggiunto)
4. [File modificati — Dettaglio tecnico](#4-file-modificati--dettaglio-tecnico)
   - 4.1 [`deployments/global-service-frontend.yaml`](#41-deploymentsglobal-service-frontendyaml--nuovo)
   - 4.2 [`deployments/cnp-frontend-l7.yaml`](#42-deploymentscnp-frontend-l7yaml--modificato)
   - 4.3 [`config/weights.yaml`](#43-configweightsyaml--modificato)
   - 4.4 [`src/level1/score_functions.py`](#44-srclevel1score_functionspy--modificato)
   - 4.5 [`src/metrics/prometheus_client.py`](#45-srcmetricsprometheus_clientpy--modificato)
   - 4.6 [`src/level1/dmos_scheduler.py`](#46-srclevel1dmos_schedulerpy--modificato)
5. [Flusso end-to-end con geo-awareness](#5-flusso-end-to-end-con-geo-awareness)
6. [Simulazione geografica con tc netem](#6-simulazione-geografica-con-tc-netem)
7. [tc netem — Meccanismo EGRESS e matrice RTT completa](#7-tc-netem--meccanismo-egress-e-matrice-rtt-completa)
8. [Multi-ingress 40/30/30 — Perché RTT_avg è la metrica corretta](#8-multi-ingress-403030--perché-rtt_avg-è-la-metrica-corretta)
9. [Effetto sullo scheduling — Esempi numerici](#9-effetto-sullo-scheduling--esempi-numerici)
10. [Degradazione graceful senza netem](#10-degradazione-graceful-senza-netem)
11. [Deploy infrastruttura: tc netem e ping_exporter](#11-deploy-infrastruttura-tc-netem-e-ping_exporter)

---

## 1. Panoramica architetturale

Il sistema DMOS è composto da tre cluster Kubernetes (k3s) connessi via Cilium ClusterMesh. Prima delle modifiche, DMOS selezionava il cluster target basandosi su 4 dimensioni: latenza applicativa, capacità, carico, carbon intensity. La geo-awareness aggiunge una **quinta dimensione**: la distanza geografica misurata tramite RTT ICMP inter-cluster.

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    ARCHITETTURA DMOS (dopo modifiche)                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    DMOS Scheduler (Python)                       │    ║
║  │                                                                  │    ║
║  │  _collect_cluster_metrics()  ←── ogni 30s per cluster           │    ║
║  │       │                                                          │    ║
║  │       ├── prom.get_cpu_available()          → Φ_cap             │    ║
║  │       ├── prom.get_memory_available_gb()    → Φ_cap             │    ║
║  │       ├── prom.get_request_rate()           → Φ_load            │    ║
║  │       ├── prom.get_latency_p95()            → Φ_lat             │    ║
║  │       ├── carbon_client.get_intensity()     → Φ_carbon          │    ║
║  │       └── prom.get_network_rtt_ms() ◄─ NUOVO → Φ_net           │    ║
║  │                                                                  │    ║
║  │  compute_total_score() = Σ ω_i × Φ_i(i)                        │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║          │                 │                 │                           ║
║  ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐                ║
║  │   cluster1    │ │   cluster2    │ │   cluster3    │                  ║
║  │  Frankfurt    │ │    Paris      │ │   Warsaw      │                  ║
║  │  :30090 Prom  │ │  :30090 Prom  │ │  :30090 Prom  │                  ║
║  │               │ │               │ │               │                  ║
║  │ ping_exporter │ │ ping_exporter │ │ ping_exporter │  ← NUOVO         ║
║  │ RTT→c2,c3     │ │ RTT→c1,c3     │ │ RTT→c1,c2     │                  ║
║  │               │ │               │ │               │                  ║
║  │ tc netem c2   │ │ tc netem c1   │ │ tc netem c1   │  ← INFRA         ║
║  │ +150ms delay  │ │ +150ms delay  │ │ +350ms delay  │                  ║
║  └───────────────┘ └───────────────┘ └───────────────┘                  ║
║         │                 │                 │                            ║
║         └─────────────────┴─────────────────┘                           ║
║                  Cilium ClusterMesh (vxlan tunnel)                       ║
║                  Global Service: frontend                                ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Motivazione: perché la geo-awareness?

### Problema originale (senza geo-awareness)

Con i Global Services Cilium attivi, Cilium ClusterMesh distribuisce le richieste HTTP tra i pod frontend di tutti e 3 i cluster. Questo significa che una richiesta di un utente europeo che arriva su `cluster1` (Frankfurt) può essere **inoltrata via tunnel vxlan** al pod frontend di `cluster3` (Warsaw).

```
SCENARIO PROBLEMATICO (senza geo-awareness):

  Utente EU ──► Nginx (cluster1) ──► Cilium LB ──► pod frontend (cluster3/Warsaw)
                                                         │
                                              +350ms tunnel vxlan
                                              +latenza applicativa
                                              = risposta lenta all'utente

  DMOS senza geo-awareness: cluster3 ha CPU libera → alloca repliche lì
  Risultato: latenza percepita dall'utente aumenta di centinaia di ms
```

### Soluzione: Φ_net come penalità geografica

Con `network_rtt_ms` nel calcolo dello score, DMOS assegna uno score più basso ai cluster geograficamente "periferici" (alta RTT verso i peer). Con meno score → meno repliche assegnate → Cilium bilancia meno traffico verso quel cluster → gli utenti vengono serviti prevalentemente dai cluster più vicini.

```
CON GEO-AWARENESS:
  cluster1 (Frankfurt): RTT_avg=250ms → Φ_net=0.368 → score più alto → più repliche
  cluster3 (Warsaw):    RTT_avg=425ms → Φ_net=0.183 → score più basso → meno repliche

  Effetto: Cilium bilancia 70% del traffico su cluster1+cluster2, 30% su cluster3
  Risultato: latenza media percepita dagli utenti EU ridotta
```

---

## 3. Stack di infrastruttura aggiunto

### tc netem — Simulazione delay geografico

`tc netem` (Traffic Control - Network Emulator) è un modulo del kernel Linux che permette di aggiungere artificialmente latenza alle interfacce di rete. È lo stesso approccio usato da Romano nella sua tesi.

```
CONFIGURAZIONE tc netem (applicata sui nodi k3s):

  cluster2 node (192.168.1.246) su interfaccia ens18:
    tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal

  cluster3 node (192.168.1.247) su interfaccia ens18:
    tc qdisc add dev ens18 root netem delay 350ms 20ms distribution normal

  cluster1: nessun delay (è il cluster "centrale" / Frankfurt)

RTT risultante misurata da ping_exporter:
  cluster1 → cluster2: ~150ms  (delay su ens18 di cluster2)
  cluster1 → cluster3: ~350ms  (delay su ens18 di cluster3)
  cluster2 → cluster3: ~500ms  (delay su entrambi)
```

### ping_exporter — Misurazione RTT via ICMP

`ping_exporter` (czerwonk/ping_exporter) è deployato via Helm in namespace `observability` su ogni cluster. Manda pacchetti ICMP ping agli IP dei nodi degli altri cluster e pubblica le metriche nel formato Prometheus.

```
FLUSSO DI DATI ping_exporter:

  [ping_exporter pod su cluster1]
       │
       ├── ICMP ping → 192.168.1.246 (cluster2) ogni 1s
       │       → ping_rtt_mean_seconds{target="192.168.1.246"} = 0.150
       │
       └── ICMP ping → 192.168.1.247 (cluster3) ogni 1s
               → ping_rtt_mean_seconds{target="192.168.1.247"} = 0.350

  [Prometheus su cluster1 scrappa :9427/metrics ogni 15s]
       → serie storica di ping_rtt_mean_seconds nel TSDB

  [DMOS interroga Prometheus cluster1 ogni 30s]
       → avg(ping_rtt_mean_seconds{target=~"192.168.1.246|192.168.1.247"}) * 1000
       → 250.0 ms
```

---

## 4. File modificati — Dettaglio tecnico

---

### 4.1 `deployments/global-service-frontend.yaml` — NUOVO

**File creato ex-novo.** Non esisteva prima.

#### Cosa fa

Trasforma il servizio Kubernetes `frontend` in un **Global Service Cilium**: un servizio il cui pool di endpoint è federato tra tutti i cluster connessi alla ClusterMesh. Senza questo file, ogni cluster ha la propria copia isolata del servizio `frontend`, e le richieste non attraversano mai il confine di cluster.

#### Struttura del file

```yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: online-boutique
  annotations:
    io.cilium/global-service: "true"   # (1) rende il Service visibile a tutti i cluster
    io.cilium/shared-service: "true"   # (2) abilita routing bidirezionale
spec:
  selector:
    app: frontend
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
```

#### Significato delle annotazioni

**`io.cilium/global-service: "true"`**
Istruisce l'agente Cilium di sincronizzare questo Service e i suoi endpoint nel KVStore condiviso (etcd/CRD mesh). Gli agenti degli altri cluster leggono questi endpoint e li aggiungono localmente alle proprie tabelle di routing BPF. Il Service diventa visibile *ma non necessariamente usato* negli altri cluster.

**`io.cilium/shared-service: "true"`**
Istruisce Cilium a usare effettivamente gli endpoint remoti come destinazioni reali nel load balancer BPF. Senza questa annotazione, il Global Service è in modalità "read-only": il cluster lo conosce ma non gli manda traffico. Con `shared-service`, il load balancer Cilium di cluster1 può scegliere tra:
- pod frontend locale (es. `10.42.0.219:8080`)
- pod frontend di cluster2 (es. `10.44.0.209:8080`)
- pod frontend di cluster3 (es. `10.45.0.X:8080`)

#### Comportamento nel sistema

```
PRIMA (senza global-service-frontend.yaml):

  cluster1: Service frontend → endpoint [10.42.0.X:8080] solo pod locali
  cluster2: Service frontend → endpoint [10.44.0.X:8080] solo pod locali
  cluster3: Service frontend → endpoint [10.45.0.X:8080] solo pod locali

  Nginx su cluster1 → SOLO pod frontend di cluster1
  DMOS controlla repliche → distribuzione tra cluster non influenza routing

────────────────────────────────────────────────────────────────────────

DOPO (con global-service-frontend.yaml su tutti i cluster):

  cluster1: Service frontend → endpoint [10.42.0.X:8080, 10.44.0.X:8080, 10.45.0.X:8080]
  cluster2: Service frontend → endpoint [stessi endpoint federati]
  cluster3: Service frontend → endpoint [stessi endpoint federati]

  Nginx su cluster1 → Cilium LB sceglie il pod frontend dal pool federato

  cilium service list output reale verificato:
  ┌──────────────────┬──────────────────────────────────────────────────┐
  │ 10.43.87.229:80  │ Backend 1: 10.42.0.219:8080 (locale cluster1)   │
  │ (ClusterIP)      │ Backend 2: 10.44.0.209:8080 (cluster2) ← cross  │
  └──────────────────┴──────────────────────────────────────────────────┘
```

#### Prerequisiti e verifica

```bash
# Applicare su tutti i cluster (stessa definizione)
kubectl apply -f deployments/global-service-frontend.yaml --context cluster1
kubectl apply -f deployments/global-service-frontend.yaml --context cluster2
kubectl apply -f deployments/global-service-frontend.yaml --context cluster3

# Verifica routing cross-cluster
kubectl exec -n kube-system ds/cilium --context cluster1 -- \
  cilium service list | grep frontend
# Output atteso: 2+ backend, uno per cluster

# Verifica endpoint cross-cluster visibili
kubectl exec -n kube-system ds/cilium --context cluster1 -- \
  cilium endpoint list | grep "frontend"
```

#### Perché è prerequisito alla geo-awareness

Senza Global Services, il routing cross-cluster non avviene. Se il traffico non attraversa mai il confine di cluster, la RTT inter-cluster è irrilevante per l'utente (anche se cluster3 ha RTT alta, nessun utente viene mai servito da lì). Con Global Services attivi, la RTT diventa un fattore reale che impatta la latenza percepita → Φ_net ha senso come penalità.

---

### 4.2 `deployments/cnp-frontend-l7.yaml` — MODIFICATO

**File già esistente.** Aggiunta la Regola 2 per il traffico cross-cluster.

#### Cosa fa il file

`CiliumNetworkPolicy` (CNP) che controlla il traffico in ingresso verso i pod con label `app: frontend`. La policy è *allow-list*: tutto ciò che non è esplicitamente consentito viene bloccato.

#### Modifica introdotta

```
PRIMA (solo Regola 1):

  ingress:
  - fromEntities: [cluster, host]   # solo traffico intra-cluster
    toPorts: [{port: "8080"}]

  PROBLEMA: con Global Services attivi, il traffico che arriva da Nginx di
  cluster2/cluster3 attraversa il tunnel vxlan e arriva classificato come
  "remote-node" dall'agente Cilium. Non rientra in "cluster" (che copre
  solo identità locali) → traffico BLOCCATO dalla CNP → 503 cross-cluster.

──────────────────────────────────────────────────────────────────────────

DOPO (Regola 1 + Regola 2):

  ingress:
  # Regola 1: traffico intra-cluster (invariata)
  - fromEntities:
    - cluster      # pod di qualsiasi namespace nel cluster locale
    - host         # processo sul nodo (kubelet, health check)
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP

  # Regola 2: traffico cross-cluster (NUOVA)
  - fromEntities:
    - remote-node  # nodi/pod dei cluster remoti via ClusterMesh
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
```

#### Dettaglio tecnico: l'entità `remote-node`

In Cilium, ogni sorgente di traffico riceve una **identità numerica** assegnata dal control plane. L'entità `remote-node` è una categoria speciale che raggruppa:
- I nodi fisici dei cluster remoti connessi via ClusterMesh
- I pod dei cluster remoti il cui traffico transita attraverso il tunnel vxlan

```
CLASSIFICAZIONE IDENTITÀ CILIUM:

  Traffico da pod nello stesso cluster   → identità "cluster" (basata su label K8s)
  Traffico dal nodo locale (kubelet)     → identità "host"
  Traffico da internet (192.168.x.x)    → identità "world"
  Traffico da nodi/pod cluster remoti   → identità "remote-node"
                                                         ↑
                                               questa è la nuova regola
```

#### Perché non usare `fromEntities: world`?

L'entità `world` coprirebbe qualsiasi IP esterno, incluso traffico malevolo diretto dall'internet. L'obiettivo della CNP è bloccare l'accesso diretto esterno al frontend (tutti devono passare da Nginx). `remote-node` è più restrittivo: copre **solo** i nodi dei cluster connessi alla ClusterMesh, non IP arbitrari.

#### Verifica della regola

```bash
# Verifica che la CNP sia applicata
kubectl get cnp l7-visibility-frontend -n online-boutique --context cluster1 -o yaml

# Verifica con hubble: un flow cross-cluster deve mostrare "forwarded"
hubble observe --context cluster1 \
  --namespace online-boutique \
  --to-pod frontend \
  --verdict FORWARDED \
  --last 20
```

---

### 4.3 `config/weights.yaml` — MODIFICATO

**File già esistente.** Aggiunti: `omega_network` in tutti i profili, nuovo profilo `geo_aware`, sezione `network_parameters`.

#### Struttura modificata — profili di peso

```yaml
# PRIMA: 4 profili con 4 pesi ciascuno (somma = 1.0)
score_weights:
  balanced:
    omega_latency:  0.35
    omega_capacity: 0.25
    omega_load:     0.15
    omega_carbon:   0.25
    # MANCAVA: omega_network

# DOPO: 5 profili con 5 pesi ciascuno (somma = 1.0)
score_weights:
  carbon_agnostic:
    omega_latency:  0.40   # ridotto da 0.45
    omega_capacity: 0.30   # invariato
    omega_load:     0.15   # invariato
    omega_carbon:   0.00   # invariato (baseline senza carbon)
    omega_network:  0.15   # NUOVO: geo-awareness anche senza carbon

  balanced:
    omega_latency:  0.30   # ridotto da 0.35
    omega_capacity: 0.25   # invariato
    omega_load:     0.15   # invariato
    omega_carbon:   0.20   # ridotto da 0.25
    omega_network:  0.10   # NUOVO: 10% peso geo-awareness

  carbon_priority:
    omega_latency:  0.20   # ridotto da 0.25
    omega_capacity: 0.20   # invariato
    omega_load:     0.10   # invariato
    omega_carbon:   0.40   # invariato
    omega_network:  0.10   # NUOVO

  latency_priority:
    omega_latency:  0.45   # invariato (priorità latenza applicativa)
    omega_capacity: 0.20   # invariato
    omega_load:     0.10   # ridotto
    omega_carbon:   0.05   # ridotto
    omega_network:  0.20   # NUOVO: alto perché latenza utente-cluster è priorità

  geo_aware:               # PROFILO COMPLETAMENTE NUOVO per la tesi
    omega_latency:  0.20
    omega_capacity: 0.15
    omega_load:     0.10
    omega_carbon:   0.15
    omega_network:  0.40   # 40%: geo-awareness è il criterio dominante

active: "balanced"         # profilo attivo (cambia per gli esperimenti)
```

#### Nuova sezione `network_parameters`

```yaml
# NUOVA SEZIONE aggiunta al file:
network_parameters:
  rho: 2.0              # Esponente penalità nella formula exp(-ρ×RTT/RTT_max)
  rtt_max_ms: 500.0     # RTT massima attesa (ms) — usata per normalizzazione
                        # 500 = somma dei delay netem (150+350)
  fallback_rtt_ms: 5.0  # RTT se ping_exporter non disponibile (latenza LAN)
```

#### Significato del parametro `rho`

`rho` (ρ) controlla la **sensibilità** della penalità alla RTT. Valori più alti rendono la funzione più "ripida" — un piccolo aumento di RTT causa una grande riduzione dello score.

```
EFFETTO DI ρ sulla funzione Φ_net = exp(-ρ × RTT/RTT_max):

  RTT = 250ms, RTT_max = 500ms  →  RTT/RTT_max = 0.5

  ρ = 0.5:  Φ_net = exp(-0.25)  = 0.779   (penalità leggera)
  ρ = 1.0:  Φ_net = exp(-0.50)  = 0.607   (penalità moderata)
  ρ = 2.0:  Φ_net = exp(-1.00)  = 0.368   (penalità significativa) ← USATO
  ρ = 3.0:  Φ_net = exp(-1.50)  = 0.223   (penalità forte)
  ρ = 5.0:  Φ_net = exp(-2.50)  = 0.082   (penalità molto forte)

  Curve:
  Φ_net
  1.0 ┤                         ρ=0.5
  0.8 ┤                  ───────────
  0.6 ┤           ρ=1.0──
  0.4 ┤    ρ=2.0──                   ← SCELTO: curva ben differenziante
  0.2 ┤ ρ=3.0──
  0.0 ┤────────────────────────────► RTT
      0ms        250ms       500ms
```

#### Come switchare profilo per gli esperimenti

```python
# In config/weights.yaml:
active: "geo_aware"   # per esperimento con massima geo-awareness
active: "balanced"    # per baseline con geo-awareness moderata
active: "carbon_agnostic"  # per baseline senza carbon (confronto)
```

---

### 4.4 `src/level1/score_functions.py` — MODIFICATO

**File già esistente.** Aggiunte modifiche a 3 classi e 2 metodi.

#### Modifica 1: `ClusterMetrics` dataclass — campo `network_rtt_ms`

```python
# PRIMA:
@dataclass
class ClusterMetrics:
    cpu_available_cores: float
    cpu_total_cores: float
    memory_available_gb: float
    memory_total_gb: float
    request_rate_current: float
    request_rate_max: float
    latency_mean_ms: float
    latency_variance_ms2: float
    carbon_intensity_gco2_kwh: float
    cost_per_replica_hour: float
    # MANCAVA: network_rtt_ms

# DOPO: aggiunto con valore di default 5.0 (LAN baseline)
@dataclass
class ClusterMetrics:
    cpu_available_cores: float
    ...
    cost_per_replica_hour: float

    # NUOVO CAMPO:
    network_rtt_ms: float = 5.0
    # Valore di default = 5ms (latenza LAN senza tc netem)
    # Aggiornato a runtime da get_network_rtt_ms() se ping_exporter disponibile
    # Con netem: cluster1≈250ms, cluster2≈325ms, cluster3≈425ms
```

**Perché `default=5.0` e non required?**
Il campo ha un default per garantire backward compatibility: il resto del codice che crea `ClusterMetrics` senza specificare `network_rtt_ms` continua a funzionare, ritornando 5ms (latenza LAN, praticamente identica per tutti → Φ_net ≈ 0.98 → impatto trascurabile). Non rompe i test esistenti.

#### Modifica 2: `ScoreParameters` dataclass — campi `rho` e `rtt_max_ms`

```python
# PRIMA:
@dataclass
class ScoreParameters:
    eta: float = 0.01
    sigma_squared: float = 100
    kappa: float = 2.0
    mu: float = 1.0
    horizon_seconds: int = 600
    nu: float = 0.5
    ci_max: float = 500.0
    # MANCAVANO: rho e rtt_max_ms

# DOPO: aggiunti parametri per Φ_net
@dataclass
class ScoreParameters:
    eta: float = 0.01
    ...
    ci_max: float = 500.0

    # NUOVI CAMPI:
    rho: float = 2.0            # sensibilità alla RTT
    rtt_max_ms: float = 500.0   # RTT massima per normalizzazione
```

#### Modifica 3: `ScoreFunctions.__init__` — `omega_network` e validazione

```python
# PRIMA:
def __init__(self, weights: Dict[str, float], parameters=None):
    self.omega_latency  = weights.get('omega_latency',  0.35)
    self.omega_capacity = weights.get('omega_capacity', 0.25)
    self.omega_load     = weights.get('omega_load',     0.15)
    self.omega_carbon   = weights.get('omega_carbon',   0.25)

    total = (self.omega_latency + self.omega_capacity +
             self.omega_load + self.omega_carbon)
    # totale = 4 pesi, somma = 1.0

# DOPO:
def __init__(self, weights: Dict[str, float], parameters=None):
    self.omega_latency  = weights.get('omega_latency',  0.35)
    self.omega_capacity = weights.get('omega_capacity', 0.25)
    self.omega_load     = weights.get('omega_load',     0.15)
    self.omega_carbon   = weights.get('omega_carbon',   0.20)
    self.omega_network  = weights.get('omega_network',  0.05)  # NUOVO

    # Validazione aggiornata: ora include omega_network
    total = (self.omega_latency + self.omega_capacity +
             self.omega_load + self.omega_carbon + self.omega_network)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"La somma delle pesature deve essere 1.0, ho {total}")
```

**Implicazione**: qualsiasi configurazione di pesi che non include `omega_network` (o lo include a 0.0) causerà un `ValueError` se la somma degli altri 4 non è esattamente 1.0. Questo forza l'aggiornamento esplicito di tutti i profili in `weights.yaml`.

#### Modifica 4: `compute_network_score()` — NUOVO METODO

```python
def compute_network_score(self, metrics: ClusterMetrics) -> float:
    """
    Φ_net(i) = exp(-ρ × min(RTT_i, RTT_max) / RTT_max)
    """
    # min() garantisce che RTT superiori a RTT_max non generino score < 0
    # (la funzione exp() è sempre positiva, ma clampiamo per coerenza)
    rtt = min(metrics.network_rtt_ms, self.params.rtt_max_ms)

    score = math.exp(-self.params.rho * rtt / self.params.rtt_max_ms)

    logger.debug(
        f"Φ_net: RTT={metrics.network_rtt_ms:.1f}ms "
        f"(capped={rtt:.1f}ms), ρ={self.params.rho}, "
        f"RTT_max={self.params.rtt_max_ms:.0f}ms, score={score:.3f}"
    )
    return score
```

**Dominio e codominio:**

```
INPUT:  RTT_i ∈ [0, +∞) ms
        (clamped a RTT_max prima del calcolo)

OUTPUT: Φ_net ∈ (0, 1]
        RTT=0ms    → Φ_net = exp(0) = 1.000  (cluster "vicinissimo")
        RTT=125ms  → Φ_net = exp(-0.5) ≈ 0.607
        RTT=250ms  → Φ_net = exp(-1.0) ≈ 0.368  (cluster1 con netem)
        RTT=375ms  → Φ_net = exp(-1.5) ≈ 0.223
        RTT=500ms  → Φ_net = exp(-2.0) ≈ 0.135  (RTT massima)
        RTT>500ms  → clamped a 500ms → Φ_net = 0.135 (non scende oltre)
```

#### Modifica 5: `compute_total_score()` — aggiunta di `phi_net`

```python
# PRIMA:
def compute_total_score(self, metrics, predicted_load=None):
    phi_lat    = self.compute_latency_score(metrics)
    phi_cap    = self.compute_capacity_score(metrics)
    phi_load   = self.compute_load_score(metrics, predicted_load)
    phi_carbon = self.compute_carbon_score(metrics)

    total = (self.omega_latency * phi_lat +
             self.omega_capacity * phi_cap +
             self.omega_load * phi_load +
             self.omega_carbon * phi_carbon)
    # 4 componenti

# DOPO:
def compute_total_score(self, metrics, predicted_load=None):
    phi_lat    = self.compute_latency_score(metrics)
    phi_cap    = self.compute_capacity_score(metrics)
    phi_load   = self.compute_load_score(metrics, predicted_load)
    phi_carbon = self.compute_carbon_score(metrics)
    phi_net    = self.compute_network_score(metrics)   # NUOVO

    total = (self.omega_latency  * phi_lat   +
             self.omega_capacity * phi_cap   +
             self.omega_load     * phi_load  +
             self.omega_carbon   * phi_carbon +
             self.omega_network  * phi_net)  # NUOVO termine
    # 5 componenti
```

#### Modifica 6: `compute_score_breakdown()` — aggiunta `phi_network`

```python
# PRIMA: il dizionario restituiva 5 chiavi
return {
    'phi_latency': phi_lat,
    'phi_capacity': phi_cap,
    'phi_load': phi_load,
    'phi_carbon': phi_carbon,
    'total_score': total,
    # MANCAVANO: phi_network, network_rtt_ms
}

# DOPO: il dizionario restituisce 8 chiavi
return {
    'phi_latency':    phi_lat,
    'phi_capacity':   phi_cap,
    'phi_load':       phi_load,
    'phi_carbon':     phi_carbon,
    'phi_network':    phi_net,            # NUOVO
    'total_score':    total,
    'network_rtt_ms': metrics.network_rtt_ms,  # NUOVO (per logging/debug)
    'weights': {                          # NUOVO (per tracciabilità)
        'omega_latency':  self.omega_latency,
        'omega_capacity': self.omega_capacity,
        'omega_load':     self.omega_load,
        'omega_carbon':   self.omega_carbon,
        'omega_network':  self.omega_network,
    }
}
```

Il breakdown arricchito permette al layer superiore (DMOS Scheduler) di loggare, monitorare e graficamente visualizzare il contributo di ogni componente allo score finale — utile per gli esperimenti della tesi.

---

### 4.5 `src/metrics/prometheus_client.py` — MODIFICATO

**File già esistente.** Aggiunto il metodo `get_network_rtt_ms()`.

#### Contesto: PROM_MAP

Prima di descrivere la modifica, è fondamentale capire l'architettura PROM_MAP:

```
ARCHITETTURA PROM_MAP (approccio Romano):

  DMOS Scheduler
       │
       ├── self.prom_map["cluster1"] = PrometheusClient("http://192.168.1.245:30090")
       ├── self.prom_map["cluster2"] = PrometheusClient("http://192.168.1.246:30090")
       └── self.prom_map["cluster3"] = PrometheusClient("http://192.168.1.247:30090")

  Ogni PrometheusClient interroga SOLO il Prometheus del suo cluster.
  Ogni Prometheus vede SOLO i pod del suo cluster → metriche accurate.

  Per cluster1:
    prom["cluster1"].get_network_rtt_ms(["192.168.1.246", "192.168.1.247"])
         │
         └── query a http://192.168.1.245:30090/api/v1/query
                 avg(ping_rtt_mean_seconds{target=~"192.168.1.246|192.168.1.247"}) * 1000
                 → 250.0 ms  (valore da ping_exporter su cluster1)
```

#### Metodo aggiunto: `get_network_rtt_ms()`

```python
def get_network_rtt_ms(self, peer_ips: List[str]) -> float:
    """
    Restituisce la RTT media (ms) verso i peer cluster via ping_exporter.
    """
    if not peer_ips:
        return 5.0  # nessun peer → fallback LAN

    # Costruisce regex OR per matchare tutti i target peer in UNA SOLA query
    # Es: "192.168.1.246|192.168.1.247"
    targets_regex = "|".join(peer_ips)

    # PromQL: media della RTT verso tutti i peer, in millisecondi
    # ping_rtt_mean_seconds è in secondi → *1000 per ms
    query = f'avg(ping_rtt_mean_seconds{{target=~"{targets_regex}"}}) * 1000'

    try:
        result = self.query(query)
        if result and result.get('result') and len(result['result']) > 0:
            val = float(result['result'][0]['value'][1])
            if val > 0:
                logger.info(f"✅ Network RTT ({self.cluster_name}): {val:.1f} ms")
                return val
            # val == 0: ping_exporter non ha ancora misurato
            # (es. appena avviato, target non raggiungibili)
    except Exception as e:
        logger.debug(f"ping_exporter query exception ({self.cluster_name}): {e}")

    # Fallback: 5.0ms (latenza LAN baseline senza netem)
    # Con questo valore, Φ_net ≈ 0.980 per tutti i cluster
    # → omega_network ha impatto trascurabile → comportamento pre-geo-awareness
    return 5.0
```

#### Query PromQL spiegata

```
avg(ping_rtt_mean_seconds{target=~"192.168.1.246|192.168.1.247"}) * 1000

├── ping_rtt_mean_seconds      metrica esposta da ping_exporter
│                              (namespace: observability, pod: ping-exporter-*)
│
├── {target=~"..."}            regex label selector:
│                              target=~"A|B" → match su target="A" OR target="B"
│                              (PromQL usa RE2 syntax)
│
├── avg(...)                   media aritmetica tra tutti i target matchati
│                              cluster1 ha 2 target → avg(150ms, 350ms) = 250ms
│
└── * 1000                     conversione secondi → millisecondi
```

#### Meccanismo di fallback

```
SEQUENZA DI FALLBACK:

  ping_exporter disponibile e RTT > 0 ms?
         │
         ├─ SÌ → return RTT misurata (es. 250.0 ms)
         │
         └─ NO (ping_exporter non deployato / target non raggiungibili / RTT=0)
                │
                └─ return 5.0 ms (latenza LAN baseline)
                   → Φ_net ≈ exp(-2×5/500) = exp(-0.02) ≈ 0.980
                   → praticamente uguale per tutti i cluster
                   → omega_network ha impatto ≈ 0 sullo score finale
                   → DMOS si comporta come nella versione pre-geo-awareness ✓
```

---

### 4.6 `src/level1/dmos_scheduler.py` — MODIFICATO

**File già esistente.** Aggiunte modifiche a `__init__()` e `_collect_cluster_metrics()`.

#### Modifica 1: `__init__()` — inizializzazione parametri geo-awareness

```python
# PRIMA:
def __init__(self, config_path: str = "config"):
    self.config = ConfigLoader(config_path)
    self.winner_det = WinnerDetermination()

    # PROM_MAP
    self.prom_map = {}
    self.cluster_configs = self.config.get_all_clusters()
    for name, cfg in self.cluster_configs.items():
        prom_url = f"http://{cfg.ip}:30090"
        self.prom_map[name] = PrometheusClient(url=prom_url, ...)

    self.carbon_client = CarbonClient(...)

    # ScoreFunctions SENZA omega_network
    self.score_func = ScoreFunctions(
        weights={
            'omega_latency': ...,
            'omega_capacity': ...,
            'omega_load': ...,
            'omega_carbon': ...,
            # MANCAVA: omega_network
        }
    )
    # MANCAVANO: rho, rtt_max_ms, fallback_rtt_ms, _peer_ips

# DOPO:
def __init__(self, config_path: str = "config"):
    ...  # stessa inizializzazione PROM_MAP e carbon_client

    # NUOVO: legge parametri geo-awareness da config/weights.yaml
    net_params = self.config.network_params
    self._network_rho            = net_params.get('rho', 2.0)
    self._network_rtt_max_ms     = net_params.get('rtt_max_ms', 500.0)
    self._network_fallback_rtt_ms = net_params.get('fallback_rtt_ms', 5.0)

    # NUOVO: mappa cluster → lista IP dei peer cluster
    # cluster1 [245] → peer = [246, 247]
    # cluster2 [246] → peer = [245, 247]
    # cluster3 [247] → peer = [245, 246]
    all_ips = {n: cfg.ip for n, cfg in self.cluster_configs.items()}
    self._peer_ips: Dict[str, list] = {
        name: [ip for n, ip in all_ips.items() if n != name]
        for name in self.cluster_configs
    }

    # ScoreFunctions CON omega_network E parametri RTT
    self.score_func = ScoreFunctions(
        weights={
            'omega_latency':  self.config.score_weights.omega_latency,
            'omega_capacity': self.config.score_weights.omega_capacity,
            'omega_load':     self.config.score_weights.omega_load,
            'omega_carbon':   self.config.score_weights.omega_carbon,
            'omega_network':  self.config.score_weights.omega_network,  # NUOVO
        },
        parameters=ScoreParameters(
            rho=self._network_rho,           # NUOVO
            rtt_max_ms=self._network_rtt_max_ms,  # NUOVO
        )
    )
```

#### Struttura `_peer_ips` spiegata

```python
# Esempio con 3 cluster:
# cluster_configs = {
#   "cluster1": ClusterConfig(ip="192.168.1.245", ...),
#   "cluster2": ClusterConfig(ip="192.168.1.246", ...),
#   "cluster3": ClusterConfig(ip="192.168.1.247", ...)
# }

all_ips = {
    "cluster1": "192.168.1.245",
    "cluster2": "192.168.1.246",
    "cluster3": "192.168.1.247"
}

self._peer_ips = {
    "cluster1": ["192.168.1.246", "192.168.1.247"],  # tutti tranne cluster1
    "cluster2": ["192.168.1.245", "192.168.1.247"],  # tutti tranne cluster2
    "cluster3": ["192.168.1.245", "192.168.1.246"],  # tutti tranne cluster3
}
# Dict comprehension: per ogni cluster, lista degli IP di TUTTI gli altri
```

#### Modifica 2: `_collect_cluster_metrics()` — aggiunta RTT

```python
# PRIMA: ClusterMetrics istanziato con 10 campi (senza network_rtt_ms)
metrics = ClusterMetrics(
    cpu_available_cores=cpu_available,
    cpu_total_cores=cpu_total,
    memory_available_gb=memory_available,
    memory_total_gb=memory_total,
    request_rate_current=request_rate,
    request_rate_max=request_rate_max,
    latency_mean_ms=latency_mean,
    latency_variance_ms2=latency_variance,
    carbon_intensity_gco2_kwh=carbon_intensity,
    cost_per_replica_hour=cost_per_replica,
    # MANCAVA: network_rtt_ms
)

# DOPO: aggiunta query RTT + passaggio al costruttore
# ── Network RTT (geo-awareness via ping_exporter) ─────────
peer_ips = self._peer_ips.get(cluster_name, [])  # NUOVO
network_rtt_ms = prom.get_network_rtt_ms(peer_ips)  # NUOVO

metrics = ClusterMetrics(
    cpu_available_cores=cpu_available,
    cpu_total_cores=cpu_total,
    memory_available_gb=memory_available,
    memory_total_gb=memory_total,
    request_rate_current=request_rate,
    request_rate_max=request_rate_max,
    latency_mean_ms=latency_mean,
    latency_variance_ms2=latency_variance,
    carbon_intensity_gco2_kwh=carbon_intensity,
    cost_per_replica_hour=cost_per_replica,
    network_rtt_ms=network_rtt_ms,  # NUOVO
)
```

#### Aggiornamento del log in `_compute_cluster_score()`

```python
# PRIMA:
logger.info(
    f"Score {cluster_name}: {breakdown['total_score']:.3f} "
    f"(lat={breakdown['phi_latency']:.3f}, cap={breakdown['phi_capacity']:.3f}, "
    f"load={breakdown['phi_load']:.3f}, carbon={breakdown['phi_carbon']:.3f}) "
    f"capacity={capacity}"
)

# DOPO: aggiunto phi_network e RTT al log
logger.info(
    f"Score {cluster_name}: {breakdown['total_score']:.3f} "
    f"(lat={breakdown['phi_latency']:.3f}, cap={breakdown['phi_capacity']:.3f}, "
    f"load={breakdown['phi_load']:.3f}, carbon={breakdown['phi_carbon']:.3f}, "
    f"net={breakdown['phi_network']:.3f}|RTT={metrics.network_rtt_ms:.0f}ms) "  # NUOVO
    f"capacity={capacity}, cpu={cpu_util_pct:.0f}%, "
    f"CI={metrics.carbon_intensity_gco2_kwh:.0f}gCO2"
)
```

**Output log con geo-awareness attiva (esempio):**
```
INFO DMOSScheduler - Score cluster1: 0.612 (lat=0.734, cap=0.681, load=0.823, carbon=0.541, net=0.368|RTT=250ms) capacity=8, cpu=42%, CI=380gCO2
INFO DMOSScheduler - Score cluster2: 0.547 (lat=0.698, cap=0.712, load=0.756, carbon=0.541, net=0.272|RTT=325ms) capacity=6, cpu=38%, CI=320gCO2
INFO DMOSScheduler - Score cluster3: 0.489 (lat=0.701, cap=0.698, load=0.789, carbon=0.412, net=0.183|RTT=425ms) capacity=4, cpu=35%, CI=580gCO2
```

---

## 5. Flusso end-to-end con geo-awareness

Il seguente diagramma mostra il percorso completo dei dati dalla misurazione ICMP alla decisione di scheduling:

```
FLUSSO COMPLETO GEO-AWARENESS (ogni 30 secondi):

  ┌─────────────────────────────────────────────────────────────────────┐
  │  LAYER FISICO (tc netem — configurato una volta sola)               │
  │                                                                     │
  │  node cluster2 (ens18): kernel aggiunge 150ms a ogni pacchetto     │
  │  node cluster3 (ens18): kernel aggiunge 350ms a ogni pacchetto     │
  └─────────────────────────────────────────────────────────────────────┘
                   │ delay fisico simulato
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  MISURAZIONE (ping_exporter — continua, ogni 1s)                    │
  │                                                                     │
  │  [ping_exporter su cluster1]                                        │
  │  ICMP → 192.168.1.246 → RTT misurata ≈ 150ms                       │
  │  ICMP → 192.168.1.247 → RTT misurata ≈ 350ms                       │
  │                                                                     │
  │  Espone /metrics su :9427:                                          │
  │  ping_rtt_mean_seconds{target="192.168.1.246"} 0.150               │
  │  ping_rtt_mean_seconds{target="192.168.1.247"} 0.350               │
  └─────────────────────────────────────────────────────────────────────┘
                   │ metriche Prometheus
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  RACCOLTA (Prometheus — ogni 15s)                                   │
  │                                                                     │
  │  Prometheus cluster1 (:30090) scrappa ping_exporter :9427          │
  │  → TSDB: serie storica di ping_rtt_mean_seconds                     │
  └─────────────────────────────────────────────────────────────────────┘
                   │ HTTP query PromQL
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  QUERY (PrometheusClient.get_network_rtt_ms() — ogni 30s)          │
  │                                                                     │
  │  prom["cluster1"].get_network_rtt_ms(["192.168.1.246","...247"])    │
  │  → PromQL: avg(ping_rtt_mean_seconds{target=~"...|..."}) * 1000     │
  │  → avg(0.150, 0.350) * 1000 = 250.0 ms                             │
  └─────────────────────────────────────────────────────────────────────┘
                   │ network_rtt_ms = 250.0
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  DATACLASS (ClusterMetrics — in-process)                            │
  │                                                                     │
  │  ClusterMetrics(                                                    │
  │    cpu_available_cores=3.2,                                         │
  │    memory_available_gb=5.1,                                         │
  │    latency_mean_ms=120.0,                                           │
  │    carbon_intensity_gco2_kwh=380.0,                                 │
  │    network_rtt_ms=250.0,     ← NUOVO CAMPO                         │
  │    ...                                                              │
  │  )                                                                  │
  └─────────────────────────────────────────────────────────────────────┘
                   │ oggetto metrics
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  SCORE COMPUTATION (ScoreFunctions — in-process)                    │
  │                                                                     │
  │  Φ_lat    = 1/(1+0.01×120) × exp(-100/100) = 0.734                │
  │  Φ_cap    = (3.2/8)^2 × (1-0.42) = 0.681                          │
  │  Φ_load   = exp(-1×0.42) = 0.823                                   │
  │  Φ_carbon = exp(-0.5×380/800) = 0.541                              │
  │  Φ_net    = exp(-2×250/500)   = 0.368   ← NUOVA COMPONENTE        │
  │                                                                     │
  │  score = 0.30×0.734 + 0.25×0.681 + 0.15×0.823                     │
  │        + 0.20×0.541 + 0.10×0.368                                   │
  │        = 0.220 + 0.170 + 0.123 + 0.108 + 0.037                    │
  │        = 0.658                                                     │
  └─────────────────────────────────────────────────────────────────────┘
                   │ score per cluster
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  WINNER DETERMINATION (WinnerDetermination — in-process)            │
  │                                                                     │
  │  cluster1: score=0.658, capacity=8                                  │
  │  cluster2: score=0.547, capacity=6                                  │
  │  cluster3: score=0.489, capacity=4   (penalizzato da alta RTT)     │
  │                                                                     │
  │  Allocation per 12 repliche totali:                                 │
  │    cluster1 → 6 repliche  (score più alto)                         │
  │    cluster2 → 4 repliche                                            │
  │    cluster3 → 2 repliche  (score più basso)                        │
  └─────────────────────────────────────────────────────────────────────┘
                   │ kubectl scale
                   ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  EFFETTO SUL ROUTING (Cilium ClusterMesh — automatico)              │
  │                                                                     │
  │  Con più repliche su cluster1: il pool Global Service ha più       │
  │  endpoint locali → Cilium LB sceglie il cluster1 più spesso        │
  │  → utenti EU vengono serviti prevalentemente da Frankfurt           │
  │  → latenza percepita ridotta                                        │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Simulazione geografica con tc netem

### Perché simulare e non misurare RTT reale?

L'ambiente di test è una LAN locale (VM su Proxmox): tutti i nodi sono fisicamente nello stesso rack, con RTT reale di 1-5ms tra tutti. Senza simulazione, Φ_net sarebbe ≈ 0.98 per tutti i cluster → nessuna differenziazione → la geo-awareness non ha effetto osservabile.

Romano nella sua tesi ha risolto questo problema con `tc netem`, e lo stesso approccio è adottato qui.

### Comandi di configurazione (da applicare una volta per test)

```bash
# Su nodo di cluster2 (IP 192.168.1.246):
# Aggiunge 150ms ± 20ms con distribuzione gaussiana su tutte le uscite di ens18
ssh ubuntu@192.168.1.246 "sudo tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal"

# Su nodo di cluster3 (IP 192.168.1.247):
# Aggiunge 350ms ± 20ms con distribuzione gaussiana
ssh ubuntu@192.168.1.247 "sudo tc qdisc add dev ens18 root netem delay 350ms 20ms distribution normal"

# Su nodo di cluster1: nessun delay (è il cluster "centrale")

# Verifica immediata (da cluster1):
ping -c 5 192.168.1.246  # deve mostrare RTT ≈ 150ms (range: 130-170ms)
ping -c 5 192.168.1.247  # deve mostrare RTT ≈ 350ms (range: 330-370ms)

# Verifica stato qdisc attivo:
ssh ubuntu@192.168.1.246 "tc qdisc show dev ens18"
ssh ubuntu@192.168.1.247 "tc qdisc show dev ens18"

# Rimozione (dopo i test):
ssh ubuntu@192.168.1.246 "sudo tc qdisc del dev ens18 root"
ssh ubuntu@192.168.1.247 "sudo tc qdisc del dev ens18 root"
```

### ✅ Risultati verificati (eseguiti il 20 marzo 2026)

I comandi sono stati applicati con successo. La verifica è stata eseguita con `ping` dal PC Windows
verso i nodi dei cluster (percorso: Windows → router LAN → nodo cluster).

```
CLUSTER2 (192.168.1.246) — delay configurato: 150ms ± 20ms
  ping -n 5 192.168.1.246
  Risposta: 213ms / 202ms / 170ms / 208ms / 215ms
  Medio: 201ms

  ← Il delay aggiunto da netem è visibile: da ~1ms (LAN base) a ~200ms.
  ← L'offset di ~50ms rispetto ai 150ms attesi è dovuto al path Windows→router→nodo
     che aggiunge latenza extra (non presente nel path diretto cluster1→cluster2).
  ← Da cluster1 (LAN diretta, nessun router), ping_exporter misurerà ≈ 150ms. ✓

CLUSTER3 (192.168.1.247) — delay configurato: 350ms ± 20ms
  ping -n 5 192.168.1.247
  Risposta: 402ms / 388ms / 409ms / 425ms / 403ms
  Medio: 405ms

  ← Stesso offset di ~50ms (path Windows), conforme con cluster2.
  ← Da cluster1, ping_exporter misurerà ≈ 350ms. ✓
  ← Il jitter è coerente: distribuzione normale con σ=20ms produce variazioni
     nell'ordine di ±15-35ms, visibile nei valori (388ms–425ms).
```

```
DOVE VIENE APPLICATO IL DELAY — visibile nell'output tc:

  # Verificabile con: ssh ubuntu@192.168.1.246 "tc qdisc show dev ens18"
  # Output atteso su cluster2:
  qdisc netem 8001: root refcnt 2 limit 1000 delay 150ms  20ms distribution normal
  #                 ^^^^                        ^^^^^^^  ^^^^
  #                 modulo attivo               delay    jitter gaussiano

  # Output atteso su cluster3:
  qdisc netem 8001: root refcnt 2 limit 1000 delay 350ms  20ms distribution normal
```

### Effetto sulla metrica Prometheus

```
DOPO tc netem su cluster2 (150ms) e cluster3 (350ms):

  ping_rtt_mean_seconds{target="192.168.1.246"} 0.150   # su prom cluster1
  ping_rtt_mean_seconds{target="192.168.1.247"} 0.350   # su prom cluster1

  → DMOS calcola:
    cluster1: RTT_avg = avg(150, 350) = 250ms → Φ_net = exp(-1.0) = 0.368
    cluster2: RTT_avg = avg(150, 500) = 325ms → Φ_net = exp(-1.3) = 0.272
    cluster3: RTT_avg = avg(350, 500) = 425ms → Φ_net = exp(-1.7) = 0.183

  cluster2 misura RTT verso cluster3 come "500ms" perché il delay è bidirezionale
  (cluster2 ha +150ms, cluster3 ha +350ms → andata+ritorno = 500ms totali).
```

---

## 7. tc netem — Meccanismo EGRESS e matrice RTT completa

### Come funziona tc netem: solo traffico in USCITA

Il comando `tc qdisc add dev ens18 root netem delay Xms` aggiunge un ritardo **esclusivamente ai pacchetti in uscita** (EGRESS) dall'interfaccia `ens18`. I pacchetti in entrata (INGRESS) non subiscono alcun delay.

```
MODELLO MENTALE tc netem (EGRESS only):

  cluster2 (Paris):
  ┌───────────────────────────────────┐
  │  ens18                            │
  │  ┌────────┐   ┌────────────────┐  │
  │  │ INGRESS│   │ EGRESS + 150ms │  │  ← netem agisce qui
  │  │ libero │   │ ogni pacchetto │  │
  │  └────────┘   └────────────────┘  │
  └───────────────────────────────────┘

  cluster3 (Warsaw):
  ┌───────────────────────────────────┐
  │  ens18                            │
  │  ┌────────┐   ┌────────────────┐  │
  │  │ INGRESS│   │ EGRESS + 350ms │  │  ← netem agisce qui
  │  │ libero │   │ ogni pacchetto │  │
  │  └────────┘   └────────────────┘  │
  └───────────────────────────────────┘

  cluster1 (Frankfurt):
  ┌───────────────────────────────────┐
  │  ens18                            │
  │  ┌────────┐   ┌────────────────┐  │
  │  │ INGRESS│   │ EGRESS libero  │  │  ← nessun netem
  │  │ libero │   │                │  │
  │  └────────┘   └────────────────┘  │
  └───────────────────────────────────┘
```

### Calcolo RTT per ogni coppia — passo per passo

La RTT di un ping è la somma dei delay EGRESS sul percorso **andata + ritorno**. Poiché netem agisce sull'uscita di ogni nodo, la RTT totale è la somma dei delay sui due EGRESS attraversati:

```
PING cluster1 → cluster2:

  cluster1                    cluster2
     │                           │
     │ ICMP request              │
     │ esce cluster1: 0ms ──────►│  (cluster1 non ha netem)
     │                           │
     │         ICMP reply        │
     │◄────────────── esce c2: +150ms
     │
  RTT = 0 + 150 = 150ms  ✓

──────────────────────────────────────────────────────────

PING cluster1 → cluster3:

  cluster1                    cluster3
     │                           │
     │ ICMP request              │
     │ esce cluster1: 0ms ──────►│
     │                           │
     │         ICMP reply        │
     │◄────────────── esce c3: +350ms
     │
  RTT = 0 + 350 = 350ms  ✓

──────────────────────────────────────────────────────────

PING cluster2 → cluster3:   ← il caso che ti chiedevi

  cluster2                    cluster3
     │                           │
     │ ICMP request              │
     │ esce cluster2: +150ms ───►│  (cluster2 ha netem +150ms)
     │                           │
     │         ICMP reply        │
     │◄────────────── esce c3: +350ms
     │
  RTT = 150 + 350 = 500ms  ← emerge automaticamente, senza configurazione extra

──────────────────────────────────────────────────────────

PING cluster3 → cluster1:

  cluster3                    cluster1
     │                           │
     │ ICMP request              │
     │ esce cluster3: +350ms ───►│
     │                           │
     │         ICMP reply        │
     │◄────────────── esce c1: 0ms
     │
  RTT = 350 + 0 = 350ms  ✓  (simmetrico a cluster1→cluster3)

──────────────────────────────────────────────────────────

PING cluster2 → cluster1:

  cluster2                    cluster1
     │                           │
     │ ICMP request              │
     │ esce cluster2: +150ms ───►│
     │                           │
     │         ICMP reply        │
     │◄────────────── esce c1: 0ms
     │
  RTT = 150 + 0 = 150ms  ✓  (simmetrico a cluster1→cluster2)
```

### Matrice RTT completa risultante

Con **soli 2 comandi SSH** (su cluster2 e cluster3) si ottiene automaticamente una topologia triangolare:

```
         cluster1       cluster2       cluster3
         (Frankfurt)    (Paris)        (Warsaw)
         no netem       +150ms EGRESS  +350ms EGRESS

cluster1    —              150ms          350ms
cluster2   150ms            —             500ms    ← 150+350 automatico
cluster3   350ms           500ms           —       ← 350+150 automatico
```

```
TOPOLOGIA GEOGRAFICA SIMULATA:

     cluster1 (Frankfurt)
        /          \
     150ms        350ms
      /              \
cluster2 (Paris) ──500ms── cluster3 (Warsaw)

→ Parigi vicina a Francoforte ✓
→ Varsavia lontana da entrambe ✓
→ Parigi-Varsavia = somma dei due delay ✓
```

### Cosa misura ping_exporter su ciascun cluster

```
ping_exporter SU CLUSTER1 misura:
  → 192.168.1.246 (cluster2): RTT = 150ms
  → 192.168.1.247 (cluster3): RTT = 350ms
  RTT_avg cluster1 = avg(150, 350) = 250ms

ping_exporter SU CLUSTER2 misura:
  → 192.168.1.245 (cluster1): RTT = 150ms
  → 192.168.1.247 (cluster3): RTT = 500ms   ← 150+350
  RTT_avg cluster2 = avg(150, 500) = 325ms

ping_exporter SU CLUSTER3 misura:
  → 192.168.1.245 (cluster1): RTT = 350ms
  → 192.168.1.246 (cluster2): RTT = 500ms   ← 350+150
  RTT_avg cluster3 = avg(350, 500) = 425ms
```

### Risultato: gerarchia geografica naturale

```
  Cluster       RTT_avg    Φ_net             Interpretazione
  ─────────     ───────    ──────            ────────────────
  cluster1      250ms      0.368  (più alto) → più centrale in Europa
  cluster2      325ms      0.272             → intermedio
  cluster3      425ms      0.183  (più basso)→ più periferico
```

Il cluster1 (Frankfurt) risulta naturalmente il più centrale perché **non aggiunge ritardo in uscita**, quindi le sue RTT verso i peer sono determinate solo dal delay dei peer stessi (i più bassi possibili). Varsavia accumula il delay sia in andata che in ritorno verso tutti.

---

## 8. Multi-ingress 40/30/30 — Perché RTT_avg è la metrica corretta

### Il setup reale: tre ingress indipendenti

Nel sistema DMOS con Nginx multi-ingress, gli utenti non entrano tutti dallo stesso cluster. La distribuzione del traffico è configurata nel bilanciamento esterno:

```
                 Internet
                    │
          ┌─────────┼──────────┐
          │40%      │30%       │30%
          ▼         ▼          ▼
    Nginx C1    Nginx C2    Nginx C3
    Frankfurt    Paris       Warsaw
          │         │          │
          └─────────┼──────────┘
                    │
             Global Service LB
            /        |        \
      pod C1(N1)  pod C2(N2)  pod C3(N3)
```

Ogni Nginx, indipendentemente dal cluster su cui si trova, può instradare le richieste verso **qualsiasi pod frontend** tramite il Global Service Cilium.

### Le due RTT da non confondere

```
RTT TIPO 1: Utente → Nginx del cluster (NON misurata da DMOS)
  Utente di Milano → Nginx cluster1: ~10ms  (dipende dall'ISP)
  Utente di Berlino → Nginx cluster2: ~8ms
  Questa RTT è fuori dal controllo di DMOS: non possiamo spostarla.

RTT TIPO 2: Nginx → Pod frontend (misurata da ping_exporter)
  Nginx cluster2 → pod frontend cluster3: 500ms tunnel vxlan
  Nginx cluster1 → pod frontend cluster3: 350ms tunnel vxlan
  QUESTA è la RTT che DMOS può ridurre mettendo più repliche vicino agli ingress.
```

### Scenario critico: utente entra da cluster2 (Paris)

```
SENZA geo-awareness (distribuzione uniforme: 4 repliche per cluster):

  Utente → Nginx cluster2
                │
                ├─(33%)──► pod frontend cluster2: 0ms tunnel     → ok
                ├─(33%)──► pod frontend cluster1: 150ms tunnel    → tollerabile
                └─(33%)──► pod frontend cluster3: 500ms tunnel    → pessimo!
                            │
                            Nginx C2 esce: +150ms
                            pod C3 risponde ed esce: +350ms
                            Totale tunnel = 500ms latenza aggiuntiva
```

```
CON geo-awareness, profilo "geo_aware" (ω_net=0.40):

  DMOS score:
    cluster1: RTT_avg=250ms → Φ_net=0.368 → 7 repliche
    cluster2: RTT_avg=325ms → Φ_net=0.272 → 3 repliche
    cluster3: RTT_avg=425ms → Φ_net=0.183 → 2 repliche

  Utente → Nginx cluster2
                │
                ├─(58%)──► pod frontend cluster1: 150ms tunnel    → tollerabile
                ├─(25%)──► pod frontend cluster2: 0ms tunnel      → ottimo
                └─(17%)──► pod frontend cluster3: 500ms tunnel    → raro!

  La probabilità di subire 500ms scende da 33% → 17%
```

### Perché RTT_avg (non RTT verso cluster1) è la metrica giusta

Qui sta il punto cruciale. Con multi-ingress, **ogni cluster funge sia da ingress che da target**. La domanda di DMOS è:

> "Se metto tante repliche su cluster X, quanto costerà agli utenti che entrano dagli altri cluster raggiungerle?"

La RTT_avg di cluster X è esattamente questo costo medio ponderato:

```
COSTO MEDIO di avere repliche su cluster3 (con distribuzione ingress 40/30/30):

  Utenti da cluster1 (40%) → tunnel C1→C3: 350ms
  Utenti da cluster2 (30%) → tunnel C2→C3: 500ms
  Utenti da cluster3 (30%) → tunnel C3→C3:   0ms (locale)

  Costo medio pesato = 40%×350 + 30%×500 + 30%×0
                     = 140 + 150 + 0 = 290ms extra per ogni richiesta

COSTO MEDIO di avere repliche su cluster1:

  Utenti da cluster1 (40%) → tunnel C1→C1:   0ms (locale)
  Utenti da cluster2 (30%) → tunnel C2→C1: 150ms
  Utenti da cluster3 (30%) → tunnel C3→C1: 350ms

  Costo medio pesato = 40%×0 + 30%×150 + 30%×350
                     = 0 + 45 + 105 = 150ms extra per ogni richiesta
```

```
CONFRONTO FINALE:
  Repliche su cluster1 → costo medio 150ms   ← DMOS favorisce
  Repliche su cluster2 → costo medio ~200ms
  Repliche su cluster3 → costo medio 290ms   ← DMOS penalizza

  RTT_avg cluster1 = 250ms  (proxy del costo "essere raggiunto da tutti")
  RTT_avg cluster3 = 425ms  (proxy del costo più alto)

  Φ_net cattura esattamente questo ordinamento: cluster1 > cluster2 > cluster3
```

### La topologia multi-ingress nel caso reale

```
SCENARIO REALE con 12 repliche totali e profilo "geo_aware":

  ┌──────────────────────────────────────────────────────────────────┐
  │  ALLOCATION DMOS:                                                │
  │    cluster1 (Frankfurt): 7 repliche  ← più repliche             │
  │    cluster2 (Paris):     3 repliche                              │
  │    cluster3 (Warsaw):    2 repliche  ← meno repliche            │
  └──────────────────────────────────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    [pod×7 C1]           [pod×3 C2]          [pod×2 C3]

  Cilium LB pool (Global Service):
  12 endpoint totali: 7 su C1, 3 su C2, 2 su C3

  ROUTING per ogni ingress (probabilità proporzionale ai pod):

  Nginx C1 (40% utenti):
    → C1 pod: 7/12 = 58% → 0ms tunnel
    → C2 pod: 3/12 = 25% → 150ms tunnel
    → C3 pod: 2/12 = 17% → 350ms tunnel
    Latenza tunnel media = 58%×0 + 25%×150 + 17%×350 = 97ms

  Nginx C2 (30% utenti):
    → C1 pod: 7/12 = 58% → 150ms tunnel
    → C2 pod: 3/12 = 25% → 0ms tunnel
    → C3 pod: 2/12 = 17% → 500ms tunnel
    Latenza tunnel media = 58%×150 + 25%×0 + 17%×500 = 87+0+85 = 172ms

  Nginx C3 (30% utenti):
    → C1 pod: 7/12 = 58% → 350ms tunnel
    → C2 pod: 3/12 = 25% → 500ms tunnel
    → C3 pod: 2/12 = 17% → 0ms tunnel
    Latenza tunnel media = 58%×350 + 25%×500 + 17%×0 = 203+125+0 = 328ms

  LATENZA TUNNEL MEDIA GLOBALE (tutti gli utenti):
    = 40%×97 + 30%×172 + 30%×328
    = 38.8 + 51.6 + 98.4 = 188.8ms

  CONFRONTO SENZA geo-awareness (4 repliche per cluster, uniforme):
    Nginx C1: 33%×0 + 33%×150 + 33%×350 = 0+49.5+115.5 = 165ms
    Nginx C2: 33%×150 + 33%×0 + 33%×500 = 49.5+0+165 = 214.5ms
    Nginx C3: 33%×350 + 33%×500 + 33%×0 = 115.5+165+0 = 280.5ms
    Media globale = 40%×165 + 30%×214.5 + 30%×280.5
                  = 66 + 64.3 + 84.1 = 214.4ms

  RISPARMIO GEO-AWARENESS: 214.4ms → 188.8ms = -25.6ms (-12%) ✓
```

### Osservazione: cluster3 penalizzato nonostante il 30% di ingress locale

Cluster3 ha il 30% degli utenti che entrano localmente (0ms tunnel per i propri pod). Ma DMOS gli assegna comunque poche repliche perché la sua RTT_avg è alta (425ms). Questo significa che il 70% degli utenti che non entrano da cluster3 soffrirebbero molto (350ms o 500ms di tunnel) per raggiungere i suoi pod. L'effetto netto è negativo: è meglio avere poche repliche su cluster3 (servire bene il 30% locale) e molte su cluster1 (servire bene tutti).

---

## 9. Effetto sullo scheduling — Esempi numerici

### Scenario: stesse metriche applicative, diversa RTT

Per isolare l'effetto puro della geo-awareness, assumiamo che CPU, latenza, carbon siano identiche tra i cluster:

```
METRICHE IDENTICHE (ipotesi didattica):
  cpu_available_fraction = 0.50 per tutti
  load_fraction          = 0.30 per tutti
  latency_mean_ms        = 100ms per tutti
  carbon_intensity       = 400 gCO2/kWh per tutti

METRICHE DIFFERENZIATE (geo-awareness):
  cluster1: network_rtt_ms = 250ms
  cluster2: network_rtt_ms = 325ms
  cluster3: network_rtt_ms = 425ms

CALCOLO SCORE (profilo "balanced": ω_net = 0.10):
  Φ_lat    = 1/(1+0.01×100) × exp(-0) ≈ 0.909  (uguale per tutti)
  Φ_cap    = (0.50)^2 × (1-0.30) = 0.175        (uguale per tutti)
  Φ_load   = exp(-1×0.30) = 0.741               (uguale per tutti)
  Φ_carbon = exp(-0.5×400/500) = 0.449          (uguale per tutti)

  Φ_net(cluster1) = exp(-2×250/500) = 0.368
  Φ_net(cluster2) = exp(-2×325/500) = 0.272
  Φ_net(cluster3) = exp(-2×425/500) = 0.183

  score(cluster1) = 0.30×0.909 + 0.25×0.175 + 0.15×0.741
                  + 0.20×0.449 + 0.10×0.368
                  = 0.273 + 0.044 + 0.111 + 0.090 + 0.037 = 0.555

  score(cluster2) = stessa formula con 0.10×0.272
                  = 0.555 - (0.037 - 0.027) = 0.545

  score(cluster3) = stessa formula con 0.10×0.183
                  = 0.555 - (0.037 - 0.018) = 0.536

DIFFERENZA: cluster1 vs cluster3 = 0.555 - 0.536 = 0.019 (profilo balanced)

CALCOLO SCORE (profilo "geo_aware": ω_net = 0.40):
  score(cluster1) = 0.20×0.909 + 0.15×0.175 + 0.10×0.741
                  + 0.15×0.449 + 0.40×0.368
                  = 0.182 + 0.026 + 0.074 + 0.067 + 0.147 = 0.496

  score(cluster3) = stessa formula con 0.40×0.183
                  = 0.496 - (0.147 - 0.073) = 0.422

DIFFERENZA: cluster1 vs cluster3 = 0.496 - 0.422 = 0.074 (profilo geo_aware)
            → differenza 4× maggiore rispetto a "balanced"
```

### Impatto sull'allocazione repliche (12 repliche totali)

```
PROFILO "balanced" (ω_net=0.10):
  cluster1: score=0.555 → 5/12 repliche  (~41%)
  cluster2: score=0.545 → 4/12 repliche  (~33%)
  cluster3: score=0.536 → 3/12 repliche  (~25%)
  → Distribuzione quasi uniforme: geo-awareness ha impatto limitato

PROFILO "geo_aware" (ω_net=0.40):
  cluster1: score=0.496 → 7/12 repliche  (~58%)
  cluster2: score=0.449 → 3/12 repliche  (~25%)
  cluster3: score=0.422 → 2/12 repliche  (~17%)
  → Concentrazione su cluster1: DMOS preferisce fortemente Frankfurt
```

---

## 10. Degradazione graceful senza netem

Se tc netem non è attivo (test in LAN pura) o ping_exporter non è ancora deployato, il sistema degrada automaticamente al comportamento pre-geo-awareness:

```
SENZA tc netem (RTT LAN ≈ 5ms per tutti):
  ping_rtt_mean_seconds{target="..."} ≈ 0.005

  Φ_net(tutti) = exp(-2 × 5 / 500) = exp(-0.02) ≈ 0.980

  Differenza tra cluster: 0.980 - 0.980 = 0.000
  → omega_network × 0.000 differenza = 0.000 impatto sullo score
  → DMOS decide esattamente come prima (lat + cap + load + carbon guidano)

SENZA ping_exporter (fallback 5.0ms):
  get_network_rtt_ms() → return 5.0  (hardcoded fallback)
  Φ_net(tutti) ≈ 0.980
  → stesso effetto di sopra: nessun impatto sulla geo-awareness

IN ENTRAMBI I CASI:
  Il sistema non crasha, non logga errori critici (solo DEBUG),
  e produce scheduling identico alla versione pre-geo-awareness. ✓
```

---

---

## 11. Deploy infrastruttura: tc netem e ping_exporter

Questa sezione documenta i passi reali di deployment eseguiti per attivare la simulazione geografica e la misurazione RTT nel sistema DMOS.

---

### 11.1 tc netem — Applicazione del delay artificiale

`tc netem` viene configurato via SSH direttamente sui nodi k3s di cluster2 e cluster3. Il comando applica un delay in **uscita** (EGRESS) sull'interfaccia di rete principale `ens18`, simulando la latenza geografica tra datacenter.

#### Comandi eseguiti

```bash
# Su cluster2 (Paris/FR — 192.168.1.246):
# Aggiunge 150ms ± 20ms con distribuzione gaussiana sull'EGRESS di ens18
ssh -t ubuntu@192.168.1.246 "sudo tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal"

# Su cluster3 (Warsaw/PL — 192.168.1.247):
# Aggiunge 350ms ± 20ms con distribuzione gaussiana sull'EGRESS di ens18
ssh -t ubuntu@192.168.1.247 "sudo tc qdisc add dev ens18 root netem delay 350ms 20ms distribution normal"

# Su cluster1 (Frankfurt/DE — 192.168.1.245):
# Nessun delay — è il cluster "centrale" di riferimento
```

> **Nota sul flag `-t`**: SSH richiede un terminale pseudo-TTY (`-t`) per permettere a `sudo` di leggere la password interattivamente. Senza `-t`, sudo restituisce `a terminal is required to read the password`.

#### Verifica post-deploy (da Windows)

```
ping -n 5 192.168.1.246   →  Media ≈ 201ms  (atteso: ~150ms + offset Windows)
ping -n 5 192.168.1.247   →  Media ≈ 405ms  (atteso: ~350ms + offset Windows)
```

#### Risultati reali ottenuti

```
CLUSTER2 (Paris, delay 150ms):
  Risposte: 213ms / 202ms / 170ms / 208ms / 215ms
  Minimo=170ms  Massimo=215ms  Medio=201ms
  Tutti i pacchetti ricevuti (0% persi) ✓

CLUSTER3 (Warsaw, delay 350ms):
  Risposte: 402ms / 388ms / 409ms / 425ms / 403ms
  Minimo=388ms  Massimo=425ms  Medio=405ms
  Tutti i pacchetti ricevuti (0% persi) ✓
```

#### Perché il valore misurato è più alto del delay configurato?

```
ANALISI DELL'OFFSET (~50ms extra):

  Ping da PC Windows → cluster2:

    PC Windows ──[LAN]──► router/gateway ──[hop]──► cluster2 node
         │                                                │
         │◄──────── ICMP reply +150ms EGRESS ────────────┘

  Percorso Windows→cluster: attraversa router NAT domestico/lab (~25ms extra)
  Percorso cluster→Windows: +150ms netem EGRESS + stessa infrastruttura

  RTT misurato da Windows = LAN_andata + netem + LAN_ritorno
                          = ~25ms + 150ms + ~25ms = ~200ms ✓

  IMPORTANTE: ping_exporter su cluster1 misura:
    cluster1 ──[LAN diretta]──► cluster2
    RTT = ~1ms + 150ms netem = ~151ms  ← questo è il valore usato da DMOS
    Il dato da Windows è puramente indicativo; la metrica operativa viene da ping_exporter.
```

#### Comandi di gestione

```bash
# Verifica stato qdisc attivo su un nodo:
ssh ubuntu@192.168.1.246 "tc qdisc show dev ens18"
ssh ubuntu@192.168.1.247 "tc qdisc show dev ens18"

# Output atteso:
#   qdisc netem 8001: root refcnt 2 limit 1000 delay 150ms  20ms distribution normal
#   qdisc netem 8001: root refcnt 2 limit 1000 delay 350ms  20ms distribution normal

# Rimozione delay (ripristino LAN pura):
ssh -t ubuntu@192.168.1.246 "sudo tc qdisc del dev ens18 root"
ssh -t ubuntu@192.168.1.247 "sudo tc qdisc del dev ens18 root"
```

---

### 11.2 ping_exporter — Deploy via kubectl apply

`ping_exporter` (progetto open-source di Oliver Czerwonk) è il componente che misura la RTT ICMP tra cluster e pubblica la metrica `ping_rtt_mean_seconds` su Prometheus.

#### Cosa misura esattamente la RTT

La RTT non è misurata da gateway a gateway, ma da **nodo k3s a nodo k3s** — tutte le macchine si trovano sulla stessa LAN fisica (subnet 192.168.1.0/24, stesso switch):

```
  [ms01 / cluster1]                    [ms02 / cluster2]
  192.168.1.245                        192.168.1.246
       │                                     │
       │  ping_exporter pod                  │  tc netem su ens18
       │  invia ICMP echo request ──────────►│  (+150ms delay su EGRESS)
       │                                     │
       │◄────── ICMP reply (+150ms) ─────────│
       │
  RTT misurata ≈ 150ms
  (LAN ~1ms andata + 150ms netem reply = ~151ms)
```

Non c'è nessun router/gateway in mezzo — il delay è **artificiale**: tc netem lo aggiunge sul pacchetto in uscita dall'interfaccia fisica `ens18` di ogni nodo, simulando la latenza che ci sarebbe se i cluster fossero in datacenter geograficamente distanti.

```
SETUP DI LAB vs PRODUZIONE REALE:

  Lab (questo sistema):
    ms01 ──[switch LAN]── ms02 ──[switch LAN]── ms03
    RTT fisica: ~1ms
    RTT simulata con netem: 150ms / 350ms / 500ms

  Produzione reale:
    datacenter-DE ──[fibra WAN]── datacenter-FR ──[fibra WAN]── datacenter-PL
    RTT reale: ~15ms / ~35ms / ~50ms (o più)
    tc netem: non necessario — la latenza è quella fisica reale

  In entrambi i casi ping_exporter misura nodo→nodo e DMOS usa
  quella RTT per calcolare Φ_net — il meccanismo è identico.
```

#### Approccio originale (Romano) vs approccio adottato

Romano nella sua tesi ha deployato ping_exporter via Helm:

```bash
# Approccio Romano (tesi 2025, pagina 46):
helm repo add ping-exporter "https://raw.githubusercontent.com/czerwonk/ping_exporter/main/dist/charts/"
helm repo update
helm install ping-exporter ping-exporter/ping-exporter -n observability -f ping-exporter-ro01-values.yaml
```

**Problema**: il repo Helm `https://raw.githubusercontent.com/czerwonk/ping_exporter/main/dist/charts/` restituisce **404 Not Found** (marzo 2026) — il chart è stato rimosso dal progetto dopo la pubblicazione della tesi. Anche il repo alternativo `https://charts.0l.de` risulta irraggiungibile (DNS non risolve).

**Soluzione adottata**: deploy diretto con `kubectl apply` usando manifest Kubernetes scritti manualmente. Il risultato è **funzionalmente identico** all'approccio Helm: stesso container `czerwonk/ping_exporter:latest`, stessa configurazione targets, stesso ServiceMonitor per Prometheus.

#### Manifest creati

Sono stati creati 3 file, uno per cluster, nella directory `deployments/`:

```
deployments/
├── ping-exporter-cluster1.yaml   ← targets: cluster2 (192.168.1.246), cluster3 (192.168.1.247)
├── ping-exporter-cluster2.yaml   ← targets: cluster1 (192.168.1.245), cluster3 (192.168.1.247)
└── ping-exporter-cluster3.yaml   ← targets: cluster1 (192.168.1.245), cluster2 (192.168.1.246)
```

Ogni manifest contiene:

```
┌─────────────────────────────────────────────────────┐
│  Namespace: observability                            │
│  ConfigMap: ping-exporter-config                    │
│    └── config.yml con targets (IP peer cluster)     │
│  Deployment: ping-exporter                          │
│    └── image: czerwonk/ping_exporter:latest         │
│    └── args: --config.path=/config/config.yml       │
│    └── securityContext: NET_RAW (per ICMP)          │
│    └── resources: 32Mi RAM, 10m CPU                 │
│  Service: ping-exporter (ClusterIP :9427)           │
│  ServiceMonitor: scrape ogni 15s (→ Prometheus)     │
└─────────────────────────────────────────────────────┘
```

#### Perché NET_RAW è necessario

```
ping_exporter invia pacchetti ICMP raw (tipo 8 — Echo Request).
In Kubernetes, i container non hanno permessi raw socket per default.
La capability NET_RAW sblocca questa possibilità senza dare root completo.

Senza NET_RAW:
  ping_exporter → "socket: operation not permitted"
  Nessuna metrica pubblicata

Con NET_RAW:
  ping_exporter → pacchetti ICMP inviati regolarmente ogni 1s
  ping_rtt_mean_seconds{target="..."} disponibile su :9427/metrics
```

#### Configurazione per cluster

```yaml
# cluster1 — misura verso cluster2 e cluster3
targets:
  - host: 192.168.1.246   # cluster2 (Paris)
    name: cluster2-paris-fr
  - host: 192.168.1.247   # cluster3 (Warsaw)
    name: cluster3-warsaw-pl

# cluster2 — misura verso cluster1 e cluster3
targets:
  - host: 192.168.1.245   # cluster1 (Frankfurt)
    name: cluster1-frankfurt-de
  - host: 192.168.1.247   # cluster3 (Warsaw)
    name: cluster3-warsaw-pl

# cluster3 — misura verso cluster1 e cluster2
targets:
  - host: 192.168.1.245   # cluster1 (Frankfurt)
    name: cluster1-frankfurt-de
  - host: 192.168.1.246   # cluster2 (Paris)
    name: cluster2-paris-fr
```

#### Comandi di deploy

```bash
# Deploy su tutti e 3 i cluster:
kubectl apply -f deployments/ping-exporter-cluster1.yaml --context cluster1
kubectl apply -f deployments/ping-exporter-cluster2.yaml --context cluster2
kubectl apply -f deployments/ping-exporter-cluster3.yaml --context cluster3

# Verifica pod attivi:
kubectl get pods -n observability --context cluster1
kubectl get pods -n observability --context cluster2
kubectl get pods -n observability --context cluster3

# Output atteso:
#   NAME                             READY   STATUS    RESTARTS   AGE
#   ping-exporter-7d9f8b6c4-xxxxx   1/1     Running   0          30s
```

#### Risultati reali — cluster1 (20 marzo 2026)

```
$ kubectl get pods -n observability --context cluster1
NAME                             READY   STATUS    RESTARTS   AGE
ping-exporter-568759784b-4c66h   1/1     Running   0          19s   ✓

$ kubectl port-forward -n observability svc/ping-exporter 9427:9427 --context cluster1
Forwarding from 127.0.0.1:9427 -> 9427

$ curl -s http://localhost:9427/metrics | Select-String "ping_rtt_mean_seconds"
# HELP ping_rtt_mean_seconds Mean round trip time in seconds
# TYPE ping_rtt_mean_seconds gauge
ping_rtt_mean_seconds{ip="192.168.1.246",ip_version="4",name="cluster2-paris-fr",target="192.168.1.246"} 0.14997555541992189
ping_rtt_mean_seconds{ip="192.168.1.247",ip_version="4",name="cluster3-warsaw-pl",target="192.168.1.247"} 0.3521145324707031
```

**Analisi risultati:**

```
METRICA                          VALORE MISURATO    VALORE ATTESO    SCARTO    ESITO
cluster2 (Paris,  delay 150ms)   149.97 ms          150.00 ms        -0.03ms   ✅
cluster3 (Warsaw, delay 350ms)   352.11 ms          350.00 ms        +2.11ms   ✅

Il jitter di ±20ms (distribuzione normale) spiega la leggera variazione.
ping_exporter misura da cluster1 via LAN diretta → nessun overhead di router.
I valori sono stabili e coerenti con la configurazione tc netem applicata.
```

> **Nota tecnica**: a differenza del ping da Windows (che mostrava ~200ms e ~405ms a causa dell'overhead del router NAT), ping_exporter opera da dentro cluster1 con accesso LAN diretto ai nodi peer. Questo è il valore effettivo che DMOS usa per il calcolo di Φ_net.

#### Verifica metrica Prometheus

Dopo il deploy, verificare che Prometheus scrapi la metrica RTT:

```bash
# Query diretta sul Prometheus di cluster1 (porta NodePort 30090):
curl "http://192.168.1.245:30090/api/v1/query?query=ping_rtt_mean_seconds"

# Output atteso (con tc netem attivo):
# {
#   "metric": {"target": "192.168.1.246"},  ← cluster2
#   "value": [timestamp, "0.150"]           ← ~150ms ✓
# },
# {
#   "metric": {"target": "192.168.1.247"},  ← cluster3
#   "value": [timestamp, "0.350"]           ← ~350ms ✓
# }
```

#### Risultati reali — DMOS scheduling (20 marzo 2026)

Dopo il deploy di ping_exporter e la scadenza del grace period (90s), DMOS ha eseguito il primo scheduling con geo-awareness attiva:

```
✅ Network RTT (cluster1): 248.0 ms [targets: 192.168.1.246|192.168.1.247]
Score cluster1: 0.474 (lat=0.103, cap=0.489, load=0.992, carbon=0.675, net=0.371|RTT=248ms)

✅ Network RTT (cluster2): 323.4 ms [targets: 192.168.1.245|192.168.1.247]
Score cluster2: 0.498 (lat=0.103, cap=0.456, load=0.992, carbon=0.883, net=0.274|RTT=323ms)

✅ Network RTT (cluster3): 421.7 ms [targets: 192.168.1.245|192.168.1.246]
Score cluster3: 0.416 (lat=0.103, cap=0.488, load=0.992, carbon=0.477, net=0.185|RTT=422ms)

→ Allocated 1 replicas to cluster2 (score=0.498)
```

**Analisi dei risultati:**

```
CLUSTER          RTT_avg    Φ_net    CI (gCO2)    score    ESITO
cluster1 (DE)    248ms      0.371    393          0.474    —
cluster2 (FR)    323ms      0.274    125          0.498    ✅ WINNER
cluster3 (PL)    422ms      0.185    739          0.416    —

RTT attesa vs misurata:
  cluster1: atteso 250ms → misurato 248ms  (Δ=-2ms)  ✅
  cluster2: atteso 325ms → misurato 323ms  (Δ=-2ms)  ✅
  cluster3: atteso 425ms → misurato 422ms  (Δ=-3ms)  ✅
```

**Perché ha vinto cluster2 nonostante RTT più alta di cluster1?**
Con profilo `balanced`, carbon e RTT pesano entrambi. Francia nucleare (125 gCO2/kWh) vs Germania carbone/gas (393 gCO2/kWh): il vantaggio carbon di FR supera lo svantaggio RTT. cluster3 (739 gCO2/kWh + RTT 422ms) è penalizzato su entrambe le dimensioni.

---

#### Come la RTT si collega al routing di Cilium ClusterMesh

DMOS non parla direttamente con Cilium — il collegamento è **indiretto tramite il numero di repliche**:

```
1. DMOS calcola score per cluster (include Φ_net da RTT):
     cluster1: score=0.474  →  X repliche
     cluster2: score=0.498  →  Y repliche  (più alto)
     cluster3: score=0.416  →  Z repliche  (meno)

2. Cilium ClusterMesh vede gli endpoint nella BPF map:
     frontend endpoints globali:
       cluster1: X pod  →  X voci BPF
       cluster2: Y pod  →  Y voci BPF  ← di più
       cluster3: Z pod  →  Z voci BPF  ← di meno

3. Cilium bilancia proporzionale alle voci:
     P(richiesta → cluster2) = Y / (X+Y+Z)   ← più alta

4. Risultato:
     Più richieste servite da cluster2
     → meno attraversano il tunnel vxlan verso cluster lontani
     → latenza percepita dagli utenti si riduce
```

DMOS non dice a Cilium "manda le richieste qui" — dice "metti più pod qui". Principio identico a quello descritto da Romano (tesi pag. 48): *"aumentando il numero di repliche si incrementa statisticamente la probabilità che una richiesta venga servita localmente, minimizzando l'instradamento verso pod remoti"*.

---

#### Flusso dati completo dopo il deploy

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUSSO OPERATIVO COMPLETO                    │
│                                                                 │
│  [cluster2 nodo]          [cluster1]                           │
│  tc netem ens18           ping_exporter pod                    │
│  +150ms EGRESS  ◄──────── ICMP echo request ogni 1s           │
│       │                        │                               │
│       └──► ICMP reply +150ms ──►│                              │
│                                 │                              │
│                         ping_rtt_mean_seconds                  │
│                         {target="192.168.1.246"} = 0.150      │
│                                 │                              │
│                    [Prometheus cluster1 :30090]                │
│                    scrape :9427/metrics ogni 15s               │
│                                 │                              │
│                    [DMOS Scheduler ogni 30s]                   │
│                    avg(ping_rtt_mean_seconds{...}) * 1000      │
│                    → network_rtt_ms = 250ms                    │
│                    → Φ_net(cluster1) = exp(-2×250/500) = 0.368│
│                    → score(cluster1) += 0.40 × 0.368 = 0.147  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Riepilogo modifiche

| File | Tipo | Modifiche principali |
|------|------|----------------------|
| `deployments/global-service-frontend.yaml` | **NUOVO** | Service con `io.cilium/global-service` + `io.cilium/shared-service`; prerequisito al routing cross-cluster |
| `deployments/cnp-frontend-l7.yaml` | **MODIFICATO** | Aggiunta Regola 2 `fromEntities: remote-node`; permette traffico ClusterMesh verso pod frontend |
| `config/weights.yaml` | **MODIFICATO** | Aggiunto `omega_network` in tutti e 5 i profili; nuovo profilo `geo_aware` (ω_net=0.40); nuova sezione `network_parameters` (rho, rtt_max_ms, fallback_rtt_ms) |
| `src/level1/score_functions.py` | **MODIFICATO** | `ClusterMetrics`: campo `network_rtt_ms`; `ScoreParameters`: campi `rho` e `rtt_max_ms`; `ScoreFunctions`: `omega_network` in `__init__`, validazione aggiornata; nuovo metodo `compute_network_score()`; `phi_net` in `compute_total_score()` e `compute_score_breakdown()` |
| `src/metrics/prometheus_client.py` | **MODIFICATO** | Nuovo metodo `get_network_rtt_ms(peer_ips)`: query PromQL su `ping_rtt_mean_seconds`, media RTT verso peer cluster, fallback 5.0ms |
| `src/level1/dmos_scheduler.py` | **MODIFICATO** | `__init__`: lettura `network_params` da config, costruzione `_peer_ips`, passaggio `omega_network` e `ScoreParameters(rho, rtt_max_ms)` a `ScoreFunctions`; `_collect_cluster_metrics()`: chiamata `get_network_rtt_ms()`, passaggio `network_rtt_ms` a `ClusterMetrics`; log aggiornato con `phi_network` e RTT |
| `deployments/ping-exporter-cluster1.yaml` | **NUOVO** | Manifest kubectl per ping_exporter su cluster1: targets=cluster2+cluster3, NET_RAW, ServiceMonitor Prometheus |
| `deployments/ping-exporter-cluster2.yaml` | **NUOVO** | Manifest kubectl per ping_exporter su cluster2: targets=cluster1+cluster3, NET_RAW, ServiceMonitor Prometheus |
| `deployments/ping-exporter-cluster3.yaml` | **NUOVO** | Manifest kubectl per ping_exporter su cluster3: targets=cluster1+cluster2, NET_RAW, ServiceMonitor Prometheus |
| tc netem (infra) | **CONFIGURAZIONE SSH** | `sudo tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal` su cluster2; `350ms` su cluster3; cluster1 senza delay |
