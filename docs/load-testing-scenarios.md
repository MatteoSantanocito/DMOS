# Load Testing Scenarios — DMOS

Documentazione degli scenari di load test implementati in
`experiments/locustfile_multiingress.py` (v5).

---

## Indice

1. [Architettura di test](#architettura-di-test)
2. [Modalità di ingress](#modalità-di-ingress)
3. [Scenari disponibili](#scenari-disponibili)
4. [Setup netem](#setup-netem)
5. [Configurazione pesi ingress](#configurazione-pesi-ingress)
6. [Output e metriche](#output-e-metriche)
7. [Piano esperimenti completo](#piano-esperimenti-completo)
8. [Confronto con Romano 2025](#confronto-con-romano-2025)

---

## Architettura di test

```
Locust (Windows, ms01)
    │
    │  ── Single Ingress ──▶  192.168.1.245:30080 (cluster1, DE)
    │                              │
    │                         Cilium Cluster Mesh
    │                         (load balance proporzionale agli endpoint)
    │                              │
    │              ┌───────────────┼───────────────┐
    │           Frontend-c1     Frontend-c2     Frontend-c3
    │           (cluster1/DE)  (cluster2/FR)  (cluster3/PL)
    │
    └── Multi-Ingress (Romano-like, probabilistico) ──▶
             60% → 192.168.1.245:30080 (c1-DE, 0ms netem)
             25% → 192.168.1.246:30080 (c2-FR, +150ms netem)
             15% → 192.168.1.247:30080 (c3-PL, +350ms netem)
```

### Differenza chiave: Single vs Multi-Ingress

| Aspetto | Single Ingress | Multi-Ingress |
|---|---|---|
| Punto di entrata | Sempre cluster1 | Probabilistico sui 3 cluster |
| Effetto netem | Solo routing interno cross-cluster | Anche percorso k6→ingress |
| Visibilità Φ_demand | Tutto traffico su c1 | Distribuzione realistica per cluster |
| Comparabilità Romano 2025 | Parziale | Diretta |
| Scenario realistico | LAN con LB centralizzato | Utenti geograficamente distribuiti |

Con il **multi-ingress**, un utente che "entra" da cluster2 (FR) subisce già
150ms di latenza di risposta prima di qualsiasi routing interno — esattamente
come un utente reale a Parigi che raggiunge un datacenter francese con RTT 150ms.
Con DMOS OFF, il frontend di cluster2 può servire internamente oppure fare una
chiamata cross-cluster verso cluster1/3 (con ulteriore netem). Con DMOS ON,
DMOS concentra le repliche frontend vicino all'ingress con più traffico,
riducendo le chiamate cross-cluster e il p95 globale.

---

## Modalità di ingress

### Single Ingress (default)

Tutte le richieste Locust vanno a `http://192.168.1.245:30080` (cluster1).
Cilium Cluster Mesh distribuisce internamente le richieste ai pod in base al
numero di endpoint attivi per cluster (proporzionale alle repliche scalate da DMOS).

**Quando usarlo:** test senza netem (LAN pura), benchmark di throughput,
scenari dove il confronto ON/OFF si vede sull'autoscaling.

### Multi-Ingress (Romano-like)

Locust seleziona l'ingress di destinazione **probabilisticamente** a ogni
richiesta, usando i pesi configurati in `MULTI_INGRESS_CONFIG`. Replica
fedelmente il setup di Romano 2025 §5.3.

**Meccanismo di selezione:**
```python
r = random.random() * peso_totale
# seleziona il primo ingress il cui peso cumulativo supera r
```

**Quando usarlo:** test con netem attivo, confronto diretto con Romano 2025,
dimostrazione geo-awareness DMOS (Φ_demand + Φ_net).

---

## Scenari disponibili

Seleziona lo scenario con la variabile d'ambiente `DMOS_SCENARIO`.

### `flash_crowd` — Single ingress, 400 utenti

```
warm-up  │ flash-spike │         sustained-peak (10min)        │ decline │ cooldown
10 users │ 10→400 (1m) │ 400 users (10min)                     │ 400→10  │ 10 users
  (2m)   │             │                                        │  (5m)   │  (2m)
─────────┴─────────────┴────────────────────────────────────────┴─────────┴─────────
Durata totale: 1200s (20 min)
```

- **Ingress:** single (cluster1)
- **Netem:** NO (LAN pura)
- **Uso:** stress test throughput, benchmark senza penalità geografiche
- **Peak users:** 400

```powershell
$env:DMOS_SCENARIO = "flash_crowd"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m
```

---

### `flash_crowd_netem` — Single ingress, 150 utenti, con netem

```
warm-up  │ flash-spike │      sustained-peak (10min)      │ decline │ cooldown
10 users │ 10→150 (1m) │ 150 users (10min)                │ 150→10  │ 10 users
  (2m)   │             │                                  │  (3m)   │  (2m)
─────────┴─────────────┴──────────────────────────────────┴─────────┴─────────
Durata totale: 1080s (18 min)
```

- **Ingress:** single (cluster1)
- **Netem:** SI (ms02=150ms, ms03=350ms)
- **Uso:** baseline comparabile con Romano 2025 Scenario 2, ma con single ingress
- **Peak users:** 150 (equivalente a ~150 req/s secondo Romano)

```powershell
$env:DMOS_SCENARIO = "flash_crowd_netem"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m
```

---

### `flash_crowd_multiingress` — Multi-ingress uniforme, 150 utenti

```
warm-up  │ flash-spike │      sustained-peak (10min)      │ decline │ cooldown
10 users │ 10→150 (1m) │ 150 users (10min)                │ 150→10  │ 10 users
  (2m)   │             │                                  │  (3m)   │  (2m)
─────────┴─────────────┴──────────────────────────────────┴─────────┴─────────
Durata totale: 1080s (18 min)

Distribuzione ingress:
  c1-DE (0ms):   33.4% delle richieste
  c2-FR (150ms): 33.3% delle richieste
  c3-PL (350ms): 33.3% delle richieste
```

- **Ingress:** multi (3 cluster, distribuzione uniforme 1/3)
- **Netem:** SI — obbligatorio per effetto geo
- **Uso:** confronto DIRETTO con Romano 2025 Scenario 2
- **Peak users:** 150

**Effetto atteso:**
- **DMOS OFF:** p95 elevato (~1500-2000ms) perché 66% delle richieste entra da
  cluster con netem (+150ms o +350ms) e può essere servito cross-cluster
- **DMOS ON:** DMOS concentra repliche su c1 (score alto: Φ_demand=33%, Φ_net alto)
  → Cilium serve localmente le richieste di c1, riducendo p95 globale

```powershell
$env:DMOS_SCENARIO = "flash_crowd_multiingress"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m
```

---

### `flash_crowd_geo` — Multi-ingress asimmetrico, 150 utenti

```
Distribuzione ingress (default: 45/25/30):
  c1-DE (0ms):   45% delle richieste  ← flash crowd su DE
  c2-FR (150ms): 25% delle richieste
  c3-PL (350ms): 30% delle richieste  ← PL pesante per baseline OFF peggiore

Penalità netem media all'entrata:
  OFF: 0.45×0 + 0.25×150 + 0.30×350 = 142ms (vs 90ms con 60/25/15)
```

- **Ingress:** multi (distribuzione asimmetrica, configurabile via env)
- **Netem:** SI
- **Uso:** dimostra che DMOS con Φ_demand + Φ_net ottimizza in modo non banale
- **Peak users:** 150

**Perché 45/25/30 e non 60/25/15:**

| Distribuzione | Penalità avg entrata | Difficoltà per DMOS | Miglioramento ON/OFF |
|---|---|---|---|
| 60/25/15 | 90ms | Bassa (Φ_demand c1 dominante) | Visibile ma prevedibile |
| **45/25/30** | **142ms** | **Media (deve bilanciare c1 vs c3)** | **Più pronunciato** |
| 33/33/33 | 165ms | Alta (nessun hotspot) | Massimo ma meno geo-aware |

Con 45/25/30, DMOS deve bilanciare due segnali contrastanti:
- `Φ_demand(c1) = 0.45` → scala su DE (flash crowd)
- `Φ_net(c3) bassa + 30% ingress su PL` → evita di mandare traffico cross-cluster verso PL

Con DMOS OFF, il 30% delle richieste che entra da PL può essere servito
cross-cluster verso c1/c2 aggiungendo ulteriore latenza ai 350ms di ingress.

```powershell
# Configura pesi (flash crowd su DE)
$env:DMOS_INGRESS_W1 = "0.60"
$env:DMOS_INGRESS_W2 = "0.25"
$env:DMOS_INGRESS_W3 = "0.15"
$env:DMOS_SCENARIO = "flash_crowd_geo"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m
```

---

### `gradual_ramp` — Rampa graduale, 300 utenti

```
warm-up │      ramp-up (10min)      │ peak (5min) │ ramp-down │ cooldown
10 usr  │ 10 → 300                  │ 300 users   │ 300 → 10  │ 10 users
  (2m)  │                           │             │   (5m)    │  (2m)
────────┴───────────────────────────┴─────────────┴───────────┴─────────
Durata totale: 1440s (24 min)
```

- **Ingress:** single
- **Uso:** osservare il comportamento DMOS durante scaling graduale

---

### `double_wave` — Doppia ondata, 250 utenti max

```
warm│ wave1-ramp │ wave1-peak │ valley-d │ valley │ wave2-ramp │ wave2-peak │ final-d │ cool
 up │ 20→200     │ 200 users  │ 200→50   │  50    │ 50→250     │ 250 users  │ 250→20  │
────┴────────────┴────────────┴──────────┴────────┴────────────┴────────────┴─────────┴────
Durata totale: 1440s (24 min)
```

- **Ingress:** single
- **Uso:** testa la capacità di DMOS di reagire a due picchi consecutivi e
  al declino intermedio (scale-up, scale-down, scale-up di nuovo)

---

### `sinusoidal` — Traffico sinusoidale

```
warm-up │         sinusoidal 30min (periodo 6min)         │ cooldown
40 usr  │ min=40, max=200, sin(2π·t/360)                  │ 40 users
  (2m)  │                                                  │  (2m)
────────┴──────────────────────────────────────────────────┴─────────
Durata totale: 2040s (34 min)
```

- **Ingress:** single
- **Uso:** osservare la risposta DMOS a variazioni periodiche prevedibili —
  testa la componente predittiva (Φ_pred) del sistema

---

## Setup netem

I seguenti scenari richiedono netem attivo su ms02 e ms03:

| Scenario | Netem richiesto |
|---|---|
| `flash_crowd` | NO |
| `flash_crowd_netem` | SI |
| `flash_crowd_multiingress` | SI (obbligatorio) |
| `flash_crowd_geo` | SI (obbligatorio) |
| `gradual_ramp` | opzionale |
| `double_wave` | opzionale |
| `sinusoidal` | opzionale |

### Applicare netem (da eseguire dopo ogni reboot)

```bash
# ms02 (cluster2, Paris — 150ms ±20ms)
sudo tc qdisc del dev ens18 root 2>/dev/null || true
sudo tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal

# ms03 (cluster3, Warsaw — 350ms ±20ms)
sudo tc qdisc del dev ens18 root 2>/dev/null || true
sudo tc qdisc add dev ens18 root netem delay 350ms 20ms distribution normal
```

### Verificare netem

```bash
# Output atteso con netem attivo:
# qdisc netem 8001: root refcnt 2 limit 1000 delay 150ms  20ms
ssh utente@192.168.1.246 "tc qdisc show dev ens18"
ssh utente@192.168.1.247 "tc qdisc show dev ens18"
```

### Rimuovere netem

```bash
sudo tc qdisc del dev ens18 root
```

### Rendere netem persistente (systemd)

```ini
# /etc/systemd/system/netem-delay.service
[Unit]
Description=Apply netem delay for geo simulation
After=network.target

[Service]
Type=oneshot
# Su ms02: 150ms; su ms03: sostituire con 350ms
ExecStart=/sbin/tc qdisc add dev ens18 root netem delay 150ms 20ms distribution normal
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable netem-delay
sudo systemctl start netem-delay
```

---

## Configurazione pesi ingress

I pesi per il multi-ingress si configurano via variabili d'ambiente **prima**
di avviare Locust. Se non specificati, la distribuzione è uniforme (33/33/33).

| Variabile | Cluster | Descrizione |
|---|---|---|
| `DMOS_INGRESS_W1` | cluster1 (DE) | Peso ingress Frankfurt (default: 0.334) |
| `DMOS_INGRESS_W2` | cluster2 (FR) | Peso ingress Paris +150ms (default: 0.333) |
| `DMOS_INGRESS_W3` | cluster3 (PL) | Peso ingress Warsaw +350ms (default: 0.333) |

I pesi **non devono sommare a 1.0**: vengono normalizzati automaticamente
internamente. Puoi usare numeri interi (1/1/1 = uniforme) o percentuali (60/25/15).

### Esempi

```powershell
# Uniforme (Romano-like) — default
# $env:DMOS_INGRESS_W1 = "0.334"  # non necessario, è il default

# Flash crowd su DE (60% DE, 25% FR, 15% PL)
$env:DMOS_INGRESS_W1 = "0.60"
$env:DMOS_INGRESS_W2 = "0.25"
$env:DMOS_INGRESS_W3 = "0.15"

# Stress su cluster ad alta carbon intensity (PL pesante)
$env:DMOS_INGRESS_W1 = "0.20"
$env:DMOS_INGRESS_W2 = "0.20"
$env:DMOS_INGRESS_W3 = "0.60"

# Ripristina default (rimuovi le variabili)
Remove-Item Env:DMOS_INGRESS_W1, Env:DMOS_INGRESS_W2, Env:DMOS_INGRESS_W3
```

---

## Output e metriche

Al termine di ogni test, Locust produce:

### Console

```
======================================================================
  GLOBAL LATENCY — flash_crowd_multiingress
======================================================================

  Requests : 85234
  Fail%    : 0.3%
  Avg      : 520ms
  p50      : 380ms
  p90      : 890ms
  p95      : 1240ms
  p99      : 2100ms
  SLO>1000ms: 8.4%

  PER-INGRESS LATENCY BREAKDOWN
  ──────────────────────────────────────────────────────────────────
  Ingress    netem        n      avg      p95   fail%    SLO%
  ──────────────────────────────────────────────────────────────────
  c1-DE        0ms    28500    210ms    420ms    0.1%    0.5%
  c2-FR      150ms    28400    540ms   1050ms    0.2%    8.2%
  c3-PL      350ms    28300    820ms   1680ms    0.6%   22.1%
  ──────────────────────────────────────────────────────────────────
```

### File CSV

| File | Contenuto |
|---|---|
| `results/multiingress/{scenario}_summary_{ts}.csv` | Metriche globali aggregate |
| `results/multiingress/{scenario}_timeseries_{ts}.csv` | Serie temporale (finestre 10s) |
| `results/multiingress/{scenario}_per_ingress_{ts}.csv` | Latenza per-ingress (solo multi-ingress) |

### JSONL (collect_metrics_simple.py)

Il collector DMOS produce un file JSONL con metriche Kubernetes/Prometheus ogni
15 secondi, incluso il traffico per-cluster (`_traffic_pct.frontend`) misurato
da Nginx Ingress Controller.

---

## Piano esperimenti completo

Per la tesi sono necessari i seguenti test (in ordine):

### Fase A — Baseline LAN (senza netem)

| Test | Scenario | DMOS | File risultati |
|---|---|---|---|
| A1 | `flash_crowd` | OFF | `flash_crowd_off_lan` |
| A2 | `flash_crowd` | ON (balanced) | `flash_crowd_on_balanced_lan` |
| A3 | `flash_crowd` | ON (romano_like) | `flash_crowd_on_romano_lan` |

### Fase B — Con netem, single ingress

| Test | Scenario | DMOS | File risultati |
|---|---|---|---|
| B1 | `flash_crowd_netem` | OFF | `flash_crowd_netem_off` |
| B2 | `flash_crowd_netem` | ON (balanced) | `flash_crowd_netem_on_balanced` |
| B3 | `flash_crowd_netem` | ON (romano_like) | `flash_crowd_netem_on_romano` |

### Fase C — Con netem, multi-ingress (confronto diretto Romano 2025)

| Test | Scenario | DMOS | Note |
|---|---|---|---|
| C1 | `flash_crowd_multiingress` | OFF | baseline Romano-like |
| C2 | `flash_crowd_multiingress` | ON (balanced) | DMOS balanced |
| C3 | `flash_crowd_multiingress` | ON (romano_like) | DMOS romano |
| C4 | `flash_crowd_geo` (60/25/15) | OFF | baseline geo-skewed |
| C5 | `flash_crowd_geo` (60/25/15) | ON (geo_aware) | DMOS geo-aware |

### Comandi PowerShell per Fase C

```powershell
# C1 — OFF baseline
$env:DMOS_SCENARIO = "flash_crowd_multiingress"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m --no-dmos

# C2 — ON balanced (imposta active: "balanced" in config/weights.yaml)
$env:DMOS_SCENARIO = "flash_crowd_multiingress"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m

# C3 — ON romano_like (imposta active: "romano_like" in config/weights.yaml)
$env:DMOS_SCENARIO = "flash_crowd_multiingress"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m

# C4 — OFF geo-skewed baseline
$env:DMOS_INGRESS_W1 = "0.60"; $env:DMOS_INGRESS_W2 = "0.25"; $env:DMOS_INGRESS_W3 = "0.15"
$env:DMOS_SCENARIO = "flash_crowd_geo"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m --no-dmos

# C5 — ON geo_aware (imposta active: "geo_aware" in config/weights.yaml)
$env:DMOS_SCENARIO = "flash_crowd_geo"
locust -f experiments/locustfile_multiingress.py --autostart --web-host 0.0.0.0 --web-port 8089 --run-time 20m
```

---

## Confronto con Romano 2025

### Setup Romano §4 (da tesi)

| Parametro | Romano 2025 | DMOS (questo progetto) |
|---|---|---|
| Cluster | 3 VM K3s + Cilium | 3 VM K3s + Cilium |
| netem c2 | 150ms ±20ms su ens18 | 150ms ±20ms su ens18 |
| netem c3 | 350ms ±20ms su ens18 | 350ms ±20ms su ens18 |
| Applicazione | Online Boutique | Online Boutique |
| Load tester | k6 | Locust |
| Ingress | 3 separati (uno per cluster) | 3 separati (multi-ingress) |
| Selezione ingress | Probabilistica (k6) | Probabilistica (Locust) |
| Metrica traffico | `hubble_http_requests_total{source="reserved:ingress"}` | `nginx_ingress_controller_requests` |
| Metrica RTT | `ping_rtt_mean_seconds` | `ping_rtt_mean_seconds` |

### Risultati Romano (Scenario 1 — traffico crescente, 10min)

| Configurazione | Avg | P90 | P95 |
|---|---|---|---|
| Metascheduler OFF | 1.09s | 1.70s | 3.88s |
| Metascheduler ON | 445ms | 1.01s | 1.23s |
| **Miglioramento** | **-59%** | **-41%** | **-68%** |

### Risultati Romano (Scenario 2 — flash crowd ~150 req/s)

| Configurazione | P95 | Errori |
|---|---|---|
| Metascheduler OFF | 1650ms | 19 |
| Metascheduler ON | 1320ms | 0 |
| **Miglioramento** | **-20%** | **-100%** |

### Differenze architetturali DMOS vs Romano

| Componente | Romano | DMOS |
|---|---|---|
| Score formula | `WT*traffic_locale + WL*latenza_remota` | 6 metriche pesate (Φ_lat, Φ_cap, Φ_load, Φ_carbon, Φ_net, Φ_demand) |
| Carbon awareness | Assente | Si (Φ_carbon via ElectricityMaps/fallback) |
| Scaling proattivo | Assente | Si (predizione DMOS con lookahead) |
| Autoscaling backends | Assente | Si (cart, catalog, checkout, recommendation) |
| Max repliche | 15 (env var) | 8 (configurabile in services.yaml) |
| Intervallo decisione | 60s | ~44s (collector interval) |

La modalità `flash_crowd_multiingress` con profilo `romano_like`
(`omega_network=0.40, omega_demand=0.35, omega_carbon=0.00`) è il confronto
più diretto con Romano 2025: stessi pesi, stessa distribuzione ingress,
stessa infrastruttura.
