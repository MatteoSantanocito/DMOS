# Throttle Check — Documentazione

> Manuale dello strumento `experiments/throttle_check.sh`: cosa verifica,
> come funziona, come si interpreta il risultato.
>
> **In una frase**: verifica che, scalando il frontend a `max_replicas` su
> un cluster, le repliche **non** vadano in CFS throttling — cioe' che il
> modello DMOS "capacita cluster = capacity_req_per_sec x max_replicas"
> sia **lineare ed esatto**.
>
> **Cross-reference**:
> - `experiments/throttle_check.sh` — lo script
> - `experiments/k6_throttle_load.js` — il load generator usato dallo script
> - `dmos-iac/HETEROGENEOUS-CLUSTER-SETUP.md` sez. 4.4 — il modello di capacita'
> - `dmos-iac/RUNBOOK.md` FASE 6 — dove sta nella pipeline
>   (`dmos-iac` è il repository esterno di provisioning del fleet, non incluso qui)

---

## Indice

1. [Cos'e' e perche' serve](#1-cose-e-perche-serve)
2. [Il problema: CFS throttling e modello lineare](#2-il-problema-cfs-throttling-e-modello-lineare)
3. [Come funziona lo script](#3-come-funziona-lo-script)
4. [La metrica chiave: CFS throttled periods](#4-la-metrica-chiave-cfs-throttled-periods)
5. [Interpretazione del verdetto](#5-interpretazione-del-verdetto)
6. [Uso: comandi e variabili env](#6-uso-comandi-e-variabili-env)
7. [Output di esempio](#7-output-di-esempio)
8. [Cosa fare se il verdetto e' FAIL](#8-cosa-fare-se-il-verdetto-e-fail)
9. [Throttle check vs Capacity test](#9-throttle-check-vs-capacity-test)
10. [Quando rieseguirlo](#10-quando-rieseguirlo)

---

## 1. Cos'e' e perche' serve

Il fleet DMOS e' eterogeneo a 3 tier (vedi `HETEROGENEOUS-CLUSTER-SETUP.md`):

| Tier | Worker | `max_replicas` frontend |
|---|---|---|
| HIGH | 8 vCPU | 6 |
| MEDIUM | 6 vCPU | 5 |
| LOW | 4 vCPU | 3 |

(HIGH: fisicamente il worker reggerebbe 8 repliche frontend; il cap e' 6
per la colocation coi backend — vedi commento in `config/services.yaml`.)

DMOS modella la capacita' di un cluster come:

```
capacita' cluster = capacity_req_per_sec (60) x max_replicas_per_cluster
```

Questo modello e' **lineare**: 8 repliche danno 8x il throughput di 1 replica.
Ma e' lineare **solo se** il worker ha CPU sufficiente perche' ogni replica
giri senza essere strozzata. Se `max_replicas` viene impostato troppo alto
per quel worker, le repliche vanno in **CFS throttling** e il throughput
diventa **sublineare** (N repliche danno meno di Nx).

`throttle_check.sh` verifica empiricamente che i `max_replicas` scelti
(6/5/3) siano sotto la soglia di throttling. E' un controllo di sanita'
**leggero** (~4 min per cluster), da eseguire dopo ogni re-provisioning.

---

## 2. Il problema: CFS throttling e modello lineare

### 2.1 Cos'e' il CFS throttling

Il kernel Linux usa il **CFS** (Completely Fair Scheduler) per ripartire
la CPU. Quando un container ha un **CPU limit** (il pod `frontend` ha
limit 500m), il kernel gli assegna una **quota** di tempo CPU per ogni
**periodo** (default 100ms):

- limit 500m = 0.5 core = 50ms di CPU ogni periodo da 100ms
- se il container usa tutti i 20ms prima della fine del periodo, viene
  **throttled**: sospeso fino all'inizio del periodo successivo

Il throttling NON e' un errore — e' il CPU limit che fa il suo lavoro.
Ma se un pod e' throttled spesso, significa che **vuole piu' CPU di
quanta gliene viene concessa** -> la sua latency sale e il suo
throughput cala.

### 2.2 Perche' impatta il modello lineare

`capacity_req_per_sec = 60` e' stato misurato con il **capacity test**
su 1 replica **isolata** (worker scarico). In quel regime il pod ottiene
sempre la CPU che gli serve.

Quando DMOS scala a N repliche, ci sono due regimi:

| Regime | Condizione | Capacita' per replica |
|---|---|---|
| Lineare | `N x 500m + altri pod <= CPU worker` | ~60 rps (uniforme) |
| Sublineare | il worker satura, le repliche competono | < 60 rps (degrada) |

Il `throttle_check` distingue i due regimi: se a `max_replicas` + carico
80% il throttle ratio e' ~0, siamo nel regime **lineare** -> il modello
DMOS e' esatto. Se il ratio e' alto, siamo nel regime **sublineare**
-> `max_replicas` e' troppo alto e va ridotto.

### 2.3 Budget CPU (perche' ci aspettiamo throttling ~0)

Calcolo del budget per i 3 tier (CPU limit pod frontend = 500m):

| Tier | Worker | frontend (max_repl x 500m) | altri pod | Totale | Utilizzo |
|---|---|---|---|---|---|
| HIGH | 8000m | 6 x 500 = 3000m | ~2200m | 5200m | 65% |
| MEDIUM | 6000m | 5 x 500 = 2500m | ~2200m | 4700m | 78% |
| LOW | 4000m | 3 x 500 = 1500m | ~2200m | 3700m | 93% |

Tutti i tier restano sotto la CPU del worker (budget worst-case a limit
pieno: l'uso reale sta ben sotto i limit) -> ci aspettiamo throttle
ratio ~0 su tutti. Il `throttle_check` lo conferma empiricamente.

---

## 3. Come funziona lo script

`throttle_check.sh`, per il cluster target, esegue 8 passi:

1. **Pre-flight** — verifica `k6`, `kubectl`, `jq`, `curl`, `python`;
   trova il kubeconfig e l'URL Prometheus del cluster.
2. **Legge `max_replicas`** del cluster da `config/services.yaml`
   (`frontend.max_replicas_per_cluster`).
3. **Scala il frontend** a `max_replicas` repliche sul cluster target,
   aspetta che il rollout sia completo.
4. **Snapshot Prometheus PRE** — legge i contatori CFS cumulativi.
5. **Genera carico** con `k6_throttle_load.js`: rate costante all'80%
   della capacita' (= 0.8 x 60 x max_replicas) per ~3 min.
6. **Snapshot Prometheus POST** — ri-legge i contatori CFS + CPU usata.
7. **Calcola il throttle ratio** e da' il verdetto.
8. **Restore** — scala il frontend a 1 replica (anche su Ctrl+C, via trap).

Carico generato per tier (80% della capacita'):

| Tier | max_replicas | Capacita' (60 x N) | Carico 80% |
|---|---|---|---|
| HIGH | 6 | 360 rps | 288 rps |
| MEDIUM | 5 | 300 rps | 240 rps |
| LOW | 3 | 180 rps | 144 rps |

---

## 4. La metrica chiave: CFS throttled periods

Lo script interroga due contatori cAdvisor (esposti da kubelet, scrapati
da Prometheus):

| Metrica | Significato |
|---|---|
| `container_cpu_cfs_periods_total` | numero totale di periodi CFS trascorsi per il container |
| `container_cpu_cfs_throttled_periods_total` | numero di periodi in cui il container e' stato throttled (ha colpito il CPU limit) |

Il **throttle ratio** e':

```
                sum(rate(container_cpu_cfs_throttled_periods_total{pod=~"frontend-.*"}[2m]))
throttle_ratio = ----------------------------------------------------------------------------
                sum(rate(container_cpu_cfs_periods_total{pod=~"frontend-.*"}[2m]))
```

Interpretazione:
- **ratio = 0.00** -> i pod frontend non colpiscono MAI il loro CPU limit.
  Hanno tutta la CPU che serve. Modello lineare perfetto.
- **ratio = 0.10** -> il 10% dei periodi CFS sono throttled. I pod a volte
  colpiscono il limit. Throttling lieve.
- **ratio = 0.50** -> meta' dei periodi throttled. I pod sono CPU-starved.
  La loro capacita' effettiva e' molto sotto i 60 rps. Modello sublineare.

Lo script usa una finestra `[2m]` (gli ultimi 2 minuti = lo steady-state
del carico generato per 3 min).

---

## 5. Interpretazione del verdetto

Lo script confronta il throttle ratio con la soglia `DMOS_THROTTLE_WARN`
(default 0.05 = 5%) e produce 3 verdetti:

| Verdetto | Condizione | Significato | Azione |
|---|---|---|---|
| **OK** | ratio < 5% | modello lineare confermato | nessuna, procedi |
| **WARN** | 5% <= ratio < 15% | throttling lieve, worker vicino alla saturazione | considera di ridurre `max_replicas` di 1, oppure accetta (Φ_cap corregge a runtime) |
| **FAIL** | ratio >= 15% | throttling significativo, modello sublineare | **riduci `max_replicas_per_cluster`** per quel cluster |

Exit code dello script: `0` = OK/WARN, `1` = FAIL, `2` = errore prerequisiti.

---

## 6. Uso: comandi e variabili env

### Comando base

```bash
cd <REPO_ROOT>
bash experiments/throttle_check.sh --cluster ms-cluster-01
```

### Validare tutti e 3 i tier (dopo un re-provisioning)

```bash
bash experiments/throttle_check.sh --cluster ms-cluster-01   # HIGH
bash experiments/throttle_check.sh --cluster ms-cluster-02   # MEDIUM
bash experiments/throttle_check.sh --cluster ms-cluster-03   # LOW
```

Basta un cluster rappresentativo per tier: gli altri dello stesso tier
hanno HW identico.

### Variabili env / flag

| Flag / env | Default | Descrizione |
|---|---|---|
| `--cluster <nome>` | ms-cluster-01 | cluster da testare |
| `--load-pct <n>` | 80 | % della capacita' da generare come carico |
| `--duration <s>` | 180 | durata del carico in secondi |
| `DMOS_CAP_PER_REPLICA` | 60 | capacita' per replica (da capacity test) |
| `DMOS_THROTTLE_WARN` | 0.05 | soglia ratio per il verdetto WARN |

Esempio: test piu' aggressivo (carico al 95%, 4 min):
```bash
bash experiments/throttle_check.sh --cluster ms-cluster-03 --load-pct 95 --duration 240
```

---

## 7. Output di esempio

```
=== Pre-flight - throttle check su ms-cluster-03 ===
  [OK]   Tool presenti
  [OK]   Cluster target: ms-cluster-03
  [OK]   IP=<CLUSTER_IP>  Prometheus=http://<CLUSTER_IP>:32090

=== Lettura max_replicas per ms-cluster-03 ===
  [OK]   max_replicas=3  capacita=180 rps  carico target (80%)=144 rps

=== Scale frontend -> 3 repliche su ms-cluster-03 ===
  [OK]   frontend Ready con 3 repliche

=== Snapshot Prometheus PRE-carico ===
  CFS throttled periods (cumul): 1240
  CFS total periods (cumul):     58300

=== Genero carico 144 rps per 180s ===
  ... (k6 constant-arrival-rate) ...

=== Snapshot Prometheus POST-carico ===
  Throttle ratio (frontend, 2m): 0.018
  CPU usata da frontend:         0.62 cores
  CPU usata da tutti i pod:      2.74 cores

=== Verdetto - ms-cluster-03 (max_replicas=3) ===
  [OK]   Throttle ratio 1.80% < soglia 5.0%
  [OK]   MODELLO LINEARE CONFERMATO: capacita = 60 x 3 = 180 rps

  Le 3 repliche frontend non sono CPU-throttled a 80% di carico.
  max_replicas_per_cluster=3 e' sotto la soglia fisica del worker. OK.
```

---

## 8. Cosa fare se il verdetto e' FAIL

Se un tier risulta FAIL (throttle ratio >= 15%):

1. **Riduci `max_replicas_per_cluster`** per quel cluster in
   `config/services.yaml`. Esempio, se LOW e' FAIL a max_replicas=3:
   ```yaml
   max_replicas_per_cluster:
     ms-cluster-03: 2     # era 3
     ms-cluster-05: 2     # era 3 (stesso tier)
   ```

2. **Ri-lancia il check** su quel cluster:
   ```bash
   bash experiments/throttle_check.sh --cluster ms-cluster-03
   ```

3. Ripeti finche' il verdetto e' OK.

Cause possibili di un FAIL:
- `max_replicas` semplicemente troppo alto per quel worker
- I backend di Online Boutique consumano piu' CPU del previsto
- Prometheus / kube-prometheus-stack sotto carico spike-a sulla CPU
- Worker con meno CPU del previsto (verifica con `kubectl get nodes`)

Nota: anche con un piccolo WARN residuo, DMOS a runtime **corregge**
comunque tramite Φ_cap (la componente `cpu_available_fraction` letta da
Prometheus in tempo reale rileva la saturazione e ferma lo scaling). Il
`throttle_check` serve a calibrare bene il modello statico, ma la
componente dinamica e' una rete di sicurezza.

---

## 9. Throttle check vs Capacity test

Sono due strumenti diversi, non confonderli:

| | Capacity test (`run_capacity.sh`) | Throttle check (`throttle_check.sh`) |
|---|---|---|
| **Domanda** | quanto regge **1 replica**? | le **N repliche** vanno in throttling? |
| **Scala il frontend a** | 1 replica (isolata) | `max_replicas` |
| **Verdetto basato su** | latency p95 **lato client** (k6) | CFS throttle ratio **lato server** (Prometheus) |
| **Output** | `capacity_req_per_sec` per services.yaml | verdetto OK/WARN/FAIL |
| **Durata** | ~17 min | ~4 min |
| **netem** | va fatto **SENZA** netem (vedi sotto) | **indifferente**, si puo' lasciare ON |
| **Quando** | una volta (capacita' del pod) | dopo ogni re-provisioning |
| **Va rifatto se** | cambia il CPU limit del pod / workload | cambia il HW dei worker o `max_replicas` |

Il capacity test stabilisce **il numero `60`**. Il throttle check verifica
che **moltiplicarlo per `max_replicas`** dia un risultato lineare reale.

### 9.1 netem: perche' il throttle check NON ne risente

Il `throttle_check` da' il verdetto in base al **CFS throttle ratio**, una
metrica CPU lato server (`container_cpu_cfs_throttled_periods_total`,
scrapata da Prometheus). Il tc netem agisce sull'**interfaccia di rete**
(egress di `eth0`): ritarda i pacchetti sul filo, ma **non fa lavorare di
piu' la CPU**. Il lavoro di "processare N richieste/s" e' identico con o
senza netem.

Quindi il throttle check si puo' eseguire con il netem attivo. Anzi: dato
che il flash crowd test gira CON il netem, fare il throttle check nello
stesso stato e' **piu' rappresentativo** della condizione reale.

Il **capacity test** invece da' il verdetto in base alla **latency p95
lato client**: il netem aggiunge millisecondi a ogni risposta -> il p95
si gonfia -> il knee-point misurato e' falso. Per questo il capacity test
va fatto **senza netem** (`bash scripts/setup-netem-and-ping.sh remove-netem`
prima, poi `netem-only` dopo).

Regola mnemonica:
- verdetto da **latency client** -> il netem falsa -> **netem OFF**
- verdetto da **metrica CPU server** -> il netem non tocca -> **netem ON ok**

---

## 10. Quando rieseguirlo

Esegui `throttle_check.sh` (3 tier):

- **Dopo ogni re-provisioning** del fleet (FASE 6 del `RUNBOOK.md`)
- **Dopo aver cambiato `max_replicas_per_cluster`** in `services.yaml`
- **Dopo aver cambiato le specifiche HW** dei worker in `ms-cluster.yaml`
- **NON** serve rieseguirlo prima di ogni flash crowd test: una volta
  che i `max_replicas` sono validati, restano validi finche' l'HW non cambia

---

*Documento creato 2026-05-20. Aggiornare se cambia la logica di
`throttle_check.sh` o il modello di capacita' di DMOS.*
