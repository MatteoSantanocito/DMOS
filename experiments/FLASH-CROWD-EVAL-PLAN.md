# Flash-Crowd Evaluation Plan — DMOS 6-Cluster

> **Documento storico (piano pre-campagna).** È il disegno sperimentale
> scritto prima dei run a 6 cluster. Da allora sono cambiati: gli arm (gli
> arm `dmos-l1` e `round-robin` qui descritti non sono implementati — il
> profilo `round_robin` non esiste in `weights.yaml`; la campagna ha usato
> OFF / L2 / L1L2 e la configurazione del paper è il modello di score `nca`),
> i regimi (sei livelli di concentrazione invece del singolo spike) e
> l'harness (run tracciabili con `run_one.sh`). Riferimenti attuali:
> [RUNBOOK.md](RUNBOOK.md), [../docs/METRICS-AND-OUTPUTS.md](../docs/METRICS-AND-OUTPUTS.md),
> [../docs/publications/2026-nca-dmos.md](../docs/publications/2026-nca-dmos.md).

> **Scope**: design dell'esperimento "Flash Crowd" per la valutazione di DMOS
> nella tesi MSc / paper IEEE in scrittura. Definisce ipotesi, metodologia,
> metriche, e procedura riproducibile. Usato come "experimental design"
> appendix nel paper.
>
> **Owner**: Matteo Santanocito (UniCT)
> **Stato**: tooling implementato e validato via smoke test (orchestratore
> dual-mode `run_flash_crowd.sh` + collector `collect_dmos_live.py`);
> in attesa dei run reali a 6 cluster.
> **Repo**: `github.com/MatteoSantanocito/dmos`

---

## 1. Obiettivo dell'esperimento

Misurare quanto DMOS migliora le prestazioni di un'applicazione cloud-native
multi-cluster (Online Boutique) durante un **flash crowd geograficamente
concentrato** — uno scenario realistico in cui un evento improvviso (es. tutti
gli utenti in DE accedono allo stesso negozio simultaneamente per un'offerta
flash) sovraccarica una sola regione mentre le altre rimangono sotto-utilizzate.

L'esperimento confronta DMOS con strategie di baseline e produce metriche
abbastanza fine-grained da supportare un paper IEEE: latency p95/p99, error
rate, replica trajectory per cluster, scheduler decision time.

---

## 2. Ipotesi sperimentali

Tre ipotesi, in ordine di importanza per il paper:

**H1 (latency)**: durante la fase di **spike sustained peak** (300-600s),
DMOS-on mantiene il p95 di latency end-to-end **<1000ms**, mentre la baseline
DMOS-off (HPA disabilitato, repliche fisse 1/1/1/1/1/1) sale oltre **3000ms**
con punti >9000ms.

**H2 (error rate)**: durante peak, DMOS-on mantiene error rate (5xx +
connection refused) **<2%**, mentre DMOS-off ha >10% con punte >30% sul
cluster sotto stress diretto (`SPIKE_REGION`).

**H3 (geo-fairness)**: DMOS-on distribuisce le repliche su tutti i 6 cluster
in funzione della **prossimita' geografica** (favorisce cluster vicini con
bassa latenza inter-cluster), non solo della capacita' residua. La varianza
di error rate tra cluster vicini ($d < 100ms$) e' **<0.02** (sostanzialmente
fair), mentre senza la componente $\Phi_{locality}$ alcuni cluster lontani
(`ms-cluster-05 SG`, +200ms netem) verrebbero scelti se hanno piu' capacita',
causando picchi di latency sui consumer.

Tre ipotesi secondarie:

**H4 (predittore L2)**: la versione DMOS-L1L2 (con predittore di traffico)
inizia a scalare **>=20s prima** della versione DMOS-L1-only durante la
transizione baseline→spike (fase 2, 180-300s), riducendo il p95 picco
del 15-30%.

**H5 (carbon)**: per workload con $\omega_{carbon}=0.20$, DMOS-on durante
le ore "verdi" (es. notte EU, sole AU) sceglie cluster con
$\Phi_{carbon}$ alto (low CO2), riducendo la carbon intensity totale del 10-20%
rispetto a una distribuzione uniforme. *Hp da validare su dataset MISO-only,
fuori scope per la prima evaluation*.

**H6 (recovery)**: dopo decay (720s), DMOS-on torna a configurazione baseline
(repliche minime, no over-provisioning residuo) entro 120s. DMOS-off lascia
le repliche al peak per sempre (no HPA = no scaling).

---

## 3. Setup sperimentale

### 3.1 Infrastruttura

- **6 cluster K3s** (`ms-cluster-01..06`) su 3 host Proxmox (`<PVE_HOST_1/2/3>`)
- **Network**: tutti sulla stessa LAN UniCT (`<LAN_SUBNET>`), con `tc netem`
  che simula latenze geografiche realistiche:

  | Cluster | Regione simulata | Delay netem | Jitter |
  |---|---|---|---|
  | ms-cluster-01 | EU-DE (Frankfurt) | 0ms (baseline) | 0ms |
  | ms-cluster-02 | EU-FR (Paris) | 30ms | 5ms |
  | ms-cluster-03 | EU-PL (Warsaw) | 50ms | 10ms |
  | ms-cluster-04 | US-VA (Virginia) | 100ms | 15ms |
  | ms-cluster-05 | APAC-SG (Singapore) | 200ms | 25ms |
  | ms-cluster-06 | EU-NO (Oslo) | 40ms | 8ms |

- **CNI**: Cilium 1.19.4 con ClusterMesh attivo (15 connessioni pairwise per N=6)
- **Workload**: Online Boutique (Google microservices-demo) deployato in
  namespace `online-boutique` su ciascun cluster, 11 microservizi
- **Service di test**: `frontend` (HTTP), esposto via Cilium Ingress
  NodePort 30080 di ciascun worker
- **Capacita' per replica**: i 4 backend gRPC principali sono stati misurati
  direttamente con k6 in-cluster (`experiments/k6_capacity_grpc.js` +
  `experiments/run_capacity_grpc.sh`, metodologia knee-point) — vedi
  `experiments/CAPACITY-GRPC.md`. Valori in `config/services.yaml`
  (`capacity_req_per_sec`): cartservice 400, productcatalogservice 1200,
  recommendationservice 40 (knee misurato 140, derated: concurrency-bound
  sotto carico reale, vedi commento in `services.yaml`), checkoutservice 160.

### 3.1.1 Cilium routing configuration (importante per il paper)

I 5 Service global (frontend + 4 backend principali: cartservice,
productcatalogservice, recommendationservice, checkoutservice) sono annotati con:

```yaml
annotations:
  io.cilium/global-service: "true"           # visibilita' cross-cluster
  service.cilium.io/affinity: "local"        # prefer local, failover remote
```

Cio' significa che ogni cluster gestisce le chiamate gRPC inter-service
**intra-cluster**, con failover automatico cross-cluster solo se 0 pod
locali sono healthy (semantica documentata su
[docs.cilium.io](https://docs.cilium.io/en/v1.19/network/clustermesh/affinity/)).

**Motivazione** (vedi `dmos-iac/DMOS-DEPLOYMENT-PIPELINE.md` sezione 10.14):
- In setup N=6 con netem variabile (30-200ms), il default Cilium
  `affinity: none` causa ~15-30% errori HTTP intermittenti dovuti a
  gRPC dial timeout cross-cluster
- `affinity: local` separa cleanly i concerns architettonici:
  scheduler decide repliche, network plane fa routing locale
- DMOS resta valutabile: la differenza dmos-on vs dmos-off emerge
  attraverso il replica allocation pattern (Phi_locality guida lo scaling)

**Differenza con la tesi originale a 3 cluster**: la tesi usava il
default `none` perche' a 3 cluster il bug del cross-cluster gRPC era
latente (solo 2/3 chance di hit remoto, vs 5/6 con N=6, e solo 2 valori
netem vs 5). Il setup N=6 richiede l'esplicita scelta `local` per
stabilita'. Il framework DMOS rimane invariato.

### 3.1.2 Topologia del carico — generatore k6 isolato (importante per la validita')

Il carico di prova **non** viene generato dal laptop che controlla
l'esperimento. Per evitare che il generatore contamini la misura
(CPU/rete del laptop come confounding factor — vedi sezione 7.1), il
load generator k6 gira su una **VM Proxmox dedicata**:

| Componente | Dove gira | Ruolo |
|---|---|---|
| `k6` (load generator) | VM `ms-k6-loadgen` (4 vCPU / 8 GB, IP statico `<K6_VM_IP>`) | genera il traffico flash-crowd verso i 6 ingress |
| DMOS (`src.dmos_main`) | laptop, in locale | scheduler L1+L2, espone `/metrics` su `:9090` |
| `run_flash_crowd.sh` | laptop | orchestratore: pre-flight, manifest, lancia k6 (via SSH) e il collector |
| `collect_dmos_live.py` | laptop, in background | scrapa DMOS `:9090` + Prometheus di ogni cluster, scrive `dmos_live.jsonl` |

L'orchestratore e' **dual-mode** (vedi sezione 4): con `DMOS_K6_REMOTE`
vuota k6 gira sul laptop (solo per smoke test / debug), con
`DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>` k6 gira sulla VM via SSH. **Tutti
i run validi per il paper usano la modalita' remota.**

### 3.2 Variabili indipendenti (controllate)

| Variabile | Valori | Note |
|---|---|---|
| Strategia di scheduling | `dmos-off`, `dmos-l1`, `dmos-l1l2`, `round-robin` | 4 run a confronto |
| Spike region | `ms-cluster-01` (EU-DE) default | Cluster che riceve 60% del traffico durante spike |
| Peak rps | 250 (default), 400, 600 | Tre intensita' per verificare scalabilita' |
| Workload mix | Online Boutique production-like | 40% GET /, 10% /cart GET, 15% /cart POST, 5% /checkout, ecc. |
| Run duration | 900s = 15 min | Fisso |

### 3.3 Variabili dipendenti (misurate)

Le metriche vengono raccolte da due collettori distinti:
- **`k6` (handleSummary)** → `flash_crowd_summary.json` — metriche client-side,
  aggregate per fase e per cluster, prodotte a fine run.
- **`collect_dmos_live.py`** → `dmos_live.jsonl` — uno snapshot ogni 5s di
  DMOS `/metrics` + Prometheus di ogni cluster, per ricostruire la timeline.

**Application-level** (k6 client-side, in `flash_crowd_summary.json`):
- Latency p50, p95, p99 — per fase e per cluster
- HTTP failure rate (5xx + connection refused, status==0) — globale e per cluster
- Throughput (rps achieved vs target) e `dropped_iterations` (gate di validita')
- Verdetto `validity.run_valid` (true se k6 ha sostenuto il rate offerto)

**System-level** (Prometheus per ogni cluster, in `dmos_live.jsonl`):
- `ready_pods` per servizio — replica trajectory (cartservice, frontend, ecc.)
- `cpu_cores` / `memory_mb` per servizio — `rate(container_cpu_usage_seconds_total)`
  e `container_memory_working_set_bytes` su `container="server"`
- `hubble_http_rps` — RPS L7 visto dal CNI Cilium
- `frontend_net_bytes_s` + `traffic_pct` — distribuzione del traffico per cluster
- `node_cpu_pct` — carico CPU a livello di nodo
- `restarts` — sanity check (nessun OOM / crash loop durante il run)

**Scheduler-level** (DMOS Prometheus exporter su `:9090`, in `dmos_live.jsonl`):
- `dmos_cluster_score{cluster,service}` — score combinato $S_i$ per cluster
- `dmos_current_replicas` / `dmos_target_replicas` — stato e decisione del scheduler
- `dmos_scaling_events_total{action}` — counter cumulativo scale up/down
- `dmos_scheduling_duration_seconds_{sum,count}` — tempo per ciclo (decision Δt)
- `dmos_actual_traffic` / `dmos_predicted_traffic` — traffico osservato vs
  predetto dal L2 (`predicted_traffic` valorizzato solo per `dmos-l1l2`)

> **Nota** — DMOS espone su `/metrics` solo lo **score combinato**
> `dmos_cluster_score`, non i singoli componenti $\Phi_{lat}/\Phi_{cap}/
> \Phi_{locality}/\Phi_{carbon}/\Phi_{pressure}$. Il breakdown per-$\Phi$ e' comunque
> tracciato: DMOS lo scrive a ogni ciclo nel log testuale
> `logs/DMOSMain.log` (riga `Score ms-cluster-NN: 0.XXX (lat=..., cap=...,
> locality=..., carbon=..., pressure=...)`). Per l'analisi qualitativa delle
> decisioni si usa quel log; per i grafici di timeline basta lo score
> combinato in `dmos_live.jsonl`.

### 3.4 Variabili confounding (controllate ma non manipolate)

- VPN UniCT attiva e stabile durante tutto il run
- Nessun altro carico di rete sui worker (verifica `top` su ciascuno)
- Tempo del giorno: tutti i run in finestra ridotta (~1h) per evitare
  drift di temperatura/throttling
- Carico baseline degli altri team su Proxmox: idealmente notte, verifica
  con `pveperf` prima

---

## 4. Procedura

> L'orchestratore `run_flash_crowd.sh` esegue **un singolo arm**. La
> sequenza dei 4 arm e la gestione dello stato di DMOS sono manuali (sotto).

### 4.1 Pre-run (una volta, prima della sessione di run)

```bash
# 1. Pipeline base — i 6 cluster up, netem e ClusterMesh ok
cd <REPO_ROOT>
.\scripts\setup-dmos-on-clusters.ps1   # Phase 3+4 (idempotente)
bash scripts/setup-netem-and-ping.sh   # Phase 5+6 (idempotente)
python scripts/dmos-ready-check.py     # Phase 7 verifica — atteso 61/61 OK

# 2. File runtime per collector e k6 (.cache/test-env.json)
bash scripts/render-runtime-files.sh

# 3. Capacita' dei backend — GIA' FATTO (vedi experiments/CAPACITY-GRPC.md).
#    config/services.yaml e' gia' aggiornato: cartservice 400,
#    productcatalogservice 1200, recommendationservice 40, checkoutservice 160.
#    Da rifare solo se l'hardware dei cluster cambia:
#      bash experiments/run_capacity_grpc.sh --all

# 4. VM del load generator accesa e raggiungibile
ssh -i ~/.ssh/<key> ubuntu@<K6_VM_IP> 'k6 version'
```

### 4.2 Run sequenziale dei 4 arm

`run_flash_crowd.sh` fa da solo: pre-flight (tool, DMOS su `:9090`,
6 ingress → 200), manifest, lancio del collector in background, lancio di
k6 (sulla VM via SSH in modo remoto), attesa, archiviazione in
`results/<ts>_<label>/`. **Lo stato di DMOS lo gestisci tu fuori dallo
script.** Imposta una volta la modalita' remota:

```bash
export DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>    # k6 gira sulla VM dedicata
```

Ordine consigliato (dal piu' semplice da debuggare):

```bash
# ── Arm A: dmos-off (baseline: nessuno scaling, repliche fisse 1) ──
# DMOS NON in esecuzione. Fissa le repliche al minimo su tutti i cluster:
for n in 01 02 03 04 05 06; do
  kubectl --kubeconfig "${KUBECONFIG_DIR:-$HOME/dmos-iac}"/kubeconfig-ms-cluster-$n \
    -n online-boutique scale deployment --all --replicas=1
done
bash experiments/run_flash_crowd.sh --label dmos-off --no-dmos

# ── Arm B: dmos-l1 (DMOS reattivo, predittore L2 disattivato) ──
# Il predittore si neutralizza via env con orizzonte 0: la predizione
# coincide col traffico corrente -> comportamento puramente reattivo.
DMOS_PREDICTOR_HORIZON_S=0 python -m src.dmos_main &   ; DMOS_PID=$!
sleep 130                        # attendi il cold-start (120s)
bash experiments/run_flash_crowd.sh --label dmos-l1
kill $DMOS_PID

# ── Arm C: dmos-l1l2 (DMOS completo, predittore L2 attivo) ──
# Predittore attivo coi default (finestra W=60s, orizzonte H=120s).
python -m src.dmos_main &        ; DMOS_PID=$!
sleep 130
bash experiments/run_flash_crowd.sh --label dmos-l1l2
kill $DMOS_PID

# ── Arm D: round-robin (multi-cluster naive) ──
# In config/weights.yaml: profilo round_robin
python -m src.dmos_main &        ; DMOS_PID=$!
sleep 130
bash experiments/run_flash_crowd.sh --label round-robin
kill $DMOS_PID
```

Note operative:
- Tra un arm e l'altro: **2 min di cooldown** (pod che tornano al baseline).
- Lo script aborta nel pre-flight se DMOS non risponde su `:9090` e manca
  `--no-dmos`: usa `--no-dmos` **solo** per l'arm dmos-off.
- Override utili: `--peak N` (default 250), `DMOS_PHASE_TIMES`,
  `DMOS_VU_FACTOR`. Per uno smoke veloce in locale: lascia `DMOS_K6_REMOTE`
  vuota e usa fasi corte, es. `DMOS_PHASE_TIMES=20,40,80,100,120`.
- Per il paper servono **3-5 ripetizioni per arm** → ripeti il blocco
  cambiando solo `--label` (es. `dmos-l1l2-rep2`).

### 4.3 Artefatti — per ogni run in `experiments/results/<ts>_<label>/`

| File | Descrizione |
|---|---|
| `manifest.json` | config del run, git commit, dirty flag, timestamp, modo k6 |
| `flash_crowd_summary.json` | metriche k6 per-fase e per-cluster + verdetto di validita' |
| `dmos_live.jsonl` | timeline scheduler+cluster, uno snapshot JSON ogni 5s |
| `k6_stdout.txt` | output console completo del run k6 |
| `k6_events.json` | eventi k6 grezzi (se `DMOS_RAW_OUT=1`, default) |
| `collector_stdout.txt` | log del collector (diagnostica) |

---

## 5. Analisi e metriche derivate

### 5.1 Tabella riassuntiva (paper Table 1)

Per ciascuno dei 4 scenari, durante la fase **spike + peak** (180-600s):

| Strategia | p95 (ms) | p99 (ms) | Error % | Avg Replicas | Decision Δt (ms) |
|---|---|---|---|---|---|
| dmos-off    | ? | ? | ? | 6 (fissi) | n/a |
| dmos-l1     | ? | ? | ? | ? | ? |
| dmos-l1l2   | ? | ? | ? | ? | ? |
| round-robin | ? | ? | ? | ? | ? |

### 5.2 Plot trajectory per il paper

La timeline viene da `dmos_live.jsonl` (uno snapshot/5s), la latency/errori
per-fase da `flash_crowd_summary.json`. Figure da generare:
- **Fig. 1**: Replica trajectory per cluster (6 linee, Y=replicas, X=tempo).
  Mostra come DMOS sposta capacita' geograficamente durante lo spike.
- **Fig. 2**: P95 latency per fase, per le 4 strategie a confronto (bar/linea).
- **Fig. 3**: HTTP failure rate per fase, per le 4 strategie.
- **Fig. 4**: Score $S_i$ + traffic distribution per cluster vs tempo
  (mostra che lo score guida l'allocazione).

### 5.3 Script di plot suggerito

`dmos_live.jsonl` e' una riga JSON per snapshot; ogni riga ha
`elapsed_s`, `services[svc].clusters[cl].{current_replicas,target_replicas,
score,ready_pods}` e `clusters[cl].{node_cpu_pct,traffic_pct,hubble_http_rps}`.

```python
# esempio minimale — la pipeline di analisi completa e'
# experiments/paper_generate_all.py (+ experiments/paper_lib.py)
import json, matplotlib.pyplot as plt
from collections import defaultdict

snaps = []
with open('experiments/results/<ts>_dmos-l1l2/dmos_live.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            snaps.append(json.loads(line))

# Replica trajectory del frontend, una linea per cluster
t = [s['elapsed_s'] for s in snaps]
series = defaultdict(list)
for s in snaps:
    for cl, cd in s['services']['frontend']['clusters'].items():
        series[cl].append(cd['current_replicas'])

fig, ax = plt.subplots(figsize=(10, 5))
for cl, ys in sorted(series.items()):
    ax.plot(t, ys, marker='.', label=cl)
ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Repliche frontend')
ax.legend()
ax.set_title('Frontend replicas per cluster — dmos-l1l2')
plt.savefig('fig1_replicas_trajectory_dmos-l1l2.pdf', bbox_inches='tight')
```

Per Fig. 2/3 si legge `flash_crowd_summary.json` → `results.phases[*]` e
`results.per_cluster[*]`.

### 5.4 Test statistici

Per validare H1, H2, H3 oltre la sola eyeball comparison:
- **Welch's t-test** sulle distribuzioni di p95 latency tra dmos-on e dmos-off
  (campioni indipendenti, no assunzione varianze uguali)
- **CIs al 95%** sui valori medi per replica trajectory tra strategie
- **Effect size (Cohen's d)** per quantificare quanto e' grande la differenza,
  non solo se e' significativa

---

## 6. Riproducibilita'

Per essere paper-ready, ogni run deve essere:

1. **Versionato**: `manifest.json` contiene `git_commit`. Se ci sono file
   dirty (`git_dirty_files > 0`), il run e' marcato come "preliminary".
2. **Configurabile**: tutti i parametri via env vars, nessun valore hardcoded.
3. **Idempotente**: lanciando 2 volte lo stesso comando con la stessa
   infrastruttura, i risultati devono essere coerenti (variance < 10% su p95).
4. **Snapshot completo**: `dmos_live.jsonl` (timeline DMOS+cluster ogni 5s) +
   `flash_crowd_summary.json` permettono replay analitico anche senza
   accesso ai cluster.

Per la pubblicazione, i risultati raw (`k6_events.json`) + i `dmos_live.jsonl`
vanno archiviati in un repo Zenodo con DOI, citato nel paper.

---

## 7. Edge cases e potenziali threats to validity

### 7.1 Threats interni

- **Cluster cohabitation**: i 6 K3s girano su 3 host Proxmox (2 per host).
  Se ms-cluster-01 e ms-cluster-02 sono entrambi sullo stesso host e si stressano
  contemporaneamente, c'e' contesa CPU/IO sull'host. **Mitigazione**: scegli
  `SPIKE_REGION` su un cluster che ha il "neighbor" su un host diverso, ed
  enforcing taints/tolerations se necessario.
- **Cilium ClusterMesh latency**: la mesh ha overhead variabile sotto carico.
  Una run con mesh degradata potrebbe falsamente attribuire latency a DMOS.
  **Mitigazione**: il pre-flight di `run_flash_crowd.sh` verifica che i 6
  ingress rispondano 200; eseguire `python scripts/dmos-ready-check.py`
  (61/61 OK) prima della sessione di run scarta gli ambienti con mesh degradata.
- **k6 client come bottleneck**: con 400+ rps e 6 destinazioni, un generatore
  co-locato col controller falserebbe la misura. **Mitigato by design**: k6
  gira su una VM Proxmox dedicata (`ms-k6-loadgen`, 4 vCPU/8 GB — sezione
  3.1.2), separata dal laptop che esegue DMOS e il collector. Lo script k6
  espone inoltre il gate `validity.run_valid` (`dropped_iterations`): un run
  VU-starved viene marcato NON VALIDO e va rilanciato con `DMOS_VU_FACTOR`
  piu' alto, invece di entrare silenziosamente nei risultati.

### 7.2 Threats esterni (generalizability)

- **Online Boutique e' un workload sintetico**: piccolo, gRPC-heavy, no DB
  reali. Risultati su workload diversi (Spark batch, ML inference) potrebbero
  differire. **Mitigazione**: nel paper, sottolineare che e' un microservices
  HTTP-centric, valido per la classe di applicazioni e-commerce ma non
  generalizzabile a HPC/batch.
- **Netem e' una simulazione**: la latency e' costante a livello di link
  (Pareto distribution opzionale), mentre la latency Internet reale ha
  jitter, packet loss, BGP convergence. **Mitigazione**: opzionalmente
  aggiungere `loss 1%` al netem per scenario realistico WAN.
- **Single tenant**: niente noisy neighbor su Proxmox, mentre in produzione
  i cluster K8s di solito condividono nodes con altre workload. **Mitigazione**:
  scenario opzionale "stress neighbor" via stress-ng su altri VM.

---

## 8. Roadmap di esperimenti correlati

Una volta validato Flash Crowd, il paper puo' includere anche:

1. **Gradual ramp** (rampa lineare 50 -> 500 rps in 10 min): valuta convergenza
   monotona dello scheduler. Gia' parzialmente esplorato nei run preliminari
   sul testbed precedente a 3 cluster.
2. **Double wave** (due picchi a distanza di 5 min): valuta stabilita' del
   predittore L2 — non deve oscillare.
3. **Sinusoidal load** (ampiezza variabile a 60s di periodo): test che il
   predittore non insegua il rumore.
4. **Disaster recovery** (kill di un cluster a meta' run): DMOS deve
   ridistribuire entro 30s.
5. **Carbon-aware scheduling**: replay con dataset MISO real per
   $\Phi_{carbon}$, verifica trade-off latency vs CO2.

---

## 9. Risultati attesi (qualitative + ranges)

Basandomi sui run preliminari del testbed precedente a 3 cluster:

| Metrica @ peak | dmos-off | dmos-l1 | dmos-l1l2 | round-robin |
|---|---|---|---|---|
| p95 latency | 3000-9000ms | 800-1500ms | **400-900ms** | 1500-3000ms |
| p99 latency | >15000ms | 2000-3000ms | **800-1500ms** | 3000-5000ms |
| Error rate | 15-30% | 3-7% | **<2%** | 5-10% |
| Avg replicas (system-wide) | 6 (fissi) | 18-24 | **20-26** | 20-26 |
| Time to first scaling | n/a | 30-60s | **15-30s** | 30-60s |

Numeri preliminari, da confermare con i nuovi run su 6 cluster. La differenza
qualitativa attesa e' che **dmos-l1l2** sia significativamente migliore di
dmos-l1 (-15-30% p95) grazie al predittore, e che round-robin sia peggio di
dmos-l1 perche' non considera Φ_cap.

---

## 10. Checklist pre-paper-submission

- [ ] 4 arm × 3-5 ripetizioni = 12-20 run totali, ognuno `validity.run_valid=true`
- [x] Capacity test dei 4 backend gRPC (vedi `experiments/CAPACITY-GRPC.md`,
      `config/services.yaml` aggiornato)
- [ ] `.cache/test-env.json` rigenerato una volta a inizio sessione, poi
      identico per tutti i run
- [ ] Plot di tutte e 4 le figures (PDF + sorgente Python in repo)
- [ ] Tabella 1 popolata con CI@95% (ripetizioni → mean ± std)
- [ ] Welch's t-test pvalues per H1, H2, H3 — riportati nel paper
- [ ] Manifest di tutte le run committate (no `dirty_files > 0`)
- [ ] Repo Zenodo con DOI per i raw data (`k6_events.json` + `dmos_live.jsonl`)
- [ ] Threats to validity section (sezione 7 → paper section)

---

*Aggiorna questo documento dopo ogni run di calibrazione. La sezione 9
(Risultati attesi) va riempita con numeri reali appena disponibili.*
