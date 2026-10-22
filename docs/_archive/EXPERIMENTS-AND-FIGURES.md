# [ARCHIVED] DMOS — Documentazione esperimenti e figure (v1)

> **Documento archiviato (ottobre 2026).** Descrive la campagna "v1" e ne
> commenta i risultati, precedenti alla configurazione finale del paper NCA
> 2026. Riferimenti attuali:
>
> - [`../METRICS-AND-OUTPUTS.md`](../METRICS-AND-OUTPUTS.md) — cosa produce un
>   run, definizioni delle metriche, ogni figura e tabella con lo script che la
>   genera;
> - [`../../experiments/RUNBOOK.md`](../../experiments/RUNBOOK.md) — come si
>   eseguono i run;
> - [`../publications/2026-nca-dmos.md`](../publications/2026-nca-dmos.md) —
>   configurazione e risultati pubblicati.
>
> Non più validi: la formula di score dell'arm `l1l2` descritta qui (la
> configurazione del paper è il modello di score `nca`), l'affermazione
> "nessun routing cross-cluster" (i Service sono global con `affinity: local`,
> vedi [`../CILIUM-CLUSTERMESH-NOTES.md`](../CILIUM-CLUSTERMESH-NOTES.md)) e i
> numeri delle sezioni 7, 9 e 10.

Questo documento descrive **come vengono eseguiti i test** di valutazione di DMOS,
**su quale infrastruttura**, e **cosa mostra ogni grafico** prodotto dall'analisi.
È pensato per capire il setup sperimentale e leggere le figure del paper (v1).

---

## 1. Infrastruttura (banco di prova)

La valutazione gira su un **fleet geo-distribuito simulato di 6 cluster Kubernetes**,
eterogeneo per hardware e per latenza di rete.

### 1.1 I 6 cluster

Ogni cluster è un cluster **K3s v1.36** (1 control-plane + 1 worker) su VM **Proxmox**,
con **Cilium** come CNI. Le VM sono sulla stessa LAN privata (gli IP reali vivono in
`config/local.yaml` / `.cache/test-env.json`, gitignored).

| Cluster | Regione | Ingress | Tier HW (worker) | Latenza rete (netem) |
|---|---|---|---|---|
| ms-cluster-01 | EU-DE (Frankfurt) | `<CLUSTER1_IP>:30080` | **HIGH** (8 vCPU) | 0 ms |
| ms-cluster-02 | EU-FR (Paris)     | `<CLUSTER2_IP>:30080` | MEDIUM (6 vCPU) | 30 ms |
| ms-cluster-03 | EU-PL (Warsaw)    | `<CLUSTER3_IP>:30080` | LOW (4 vCPU) | 50 ms |
| ms-cluster-04 | US-VA (Virginia)  | `<CLUSTER4_IP>:30080` | **HIGH** (8 vCPU) | 100 ms |
| ms-cluster-05 | APAC-SG (Singapore)| `<CLUSTER5_IP>:30080` | LOW (4 vCPU) | 200 ms |
| ms-cluster-06 | EU-NO (Oslo)      | `<CLUSTER6_IP>:30080` | MEDIUM (6 vCPU) | 40 ms |

- **Eterogeneità HW**: 3 tier (HIGH/MEDIUM/LOW) → ogni servizio ha `max_replicas_per_cluster`
  diverso per tier (es. frontend HIGH=6, MEDIUM=5, LOW=3 — cap colocation-aware da
  `services.yaml`: su HIGH il tetto è 8→6 per lasciare headroom al co-scaling dei backend).
  DMOS deve tenerne conto.
- **Latenza di rete**: simulata con `tc netem` su `eth0` (one-way delay). Modella la
  distanza geografica. **Si azzera al reboot della VM** → va ri-applicata col preflight.
- **Ingress**: ogni cluster espone l'app via **Cilium Ingress** (proxy Envoy) su
  **NodePort 30080**. Il client (k6) colpisce *direttamente* l'ingress di ogni cluster
  (nessun routing cross-cluster: ogni cluster serve il proprio traffico in locale).

### 1.2 Le macchine di controllo e carico

| Ruolo | Macchina | Cosa fa |
|---|---|---|
| **Control plane** | Laptop (Windows, `venv` Python) | Esegue **DMOS** (`src.dmos_main`, espone `/metrics` su `localhost:9090`), gli orchestratori di test (bash), e il **collector**. Ha i kubeconfig dei 6 cluster. |
| **Generatore di carico** | VM dedicata `ubuntu@<K6_VM_IP>` | Esegue **k6** (via SSH, detached). Separata dal control-plane per non falsare le misure. |
| **Osservabilità** | su ogni cluster | **Prometheus** (NodePort 32090) + **Hubble** (metriche L7 HTTP) + **ping_exporter** (RTT inter-cluster). |

### 1.3 Osservabilità usata da DMOS

- **Hubble L7** (`hubble_http_requests_total`, `hubble_http_request_duration_seconds`):
  rps e latenza per-servizio/per-edge. DMOS legge l'rps in ingresso al frontend
  (`reporter="server", traffic_direction="ingress", source_workload=""` = solo traffico
  esterno, esclude il pod-to-pod interno).
- **Prometheus per-cluster**: CPU/mem dei pod, repliche.
- **ping_exporter**: RTT verso i peer → proxy della distanza geografica (Φ_net).

---

## 2. DMOS e i 3 "arm" confrontati

DMOS è un orchestratore multi-cluster a due livelli:
- **L1 (placement)**: scheduler geo-aware multi-obiettivo. Decide **quante repliche per
  cluster** con uno score `S_i` (latenza Φ_lat, capacità Φ_cap, località/geo Φ_loc·Φ_net,
  carbonio Φ_car, predittivo Φ_pre) + un **demand-floor** (ogni cluster dimensionato sulla
  propria domanda locale).
- **L2 (scaling/feedback)**: controllo **Proporzionale-Derivativo** + un **predictor**
  trend-based (`Λ_pred = Λ + dΛ/dt·horizon`) per anticipare, e **co-scaling
  dependency-aware** dei backend (fanout-aware).

I test confrontano **3 configurazioni (arm)**:

| Arm | Cosa fa | Colore figure |
|---|---|---|
| **DMOS OFF** | Nessun DMOS. Repliche **fisse a 1** per cluster (baseline statico). | rosso |
| **DMOS (L2)** | Solo scaling+feedback (`--no-scheduling`): scala ma **piazza in modo uniforme** (N/K), senza geo-awareness. | arancio |
| **DMOS (L1+L2)** | Sistema **completo**: placement geo-aware + scaling + co-scaling. | blu |

L'obiettivo è isolare il contributo del **placement L1**: a parità di scaling (L2), cosa
aggiunge il piazzamento demand-aware?

---

## 3. Il workload e lo scenario di test

### 3.1 Applicazione

**Online Boutique** (microservizi Google, gRPC). Il `frontend` (HTTP) è il punto di
ingresso utente; chiama a cascata i backend (cart, productcatalog, currency,
recommendation, checkout, ad, shipping...). Una pagina prodotto genera ~7 hop interni.
`redis-cart` è il datastore di cart (stateful, escluso dallo scaling DMOS).

### 3.2 Test "Flash Crowd"

Un singolo run dura **15 minuti** (900s) e simula un picco di traffico improvviso,
diviso in **5 fasi** (env `DMOS_PHASE_TIMES=180,300,600,720,900`):

| Fase | Tempo | Descrizione |
|---|---|---|
| **baseline** | 0–180s | Carico basso (~50 rps), regime di riposo. |
| **spike** | 180–300s | **Rampa**: il traffico sale rapidamente verso il picco. |
| **peak** | 300–600s | **Picco sostenuto** (350 rps nella campagna, via `DMOS_PEAK_RPS`; il default dello script è 400). La fase che conta. |
| **decay** | 600–720s | Rientro graduale. |
| **cooldown** | 720–900s | Ritorno a basso carico. |

### 3.3 Regimi di domanda (demand skew)

Il parametro chiave è la **concentrazione** del picco su un cluster ("hottest-cluster
share"). I pesi per-cluster (`DMOS_SPIKE_W1..6`) la controllano; W1 = cluster caldo (EU-DE).

| Regime | Pesi (W1/altri) | Hottest share | Significato |
|---|---|---|---|
| `uni` | 1/1/1/1/1/1 | 0.167 | Uniforme su 6 cluster (nessuno stress localizzato). |
| `bal` | 40/15/12/12/11/10 | 0.40 | Realistico EU-heavy. |
| `s50` | 50/10×5 | 0.50 | *(sweep)* |
| `s65` | 65/7×5 | 0.65 | *(sweep)* |
| `s75` | 75/5×5 | 0.75 | *(sweep)* |
| `con` | 85/3/3/3/3/3 | 0.85 | **Flash crowd estremo** su un cluster. Lo scenario di stress principale. |

Il **peak rps è fisso a 350** in tutti i regimi → varia *solo* la concentrazione, così
l'asse "skew" è pulito. (Un ex-regime `con400` a 400 rps è stato **rimosso**: a 340 rps su
un solo ingress il collo è l'Envoy non-scalabile, non il placement → non differenziava.)

### 3.4 Ripetizioni

Ogni cella `(arm, regime)` è ripetuta **3 volte** (reps) in sessioni separate; le figure
aggregate usano media ± deviazione. Lo sweep usa 1 rep/cella (basta per la *forma* della
curva).

---

## 4. Come si esegue un test

### 4.1 Un singolo run — `run_flash_crowd.sh`

```bash
DMOS_K6_REMOTE=ubuntu@<K6_VM_IP> bash experiments/run_flash_crowd.sh --label dmos-l1l2
```
Cosa fa:
1. **Pre-flight**: verifica tool, raggiungibilità dei 6 cluster, DMOS su :9090.
2. Avvia il **collector** (`collect_dmos_live.py`) sul laptop → campiona ogni 5s lo score
   DMOS, repliche current/target per cluster, CPU/mem, eventi di scaling → `dmos_live.jsonl`.
3. Lancia **k6** sulla VM remota (detached via SSH): genera il carico flash-crowd con i
   pesi del regime, colpendo l'ingress di ogni cluster.
4. Raccoglie gli artefatti nella cartella del run `experiments/results/<timestamp>_<label>/`.

**Artefatti per run**:
- `flash_crowd_summary.json` — metriche per-fase e per-cluster (p50/p95/p99, error rate, rps).
- `k6_events.json` — eventi grezzi per-richiesta (~1.2 GB): timestamp, durata, cluster, fase,
  status, `expected_response` (ok/fail). Serve per le figure time-series/CDF/SLO.
- `dmos_live.jsonl` — timeline scheduler+cluster (ogni 5s).
- `manifest.json` — config + git commit (riproducibilità).

### 4.2 Sessioni multi-cella (campagne)

Per non inquinare i run, ogni cella riparte da uno **stato pristino** con prep:
`kill DMOS → reset_test.ps1 (scala tutto a 1 e attende) → settle 75s → health-gate`
(verifica i 6 cluster a 200; se uno ha l'Envoy piantato, lo recupera con `rollout restart
frontend + cilium-envoy`). L'ordine è **equo** (off/l2/l1l2 consecutivi per regime).

| Script | Cosa lancia |
|---|---|
| `run_matrix_session.sh` | Matrice completa (uni/bal/con + ablazioni) — 1 rep/sessione. |
| `run_skew_sweep.sh` | Sweep di concentrazione 50/65/75% × off/l2/l1l2 (curva crossover). |

### 4.3 Hardening del banco (riproducibile)

I manifest di Online Boutique non sono nel repo (upstream), quindi le patch di risorse/probe
del banco vivono in **script versionati** (fonte di verità):
- `apply_resource_hardening.sh` — memoria anti-OOM (redis 1Gi, cart 512Mi, **recommendation
  1Gi + CPU 500m**) + liveness/readiness rilassate (5s/15s/fail6, anti throttle-kill).
- `apply_pod_lifecycle.sh` — `preStop` 15s + grace 30s (drain connessioni: riduce i timeout
  da scale-down a cooldown).

Questi toccano solo **malfunzioni del banco** (OOM, kill da throttle), **non** CPU/capacity
dei servizi (= il problema di saturazione che il paper studia, lasciato intatto).

### 4.4 Validità di un run

Un run è scartato dall'analisi se k6 ha **droppato ≥0.5%** delle iterazioni offerte (VU
starvation = carico non consegnato → arm incomparabile).

---

## 5. Le metriche

| Metrica | Definizione | Perché |
|---|---|---|
| **Error rate** | % richieste fallite (status≠2xx). | Completamenti. |
| **p95 latency (tutte)** | 95° percentile su *tutte* le richieste. | ⚠️ **fuorviante**: i fallimenti-lampo (503 in pochi ms) la deflazionano per gli arm che scaricano carico. |
| **p95 latency (riuscite)** | 95° percentile sulle *sole* richieste riuscite. | Latenza reale di chi viene servito. |
| **SLO violation** | % richieste **lente (>soglia, default 2000ms) OPPURE fallite**. | La metrica **onesta**: un 503 veloce è comunque una violazione. |
| **Goodput** | rps *riuscite* (= rps × (1−fail)). | Throughput utile. |

> **Bias da load-shedding (importante)**: gli arm sotto-provisionati (off/l2) **falliscono
> in fretta** (es. L2 fallisce con mediana 26ms). Questi fallimenti-lampo abbassano
> artificialmente la p95 "su tutte le richieste". L1+L2 invece **ammette e serve** le
> richieste dure (fallisce poco, ma quelle che falliscono hanno provato fino al timeout) →
> p95 più alta perché serve un workload più difficile. Per questo l'**headline è
> error/SLO/goodput**, e la p95 va letta *insieme* all'error rate.

---

## 6. I grafici (cosa analizza ognuno)

L'analisi (`paper_plot_detailed.py`) produce, in `paper_artifacts/detailed/`:
- una **cartella per regime** (`uni/`, `bal/`, `s50/`, `s65/`, `s75/`, `con/`) con 6 figure;
- **3 figure cross-regime** al livello superiore.

Le serie temporali usano il **rep rappresentativo** (mediano per error rate al picco) di
ogni cella; le barre aggregate usano la media sulle reps. Colori: blu=L1+L2, arancio=L2,
rosso=OFF. Le fasi sono evidenziate con bande colorate.

### 6.1 Per regime (in `detailed/<regime>/`)

| File | Cosa mostra | Cosa ci dice |
|---|---|---|
| **`key_metrics.png`** | 4 barre per arm: error rate (picco), **p95 tutte vs riuscite**, goodput, fallimenti+dropped. | Il riassunto headline di un regime: chi completa di più, chi è più veloce, chi spreca. |
| **`latency_timeseries.png`** | p50 / p95 / p99 nel tempo (3 pannelli). | *Quando* la latenza degrada nelle 5 fasi; transitori (es. lo spike di rampa di L1+L2). |
| **`error_throughput.png`** | Error rate / throughput riuscito / VUs nel tempo. | Dinamica errori e throughput; i VUs mostrano quanto carico k6 sostiene. |
| **`per_cluster.png`** | Per ciascuno dei 6 cluster: p95 + error rate nel tempo. | **Dove** stanno gli errori. A `con` si vede EU-DE (caldo) soffrire e gli altri 5 a 0% (asse error condiviso = sani). |
| **`error_heatmap.png`** | Heatmap error rate: arm×fase e arm×cluster (picco). | Sintesi a colpo d'occhio: in quale fase/cluster ogni arm fallisce. |
| **`cdf.png`** | CDF della latenza, globale + per-cluster. | Distribuzione completa (non solo i percentili); confronto della coda. |

### 6.2 Cross-regime (in `detailed/`)

| File | Cosa mostra | Cosa ci dice |
|---|---|---|
| **`slo_violation.png`** | 3+ pannelli (uni/bal/.../con): % SLO violation (picco) per arm. | La **headline**: a `con` L1+L2 ≈22% ≪ L2 ≈55% ≪ OFF ≈67%. A uni/bal ~uguali (nessuno stress). |
| **`slo_per_phase.png`** | Heatmap %SLO per fase × arm, un pannello per regime. | Dove (in quale fase) si concentrano le violazioni per ogni arm. |
| **`skew_robustness.png`** | %SLO (picco) vs **skew della domanda** (0.167→0.85), una linea per arm. | La curva del **crossover**: a che concentrazione off/l2 iniziano a degradare mentre L1+L2 resta piatto = **robustezza allo skew** (il valore del placement). |
| `slo_data.json` | I numeri dietro le figure SLO. | — |

### 6.3 Come rigenerare le figure

```bash
# tutte (dopo aver completato i run):
python experiments/paper_plot_detailed.py --output-dir paper_artifacts/detailed
# solo un regime (rigenerazione selettiva, salta le figure cross-regime):
python experiments/paper_plot_detailed.py --output-dir paper_artifacts/detailed --regimes con
```
> ⚠️ Non rigenerare **mentre un test è in corso**: il parsing dei `k6_events.json` (~1.2 GB)
> sul laptop ruba CPU a DMOS e può perturbare la cella in esecuzione.

La pipeline completa del paper (tabelle + report + altre figure) è
`paper_generate_all.py`, che invoca anche `paper_plot_detailed.py` (Step 2f) e
`paper_slo_compliance.py` (cache SLO per Table 1 / H2, con la stessa definizione
violazione = lenta o fallita).

---

## 7. Risultato sintetico (v1)

- **`con` (stress)**: L1+L2 vince nettamente su ogni metrica che conta — error 4% ≪ 26% ≪ 42%,
  SLO violation 22% ≪ 55% ≪ 67%, goodput 221 > 192 > 172. La p95 grezza più alta di L1+L2 è
  l'artefatto del load-shedding (serve le richieste dure che gli altri buttano via).
- **`uni`/`bal`**: tutti ~0% → **nessuna regressione** dove non c'è stress localizzato.
- **Sweep**: la curva mostra *dove* (a quale skew) il placement inizia a contare.
- **Limiti onesti** → future work (v2): transitorio di rampa (predictor più anticipatorio),
  steering geo/carbonio a livello GSLB (overflow), ablazione degli obiettivi (richiede
  floor-off/steering per differenziare).

---

## 8. Lo Skew Sweep test (cos'è e cosa fa)

### 8.1 Motivazione

Il regime `con` (skew 0.85) dimostra che, sotto un flash crowd **estremo**, il placement
geo-aware vince. Ma da solo è **un punto isolato**: un reviewer può chiedersi *"è un caso
limite o un comportamento robusto? A che livello di concentrazione il vantaggio emerge?"*.

Lo **Skew Sweep** risponde misurando la stessa metrica a **concentrazioni crescenti**,
trasformando il punto singolo in una **curva**.

### 8.2 Cosa fa, in pratica

Esegue il test flash-crowd **a peak fisso (350 rps)** variando **solo la concentrazione**
del picco sul cluster caldo (EU-DE):

| skew (hottest share) | regime | pesi W1/altri | rps sul cluster caldo |
|---|---|---|---|
| 0.167 | uni | 1/1/1/1/1/1 | ~58 |
| 0.40 | bal | 40/15/12/12/11/10 | ~140 |
| 0.50 | s50 | 50/10×5 | ~175 |
| 0.65 | s65 | 65/7×5 | ~228 |
| 0.75 | s75 | 75/5×5 | ~263 |
| 0.85 | con | 85/3×5 | ~297 |

Per ogni livello esegue i **3 arm** (off/l2/l1l2). Tenendo il peak costante e variando solo
lo skew, l'asse X (`hottest-cluster share`) isola **l'effetto della concentrazione** dal
volume → confronto pulito. Script: `experiments/run_skew_sweep.sh` (9 celle, ~2.5-3h, 1 rep/cella:
per una *curva* conta la forma, non il CI). I punti uni/bal/con vengono dai dataset già esistenti.

### 8.3 Perché è una prova forte

Una **curva piatta vs due curve che salgono** è la dimostrazione visiva della **robustezza
allo skew**: se l1l2 resta basso mentre off/l2 degradano, il placement geo-aware sta
*assorbendo* la concentrazione che gli altri non gestiscono. Il **punto di crossover**
(dove le curve divergono) dice *a quale livello di stress* il contributo inizia a contare.

---

## 9. Analisi dettagliata di ogni figura (risultati v1)

> **Premessa metodologica.** Le serie temporali e le distribuzioni usano il **rep
> rappresentativo** (mediano per error-rate al picco) di ogni cella; le barre aggregate la
> media sulle reps. Soglia SLO = **2000 ms**, e **violazione = lenta (>2s) OPPURE fallita**.
> Colori: **blu = L1+L2**, **arancio = L2**, **rosso = OFF**. Nelle figure SLO compare un
> ~13% di violazioni in *baseline/cooldown* **uguale per tutti gli arm**: è un artefatto del
> banco (coda dell'ingress Envoy a basso carico, richieste >2s ma non fallite) — non
> dipende dall'arm, quindi non sporca il confronto, che si gioca nella fase **peak**.

### 9.1 Figure del regime `con` (flash crowd estremo, skew 0.85)

#### `con/key_metrics.png` — il riassunto headline
**Cosa analizza:** 4 barre per arm — error rate (peak), p95 (tutte vs riuscite), goodput,
fallimenti+dropped — medie sulle 3 reps.
**Come è andata:** error rate **4.06% (l1l2) / 25.68% (l2) / 41.67% (off)**; goodput
**221 / 192 / 172** rps utili; richieste fallite **5.196 / 29.000 / 46.177**. La p95
mostra il caso "scomodo": l1l2 ~6450 ms vs l2 ~3980 / off ~4500.
**Vantaggio dimostrato:** L1+L2 **completa ~9× più richieste** di off e con goodput più alto.
La p95 più alta è il **bias da load-shedding** (vedi §5): off/l2 *sembrano* più veloci perché
rifiutano in fretta le richieste dure; il pannello "tutte vs riuscite" e l'error rate accanto
rendono il confronto onesto.

#### `con/latency_timeseries.png` — la dinamica della latenza (p50/p95/p99)
**Cosa analizza:** i percentili di latenza nel tempo, fase per fase.
**Come è andata:** **p50** — l1l2 (blu) è la **più bassa** durante il picco (~300-600 ms vs
~800-1500 ms di off/l2): *la richiesta mediana, con l1l2, è servita veloce*. **p95** — l1l2
ha lo **spike di rampa a ~14.000 ms intorno a t≈6 min** (transitorio: accoda mentre scala),
ma poi in **decay/cooldown crolla a ~500 ms PIATTO**, mentre off/l2 restano inchiodati a
~7000 ms ben oltre il picco.
**Vantaggio dimostrato:** due cose. (1) Il **corpo** della distribuzione (p50) è nettamente
migliore con l1l2. (2) **L1+L2 si riprende subito** dopo il picco (p95→500 ms), off/l2
restano saturi a lungo (recovery lenta). Lo spike di rampa è l'unico neo, inquadrato come
transitorio (→ future work: predictor più anticipatorio).

#### `con/error_throughput.png` — errori, throughput e VU nel tempo
**Cosa analizza:** error rate, RPS riuscito e Virtual Users attivi nel tempo.
**Come è andata:** **error rate** — off/l2 sostengono **20-40% di errori per tutta la fase
peak**; l1l2 resta **~0** (a parte lo spike di rampa ~27% e qualche picco <15%). **Throughput**
— l1l2 eroga **~400 RPS riusciti** contro i ~300 di off/l2. **VU** — l1l2 lavora con **meno
VU** (le richieste si chiudono prima); off resta "incollato" a VU alti (le richieste si
accumulano perché lente/fallite).
**Vantaggio dimostrato:** l1l2 trasforma lo stesso carico offerto in **più lavoro utile
completato** (throughput ↑) con **meno errori**, e i VU bassi confermano che il sistema non
si ingolfa. È la prova "di sistema" del vantaggio.

#### `con/per_cluster.png` — dove stanno gli errori (6 cluster)
**Cosa analizza:** p95 + error rate nel tempo per **ciascuno** dei 6 cluster (asse Y degli
error condiviso, così i cluster sani mostrano una linea piatta a 0).
**Come è andata:** **tutti gli errori sono su EU-DE** (il cluster caldo, 85% del traffico):
off ~40-55%, l2 ~30%, l1l2 basso (+ spike di rampa). Gli **altri 5 cluster sono a 0%** e con
p95 bassa (EU-NO ~150 ms, ecc.).
**Vantaggio dimostrato:** isola la **causa**: lo stress è *localizzato* su un cluster, e su
quel cluster l1l2 lo gestisce molto meglio. Visivamente smonta l'idea che "il test è troppo
pesante ovunque": 5 cluster su 6 sono sereni — è un problema di **placement locale**, che è
esattamente ciò che L1 risolve.

#### `con/error_heatmap.png` — sintesi fase × cluster
**Cosa analizza:** due heatmap dell'error rate: arm×fase e arm×cluster (peak).
**Come è andata:** **arm×fase** — gli errori si concentrano in **peak** (l1l2 4.27 / l2 22.98 /
off 32.53) e un po' in **spike** (0.51 / 7.36 / 7.91); baseline/decay/cooldown = 0.
**arm×cluster** — solo **EU-DE** (l1l2 5.0 / l2 27.2 / off 38.5), tutti gli altri 0.
**Vantaggio dimostrato:** è la "mappa di calore" che riassume in un colpo d'occhio *quando*
(peak) e *dove* (EU-DE) il sistema soffre, e quanto meno soffre con l1l2. Ottima per il paper
come figura-sintesi.

#### `con/cdf.png` — la distribuzione completa della latenza
**Cosa analizza:** CDF della latenza, globale e per-cluster (linea=arm, colore=cluster).
**Come è andata:** **globale** — la curva l1l2 **domina** (sta sopra/a sinistra): a 3000 ms
ha già servito **~90%** delle richieste, contro ~83% (l2) e ~80% (off). **Per-cluster** — le
curve di **EU-DE (verde)** sono le lente (arrivano solo a 0.78-0.85 a 3000 ms), gli altri
cluster salgono a 1.0 quasi subito; **dentro EU-DE**, l1l2 (linea piena) è la migliore, off
(punteggiata) la peggiore.
**Vantaggio dimostrato:** mostra il vantaggio **sull'intera distribuzione**, non solo sui
percentili scelti — robusto a critiche del tipo "hai scelto la metrica giusta". L1+L2
stocasticamente domina.

### 9.2 Skew Sweep — `skew_robustness.png`
**Cosa analizza:** % SLO violation (peak, >2s o fail) in funzione dello **skew della domanda**
(0.167 → 0.85), una linea per arm.
**Come è andata:**

| skew | OFF | L2 | L1+L2 |
|---|---|---|---|
| 0.167 (uni) | 3.9 | 3.9 | 3.8 |
| 0.40 (bal) | 6.5 | 5.0 | 4.2 |
| 0.50 (s50) | 12.2 | 1.7 | **0.3** |
| 0.65 (s65) | 50.8 | 11.8 | **2.9** |
| 0.75 (s75) | 59.1 | 12.9 | **1.1** |
| 0.85 (con) | 67.4 | 55.4 | **22.0** |

OFF si impenna già da **0.50** e arriva al **67%**; L2 degrada più dolcemente ma poi
**collassa a 0.85 (55%)**; **L1+L2 resta PIATTO ~0-3% fino a 0.75**, e si "sente" solo
all'estremo (22%).
**Vantaggio dimostrato:** è **la figura più forte del paper**. La curva piatta di l1l2 contro
le due che salgono = **robustezza allo skew** dimostrata. Il **crossover ~0.50-0.65** dice
*esattamente* da che concentrazione il placement geo-aware diventa decisivo. Trasforma un
singolo risultato in una **legge di comportamento**.

### 9.3 SLO — `slo_violation.png` e `slo_per_phase.png`

#### `slo_violation.png` — la headline onesta, per regime
**Cosa analizza:** % SLO violation (peak, **>2s o fail**) per arm, un pannello per regime
(uni/bal/s50/s65/s75/con).
**Come è andata:** uni/bal **indistinguibili** (~4-6% per tutti: nessuno stress + coda idle);
poi il divario si apre — s50 **0.3 / 1.7 / 12.2**, s65 **2.9 / 11.8 / 50.8**, s75
**1.1 / 12.9 / 59.1**, con **22.0 / 55.4 / 67.4** (l1l2/l2/off).
**Vantaggio dimostrato:** è la metrica **giusta** (conta i fallimenti come violazioni, non
premia il fail-fast). Dimostra che l1l2 mantiene lo SLO dove gli altri lo violano
massicciamente, e che a basso skew **non c'è regressione** (tutti pari).

#### `slo_per_phase.png` — *quando* si viola lo SLO
**Cosa analizza:** heatmap % SLO violation per **fase × arm**, un pannello per regime.
**Come è andata:** le violazioni si concentrano in **peak** (e in parte **spike** = la rampa):
es. a s75 peak off **59.1** / l2 **12.9** / l1l2 **1.1**; a con peak off **67.4** / l2 **55.4** /
l1l2 **22.0**. Le righe baseline/cooldown mostrano il ~13% **uguale per tutti** (coda idle del
banco, non un effetto-arm).
**Vantaggio dimostrato:** localizza il problema **nel tempo** e separa il segnale (peak/spike,
dove l1l2 vince) dal rumore di banco (baseline/cooldown, uguale per tutti) — trasparenza che
rafforza la credibilità del confronto.

### 9.4 Lettura d'insieme
Le figure raccontano una storia **coerente e a più livelli**:
1. **Cosa** succede (key_metrics, slo_violation): l1l2 completa di più e viola meno.
2. **Dove** (per_cluster, heatmap arm×cluster): tutto su EU-DE, gli altri 5 sereni → problema
   di placement locale.
3. **Quando** (latency_timeseries, error_throughput, slo_per_phase): in peak/spike; e l1l2
   recupera subito in decay.
4. **Come scala col carico** (skew_robustness): robustezza allo skew, con crossover ~0.5-0.65.
5. **Su tutta la distribuzione** (cdf): dominanza stocastica, non un artefatto di percentile.
L'unico neo onesto (lo spike di p95 in rampa di l1l2) è inquadrato come transitorio e
indirizzato in future work.

---

## 10. Scheda tecnica e analisi approfondita per figura

Questa sezione, per **ogni** figura, specifica: **assi** (cosa rappresentano X e Y, unità,
scala), **che metrica è e perché la analizziamo**, **da dove sono presi i dati** (file +
campo + calcolo esatto), i **numeri specifici** e la **lettura**.

### Glossario delle fonti dati (valido per tutte le figure)
- **`k6_events.json`** (per-run, ~1.2 GB, NDJSON): un *Point* per ogni evento k6. Campi usati:
  - `metric` — quale misura (`http_req_duration` = durata richiesta in ms; `vus` = utenti virtuali).
  - `data.value` — il valore (ms per la durata; conteggio per i VU).
  - `data.time` — timestamp ISO; convertito in **`elapsed_s`** = secondi dall'inizio del run.
  - `data.tags.cluster` — quale cluster ha servito (`ms-cluster-01..06`).
  - `data.tags.expected_response` — `"true"`/`"false"` → booleano **ok** (richiesta riuscita o no).
  - `data.tags.status` — codice HTTP.
  - `data.tags.phase` — fase del test (`baseline`/`spike`/`peak`/`decay`/`cooldown`).
- **`flash_crowd_summary.json`** (per-run): aggregati già calcolati da k6 a fine run, per fase
  e per cluster (`p50/p95/p99`, `errorRate`, `rps`, `requests`).
- **Binning temporale**: per le serie nel tempo, le richieste sono raggruppate in **finestre
  di 5 s** (`elapsed_s // 5`), poi si calcola la statistica per finestra.
- **Rep rappresentativo**: per le figure basate su un singolo run (serie temporali, CDF,
  heatmap, SLO) si usa la ripetizione con **error-rate al picco mediano** della cella.

---

### 10.1 `con/key_metrics.png` — 4 metriche sintetiche a barre

Figura a 4 pannelli; in ognuno l'asse **X è categoriale** (i 3 arm: L1+L2, L2, OFF).

**Pannello A — Error Rate (peak)**
- **Y**: error rate in **%** (0–~45), barra = **media sulle 3 reps**, baffo = ±deviazione standard.
- **Metrica / perché**: frazione di richieste **fallite** (status ≠ 2xx) durante la fase *peak*.
  È la metrica primaria di **disponibilità**: quante richieste il sistema NON completa sotto stress.
- **Dati**: `flash_crowd_summary.json` → `results.per_phase.peak.errorRate`, media sulle reps.
- **Numeri**: **4.06%** (L1+L2) / **25.68%** (L2) / **41.67%** (OFF).

**Pannello B — p95 Response Latency (peak): all vs successful**
- **Y**: 95° percentile della latenza in **ms**. Per ogni arm **2 barre**: tratteggiata = *tutte*
  le richieste, piena = *solo riuscite*.
- **Metrica / perché**: la latenza che il 95% degli utenti NON supera. Mostriamo *all* e
  *successful* perché la p95 "all" è **deflazionata dai fallimenti-lampo** (un 503 in pochi ms
  abbassa il percentile): vedere entrambe rende il confronto onesto.
- **Dati**: `k6_events.json` (rep) → `http_req_duration.value` con `phase=peak`; `np.percentile(·,95)`
  su tutte e sul sottoinsieme `expected_response=true`.
- **Numeri**: L1+L2 6453/6377 · L2 3984/4091 · OFF 4499/4878 ms (all/successful).

**Pannello C — Effective Throughput**
- **Y**: RPS **riusciti** (goodput).
- **Metrica / perché**: richieste andate a buon fine al secondo = `avg_rps × (1 − failure_rate)`.
  È il lavoro **utile** prodotto, non il carico offerto.
- **Dati**: summary `results.global.avg_rps` × (1 − `results.global.http_failure_rate`), media reps.
- **Numeri**: **221.1** / 191.8 / 172.5 RPS.

**Pannello D — Failed Requests & Dropped Iterations**
- **Y**: conteggio assoluto; 2 barre (richieste fallite; iterazioni droppate da k6).
- **Metrica / perché**: dimensione **assoluta** del danno (# fallimenti) + check di validità
  (dropped basso = il carico è stato consegnato, test valido — vedi §4.4).
- **Dati**: summary `results.global.total_requests × http_failure_rate`; `validity.dropped_iterations`.
- **Numeri**: fallite 5.196 / 29.000 / 46.177; dropped 328 / 178 / 239.

**Lettura:** L1+L2 completa ~9× più richieste di OFF con goodput più alto; la p95 più alta è
l'artefatto del load-shedding, reso esplicito dal confronto all-vs-successful + l'error rate.

---

### 10.2 `con/latency_timeseries.png` — percentili di latenza nel tempo

- **X**: **tempo in minuti** (0–15) = `elapsed_s/60`. Le bande colorate sullo sfondo sono le 5
  fasi. Asse X condiviso fra i 3 pannelli.
- **Y**: **latenza in ms** (scala lineare 0–~14000). Tre pannelli impilati: **p50**, **p95**, **p99**.
- **Metrica / perché**: percentili della **latenza end-to-end** vista dal client k6, calcolati su
  finestre di 5 s. p50 = esperienza *tipica* (mediana), p95/p99 = *coda* (worst-case). La media
  nascondereb­be la coda; i percentili nel tempo mostrano **quando** la latenza degrada e quanto
  rapidamente il sistema si **riprende**.
- **Dati**: `k6_events.json` (rep) → `http_req_duration.value`; bin 5 s; `quantile(0.50/0.95/0.99)`
  per bin; tutte le richieste (ok+fail).
- **Numeri/lettura**: p50 — L1+L2 la più bassa al picco (~300-600 ms vs 800-1500 di OFF/L2):
  la mediana è veloce. p95 — L1+L2 ha lo **spike di rampa ~14.000 ms a t≈6 min** (accoda mentre
  scala), poi in **decay crolla a ~500 ms PIATTO**, mentre OFF/L2 restano a ~7000 ms ben oltre il
  picco → **L1+L2 recupera subito, gli altri restano saturi**.

---

### 10.3 `con/error_throughput.png` — errori, throughput, VU nel tempo

Tre pannelli impilati, **X = tempo in minuti** (0–15) per tutti.

- **Error Rate** — **Y**: error rate **%** (0–50). Metrica: frazione fallita per finestra di 5 s.
  Dati: `http_req_duration` → per bin `(count − Σok)/count × 100`.
- **Throughput** — **Y**: **RPS riusciti**. Metrica: # richieste ok al secondo. Dati: solo
  `expected_response=true`, `count(ok)/5 s` per bin.
- **Virtual Users** — **Y**: **# VU attivi**. Metrica: utenti virtuali k6 (k6 ne *aggiunge* per
  tenere il rate quando le risposte rallentano → VU alti = richieste lente/accumulate). Dati:
  metrica `vus.value` nel tempo.
- **Perché**: dinamica **congiunta** — quanti errori, quanto lavoro utile, e quanto k6 deve
  "spingere". È la prova "di sistema" del comportamento.
- **Numeri/lettura**: OFF/L2 sostengono **20-40% errori** per tutto il peak; L1+L2 ~0 (+ spike di
  rampa ~27%). Throughput L1+L2 **~400** vs ~300 RPS. VU L1+L2 più **bassi** (richieste si chiudono
  prima); OFF resta incollato a VU alti (sistema ingolfato).

---

### 10.4 `con/per_cluster.png` — p95 + error per ciascuno dei 6 cluster

- **Layout**: 6 righe (un cluster per riga) × 2 colonne. **X = tempo in minuti** (0–15), condiviso
  per colonna.
- **Y colonna sinistra**: **p95 latenza (ms)**, scala **indipendente per riga** (i cluster hanno
  range diversi: EU-NO ~150 ms vs EU-DE ~12000 ms). **Y colonna destra**: **error rate %**, scala
  **condivisa fra i cluster** (0–~55) — così i cluster a 0% mostrano una linea piatta pulita
  invece di un asse ±0.04 fuorviante.
- **Metrica / perché**: p95 ed error rate **per-cluster** nel tempo. Serve a **localizzare
  geograficamente** lo stress: a `con` l'85% del traffico va su EU-DE, e vogliamo vedere se il
  problema è ovunque o concentrato.
- **Dati**: `http_req_duration` **filtrato per tag `cluster`**; bin 5 s; sinistra `quantile(0.95)`,
  destra `(count − Σok)/count`.
- **Numeri/lettura**: **tutti** gli errori su **EU-DE** (OFF 40-55%, L2 ~30%, L1+L2 basso + spike);
  gli **altri 5 cluster a 0%** e p95 bassa. Prova che lo stress è *localizzato* → problema di
  **placement locale**, esattamente ciò che L1 risolve (e smonta "il test è troppo pesante ovunque").

---

### 10.5 `con/error_heatmap.png` — sintesi 2D quando × dove

Due heatmap affiancate; colore **RdYlGn_r** (verde = basso, rosso = alto error rate).

- **Sinistra — Arm × Fase**: **Y** = arm, **X** = fase (`baseline/spike/peak/decay/cooldown`),
  cella = error % in quella fase.
- **Destra — Arm × Cluster (peak)**: **Y** = arm, **X** = cluster (i 6), cella = error % **nella
  sola fase peak**.
- **Metrica / perché**: error rate aggregato su due tagli (temporale e geografico). Risponde "in
  quale **fase** e in quale **cluster** ogni arm fallisce", in un colpo d'occhio.
- **Dati**: `http_req_duration` (rep), raggruppato per tag `phase` (sx) e per `cluster` con
  `phase=peak` (dx); cella = `(Σ¬ok)/n × 100`.
- **Numeri/lettura**: peak **4.27 / 22.98 / 32.53%**; spike 0.51 / 7.36 / 7.91%; baseline/decay/
  cooldown = 0. EU-DE@peak **5.0 / 27.2 / 38.5%**, tutti gli altri cluster 0. → errori concentrati
  in **peak** e su **EU-DE**, con L1+L2 molto più basso.

---

### 10.6 `con/cdf.png` — distribuzione cumulativa della latenza

- **Sinistra (globale)**: **X** = latenza in **ms** (0–3000), **Y** = **CDF** (0–1) = frazione
  cumulata di richieste con latenza ≤ X.
- **Destra (per-cluster)**: stessi assi; **stile linea = arm**, **colore = cluster**.
- **Metrica / perché**: la **CDF** = P(latenza ≤ x). Mostra l'**intera distribuzione**, non solo
  i percentili scelti → permette di affermare la **dominanza stocastica** (una curva sopra l'altra
  ovunque) ed è robusta all'accusa "hai scelto la metrica comoda".
- **Dati**: `http_req_duration.value` ordinato; `CDF = rango/N`. (Per-cluster: filtrato per `cluster`.)
- **Numeri/lettura**: globale — L1+L2 **domina** (sta sopra/a sinistra): a 3000 ms ha servito
  **~90%** delle richieste vs ~83% (L2) / ~80% (OFF). Per-cluster — le curve **EU-DE (verde)** sono
  le lente (arrivano solo a 0.78-0.85 a 3000 ms), gli altri salgono a 1.0 subito; **dentro EU-DE**,
  L1+L2 (linea piena) è la migliore, OFF (punteggiata) la peggiore.

---

### 10.7 `skew_robustness.png` — la curva del crossover (Skew Sweep)

- **X**: **skew della domanda** = *hottest-cluster share* (0.167 → 0.85), scala **continua**.
  È la frazione di traffico concentrata sul cluster più caldo (0.167 = uniforme su 6 cluster,
  0.85 = flash crowd estremo). Linee verticali tratteggiate marcano i regimi.
- **Y**: **% SLO violation** nella fase peak (0–70), dove violazione = latenza **> 2000 ms OPPURE
  richiesta fallita**.
- **Metrica / perché**: misura come la **qualità del servizio degrada al crescere della
  concentrazione**. È la prova di **robustezza allo skew**: una curva piatta (l1l2) contro due che
  salgono (off/l2) dimostra che il placement assorbe la concentrazione; il **crossover** indica
  *da quale livello di stress* il contributo diventa decisivo.
- **Dati**: per ogni (regime, arm) il rep rappresentativo → `slo_from_df` sulla fase peak
  (`frazione con value > 2000 OR ¬ok`); l'ascissa da `paper_lib.REGIME_SKEW`.
- **Numeri**: (OFF / L2 / L1+L2) — 0.167: 3.9/3.9/3.8 · 0.40: 6.5/5.0/4.2 · 0.50: 12.2/1.7/**0.3** ·
  0.65: 50.8/11.8/**2.9** · 0.75: 59.1/12.9/**1.1** · 0.85: 67.4/55.4/**22.0**.
- **Lettura**: OFF si impenna già da 0.50; L2 degrada dolce poi collassa a 0.85; **L1+L2 resta
  piatto ~0-3% fino a 0.75**. **Crossover ~0.50-0.65.** La figura più forte del paper.

---

### 10.8 `slo_violation.png` — SLO violation per regime (barre)

- **Layout**: un pannello per regime (uni, bal, s50, s65, s75, con).
- **X**: arm (categoriale). **Y**: **% SLO violation** nel peak (> 2 s o fail), scala **condivisa**
  fra i pannelli (0–70) per confronto immediato.
- **Metrica / perché**: la metrica di qualità **onesta** (conta i fallimenti come violazioni, così
  non premia il fail-fast — vedi §5). Mostra il confronto diretto per regime e il **no-regression**
  a basso skew.
- **Dati**: dizionario SLO (`slo_from_df`, fase peak) per ogni cella.
- **Numeri**: uni 3.8/3.9/3.9 · bal 4.2/5.0/6.5 · s50 0.3/1.7/12.2 · s65 2.9/11.8/50.8 ·
  s75 1.1/12.9/59.1 · con 22.0/55.4/67.4 (L1+L2/L2/OFF).
- **Lettura**: uni/bal indistinguibili (~4-6% per tutti: nessuno stress + coda idle); il divario si
  apre da s50 e diventa enorme a con. L1+L2 sempre il più basso.

---

### 10.9 `slo_per_phase.png` — SLO violation per fase (heatmap)

- **Layout**: un pannello (heatmap) per regime; colore RdYlGn_r, scala condivisa.
- **X**: **fase** (`baseline/spike/peak/decay/cooldown`). **Y**: arm. Cella = **% SLO violation**
  in quella fase.
- **Metrica / perché**: scompone la violazione SLO **nel tempo**. Serve a separare il **segnale**
  (peak/spike, dove l1l2 vince) dal **rumore di banco** (baseline/cooldown ~13% **uguale per tutti
  gli arm** = coda dell'ingress Envoy a basso carico, richieste >2 s ma non fallite) → trasparenza
  che rafforza la credibilità del confronto.
- **Dati**: `slo_from_df` campo `per_phase` per ogni cella.
- **Numeri/lettura**: le violazioni si concentrano in **peak** (e in parte **spike** = rampa):
  es. s75 peak OFF **59.1** / L2 **12.9** / L1+L2 **1.1**; con peak **67.4 / 55.4 / 22.0**. Le righe
  baseline/cooldown mostrano il ~13% uniforme (artefatto, non effetto-arm).

---

### 10.10 Tabella riassuntiva: figura → metrica → fonte

| Figura | Asse X | Asse Y | Metrica | Fonte dati |
|---|---|---|---|---|
| key_metrics | arm | %, ms, RPS, count | error/p95/goodput/fail | summary + k6_events(p95) |
| latency_timeseries | tempo (min) | latenza ms | p50/p95/p99 vs tempo | k6_events `http_req_duration` |
| error_throughput | tempo (min) | %, RPS, #VU | err/goodput/VU vs tempo | k6_events `http_req_duration`+`vus` |
| per_cluster | tempo (min) | p95 ms / error % | per-cluster vs tempo | k6_events filtrato `cluster` |
| error_heatmap | fase / cluster | arm | error % (2D) | k6_events `phase`/`cluster` |
| cdf | latenza ms | CDF (0-1) | distribuzione latenza | k6_events `http_req_duration` |
| skew_robustness | skew (0.167-0.85) | % SLO violation | robustezza allo skew | `slo_from_df` peak, tutti i regimi |
| slo_violation | arm | % SLO violation | SLO per regime | `slo_from_df` peak |
| slo_per_phase | fase | arm | SLO per fase | `slo_from_df` per_phase |
