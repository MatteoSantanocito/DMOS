# DMOS Experiment Runbook

Flusso operativo per i run di valutazione flash-crowd (paper DMOS) sul
fleet a **6 cluster** (`ms-cluster-01..06`, K3s su Proxmox, Cilium
ClusterMesh, latenze geografiche emulate con `tc netem`).

## Strumenti attuali

```
$DMOS_REPO/
├── experiments/
│   ├── preflight_runs.sh            ← gate GO/NO-GO prima di una sessione
│   ├── run_naming.py                ← NOMI dei test: arm × regime → configurazione
│   ├── run_matrix_session.sh        ← UNA ripetizione di una matrice arm × regime
│   ├── run_skew_sweep.sh            ← scorciatoia: regimi s50 s65 s75
│   ├── run_one.sh                   ← una singola cella (arm × regime)
│   ├── run_flash_crowd.sh           ← orchestratore di UN run (k6 + collector)
│   ├── lib/run_common.sh            ← run_cell / run_named: prep + provenienza (comune a tutti)
│   ├── capture_env_state.py         ← snapshot testbed pre/post (netem, affinity, Cilium, HPA, loadgenerator)
│   ├── write_manifest.py            ← manifest di provenienza del run
│   ├── run_provenance.py            ← show / audit / diff della provenienza
│   ├── k6_flash_crowd.js            ← load generator k6 (5 fasi, 6 ingress)
│   ├── collect_dmos_live.py         ← collector timeline DMOS + Prometheus
│   ├── validate_run.sh              ← verdetto CLEAN / SUSPECT / DEGRADED
│   ├── paper_lib.py                 ← libreria condivisa dell'analisi
│   ├── paper_generate_all.py        ← pipeline completa figure + tabelle
│   └── results/                     ← un run = una cartella (gitignored)
├── .cache/
│   └── test-env.json                ← IP/kubeconfig/k6 VM (gitignored,
│                                      generato da scripts/render-runtime-files.sh)
└── paper_artifacts/                 ← output di paper_generate_all.py
```

> **Tutti i comandi vanno eseguiti da Git Bash, dalla root del repo**
> (`$DMOS_REPO`), salvo dove indicato. Gli script leggono topologia e
> credenziali da `.cache/test-env.json`: niente IP hardcoded. Se il file
> manca, rigeneralo con `bash scripts/render-runtime-files.sh` (parte da
> `config/local.yaml`).

## Prerequisiti

- `jq`, `kubectl`, `k6` (solo per prove locali), `python` con venv in
  `venv/Scripts` (gli script lo trovano da soli).
- Kubeconfig dei 6 cluster: `$KUBECONFIG_DIR/kubeconfig-ms-cluster-NN`
  (path registrati in `.cache/test-env.json`).
- VM k6 dedicata raggiungibile via SSH: gli script leggono
  `DMOS_K6_REMOTE` dall'ambiente oppure `.k6.remote` da
  `.cache/test-env.json` (es. `ubuntu@<K6_VM_IP>`), con chiave
  `.ssh.key` (override: `DMOS_K6_SSH_KEY`). `render-runtime-files.sh` non
  scrive `.k6.remote`: conviene esportare `DMOS_K6_REMOTE`. Senza
  `DMOS_K6_REMOTE`, `run_flash_crowd.sh` esegue k6 in locale (solo per prove).
- Data plane come nel paper: Service `frontend` global, shared, `affinity:
  none` (`deployments/global-service-frontend.yaml`). Ogni cella lo verifica
  prima di partire.
- **Misura della domanda decisa prima della campagna**
  (`config/services.yaml → application.demand`). Il default conta ogni
  client del cluster che chiama il frontend, come nei run NCA 2026. Per
  contare solo l'ingresso, guarda prima su un cluster chi chiama il frontend
  (Prometheus del cluster, durante un po' di carico):
  `sum by (source_workload) (rate(hubble_http_requests_total{destination_workload="frontend", reporter="client"}[1m]))`
  e metti il nome dell'ingresso in `source_workloads`. Non cambiarla a metà
  campagna: cambia la domanda misurata e il `config_id`.
- **Nessuna variabile `DMOS_*` di scoring esportata nella shell**
  (`DMOS_SCORE_*`, `DMOS_WEIGHTS_COHERENT`, `DMOS_LOCALITY_PURE_NET`,
  `DMOS_DEMAND_FLOOR`): la configurazione di ogni arm la decide il suo nome.
  Controllo: `env | grep '^DMOS_'`.

---

## 0. Pre-flight (fare SEMPRE prima di una sessione)

Un reboot/manutenzione dei cluster azzera stato volatile (netem in
primis) e invalida i run geo-aware senza errori evidenti. Il gate:

```bash
bash experiments/preflight_runs.sh            # verdetto GO / NO-GO
bash experiments/preflight_runs.sh --watch    # attende che i 6 cluster tornino su
bash experiments/preflight_runs.sh --fix      # riapplica netem se resettato
bash experiments/preflight_runs.sh --skip-dmos
```

Controlla: raggiungibilità ingress+Prometheus dei 6 cluster, nodi Ready,
**netem/geo-RTT** (il check chiave), sanity `ping_rtt`, annotation
Cilium dei 5 Service gestiti, stato pod, layer DMOS. Exit 0 = GO,
1 = NO-GO (non lanciare i run), 2 = prerequisiti mancanti.

Per un reset profondo del fleet prima di una sessione:
`bash experiments/clean_reset.sh [--deep]`.

---

## 1. Nomi dei test

Ogni cella ha un nome  **`dmos-<arm>__<regime>`**, che è anche il nome della
cartella del run (`experiments/results/<ts>_dmos-<arm>__<regime>/`). Il
significato di ogni nome è definito in **un solo punto**,
`experiments/run_naming.py`, ed è identico in tutti gli script e
nell'analisi: un nome non cambia mai significato fra una campagna e l'altra.

```bash
python experiments/run_naming.py list              # tutti gli arm e i regimi
python experiments/run_naming.py show nca con-sg   # cosa esegue una cella
bash experiments/run_matrix_session.sh --arms "off l2 l1l2" --regimes "bal con" --dry-run
```

**Arm** — cosa gira:

| Arm | Cosa esegue |
|---|---|
| `off` | DMOS spento: 1 replica per servizio e cluster |
| `l2` | solo Level 2 (`--no-scheduling`): budget N e controllo, distribuzione uniforme N/K |
| `l1l2` | DMOS completo **come nella campagna NCA 2026**: gate g = 2, Φ_loc = Φ_net, Φ_pre spento, pesi lat/cap/net/car = 0.25/0.20/0.25/0.30, demand floor spento (letto dai log DMOS del 15–17 giugno 2026) |
| `nca`, `gate_g0`, `gate_g1`, `gate_g3`, `additive_dem50`, `locality`, `latency_first`, `capacity_first`, `carbon_first`, … | DMOS completo con un **modello di score** di `config/weights.yaml → score_models` (`DMOS_SCORE_MODEL=<nome>`). `nca` = configurazione *descritta* nel paper (pesi 0.25/0.35/0.30/0.10) |
| `profile-<nome>` | profilo legacy di `score_weights` (`DMOS_SCORE_PROFILE`), es. `profile-latonly` |
| `legacy-default`, `legacy-gate1`, `legacy-nofloor` | configurazioni storiche: default del codice (floor acceso), gate g = 1, score additivo senza floor |
| `hpa-none`, `hpa-local`, `karmada-fhpa`, `karmada-budget` | riservati a E1, **non ancora implementati** (lo script si ferma) |

**Regime** — che carico: `<domanda>[-<cluster caldo>][-p<picco>]`

| Parte | Valori |
|---|---|
| domanda | `uni` (1/6 a testa), `bal` (0.40), `s50`, `s65`, `s75`, `con` (0.85): quota del cluster caldo nel picco |
| cluster caldo | `de` (default, ms-cluster-01), `fr`, `pl`, `us`, `sg`, `no`: riceve la quota del cluster caldo (i pesi si scambiano con quelli di DE) |
| picco | `p<req/s>`, default 350; sopra 350 il fattore VU di k6 passa a 30 |

Esempi: `con`, `bal-sg`, `con-pl`, `s50-p500`, `con-sg-p200`. La forma
canonica omette i default (`con-de-p350` diventa `con`).

**Nomi vecchi** (rifiutati con il nome nuovo da usare): `l1l2coh` → `l1l2`,
`l1l2gate` → `legacy-gate1`, `l1l2nofloor` → `legacy-nofloor`,
`l1l2floor` → `legacy-default`, `latonly`/`netonly`/`carbon` →
`profile-latonly`/`profile-netonly`/`profile-carbon`, regime `con400` →
`con-p400`. Le cartelle delle campagne passate con i nomi vecchi restano
leggibili dall'analisi.

> **Pesi di DMOS.** Il paper NCA dichiara (0.25, 0.35, 0.30, 0.10), ma la
> campagna ha girato con (0.25, 0.20, 0.25, 0.30). `l1l2` riproduce la
> campagna, `nca` il testo; i modelli di E2/E3 (`gate_g*`, `additive_dem50`,
> `*_first`) partono dai pesi di `nca`. Va deciso quale dei due è "DMOS" per
> la rivista prima di F1 (vedi `docs/publications/2026-nca-dmos.md`).

---

## 2. Sessione (una ripetizione di una matrice)

**Filosofia: UNA rep per sessione.** Rilancia lo stesso comando in sessioni
separate per accumulare ripetizioni: i run con la stessa label in timestamp
diversi sono ripetizioni per `paper_generate_all.py`, che avvisa se la stessa
label raccoglie `config_id` diversi.

```bash
bash experiments/run_matrix_session.sh [--arms "A B ..."] [--regimes "R ..."] [--dry-run]
```

Default: `--arms "off l2 l1l2" --regimes "uni bal s50 s65 s75 con"` (il core
del paper NCA, 18 celle). Cosa fa:

1. **Valida tutti i nomi** prima di toccare il testbed.
2. **Ordine equo**: per ogni regime gli arm girano consecutivi, così
   condividono lo stesso stato del cluster.
3. **Stato pristino per cella**: arresto di DMOS → `scripts/reset_test.ps1`
   (tutto a 1 replica, loadgenerator a 0) → quiescenza `SETTLE` (default
   75 s) → **health-gate** (i 6 ingress devono rispondere 200;
   auto-recovery con restart frontend + cilium-envoy sui cluster giù).
4. **Tracciabilità** (`experiments/lib/run_common.sh → run_cell`): snapshot
   del testbed prima del run (`env_state_pre.json`) con **gate**: la cella
   non parte (cartella in `results/_aborted/`) se netem manca, se il frontend
   non ha l'affinity attesa (`none`) o se il loadgenerator gira; DMOS avviato
   con log e configurazione effettiva nella cartella del run (`dmos.log`,
   `dmos_config.json` con `config_id`); snapshot dopo il run
   (`env_state_post.json`); manifest completato con i controlli di
   provenienza. Dettagli: [`docs/RUN-PROVENANCE.md`](../docs/RUN-PROVENANCE.md).

`run_skew_sweep.sh` è la stessa cosa con `--regimes "s50 s65 s75"`.

Variabili opzionali: `SETTLE`, `DMOS_K6_REMOTE`, `DMOS_REQUIRE_NETEM`
(default 1), `DMOS_ENV_CAPTURE` (default 1), `DMOS_EXPECT_FRONTEND_AFFINITY`
(default `none`), `DMOS_ALLOW_LOADGENERATOR` (default 0).

## 3. Run singolo (una cella)

Per un pilot o per rifare un run contaminato senza lanciare una sessione:

```bash
bash experiments/run_one.sh ARM REGIME [--dry-run]
bash experiments/run_one.sh --list
# es.
bash experiments/run_one.sh l1l2 con       # DMOS come nella campagna NCA
bash experiments/run_one.sh nca  con       # DMOS come descritto nel paper
bash experiments/run_one.sh gate_g0 s50
bash experiments/run_one.sh l2 con-sg
```

Lo script fa da solo prep pristino + health-gate, avvia/ferma DMOS (tranne
per `off`) e delega a `run_flash_crowd.sh`.

## 4. Comandi per fase (piano di test della rivista)

Ogni comando è **una** ripetizione: lancialo una volta per sessione, tante
volte quante sono le ripetizioni previste (5 nei blocchi principali, 3 in
quelli secondari). Prima di ogni fase: pre-flight (§0).

```bash
# F0 — pilot: un run per configurazione nuova a CON, FUORI dal dataset
bash experiments/run_one.sh nca con
bash experiments/run_one.sh gate_g0 con
bash experiments/run_one.sh additive_dem50 con
bash experiments/run_one.sh l1l2 con-sg
bash experiments/run_one.sh l1l2 con-pl
#   poi spostali fuori dal dataset (l'analisi legge solo le cartelle di primo livello):
mkdir -p experiments/results/_pilot && mv experiments/results/<ts>_<label> experiments/results/_pilot/

# F1 — core OFF / L2 / L1L2 sui 6 regimi (18 celle)
bash experiments/run_matrix_session.sh
#   baseline HPA (affinity none / local): arm hpa-none / hpa-local non ancora implementati

# F2 / E2 — gate sì o no e quanto forte, a BAL, 0.50, CON (15 celle)
bash experiments/run_matrix_session.sh --arms "gate_g0 gate_g1 nca gate_g3 additive_dem50" --regimes "bal s50 con"

# F2 / E3 — profili di pesi a BAL e CON (8 celle)
bash experiments/run_matrix_session.sh --arms "nca latency_first capacity_first carbon_first" --regimes "bal con"

# F3 / E4 — cluster caldo PL e SG (18 celle)
bash experiments/run_matrix_session.sh --arms "off l2 l1l2" --regimes "bal-pl s50-pl con-pl bal-sg s50-sg con-sg"

# F3 / E4 — volume a skew fisso, 200 / 350 / 500 req/s (9 celle)
bash experiments/run_matrix_session.sh --arms "off l2 l1l2" --regimes "con-p200 con con-p500"
```

Le forme di picco (rampa veloce/lenta, picchi multipli, hotspot che si
sposta), MuBench, HPA e Karmada (E1, F4) non sono ancora nello schema dei
nomi: vanno aggiunti a `run_naming.py` quando saranno implementati. Per
MuBench DMOS è già pronto: basta un `services.yaml` con la sua sezione
`application` (servizio d'ingresso, servizi co-scalati, namespace) e le
capacità misurate; restano specifici di Online Boutique il collector
(`collect_dmos_live.py`), lo script k6 e gli script di controllo
(`dmos-ready-check.py`).

## 5. Cosa succede sotto: run_flash_crowd.sh

Ogni run esegue `k6_flash_crowd.js` — 5 fasi in 900 s: baseline (50 rps),
spike (ramp verso il picco in 120 s), peak, decay, cooldown — contro i 6
ingress con i pesi del regime, e in parallelo il collector
`collect_dmos_live.py` (sempre sul laptop) che scrapa il `/metrics` di
DMOS (`localhost:9090`) e il Prometheus di ogni cluster.

Output: una cartella per run in `experiments/results/<TS>_<label>/`:

| File | Contenuto |
|---|---|
| `manifest.json` | config k6, pesi regime, arm/regime, git (+ diff se sporco), host, **controlli di provenienza** |
| `dmos_config.json` | configurazione **effettiva** di DMOS (formula, gate, pesi, L2, safeguard, fonte del carbonio per cluster) e `config_id` |
| `dmos.log` | log completo di DMOS |
| `env_state_pre.json` / `env_state_post.json` | stato del testbed: netem misurato, affinity dei Service, Cilium, HPA, loadgenerator |
| `dmos_config_expected.json` | configurazione attesa dall'arm (pre-check offline) |
| `config_snapshot/` | copia delle `config/*.yaml` usate (`local.yaml` sanificato) |
| `flash_crowd_summary.json` | metriche k6 per-fase / per-cluster |
| `dmos_live.jsonl` | timeline scheduler+cluster (score Φ, repliche, eventi, `config_id`) |
| `k6_stdout.txt` | stdout completo di k6 |
| `k6_events.json` | eventi grezzi k6 (uno per richiesta), base delle metriche SLO |

Ispezione: `python experiments/run_provenance.py show <cartella_run>`
(tutto il run), `... audit` (tutti i run: validità, label che mescolano
configurazioni), `... diff <A> <B>`. Per verificare la configurazione di
un'arm PRIMA di lanciarla, senza cluster:
`python experiments/run_naming.py show <arm> <regime>`, oppure
`DMOS_SCORE_MODEL=nca python -m src.dmos_main --dump-config-only /tmp/cfg.json`
(`run_cell` lo fa da solo e salva il risultato in `dmos_config_expected.json`).
Il significato di ogni file e metrica è in
[docs/METRICS-AND-OUTPUTS.md](../docs/METRICS-AND-OUTPUTS.md).

Uso diretto (di norma non serve: passano da `run_one.sh`/matrice):

```bash
bash experiments/run_flash_crowd.sh --label dmos-off__con --no-dmos
```

Il collector si può anche lanciare a mano (es. baseline senza DMOS):

```bash
python experiments/collect_dmos_live.py 900 --scenario flash_crowd_off --no-dmos
```

## 6. Validazione del run (subito dopo ogni run)

Su questo testbed il data-plane può degradare a metà run: non si
previene, si **detecta e si scarta**.

```bash
bash experiments/validate_run.sh              # valida l'ULTIMO run
bash experiments/validate_run.sh experiments/results/<TS>_<label>
```

Verdetto: `CLEAN` (exit 0), `SUSPECT` (exit 2: peak err 6-10% o coda hot
anomala), `DEGRADED` (exit 3: peak err > 10% o un cluster non-hot con
err > 2%). Tieni i CLEAN, rifai i DEGRADED.

> ⚠️ `validate_run.sh` giudica le **prestazioni** dell'arm, non la salute del
> testbed: a `con` le baseline OFF/L2 superano il 10% di errori *per
> costruzione* e risultano DEGRADED pur essendo run validi. La validità del
> testbed (netem pre/post, configurazione DMOS nota e costante, niente DMOS
> attivo in un arm OFF, loadgenerator spento) è nei controlli di provenienza
> del manifest: `python experiments/run_provenance.py show <cartella_run>`.

---

## 7. Analisi risultati

Tutta l'analisi passa da `paper_lib.py` + `paper_generate_all.py`, che
genera figure, tabelle LaTeX e report HTML in
`paper_artifacts/<YYYYMMDD_HHMMSS>/`:

```bash
python experiments/paper_generate_all.py
python experiments/paper_generate_all.py --tag "v1-final"
python experiments/paper_generate_all.py --scenarios dmos-off__con dmos-l2__con dmos-l1l2__con
```

Le figure del core hanno come default gli arm `dmos-off`/`dmos-l2`/`dmos-l1l2`;
per altri arm o regimi:

```bash
python experiments/paper_plot_skew_robustness.py --arms dmos-gate_g0 dmos-nca dmos-additive_dem50 --regimes bal s50 con
python experiments/paper_plot_hotshare.py --dmos-arm dmos-l1l2 --hot sg     # cluster caldo SG
python experiments/paper_plot_detailed.py --dmos-arm dmos-nca
python experiments/paper_ablation_compare.py --regime con                   # E3
DMOS_H_ARM=dmos-nca python experiments/paper_report.py                      # ipotesi H1–H4
```

Elenco di figure, tabelle e definizioni delle metriche:
[docs/METRICS-AND-OUTPUTS.md](../docs/METRICS-AND-OUTPUTS.md).

La pipeline aggrega automaticamente le ripetizioni (stessa label,
timestamp diversi) e avvisa se uno scenario ha ≠ 3 reps. Gli step
best-effort vengono saltati se mancano i run relativi.

---

## Note operative

- **Avvio manuale di DMOS** (di norma lo gestiscono gli script):
  `python -m src.dmos_main` dalla root; endpoint metriche su
  `http://localhost:9090/metrics`. Stop: Ctrl+C.
- **Tra un run e l'altro** non serve reset manuale: prep pristino,
  settle e health-gate sono dentro `run_one.sh`/`run_matrix_session.sh`.
- **netem**: se il pre-flight segnala RTT piatta, `--fix` riapplica i
  delay (`scripts/setup-netem-and-ping.sh`).
- **loadgenerator** di Online Boutique: genera traffico verso il frontend
  dall'interno del cluster, che k6 non conta ma DMOS vede come domanda.
  Il reset (`scripts/reset_test.ps1`, `clean_reset.sh`) porta tutti i
  Deployment a 1 replica e poi il `loadgenerator` a 0; ogni cella si ferma
  se lo trova acceso (`DMOS_ALLOW_LOADGENERATOR=1` per disattivare il
  controllo) e il manifest lo segna come errore. Il restart di frontend e
  cilium-envoy dell'health-gate resta: serve per i pod che si bloccano per
  colpa di Cilium ed è indipendente dal loadgenerator.
- **Arresto di DMOS negli script**: `kill_dmos` (in `lib/run_common.sh`)
  ferma **solo** i processi Python la cui riga di comando contiene
  `-m src.dmos_main` (su Windows via `Win32_Process`; su Linux il PID salvato
  in `.cache/dmos.pid`, verificato, più `pkill -f`). Gli altri programmi
  Python della macchina non vengono toccati.
- **Capacity test** (calibrazione `capacity_req_per_sec`): procedura
  separata, vedi `docs/capacity-testing.md` e
  `experiments/CAPACITY-GRPC.md`.
