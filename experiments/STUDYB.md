# Studio B — Trade-off carbon vs P95

> **Stato: studio esplorativo, fuori dai paper pubblicati.** Usa l'arm legacy
> `l1l2coh` (variabili d'ambiente) e una redistribuzione GSLB tramite una
> `CiliumClusterwideEnvoyConfig`; gli script `gslb_cec_writer.py`,
> `studyb_weights.py` e `studyb_collect.py` vivono nel repository esterno
> `dmos-iac`, non qui. La CEC GSLB è una risorsa dei cluster: non va
> modificata o rimossa senza conferma. Per il ruolo dei pesi nella
> estensione su rivista si usano i modelli di score `*_first`
> ([../docs/dmos-architecture.md §7.4](../docs/dmos-architecture.md#74-score-models-the-role-of-demand)).
> Carbonio: le zone senza traccia (oggi `US-VA` e `SG`) usano profili
> orari dipendenti dall'ora reale ([../docs/dmos-architecture.md §5.3](../docs/dmos-architecture.md#53-carbon-intensity)).
> **Naming**: l'arm `l1l2coh` oggi si chiama `l1l2` (`experiments/run_naming.py`)
> e fissa da sé `DMOS_WEIGHTS_COHERENT=0.25,0.20,0.25,0.30`: un valore
> esportato nella shell viene sovrascritto. Per rifare questo studio definisci
> i due vettori di pesi come modelli di score in `config/weights.yaml`
> (`demand_mode: gate`, `gate: {eps: 0.05, gamma: 2}`) e lanciali per nome.

Dimostra che lo **stesso score multi-obiettivo** di DMOS è *azionabile*: cambiando
il profilo di pesi cambia sia lo scaling L1/L2 sia i pesi Envoy del 2° livello
(redistribuzione GSLB), e con essi **dove** viene servito il traffico → cambia la
carbon-intensity efficace, a un costo misurabile in latenza P95.

Due run identiche tranne il **vettore di pesi coerente** `DMOS_WEIGHTS_COHERENT`
(arm `l1l2coh` → i pesi vengono da qui, non da `weights.yaml`). Ordine dei 4
valori: **lat,cap,net,car**.

| Run       | `DMOS_WEIGHTS_COHERENT` (lat,cap,net,car) | Atteso |
|-----------|-------------------------------------------|--------|
| `latency` | `0.55,0.20,0.20,0.05` | traffico → DE (veloce, no netem): P95 ↓, CI ↑ |
| `carbon`  | `0.10,0.20,0.20,0.50` | traffico → FR (nucleare, +150ms netem): CI ↓, P95 ↑ |

(cap e net restano uguali nelle due run → l'unica variabile è lo scambio
lat↔car. I valori sommano a 1; aggiustabili.)

Il trade-off vive su **cluster1 (DE, 350 gCO₂, 0 ms)** ↔ **cluster2 (FR, 80 gCO₂,
150 ms)**; cluster3 (PL, 650 gCO₂, 350 ms) è dominato.

> **Scope**: lo Studio B usa i 3 cluster con regione carbon definita
> (cluster1/2/3 = DE/FR/PL). I cluster 04–06 ricevono peso GSLB 0.

---

## Domanda UNIFORME (requisito di design — leggere prima)

DMOS usa un **gate moltiplicativo** sulla domanda: `S_i = G(d_i)·Σ_k ω_k Φ_k`,
con `G(d)=(ε+(1−ε)·Φ_dem)^γ`. Sotto domanda **concentrata** (es. flash crowd su
DE) il gate azzera lo score dei cluster senza domanda locale → il cluster verde
(FR) verrebbe gatato a ~0 e il termine carbon **non sarebbe azionabile**, né nello
scaling né nei pesi GSLB (che derivano dallo score).

⇒ Lo Studio B si esegue sotto **domanda uniforme** (`uni`: stesso carico su tutti
i cluster). Con `d_i = 1/N` il gate è lo **stesso fattore costante** per tutti →
si semplifica nel rapporto degli score → a decidere è solo `Σ_k ω_k Φ_k`, cioè i
pesi multi-obiettivo. **Stessa identica formula dello Studio A**: cambia solo il
regime di domanda (uniforme invece di concentrato) e il profilo di pesi.
`studyb_weights.py` normalizza i pesi allo score minimo, quindi è invariante al
fattore di gate costante e conserva la differenziazione carbon↔latency.

- k6: carico **uniforme** sugli ingress dei 3 cluster (stesso rps per cluster).
- arm DMOS: lo stesso dello Studio A (`l1l2coh`, gate ON) — non cambiare il design.

## Validità sperimentale (leggere prima)

1. **Stessa CI nelle due run.** Altrimenti il risparmio si confonde con la deriva
   oraria della carbon-intensity. Due opzioni:
   - in `config/carbon.yaml` metti `realistic.noise_enabled: false` ed esegui le
     due run **nella stessa ora** (il profilo orario dà lo stesso valore), **oppure**
   - passa `--ci 01=350,02=80,03=650` **identico** nelle due `finalize`.
2. **Stessa offered load, stessa durata, stesso target k6** nelle due run. L'unica
   variabile è il profilo di pesi.
3. **Snapshot solo a regime.** Fai `snapshot before` *dopo* il warmup e dopo che
   Envoy ha recepito i nuovi pesi; `snapshot after` a fine finestra di misura. Il
   delta dei contatori cumulativi isola la finestra (immune al ring-buffer Hubble).
4. **Spegni il `loadgenerator` interno** di Online Boutique durante le run: anche
   il suo traffico passa dal CEC e verrebbe redistribuito, contaminando le
   frazioni servite. Guida con il solo k6:
   ```bash
   for c in 01 02 03 04 05 06; do
     kubectl --kubeconfig "${KUBECONFIG_DIR:-$HOME/dmos-iac}"/kubeconfig-ms-cluster-$c -n online-boutique \
       scale deploy/loadgenerator --replicas=0
   done
   ```
   (ripristina a fine studio con `--replicas=1`).

---

## Prerequisiti

- CEC del GSLB attivo su tutti i cluster (`gslb_cec_writer.py`).
- DMOS in esecuzione con exporter su `:9090/metrics`.
- `studyb_weights.py` e `studyb_collect.py` sono in `dmos-iac/` (accanto a
  `gslb_cec_writer.py`, stessa risoluzione dei kubeconfig); `studyb_plot.py` qui
  in `experiments/`.

---

## Procedura (ripeti per ogni profilo: latency, poi carbon)

`$WC` = vettore coerente del profilo (vedi tabella). Esempio `latency`.

```bash
# === A) avvia DMOS col profilo (dal repo Voda) ============================
cd <REPO_ROOT>
PY=venv/Scripts/python.exe; [ -f "$PY" ] || PY=python
WC=0.55,0.20,0.20,0.05          # latency  (carbon: 0.10,0.20,0.20,0.50)
env DMOS_DEMAND_FLOOR=0 DMOS_SCORE_GATE=1 DMOS_SCORE_GATE_GAMMA=2 \
    DMOS_LOCALITY_PURE_NET=1 DMOS_WEIGHTS_COHERENT=$WC \
    "$PY" -m src.dmos_main > /tmp/dmos_studyb_latency.log 2>&1 &
DMOS_PID=$!

# attendi ~150 s: nei primi 120 s DMOS è nel periodo di grazia (osserva ma
# non scala né calcola lo score), quindi i due profili non si differenziano.
sleep 150
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:9090/metrics   # atteso 200

# === B) pesi GSLB dallo score DMOS -> applica al CEC ======================
cd $HOME/dmos-iac
python studyb_weights.py --apply        # weight[c]=round(score[c]/min_score*10)
sleep 15                                 # Envoy recepisce i pesi

# === C) snapshot PRIMA, k6 uniforme, snapshot DOPO =======================
python studyb_collect.py snapshot --out before_lat.json

# k6 a carico UNIFORME sui 3 cluster (W tutti =1), DMOS resta su:
cd <REPO_ROOT>
env DMOS_SPIKE_W1=1 DMOS_SPIKE_W2=1 DMOS_SPIKE_W3=1 \
    DMOS_SPIKE_W4=0 DMOS_SPIKE_W5=0 DMOS_SPIKE_W6=0 DMOS_PEAK_RPS=150 \
    bash experiments/run_flash_crowd.sh --label studyb-latency
# -> P95 in experiments/results/<ts>_studyb-latency/flash_crowd_summary.json
#    (.results.global.p95_ms)

cd $HOME/dmos-iac
python studyb_collect.py snapshot --out after_lat.json

# === D) finalize (CI identica nelle due run) =============================
P95=$(python -c "import json,glob,os;d=sorted(glob.glob('<REPO_ROOT>/experiments/results/*_studyb-latency'))[-1];print(json.load(open(d+'/flash_crowd_summary.json'))['results']['global']['p95_ms'])")
python studyb_collect.py finalize --before before_lat.json --after after_lat.json \
    --label latency --p95-ms "$P95" --ci 01=350,02=80,03=650 \
    --out studyb_latency.json

# === E) ferma DMOS e ripeti per "carbon" =================================
kill $DMOS_PID 2>/dev/null
# ripeti A–D con WC=0.10,0.20,0.20,0.50, label studyb-carbon, file *_car.json
#   -> studyb_carbon.json   (stesso PEAK_RPS, stessa ora, stessa CI)
```

## Figura

```bash
cd <REPO_ROOT>
python experiments/studyb_plot.py \
    --latency $HOME/dmos-iac/studyb_latency.json \
    --carbon  $HOME/dmos-iac/studyb_carbon.json \
    --output-dir paper_artifacts/studyb
# -> fig_studyb_tradeoff.{pdf,png}  +  stampa "−X% CO2 / +Y ms P95"
```

`fig_studyb_tradeoff`: (a) barre impilate della frazione di traffico servita per
cluster nelle due run (mostra lo spostamento DE→FR), (b) trade-off a 2 punti
CI_eff vs P95 con annotato risparmio carbon e costo latenza.

---

## Metriche

- **CI efficace** (pesata sul traffico servito):
  `CI_eff = Σ_c served_fraction[c] · CI[c]`, con `served_fraction` dai contatori
  Envoy `rq_total` per-endpoint del cluster `frontend-global`, attribuiti al
  cluster proprietario dell'IP e sommati su tutti gli Envoy della mesh.
- **Risparmio carbon** = `(CI_eff^latency − CI_eff^carbon) / CI_eff^latency`.
- **P95** end-to-end dal `handleSummary` di k6 (stesso campo usato nello Studio A).
