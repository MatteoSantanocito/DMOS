# DMOS Experiment Runbook

## Struttura directory

```
Voda/
├── experiments/
│   ├── locustfile_multiingress.py   ← script Locust
│   ├── collect_metrics_simple.py    ← collector DMOS + Locust API
│   ├── analyze_test_complete.py     ← analisi JSONL → report + plot
│   └── plot_cluster_latency.py      ← plot per-cluster latency
├── results/
│   ├── *.jsonl                      ← output collector (un file per test)
│   └── multiingress/
│       ├── *_cluster_latency_*.csv  ← output Locust (latenze per cluster)
│       └── *_timeseries_*.csv       ← output Locust (serie temporale)
└── .kube/
    ├── cluster1.yaml
    ├── cluster2.yaml
    └── cluster3.yaml
```

> **Tutti i comandi vanno eseguiti dalla root `C:\Users\matte\Desktop\Voda\`**
> salvo dove indicato diversamente.

---

## Setup pre-test (fare SEMPRE prima di ogni test)

### 1. Reset repliche a 1 su tutti i cluster

```powershell
# Da: C:\Users\matte\Desktop\Voda\
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    foreach ($svc in @("frontend","cartservice","productcatalogservice","checkoutservice","recommendationservice")) {
        kubectl scale deployment/$svc -n online-boutique --replicas=1 --kubeconfig .kube\$ctx.yaml
    }
}
```

### 2. Verifica repliche a 1

```powershell
foreach ($ctx in @("cluster1","cluster2","cluster3")) {
    Write-Host "=== $ctx ===" -ForegroundColor Cyan
    kubectl get deployment -n online-boutique --kubeconfig .kube\$ctx.yaml `
        -o custom-columns="NAME:.metadata.name,READY:.status.readyReplicas,DESIRED:.spec.replicas"
}
```

### 3. Per test DMOS ON — avvia DMOS (su WSL/Linux)

```bash
cd ~/Voda   # o percorso del progetto su WSL
python src/dmos_main.py
```

### 4. Per test DMOS OFF — ferma DMOS

Nel terminale dove gira `dmos_main.py`:
```
Ctrl+C
```

Solo se gira in background (avviato con `&` o `nohup`):
```bash
pkill -f dmos_main.py
```

Verifica che sia fermo:
```bash
curl http://localhost:9090/metrics   # deve fallire / connection refused
```

---

## Analisi risultati

```powershell
# Da: C:\Users\matte\Desktop\Voda\
python experiments\analyze_test_complete.py results\<JSONL_FILE>

# Esempio:
python experiments\analyze_test_complete.py results\091012_20260326_flash_crowd_on.jsonl
```

### Plot per-cluster latency

```powershell
python experiments\plot_cluster_latency.py `
    results\multiingress\<scenario>_timeseries_<timestamp>.csv `
    results\multiingress\<scenario>_cluster_latency_<timestamp>.csv
```

---

## Stato test

| Scenario       | DMOS ON | DMOS OFF |
|----------------|---------|----------|
| gradual_ramp   | ✅ FATTO | ✅ FATTO |
| flash_crowd    | ✅ FATTO | ⏳        |
| double_wave    | ⏳        | ⏳        |
| sinusoidal     | ⏳        | ⏳        |

---

## GRADUAL RAMP

> Carico che sale gradualmente da 10 a 350 utenti in ~10 minuti, poi scende.

### DMOS ON ✅ (già eseguito)
- JSONL: `results/235039_20260325_gradual_ramp_on.jsonl`
- CSV:   `results/multiingress/gradual_ramp_cluster_latency_20260326_001446.csv`

### DMOS OFF ✅ (già eseguito)
- JSONL: `results/003634_20260326_gradual_ramp_off.jsonl`
- CSV:   `results/multiingress/gradual_ramp_cluster_latency_20260326_010039.csv`

---

## FLASH CROWD

> Spike improvviso: 10 utenti → 320 in ~2 minuti, poi scende rapidamente.

### DMOS ON ✅ (già eseguito)
- JSONL: `results/091012_20260326_flash_crowd_on.jsonl`
- CSV:   `results/multiingress/flash_crowd_cluster_latency_20260326_092719.csv`

### DMOS OFF ⏳

**Pre-test:** ferma DMOS, reset repliche a 1.

**Terminale 1** — dalla cartella `experiments\`:
```powershell
cd C:\Users\matte\Desktop\Voda\experiments
$env:DMOS_SCENARIO="flash_crowd"
locust -f locustfile_multiingress.py `
    --autostart --users 350 --spawn-rate 10 `
    --web-host 0.0.0.0 --web-port 8089 `
    --run-time 26m
```

**Terminale 2** — dalla root `Voda\`:
```powershell
cd C:\Users\matte\Desktop\Voda
python experiments\collect_metrics_simple.py 1560 --scenario flash_crowd_off --no-dmos
```

---

## DOUBLE WAVE

> Due picchi separati: primo picco ~300 utenti, discesa, secondo picco ~350 utenti.

### DMOS ON ⏳

**Pre-test:** avvia DMOS, reset repliche a 1.

**Terminale 1** — dalla cartella `experiments\`:
```powershell
cd C:\Users\matte\Desktop\Voda\experiments
$env:DMOS_SCENARIO="double_wave"
locust -f locustfile_multiingress.py `
    --autostart --users 350 --spawn-rate 10 `
    --web-host 0.0.0.0 --web-port 8089 `
    --run-time 26m
```

**Terminale 2** — dalla root `Voda\`:
```powershell
cd C:\Users\matte\Desktop\Voda
python experiments\collect_metrics_simple.py 1560 --scenario double_wave_on
```

### DMOS OFF ⏳

**Pre-test:** ferma DMOS, reset repliche a 1.

**Terminale 1** — dalla cartella `experiments\`:
```powershell
cd C:\Users\matte\Desktop\Voda\experiments
$env:DMOS_SCENARIO="double_wave"
locust -f locustfile_multiingress.py `
    --autostart --users 350 --spawn-rate 10 `
    --web-host 0.0.0.0 --web-port 8089 `
    --run-time 26m
```

**Terminale 2** — dalla root `Voda\`:
```powershell
cd C:\Users\matte\Desktop\Voda
python experiments\collect_metrics_simple.py 1560 --scenario double_wave_off --no-dmos
```

---

## SINUSOIDAL

> Carico sinusoidale periodico: oscilla tra 50 e 350 utenti con ciclo ~8 minuti.

### DMOS ON ⏳

**Pre-test:** avvia DMOS, reset repliche a 1.

**Terminale 1** — dalla cartella `experiments\`:
```powershell
cd C:\Users\matte\Desktop\Voda\experiments
$env:DMOS_SCENARIO="sinusoidal"
locust -f locustfile_multiingress.py `
    --autostart --users 350 --spawn-rate 10 `
    --web-host 0.0.0.0 --web-port 8089 `
    --run-time 26m
```

**Terminale 2** — dalla root `Voda\`:
```powershell
cd C:\Users\matte\Desktop\Voda
python experiments\collect_metrics_simple.py 1560 --scenario sinusoidal_on
```

### DMOS OFF ⏳

**Pre-test:** ferma DMOS, reset repliche a 1.

**Terminale 1** — dalla cartella `experiments\`:
```powershell
cd C:\Users\matte\Desktop\Voda\experiments
$env:DMOS_SCENARIO="sinusoidal"
locust -f locustfile_multiingress.py `
    --autostart --users 350 --spawn-rate 10 `
    --web-host 0.0.0.0 --web-port 8089 `
    --run-time 26m
```

**Terminale 2** — dalla root `Voda\`:
```powershell
cd C:\Users\matte\Desktop\Voda
python experiments\collect_metrics_simple.py 1560 --scenario sinusoidal_off --no-dmos
```

---

## Note operative

- **Tra un test e l'altro**: aspetta 2-3 minuti, poi reset repliche a 1
- **Safety watchdog**: il locustfile ferma automaticamente il test se p95 globale > 8000ms per 90s consecutivi (configurabile con `$env:SAFETY_P95_MS` e `$env:SAFETY_CONSECUTIVE`)
- **CSV Locust**: vengono salvati automaticamente in `results/multiingress/` al termine di ogni test Locust
- **JSONL collector**: viene salvato in `results/` al termine del collector
- **Analisi**: eseguire `analyze_test_complete.py` dalla root `Voda\` passando il path del JSONL
