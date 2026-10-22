# ClusterMesh Overflow — setup "frontend global + affinity-local + failover su saturazione"

Documentazione completa del passaggio dalla config **v1 (local-serving)** alla
config **v2 (overflow cross-cluster)**, per il pilota DMOS.

> **Stato:** pilota opzionale, non usato nelle campagne di valutazione. La
> redistribuzione è possibile solo est-ovest (frontend → frontend attraverso la
> mesh); il percorso nord-sud (mandare l'utente all'Ingress di un altro
> cluster) non è realizzabile con questo meccanismo. Il quadro complessivo e i
> punti ancora da verificare sono in
> [`docs/CILIUM-CLUSTERMESH-NOTES.md`](../../docs/CILIUM-CLUSTERMESH-NOTES.md).

---

## 1. Perché (obiettivo dell'esperimento)

Nella config **v1** (attuale) i Service sono di fatto **cluster-local**:
- mesh connesso (5/5 remote clusters ready), ma `clustermesh-enable-endpoint-sync=false`
  → i global service **non condividono endpoint** cross-cluster;
- frontend: `service.cilium.io/global=true` ma `affinity=none`; 4 backend global+local; resto local;
- verifica diretta: il Service `frontend` ha **1 solo backend, locale** → nessun routing cross-cluster.

Conseguenza misurata: a `con` (85% domanda su cluster-01) DMOS può solo **scalare
localmente** il cluster caldo. Vantaggio l1l2 vs l2 reale ma **modesto** (errori
22%→17%), perché 85% su UN cluster servito localmente ha un **tetto fisico**
(concurrency-bound): nemmeno scalando si scende sotto ~17% errori.

**Ipotesi v2 (overflow):** se l'eccesso di carico **sfocia** sui cluster vicini
quando cluster-01 satura, l'errore crolla — e il vantaggio di l1l2 si **amplifica**,
perché l1l2 ha **pre-posizionato** capacità sui cluster di overflow (placement
multi-obiettivo), mentre off/l2 no. Questo renderebbe il multi-cluster **reale nel
data-plane**.

---

## 2. Ricerca Cilium (cosa dicono le doc) e scelte di design

Fonti: doc ufficiali Cilium 1.19 (ClusterMesh Affinity, Load-balancing & Service
Discovery, L7 Circuit Breaking).

1. **Annotation global (moderna, 1.13+):** `service.cilium.io/global: "true"`.
   La legacy `io.cilium/global-service` è auto-tradotta da Cilium 1.19 (il frontend
   ha già entrambe). → usiamo la **moderna**.

2. **Affinity** (`service.cilium.io/affinity`):
   - `local`  = LB sui backend **locali sani**; usa i **remoti SOLO se tutti i
     locali sono non disponibili/unhealthy** (failover **health-based**).
   - `remote` = preferisce i remoti.
   - `none`   = LB su **tutti** i backend (locali+remoti) senza preferenza.
   → vogliamo **`local`** (servi locale; sfoga sui remoti solo quando i locali
     "cadono").

3. **Il limite chiave:** `affinity: local` fa failover **solo se i backend locali
   sono unhealthy/down**, NON sotto carico. Un pod *Ready ma saturo* resta in
   servizio → niente failover. **Cilium non ha failover su soglia CPU.**

4. **La soluzione: outlier detection (Envoy via `CiliumClusterwideEnvoyConfig`).**
   Envoy **espelle** gli endpoint che ritornano 5xx consecutivi. Sotto saturazione
   il frontend locale erra (a `con` misurato ~38% errori) → outlier detection lo
   **espelle** → il traffico va sui remoti (failover **error/overload-based**).
   Con `max_ejection_percent: 100` può espellere tutti i locali → failover completo.

5. **Condivisione endpoint cross-cluster:** richiede `clustermesh-enable-endpoint-sync=true`
   (oggi `false`). Senza, i global service restano local-only (è il nostro caso v1).

6. **Backend local:** togliamo l'annotation global dai backend → ogni frontend
   (anche quello su cui sfocia il traffico) usa i **propri backend locali**
   → niente gRPC cross-cluster (evita la cascata di latenza che aveva spinto a
   disattivare l'overflow in origine).

**Catena risultante (v2):**
`utente → ingress cluster-i → frontend locale` → se il frontend locale satura
(5xx) → outlier detection lo espelle → `failover → frontend di un cluster remoto`
(round-robin sui remoti sani; DMOS, avendo messo **più repliche** sui cluster
preparati, ne attira di più) → quel frontend usa i **suoi backend locali**.

---

## 3. Caveat onesti (da tenere a mente nel pilota)

- **Esito incerto:** l'overflow potrebbe migliorare `con` ma **peggiorare uni/bal**
  (coda cross-cluster verso i cluster a 200ms). Il pilota serve a misurarlo.
- **Outlier detection è error-based**, non CPU-based: dipende dal fatto che il
  frontend saturo **ritorni 5xx** (a `con` lo fa, ~38%). Se non erra abbastanza,
  l'ejection non scatta.
- **Round-robin, non load-aware:** DMOS non dirige il failover esplicitamente;
  lo influenza **indirettamente** via numero di endpoint (più repliche su un
  cluster = più traffico di failover lì).
- **Rollout disruptivo:** abilitare endpoint-sync richiede `rollout restart` dei
  cilium-agent su tutti i cluster (breve disruzione di rete).
- **Flapping/thundering-herd possibili** quando un cluster viene espulso in blocco.

---

## 4. Procedura (in ordine)

### (a) Snapshot della config v1 (local-serving)
```bash
bash deployments/clustermesh-overflow/snapshot-config.sh > CONFIG_V1_SNAPSHOT.txt
```
Salva uno snapshot testuale della config Cilium corrente, utile come riferimento
per il confronto v1/v2 e per verificare il rollback.

### (b) Applica la config v2 (overflow) su tutti e 6 i cluster
```bash
bash deployments/clustermesh-overflow/apply-overflow.sh
```
Lo script: frontend→`affinity=local`; backend→toglie global (local); cilium-config
`clustermesh-enable-endpoint-sync=true` + rollout cilium; applica il CCEC outlier
detection. **Idempotente.** Per tornare a v1: `revert-overflow.sh`.

### (c) Verifica che l'overflow sia attivo
```bash
bash experiments/preflight_runs.sh        # check [4b]: frontend deve avere >1 backend (endpoint remoti)
```
Atteso nel check [4b]:
- `endpoint-sync = true`
- con `affinity=local` i backend remoti restano in **standby** (peso 0) e a
  riposo non compaiono tra quelli attivi: il check stampa
  `frontend: 1 backend attivo + remoti in STANDBY`. Il failover verso i remoti
  si vede solo sotto saturazione, quando l'outlier detection espelle i locali.

### (d) Pilota: confronto v1 vs v2 sotto carico concentrato
Esegui lo stesso scenario flash crowd (regime concentrato `con`) nelle tre
modalità (off, solo-L2, L1+L2) in entrambe le config, e confronta `p95` ed
`errorRate` della fase di picco dal `flash_crowd_summary.json` di ciascun run.

**Lettura del risultato:**
- se a `con` l'overflow crolla gli errori (es. 17%→<5%) e il gap l1l2 vs l2
  si allarga → la config v2 è validata;
- se non aiuta o peggiora (latenza cross-cluster) → **revert a v1** con
  `revert-overflow.sh`.

---

## 5. Rollback
```bash
bash deployments/clustermesh-overflow/revert-overflow.sh
bash experiments/preflight_runs.sh    # [4b] deve tornare: endpoint-sync=false, frontend 1 backend
```

---

## 6. File in questa cartella
- `README.md` — questo documento.
- `ccec-frontend-outlier.yaml` — CiliumClusterwideEnvoyConfig (outlier detection sul frontend).
- `apply-overflow.sh` — applica la config v2 sui 6 cluster (idempotente).
- `revert-overflow.sh` — ripristina la config v1 (local-serving).
- `snapshot-config.sh` — salva uno snapshot testuale della config Cilium corrente.
