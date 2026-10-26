# Cilium ClusterMesh — note operative

Stato di Cilium ClusterMesh sui 6 cluster DMOS, la configurazione del data
plane usata dal paper NCA 2026 e dagli esperimenti, e le configurazioni
precedenti. Il ruolo del data plane nell'architettura è in
[`dmos-architecture.md` §4](dmos-architecture.md#4-data-plane).

## 1. Componenti

- Cilium **v1.19.4** (installato da proxmox-k3s, aggiornato per DMOS da
  `scripts/setup-dmos-on-clusters.ps1`), kube-proxy replacement, Hubble.
- ClusterMesh connesso: `cilium clustermesh status` → `5/5 remote clusters ready`
  su ogni cluster (connettività pod-to-pod e identità condivise).
- Flag rilevanti in `cilium-config` (`kubectl -n kube-system get cm cilium-config -o yaml`):

```yaml
clustermesh-enable-endpoint-sync: "false"   # "true" solo nel pilota overflow
clustermesh-enable-mcs-api:       "false"
```

- Annotazioni Cilium dei Service (documentazione Cilium 1.19, *Load-balancing
  & Service Discovery* e *Service Affinity*):

| Annotazione | Effetto |
|---|---|
| `service.cilium.io/global: "true"` | Service globale: Cilium unisce gli endpoint dei Service con lo stesso nome e namespace in tutti i cluster della mesh |
| `service.cilium.io/shared` | se gli endpoint di questo cluster sono offerti agli altri cluster; per un Service globale il default è `"true"` |
| `service.cilium.io/affinity` | `none` (default): nessuna preferenza, tutto il pool; `local`: endpoint locali sani, remoti solo se nessun locale è disponibile; `remote`: l'opposto |

Le chiavi legacy `io.cilium/global-service` e `io.cilium/shared-service` non
compaiono più nella documentazione: i manifest usano le chiavi
`service.cilium.io/*`. La sincronizzazione delle EndpointSlice
(`service.cilium.io/global-sync-endpoint-slices`, Helm
`clustermesh.enableEndpointSliceSynchronization`, default spenta) serve solo a
esporre gli endpoint remoti ad altri consumatori (es. un Ingress controller
esterno): **non** è richiesta per il bilanciamento dei Service globali.

## 2. Configurazione del paper (in uso)

Il paper NCA 2026 descrive i servizi esposti come **global service con
`affinity: none`**: il data plane ClusterMesh distribuisce ogni connessione,
in modo uniforme per endpoint, sull'unione dei pod pronti di tutti i
cluster. La quota di traffico servita da un cluster è quindi la sua quota di
repliche pronte: è la leva su cui agisce DMOS.

| Oggetto | Manifest | Stato |
|---|---|---|
| Service `frontend` | `deployments/global-service-frontend.yaml` | `global: "true"`, `shared: "true"`, `affinity: "none"` |
| cartservice, productcatalogservice, checkoutservice, recommendationservice | `deployments/global-service-backends.yaml` | global, `affinity: "local"` |
| currencyservice, adservice, shippingservice, gli altri | — | Service locali |
| `clustermesh-enable-endpoint-sync` | `cilium-config` | `"false"` |

Note:

- La CiliumNetworkPolicy del frontend (`deployments/cnp-frontend-l7.yaml`)
  ammette l'entità `remote-node`: senza, un frontend non riceverebbe le
  richieste entrate da un altro cluster.
- La domanda è misurata lato client sull'Ingress (`reporter="client"`), quindi
  resta quella che *entra* in ogni cluster anche se a servirla è un frontend
  remoto ([traffic-metrics-architecture.md](traffic-metrics-architecture.md)).
- I backend restano `affinity: local` (i loro commenti riportano timeout di
  dial gRPC con `none`). Il paper dice "the services are exposed as global
  services with affinity: none": **da decidere** se la frase vale anche per i
  backend (§4).
- Ogni cella dell'harness controlla, prima di partire, che il frontend abbia
  l'affinity attesa su tutti i cluster (default `none`,
  `DMOS_EXPECT_FRONTEND_AFFINITY`), e registra annotazioni e `cilium-config`
  in `env_state_pre.json` / `env_state_post.json`
  ([RUN-PROVENANCE.md](RUN-PROVENANCE.md)).

Applicazione (su ogni cluster):

```bash
for kc in ~/dmos-iac/kubeconfig-ms-cluster-0{1..6}; do
  kubectl --kubeconfig "$kc" apply -f deployments/global-service-frontend.yaml
done
```

## 3. Configurazioni precedenti

### v1 — "local-serving"

Stato ripristinato da `deployments/clustermesh-overflow/revert-overflow.sh`
(frontend global `affinity=none`, backend global `affinity=local`,
endpoint-sync spento). Una misura di allora (check [4b] di
`experiments/preflight_runs.sh`, che conta i backend del Service frontend nel
datapath Cilium) trovava **un solo backend, locale**: il data plane si
comportava come cluster-local. Prima delle correzioni del data plane per il
paper, inoltre, il manifest del frontend nel repository diceva `affinity:
"local"`. Con la configurazione attuale il check deve trovare nel datapath
tutti i frontend pronti dei 6 cluster.

### v2 — overflow (pilota, non usato nelle campagne)

`deployments/clustermesh-overflow/apply-overflow.sh`: endpoint-sync `true`
(con rollout dei cilium-agent), frontend global con `affinity=local`, backend
resi locali, e una `CiliumClusterwideEnvoyConfig` con outlier detection che
espelle i frontend locali che rispondono 5xx, così che il traffico in eccesso
sfoci sui frontend remoti (solo est-ovest). Procedura, ipotesi e caveat:
[`../deployments/clustermesh-overflow/README.md`](../deployments/clustermesh-overflow/README.md).
Non va applicato insieme alla configurazione del paper.

## 4. Da verificare sul testbed (G0)

1. **Pool del frontend.** `cilium-dbg service list` su un nodo: il Service
   frontend deve elencare come backend i pod frontend pronti di tutti e 6 i
   cluster (con 1 replica per cluster, 6 backend).
2. **Distribuzione.** Con k6 che entra da un solo cluster, Hubble deve
   mostrare richieste servite da frontend di tutti i cluster in proporzione
   alle loro repliche.
3. **Backend.** Se i backend passano a `affinity: none`, ricontrollare i
   timeout di dial gRPC che avevano portato a `local`.

## 5. Costo della distribuzione su tutto il pool

Una richiesta entrata nel cluster *i* e servita da un frontend del cluster *j*
paga in più circa `ritardo_i + ritardo_j` (andata e ritorno tra i due nodi,
netem); con i ritardi del testbed sono da 30 ms (DE↔FR) a 300 ms (US-VA↔SG).
È il motivo per cui il placement deve seguire la domanda: con `affinity:
none` DMOS sposta il traffico servito spostando le repliche.

## 6. Comandi

```bash
# stato della mesh e del flag
cilium clustermesh status --context <ctx>
kubectl -n kube-system get cm cilium-config -o jsonpath='{.data.clustermesh-enable-endpoint-sync}'

# annotazioni dei Service
kubectl -n online-boutique get svc -o custom-columns='NAME:.metadata.name,ANN:.metadata.annotations'

# backend effettivi del frontend nel datapath (da un pod cilium)
kubectl -n kube-system exec ds/cilium -- cilium-dbg service list | grep -A8 frontend

# check completo pre-run
bash experiments/preflight_runs.sh
```

## Riferimenti

- Cilium ClusterMesh — Load-balancing & Service Discovery:
  <https://docs.cilium.io/en/stable/network/clustermesh/services/>
- Cilium ClusterMesh — Service Affinity:
  <https://docs.cilium.io/en/stable/network/clustermesh/affinity/>
