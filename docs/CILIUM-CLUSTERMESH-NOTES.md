# Cilium ClusterMesh — operational notes

State of Cilium ClusterMesh on the 6 DMOS clusters, the data-plane
configuration used by the experiments, and the earlier configurations. The
role of the data plane in the architecture is described in
[`dmos-architecture.md` §4](dmos-architecture.md#4-data-plane).

## 1. Components

- Cilium **v1.19.4** (installed by proxmox-k3s, upgraded for DMOS by
  `scripts/setup-dmos-on-clusters.ps1`), kube-proxy replacement, Hubble.
- ClusterMesh connected: `cilium clustermesh status` → `5/5 remote clusters ready`
  on every cluster (pod-to-pod connectivity and shared identities).
- Relevant flags in `cilium-config` (`kubectl -n kube-system get cm cilium-config -o yaml`):

```yaml
clustermesh-enable-endpoint-sync: "false"   # "true" only in the overflow pilot
clustermesh-enable-mcs-api:       "false"
```

- Cilium Service annotations (Cilium 1.19 documentation, *Load-balancing
  & Service Discovery* and *Service Affinity*):

| Annotation | Effect |
|---|---|
| `service.cilium.io/global: "true"` | Global Service: Cilium merges the endpoints of the Services with the same name and namespace across all clusters of the mesh |
| `service.cilium.io/shared` | whether this cluster's endpoints are offered to the other clusters; for a global Service the default is `"true"` |
| `service.cilium.io/affinity` | `none` (default): no preference, the whole pool; `local`: healthy local endpoints, remote ones only if no local one is available; `remote`: the opposite |

The legacy keys `io.cilium/global-service` and `io.cilium/shared-service` no
longer appear in the documentation. The frontend manifest uses the
`service.cilium.io/*` keys; the backend manifest
(`deployments/global-service-backends.yaml`) still marks the backends global
with the legacy `io.cilium/global-service` key, next to
`service.cilium.io/affinity`. EndpointSlice synchronisation
(`service.cilium.io/global-sync-endpoint-slices`, Helm
`clustermesh.enableEndpointSliceSynchronization`, off by default) only serves
to expose the remote endpoints to other consumers (e.g. an external Ingress
controller): it is **not** required for global Service load balancing.

## 2. Current configuration

The frontend is exposed as a **global service with `affinity: none`**: the
ClusterMesh data plane spreads each connection, uniformly per endpoint, over
the union of the ready pods of all clusters. The share of traffic a cluster
serves is therefore its share of ready replicas: this is the lever DMOS acts
on.

| Object | Manifest | State |
|---|---|---|
| Service `frontend` | `deployments/global-service-frontend.yaml` | `global: "true"`, `shared: "true"`, `affinity: "none"` |
| cartservice, productcatalogservice, checkoutservice, recommendationservice | `deployments/global-service-backends.yaml` | global, `affinity: "local"` |
| currencyservice, adservice, shippingservice, the others | — | local Services |
| `clustermesh-enable-endpoint-sync` | `cilium-config` | `"false"` |

Notes:

- The frontend's CiliumNetworkPolicy (`deployments/cnp-frontend-l7.yaml`)
  admits the `remote-node` entity: without it, a frontend would not receive
  the requests that entered another cluster.
- Demand is measured client-side at the Ingress (`reporter="client"`), so it
  remains the demand that *enters* each cluster even when a remote frontend
  serves it ([traffic-metrics-architecture.md](traffic-metrics-architecture.md)).
- Only the frontend is pooled. The backends are not: cartservice,
  productcatalogservice, checkoutservice and recommendationservice are global
  with `affinity: local` (remote endpoints are used only as failover when no
  local endpoint is healthy; the manifest comments report gRPC dial timeouts
  with `none`), and currencyservice, adservice and shippingservice are plain
  cluster-local Services. A frontend therefore calls the backends of its own
  cluster, which is why DMOS co-scales the backends per cluster from the
  frontend replicas of that cluster
  ([BACKEND-CO-SCALING.md](BACKEND-CO-SCALING.md)).
- Before starting, every harness cell checks that the frontend has the
  expected affinity on every cluster (default `none`,
  `DMOS_EXPECT_FRONTEND_AFFINITY`), and records the annotations and
  `cilium-config` in `env_state_pre.json` / `env_state_post.json`
  ([RUN-PROVENANCE.md](RUN-PROVENANCE.md)).

Applying it (on every cluster):

```bash
for kc in <dmos-iac>/kubeconfig-ms-cluster-0{1..6}; do
  kubectl --kubeconfig "$kc" apply -f deployments/global-service-frontend.yaml
done
```

## 3. Earlier configurations

### v1 — "local-serving"

Same annotations as today (frontend global with `affinity=none`, backends
global with `affinity=local`, endpoint-sync off). A measurement taken at the
time (check [4b] of `experiments/preflight_runs.sh`, which counts the backends
of the frontend Service in the Cilium datapath) found **a single, local
backend**: the data plane behaved as cluster-local (see §3b). Before the
data-plane fixes, moreover, the frontend manifest in the repository said
`affinity: "local"`. With the current configuration the check must find all
the ready frontends of the 6 clusters in the datapath.

### v2 — overflow (pilot, not used in the campaigns)

Endpoint-sync `true` (with a rollout of the cilium-agents), frontend global
with `affinity=local`, backends made local, and a
`CiliumClusterwideEnvoyConfig` with outlier detection that ejects the local
frontends answering 5xx, so that the excess traffic spills over to the remote
frontends (east-west only). The pilot's manifests are not part of this
release; it must not be combined with the current configuration.

## 3b. Cilium bug with an explicit `affinity: "none"`

With `service.cilium.io/affinity: "none"` written in the manifest, Cilium
discards all the remote backends of the global service: the frontend only
sees the pods of its own cluster, and a cluster with no local pods no longer
serves its users
([issue #46623](https://github.com/cilium/cilium/issues/46623), fix
[PR #46635](https://github.com/cilium/cilium/pull/46635), backported to the
1.19 branch after 23 June 2026 with [PR #46691](https://github.com/cilium/cilium/pull/46691)).
**1.19.4** (13 May 2026) has the bug.

In the archived June 2026 runs (including those of an earlier version of
DMOS) every cluster served the demand that entered it, whatever its share of
replicas (served share = demand share, confirmed by the frontend CPU): local
serving, consistent with the bug (or with a local configuration).

The manifest stays `affinity: "none"`: real distribution requires a Cilium
version that includes the fix. Before running experiments that depend on
cross-cluster serving, check the version (`cilium version`, the DaemonSet
image) and the pool (§4, item 1). The arms with a minimum of 0 (`-min0`)
strictly require it: with the bug, a cluster with 0 frontends leaves its
users without service.

## 4. Checks on the testbed

1. **Frontend pool.** `cilium-dbg service list` on a node: the frontend
   Service must list the ready frontend pods of all 6 clusters as backends
   (with 1 replica per cluster, 6 backends).
2. **Distribution.** With k6 entering from a single cluster, Hubble must show
   requests served by the frontends of all clusters in proportion to their
   replicas.
3. **Backends.** If the backends are switched to `affinity: none`, recheck the
   gRPC dial timeouts that led to `local`.

## 5. Cost of distributing over the whole pool

A request that enters cluster *i* and is served by a frontend of cluster *j*
pays roughly an extra `delay_i + delay_j` (round trip between the two nodes,
netem); with the testbed delays this ranges from 30 ms (DE↔FR) to 300 ms
(US-VA↔SG). This is why placement must follow demand: with `affinity: none`
DMOS moves the served traffic by moving the replicas.

## 6. Commands

```bash
# mesh state and flag
cilium clustermesh status --context <ctx>
kubectl -n kube-system get cm cilium-config -o jsonpath='{.data.clustermesh-enable-endpoint-sync}'

# Service annotations
kubectl -n online-boutique get svc -o custom-columns='NAME:.metadata.name,ANN:.metadata.annotations'

# effective frontend backends in the datapath (from a cilium pod)
kubectl -n kube-system exec ds/cilium -- cilium-dbg service list | grep -A8 frontend

# full pre-run check
bash experiments/preflight_runs.sh
```

## References

- Cilium ClusterMesh — Load-balancing & Service Discovery:
  <https://docs.cilium.io/en/stable/network/clustermesh/services/>
- Cilium ClusterMesh — Service Affinity:
  <https://docs.cilium.io/en/stable/network/clustermesh/affinity/>
