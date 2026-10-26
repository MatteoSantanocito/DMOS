# Global Services with Cilium ClusterMesh — Configuration and Verification

**Date**: March 20, 2026
**Context**: Extension of the DMOS system to support cross-cluster routing of the frontend service
**Motivation**: Advisor's suggestion — introduction of geo-awareness as a scheduling criterion

> **[ARCHIVED] Historical document — does not describe the current state.**
> This page records the original Global Services experiment on the old
> **3-cluster / NGINX-ingress** fleet, where `cilium service list` showed a
> remote frontend backend and cross-cluster routing was observed. On the
> current 6-cluster fleet (`ms-cluster-01..06`, Cilium 1.19) the configuration
> documented for the campaigns ("v1": `clustermesh-enable-endpoint-sync:
> false`, frontend global with `affinity=none`) was measured to give the
> frontend only its local backend, so each cluster serves the traffic that
> enters it. Cross-cluster overflow is the optional **v2 pilot** in
> `deployments/clustermesh-overflow/` (east-west only). Current configuration
> and open checks: [`../CILIUM-CLUSTERMESH-NOTES.md`](../CILIUM-CLUSTERMESH-NOTES.md).
> The statement below that `shared-service` defaults to `false` is not what
> the Cilium documentation says (global services are shared by default).

---

## 1. Context and Motivation

### 1.1 Advisor's observations

During a meeting with the advisor, the following observations on the system architecture emerged:

- **Clarify the optimization objective**: the system must explicitly declare what it optimizes (response time p95 < 200ms under capacity constraint)
- **Geo-awareness**: before deciding which cluster to schedule on, consider the geographic distribution of users and the network latency cluster→user
- **Network latency via ping**: use the RTT measured via ping as a proxy for user-cluster geographic proximity
- **Microservice placement**: as a future direction, evaluate inter-cluster latency to decide where to schedule microservices in the application chain

### 1.2 Comparison with Romano's thesis (2025)

The thesis by Romano Giuseppe (2025) implements a similar system with the following architectural differences:

| Aspect | Romano | This system (before) |
|---|---|---|
| Ingress controller | Cilium Ingress | Nginx Ingress |
| Global Services | enabled | not configured |
| Cross-cluster routing | Via Cilium ClusterMesh + Global Services | Each cluster independent |
| Geo-awareness | `ping_exporter` + `ping_rtt_mean_seconds` | Not present |
| Traffic metric | `hubble_http_requests_total{source="reserved:ingress"}` | `hubble_http_requests_total{destination_workload="frontend"}` |

Romano measures **inter-cluster latency** (cluster→cluster) via `ping_exporter` deployed through Helm, with targets configured on the node IPs of the other clusters. He uses `tc netem` to artificially inject network delay and simulate realistic geographic distances, since all clusters are physically on the same LAN.

### 1.3 Why Global Services are a prerequisite for geo-awareness

Without Global Services, each cluster serves its own requests in a completely independent way:

```
User → Nginx cluster1 → frontend pod cluster1  (always)
User → Nginx cluster2 → frontend pod cluster2  (always)
```

In this scenario, inter-cluster latency has no impact on the user: every request is served entirely by the cluster that receives it. Inter-cluster ping does not add useful information for scheduling.

With Global Services active, Cilium ClusterMesh forms a **single pool** of frontend pods across all clusters:

```
User → Nginx cluster1 → frontend pod cluster1  ←
                      → frontend pod cluster2  ← Cilium chooses
                      → frontend pod cluster3  ←
```

In this scenario, inter-cluster latency actually impacts the user experience: a request that enters on cluster1 may be served by a pod on cluster2, adding the latency of the ClusterMesh vxlan tunnel. Inter-cluster ping therefore becomes a relevant input for scheduling.

<!-- TODO(review): possibly stale — current DMOS uses Cilium Ingress with NodePort 30080, not Nginx; this section still describes the pre-migration Nginx-based topology. -->

---

## 2. Pre-configuration state

### 2.1 ClusterMesh already active

The system already had Cilium ClusterMesh configured and operational on all 3 clusters, with vxlan connectivity between the nodes:

| Cluster | Node IP | Simulated region | Carbon Intensity |
|---|---|---|---|
| cluster1 | <CLUSTER1_IP> | Frankfurt (DE) | 350 gCO2/kWh |
| cluster2 | <CLUSTER2_IP> | Paris (FR) | 80 gCO2/kWh |
| cluster3 | <CLUSTER3_IP> | Warsaw (PL) | 650 gCO2/kWh |

ClusterMesh connectivity verification (executed on all clusters):
```
✅ Service "clustermesh-apiserver" of type "NodePort" found
✅ All 1 nodes are connected to all clusters [min:2 / avg:2.0 / max:2]
🔌 cluster2: 1/1 configured, 1/1 connected - KVStoreMesh: 1/1 configured, 1/1 connected
🔌 cluster3: 1/1 configured, 1/1 connected - KVStoreMesh: 1/1 configured, 1/1 connected
```

### 2.2 Services without Global Service annotations

Before configuration, the `frontend` service in the `online-boutique` namespace was a simple ClusterIP without Cilium annotations. Each cluster operated in complete isolation: no cross-cluster endpoint was synchronized via ClusterMesh.

```bash
kubectl get svc frontend -n online-boutique --context cluster1 --show-labels
# No io.cilium/global-service annotation
```

### 2.3 Original CiliumNetworkPolicy

The CNP `l7-visibility-frontend` allowed ingress to the frontend only from:
- `fromEntities: cluster` — pods of the local cluster (any namespace, including `ingress-nginx`)
- `fromEntities: host` — kubelet process on the local node

Traffic coming from nodes of remote clusters (classified by Cilium as `remote-node`) was **blocked** by the policy, making cross-cluster routing impossible even if ClusterMesh had been configured to do so.

---

## 3. Changes made

### 3.1 New file: `deployments/global-service-frontend.yaml`

Created a Service manifest with the annotations required to enable Global Services:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: online-boutique
  annotations:
    io.cilium/global-service: "true"
    # io.cilium/shared-service NOT set (deliberate — see below)
spec:
  selector:
    app: frontend
  ports:
    - name: http
      port: 80
      targetPort: 8080
  type: ClusterIP
```

**Meaning of the annotations:**

- `io.cilium/global-service: "true"` — makes the Service visible and reachable from all clusters connected to the ClusterMesh. Cilium synchronizes the frontend endpoints across clusters via the distributed KVStore.
- `io.cilium/shared-service` — **deliberately NOT set** on the frontend. With `shared-service: "true"`, Cilium ClusterMesh would unify the frontend pods of all clusters into a single virtual pool and would automatically balance the requests arriving on each cluster across the pods of *all* clusters, proportionally to the number of pods. This would create a **positive feedback loop** on the Φ_demand component of DMOS: more pods on a cluster → Cilium forwards more traffic to it → Hubble measures more ingress → DMOS assigns it even more replicas → loop. Without `shared-service`, the Cilium Ingress of each cluster routes requests *only* to local frontend pods and Hubble measures the real demand per region. For backends, the same choice produces **local affinity with cross-cluster failover** (Cilium redirects only if 0 local pods are available).

**Application:**
```bash
kubectl apply -f deployments/global-service-frontend.yaml --context cluster1
kubectl apply -f deployments/global-service-frontend.yaml --context cluster2
kubectl apply -f deployments/global-service-frontend.yaml --context cluster3
```

Output: `service/frontend configured` on all 3 clusters.

### 3.2 Modification: `deployments/cnp-frontend-l7.yaml`

Added a second ingress rule to allow traffic from the nodes of remote clusters:

**Before (original ingress):**
```yaml
ingress:
- fromEntities:
  - cluster   # pods of the local cluster
  - host      # local kubelet
  toPorts:
  - ports:
    - port: "8080"
      protocol: TCP
```

**After (with cross-cluster rule):**
```yaml
ingress:
# Rule 1: intra-cluster traffic
- fromEntities:
  - cluster      # pods of any namespace in the local cluster
  - host         # process on the node (kubelet, health check)
  toPorts:
  - ports:
    - port: "8080"
      protocol: TCP

# Rule 2: cross-cluster traffic via Cilium ClusterMesh (Global Services)
- fromEntities:
  - remote-node  # nodes/pods of the remote clusters connected via ClusterMesh
  toPorts:
  - ports:
    - port: "8080"
      protocol: TCP
```

**Motivation:** With Global Services active, the Nginx of cluster2 or cluster3 can route requests towards frontend pods of cluster1 through the ClusterMesh vxlan tunnel. In Cilium, this traffic coming from remote nodes is classified as the `remote-node` entity. Without this rule, all cross-cluster traffic was silently blocked by the policy at the L3 level, making Global Services inoperative despite the correct configuration of the annotations.

**Application:**
```bash
kubectl apply -f deployments/cnp-frontend-l7.yaml --context cluster1
kubectl apply -f deployments/cnp-frontend-l7.yaml --context cluster2
kubectl apply -f deployments/cnp-frontend-l7.yaml --context cluster3
```

Output: `ciliumnetworkpolicy.cilium.io/l7-visibility-frontend configured` on all 3 clusters.

---

## 4. Verification and results

### 4.1 Annotation verification

```bash
kubectl get svc frontend -n online-boutique --context cluster1 \
  -o jsonpath='{.metadata.annotations.io\.cilium/global-service}'
# Output: true
```

Verified on all 3 clusters with output `true`.

### 4.2 ClusterMesh status verification

```
cilium clustermesh status --context cluster1

✅ Service "clustermesh-apiserver" of type "NodePort" found
✅ Cluster access information is available: <CLUSTER1_IP>:32379
✅ Deployment clustermesh-apiserver is ready
ℹ️  KVStoreMesh is enabled
✅ All 1 nodes are connected to all clusters [min:2 / avg:2.0 / max:2]
✅ All 1 KVStoreMesh replicas are connected to all clusters [min:2 / avg:2.0 / max:2]
🔌 cluster2: 1/1 configured, 1/1 connected
🔌 cluster3: 1/1 configured, 1/1 connected
```

### 4.3 Cilium service table verification — cross-cluster endpoints

Running `cilium service list` inside the agent pod of cluster1, the frontend row shows:

```
ID   Frontend              Service Type   Backend
17   10.43.87.229:80       ClusterIP      1 => 10.42.0.219:8080 (active)
                                          2 => 10.44.0.209:8080 (active)
```

**Analysis:**
- `10.42.0.x` — pod CIDR of cluster1. This is the local frontend pod.
- `10.44.0.x` — pod CIDR of a remote cluster (cluster2 or cluster3). This endpoint was synchronized via ClusterMesh after the application of the Global Service annotations.

Cilium is already balancing traffic between the two backends. This is the direct signal that the ClusterMesh has synchronized the endpoints and that the Cilium dataplane is configured for cross-cluster routing.

**Important note:** `kubectl get endpoints` continues to show only one endpoint (the local pod). This is expected behavior: Kubernetes is not aware of cross-cluster endpoints. Synchronization happens exclusively at the Cilium dataplane level, invisible to the standard Kubernetes APIs.

### 4.4 Cross-cluster routing verification with Hubble

With active traffic (generated via Locust with the `flash_crowd` scenario, 50 initial users distributed across the 3 clusters with weights 40/35/25), the output of `hubble observe --last 50` on cluster1 shows **two distinct frontend identities** receiving traffic from Nginx:

```
# LOCAL frontend pod of cluster1 (identity 91770 — present in cilium endpoint list)
ingress-nginx/ingress-nginx-controller-644c9c6b99-mc77d → online-boutique/frontend-76dbb9564c-dcg4b:8080   FORWARDED

# Frontend pod of REMOTE CLUSTER (identity 138068 — not present in cluster1's cilium endpoint list)
ingress-nginx/ingress-nginx-controller-644c9c6b99-wrgws → online-boutique/frontend-fc6cd7cb8-jgmt8:8080   FORWARDED
```

The pod `frontend-fc6cd7cb8-jgmt8` with Cilium identity `138068` is not present in cluster1's local endpoint list — it is a pod that physically resides on cluster2 or cluster3, reached through the ClusterMesh vxlan tunnel.

Further confirmation comes from the presence of active ClusterMesh traffic toward the clustermesh-apiserver:
```
10.44.0.76:60860 (host) -> kube-system/clustermesh-apiserver-56746c559d-vwbqf:9881   FORWARDED
```

`10.44.0.76` is a node of a remote cluster that maintains the connection toward cluster1's clustermesh-apiserver for the continuous synchronization of endpoints.

### 4.5 Application integrity verification

During the Locust test, the checkoutservice logs on cluster1 show orders completed correctly:

```json
{"message":"payment went through (transaction_id: d7ee9347-...)","severity":"info"}
{"message":"order confirmation email sent to \"test@example.com\"","severity":"info"}
```

The failures recorded by Locust on `/cart/checkout` (~20%) are **client-side timeouts** — Locust records as failures the requests that exceed the configured timeout, but the orders are still completed server-side. This behavior was already present before the introduction of Global Services and is due to the gRPC call chain of checkout (frontend → checkoutservice → paymentservice → shippingservice → emailservice), which under load can exceed the HTTP client timeout.

---

## 5. Impact on the DMOS architecture

### 5.1 Change in the meaning of Hubble metrics

With Global Services active, the metric `hubble_http_requests_total{destination_workload="frontend"}` on cluster1 measures the requests **served by cluster1's frontend pods**, including those forwarded by Nginx of cluster2 and cluster3. It no longer measures only the requests that "entered" through cluster1.

This change is consistent with the system's objective: DMOS must know how much work each cluster is doing, not just how many requests arrived at its ingress.

### 5.2 DMOS's role with Global Services

The DMOS system maintains its role unchanged: it decides the **number of replicas** per cluster and **which cluster** to scale on, based on the multi-dimensional score (latency, capacity, load, carbon). The routing of individual requests is delegated to the Cilium ClusterMesh dataplane, which automatically distributes traffic among all available frontend pods.

The combination is therefore:
- **DMOS** → decides the distribution of replicas across clusters (high-level scheduling)
- **Cilium ClusterMesh** → performs the actual routing of individual requests between pods (low-level forwarding)

### 5.3 Prerequisite for geo-awareness (implemented)

With Global Services active, inter-cluster ping becomes a relevant input for scheduling. If DMOS knows that cluster2 has an average latency of 150ms toward users and cluster1 has a latency of 8ms, it can:
1. Allocate more replicas on cluster1 (the cluster closest to users)
2. Indirectly weight Cilium's routing through the number of replicas available per cluster

The next steps have been completed in the current system:
1. Deploy of `ping_exporter` (project `czerwonk/ping_exporter`) on each cluster — see `deployments/ping-exporter-cluster{1,2,3}.yaml`
2. Simulation of geographic distances with `tc netem` on the node interfaces (cluster2 = 150 ms, cluster3 = 350 ms)
3. Addition of the score function `Φ_net` and the multiplicative term `Φ_geo = Φ_load · Φ_net` in `src/level1/score_functions.py`
4. Update of `config/weights.yaml` with the weight `omega_geo_load` and the `network_parameters` (ρ, RTT_max). Note: the initial version had a separate additive weight `omega_network`, later replaced by the current multiplicative form (see `dmos-architecture.md` for the complete formula).

---

## 6. Summary of modified files

| File | Type | Change description |
|---|---|---|
| `deployments/global-service-frontend.yaml` | New | Service manifest with `io.cilium/global-service: "true"`. **`io.cilium/shared-service` deliberately NOT set** (default: `false`) to avoid the Cilium feedback loop on Φ_demand discussed in §3 |
| `deployments/cnp-frontend-l7.yaml` | Modified | Added ingress rule `fromEntities: remote-node` for cross-cluster traffic |

---

## 7. Useful commands for maintenance

```bash
# Verify Global Service annotations on all clusters
for ctx in cluster1 cluster2 cluster3; do
  echo -n "$ctx: "
  kubectl get svc frontend -n online-boutique --context $ctx \
    -o jsonpath='{.metadata.annotations.io\.cilium/global-service}'
  echo
done

# Verify cross-cluster backends in the Cilium dataplane
kubectl exec -n kube-system ds/cilium --context cluster1 -- \
  cilium service list | grep "10.43.87.229"

# Observe cross-cluster flows in real time
cilium hubble port-forward --context cluster1 &
hubble observe --follow --namespace online-boutique \
  --to-label app=frontend --server 127.0.0.1:4245

# Full rollback (removes Global Services, returns to per-cluster isolation)
kubectl annotate svc frontend -n online-boutique \
  io.cilium/global-service- io.cilium/shared-service- \
  --context cluster1 --context cluster2 --context cluster3
```
