# Cilium Network Policies — Complete Documentation
## Context: DMOS Multi-Cluster with Cilium ClusterMesh

> Originally written 2026-02-27 following the debugging session that resolved
> the 100% timeout issue in multiingress Locust tests. Updated for the current
> Cilium-Ingress-based DMOS deployment.

---

## Index

1. [Cluster architecture](#1-cluster-architecture)
2. [What is a CiliumNetworkPolicy](#2-what-is-a-ciliumnetworkpolicy)
3. [Identities and Entities in Cilium](#3-identities-and-entities-in-cilium)
4. [L7 Rules and the Envoy proxy](#4-l7-rules-and-the-envoy-proxy)
5. [Active CNPs in DMOS](#5-active-cnps-in-dmos)
6. [Cilium with NodePort traffic](#6-cilium-with-nodeport-traffic)
7. [Testbed hygiene that affects the policies](#7-testbed-hygiene-that-affects-the-policies)
8. [Pre-run checks](#8-pre-run-checks)
9. [Useful reference commands](#9-useful-reference-commands)

---

## 1. Cluster architecture

The DMOS setup uses **6 Kubernetes clusters** (`ms-cluster-01` …
`ms-cluster-06`, one K3s cluster per Proxmox VM) connected via
**Cilium ClusterMesh**:

```
Test client (<CLIENT_IP>)  <-- test traffic -->  ms-cluster-01 (<CLUSTER1_IP>)
                                                 ms-cluster-02 (<CLUSTER2_IP>)
                                                 ...
                                                 ms-cluster-06 (<CLUSTER6_IP>)

Each cluster:
  - NodePort 30080  -> Cilium Ingress (entry point for external traffic)
  - NodePort 32090  -> Prometheus (kube-prometheus-stack, ns observability)

Cilium replaces kube-proxy (eBPF full replacement mode).
Cilium ClusterMesh connects the 6 clusters for service discovery and
cross-cluster policy.
```

**Exposed services:**

| Service | Type | Internal port | NodePort |
|---|---|---|---|
| `frontend` | ClusterIP | 80 → pod:8080 | — |
| Cilium Ingress | NodePort | 80 → frontend:8080 | **30080** |
| `prometheus` | NodePort | 9090 | 32090 |

> **Note**: external traffic always enters the cluster through the **Cilium
> Ingress** Envoy listener on NodePort 30080. The frontend `Service` itself is
> ClusterIP and is not exposed directly to the network.

---

## 2. What is a CiliumNetworkPolicy

A `CiliumNetworkPolicy` (CNP) is a Kubernetes Custom Resource (CRD) that extends
the standard Kubernetes `NetworkPolicy` with advanced Cilium features:

- **L3/L4**: filtering by IP, CIDR, port, protocol
- **L7**: HTTP, gRPC, Kafka inspection (via the integrated Envoy proxy)
- **Identity-based**: filtering by pod label rather than by IP (more robust in
  dynamic environments)
- **Special entities**: `world`, `host`, `cluster`, `remote-node`, `ingress`, etc.

### Why DMOS uses CNPs

> **CNPs in DMOS are NOT used to restrict traffic for security reasons.** They
> exist purely to enable **L7 observability** — flipping the Cilium Envoy proxy
> on for selected hops so that Hubble can produce
> `hubble_http_requests_total` and `hubble_http_request_duration_seconds_bucket`
> metrics, which DMOS reads from Prometheus for the ingress demand and Φ_lat
> (see [dmos-architecture.md §5](dmos-architecture.md#5-observed-state-metrics-and-queries)).
> The only flows they drop on purpose are those reaching the frontend from
> outside the Cilium Ingress (e.g. its NodePort).

### Fundamental semantics

```
No policy -> Default ALLOW (everything permitted)
At least one policy selects an endpoint -> Default DENY (everything blocked
except what is explicitly allowed)
```

This is the most critical point: **as soon as a CNP is applied to a pod, all
traffic not explicitly permitted is silently dropped** (no RST, no ICMP
unreachable — the sender sees only a timeout). This is why a CNP that is
intended only for observability must still cover every legitimate source of
traffic to the selected pods.

### Basic structure

```yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: example
  namespace: online-boutique
spec:
  # Which pods this policy applies to
  endpointSelector:
    matchLabels:
      app: frontend

  # Permitted INGRESS traffic
  ingress:
  - fromEndpoints:        # permitted sources (other pods)
    - matchLabels:
        app: checkoutservice
    toPorts:              # permitted ports/protocols
    - ports:
      - port: "8080"
        protocol: TCP

  # Permitted EGRESS traffic
  egress:
  - toEndpoints:
    - {}                  # all pods in the cluster (namespace-scoped, see below)
```

### Logical operators in rules

Within a single `ingress` rule, **different source fields are OR'd together**
ONLY when they are of the same type. Across different types (`fromEndpoints` +
`fromEntities`) inside the same rule block, the behavior is **AND** in Cilium.

**WRONG** (AND — impossible condition for external traffic):
```yaml
ingress:
- fromEndpoints:      # must be a pod
  - {}
  fromEntities:       # AND must also be world/host
  - world
  - host
```

**CORRECT** (OR — two separate rules):
```yaml
ingress:
- fromEndpoints:      # rule 1: traffic from pods
  - {}
  toPorts: [...]
- fromEntities:       # rule 2: external traffic
  - world
  - host
  toPorts: [...]
```

---

## 3. Identities and Entities in Cilium

Cilium does not use IPs to identify traffic sources (IPs change with pod
restarts). It uses **identities** instead, derived from Kubernetes labels.

### Special (non-pod) entities

| Entity | Represents |
|---|---|
| `world` | All traffic from outside the cluster (internet, external LAN) |
| `host` | The Kubernetes node itself (the node's network namespace) |
| `cluster` | All endpoints inside the cluster |
| `remote-node` | Nodes (and pods) of remote clusters connected via ClusterMesh |
| `ingress` | The Cilium Ingress Envoy proxy (`reserved:ingress`) |
| `init` | Pods being initialised (no identity yet) |

### How Cilium classifies external traffic in DMOS

External clients (the k6 load generator) hit the Cilium Ingress NodePort 30080.
Cilium Ingress is implemented by the same Envoy that runs in-process inside
`cilium-agent`. The Envoy listener accepts the connection and, when it forwards
the request to the `frontend` pod, the source identity from the policy
engine's perspective is `reserved:ingress` (entity `ingress`), not `world`.
This is why the active CNPs admit `fromEntities: ingress` for the
external-traffic ingress path.

---

## 4. L7 Rules and the Envoy proxy

### What L7 rules do

L7 rules let Cilium inspect the contents of HTTP/gRPC requests, not just IPs
and ports. They are useful for:
- Visibility in **Hubble** (HTTP tracing: methods, URLs, status codes)
- Policies based on HTTP path (`/api/v1/*`)
- Per-endpoint rate limiting

```yaml
toPorts:
- ports:
  - port: "8080"
    protocol: TCP
  rules:
    http:           # <- L7 rule: enables HTTP inspection
    - {}            # {} = allow any HTTP (no filter on path/method)
```

In DMOS, every L7 rule is the empty match `rules.http: [{}]` — a single empty
predicate that accepts any HTTP request. The point is not to filter, but to
force the connection through Envoy so that Hubble can observe and emit L7
metrics.

### The Envoy proxy in Cilium

When an L7 rule is active on a port, Cilium inserts an **Envoy proxy** (running
in-process inside `cilium-agent`) into the traffic path:

```
Pod A --> Envoy (Cilium L7 proxy) --> Pod B (port 8080)
```

Envoy decodes the HTTP, applies the L7 rules, collects metrics, and then
forwards the request to the destination pod.

### Critical behaviour: L7 contaminates all traffic on the port

If **any** ingress rule for a given port uses `rules: http`, Cilium will route
**all traffic on that port** through Envoy — regardless of which rule matched.
This is why the active CNPs in DMOS are written with care to ensure that every
admitted source on the port is one that Envoy can handle correctly (intra-cluster
traffic and the `reserved:ingress` proxy itself).

> **Historical note**: in the NGINX Ingress era, this same property combined
> with NodePort `world` traffic to produce 100% timeouts during Locust load
> tests. With Cilium Ingress, external traffic is intercepted by Envoy *before*
> the policy engine evaluates it, so the failure mode no longer applies. See
> [`_archive/nginx-ingress-hubble-l7.md`](_archive/nginx-ingress-hubble-l7.md)
> for the full archived narrative.

---

## 5. Active CNPs in DMOS

DMOS ships two CNPs in `deployments/`. Both exist purely to enable Hubble L7
observability on internal hops; neither is used to restrict traffic. They are
applied to all six clusters.

### `cnp-frontend-l7.yaml` — frontend pod

**Purpose**: enable `hubble_http_requests_total` and
`hubble_http_request_duration_seconds_bucket` for traffic destined to the
`frontend` pod. DMOS reads the requests sent by the ingress to the frontend
(`reporter="client"`, `traffic_direction="egress"`) as the demand entering
the cluster, and the duration histogram for Φ_lat. (The comments inside the
manifest mention a `source="reserved:ingress"` label: Hubble exposes no such
label; external requests appear with `source_workload=""`.)

**Applied to**: pods with label `app=frontend` in namespace `online-boutique`.

```yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: l7-visibility-frontend
  namespace: online-boutique
spec:
  endpointSelector:
    matchLabels:
      app: frontend

  # No egress section -> default allow all egress.
  # Reason: toEntities:cluster only covers pod IPs (Cilium-managed identities),
  # NOT Kubernetes Service ClusterIPs (10.43.0.0/16). Restricting egress would
  # break DNS (10.43.0.10:53) and all gRPC backend connections (which use
  # ClusterIP). Removing egress is the correct choice for k3s clusters whose
  # pod CIDR and service CIDR differ across clusters.

  ingress:
  # Rule 1: traffic from the Cilium Ingress Envoy proxy (reserved:ingress).
  #   - Every load-generator request -> NodePort 30080 -> Cilium Envoy -> frontend:8080
  #   - With rules: http, Hubble records the hop at L7
  #   - DMOS reads its client side as the demand entering the cluster
  - fromEntities:
    - ingress
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - {}

  # Rule 2: intra-cluster and cross-cluster traffic.
  #   - Internal microservices, DMOS health checks, kubelet probes.
  #   - With rules: http, enables Hubble L7 for Φ_lat (p95 latency).
  #   - remote-node covers the cross-cluster frontend pods reached via
  #     ClusterMesh Global Services.
  - fromEntities:
    - cluster
    - host
    - remote-node
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
      rules:
        http:
        - {}
```

### `cnp-backends-l7.yaml` — backend services

**Purpose**: enable Hubble L7 on the Online Boutique backend microservices.
gRPC is HTTP/2, so Hubble captures it as `httpV2` exactly like the frontend's
HTTP traffic. The per-service rates and latencies are used to measure the
call fan-out (`calls_per_request` in `config/services.yaml`), in the
collector's per-edge p95 (`l7_edge_p95_ms`) and in the namespace-wide p95 of
Φ_lat; DMOS does not size the backends from them (they are co-scaled from the
frontend, see [BACKEND-CO-SCALING.md](BACKEND-CO-SCALING.md)).

**Applied to**: every pod in the namespace **except** `frontend`,
`loadgenerator`, and `redis-cart` (Redis is plain TCP, not HTTP — applying L7
to it does not make sense).

```yaml
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: l7-visibility-backends
  namespace: online-boutique
spec:
  endpointSelector:
    matchExpressions:
      - key: app
        operator: NotIn
        values:
          - frontend
          - loadgenerator
          - redis-cart

  ingress:
    # Inter-service gRPC traffic (frontend -> backend, backend -> backend).
    # With rules: http, Hubble intercepts HTTP/2 and emits hubble_http_requests_total.
    - fromEndpoints:
        - {}   # any pod in the namespace (frontend, other backends)
      toPorts:
        - ports:
            - port: "3550"    # productcatalogservice
              protocol: TCP
            - port: "5050"    # checkoutservice
              protocol: TCP
            - port: "7000"    # currencyservice
              protocol: TCP
            - port: "7070"    # cartservice
              protocol: TCP
            - port: "8080"    # recommendationservice, emailservice
              protocol: TCP
            - port: "9555"    # adservice
              protocol: TCP
            - port: "50051"   # paymentservice, shippingservice
              protocol: TCP
          rules:
            http:
              - {}
    # Cross-cluster traffic (global-service-backends.yaml, affinity=local):
    # used only if a remote cluster has no available local backend and the
    # backends are shared across clusters (see CILIUM-CLUSTERMESH-NOTES.md).
    # Such packets arrive with identity "remote-node".
    - fromEntities:
        - remote-node
      toPorts:
        - ports:
            - port: "3550"
              protocol: TCP
            - port: "5050"
              protocol: TCP
            - port: "7000"
              protocol: TCP
            - port: "7070"
              protocol: TCP
            - port: "8080"
              protocol: TCP
            - port: "9555"
              protocol: TCP
            - port: "50051"
              protocol: TCP
          rules:
            http:
              - {}

  egress:
    # Backends must reply to requests and call each other.
    # toEntities:cluster handles pod IPs that Cilium has not yet resolved
    # (e.g. pods just rescheduled by DMOS whose IP is briefly classified as
    #  "world" -> reply packets would otherwise be dropped).
    - toEndpoints:
        - {}
    - toEntities:
        - cluster
```

### Apply on all clusters

```powershell
# Kubeconfigs live in <KUBECONFIG_DIR> (default: $HOME/dmos-iac), one per cluster.
foreach ($kc in Get-ChildItem "$env:USERPROFILE\dmos-iac\kubeconfig-ms-cluster-*") {
    kubectl --kubeconfig $kc apply -f deployments/cnp-frontend-l7.yaml
    kubectl --kubeconfig $kc apply -f deployments/cnp-backends-l7.yaml
    kubectl --kubeconfig $kc rollout restart deployment/frontend -n online-boutique
}
Start-Sleep 30
```

### Historical: NGINX Ingress + frontend CNP

The previous DMOS deployment used NGINX Ingress as the entry point and required
a different `l7-visibility-frontend` CNP shape (with `fromEntities: cluster, host`
to admit NGINX). That setup, including the diagnosis of the 100% Locust timeout
issue and the three CNP bugs that came out of it, is archived at
[`_archive/nginx-ingress-hubble-l7.md`](_archive/nginx-ingress-hubble-l7.md).

---

## 6. Cilium with NodePort traffic

### How the packet path works (kube-proxy replacement mode)

```
Client (test machine)               Node (ms-cluster-01)
<CLIENT_IP>:XXXXX  -------------->  <CLUSTER1_IP>:30080
                                         |
                                    [Cilium eBPF - XDP/tc hook]
                                         |  Cilium Ingress listener
                                         v
                                    Envoy (in-process in cilium-agent)
                                         |  HTTP routing
                                         v
                                    frontend pod 10.42.x.y:8080
                                         |
                                    [Cilium policy engine]
                                    src identity: reserved:ingress
                                    dst port: 8080
                                    -> evaluates ingress rules
```

### Critical points

1. **Policy is evaluated AFTER the Envoy hop** -> the relevant port is 8080
   (not 30080).
2. **The source identity is `reserved:ingress`** for traffic that came in
   through Cilium Ingress -> the frontend CNP admits `fromEntities: ingress`
   for that path.
3. **`world` is essentially never seen on port 8080** in normal operation
   because external traffic always traverses the Ingress listener first. Direct
   pod-IP / NodePort access to the frontend is not part of the supported path.

### `world`, `host`, `ingress`, `remote-node`

| Entity | When to use it |
|---|---|
| `world` | Traffic from IPs outside the cluster (internet, external LAN). Not used by the active DMOS CNPs because external traffic enters via Cilium Ingress. |
| `host` | Traffic originating from the node itself (kubelet probes, NodePort with SNAT, health checks) |
| `ingress` | The Cilium Ingress Envoy proxy (`reserved:ingress`). Required so external traffic can reach the frontend pod under the L7 rule. |
| `remote-node` | Nodes (and pods reachable through them) of remote clusters connected via ClusterMesh. Required for Global Services failover. |

---

## 7. Testbed hygiene that affects the policies

- **Pod resources.** Online Boutique's default memory limits caused
  OOMKills under load (currencyservice, paymentservice, cartservice,
  recommendationservice). The limits in use are set by
  `experiments/apply_resource_hardening.sh` (and the drain settings by
  `experiments/apply_pod_lifecycle.sh`); `config/services.yaml` documents them.
- **Online Boutique `loadgenerator`.** It generates internal traffic to the
  frontend (not through the Ingress), which mixes with the experiment's load
  in the served rates and in the latency DMOS observes. It must be at 0
  replicas during the runs: the harness's reset scales it to 0 and every
  cell refuses to start otherwise (state in each run's
  `env_state_pre.json`).
- **After applying or changing a policy**, restart the selected Deployments
  (`kubectl rollout restart deployment/<svc> -n online-boutique`) so that
  existing connections go through the proxy.

## 8. Pre-run checks

The automated pre-run checks are in `experiments/preflight_runs.sh` (Ingress
reachability, pods, netem, ClusterMesh, Hubble) and in the run harness's
health gate ([../experiments/RUNBOOK.md](../experiments/RUNBOOK.md)). The
policy-specific checks by hand:

```bash
KC=<kubeconfig>
# both policies VALID=True
kubectl --kubeconfig $KC get cnp -n online-boutique
# the frontend policy has the L7 rule: {"http":[{}]}
kubectl --kubeconfig $KC get cnp l7-visibility-frontend -n online-boutique -o jsonpath='{.spec.ingress[0].toPorts[0].rules}'
# the Ingress answers 200
curl -s -o /dev/null -w '%{http_code}\n' http://<cluster_ip>:30080/
# Hubble shows HTTP events
kubectl --kubeconfig $KC -n kube-system exec ds/cilium -- hubble observe --type l7 --namespace online-boutique --last 5
```

---

## 9. Useful reference commands

`$KC` is the kubeconfig of one cluster (e.g. `$HOME\dmos-iac\kubeconfig-ms-cluster-01`).

### CiliumNetworkPolicy management

```powershell
# List all CNPs in a namespace
kubectl --kubeconfig $KC get ciliumnetworkpolicy -n online-boutique

# View the contents of a CNP
kubectl --kubeconfig $KC get ciliumnetworkpolicy l7-visibility-frontend `
  -n online-boutique -o yaml

# View only the ingress section (formatted)
kubectl --kubeconfig $KC get ciliumnetworkpolicy l7-visibility-frontend `
  -n online-boutique -o jsonpath='{.spec.ingress}' | python -m json.tool

# Apply a CNP from a PowerShell string
$yaml = @' ... '@
$yaml | kubectl --kubeconfig $KC apply -f -

# Apply on every cluster
foreach ($kc in Get-ChildItem "$env:USERPROFILE\dmos-iac\kubeconfig-ms-cluster-*") { $yaml | kubectl --kubeconfig $kc apply -f - }

# Delete a CNP (warning: returns the pod to default ALLOW)
kubectl --kubeconfig $KC delete ciliumnetworkpolicy l7-visibility-frontend -n online-boutique
```

### Connectivity diagnostics

```powershell
# Test the Cilium Ingress NodePort from outside
curl -v --max-time 5 http://<CLUSTER1_IP>:30080/

# List every NodePort
kubectl --kubeconfig $KC get svc -A --field-selector spec.type=NodePort

# Verify a service's endpoints
kubectl --kubeconfig $KC get endpoints frontend -n online-boutique

# Test from inside the cluster (bypasses host firewall, NOT CNP)
kubectl --kubeconfig $KC run nettest --image=nicolaka/netshoot --rm -it `
  --restart=Never -n online-boutique -- curl -v http://10.43.195.100:80/

# Logs from the previous pod (--previous = last crash)
kubectl --kubeconfig $KC logs -n online-boutique <pod-name> --previous --tail=50

# Detailed pod state (look for OOMKill)
kubectl --kubeconfig $KC describe pod -n online-boutique <pod-name> `
  | Select-String -Pattern "State|Reason|Exit|OOM|Limit" -Context 2
```

### Replicas and resources

```powershell
# Scale a deployment
kubectl --kubeconfig $KC scale deployment frontend -n online-boutique --replicas=2

# Change memory limit
kubectl --kubeconfig $KC patch deployment currencyservice -n online-boutique `
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"server","resources":
  {"limits":{"memory":"256Mi"},"requests":{"memory":"128Mi"}}}]}}}}'

# Stop / restart the loadgenerator
kubectl --kubeconfig $KC scale deployment loadgenerator -n online-boutique --replicas=0
kubectl --kubeconfig $KC scale deployment loadgenerator -n online-boutique --replicas=1
```

---

## Glossary

| Term | Definition |
|---|---|
| **CNP** | CiliumNetworkPolicy — Cilium's custom Kubernetes resource |
| **eBPF** | Extended Berkeley Packet Filter — kernel technology used by Cilium for high-performance packet processing |
| **Envoy** | Open-source L7 proxy (used by Istio, Cilium) for HTTP/gRPC inspection and routing. In Cilium it runs in-process inside `cilium-agent`. |
| **Cilium Ingress** | Cilium's built-in Kubernetes Ingress implementation. Uses the in-process Envoy as the data plane; replaces external Ingress controllers such as NGINX. |
| **L4** | Layer 4 (Transport) — filtering by IP and port |
| **L7** | Layer 7 (Application) — filtering by HTTP content (path, method, headers) |
| **NodePort** | Kubernetes Service type that exposes a port on every cluster node |
| **ClusterIP** | Kubernetes Service type reachable only from inside the cluster |
| **DNAT** | Destination NAT — modifies a packet's destination IP/port |
| **SNAT** | Source NAT — modifies a packet's source IP |
| **OOMKilled** | Out Of Memory Killed — the Linux kernel terminates a process that exceeds its memory limit |
| **ClusterMesh** | Cilium feature for connecting multiple Kubernetes clusters with shared service discovery and policy |
| **Hubble** | Cilium's observability tool for visualising network traffic in real time |
| **default deny** | Cilium behaviour: if a policy selects a pod, all traffic not explicitly permitted is blocked |
| **reserved:ingress** | Special identity assigned by Cilium to traffic forwarded by the Cilium Ingress Envoy proxy |
