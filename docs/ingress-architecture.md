# Ingress architecture

Every cluster exposes the Online Boutique frontend through the **Cilium
Ingress controller** (Envoy embedded in the Cilium agent) on NodePort
**30080**. This page describes how it is configured, how its traffic becomes
the Hubble metrics DMOS reads, and why it replaced NGINX.

## 1. Configuration

**Controller** — enabled by `scripts/setup-dmos-on-clusters.ps1` with a Helm
upgrade of Cilium:

```yaml
ingressController:
  enabled: true
  loadbalancerMode: shared        # one Envoy listener shared by all Ingresses
  service:
    type: NodePort
    insecureNodePort: 30080
```

(The Cilium values rendered by `scripts/render-runtime-files.sh` into
`.cache/cilium-<cluster>-values.yaml`, used by the `bootstrap-assisted.sh`
installation path, cover ClusterMesh and the Hubble metrics but not the
Ingress controller: enable it with the block above.)

**Ingress resource** — `deployments/ingress-frontend-cilium.yaml`:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: frontend-ingress-cilium
  namespace: online-boutique
spec:
  ingressClassName: cilium
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service: {name: frontend, port: {number: 80}}
```

**Request path:**

```
k6 ──► <node_ip>:30080 ──► Envoy (Cilium Ingress, reserved:ingress identity)
                              │  load-balances over the frontend Service endpoints
                              ▼
                         frontend pod :8080 ──gRPC──► backends
```

## 2. L7 visibility

Hubble produces HTTP metrics for a flow only when the flow goes through an
L7 policy. `deployments/cnp-frontend-l7.yaml` (`l7-visibility-frontend`)
selects the frontend pods and allows port 8080, with `rules: http: [{}]`
(match any request), from:

- the `ingress` entity — the Cilium Ingress Envoy;
- the `cluster`, `host` and `remote-node` entities — pods of the cluster,
  the kubelet (probes), and nodes of the other clusters (ClusterMesh).

Anything else (e.g. the frontend's own NodePort reached from outside) is
dropped. `deployments/cnp-backends-l7.yaml` does the same for the gRPC ports
of the backends, so backend flows also appear in the Hubble metrics.

The Hubble metric `httpV2` must carry the labels DMOS filters on
(`labelsContext=source_namespace,source_workload,destination_namespace,destination_workload,traffic_direction`);
the Prometheus of each cluster scrapes it every 15 s through
`deployments/hubble-servicemonitor-monitoring.yaml`. Troubleshooting:
[cilium-hubble-l7-visibility.md](cilium-hubble-l7-visibility.md).

## 3. What DMOS reads

| Question | Filter on `hubble_http_requests_total` / `..._duration_seconds_bucket` |
|---|---|
| Demand entering the cluster (λⁱⁿ) | `destination_workload="frontend", reporter="client", traffic_direction="egress"`; fallback `reporter="server", traffic_direction="ingress", source_workload=""` |
| Requests served by the frontend pods | `destination_workload="frontend"` |
| p95 latency (Φ_lat) | all flows of `destination_namespace="online-boutique"` |

Details and the reasons for the client-side demand count:
[traffic-metrics-architecture.md](traffic-metrics-architecture.md) and
[dmos-architecture.md §5](dmos-architecture.md#5-observed-state-metrics-and-queries).

## 4. Multi-ingress load

The load generator sends to the six ingresses at once, choosing the cluster
of each request with per-cluster weights
([METRICS-AND-OUTPUTS.md §1](METRICS-AND-OUTPUTS.md#1-the-workload-and-its-phases)).
Which frontend serves a request entering cluster *i* depends on the
ClusterMesh configuration of the frontend Service; in the configuration
documented for the campaigns it is a frontend of cluster *i*
([CILIUM-CLUSTERMESH-NOTES.md](CILIUM-CLUSTERMESH-NOTES.md)). DMOS acts by
changing how many frontends each cluster runs.

## 5. Why not NGINX

An earlier iteration used the NGINX Ingress controller with a policy for L7
visibility ([_archive/nginx-ingress-hubble-l7.md](_archive/nginx-ingress-hubble-l7.md)).
The Cilium Ingress removes a component and a hop (Envoy runs in the Cilium
agent), works natively with ClusterMesh, and makes external requests
distinguishable: they reach the frontend from the `reserved:ingress`
identity with an empty `source_workload`, whereas with NGINX every flow came
from the NGINX pod.

## 6. References

- [`deployments/ingress-frontend-cilium.yaml`](../deployments/ingress-frontend-cilium.yaml),
  [`deployments/cnp-frontend-l7.yaml`](../deployments/cnp-frontend-l7.yaml),
  [`deployments/cnp-backends-l7.yaml`](../deployments/cnp-backends-l7.yaml)
- [`scripts/setup-dmos-on-clusters.ps1`](../scripts/setup-dmos-on-clusters.ps1) — Cilium Helm values
- [`src/metrics/prometheus_client.py`](../src/metrics/prometheus_client.py) — `get_ingress_rate()`
