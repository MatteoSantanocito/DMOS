# Hubble L7 visibility — setup and troubleshooting

DMOS reads HTTP-level metrics produced by Hubble (`hubble_http_requests_total`,
`hubble_http_request_duration_seconds_bucket`). This page explains what has
to be in place for those metrics to exist, how to check it, and the pitfalls
met along the way. How DMOS uses the metrics:
[traffic-metrics-architecture.md](traffic-metrics-architecture.md) and
[dmos-architecture.md §5](dmos-architecture.md#5-observed-state-metrics-and-queries).

## 1. The three requirements

| # | Requirement | Provided by |
|---|---|---|
| 1 | Hubble exports the `httpV2` metric with the context labels DMOS filters on | Cilium Helm values (`hubble.metrics.enabled`) |
| 2 | The HTTP flows go through Envoy, so that Hubble sees them at L7 | CiliumNetworkPolicies with `rules: http` on the destination pods |
| 3 | Each cluster's Prometheus scrapes the Hubble metrics often enough | ServiceMonitor, 15 s |

### 1.1 Hubble metrics

On the proxmox-k3s testbed `scripts/setup-dmos-on-clusters.ps1` upgrades
Cilium with:

```yaml
hubble:
  metrics:
    enableOpenMetrics: true
    enabled:
      - 'httpV2:exemplars=true;labelsContext=source_namespace,source_workload,destination_namespace,destination_workload,traffic_direction'
      - dns
      - drop
      - tcp
      - flow
      - icmp
    serviceMonitor:
      enabled: true
      labels: {release: prometheus-stack}
```

The values rendered for the `bootstrap-assisted.sh` path
(`.cache/cilium-<cluster>-values.yaml`) contain the same list.
`config/cilium-hubble-values.yaml` is an older stand-alone version of these
values (it also adds `destination_port`); no script applies it.

Resulting labels on `hubble_http_requests_total`: the context labels
`source_namespace`, `source_workload`, `destination_namespace`,
`destination_workload`, `traffic_direction`, plus the labels of the `httpV2`
metric itself, among which `reporter` (`client`/`server`), the method and
the status code. The duration histogram carries the same context labels plus
`le`.

### 1.2 L7 policies

Cilium sends a flow through its Envoy proxy (and Hubble records it at L7)
only when the destination pod has a policy with an L7 rule on that port. An
empty rule `http: [{}]` allows every request while enabling the inspection.

- **`deployments/cnp-frontend-l7.yaml`** (`l7-visibility-frontend`) — frontend
  pods, port 8080, two ingress rules with `rules: http: [{}]`: one from the
  `ingress` entity (the Cilium Ingress Envoy), one from the `cluster`, `host`
  and `remote-node` entities. Any other source is dropped.
- **`deployments/cnp-backends-l7.yaml`** (`l7-visibility-backends`) — all pods
  except `frontend`, `loadgenerator` and `redis-cart`, on the gRPC ports
  (3550, 5050, 7000, 7070, 8080, 9555, 50051), from any pod of the namespace
  and from `remote-node`; egress allowed to any pod and to the `cluster`
  entity. Redis is excluded because it does not speak HTTP.

Both are applied by `setup-dmos-on-clusters.ps1`. A policy with an L7 rule
turns on the proxy for **all** traffic to that port: L7 for some sources and
plain TCP for others on the same port is not possible.

### 1.3 Scrape

`deployments/hubble-servicemonitor-monitoring.yaml` (ServiceMonitor
`hubble-metrics` in `observability`, label `release: prometheus-stack`)
scrapes port `hubble-metrics` (9965) of the Cilium agents every **15 s**. The
Cilium chart can also create its own ServiceMonitor (`serviceMonitor.enabled`);
check that each Hubble series is ingested once (§3).

## 2. Distinguishing external from internal traffic

Requests from the Cilium Ingress come from the `reserved:ingress` identity and
have no source workload, so `source_workload=""`; requests between pods carry
the caller's workload. DMOS counts the demand on the client side
(`reporter="client"`, `traffic_direction="egress"`, destination `frontend`),
with the server-side `source_workload=""` count as fallback
([traffic-metrics-architecture.md](traffic-metrics-architecture.md)).

## 3. Checks

```bash
KC=<kubeconfig of the cluster>

# policies present and valid
kubectl --kubeconfig $KC get cnp -n online-boutique
#   l7-visibility-frontend   … VALID True
#   l7-visibility-backends   … VALID True

# L7 events in Hubble
kubectl --kubeconfig $KC -n kube-system exec ds/cilium -- \
  hubble observe --type l7 --namespace online-boutique --last 20

# proxy enabled on the frontend endpoints
kubectl --kubeconfig $KC -n kube-system exec ds/cilium -- cilium-dbg endpoint list | grep frontend
```

On the cluster's Prometheus (`http://<ip>:32090`), under load:

```promql
sum(rate(hubble_http_requests_total{destination_workload="frontend",reporter="client",traffic_direction="egress"}[1m]))   # demand entering
sum(rate(hubble_http_requests_total{destination_workload="frontend"}[1m]))                                                # served by the frontend
count by (job) (hubble_http_requests_total)                                                                               # one job per series expected
```

The first should match the rate k6 sends to that cluster. `scripts/validate-setup.sh`
and `experiments/preflight_runs.sh` include presence checks.

## 4. Pitfalls and fixes

| Symptom | Cause | Fix |
|---|---|---|
| no `hubble_http_*` series at all | `httpV2` not in `hubble.metrics.enabled`, or not scraped | re-run `setup-dmos-on-clusters.ps1` (Cilium upgrade); check the ServiceMonitor label `release` matches the Prometheus Operator |
| series exist but miss `traffic_direction`/`source_workload` | `labelsContext` without those labels | fix the Helm values and restart the agents |
| frontend counter flat | L7 policy not applied on the frontend | `kubectl apply -f deployments/cnp-frontend-l7.yaml` |
| backend counters flat | backends policy missing | `kubectl apply -f deployments/cnp-backends-l7.yaml` |
| 5xx/timeouts right after applying a policy | existing connections broken when the proxy is inserted | `kubectl rollout restart deployment/<svc> -n online-boutique` |
| `rate()` intermittently 0 | scrape interval too long for `[1m]` | keep the 15 s scrape (or widen the windows) |
| traffic blocked after a policy change | `fromEndpoints: {}` matches only the policy's namespace; `toEntities: cluster` does not cover Service ClusterIPs | allow the needed entities explicitly (see the archived NGINX notes) |

`scripts/enable_hubble_http.sh` belongs to the three-cluster testbed: it
patches the `cilium-config` ConfigMap directly and restarts the agents. On a
Helm-managed Cilium use the Helm values above instead.

## 5. History

The NGINX-ingress setup and its debugging (namespace-scoped endpoint
selectors, ClusterIP identities, proxy activation per port) are in
[_archive/nginx-ingress-hubble-l7.md](_archive/nginx-ingress-hubble-l7.md).
