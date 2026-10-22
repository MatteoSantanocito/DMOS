# Measuring demand: the ingress rate

DMOS sizes and places the frontend from one signal: the rate of requests
that **enter** each cluster through its Cilium Ingress, λᵢⁱⁿ. This page
explains how that rate is measured, why it is measured on the client side,
what it is used for, and how to check it. The rest of the observed state is
in [dmos-architecture.md §5](dmos-architecture.md#5-observed-state-metrics-and-queries).

## 1. Who decides where traffic goes

DMOS does not route requests. Which cluster a user enters is decided outside
DMOS — in the experiments by the k6 ingress weights (one weight per
cluster and per phase, [METRICS-AND-OUTPUTS.md §1](METRICS-AND-OUTPUTS.md#1-the-workload-and-its-phases)),
in production by GeoDNS, anycast or a global load balancer. DMOS observes
that split and decides how many replicas each cluster runs. Cilium then
load-balances the entering requests over the frontend endpoints of the
global service: with `affinity: none`, as in the paper, over the ready
frontend pods of **all** clusters, uniformly per endpoint
([architecture §4](dmos-architecture.md#4-data-plane)). This is why demand is
measured where the request enters (the ingress → frontend hop, client side)
and not where it is served.

## 2. The signal

The Cilium Ingress is an Envoy proxy embedded in the Cilium agent. With the
L7 visibility policy on the frontend (`deployments/cnp-frontend-l7.yaml`,
`rules: http: [{}]`), Hubble records every HTTP request from the ingress to
the frontend in `hubble_http_requests_total`, with the labels enabled in the
Hubble `httpV2` metric (`source_workload`, `destination_workload`,
`traffic_direction`, namespaces, `reporter`).

`PrometheusClient.get_ingress_rate()` queries each cluster's Prometheus:

```promql
# primary: client-reported, recorded on the cluster the request entered
sum(rate(hubble_http_requests_total{
    destination_workload="frontend", destination_namespace="online-boutique",
    reporter="client", traffic_direction="egress"}[1m]))

# fallback (only if the primary returns 0): server-side, external clients only
sum(rate(hubble_http_requests_total{
    destination_workload="frontend", destination_namespace="online-boutique",
    reporter="server", traffic_direction="ingress", source_workload=""}[1m]))
```

**Why client-side.** If a request entering cluster *i* were served by a
frontend pod on cluster *j* (possible with a global service), a server-side
count would attribute it to *j*. More replicas on *j* would then look like
more demand on *j*, which would attract more replicas: a feedback loop. The
client-reported flow is recorded where the request entered, wherever it is
served. The server-side fallback is equivalent only while every request is
served by the cluster it entered.

`source_workload=""` identifies requests from outside the cluster (the
Envoy ingress has no workload identity); requests between pods carry the
calling workload.

The rate uses a `[1m]` window over a 15 s scrape (four samples), so a step
in the offered load is fully visible in λᵢⁱⁿ after about a minute and
partially after 15–30 s.

## 3. What the signal is used for

| Use | How | Code |
|---|---|---|
| Trigger of a cycle | total λ and the per-cluster predictions against 30 / 10 req/s | `periodic_check_thread` |
| Replica budget N | Σᵢ clip(⌈λᵢⁱⁿ / 60⌉, 1, c_maxᵢ) | `schedule_service` |
| Demand floor (when enabled) | ⌈λᵢⁱⁿ / 60⌉ per cluster | `schedule_service` → `allocate` |
| Demand share d | dᵢ = λᵢⁱⁿ / Σⱼ λⱼⁱⁿ (1/K if the total is 0) | `DMOSScheduler.collect_scores` |
| Level 2 | predictor input and PD reference per cluster | `schedule_service` (Step 2b) |
| Cold-phase request rate | `request_rate_current` = λᵢⁱⁿ | `_collect_cluster_metrics` |

The demand share enters the score according to the score model (gate,
additive term or proximity weight, [architecture §7.4](dmos-architecture.md#74-score-models-the-role-of-demand)).
It is the only score input that needs all clusters before any cluster's
score can be computed (its denominator is a sum over clusters), which is why
`collect_scores` queries every ingress first.

Other rates, not used for demand:

| Metric | Measures | Used by |
|---|---|---|
| `hubble_http_requests_total{destination_workload="frontend"}` (all reporters/directions) | requests served by the frontend pods of the cluster, from anywhere | Φ_pressure (legacy path), warm phase |
| collector `frontend_net_bytes_s` / `traffic_pct` | frontend requests from the ingress, server-side | timeline of the served share ([METRICS-AND-OUTPUTS.md §4](METRICS-AND-OUTPUTS.md#4-scheduler-and-cluster-timeline-dmos_livejsonl)) |
| collector `hubble_http_rps` | all L7 requests in the namespace (frontend + gRPC backends) | timeline |

## 4. Checking it

On a cluster's Prometheus (`http://<ip>:32090`), during load:

```promql
sum(rate(hubble_http_requests_total{destination_workload="frontend",reporter="client",traffic_direction="egress"}[1m]))
```

should match the rate k6 sends to that cluster (offered rate × its ingress
weight). In `dmos.log` every cycle prints

```
Ingress (origin) rate (ms-cluster-01): 297.4 req/s [hubble L7 client-reported]
Ingress rates: ms-cluster-01=297.4 req/s, … | total=349.8 req/s
```

and a line `Ingress rate (…) … [FALLBACK server-side …]` when the fallback was
used. If no ingress metric is available at all, the shares become uniform
(`Φ_demand uniforme (1/N per cluster)` at DEBUG level) and the budget falls
back to 30 req/s spread uniformly.

Typical causes of a missing signal: the L7 policy is not applied (no
`rules: http`), the Hubble `httpV2` metric lacks the
`traffic_direction`/`source_workload` labels, or the Hubble ServiceMonitor is
not scraped. See [cilium-hubble-l7-visibility.md](cilium-hubble-l7-visibility.md)
and [troubleshooting.md](troubleshooting.md).

## 5. History

- The first testbed (three clusters) used the frontend NodePort directly and
  counted `source="reserved:ingress"` flows; Envoy was not engaged on that
  path, so L7 metrics were incomplete.
- An NGINX Ingress phase used `nginx_ingress_controller_requests`
  ([_archive/nginx-ingress-hubble-l7.md](_archive/nginx-ingress-hubble-l7.md)).
- With the Cilium Ingress the server-side `source_workload=""` count became
  the demand signal; the client-reported form replaced it as primary to stay
  correct when requests can be served across clusters.
