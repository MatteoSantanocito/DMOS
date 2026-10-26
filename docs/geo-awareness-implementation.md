# Network proximity (Φ_net) and the emulated geography

How DMOS gives the clusters a network position, how that position enters
the score, and how the testbed emulates geography with `tc netem`. The full
score and the other components are in
[dmos-architecture.md §7](dmos-architecture.md#7-level-1--placement).

## 1. What Φ_net measures

The testbed's VMs share one LAN (sub-millisecond RTT). Each cluster is given
an emulated one-way delay on its worker's interface, and every cluster
measures its RTT to all the others with `ping_exporter`. DMOS turns the mean
of those RTTs into the proximity score:

```
RTTᵢ  = mean over peers j of ping_rtt_mean_seconds{target = IPⱼ}   (ms)
Φ_net = exp(−ρ · min(RTTᵢ, RTT_max) / RTT_max)          ρ = 3.0, RTT_max = 500 ms
```

`Φ_net` is therefore a measure of how **central** a cluster is within the
federation (how far it is, on average, from the other clusters), used as a
proxy for its distance from users and dependencies. It is not the distance
between a cluster and the users that enter it: the load generator is on the
same LAN for every cluster, and the users' delay to "their" cluster is the
netem delay of that cluster's responses.

When no ping data is available `RTTᵢ` = 5 ms (Φ_net ≈ 0.97 for everyone).

## 2. Expected values on the 6-cluster fleet

netem delays every packet that leaves the worker, so the RTT between *i* and
*j* is about `delayᵢ + delayⱼ` (plus jitter):

| Cluster | delay (ms) | mean RTT to peers (ms) | Φ_net, ρ = 3 (code) | Φ_net, ρ = 2 (paper Tab. I) |
|---|---:|---:|---:|---:|
| ms-cluster-01 (DE) | 0 | 84 | 0.604 | 0.715 |
| ms-cluster-02 (FR) | 30 | 108 | 0.523 | 0.649 |
| ms-cluster-03 (PL) | 50 | 124 | 0.475 | 0.609 |
| ms-cluster-04 (US-VA) | 100 | 164 | 0.374 | 0.519 |
| ms-cluster-05 (SG) | 200 | 244 | 0.231 | 0.377 |
| ms-cluster-06 (NO) | 40 | 116 | 0.499 | 0.629 |

Measured values are in each run's `dmos_live.jsonl` (`ping_rtt_ms` per
peer) and in `dmos.log` ("Network RTT (<cluster>): … ms"). The ratio between
the most and the least central cluster is 2.6 with ρ = 3 and 1.9 with ρ = 2;
`network_parameters.rho` in `weights.yaml` decides which applies (a score
model can override it with `parameters: {rho: …}`).

## 3. How proximity combines with demand

Proximity alone would pull replicas toward the central clusters regardless
of where the users are. How demand corrects this depends on the score model
(`demand_mode`, see [architecture §7.4](dmos-architecture.md#74-score-models-the-role-of-demand)):

| Mode | Proximity term | Effect |
|---|---|---|
| `gate` (NCA design) | ω_net · Φ_net inside the sum, the sum multiplied by G(d) | proximity ranks the clusters that have demand; a central cluster without demand is suppressed by the gate |
| `locality` | ω_net · (d · Φ_net) | only the proximity term is demand-weighted; latency, capacity and carbon still attract replicas to clusters without demand |
| `additive` | ω_net · Φ_net + ω_dem · d | demand compensates proximity linearly; a central cluster without demand keeps its proximity score |

The multiplicative forms avoid the "near but empty" pathology of a sum. With
two separate terms, a cluster with little demand but high proximity scores
like one with high demand but low proximity:

| Cluster | d | Φ_net | (d + Φ_net)/2 | d · Φ_net |
|---|---:|---:|---:|---:|
| no demand, near | 0.20 | 0.90 | 0.55 | 0.18 |
| high demand, far | 0.95 | 0.20 | 0.575 | 0.19 |
| high demand, near | 0.95 | 0.90 | 0.925 | 0.855 |

The legacy profiles (`flash_crowd_aware` and others) use the `locality`
form under the key `omega_geo_load`.

*History.* Earlier versions multiplied Φ_net by a decreasing load score
(`Φ_load · Φ_net`, "unloaded and close"). Under a flash crowd the saturated
cluster's Φ_load collapsed and replicas moved away from it; the load term was
then inverted into Φ_pressure (legacy path only) and the demand share took
its place in the product.

## 4. Infrastructure

### 4.1 netem

`scripts/setup-netem-and-ping.sh` reads `.cache/test-env.json` and, for each
cluster with `netem.delay_ms > 0`, runs on the worker over SSH:

```bash
sudo tc qdisc replace dev <interface> root netem delay <delay>ms <jitter>ms distribution normal
```

| Command | Effect |
|---|---|
| `bash scripts/setup-netem-and-ping.sh` | netem and ping_exporter |
| `bash scripts/setup-netem-and-ping.sh netem-only` | netem only (needed after every VM reboot: the qdisc is not persistent) |
| `bash scripts/setup-netem-and-ping.sh verify` | check only |
| `bash scripts/setup-netem-and-ping.sh remove-netem` | back to plain LAN |

The run harness checks the qdisc of every worker before and after each run
and refuses to start a run without it
([RUN-PROVENANCE.md](RUN-PROVENANCE.md)).

### 4.2 ping_exporter

Installed by the same script with Helm (chart `czerwonk/ping_exporter`), one
release per cluster in namespace `observability`, with the values rendered
in `.cache/ping-exporter-<cluster>-values.yaml`: the targets are the IPs of
the other clusters, one ICMP probe per second, and a ServiceMonitor
(`release: prometheus-stack`) so that the cluster's Prometheus scrapes
`ping_rtt_mean_seconds`.

### 4.3 In DMOS

| Step | Code |
|---|---|
| peer list of each cluster (all the other clusters' `ip`) | `DMOSScheduler.__init__` (`_peer_ips`) |
| query `avg(ping_rtt_mean_seconds{target=~"<peers>"}) * 1000` | `PrometheusClient.get_network_rtt_ms` |
| `ClusterMetrics.network_rtt_ms` | `DMOSScheduler._collect_cluster_metrics` |
| Φ_net | `ScoreFunctions.compute_network_score` |
| log line `Score <cluster>: … locality=…[Φ_dem=…*Φ_net=…|RTT=…ms]` | `DMOSScheduler._compute_cluster_score` |

## 5. Degraded cases

| Situation | Result |
|---|---|
| no netem | RTT ≈ 1 ms → Φ_net ≈ 0.99 everywhere: proximity does not differentiate |
| no ping_exporter data | RTT = 5 ms → Φ_net ≈ 0.97 everywhere |
| one peer unreachable | the mean uses the targets that report |

In all cases DMOS keeps scheduling on the other components; the provenance
check on netem prevents an unintended run without emulated geography.

## 6. Design notes

- **Mean over peers vs per-source RTT.** A per-request or demand-weighted
  proximity (distance from where the demand enters) would be closer to the
  intent; the mean keeps the score a per-cluster quantity.
- **Choice of ρ.** ρ = 2 gives a 1.9× spread between the most and the least
  central cluster, ρ = 3 gives 2.6×. The code uses 3.0, the NCA paper reports
  2.0 (to be aligned).
