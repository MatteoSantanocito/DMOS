# DMOS on a Kubephos multicluster

Kubephos builds K3s clusters on Proxmox with Cilium (ClusterMesh between
the clusters, Hubble, Gateway API), kube-prometheus-stack on a dedicated
management node and an application pool. `scripts/setup-kubephos-fleet.sh`
takes the kubeconfig Kubephos gives (one context per cluster) and prepares any
number of clusters for DMOS, checks them and runs a first test.

```bash
bash scripts/setup-kubephos-fleet.sh --kubeconfig ~/Downloads/kubeconfig.yaml --name myfleet
```

Run it from Git Bash in the repository root. Inputs:

| Option | Default | |
|---|---|---|
| `--kubeconfig` | required | kubeconfig downloaded from Kubephos |
| `--name` | kubeconfig file name | fleet name (files, title of `local.yaml`) |
| `--out-dir` | a `dmos-iac/<name>` folder on the desktop of the current user | per-cluster kubeconfigs, `fleet.json`, `local.<name>.yaml`, backups of the previous `local.yaml`, run logs. Outside the repository: the kubeconfigs hold admin credentials |
| `--contexts` | all, kubeconfig order | subset and order of the clusters; the order is the order of the k6 weights, the first cluster is the hot spot of the test runs |
| `--regions` | `DE,FR,PL,US-VA,SG,NO` truncated to N | region of each cluster (carbon zone, k6 weights); required beyond 6 clusters |
| `--baseline R=G` | | carbon baseline (gCO2/kWh) of a region outside the reference list |
| `--peak-rps` | 250 | peak of the test runs |
| `--from STEP` | 1 | restart from a step (e.g. `--from 6` to re-check) |
| `--calibrate` / `--no-calibrate` | measure when `calibration.json` is missing | measure k again / skip the measurement |
| `--k MCORE`, `--u-cpu U` | measured, 0.8 | override k / the target CPU utilisation of the replica cap |
| `--no-hubble-fix` | | do not change Cilium; stop if the Hubble labels are missing |
| `--no-runs` | | stop before the runs |

Nothing else is asked: the rest is read from the clusters.

## What is read from the clusters

| `local.yaml` / setup | Source |
|---|---|
| `ip` | address of the API server in the kubeconfig (the address Kubephos shows) |
| `resources.cpu_cores`, `memory_gb` | sum of the capacity of the nodes `kubephos.dev/pool=app` (VM sizes). DMOS does not read them from the cluster: they feed Φ_cap and L0, so the script writes them |
| `prometheus_port`, ServiceMonitor namespace and `release` | the Prometheus operator CR and its NodePort Service |
| GatewayClass | the Cilium GatewayClass whose `CiliumGatewayClassConfig` exposes a NodePort (`kubephos-cilium`) |
| `ingress_port` | NodePort of the Service Cilium creates for the Gateway (assigned by Kubernetes) |
| `worker_ips` | the app-pool nodes: DMOS measures free CPU and memory there. Without it DMOS would measure the node of `ip`, the control plane (4 GB): in a first test that showed 90 % memory used and excluded every cluster from Level 1 |
| `ingress_demand_match` | `namespace` when `ip` is not an app node (the Gateway Envoy then runs next to no frontend, [traffic-metrics-architecture.md §2](traffic-metrics-architecture.md#2-the-signal)) |
| Cilium version (for the Hubble fix) | the `cilium` Helm release |
| `max_replicas` (entry service) | capacity model below: app-pool size, idle use measured in Prometheus, k from the calibration |

## Steps

1. **discover** (read only) — one kubeconfig per cluster, `fleet.json`. Blocks on:
   nodes not Ready, no app pool, no NodePort GatewayClass, metrics-server not
   Available, duplicated Cilium cluster-id, ClusterMesh not connected to all the
   other clusters or peers not ready, Prometheus namespace/release different
   between clusters.
2. **hubble** — where `httpV2` lacks
   `labelsContext=source_namespace,source_workload,destination_namespace,destination_workload,traffic_direction`:
   `helm upgrade` of the installed chart version with `--reuse-values` (only
   the Hubble metric list changes), restart of the Cilium agents, one cluster
   at a time. Once Kubephos sets these labels by default this step does
   nothing.
3. **workload** — `scripts/setup-dmos-on-clusters.ps1 -IngressMode gateway`:
   Online Boutique on the app pool (nodeSelector), L7 visibility policies,
   global services (frontend `affinity: none`), Hubble ServiceMonitor,
   Gateway + HTTPRoute ([ingress-architecture.md §7](ingress-architecture.md#7-gateway-api-variant)).
4. **local.yaml** — written from what step 1 found and the Gateway NodePorts;
   the previous `config/local.yaml` is saved in `--out-dir`; then
   `scripts/render-runtime-files.sh`.
5. **prepare** — `apply_resource_hardening.sh`, `apply_pod_lifecycle.sh`,
   `clean_reset.sh`.
6. **check** — per cluster: Hubble labels, Deployments available, pods only on
   the app pool, Gateway Programmed and answering 200 at the URL k6 uses,
   Hubble scraped by exactly one scrape pool with all targets up (two pools
   would double every rate), ClusterMesh peers. Then a probe, only if no
   traffic is entering the fleet: ~6 req/s to the first Gateway for 45 s; DMOS
   must measure it client-side on that cluster, and the frontends of the other
   clusters must serve part of it (global service).
7. **calibrate** (~8 min, only when `--out-dir` has no `calibration.json`, or
   with `--calibrate`) — measures k (below), then writes `local.yaml` again
   with the replica cap of every cluster.
8. **runs** (~40 min) — flash crowd with DMOS off, reset, DMOS (`nca`, the legacy demand gate) on,
   reset, comparison: p95 / error rate per phase, max frontend replicas per
   cluster, and for the peak phase the app-node CPU per cluster and the worst
   pod CPU throttling (`kubephos_fleet.py compare RUN... --out-dir DIR`). The script refuses to start while k6 is running.

`python scripts/kubephos_fleet.py {discover,hubble-fix,write-local,calibrate,check,compare}`
runs a single step.

## Replica cap

DMOS needs a ceiling on the replicas of the entry service per cluster: its
budget follows the demand (about ⌈λ/r⌉), and replicas that do not fit the
nodes become Pending pods or CPU contention. On the reference testbed 8
frontends (plus the backends that scale with them) on an 8 vCPU worker gave
20 % errors and a p95 of 6.9 s; 6 did not. The script derives the ceiling from
the capacity of the app pool instead of fixing it by hand:

```
cap_cpu = floor((u·V − C0) / (r·k + c_rep))
cap_mem = floor((um·M − M0) / m)
cap     = max(1, min(cap_cpu, cap_mem))
```

| | |
|---|---|
| V, M | vCPU and GB of the app pool (`kubephos.dev/pool=app`) |
| C0, M0 | CPU and memory used on the app nodes at idle, one replica per service |
| u, um | target utilisation, 0.8 and 0.85: beyond them CPU throttling and the latency knee start |
| r | req/s one entry replica serves at 70 % of its CPU limit (calibrated: limit × 0.7 / CPU per request of the entry pod); without a calibration, `capacity_req_per_sec` of `services.yaml` |
| k | **marginal** app-node CPU per request of the whole path: entry service, the backends it calls, and the Cilium agent and Envoy, which parse every HTTP/gRPC call because of the L7 visibility policies |
| c_rep | idle app-node CPU of one more replica of the entry service and of each co-scaled backend (runtime, probes, Cilium endpoint): a fixed cost per replica, independent of the load |
| m | memory limits of one entry replica plus its share of the co-scaled backends: Σ_b calls_b · r / capacity_b · limit_b (≈ 0.9 GB with Online Boutique, so memory binds only on small pools) |

**Calibration** (`kubephos_fleet.py calibrate`, ~11 min, DMOS stopped, no
traffic): C0 with one replica per service; the entry service and the
co-scaled backends go to 2 replicas and the idle CPU is measured again
(c_rep = the difference); k6 (`k6_flash_crowd.js`, the request mix of the
test runs) sends 25 % and then 50 % of the capacity of those replicas, spread
evenly; k = Δ app-node CPU / Δ requests served between the two loads (the
slope, so fixed costs stay out of it); r from the CPU of the entry pods at the
higher load. The replicas are restored at the end; `calibration.json` keeps
every measurement.

On the six-cluster fleet of the paper (`ms-01-cluster-01` … `ms-01-cluster-06`,
DE, FR, PL, US, SG, NO) the calibration gave **r = 33 req/s** per frontend
replica and the frontend caps **4/3/2/4/2/3**; these are the values of the
campaign's `local.yaml`.

Why the slope and c_rep: the first version measured k at one load against the
idle CPU with one replica per service. It gave 26.5 mcore/req
because the idle cost of the doubled replicas ended up in k, while the
DMOS-off run at 450 req/s showed about 11 mcore/req: the cap came out at 3,
the frontends of every cluster stayed at the cap and were throttled at the
peak (86 %), and DMOS lost to the baseline. The same tests showed that the
fixed cost is real: DMOS with 8 frontends and the co-scaled backends on one
8 vCPU cluster spent 2.4 cores in Cilium alone and collapsed (p95 10 s,
29 % errors).

**Known limit: gRPC connections are pinned.** Kubernetes Services balance
connections, not requests, and the Online Boutique clients keep their gRPC
connections open: a new backend replica receives little traffic from the
frontends already connected (in one test one currencyservice pod of three was
throttled 87–93 %, the others 2–25 %). Request-level balancing needs the Cilium
Envoy L7 load balancer (`loadBalancer.l7.backend: envoy` and
`service.cilium.io/lb-l7: enabled` on the backends), not enabled on Kubephos.

The cap is written in `local.yaml` as `max_replicas: {frontend: N}` per
cluster (and r as `capacity_req_per_sec: {frontend: r}` after a calibration), with the numbers of the formula in the header; it overrides
`services.yaml → max_replicas_per_cluster` for that cluster in DMOS, the HPA
and Karmada baselines and the simulator. The co-scaled backends keep their
`services.yaml` caps.

## Cilium is re-applied by K3s every 10 hours

Cilium is installed by a K3s `HelmChart` (`kube-system/cilium`). The K3s
helm-controller re-applies it every 10 hours from the start of K3s (on the
paper fleet in windows of about 80 min, one cluster every ~15 min; see
`DMOS_REAPPLY_ANCHOR` in [configuration-reference.md §7.2](configuration-reference.md#72-run-harness-run_onesh-run_campaignsh-run_flash_crowdsh)),
with the chart's `valuesContent` merged with the `HelmChartConfig`
`kube-system/cilium`. Everything set outside them with `helm upgrade` is lost:
once this removed the ClusterMesh (the global frontend kept
only its local backend) and the Hubble labels on all six clusters.

* The ClusterMesh must live in the `HelmChartConfig` (done by recent Kubephos
  revisions).
* The Hubble labels (`hubble.metrics.enabled` with the `httpV2:...labelsContext=`
  entry of `kubephos_fleet.py hubble-fix`) should live there too; until then
  the run harness puts them back.
* **Run gate.** With `DMOS_FLEET_DIR=<kubephos_fleet out-dir>` every cell of
  `run_one.sh` runs `kubephos_fleet.py check` before the load (missing Hubble
  labels → `discover` + `hubble-fix`; ClusterMesh not 5/5, a Gateway not 200 or
  pods off the app pool → the cell does not start) and after it (a change
  during the run → `INVALID_DATAPLANE.txt` in the run directory).

## Load generator next to the clusters

k6 must run on a VM in the same network as the clusters
(`DMOS_K6_REMOTE=ubuntu@<K6_VM_IP>`, `DMOS_RAW_OUT=0` to keep the 1 GB raw
event file on the VM). From a remote client behind a VPN the client path adds
~55 ms per request and saturates under load (peak p99 ×5 with unchanged node
CPU, k6 spending up to 2.5 s receiving bodies). The paper fleet uses a
dedicated load-generator VM (4 vCPU, 3 GB, outside the clusters and without
netem: a client in DE).

## Limits of a Kubephos fleet

- **Emulated geography.** `kubephos_fleet.py write-local --netem` +
  `setup-netem-and-ping.sh` shape every node of a cluster but only the traffic
  leaving the cluster (and not the NFS storage of the PersistentVolumes), see
  `scripts/setup-netem-and-ping.sh`.
- **Replica caps of the backends.** Only the entry service gets a cap from the
  capacity model; the backends keep the global caps of `services.yaml`
  (`max_replicas_per_cluster` is keyed by the reference fleet names).
- **Platform components on the app nodes.** Chaos Mesh and Mentat run on every
  node: a small background load to declare.
- **Campaign.** `run_one.sh` / `run_campaign.sh` need the 6-cluster fleet
  of the regimes ([RUNBOOK](../experiments/RUNBOOK.md)); a 6-cluster Kubephos
  fleet with `--regions DE,FR,PL,US-VA,SG,NO` satisfies the count; the tiers of
  the reference testbed are not reproduced (all app pools of the same size get
  the same cap, unless Kubephos builds pools of different sizes).
