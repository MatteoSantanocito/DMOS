# Live flow map

`experiments/live_flow_map.py` shows, while a run is in progress, where the
requests enter the fleet and which cluster serves them, together with the
frontend replicas and the decisions of DMOS.

```bash
python experiments/live_flow_map.py          # then open http://127.0.0.1:8765
python experiments/live_flow_map.py --demo   # synthetic data, to see the page without a fleet
```

## What the page shows

* **Map**: k6 → entry cluster *j* → cluster *i* whose frontend serves the
  request. A moving dot is a sampled request (about 40 per second), coloured
  by its latency; the thickness of an arc is the request rate *j → i*; the
  circle of a cluster grows with its ready frontend replicas, its ring with
  the requests it serves locally.
* **Matrix entered → served** over the last 10 s (req/s and p95). With the
  pooled data plane (affinity `none`) each row is spread over the columns in
  proportion to the replicas: this is the direct measurement of what
  `check_dataplane.py` infers from the CPU per replica.
* **Replicas**: ready and desired frontend replicas per cluster
  (kube-state-metrics) and the target of DMOS (its `/metrics` on port 9090).
* **Demand vs served share**: the demand entering each cluster (the
  Prometheus query DMOS uses) next to the share each cluster serves.
* **DMOS log** of the newest `experiments/results/*/dmos.log`: budget
  (`Total replicas needed`), scaling actions (`cluster/service a → b`, green
  up, orange down), reach-weighted demand, warnings and errors.

## How a request is attributed

Every cluster's Hubble Relay is read with `hubble observe --follow --type l7`
over a `kubectl port-forward`. A request is counted once, from the L7
RESPONSE of a frontend pod seen by the relay of the cluster that runs that
pod (serving cluster *i*). The entry cluster *j* is the cluster that owns the
client IP of that response, i.e. the Gateway/Ingress Envoy where the request
entered; IPs are mapped to clusters with the pod CIDRs and node addresses of
the CiliumNodes (ClusterMesh requires distinct pod CIDRs). The same response
seen by the relay of the entry cluster (client side) is not counted again
(`client_side` under the matrix).

## Requirements

* `kubectl` and the **hubble CLI** in PATH (Windows: `hubble-windows-amd64.zip`
  from https://github.com/cilium/hubble/releases, put `hubble.exe` in a PATH folder).
* **Hubble Relay** in every cluster: it is there if the Hubble UI works.
  The script opens the port-forwards itself (`svc/hubble-relay` in
  `kube-system`; `--relay-namespace`, `--relay-service`, `--relay-tls` if
  different).
* `.cache/test-env.json` (kubeconfigs and Prometheus of the clusters).

Reading every flow loads the Cilium agents, the netem queues (the streams
leave the clusters towards the operator machine) and the operator machine,
which also runs DMOS: use
the map in pilot or demonstration runs, not while measuring, unless its
effect has been measured:

```bash
bash experiments/run_map_overhead.sh     # 4 pairs of reach-pool-ps con, ABBA order, ~2 h 40
```

Each run folder gets `LIVE_MAP_on.txt` or `LIVE_MAP_off.txt`; the "on" runs
also get `live_map_state.json` (relays connected, flows read). The map stops
cleanly with `GET /quit` or `--duration SECONDS`, terminating its
port-forwards and hubble processes.
