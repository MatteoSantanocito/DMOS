# DMOS Troubleshooting Guide

Reference catalog of failure modes, root causes, and fixes for the DMOS
multi-cluster Kubernetes scheduler. Each `### Symptom` block names the exact
error string the user is likely to see, identifies the root cause, and lists
the commands to fix it.

DMOS runs locally on the operator's laptop and reaches a fleet of N K3s
clusters — 6 in the reference testbed (`ms-cluster-01` … `ms-cluster-06`,
Proxmox VMs across 3 hardware tiers) — each with Cilium + Hubble L7 +
Prometheus + ping_exporter + Online Boutique. The canonical smoke test is
`scripts/validate-setup.sh`; before experiments, `experiments/preflight_runs.sh`
checks the same things plus netem. Command examples below use the generic
`cluster1`, `cluster2`, … names of the assisted-bootstrap path
(`.kube/clusterN.yaml`) — on the reference testbed substitute the kubeconfigs
`~/dmos-iac/kubeconfig-ms-cluster-NN` and extend the `for` loops to your N.

---

## Table of contents

1. [How to diagnose](#1-how-to-diagnose)
2. [Cluster connectivity issues](#2-cluster-connectivity-issues)
3. [Cilium / Hubble issues](#3-cilium--hubble-issues)
4. [ping_exporter issues](#4-ping_exporter-issues)
5. [Online Boutique issues](#5-online-boutique-issues)
6. [DMOS scheduler runtime issues](#6-dmos-scheduler-runtime-issues)
7. [`tc netem` simulation issues](#7-tc-netem-simulation-issues)
8. [Recovery procedures](#8-recovery-procedures)
9. [When all else fails](#9-when-all-else-fails)

---

## 1. How to diagnose

Run the canonical smoke test from the repo root:

```bash
bash scripts/validate-setup.sh
```

It exits `0` if the fleet is healthy. Every failure prints a short hint that
points at a section of this document. Each hint shipped by the script
corresponds to a documented symptom below.

For deeper introspection on a per-cluster basis (what's installed, how it's
installed, whether scrape targets are wired) run:

```bash
bash scripts/diagnose_cluster.sh
```

This is read-only — it does not change anything in the clusters.

### Quick map: validate-setup hint -> symptom in this document

| Hint printed by `validate-setup.sh`                                 | See                                                        |
|---------------------------------------------------------------------|------------------------------------------------------------|
| `config/local.yaml not found`                                    | [§2.1](#symptom-configlocalyaml-not-found)              |
| `<tool> not on PATH`                                                | [§6.6](#symptom-tool-not-on-path)                          |
| `Could not parse cluster IPs from config/local.yaml`             | [§6.7](#symptom-could-not-parse-cluster-ips-from-configlocalyaml)               |
| `.kube/<name>.yaml missing`                                         | [§2.2](#symptom-kubenameyaml-missing)                      |
| `K8s API unreachable`                                               | [§2.3](#symptom-k8s-api-unreachable)                       |
| `Cilium pods not Running`                                           | [§3.1](#symptom-cilium-pods-not-running)                   |
| `ClusterMesh not connected`                                         | [§2.4](#symptom-clustermesh-not-connected)                 |
| `Prometheus NOT reachable on <ip>`                                  | [§2.5](#symptom-prometheus-not-reachable-on-ip)            |
| `ping_exporter not deployed in 'observability' namespace`           | [§4.1](#symptom-ping_exporter-not-deployed)                |
| `Online Boutique frontend not Ready`                                | [§5.1](#symptom-online-boutique-frontend-not-ready)        |
| `Hubble L7 metric absent`                                           | [§3.2](#symptom-hubble-l7-metric-absent)                   |
| `ping_rtt_mean_seconds absent`                                      | [§4.2](#symptom-ping_rtt_mean_seconds-absent)              |
| `./venv missing`                                                    | [§6.5](#symptom-venv-missing)                              |
| `ConfigLoader failed`                                               | [§6.1](#symptom-configloader-failed)                       |

---

## 2. Cluster connectivity issues

### Symptom: `config/local.yaml not found`

**Root cause.** Bootstrap was never completed; DMOS has nothing to talk to.

**Fix.**

```bash
bash scripts/bootstrap-assisted.sh
# or copy the example by hand:
cp config/local.yaml.example config/local.yaml
$EDITOR config/local.yaml   # fill in the node IPs
```

### Symptom: `.kube/<name>.yaml missing`

**Root cause.** Kubeconfigs were not copied off the K3s nodes.

**Fix.** Re-run the kubeconfig copy step from the setup guide:

```bash
mkdir -p .kube
for n in 1 2 3; do
    var="CLUSTER${n}_IP"; ip="${!var}"
    scp ubuntu@${ip}:/etc/rancher/k3s/k3s.yaml .kube/cluster${n}.yaml
    sed -i "s/127.0.0.1/${ip}/g" .kube/cluster${n}.yaml
done
```

### Symptom: `K8s API unreachable`

Full message from `validate-setup.sh`:

```
✗ K8s API unreachable — check firewall on TCP 6443
```

**Root causes (any of):**

1. Host firewall on the node blocks TCP 6443.
2. The kubeconfig still points at `127.0.0.1` (the K3s default) instead of
   the node's routable IP.
3. The serving certificate's SAN list does not include the IP your laptop is
   using to reach the API. K3s must be installed with `--tls-san=<NODE_IP>`.

**Fix.**

```bash
# 1. Probe the API socket
curl -k --max-time 5 https://<NODE_IP>:6443/healthz

# 2. Open the port (Ubuntu / ufw)
ssh ubuntu@<NODE_IP> 'sudo ufw allow 6443/tcp || true; sudo ufw status'

# 3. Inspect the kubeconfig server URL
grep '^    server:' .kube/cluster1.yaml
# must NOT be https://127.0.0.1:6443

# 4. If the SAN is wrong, re-install K3s with --tls-san=<NODE_IP>
#    (see Recovery procedures below)
```

### Symptom: `ClusterMesh not connected`

Full message from `validate-setup.sh`:

```
✗ ClusterMesh not connected — check NodePort 32379 between nodes
```

**Root causes (any of):**

1. NodePort `32379` (the `clustermesh-apiserver`) is not reachable between
   the nodes — usually a firewall on the LAN.
2. The `cilium-clustermesh` Secret has not been pushed to one of the peers.
3. `cilium clustermesh enable` / `cilium clustermesh connect` was never run
   for one of the cluster pairs.

**Fix.**

```bash
# Verify port reachability node-to-node
ssh ubuntu@<CLUSTER2_IP> 'curl -k --max-time 5 https://<CLUSTER1_IP>:32379/' || true

# Inspect mesh status
KUBECONFIG=.kube/cluster1.yaml cilium clustermesh status --verbose

# Re-arm the handshake
KUBECONFIG=.kube/cluster1.yaml cilium clustermesh enable
KUBECONFIG=.kube/cluster2.yaml cilium clustermesh enable
KUBECONFIG=.kube/cluster3.yaml cilium clustermesh enable

cilium clustermesh connect --kubeconfig .kube/cluster1.yaml --destination-context cluster2
cilium clustermesh connect --kubeconfig .kube/cluster1.yaml --destination-context cluster3
cilium clustermesh connect --kubeconfig .kube/cluster2.yaml --destination-context cluster3

# Confirm
KUBECONFIG=.kube/cluster1.yaml cilium clustermesh status --wait
```

### Symptom: `Prometheus NOT reachable on <ip>`

`validate-setup.sh` probes NodePort `32090` (the kube-prometheus-stack
default in this setup) and falls back to the legacy `30090` before
declaring failure.

**Root causes (any of):**

1. The `prometheus` Service exists as `ClusterIP` instead of `NodePort 32090`
   (kube-prometheus-stack defaults).
2. The kube-prometheus-stack pods (`prometheus-prometheus-0`,
   `prometheus-operator-*`) are not Running.
3. NodePort 32090 is blocked between your laptop and the node.

**Fix.**

```bash
# 1. Service type / NodePort
KUBECONFIG=.kube/cluster1.yaml kubectl -n observability get svc \
    -l app.kubernetes.io/name=prometheus -o wide
# Expect: TYPE=NodePort  PORT(S)=9090:32090/TCP

# 2. Pod status
KUBECONFIG=.kube/cluster1.yaml kubectl -n observability get pods

# 3. Direct probe from the laptop
curl -fsS --max-time 5 http://<NODE_IP>:32090/-/ready

# 4. If the Service type is wrong, re-apply Helm values
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
    --kubeconfig .kube/cluster1.yaml -n observability \
    -f config/prometheus-helm-values.yaml
```

---

## 3. Cilium / Hubble issues

### Symptom: `Cilium pods not Running`

Full message from `validate-setup.sh`:

```
✗ Cilium pods not Running (N bad) — check 'kubectl logs -n kube-system -l k8s-app=cilium'
```

**Root causes (any of):**

1. **CNI conflict.** K3s was installed without `--flannel-backend=none` and
   `--disable-network-policy`. Flannel and Cilium both try to manage the
   pod network and trip over each other.
2. **MTU mismatch.** VXLAN overlay adds 50 bytes; if the underlay MTU is
   already 1500, pod traffic ends up at 1450, and large pod-to-pod packets
   silently drop.
3. **IPAM exhausted / wrong CIDR.** Each cluster must have a distinct
   `--cluster-cidr` (`10.42.0.0/16`, `10.43.0.0/16`, `10.44.0.0/16`). A
   collision causes pods to be assigned IPs that conflict across the mesh.
4. **Image pull / registry rate limits.**

**Fix.**

```bash
# Inspect the cilium-agent logs
KUBECONFIG=.kube/cluster1.yaml kubectl -n kube-system \
    logs -l k8s-app=cilium --tail=200

# Check Cilium's own self-diagnosis
KUBECONFIG=.kube/cluster1.yaml cilium status --verbose

# Confirm Flannel is gone
ssh ubuntu@<NODE_IP> 'ls /var/lib/rancher/k3s/agent/etc/flannel 2>/dev/null && echo FLANNEL_PRESENT'

# Apply MTU fix (edit the rendered values then upgrade; re-render from
# config/local.yaml with scripts/render-runtime-files.sh if you regenerate)
$EDITOR .cache/cilium-cluster1-values.yaml   # set: mtu: 1450
helm upgrade cilium cilium/cilium \
    --kubeconfig .kube/cluster1.yaml -n kube-system \
    -f .cache/cilium-cluster1-values.yaml
```

### Symptom: `Hubble L7 metric absent`

Full message from `validate-setup.sh`:

```
✗ Hubble L7 metric absent — apply deployments/cnp-*-l7.yaml and helm upgrade Cilium con hubble.metrics.enabled
```

**Root causes (any of):**

1. The httpV2 Hubble exporter is not enabled in the Cilium Helm values
   (`hubble.metrics.enabled` must include `httpV2` with the context labels;
   set by `scripts/setup-dmos-on-clusters.ps1` on the reference testbed, or
   in `.cache/cilium-<cluster>-values.yaml` on the bootstrap path). Without
   it Cilium never emits `hubble_http_requests_total`.
2. The helper CNPs that flip Envoy on for the relevant pods
   (`deployments/cnp-frontend-l7.yaml`, `deployments/cnp-backends-l7.yaml`)
   were never applied. East-west pod-to-pod HTTP stays L4-only and no L7
   metric is produced.
3. The `hubble-metrics` Service does not exist, or the ServiceMonitor
   (`deployments/hubble-servicemonitor-monitoring.yaml`, label
   `release: prometheus-stack`) is not selected by the Prometheus Operator.
4. No traffic — Hubble counters only advance once at least one HTTP request
   has been observed.

**Fix.**

```bash
# 1. Verify the deployed Helm values include httpV2
KUBECONFIG=.kube/cluster1.yaml helm get values cilium -n kube-system | grep -n httpV2 || echo "MISSING"
#    if missing: pwsh scripts/setup-dmos-on-clusters.ps1 -SkipOnlineBoutique -SkipDmosManifests

# 2. Apply (or re-apply) the L7 helper CNPs and the ServiceMonitor to every cluster
for n in 1 2 3; do
    KUBECONFIG=.kube/cluster${n}.yaml kubectl apply -f deployments/cnp-frontend-l7.yaml
    KUBECONFIG=.kube/cluster${n}.yaml kubectl apply -f deployments/cnp-backends-l7.yaml
    KUBECONFIG=.kube/cluster${n}.yaml kubectl apply -f deployments/hubble-servicemonitor-monitoring.yaml
done

# 3. (scripts/enable_hubble_http.sh patches the cilium-config ConfigMap
#    directly; it belongs to the three-cluster testbed — on a Helm-managed
#    Cilium prefer step 1)

# 4. Drive a single request through Cilium Ingress, then re-query
curl -sS http://<NODE_IP>:30080/ > /dev/null
curl -sS "http://<NODE_IP>:32090/api/v1/query?query=hubble_http_requests_total" \
    | python3 -m json.tool | head -30
```

> Note: applying or restarting these CNPs causes a brief Envoy
> re-injection. `kubectl rollout restart deployment/<svc> -n online-boutique`
> is the canonical way to clear stale connections afterwards.

### Symptom: `source_workload="" returns no data` in Hubble queries

**Root cause.** External traffic is no longer flowing through **Cilium
Ingress**. Hubble only fills in `source_workload=""` for traffic that
arrived through the in-process Envoy listener; if the test client is
hitting the frontend pod directly (or via a different ingress), that label
is never the empty string.

**Fix.**

```bash
# 1. Confirm Cilium Ingress is the entry point
KUBECONFIG=.kube/cluster1.yaml kubectl -n online-boutique get ingress
# expect: ingressClassName: cilium  +  NodePort 30080

# 2. Confirm 30080 is reachable from the test client
curl -fsS --max-time 5 http://<NODE_IP>:30080/

# 3. Drive a request and re-query
curl -sS "http://<NODE_IP>:32090/api/v1/query?query=sum(rate(hubble_http_requests_total{destination_namespace=\"online-boutique\",source_workload=\"\"}[1m]))"
```

> If you see this symptom together with anything NGINX-shaped, you've
> reverted to the legacy NGINX Ingress setup. Switch back to Cilium Ingress
> per `docs/ingress-architecture.md` — DMOS' L7 path is designed for it.
> The historical NGINX guidance is preserved in
> `docs/_archive/nginx-ingress-hubble-l7.md` for reference only and should
> not be promoted into the live setup.

### Symptom: Counter is flat for some internal hops only

**Root cause.** The relevant pod is not covered by either CNP. Because L7
metrics are only emitted when Cilium inserts Envoy on the destination's
port, a pod that no CNP selects produces no L7 metric.

**Fix.** Extend `cnp-backends-l7.yaml` to include the missing port and
re-apply on all clusters. Confirm Envoy is active:

```bash
KUBECONFIG=.kube/cluster1.yaml kubectl exec -n kube-system <cilium-pod> \
    -- hubble observe --type l7 --namespace online-boutique --last 5
```

---

## 4. ping_exporter issues

### Symptom: `ping_exporter not deployed`

Full message from `validate-setup.sh`:

```
✗ ping_exporter not deployed in 'observability' namespace
```

**Root cause.** The Helm chart was never installed on this cluster, or it
was installed in a different namespace.

**Fix.**

```bash
helm repo add czerwonk https://czerwonk.github.io/ping_exporter && helm repo update
helm upgrade --install ping-exporter czerwonk/ping-exporter \
    --kubeconfig .kube/cluster1.yaml \
    -n observability --create-namespace \
    -f .cache/ping-exporter-<cluster>-values.yaml
# or, for the whole fleet: bash scripts/setup-netem-and-ping.sh ping-only
```

### Symptom: `ping_rtt_mean_seconds absent`

Full message from `validate-setup.sh`:

```
✗ ping_rtt_mean_seconds absent — check ping_exporter logs and ServiceMonitor labels
```

**Root causes (any of):**

1. **ServiceMonitor label mismatch.** kube-prometheus-stack only picks up
   ServiceMonitors whose `release` label matches its release name
   (`prometheus-stack` on the reference testbed; the rendered ping_exporter
   values use that label).
2. **ICMP egress is blocked** by the host firewall on the node. The pod
   can't ping the peers.
3. **Wrong target IPs** in `.cache/ping-exporter-<cluster>-values.yaml`
   (typo in `config/local.yaml`, leftover placeholder, stale IP after a
   node moved — fix `local.yaml` and re-run `scripts/render-runtime-files.sh`).
4. **Pod cannot reach the peer pod-CIDR / node-IP.**

**Fix.**

```bash
# 1. Inspect logs for "ping: permission denied" or "no route to host"
KUBECONFIG=.kube/cluster1.yaml kubectl -n observability \
    logs -l app.kubernetes.io/name=ping-exporter --tail=100

# 2. Check ServiceMonitor labels match what Prometheus selects
KUBECONFIG=.kube/cluster1.yaml kubectl -n observability \
    get servicemonitor -o yaml | grep -A2 labels:

# 3. Probe from inside the pod network
KUBECONFIG=.kube/cluster1.yaml kubectl -n observability run -it --rm probe \
    --image=busybox --restart=Never -- ping -c 3 <PEER_IP>

# 4. Local /metrics smoke test
KUBECONFIG=.kube/cluster1.yaml kubectl -n observability \
    port-forward svc/ping-exporter 9427:9427 &
curl -sS http://localhost:9427/metrics | grep ping_rtt_mean_seconds
kill %1
```

---

## 5. Online Boutique issues

### Symptom: `Online Boutique frontend not Ready`

Full message from `validate-setup.sh`:

```
✗ Online Boutique frontend not Ready
```

**Root causes (any of):**

1. **Image pull failure** (registry rate-limited, no internet egress on the
   node).
2. **Namespace missing** — `kubectl create namespace online-boutique` was
   skipped.
3. **Replicas spec is `0`** because DMOS already scaled the deployment down
   on a previous run.
4. **Init delays / readiness probes** — the gRPC backends need `>30s` to
   warm up; the frontend's readiness gate fails until they answer.
5. **OOMKilled** on `currencyservice`, `paymentservice`, `cartservice` or
   `recommendationservice` (default memory limits too low under load) — run
   `experiments/apply_resource_hardening.sh`.

**Fix.**

```bash
# 1. Inspect deploy + pods
KUBECONFIG=.kube/cluster1.yaml kubectl -n online-boutique get deploy,pods

# 2. Look at events (image pull, OOM)
KUBECONFIG=.kube/cluster1.yaml kubectl -n online-boutique \
    describe pod -l app=frontend | grep -A2 -E 'State|Reason|Exit|OOM'

# 3. Restore replicas
KUBECONFIG=.kube/cluster1.yaml kubectl -n online-boutique \
    scale deploy frontend --replicas=1

# 4. Re-apply the resource and lifecycle hardening (all clusters)
bash experiments/apply_resource_hardening.sh
bash experiments/apply_pod_lifecycle.sh

# 5. Wait
KUBECONFIG=.kube/cluster1.yaml kubectl -n online-boutique \
    wait --for=condition=available --timeout=300s deploy --all
```

---

## 6. DMOS scheduler runtime issues

### Symptom: `ConfigLoader failed`

Full message from `validate-setup.sh`:

```
✗ ConfigLoader failed — check config/ files and weights.yaml
```

**Root causes (any of):**

1. The active weight profile, or the selected score model
   (`DMOS_SCORE_MODEL`), in `config/weights.yaml` is invalid (weights not
   summing to 1, unknown keys, unknown model name): DMOS refuses to start
   and says why.
2. A required field (`name`, `region`, `location`, `ip`, `kubeconfig_path`)
   is missing in `config/local.yaml`.
3. Malformed YAML — a stray tab, an unclosed quote, a list item without a
   leading `-`.

**Fix.**

```bash
# 1. Re-run the loader explicitly to surface the underlying exception
source venv/bin/activate
python -c "from src.utils.config_loader import ConfigLoader; ConfigLoader('config')"

# 2. Build the whole effective configuration offline (validates models and weights)
DMOS_SCORE_MODEL=nca python -m src.dmos_main --dump-config-only /tmp/cfg.json

# 3. YAML lint
python -c "import yaml; yaml.safe_load(open('config/local.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/services.yaml'))"
```

### Symptom: `kubectl scale` returns 403 / 404 from the scheduler

Typical scheduler log lines:

```
ERROR | ... K8s API error: 403 Forbidden
ERROR | ... K8s API error: 404 Not Found
ERROR | ... Cluster <name> not configured
```

**Root causes (any of):**

1. **Wrong context.** The kubeconfig in `.kube/clusterN.yaml` does not
   actually point at cluster N — typically because the post-`scp` `sed`
   step substituted the wrong IP.
2. **RBAC.** The kubeconfig holds a token without `patch`/`update` on
   `apps/v1.deployments` in `online-boutique`.
3. **Deployment doesn't exist on that cluster.** Online Boutique was only
   applied to a subset of clusters.

**Fix.**

```bash
# 1. Confirm contexts match expected cluster
for n in 1 2 3; do
    KUBECONFIG=.kube/cluster${n}.yaml kubectl config view --minify \
        | grep -E 'name:|server:'
done

# 2. Probe RBAC explicitly
KUBECONFIG=.kube/cluster2.yaml kubectl auth can-i patch deploy \
    -n online-boutique
# expect: yes

# 3. Confirm the deployment is there
KUBECONFIG=.kube/cluster3.yaml kubectl -n online-boutique get deploy cartservice
```

### Symptom: replicas always concentrate on the same cluster

In `dmos.log` the allocations (`Allocated N replicas to <cluster>`) keep
favouring one cluster.

**Root causes (any of):**

1. **Expected behaviour.** With the demand gate (score model `nca`) replicas
   follow the demand share: under a concentrated regime most of the budget
   goes to the hot cluster, up to its cap. Check the `Ingress rates:` line.
2. **Demand not measured on some clusters** (Hubble ingress metric
   missing): their share is 0 and the gate suppresses them. Look for
   clusters with `0.0 req/s` while k6 sends to them (§3.2).
3. **A weight profile dominated by one objective** on which one cluster
   always wins (e.g. `carbon_first`, or the legacy `latonly`/`carbon`).
4. **Prometheus not answering on some clusters**: their CPU/memory read as
   0 and L0 excludes them (`Cluster esclusi:` in the log). Re-check §2.5.

**Checks.**

```bash
# score and its components per cluster, every cycle
grep -E "Ingress rates:|Score ms-cluster|Allocated|Cluster esclusi" experiments/results/<run>/dmos.log | tail -40
# effective score of the run
python experiments/run_provenance.py show experiments/results/<run>
```

### Symptom: replicas oscillate between cycles

**Root causes (any of):**

1. **Two clusters very close on score** — small changes in latency, RTT or
   carbon tip the proportional allocation back and forth (the geo-boost
   moves one replica to the best-scored cluster on ties).
2. **Traffic near a trigger threshold** — cycles start and stop as the total
   traffic crosses 10/30 req/s.
3. **Safeguards too permissive for the workload** — they are hard-coded in
   `src/dmos_main.py`: dead zone 10 %, scale-down cooldown 90 s, scale-up
   protection 90 s, +6/−1 per cycle, churn budgets 8/2 (frontend) and 12/6
   (co-scaling). See [dmos-architecture.md §10](dmos-architecture.md#10-safeguards-and-timing-constants).

**Fix.** Change the constants in `src/dmos_main.py` (they are recorded in
each run's `dmos_config.json`) and restart the scheduler.
`global_autoscaling` in `services.yaml` has no effect.

### Symptom: `./venv missing`

Full message from `validate-setup.sh`:

```
✗ ./venv missing — run: python3.12 -m venv venv && pip install -r requirements.txt
```

**Fix.**

```bash
python3.12 -m venv venv
source venv/bin/activate           # Linux / macOS
# .\venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
pip install -e .
```

### Symptom: `<tool> not on PATH`

`kubectl`, `helm`, `curl`, or `python3` is missing.

**Fix.** Install the missing tool per `docs/setup-guide.md` §2 (operator
prerequisites), then re-run `validate-setup.sh`.

### Symptom: `Could not parse cluster IPs from config/local.yaml`

**Root cause.** The YAML is syntactically valid for `yaml.safe_load`, but
the `clusters:` field is missing or has an unexpected shape (e.g. a string
instead of a list/dict, or `name`/`ip` keys absent). Re-check against
`config/local.yaml.example`.

---

## 7. `tc netem` simulation issues

### Symptom: ping RTT does not match the configured netem delay

E.g. `config/local.yaml` declares `netem.delay_ms: 100` for a cluster but
`ping_rtt_mean_seconds` toward it reports 1–2 ms.

**Root causes (any of):**

1. **Wrong interface.** `scripts/setup-netem-and-ping.sh` applies the rule
   to the `netem.interface` declared per cluster in `config/local.yaml`
   (Proxmox default `ens18`). On bare metal the interface is usually
   `eth0` or `enp1s0` and the rule is silently applied to a non-existent
   device.
2. **Qdisc was cleared** by another process (e.g. a network restart) and
   the rule is no longer present. `tc netem` is not persistent across
   reboots by default.
3. **Direction confusion.** `tc netem` only delays *outgoing* traffic from
   the interface; if the rule is on the wrong node, RTT is measured in the
   wrong direction.

**Fix.**

```bash
# 1. Find the right interface
ssh ubuntu@<NODE_IP> 'ip -br link | grep -v "lo "'

# 2. Inspect current qdisc (script-provided check across the fleet)
bash scripts/setup-netem-and-ping.sh verify

# 3. Re-apply (after fixing netem.interface in config/local.yaml and
#    re-running scripts/render-runtime-files.sh if needed)
bash scripts/setup-netem-and-ping.sh netem-only

# 4. Validate end-to-end with ping
ssh ubuntu@<NODE_IP> 'ping -c 5 -W 2 <PEER_IP>'
```

> If you need persistence across reboot, wire
> `setup-netem-and-ping.sh netem-only` into a `systemd` oneshot. The repo
> does not ship one — production-grade simulation is out of scope.

---

## 8. Recovery procedures

### How do I fully reset cluster N?

When a single cluster is so wedged that piecemeal fixes are not worth it:

```bash
# 1. Wipe K3s on the node
ssh ubuntu@<CLUSTER_N_IP> '/usr/local/bin/k3s-uninstall.sh'

# 2. Re-run bootstrap from the top (re-installs K3s with the right flags,
#    Cilium with the cluster's Helm values, kube-prometheus-stack,
#    ping_exporter, Online Boutique, and the L7 CNPs).
#    bootstrap-assisted.sh prompts for IPs and writes the full command
#    transcript to bootstrap-commands.sh for piece-by-piece execution.
bash scripts/bootstrap-assisted.sh
bash bootstrap-commands.sh   # generated transcript — run section by section

# 3. Re-collect the kubeconfig
scp ubuntu@<CLUSTER_N_IP>:/etc/rancher/k3s/k3s.yaml .kube/clusterN.yaml
sed -i "s/127.0.0.1/<CLUSTER_N_IP>/g" .kube/clusterN.yaml

# 4. Re-connect ClusterMesh
KUBECONFIG=.kube/clusterN.yaml cilium clustermesh enable
cilium clustermesh connect --kubeconfig .kube/cluster1.yaml \
    --destination-context clusterN

# 5. Re-validate
bash scripts/validate-setup.sh
```

### How do I clear stale Envoy state after a CNP change?

Applying or modifying a CNP that toggles L7 mode causes Cilium to
re-inject Envoy on existing connections, which can stall in-flight HTTP/2
streams:

```bash
for n in 1 2 3; do
    KUBECONFIG=.kube/cluster${n}.yaml kubectl -n online-boutique \
        rollout restart deploy
done
```

### How do I reset only the DMOS local state?

DMOS keeps no on-disk state besides logs. Restart is `Ctrl-C` and re-launch:

```bash
source venv/bin/activate
python -m src.dmos_main
```

If the scheduler scaled deployments and you want a clean baseline before
the next experiment, use the harness's reset (all Deployments of
`online-boutique` to 1 replica on every cluster, then waits):

```powershell
pwsh scripts/reset_test.ps1
```

Afterwards it scales the Online Boutique `loadgenerator` back to 0 (if the
Deployment exists): its internal traffic must not run during the
experiments, and every harness cell refuses to start while it has
replicas.

---

## 9. When all else fails

In rough order of usefulness, in increasing depth:

```bash
# 1. Cross-cluster snapshot (read-only, what's installed and how)
bash scripts/diagnose_cluster.sh

# 2. Smoke test (re-run after each fix)
bash scripts/validate-setup.sh

# 3. Cilium agent logs on each cluster
for n in 1 2 3; do
    echo "== cluster${n} =="
    KUBECONFIG=.kube/cluster${n}.yaml kubectl -n kube-system \
        logs -l k8s-app=cilium --tail=80
done

# 4. Cilium self-diagnosis
for n in 1 2 3; do
    KUBECONFIG=.kube/cluster${n}.yaml cilium status --verbose
done

# 5. Live Hubble L7 events (drives the test traffic, then watches)
curl -sS http://<CLUSTER1_IP>:30080/ > /dev/null
KUBECONFIG=.kube/cluster1.yaml kubectl exec -n kube-system \
    "$(KUBECONFIG=.kube/cluster1.yaml kubectl -n kube-system get pod \
        -l k8s-app=cilium -o jsonpath='{.items[0].metadata.name}')" \
    -- hubble observe --type l7 --namespace online-boutique --last 10

# 6. Scheduler log: stdout/stderr (dmos.log in each run directory) and
#    logs/DMOSMain.log. The level is INFO; DEBUG lines (per-component
#    details) require changing the level passed to setup_logging/get_logger
#    in the code — there is no environment switch.
source venv/bin/activate
python -m src.dmos_main
```

If the failure is reproducible and not covered above, capture:

- Output of `bash scripts/validate-setup.sh` (full).
- Output of `bash scripts/diagnose_cluster.sh` (full).
- Last 200 lines of `cilium-agent` logs from the affected cluster.
- The exact `ConfigLoader` traceback (if applicable).

…and open an issue against the repo.

