# Integration guide

For people building automation around DMOS — provisioning, CI, other
schedulers to compare with, dashboards. It lists every input DMOS takes and
every output it gives, with pointers to the detailed references. To deploy
DMOS by hand, start from [setup-guide.md](setup-guide.md).

## 1. Surfaces

| Surface | Where | Reference |
|---|---|---|
| Testbed description | `config/local.yaml` (gitignored; template `config/local.yaml.example`) | [configuration-reference.md §2](configuration-reference.md#2-localyaml--testbed) |
| Algorithm configuration | `config/weights.yaml` (score models, legacy profiles), `config/services.yaml` (service model), `config/carbon.yaml` | [configuration-reference.md §3–§5](configuration-reference.md#3-weightsyaml--score) |
| Generated runtime files | `.cache/test-env.json`, `.cache/test-env.sh`, `.cache/cilium-<cluster>-values.yaml`, `.cache/ping-exporter-<cluster>-values.yaml` | [configuration-reference.md §6](configuration-reference.md#6-helm-values-and-generated-files) |
| Cluster resources DMOS expects | `deployments/*.yaml` (§3) | [ingress-architecture.md](ingress-architecture.md), [cilium-hubble-l7-visibility.md](cilium-hubble-l7-visibility.md) |
| Process interface | CLI flags and `DMOS_*` environment variables | [configuration-reference.md §7](configuration-reference.md#7-environment-variables) |
| Runtime interface | HTTP `:8081`, Prometheus `:9090` (§4) | [dmos-architecture.md §11](dmos-architecture.md#11-interfaces-ports-endpoints-exported-metrics) |
| Experiment outputs | `experiments/results/<run>/` | [METRICS-AND-OUTPUTS.md](METRICS-AND-OUTPUTS.md), [RUN-PROVENANCE.md](RUN-PROVENANCE.md) |

The scheduler runs outside the clusters and needs, per cluster, a kubeconfig
with permission to read and scale Deployments in `online-boutique`, and HTTP
access to the cluster's Prometheus (`http://<ip>:<prometheus_port>`).

## 2. Producing the configuration

A tool that provisions clusters has to:

1. write `config/local.yaml` (schema: `local.yaml.example` and
   `ClusterConfig` in `src/utils/config_loader.py`);
2. run `bash scripts/render-runtime-files.sh` (or produce the same
   `.cache/test-env.json`: `ssh`, and per cluster `name`, `ip`, `region`,
   `display_name`, `ingress_url`, `prometheus_url`, `kubeconfig_path`,
   `netem`);
3. keep `config/services.yaml` consistent with the Deployments actually
   running (CPU/memory requests, per-cluster replica caps).

`python -m src.dmos_main --dump-config-only <file>` validates the whole
configuration without contacting the clusters and writes the effective
configuration (with its `config_id`) — a cheap CI check.

## 3. Cluster prerequisites

| Component | Reference testbed | Needed for |
|---|---|---|
| K3s | provisioned by proxmox-k3s | — |
| Cilium with kube-proxy replacement, ClusterMesh, Hubble `httpV2` metrics with context labels, Cilium Ingress on NodePort 30080 | Cilium 1.19.4 | demand and latency signals, entry point |
| kube-prometheus-stack, NodePort 32090, release `prometheus-stack` | from proxmox-k3s | all metrics |
| ping_exporter (chart `czerwonk/ping-exporter`), targets = the other clusters | `scripts/setup-netem-and-ping.sh` | Φ_net |
| Online Boutique in `online-boutique` | upstream release manifests | workload |
| `tc netem` on the workers (optional) | `scripts/setup-netem-and-ping.sh` | emulated geography |

Manifests to apply on every cluster (all applied by
`scripts/setup-dmos-on-clusters.ps1`):

| File | Resource | Role |
|---|---|---|
| `00-namespace.yaml` | Namespace | `online-boutique` |
| `cnp-frontend-l7.yaml` | CiliumNetworkPolicy | L7 visibility on the frontend |
| `cnp-backends-l7.yaml` | CiliumNetworkPolicy | L7 visibility on the backends |
| `ingress-frontend-cilium.yaml` | Ingress | entry point |
| `global-service-frontend.yaml`, `global-service-backends.yaml` | Services | ClusterMesh global services |
| `hubble-servicemonitor-monitoring.yaml` | ServiceMonitor | Hubble scrape every 15 s |

Optional: `hpa-baseline.yaml` (HPA baseline for comparisons),
`alertmanager-config.yaml.template` (Alertmanager → DMOS webhook). Not used
by the current setup: `dmos-scheduler.yaml` / `dmos-config.yaml` (in-cluster
DMOS, no image published), `prometheus-rule.yaml`, `prometheus-alerts.yaml`,
`grafana-dashboard-dmos.json`.

Post-conditions: `bash scripts/validate-setup.sh` exits 0, and
`bash experiments/preflight_runs.sh` passes (it also checks netem and
ClusterMesh).

## 4. Runtime interface

| Endpoint | Use |
|---|---|
| `GET :8081/health` | liveness, queue size, `config_id` |
| `GET :8081/config` | effective configuration of the running process |
| `POST :8081/api/schedule` `{"service": "frontend"}` | trigger a cycle (subject to the 30 s debounce) |
| `POST :8081/webhook/alert` | Alertmanager webhook; alerts with a `service` label become events |
| `GET :9090/metrics` | `dmos_actual_traffic`, `dmos_predicted_traffic`, `dmos_current_replicas`, `dmos_target_replicas`, `dmos_cluster_score`, `dmos_scaling_events_total`, `dmos_scheduling_duration_seconds`, `dmos_config_info` |

DMOS reads these Prometheus series on each cluster:
`hubble_http_requests_total`, `hubble_http_request_duration_seconds_bucket`,
`ping_rtt_mean_seconds`, `kube_node_status_capacity`,
`container_cpu_usage_seconds_total`, `container_memory_working_set_bytes`
(queries in [dmos-architecture.md §5.1](dmos-architecture.md#51-queries)).
It writes only `scale` on the Deployments of `online-boutique`.

## 5. Running experiments from automation

```bash
export DMOS_K6_REMOTE=ubuntu@<k6-vm>
bash experiments/run_one.sh <arm> <regime>     # one traceable cell
python experiments/run_provenance.py audit      # exit 2 if a run is invalid
python experiments/paper_generate_all.py        # figures and tables
```

`run_one.sh` exits non-zero when the cell could not run (invalid
configuration, health gate, missing netem); the run directory's
`manifest.json → provenance.ok` says whether a completed run is usable.
Details: [../experiments/RUNBOOK.md](../experiments/RUNBOOK.md).
