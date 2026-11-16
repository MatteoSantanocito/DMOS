# Karmada baseline (arms `karmada-fhpa`, `karmada-agg`)

> **Optional tool, not used in the paper's campaign.** The Karmada arms are
> kept as an optional external baseline; the campaign of the paper
> (`experiments/plans/2026-11-02-definitive.plan`) does not include them, and
> no figure or table of the paper depends on them.

Karmada is the CNCF multi-cluster orchestrator. Here it serves as an
external multi-cluster baseline: same 6 clusters, same Online Boutique,
same Cilium data plane (frontend global service, `affinity: "none"`), same
k6 load. DMOS does not run in these arms and nothing of DMOS changes.

**How the comparison works.** Karmada runs outside the 6 clusters (like DMOS,
which runs on the operator machine) and reaches them in Push mode: no Karmada pod on
the measured nodes. Only during a Karmada cell it takes over the 8
Deployments that DMOS scales, scales them with FederatedHPA and divides the
replicas among the clusters with its scheduler; at the end of the cell it
gives them back untouched and the next cell starts from the usual reset.
Everything is done by `experiments/karmada_baseline.py`, called by the
harness (`experiments/lib/run_common.sh`).

Version: **Karmada v1.19.0** (docs v1.19 and source at tag `v1.19.0`;
sources at the end).

## 1. The arms

| Arm | Karmada configuration |
|---|---|
| `karmada-fhpa` | FederatedHPA per service (CPU utilization 70 %, min / max = sum over the clusters of the per-cluster limits of `services.yaml`, the same numbers as the `hpa-none` HPAs, Kubernetes default behaviour) + PropagationPolicy `Divided` / `Weighted` / `dynamicWeight: AvailableReplicas` (replicas in proportion to how many each cluster can still host), scheduler estimator on, failover off |
| `karmada-agg` | as above with `replicaDivisionPreference: Aggregated` (as few clusters as possible) |

FederatedHPA limits (`python experiments/karmada_baseline.py show`):
frontend 6–28, cartservice 6–12, productcatalogservice 6–18, checkoutservice
6–12, recommendationservice 6–24, currencyservice 6–26, adservice 6–18,
shippingservice 6–12.

Karmada does not know DMOS's per-cluster maxima or where the demand enters:
it divides by free capacity. That is the behaviour being compared, and it is
reported, not corrected. No `spreadConstraints`: forcing one replica per
cluster would make Karmada look more like DMOS than it is.

## 2. Karmada facts the harness relies on

| Fact (v1.19.0) | Source | Consequence |
|---|---|---|
| A PropagationPolicy with `conflictResolution: Overwrite` over Deployments that already exist in the members takes them over in place: pods not recreated | migrate-in-batch, resource-migration tutorial | Karmada is switched on per cell, without reinstalling Online Boutique |
| A member object applied by Karmada carries the **label `karmada.io/managed=true`** (`karmadactl get --operation-scope=all` shows it as `ADOPTION=Y`); `work.karmada.io/name` / `namespace` are **annotations** | `pkg/util/constants.go`, `pkg/controllers/ctrlutil/work.go`, `pkg/karmadactl/get/get.go` | the env gate and the release check use that label |
| With `preserveResourcesOnDeletion: true`, deleting the resource template keeps the member objects and **removes Karmada's labels and annotations** from them | PropagationPolicy API reference; `execution_controller.go` (`cleanupPolicyClaimMetadata`) | release = delete FederatedHPAs, then templates, then the policy; the member Deployments stay, unmanaged |
| Deleting the policy alone leaves templates and member objects unchanged; `unjoin` deletes what Karmada propagated to that member | PropagationPolicy API reference; cluster-registration | never `unjoin` while something is managed |
| The scheduler **drops a cluster given 0 replicas** from `ResourceBinding.spec.clusters` (`removeZeroReplicasCluster`); its Work is then deleted, so the member Deployment is deleted, or, if preserved, left unmanaged with its old replicas. With `--enable-empty-workload-propagation=true` (default false) the cluster stays in the binding with 0 replicas and the Deployment is scaled to 0 | `pkg/scheduler/core/{common,util,generic_scheduler}.go`, `pkg/controllers/binding` | **required scheduler flag** (§4.4): it only changes how a 0 is applied, not the scheduling decision |
| ResourceBinding of Deployment `frontend` = `frontend-deployment`; conditions `Scheduled`, `FullyApplied`; Works in `karmada-es-<cluster>`, labelled `resourcebinding.karmada.io/permanent-id` | `pkg/util/names/names.go`, `binding_types.go` | the trace reads the bindings; the release finds every Work and checks it preserves resources |
| FederatedHPA writes the replicas into the template's scale subresource; status conditions are the `autoscaling/v2` ones (`ScalingActive=True`, reason `ValidMetricFound`); defaults: sync 15 s, tolerance 0.1, scale-down stabilization 5 min | `federatedhpa_controller.go`; karmada-controller-manager reference | `verify` requires `ScalingActive=True` |
| FederatedHPA needs `karmada-metrics-adapter` (APIServices `v1beta1.metrics.k8s.io`, `v1beta1/v1beta2.custom.metrics.k8s.io` in the Karmada apiserver) and metrics-server in the members; it does not need MultiClusterService | `addons/metricsadapter`; autoscaling-with-resource-metrics tutorial | `check` requires `v1beta1.metrics.k8s.io` Available |
| `karmadactl addons enable karmada-scheduler-estimator` does **not** add `--enable-scheduler-estimator=true` to the scheduler | `addons/estimator/estimator.go`; scheduler-estimator guide | set by hand (§4.4) |
| Template labels and annotations are copied to the member objects; member annotations Karmada did not set (e.g. `deployment.kubernetes.io/revision`) are retained | `objectwatcher.retainClusterFields` | templates carry no label of ours; a rollout is detected by the revision annotation |
| Namespaces created in the Karmada apiserver are propagated to all members unless labelled `namespace.karmada.io/skip-auto-propagation: "true"` | `namespace_sync_controller.go` | `online-boutique` is created in Karmada with that label |
| Feature gate `Failover` off by default; NoExecute taint eviction off | controller-manager reference; cluster-failover | left at the defaults: no migration mid-cell |
| Compatibility: each release tested on the latest 10 Kubernetes versions; `karmadactl init` default control plane v1.36.2 | compatibility page; CHANGELOG-1.19 | check the member K3s version (`kubectl get clusters`, column VERSION) |

## 3. Prerequisites

- Cilium with the fix of [#46623](https://github.com/cilium/cilium/issues/46623)
  on all clusters (as for the pooled data plane): with Karmada some clusters can
  be left with 0 frontend replicas, and their requests must be served by the
  global pool.
- metrics-server on every member (as for `hpa-none`).
- A small Linux VM next to the clusters for the control plane (2 vCPU, 4 GB, 20 GB):
  never inside one of the 6 clusters (it would consume CPU and energy of a
  measured cluster). `karmadactl` has no Windows build.
- Network: VM → member apiservers (6443); operator machine → VM (32443 for the Karmada
  apiserver, 6443 for the K3s host apiserver used by the host check).
- The 6 member kubeconfigs on the VM.
- Permission to install Karmada's service account in each member (join) and
  to let Karmada manage the 8 Deployments temporarily during the Karmada
  cells.

## 4. Installation (on the VM, about 1 hour)

### 4.1 Host cluster and CLI

```bash
curl -sfL https://get.k3s.io | sh -
export HOSTKC=/etc/rancher/k3s/k3s.yaml
V=v1.19.0
curl -LO https://github.com/karmada-io/karmada/releases/download/$V/karmadactl-linux-amd64.tgz
curl -LO https://github.com/karmada-io/karmada/releases/download/$V/karmadactl-linux-amd64.tgz.sha256
sha256sum -c karmadactl-linux-amd64.tgz.sha256
tar -xzf karmadactl-linux-amd64.tgz && sudo install karmadactl /usr/local/bin/
karmadactl version
```

### 4.2 Control plane, pinned to v1.19.0

The reference page shows `v0.0.0-master` as image defaults: pin images and
CRDs explicitly. `VM_IP` is the address the operator machine reaches.

```bash
sudo karmadactl init --kubeconfig $HOSTKC \
  --karmada-apiserver-advertise-address VM_IP \
  --crds https://github.com/karmada-io/karmada/releases/download/v1.19.0/crds.tar.gz \
  --karmada-aggregated-apiserver-image docker.io/karmada/karmada-aggregated-apiserver:v1.19.0 \
  --karmada-controller-manager-image  docker.io/karmada/karmada-controller-manager:v1.19.0 \
  --karmada-scheduler-image           docker.io/karmada/karmada-scheduler:v1.19.0 \
  --karmada-webhook-image             docker.io/karmada/karmada-webhook:v1.19.0 \
  --etcd-storage-mode hostPath --etcd-replicas 1
sudo k3s kubectl -n karmada-system get pods          # all Running
sudo cp /etc/karmada/karmada-apiserver.config ~/karmada.config && sudo chown $USER ~/karmada.config
export KARMADA=~/karmada.config
```

If Karmada was installed before on the VM: `sudo rm -rf /var/lib/karmada-etcd`
first (otherwise `x509: certificate signed by unknown authority`).

### 4.3 Members, Push mode

```bash
REG=(x DE FR PL US-VA SG NO)
for i in 1 2 3 4 5 6; do
  KC=~/dmos-iac/kubeconfig-ms-cluster-0$i; CTX=$(kubectl --kubeconfig $KC config current-context)
  karmadactl join ms-cluster-0$i --kubeconfig $KARMADA \
    --cluster-kubeconfig $KC --cluster-context $CTX --cluster-region ${REG[$i]}
done
kubectl --kubeconfig $KARMADA get clusters      # 6 rows, MODE Push, READY True
```

The cluster names must be the harness names (`ms-cluster-0N`): the policy
selects clusters by name.

### 4.4 Addons and the two scheduler flags

```bash
karmadactl addons enable karmada-metrics-adapter --kubeconfig $HOSTKC --karmada-kubeconfig $KARMADA \
  --karmada-metrics-adapter-image docker.io/karmada/karmada-metrics-adapter:v1.19.0
for n in 01 02 03 04 05 06; do
  KC=~/dmos-iac/kubeconfig-ms-cluster-$n; CTX=$(kubectl --kubeconfig $KC config current-context)
  karmadactl addons enable karmada-scheduler-estimator -C ms-cluster-$n \
    --member-kubeconfig $KC --member-context $CTX \
    --kubeconfig $HOSTKC --karmada-kubeconfig $KARMADA \
    --karmada-scheduler-estimator-image docker.io/karmada/karmada-scheduler-estimator:v1.19.0
done
sudo k3s kubectl -n karmada-system edit deploy karmada-scheduler
#   add to the karmada-scheduler container args:
#     --enable-scheduler-estimator=true
#     --enable-empty-workload-propagation=true
kubectl --kubeconfig $KARMADA get apiservice v1beta1.metrics.k8s.io      # Available True
```

Why `--enable-empty-workload-propagation=true`: §2. Without it a cluster
given 0 replicas leaves the binding and its Deployment is deleted (or left
running with its old replicas, outside Karmada's control). The harness
refuses to start a Karmada cell without the flag (host check) and, at every
sample, checks that every binding still lists all 6 clusters.

### 4.5 On the operator machine

```bash
scp VM:~/karmada.config ~/dmos-iac/karmada.config                 # admin credentials: never in the repo
scp VM:/etc/rancher/k3s/k3s.yaml ~/dmos-iac/karmada-host.yaml     # then set server: https://VM_IP:6443
export DMOS_KARMADA_KUBECONFIG=~/dmos-iac/karmada.config
export DMOS_KARMADA_HOST_KUBECONFIG=~/dmos-iac/karmada-host.yaml
python experiments/karmada_baseline.py check      # must end with: check: ok
```

`check` verifies: the 6 clusters registered, Ready, Push; the metrics API;
the namespace (absent, or present with the skip label); on the host: the two
scheduler flags, one ready estimator per member, the image versions
(v1.19.0), failover and NoExecute eviction off. If the operator machine cannot
reach the host apiserver, `DMOS_KARMADA_SKIP_HOST_CHECK=1` lets the cell run (the
run gets the warning `karmada_host_unchecked`; the per-sample binding check
stays).

## 5. What happens in a Karmada cell

1. **prep** (every cell, every arm): `karmada_baseline.py release` —
   idempotent; without `DMOS_KARMADA_KUBECONFIG` it only checks that no member
   Deployment has `karmada.io/managed`. If something stays managed, the cell
   does not start.
2. **pre snapshot** with `--expect-no-karmada` (every arm): no Deployment
   managed by Karmada.
3. **apply** `--variant fhpa|agg`:
   - nothing of ours left in the Karmada namespace, no member Deployment
     managed, all 48 Deployments present;
   - the Deployment spec (without replicas) must be identical in the 6
     clusters, otherwise Overwrite would roll the pods out: it stops and lists
     the differing fields;
   - templates exported from cluster 01 (no runtime fields, no
     `deployment.kubernetes.io/revision`, no `last-applied-configuration`,
     replicas = FederatedHPA minimum), then the PropagationPolicy, created in
     the Karmada apiserver (documented migration order);
   - waits (≤ 300 s) until every binding is `Scheduled` and `FullyApplied`,
     lists the 6 clusters, every member Deployment is managed, has the
     replicas of the binding and is ready; a changed `uid` or revision (pods
     recreated or rolled out) is an error;
   - then the 8 FederatedHPAs. Record: `karmada_baseline.json → apply`
     (initial placement, manifests, component versions).
4. **trace** every 15 s during the load (`karmada_trace.jsonl`): binding
   placement per cluster, FederatedHPA current / desired replicas and
   `ScalingActive`, cluster readiness.
5. **verify** after the load: every FederatedHPA `ScalingActive=True`, all 48
   Deployments still managed, no cluster ever dropped from a binding; from the
   trace: replica range, clusters used, whether the FederatedHPAs scaled.
6. **post snapshot**, then **release**: delete the FederatedHPAs; check that
   every binding and every Work (found by the binding's permanent-id) has
   `preserveResourcesOnDeletion: true` — if not, the templates are **not**
   deleted; delete the 8 templates; wait until no member Deployment is managed
   and all 48 still exist; delete the policy.

Provenance checks (`manifest.json`, docs/RUN-PROVENANCE.md):
`karmada_baseline_missing`, `karmada_not_adopted`, `karmada_pod_restarted`,
`karmada_not_verified`, `karmada_cluster_dropped`, `fhpa_not_active`,
`karmada_lost_management` (errors); `karmada_cluster_not_ready`,
`fhpa_never_scaled`, `karmada_host_unchecked`, `karmada_version_unexpected`,
`karmada_not_released` (warnings); `karmada_present` (error, any non-Karmada
arm with a managed Deployment).

## 6. Manual test

With DMOS off, after `bash experiments/clean_reset.sh`:

```bash
python experiments/karmada_baseline.py apply --variant fhpa --out /tmp/kb.json
python experiments/karmada_baseline.py status          # managed_members: 48 entries, bindings, fhpa
kubectl --kubeconfig ~/dmos-iac/kubeconfig-ms-cluster-01 -n online-boutique get pods -L pod-template-hash
python experiments/karmada_baseline.py release --out /tmp/kb.json
python experiments/karmada_baseline.py status          # managed_members: []
bash experiments/run_one.sh reach-pool-ps con          # DMOS scales and nobody reverts it
```

On the VM, the same picture from Karmada:
`karmadactl --kubeconfig $KARMADA get deploy -n online-boutique --operation-scope=all`
(`ADOPTION=Y` while managed, `N` after the release).

## 7. Running the Karmada arms

```bash
bash experiments/run_one.sh karmada-fhpa con
bash experiments/run_one.sh reach-pool-ps con          # right after: the gate finds nothing managed
bash experiments/run_one.sh karmada-fhpa con-tosg      # hot spot that moves
```

## 8. Uninstall

```bash
python experiments/karmada_baseline.py release
python experiments/karmada_baseline.py status          # managed_members: [] — only then:
for n in 01 02 03 04 05 06; do
  KC=~/dmos-iac/kubeconfig-ms-cluster-$n; CTX=$(kubectl --kubeconfig $KC config current-context)
  karmadactl unjoin ms-cluster-$n --kubeconfig $KARMADA --cluster-kubeconfig $KC --cluster-context $CTX
done
sudo karmadactl deinit --kubeconfig $HOSTKC --purge-namespace
sudo rm -rf /var/lib/karmada-etcd /etc/karmada
bash experiments/clean_reset.sh && bash experiments/preflight_runs.sh
```

Never `unjoin` while `status` lists managed Deployments: unjoin deletes what
Karmada propagated to that member, i.e. Online Boutique.

## Sources

Documentation (v1.19): [installation](https://karmada.io/docs/installation/install-with-cli),
[karmadactl init](https://karmada.io/docs/reference/karmadactl/karmadactl-commands/karmadactl_init),
[join](https://karmada.io/docs/reference/karmadactl/karmadactl-commands/karmadactl_join),
[addons enable](https://karmada.io/docs/reference/karmadactl/karmadactl-commands/karmadactl_addons_enable),
[cluster registration](https://karmada.io/docs/userguide/clustermanager/cluster-registration),
[PropagationPolicy](https://karmada.io/docs/userguide/scheduling/propagation-policy) and its
[API reference](https://karmada.io/docs/reference/karmada-api/policy-resources/propagation-policy-v1alpha1),
[FederatedHPA](https://karmada.io/docs/userguide/autoscaling/federatedhpa) and its
[API reference](https://karmada.io/docs/reference/karmada-api/autoscaling-resources/federated-hpa-v1alpha1),
[resource metrics tutorial](https://karmada.io/docs/tutorials/autoscaling-with-resource-metrics),
[scheduler estimator](https://karmada.io/docs/userguide/scheduling/scheduler-estimator),
[karmada-scheduler](https://karmada.io/docs/reference/components/karmada-scheduler),
[karmada-controller-manager](https://karmada.io/docs/reference/components/karmada-controller-manager),
[migrate in batch](https://karmada.io/docs/administrator/migration/migrate-in-batch),
[resource migration tutorial](https://karmada.io/docs/tutorials/resource-migration),
[object association mapping](https://karmada.io/docs/reference/object-association-mapping),
[cluster failover](https://karmada.io/docs/userguide/failover/cluster-failover),
[compatibility](https://karmada.io/docs/administrator/compatibility/),
[release v1.19.0](https://github.com/karmada-io/karmada/releases/tag/v1.19.0).
Source at tag `v1.19.0` (where the docs are silent): `pkg/util/constants.go`,
`pkg/util/names/names.go`, `pkg/apis/work/v1alpha2/{well_known_constants,binding_types}.go`,
`pkg/scheduler/core/{common,util,generic_scheduler}.go`,
`cmd/scheduler/app/options/options.go`, `pkg/controllers/binding/{binding_controller,common}.go`,
`pkg/controllers/execution/execution_controller.go`, `pkg/util/objectwatcher/objectwatcher.go`,
`pkg/controllers/federatedhpa/federatedhpa_controller.go`,
`pkg/karmadactl/addons/{metricsadapter,estimator}`, `pkg/karmadactl/get/get.go`,
`pkg/controllers/namespace/namespace_sync_controller.go`.
