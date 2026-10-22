#!/usr/bin/env bash
# snapshot-config.sh — salva uno snapshot testuale della config ClusterMesh
# corrente (annotation Service + endpoint-sync + stato mesh) per archiviazione.
# Uso:  bash deployments/clustermesh-overflow/snapshot-config.sh > CONFIG_Vx_SNAPSHOT.txt
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_JSON="${REPO_ROOT}/.cache/test-env.json"
NS="online-boutique"
kc_of() { local kc="$1"; kc="${kc/#\~/$HOME}";
          [[ "$kc" != /* && "$kc" != [A-Za-z]:* ]] && kc="${REPO_ROOT}/${kc}"; printf '%s' "$kc"; }

echo "ClusterMesh config snapshot — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
first_kc="$(kc_of "$(jq -r '.clusters[0].kubeconfig_path' "$ENV_JSON" | tr -d '\r')")"
echo
echo "== cilium-config: clustermesh-enable-endpoint-sync =="
kubectl --kubeconfig "$first_kc" -n kube-system get cm cilium-config \
    -o jsonpath='{.data.clustermesh-enable-endpoint-sync}{"\n"}'
echo
echo "== annotation Service ($NS) =="
kubectl --kubeconfig "$first_kc" -n "$NS" get svc \
    -o custom-columns='NAME:.metadata.name,GLOBAL:.metadata.annotations.service\.cilium\.io/global,AFFINITY:.metadata.annotations.service\.cilium\.io/affinity,LEGACY:.metadata.annotations.io\.cilium/global-service'
echo
echo "== CiliumClusterwideEnvoyConfig presenti =="
kubectl --kubeconfig "$first_kc" get ciliumclusterwideenvoyconfigs.cilium.io 2>/dev/null || echo "(nessuno)"
echo
echo "== frontend ClusterIP + n. backend (locale vs cross-cluster) =="
fip=$(kubectl --kubeconfig "$first_kc" -n "$NS" get svc frontend -o jsonpath='{.spec.clusterIP}')
echo "frontend ClusterIP = $fip"
kubectl --kubeconfig "$first_kc" -n kube-system exec ds/cilium -- cilium-dbg service list 2>/dev/null \
  | awk -v ip="$fip" '/^[0-9]+[[:space:]]/{inblk=($0~ip)?1:0} inblk' | head -20
echo
echo "== Cilium version =="
kubectl --kubeconfig "$first_kc" -n kube-system get ds cilium -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
