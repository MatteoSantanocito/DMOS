#!/usr/bin/env bash
# =============================================================================
# revert-overflow.sh — ripristina la config v1 (local-serving) sui 6 cluster
# =============================================================================
# Inverte apply-overflow.sh:
#   1. frontend  → affinity=none  (com'era in v1)
#   2. backend   → ri-aggiunge global + affinity=local (com'erano in v1)
#   3. cilium-config: clustermesh-enable-endpoint-sync=false + rollout cilium
#   4. rimuove il CiliumClusterwideEnvoyConfig
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_JSON="${REPO_ROOT}/.cache/test-env.json"
CCEC="${SCRIPT_DIR}/ccec-frontend-outlier.yaml"
NS="online-boutique"
BACKENDS=(cartservice checkoutservice productcatalogservice recommendationservice)

kc_of() { local kc="$1"; kc="${kc/#\~/$HOME}";
          [[ "$kc" != /* && "$kc" != [A-Za-z]:* ]] && kc="${REPO_ROOT}/${kc}";
          printf '%s' "$kc"; }

echo "### Ripristino config v1 (local-serving) ###"
while IFS=$'\t' read -r name kcp; do
  KC="$(kc_of "$kcp")"; [[ -f "$KC" ]] || { echo "[$name] kubeconfig mancante — skip"; continue; }
  echo "── $name ──"
  kubectl --kubeconfig "$KC" -n "$NS" annotate svc frontend \
      service.cilium.io/affinity=none --overwrite >/dev/null && echo "  [ok] frontend affinity=none"
  for b in "${BACKENDS[@]}"; do
    kubectl --kubeconfig "$KC" -n "$NS" annotate svc "$b" \
        service.cilium.io/global=true service.cilium.io/affinity=local --overwrite >/dev/null 2>&1
  done
  echo "  [ok] backend ${BACKENDS[*]} → global+affinity=local (v1)"
  kubectl --kubeconfig "$KC" -n kube-system patch configmap cilium-config --type merge \
      -p '{"data":{"clustermesh-enable-endpoint-sync":"false"}}' >/dev/null \
      && echo "  [ok] endpoint-sync=false"
  kubectl --kubeconfig "$KC" -n kube-system rollout restart ds/cilium >/dev/null
  kubectl --kubeconfig "$KC" -n kube-system rollout restart deploy/clustermesh-apiserver >/dev/null 2>&1
  kubectl --kubeconfig "$KC" delete -f "$CCEC" >/dev/null 2>&1 && echo "  [ok] CCEC rimosso"
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')

echo
echo "Attendo rollout cilium..."
while IFS=$'\t' read -r name kcp; do
  KC="$(kc_of "$kcp")"; [[ -f "$KC" ]] || continue
  kubectl --kubeconfig "$KC" -n kube-system rollout status ds/cilium --timeout=180s >/dev/null 2>&1 \
      && echo "  [ok] $name" || echo "  [!] $name rollout non confermato"
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')
echo "FATTO. Verifica: bash experiments/preflight_runs.sh (atteso endpoint-sync=false, frontend 1 backend)."
