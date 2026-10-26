#!/usr/bin/env bash
# =============================================================================
# apply-overflow.sh — passa alla config v2 (overflow cross-cluster) sui 6 cluster
# =============================================================================
# Cosa fa (idempotente), su OGNI cluster:
#   1. frontend  → service.cilium.io/affinity=local   (servi locale, failover su unhealthy)
#   2. backend   → toglie le annotation global (→ cluster-local: ogni frontend usa i SUOI backend)
#   3. cilium-config: clustermesh-enable-endpoint-sync=true  + rollout restart cilium
#   4. applica il CiliumClusterwideEnvoyConfig (outlier detection sul frontend)
#
# Rollback:  revert-overflow.sh
# Verifica:  bash experiments/preflight_runs.sh   (check [4b])
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_JSON="${REPO_ROOT}/.cache/test-env.json"
CCEC="${SCRIPT_DIR}/ccec-frontend-outlier.yaml"
NS="online-boutique"
FRONTEND="frontend"
BACKENDS=(cartservice checkoutservice productcatalogservice recommendationservice)

command -v kubectl >/dev/null || { echo "kubectl mancante"; exit 2; }
command -v jq      >/dev/null || { echo "jq mancante"; exit 2; }
[[ -f "$ENV_JSON" ]] || { echo ".cache/test-env.json assente"; exit 2; }
[[ -f "$CCEC" ]]     || { echo "manifest CCEC mancante: $CCEC"; exit 2; }

kc_of() { local kc="$1"; kc="${kc/#\~/$HOME}";
          [[ "$kc" != /* && "$kc" != [A-Za-z]:* ]] && kc="${REPO_ROOT}/${kc}";
          printf '%s' "$kc"; }

echo "### Config v2 (overflow) — applico su tutti i cluster ###"
while IFS=$'\t' read -r name kcp; do
  KC="$(kc_of "$kcp")"
  [[ -f "$KC" ]] || { echo "[$name] kubeconfig mancante: $KC — skip"; continue; }
  echo "── $name ──"

  # 1) frontend: global resta, affinity → local
  kubectl --kubeconfig "$KC" -n "$NS" annotate svc "$FRONTEND" \
      service.cilium.io/global=true service.cilium.io/affinity=local --overwrite >/dev/null \
      && echo "  [ok] frontend: global=true, affinity=local"

  # 2) backend → local (rimuovo tutte le annotation global/affinity, legacy e moderne)
  for b in "${BACKENDS[@]}"; do
    kubectl --kubeconfig "$KC" -n "$NS" annotate svc "$b" \
        service.cilium.io/global- io.cilium/global-service- \
        service.cilium.io/affinity- io.cilium/shared-service- \
        service.cilium.io/shared- >/dev/null 2>&1
  done
  echo "  [ok] backend ${BACKENDS[*]} → local (global rimosso)"

  # 3) endpoint-sync = true + rollout cilium-agent E clustermesh-apiserver.
  #    NB: il flag va riletto da ENTRAMBI: l'agent IMPORTA gli endpoint remoti,
  #    ma il clustermesh-apiserver li ESPORTA verso la KVStore. Senza riavviare
  #    l'apiserver, nessun cluster esporta → i global service restano local-only.
  kubectl --kubeconfig "$KC" -n kube-system patch configmap cilium-config --type merge \
      -p '{"data":{"clustermesh-enable-endpoint-sync":"true"}}' >/dev/null \
      && echo "  [ok] cilium-config: clustermesh-enable-endpoint-sync=true"
  kubectl --kubeconfig "$KC" -n kube-system rollout restart ds/cilium >/dev/null \
      && echo "  [..] rollout cilium-agent avviato"
  kubectl --kubeconfig "$KC" -n kube-system rollout restart deploy/clustermesh-apiserver >/dev/null 2>&1 \
      && echo "  [..] rollout clustermesh-apiserver avviato (export endpoint)"

  # 4) CCEC outlier detection
  kubectl --kubeconfig "$KC" apply -f "$CCEC" >/dev/null \
      && echo "  [ok] CiliumClusterwideEnvoyConfig applicato"
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')

echo
echo "### Attendo il rollout dei cilium-agent (può richiedere 1-2 min) ###"
while IFS=$'\t' read -r name kcp; do
  KC="$(kc_of "$kcp")"; [[ -f "$KC" ]] || continue
  kubectl --kubeconfig "$KC" -n kube-system rollout status ds/cilium --timeout=180s >/dev/null 2>&1 \
      && echo "  [ok] $name cilium pronto" || echo "  [!] $name rollout non confermato (controlla)"
  kubectl --kubeconfig "$KC" -n kube-system rollout status deploy/clustermesh-apiserver --timeout=180s >/dev/null 2>&1 \
      && echo "  [ok] $name apiserver pronto" || echo "  [!] $name apiserver non confermato"
done < <(jq -r '.clusters[] | [.name, .kubeconfig_path] | @tsv' "$ENV_JSON" | tr -d '\r')

echo
echo "FATTO. Verifica ora con:  bash experiments/preflight_runs.sh   (check [4b])"
echo "Atteso: endpoint-sync=true e 'frontend: N backend (>1 → endpoint cross-cluster ATTIVI)'."
