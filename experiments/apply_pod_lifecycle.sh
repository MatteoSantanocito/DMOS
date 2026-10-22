#!/usr/bin/env bash
# =============================================================================
# apply_pod_lifecycle.sh — Fix B: lifecycle dei pod Online Boutique
# =============================================================================
# Applica a tutti i deployment gestiti da DMOS, su tutti e 6 i cluster:
#   - preStop con drain di 15s (azione nativa k8s `sleep`, richiede k8s >=1.30;
#     i cluster sono v1.36 -> ok anche con immagini distroless senza shell).
#     Il pod, una volta marcato Terminating, resta fermo 15s prima del SIGTERM:
#     in quei 15s la rimozione dall'endpoint si propaga a Envoy/Cilium e le
#     richieste gRPC in volo si svuotano -> niente connessioni spezzate.
#   - terminationGracePeriodSeconds = 30s (cart/productcatalog/recommendation
#     erano a 5s: SIGKILL prima che il drain finisse).
#
# Idempotente: ri-eseguibile senza effetti collaterali.
# Ogni patch cambia il pod template -> rolling update del deployment (atteso).
#
# Uso:
#   bash experiments/apply_pod_lifecycle.sh            # applica
#   bash experiments/apply_pod_lifecycle.sh --verify   # solo verifica
# =============================================================================
set -uo pipefail

KUBECONFIG_DIR="${DMOS_KUBECONFIG_DIR:-$HOME/Desktop/dmos-iac}"
NS="online-boutique"
SERVICES=(frontend cartservice productcatalogservice checkoutservice recommendationservice)
CLUSTERS=(01 02 03 04 05 06)

read -r -d '' PATCH <<'EOF'
{
  "spec": {
    "template": {
      "spec": {
        "terminationGracePeriodSeconds": 30,
        "containers": [
          { "name": "server", "lifecycle": { "preStop": { "sleep": { "seconds": 15 } } } }
        ]
      }
    }
  }
}
EOF

verify_one() {
  local kc="$1" svc="$2"
  kubectl --kubeconfig "$kc" -n "$NS" get deploy "$svc" \
    -o jsonpath='{.spec.template.spec.terminationGracePeriodSeconds}s/preStop{.spec.template.spec.containers[0].lifecycle.preStop.sleep.seconds}s' 2>/dev/null
}

if [[ "${1:-}" == "--verify" ]]; then
  echo "=== Verifica lifecycle (grace/preStop) ==="
  for n in "${CLUSTERS[@]}"; do
    kc="${KUBECONFIG_DIR}/kubeconfig-ms-cluster-${n}"
    printf "  ms-cluster-%s: " "$n"
    for svc in "${SERVICES[@]}"; do
      printf "%s=%s  " "$svc" "$(verify_one "$kc" "$svc")"
    done
    echo
  done
  exit 0
fi

ok=0; fail=0
for n in "${CLUSTERS[@]}"; do
  kc="${KUBECONFIG_DIR}/kubeconfig-ms-cluster-${n}"
  if [[ ! -f "$kc" ]]; then
    echo "=== ms-cluster-${n} === [x] kubeconfig non trovato: $kc"
    fail=$((fail+5)); continue
  fi
  echo "=== ms-cluster-${n} ==="
  for svc in "${SERVICES[@]}"; do
    if kubectl --kubeconfig "$kc" -n "$NS" patch deployment "$svc" \
         --type strategic --patch "$PATCH" >/dev/null 2>&1; then
      echo "  [ok] $svc"
      ok=$((ok+1))
    else
      echo "  [x]  $svc — patch fallita"
      fail=$((fail+1))
    fi
  done
done

echo
echo "Patch applicate: ${ok} ok, ${fail} fallite (atteso 30 ok)"
echo "Rolling update in corso sui deployment. Verifica fra ~1 min con:"
echo "  bash experiments/apply_pod_lifecycle.sh --verify"
