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
# Servizi: il frontend e TUTTI i servizi che DMOS co-scala
# (config/services.yaml → application.co_scaled_services), perche' DMOS li
# toglie negli scale-down. Fino al 31/10/2026 la lista ne copriva 5: currency,
# ad (grace 5 s) e shipping venivano tolti senza drenaggio, e ogni run DMOS
# aveva 8-25 richieste fallite nel cooldown.
# Un test (tests/test_pod_lifecycle_script.py) verifica che la lista resti
# allineata a services.yaml. DMOS_LIFECYCLE_SERVICES="a b" la sostituisce.
#
# La patch e' uno strategic merge sul container di nome "server": prima di
# applicarla lo script controlla che il Deployment abbia quel container (con
# un nome diverso lo strategic merge AGGIUNGEREBBE un secondo container).
# Dopo le patch attende il rollout (--no-wait per non attendere).
#
# Idempotente: ri-eseguibile senza effetti collaterali.
# Ogni patch cambia il pod template -> rolling update del deployment (atteso).
#
# Uso:
#   bash experiments/apply_pod_lifecycle.sh            # applica
#   bash experiments/apply_pod_lifecycle.sh --verify   # solo verifica
# =============================================================================
set -uo pipefail

# Cluster: experiments/lib/fleet.sh (.cache/test-env.json, o DMOS_KUBECONFIG_DIR)
source "$(dirname "${BASH_SOURCE[0]}")/lib/fleet.sh"; fleet_load
NS="online-boutique"
SERVICES=(frontend cartservice productcatalogservice checkoutservice recommendationservice
          currencyservice adservice shippingservice)
if [[ -n "${DMOS_LIFECYCLE_SERVICES:-}" ]]; then
  read -r -a SERVICES <<< "$DMOS_LIFECYCLE_SERVICES"
fi
WAIT=1
for a in "$@"; do [[ "$a" == "--no-wait" ]] && WAIT=0; done

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
    -o jsonpath='{.spec.template.spec.terminationGracePeriodSeconds}s/preStop{.spec.template.spec.containers[?(@.name=="server")].lifecycle.preStop.sleep.seconds}s' 2>/dev/null
}

has_server_container() {  # $1 kubeconfig, $2 deployment
  kubectl --kubeconfig "$1" -n "$NS" get deploy "$2" \
    -o jsonpath='{.spec.template.spec.containers[*].name}' 2>/dev/null | tr ' ' '\n' | grep -qx server
}

if [[ "${1:-}" == "--verify" ]]; then
  echo "=== Verifica lifecycle (grace/preStop) ==="
  for i in "${!FLEET_NAMES[@]}"; do
    kc="${FLEET_KCS[$i]}"
    printf "  %s: " "${FLEET_NAMES[$i]}"
    for svc in "${SERVICES[@]}"; do
      printf "%s=%s  " "$svc" "$(verify_one "$kc" "$svc")"
    done
    echo
  done
  exit 0
fi

ok=0; fail=0
for i in "${!FLEET_NAMES[@]}"; do
  kc="${FLEET_KCS[$i]}"
  if [[ ! -f "$kc" ]]; then
    echo "=== ${FLEET_NAMES[$i]} === [x] kubeconfig non trovato: $kc"
    fail=$((fail+${#SERVICES[@]})); continue
  fi
  echo "=== ${FLEET_NAMES[$i]} ==="
  for svc in "${SERVICES[@]}"; do
    if ! has_server_container "$kc" "$svc"; then
      echo "  [skip] $svc — nessun container 'server' (o Deployment assente): non patchato"
      fail=$((fail+1)); continue
    fi
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

if [[ "$WAIT" == "1" ]]; then
  echo
  echo "Attendo il rollout (pod nuovi pronti) ..."
  for i in "${!FLEET_NAMES[@]}"; do
    kc="${FLEET_KCS[$i]}"
    [[ -f "$kc" ]] || continue
    for svc in "${SERVICES[@]}"; do
      if ! kubectl --kubeconfig "$kc" -n "$NS" rollout status "deploy/$svc" --timeout=240s >/dev/null 2>&1; then
        echo "  [x] ${FLEET_NAMES[$i]}/$svc: rollout non completato in 240 s"
        fail=$((fail+1))
      fi
    done
  done
fi

echo
echo "Patch applicate: ${ok} ok, ${fail} fallite (attese $(( ${#FLEET_NAMES[@]} * ${#SERVICES[@]} )) ok)"
echo "Verifica (30s/preStop15s atteso per ogni servizio):"
echo "  bash experiments/apply_pod_lifecycle.sh --verify"
[[ "$fail" -eq 0 ]]
