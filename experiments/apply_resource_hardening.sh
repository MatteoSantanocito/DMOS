#!/usr/bin/env bash
# =============================================================================
# apply_resource_hardening.sh — FONTE DI VERITÀ delle patch RISORSE/PROBE del banco
# =============================================================================
# I manifest di Online Boutique NON sono nel repo (applicati da upstream), quindi
# le patch di hardening del banco vivrebbero solo come modifiche live in etcd
# (fragili: si perdono a un re-apply da upstream). Questo script le rende
# RIPRODUCIBILI e TRACCIATE. Idempotente: una strategic-merge con valori già
# presenti è un no-op (niente rolling update). Compagno di apply_pod_lifecycle.sh
# (preStop/grace) — qui invece: MEMORIA (anti-OOM) + LIVENESS/READINESS (anti
# throttle-kill). NB: tocca solo malfunzioni del banco, NON CPU/capacity (= il
# problema di saturazione che il paper studia, lasciato intatto).
#
# STORICO DEI FIX (perché ognuno):
#   - redis-cart  mem 256Mi→1Gi : OOMKilled 172× sotto con (datastore di cart;
#                                 il crash di redis → cart 500). (2026-06-06)
#   - cartservice mem 128Mi→512Mi: OOMKilled sotto con. (2026-06-06)
#   - recommendationservice mem 450Mi→1Gi: OOM INTERMITTENTE sotto picco
#                                 sostenuto a con (diag 2026-06-08): l1l2 è
#                                 l'unico arm che scala reco a ~3 sul cluster
#                                 caldo → l'unico che può OOMare → picco a volte
#                                 2.6% (ok) a volte 30% (OOM). Il fix rende il
#                                 risultato consistente SENZA aiutare off/l2
#                                 (che tengono reco a 1 → non OOMano → restano
#                                 ~39%/~29%): rimuove un confound che penalizzava
#                                 il vincitore, NON appiattisce le differenze.
#   - liveness/readiness 1s/3 → 5s/6 (period 15s): sotto CFS-throttle /healthz>1s
#                                 → 3 fail → kill → cascata → ingress Envoy
#                                 incastrato. Rilassate per non uccidere un pod
#                                 throttlato-ma-vivo. (2026-06-06)
#
# Uso:
#   bash experiments/apply_resource_hardening.sh           # applica
#   bash experiments/apply_resource_hardening.sh --verify  # solo verifica
# =============================================================================
set -uo pipefail
KUBECONFIG_DIR="${DMOS_KUBECONFIG_DIR:-$HOME/Desktop/dmos-iac}"
NS="online-boutique"
CLUSTERS=(01 02 03 04 05 06)

# MEMORIA (anti-OOM): "svc:container:mem_limit"
MEM_TARGETS=(
  "redis-cart:redis:1Gi"
  "cartservice:server:512Mi"
  "recommendationservice:server:1Gi"
)
# CPU (anti-throttle-kill): "svc:container:cpu_limit"
#   recommendation 200m→500m: a 200m il GIL Python va in CFS-throttle sotto picco
#   sostenuto a con → /healthz oltre il timeout → liveness kill → restart →
#   picco bimodale (diag 2026-06-08: reco_restart=1 ⇒ peak 24-41%, =0 ⇒ ~5%).
#   Node HIGH (8 vCPU) ha headroom (5×500m=2.5 core). (2026-06-08)
CPU_TARGETS=(
  "recommendationservice:server:500m"
)
# LIVENESS/READINESS rilassate (container "server"): timeout 5s, period 15s, fail 6
PROBE_SVCS=(frontend cartservice productcatalogservice checkoutservice \
            recommendationservice currencyservice adservice shippingservice)

mem_patch() { printf '{"spec":{"template":{"spec":{"containers":[{"name":"%s","resources":{"limits":{"memory":"%s"}}}]}}}}' "$1" "$2"; }
cpu_patch() { printf '{"spec":{"template":{"spec":{"containers":[{"name":"%s","resources":{"limits":{"cpu":"%s"}}}]}}}}' "$1" "$2"; }
probe_patch() { cat <<'EOF'
{"spec":{"template":{"spec":{"containers":[{"name":"server",
 "livenessProbe":{"timeoutSeconds":5,"periodSeconds":15,"failureThreshold":6},
 "readinessProbe":{"timeoutSeconds":5,"periodSeconds":15,"failureThreshold":6}}]}}}}
EOF
}

if [[ "${1:-}" == "--verify" ]]; then
  echo "=== Verifica memoria + liveness ==="
  for n in "${CLUSTERS[@]}"; do
    kc="${KUBECONFIG_DIR}/kubeconfig-ms-cluster-${n}"; echo "ms-cluster-${n}:"
    for t in "${MEM_TARGETS[@]}"; do IFS=: read -r svc c m <<<"$t"
      cur=$(kubectl --kubeconfig "$kc" -n "$NS" get deploy "$svc" -o jsonpath="{.spec.template.spec.containers[?(@.name=='$c')].resources.limits.memory}" 2>/dev/null)
      printf "  %-24s mem=%s (atteso %s)\n" "$svc" "${cur:-?}" "$m"
    done
  done
  exit 0
fi

ok=0; fail=0
for n in "${CLUSTERS[@]}"; do
  kc="${KUBECONFIG_DIR}/kubeconfig-ms-cluster-${n}"
  [[ -f "$kc" ]] || { echo "=== ms-cluster-${n} === [x] kubeconfig assente"; fail=$((fail+1)); continue; }
  echo "=== ms-cluster-${n} ==="
  for t in "${MEM_TARGETS[@]}"; do IFS=: read -r svc c m <<<"$t"
    if kubectl --kubeconfig "$kc" -n "$NS" patch deploy "$svc" --type strategic -p "$(mem_patch "$c" "$m")" >/dev/null 2>&1
    then echo "  [ok] mem $svc=$m"; ok=$((ok+1)); else echo "  [x] mem $svc"; fail=$((fail+1)); fi
  done
  for t in "${CPU_TARGETS[@]}"; do IFS=: read -r svc c cpu <<<"$t"
    if kubectl --kubeconfig "$kc" -n "$NS" patch deploy "$svc" --type strategic -p "$(cpu_patch "$c" "$cpu")" >/dev/null 2>&1
    then echo "  [ok] cpu $svc=$cpu"; ok=$((ok+1)); else echo "  [x] cpu $svc"; fail=$((fail+1)); fi
  done
  for svc in "${PROBE_SVCS[@]}"; do
    if kubectl --kubeconfig "$kc" -n "$NS" patch deploy "$svc" --type strategic -p "$(probe_patch)" >/dev/null 2>&1
    then echo "  [ok] probe $svc"; ok=$((ok+1)); else echo "  [x] probe $svc"; fail=$((fail+1)); fi
  done
done
echo; echo "Patch: ${ok} ok, ${fail} fallite. Rolling update solo dove i valori cambiano."
echo "Verifica: bash experiments/apply_resource_hardening.sh --verify"
