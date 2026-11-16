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
#   - readiness period 15s → 5s, fail 6 → 18 (stessa tolleranza: 90 s di probe
#                                 fallite prima di togliere il pod dagli endpoint):
#                                 con period 15s un pod NUOVO diventava Ready fino a
#                                 15 s dopo essere pronto, e lo scaling (di tutti gli
#                                 arm che scalano) serviva lo spike in ritardo. La
#                                 liveness resta a 15s/6 (e' quella che evita il
#                                 kill dei pod throttlati). (2026-10-28,
#                                 docs/PREDICTIVE-SCALER.md)
#
# Uso:
#   bash experiments/apply_resource_hardening.sh           # applica
#   bash experiments/apply_resource_hardening.sh --verify  # solo verifica
# =============================================================================
set -uo pipefail
# Cluster: experiments/lib/fleet.sh (.cache/test-env.json, o DMOS_KUBECONFIG_DIR)
source "$(dirname "${BASH_SOURCE[0]}")/lib/fleet.sh"; fleet_load
NS="online-boutique"

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
#   currencyservice 200m→500m: chiamato 3.4 volte per richiesta; le connessioni
#   gRPC dei frontend restano agganciate a un pod (il Service bilancia
#   connessioni, non richieste), quindi le repliche nuove ricevono poco traffico
#   e un pod resta al 50-94% di throttling sotto picco mentre gli altri stanno
#   all'11-24% (Kubephos, 450 req/s, 28/10/2026). Il limit piu' alto da' a quel
#   pod la CPU per servirle; vale per tutti gli arm. (2026-10-28)
CPU_TARGETS=(
  "recommendationservice:server:500m"
  "currencyservice:server:500m"
)
# LIVENESS rilassata (container "server"): timeout 5s, period 15s, fail 6
# READINESS: timeout 4s, period 5s, fail 18 (stessa tolleranza di 90 s, Ready prima)
PROBE_SVCS=(frontend cartservice productcatalogservice checkoutservice \
            recommendationservice currencyservice adservice shippingservice)

mem_patch() { printf '{"spec":{"template":{"spec":{"containers":[{"name":"%s","resources":{"limits":{"memory":"%s"}}}]}}}}' "$1" "$2"; }
cpu_patch() { printf '{"spec":{"template":{"spec":{"containers":[{"name":"%s","resources":{"limits":{"cpu":"%s"}}}]}}}}' "$1" "$2"; }
probe_patch() { cat <<'EOF'
{"spec":{"template":{"spec":{"containers":[{"name":"server",
 "livenessProbe":{"timeoutSeconds":5,"periodSeconds":15,"failureThreshold":6},
 "readinessProbe":{"timeoutSeconds":4,"periodSeconds":5,"failureThreshold":18}}]}}}}
EOF
}

if [[ "${1:-}" == "--verify" ]]; then
  echo "=== Verifica memoria + liveness ==="
  for i in "${!FLEET_NAMES[@]}"; do
    kc="${FLEET_KCS[$i]}"; echo "${FLEET_NAMES[$i]}:"
    for t in "${MEM_TARGETS[@]}"; do IFS=: read -r svc c m <<<"$t"
      cur=$(kubectl --kubeconfig "$kc" -n "$NS" get deploy "$svc" -o jsonpath="{.spec.template.spec.containers[?(@.name=='$c')].resources.limits.memory}" 2>/dev/null)
      printf "  %-24s mem=%s (atteso %s)\n" "$svc" "${cur:-?}" "$m"
    done
  done
  exit 0
fi

ok=0; fail=0
for i in "${!FLEET_NAMES[@]}"; do
  kc="${FLEET_KCS[$i]}"
  [[ -f "$kc" ]] || { echo "=== ${FLEET_NAMES[$i]} === [x] kubeconfig assente: $kc"; fail=$((fail+1)); continue; }
  echo "=== ${FLEET_NAMES[$i]} ==="
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
