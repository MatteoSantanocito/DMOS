#!/usr/bin/env bash
# clean_reset.sh — reset PULITO e profondo del fleet a 6 cluster prima dei run.
# Obiettivo: partire sempre da uno stato non degradato (niente Envoy listener
# hang, niente repliche residue), così i risultati non sono contaminati.
#
# USO:
#   bash experiments/clean_reset.sh --check    # SOLO verifica i 6 ingress :30080 e
#                                              # recupera SOLO i cluster non-200 (envoy
#                                              # hang/500). Veloce, NON tocca i sani,
#                                              # NON resetta i deployment. ← TRA un run
#                                              # e l'altro, per garantire no-500.
#   bash experiments/clean_reset.sh            # reset standard (deployments→1 +
#                                              # health-gate con recovery reattiva)
#   bash experiments/clean_reset.sh --deep     # come sopra MA prima riavvia in
#                                              # modo PROATTIVO cilium-envoy+frontend
#                                              # su TUTTI i cluster (azzera i listener
#                                              # hang latenti). ← a INIZIO sessione /
#                                              # dopo un run uscito degradato.
#
# Cosa fa:
#   1) uccide eventuale DMOS zombie su :9090
#   2) [--deep] rollout restart cilium-envoy + frontend su tutti i 6 cluster
#   3) scala tutti i deployment di online-boutique a 1 replica su tutti i cluster
#   4) attende lo scale-down (pod extra terminati)
#   5) health-gate :30080 con recovery automatica dei cluster non-200
#   6) settle + verifica finale (6/6 a 200 + pod tutti Running)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

MODE="standard"
case "${1:-}" in
  --deep)  MODE="deep"  ;;
  --check) MODE="check" ;;
  "")      MODE="standard" ;;
  *) echo "uso: clean_reset.sh [--check|--deep]"; exit 2 ;;
esac
DEEP=0; [ "$MODE" = "deep" ] && DEEP=1
NS="online-boutique"
SETTLE="${SETTLE:-60}"
# Fleet (IP + kubeconfig): letti da .cache/test-env.json (gitignored, vedi
# preflight_runs.sh / scripts/render-runtime-files.sh) oppure via env:
#   CLUSTERS_SPEC="01:<ip> 02:<ip> ..."          coppie NN:IP
#   KCBASE=<dir>/kubeconfig-ms-cluster           base dei kubeconfig (senza -NN)
#   DMOS_IAC=<dir>                               da cui KCBASE=$DMOS_IAC/kubeconfig-ms-cluster
ENV_JSON=".cache/test-env.json"
CLUSTERS=()
if [ -n "${CLUSTERS_SPEC:-}" ]; then
  read -r -a CLUSTERS <<< "$CLUSTERS_SPEC"
elif [ -f "$ENV_JSON" ] && command -v jq >/dev/null 2>&1; then
  while IFS= read -r entry; do [ -n "$entry" ] && CLUSTERS+=("$entry"); done \
    < <(jq -r '.clusters[] | (.name | sub("^ms-cluster-"; "")) + ":" + .ip' "$ENV_JSON" | tr -d '\r')
fi
[ ${#CLUSTERS[@]} -gt 0 ] || { echo "[ERR] fleet sconosciuto: manca $ENV_JSON (genera con scripts/render-runtime-files.sh) e CLUSTERS_SPEC non e' impostata"; exit 2; }
KCBASE="${KCBASE:-${DMOS_IAC:+$DMOS_IAC/kubeconfig-ms-cluster}}"
if [ -z "$KCBASE" ] && [ -f "$ENV_JSON" ] && command -v jq >/dev/null 2>&1; then
  KCBASE="$(jq -r '.clusters[0].kubeconfig_path' "$ENV_JSON" | tr -d '\r' | sed -E 's/-[0-9]+$//')"
fi
[ -n "$KCBASE" ] || { echo "[ERR] KCBASE non determinabile: esporta KCBASE o DMOS_IAC, oppure genera $ENV_JSON"; exit 2; }
kc() { echo "$KCBASE-$1"; }

kill_dmos() {
  echo "==> [1] stop DMOS (solo i processi python -m src.dmos_main)"
  # Come kill_dmos in lib/run_common.sh: nessun altro processo Python toccato.
  pwsh -NoProfile -Command 'Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "python*" -and $_.CommandLine -match "-m\s+src\.dmos_main" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }' >/dev/null 2>&1
  for i in $(seq 1 12); do
    [ "$(curl -s -o /dev/null -m2 -w '%{http_code}' http://localhost:9090/metrics 2>/dev/null)" = "000" ] && { echo "    :9090 libera"; return 0; }
    sleep 1
  done
  echo "    [warn] :9090 ancora occupata"
}

restart_dataplane() {  # $1 = cluster — recovery GENTILE: solo cilium-envoy + frontend.
  # NON tocca il cilium-agent: il suo restart cicla il networking del nodo e fa
  # riavviare i POD BACKEND (exit 255, connessioni severed) → 5xx sotto carico.
  # Usato come primo tentativo; l'escalation con l'agent è in restart_dataplane_hard.
  local n="$1"; local kc; kc="$(kc "$n")"
  kubectl --kubeconfig "$kc" -n kube-system  rollout restart ds/cilium-envoy      >/dev/null 2>&1
  kubectl --kubeconfig "$kc" -n "$NS"        rollout restart deploy/frontend      >/dev/null 2>&1
  kubectl --kubeconfig "$kc" -n kube-system  rollout status  ds/cilium-envoy  --timeout=120s >/dev/null 2>&1
  kubectl --kubeconfig "$kc" -n "$NS"        rollout status  deploy/frontend  --timeout=120s >/dev/null 2>&1
}
restart_dataplane_hard() {  # $1 = cluster — ESCALATION: + cilium-agent (cura l'hang
  # xDS cronico, ma disturba i backend → usato SOLO se il gentile non basta).
  local n="$1"; local kc; kc="$(kc "$n")"
  kubectl --kubeconfig "$kc" -n kube-system  rollout restart ds/cilium            >/dev/null 2>&1
  kubectl --kubeconfig "$kc" -n kube-system  rollout status  ds/cilium        --timeout=150s >/dev/null 2>&1
  restart_dataplane "$n"
}

deep_recover_all() {
  echo "==> [2] DEEP: rollout restart cilium-envoy + frontend su tutti i cluster"
  for entry in "${CLUSTERS[@]}"; do
    local n="${entry%%:*}"
    echo "    cluster-$n ..."
    restart_dataplane "$n"
  done
  echo "    data-plane riavviato ovunque"
}

reset_deployments() {
  echo "==> [3] scale-down: tutti i deployment di '$NS' → 1 replica"
  for entry in "${CLUSTERS[@]}"; do
    local n="${entry%%:*}"; local kc; kc="$(kc "$n")"
    if kubectl --kubeconfig "$kc" -n "$NS" scale deployment --all --replicas=1 >/dev/null 2>&1; then
      # il loadgenerator di Online Boutique resta spento (traffico interno non k6)
      if kubectl --kubeconfig "$kc" -n "$NS" get deploy loadgenerator >/dev/null 2>&1; then
        kubectl --kubeconfig "$kc" -n "$NS" scale deploy loadgenerator --replicas=0 >/dev/null 2>&1
      fi
      echo "    cluster-$n: scalato a 1 (loadgenerator a 0)"
    else
      echo "    [warn] cluster-$n: scale fallito (cluster raggiungibile?)"
    fi
  done
  echo "==> [4] attendo terminazione pod extra (max 90s)"
  for s in $(seq 1 18); do
    local pending=0 entry n kc extra
    for entry in "${CLUSTERS[@]}"; do
      n="${entry%%:*}"; kc="$(kc "$n")"
      # somma (replicas - 1) sui deployment con >1 replica
      extra=$(kubectl --kubeconfig "$kc" -n "$NS" get deploy -o jsonpath='{range .items[*]}{.spec.replicas} {end}' 2>/dev/null \
              | tr ' ' '\n' | awk '{if($1>1) s+=($1-1)} END{print s+0}')
      pending=$((pending + extra))
    done
    [ "$pending" -eq 0 ] && { echo "    tutti i deployment a 1 replica"; return 0; }
    sleep 5
  done
  echo "    [warn] alcuni deployment ancora >1 (procedo comunque)"
}

ingress_code() {  # $1 = ip
  curl -s -o /dev/null -m6 -w '%{http_code}' "http://$1:30080/" 2>/dev/null
}
wait_for_stable() {  # $1 = ip, $2 = timeout_s — polla finché STABILE (stable_200) o timeout
  local ip="$1" tmo="${2:-180}" t=0
  while [ "$t" -lt "$tmo" ]; do
    stable_200 "$ip" && return 0
    t=$((t+15))   # stable_200 impiega ~10s; +pausa
  done
  return 1
}
stable_200() {  # $1 = ip — 3 probe a 5s; sano se gli ULTIMI 2 sono 200
  #   (tollera un 000 transitorio iniziale; BLOCCA il flapping 200↔000/down)
  local ip="$1" a b c
  a=$(ingress_code "$ip"); sleep 5
  b=$(ingress_code "$ip"); sleep 5
  c=$(ingress_code "$ip")
  RESULT_TRIPLE="$a/$b/$c"
  [ "$b" = "200" ] && [ "$c" = "200" ]
}
ip_of() { local entry; for entry in "${CLUSTERS[@]}"; do [ "${entry%%:*}" = "$1" ] && { echo "${entry#*:}"; return; }; done; }

health_gate() {
  # Loop unico: stable_200 (3 probe) è l'UNICO rilevatore, e il suo elenco di
  # cluster instabili guida il recovery. Re-loop fino a 3 round: se un cluster
  # cade DOPO essere sembrato sano (flap/timing), il round successivo lo coglie
  # e lo recupera (prima censimento e verify usavano sonde diverse e non si
  # parlavano → blocco senza azione di recovery).
  echo "==> [5] health-gate ingress :30080 (rilevatore stabile + recovery in loop)"
  local attempt entry n ip bad
  for attempt in 1 2 3; do
    bad=()
    for entry in "${CLUSTERS[@]}"; do
      n="${entry%%:*}"; ip="${entry#*:}"
      stable_200 "$ip" || bad+=("$n")
    done
    if [ ${#bad[@]} -eq 0 ]; then
      echo "    6/6 cluster 200 stabile OK (round $attempt)"; return 0
    fi
    echo "    round $attempt — instabili/giù: ${bad[*]} → recovery"
    for n in "${bad[@]}"; do
      ip="$(ip_of "$n")"
      if [ "$attempt" -eq 1 ]; then
        echo "      cluster-$n: restart envoy+frontend (gentile) → attendo 200 STABILE (≤210s)"
        restart_dataplane "$n"
      else
        echo "      cluster-$n: ESCALATION restart cilium-agent+envoy+frontend → attendo 200 STABILE (≤210s)"
        restart_dataplane_hard "$n"
      fi
      wait_for_stable "$ip" 210 \
        && echo "      cluster-$n: 200 stabile OK" \
        || echo "      cluster-$n: ancora instabile (riprovo al round successivo)"
    done
  done
  # verifica finale dopo i round
  bad=()
  for entry in "${CLUSTERS[@]}"; do
    n="${entry%%:*}"; stable_200 "${entry#*:}" || bad+=("$n")
  done
  [ ${#bad[@]} -eq 0 ] && { echo "    tutti i cluster stabili"; return 0; }
  echo "    [FAIL] cluster ancora instabili dopo 3 round: ${bad[*]}"; return 1
}
verify_final() {
  echo "==> [6] verifica finale (ingress STABILE + pod non-Running)"
  local entry n ip bad_pods ok=0
  for entry in "${CLUSTERS[@]}"; do
    n="${entry%%:*}"; ip="${entry#*:}"
    if stable_200 "$ip"; then ok=$((ok+1)); fi
    bad_pods=$(kubectl --kubeconfig "$(kc "$n")" -n "$NS" get pods --no-headers 2>/dev/null \
               | grep -vcE "Running|Completed")
    printf "    cluster-%s: ingress(3 probe)=%s  pod-non-Running=%s\n" "$n" "$RESULT_TRIPLE" "${bad_pods:-?}"
  done
  echo "    → $ok/6 ingress STABILE a 200"
  [ "$ok" -eq 6 ] && return 0 || return 1
}

echo "############ clean_reset.sh ($MODE) ############"

# --check: solo health-gate + verifica, niente kill/reset/settle. Per uso TRA i run.
# Esce 0 SOLO se 6/6 sano → con `&& run_one.sh` il run parte solo a fleet OK.
if [ "$MODE" = "check" ]; then
  health_gate || true
  if verify_final; then
    echo "############ FLEET OK — puoi lanciare il prossimo run ############"; exit 0
  else
    echo "############ ATTENZIONE: fleet NON sano — run BLOCCATO ############"; exit 1
  fi
fi

kill_dmos
[ $DEEP -eq 1 ] && deep_recover_all
reset_deployments
echo "==> settle ${SETTLE}s"; sleep "$SETTLE"
health_gate || true
if verify_final; then
  echo "############ FLEET PULITO — pronto per i run ############"; exit 0
else
  echo "############ ATTENZIONE: fleet NON completamente sano — run BLOCCATO ############"; exit 1
fi
