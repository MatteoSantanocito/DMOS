#!/usr/bin/env bash
# =============================================================================
# run_flash_incluster.sh — flash crowd EST-OVEST: 1 Job k6 per cluster -> Service
# `frontend`, con overflow GSLB ATTIVO. Volume totale costante, varia lo skew.
# =============================================================================
#
# MODELLO (vedi k6_flash_crowd_incluster.js):
#   - un Job k6 per cluster, schedulato sul worker, colpisce http://frontend
#     (ClusterIP, est-ovest) -> la CEC redistribuisce cross-cluster.
#   - il "peso" del cluster = il suo PEAK_RPS (frazione del totale 350).
#   - baseline uniforme; il regime modella la concentrazione del PICCO.
#
# USO:
#   bash experiments/run_flash_incluster.sh <regime> [run_label]
#     regime ∈ uni | bal | s50 | s65 | s75 | con     (default con)
#     run_label default: fc-incluster-<regime>
#
# PREREQUISITI: overflow sano (lancia prima `cd dmos-iac && python gslb_check.py`).
#   DMOS gia' in esecuzione se vuoi pesi GSLB = score (altrimenti pesi uniformi).
# ENV: KUBECONFIG_DIR (default: ricavata da .cache/test-env.json)
#      DMOS_PEAK_TOTAL(350)  DMOS_BASE_TOTAL(50)  DMOS_K6_IMAGE
#      DMOS_PHASE_TIMES(180,300,600,720,900)  DMOS_SLO_MS(2000)
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"; cd "$REPO_ROOT"
RESULTS_DIR="${REPO_ROOT}/experiments/results"; mkdir -p "$RESULTS_DIR"

NS="online-boutique"
# KUBECONFIG_DIR: env, altrimenti ricavata dal kubeconfig_path in
# .cache/test-env.json (gitignored, vedi preflight_runs.sh).
KUBECONFIG_DIR="${KUBECONFIG_DIR:-}"
if [ -z "$KUBECONFIG_DIR" ] && [ -f "${REPO_ROOT}/.cache/test-env.json" ] && command -v jq >/dev/null 2>&1; then
  KUBECONFIG_DIR="$(dirname "$(jq -r '.clusters[0].kubeconfig_path' "${REPO_ROOT}/.cache/test-env.json" | tr -d '\r')")"
fi
[ -n "$KUBECONFIG_DIR" ] || { echo "[ERR] KUBECONFIG_DIR non impostata e .cache/test-env.json assente — esporta KUBECONFIG_DIR=<dir dei kubeconfig-ms-cluster-NN> (es. \$HOME/dmos-iac)"; exit 2; }
K6_IMAGE="${DMOS_K6_IMAGE:-grafana/k6:latest}"
PEAK_TOTAL="${DMOS_PEAK_TOTAL:-350}"
BASE_TOTAL="${DMOS_BASE_TOTAL:-50}"
PHASE_TIMES="${DMOS_PHASE_TIMES:-180,300,600,720,900}"
SLO_MS="${DMOS_SLO_MS:-2000}"
CM_NAME="dmos-k6-fc-script"
CLUSTERS=(01 02 03 04 05 06)
REGIME="${1:-con}"
RUN_LABEL="${2:-fc-incluster-${REGIME}}"

# pesi di PICCO per regime (come run_one.sh)
case "$REGIME" in
  uni) W=(1 1 1 1 1 1);;
  bal) W=(40 15 12 12 11 10);;
  s50) W=(50 10 10 10 10 10);;
  s65) W=(65 7 7 7 7 7);;
  s75) W=(75 5 5 5 5 5);;
  con) W=(85 3 3 3 3 3);;
  *) echo "regime sconosciuto: $REGIME"; exit 2;;
esac
WSUM=0; for w in "${W[@]}"; do WSUM=$((WSUM+w)); done
TEST_DUR="${PHASE_TIMES##*,}"

kc() { echo "${KUBECONFIG_DIR}/kubeconfig-ms-cluster-$1"; }

echo "### run_flash_incluster — regime=$REGIME label=$RUN_LABEL (peak ${PEAK_TOTAL}rps) ###"
# baseline uniforme per cluster
BASE_RPS=$(awk "BEGIN{printf \"%d\", ($BASE_TOTAL/6)+0.5}")
echo "  baseline/cluster=${BASE_RPS}rps  | pesi picco: ${W[*]} (sum=$WSUM)"

cleanup() {
  for c in "${CLUSTERS[@]}"; do
    kubectl --kubeconfig "$(kc $c)" -n "$NS" delete job "dmos-k6-fc-$c" --ignore-not-found --wait=false >/dev/null 2>&1 || true
    kubectl --kubeconfig "$(kc $c)" -n "$NS" delete configmap "$CM_NAME" --ignore-not-found >/dev/null 2>&1 || true
  done
}
restore_loadgen() {
  for c in "${CLUSTERS[@]}"; do
    kubectl --kubeconfig "$(kc $c)" -n "$NS" scale deploy/loadgenerator --replicas=1 >/dev/null 2>&1 || true
  done
}
trap 'echo; echo "[interrotto] cleanup"; cleanup; restore_loadgen; exit 130' INT TERM

# loadgenerator OFF: e' anch'esso egress->frontend e falserebbe la domanda d'origine
echo "== spengo loadgenerator (clean demand) =="
for c in "${CLUSTERS[@]}"; do
  kubectl --kubeconfig "$(kc $c)" -n "$NS" scale deploy/loadgenerator --replicas=0 >/dev/null 2>&1 \
    && echo "  c$c loadgen->0" || echo "  c$c loadgen non scalato"
done

# 1) ConfigMap (script) + Job su ogni cluster
for i in "${!CLUSTERS[@]}"; do
  c="${CLUSTERS[$i]}"; KC="$(kc $c)"
  peak=$(awk "BEGIN{printf \"%d\", (${W[$i]}/$WSUM)*$PEAK_TOTAL+0.5}")
  [ "$peak" -lt 1 ] && peak=1
  echo "  c$c: peak=${peak}rps (w=${W[$i]})"
  kubectl --kubeconfig "$KC" -n "$NS" delete configmap "$CM_NAME" --ignore-not-found >/dev/null 2>&1
  kubectl --kubeconfig "$KC" -n "$NS" create configmap "$CM_NAME" \
    --from-file=k6_flash_crowd_incluster.js=experiments/k6_flash_crowd_incluster.js >/dev/null
  kubectl --kubeconfig "$KC" -n "$NS" delete job "dmos-k6-fc-$c" --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl --kubeconfig "$KC" -n "$NS" apply -f - >/dev/null <<EOF
apiVersion: batch/v1
kind: Job
metadata: {name: dmos-k6-fc-$c, namespace: $NS, labels: {app: dmos-k6-fc}}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    metadata: {labels: {app: dmos-k6-fc}}
    spec:
      restartPolicy: Never
      containers:
      - name: k6
        image: $K6_IMAGE
        command: ["k6","run","/scripts/k6_flash_crowd_incluster.js"]
        env:
        - {name: DMOS_RUN_LABEL, value: "$RUN_LABEL"}
        - {name: DMOS_CLUSTER, value: "ms-cluster-$c"}
        - {name: DMOS_TARGET_URL, value: "http://frontend"}
        - {name: DMOS_BASELINE_RPS, value: "$BASE_RPS"}
        - {name: DMOS_PEAK_RPS, value: "$peak"}
        - {name: DMOS_PHASE_TIMES, value: "$PHASE_TIMES"}
        - {name: DMOS_SLO_MS, value: "$SLO_MS"}
        resources: {requests: {cpu: "250m", memory: "256Mi"}, limits: {cpu: "2", memory: "2Gi"}}
        volumeMounts: [{name: scripts, mountPath: /scripts}]
      volumes: [{name: scripts, configMap: {name: $CM_NAME}}]
EOF
done
echo "  6 Job applicati. Durata ~$((TEST_DUR/60)) min + pull immagine."

# 2) attendi che TUTTI i Job siano terminali (poll)
echo "== attendo i 6 Job (poll ogni 20s) =="
deadline=$(( $(date +%s) + TEST_DUR + 600 ))
while :; do
  done=0
  for c in "${CLUSTERS[@]}"; do
    s=$(kubectl --kubeconfig "$(kc $c)" -n "$NS" get job "dmos-k6-fc-$c" -o jsonpath='{.status.succeeded}{"/"}{.status.failed}' 2>/dev/null)
    case "$s" in 1/*|*/1|1/) done=$((done+1));; esac
  done
  echo "  $(date +%H:%M:%S) terminati: $done/6"
  [ "$done" -ge 6 ] && break
  [ "$(date +%s)" -gt "$deadline" ] && { echo "  [warn] timeout: proseguo coi summary disponibili"; break; }
  sleep 20
done

# 3) raccogli i summary JSON dai log
TS="$(date +%Y%m%d_%H%M%S)"; OUTDIR="${RESULTS_DIR}/${TS}_${RUN_LABEL}"; mkdir -p "$OUTDIR"
for c in "${CLUSTERS[@]}"; do
  kubectl --kubeconfig "$(kc $c)" -n "$NS" logs "job/dmos-k6-fc-$c" 2>/dev/null \
    | awk '/===DMOS_FC_JSON_BEGIN===/{f=1;next} /===DMOS_FC_JSON_END===/{f=0} f' \
    | tr -d '\r' > "$OUTDIR/cluster-$c.json"
  [ -s "$OUTDIR/cluster-$c.json" ] && echo "  c$c summary ok" || echo "  c$c summary MANCANTE"
done

# 4) aggrega (SLO/goodput esatti, p95 pesato sul picco)
python - "$OUTDIR" "$RUN_LABEL" "$REGIME" <<'PY'
import json, glob, os, sys
outdir, label, regime = sys.argv[1], sys.argv[2], sys.argv[3]
tot_req=tot_viol=0; good=0.0; pk_num=0.0; pk_den=0; per=[]
for f in sorted(glob.glob(os.path.join(outdir,"cluster-*.json"))):
    try: d=json.load(open(f))
    except Exception: continue
    t=d.get("totals",{}); g=d.get("global") or {}; pk=d.get("peak") or {}
    tot_req+=t.get("requests",0); tot_viol+=t.get("slo_violations",0)
    good+=g.get("success_rps",0) or 0
    if pk.get("requests"): pk_num+=pk["p95"]*pk["requests"]; pk_den+=pk["requests"]
    per.append({"cluster":d.get("cluster"),"req":t.get("requests",0),
                "slo%":round(100*t.get("slo_violation_rate",0),2),
                "peak_p95":round(pk.get("p95",0)) if pk else None,
                "valid":d.get("validity",{}).get("run_valid")})
agg={"run_label":label,"regime":regime,
     "global":{"total_requests":tot_req,
               "slo_violation_rate":round(tot_viol/tot_req,4) if tot_req else None,
               "goodput_rps":round(good,1),
               "peak_p95_ms_weighted":round(pk_num/pk_den) if pk_den else None},
     "per_cluster":per}
json.dump(agg, open(os.path.join(outdir,"aggregate.json"),"w"), indent=2)
print("\n  === AGGREGATO (%s, %s) ==="%(label,regime))
print("  richieste totali: %d"%tot_req)
print("  SLO violation:    %s%%"%(round(100*tot_viol/tot_req,2) if tot_req else "n/a"))
print("  goodput:          %.1f req/s"%good)
print("  peak p95 (pesato):%s ms"%(round(pk_num/pk_den) if pk_den else "n/a"))
print("  per-cluster:")
for p in per: print("    %-14s req=%-7d slo=%-5s%% peak_p95=%-5s valid=%s"%(
    p["cluster"],p["req"],p["slo%"],p["peak_p95"],p["valid"]))
print("  -> %s/aggregate.json"%outdir)
PY

cleanup
restore_loadgen
echo "  loadgenerator ripristinato a 1 su tutti i cluster."
echo "### FATTO: $RUN_LABEL ###"
