#!/bin/bash
# ============================================================================
# enable_hubble_http.sh
# Abilita Hubble con metriche HTTP L7 su tutti e 3 i cluster.
#
# Cosa fa:
#   1. Aggiorna il ConfigMap cilium-config per abilitare hubble-metrics
#   2. Riavvia il DaemonSet cilium per applicare le modifiche
#   3. Verifica che le metriche siano disponibili
#
# NOTA: Hubble deve già essere installato (viene con Cilium).
#       Questo script lo CONFIGURA per esporre metriche HTTP L7.
#
# Prerequisiti:
#   - kubectl configurato
#   - Cilium installato come CNI (verifica: kubectl -n kube-system get pods | grep cilium)
#
# Usage:
#   bash scripts/enable_hubble_http.sh
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CLUSTERS=(cluster1 cluster2 cluster3)
CLUSTER_IPS=(192.168.1.245 192.168.1.246 192.168.1.247)

log()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m $*"; }

# ── Metriche Hubble da abilitare ───────────────────────────────────────────
# httpV2: metriche HTTP L7 con label dettagliate (necessarie per DMOS)
#   - labelsContext: aggiunge source/destination namespace e workload
# drop: pacchetti scartati
# tcp: connessioni TCP
# flow: flussi generici
# icmp: ping/icmp
# dns: query DNS
HUBBLE_METRICS="httpV2:exemplars=true;labelsContext=source_namespace,source_workload,destination_namespace,destination_workload,traffic_direction,destination_port drop tcp flow icmp dns"
HUBBLE_METRICS_PORT="9091"

for i in "${!CLUSTERS[@]}"; do
    CLUSTER="${CLUSTERS[$i]}"
    IP="${CLUSTER_IPS[$i]}"
    KUBECONFIG_PATH="$ROOT_DIR/.kube/${CLUSTER}.yaml"

    echo ""
    log "============================================================"
    log "  Configuro Hubble su: $CLUSTER  (IP: $IP)"
    log "============================================================"

    if [ ! -f "$KUBECONFIG_PATH" ]; then
        err "kubeconfig non trovato: $KUBECONFIG_PATH — salto"
        continue
    fi
    export KUBECONFIG="$KUBECONFIG_PATH"

    if ! kubectl cluster-info --request-timeout=5s &>/dev/null; then
        err "Cluster $CLUSTER non raggiungibile — salto"
        continue
    fi

    # ── Controlla lo stato attuale di Hubble ──────────────────────────
    echo ""
    log "[1] Stato attuale Hubble su $CLUSTER:"
    kubectl -n kube-system get configmap cilium-config -o jsonpath='{.data.hubble-enabled}' 2>/dev/null \
        && echo "" || warn "Chiave hubble-enabled non trovata nel configmap"
    kubectl -n kube-system get configmap cilium-config -o jsonpath='{.data.hubble-metrics}' 2>/dev/null \
        && echo "" || warn "Chiave hubble-metrics non trovata — Hubble HTTP non configurato"

    # ── Metodo 1: Aggiorna ConfigMap direttamente ──────────────────────
    echo ""
    log "[2] Aggiorno cilium-config per abilitare Hubble HTTP..."

    # Patch il ConfigMap cilium-config
    kubectl -n kube-system patch configmap cilium-config \
        --type merge \
        -p "{
            \"data\": {
                \"hubble-enabled\": \"true\",
                \"hubble-listen-address\": \":4244\",
                \"hubble-metrics-server\": \":$HUBBLE_METRICS_PORT\",
                \"hubble-metrics\": \"$HUBBLE_METRICS\",
                \"hubble-export-file-max-size-mb\": \"10\",
                \"hubble-export-file-max-backups\": \"5\"
            }
        }"

    log "ConfigMap cilium-config aggiornato ✓"

    # ── Riavvia Cilium DaemonSet per applicare le modifiche ───────────
    echo ""
    log "[3] Riavvio DaemonSet cilium per applicare le modifiche..."
    kubectl -n kube-system rollout restart daemonset/cilium
    log "Attendo che cilium sia pronto (max 120s)..."
    kubectl -n kube-system rollout status daemonset/cilium --timeout=120s

    # ── Verifica che il Service hubble-metrics esista ─────────────────
    echo ""
    log "[4] Verifico/creo Service hubble-metrics..."

    # Crea un Service per esporre le metriche Hubble (se non esiste)
    kubectl -n kube-system get service hubble-metrics &>/dev/null && \
        log "Service hubble-metrics già presente" || \
        kubectl -n kube-system apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: hubble-metrics
  namespace: kube-system
  labels:
    k8s-app: hubble-metrics
spec:
  selector:
    k8s-app: cilium
  ports:
    - name: hubble-metrics
      port: $HUBBLE_METRICS_PORT
      targetPort: $HUBBLE_METRICS_PORT
      protocol: TCP
  type: ClusterIP
EOF

    # ── Verifica che le metriche siano disponibili ────────────────────
    echo ""
    log "[5] Verifica metriche Hubble su $IP:$HUBBLE_METRICS_PORT..."
    sleep 10  # Aspetta che cilium si riavvii completamente

    # Trova un pod cilium per fare port-forward del test
    CILIUM_POD=$(kubectl -n kube-system get pods -l k8s-app=cilium -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$CILIUM_POD" ]; then
        log "Controllo metriche su pod $CILIUM_POD..."
        # Usa exec per fare un curl dall'interno del pod verso localhost:9091
        if kubectl -n kube-system exec "$CILIUM_POD" -- \
            curl -s --max-time 5 "http://localhost:$HUBBLE_METRICS_PORT/metrics" 2>/dev/null | \
            grep -q "hubble_http"; then
            log "✅ hubble_http_* metriche disponibili sul pod $CILIUM_POD"
        else
            warn "⚠️  hubble_http_* non ancora disponibili — potrebbe servire traffico L7 attivo"
            warn "   Prova: kubectl -n kube-system exec $CILIUM_POD -- curl localhost:9091/metrics | grep hubble"
        fi
    else
        warn "Nessun pod cilium trovato — verifica l'installazione"
    fi

    # ── Verifica che Prometheus stia già scraping Hubble ─────────────
    echo ""
    log "[6] Verifica scraping Hubble in Prometheus ($IP:30090)..."
    sleep 3
    RESULT=$(curl -s --max-time 5 \
        "http://$IP:30090/api/v1/query?query=hubble_http_requests_total" 2>/dev/null)
    if echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if d.get('data',{}).get('result') else 'EMPTY')" 2>/dev/null | grep -q "OK"; then
        log "✅ Prometheus già scraping hubble_http_requests_total su $CLUSTER"
    else
        warn "⚠️  hubble_http_requests_total non ancora in Prometheus"
        warn "   Aspetta 1-2 cicli di scraping (15-30s) e ricontrolla con:"
        warn "   curl 'http://$IP:30090/api/v1/query?query=hubble_http_requests_total'"
        warn "   O verifica che prometheus-helm-values.yaml abbia il job 'hubble'"
    fi

    log "Configurazione Hubble completata su $CLUSTER ✓"
done

echo ""
log "============================================================"
log "  RIEPILOGO CONFIGURAZIONE HUBBLE"
log "============================================================"
echo ""
echo "  Metriche abilitate:"
echo "    • hubble_http_requests_total{destination_namespace, destination_workload,"
echo "      source_namespace, source_workload, traffic_direction, method, status_code}"
echo "    • hubble_http_request_duration_seconds_bucket{...} (per p95 latency)"
echo "    • hubble_drop_total, hubble_tcp_*, hubble_flows_*, hubble_dns_*"
echo ""
echo "  Prossimi passi:"
echo "    1. Genera traffico verso online-boutique per vedere le metriche"
echo "    2. Verifica: python scripts/check_prometheus.py"
echo "    3. Test query manuale:"
echo "       curl 'http://192.168.1.245:30090/api/v1/query?query=sum(rate(hubble_http_requests_total{destination_namespace=\"online-boutique\"}[1m]))'"
echo ""
warn "NOTA: se usi helm per Cilium, è preferibile l'upgrade via helm:"
warn "  helm upgrade cilium cilium/cilium -n kube-system --reuse-values \\"
warn "    --set hubble.enabled=true \\"
warn "    --set hubble.metrics.enableOpenMetrics=true \\"
warn "    --set 'hubble.metrics.enabled={httpV2:labelsContext=source_namespace\\,source_workload\\,destination_namespace\\,destination_workload\\,traffic_direction,drop,tcp,flow,icmp,dns}'"
