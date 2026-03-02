#!/bin/bash
# ============================================================================
# diagnose_cluster.sh
# Controlla COSA è già installato su ogni cluster PRIMA di eseguire
# qualsiasi setup script. Risponde alle domande:
#
#   1. Come è installato Prometheus? (helm chart / manifesti manuali / operator)
#   2. kube-state-metrics è già presente?
#   3. Cilium/Hubble è installato e come?
#   4. Hubble metrics è già abilitato?
#   5. C'è già un service hubble-metrics?
#
# NON modifica nulla — solo lettura.
#
# Usage:
#   bash scripts/diagnose_cluster.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CLUSTERS=(cluster1 cluster2 cluster3)
CLUSTER_IPS=(192.168.1.245 192.168.1.246 192.168.1.247)

log()  { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
ok()   { echo -e "    \033[92m✅ $*\033[0m"; }
fail() { echo -e "    \033[91m❌ $*\033[0m"; }
info() { echo -e "    \033[96mℹ️  $*\033[0m"; }

for i in "${!CLUSTERS[@]}"; do
    CLUSTER="${CLUSTERS[$i]}"
    IP="${CLUSTER_IPS[$i]}"
    KUBECONFIG_PATH="$ROOT_DIR/.kube/${CLUSTER}.yaml"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CLUSTER: $CLUSTER  ($IP)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "$KUBECONFIG_PATH" ]; then
        fail "kubeconfig non trovato: $KUBECONFIG_PATH"
        continue
    fi
    export KUBECONFIG="$KUBECONFIG_PATH"

    if ! kubectl cluster-info --request-timeout=5s &>/dev/null; then
        fail "Cluster non raggiungibile"
        continue
    fi

    # ── 1. PROMETHEUS ──────────────────────────────────────────────────
    echo ""
    echo "  [PROMETHEUS]"

    # Cerca release helm
    HELM_RELEASES=$(helm list -A 2>/dev/null | grep -E "prometheus|kube-prom" || true)
    if [ -n "$HELM_RELEASES" ]; then
        ok "Helm releases trovate:"
        echo "$HELM_RELEASES" | while read line; do echo "      $line"; done

        # Identifica chart specifico
        if helm list -A 2>/dev/null | grep -q "kube-prometheus-stack"; then
            warn "    → kube-prometheus-stack (operator) — NON usare setup_prometheus_cluster.sh!"
            warn "      Per aggiungere scrape Hubble, usa un additionalScrapeConfig separato."
        elif helm list -A 2>/dev/null | grep -q "prometheus-community/prometheus"; then
            ok "    → prometheus standalone (chart prometheus-community/prometheus)"
            info "   setup_prometheus_cluster.sh è SICURO (farà upgrade)"
        fi
    else
        # Cerca pod prometheus manualmente installati
        PROM_PODS=$(kubectl get pods -A 2>/dev/null | grep -i prometheus | grep -v kube-state | grep -v node-exporter | grep -v alertmanager || true)
        if [ -n "$PROM_PODS" ]; then
            warn "Prometheus trovato ma NON installato via helm:"
            echo "$PROM_PODS" | while read line; do echo "      $line"; done
            warn "    → Installazione manuale — NON usare setup_prometheus_cluster.sh!"
            warn "      Usa invece il ConfigMap extra scrape mostrato sotto."
        else
            info "Prometheus non trovato — setup_prometheus_cluster.sh installerà fresh"
        fi
    fi

    # Verifica raggiungibilità NodePort 30090
    if curl -s --max-time 3 "http://$IP:30090/-/healthy" &>/dev/null; then
        ok "NodePort 30090 raggiungibile su $IP"
        # Mostra versione prometheus
        VERSION=$(curl -s --max-time 3 "http://$IP:30090/api/v1/status/buildinfo" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['version'])" 2>/dev/null || echo "sconosciuta")
        info "Versione Prometheus: $VERSION"
    else
        fail "NodePort 30090 NON raggiungibile su $IP"
    fi

    # ── 2. KUBE-STATE-METRICS ─────────────────────────────────────────
    echo ""
    echo "  [KUBE-STATE-METRICS]"

    KSM_PODS=$(kubectl get pods -A -l app.kubernetes.io/name=kube-state-metrics 2>/dev/null | grep -v "^NAMESPACE" || true)
    KSM_PODS2=$(kubectl get pods -A -l app=kube-state-metrics 2>/dev/null | grep -v "^NAMESPACE" || true)

    if [ -n "$KSM_PODS" ] || [ -n "$KSM_PODS2" ]; then
        ok "kube-state-metrics già presente"
        [ -n "$KSM_PODS" ] && echo "$KSM_PODS" | while read line; do echo "      $line"; done
        [ -n "$KSM_PODS2" ] && echo "$KSM_PODS2" | while read line; do echo "      $line"; done
        warn "    → NON aggiungere kube-state-metrics nel helm values (già c'è)!"
        warn "      In prometheus-helm-values.yaml, imposta: kube-state-metrics.enabled: false"
    else
        info "kube-state-metrics NON presente"
        info "→ Lascia kube-state-metrics.enabled: true in prometheus-helm-values.yaml"
    fi

    # Verifica che kube_node_status_capacity sia disponibile in Prometheus
    RESULT=$(curl -s --max-time 5 "http://$IP:30090/api/v1/query?query=kube_node_status_capacity%7Bresource%3D%22cpu%22%7D" 2>/dev/null)
    if echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data',{}).get('result') else 1)" 2>/dev/null; then
        ok "kube_node_status_capacity disponibile in Prometheus"
    else
        fail "kube_node_status_capacity NON in Prometheus — kube-state-metrics non scraped"
    fi

    # ── 3. CILIUM / HUBBLE ────────────────────────────────────────────
    echo ""
    echo "  [CILIUM / HUBBLE]"

    # Come è installato Cilium?
    CILIUM_HELM=$(helm list -n kube-system 2>/dev/null | grep cilium || true)
    if [ -n "$CILIUM_HELM" ]; then
        ok "Cilium installato via Helm:"
        echo "      $CILIUM_HELM"
        info "→ Per Hubble: usa 'helm upgrade cilium ...' (NON patch ConfigMap)"
        CILIUM_HELM_VERSION=$(helm list -n kube-system 2>/dev/null | grep cilium | awk '{print $NF}')
        info "  Versione chart: $CILIUM_HELM_VERSION"
    else
        CILIUM_PODS=$(kubectl get pods -n kube-system -l k8s-app=cilium 2>/dev/null | grep -v "^NAME" || true)
        if [ -n "$CILIUM_PODS" ]; then
            warn "Cilium installato ma NON via Helm (manifesti manuali o altro):"
            echo "$CILIUM_PODS" | head -3 | while read line; do echo "      $line"; done
            info "→ Puoi usare patch ConfigMap (enable_hubble_http.sh)"
        else
            fail "Cilium NON trovato — verifica il CNI installato"
        fi
    fi

    # Hubble già abilitato?
    echo ""
    echo "    Stato Hubble:"
    HUBBLE_ENABLED=$(kubectl -n kube-system get configmap cilium-config -o jsonpath='{.data.hubble-enabled}' 2>/dev/null || echo "non trovato")
    HUBBLE_METRICS=$(kubectl -n kube-system get configmap cilium-config -o jsonpath='{.data.hubble-metrics}' 2>/dev/null || echo "")
    HUBBLE_METRICS_SERVER=$(kubectl -n kube-system get configmap cilium-config -o jsonpath='{.data.hubble-metrics-server}' 2>/dev/null || echo "")

    if [ "$HUBBLE_ENABLED" = "true" ]; then
        ok "hubble-enabled: true"
    else
        fail "hubble-enabled: $HUBBLE_ENABLED"
    fi

    if [ -n "$HUBBLE_METRICS" ]; then
        ok "hubble-metrics configurato: $HUBBLE_METRICS"
        if echo "$HUBBLE_METRICS" | grep -q "http"; then
            ok "HTTP metrics ABILITATO"
        else
            warn "HTTP metrics NON in hubble-metrics — aggiungere 'httpV2'"
        fi
    else
        fail "hubble-metrics: non configurato — Hubble HTTP non attivo"
    fi

    if [ -n "$HUBBLE_METRICS_SERVER" ]; then
        ok "hubble-metrics-server: $HUBBLE_METRICS_SERVER"
    else
        warn "hubble-metrics-server non configurato (default: non esposto)"
    fi

    # Service hubble-metrics
    HM_SVC=$(kubectl -n kube-system get service hubble-metrics 2>/dev/null | grep -v "^NAME" || true)
    if [ -n "$HM_SVC" ]; then
        ok "Service hubble-metrics presente: $HM_SVC"
    else
        warn "Service hubble-metrics NON presente — Prometheus non può scrape Hubble"
    fi

    # Metriche hubble disponibili in Prometheus?
    RESULT=$(curl -s --max-time 5 "http://$IP:30090/api/v1/query?query=hubble_http_requests_total" 2>/dev/null)
    if echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('data',{}).get('result') else 1)" 2>/dev/null; then
        ok "hubble_http_requests_total disponibile in Prometheus"
    else
        fail "hubble_http_requests_total NON in Prometheus"
    fi

    # ── 4. NAMESPACE E DEPLOYMENTS ────────────────────────────────────
    echo ""
    echo "  [ONLINE BOUTIQUE]"
    PODS=$(kubectl -n online-boutique get pods 2>/dev/null | grep -v "^NAME" | wc -l || echo "0")
    ok "Pod in online-boutique: $PODS"

    echo ""
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RIEPILOGO — cosa fare in base ai risultati:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  SCENARIO A — Prometheus NON installato:"
echo "    → bash scripts/setup_prometheus_cluster.sh   (sicuro)"
echo ""
echo "  SCENARIO B — Prometheus già installato via helm prometheus-community/prometheus:"
echo "    → helm upgrade con -f config/prometheus-helm-values.yaml  (sicuro)"
echo "    → Oppure: bash scripts/setup_prometheus_cluster.sh         (fa upgrade automatico)"
echo ""
echo "  SCENARIO C — Prometheus installato via kube-prometheus-stack (operator):"
echo "    → NON usare setup_prometheus_cluster.sh"
echo "    → Aggiungi scrape Hubble con un AdditionalScrapeConfig secret:"
echo "       kubectl create secret generic hubble-scrape -n monitoring \\"
echo "         --from-file=hubble.yaml=deployments/hubble-scrape-config.yaml"
echo ""
echo "  SCENARIO D — Cilium installato via Helm:"
echo "    → Usa helm upgrade per abilitare Hubble (più sicuro del patch ConfigMap)"
echo "    → enable_hubble_http.sh funziona lo stesso ma patch è meno 'pulito'"
echo ""
echo "  SCENARIO E — Cilium installato con manifesti (NON helm):"
echo "    → bash scripts/enable_hubble_http.sh   (patch ConfigMap + restart DaemonSet)"
echo "    → ATTENZIONE: il restart cilium causa ~30s di interruzione rete per i pod"
echo "    → Fai quando Online Boutique non è sotto test"
echo ""
