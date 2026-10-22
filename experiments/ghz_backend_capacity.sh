#!/bin/bash
# ============================================================================
# ghz Backend Capacity Test — Per-Service gRPC Profiling
# ============================================================================
#
# Scopo: misurare la capacita reale (rps/replica) di ciascun backend gRPC,
#        con la stessa metodologia stepped-load + knee detection usata da
#        k6_capacity_test.js per il frontend HTTP.
#
# Cosa fa ghz:
#   ghz e' un tool di benchmarking gRPC — l'equivalente di k6/Locust per
#   servizi gRPC. Manda richieste direttamente a un singolo servizio
#   bypassando il frontend. ghz SOSTITUISCE il frontend come generatore
#   di richieste: ghz -> gRPC -> backend (invece di HTTP -> frontend -> backend).
#
# DIPENDENZE (IMPORTANTE):
#   ghz NON isola il servizio dalle dipendenze downstream.
#   Il servizio chiamato chiama comunque i suoi servizi downstream.
#   Questo e' CORRETTO: misuri la capacita reale incluso il tempo
#   delle dipendenze, come nel sistema reale.
#
#   Call graph:
#     cartservice          -> redis-cart                    (1 dipendenza)
#     productcatalogservice -> (nessuna, in-memory)         (0 dipendenze)
#     recommendationservice -> productcatalogservice        (1 dipendenza)
#     checkoutservice       -> cart + catalog + currency    (6 dipendenze)
#                              + shipping + payment + email
#
#   Setup critico:
#     - Servizio TARGET: 1 replica (quello che misuri)
#     - Dipendenze: ATTIVE e NON SATURATE
#     - Servizi non scalati da DMOS (redis, currency, shipping, payment,
#       email) hanno gia repliche fisse — non sono un problema
#     - DMOS FERMO durante il test
#
# Ordine consigliato:
#   1. productcatalogservice  (zero dipendenze, misura pura)
#   2. cartservice            (solo redis-cart locale)
#   3. recommendationservice  (dipende da productcatalog, gia testato)
#   4. checkoutservice        (fan-out completo, testare per ultimo)
#
# Metodologia:
#   Step-load crescente con ghz.
#   Ogni step dura STEP_DURATION secondi a un rate costante.
#   Il rate aumenta di STEP_INCREMENT rps ad ogni step.
#   Il test si ferma quando la latency p95 supera SLA o error rate > 2%.
#   Il knee point e' l'ultimo step con SLA rispettato.
#
# Prerequisiti:
#   1. ghz installato: go install github.com/bojand/ghz/cmd/ghz@latest
#      oppure: https://github.com/bojand/ghz/releases
#   2. Proto files scaricati (lo script li scarica automaticamente)
#   3. kubectl configurato con accesso ai 3 cluster
#   4. DMOS FERMO — repliche fissate a 1 per servizio target
#
# Uso:
#   # Test tutti i backend su cluster1 (ordine: catalog->cart->reco->checkout):
#   bash experiments/ghz_backend_capacity.sh cluster1
#
#   # Test singolo servizio:
#   bash experiments/ghz_backend_capacity.sh cluster1 cartservice
#
#   # Con parametri personalizzati:
#   STEP_DURATION=90 MAX_RATE=200 bash experiments/ghz_backend_capacity.sh cluster1
#
# Output:
#   results/capacity/ghz_{service}_{cluster}_{timestamp}.json
#   Console: tabella per step + knee point
#
# Riferimenti:
#   - ghz: https://ghz.sh/
#   - Gan et al. (ASPLOS 2019): microservice dependency-aware profiling
#   - Sriraman & Dhanotia (ASPLOS 2020): end-to-end tail latency
#   - USL (Gunther): throughput saturation under contention
# ============================================================================

set -euo pipefail

# ── Parametri ──────────────────────────────────────────────────────────────

CLUSTER="${1:-cluster1}"
TARGET_SERVICE="${2:-all}"

STEP_DURATION="${STEP_DURATION:-60}"       # secondi per step
STEP_INCREMENT="${STEP_INCREMENT:-10}"     # rps incremento per step
MAX_RATE="${MAX_RATE:-300}"                # rate massima da testare
WARMUP_DURATION="${WARMUP_DURATION:-10}"   # secondi di warmup a rate minima
SLA_P95_MS="${SLA_P95_MS:-0}"              # 0 = usa SLA per-servizio da tabella
MAX_CONSECUTIVE_FAIL="${MAX_CONSECUTIVE_FAIL:-2}"

OUTPUT_DIR="results/capacity"
PROTO_DIR="experiments/protos"
PROTO_FILE="${PROTO_DIR}/demo.proto"

# ── Cluster → IP mapping (read from .cache/test-env.sh) ────────────────
ENV_FILE="$(dirname "$0")/../.cache/test-env.sh"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERRORE: $ENV_FILE not found."
    echo "Run 'bash scripts/render-runtime-files.sh' first (it generates"
    echo "this file from config/local.yaml)."
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

declare -A CLUSTER_IPS
for ((idx = 0; idx < DMOS_CLUSTER_COUNT; idx++)); do
    name="${DMOS_CLUSTER_NAMES[$idx]}"
    CLUSTER_IPS["$name"]="${DMOS_CLUSTER_IPS[$idx]}"
done

CLUSTER_IP="${CLUSTER_IPS[$CLUSTER]:-}"
if [[ -z "$CLUSTER_IP" ]]; then
    echo "ERRORE: Cluster sconosciuto: $CLUSTER"
    echo "Disponibili: ${DMOS_CLUSTER_NAMES[*]}"
    exit 1
fi

# ── Servizi backend e configurazione RPC ─────────────────────────────────
#
# Ogni servizio ha:
#   port: porta gRPC del Service K8s
#   rpc: metodo gRPC da chiamare (il piu rappresentativo del workload reale)
#   data: payload JSON per la richiesta
#   sla_p95: soglia SLA in millisecondi
#
# Le RPC scelte sono quelle piu chiamate dal frontend nel workload mix:
#   - CartService: GetCart (il frontend lo chiama su ogni page view con carrello)
#   - ProductCatalogService: GetProduct (chiamato per ogni pagina prodotto)
#   - CheckoutService: PlaceOrder (l'unica RPC, path pesante ma poco frequente)
#   - RecommendationService: ListRecommendations (chiamato sulla home e prodotti)
#
# DIPENDENZE quando ghz chiama ciascun servizio:
#
#   ghz -> cartservice          -> redis-cart              (1 dep, locale)
#   ghz -> productcatalogservice -> (nessuna, in-memory)   (0 dep, pura)
#   ghz -> recommendationservice -> productcatalogservice  (1 dep, gRPC)
#   ghz -> checkoutservice       -> cart + catalog +       (6 dep, fan-out)
#                                   currency + shipping +
#                                   payment + email

declare -A SVC_PORT SVC_RPC SVC_DATA SVC_SLA SVC_DEPS

# cartservice: dipende solo da redis-cart (locale, non scalato da DMOS)
SVC_PORT["cartservice"]=7070
SVC_RPC["cartservice"]="hipstershop.CartService/GetCart"
SVC_DATA["cartservice"]='{"user_id":"capacity-test-user"}'
SVC_SLA["cartservice"]=50
SVC_DEPS["cartservice"]="redis-cart"

# productcatalogservice: ZERO dipendenze (catalogo in-memory Go)
SVC_PORT["productcatalogservice"]=3550
SVC_RPC["productcatalogservice"]="hipstershop.ProductCatalogService/GetProduct"
SVC_DATA["productcatalogservice"]='{"id":"OLJCESPC7Z"}'
SVC_SLA["productcatalogservice"]=75
SVC_DEPS["productcatalogservice"]="nessuna"

# checkoutservice: fan-out completo (6 dipendenze downstream)
# ATTENZIONE: testare per ULTIMO — se le dipendenze sono saturate,
# il bottleneck e' la dipendenza, non il checkoutservice
SVC_PORT["checkoutservice"]=5050
SVC_RPC["checkoutservice"]="hipstershop.CheckoutService/PlaceOrder"
SVC_DATA["checkoutservice"]='{"user_id":"capacity-test","user_currency":"USD","address":{"street_address":"123 Test","city":"NY","state":"NY","country":"US","zip_code":"10001"},"email":"test@example.com","credit_card":{"credit_card_number":"4432801561520454","credit_card_cvv":672,"credit_card_expiration_year":2030,"credit_card_expiration_month":1}}'
SVC_SLA["checkoutservice"]=200
SVC_DEPS["checkoutservice"]="cart,catalog,currency,shipping,payment,email"

# recommendationservice: dipende da productcatalogservice
SVC_PORT["recommendationservice"]=8080
SVC_RPC["recommendationservice"]="hipstershop.RecommendationService/ListRecommendations"
SVC_DATA["recommendationservice"]='{"user_id":"capacity-test","product_ids":["OLJCESPC7Z"]}'
SVC_SLA["recommendationservice"]=100
SVC_DEPS["recommendationservice"]="productcatalogservice"

# Ordine: dalle zero dipendenze al fan-out completo
ALL_SERVICES="productcatalogservice cartservice recommendationservice checkoutservice"

# ── Funzioni utility ─────────────────────────────────────────────────────

log() { echo "[$(date +%H:%M:%S)] $*"; }

check_prerequisites() {
    log "Verifica prerequisiti..."

    # ghz
    if ! command -v ghz &>/dev/null; then
        echo ""
        echo "ERRORE: ghz non trovato."
        echo ""
        echo "Installazione:"
        echo "  # Con Go:"
        echo "  go install github.com/bojand/ghz/cmd/ghz@latest"
        echo ""
        echo "  # Oppure scarica il binario:"
        echo "  https://github.com/bojand/ghz/releases"
        echo ""
        exit 1
    fi
    log "  ghz: $(ghz --version 2>&1 | head -1)"

    # kubectl
    if ! command -v kubectl &>/dev/null; then
        echo "ERRORE: kubectl non trovato"
        exit 1
    fi

    # Proto file
    if [[ ! -f "$PROTO_FILE" ]]; then
        log "  Proto file non trovato, download in corso..."
        download_protos
    fi
    log "  Proto file: $PROTO_FILE"
}

download_protos() {
    mkdir -p "$PROTO_DIR"

    # Scarica demo.proto dal repository Google Online Boutique
    local PROTO_URL="https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/protos/demo.proto"

    if command -v curl &>/dev/null; then
        curl -sL "$PROTO_URL" -o "$PROTO_FILE"
    elif command -v wget &>/dev/null; then
        wget -q "$PROTO_URL" -O "$PROTO_FILE"
    else
        echo "ERRORE: ne curl ne wget disponibili per scaricare il proto file"
        echo "Scarica manualmente: $PROTO_URL -> $PROTO_FILE"
        exit 1
    fi

    if [[ ! -s "$PROTO_FILE" ]]; then
        echo "ERRORE: download proto file fallito"
        exit 1
    fi
    log "  Proto file scaricato: $PROTO_FILE"
}

# ── Test singolo servizio ────────────────────────────────────────────────

test_service() {
    local svc="$1"
    local port="${SVC_PORT[$svc]}"
    local rpc="${SVC_RPC[$svc]}"
    local data="${SVC_DATA[$svc]}"
    local sla="${SLA_P95_MS:-0}"
    [[ "$sla" == "0" ]] && sla="${SVC_SLA[$svc]}"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local result_file="${OUTPUT_DIR}/ghz_${svc}_${CLUSTER}_${timestamp}.json"

    mkdir -p "$OUTPUT_DIR"

    local deps="${SVC_DEPS[$svc]:-nessuna}"

    echo ""
    echo "================================================================"
    echo "  CAPACITY TEST: $svc @ $CLUSTER ($CLUSTER_IP)"
    echo "================================================================"
    echo "  Porta gRPC:    $port"
    echo "  RPC:           $rpc"
    echo "  SLA p95:       ${sla}ms"
    echo "  Step:          ${STEP_INCREMENT} rps x ${STEP_DURATION}s"
    echo "  Max rate:      ${MAX_RATE} rps"
    echo "  Dipendenze:    $deps"
    echo "================================================================"
    if [[ "$deps" != "nessuna" ]]; then
        echo "  Le dipendenze [$deps] verranno colpite durante il test."
        echo "  Assicurati che siano attive e non saturate."
    fi
    echo ""

    # Setup port-forward in background
    log "Avvio port-forward: $svc:$port -> localhost:$port"
    kubectl port-forward "svc/$svc" "$port:$port" \
        -n online-boutique \
        --context "$CLUSTER" &
    local PF_PID=$!
    sleep 2

    # Verifica port-forward attivo
    if ! kill -0 "$PF_PID" 2>/dev/null; then
        log "ERRORE: port-forward fallito per $svc"
        return 1
    fi

    # Header tabella
    printf "  %4s | %6s | %8s | %8s | %8s | %8s | %5s | %s\n" \
        "Step" "Rate" "Actual" "p50" "p95" "p99" "Err%" "SLA"
    printf "  %s\n" "$(printf -- '-%.0s' {1..70})"

    local knee_rps=0
    local knee_rate=0
    local consecutive_fail=0
    local step=0
    local all_steps="["

    for rate in $(seq "$STEP_INCREMENT" "$STEP_INCREMENT" "$MAX_RATE"); do
        step=$((step + 1))

        # Esegui ghz per questo step
        local ghz_output
        ghz_output=$(ghz \
            --proto "$PROTO_FILE" \
            --call "$rpc" \
            --data "$data" \
            --rps "$rate" \
            --duration "${STEP_DURATION}s" \
            --connections 50 \
            --concurrency 100 \
            --format json \
            "localhost:$port" 2>/dev/null) || true

        if [[ -z "$ghz_output" ]]; then
            log "ERRORE: ghz non ha prodotto output per step $step"
            continue
        fi

        # Parse risultati JSON di ghz
        local actual_rps p50 p95 p99 total_count error_count error_rate
        actual_rps=$(echo "$ghz_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"rps\"]:.1f}')" 2>/dev/null || echo "0")
        p50=$(echo "$ghz_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"latencyDistribution\"][4][\"latency\"]/1e6:.0f}' if len(d.get(\"latencyDistribution\",[]))>4 else '0')" 2>/dev/null || echo "0")
        p95=$(echo "$ghz_output" | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'{e[\"latency\"]/1e6:.0f}') for e in d.get(\"latencyDistribution\",[]) if e.get(\"percentage\")==95]" 2>/dev/null | head -1 || echo "0")
        p99=$(echo "$ghz_output" | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'{e[\"latency\"]/1e6:.0f}') for e in d.get(\"latencyDistribution\",[]) if e.get(\"percentage\")==99]" 2>/dev/null | head -1 || echo "0")
        total_count=$(echo "$ghz_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['count'])" 2>/dev/null || echo "0")
        error_count=$(echo "$ghz_output" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('errorDistribution',{})))" 2>/dev/null || echo "0")

        # Calcola error rate
        if [[ "$total_count" -gt 0 ]]; then
            error_rate=$(python3 -c "print(f'{int(${error_count})/int(${total_count})*100:.1f}')" 2>/dev/null || echo "0.0")
        else
            error_rate="0.0"
        fi

        # Valuta SLA
        local sla_ok="OK"
        p95_num="${p95:-0}"
        err_num="${error_rate:-0}"
        if python3 -c "exit(0 if float('$p95_num') < $sla or float('$err_num') < 2.0 else 1)" 2>/dev/null; then
            sla_ok=" OK "
            consecutive_fail=0
            knee_rps="$actual_rps"
            knee_rate="$rate"
        else
            sla_ok="FAIL"
            consecutive_fail=$((consecutive_fail + 1))
        fi

        printf "  %4d | %4dr/s | %6sr/s | %5sms | %5sms | %5sms | %4s%% | %s\n" \
            "$step" "$rate" "$actual_rps" "$p50" "$p95" "$p99" "$error_rate" "$sla_ok"

        # Salva step per JSON
        [[ "$step" -gt 1 ]] && all_steps+=","
        all_steps+="{\"step\":$step,\"target_rate\":$rate,\"actual_rps\":$actual_rps,\"p50_ms\":$p50,\"p95_ms\":$p95,\"p99_ms\":$p99,\"error_rate\":$error_rate}"

        # Stop se SLA violato per N step consecutivi
        if [[ "$consecutive_fail" -ge "$MAX_CONSECUTIVE_FAIL" ]]; then
            echo ""
            log "SLA violato per $consecutive_fail step consecutivi -> saturazione"
            break
        fi
    done

    all_steps+="]"

    # Cleanup port-forward
    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true

    # Report
    echo ""
    echo "================================================================"
    if [[ "$knee_rps" != "0" ]]; then
        local recommended
        recommended=$(python3 -c "print(max(1, round(float('$knee_rps') * $SAFETY_MARGIN)))")
        echo "  Knee point:            $knee_rps rps/replica (a ${knee_rate} rps target)"
        echo "  Capacita raccomandata: $recommended rps/replica (80% del knee)"
        echo ""
        echo "  -> Aggiorna config/services.yaml:"
        echo "     $svc:"
        echo "       capacity_req_per_sec: $recommended"
    else
        echo "  Knee point: N/A - SLA violato gia al primo step"
        echo "  Verifica che il servizio sia attivo e raggiungibile"
    fi
    echo "================================================================"

    # Salva JSON
    cat > "$result_file" <<ENDJSON
{
  "service": "$svc",
  "cluster": "$CLUSTER",
  "cluster_ip": "$CLUSTER_IP",
  "rpc": "$rpc",
  "sla_p95_ms": $sla,
  "step_duration_s": $STEP_DURATION,
  "step_increment_rps": $STEP_INCREMENT,
  "knee_rps": $knee_rps,
  "knee_rate": $knee_rate,
  "recommended_capacity": $(python3 -c "print(max(1, round(float('$knee_rps') * $SAFETY_MARGIN)))" 2>/dev/null || echo "null"),
  "timestamp": "$(date -Iseconds)",
  "steps": $all_steps
}
ENDJSON

    log "Risultati salvati: $result_file"
}

# ── Main ─────────────────────────────────────────────────────────────────

main() {
    check_prerequisites

    echo ""
    echo "================================================================"
    echo "  ghz BACKEND CAPACITY TEST"
    echo "================================================================"
    echo "  Cluster:   $CLUSTER ($CLUSTER_IP)"
    echo "  Target:    $TARGET_SERVICE"
    echo "  Step:      ${STEP_INCREMENT} rps x ${STEP_DURATION}s"
    echo "  Max rate:  ${MAX_RATE} rps"
    echo "  Output:    ${OUTPUT_DIR}/"
    echo "================================================================"
    echo ""
    echo "  PREREQUISITI:"
    echo "    1. DMOS FERMO"
    echo "    2. Servizio target: esattamente 1 replica su $CLUSTER"
    echo "    3. Dipendenze downstream ATTIVE e NON SATURATE:"
    echo "       - redis-cart, currencyservice, shippingservice,"
    echo "         paymentservice, emailservice (repliche fisse, non scalati)"
    echo "       - productcatalogservice, cartservice (fissare a 1 replica)"
    echo ""
    echo "  NOTA: ghz NON isola dalle dipendenze. Il servizio chiamato"
    echo "  invoca comunque i suoi servizi downstream. La capacita misurata"
    echo "  include il tempo delle dipendenze = comportamento reale."
    echo ""
    if [[ "$TARGET_SERVICE" == "all" ]]; then
        echo "  ORDINE: productcatalog -> cart -> recommendation -> checkout"
        echo "  (dalle zero dipendenze al fan-out completo)"
        echo ""
    fi

    local services_to_test
    if [[ "$TARGET_SERVICE" == "all" ]]; then
        services_to_test="$ALL_SERVICES"
    else
        services_to_test="$TARGET_SERVICE"
    fi

    for svc in $services_to_test; do
        if [[ -z "${SVC_PORT[$svc]:-}" ]]; then
            log "ERRORE: servizio sconosciuto: $svc"
            log "Servizi validi: $ALL_SERVICES"
            exit 1
        fi

        # Verifica che il servizio abbia 1 replica
        local replicas
        replicas=$(kubectl get deployment "$svc" -n online-boutique \
            --context "$CLUSTER" \
            -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "?")
        if [[ "$replicas" != "1" ]]; then
            log "WARNING: $svc ha $replicas repliche su $CLUSTER (atteso: 1)"
            log "Esegui: kubectl scale deployment $svc -n online-boutique --replicas=1 --context $CLUSTER"
            read -p "  Continuare comunque? (y/N) " -n 1 -r
            echo
            [[ ! "$REPLY" =~ ^[Yy]$ ]] && continue
        fi

        test_service "$svc"
        echo ""
    done

    echo ""
    log "Test completati. Risultati in: $OUTPUT_DIR/"
}

main
