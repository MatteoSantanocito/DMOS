import grpc from 'k6/net/grpc';
import { check } from 'k6';
import { Trend, Counter } from 'k6/metrics';
import exec from 'k6/execution';

// ============================================================================
// k6 Capacity Test — Backend gRPC (Online Boutique)
// ============================================================================
//
// Scopo
// -----
// Misurare la capacita' reale (rps/replica) dei microservizi BACKEND gRPC di
// Online Boutique, per calibrare `capacity_req_per_sec` in config/services.yaml.
// I backend sono gRPC, quindi il k6_capacity_test.js (HTTP, via Ingress) non li
// puo' testare: serve un client gRPC.
//
// Design — identico a k6_capacity_test.js ma su gRPC:
//   - scenari `constant-arrival-rate` consecutivi (rate FISSO per step)
//   - ogni step: SETTLE (scartato) + MEASURE (registrato)
//   - knee-point = ultimo step con p95 < SLA, p99 < 4xSLA e err < SLA
//   - la connessione gRPC e' aperta UNA volta per VU (no overhead connect)
//
// Backend supportati e metodo gRPC usato per il test:
//   cartservice            -> hipstershop.CartService/GetCart
//   productcatalogservice  -> hipstershop.ProductCatalogService/ListProducts
//   recommendationservice  -> hipstershop.RecommendationService/ListRecommendations
//   checkoutservice        -> hipstershop.CheckoutService/PlaceOrder
//
// IMPORTANTE — questo script gira come Job DENTRO il cluster (pod k6 nel
// namespace online-boutique) e colpisce il backend via ClusterIP, es.
// cartservice:7070. NON usa kubectl port-forward: misurare da fuori il
// cluster aggiungeva ~47ms di tassa fissa per richiesta (round-trip via API
// server) e il listener del port-forward crollava sotto carico. In-cluster il
// gRPC e' pod-to-pod (sub-ms) e attraversa lo stesso proxy L7 Cilium del
// traffico reale frontend->backend. Lo orchestra run_capacity_grpc.sh.
//
// Uso (di solito via run_capacity_grpc.sh, che crea ConfigMap + Job):
//   k6 run \
//       -e DMOS_GRPC_TARGET=cartservice:7070 \
//       -e DMOS_GRPC_METHOD=hipstershop.CartService/GetCart \
//       -e DMOS_SVC_NAME=cartservice \
//       k6_capacity_grpc.js
//
// Variabili env
// -------------
//   DMOS_GRPC_TARGET     : host:port del backend (default cartservice:7070)
//   DMOS_GRPC_METHOD     : metodo gRPC completo (default CartService/GetCart)
//   DMOS_SVC_NAME        : nome del servizio per i report (default cartservice)
//   DMOS_STEPS           : lista rps per step (default 25,50,75,100,150,200,300,400)
//   DMOS_STEP_DURATION_S : durata measure per step (default 90)
//   DMOS_SETTLE_S        : durata settle per step (default 15)
//   DMOS_SLA_P95_MS      : SLA p95 per il knee (default 100)
//   DMOS_SLA_ERR_RATE    : SLA error rate per il knee (default 0.02)
//   DMOS_P99_SLA_FACTOR  : knee scartato se p99 >= fattore x SLA p95 (default 4)
//
// Output
// ------
// Tabella per-step + knee-point + suggerimento per services.yaml. Il JSON
// machine-readable e' stampato su stdout fra i marcatori
// ===DMOS_CAPACITY_JSON_BEGIN=== / ===DMOS_CAPACITY_JSON_END=== cosi'
// run_capacity_grpc.sh lo estrae da `kubectl logs`.
// ============================================================================

// ── Configurazione (env-driven) ─────────────────────────────────────────────

const GRPC_TARGET     = __ENV.DMOS_GRPC_TARGET  || 'cartservice:7070';
const GRPC_METHOD     = __ENV.DMOS_GRPC_METHOD  || 'hipstershop.CartService/GetCart';
const SVC_NAME        = __ENV.DMOS_SVC_NAME     || 'cartservice';
const STEP_DURATION_S = parseInt(__ENV.DMOS_STEP_DURATION_S || '90');
const SETTLE_S        = parseInt(__ENV.DMOS_SETTLE_S        || '15');
const SLA_P95_MS      = parseInt(__ENV.DMOS_SLA_P95_MS      || '100');
const SLA_ERR_RATE    = parseFloat(__ENV.DMOS_SLA_ERR_RATE  || '0.02');
const P99_SLA_FACTOR  = parseFloat(__ENV.DMOS_P99_SLA_FACTOR || '4');

const DEFAULT_STEPS = '25,50,75,100,150,200,300,400';
const STEPS = (__ENV.DMOS_STEPS || DEFAULT_STEPS)
    .split(',').map((s) => parseInt(s.trim()));

// ── Carica il proto ─────────────────────────────────────────────────────────
// k6 risolve i path relativi alla cartella di QUESTO file .js. demo.proto sta
// accanto allo script (in locale experiments/demo.proto; in-cluster nella
// stessa ConfigMap montata su /scripts) -> import path '.'.
const grpcClient = new grpc.Client();
grpcClient.load(['.'], 'demo.proto');

// ── Build scenarios: 1 settle + 1 measure (constant-arrival-rate) per step ──

// VU sizing — il test gira IN-CLUSTER (pod k6 -> ClusterIP del backend), dove
// la latenza gRPC e' bassa (~1-10ms): servono pochissime VU. preAllocatedVUs
// basso = avvio rapido; maxVUs e' il tetto a cui k6 cresce su richiesta verso
// il knee. Entrambi sono limitati a 1000: ai rate alti (test che spingono fino
// a migliaia di rps) k6 alloca solo le VU che servono davvero, ma il tetto
// protegge la RAM del pod k6 dall'OOM nei 2 step di saturazione dopo il knee.
const scenarios = {};
let cursor = 0;
for (let i = 0; i < STEPS.length; i++) {
    const rate = STEPS[i];
    const tag = String(rate).padStart(4, '0');
    const preVUs = Math.min(Math.max(20, Math.ceil(rate * 0.05)), 1000);
    const capVUs = Math.min(Math.max(100, rate), 1000);

    scenarios[`step${String(i).padStart(2, '0')}_${tag}_settle`] = {
        executor: 'constant-arrival-rate',
        rate: rate, timeUnit: '1s', duration: `${SETTLE_S}s`,
        preAllocatedVUs: preVUs, maxVUs: capVUs,
        startTime: `${cursor}s`, exec: 'doRequest',
        tags: { step: tag, phase: 'settle' },
    };
    cursor += SETTLE_S;

    scenarios[`step${String(i).padStart(2, '0')}_${tag}_measure`] = {
        executor: 'constant-arrival-rate',
        rate: rate, timeUnit: '1s', duration: `${STEP_DURATION_S}s`,
        preAllocatedVUs: preVUs, maxVUs: capVUs,
        startTime: `${cursor}s`, exec: 'doRequest',
        tags: { step: tag, phase: 'measure' },
    };
    cursor += STEP_DURATION_S;
}

export const options = {
    scenarios: scenarios,
    summaryTrendStats: ['avg', 'min', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

// ── Per-step custom metrics ─────────────────────────────────────────────────

const stepLatency  = {};
const stepRequests = {};
const stepErrors   = {};
for (const rate of STEPS) {
    const key = String(rate).padStart(4, '0');
    stepLatency[key]  = new Trend(`step_${key}_latency`, true);
    stepRequests[key] = new Counter(`step_${key}_requests`);
    stepErrors[key]   = new Counter(`step_${key}_errors`);
}

// ── Payload builder per metodo gRPC ─────────────────────────────────────────

const PRODUCT_IDS = [
    'OLJCESPC7Z', '66VCHSJNUP', '1YMWWN1N4O', 'L9ECAV7KIM', '2ZYFJ3GM2N',
    '0PUK6V6EV0', 'LS4PSXUNUM', '9SIQT8TOJO', '6E92ZMYYFZ',
];
function randomProduct() {
    return PRODUCT_IDS[Math.floor(Math.random() * PRODUCT_IDS.length)];
}
function randomUser() {
    return 'k6-user-' + Math.floor(Math.random() * 100000);
}

// Ritorna il payload corretto per il metodo gRPC corrente.
function buildPayload(method) {
    if (method.endsWith('CartService/GetCart')) {
        return { user_id: randomUser() };
    }
    if (method.endsWith('CartService/AddItem')) {
        return { user_id: randomUser(), item: { product_id: randomProduct(), quantity: 1 } };
    }
    if (method.endsWith('ProductCatalogService/ListProducts')) {
        return {};  // input Empty
    }
    if (method.endsWith('ProductCatalogService/GetProduct')) {
        return { id: randomProduct() };
    }
    if (method.endsWith('RecommendationService/ListRecommendations')) {
        return { user_id: randomUser(), product_ids: [randomProduct()] };
    }
    if (method.endsWith('CheckoutService/PlaceOrder')) {
        return {
            user_id: randomUser(),
            user_currency: 'USD',
            address: {
                street_address: '1600 Amphitheatre Parkway',
                city: 'Mountain View', state: 'CA', country: 'USA',
                zip_code: 94043,
            },
            email: 'k6-test@example.com',
            credit_card: {
                credit_card_number: '4432801561520454',
                credit_card_cvv: 672,
                credit_card_expiration_year: 2030,
                credit_card_expiration_month: 1,
            },
        };
    }
    throw new Error(`Metodo gRPC non gestito da buildPayload: ${method}`);
}

// ── VU exec function ────────────────────────────────────────────────────────
// La connessione gRPC e' aperta UNA volta per VU (variabile a livello modulo:
// in k6 ogni VU ha la sua copia). Cosi' misuriamo il tempo di invoke, non
// quello di connect.

let _connected = false;

export function doRequest() {
    if (!_connected) {
        grpcClient.connect(GRPC_TARGET, { plaintext: true, timeout: '10s' });
        _connected = true;
    }

    const payload = buildPayload(GRPC_METHOD);

    const t0 = Date.now();
    let resp;
    try {
        resp = grpcClient.invoke(GRPC_METHOD, payload, { timeout: '30s' });
    } catch (e) {
        resp = { status: -1, error: String(e) };
    }
    const dur = Date.now() - t0;

    const isOk = resp && resp.status === grpc.StatusOK;

    check(resp, {
        'grpc status OK': (r) => r && r.status === grpc.StatusOK,
    });

    // Attribuisci la misura allo step corrente (solo phase=measure).
    const m = exec.scenario.name.match(/step\d+_(\d+)_measure/);
    if (m) {
        const key = m[1];
        if (stepLatency[key]) {
            stepLatency[key].add(dur);
            stepRequests[key].add(1);
            if (!isOk) stepErrors[key].add(1);
        }
    }
}

// ── Custom summary ──────────────────────────────────────────────────────────

export function handleSummary(data) {
    const lines = [];
    const totalMin = ((STEPS.length * (SETTLE_S + STEP_DURATION_S)) / 60).toFixed(1);

    lines.push('');
    lines.push('='.repeat(80));
    lines.push(`  CAPACITY TEST gRPC — ${SVC_NAME}`);
    lines.push(`  Metodo: ${GRPC_METHOD}`);
    lines.push(`  Target: ${GRPC_TARGET}`);
    lines.push(`  SLA: p95 < ${SLA_P95_MS}ms, p99 < ${SLA_P95_MS * P99_SLA_FACTOR}ms, err < ${(SLA_ERR_RATE * 100).toFixed(1)}% | Tot ${totalMin} min`);
    lines.push('='.repeat(80));
    lines.push('');
    lines.push('  Step | rps tgt | rps obt | rps/rep |   p50  |   p95  |   p99  | Err%  | SLA');
    lines.push('  ' + '-'.repeat(76));

    let kneeRps = null;
    let kneeRate = null;
    const results = [];
    let consecutiveFails = 0;

    for (let i = 0; i < STEPS.length; i++) {
        const rate = STEPS[i];
        const key = String(rate).padStart(4, '0');
        const lat = data.metrics[`step_${key}_latency`];
        const req = data.metrics[`step_${key}_requests`];
        const err = data.metrics[`step_${key}_errors`];

        if (!lat || !req || req.values.count === 0) {
            lines.push(`  ${String(i + 1).padStart(4)} | ${String(rate).padStart(5)}r/s | (no data)`);
            continue;
        }

        const totalReqs = req.values.count;
        const totalErrs = err ? err.values.count : 0;
        const errorRate = totalReqs > 0 ? totalErrs / totalReqs : 0;
        const p50 = lat.values['p(50)'] || 0;
        const p95 = lat.values['p(95)'] || 0;
        const p99 = lat.values['p(99)'] || 0;
        const rpsObt = totalReqs / STEP_DURATION_S;

        // Uno step e' "sano" solo se anche la CODA regge: un p99 esploso (es.
        // da CFS throttling) con p95 ancora basso NON e' un punto operativo
        // valido. Per questo il knee richiede anche p99 < P99_SLA_FACTOR x SLA.
        const slaOk = p95 < SLA_P95_MS
                   && p99 < SLA_P95_MS * P99_SLA_FACTOR
                   && errorRate < SLA_ERR_RATE;
        results.push({ rate, rpsObt, p50, p95, p99, errorRate, slaOk });

        if (slaOk) {
            consecutiveFails = 0;
            kneeRps = rpsObt;
            kneeRate = rate;
        } else {
            consecutiveFails++;
        }

        lines.push(
            `  ${String(i + 1).padStart(4)} | ` +
            `${String(rate).padStart(5)}r/s | ` +
            `${rpsObt.toFixed(1).padStart(7)} | ` +
            `${rpsObt.toFixed(1).padStart(7)} | ` +
            `${p50.toFixed(0).padStart(5)}ms | ` +
            `${p95.toFixed(0).padStart(5)}ms | ` +
            `${p99.toFixed(0).padStart(5)}ms | ` +
            `${(errorRate * 100).toFixed(1).padStart(4)}% | ` +
            `${slaOk ? ' OK ' : 'FAIL'}`
        );

        if (consecutiveFails >= 2) {
            lines.push('');
            lines.push('  -> SLA violato per 2 step consecutivi: saturazione confermata');
            break;
        }
    }

    lines.push('');
    lines.push('='.repeat(80));

    if (kneeRps !== null) {
        const recommended = Math.max(1, Math.round(kneeRps * 0.80));
        lines.push(`  Knee-point:             ${kneeRps.toFixed(1)} rps/replica  (a ${kneeRate} rps target)`);
        lines.push(`  Capacita raccomandata:  ${recommended} rps/replica  (80% del knee)`);
        lines.push('');
        lines.push(`  -> config/services.yaml, sezione ${SVC_NAME}:`);
        lines.push(`     capacity_req_per_sec: ${recommended}`);
    } else {
        lines.push('  Knee-point: N/A (SLA gia violato al primo step)');
        lines.push('  Verifica che il port-forward sia attivo e il backend Ready.');
    }
    lines.push('='.repeat(80));
    lines.push('');

    const machineReport = {
        service: SVC_NAME,
        grpc_method: GRPC_METHOD,
        grpc_target: GRPC_TARGET,
        sla: { p95_ms: SLA_P95_MS, p99_ms: SLA_P95_MS * P99_SLA_FACTOR, error_rate: SLA_ERR_RATE },
        steps: STEPS,
        results: results,
        knee: kneeRps !== null
            ? { rps_per_replica: kneeRps, target_rate: kneeRate,
                recommended_capacity_req_per_sec: Math.round(kneeRps * 0.80) }
            : null,
        timestamp_iso: new Date().toISOString(),
    };

    // Il test gira come Job dentro il cluster: l'unico canale di ritorno e'
    // stdout (-> kubectl logs). Stampiamo il JSON machine-readable fra due
    // marcatori, cosi' run_capacity_grpc.sh lo estrae dai log. Niente file
    // output: dentro il pod la cartella experiments/results/ non esiste.
    const jsonBlock =
        '\n===DMOS_CAPACITY_JSON_BEGIN===\n' +
        JSON.stringify(machineReport) +
        '\n===DMOS_CAPACITY_JSON_END===\n';

    return {
        stdout: lines.join('\n') + '\n' + grpcSummaryFallback(data) + jsonBlock,
    };
}

function grpcSummaryFallback(data) {
    const lines = ['  --- Standard k6 gRPC Summary ---', ''];
    for (const name of ['grpc_req_duration', 'grpc_reqs', 'iterations', 'vus', 'vus_max', 'checks']) {
        const m = data.metrics[name];
        if (!m) continue;
        if (m.type === 'trend') {
            const v = m.values;
            lines.push(`  ${name.padEnd(22)} avg=${(v.avg||0).toFixed(2)}ms ` +
                `p50=${(v['p(50)']||0).toFixed(2)}ms p95=${(v['p(95)']||0).toFixed(2)}ms ` +
                `p99=${(v['p(99)']||0).toFixed(2)}ms`);
        } else if (m.type === 'counter') {
            lines.push(`  ${name.padEnd(22)} count=${m.values.count} rate=${(m.values.rate||0).toFixed(2)}/s`);
        } else if (m.type === 'rate') {
            lines.push(`  ${name.padEnd(22)} rate=${((m.values.rate||0)*100).toFixed(2)}%`);
        }
    }
    lines.push('');
    return lines.join('\n');
}
