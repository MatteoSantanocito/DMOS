import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter } from 'k6/metrics';
import exec from 'k6/execution';

// ============================================================================
// k6 Capacity Test — Per-Cluster, con netem
// ============================================================================
//
// Scopo: misurare la capacità reale (rps/replica) di un singolo cluster
//        con netem attivo, per calibrare capacity_per_cluster in services.yaml.
//
// Uso:
//   # 1. Scala il servizio a 1 replica SOLO sul cluster target:
//   kubectl scale deployment frontend -n online-boutique --replicas=1 --context cluster1
//   kubectl scale deployment frontend -n online-boutique --replicas=0 --context cluster2
//   kubectl scale deployment frontend -n online-boutique --replicas=0 --context cluster3
//
//   # 2. Lancia il test verso quel cluster:
//   k6 run -e CLUSTER=cluster1 experiments/k6_capacity_test.js
//   k6 run -e CLUSTER=cluster2 experiments/k6_capacity_test.js
//   k6 run -e CLUSTER=cluster3 experiments/k6_capacity_test.js
//
//   # 3. Opzioni:
//   k6 run -e CLUSTER=cluster2 -e REPLICAS=2 -e SLA_P95=500 experiments/k6_capacity_test.js
//
// Output:
//   Tabella per-step con rps, p50, p95, p99, error rate, SLA
//   Knee point e valore raccomandato per capacity_per_cluster
// ============================================================================

// ── Configurazione ──────────────────────────────────────────────────────────

const CLUSTER = __ENV.CLUSTER || 'cluster1';
const REPLICAS = parseInt(__ENV.REPLICAS || '1');
const SLA_P95_MS = parseInt(__ENV.SLA_P95 || '500');

const CLUSTERS = {
    cluster1: 'http://192.168.1.245:30080',
    cluster2: 'http://192.168.1.246:30080',
    cluster3: 'http://192.168.1.247:30080',
};

const BASE_URL = CLUSTERS[CLUSTER];
if (!BASE_URL) {
    throw new Error(`Cluster sconosciuto: ${CLUSTER}. Usa: cluster1, cluster2, cluster3`);
}

// Step configuration
const STEP_DURATION_S = 60;      // Durata di ogni step in secondi
const STEP_INCREMENT = 2;        // Incremento rps per step
const MAX_RATE = 50;             // Rate massima da testare (rps)
const WARMUP_DURATION_S = 30;    // Warmup a rate minima (non misurato)

// ── Genera stages ───────────────────────────────────────────────────────────
// Warmup + stepped ramp: 2 → 4 → 6 → ... → MAX_RATE rps

const stages = [
    { duration: `${WARMUP_DURATION_S}s`, target: STEP_INCREMENT },  // warmup
];
for (let rate = STEP_INCREMENT; rate <= MAX_RATE; rate += STEP_INCREMENT) {
    stages.push({ duration: `${STEP_DURATION_S}s`, target: rate });
}

const NUM_STEPS = Math.floor(MAX_RATE / STEP_INCREMENT);  // steps misurati (escluso warmup)

export const options = {
    scenarios: {
        capacity: {
            executor: 'ramping-arrival-rate',
            startRate: STEP_INCREMENT,
            timeUnit: '1s',
            preAllocatedVUs: 200,
            maxVUs: 500,
            stages: stages,
        },
    },
    summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],
};

// ── Custom metrics per step ─────────────────────────────────────────────────
// Ogni step ha le proprie metriche per calcolare p95 e error rate isolati.

const stepLatency = [];
const stepErrors = [];
const stepRequests = [];

for (let i = 0; i < NUM_STEPS; i++) {
    const rate = (i + 1) * STEP_INCREMENT;
    stepLatency.push(new Trend(`step_${String(rate).padStart(2, '0')}rps_latency`, true));
    stepErrors.push(new Counter(`step_${String(rate).padStart(2, '0')}rps_errors`));
    stepRequests.push(new Counter(`step_${String(rate).padStart(2, '0')}rps_requests`));
}

// ── Endpoint mix (identico a k6_flash_crowd.js) ────────────────────────────

const CHECKOUT_DATA = {
    email: "test@example.com", street_address: "123 Test St", zip_code: "10001",
    city: "New York", state: "NY", country: "US",
    credit_card_number: "4432801561520454", credit_card_expiration_month: "1",
    credit_card_expiration_year: "2030", credit_card_cvv: "672",
};

const ENDPOINTS = [
    { path: "/",                   weight: 0.40, method: "GET",  body: null },
    { path: "/product/OLJCESPC7Z", weight: 0.15, method: "GET",  body: null },
    { path: "/product/66VCHSJNUP", weight: 0.10, method: "GET",  body: null },
    { path: "/cart",               weight: 0.10, method: "GET",  body: null },
    { path: "/cart",               weight: 0.15, method: "POST", body: { product_id: "OLJCESPC7Z", quantity: 1 } },
    { path: "/setCurrency",        weight: 0.05, method: "POST", body: { currency_code: "EUR" } },
    { path: "/cart/checkout",      weight: 0.05, method: "POST", body: CHECKOUT_DATA },
];

function pickWeighted(items) {
    const totalWeight = items.reduce((acc, item) => acc + item.weight, 0);
    let r = Math.random() * totalWeight;
    for (let item of items) {
        r -= item.weight;
        if (r <= 0) return item;
    }
    return items[items.length - 1];
}

// ── VU function ─────────────────────────────────────────────────────────────

export default function () {
    const ep = pickWeighted(ENDPOINTS);
    const url = BASE_URL + ep.path;

    const params = {
        tags: { cluster: CLUSTER },
        timeout: '30s',
    };

    let res;
    if (ep.method === "POST") {
        res = http.post(url, ep.body, params);
    } else {
        res = http.get(url, params);
    }

    const is5xx = res.status >= 500 || res.status === 0;

    check(res, {
        'status is not 5xx': (r) => r.status < 500,
        'status is not 0 (connection error)': (r) => r.status !== 0,
    });

    // Determina lo step corrente in base al progresso dello scenario.
    // progress va da 0.0 a 1.0 sull'intera durata (warmup + steps).
    // Sottrai la frazione di warmup per ottenere l'indice dello step misurato.
    const totalDuration = WARMUP_DURATION_S + NUM_STEPS * STEP_DURATION_S;
    const warmupFraction = WARMUP_DURATION_S / totalDuration;
    const progress = exec.scenario.progress;  // 0.0 → 1.0

    if (progress > warmupFraction) {
        const measuredProgress = (progress - warmupFraction) / (1 - warmupFraction);
        const stepIdx = Math.min(
            Math.floor(measuredProgress * NUM_STEPS),
            NUM_STEPS - 1
        );

        stepLatency[stepIdx].add(res.timings.duration);
        stepRequests[stepIdx].add(1);
        if (is5xx) {
            stepErrors[stepIdx].add(1);
        }
    }

    sleep(Math.random() * 0.5 + 0.25);
}

// ── Custom Summary ──────────────────────────────────────────────────────────

export function handleSummary(data) {
    const lines = [];
    lines.push('');
    lines.push('='.repeat(75));
    lines.push(`  CAPACITY TEST — ${CLUSTER} (${BASE_URL})`);
    lines.push(`  Repliche: ${REPLICAS} | SLA: p95 < ${SLA_P95_MS}ms | Step: ${STEP_DURATION_S}s`);
    lines.push('='.repeat(75));
    lines.push('');
    lines.push(
        '  Step |  Rate  |   RPS  | rps/rep |   p50  |   p95  |   p99  | Err%  | SLA'
    );
    lines.push('  ' + '-'.repeat(71));

    let kneeRps = null;
    let kneeRate = null;
    const results = [];

    for (let i = 0; i < NUM_STEPS; i++) {
        const rate = (i + 1) * STEP_INCREMENT;
        const prefix = `step_${String(rate).padStart(2, '0')}rps`;

        const latencyKey = `${prefix}_latency`;
        const requestsKey = `${prefix}_requests`;
        const errorsKey = `${prefix}_errors`;

        const latencyMetric = data.metrics[latencyKey];
        const requestsMetric = data.metrics[requestsKey];
        const errorsMetric = data.metrics[errorsKey];

        if (!latencyMetric || !requestsMetric) {
            continue;
        }

        const totalReqs = requestsMetric.values.count;
        const totalErrs = errorsMetric ? errorsMetric.values.count : 0;
        const errorRate = totalReqs > 0 ? totalErrs / totalReqs : 0;

        const p50 = latencyMetric.values['p(50)'] || 0;
        const p95 = latencyMetric.values['p(95)'] || 0;
        const p99 = latencyMetric.values['p(99)'] || 0;
        const avg = latencyMetric.values['avg'] || 0;

        const rps = totalReqs / STEP_DURATION_S;
        const rpsPerReplica = rps / REPLICAS;

        const slaOk = p95 < SLA_P95_MS && errorRate < 0.02;
        const slaStr = slaOk ? ' OK ' : 'FAIL';

        results.push({
            rate, rps, rpsPerReplica, p50, p95, p99, errorRate, slaOk, totalReqs,
        });

        if (slaOk) {
            kneeRps = rpsPerReplica;
            kneeRate = rate;
        }

        lines.push(
            `  ${String(i + 1).padStart(4)} | ` +
            `${String(rate).padStart(4)}r/s | ` +
            `${rps.toFixed(1).padStart(6)} | ` +
            `${rpsPerReplica.toFixed(1).padStart(7)} | ` +
            `${p50.toFixed(0).padStart(5)}ms | ` +
            `${p95.toFixed(0).padStart(5)}ms | ` +
            `${p99.toFixed(0).padStart(5)}ms | ` +
            `${(errorRate * 100).toFixed(1).padStart(4)}% | ` +
            `${slaStr}`
        );

        // Se SLA fallito per 2 step consecutivi, stop analisi
        if (!slaOk && i > 0 && results.length >= 2 && !results[results.length - 2].slaOk) {
            lines.push('');
            lines.push(`  SLA VIOLATO per 2 step consecutivi → saturazione confermata`);
            break;
        }
    }

    lines.push('');
    lines.push('='.repeat(75));

    if (kneeRps !== null) {
        const recommended = Math.max(1, Math.round(kneeRps * 0.80));
        lines.push(`  Knee point:             ${kneeRps.toFixed(1)} rps/replica  (a ${kneeRate} rps target)`);
        lines.push(`  Capacita raccomandata:  ${recommended} rps/replica  (80% del knee point)`);
        lines.push('');
        lines.push(`  → Aggiorna config/services.yaml:`);
        lines.push(`    capacity_per_cluster:`);
        lines.push(`      ${CLUSTER}: ${recommended}`);
    } else {
        lines.push(`  Knee point: N/A — SLA violato gia al primo step`);
        lines.push(`  Il cluster potrebbe essere sovraccarico o non raggiungibile.`);
    }

    lines.push('='.repeat(75));
    lines.push('');

    // Stampa il report personalizzato + il summary standard di k6
    const customReport = lines.join('\n');

    return {
        stdout: customReport + '\n\n' + textSummaryFallback(data),
    };
}

// ── Fallback text summary (senza jslib import) ─────────────────────────────
// k6 non supporta import di URL su tutti gli ambienti, quindi implementiamo
// un summary minimale inline.

function textSummaryFallback(data) {
    const lines = ['  --- Standard k6 Summary ---', ''];
    const interesting = [
        'http_req_duration', 'http_req_failed', 'http_reqs',
        'iteration_duration', 'iterations', 'vus', 'vus_max',
        'checks',
    ];

    for (const name of interesting) {
        const m = data.metrics[name];
        if (!m) continue;

        if (m.type === 'trend') {
            const v = m.values;
            lines.push(
                `  ${name.padEnd(30)} avg=${(v.avg || 0).toFixed(2)}ms ` +
                `p50=${(v['p(50)'] || 0).toFixed(2)}ms ` +
                `p95=${(v['p(95)'] || 0).toFixed(2)}ms ` +
                `p99=${(v['p(99)'] || 0).toFixed(2)}ms`
            );
        } else if (m.type === 'counter') {
            lines.push(`  ${name.padEnd(30)} count=${m.values.count}  rate=${(m.values.rate || 0).toFixed(2)}/s`);
        } else if (m.type === 'rate') {
            lines.push(`  ${name.padEnd(30)} rate=${((m.values.rate || 0) * 100).toFixed(2)}%`);
        } else if (m.type === 'gauge') {
            lines.push(`  ${name.padEnd(30)} value=${m.values.value}  min=${m.values.min}  max=${m.values.max}`);
        }
    }
    lines.push('');
    return lines.join('\n');
}
