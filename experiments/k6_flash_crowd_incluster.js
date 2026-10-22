import http from 'k6/http';
import { check, randomSeed } from 'k6';
import { Trend, Counter } from 'k6/metrics';
import exec from 'k6/execution';

// ============================================================================
// k6 Flash-Crowd IN-CLUSTER — un Job per cluster, carico EST-OVEST
// ============================================================================
//
// PERCHE'
// -------
// Il k6 esterno via Ingress e' traffico nord-sud e BYPASSA la CEC del GSLB
// (overflow). Per avere overflow ATTIVO il carico va generato DENTRO il
// cluster, verso il Service `frontend` (est-ovest): la CEC lo redistribuisce
// cross-cluster per peso x repliche (= score DMOS).
//
// MODELLO
// -------
// Un Job k6 per cluster colpisce `http://frontend` (la sua ClusterIP) a un
// RATE proprio = il "peso" di domanda di quel cluster. Stesso profilo a 5 fasi
// del test esterno; cambia solo il rate (startRate=baseline, picco=peak).
//
//   c01 (caldo, regime con): peak alto  ;  altri: peak basso
//   uni: tutti uguali
//
// La domanda la genera questo Job (origine), e DMOS la misura lato client.
// Il serving e' deciso dalla CEC: se lo score concentra, resta locale.
//
// OUTPUT
// ------
// Un Job non puo' scrivere file sul laptop: il summary JSON e' emesso sullo
// stdout fra i marcatori ===DMOS_FC_JSON_BEGIN=== / ===DMOS_FC_JSON_END===,
// e il driver (run_flash_incluster.sh) lo estrae dai log e aggrega i 6 Job.
//
// METRICA PRIMARIA: violazione SLO = frazione di richieste >SLO_MS o fallite.
// Esposta come CONTEGGI (slo_violations, total) cosi' l'aggregazione globale
// fra i 6 Job e' ESATTA (i percentili invece si combinano per media pesata).
//
// ENV
//   DMOS_TARGET_URL   default http://frontend   (Service ClusterIP, est-ovest)
//   DMOS_RUN_LABEL    etichetta run
//   DMOS_CLUSTER      nome cluster (per il summary)
//   DMOS_BASELINE_RPS rate baseline DI QUESTO cluster
//   DMOS_PEAK_RPS     rate di picco DI QUESTO cluster
//   DMOS_PHASE_TIMES  "180,300,600,720,900"
//   DMOS_SLO_MS       soglia violazione SLO (default 2000)
//   DMOS_VU_FACTOR DMOS_REQ_TIMEOUT_S DMOS_RNG_SEED
// ============================================================================

const RUN_LABEL    = __ENV.DMOS_RUN_LABEL    || 'default';
const CLUSTER      = __ENV.DMOS_CLUSTER      || 'unknown';
const TARGET_URL   = __ENV.DMOS_TARGET_URL   || 'http://frontend';
const BASELINE_RPS = Math.max(1, Math.round(parseFloat(__ENV.DMOS_BASELINE_RPS || '10')));
const PEAK_RPS     = Math.max(1, Math.round(parseFloat(__ENV.DMOS_PEAK_RPS     || '60')));
const SLO_MS       = parseInt(__ENV.DMOS_SLO_MS       || '2000');
const VU_FACTOR    = parseFloat(__ENV.DMOS_VU_FACTOR  || '20');
const REQ_TIMEOUT_S= parseInt(__ENV.DMOS_REQ_TIMEOUT_S|| '14');
const RNG_SEED     = parseInt(__ENV.DMOS_RNG_SEED     || '20260521');

const _phaseDefaults = '180,300,600,720,900';
const PHASE_TIMES = (__ENV.DMOS_PHASE_TIMES || _phaseDefaults)
    .split(',').map((s) => parseInt(s.trim()));
const [T_END_BASELINE, T_END_SPIKE, T_END_PEAK, T_END_DECAY, T_END_COOLDOWN] = PHASE_TIMES;

export const options = {
    scenarios: {
        flash_crowd: {
            executor: 'ramping-arrival-rate',
            startRate: BASELINE_RPS,
            timeUnit: '1s',
            preAllocatedVUs: Math.max(50, PEAK_RPS * 2),
            maxVUs: Math.ceil(Math.max(PEAK_RPS, BASELINE_RPS) * VU_FACTOR),
            stages: [
                { duration: `${T_END_BASELINE}s`, target: BASELINE_RPS },
                { duration: `${T_END_SPIKE - T_END_BASELINE}s`, target: PEAK_RPS },
                { duration: `${T_END_PEAK - T_END_SPIKE}s`, target: PEAK_RPS },
                { duration: `${T_END_DECAY - T_END_PEAK}s`, target: BASELINE_RPS },
                { duration: `${T_END_COOLDOWN - T_END_DECAY}s`, target: BASELINE_RPS },
            ],
            gracefulStop: '30s',
        },
    },
    summaryTrendStats: ['avg', 'min', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
    tags: { run_label: RUN_LABEL, cluster: CLUSTER },
};

const PHASES = ['baseline', 'spike', 'peak', 'decay', 'cooldown'];
const phaseLatency = {}, phaseReq = {}, phaseSlo = {};
for (const ph of PHASES) {
    phaseLatency[ph] = new Trend(`phase_${ph}_latency`, true);
    phaseReq[ph]     = new Counter(`phase_${ph}_requests`);
    phaseSlo[ph]     = new Counter(`phase_${ph}_slo_violations`);
}

const CHECKOUT_DATA = {
    email: 'test@example.com', street_address: '123 Test St', zip_code: '10001',
    city: 'New York', state: 'NY', country: 'US',
    credit_card_number: '4432801561520454', credit_card_expiration_month: '1',
    credit_card_expiration_year: '2030', credit_card_cvv: '672',
};
const ENDPOINTS = [
    { path: '/',                   weight: 0.40, method: 'GET',  body: null },
    { path: '/product/OLJCESPC7Z', weight: 0.15, method: 'GET',  body: null },
    { path: '/product/66VCHSJNUP', weight: 0.10, method: 'GET',  body: null },
    { path: '/cart',               weight: 0.10, method: 'GET',  body: null },
    { path: '/cart',               weight: 0.15, method: 'POST', body: { product_id: 'OLJCESPC7Z', quantity: 1 } },
    { path: '/setCurrency',        weight: 0.05, method: 'POST', body: { currency_code: 'EUR' } },
    { path: '/cart/checkout',      weight: 0.05, method: 'POST', body: CHECKOUT_DATA },
];
function pickWeighted(items) {
    const total = items.reduce((a, it) => a + it.weight, 0);
    let r = Math.random() * total;
    for (const it of items) { r -= it.weight; if (r <= 0) return it; }
    return items[items.length - 1];
}
function currentPhase(s) {
    if (s < T_END_BASELINE) return 'baseline';
    if (s < T_END_SPIKE)    return 'spike';
    if (s < T_END_PEAK)     return 'peak';
    if (s < T_END_DECAY)    return 'decay';
    return 'cooldown';
}

export function setup() {
    console.log(`[FC-INCLUSTER] ${RUN_LABEL} cluster=${CLUSTER} target=${TARGET_URL} ` +
                `baseline=${BASELINE_RPS} peak=${PEAK_RPS} SLO=${SLO_MS}ms`);
    return {};
}

let _seeded = false;
export default function () {
    if (!_seeded) { randomSeed(RNG_SEED + exec.vu.idInTest); _seeded = true; }
    const elapsed = exec.instance.currentTestRunDuration / 1000;
    const phase = currentPhase(elapsed);
    const ep = pickWeighted(ENDPOINTS);
    const url = TARGET_URL + ep.path;
    const params = { timeout: `${REQ_TIMEOUT_S}s`,
        tags: { endpoint: ep.path, phase: phase, method: ep.method } };
    const res = ep.method === 'POST' ? http.post(url, ep.body, params)
                                     : http.get(url, params);
    const is5xx = res.status >= 500 || res.status === 0;
    const sloViol = is5xx || res.timings.duration > SLO_MS;
    check(res, { 'status<500': (r) => r.status < 500 });
    phaseLatency[phase].add(res.timings.duration);
    phaseReq[phase].add(1);
    if (sloViol) phaseSlo[phase].add(1);
}

export function handleSummary(data) {
    const phaseDur = (ph) => (
        ph === 'baseline' ? T_END_BASELINE :
        ph === 'spike'    ? T_END_SPIKE - T_END_BASELINE :
        ph === 'peak'     ? T_END_PEAK - T_END_SPIKE :
        ph === 'decay'    ? T_END_DECAY - T_END_PEAK :
                            T_END_COOLDOWN - T_END_DECAY);
    const perPhase = {};
    for (const ph of PHASES) {
        const lat = data.metrics[`phase_${ph}_latency`];
        const req = data.metrics[`phase_${ph}_requests`];
        const slo = data.metrics[`phase_${ph}_slo_violations`];
        const n = req ? req.values.count : 0;
        if (!lat || n === 0) continue;
        const viol = slo ? slo.values.count : 0;
        perPhase[ph] = {
            requests: n, slo_violations: viol,
            slo_violation_rate: n > 0 ? viol / n : 0,
            rps: n / phaseDur(ph),
            p50: lat.values['p(50)'] || 0,
            p95: lat.values['p(95)'] || 0,
            p99: lat.values['p(99)'] || 0,
        };
    }
    const gLat = data.metrics.http_req_duration;
    const gReq = data.metrics.http_reqs;
    const gFail = data.metrics.http_req_failed;
    const droppedIter = data.metrics.dropped_iterations ? data.metrics.dropped_iterations.values.count : 0;
    const doneIter = data.metrics.iterations ? data.metrics.iterations.values.count : 0;
    const offered = droppedIter + doneIter;
    const droppedPct = offered > 0 ? droppedIter / offered : 0;

    // totale violazioni SLO sull'intero run (somma delle fasi)
    let totReq = 0, totViol = 0;
    for (const ph of PHASES) {
        const req = data.metrics[`phase_${ph}_requests`];
        const slo = data.metrics[`phase_${ph}_slo_violations`];
        if (req) totReq += req.values.count;
        if (slo) totViol += slo.values.count;
    }

    const report = {
        run_label: RUN_LABEL, cluster: CLUSTER, target: TARGET_URL,
        timestamp_iso: new Date().toISOString(),
        config: { baseline_rps: BASELINE_RPS, peak_rps: PEAK_RPS,
                  slo_ms: SLO_MS, phase_times_s: PHASE_TIMES },
        validity: { run_valid: droppedPct < 0.005, dropped_iterations: droppedIter,
                    dropped_pct: droppedPct },
        // conteggi per aggregazione ESATTA fra i 6 Job:
        totals: { requests: totReq, slo_violations: totViol,
                  slo_violation_rate: totReq > 0 ? totViol / totReq : 0 },
        peak: perPhase.peak || null,
        per_phase: perPhase,
        global: gLat ? {
            total_requests: gReq.values.count,
            success_rps: gReq.values.count * (gFail ? (1 - gFail.values.rate) : 1) / T_END_COOLDOWN,
            p50_ms: gLat.values['p(50)'], p95_ms: gLat.values['p(95)'],
            p99_ms: gLat.values['p(99)'],
            http_failure_rate: gFail ? gFail.values.rate : null,
        } : null,
    };

    const json = JSON.stringify(report);
    const out = [
        '', '===DMOS_FC_JSON_BEGIN===', json, '===DMOS_FC_JSON_END===', '',
        `[FC-INCLUSTER] ${CLUSTER}: req=${totReq} SLO_viol=${totViol} ` +
        `(${(report.totals.slo_violation_rate * 100).toFixed(2)}%) ` +
        `peak_p95=${perPhase.peak ? perPhase.peak.p95.toFixed(0) : 'n/a'}ms ` +
        `valid=${report.validity.run_valid}`, '',
    ].join('\n');
    return { stdout: out };
}
