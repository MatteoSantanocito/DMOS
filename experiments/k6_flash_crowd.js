import http from 'k6/http';
import { check, randomSeed } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';
import exec from 'k6/execution';

// ============================================================================
// k6 Flash-Crowd Test — Paper-Quality Multi-Cluster Eval
// ============================================================================
//
// Scopo
// -----
// Generare un workload realistico tipo "flash crowd" (viral event, Black
// Friday, ddos-burst) per valutare DMOS contro baseline (single cluster /
// round-robin / no-predictor). Output e' progettato per essere riproducibile
// e fornire metriche e plot per paper IEEE.
//
// Design del workload (5 fasi, totale 15 min)
// -------------------------------------------
//
//   Fase 1: BASELINE       [0  -180s]    50 rps    geo: EU-heavy (DE 30%, FR 20%, PL 15%, NO 15%, US-VA 10%, SG 10%)
//   Fase 2: SPIKE          [180-300s]    ramp 50 -> PEAK_RPS in 120s
//                                        geo shift: SPIKE_REGION arriva a 60%
//                                        (simula "trending sul forum DE" o ddos
//                                        regionale)
//   Fase 3: SUSTAINED PEAK [300-600s]    PEAK_RPS rps    geo: SPIKE_REGION 60%
//                                        durata 5 min per dare al predittore L2
//                                        tempo di adattarsi
//   Fase 4: DECAY          [600-720s]    ramp PEAK_RPS -> 50 rps in 120s
//                                        geo: torna alla distribuzione baseline
//   Fase 5: COOLDOWN       [720-900s]    50 rps        verifica che il sistema
//                                        torni a stato baseline pulito
//
// Hyp: DMOS-on dovrebbe:
//   - Mantenere p95 < 1s durante SPIKE (DMOS-off lo lascia esplodere)
//   - Mantenere error rate < 2% (DMOS-off ne ha 10-30%)
//   - Ridistribuire repliche geograficamente: SPIKE_REGION non passa max_replicas,
//     i cluster vicini si gonfiano per assorbire
//
// Comparison strategies (esegui lo stesso test variando --tag run_label)
// ----------------------------------------------------------------------
//   1. dmos-off       : DMOS spento, repliche fisse 1/1/1/1/1/1
//   2. dmos-l1-only   : DMOS L1 (reattivo, no predittore)
//   3. dmos-l1l2      : DMOS completo (L1 + L2 predittore)
//   4. round-robin    : naive ingress weights uniformi, no scheduler
//   5. nearest-only   : tutti gli utenti vanno al cluster "piu' vicino" (single)
//
// Variabili env
// -------------
//   DMOS_RUN_LABEL       : etichetta del run, finisce nel JSON + tag (default "default")
//   DMOS_BASELINE_RPS    : rps di baseline (default 50)
//   DMOS_PEAK_RPS        : rps di picco durante SPIKE+SUSTAINED (default 400)
//   DMOS_SPIKE_REGION    : cluster che riceve il 60% del traffico durante spike
//                          (default ms-cluster-01)
//   DMOS_SPIKE_WEIGHT    : peso del cluster sotto spike (default 0.60)
//   DMOS_PHASE_TIMES     : override timestamps in secondi, "180,300,600,720,900"
//                          (default: i valori sopra)
//   DMOS_VU_FACTOR       : maxVUs = PEAK_RPS x questo (default 20). Alzalo se il
//                          summary segnala "RUN NON VALIDO" (VU starvation).
//                          Worst case: PEAK_RPS x REQ_TIMEOUT_S VU servono se
//                          ogni richiesta arriva al timeout (scenario dmos-off
//                          saturo). 20 copre 400 rps x 14s = 5600 con margine.
//   DMOS_REQ_TIMEOUT_S   : timeout per richiesta in secondi (default 14)
//   DMOS_RNG_SEED        : seed RNG per riproducibilita' (default 20260521)
//
// Uso tipico
// ----------
//   # Run completo dmos-l1l2 con output JSON archiviato:
//   k6 run \
//       -e DMOS_RUN_LABEL=dmos-l1l2 \
//       -e DMOS_PEAK_RPS=400 \
//       --out json=experiments/results/flash_crowd_$(date +%Y%m%d_%H%M)_dmos-l1l2.json \
//       experiments/k6_flash_crowd.js
//
//   # Confronto baseline dmos-off:
//   k6 run \
//       -e DMOS_RUN_LABEL=dmos-off \
//       --out json=experiments/results/flash_crowd_$(date +%Y%m%d_%H%M)_dmos-off.json \
//       experiments/k6_flash_crowd.js
//
// Tag esposti per slicing
// -----------------------
//   ingress  : nome cluster (es. ms-cluster-01-DE)
//   endpoint : path richiesto (es. /, /cart, /cart/checkout)
//   run_label: valore di DMOS_RUN_LABEL (per joinare run multipli)
//
// Output
// ------
// 1. JSON con --out json (k6 events: timestamp, latency, status, tag)
// 2. Custom metrics per-phase: phase_<name>_latency, phase_<name>_errors
// 3. Custom summary stdout: tabella per-fase con p50/p95/p99, error rate, throughput
// 4. JSON summary: experiments/results/flash_crowd_summary.json
// ============================================================================

// ── Env config ──────────────────────────────────────────────────────────────

const RUN_LABEL     = __ENV.DMOS_RUN_LABEL     || 'default';
const BASELINE_RPS  = parseInt(__ENV.DMOS_BASELINE_RPS  || '50');
const PEAK_RPS      = parseInt(__ENV.DMOS_PEAK_RPS      || '400');
const SPIKE_REGION  = __ENV.DMOS_SPIKE_REGION  || 'ms-cluster-01';
const SPIKE_WEIGHT  = parseFloat(__ENV.DMOS_SPIKE_WEIGHT || '0.60');

// VU pool: deve sostenere PEAK_RPS anche a latenza alta (vedi options.scenarios).
const VU_FACTOR     = parseFloat(__ENV.DMOS_VU_FACTOR   || '20');
const REQ_TIMEOUT_S = parseInt(__ENV.DMOS_REQ_TIMEOUT_S || '14');
// Seed RNG per riproducibilita' delle scelte casuali (endpoint/ingress mix).
const RNG_SEED      = parseInt(__ENV.DMOS_RNG_SEED      || '20260521');

// Phase timestamps in seconds (cumulative from t=0)
const _phaseDefaults = '180,300,600,720,900';
const PHASE_TIMES = (__ENV.DMOS_PHASE_TIMES || _phaseDefaults)
    .split(',').map((s) => parseInt(s.trim()));
const [T_END_BASELINE, T_END_SPIKE, T_END_PEAK, T_END_DECAY, T_END_COOLDOWN] = PHASE_TIMES;

// ── Load cluster info from cache (single source of truth) ──────────────────
// k6 open() risolve i path relativi a QUESTO .js (non al CWD), quindi
// dal experiments/ dobbiamo salire di una dir.

// Path del test-env: di default ../.cache/test-env.json (esecuzione dal repo);
// in modo remoto (k6 sulla VM) run_flash_crowd.sh lo copia accanto allo script
// e imposta DMOS_TEST_ENV=test-env.json.
const _envFile = __ENV.DMOS_TEST_ENV || '../.cache/test-env.json';
let _envData;
try {
    _envData = JSON.parse(open(_envFile));
} catch (e) {
    throw new Error(
        `Cannot read ${_envFile}. Run 'bash scripts/render-runtime-files.sh' first.`
    );
}
const _clusters = _envData.clusters;
if (!_clusters || _clusters.length < 2) {
    throw new Error(`${_envFile} contains fewer than 2 clusters; need at least 2.`);
}

// ── Geo weights for baseline vs spike phase ────────────────────────────────
//
// Baseline weights: EU-heavy realistic distribution.
// Custom override via env DMOS_INGRESS_W1..WN (1-indexed by cluster order).
//
// During SPIKE/PEAK phase: SPIKE_REGION gets SPIKE_WEIGHT, others share the rest.

function defaultBaselineWeights() {
    // Mapping intelligente: piu' peso ai cluster "vicini" e con bassa latenza
    // (DE/FR/PL/NO sono Europa, US-VA e SG sono lontani).
    const regionWeights = {
        'DE': 0.30, 'FR': 0.20, 'PL': 0.15, 'NO': 0.15,
        'US-VA': 0.10, 'SG': 0.10,
        // Catch-all per region non previste:
    };
    const ws = _clusters.map((c) => regionWeights[c.region] !== undefined
        ? regionWeights[c.region]
        : 1.0 / _clusters.length);
    // Normalize so they sum to 1.0
    const sum = ws.reduce((a, b) => a + b, 0);
    return ws.map((w) => w / sum);
}

function envOverrideWeights() {
    const overrides = _clusters.map((_, i) => __ENV[`DMOS_INGRESS_W${i + 1}`]);
    if (overrides.some((v) => v !== undefined)) {
        return overrides.map((v, i) => v !== undefined ? parseFloat(v) : 1.0 / _clusters.length);
    }
    return null;
}

const BASELINE_WEIGHTS = envOverrideWeights() || defaultBaselineWeights();

// Spike phase weights: override via DMOS_SPIKE_W{i} (1-indexed), oppure
// default da SPIKE_REGION + SPIKE_WEIGHT (~legacy single-spike).
//
// Override usage:
//   DMOS_SPIKE_W1=0.25 DMOS_SPIKE_W2=0.12 DMOS_SPIKE_W3=0.10 \
//   DMOS_SPIKE_W4=0.25 DMOS_SPIKE_W5=0.13 DMOS_SPIKE_W6=0.15
// Pesi non normalizzati vengono normalizzati per somma. Se mancano alcuni
// W{i}, quei cluster ricevono 0 (esplicito).
//
// Multi-modal flash crowds richiedono questa override per testare il
// claim multi-objective scheduling (vedi metodologia Bodík SoCC'10).
function envOverrideSpikeWeights() {
    const overrides = _clusters.map((_, i) => __ENV[`DMOS_SPIKE_W${i + 1}`]);
    if (!overrides.some((v) => v !== undefined)) return null;
    const parsed = overrides.map((v) => v !== undefined ? parseFloat(v) : 0.0);
    const sum = parsed.reduce((a, b) => a + b, 0);
    if (sum <= 0) {
        throw new Error('DMOS_SPIKE_W* override sums to 0; specifica almeno un peso > 0');
    }
    return parsed.map((w) => w / sum);
}

let SPIKE_WEIGHTS;
const _spikeOverride = envOverrideSpikeWeights();
if (_spikeOverride !== null) {
    SPIKE_WEIGHTS = _spikeOverride;
} else {
    // Legacy single-spike: SPIKE_REGION gets SPIKE_WEIGHT, rest splits remainder.
    const _spikeIdx = _clusters.findIndex((c) => c.name === SPIKE_REGION);
    if (_spikeIdx < 0) {
        throw new Error(
            `DMOS_SPIKE_REGION="${SPIKE_REGION}" non trovato nel cluster fleet. ` +
            `Disponibili: ${_clusters.map((c) => c.name).join(', ')}`
        );
    }
    SPIKE_WEIGHTS = _clusters.map((_, i) => {
        if (i === _spikeIdx) return SPIKE_WEIGHT;
        const others = _clusters.length - 1;
        return (1.0 - SPIKE_WEIGHT) / others;
    });
}

const INGRESSES = _clusters.map((c, i) => ({
    name: `${c.name}-${c.region}`,
    cluster_name: c.name,
    region: c.region,
    url: c.ingress_url,
    weight_baseline: BASELINE_WEIGHTS[i],
    weight_spike: SPIKE_WEIGHTS[i],
}));

// ── k6 scenario: ramping-arrival-rate with phase shape ─────────────────────

export const options = {
    scenarios: {
        flash_crowd: {
            executor: 'ramping-arrival-rate',
            startRate: BASELINE_RPS,
            timeUnit: '1s',
            // Pool VU — CRITICO per la validita' del confronto. Sotto
            // saturazione (scenario dmos-off) la latenza puo' arrivare al
            // timeout: per sostenere comunque PEAK_RPS servono ~PEAK_RPS x
            // latenza[s] VU. Se il pool e' troppo piccolo k6 NON regge il rate
            // -> dropped_iterations -> dmos-off riceve meno carico di dmos-on
            // e il confronto e' falsato. maxVUs = PEAK_RPS x VU_FACTOR.
            preAllocatedVUs: Math.max(200, PEAK_RPS * 2),
            maxVUs: Math.ceil(PEAK_RPS * VU_FACTOR),
            stages: [
                // Fase 1 BASELINE: 0 - T_END_BASELINE = 180s a BASELINE_RPS
                { duration: `${T_END_BASELINE}s`, target: BASELINE_RPS },
                // Fase 2 SPIKE: ramp BASELINE -> PEAK in (T_END_SPIKE - T_END_BASELINE)s = 120s
                { duration: `${T_END_SPIKE - T_END_BASELINE}s`, target: PEAK_RPS },
                // Fase 3 SUSTAINED PEAK: hold PEAK_RPS for (T_END_PEAK - T_END_SPIKE)s = 300s
                { duration: `${T_END_PEAK - T_END_SPIKE}s`, target: PEAK_RPS },
                // Fase 4 DECAY: ramp PEAK -> BASELINE in (T_END_DECAY - T_END_PEAK)s = 120s
                { duration: `${T_END_DECAY - T_END_PEAK}s`, target: BASELINE_RPS },
                // Fase 5 COOLDOWN: hold BASELINE for (T_END_COOLDOWN - T_END_DECAY)s = 180s
                { duration: `${T_END_COOLDOWN - T_END_DECAY}s`, target: BASELINE_RPS },
            ],
            gracefulStop: '30s',
        },
    },
    // Soglia di VALIDITA' del run: se k6 ha droppato iterazioni il pool VU non
    // ha retto il rate offerto -> il run NON e' confrontabile. Fa uscire k6 con
    // codice != 0, cosi' run_flash_crowd.sh puo' marcare il run come invalido.
    thresholds: {
        // Guard grossolano (~1% di ~192k iterazioni). Il gate FINE e'
        // percentuale, in handleSummary (vedi MAX_DROPPED_PCT).
        dropped_iterations: ['count<2000'],
    },
    summaryTrendStats: ['avg', 'min', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
    // Tag globale per identificare il run nel JSON output
    tags: { run_label: RUN_LABEL },
};

// ── Per-phase custom metrics ───────────────────────────────────────────────

const PHASES = ['baseline', 'spike', 'peak', 'decay', 'cooldown'];
const phaseLatency = {};
const phaseRequests = {};
const phaseErrors = {};
for (const ph of PHASES) {
    phaseLatency[ph]  = new Trend(`phase_${ph}_latency`, true);
    phaseRequests[ph] = new Counter(`phase_${ph}_requests`);
    phaseErrors[ph]   = new Counter(`phase_${ph}_errors`);
}

// Per-cluster latency and error rate (slice by ingress).
// k6 NON accetta `-` nei nomi delle metriche, quindi sostituiamo con `_`.
// La chiave dell'oggetto JS resta `cluster_name` (con trattini) per lookup
// comodo dal pickIngress; solo il nome della metrica esposta a Prometheus/JSON
// e' sanitized.
function _safeName(name) { return name.replace(/[^A-Za-z0-9_]/g, '_'); }
const clusterLatency = {};
const clusterErrors = {};
const clusterRequests = {};
for (const ing of INGRESSES) {
    const safe = _safeName(ing.cluster_name);
    clusterLatency[ing.cluster_name]  = new Trend(`cluster_${safe}_latency`, true);
    clusterErrors[ing.cluster_name]   = new Rate(`cluster_${safe}_errors`);
    clusterRequests[ing.cluster_name] = new Counter(`cluster_${safe}_requests`);
}

// ── Endpoint mix (Online Boutique production-like) ─────────────────────────

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

function pickWeighted(items, weightField) {
    const total = items.reduce((acc, it) => acc + (weightField ? it[weightField] : it.weight), 0);
    let r = Math.random() * total;
    for (const it of items) {
        r -= weightField ? it[weightField] : it.weight;
        if (r <= 0) return it;
    }
    return items[items.length - 1];
}

// ── setup() logs the config once ───────────────────────────────────────────

export function setup() {
    const baseDetails = INGRESSES.map((ing) =>
        `${ing.name}=${ing.weight_baseline.toFixed(3)}`
    ).join(', ');
    const spikeDetails = INGRESSES.map((ing) =>
        `${ing.name}=${ing.weight_spike.toFixed(3)}`
    ).join(', ');
    console.log(`[FLASH CROWD] Run label: ${RUN_LABEL}`);
    console.log(`[FLASH CROWD] Baseline ${BASELINE_RPS} rps, Peak ${PEAK_RPS} rps`);
    console.log(`[FLASH CROWD] Phases (s): ${PHASE_TIMES.join(' -> ')}`);
    console.log(`[FLASH CROWD] Spike region: ${SPIKE_REGION} (weight ${SPIKE_WEIGHT})`);
    console.log(`[FLASH CROWD] Baseline weights: ${baseDetails}`);
    console.log(`[FLASH CROWD] Spike weights:    ${spikeDetails}`);
    return { startTime: Date.now() };
}

// ── currentPhase: identifica la fase corrente in base al tempo elapsed ─────

function currentPhase(elapsedSec) {
    if (elapsedSec < T_END_BASELINE) return 'baseline';
    if (elapsedSec < T_END_SPIKE)    return 'spike';
    if (elapsedSec < T_END_PEAK)     return 'peak';
    if (elapsedSec < T_END_DECAY)    return 'decay';
    return 'cooldown';
}

// During SPIKE/PEAK use spike weights, else use baseline weights.
function pickIngress(phase) {
    const useSpike = (phase === 'spike' || phase === 'peak');
    return pickWeighted(INGRESSES, useSpike ? 'weight_spike' : 'weight_baseline');
}

// ── Default VU function ─────────────────────────────────────────────────────

let _seeded = false;

export default function (data) {
    // Seed dell'RNG per-VU: rende riproducibili le scelte casuali (endpoint e
    // ingress mix). Seed distinto per VU cosi' le VU non sono identiche.
    if (!_seeded) { randomSeed(RNG_SEED + exec.vu.idInTest); _seeded = true; }

    // Tempo dal clock INTERNO di k6 (lo stesso che guida il ramp): fase e
    // geo-shift restano sincronizzati col carico. Date.now()-startTime derivava.
    const elapsedSec = exec.instance.currentTestRunDuration / 1000;
    const phase = currentPhase(elapsedSec);

    const myIngress = pickIngress(phase);
    const ep = pickWeighted(ENDPOINTS);
    const url = myIngress.url + ep.path;

    const params = {
        timeout: `${REQ_TIMEOUT_S}s`,
        tags: {
            ingress:  myIngress.name,
            cluster:  myIngress.cluster_name,
            endpoint: ep.path,
            phase:    phase,
            method:   ep.method,
        },
    };

    let res;
    if (ep.method === 'POST') {
        res = http.post(url, ep.body, params);
    } else {
        res = http.get(url, params);
    }

    const is5xx = res.status >= 500 || res.status === 0;

    check(res, {
        'status is not 5xx':           (r) => r.status < 500,
        'status is not 0 (conn err)':  (r) => r.status !== 0,
        'status is not 4xx unexpected': (r) => r.status < 400 || r.status === 404,
    });

    // Per-phase metrics
    phaseLatency[phase].add(res.timings.duration);
    phaseRequests[phase].add(1);
    if (is5xx) phaseErrors[phase].add(1);

    // Per-cluster metrics
    clusterLatency[myIngress.cluster_name].add(res.timings.duration);
    clusterErrors[myIngress.cluster_name].add(is5xx);
    clusterRequests[myIngress.cluster_name].add(1);
}

// ── Custom summary: tabella per-fase + per-cluster ─────────────────────────

export function handleSummary(data) {
    const lines = [];
    lines.push('');
    lines.push('='.repeat(80));
    lines.push(`  FLASH CROWD TEST — ${RUN_LABEL}`);
    lines.push(`  ${_clusters.length} cluster, baseline ${BASELINE_RPS} rps, peak ${PEAK_RPS} rps`);
    lines.push(`  Spike region: ${SPIKE_REGION} (weight ${SPIKE_WEIGHT})`);
    lines.push(`  Phases (s): baseline 0-${T_END_BASELINE}, spike ${T_END_BASELINE}-${T_END_SPIKE}, ` +
               `peak ${T_END_SPIKE}-${T_END_PEAK}, decay ${T_END_PEAK}-${T_END_DECAY}, ` +
               `cooldown ${T_END_DECAY}-${T_END_COOLDOWN}`);
    lines.push('='.repeat(80));
    lines.push('');

    // Tabella per-fase
    lines.push('  PER-PHASE METRICS');
    lines.push('  ' + '-'.repeat(75));
    lines.push('  Phase     | Requests | rps   |   p50  |   p95  |   p99  | Err%');
    lines.push('  ' + '-'.repeat(75));

    const phaseRes = {};
    for (const ph of PHASES) {
        const lat = data.metrics[`phase_${ph}_latency`];
        const req = data.metrics[`phase_${ph}_requests`];
        const err = data.metrics[`phase_${ph}_errors`];
        if (!lat || !req || req.values.count === 0) {
            lines.push(`  ${ph.padEnd(9)} | (no data)`);
            continue;
        }
        const phaseDuration = (
            ph === 'baseline' ? T_END_BASELINE :
            ph === 'spike'    ? T_END_SPIKE - T_END_BASELINE :
            ph === 'peak'     ? T_END_PEAK - T_END_SPIKE :
            ph === 'decay'    ? T_END_DECAY - T_END_PEAK :
                                T_END_COOLDOWN - T_END_DECAY
        );
        const totalReqs = req.values.count;
        const totalErrs = err ? err.values.count : 0;
        const errorRate = totalReqs > 0 ? totalErrs / totalReqs : 0;
        const rps = totalReqs / phaseDuration;
        const p50 = lat.values['p(50)'] || 0;
        const p95 = lat.values['p(95)'] || 0;
        const p99 = lat.values['p(99)'] || 0;
        phaseRes[ph] = { requests: totalReqs, rps, p50, p95, p99, errorRate };

        lines.push(
            `  ${ph.padEnd(9)} | ` +
            `${String(totalReqs).padStart(8)} | ` +
            `${rps.toFixed(1).padStart(5)} | ` +
            `${p50.toFixed(0).padStart(5)}ms | ` +
            `${p95.toFixed(0).padStart(5)}ms | ` +
            `${p99.toFixed(0).padStart(5)}ms | ` +
            `${(errorRate * 100).toFixed(2).padStart(5)}%`
        );
    }
    lines.push('');

    // Tabella per-cluster (durante SPIKE+PEAK, le fasi piu' indicative)
    lines.push('  PER-CLUSTER METRICS (intero run)');
    lines.push('  ' + '-'.repeat(75));
    lines.push('  Cluster              | Requests | p50    | p95    | p99    | Err%');
    lines.push('  ' + '-'.repeat(75));
    const clusterRes = {};
    for (const ing of INGRESSES) {
        const safe = _safeName(ing.cluster_name);
        const lat = data.metrics[`cluster_${safe}_latency`];
        const req = data.metrics[`cluster_${safe}_requests`];
        const errRate = data.metrics[`cluster_${safe}_errors`];
        // I Trend di k6 NON espongono values.count: per capire se c'e' stato
        // traffico si controlla il Counter cluster_*_requests (come per le fasi).
        if (!lat || !req || (req.values.count || 0) === 0) {
            lines.push(`  ${ing.name.padEnd(20)} | (no data)`);
            continue;
        }
        const reqs = req.values.count || 0;
        const p50  = lat.values['p(50)'] || 0;
        const p95  = lat.values['p(95)'] || 0;
        const p99  = lat.values['p(99)'] || 0;
        const er   = errRate ? (errRate.values.rate || 0) : 0;
        clusterRes[ing.cluster_name] = { requests: reqs, p50, p95, p99, errorRate: er };

        lines.push(
            `  ${ing.name.padEnd(20)} | ` +
            `${String(reqs).padStart(8)} | ` +
            `${p50.toFixed(0).padStart(4)}ms | ` +
            `${p95.toFixed(0).padStart(4)}ms | ` +
            `${p99.toFixed(0).padStart(4)}ms | ` +
            `${(er * 100).toFixed(2).padStart(5)}%`
        );
    }

    lines.push('');
    lines.push('='.repeat(80));
    lines.push('  GLOBAL HTTP STATS');
    lines.push('='.repeat(80));
    const globalLat = data.metrics.http_req_duration;
    const globalReqs = data.metrics.http_reqs;
    const globalFail = data.metrics.http_req_failed;
    if (globalLat && globalReqs) {
        const ttotal = T_END_COOLDOWN;
        lines.push(`  Total requests:     ${globalReqs.values.count}`);
        lines.push(`  Avg throughput:     ${(globalReqs.values.count / ttotal).toFixed(1)} rps`);
        lines.push(`  p50/p95/p99:        ${globalLat.values['p(50)'].toFixed(0)}ms / ` +
                   `${globalLat.values['p(95)'].toFixed(0)}ms / ` +
                   `${globalLat.values['p(99)'].toFixed(0)}ms`);
        if (globalFail) {
            lines.push(`  HTTP failure rate:  ${(globalFail.values.rate * 100).toFixed(2)}%`);
        }
    }
    lines.push('='.repeat(80));
    lines.push('');

    // ── Verdetto di VALIDITA' del run ───────────────────────────────────────
    // Se k6 ha droppato iterazioni, il pool VU non ha retto il rate offerto:
    // l'arm ha ricevuto meno carico del previsto e il confronto e' falsato.
    const droppedIter = data.metrics.dropped_iterations
        ? data.metrics.dropped_iterations.values.count : 0;
    const doneIter = data.metrics.iterations ? data.metrics.iterations.values.count : 0;
    const offeredIter = droppedIter + doneIter;
    const droppedPct = offeredIter > 0 ? droppedIter / offeredIter : 0;
    // Gate di validita' PERCENTUALE: il run e' valido se k6 ha consegnato
    // >= 99.5% del rate offerto. Un conteggio assoluto (es. <100) era troppo
    // severo su run lunghi (~192k iterazioni a 400 rps): 100 drop = 0.05%,
    // un'inezia che non altera p95/error-rate aggregati e non falsa il confronto.
    const MAX_DROPPED_PCT = 0.005;   // 0.5%
    const runValid = droppedPct < MAX_DROPPED_PCT;
    lines.push('  VALIDITA DEL RUN');
    lines.push('  ' + '-'.repeat(75));
    lines.push(`  Iterazioni: offerte ${offeredIter}, eseguite ${doneIter}, droppate ${droppedIter}`);
    if (runValid) {
        lines.push(`  -> RUN VALIDO: k6 ha sostenuto il rate offerto (drop ${(droppedPct * 100).toFixed(3)}%).`);
    } else {
        lines.push(`  -> RUN NON VALIDO: ${(droppedPct * 100).toFixed(2)}% iterazioni droppate -- il`);
        lines.push(`     pool VU non ha retto il carico, il confronto e' falsato. Rilancia con`);
        lines.push(`     DMOS_VU_FACTOR piu' alto o k6 su una macchina piu' capace.`);
    }
    lines.push('='.repeat(80));
    lines.push('');

    const machineReport = {
        run_label: RUN_LABEL,
        timestamp_iso: new Date().toISOString(),
        validity: {
            run_valid: runValid,
            dropped_iterations: droppedIter,
            dropped_pct: droppedPct,
        },
        config: {
            baseline_rps: BASELINE_RPS,
            peak_rps: PEAK_RPS,
            spike_region: SPIKE_REGION,
            spike_weight: SPIKE_WEIGHT,
            phase_times_s: PHASE_TIMES,
            num_clusters: _clusters.length,
            baseline_weights: BASELINE_WEIGHTS,
            spike_weights: SPIKE_WEIGHTS,
        },
        clusters: INGRESSES.map((ing) => ({
            name: ing.cluster_name, region: ing.region,
            ingress_url: ing.url,
            weight_baseline: ing.weight_baseline,
            weight_spike: ing.weight_spike,
        })),
        results: {
            per_phase: phaseRes,
            per_cluster: clusterRes,
            global: globalLat ? {
                total_requests: globalReqs.values.count,
                avg_rps: globalReqs.values.count / T_END_COOLDOWN,
                p50_ms: globalLat.values['p(50)'],
                p95_ms: globalLat.values['p(95)'],
                p99_ms: globalLat.values['p(99)'],
                avg_ms: globalLat.values.avg,
                http_failure_rate: globalFail ? globalFail.values.rate : null,
            } : null,
        },
    };

    return {
        stdout: lines.join('\n') + '\n' + textSummaryFallback(data),
        'flash_crowd_summary.json': JSON.stringify(machineReport, null, 2),
    };
}

function textSummaryFallback(data) {
    const lines = ['  --- Standard k6 Summary ---', ''];
    const interesting = [
        'http_req_duration', 'http_req_failed', 'http_reqs', 'dropped_iterations',
        'iteration_duration', 'iterations', 'vus', 'vus_max', 'checks',
    ];
    for (const name of interesting) {
        const m = data.metrics[name];
        if (!m) continue;
        if (m.type === 'trend') {
            const v = m.values;
            lines.push(
                `  ${name.padEnd(28)} avg=${(v.avg || 0).toFixed(2)}ms ` +
                `p50=${(v['p(50)'] || 0).toFixed(2)}ms ` +
                `p95=${(v['p(95)'] || 0).toFixed(2)}ms ` +
                `p99=${(v['p(99)'] || 0).toFixed(2)}ms`
            );
        } else if (m.type === 'counter') {
            lines.push(`  ${name.padEnd(28)} count=${m.values.count}  rate=${(m.values.rate || 0).toFixed(2)}/s`);
        } else if (m.type === 'rate') {
            lines.push(`  ${name.padEnd(28)} rate=${((m.values.rate || 0) * 100).toFixed(2)}%`);
        }
    }
    lines.push('');
    return lines.join('\n');
}
