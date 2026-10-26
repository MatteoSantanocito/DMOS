import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter } from 'k6/metrics';
import exec from 'k6/execution';

// ============================================================================
// k6 Capacity Test — Per-Cluster, per-Replica
// ============================================================================
//
// Scopo
// -----
// Misurare la capacita' reale (rps/replica) di un singolo cluster, ottenendo
// il valore da inserire come `capacity_req_per_sec` in config/services.yaml
// per ciascun servizio (default: frontend).
//
// Design
// ------
// - Usa scenari `constant-arrival-rate` consecutivi (NON ramping): ogni step
//   gira a un rate FISSO per N secondi. La misura non e' contaminata dalla
//   transizione, a differenza di `ramping-arrival-rate` dove il rate cambia
//   continuamente dentro lo stage.
// - Range di step: 5, 10, 20, 30, 40, 50, 60, 70, 80, 100 rps (10 step).
//   Override via env DMOS_CAPACITY_STEPS="5,10,20,...".
// - Ogni step: 15s settle + 90s misura = 105s. Totale ~17.5 min per cluster.
// - Tag custom `step` permette analisi post-test slicing per rate.
// - JSON output via flag `--out json=results/capacity_<cluster>_<ts>.json`.
//
// Preparazione (per ogni cluster da testare)
// ------------------------------------------
//   # Una sola replica del servizio target (cosi misuriamo la capacita' di 1 rep):
//   kubectl --kubeconfig <kc-target> -n online-boutique scale deployment <svc> --replicas=1
//
//   # Sugli altri 5 cluster, scala a 0 per evitare contaminazione da ClusterMesh
//   # (i Global Service di Cilium ridistribuiscono se non hai endpoint locali):
//   for n in 02 03 04 05 06; do
//       kubectl --kubeconfig ../dmos-iac/kubeconfig-ms-cluster-$n \
//           -n online-boutique scale deployment frontend --replicas=0
//   done
//
// Uso
// ---
//   # Default: cluster ms-cluster-01, frontend
//   k6 run experiments/k6_capacity_test.js
//
//   # Con override (cluster diverso, custom steps, JSON output):
//   k6 run \
//       -e DMOS_CLUSTER=ms-cluster-02 \
//       -e DMOS_STEPS="10,20,30,40,50" \
//       -e DMOS_STEP_DURATION_S=60 \
//       -e DMOS_SLA_P95_MS=200 \
//       --out json=experiments/results/capacity_ms-cluster-02_$(date +%s).json \
//       experiments/k6_capacity_test.js
//
// Variabili env
// -------------
//   DMOS_CLUSTER         : nome cluster (default ms-cluster-01)
//   DMOS_STEPS           : lista rps separati da virgola (default 5,10,20,30,40,50,60,70,80,100)
//   DMOS_STEP_DURATION_S : durata misurata per step in s (default 90)
//   DMOS_SETTLE_S        : durata settle pre-misura per step (default 15)
//   DMOS_SLA_P95_MS      : SLA p95 latency in ms per knee-point (default 200)
//   DMOS_SLA_ERR_RATE    : SLA error rate per knee-point (default 0.02 = 2%)
//   DMOS_REPLICAS        : numero repliche del servizio (per rps/replica, default 1)
//
// Output
// ------
// 1. Tabella stdout con per-step: rps target, rps misurato, p50/p95/p99, error%, SLA OK?
// 2. Knee-point detection: ultimo step con p95 < SLA_P95 AND error < SLA_ERR
// 3. Raccomandazione per services.yaml (80% del knee come margine)
// 4. JSON dettagliato (se --out json=path) per post-analysis grafici nel paper
// ============================================================================

// ── Configurazione (env-driven) ─────────────────────────────────────────────

const CLUSTER          = __ENV.DMOS_CLUSTER         || 'ms-cluster-01';
const STEP_DURATION_S  = parseInt(__ENV.DMOS_STEP_DURATION_S || '90');
const SETTLE_S         = parseInt(__ENV.DMOS_SETTLE_S        || '15');
const SLA_P95_MS       = parseInt(__ENV.DMOS_SLA_P95_MS      || '200');
const SLA_ERR_RATE     = parseFloat(__ENV.DMOS_SLA_ERR_RATE  || '0.02');
const REPLICAS         = parseInt(__ENV.DMOS_REPLICAS        || '1');

// Steps: rate per step in rps
const DEFAULT_STEPS = '5,10,20,30,40,50,60,70,80,100';
const STEPS = (__ENV.DMOS_STEPS || DEFAULT_STEPS)
    .split(',').map((s) => parseInt(s.trim()));

// Cluster URL: read from .cache/test-env.json (single source of truth).
// k6 open() resolves paths relative to THIS .js file (not CWD), quindi
// dal experiments/ dobbiamo salire di una dir.
const _envFile = '../.cache/test-env.json';
let _envData;
try {
    _envData = JSON.parse(open(_envFile));
} catch (e) {
    throw new Error(
        `Cannot read ${_envFile}. Run 'bash scripts/render-runtime-files.sh' first.`
    );
}
const CLUSTERS = Object.fromEntries(
    _envData.clusters.map((c) => [c.name, c.ingress_url])
);
const BASE_URL = CLUSTERS[CLUSTER];
if (!BASE_URL) {
    throw new Error(
        `Cluster sconosciuto: ${CLUSTER}. Disponibili: ${Object.keys(CLUSTERS).join(', ')}`
    );
}

// ── Build k6 scenarios: 1 scenario constant-arrival-rate per step ──────────
//
// Sequenza temporale:
//   step_0_settle:  [0, SETTLE_S]                          → rate = STEPS[0], non misurato
//   step_0_measure: [SETTLE_S, SETTLE_S + STEP_DURATION_S] → rate = STEPS[0], misurato
//   step_1_settle:  ...
//   ...
// La separazione settle/measure evita di catturare il transitorio iniziale
// (warmup connessioni TCP, JIT, ecc.) come parte della misura.

const scenarios = {};
let cursor = 0;
for (let i = 0; i < STEPS.length; i++) {
    const rate = STEPS[i];
    const settleName = `step${String(i).padStart(2, '0')}_${String(rate).padStart(3, '0')}rps_settle`;
    const measureName = `step${String(i).padStart(2, '0')}_${String(rate).padStart(3, '0')}rps_measure`;

    // Pool VU: dimensionato per il rate piu' alto + margine
    const maxVUs = Math.max(50, Math.ceil(rate * 3));

    scenarios[settleName] = {
        executor: 'constant-arrival-rate',
        rate: rate,
        timeUnit: '1s',
        duration: `${SETTLE_S}s`,
        preAllocatedVUs: maxVUs,
        maxVUs: maxVUs * 2,
        startTime: `${cursor}s`,
        tags: { step: String(rate).padStart(3, '0'), phase: 'settle' },
        exec: 'doRequest',
    };
    cursor += SETTLE_S;

    scenarios[measureName] = {
        executor: 'constant-arrival-rate',
        rate: rate,
        timeUnit: '1s',
        duration: `${STEP_DURATION_S}s`,
        preAllocatedVUs: maxVUs,
        maxVUs: maxVUs * 2,
        startTime: `${cursor}s`,
        tags: { step: String(rate).padStart(3, '0'), phase: 'measure' },
        exec: 'doRequest',
    };
    cursor += STEP_DURATION_S;
}

export const options = {
    scenarios: scenarios,
    summaryTrendStats: ['avg', 'min', 'p(50)', 'p(90)', 'p(95)', 'p(99)', 'max'],
    // No thresholds globali: il knee-point e' calcolato in handleSummary.
};

// ── Per-step custom metrics ─────────────────────────────────────────────────
// Solo per phase=measure (non settle).

const stepLatency  = {};
const stepRequests = {};
const stepErrors   = {};
for (const rate of STEPS) {
    const key = String(rate).padStart(3, '0');
    stepLatency[key]  = new Trend(`step_${key}rps_latency`, true);
    stepRequests[key] = new Counter(`step_${key}rps_requests`);
    stepErrors[key]   = new Counter(`step_${key}rps_errors`);
}

// ── Endpoint mix (production-like Online Boutique workload) ────────────────

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
    const total = items.reduce((acc, it) => acc + it.weight, 0);
    let r = Math.random() * total;
    for (const it of items) {
        r -= it.weight;
        if (r <= 0) return it;
    }
    return items[items.length - 1];
}

// ── VU exec function (called from each scenario) ───────────────────────────

export function doRequest() {
    const ep = pickWeighted(ENDPOINTS);
    const url = BASE_URL + ep.path;

    const params = {
        timeout: '30s',
        tags: { endpoint: ep.path },
    };

    let res;
    if (ep.method === 'POST') {
        res = http.post(url, ep.body, params);
    } else {
        res = http.get(url, params);
    }

    const is5xx = res.status >= 500 || res.status === 0;

    check(res, {
        'status is not 5xx': (r) => r.status < 500,
        'status is not 0 (connection error)': (r) => r.status !== 0,
    });

    // Estrai il tag `step` dallo scenario corrente per attribuire la misura
    // alla custom metric giusta. Il tag `phase` ci dice se siamo in misura.
    // Nota: i tag di scenario sono ereditati automaticamente da ogni req.
    // Possiamo leggerli via __ENV o exec.scenario.* ma il pattern piu' robusto
    // e' identificare il rate dal nome dello scenario.
    // k6 espone exec.scenario.name a runtime.
    const scenarioName = exec.scenario.name; // es. "step01_010rps_measure"
    const measureMatch = scenarioName.match(/step\d+_(\d+)rps_measure/);
    if (measureMatch) {
        const key = measureMatch[1]; // "010"
        if (stepLatency[key]) {
            stepLatency[key].add(res.timings.duration);
            stepRequests[key].add(1);
            if (is5xx) stepErrors[key].add(1);
        }
    }

    // Nessun sleep: il rate e' controllato da constant-arrival-rate.
}

// ── Custom summary: tabella per-step + knee-point + suggerimento services.yaml

export function handleSummary(data) {
    const lines = [];
    const totalSec = STEPS.length * (SETTLE_S + STEP_DURATION_S);
    const totalMin = (totalSec / 60).toFixed(1);

    lines.push('');
    lines.push('='.repeat(80));
    lines.push(`  CAPACITY TEST — ${CLUSTER}`);
    lines.push(`  URL: ${BASE_URL}`);
    lines.push(`  Replicas: ${REPLICAS} | SLA: p95 < ${SLA_P95_MS}ms, err < ${(SLA_ERR_RATE * 100).toFixed(1)}%`);
    lines.push(`  Steps: ${STEPS.length} | Settle ${SETTLE_S}s + measure ${STEP_DURATION_S}s per step | Tot ${totalMin} min`);
    lines.push('='.repeat(80));
    lines.push('');
    lines.push(
        '  Step | rps tgt | rps obt | rps/rep |   p50  |   p95  |   p99  | Err%  | SLA'
    );
    lines.push('  ' + '-'.repeat(76));

    let kneeRpsPerRep = null;
    let kneeRate = null;
    const results = [];
    let consecutiveFails = 0;

    for (let i = 0; i < STEPS.length; i++) {
        const rate = STEPS[i];
        const key = String(rate).padStart(3, '0');
        const lat = data.metrics[`step_${key}rps_latency`];
        const req = data.metrics[`step_${key}rps_requests`];
        const err = data.metrics[`step_${key}rps_errors`];

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
        const rpsPerRep = rpsObt / REPLICAS;

        const slaOk = p95 < SLA_P95_MS && errorRate < SLA_ERR_RATE;
        const slaStr = slaOk ? ' OK ' : 'FAIL';

        results.push({ rate, rpsObt, rpsPerRep, p50, p95, p99, errorRate, slaOk, totalReqs });

        // Knee-point: ultimo step con SLA OK (occorre che il PRECEDENTE fosse OK,
        // per scartare risultati spuri come "tutto verde fino a 10 rps poi rosso a 20").
        if (slaOk) {
            consecutiveFails = 0;
            kneeRpsPerRep = rpsPerRep;
            kneeRate = rate;
        } else {
            consecutiveFails++;
        }

        lines.push(
            `  ${String(i + 1).padStart(4)} | ` +
            `${String(rate).padStart(5)}r/s | ` +
            `${rpsObt.toFixed(1).padStart(7)} | ` +
            `${rpsPerRep.toFixed(1).padStart(7)} | ` +
            `${p50.toFixed(0).padStart(5)}ms | ` +
            `${p95.toFixed(0).padStart(5)}ms | ` +
            `${p99.toFixed(0).padStart(5)}ms | ` +
            `${(errorRate * 100).toFixed(1).padStart(4)}% | ` +
            `${slaStr}`
        );

        // Early stop se SLA fallito per 2 step consecutivi (saturazione confermata)
        if (consecutiveFails >= 2) {
            lines.push('');
            lines.push(`  -> SLA violato per 2 step consecutivi: saturazione confermata`);
            break;
        }
    }

    lines.push('');
    lines.push('='.repeat(80));

    if (kneeRpsPerRep !== null) {
        const recommended = Math.max(1, Math.round(kneeRpsPerRep * 0.80));
        lines.push(`  Knee-point:                ${kneeRpsPerRep.toFixed(1)} rps/replica  (a ${kneeRate} rps target, ${REPLICAS} replica/e)`);
        lines.push(`  Capacita raccomandata:     ${recommended} rps/replica  (80% del knee, margine per spike)`);
        lines.push('');
        lines.push('  -> Aggiorna config/services.yaml:');
        lines.push('     # Frontend (Online Boutique)');
        lines.push(`     capacity_req_per_sec: ${recommended}`);
        lines.push('');
        lines.push('  (opzionale, per per-cluster override:)');
        lines.push('     capacity_per_cluster:');
        lines.push(`       ${CLUSTER}: ${recommended}`);
    } else {
        lines.push('  Knee-point: N/A (SLA gia violato al primo step)');
        lines.push('  Verifica che il cluster sia raggiungibile e il frontend Ready.');
    }

    lines.push('='.repeat(80));
    lines.push('');

    // Costruisci report custom + standard summary
    const customReport = lines.join('\n');
    const stdSummary = textSummaryFallback(data);

    // Output structure: machine-readable JSON + human-readable stdout
    const machineReport = {
        cluster: CLUSTER,
        base_url: BASE_URL,
        replicas: REPLICAS,
        sla: { p95_ms: SLA_P95_MS, error_rate: SLA_ERR_RATE },
        steps: STEPS,
        step_duration_s: STEP_DURATION_S,
        settle_s: SETTLE_S,
        results: results,
        knee: kneeRpsPerRep !== null
            ? { rps_per_replica: kneeRpsPerRep, target_rate: kneeRate,
                recommended_capacity_req_per_sec: Math.round(kneeRpsPerRep * 0.80) }
            : null,
        timestamp_iso: new Date().toISOString(),
    };

    return {
        stdout: customReport + '\n' + stdSummary,
        'experiments/results/capacity_summary.json': JSON.stringify(machineReport, null, 2),
    };
}

function textSummaryFallback(data) {
    const lines = ['  --- Standard k6 Summary ---', ''];
    const interesting = [
        'http_req_duration', 'http_req_failed', 'http_reqs',
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
