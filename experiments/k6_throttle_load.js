import http from 'k6/http';
import { check } from 'k6';

// ============================================================================
// k6 Throttle-Check Load Generator
// ============================================================================
//
// Load generator MINIMALE usato da experiments/throttle_check.sh per verificare
// che, scalando il frontend a max_replicas su un cluster, le repliche NON
// vadano in CFS throttling (cioe' che il modello "capacita = 60 x max_replicas"
// sia lineare ed esatto).
//
// NON e' un capacity test: gira a un rate FISSO (constant-arrival-rate) per una
// durata fissa. Le metriche di throttling si leggono da Prometheus, non da k6.
//
// Variabili env (impostate da throttle_check.sh):
//   DMOS_CLUSTER   : nome cluster target (default ms-cluster-01)
//   DMOS_RATE      : rps costanti da generare (default 100)
//   DMOS_DURATION  : durata del carico, es. "180s" (default "180s")
//
// Uso standalone (di solito lo lancia throttle_check.sh):
//   k6 run -e DMOS_CLUSTER=ms-cluster-01 -e DMOS_RATE=384 -e DMOS_DURATION=180s \
//       experiments/k6_throttle_load.js
// ============================================================================

const CLUSTER  = __ENV.DMOS_CLUSTER  || 'ms-cluster-01';
const RATE     = parseInt(__ENV.DMOS_RATE || '100');
const DURATION = __ENV.DMOS_DURATION || '180s';

// k6 open() risolve i path relativi a QUESTO .js file (esperiments/), quindi
// saliamo di una dir per raggiungere .cache/ nella repo root.
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

export const options = {
    scenarios: {
        throttle_load: {
            executor: 'constant-arrival-rate',
            rate: RATE,
            timeUnit: '1s',
            duration: DURATION,
            // Pool VU: dimensionato per il rate con margine ampio (latency
            // intra-cluster ~100ms -> ~RATE/10 VU concorrenti, margine 3-6x).
            preAllocatedVUs: Math.max(50, RATE * 3),
            maxVUs: Math.max(100, RATE * 6),
        },
    },
    summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)'],
};

// Endpoint mix leggero (GET-heavy, niente checkout pesante: il throttle-check
// vuole stressare la CPU del frontend, non testare la chain completa).
const ENDPOINTS = [
    { path: '/',                   weight: 0.55 },
    { path: '/product/OLJCESPC7Z', weight: 0.25 },
    { path: '/cart',               weight: 0.20 },
];

function pickEndpoint() {
    let r = Math.random();
    for (const ep of ENDPOINTS) {
        r -= ep.weight;
        if (r <= 0) return ep;
    }
    return ENDPOINTS[0];
}

export default function () {
    const ep = pickEndpoint();
    const res = http.get(BASE_URL + ep.path, { timeout: '15s' });
    check(res, {
        'status is 200': (r) => r.status === 200,
        'status is not 5xx': (r) => r.status < 500,
    });
}
