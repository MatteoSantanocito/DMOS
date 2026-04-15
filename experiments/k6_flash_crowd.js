import http from 'k6/http';
import { sleep, check } from 'k6';

// 1. SCENARIO DI CARICO 
export const options = {
    scenarios: {
        flash_crowd: {
            executor: 'ramping-arrival-rate',
            startRate: 10,
            timeUnit: '1s', 
            preAllocatedVUs: 150,
            maxVUs: 300,
            stages: [
                { duration: '120s', target: 10 },   // warm-up: DMOS grace period
                { duration: '60s',  target: 100 },   // flash-spike
                { duration: '480s', target: 150 },   // Sustained-ramp
                { duration: '120s', target: 10 },    // decline
                { duration: '60s',  target: 10 },    // cooldown
            ],
        },
    },

    // Percentili da calcolare nel summary finale
    summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],
};

// 2. CONFIGURAZIONE INGRESS E PESI
// Default: distribuzione uniforme 33/33/33
const w1 = __ENV.DMOS_INGRESS_W1 ? parseFloat(__ENV.DMOS_INGRESS_W1) : 0.333;
const w2 = __ENV.DMOS_INGRESS_W2 ? parseFloat(__ENV.DMOS_INGRESS_W2) : 0.333;
const w3 = __ENV.DMOS_INGRESS_W3 ? parseFloat(__ENV.DMOS_INGRESS_W3) : 0.333;

const INGRESSES = [
    { name: "c1-DE", url: "http://192.168.1.245:30080", weight: w1 },
    { name: "c2-FR", url: "http://192.168.1.246:30080", weight: w2 },
    { name: "c3-PL", url: "http://192.168.1.247:30080", weight: w3 },
];

// Log pesi una sola volta (in setup, non per ogni VU)
export function setup() {
    console.log(`[DMOS k6] Ingress weights: C1=${w1}, C2=${w2}, C3=${w3} (sum=${(w1+w2+w3).toFixed(3)})`);
}

// 3. PROFILO ENDPOINT Online Boutique
const CHECKOUT_DATA = {
    email: "test@example.com", street_address: "123 Test St", zip_code: "10001",
    city: "New York", state: "NY", country: "US",
    credit_card_number: "4432801561520454", credit_card_expiration_month: "1",
    credit_card_expiration_year: "2030", credit_card_cvv: "672",
};

const ENDPOINTS = [
    { path: "/",                    weight: 0.40, method: "GET",  body: null },
    { path: "/product/OLJCESPC7Z",  weight: 0.15, method: "GET",  body: null },
    { path: "/product/66VCHSJNUP",  weight: 0.10, method: "GET",  body: null },
    { path: "/cart",                weight: 0.10, method: "GET",  body: null },
    { path: "/cart",                weight: 0.15, method: "POST", body: { product_id: "OLJCESPC7Z", quantity: 1 } },
    { path: "/setCurrency",         weight: 0.05, method: "POST", body: { currency_code: "EUR" } },
    { path: "/cart/checkout",       weight: 0.05, method: "POST", body: CHECKOUT_DATA },
];

// Selezione Pesata
function pickWeighted(items) {
    const totalWeight = items.reduce((acc, item) => acc + item.weight, 0);
    let r = Math.random() * totalWeight;
    for (let item of items) {
        r -= item.weight;
        if (r <= 0) return item;
    }
    return items[items.length - 1];
}

// 4. CICLO PRINCIPALE
// Ogni iterazione sceglie un Ingress pesato indipendentemente.
// Prima: VU fissa a un cluster → se C3 rallenta (netem 300ms), le VU
// assegnate a C3 si bloccano tutte → k6 esaurisce le 300 VU max →
// smette di mandare richieste anche a C1/C2 → cascata artificiale.
// Ora: la distribuzione globale rispetta i pesi W1/W2/W3 ma nessuna
// VU resta "incastrata" su un cluster lento.
export default function () {
    const myIngress = pickWeighted(INGRESSES);
    const ep = pickWeighted(ENDPOINTS);
    const url = myIngress.url + ep.path;

    const params = {

        tags: { ingress: myIngress.name },

        timeout: '10s',
    };

    let res;
    if (ep.method === "POST") {
        res = http.post(url, ep.body, params);
    } else {
        res = http.get(url, params);
    }

    check(res, {
        'status is not 5xx': (r) => r.status < 500,
        'status is not 0 (connection error)': (r) => r.status !== 0,
    });

}