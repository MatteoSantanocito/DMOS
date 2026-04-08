import http from 'k6/http';
import { sleep, check } from 'k6';

// 1. SCENARIO DI CARICO ESTREMO (Stress Test)
// Questo executor forza k6 a generare un numero fisso di ARRIVI al secondo,
// ignorando quanto tempo il server impiega a rispondere. È il modo corretto 
// per testare il collasso dell'infrastruttura in assenza di DMOS.
export const options = {
    scenarios: {
        flash_crowd: {
            executor: 'ramping-arrival-rate',
            startRate: 10,
            timeUnit: '1s', // La rate si intende "al secondo"
            preAllocatedVUs: 150,
            maxVUs: 300,
            stages: [
                { duration: '120s', target: 10 },   // warm-up: DMOS grace period
                { duration: '60s',  target: 100 },   // flash-spike
                { duration: '480s', target: 150 },   // sustained-peak
                { duration: '120s', target: 10 },    // decline
                { duration: '60s',  target: 10 },    // cooldown
            ],
        },
    },

    // Percentili da calcolare nel summary finale
    summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],
};

// 2. CONFIGURAZIONE INGRESS E PESI
//
// Variabili d'ambiente configurabili per simulare flash crowd geografica:
//   k6 run -e DMOS_INGRESS_W1=0.80 -e DMOS_INGRESS_W2=0.10 -e DMOS_INGRESS_W3=0.10 k6_stress_test.js
//
// Default: distribuzione uniforme 33/33/33
const w1 = __ENV.DMOS_INGRESS_W1 ? parseFloat(__ENV.DMOS_INGRESS_W1) : 0.333;
const w2 = __ENV.DMOS_INGRESS_W2 ? parseFloat(__ENV.DMOS_INGRESS_W2) : 0.333;
const w3 = __ENV.DMOS_INGRESS_W3 ? parseFloat(__ENV.DMOS_INGRESS_W3) : 0.333;

const INGRESSES = [
    { name: "c1-DE", url: "http://192.168.1.245:30080", weight: w1 },
    { name: "c2-FR", url: "http://192.168.1.246:30080", weight: w2 },
    { name: "c3-PL", url: "http://192.168.1.247:30080", weight: w3 },
];

// 3. PROFILO ENDPOINT Online Boutique
// FIX: Online Boutique accetta solo application/x-www-form-urlencoded per i POST.
// Passare oggetti (non JSON.stringify) fa sì che k6 li invii come form-encoded,
// esattamente come fa Locust con requests.post(data=dict). Con JSON → 400 Bad Request.
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

// Helper: selezione pesata
function pickWeighted(items) {
    const totalWeight = items.reduce((acc, item) => acc + item.weight, 0);
    let r = Math.random() * totalWeight;
    for (let item of items) {
        r -= item.weight;
        if (r <= 0) return item;
    }
    return items[items.length - 1];
}

// 4. INIZIALIZZAZIONE PER VIRTUAL USER
// Assegniamo probabilisticamente ogni utente a un Ingress specifico
const myIngress = pickWeighted(INGRESSES);

// 5. CICLO PRINCIPALE
export default function () {
    const ep = pickWeighted(ENDPOINTS);
    const url = myIngress.url + ep.path;

    const params = {
        // Nessun Content-Type fisso: k6 usa automaticamente
        //   application/x-www-form-urlencoded per body oggetto (POST form)
        //   e nessun header aggiuntivo per GET.
        // Tag per disaggregare le metriche per cluster nel summary
        tags: { ingress: myIngress.name },
        // Timeout: 10s è sufficiente per catturare latenze reali senza bloccare
        // VU per minuti su pod non responsivi. Request > 10s sono comunque
        // fuori SLA e vengono contate come errori.
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

    // Con ramping-arrival-rate, lo sleep blocca il VU e riduce il numero di VU
    // disponibili per nuove iterazioni. A 150 iter/s con sleep(1s) servirebbero
    // 150×(0.4+1.0)=210 VU, ma k6 fatica ad allocarli → rate effettivo crolla
    // a ~75 iter/s. Senza sleep, ogni VU si libera subito dopo la request e k6
    // riesce a mantenere il target rate con molti meno VU.
    // Think-time: NON necessario con arrival-rate perché l'executor controlla
    // il rate di arrivi indipendentemente dalla durata dell'iterazione.
}