import http from 'k6/http';
import { sleep, check } from 'k6';

// 1. SCENARIO DI CARICO (Equivalente al tuo flash_crowd_multiingress)
export const options = {
    scenarios: {
        flash_crowd: {
            executor: 'ramping-vus',
            startVUs: 10,
            stages: [
                { duration: '120s', target: 10 },  // warm-up
                { duration: '60s',  target: 150 }, // flash-spike
                { duration: '600s', target: 150 }, // sustained-peak
                { duration: '180s', target: 10 },  // decline
                { duration: '120s', target: 10 },  // cooldown
            ],
        },
    },
    // Diciamo a k6 di calcolare esattamente questi percentili
    summaryTrendStats: ['avg', 'p(50)', 'p(90)', 'p(95)', 'p(99)'],
};

// 2. CONFIGURAZIONE INGRESS E PESI
// Leggiamo le variabili d'ambiente (se presenti), altrimenti usiamo 33/33/33
const w1 = __ENV.DMOS_INGRESS_W1 ? parseFloat(__ENV.DMOS_INGRESS_W1) : 0.333;
const w2 = __ENV.DMOS_INGRESS_W2 ? parseFloat(__ENV.DMOS_INGRESS_W2) : 0.333;
const w3 = __ENV.DMOS_INGRESS_W3 ? parseFloat(__ENV.DMOS_INGRESS_W3) : 0.333;

const INGRESSES = [
    { name: "c1-DE", url: "http://192.168.1.245:30080", weight: w1 },
    { name: "c2-FR", url: "http://192.168.1.246:30080", weight: w2 },
    { name: "c3-PL", url: "http://192.168.1.247:30080", weight: w3 },
];

// 3. DATI E PROBABILITÀ DEGLI ENDPOINT
const CHECKOUT_DATA = JSON.stringify({
    email: "test@example.com", street_address: "123 Test St", zip_code: "10001",
    city: "New York", state: "NY", country: "US",
    credit_card_number: "4432801561520454", credit_card_expiration_month: "1",
    credit_card_expiration_year: "2030", credit_card_cvv: "672",
});

const ENDPOINTS = [
    { path: "/", weight: 0.40, method: "GET", body: null },
    { path: "/product/OLJCESPC7Z", weight: 0.15, method: "GET", body: null },
    { path: "/product/66VCHSJNUP", weight: 0.10, method: "GET", body: null },
    { path: "/cart", weight: 0.10, method: "GET", body: null },
    { path: "/cart", weight: 0.15, method: "POST", body: JSON.stringify({ product_id: "OLJCESPC7Z", quantity: 1 }) },
    { path: "/setCurrency", weight: 0.05, method: "POST", body: JSON.stringify({ currency_code: "EUR" }) },
    { path: "/cart/checkout", weight: 0.05, method: "POST", body: CHECKOUT_DATA },
];

// Funzione Helper per selezione basata sui pesi
function pickWeighted(items) {
    let r = Math.random();
    let cumulative = 0.0;
    // Normalizziamo i pesi al volo se la somma non è 1.0 esatto
    let totalWeight = items.reduce((acc, item) => acc + item.weight, 0);
    r = r * totalWeight;

    for (let item of items) {
        cumulative += item.weight;
        if (r <= cumulative) return item;
    }
    return items[items.length - 1];
}

// 4. INIZIALIZZAZIONE VIRTUAL USER (Il trucco per simulare l'utente geografico)
// In k6, il codice fuori dalla funzione "default" gira UNA SOLA VOLTA per ogni utente.
// Scegliamo a quale cluster assegnare questo utente per tutta la sua sessione.
const myIngress = pickWeighted(INGRESSES);


// 5. CICLO PRINCIPALE DI TEST
export default function () {
    // Scegliamo l'operazione da fare
    let ep = pickWeighted(ENDPOINTS);
    let url = myIngress.url + ep.path;

    let params = {
        headers: { 'Content-Type': 'application/json' },
        // LA MAGIA DI K6: Taggare la richiesta con il nome dell'ingress
        // Questo sdoppierà automaticamente le metriche p95 per ogni cluster!
        tags: { ingress: myIngress.name }, 
    };

    let res;
    if (ep.method === "POST") {
        res = http.post(url, ep.body, params);
    } else {
        res = http.get(url, params);
    }

    // Segnaliamo gli errori 5xx
    check(res, {
        'status is not 5xx': (r) => r.status < 500,
    });

    // Pausa equivalente a between(0.5, 1.5) di Locust
    sleep(Math.random() * (1.5 - 0.5) + 0.5);
}