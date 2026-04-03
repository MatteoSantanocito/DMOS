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
            preAllocatedVUs: 800, // Utenti virtuali pronti a sparare
            maxVUs: 4000, // Se il server rallenta, k6 allocherà fino a 800 VUs per mantenere la rate!
            stages: [
                { duration: '240s', target: 10 },  // warm-up esteso: 120s grace DMOS + 120s per scaling iniziale
                { duration: '60s',  target: 150 }, // flash-spike a 150 req/s (ridotto da 200: workload reale genera ~3.5x su productcatalogservice → 525 req/s < 540 max capacity)
                { duration: '600s', target: 150 }, // sustained-peak a 150 req/s
                { duration: '180s', target: 10 },  // decline a 10 req/s
                { duration: '120s', target: 10 },  // cooldown a 10 req/s
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
        // Timeout molto alto per permettere a k6 di misurare i colli di bottiglia 
        // invece di interrompere la connessione (simula la vera latenza p95)
        timeout: '60s',
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
    
    // In arrival-rate, lo sleep finale non incide sul numero di richieste generate
    // dal load generator, ma simula solo il think-time per la singola sessione.
    sleep(Math.random() * 1.0 + 0.5);
}