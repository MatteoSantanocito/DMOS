"""
DMOS Global-Ingress Scenario Load Test — v5
=============================================
Supporta due modalità di ingress:

1. SINGLE INGRESS (default)
   Usa il Cilium Ingress globale su cluster1 (192.168.1.245:30080).
   Cilium Cluster Mesh distribuisce le richieste ai pod in base al numero
   di endpoint attivi (= repliche scalate da DMOS).
   → Scenari: flash_crowd, flash_crowd_netem, gradual_ramp, double_wave, sinusoidal

2. MULTI INGRESS (Romano-like)
   Usa 3 ingress separati (uno per cluster). Locust seleziona l'ingress
   di destinazione in modo probabilistico, simulando utenti geograficamente
   distribuiti — replica esattamente il setup di Romano 2025 (§5.3).
   Con netem attivo: cluster2 risponde con +150ms, cluster3 con +350ms
   anche sul percorso k6→ingress (non solo interno cross-cluster).
   → Scenari: flash_crowd_multiingress, flash_crowd_geo

Pesi ingress configurabili via env vars (DMOS_INGRESS_W1/W2/W3).
Tracking per-ingress automatico per scenari multi-ingress.

Con DMOS ON:  DMOS scala più repliche sul cluster col maggiore ingress rate
              (Φ_demand) e migliore RTT (Φ_net) → Cilium privilegia pod locali.
Con DMOS OFF: repliche fisse → distribuzione proporzionale agli endpoint attivi.

Usage (PowerShell):
  # Single ingress (default)
  $env:DMOS_SCENARIO="flash_crowd"
  locust -f experiments/locustfile_multiingress.py

  # Multi-ingress uniforme (Romano-like, 33/33/33)
  $env:DMOS_SCENARIO="flash_crowd_multiingress"
  locust -f experiments/locustfile_multiingress.py

  # Multi-ingress geo-skewed (flash crowd su DE, 60/25/15)
  $env:DMOS_SCENARIO="flash_crowd_geo"
  $env:DMOS_INGRESS_W1="0.60"; $env:DMOS_INGRESS_W2="0.25"; $env:DMOS_INGRESS_W3="0.15"
  locust -f experiments/locustfile_multiingress.py
"""

import netrc  # pre-import per evitare AssertionError gevent al primo uso di requests
import os
import csv
import math
import time
import random
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import requests as _requests
from gevent.threadpool import ThreadPoolExecutor as _GeventThreadPool
from locust import HttpUser, task, between, LoadTestShape, events

# Thread pool: HTTP in thread OS reali → r.elapsed accurato su Windows
# (immune al gevent select() polling delay che azzerava le latenze)
_request_pool = _GeventThreadPool(max_workers=800)
_thread_local = threading.local()


def _get_session():
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _requests.Session()
    return _thread_local.session


def _blocking_get(url):
    resp = _get_session().get(url, timeout=10, allow_redirects=True)
    return resp.elapsed.total_seconds() * 1000, resp.status_code


def _blocking_post(url, data):
    resp = _get_session().post(url, data=data, timeout=10, allow_redirects=True)
    return resp.elapsed.total_seconds() * 1000, resp.status_code


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO = os.environ.get("DMOS_SCENARIO", "flash_crowd").lower()

# ── Single-ingress (default) ───────────────────────────────────────────────────
# Cilium Ingress globale su cluster1; distribuisce ai pod in base agli endpoint.
GLOBAL_INGRESS = "http://192.168.1.245:30080"

# ── Multi-ingress configuration (Romano-like) ──────────────────────────────────
# Un ingress per cluster. La selezione avviene probabilisticamente per simulare
# utenti distribuiti geograficamente — replica il setup di Romano 2025 §5.3.
#
# Con netem attivo (ms02=150ms, ms03=350ms), un utente che entra da cluster2
# subisce già 150ms di RTT prima che la risposta arrivi; da cluster3, 350ms.
#
# I pesi di default sono definiti PER SCENARIO qui sotto.
# Possono essere sovrascritti a runtime via env vars:
#   DMOS_INGRESS_W1  cluster1 (DE, Frankfurt, 0ms netem)
#   DMOS_INGRESS_W2  cluster2 (FR, Paris,    150ms netem)
#   DMOS_INGRESS_W3  cluster3 (PL, Warsaw,   350ms netem)

# Pesi di default per scenario: (w1_DE, w2_FR, w3_PL)
# Vengono usati se le env vars non sono impostate.
_SCENARIO_DEFAULT_WEIGHTS = {
    "flash_crowd_multiingress": (1/3, 1/3, 1/3),   # Romano-like: distribuzione uniforme
    "flash_crowd_geo":          (0.45, 0.25, 0.30), # Flash crowd su DE, PL pesante (142ms avg penalty)
    # Perché 45/25/30 e non 60/25/15:
    #   - PL al 30% → baseline OFF più pesante (+52ms vs 60/25/15) → miglioramento DMOS più visibile
    #   - DE al 45% → flash crowd riconoscibile ma non banale per Φ_demand
    #   - DMOS deve bilanciare Φ_demand(c1=0.45) vs Φ_net(c3 penalizzato) → scenario più ricco
}
_dw = _SCENARIO_DEFAULT_WEIGHTS.get(SCENARIO, (1/3, 1/3, 1/3))

MULTI_INGRESS_CONFIG = [
    {
        "url":    "http://192.168.1.245:30080",
        "name":   "c1-DE",
        "region": "DE",
        "netem":  0,
        "weight": float(os.environ.get("DMOS_INGRESS_W1", str(_dw[0]))),
    },
    {
        "url":    "http://192.168.1.246:30080",
        "name":   "c2-FR",
        "region": "FR",
        "netem":  150,
        "weight": float(os.environ.get("DMOS_INGRESS_W2", str(_dw[1]))),
    },
    {
        "url":    "http://192.168.1.247:30080",
        "name":   "c3-PL",
        "region": "PL",
        "netem":  350,
        "weight": float(os.environ.get("DMOS_INGRESS_W3", str(_dw[2]))),
    },
]

# Attivo solo per scenari il cui nome contiene "multiingress" o "geo"
IS_MULTI_INGRESS = any(kw in SCENARIO for kw in ("multiingress", "geo"))

# Pesi cumulativi per selezione probabilistica O(1).
# Esempio con flash_crowd_geo (0.60 / 0.25 / 0.15):
#   _ingress_cumulative = [0.60, 0.85, 1.00]
#   r = random() * 1.00
#   r ∈ [0.00, 0.60) → cluster1 (DE)   ← 60% delle volte
#   r ∈ [0.60, 0.85) → cluster2 (FR)   ← 25% delle volte
#   r ∈ [0.85, 1.00) → cluster3 (PL)   ← 15% delle volte
_ingress_cumulative = []
_running = 0.0
for _cfg in MULTI_INGRESS_CONFIG:
    _running += _cfg["weight"]
    _ingress_cumulative.append(_running)
_ingress_total = _running  # somma pesi (normalizzazione automatica se ≠ 1.0)

# Tracking latenza per-ingress (popolato solo in modalità IS_MULTI_INGRESS)
_per_ingress_stats = {
    cfg["url"]: {
        "name": cfg["name"],
        "netem": cfg["netem"],
        "count": 0,
        "failures": 0,
        "slo_violations": 0,
        "response_times": [],
    }
    for cfg in MULTI_INGRESS_CONFIG
}


# _pick_ingress() rimosso: la selezione dell'ingress ora avviene a livello di
# user class (DEUser/FRUser/PLUser), non per singola richiesta. Questo è più
# realistico: un utente parigino resta sempre su cluster2, non salta cluster
# a ogni request. I pesi diventano il `weight` di ogni HttpUser class.

PRODUCTS = [
    "OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O", "L9ECAV7KIM",
    "2ZYFJ3GM2N", "0PUK6V6EV0", "LS4PSXUNUM", "9SIQT8TOJO", "6E92ZMYYFZ",
]
CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CAD"]

# ═══════════════════════════════════════════════════════════════════════════════
# Global Latency Tracking
# ═══════════════════════════════════════════════════════════════════════════════

_stats_lock = threading.Lock()
_global_stats = {"count": 0, "failures": 0, "slo_violations": 0, "response_times": []}
_WINDOW_SIZE = 10
_windowed_stats = defaultdict(lambda: {"count": 0, "failures": 0, "slo_violations": 0, "response_times": []})

# SLO threshold: 1s
SLO_THRESHOLD_MS = 1000

# Safety: ferma il test se il p95 globale supera questa soglia per N check consecutivi
SAFETY_P95_LIMIT_MS = float(os.environ.get("SAFETY_P95_MS", "8000"))
SAFETY_CONSECUTIVE  = int(os.environ.get("SAFETY_CONSECUTIVE", "3"))
SAFETY_CHECK_SEC    = 30


def track_request(response_time_ms, is_failure=False):
    window = int(time.time()) // _WINDOW_SIZE
    with _stats_lock:
        _global_stats["count"] += 1
        _global_stats["response_times"].append(response_time_ms)
        if is_failure:
            _global_stats["failures"] += 1
        if response_time_ms > SLO_THRESHOLD_MS:
            _global_stats["slo_violations"] += 1
        _windowed_stats[window]["count"] += 1
        _windowed_stats[window]["response_times"].append(response_time_ms)
        if is_failure:
            _windowed_stats[window]["failures"] += 1
        if response_time_ms > SLO_THRESHOLD_MS:
            _windowed_stats[window]["slo_violations"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "gradual_ramp": [
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 600, "users_start": 10,  "users_end": 300, "spawn_rate": 5,  "label": "ramp-up"},
        {"duration": 300, "users_start": 300, "users_end": 300, "spawn_rate": 10, "label": "peak"},
        {"duration": 300, "users_start": 300, "users_end": 10,  "spawn_rate": 5,  "label": "ramp-down"},
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "cooldown"},
    ],
    "flash_crowd": [
        # Spike 400 utenti in 60s → plateau 10 min → declino
        # Durata totale: 120+60+600+300+120 = 1200s = 20min
        # Con global ingress: Cilium distribuisce agli endpoint attivi.
        # DMOS ON:  scala repliche su FR → più endpoint FR → più traffico a FR
        # DMOS OFF: 1 replica/cluster → ~33%/33%/33% fisso
        # ⚠️  Usare solo SENZA netem (LAN pura). Con netem usare flash_crowd_netem.
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 60,  "users_start": 10,  "users_end": 400, "spawn_rate": 60, "label": "flash-spike"},
        {"duration": 600, "users_start": 400, "users_end": 400, "spawn_rate": 10, "label": "sustained-peak"},
        {"duration": 300, "users_start": 400, "users_end": 10,  "spawn_rate": 8,  "label": "decline"},
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "cooldown"},
    ],
    "flash_crowd_netem": [
        # Flash crowd con picco 150 utenti — calibrato per test CON netem attivo
        # (ms02=150ms, ms03=350ms). Equivalente allo Scenario 2 di Romano 2025
        # (~150 req/s, 3 minuti di plateau).
        # Durata totale: 120+60+600+180+120 = 1080s ≈ 18min
        # DMOS OFF: distribuzione uniforme 33/33/33 + penalità cross-cluster netem
        # DMOS ON:  concentra repliche su cluster1 (no netem) → riduce penalità
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 60,  "users_start": 10,  "users_end": 150, "spawn_rate": 30, "label": "flash-spike"},
        {"duration": 600, "users_start": 150, "users_end": 150, "spawn_rate": 10, "label": "sustained-peak"},
        {"duration": 180, "users_start": 150, "users_end": 10,  "spawn_rate": 8,  "label": "decline"},
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "cooldown"},
    ],
    "double_wave": [
        {"duration": 120, "users_start": 20,  "users_end": 20,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 180, "users_start": 20,  "users_end": 200, "spawn_rate": 8,  "label": "wave1-ramp"},
        {"duration": 240, "users_start": 200, "users_end": 200, "spawn_rate": 10, "label": "wave1-peak"},
        {"duration": 180, "users_start": 200, "users_end": 50,  "spawn_rate": 5,  "label": "valley-descent"},
        {"duration": 120, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "valley"},
        {"duration": 180, "users_start": 50,  "users_end": 250, "spawn_rate": 8,  "label": "wave2-ramp"},
        {"duration": 240, "users_start": 250, "users_end": 250, "spawn_rate": 10, "label": "wave2-peak"},
        {"duration": 180, "users_start": 250, "users_end": 20,  "spawn_rate": 5,  "label": "final-decline"},
        {"duration": 120, "users_start": 20,  "users_end": 20,  "spawn_rate": 5,  "label": "cooldown"},
    ],
    "sinusoidal": [
        {"duration": 120,  "users_start": 40, "users_end": 40,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 1800, "users_start": 40, "users_end": 200, "spawn_rate": 10, "label": "sinusoidal",
         "type": "sinusoidal", "period": 360, "min_users": 40, "max_users": 200},
        {"duration": 120,  "users_start": 40, "users_end": 40,  "spawn_rate": 5,  "label": "cooldown"},
    ],

    # ── Multi-ingress scenarios (Romano-like, 3 ingress separati) ──────────────
    #
    # PREREQUISITI:
    #   - netem ATTIVO su ms02 (150ms) e ms03 (350ms)
    #   - Online Boutique esposto su NodePort 30080 anche su cluster2 e cluster3
    #     (o tramite Cilium Ingress locale per ogni cluster)
    #
    # In questi scenari Locust invia le richieste probabilisticamente ai 3 ingress,
    # simulando utenti che entrano da cluster geograficamente diversi. Questo
    # replica fedelmente il setup di Romano 2025 §5.3 (k6 + lista ingress).
    #
    # Il tracking per-ingress (latenza, fail%, SLO) viene riportato nel summary
    # finale e nel CSV, permettendo di vedere l'effetto di DMOS cluster per cluster.

    "flash_crowd_multiingress": [
        # Romano-like flash crowd con 3 ingress — distribuzione UNIFORME 33/33/33.
        # Uso: confronto diretto con Romano 2025 Scenario 2 (~150 req/s, netem attivo).
        # Peak: 150 utenti (come Romano). Durata totale: 1080s ≈ 18min.
        #
        # Con DMOS OFF: ogni cluster riceve ~33% del traffico. Gli utenti che
        #   entrano da c2/c3 subiscono 150/350ms solo per ricevere la risposta.
        # Con DMOS ON: DMOS concentra repliche su c1 (Φ_demand alto, Φ_net basso) →
        #   Cilium serve localmente chi entra da c1, riducendo il p95 globale.
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 60,  "users_start": 10,  "users_end": 150, "spawn_rate": 30, "label": "flash-spike"},
        {"duration": 600, "users_start": 150, "users_end": 150, "spawn_rate": 10, "label": "sustained-peak"},
        {"duration": 180, "users_start": 150, "users_end": 10,  "spawn_rate": 8,  "label": "decline"},
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "cooldown"},
    ],

    "flash_crowd_geo": [
        # Geo-aware flash crowd: distribuzione ASIMMETRICA (default 45/25/30).
        # DE riceve il 45% (flash crowd), PL il 30% (penalità alta 350ms).
        #
        # Perché 45/25/30 e non 60/25/15:
        #   - Penalità netem media entrata: 142ms vs 90ms → baseline OFF più pesante
        #   - PL al 30% amplifica il danno del routing cross-cluster in OFF
        #   - DE al 45% è ancora un flash crowd riconoscibile per Φ_demand
        #   - DMOS deve bilanciare Φ_demand(c1=0.45) vs ridurre routing verso PL
        #     → scenario più ricco di quello banale con c1=0.60 dominante
        #
        # Pesi sovrascrivibili via env vars se necessario:
        #   $env:DMOS_INGRESS_W1="0.45"   # già default
        #   $env:DMOS_INGRESS_W2="0.25"   # già default
        #   $env:DMOS_INGRESS_W3="0.30"   # già default
        #   $env:DMOS_SCENARIO="flash_crowd_geo"
        # Peak: 150 utenti. Durata totale: 1080s ≈ 18min.
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "warm-up"},
        {"duration": 60,  "users_start": 10,  "users_end": 150, "spawn_rate": 30, "label": "flash-spike"},
        {"duration": 600, "users_start": 150, "users_end": 150, "spawn_rate": 10, "label": "sustained-peak"},
        {"duration": 180, "users_start": 150, "users_end": 10,  "spawn_rate": 8,  "label": "decline"},
        {"duration": 120, "users_start": 10,  "users_end": 10,  "spawn_rate": 5,  "label": "cooldown"},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Traffic Shape
# ═══════════════════════════════════════════════════════════════════════════════

class DMOSScenarioShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        self.phases = SCENARIOS.get(SCENARIO, SCENARIOS["flash_crowd"])
        self._phase_offsets = []
        offset = 0
        for p in self.phases:
            self._phase_offsets.append(offset)
            offset += p["duration"]
        self._total_duration = offset
        print(f"\n{'='*70}")
        print(f"  DMOS Global-Ingress Load Test v5")
        print(f"  Scenario:      {SCENARIO} | Duration: {self._total_duration/60:.0f} min")
        if IS_MULTI_INGRESS:
            print(f"  Ingress mode:  MULTI-INGRESS (3 user classes con host fisso)")
            total_w = sum(c["weight"] for c in MULTI_INGRESS_CONFIG)
            classes = ["DEUser", "FRUser", "PLUser"]
            for cfg, cls in zip(MULTI_INGRESS_CONFIG, classes):
                pct = cfg["weight"] / total_w * 100
                w   = max(1, round(cfg["weight"] / total_w * 100))
                netem_str = f"+{cfg['netem']}ms netem" if cfg["netem"] > 0 else "0ms (no netem)"
                print(f"    {cls:8s} ({cfg['name']}): {cfg['url']}  {pct:4.1f}%  weight={w}  [{netem_str}]")
        else:
            print(f"  Ingress mode:  SINGLE (Cilium global ingress, cluster1)")
            print(f"  Ingress URL:   {GLOBAL_INGRESS}")
        print(f"  Traffic split:  controlled by Cilium Cluster Mesh (proportional to replicas)")
        print(f"{'='*70}\n")

    def tick(self):
        run_time = self.get_run_time()
        if run_time >= self._total_duration:
            return None
        for i, phase in enumerate(self.phases):
            phase_start = self._phase_offsets[i]
            phase_end   = phase_start + phase["duration"]
            if run_time < phase_end:
                t_in_phase = run_time - phase_start
                if t_in_phase < 1:
                    print(f"\n  Phase: {phase['label']} | "
                          f"Users: {phase['users_start']} -> {phase['users_end']} | "
                          f"Duration: {phase['duration']}s")
                if phase.get("type") == "sinusoidal":
                    period = phase["period"]
                    amp    = (phase["max_users"] - phase["min_users"]) / 2
                    mid    = (phase["max_users"] + phase["min_users"]) / 2
                    return (int(mid + amp * math.sin(2 * math.pi * t_in_phase / period)),
                            phase["spawn_rate"])
                progress = t_in_phase / phase["duration"]
                current  = int(phase["users_start"] +
                               (phase["users_end"] - phase["users_start"]) * progress)
                return (current, phase["spawn_rate"])
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints & Request Logic
# ═══════════════════════════════════════════════════════════════════════════════

_CHECKOUT_DATA = {
    "email": "test@example.com",
    "street_address": "123 Test St",
    "zip_code": "10001",
    "city": "New York",
    "state": "NY",
    "country": "US",
    "credit_card_number": "4432801561520454",
    "credit_card_expiration_month": "1",
    "credit_card_expiration_year": "2030",
    "credit_card_cvv": "672",
}

# (path, weight, method, data)
_ENDPOINTS = [
    ("/",                   0.40, "GET",  None),
    ("/product/OLJCESPC7Z", 0.15, "GET",  None),
    ("/product/66VCHSJNUP", 0.10, "GET",  None),
    ("/cart",               0.10, "GET",  None),
    ("/cart",               0.15, "POST", {"product_id": "OLJCESPC7Z", "quantity": 1}),
    ("/setCurrency",        0.05, "POST", {"currency_code": "EUR"}),
    ("/cart/checkout",      0.05, "POST", _CHECKOUT_DATA),
]


def _weighted_choice():
    r = random.random()
    cumulative = 0.0
    for path, weight, method, data in _ENDPOINTS:
        cumulative += weight
        if r <= cumulative:
            return path, method, data
    path, _, method, data = _ENDPOINTS[-1]
    return path, method, data


def _do_request(base_url=None):
    """Esegue una richiesta HTTP verso base_url (o GLOBAL_INGRESS se None).
    Chiamato da ogni user class con il proprio host fisso."""
    if base_url is None:
        base_url = GLOBAL_INGRESS
    path, method, data = _weighted_choice()
    url = base_url + path

    try:
        if method == "POST":
            elapsed_ms, status = _request_pool.submit(_blocking_post, url, data).result()
        else:
            elapsed_ms, status = _request_pool.submit(_blocking_get, url).result()
    except Exception:
        elapsed_ms, status = 10000.0, 500

    is_failure = status >= 500
    track_request(elapsed_ms, is_failure)

    # Tracking per-ingress (solo modalità multi-ingress)
    if IS_MULTI_INGRESS and base_url in _per_ingress_stats:
        with _stats_lock:
            s = _per_ingress_stats[base_url]
            s["count"] += 1
            s["response_times"].append(elapsed_ms)
            if is_failure:
                s["failures"] += 1
            if elapsed_ms > SLO_THRESHOLD_MS:
                s["slo_violations"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
# User Classes
# ═══════════════════════════════════════════════════════════════════════════════
#
# SINGLE INGRESS: GlobalUser — tutti gli utenti verso cluster1.
#
# MULTI INGRESS:  3 classi separate (DEUser / FRUser / PLUser).
#   Ogni classe ha host fisso → un utente "parigino" resta sempre su cluster2,
#   non salta cluster a ogni request (sessione geograficamente vincolata).
#   Il `weight` di ogni classe determina la proporzione di utenti allocati
#   da Locust, riflettendo i pesi di MULTI_INGRESS_CONFIG.
#
#   Locust UI mostra le 3 classi separate con stats indipendenti.

if IS_MULTI_INGRESS:
    _w_total = sum(c["weight"] for c in MULTI_INGRESS_CONFIG)

    class DEUser(HttpUser):
        """Utenti geograficamente vincolati a cluster1 (DE/Frankfurt, 0ms netem)."""
        host      = MULTI_INGRESS_CONFIG[0]["url"]
        wait_time = between(0.5, 1.5)
        weight    = max(1, round(MULTI_INGRESS_CONFIG[0]["weight"] / _w_total * 100))

        @task
        def browse(self):
            _do_request(self.host)

    class FRUser(HttpUser):
        """Utenti geograficamente vincolati a cluster2 (FR/Paris, +150ms netem)."""
        host      = MULTI_INGRESS_CONFIG[1]["url"]
        wait_time = between(0.5, 1.5)
        weight    = max(1, round(MULTI_INGRESS_CONFIG[1]["weight"] / _w_total * 100))

        @task
        def browse(self):
            _do_request(self.host)

    class PLUser(HttpUser):
        """Utenti geograficamente vincolati a cluster3 (PL/Warsaw, +350ms netem)."""
        host      = MULTI_INGRESS_CONFIG[2]["url"]
        wait_time = between(0.5, 1.5)
        weight    = max(1, round(MULTI_INGRESS_CONFIG[2]["weight"] / _w_total * 100))

        @task
        def browse(self):
            _do_request(self.host)

else:
    class GlobalUser(HttpUser):
        """Utente singolo per scenari single-ingress (tutti su cluster1)."""
        host      = GLOBAL_INGRESS
        wait_time = between(0.5, 1.5)

        @task
        def browse(self):
            _do_request()


# ═══════════════════════════════════════════════════════════════════════════════
# Safety Watchdog
# ═══════════════════════════════════════════════════════════════════════════════

def _safety_watchdog(environment):
    consecutive = 0
    while not environment.runner.state == "stopped":
        time.sleep(SAFETY_CHECK_SEC)
        with _stats_lock:
            all_rts = list(_global_stats["response_times"])
        if len(all_rts) < 50:
            continue
        p95 = sorted(all_rts)[int(len(all_rts) * 0.95)]
        if p95 > SAFETY_P95_LIMIT_MS:
            consecutive += 1
            print(f"\n  ⚠️  SAFETY: p95 = {p95:.0f}ms > {SAFETY_P95_LIMIT_MS:.0f}ms "
                  f"({consecutive}/{SAFETY_CONSECUTIVE})")
            if consecutive >= SAFETY_CONSECUTIVE:
                print(f"\n  🛑 SAFETY STOP: p95 troppo alto per {SAFETY_CONSECUTIVE} check — stop\n")
                environment.runner.quit()
                return
        else:
            if consecutive > 0:
                print(f"  ✅ SAFETY: p95 tornato a {p95:.0f}ms — reset contatore")
            consecutive = 0


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print(f"\n  Test started: {SCENARIO} | Global tracking: ACTIVE")
    if IS_MULTI_INGRESS:
        total_w = sum(c["weight"] for c in MULTI_INGRESS_CONFIG)
        classes = ["DEUser", "FRUser", "PLUser"]
        weights_str = " | ".join(
            f"{cls}({c['name']})={c['weight']/total_w*100:.1f}%"
            for c, cls in zip(MULTI_INGRESS_CONFIG, classes)
        )
        print(f"  User classes:    {weights_str}")
    print(f"  Per-cluster distribution: via Prometheus in collect_metrics_simple.py\n")
    t = threading.Thread(target=_safety_watchdog, args=(environment,), daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    output_dir = Path(__file__).parent.parent / "results" / "multiingress"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    stats = _global_stats
    rts   = sorted(stats["response_times"]) if stats["response_times"] else []
    n     = len(rts)

    print(f"\n{'='*70}")
    print(f"  GLOBAL LATENCY — {SCENARIO}")
    print(f"{'='*70}")

    if n > 0:
        avg      = sum(rts) / n
        p50      = rts[int(n * 0.50)]
        p90      = rts[int(n * 0.90)]
        p95      = rts[int(n * 0.95)]
        p99      = rts[min(int(n * 0.99), n - 1)]
        fail_pct = stats["failures"]      / n * 100
        slo_pct  = stats["slo_violations"] / n * 100

        print(f"\n  Requests : {n}")
        print(f"  Fail%    : {fail_pct:.1f}%")
        print(f"  Avg      : {avg:.0f}ms")
        print(f"  p50      : {p50:.0f}ms")
        print(f"  p90      : {p90:.0f}ms")
        print(f"  p95      : {p95:.0f}ms")
        print(f"  p99      : {p99:.0f}ms")
        print(f"  SLO>1000ms: {slo_pct:.1f}%")
        print(f"\n  Note: per-cluster traffic split in collect_metrics_simple.py JSONL")
        print(f"        (resources._traffic_pct.frontend per ogni cluster)")
    else:
        print("  No data collected.")

    # ── Per-ingress breakdown (solo scenari multi-ingress) ───────────────────
    if IS_MULTI_INGRESS:
        print(f"\n  PER-INGRESS LATENCY BREAKDOWN")
        print(f"  {'─'*66}")
        print(f"  {'Ingress':<10} {'netem':>7}  {'n':>7}  {'avg':>7}  {'p95':>7}  {'fail%':>6}  {'SLO%':>6}")
        print(f"  {'─'*66}")
        for url, s in _per_ingress_stats.items():
            rts_i = sorted(s["response_times"]) if s["response_times"] else []
            ni = len(rts_i)
            if ni > 0:
                avg_i  = sum(rts_i) / ni
                p95_i  = rts_i[int(ni * 0.95)]
                fail_i = s["failures"]       / ni * 100
                slo_i  = s["slo_violations"] / ni * 100
                print(
                    f"  {s['name']:<10} {s['netem']:>5}ms  {ni:>7}  "
                    f"{avg_i:>6.0f}ms  {p95_i:>6.0f}ms  {fail_i:>5.1f}%  {slo_i:>5.1f}%"
                )
            else:
                print(f"  {s['name']:<10} {s['netem']:>5}ms  {'(no data)':>7}")
        print(f"  {'─'*66}")

    print(f"\n{'='*70}\n")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    summary_file = output_dir / f"{SCENARIO}_summary_{timestamp}.csv"
    if n > 0:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scenario", "requests", "failures", "fail_pct",
                "avg_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "slo_pct",
            ])
            writer.writeheader()
            writer.writerow({
                "scenario":  SCENARIO,
                "requests":  n,
                "failures":  stats["failures"],
                "fail_pct":  round(fail_pct, 2),
                "avg_ms":    round(avg, 1),
                "p50_ms":    round(p50, 1),
                "p90_ms":    round(p90, 1),
                "p95_ms":    round(p95, 1),
                "p99_ms":    round(p99, 1),
                "slo_pct":   round(slo_pct, 2),
            })
        print(f"  Summary CSV:     {summary_file}")

    # ── Timeseries CSV ───────────────────────────────────────────────────────
    # Colonne: timestamp, count, avg_ms, p95_ms, slo_pct, fail_pct
    # Per-cluster traffic %: nel JSONL di collect_metrics_simple.py
    ts_file = output_dir / f"{SCENARIO}_timeseries_{timestamp}.csv"
    with open(ts_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "count", "avg_ms", "p95_ms", "slo_pct", "fail_pct"])
        for window_ts in sorted(_windowed_stats.keys()):
            ws = _windowed_stats[window_ts]
            if ws["response_times"]:
                wrts  = sorted(ws["response_times"])
                nw    = len(wrts)
                p95_w      = round(wrts[int(nw * 0.95)], 1) if nw >= 2 else round(wrts[-1], 1)
                slo_pct_w  = round(ws["slo_violations"] / nw * 100, 1)
                fail_pct_w = round(ws["failures"]       / nw * 100, 1)
                writer.writerow([
                    datetime.fromtimestamp(window_ts * _WINDOW_SIZE).strftime("%H:%M:%S"),
                    nw,
                    round(sum(wrts) / nw, 1),
                    p95_w,
                    slo_pct_w,
                    fail_pct_w,
                ])
    print(f"  Timeseries CSV:  {ts_file}")

    # ── Per-ingress CSV (solo scenari multi-ingress) ──────────────────────────
    if IS_MULTI_INGRESS:
        ingress_file = output_dir / f"{SCENARIO}_per_ingress_{timestamp}.csv"
        with open(ingress_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ingress_name", "region", "netem_ms", "weight_pct",
                "requests", "failures", "fail_pct",
                "avg_ms", "p95_ms", "slo_pct",
            ])
            writer.writeheader()
            total_w = sum(c["weight"] for c in MULTI_INGRESS_CONFIG)
            for cfg in MULTI_INGRESS_CONFIG:
                url = cfg["url"]
                s   = _per_ingress_stats[url]
                rts_i = sorted(s["response_times"]) if s["response_times"] else []
                ni = len(rts_i)
                if ni > 0:
                    avg_i  = round(sum(rts_i) / ni, 1)
                    p95_i  = round(rts_i[int(ni * 0.95)], 1)
                    fail_i = round(s["failures"]       / ni * 100, 2)
                    slo_i  = round(s["slo_violations"] / ni * 100, 2)
                else:
                    avg_i = p95_i = fail_i = slo_i = 0.0
                writer.writerow({
                    "ingress_name": cfg["name"],
                    "region":       cfg["region"],
                    "netem_ms":     cfg["netem"],
                    "weight_pct":   round(cfg["weight"] / total_w * 100, 1),
                    "requests":     ni,
                    "failures":     s["failures"],
                    "fail_pct":     fail_i,
                    "avg_ms":       avg_i,
                    "p95_ms":       p95_i,
                    "slo_pct":      slo_i,
                })
        print(f"  Per-ingress CSV: {ingress_file}")

    print(f"\n  Done! Per-cluster split: usa collect_metrics_simple.py JSONL\n")
