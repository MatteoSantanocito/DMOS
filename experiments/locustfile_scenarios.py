"""
DMOS Multi-Scenario Traffic Load Test
======================================
File unico con scenari multipli per la validazione sperimentale di DMOS.

Scenari disponibili:
  1. gradual_ramp    — Rampa graduale con doppia onda (scenario originale, migliorato)
  2. flash_crowd     — Spike improvviso (flash crowd / evento virale)
  3. double_wave     — Doppia onda con valle intermedia
  4. sinusoidal      — Traffico oscillante continuo (stress test anti-flapping)
  5. slow_ramp       — Rampa lentissima con plateau lungo (efficienza stazionaria)
  6. sawtooth        — Dente di sega ripetuto (pattern ciclico)

═══════════════════════════════════════════════════════════════════════════════
USO:

  Scegli lo scenario impostando la variabile SCENARIO qui sotto,
  oppure usa la variabile d'ambiente DMOS_SCENARIO:

    # Opzione 1: modifica SCENARIO nel file
    SCENARIO = "flash_crowd"

    # Opzione 2: variabile d'ambiente
    export DMOS_SCENARIO=flash_crowd

  Poi lancia:
    locust -f locustfile_scenarios.py --host http://192.168.1.245:30007

  Il test si ferma automaticamente alla fine dello scenario.
  --users e --spawn-rate sono ignorati: è il LoadTestShape che controlla tutto.

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import math
import random
import datetime
from locust import FastHttpUser, task, between, LoadTestShape


# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    SCEGLI LO SCENARIO QUI                              │
# └─────────────────────────────────────────────────────────────────────────┘
SCENARIO = os.environ.get("DMOS_SCENARIO", "gradual_ramp")

# Product IDs from Online Boutique catalog
PRODUCTS = [
    '0PUK6V6EV0', '1YMWWN1N4O', '2ZYFJ3GM2N', '66VCHSJNUP',
    '6E92ZMYYFZ', '9SIQT8TOJO', 'L9ECAV7KIM', 'LS4PSXUNUM', 'OLJCESPC7Z'
]
CURRENCIES = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY']


# ═══════════════════════════════════════════════════════════════════════════════
# User Definition (basato sul locustfile ufficiale Google Online Boutique)
# ═══════════════════════════════════════════════════════════════════════════════

class FrontendUser(FastHttpUser):
    """Simula un utente che naviga Online Boutique (basato sul locustfile ufficiale)."""
    wait_time = between(1, 5)

    @task(10)
    def browse_product(self):
        self.client.get("/product/" + random.choice(PRODUCTS))

    @task(5)
    def browse_homepage(self):
        self.client.get("/")

    @task(3)
    def view_cart(self):
        self.client.get("/cart")

    @task(2)
    def set_currency(self):
        self.client.post("/setCurrency", {
            'currency_code': random.choice(CURRENCIES)
        })

    @task(2)
    def add_to_cart(self):
        product = random.choice(PRODUCTS)
        self.client.get("/product/" + product)
        self.client.post("/cart", {
            'product_id': product,
            'quantity': random.randint(1, 5)
        })

    @task(1)
    def checkout(self):
        # Prima aggiungi un prodotto al carrello (come fa il locustfile ufficiale)
        product = random.choice(PRODUCTS)
        self.client.get("/product/" + product)
        self.client.post("/cart", {
            'product_id': product,
            'quantity': 1
        })
        # Poi checkout
        current_year = datetime.datetime.now().year + 1
        self.client.post("/cart/checkout", {
            'email': f'test{random.randint(1,9999)}@example.com',
            'street_address': '123 Test Street',
            'zip_code': '10001',
            'city': 'New York',
            'state': 'NY',
            'country': 'US',
            'credit_card_number': '4432-8015-6152-0454',
            'credit_card_expiration_month': str(random.randint(1, 12)),
            'credit_card_expiration_year': str(random.randint(current_year, current_year + 5)),
            'credit_card_cvv': str(random.randint(100, 999)),
        })


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: interpolazione lineare tra fasi
# ═══════════════════════════════════════════════════════════════════════════════

def compute_users_from_phases(phases, run_time):
    """
    Data una lista di fasi e il tempo corrente, calcola (users, spawn_rate).
    Ogni fase: {"duration": sec, "users_start": N, "users_end": N, "spawn_rate": N, "label": str}
    Interpola linearmente tra users_start e users_end durante la fase.
    """
    elapsed = 0
    for phase in phases:
        phase_end = elapsed + phase["duration"]
        if run_time < phase_end:
            # Siamo in questa fase
            progress = (run_time - elapsed) / phase["duration"]
            current_users = int(
                phase["users_start"] + (phase["users_end"] - phase["users_start"]) * progress
            )
            return max(1, current_users), phase["spawn_rate"]
        elapsed = phase_end
    # Test finito
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Rampa Graduale con Doppia Onda (originale migliorato)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  300 ┤                                    ╭──────╮
#  250 ┤              ╭──────╮              │      │
#  200 ┤            ╱─╯      ╰─╮            │      │
#  150 ┤          ╱─╯           ╰─╮       ╱─╯      ╰─╮
#  100 ┤        ╱─╯               ╰─╮  ╱──╯           ╰─╮
#   50 ┤──────╱─╯                    ╰─╯                  ╰──────
#    0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────
#      0    3    6    9   12   15   18   21   24   27   30  min
#
# Durata: ~28 minuti
# Obiettivo: test completo proattività, stabilità, efficienza
#

PHASES_GRADUAL_RAMP = [
    {"duration": 180, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "warm-up"},
    {"duration": 180, "users_start": 50,  "users_end": 250, "spawn_rate": 5,  "label": "ramp-up"},
    {"duration": 240, "users_start": 250, "users_end": 250, "spawn_rate": 10, "label": "peak-1"},
    {"duration": 180, "users_start": 250, "users_end": 80,  "spawn_rate": 5,  "label": "ramp-down"},
    {"duration": 120, "users_start": 80,  "users_end": 80,  "spawn_rate": 10, "label": "valley"},
    {"duration": 120, "users_start": 80,  "users_end": 300, "spawn_rate": 20, "label": "spike"},
    {"duration": 180, "users_start": 300, "users_end": 300, "spawn_rate": 10, "label": "peak-2"},
    {"duration": 180, "users_start": 300, "users_end": 50,  "spawn_rate": 5,  "label": "cooldown"},
    {"duration": 120, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "baseline"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: Flash Crowd (Spike Improvviso)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  300 ┤              ┌──────────────────────┐
#  250 ┤              │                      │
#  200 ┤              │                      │
#  150 ┤              │                      │
#  100 ┤              │                      │
#   50 ┤──────────────┘                      └─────────────
#    0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────
#      0    2    4    6    8   10   12   14   16   18  min
#
# Durata: ~18 minuti
# Obiettivo: worst-case per il predittore — salto istantaneo da 50 a 300.
#   DMOS non può anticipare lo spike, quindi il TtS sarà positivo (reattivo).
#   Dimostra che il sistema gestisce il caso peggiore senza flapping e
#   converge al provisioning corretto in 1-2 cicli.
#

PHASES_FLASH_CROWD = [
    {"duration": 240, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "baseline-stabile"},
    {"duration": 30,  "users_start": 50,  "users_end": 300, "spawn_rate": 50, "label": "flash-spike"},
    {"duration": 360, "users_start": 300, "users_end": 300, "spawn_rate": 10, "label": "sustained-peak"},
    {"duration": 30,  "users_start": 300, "users_end": 50,  "spawn_rate": 50, "label": "crash-down"},
    {"duration": 240, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "post-crash-baseline"},
    {"duration": 120, "users_start": 50,  "users_end": 0,   "spawn_rate": 10, "label": "drain"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3: Doppia Onda con Valle Profonda
# ═══════════════════════════════════════════════════════════════════════════════
#
#  300 ┤                                          ╭──────╮
#  250 ┤         ╭──────╮                       ╱─╯      ╰─╮
#  200 ┤       ╱─╯      ╰─╮                  ╱──╯           ╰─╮
#  150 ┤     ╱─╯           ╰─╮            ╱──╯                 ╰─╮
#  100 ┤   ╱─╯               ╰─╮       ╱──╯                      ╰─╮
#   50 ┤──╱                      ╰─────╯                             ╰───
#    0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────
#      0    3    6    9   12   15   18   21   24   27   30   33  min
#
# Durata: ~32 minuti
# Obiettivo: la seconda onda è più alta della prima. Se il predittore
#   "ricorda" il pattern della prima onda, il TtS sulla seconda dovrebbe
#   essere migliore. Confronta proactive % tra le due onde.
#

PHASES_DOUBLE_WAVE = [
    # Prima onda: picco a 250
    {"duration": 120, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "warm-up"},
    {"duration": 180, "users_start": 50,  "users_end": 250, "spawn_rate": 5,  "label": "onda1-salita"},
    {"duration": 180, "users_start": 250, "users_end": 250, "spawn_rate": 10, "label": "onda1-picco"},
    {"duration": 180, "users_start": 250, "users_end": 50,  "spawn_rate": 5,  "label": "onda1-discesa"},
    # Valle profonda
    {"duration": 180, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "valle"},
    # Seconda onda: picco a 300 (più alta)
    {"duration": 180, "users_start": 50,  "users_end": 300, "spawn_rate": 5,  "label": "onda2-salita"},
    {"duration": 240, "users_start": 300, "users_end": 300, "spawn_rate": 10, "label": "onda2-picco"},
    {"duration": 180, "users_start": 300, "users_end": 50,  "spawn_rate": 5,  "label": "onda2-discesa"},
    # Coda
    {"duration": 120, "users_start": 50,  "users_end": 0,   "spawn_rate": 10, "label": "drain"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4: Traffico Sinusoidale (Oscillante)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  250 ┤    ╭─╮       ╭─╮       ╭─╮       ╭─╮       ╭─╮
#  200 ┤  ╱─╯ ╰─╮  ╱──╯ ╰─╮  ╱──╯ ╰─╮  ╱──╯ ╰─╮  ╱──╯ ╰─╮
#  150 ┤╱─╯     ╰─╱╯       ╰─╱╯       ╰─╱╯       ╰─╱╯       ╰
#  100 ┤                                                        
#   50 ┤
#    0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────
#      0    3    6    9   12   15   18   21   24   27   30  min
#
# Durata: ~25 minuti
# Obiettivo: stress test anti-flapping. Il traffico oscilla tra 80 e 250
#   con periodo di ~5 minuti. Un autoscaler instabile scalerebbe ad ogni
#   ciclo. DMOS con il PD controller dovrebbe trovare un livello di
#   repliche stabile senza inseguire ogni oscillazione.
#   Metrica chiave: flapping_windows deve rimanere 0.
#

# Per la sinusoide usiamo un approccio diverso: tick() diretto con math.sin
SINUSOIDAL_CONFIG = {
    "duration_minutes": 25,
    "min_users": 80,
    "max_users": 250,
    "period_minutes": 5,     # Un ciclo completo ogni 5 minuti
    "spawn_rate": 10,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5: Rampa Lenta con Plateau Lungo
# ═══════════════════════════════════════════════════════════════════════════════
#
#  200 ┤                    ╭────────────────────────────╮
#  150 ┤               ╱────╯                            ╰────╮
#  100 ┤          ╱────╯                                       ╰────╮
#   50 ┤─────────╯                                                   ╰───
#    0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────
#      0    3    6    9   12   15   18   21   24   27   30   33  min
#
# Durata: ~33 minuti
# Obiettivo: testare l'efficienza stazionaria. Con una rampa di 10 minuti
#   e un plateau di 15 minuti, il sistema ha tutto il tempo di convergere.
#   Il provisioning ratio a regime dovrebbe essere molto vicino a 1.0-1.15.
#   Anche il MAPE dovrebbe essere basso perché il traffico è prevedibile.
#

PHASES_SLOW_RAMP = [
    {"duration": 180, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "baseline"},
    {"duration": 600, "users_start": 50,  "users_end": 200, "spawn_rate": 2,  "label": "rampa-lenta"},
    {"duration": 900, "users_start": 200, "users_end": 200, "spawn_rate": 10, "label": "plateau-lungo"},
    {"duration": 300, "users_start": 200, "users_end": 50,  "spawn_rate": 3,  "label": "discesa-graduale"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 6: Dente di Sega (Sawtooth)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  250 ┤    ╱│     ╱│     ╱│     ╱│     ╱│
#  200 ┤  ╱─ │   ╱─ │   ╱─ │   ╱─ │   ╱─ │
#  150 ┤╱──  │ ╱──  │ ╱──  │ ╱──  │ ╱──  │
#  100 ┤     │╱     │╱     │╱     │╱     │╱
#   50 ┤     ╱      ╱      ╱      ╱      ╱
#    0 ┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────
#      0    3    6    9   12   15   18   21   24   27  min
#
# Durata: ~25 minuti
# Obiettivo: pattern ciclico con crollo rapido e risalita graduale.
#   Testa la capacità di DMOS di non sovra-reagire ai crolli improvvisi
#   (il cooldown asimmetrico dovrebbe trattenere le repliche) e di
#   anticipare la risalita successiva (pattern ripetitivo = prevedibile).
#

PHASES_SAWTOOTH = [
    {"duration": 60,  "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "warm-up"},
    # Dente 1
    {"duration": 240, "users_start": 80,  "users_end": 250, "spawn_rate": 5,  "label": "dente1-salita"},
    {"duration": 30,  "users_start": 250, "users_end": 80,  "spawn_rate": 30, "label": "dente1-crollo"},
    # Dente 2
    {"duration": 240, "users_start": 80,  "users_end": 250, "spawn_rate": 5,  "label": "dente2-salita"},
    {"duration": 30,  "users_start": 250, "users_end": 80,  "spawn_rate": 30, "label": "dente2-crollo"},
    # Dente 3
    {"duration": 240, "users_start": 80,  "users_end": 250, "spawn_rate": 5,  "label": "dente3-salita"},
    {"duration": 30,  "users_start": 250, "users_end": 80,  "spawn_rate": 30, "label": "dente3-crollo"},
    # Dente 4
    {"duration": 240, "users_start": 80,  "users_end": 250, "spawn_rate": 5,  "label": "dente4-salita"},
    {"duration": 30,  "users_start": 250, "users_end": 80,  "spawn_rate": 30, "label": "dente4-crollo"},
    # Dente 5
    {"duration": 240, "users_start": 80,  "users_end": 250, "spawn_rate": 5,  "label": "dente5-salita"},
    {"duration": 30,  "users_start": 250, "users_end": 80,  "spawn_rate": 30, "label": "dente5-crollo"},
    # Drain
    {"duration": 120, "users_start": 80,  "users_end": 0,   "spawn_rate": 10, "label": "drain"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario Registry
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    "gradual_ramp": {"phases": PHASES_GRADUAL_RAMP,  "type": "phases"},
    "flash_crowd":  {"phases": PHASES_FLASH_CROWD,   "type": "phases"},
    "double_wave":  {"phases": PHASES_DOUBLE_WAVE,    "type": "phases"},
    "sinusoidal":   {"config": SINUSOIDAL_CONFIG,     "type": "sinusoidal"},
    "slow_ramp":    {"phases": PHASES_SLOW_RAMP,      "type": "phases"},
    "sawtooth":     {"phases": PHASES_SAWTOOTH,       "type": "phases"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# LoadTestShape — Router tra scenari
# ═══════════════════════════════════════════════════════════════════════════════

class DMOSScenarioShape(LoadTestShape):
    """
    Shape universale che seleziona lo scenario in base alla variabile SCENARIO.
    Locust richiede esattamente una classe LoadTestShape nel file.
    """

    def __init__(self):
        super().__init__()
        self.scenario_name = SCENARIO
        self.scenario = SCENARIOS.get(self.scenario_name)
        if not self.scenario:
            raise ValueError(
                f"Scenario '{self.scenario_name}' non trovato. "
                f"Disponibili: {list(SCENARIOS.keys())}"
            )
        
        # Calcola durata totale per logging
        if self.scenario["type"] == "phases":
            total = sum(p["duration"] for p in self.scenario["phases"])
        else:
            total = self.scenario["config"]["duration_minutes"] * 60
        
        print(f"\n{'='*70}")
        print(f"  📊 DMOS SCENARIO: {self.scenario_name}")
        print(f"  ⏱  Durata: {total // 60} minuti {total % 60} secondi")
        if self.scenario["type"] == "phases":
            print(f"  📋 Fasi:")
            elapsed = 0
            for p in self.scenario["phases"]:
                t_start = f"{elapsed // 60}:{elapsed % 60:02d}"
                elapsed += p["duration"]
                t_end = f"{elapsed // 60}:{elapsed % 60:02d}"
                print(f"      {t_start:>6} → {t_end:>6}  "
                      f"{p['users_start']:>3}→{p['users_end']:>3} utenti  "
                      f"({p['label']})")
        else:
            cfg = self.scenario["config"]
            print(f"  🌊 Sinusoide: {cfg['min_users']}↔{cfg['max_users']} utenti, "
                  f"periodo {cfg['period_minutes']}min")
        print(f"{'='*70}\n")

    def tick(self):
        run_time = self.get_run_time()

        if self.scenario["type"] == "phases":
            result = compute_users_from_phases(self.scenario["phases"], run_time)
            return result  # (users, spawn_rate) or None

        elif self.scenario["type"] == "sinusoidal":
            cfg = self.scenario["config"]
            total_duration = cfg["duration_minutes"] * 60

            if run_time > total_duration:
                return None  # Fine test

            amplitude = (cfg["max_users"] - cfg["min_users"]) / 2
            center = (cfg["max_users"] + cfg["min_users"]) / 2
            period_sec = cfg["period_minutes"] * 60

            users = int(center + amplitude * math.sin(2 * math.pi * run_time / period_sec))
            return max(1, users), cfg["spawn_rate"]

        return None