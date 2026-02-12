"""
DMOS Variable Traffic Load Test
================================
Designed to test DMOS predictive scaling capabilities.

Traffic Pattern (total ~25 minutes):
  Phase 1:  0:00 - 3:00   Warm-up          50 users   (low baseline)
  Phase 2:  3:00 - 6:00   Gradual ramp     50 → 250   (tests proactive scale-up)
  Phase 3:  6:00 - 10:00  Sustained peak   250 users   (tests stability at load)
  Phase 4: 10:00 - 13:00  Gradual decline  250 → 80    (tests scale-down cooldown)
  Phase 5: 13:00 - 15:00  Low traffic      80 users    (tests over-provisioning)
  Phase 6: 15:00 - 17:00  Sudden spike     80 → 300    (tests reactive vs proactive)
  Phase 7: 17:00 - 20:00  High sustained   300 users   (peak load stability)
  Phase 8: 20:00 - 23:00  Gradual cooldown 300 → 50    (final scale-down)
  Phase 9: 23:00 - 25:00  Baseline return  50 users    (back to baseline)

The gradual ramps are the key phases for demonstrating DMOS value:
- If DMOS is proactive, it will scale BEFORE the ramp reaches the threshold
- A reactive system (HPA) would only scale AFTER latency spikes

Usage:
  locust -f locustfile_variable.py --host http://192.168.1.245:30007 --headless \
         --users 300 --spawn-rate 10 --run-time 25m
  
  OR with web UI:
  locust -f locustfile_variable.py --host http://192.168.1.245:30007

  Then start the test from the web UI — the LoadTestShape controls the user count
  automatically, so the --users flag is only the max.
"""

import math
import time
from locust import HttpUser, task, between, LoadTestShape


# ═══════════════════════════════════════════════════════════════════════════════
# Traffic Shape Definition
# ═══════════════════════════════════════════════════════════════════════════════

class DMOSTrafficShape(LoadTestShape):
    """
    Custom traffic shape with gradual ramps and sudden spikes.
    
    Returns (user_count, spawn_rate) at each time step.
    Locust calls tick() every second to get the current desired state.
    """
    
    # Define phases as (duration_seconds, start_users, end_users, label)
    phases = [
        # Phase 1: Warm-up baseline
        {"duration": 180, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "warm-up"},
        # Phase 2: Gradual ramp up (key for proactive scaling test)
        {"duration": 180, "users_start": 50,  "users_end": 250, "spawn_rate": 5,  "label": "ramp-up"},
        # Phase 3: Sustained peak
        {"duration": 240, "users_start": 250, "users_end": 250, "spawn_rate": 10, "label": "peak"},
        # Phase 4: Gradual decline
        {"duration": 180, "users_start": 250, "users_end": 80,  "spawn_rate": 5,  "label": "ramp-down"},
        # Phase 5: Low traffic valley
        {"duration": 120, "users_start": 80,  "users_end": 80,  "spawn_rate": 10, "label": "valley"},
        # Phase 6: Sudden spike (tests reaction speed)
        {"duration": 120, "users_start": 80,  "users_end": 300, "spawn_rate": 20, "label": "spike"},
        # Phase 7: High sustained
        {"duration": 180, "users_start": 300, "users_end": 300, "spawn_rate": 10, "label": "high-sustained"},
        # Phase 8: Final gradual cooldown
        {"duration": 180, "users_start": 300, "users_end": 50,  "spawn_rate": 5,  "label": "cooldown"},
        # Phase 9: Return to baseline
        {"duration": 120, "users_start": 50,  "users_end": 50,  "spawn_rate": 10, "label": "baseline-return"},
    ]
    
    def tick(self):
        """Called every second by Locust. Returns (user_count, spawn_rate) or None to stop."""
        run_time = self.get_run_time()
        
        elapsed = 0
        for phase in self.phases:
            phase_end = elapsed + phase["duration"]
            
            if run_time < phase_end:
                # We're in this phase
                phase_progress = (run_time - elapsed) / phase["duration"]
                
                # Linear interpolation between start and end users
                current_users = int(
                    phase["users_start"] + 
                    (phase["users_end"] - phase["users_start"]) * phase_progress
                )
                
                # Log phase transitions
                if run_time - elapsed < 1:  # First second of phase
                    total_min = phase_end / 60
                    print(f"\n🔄 Phase: {phase['label']} | "
                          f"Users: {phase['users_start']} → {phase['users_end']} | "
                          f"Duration: {phase['duration']}s | "
                          f"Total: {total_min:.0f}min")
                
                return (current_users, phase["spawn_rate"])
            
            elapsed = phase_end
        
        # All phases complete
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# User Behavior
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineBoutiqueUser(HttpUser):
    """
    Simulates a realistic user browsing the Online Boutique.
    
    Behavior mix:
      - 40% Browse homepage (heaviest page, loads product catalog)
      - 25% View product details  
      - 15% Add to cart
      - 10% View cart
      - 5%  Set currency
      - 5%  Checkout (most expensive operation)
    """
    
    wait_time = between(1, 3)  # 1-3 seconds between requests
    
    # Product IDs from Online Boutique catalog
    PRODUCTS = [
        "OLJCESPC7Z",  # Sunglasses
        "66VCHSJNUP",  # Tank Top
        "1YMWWN1N4O",  # Watch
        "L9ECAV7KIM",  # Loafers
        "2ZYFJ3GM2N",  # Hairdryer
        "0PUK6V6EV0",  # Candle
        "LS4PSXUNUM",  # Salt & Pepper Shakers
        "9SIQT8TOJO",  # Bamboo Glass Jar
        "6E92ZMYYFZ",  # Mug
    ]
    
    CURRENCIES = ["EUR", "USD", "JPY", "GBP", "CAD"]
    
    @task(40)
    def browse_homepage(self):
        """Browse the homepage — triggers productcatalog + recommendation + ad services."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Homepage: {response.status_code}")
    
    @task(25)
    def view_product(self):
        """View a product detail page — triggers productcatalog + recommendation."""
        import random
        product_id = random.choice(self.PRODUCTS)
        with self.client.get(f"/product/{product_id}", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Product {product_id}: {response.status_code}")
    
    @task(15)
    def add_to_cart(self):
        """Add item to cart — triggers cartservice."""
        import random
        product_id = random.choice(self.PRODUCTS)
        with self.client.post("/cart", data={
            "product_id": product_id,
            "quantity": random.randint(1, 3),
        }, catch_response=True) as response:
            if response.status_code not in [200, 302]:
                response.failure(f"Add to cart: {response.status_code}")
    
    @task(10)
    def view_cart(self):
        """View cart page — triggers cartservice + recommendation."""
        with self.client.get("/cart", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Cart: {response.status_code}")
    
    @task(5)
    def set_currency(self):
        """Change currency — triggers currencyservice."""
        import random
        currency = random.choice(self.CURRENCIES)
        with self.client.post("/setCurrency", data={
            "currency_code": currency,
        }, catch_response=True) as response:
            if response.status_code not in [200, 302]:
                response.failure(f"Currency: {response.status_code}")
    
    @task(5)
    def checkout(self):
        """Full checkout flow — triggers checkout, payment, shipping, email services."""
        # First add something to cart
        import random
        product_id = random.choice(self.PRODUCTS)
        self.client.post("/cart", data={
            "product_id": product_id,
            "quantity": 1,
        })
        
        # Then checkout
        with self.client.post("/cart/checkout", data={
            "email": "test@example.com",
            "street_address": "123 Test St",
            "zip_code": "10001",
            "city": "New York",
            "state": "NY",
            "country": "US",
            "credit_card_number": "4432-8015-6152-0454",
            "credit_card_expiration_month": "1",
            "credit_card_expiration_year": "2030",
            "credit_card_cvv": "672",
        }, catch_response=True) as response:
            if response.status_code not in [200, 302]:
                response.failure(f"Checkout: {response.status_code}")