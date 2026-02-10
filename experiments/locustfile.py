"""
Locust load test for Online Boutique
Simulates realistic user behavior
"""

from locust import HttpUser, task, between, events
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OnlineBoutiqueUser(HttpUser):
    """
    Simulated user for Online Boutique
    """
    wait_time = between(1, 3)  # Wait 1-3s between requests
    
    def on_start(self):
        """Called when user starts"""
        logger.info(f"User started: {self.environment.runner.user_count} users")
    
    @task(5)  # Weight 5 (most common)
    def view_homepage(self):
        """View homepage"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Got {response.status_code}")
    
    @task(3)
    def view_product(self):
        """View random product"""
        products = [
            "OLJCESPC7Z",  # Vintage Camera Lens
            "66VCHSJNUP",  # Vintage Record Player
            "1YMWWN1N4O",  # Home Barista Kit
            "L9ECAV7KIM",  # Loafers
            "2ZYFJ3GM2N",  # Tank Top
            "0PUK6V6EV0",  # Vintage Typewriter
        ]
        product_id = random.choice(products)
        
        with self.client.get(f"/product/{product_id}", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
    
    @task(2)
    def add_to_cart(self):
        """Add product to cart"""
        products = ["OLJCESPC7Z", "66VCHSJNUP", "1YMWWN1N4O"]
        product_id = random.choice(products)
        quantity = random.randint(1, 3)
        
        payload = {
            "product_id": product_id,
            "quantity": quantity
        }
        
        with self.client.post("/cart", data=payload, catch_response=True) as response:
            if response.status_code in [200, 302]:  # 302 redirect OK
                response.success()
    
    @task(1)
    def view_cart(self):
        """View cart"""
        with self.client.get("/cart", catch_response=True) as response:
            if response.status_code == 200:
                response.success()


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts"""
    logger.info("=" * 70)
    logger.info("Load test starting...")
    logger.info("=" * 70)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops"""
    logger.info("=" * 70)
    logger.info("Load test completed")
    logger.info(f"Total users: {environment.runner.user_count}")
    logger.info("=" * 70)