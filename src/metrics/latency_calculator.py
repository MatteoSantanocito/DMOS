"""
Latency calculator for geographic distance estimation
"""

import math
from typing import Dict, Tuple
from ..utils.logger import get_logger

logger = get_logger("LatencyCalculator")


class LatencyCalculator:
    """
    Calculate network latency based on geographic distance
    """
    
    # Approximate coordinates (latitude, longitude)
    CITY_COORDS = {
        'Frankfurt': (50.1109, 8.6821),
        'Virginia': (38.0336, -78.5080),
        'Singapore': (1.3521, 103.8198),
        'London': (51.5074, -0.1278),
        'Tokyo': (35.6762, 139.6503),
        'Sydney': (-33.8688, 151.2093),
    }
    
    # Speed of light in fiber: ~200,000 km/s (2/3 of c)
    # Round-trip latency: ~10 ms per 1000 km
    LATENCY_PER_1000KM = 10.0  # ms
    
    def __init__(self):
        """Initialize latency calculator"""
        pass
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points on Earth
        
        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)
        
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in km
        r = 6371
        
        return c * r
    
    def calculate_latency(self, city1: str, city2: str) -> float:
        """
        Calculate estimated latency between two cities
        
        Args:
            city1: First city name
            city2: Second city name
        
        Returns:
            Estimated round-trip latency in milliseconds
        """
        if city1 not in self.CITY_COORDS or city2 not in self.CITY_COORDS:
            logger.warning(f"Unknown city: {city1} or {city2}, using default 100ms")
            return 100.0
        
        if city1 == city2:
            return 2.0  # Same city, minimal latency
        
        lat1, lon1 = self.CITY_COORDS[city1]
        lat2, lon2 = self.CITY_COORDS[city2]
        
        distance_km = self.haversine_distance(lat1, lon1, lat2, lon2)
        latency_ms = (distance_km / 1000.0) * self.LATENCY_PER_1000KM
        
        logger.debug(f"Latency {city1} ↔ {city2}: {latency_ms:.1f}ms ({distance_km:.0f}km)")
        
        return latency_ms
    
    def calculate_latency_matrix(self, cities: list) -> Dict[Tuple[str, str], float]:
        """
        Calculate latency matrix for all pairs of cities
        
        Args:
            cities: List of city names
        
        Returns:
            Dict mapping (city1, city2) to latency in ms
        """
        matrix = {}
        
        for i, city1 in enumerate(cities):
            for city2 in cities[i:]:  # Only calculate upper triangle
                latency = self.calculate_latency(city1, city2)
                matrix[(city1, city2)] = latency
                matrix[(city2, city1)] = latency  # Symmetric
        
        return matrix