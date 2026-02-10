"""
Carbon intensity client
Supports ElectricityMaps API and mock data
"""

import requests
from typing import Dict, Optional
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger("CarbonClient")


class CarbonClient:
    """
    Client for retrieving carbon intensity data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize carbon client
        
        Args:
            config: Carbon configuration dict from config_loader
        """
        self.provider = config.get('provider', 'mock')
        self.config = config
        self.cache: Dict[str, tuple] = {}  # {region: (value, timestamp)}
        self.cache_ttl = config.get('cache', {}).get('ttl_seconds', 300)
        
        logger.info(f"Carbon client initialized with provider: {self.provider}")
    
    def get_carbon_intensity(self, region_code: str) -> Optional[float]:
        """
        Get current carbon intensity for a region
        
        Args:
            region_code: Region code (e.g., "DE", "US-VA", "SG")
        
        Returns:
            Carbon intensity in gCO2/kWh or None
        """
        # Check cache
        if region_code in self.cache:
            value, timestamp = self.cache[region_code]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                logger.debug(f"Cache hit for {region_code}: {value} gCO2/kWh")
                return value
        
        # Fetch new value
        if self.provider == 'mock':
            value = self._get_mock_value(region_code)
        elif self.provider == 'electricitymaps':
            value = self._get_electricitymaps_value(region_code)
        elif self.provider == 'watttime':
            value = self._get_watttime_value(region_code)
        else:
            logger.error(f"Unknown carbon provider: {self.provider}")
            value = None
        
        # Update cache
        if value is not None:
            self.cache[region_code] = (value, datetime.now())
        
        return value
    
    def _get_mock_value(self, region_code: str) -> Optional[float]:
        """
        Get mock carbon intensity value
        
        Args:
            region_code: Region code
        
        Returns:
            Mock carbon intensity
        """
        mock_config = self.config.get('mock', {})
        
        if not mock_config.get('enabled', False):
            logger.warning("Mock provider not enabled")
            return None
        
        # Get baseline value
        baseline = mock_config.get('values', {}).get(region_code)
        if baseline is None:
            logger.warning(f"No mock value for region {region_code}")
            return None
        
        # Apply hourly variation if enabled
        if mock_config.get('hourly_variation', {}).get('enabled', False):
            current_hour = datetime.now().hour
            pattern = mock_config.get('hourly_variation', {}).get('pattern', {})
            
            # Get multiplier for current hour (default 1.0)
            multiplier = pattern.get(current_hour, 1.0)
            value = baseline * multiplier
            
            logger.debug(f"Mock CI for {region_code}: {value:.1f} gCO2/kWh (baseline={baseline}, hour={current_hour}, mult={multiplier})")
            return value
        
        return float(baseline)
    
    def _get_electricitymaps_value(self, region_code: str) -> Optional[float]:
        """
        Get carbon intensity from ElectricityMaps API
        
        Args:
            region_code: Region code
        
        Returns:
            Carbon intensity or None
        """
        em_config = self.config.get('electricitymaps', {})
        api_url = em_config.get('api_url')
        api_key = em_config.get('api_key')
        
        if not api_url or not api_key:
            logger.error("ElectricityMaps API not configured")
            return None
        
        try:
            headers = {'auth-token': api_key}
            params = {'zone': region_code}
            
            r = requests.get(api_url, headers=headers, params=params, timeout=10)
            r.raise_for_status()
            
            data = r.json()
            carbon_intensity = data.get('carbonIntensity')
            
            if carbon_intensity is not None:
                logger.info(f"ElectricityMaps CI for {region_code}: {carbon_intensity} gCO2/kWh")
                return float(carbon_intensity)
            else:
                logger.warning(f"No carbon intensity in response for {region_code}")
                return None
                
        except Exception as e:
            logger.error(f"ElectricityMaps API error: {e}")
            return None
    
    def _get_watttime_value(self, region_code: str) -> Optional[float]:
        """
        Get carbon intensity from WattTime API
        
        Args:
            region_code: Region code
        
        Returns:
            Carbon intensity or None
        """
        wt_config = self.config.get('watttime', {})
        # TODO: Implement WattTime API client
        logger.warning("WattTime API not yet implemented")
        return None
    
    def get_all_regions(self) -> Dict[str, float]:
        """
        Get carbon intensity for all configured regions
        
        Returns:
            Dict mapping region_code to carbon intensity
        """
        if self.provider == 'mock':
            mock_values = self.config.get('mock', {}).get('values', {})
            result = {}
            for region in mock_values.keys():
                ci = self.get_carbon_intensity(region)
                if ci is not None:
                    result[region] = ci
            return result
        else:
            logger.warning("get_all_regions only supported for mock provider")
            return {}