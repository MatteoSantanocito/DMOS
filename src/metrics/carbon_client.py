"""
Carbon intensity client - Enhanced version
Supports ElectricityMaps API, realistic simulation profiles, and mock data

The realistic profiles are based on actual carbon intensity patterns from
Electricity Maps data (2024-2025 averages) for each region, with hourly
variation that reflects real renewable energy production cycles.
"""

import bisect
import csv
import glob
import math
import os
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from ..utils.logger import get_logger

logger = get_logger("CarbonClient")


# ============================================================================
# Realistic Carbon Intensity Profiles
# Based on Electricity Maps historical averages (2024-2025)
# Values in gCO2eq/kWh
# ============================================================================

REALISTIC_PROFILES = {
    # Germany (DE) - High coal/gas, significant solar midday dip
    # Average: ~350 gCO2/kWh, range: 200-500
    "DE": {
        "baseline": 350,
        "hourly": {
            0: 380, 1: 375, 2: 370, 3: 365, 4: 360, 5: 370,
            6: 390, 7: 400, 8: 380, 9: 340, 10: 300, 11: 270,
            12: 250, 13: 240, 14: 260, 15: 290, 16: 330, 17: 370,
            18: 400, 19: 410, 20: 400, 21: 395, 22: 390, 23: 385
        },
        "description": "Germany: coal/gas heavy, solar dip midday"
    },
    
    # US Virginia (US-VA / PJM interconnection) - Gas + nuclear + some renewables
    # Average: ~280 gCO2/kWh, range: 200-380
    "US-VA": {
        "baseline": 280,
        "hourly": {
            0: 260, 1: 250, 2: 245, 3: 240, 4: 245, 5: 260,
            6: 290, 7: 310, 8: 320, 9: 300, 10: 280, 11: 260,
            12: 250, 13: 245, 14: 255, 15: 270, 16: 295, 17: 320,
            18: 340, 19: 350, 20: 330, 21: 310, 22: 290, 23: 275
        },
        "description": "US Virginia (PJM): gas + nuclear mix"
    },
    
    # Singapore (SG) - Almost entirely natural gas, very stable
    # Average: ~420 gCO2/kWh, range: 380-460
    "SG": {
        "baseline": 420,
        "hourly": {
            0: 410, 1: 405, 2: 400, 3: 395, 4: 400, 5: 410,
            6: 420, 7: 435, 8: 445, 9: 450, 10: 455, 11: 450,
            12: 445, 13: 440, 14: 445, 15: 450, 16: 455, 17: 460,
            18: 455, 19: 445, 20: 435, 21: 425, 22: 420, 23: 415
        },
        "description": "Singapore: natural gas dominated, stable"
    },
    
    # Italy North (IT-NO) - Mix of gas + hydro + solar
    # Average: ~300 gCO2/kWh, range: 150-400
    "IT-NO": {
        "baseline": 300,
        "hourly": {
            0: 340, 1: 335, 2: 330, 3: 325, 4: 320, 5: 330,
            6: 350, 7: 360, 8: 330, 9: 280, 10: 240, 11: 200,
            12: 180, 13: 170, 14: 190, 15: 230, 16: 280, 17: 330,
            18: 370, 19: 380, 20: 370, 21: 360, 22: 355, 23: 345
        },
        "description": "Italy North: gas + hydro + significant solar"
    },
    
    # France (FR) - Nuclear dominated, very low carbon
    # Average: ~80 gCO2/kWh, range: 40-150
    "FR": {
        "baseline": 80,
        "hourly": {
            0: 60, 1: 55, 2: 50, 3: 45, 4: 50, 5: 60,
            6: 80, 7: 95, 8: 100, 9: 90, 10: 75, 11: 65,
            12: 60, 13: 55, 14: 60, 15: 70, 16: 85, 17: 105,
            18: 120, 19: 130, 20: 115, 21: 100, 22: 85, 23: 70
        },
        "description": "France: nuclear dominated, very low carbon"
    },
    
    # Norway (NO) - Almost 100% hydro, extremely low
    # Average: ~25 gCO2/kWh, range: 15-50
    "NO": {
        "baseline": 25,
        "hourly": {
            0: 20, 1: 18, 2: 17, 3: 16, 4: 17, 5: 20,
            6: 25, 7: 30, 8: 32, 9: 28, 10: 24, 11: 22,
            12: 20, 13: 19, 14: 21, 15: 24, 16: 28, 17: 35,
            18: 40, 19: 42, 20: 38, 21: 33, 22: 28, 23: 23
        },
        "description": "Norway: hydro dominated, extremely clean"
    },
    
    # Poland (PL) - Coal heavy, high carbon
    # Average: ~650 gCO2/kWh, range: 500-780
    "PL": {
        "baseline": 650,
        "hourly": {
            0: 620, 1: 610, 2: 600, 3: 590, 4: 600, 5: 620,
            6: 660, 7: 700, 8: 720, 9: 700, 10: 670, 11: 640,
            12: 620, 13: 610, 14: 630, 15: 660, 16: 700, 17: 740,
            18: 760, 19: 770, 20: 750, 21: 720, 22: 680, 23: 650
        },
        "description": "Poland: coal dominated, high carbon"
    }
}

# Seasonal adjustment factors (month -> multiplier)
# Reflects renewable energy availability variations
SEASONAL_FACTORS = {
    1: 1.15,  # January: less solar, more heating
    2: 1.10,
    3: 1.00,
    4: 0.90,
    5: 0.85,  # May: good solar + hydro
    6: 0.80,  # June: peak solar
    7: 0.82,
    8: 0.85,
    9: 0.90,
    10: 0.95,
    11: 1.05,
    12: 1.12   # December: less solar, more heating
}


class CarbonClient:
    """
    Client for retrieving carbon intensity data
    
    Supports three providers:
    - "electricitymaps": Real API data from Electricity Maps
    - "realistic": Simulated data based on real-world patterns
    - "mock": Simple static/hourly-varying mock data
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
        
        # For realistic provider: optional random noise
        self.noise_enabled = config.get('realistic', {}).get('noise_enabled', True)
        self.noise_percent = config.get('realistic', {}).get('noise_percent', 5)

        # For trace provider: lazy-loaded per-zone hourly series + replay state.
        self._trace_cfg = config.get('trace', {})
        self._trace_data: Dict[str, List[Tuple[float, float]]] = {}  # zone -> [(epoch_s, ci)]
        self._trace_missing: set = set()   # zone codes con file mancante (warn una volta)
        self._trace_t0: Optional[datetime] = None  # ancora real-time del replay (primo accesso)

        logger.info(f"Carbon client initialized with provider: {self.provider}")
        
        if self.provider == 'realistic':
            logger.info(f"Realistic profiles available: {list(REALISTIC_PROFILES.keys())}")
    
    def get_carbon_intensity(self, region_code: str) -> Optional[float]:
        """
        Get current carbon intensity for a region
        
        Args:
            region_code: Region code (e.g., "DE", "US-VA", "SG", "IT-NO")
        
        Returns:
            Carbon intensity in gCO2eq/kWh or None
        """
        # Check cache — saltata per 'trace': il replay (eventualmente accelerato)
        # deve restituire un valore fresco a ogni ciclo, non uno congelato per ttl.
        if region_code in self.cache and self.provider != 'trace':
            value, timestamp = self.cache[region_code]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                logger.debug(f"Cache hit for {region_code}: {value} gCO2/kWh")
                return value

        # Fetch new value based on provider
        if self.provider == 'mock':
            value = self._get_mock_value(region_code)
        elif self.provider == 'realistic':
            value = self._get_realistic_value(region_code)
        elif self.provider == 'trace':
            value = self._get_trace_value(region_code)
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
    
    def _get_realistic_value(self, region_code: str) -> Optional[float]:
        """
        Get realistic carbon intensity based on real-world patterns
        
        Uses hourly profiles derived from Electricity Maps historical data,
        with seasonal adjustment and optional random noise.
        
        Args:
            region_code: Region code (e.g., "DE", "US-VA", "SG")
        
        Returns:
            Realistic carbon intensity in gCO2eq/kWh
        """
        profile = REALISTIC_PROFILES.get(region_code)
        
        if profile is None:
            logger.warning(f"No realistic profile for {region_code}, "
                         f"available: {list(REALISTIC_PROFILES.keys())}")
            # Fallback to mock
            return self._get_mock_value(region_code)
        
        now = datetime.now()
        current_hour = now.hour
        
        # Get hourly value
        hourly_values = profile["hourly"]
        value = hourly_values.get(current_hour, profile["baseline"])
        
        # Apply seasonal factor
        seasonal = SEASONAL_FACTORS.get(now.month, 1.0)
        value *= seasonal
        
        # Apply optional noise (deterministic based on minute for reproducibility)
        if self.noise_enabled:
            # Use minute as seed for reproducible but varying noise
            noise_seed = (now.minute * 7 + now.second // 15) % 100
            noise_factor = 1.0 + (noise_seed - 50) / 50.0 * (self.noise_percent / 100.0)
            value *= noise_factor
        
        value = round(value, 1)
        
        logger.info(f"Realistic CI for {region_code}: {value} gCO2/kWh "
                    f"(hour={current_hour}, season={seasonal:.2f}, "
                    f"profile={profile['description']})")
        
        return value
    
    # ========================================================================
    # Trace provider — replay di tracce orarie REALI (Electricity Maps data
    # portal). Dà valori reali E riproducibili: l'opzione consigliata per il
    # paper. La traccia per-zona è caricata pigramente una sola volta.
    # ========================================================================

    @staticmethod
    def _epoch_utc(dt: datetime) -> float:
        """Epoch (s) interpretando dt come UTC, indipendente dal fuso locale."""
        return (dt - datetime(1970, 1, 1)).total_seconds()

    @staticmethod
    def _parse_dt(raw: str) -> Optional[datetime]:
        """Parsa i formati datetime usati dai CSV Electricity Maps."""
        s = raw.strip().replace("Z", "").replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    def trace_zone(self, region_code: str) -> str:
        """Electricity Maps zone of the trace used for a cluster region code.

        `trace.zone_aliases` in carbon.yaml maps region codes to trace zones
        when they differ (e.g. NO -> NO-NO1, US-VA -> US-MIDA-PJM); otherwise
        the region code itself is the zone.
        """
        aliases = self._trace_cfg.get('zone_aliases') or {}
        return str(aliases.get(region_code, region_code))

    def trace_file(self, region_code: str) -> Optional[str]:
        """Path of the trace CSV for a region code, or None if there is none."""
        zone = self.trace_zone(region_code)
        data_dir = self._trace_cfg.get('data_dir', 'data/carbon_traces')
        # Risolvi relativo alla repo root (questo file è in src/metrics/)
        if not os.path.isabs(data_dir):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            data_dir = os.path.join(repo_root, data_dir)
        for pat in (f"{zone}_*hourly*.csv", f"{zone}_*.csv", f"{zone}.csv"):
            matches = sorted(glob.glob(os.path.join(data_dir, pat)))
            if matches:
                return matches[0]
        return None

    def source_for(self, region_code: str) -> str:
        """Where the intensity of a region comes from, without loading data:
        'trace:<file>' | 'realistic' | 'mock' | 'default-300' | '<provider>'."""
        if self.provider == 'trace':
            path = self.trace_file(region_code)
            if path:
                return f"trace:{os.path.basename(path)}"
            if region_code in REALISTIC_PROFILES:
                return "realistic"
            mock = self.config.get('mock', {}) or {}
            if mock.get('enabled') and region_code in (mock.get('values') or {}):
                return "mock"
            return "default-300"
        return str(self.provider)

    def _load_trace_csv(self, zone: str) -> List[Tuple[float, float]]:
        """
        Carica la serie oraria (epoch_s, CI life-cycle) per una regione da
        <data_dir>/<zona>*.csv (formato Electricity Maps data portal), dove la
        zona è region_code tradotto con trace.zone_aliases.

        Rileva automaticamente la colonna 'Carbon Intensity ... (Life cycle)'
        (fallback su 'direct') e la colonna datetime. Ritorna la serie ordinata
        per tempo, o [] se il file manca/è illeggibile.
        """
        path = self.trace_file(zone)
        if path is None:
            logger.warning(f"Trace: nessun CSV per zona '{self.trace_zone(zone)}' "
                           f"(region_code '{zone}') in "
                           f"{self._trace_cfg.get('data_dir', 'data/carbon_traces')}")
            return []
        matches = [path]

        series: List[Tuple[float, float]] = []
        try:
            with open(matches[0], newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                # colonna CI: preferisci life-cycle, poi direct
                ci_col = next((h for h in headers if 'life cycle' in h.lower()), None)
                if ci_col is None:
                    ci_col = next((h for h in headers if 'direct' in h.lower()
                                   and 'carbon' in h.lower()), None)
                dt_col = next((h for h in headers if 'datetime' in h.lower()), None)
                if ci_col is None or dt_col is None:
                    logger.warning(f"Trace {zone}: colonne CI/datetime non trovate "
                                   f"in {os.path.basename(matches[0])} ({headers})")
                    return []
                for row in reader:
                    dt = self._parse_dt(row.get(dt_col, ''))
                    raw_ci = row.get(ci_col, '')
                    if dt is None or raw_ci in (None, '', 'NA', 'nan'):
                        continue
                    try:
                        ci = float(raw_ci)
                    except ValueError:
                        continue
                    series.append((self._epoch_utc(dt), ci))
        except Exception as e:
            logger.error(f"Trace {zone}: errore lettura {matches[0]}: {e}")
            return []

        series.sort(key=lambda x: x[0])
        logger.info(f"Trace {zone}: caricati {len(series)} campioni orari da "
                    f"{os.path.basename(matches[0])}")
        return series

    def _get_trace_value(self, region_code: str) -> Optional[float]:
        """
        CI(t) per replay a finestra fissa di una traccia oraria reale.

            trace_time = window_start + speedup · (now − t0)

        - window_start: istante fisso nella traccia (config 'window_start').
        - speedup: compressione temporale (1.0 = tempo reale; >1 = un run breve
          spazza piu' ore di traccia, rendendo Φ_carbon dinamico). Default 1.0.
        - t0: ancora real-time fissata al primo accesso (→ riproducibile: dipende
          solo dal tempo trascorso, non dall'orologio assoluto).
        Oltre la fine della traccia si avvolge (modulo) sulla finestra disponibile.
        Interpolazione lineare fra i due campioni orari adiacenti.
        Fallback al provider 'realistic' se la traccia manca.
        """
        if region_code not in self._trace_data:
            self._trace_data[region_code] = self._load_trace_csv(region_code)
        series = self._trace_data[region_code]
        if not series:
            if region_code not in self._trace_missing:
                self._trace_missing.add(region_code)
                logger.warning(f"Trace {region_code}: assente → fallback 'realistic'")
            return self._get_realistic_value(region_code)

        if self._trace_t0 is None:
            self._trace_t0 = datetime.now()
            # Provenance: the anchor fixes the whole replay of this run
            # (trace_time = window_start + speedup·(now − t0)); logging it
            # lets the run's carbon timeline be reconstructed offline.
            logger.info(
                "Trace replay anchored: t0=%s (local), window_start=%s, "
                "speedup=%s" % (self._trace_t0.isoformat(timespec='seconds'),
                                self._trace_cfg.get('window_start'),
                                self._trace_cfg.get('speedup', 1.0)))
        elapsed = (datetime.now() - self._trace_t0).total_seconds()

        speedup = float(self._trace_cfg.get('speedup', 1.0))
        ws_raw = self._trace_cfg.get('window_start')
        ws_dt = self._parse_dt(ws_raw) if ws_raw else None
        t_first, t_last = series[0][0], series[-1][0]
        span = max(1.0, t_last - t_first)
        base = self._epoch_utc(ws_dt) if ws_dt else t_first

        trace_t = base + speedup * elapsed
        # wrap-around dentro [t_first, t_last]
        trace_t = t_first + ((trace_t - t_first) % span)

        # interpolazione lineare sui due campioni adiacenti
        times = [s[0] for s in series]
        j = bisect.bisect_right(times, trace_t)
        if j <= 0:
            value = series[0][1]
        elif j >= len(series):
            value = series[-1][1]
        else:
            t0_, c0_ = series[j - 1]
            t1_, c1_ = series[j]
            frac = 0.0 if t1_ == t0_ else (trace_t - t0_) / (t1_ - t0_)
            value = c0_ + frac * (c1_ - c0_)

        value = round(value, 1)
        logger.info(f"Trace CI {region_code}: {value} gCO2/kWh "
                    f"(elapsed={elapsed:.0f}s, speedup={speedup:g})")
        return value

    def _get_mock_value(self, region_code: str) -> Optional[float]:
        """
        Get mock carbon intensity value (original implementation)
        """
        mock_config = self.config.get('mock', {})
        
        if not mock_config.get('enabled', False):
            logger.warning("Mock provider not enabled")
            return None
        
        baseline = mock_config.get('values', {}).get(region_code)
        if baseline is None:
            logger.warning(f"No mock value for region {region_code}")
            return None
        
        if mock_config.get('hourly_variation', {}).get('enabled', False):
            current_hour = datetime.now().hour
            pattern = mock_config.get('hourly_variation', {}).get('pattern', {})
            multiplier = pattern.get(current_hour, 1.0)
            value = baseline * multiplier
            
            logger.debug(f"Mock CI for {region_code}: {value:.1f} gCO2/kWh "
                        f"(baseline={baseline}, hour={current_hour}, mult={multiplier})")
            return value
        
        return float(baseline)
    
    def _get_electricitymaps_value(self, region_code: str) -> Optional[float]:
        """
        Get carbon intensity from ElectricityMaps API
        
        API endpoint: GET /v3/carbon-intensity/latest?zone={zone}
        Header: auth-token: {api_key}
        Response: {"carbonIntensity": 350, "zone": "DE", ...}
        """
        em_config = self.config.get('electricitymaps', {})
        api_url = em_config.get('api_url')
        api_key = em_config.get('api_key')
        
        if not api_url or not api_key or api_key == "YOUR_API_KEY_HERE":
            logger.warning(f"ElectricityMaps API not configured, "
                          f"falling back to realistic profile for {region_code}")
            return self._get_realistic_value(region_code)
        
        try:
            headers = {'auth-token': api_key}
            params = {'zone': region_code}
            
            r = requests.get(api_url, headers=headers, params=params, timeout=10)
            r.raise_for_status()
            
            data = r.json()
            carbon_intensity = data.get('carbonIntensity')
            
            if carbon_intensity is not None:
                logger.info(f"ElectricityMaps CI for {region_code}: "
                          f"{carbon_intensity} gCO2/kWh "
                          f"(fossil%={data.get('fossilFuelPercentage', 'N/A')})")
                return float(carbon_intensity)
            else:
                logger.warning(f"No carbonIntensity in response for {region_code}")
                return self._get_realistic_value(region_code)
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("ElectricityMaps: Invalid API key")
            elif e.response.status_code == 429:
                logger.warning("ElectricityMaps: Rate limit exceeded, using realistic fallback")
            else:
                logger.error(f"ElectricityMaps HTTP error: {e}")
            return self._get_realistic_value(region_code)
            
        except Exception as e:
            logger.error(f"ElectricityMaps API error: {e}")
            return self._get_realistic_value(region_code)
    
    def _get_watttime_value(self, region_code: str) -> Optional[float]:
        """
        Get carbon intensity from WattTime API
        TODO: Implement WattTime API client
        """
        logger.warning("WattTime API not yet implemented, using realistic fallback")
        return self._get_realistic_value(region_code)
    
    def get_all_regions(self) -> Dict[str, float]:
        """
        Get carbon intensity for all configured regions
        """
        if self.provider == 'realistic':
            result = {}
            for region in REALISTIC_PROFILES.keys():
                ci = self.get_carbon_intensity(region)
                if ci is not None:
                    result[region] = ci
            return result
        elif self.provider == 'mock':
            mock_values = self.config.get('mock', {}).get('values', {})
            result = {}
            for region in mock_values.keys():
                ci = self.get_carbon_intensity(region)
                if ci is not None:
                    result[region] = ci
            return result
        else:
            # For API providers, query only configured cluster regions
            result = {}
            # Try common regions
            for region in ['DE', 'US-VA', 'SG', 'IT-NO']:
                ci = self.get_carbon_intensity(region)
                if ci is not None:
                    result[region] = ci
            return result
    
    def get_carbon_savings_estimate(
        self, 
        region_a: str, 
        region_b: str, 
        power_watts: float,
        duration_hours: float = 1.0
    ) -> Dict:
        """
        Estimate carbon savings from placing workload in region_a vs region_b
        
        Args:
            region_a: First region code
            region_b: Second region code
            power_watts: Power consumption in watts
            duration_hours: Duration in hours
            
        Returns:
            Dict with carbon comparison and savings estimate
        """
        ci_a = self.get_carbon_intensity(region_a)
        ci_b = self.get_carbon_intensity(region_b)
        
        if ci_a is None or ci_b is None:
            return {"error": "Could not get carbon intensity for one or both regions"}
        
        power_kwh = (power_watts / 1000.0) * duration_hours
        
        emissions_a = ci_a * power_kwh  # gCO2eq
        emissions_b = ci_b * power_kwh  # gCO2eq
        savings = emissions_b - emissions_a  # positive = A is greener
        
        return {
            "region_a": {"code": region_a, "ci": ci_a, "emissions_gco2": round(emissions_a, 2)},
            "region_b": {"code": region_b, "ci": ci_b, "emissions_gco2": round(emissions_b, 2)},
            "savings_gco2": round(savings, 2),
            "savings_percent": round((savings / emissions_b) * 100, 1) if emissions_b > 0 else 0,
            "greener_region": region_a if ci_a < ci_b else region_b
        }