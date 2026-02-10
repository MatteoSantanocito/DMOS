"""
Traffic Predictor
Trend-based prediction for request rate
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from ..utils.logger import get_logger

logger = get_logger("TrafficPredictor")


@dataclass
class TrafficSample:
    """Single traffic measurement"""
    timestamp: datetime
    request_rate: float  # req/s


class TrafficPredictor:
    """
    Trend-based traffic predictor
    
    From paper:
    Λ^pred(t) = Λ(t) + (dΛ/dt) · Δt_horizon
    
    Derivative approximation:
    dΛ/dt ≈ (Λ(t) - Λ(t - τ)) / τ
    """
    
    def __init__(
        self,
        window_seconds: int = 300,    # 5 minutes for derivative
        horizon_seconds: int = 300,   # 5 minutes prediction
        smoothing_alpha: float = 0.3  # EMA smoothing (optional)
    ):
        """
        Initialize traffic predictor
        
        Args:
            window_seconds: Time window for derivative calculation
            horizon_seconds: Prediction horizon
            smoothing_alpha: EMA smoothing coefficient (0-1)
        """
        self.window_seconds = window_seconds
        self.horizon_seconds = horizon_seconds
        self.smoothing_alpha = smoothing_alpha
        
        # History storage
        self.history: List[TrafficSample] = []
        
        logger.info(f"Traffic predictor initialized: "
                   f"window={window_seconds}s, horizon={horizon_seconds}s")
    
    def add_sample(self, timestamp: datetime, request_rate: float):
        """
        Add traffic sample to history
        
        Args:
            timestamp: Sample timestamp
            request_rate: Request rate (req/s)
        """
        sample = TrafficSample(timestamp=timestamp, request_rate=request_rate)
        self.history.append(sample)
        
        # Keep only relevant history (2x window for safety)
        cutoff = datetime.now() - timedelta(seconds=self.window_seconds * 2)
        self.history = [s for s in self.history if s.timestamp > cutoff]
        
        logger.debug(f"Added sample: {request_rate:.1f} req/s at {timestamp}, "
                    f"history size={len(self.history)}")
    
    def _compute_derivative(self, current_rate: float) -> Optional[float]:
        """
        Compute traffic derivative (trend)
        
        dΛ/dt ≈ (Λ(t) - Λ(t - τ)) / τ
        
        Args:
            current_rate: Current traffic rate
        
        Returns:
            Derivative in req/s² or None if insufficient data
        """
        if len(self.history) < 2:
            logger.warning("Insufficient history for derivative")
            return None
        
        # Find sample approximately window_seconds ago
        target_time = datetime.now() - timedelta(seconds=self.window_seconds)
        
        # Get closest sample to target time
        past_sample = min(
            self.history,
            key=lambda s: abs((s.timestamp - target_time).total_seconds())
        )
        
        # Calculate derivative
        dt = (datetime.now() - past_sample.timestamp).total_seconds()
        if dt == 0:
            return 0.0
        
        d_rate = current_rate - past_sample.request_rate
        derivative = d_rate / dt
        
        logger.debug(f"Derivative: Δrate={d_rate:.2f}, Δt={dt:.0f}s, "
                    f"d/dt={derivative:.4f} req/s²")
        
        return derivative
    
    def predict(
        self, 
        current_rate: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, dict]:
        """
        Predict future traffic rate
        
        Args:
            current_rate: Current traffic rate (req/s)
            timestamp: Current timestamp (default: now)
        
        Returns:
            Tuple of (predicted_rate, metadata_dict)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add current sample to history
        self.add_sample(timestamp, current_rate)
        
        # Compute derivative
        derivative = self._compute_derivative(current_rate)
        
        if derivative is None:
            # Fallback: no prediction, return current
            logger.warning("No derivative, using current rate as prediction")
            return current_rate, {
                'method': 'fallback',
                'derivative': 0.0,
                'horizon_seconds': 0
            }
        
        # Linear extrapolation
        predicted_rate = current_rate + derivative * self.horizon_seconds
            # Damping factor per trend
        if derivative > 0:
            # Scale-up: dampen positive trends
            trend_factor = 0.3  # ← Solo 30% del trend
        else:
            # Scale-down: più reattivo
            trend_factor = 0.7  # ← 70% del trend negativo
        
        
        predicted_rate = current_rate + (derivative * self.horizon_seconds * trend_factor)
    
        # Cap prediction a 2x current
        predicted_rate = min(predicted_rate, current_rate * 2.0)
    
        # Ensure non-negative
        predicted_rate = max(0.0, predicted_rate)
        
        logger.info(f"Prediction: current={current_rate:.1f} req/s, "
                   f"derivative={derivative:.4f}, "
                   f"predicted={predicted_rate:.1f} req/s "
                   f"(+{self.horizon_seconds}s)")
        
        metadata = {
            'method': 'trend-based',
            'current_rate': current_rate,
            'derivative': derivative,
            'horizon_seconds': self.horizon_seconds,
            'predicted_rate': predicted_rate,
            'history_size': len(self.history)
        }
        
        return predicted_rate, metadata
    
    def predict_with_ema_smoothing(
        self,
        current_rate: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, dict]:
        """
        Predict with Exponential Moving Average smoothing
        
        Args:
            current_rate: Current rate
            timestamp: Current timestamp
        
        Returns:
            Tuple of (smoothed_prediction, metadata)
        """
        # Get base prediction
        predicted_rate, metadata = self.predict(current_rate, timestamp)
        
        # Apply EMA smoothing
        if len(self.history) >= 2:
            previous_rate = self.history[-2].request_rate
            smoothed = (self.smoothing_alpha * predicted_rate + 
                       (1 - self.smoothing_alpha) * previous_rate)
        else:
            smoothed = predicted_rate
        
        metadata['smoothed_rate'] = smoothed
        metadata['smoothing_alpha'] = self.smoothing_alpha
        
        logger.debug(f"EMA smoothing: raw={predicted_rate:.1f}, "
                    f"smoothed={smoothed:.1f}")
        
        return smoothed, metadata
    
    def get_statistics(self) -> dict:
        """
        Get statistics about traffic history
        
        Returns:
            Dict with mean, std, min, max
        """
        if not self.history:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        rates = [s.request_rate for s in self.history]
        
        return {
            'count': len(rates),
            'mean': np.mean(rates),
            'std': np.std(rates),
            'min': np.min(rates),
            'max': np.max(rates)
        }