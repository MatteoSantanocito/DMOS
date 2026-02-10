"""
PD Controller for replica adjustment
Proportional-Derivative control to smooth scaling
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger

logger = get_logger("PDController")


@dataclass
class ControlState:
    """Controller internal state"""
    last_error: Optional[float] = None
    last_timestamp: Optional[datetime] = None


class PDController:
    """
    Proportional-Derivative Controller
    
    From paper:
    Δx(t) = K_p · e(t) + K_d · (de(t)/dt)
    
    where:
    - e(t) = error between predicted and actual
    - K_p = proportional gain
    - K_d = derivative gain
    """
    
    def __init__(
        self,
        kp: float = 2.0,
        kd: float = 150.0,
        output_min: float = -3.0,
        output_max: float = 3.0
    ):
        """
        Initialize PD controller
        
        Args:
            kp: Proportional gain (default: 5.0)
            kd: Derivative gain (default: 300.0)
            output_min: Minimum output (replica delta)
            output_max: Maximum output (replica delta)
        """
        self.kp = kp
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        self.state = ControlState()
        
        logger.info(f"PD Controller initialized: Kp={kp}, Kd={kd}, "
                   f"output=[{output_min}, {output_max}]")
    
    def compute(
        self,
        predicted_traffic: float,
        actual_traffic: float,
        timestamp: Optional[datetime] = None
    ) -> int:
        """
        Compute control output (replica adjustment)
        
        Args:
            predicted_traffic: Predicted traffic (req/s)
            actual_traffic: Actual observed traffic (req/s)
            timestamp: Current timestamp (default: now)
        
        Returns:
            Replica delta (int) to apply
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate error (normalized)
        if predicted_traffic == 0:
            error = 0.0
        else:
            error = (actual_traffic - predicted_traffic) / predicted_traffic
        
        # Proportional term
        p_term = self.kp * error
        
        # Derivative term
        d_term = 0.0
        if self.state.last_error is not None and self.state.last_timestamp is not None:
            dt = (timestamp - self.state.last_timestamp).total_seconds()
            if dt > 0:
                d_error = error - self.state.last_error
                d_term = self.kd * (d_error / dt)
        
        # Total output
        output = p_term + d_term
        
        # Apply limits (anti-windup)
        output = max(self.output_min, min(self.output_max, output))
        
        # Round to integer replicas
        replica_delta = int(round(output))
        
        logger.info(f"PD control: error={error:.3f}, "
                   f"P={p_term:.2f}, D={d_term:.2f}, "
                   f"output={output:.2f}, delta={replica_delta}")
        
        # Update state
        self.state.last_error = error
        self.state.last_timestamp = timestamp
        
        return replica_delta
    
    def reset(self):
        """Reset controller state"""
        self.state = ControlState()
        logger.info("Controller state reset")