"""
Replica Scaler
Combines prediction + PD control + constraints to determine replica count
"""

from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger
from .predictor import TrafficPredictor
from .pd_controller import PDController

logger = get_logger("ReplicaScaler")


@dataclass
class ScalingDecision:
    """Result of scaling computation"""
    current_replicas: int
    target_replicas: int
    delta_replicas: int
    predicted_traffic: float
    base_replicas: float
    safety_replicas: int
    pd_adjustment: int
    constrained: bool
    metadata: Dict


class ReplicaScaler:
    """
    Replica scaler combining prediction + PD control
    
    Algorithm:
    1. Predict future traffic (trend-based)
    2. Calculate base replicas = predicted / capacity_per_replica
    3. Apply safety margin (15%)
    4. Apply PD correction
    5. Apply constraints (min, max, rate limit)
    """
    
    def __init__(
        self,
        capacity_per_replica: float,
        min_replicas: int = 1,
        max_replicas: int = 20,
        safety_margin: float = 0.10,
        max_delta_per_cycle: int = 3,
        predictor: Optional[TrafficPredictor] = None,
        controller: Optional[PDController] = None
    ):
        """
        Initialize replica scaler
        
        Args:
            capacity_per_replica: Request/s handled by one replica
            min_replicas: Minimum replicas (HA)
            max_replicas: Maximum replicas
            safety_margin: Safety buffer (default: 15%)
            max_delta_per_cycle: Max replica change per cycle
            predictor: Traffic predictor (creates default if None)
            controller: PD controller (creates default if None)
        """
        self.capacity_per_replica = capacity_per_replica
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.safety_margin = safety_margin
        self.max_delta_per_cycle = max_delta_per_cycle
        
        # Initialize predictor and controller
        self.predictor = predictor or TrafficPredictor()
        self.controller = controller or PDController()
        
        logger.info(f"Replica scaler inizializzato: "
                   f"capacità={capacity_per_replica} req/s, "
                   f"repliche=[{min_replicas}, {max_replicas}], "
                   f"margine di sicurezza={safety_margin*100}%, "
                   f"max_delta={max_delta_per_cycle}")
    
    def compute_target_replicas(
        self,
        current_replicas: int,
        current_traffic: float,
        actual_traffic: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> ScalingDecision:
        """
        Compute target replica count
        
        Args:
            current_replicas: Current number of replicas
            current_traffic: Current observed traffic (req/s)
            actual_traffic: Actual traffic for PD error (if None, uses current)
            timestamp: Current timestamp
        
        Returns:
            ScalingDecision with target replicas and metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if actual_traffic is None:
            actual_traffic = current_traffic
        
        # Step 1: Predict future traffic
        predicted_traffic, pred_metadata = self.predictor.predict(
            current_rate=current_traffic,
            timestamp=timestamp
        )
        
        # Step 2: Calculate base replicas
        base_replicas_raw = predicted_traffic / self.capacity_per_replica
        
        # Step 3: Apply safety margin
        base_replicas_safe = base_replicas_raw * (1 + self.safety_margin)
        base_replicas = int(max(1, round(base_replicas_safe)))
        
        # Step 4: PD adjustment
        pd_adjustment = self.controller.compute(
            predicted_traffic=predicted_traffic,
            actual_traffic=actual_traffic,
            timestamp=timestamp
        )
        
        # Step 5: Combine
        target_replicas_raw = base_replicas + pd_adjustment
        
        # Step 6: Apply constraints
        # Min/max bounds
        target_replicas = max(self.min_replicas, 
                             min(self.max_replicas, target_replicas_raw))
        
        # Rate limiting (max change per cycle)
        delta = target_replicas - current_replicas
        if abs(delta) > self.max_delta_per_cycle:
            if delta > 0:
                target_replicas = current_replicas + self.max_delta_per_cycle
            else:
                target_replicas = current_replicas - self.max_delta_per_cycle
        
        final_delta = target_replicas - current_replicas
        constrained = (target_replicas != target_replicas_raw)
        
        logger.info(f"Decisione di scaling: {current_replicas} → {target_replicas} "
                   f"(Δ={final_delta:+d}) | "
                   f"traffico: corrente={current_traffic:.1f}, "
                   f"predetto={predicted_traffic:.1f} | "
                   f"base={base_replicas}, PD={pd_adjustment:+d}, "
                   f"vincolato={constrained}")
        
        return ScalingDecision(
            current_replicas=current_replicas,
            target_replicas=target_replicas,
            delta_replicas=final_delta,
            predicted_traffic=predicted_traffic,
            base_replicas=base_replicas_raw,
            safety_replicas=base_replicas,
            pd_adjustment=pd_adjustment,
            constrained=constrained,
            metadata={
                'prediction': pred_metadata,
                'current_traffic': current_traffic,
                'actual_traffic': actual_traffic,
                'capacity_per_replica': self.capacity_per_replica,
                'safety_margin': self.safety_margin,
                'timestamp': timestamp.isoformat()
            }
        )