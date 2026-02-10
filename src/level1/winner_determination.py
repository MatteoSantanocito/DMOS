"""
Winner determination algorithm
Greedy allocation based on scores
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from ..utils.logger import get_logger

logger = get_logger("WinnerDetermination")


@dataclass
class ClusterBid:
    """
    Bid from a cluster
    """
    cluster_name: str
    score: float
    capacity: int  # Max replicas this cluster can handle
    
    def __repr__(self):
        return f"Bid({self.cluster_name}, score={self.score:.3f}, cap={self.capacity})"


@dataclass
class Allocation:
    """
    Allocation result for a cluster
    """
    cluster_name: str
    replicas: int      # Number of replicas allocated
    quota: float       # Quota as fraction of total demand
    score: float       # Score that won
    
    def __repr__(self):
        return f"Allocation({self.cluster_name}, replicas={self.replicas}, quota={self.quota:.2%}, score={self.score:.3f})"


class WinnerDetermination:
    """
    Greedy winner determination algorithm
    
    From paper Algorithm 1: Greedy Allocation by Score
    
    Complexity: O(N log N) dominated by sorting
    """
    
    def __init__(self):
        """Initialize winner determination"""
        pass
    
    def allocate(self, bids: List[ClusterBid], total_replicas: int) -> Tuple[List[Allocation], bool]:
        """
        Allocate replicas PROPORTIONALLY across clusters by score
        """
        if not bids or total_replicas <= 0:
            return [], False
        
        logger.info(f"Allocating {total_replicas} replicas among {len(bids)} clusters")
        
        # Sort by score descending
        sorted_bids = sorted(bids, key=lambda b: b.score, reverse=True)
        
        # Total score
        total_score = sum(bid.score for bid in sorted_bids)
        
        if total_score == 0:
            logger.error("Total score is zero")
            return [], False
        
        allocations = []
        remaining = total_replicas
        allocated_so_far = 0
        
        # Proportional allocation
        for i, bid in enumerate(sorted_bids):
            if i == len(sorted_bids) - 1:
                # Ultimo cluster: tutto il rimanente
                allocated = remaining
            else:
                # Proporzionale allo score
                quota = bid.score / total_score
                allocated = int(round(total_replicas * quota))
                
                # Almeno 1 se c'è capacità
                if allocated == 0 and remaining > 0 and bid.capacity > 0:
                    allocated = 1
            
            # Limita a capacità e rimanenti
            allocated = min(allocated, bid.capacity, remaining)
            
            if allocated > 0:
                allocations.append(Allocation(
                    cluster_name=bid.cluster_name,
                    replicas=allocated,
                    quota=allocated / total_replicas,
                    score=bid.score
                ))
                remaining -= allocated
                allocated_so_far += allocated
                
                logger.info(f"  Allocated {allocated} replicas to {bid.cluster_name} "
                        f"(quota={allocated/total_replicas*100:.1f}%, score={bid.score:.3f})")
        
        success = remaining == 0
        
        if success:
            logger.info(f"✅ Successfully allocated all {total_replicas} replicas across {len(allocations)} clusters")
        else:
            logger.warning(f"⚠️  Could not allocate {remaining} replicas")
        
        return allocations, success

    def allocate_with_vickrey_pricing(
        self,
        bids: List[ClusterBid],
        demand: int
    ) -> Tuple[List[Allocation], Dict[str, float], bool]:
        """
        Allocate with Vickrey (second-price) auction pricing
        
        Each winner pays the price of the next-best alternative
        (Not strictly needed for replica allocation, but included for completeness)
        
        Args:
            bids: List of cluster bids
            demand: Total replicas needed
        
        Returns:
            Tuple of (allocations, prices, success)
            - allocations: List of Allocation objects
            - prices: Dict mapping cluster_name to "price paid" (second-price score)
            - success: True if demand satisfied
        """
        allocations, success = self.allocate(bids, demand)
        
        if not allocations:
            return [], {}, False
        
        # Vickrey pricing: winner pays second-highest price
        sorted_bids = sorted(bids, key=lambda b: b.score, reverse=True)
        
        prices = {}
        for i, allocation in enumerate(allocations):
            # Find the "price" (next best score)
            winner_index = next(
                idx for idx, bid in enumerate(sorted_bids) 
                if bid.cluster_name == allocation.cluster_name
            )
            
            # Price is the next-best score (or 0 if last)
            if winner_index + 1 < len(sorted_bids):
                second_price = sorted_bids[winner_index + 1].score
            else:
                second_price = 0.0
            
            prices[allocation.cluster_name] = second_price
            
            logger.debug(f"Vickrey price for {allocation.cluster_name}: "
                        f"{second_price:.3f} (won with {allocation.score:.3f})")
        
        return allocations, prices, success
    
    def compute_fairness_jain_index(self, allocations: List[Allocation]) -> float:
        """
        Compute Jain's fairness index for allocation
        
        From paper:
        J(X) = (Σ U_i)² / (N * Σ U_i²)
        
        where U_i is utilization fraction
        
        Args:
            allocations: List of allocations
        
        Returns:
            Jain index in [1/N, 1] where 1 is perfect fairness
        """
        if not allocations:
            return 0.0
        
        # For replica allocation, use quotas as utilization
        quotas = [alloc.quota for alloc in allocations]
        
        n = len(quotas)
        sum_u = sum(quotas)
        sum_u_squared = sum(q**2 for q in quotas)
        
        if sum_u_squared == 0:
            return 0.0
        
        jain_index = (sum_u ** 2) / (n * sum_u_squared)
        
        logger.debug(f"Jain index: {jain_index:.3f} (1.0 = perfect fairness)")
        
        return jain_index