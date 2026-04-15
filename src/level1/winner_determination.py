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
        Alloca le repliche PROPORZIONALMENTE tra i cluster in base allo score.

        Algoritmo: Hamilton method (Largest Remainder Method) con multi-pass
        per gestire i vincoli di capacità.

        Fase 1 — multi-pass capacity-aware:
          Calcola la quota ideale proporzionale per ogni cluster attivo.
          Se un cluster supera la propria capacità residua, viene cappato
          e rimosso dagli attivi. La domanda eccedente viene redistribuita
          proporzionalmente ai cluster rimanenti (loop fino a stabilità).

        Fase 2 — Largest Remainder Method (Hamilton):
          Quando nessun cluster è over-capacity, usa floor() per le quote
          intere e assegna le repliche rimanenti (rimanze da arrotondamento)
          una alla volta ai cluster con la parte frazionaria più alta.

        Proprietà:
          - Ogni cluster riceve almeno floor(quota_proporzionale) repliche
          - Ogni cluster riceve al massimo floor(quota_proporzionale) + 1
          - Le repliche extra vanno ai cluster con la frazione più alta
            (= quelli che hanno "più credito" proporzionale non ancora soddisfatto)
          - La concentrazione su un singolo cluster avviene SOLO quando gli
            altri sono effettivamente a piena capacità (non per arrotondamento)
        """
        if not bids or total_replicas <= 0:
            return [], False

        logger.info(f"Allocating {total_replicas} replicas among {len(bids)} clusters")

        # Sort by score descending (log + tie-breaking LRM)
        sorted_bids = sorted(bids, key=lambda b: b.score, reverse=True)

        total_score = sum(bid.score for bid in sorted_bids)
        if total_score == 0:
            logger.error("Total score is zero")
            return [], False

        # ── Multi-pass: redistribuisci la domanda quando un cluster è over-capacity ──
        assigned: Dict[str, int] = {bid.cluster_name: 0 for bid in sorted_bids}
        capped: set = set()
        active = list(sorted_bids)
        demand = total_replicas

        while demand > 0 and active:
            active_score = sum(bid.score for bid in active)
            if active_score == 0:
                break

            # Quota ideale (reale, non intera) per ogni cluster attivo
            ideal: Dict[str, float] = {
                bid.cluster_name: demand * bid.score / active_score
                for bid in active
            }

            # Cluster la cui quota ideale supera la capacità residua
            over_cap = [
                bid for bid in active
                if ideal[bid.cluster_name] > (bid.capacity - assigned[bid.cluster_name]) + 1e-9
            ]

            if over_cap:
                # Cappa i cluster over-capacity e ricalcola la domanda residua
                for bid in over_cap:
                    spare = bid.capacity - assigned[bid.cluster_name]
                    if spare > 0:
                        assigned[bid.cluster_name] = bid.capacity
                        demand -= spare
                        logger.info(
                            f"  Capped {bid.cluster_name} → {bid.capacity} replicas "
                            f"(ideal={ideal[bid.cluster_name]:.2f}, assigned={spare}, "
                            f"excess={ideal[bid.cluster_name]-spare:.2f} redistributed to remaining clusters)"
                        )
                    capped.add(bid.cluster_name)
                active = [b for b in active if b.cluster_name not in capped]

            else:
                # ── Largest Remainder Method (Hamilton): nessuno over-capacity ──
                # Fase 2a: floor allocation
                for bid in active:
                    floored = int(ideal[bid.cluster_name])
                    spare = bid.capacity - assigned[bid.cluster_name]
                    add = min(floored, spare)
                    assigned[bid.cluster_name] += add
                    demand -= add

                # Fase 2b: distribuisci le rimanze ai cluster con frazione più alta
                fractions = sorted(
                    active,
                    key=lambda b: ideal[b.cluster_name] - int(ideal[b.cluster_name]),
                    reverse=True
                )
                for bid in fractions:
                    if demand <= 0:
                        break
                    spare = bid.capacity - assigned[bid.cluster_name]
                    if spare > 0:
                        assigned[bid.cluster_name] += 1
                        demand -= 1
                        frac = ideal[bid.cluster_name] - int(ideal[bid.cluster_name])
                        logger.info(
                            f"  +1 LRM remainder (frac={frac:.3f}) → {bid.cluster_name}"
                        )

                # ── Geo-boost: garantisci che lo score ordering si rifletta ──
                # Se il cluster con score più alto ha le stesse repliche del
                # cluster con score più basso, trasferisci 1 replica dal peggiore
                # al migliore. Questo risolve il problema della quantizzazione
                # Hamilton con score quasi identici (es. 0.12/0.12/0.13 → 3/3/3).
                if len(active) >= 2:
                    best = active[0]   # già ordinati per score desc
                    worst = active[-1]
                    best_r = assigned[best.cluster_name]
                    worst_r = assigned[worst.cluster_name]
                    # Boost solo se: (1) stesse repliche, (2) score effettivamente diversi,
                    # (3) il peggiore mantiene almeno 1 replica dopo il trasferimento
                    if (best_r <= worst_r
                            and best.score > worst.score + 1e-9
                            and worst_r >= 2):
                        assigned[best.cluster_name] += 1
                        assigned[worst.cluster_name] -= 1
                        logger.info(
                            f"  Geo-boost: +1 {best.cluster_name} "
                            f"(score={best.score:.3f}), "
                            f"-1 {worst.cluster_name} "
                            f"(score={worst.score:.3f})"
                        )

                break  # allocazione completata

        # ── Costruisci risultato ──────────────────────────────────────────────
        alloc_map: Dict[str, Allocation] = {}
        for bid in sorted_bids:
            a = assigned.get(bid.cluster_name, 0)
            if a > 0:
                alloc_map[bid.cluster_name] = Allocation(
                    cluster_name=bid.cluster_name,
                    replicas=a,
                    quota=a / total_replicas,
                    score=bid.score
                )
                logger.info(
                    f"  Allocated {a} replicas to {bid.cluster_name} "
                    f"(quota={a/total_replicas*100:.1f}%, score={bid.score:.3f})"
                )

        allocations = list(alloc_map.values())
        success = demand == 0

        if success:
            logger.info(
                f"Successfully allocated all {total_replicas} replicas "
                f"across {len(allocations)} clusters"
            )
        else:
            logger.warning(
                f"  Could not allocate {demand} replicas "
                f"(clusters at full capacity)"
            )

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