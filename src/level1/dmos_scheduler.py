"""
DMOS Scheduler (Centralized)
Queries cluster estimators, performs winner determination, sends scaling decisions

Like Karmada Scheduler but with multi-objective optimization
"""

import requests
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.logger import get_logger
from ..utils.config_loader import ConfigLoader
from .winner_determination import WinnerDetermination, ClusterBid, Allocation

logger = get_logger("DMOSScheduler")


class DMOSScheduler:
    """
    DMOS Centralized Scheduler
    
    Architecture (Karmada-like):
    1. Central scheduler queries cluster estimators
    2. Collects scores in parallel (via threads)
    3. Performs winner determination (greedy allocation)
    4. Returns scheduling decisions
    
    Note: We use threading for parallel queries, but collection is still
    centralized (not truly distributed like p2p auction)
    """
    
    def __init__(self, config_path: str = "config"):
        """
        Initialize DMOS scheduler
        
        Args:
            config_path: Path to config directory
        """
        self.config = ConfigLoader(config_path)
        
        # Winner determination algorithm
        self.winner_det = WinnerDetermination()
        
        # Cluster estimator endpoints
        self.estimator_endpoints = self._build_estimator_endpoints()
        
        # Query timeout
        self.query_timeout = self.config.global_config.get(
            'score_request_timeout_seconds', 5
        )
        
        logger.info(f"DMOS Scheduler initialized with {len(self.estimator_endpoints)} clusters")
    
    def _build_estimator_endpoints(self) -> Dict[str, str]:
        """
        Build estimator endpoint URLs for each cluster
        
        Returns:
            Dict mapping cluster_name to estimator URL
        """
        endpoints = {}
        estimator_port = self.config.global_config.get('score_agent_port', 8080)
        
        for cluster_name, cluster_config in self.config.clusters.items():
            url = f"http://{cluster_config.ip}:{estimator_port}"
            endpoints[cluster_name] = url
            logger.debug(f"Cluster {cluster_name} estimator: {url}")
        
        return endpoints
    
    def _query_estimator(
        self, 
        cluster_name: str, 
        service_name: str,
        predicted_load: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Query a single cluster estimator
        
        Args:
            cluster_name: Cluster to query
            service_name: Service name
            predicted_load: Optional predicted load
        
        Returns:
            Estimator response dict or None if failed
        """
        url = self.estimator_endpoints.get(cluster_name)
        if not url:
            logger.error(f"No estimator endpoint for cluster {cluster_name}")
            return None
        
        endpoint = f"{url}/estimate"
        payload = {'service_name': service_name}
        if predicted_load is not None:
            payload['predicted_load'] = predicted_load
        
        try:
            start_time = time.time()
            response = requests.post(
                endpoint,
                json=payload,
                timeout=self.query_timeout
            )
            elapsed = (time.time() - start_time) * 1000  # ms
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Query {cluster_name}: score={data['score']:.3f}, "
                       f"capacity={data['capacity']}, latency={elapsed:.0f}ms")
            
            return data
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout querying {cluster_name} (>{self.query_timeout}s)")
            return None
        except Exception as e:
            logger.error(f"Error querying {cluster_name}: {e}")
            return None
    
    def collect_scores(
        self, 
        service_name: str,
        predicted_load: Optional[float] = None
    ) -> List[ClusterBid]:
        """
        Collect scores from all cluster estimators in PARALLEL
        
        Args:
            service_name: Service to schedule
            predicted_load: Optional predicted load
        
        Returns:
            List of ClusterBid objects
        """
        logger.info(f"Collecting scores for service '{service_name}' from all clusters...")
        
        bids = []
        
        # Use ThreadPoolExecutor for parallel queries
        with ThreadPoolExecutor(max_workers=len(self.estimator_endpoints)) as executor:
            # Submit all queries
            future_to_cluster = {
                executor.submit(
                    self._query_estimator, 
                    cluster_name, 
                    service_name, 
                    predicted_load
                ): cluster_name
                for cluster_name in self.estimator_endpoints.keys()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_cluster):
                cluster_name = future_to_cluster[future]
                
                try:
                    result = future.result()
                    
                    if result:
                        bid = ClusterBid(
                            cluster_name=result['cluster_name'],
                            score=result['score'],
                            capacity=result['capacity']
                        )
                        bids.append(bid)
                        logger.debug(f"Received bid from {cluster_name}: {bid}")
                    else:
                        logger.warning(f"No response from {cluster_name}")
                        
                except Exception as e:
                    logger.error(f"Exception processing {cluster_name}: {e}")
        
        logger.info(f"Collected {len(bids)}/{len(self.estimator_endpoints)} bids")
        
        return bids
    
    def schedule_service(
        self, 
        service_name: str, 
        total_replicas: int,
        predicted_load: Optional[float] = None
    ) -> Tuple[List[Allocation], bool]:
        """
        Schedule a service across clusters
        
        Args:
            service_name: Service to schedule
            total_replicas: Total number of replicas needed
            predicted_load: Optional predicted load
        
        Returns:
            Tuple of (allocations, success)
        """
        logger.info(f"Scheduling service '{service_name}' with {total_replicas} replicas")
        
        # Step 1: Collect scores from estimators
        start_time = time.time()
        bids = self.collect_scores(service_name, predicted_load)
        collection_time = (time.time() - start_time) * 1000
        
        if not bids:
            logger.error("No bids received from any cluster!")
            return [], False
        
        # Step 2: Winner determination
        allocations, success = self.winner_det.allocate(bids, total_replicas)
        
        # Step 3: Log results
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Scheduling completed in {total_time:.0f}ms "
                   f"(collection={collection_time:.0f}ms, "
                   f"winner_det={(total_time-collection_time):.0f}ms)")
        
        if success:
            logger.info(f"✅ Allocated {total_replicas} replicas across {len(allocations)} clusters:")
            for alloc in allocations:
                logger.info(f"   {alloc}")
            
            # Compute fairness
            jain = self.winner_det.compute_fairness_jain_index(allocations)
            logger.info(f"   Jain fairness index: {jain:.3f}")
        else:
            logger.error(f"❌ Failed to satisfy demand ({total_replicas} replicas)")
        
        return allocations, success