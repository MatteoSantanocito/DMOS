"""
Kubernetes Client for multi-cluster operations
"""

import subprocess
from typing import Dict, Optional, List
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import tempfile
import yaml

from ..utils.logger import get_logger

logger = get_logger("K8sClient")


class KubernetesClient:
    """
    Multi-cluster Kubernetes client with kubeconfig context handling
    """
    
    def __init__(self, cluster_configs: Dict[str, Dict[str, str]]):
        """
        Initialize K8s client
        
        Args:
            cluster_configs: Dict mapping cluster_name -> {
                'kubeconfig_path': '/path/to/kubeconfig',
                'server': 'https://192.168.1.245:6443'
            }
        """
        self.cluster_configs = cluster_configs
        self.api_clients: Dict[str, client.AppsV1Api] = {}
        
        for cluster_name, cfg in cluster_configs.items():
            try:
                self._init_cluster_client(
                    cluster_name, 
                    cfg['kubeconfig_path'],
                    cfg.get('server')
                )
                logger.info(f"✅ Initialized K8s client for {cluster_name}")
            except Exception as e:
                logger.error(f"❌ Failed to init {cluster_name}: {e}")
    
    def _fix_kubeconfig(
        self, 
        kubeconfig_path: str, 
        cluster_name: str,
        server_url: Optional[str] = None
    ) -> str:
        """
        Fix kubeconfig: unique context + remote server URL
        
        Args:
            kubeconfig_path: Original kubeconfig
            cluster_name: Unique cluster name
            server_url: Optional server URL (e.g., https://192.168.1.245:6443)
        
        Returns:
            Path to fixed temp kubeconfig
        """
        with open(kubeconfig_path, 'r') as f:
            kc = yaml.safe_load(f)
        
        # Fix context name
        if kc.get('contexts'):
            for ctx in kc['contexts']:
                ctx['name'] = cluster_name
                ctx['context']['cluster'] = cluster_name
                ctx['context']['user'] = cluster_name
        
        # Fix cluster name + server URL
        if kc.get('clusters'):
            for clust in kc['clusters']:
                clust['name'] = cluster_name
                if server_url:
                    clust['cluster']['server'] = server_url
        
        # Fix user name
        if kc.get('users'):
            for usr in kc['users']:
                usr['name'] = cluster_name
        
        kc['current-context'] = cluster_name
        
        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'-{cluster_name}.yaml',
            delete=False
        )
        yaml.dump(kc, temp_file)
        temp_file.flush()
        
        return temp_file.name
    
    def _init_cluster_client(
        self, 
        cluster_name: str, 
        kubeconfig_path: str,
        server_url: Optional[str] = None
    ):
        """
        Initialize K8s API client for a cluster
        
        Args:
            cluster_name: Cluster name
            kubeconfig_path: Path to kubeconfig
            server_url: Optional server URL override
        """
        import os
        kubeconfig_path = os.path.expanduser(kubeconfig_path)
        fixed_kubeconfig = self._fix_kubeconfig(
            kubeconfig_path, 
            cluster_name,
            server_url
        )
        
        k8s_client = config.new_client_from_config(
            config_file=fixed_kubeconfig,
            context=cluster_name
        )
        
        apps_api = client.AppsV1Api(api_client=k8s_client)
        self.api_clients[cluster_name] = apps_api
    
    def scale_deployment(
        self,
        cluster: str,
        deployment: str,
        replicas: int,
        namespace: str = "online-boutique"
    ) -> bool:
        """
        Scale a deployment
        
        Args:
            cluster: Cluster name
            deployment: Deployment name
            replicas: Target replicas
            namespace: Namespace
        
        Returns:
            True if successful
        """
        if cluster not in self.api_clients:
            logger.error(f"Cluster {cluster} not configured")
            return False
        
        apps_api = self.api_clients[cluster]
        
        try:
            body = {'spec': {'replicas': replicas}}
            
            apps_api.patch_namespaced_deployment_scale(
                name=deployment,
                namespace=namespace,
                body=body
            )
            
            logger.info(f"✅ Scaled {cluster}/{deployment} → {replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"K8s API error: {e.status} {e.reason}")
            return False
        except Exception as e:
            logger.error(f"Error scaling: {e}")
            return False
    
    def get_deployment_replicas(
        self,
        cluster: str,
        deployment: str,
        namespace: str = "online-boutique"
    ) -> Optional[int]:
        """
        Get current replica count
        
        Args:
            cluster: Cluster name
            deployment: Deployment name
            namespace: Namespace
        
        Returns:
            Replica count or None
        """
        if cluster not in self.api_clients:
            return None
        
        apps_api = self.api_clients[cluster]
        
        try:
            dep = apps_api.read_namespaced_deployment(
                name=deployment,
                namespace=namespace
            )
            return dep.spec.replicas
            
        except Exception as e:
            logger.error(f"Error reading replicas: {e}")
            return None