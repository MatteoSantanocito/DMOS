# Terminal 2: Reset TOTALE (Cluster 1, 2, 3)
from src.k8s.client import KubernetesClient
import yaml

# Carica config corretta
with open('config/clusters.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Crea client mappando i dati dal yaml
k8s_config = {}
for name, data in config['clusters'].items():
    k8s_config[name] = {
        'kubeconfig_path': data['kubeconfig_path'],
        'server': data['server'] # Questo userà i nuovi IP corretti!
    }

print('🔌 Connessione ai cluster...')
k8s = KubernetesClient(k8s_config)

# Resetta TUTTI i cluster a 1 replica
for cluster in ['cluster1', 'cluster2', 'cluster3']:
    print(f'🧹 Resetting {cluster}...')
    k8s.scale_deployment(cluster, 'frontend', 1, 'online-boutique')

print('✅ Reset Multi-Cluster Completato!')
