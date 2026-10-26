import json
import os
import sys

import requests

# Cluster topology read from .cache/test-env.json (rendered from config/local.yaml).
_env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".cache", "test-env.json",
)
try:
    with open(_env_path, "r", encoding="utf-8") as _f:
        _env = json.load(_f)
except FileNotFoundError:
    sys.exit(
        f"{_env_path} not found. Run 'bash scripts/render-runtime-files.sh' first."
    )

clusters = {
    c["name"]: (
        c["prometheus_url"],
        [p["ip"] for p in _env["clusters"] if p["name"] != c["name"]],
    )
    for c in _env["clusters"]
}

def q(url, query):
    try:
        r = requests.get(f'{url}/api/v1/query', params={'query': query}, timeout=5)
        d = r.json()
        res = d['data']['result']
        return float(res[0]['value'][1]) if res else 0.0
    except:
        return 0.0

print('=== Network bytes fallback ===')
net_rates = {}
for name, (url, _) in clusters.items():
    b = q(url, 'sum(rate(container_network_receive_bytes_total{namespace="online-boutique", pod=~"frontend.*"}[1m]))')
    net_rates[name] = b / 4000
    print(f'  {name}: {b:.0f} B/s -> {b/4000:.2f} rps')

print()
print('=== Hubble HTTP totale ===')
hubble_rates = {}
for name, (url, _) in clusters.items():
    r = q(url, 'sum(rate(hubble_http_requests_total{destination_workload="frontend",destination_namespace="online-boutique"}[1m]))')
    hubble_rates[name] = r
    print(f'  {name}: {r:.2f} rps')

print()
print('=== Pod frontend ===')
for name, (url, _) in clusters.items():
    p = q(url, 'count(kube_pod_info{namespace="online-boutique", created_by_name=~"frontend-.*"})')
    print(f'  {name}: {int(p)} pods')

print()
print('=== RTT (Phi_net) ===')
rtts = {}
for name, (url, peers) in clusters.items():
    targets = '|'.join(peers)
    rtt = q(url, 'avg(ping_rtt_mean_seconds{target=~"' + targets + '"}) * 1000')
    rtts[name] = rtt
    print(f'  {name}: {rtt:.1f}ms')

phi_net = {n: 1.0/(1.0+rtts[n]/1000.0) for n in rtts}
tot = sum(phi_net.values())
phi_norm = {n: phi_net[n]/tot for n in phi_net}
print('  Phi_net norm:', {n: f'{v:.3f}' for n, v in phi_norm.items()})

print()
tot_net = sum(net_rates.values())
tot_hub = sum(hubble_rates.values())
print('=== Phi_demand ===')
for name in clusters:
    nd = net_rates[name]/tot_net if tot_net > 0 else 0
    hd = hubble_rates[name]/tot_hub if tot_hub > 0 else 0
    print(f'  {name}: network={nd:.3f}  hubble={hd:.3f}')

print()
print('=== Distribuzione proporzionale attesa solo Phi_net, N=10 ===')
for n, phi in phi_norm.items():
    print(f'  {n}: score={phi:.3f} -> {10*phi:.2f} repliche ideali')
