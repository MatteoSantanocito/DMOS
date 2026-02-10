#!/usr/bin/env python3
"""
Test connessione Prometheus e query base
"""

import requests
import json
from datetime import datetime

PROMETHEUS_URL = "http://192.168.1.245:30090"

def test_prometheus():
    print("=== Test Prometheus ===\n")
    
    # 1. Health check
    print("1. Health Check...")
    try:
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=5)
        if r.status_code == 200:
            print("   ✅ Prometheus is healthy\n")
        else:
            print(f"   ❌ HTTP {r.status_code}\n")
            return
    except Exception as e:
        print(f"   ❌ Connection failed: {e}\n")
        return
    
    # 2. Test query: CPU capacity
    print("2. Query CPU Capacity...")
    query = 'sum(kube_node_status_capacity{resource="cpu"})'
    params = {'query': query}
    
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params=params, timeout=5)
        data = r.json()
        
        if data['status'] == 'success':
            result = data['data']['result']
            if result:
                cpu_total = float(result[0]['value'][1])
                print(f"   ✅ Total CPU Capacity: {cpu_total} cores\n")
            else:
                print("   ⚠️  No data returned (kube-state-metrics missing?)\n")
        else:
            print(f"   ❌ Query failed: {data}\n")
    except Exception as e:
        print(f"   ❌ Query error: {e}\n")
    
    # 3. Test query: Memory capacity
    print("3. Query Memory Capacity...")
    query = 'sum(kube_node_status_capacity{resource="memory"})'
    params = {'query': query}
    
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params=params, timeout=5)
        data = r.json()
        
        if data['status'] == 'success':
            result = data['data']['result']
            if result:
                mem_bytes = float(result[0]['value'][1])
                mem_gb = mem_bytes / (1024**3)
                print(f"   ✅ Total Memory: {mem_gb:.2f} GB\n")
            else:
                print("   ⚠️  No data returned\n")
        else:
            print(f"   ❌ Query failed\n")
    except Exception as e:
        print(f"   ❌ Query error: {e}\n")
    
    # 4. Test query: Pods running
    print("4. Query Running Pods...")
    query = 'count(kube_pod_info{namespace="online-boutique"})'
    params = {'query': query}
    
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params=params, timeout=5)
        data = r.json()
        
        if data['status'] == 'success':
            result = data['data']['result']
            if result:
                pod_count = int(float(result[0]['value'][1]))
                print(f"   ✅ Online Boutique Pods: {pod_count}\n")
            else:
                print("   ⚠️  No pods found in online-boutique namespace\n")
        else:
            print(f"   ❌ Query failed\n")
    except Exception as e:
        print(f"   ❌ Query error: {e}\n")

if __name__ == "__main__":
    test_prometheus()