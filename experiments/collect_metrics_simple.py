"""
Collector metriche semplificato
Salva snapshot DMOS metrics ogni 15s
"""

import requests
import time
from datetime import datetime
from pathlib import Path

METRICS_URL = "http://localhost:9090/metrics"
OUTPUT_DIR = Path("results")

def collect_snapshot():
    """Raccoglie snapshot metriche DMOS"""
    try:
        r = requests.get(METRICS_URL, timeout=5)
        return r.text
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_metrics(snapshot: str) -> dict:
    """Parse metriche chiave dallo snapshot"""
    metrics = {
        'traffic': None,
        'replicas_cluster1': 0,
        'replicas_cluster2': 0,
        'replicas_cluster3': 0,
        'replicas_total': 0
    }
    
    for line in snapshot.split('\n'):
        # Skip comments
        if line.startswith('#') or not line.strip():
            continue
        
        try:
            # Actual traffic
            if 'dmos_actual_traffic{service="frontend"}' in line:
                parts = line.split()
                if len(parts) >= 2:
                    metrics['traffic'] = float(parts[-1])
            
            # Current replicas per cluster
            elif 'dmos_current_replicas{cluster="cluster1"' in line:
                parts = line.split()
                if len(parts) >= 2:
                    metrics['replicas_cluster1'] = int(float(parts[-1]))
            
            elif 'dmos_current_replicas{cluster="cluster2"' in line:
                parts = line.split()
                if len(parts) >= 2:
                    metrics['replicas_cluster2'] = int(float(parts[-1]))
            
            elif 'dmos_current_replicas{cluster="cluster3"' in line:
                parts = line.split()
                if len(parts) >= 2:
                    metrics['replicas_cluster3'] = int(float(parts[-1]))
        
        except (ValueError, IndexError):
            continue
    
    # Calcola totale
    metrics['replicas_total'] = (
        metrics['replicas_cluster1'] + 
        metrics['replicas_cluster2'] + 
        metrics['replicas_cluster3']
    )
    
    return metrics

def run_collector(duration_minutes: int = 20):
    """
    Raccoglie metriche per durata specificata
    
    Args:
        duration_minutes: Durata raccolta in minuti (default: 20)
    """
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"metrics_timeseries_{test_id}.txt"
    
    print("="*70)
    print(f"DMOS Metrics Collector")
    print(f"Duration: {duration_minutes} minutes")
    print(f"Interval: 15 seconds")
    print(f"Output: {output_file}")
    print("="*70)
    print("\nCollecting snapshots...\n")
    
    iterations = (duration_minutes * 60) // 15
    
    with open(output_file, 'w') as f:
        for i in range(iterations):
            timestamp = datetime.now().isoformat()
            
            print(f"[{i+1}/{iterations}] {timestamp}", end=" ")
            
            snapshot = collect_snapshot()
            
            if snapshot:
                # Scrivi timestamp + snapshot
                f.write(f"\n{'='*70}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"{'='*70}\n")
                f.write(snapshot)
                f.write("\n")
                f.flush()
                
                # Parse metriche
                metrics = parse_metrics(snapshot)
                
                # Stampa summary
                if metrics['traffic'] is not None and metrics['traffic'] > 0:
                    print(f"Traffic: {metrics['traffic']:6.1f} req/s "
                          f"Replicas: {metrics['replicas_total']} "
                          f"(C1:{metrics['replicas_cluster1']} "
                          f"C2:{metrics['replicas_cluster2']} "
                          f"C3:{metrics['replicas_cluster3']}) ✓")
                else:
                    print("✓")
            else:
                print("✗")
            
            if i < iterations - 1:
                time.sleep(15)
    
    print("\n" + "="*70)
    print(f"Collection complete!")
    print(f"Saved to: {output_file}")
    print("="*70)
    
    return output_file

if __name__ == "__main__":
    import sys
    
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    
    print("\n🚀 DMOS Metrics Collector")
    print(f"   Duration: {duration} minutes")
    print(f"   Interval: 15 seconds")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    run_collector(duration_minutes=duration)