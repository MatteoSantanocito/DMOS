"""
Analisi Completa Test DMOS
Genera statistiche, grafici e report
"""

import re
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

def parse_metrics_file(filepath: str):
    """Parse metrics file - fixed chunking"""
    
    print(f"📂 Opening: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📄 Size: {len(content)} chars")
    
    # Pattern: trova blocchi timestamp + metriche
    # Usa regex per trovare pattern completo
    pattern = r'Timestamp: ([\d\-T:\.]+)\s*={70,}\s*(.*?)(?=Timestamp:|$)'
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"🕐 Found {len(matches)} complete snapshots")
    
    if not matches:
        print("❌ No snapshots found!")
        return []
    
    data = []
    
    for timestamp_str, metrics_block in matches:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except Exception as e:
            print(f"⚠️  Bad timestamp: {timestamp_str}")
            continue
        
        # Initialize metrics
        metrics = {
            'timestamp': timestamp,
            'traffic': 0.0,
            'replicas_cluster1': 0,
            'replicas_cluster2': 0,
            'replicas_cluster3': 0,
            'replicas_total': 0,
            'scale_up_events': 0,
            'scale_down_events': 0,
            'cluster1_score': 0.0,
            'cluster2_score': 0.0,
            'cluster3_score': 0.0,
            'predicted_traffic': 0.0
        }
        
        # Parse metrics block
        for line in metrics_block.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            try:
                # Actual traffic
                if 'dmos_actual_traffic{service="frontend"}' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['traffic'] = float(match.group(1))
                
                # Current replicas
                elif 'dmos_current_replicas{cluster="cluster1"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['replicas_cluster1'] = int(float(match.group(1)))
                
                elif 'dmos_current_replicas{cluster="cluster2"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['replicas_cluster2'] = int(float(match.group(1)))
                
                elif 'dmos_current_replicas{cluster="cluster3"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['replicas_cluster3'] = int(float(match.group(1)))
                
                # Scaling events (cumulative counters)
                elif 'dmos_scaling_events_total{action="scale_up"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['scale_up_events'] += int(float(match.group(1)))
                
                elif 'dmos_scaling_events_total{action="scale_down"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['scale_down_events'] += int(float(match.group(1)))
                
                # Cluster scores
                elif 'dmos_cluster_score{cluster="cluster1"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['cluster1_score'] = float(match.group(1))
                
                elif 'dmos_cluster_score{cluster="cluster2"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['cluster2_score'] = float(match.group(1))
                
                elif 'dmos_cluster_score{cluster="cluster3"' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['cluster3_score'] = float(match.group(1))
                
                # Predicted traffic (sum all clusters)
                elif 'dmos_predicted_traffic{cluster=' in line:
                    match = re.search(r'([\d\.]+)$', line)
                    if match:
                        metrics['predicted_traffic'] += float(match.group(1))
            
            except Exception:
                pass
        
        # Total replicas
        metrics['replicas_total'] = (
            metrics['replicas_cluster1'] + 
            metrics['replicas_cluster2'] + 
            metrics['replicas_cluster3']
        )
        
        data.append(metrics)
    
    print(f"✅ Parsed {len(data)} snapshots")
    
    if data:
        print("\n📊 First snapshot:")
        s = data[0]
        print(f"  Traffic: {s['traffic']:.2f} req/s")
        print(f"  Replicas: C1={s['replicas_cluster1']}, C2={s['replicas_cluster2']}, C3={s['replicas_cluster3']}")
        print(f"  Total events: up={s['scale_up_events']}, down={s['scale_down_events']}")
    
    return data

def print_statistics(data):
    """Stampa statistiche dettagliate"""
    
    print("\n" + "="*70)
    print("📊 DMOS TEST - STATISTICAL ANALYSIS")
    print("="*70)
    
    if not data:
        print("\n❌ No data found")
        return
    
    print(f"\nTest Duration:")
    print(f"  Start: {data[0]['timestamp']}")
    print(f"  End:   {data[-1]['timestamp']}")
    print(f"  Duration: {(data[-1]['timestamp'] - data[0]['timestamp']).total_seconds() / 60:.1f} minutes")
    print(f"  Snapshots: {len(data)} (every ~15s)")
    
    # Traffic statistics
    print("\n" + "="*70)
    print("1. TRAFFIC ANALYSIS")
    print("="*70)
    
    traffic_values = [d['traffic'] for d in data if d['traffic'] > 0]
    
    if traffic_values:
        print(f"\n  Samples with traffic: {len(traffic_values)}/{len(data)}")
        print(f"  Min:    {min(traffic_values):6.2f} req/s")
        print(f"  Max:    {max(traffic_values):6.2f} req/s")
        print(f"  Mean:   {sum(traffic_values)/len(traffic_values):6.2f} req/s")
        print(f"  Median: {sorted(traffic_values)[len(traffic_values)//2]:6.2f} req/s")
        
        # Percentiles
        sorted_traffic = sorted(traffic_values)
        p50 = sorted_traffic[len(sorted_traffic)//2]
        p95 = sorted_traffic[int(len(sorted_traffic)*0.95)]
        p99 = sorted_traffic[int(len(sorted_traffic)*0.99)]
        
        print(f"\n  Percentiles:")
        print(f"    p50: {p50:.2f} req/s")
        print(f"    p95: {p95:.2f} req/s")
        print(f"    p99: {p99:.2f} req/s")
    else:
        print("\n  ⚠️  No traffic data captured")
    
    # Replica statistics
    print("\n" + "="*70)
    print("2. REPLICA DISTRIBUTION")
    print("="*70)
    
    replicas_total = [d['replicas_total'] for d in data]
    
    if any(r > 0 for r in replicas_total):
        print(f"\n  Total Replicas:")
        print(f"    Min:  {min(replicas_total)}")
        print(f"    Max:  {max(replicas_total)}")
        print(f"    Mean: {sum(replicas_total)/len(replicas_total):.1f}")
        
        # Per cluster breakdown
        c1_reps = [d['replicas_cluster1'] for d in data if d['replicas_cluster1'] > 0]
        c2_reps = [d['replicas_cluster2'] for d in data if d['replicas_cluster2'] > 0]
        c3_reps = [d['replicas_cluster3'] for d in data if d['replicas_cluster3'] > 0]
        
        print(f"\n  Cluster Distribution:")
        if c1_reps:
            print(f"    Cluster1: avg={sum(c1_reps)/len(c1_reps):.1f}, max={max(c1_reps)}")
        if c2_reps:
            print(f"    Cluster2: avg={sum(c2_reps)/len(c2_reps):.1f}, max={max(c2_reps)}")
        if c3_reps:
            print(f"    Cluster3: avg={sum(c3_reps)/len(c3_reps):.1f}, max={max(c3_reps)}")
        
        if not c2_reps and not c3_reps:
            print(f"    ⚠️  Only Cluster1 used (winner-takes-all)")
    
    # Scaling activity
    print("\n" + "="*70)
    print("3. SCALING ACTIVITY")
    print("="*70)
    
    if data:
        total_scale_up = data[-1]['scale_up_events'] - data[0]['scale_up_events']
        total_scale_down = data[-1]['scale_down_events'] - data[0]['scale_down_events']
        total_events = total_scale_up + total_scale_down
        
        print(f"\n  Scale-up events:   {total_scale_up}")
        print(f"  Scale-down events: {total_scale_down}")
        print(f"  Total events:      {total_events}")
        
        if total_events > 0:
            # Find scaling transitions
            transitions = []
            for i in range(1, len(data)):
                prev = data[i-1]['replicas_total']
                curr = data[i]['replicas_total']
                if prev != curr:
                    transitions.append({
                        'time': data[i]['timestamp'],
                        'from': prev,
                        'to': curr,
                        'delta': curr - prev,
                        'traffic': data[i]['traffic']
                    })
            
            if transitions:
                print(f"\n  Scaling Transitions ({len(transitions)}):")
                for t in transitions[:10]:  # Show first 10
                    direction = "↗️ UP" if t['delta'] > 0 else "↘️ DOWN"
                    print(f"    {t['time'].strftime('%H:%M:%S')} | {t['from']} → {t['to']} ({t['delta']:+d}) {direction} | Traffic: {t['traffic']:.1f} req/s")
                
                if len(transitions) > 10:
                    print(f"    ... and {len(transitions) - 10} more")
    
    # Prediction accuracy
    print("\n" + "="*70)
    print("4. PREDICTION ACCURACY")
    print("="*70)
    
    predictions = [(d['traffic'], d['predicted_traffic']) 
                   for d in data 
                   if d['traffic'] > 0 and d['predicted_traffic'] > 0]
    
    if predictions:
        errors = []
        for actual, predicted in predictions:
            if actual > 0:
                error = abs((predicted - actual) / actual) * 100
                errors.append(error)
        
        if errors:
            mape = sum(errors) / len(errors)
            print(f"\n  Prediction samples: {len(errors)}")
            print(f"  MAPE (Mean Abs % Error): {mape:.1f}%")
            print(f"  Accuracy: {100 - mape:.1f}%")
            
            if mape < 10:
                print(f"  ✅ Excellent prediction accuracy")
            elif mape < 20:
                print(f"  ✓ Good prediction accuracy")
            elif mape < 30:
                print(f"  ⚠️  Fair prediction accuracy")
            else:
                print(f"  ⚠️  Poor prediction accuracy (needs tuning)")
    else:
        print("\n  ⚠️  No prediction data available")
    
    # Resource efficiency
    print("\n" + "="*70)
    print("5. RESOURCE EFFICIENCY")
    print("="*70)
    
    if traffic_values and replicas_total:
        # Calculate resource utilization
        avg_traffic = sum(traffic_values) / len(traffic_values)
        avg_replicas = sum(replicas_total) / len(replicas_total)
        
        # Assume 50 req/s capacity per replica (from config)
        theoretical_capacity = avg_replicas * 50
        utilization = (avg_traffic / theoretical_capacity * 100) if theoretical_capacity > 0 else 0
        
        print(f"\n  Average traffic:     {avg_traffic:.1f} req/s")
        print(f"  Average replicas:    {avg_replicas:.1f}")
        print(f"  Theoretical capacity: {theoretical_capacity:.0f} req/s")
        print(f"  Utilization:         {utilization:.1f}%")
        
        if 60 <= utilization <= 80:
            print(f"  ✅ Optimal utilization (60-80%)")
        elif utilization < 60:
            print(f"  ⚠️  Under-utilized (consider reducing safety margin)")
        else:
            print(f"  ⚠️  Over-utilized (may need faster scaling)")
        
        # Over-provisioning
        over_provision = ((theoretical_capacity - avg_traffic) / avg_traffic * 100) if avg_traffic > 0 else 0
        print(f"  Over-provisioning:   {over_provision:.1f}%")
    
    print("\n" + "="*70)


def generate_plots(data, output_dir: Path):
    """Genera grafici completi"""
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not data:
        print("No data to plot")
        return
    
    timestamps = [d['timestamp'] for d in data]
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Traffic over time
    ax1 = plt.subplot(3, 2, 1)
    traffic = [d['traffic'] for d in data]
    ax1.plot(timestamps, traffic, 'b-', linewidth=2, label='Actual Traffic')
    
    predicted = [d['predicted_traffic'] for d in data]
    if any(p > 0 for p in predicted):
        ax1.plot(timestamps, predicted, 'g--', linewidth=1.5, alpha=0.7, label='Predicted Traffic')
    
    ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='High Threshold (30 req/s)')
    ax1.axhline(y=10, color='cyan', linestyle='--', alpha=0.5, label='Low Threshold (10 req/s)')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Traffic (req/s)')
    ax1.set_title('Traffic Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Plot 2: Replicas over time
    ax2 = plt.subplot(3, 2, 2)
    replicas_total = [d['replicas_total'] for d in data]
    ax2.plot(timestamps, replicas_total, 'r-', linewidth=2, marker='o', markersize=4, label='Total Replicas')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Replica Count')
    ax2.set_title('Replica Count Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Plot 3: Traffic vs Replicas (dual axis)
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(timestamps, traffic, 'b-', linewidth=2, label='Traffic')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Traffic (req/s)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(timestamps, replicas_total, 'r-', linewidth=2, marker='o', markersize=3, label='Replicas')
    ax3_twin.set_ylabel('Replicas', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    ax3.set_title('Traffic vs Replicas Correlation')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Plot 4: Cluster distribution (stacked area)
    ax4 = plt.subplot(3, 2, 4)
    c1 = [d['replicas_cluster1'] for d in data]
    c2 = [d['replicas_cluster2'] for d in data]
    c3 = [d['replicas_cluster3'] for d in data]
    
    ax4.fill_between(timestamps, 0, c1, label='Cluster1', alpha=0.7, color='#1f77b4')
    ax4.fill_between(timestamps, c1, [c1[i]+c2[i] for i in range(len(c1))], label='Cluster2', alpha=0.7, color='#ff7f0e')
    ax4.fill_between(timestamps, [c1[i]+c2[i] for i in range(len(c1))], replicas_total, label='Cluster3', alpha=0.7, color='#2ca02c')
    
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Replicas')
    ax4.set_title('Replica Distribution Across Clusters')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Plot 5: Scaling events timeline
    ax5 = plt.subplot(3, 2, 5)
    
    # Find transitions
    transitions_up = []
    transitions_down = []
    for i in range(1, len(data)):
        delta = data[i]['replicas_total'] - data[i-1]['replicas_total']
        if delta > 0:
            transitions_up.append((timestamps[i], delta))
        elif delta < 0:
            transitions_down.append((timestamps[i], abs(delta)))
    
    if transitions_up:
        up_times, up_deltas = zip(*transitions_up)
        ax5.scatter(up_times, up_deltas, color='green', s=100, marker='^', label='Scale Up', alpha=0.7)
    
    if transitions_down:
        down_times, down_deltas = zip(*transitions_down)
        ax5.scatter(down_times, [-d for d in down_deltas], color='red', s=100, marker='v', label='Scale Down', alpha=0.7)
    
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Replica Change (Δ)')
    ax5.set_title('Scaling Events Timeline')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Plot 6: Prediction accuracy
    ax6 = plt.subplot(3, 2, 6)
    
    actual_vals = []
    predicted_vals = []
    for d in data:
        if d['traffic'] > 0 and d['predicted_traffic'] > 0:
            actual_vals.append(d['traffic'])
            predicted_vals.append(d['predicted_traffic'])
    
    if actual_vals:
        ax6.scatter(actual_vals, predicted_vals, alpha=0.5, s=30)
        
        # Perfect prediction line
        max_val = max(max(actual_vals), max(predicted_vals))
        ax6.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        ax6.set_xlabel('Actual Traffic (req/s)')
        ax6.set_ylabel('Predicted Traffic (req/s)')
        ax6.set_title('Prediction Accuracy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_dir / "dmos_analysis.png"
    plt.savefig(plot_file, dpi=200, bbox_inches='tight')
    print(f"\n📊 Plots saved: {plot_file}")
    
    plt.close()


def export_summary_json(data, output_file: Path):
    """Export summary statistics as JSON"""
    
    if not data:
        return
    
    traffic_values = [d['traffic'] for d in data if d['traffic'] > 0]
    replicas_total = [d['replicas_total'] for d in data]
    
    summary = {
        "test_info": {
            "start": data[0]['timestamp'].isoformat(),
            "end": data[-1]['timestamp'].isoformat(),
            "duration_minutes": (data[-1]['timestamp'] - data[0]['timestamp']).total_seconds() / 60,
            "snapshots": len(data)
        },
        "traffic": {
            "min": min(traffic_values) if traffic_values else 0,
            "max": max(traffic_values) if traffic_values else 0,
            "mean": sum(traffic_values) / len(traffic_values) if traffic_values else 0,
            "samples": len(traffic_values)
        },
        "replicas": {
            "min": min(replicas_total),
            "max": max(replicas_total),
            "mean": sum(replicas_total) / len(replicas_total)
        },
        "scaling_events": {
            "scale_up": data[-1]['scale_up_events'] - data[0]['scale_up_events'],
            "scale_down": data[-1]['scale_down_events'] - data[0]['scale_down_events'],
            "total": (data[-1]['scale_up_events'] - data[0]['scale_up_events']) + 
                    (data[-1]['scale_down_events'] - data[0]['scale_down_events'])
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary JSON saved: {output_file}")


def analyze_test(filepath: str):
    """Main analysis function"""
    
    print("="*70)
    print("🔬 DMOS Test Analysis")
    print("="*70)
    print(f"\nAnalyzing: {filepath}")
    
    # Parse data
    data = parse_metrics_file(filepath)
    
    if not data:
        print("\n❌ No data found in file")
        return
    
    # Print statistics
    print_statistics(data)
    
    # Generate plots
    output_dir = Path("results/plots")
    generate_plots(data, output_dir)
    
    # Export summary
    summary_file = Path(filepath).with_suffix('.summary.json')
    export_summary_json(data, summary_file)
    
    print("\n" + "="*70)
    print("✅ Analysis Complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - Plots: results/plots/dmos_analysis.png")
    print(f"  - Summary: {summary_file}")
    print("\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        # Find latest file
        results_dir = Path("results")
        files = list(results_dir.glob("metrics_timeseries_*.txt"))
        
        if not files:
            print("❌ No metrics files found in results/")
            print("\nUsage: python analyze_test_complete.py <metrics_file.txt>")
            sys.exit(1)
        
        latest = max(files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Using latest file: {latest.name}\n")
        analyze_test(str(latest))
    else:
        analyze_test(sys.argv[1])