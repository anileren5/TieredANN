#!/usr/bin/env python3
"""
High-quality visualization script for QVCache experiment logs.
Produces publication-ready plots for paper submission.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Set high-quality matplotlib parameters for publication
matplotlib.rcParams['figure.dpi'] = 600
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['font.size'] = 6
matplotlib.rcParams['axes.labelsize'] = 6
matplotlib.rcParams['axes.titlesize'] = 6
matplotlib.rcParams['xtick.labelsize'] = 5
matplotlib.rcParams['ytick.labelsize'] = 5
matplotlib.rcParams['legend.fontsize'] = 5
matplotlib.rcParams['figure.titlesize'] = 7
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
matplotlib.rcParams['axes.linewidth'] = 0.7
matplotlib.rcParams['grid.linewidth'] = 0.35
matplotlib.rcParams['lines.linewidth'] = 0.7
matplotlib.rcParams['lines.markersize'] = 2.5
matplotlib.rcParams['patch.linewidth'] = 0.35
matplotlib.rcParams['xtick.major.width'] = 0.7
matplotlib.rcParams['ytick.major.width'] = 0.7
matplotlib.rcParams['xtick.minor.width'] = 0.35
matplotlib.rcParams['ytick.minor.width'] = 0.35


def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """Parse JSON log file and extract split_metrics events."""
    metrics = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
                if data.get('event') == 'split_metrics':
                    metrics.append(data)
            except json.JSONDecodeError:
                continue
    return metrics


def create_all_metrics_plot(metrics: List[Dict], output_dir: Path):
    """Create a single vertical chart with selected metrics."""
    iterations = np.arange(len(metrics))
    
    # Extract only the requested data
    hit_ratios = [m['hit_ratio'] for m in metrics]
    avg_latency = [m['avg_latency_ms'] for m in metrics]
    avg_hit_latency = [m['avg_hit_latency_ms'] if m['avg_hit_latency_ms'] > 0 else None for m in metrics]
    p50_latency = [m.get('tail_latency_ms', {}).get('p50', None) for m in metrics]
    p99_latency = [m.get('tail_latency_ms', {}).get('p99', None) for m in metrics]
    qps = [m['qps'] for m in metrics]
    memory_active = [m['memory_active_vectors'] for m in metrics]
    recall_all = [m['recall_all'] for m in metrics]
    
    # Create figure with 8 vertical subplots - optimized for side-by-side placement (5 plots)
    fig, axes = plt.subplots(8, 1, figsize=(2.8, 6.5), sharex=True)
    fig.subplots_adjust(hspace=0.25, left=0.15, right=0.95, top=0.98, bottom=0.08)  # Very tight spacing
    
    # 1. Hit Ratio - Blue
    axes[0].plot(iterations, hit_ratios, 'b-', linewidth=0.7, color='blue')
    axes[0].set_ylabel('Hit Ratio', fontsize=6)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    axes[0].set_title('Cache Hit Ratio', fontsize=6, fontweight='bold', pad=2)
    
    # 2. Average Latency - Red
    axes[1].plot(iterations, avg_latency, 'r-', linewidth=0.7, color='red')
    axes[1].set_ylabel('Latency (ms)', fontsize=6)
    axes[1].set_title('Average Latency', fontsize=6, fontweight='bold', pad=2)
    axes[1].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    
    # 3. Average Hit Latency - Green
    valid_hit_indices = [i for i, lat in enumerate(avg_hit_latency) if lat is not None]
    if valid_hit_indices:
        valid_hit_latency = [avg_hit_latency[i] for i in valid_hit_indices]
        axes[2].plot([iterations[i] for i in valid_hit_indices], valid_hit_latency, 
                'g-', linewidth=0.7, color='green')
    axes[2].set_ylabel('Latency (ms)', fontsize=6)
    axes[2].set_title('Average Hit Latency', fontsize=6, fontweight='bold', pad=2)
    axes[2].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    
    # 4. P50 Latency - Brown/Dark Red
    valid_p50_indices = [i for i, lat in enumerate(p50_latency) if lat is not None]
    if valid_p50_indices:
        valid_p50_latency = [p50_latency[i] for i in valid_p50_indices]
        axes[3].plot([iterations[i] for i in valid_p50_indices], valid_p50_latency, 
                linewidth=0.7, color='brown')
    axes[3].set_ylabel('Latency (ms)', fontsize=6)
    axes[3].set_title('P50 Latency', fontsize=6, fontweight='bold', pad=2)
    axes[3].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    
    # 5. P99 Latency - Dark Red
    valid_p99_indices = [i for i, lat in enumerate(p99_latency) if lat is not None]
    if valid_p99_indices:
        valid_p99_latency = [p99_latency[i] for i in valid_p99_indices]
        axes[4].plot([iterations[i] for i in valid_p99_indices], valid_p99_latency, 
                linewidth=0.7, color='darkred')
    axes[4].set_ylabel('Latency (ms)', fontsize=6)
    axes[4].set_title('P99 Latency', fontsize=6, fontweight='bold', pad=2)
    axes[4].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    
    # 6. QPS - Orange
    axes[5].plot(iterations, qps, linewidth=0.7, color='orange')
    axes[5].set_ylabel('QPS', fontsize=6)
    axes[5].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    axes[5].set_title('Query Throughput', fontsize=6, fontweight='bold', pad=2)
    
    # 7. Memory Active Vectors - Purple
    axes[6].plot(iterations, memory_active, linewidth=0.7, color='purple')
    axes[6].set_ylabel('Vectors', fontsize=6)
    axes[6].set_title('Memory Active Vectors', fontsize=6, fontweight='bold', pad=2)
    axes[6].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    
    # 8. Recall All - Teal
    axes[7].plot(iterations, recall_all, linewidth=0.7, color='teal')
    axes[7].set_xlabel('Iteration', fontsize=6)
    axes[7].set_ylabel('Recall', fontsize=6)
    axes[7].set_ylim([0, 1.05])
    axes[7].set_title('Recall', fontsize=6, fontweight='bold', pad=2)
    axes[7].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    
    # Remove main title for side-by-side placement - each plot can have its own context
    plt.savefig(output_dir / 'all_metrics.png', bbox_inches='tight', facecolor='white', dpi=600, pad_inches=0.05)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize QVCache experiment logs')
    parser.add_argument('--log', type=str, default='qdrant.log', help='Path to log file')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Parsing log file: {args.log}")
    metrics = parse_log_file(args.log)
    print(f"Found {len(metrics)} split_metrics events")
    
    if not metrics:
        print("No metrics found in log file!")
        return
    
    print("Generating combined plot with all metrics...")
    create_all_metrics_plot(metrics, output_dir)
    print("  ✓ All metrics plot")
    
    print(f"\nPlot saved to: {output_dir.absolute() / 'all_metrics.png'}")


if __name__ == '__main__':
    main()

