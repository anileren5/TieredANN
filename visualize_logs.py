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
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
matplotlib.rcParams['axes.linewidth'] = 1.0
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 4
matplotlib.rcParams['patch.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.width'] = 1.0
matplotlib.rcParams['ytick.major.width'] = 1.0
matplotlib.rcParams['xtick.minor.width'] = 0.5
matplotlib.rcParams['ytick.minor.width'] = 0.5


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


def create_hit_ratio_plot(metrics: List[Dict], output_dir: Path):
    """Plot hit ratio over iterations."""
    iterations = np.arange(len(metrics))
    hit_ratios = [m['hit_ratio'] for m in metrics]
    hits = [m['hits'] for m in metrics]
    total_queries = [m['total_queries'] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Hit ratio
    ax1.plot(iterations, hit_ratios, 'b-', linewidth=1.5, label='Hit Ratio')
    ax1.set_ylabel('Hit Ratio', fontsize=12)
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best')
    ax1.set_title('Cache Hit Ratio Over Iterations', fontsize=13, fontweight='bold')
    
    # Hit count
    ax2.plot(iterations, hits, 'g-', linewidth=1.5, label='Cache Hits', alpha=0.7)
    ax2.plot(iterations, total_queries, 'r--', linewidth=1.5, label='Total Queries', alpha=0.7)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Number of Queries', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hit_ratio.png', bbox_inches='tight', facecolor='white')
    plt.close()


def create_latency_plot(metrics: List[Dict], output_dir: Path):
    """Plot latency metrics over iterations."""
    iterations = np.arange(len(metrics))
    avg_latency = [m['avg_latency_ms'] for m in metrics]
    avg_hit_latency = [m['avg_hit_latency_ms'] if m['avg_hit_latency_ms'] > 0 else None for m in metrics]
    p90_latency = [m['tail_latency_ms']['p90'] for m in metrics]
    p95_latency = [m['tail_latency_ms']['p95'] for m in metrics]
    p99_latency = [m['tail_latency_ms']['p99'] for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, avg_latency, 'b-', linewidth=1.5, label='Average Latency', marker='o', markersize=3, markevery=max(1, len(iterations)//20))
    ax.plot(iterations, p90_latency, 'g--', linewidth=1.2, label='P90 Latency', alpha=0.8)
    ax.plot(iterations, p95_latency, 'orange', linestyle='--', linewidth=1.2, label='P95 Latency', alpha=0.8)
    ax.plot(iterations, p99_latency, 'r--', linewidth=1.2, label='P99 Latency', alpha=0.8)
    
    # Plot hit latency only where it's valid
    valid_hit_indices = [i for i, lat in enumerate(avg_hit_latency) if lat is not None]
    if valid_hit_indices:
        valid_hit_latency = [avg_hit_latency[i] for i in valid_hit_indices]
        ax.plot([iterations[i] for i in valid_hit_indices], valid_hit_latency, 
                'm:', linewidth=1.5, label='Average Hit Latency', alpha=0.9)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Query Latency Metrics Over Iterations', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'latency.png', bbox_inches='tight', facecolor='white')
    plt.close()


def create_qps_plot(metrics: List[Dict], output_dir: Path):
    """Plot QPS metrics over iterations."""
    iterations = np.arange(len(metrics))
    qps = [m['qps'] for m in metrics]
    qps_per_thread = [m['qps_per_thread'] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # Overall QPS
    ax1.plot(iterations, qps, 'b-', linewidth=1.5, label='Overall QPS', marker='o', markersize=3, markevery=max(1, len(iterations)//20))
    ax1.set_ylabel('Queries Per Second', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best')
    ax1.set_title('Query Throughput Over Iterations', fontsize=13, fontweight='bold')
    
    # QPS per thread
    ax2.plot(iterations, qps_per_thread, 'g-', linewidth=1.5, label='QPS per Thread', marker='s', markersize=3, markevery=max(1, len(iterations)//20))
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Queries Per Second per Thread', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'qps.png', bbox_inches='tight', facecolor='white')
    plt.close()


def create_recall_plot(metrics: List[Dict], output_dir: Path):
    """Plot recall metrics over iterations."""
    iterations = np.arange(len(metrics))
    recall_all = [m['recall_all'] for m in metrics]
    recall_cache_hits = [m['recall_cache_hits'] if m['recall_cache_hits'] is not None else None for m in metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(iterations, recall_all, 'b-', linewidth=1.5, label='Recall (All Queries)', marker='o', markersize=3, markevery=max(1, len(iterations)//20))
    
    # Plot cache hit recall only where it's valid
    valid_recall_indices = [i for i, r in enumerate(recall_cache_hits) if r is not None]
    if valid_recall_indices:
        valid_recall = [recall_cache_hits[i] for i in valid_recall_indices]
        ax.plot([iterations[i] for i in valid_recall_indices], valid_recall, 
                'g--', linewidth=1.5, label='Recall (Cache Hits)', marker='s', markersize=3, markevery=max(1, len(valid_recall_indices)//20))
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_title('Recall Metrics Over Iterations', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'recall.png', bbox_inches='tight', facecolor='white')
    plt.close()


def create_memory_plot(metrics: List[Dict], output_dir: Path):
    """Plot memory usage over iterations."""
    iterations = np.arange(len(metrics))
    memory_active = [m['memory_active_vectors'] for m in metrics]
    memory_max = [m['memory_max_points'] for m in metrics]
    pca_regions = [m['pca_active_regions'] for m in metrics]
    
    # Get index vectors
    num_indices = max([len([k for k in m.keys() if k.startswith('index_') and k.endswith('_vectors')]) for m in metrics])
    index_vectors = {}
    for i in range(num_indices):
        key = f'index_{i}_vectors'
        index_vectors[i] = [m.get(key, 0) for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Memory usage
    ax1.plot(iterations, memory_active, 'b-', linewidth=1.5, label='Active Vectors in Memory', marker='o', markersize=3, markevery=max(1, len(iterations)//20))
    ax1.axhline(y=memory_max[0], color='r', linestyle='--', linewidth=1.5, label=f'Max Capacity ({memory_max[0]:,})', alpha=0.7)
    ax1.set_ylabel('Number of Vectors', fontsize=12)
    ax1.set_title('Memory Usage Over Iterations', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best')
    
    # Index distribution
    colors = plt.cm.tab10(np.linspace(0, 1, num_indices))
    for i in range(num_indices):
        ax2.plot(iterations, index_vectors[i], linewidth=1.5, label=f'Index {i}', 
                color=colors[i], marker='o', markersize=2, markevery=max(1, len(iterations)//30))
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Number of Vectors', fontsize=12)
    ax2.set_title('Vector Distribution Across Mini-Indexes', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', ncol=min(4, num_indices))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'memory.png', bbox_inches='tight', facecolor='white')
    plt.close()


def create_comprehensive_plot(metrics: List[Dict], output_dir: Path):
    """Create a comprehensive multi-panel plot."""
    iterations = np.arange(len(metrics))
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Hit ratio
    ax1 = fig.add_subplot(gs[0, 0])
    hit_ratios = [m['hit_ratio'] for m in metrics]
    ax1.plot(iterations, hit_ratios, 'b-', linewidth=1.5)
    ax1.set_ylabel('Hit Ratio', fontsize=11)
    ax1.set_title('Cache Hit Ratio', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([-0.05, 1.05])
    
    # Latency
    ax2 = fig.add_subplot(gs[0, 1])
    avg_latency = [m['avg_latency_ms'] for m in metrics]
    p99_latency = [m['tail_latency_ms']['p99'] for m in metrics]
    ax2.plot(iterations, avg_latency, 'b-', linewidth=1.5, label='Avg', marker='o', markersize=2, markevery=max(1, len(iterations)//30))
    ax2.plot(iterations, p99_latency, 'r--', linewidth=1.2, label='P99', alpha=0.8)
    ax2.set_ylabel('Latency (ms)', fontsize=11)
    ax2.set_title('Query Latency', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best')
    
    # QPS
    ax3 = fig.add_subplot(gs[1, 0])
    qps = [m['qps'] for m in metrics]
    ax3.plot(iterations, qps, 'g-', linewidth=1.5, marker='s', markersize=2, markevery=max(1, len(iterations)//30))
    ax3.set_ylabel('QPS', fontsize=11)
    ax3.set_title('Query Throughput', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # Recall
    ax4 = fig.add_subplot(gs[1, 1])
    recall_all = [m['recall_all'] for m in metrics]
    ax4.plot(iterations, recall_all, 'purple', linewidth=1.5, marker='^', markersize=2, markevery=max(1, len(iterations)//30))
    ax4.set_ylabel('Recall', fontsize=11)
    ax4.set_title('Recall (All Queries)', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # Memory
    ax5 = fig.add_subplot(gs[2, 0])
    memory_active = [m['memory_active_vectors'] for m in metrics]
    memory_max = [m['memory_max_points'] for m in metrics]
    ax5.plot(iterations, memory_active, 'orange', linewidth=1.5, marker='d', markersize=2, markevery=max(1, len(iterations)//30))
    ax5.axhline(y=memory_max[0], color='r', linestyle='--', linewidth=1.2, alpha=0.7, label='Max Capacity')
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('Vectors in Memory', fontsize=11)
    ax5.set_title('Memory Usage', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(loc='best')
    
    # Latency vs Hit Ratio (scatter)
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(hit_ratios, avg_latency, c=iterations, cmap='viridis', 
                         s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Hit Ratio', fontsize=11)
    ax6.set_ylabel('Average Latency (ms)', fontsize=11)
    ax6.set_title('Latency vs Hit Ratio', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Iteration', fontsize=10)
    
    plt.suptitle('QVCache Performance Metrics Over Iterations', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_dir / 'comprehensive.png', bbox_inches='tight', facecolor='white', dpi=300)
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
    
    print("Generating plots...")
    create_hit_ratio_plot(metrics, output_dir)
    print("  ✓ Hit ratio plot")
    
    create_latency_plot(metrics, output_dir)
    print("  ✓ Latency plot")
    
    create_qps_plot(metrics, output_dir)
    print("  ✓ QPS plot")
    
    create_recall_plot(metrics, output_dir)
    print("  ✓ Recall plot")
    
    create_memory_plot(metrics, output_dir)
    print("  ✓ Memory plot")
    
    create_comprehensive_plot(metrics, output_dir)
    print("  ✓ Comprehensive plot")
    
    print(f"\nAll plots saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

