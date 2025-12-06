#!/usr/bin/env python3
"""
High-quality visualization script for QVCache experiment logs.
Produces publication-ready plots for paper submission.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter, LogFormatter, LogLocator, FixedLocator, NullLocator
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Set super high-quality matplotlib parameters for publication
matplotlib.rcParams['figure.dpi'] = 300  # High DPI for publication
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts for better quality
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 7
matplotlib.rcParams['axes.labelsize'] = 7
matplotlib.rcParams['axes.titlesize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 6
matplotlib.rcParams['ytick.labelsize'] = 6
matplotlib.rcParams['legend.fontsize'] = 6
matplotlib.rcParams['figure.titlesize'] = 9
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman', 'Liberation Serif']
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['grid.linewidth'] = 0.4
matplotlib.rcParams['lines.linewidth'] = 1.0
matplotlib.rcParams['lines.markersize'] = 3.0
matplotlib.rcParams['patch.linewidth'] = 0.4
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8
matplotlib.rcParams['xtick.minor.width'] = 0.4
matplotlib.rcParams['ytick.minor.width'] = 0.4


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


def extract_metric(metrics: List[Dict], key: str, default=None):
    """Extract metric from list, handling missing keys."""
    result = []
    for m in metrics:
        if key in m:
            val = m[key]
            if isinstance(val, (int, float)) and val > 0:
                result.append(val)
            elif val is not None:
                result.append(val)
            else:
                result.append(default)
        else:
            result.append(default)
    return result

def format_log_tick(value, pos):
    """Format log scale ticks as regular numbers without scientific notation."""
    if value == 0:
        return '0'
    # Format as integer if it's a whole number, otherwise as float with appropriate precision
    if value >= 1000:
        return f'{int(value):,}'
    elif value >= 1:
        if value == int(value):
            return str(int(value))
        else:
            return f'{value:.1f}'
    elif value >= 0.1:
        return f'{value:.2f}'
    elif value >= 0.01:
        return f'{value:.3f}'
    else:
        return f'{value:.4f}'

def create_individual_plots(backend_metrics: List[Dict], qvcache_metrics: List[Dict], output_dir: Path):
    """Create individual plot files for each metric with comparison."""
    backend_iterations = np.arange(len(backend_metrics)) if backend_metrics else []
    qvcache_iterations = np.arange(len(qvcache_metrics)) if qvcache_metrics else []
    
    # Extract data - handle missing keys gracefully
    backend_avg_latency = extract_metric(backend_metrics, 'avg_latency_ms')
    qvcache_avg_latency = extract_metric(qvcache_metrics, 'avg_latency_ms')
    
    backend_avg_hit_latency = extract_metric(backend_metrics, 'avg_hit_latency_ms', None)
    qvcache_avg_hit_latency = extract_metric(qvcache_metrics, 'avg_hit_latency_ms', None)
    
    backend_p50 = [m.get('tail_latency_ms', {}).get('p50', None) if 'tail_latency_ms' in m else None for m in backend_metrics]
    qvcache_p50 = [m.get('tail_latency_ms', {}).get('p50', None) if 'tail_latency_ms' in m else None for m in qvcache_metrics]
    
    backend_p99 = [m.get('tail_latency_ms', {}).get('p99', None) if 'tail_latency_ms' in m else None for m in backend_metrics]
    qvcache_p99 = [m.get('tail_latency_ms', {}).get('p99', None) if 'tail_latency_ms' in m else None for m in qvcache_metrics]
    
    backend_qps = extract_metric(backend_metrics, 'qps')
    qvcache_qps = extract_metric(qvcache_metrics, 'qps')
    
    backend_recall = extract_metric(backend_metrics, 'recall_all')
    qvcache_recall = extract_metric(qvcache_metrics, 'recall_all')
    
    # QVCache-only metrics
    qvcache_hit_ratios = extract_metric(qvcache_metrics, 'hit_ratio', None)
    qvcache_memory_active = extract_metric(qvcache_metrics, 'memory_active_vectors', None)
    
    # Define plot configurations: (backend_data, qvcache_data, ylabel, title, color, ylim, filename, show_legend, use_log_scale)
    # Using professional, colorblind-friendly color palette with darker, more saturated colors
    plot_configs = [
        (None, qvcache_hit_ratios, 'Hit Ratio', 'Cache Hit Ratio', '#1f4e79', (-0.05, 1.05), 'hit_ratio.pdf', True, False),
        (backend_avg_hit_latency, qvcache_avg_hit_latency, 'Latency (ms)', 'Average Hit Latency', '#548235', None, 'avg_hit_latency.pdf', True, False),
        (backend_p50, qvcache_p50, 'Latency (ms)', 'Latency (P50)', '#c55a11', None, 'p50_latency.pdf', True, True),
        (backend_qps, qvcache_qps, 'QPS', 'Query Throughput', '#bf8f00', None, 'qps.pdf', True, True),
        (None, qvcache_memory_active, 'Vectors', 'Vectors In Cache', '#5b2c6f', None, 'memory_active_vectors.pdf', True, False),
        (backend_recall, qvcache_recall, 'Recall@10', 'Recall@10', '#2e75b6', None, 'recall.pdf', True, False),
    ]
    
    for backend_data, qvcache_data, ylabel, title, color, ylim, filename, show_legend, use_log_scale in plot_configs:
        # Make plots bigger for publication quality
        fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.2))
        # Adjust margins for log scale plots to accommodate longer tick labels
        # Also leave room below for legend if shown
        if show_legend:
            if use_log_scale:
                fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.50)
            else:
                fig.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.50)
        else:
            if use_log_scale:
                fig.subplots_adjust(left=0.22, right=0.95, top=0.85, bottom=0.25)
            else:
                fig.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.25)
        
        # Set log scale if requested
        if use_log_scale:
            ax.set_yscale('log')
            # Use custom formatter for better tick labels
            ax.yaxis.set_major_formatter(FuncFormatter(format_log_tick))
            # Disable minor ticks
            ax.yaxis.set_minor_locator(NullLocator())
            
            # Set custom tick locations: backend at average, QVCache at min/max
            tick_locations = []
            tick_labels = []
            
            # For average latency, add 1ms reference tick
            if filename == 'avg_latency.pdf':
                tick_locations.append(1.0)
                tick_labels.append('1')
            
            # Backend: tick at average
            if backend_data and any(x is not None and x > 0 for x in backend_data):
                backend_values = [x for x in backend_data if x is not None and x > 0]
                if backend_values:
                    backend_avg = np.mean(backend_values)
                    tick_locations.append(backend_avg)
                    tick_labels.append(format_log_tick(backend_avg, None))
            
            # QVCache: tick at min for latency, max for QPS
            if qvcache_data and any(x is not None and x > 0 for x in qvcache_data):
                qvcache_values = [x for x in qvcache_data if x is not None and x > 0]
                if qvcache_values:
                    # Determine if this is a latency metric (min) or QPS (max)
                    is_latency = 'latency' in filename.lower() or 'latency' in title.lower()
                    if is_latency:
                        qvcache_tick_val = min(qvcache_values)
                    else:  # QPS
                        qvcache_tick_val = max(qvcache_values)
                    tick_locations.append(qvcache_tick_val)
                    tick_labels.append(format_log_tick(qvcache_tick_val, None))
            
            if tick_locations:
                ax.yaxis.set_major_locator(FixedLocator(tick_locations))
                # Set custom labels with actual values
                ax.set_yticks(tick_locations)
                ax.set_yticklabels(tick_labels, fontsize=6)
            else:
                # Fallback to default if no data
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            
            ax.tick_params(axis='y', labelsize=6, pad=2, which='major')
            ax.tick_params(axis='y', which='minor', length=0)  # Hide minor ticks
        
        # Plot backend data if available
        if backend_data and any(x is not None for x in backend_data):
            valid_indices = [i for i, val in enumerate(backend_data) if val is not None and val > 0]
            if valid_indices:
                valid_data = [backend_data[i] for i in valid_indices]
                valid_iterations = [backend_iterations[i] for i in valid_indices]
                ax.plot(valid_iterations, valid_data, linewidth=1.0, color='#E74C3C', linestyle='--', label='Backend only', alpha=0.8)
        
        # Plot QVCache data if available
        if qvcache_data and any(x is not None for x in qvcache_data):
            # For hit ratio, allow 0 values (0% hit rate is valid)
            # For other metrics, filter out 0 values
            if filename == 'hit_ratio.pdf':
                valid_indices = [i for i, val in enumerate(qvcache_data) if val is not None and val >= 0]
            else:
                valid_indices = [i for i, val in enumerate(qvcache_data) if val is not None and val > 0]
            if valid_indices:
                valid_data = [qvcache_data[i] for i in valid_indices]
                valid_iterations = [qvcache_iterations[i] for i in valid_indices]
                ax.plot(valid_iterations, valid_data, linewidth=1.0, color=color, linestyle='-', label='With QVCache', alpha=0.9)
        
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlabel('Splits', fontsize=7, labelpad=4)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=8)
        # Adjust tick parameters to prevent overlap
        ax.tick_params(axis='x', labelsize=6, pad=2)
        if not use_log_scale:
            ax.tick_params(axis='y', labelsize=6, pad=2)
        
        # Special handling for recall: use dynamic scale based on data range
        if filename == 'recall.pdf':
            all_values = []
            if backend_data:
                all_values.extend([x for x in backend_data if x is not None])
            if qvcache_data:
                all_values.extend([x for x in qvcache_data if x is not None])
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                # Add 5% padding on each side, but ensure we don't go below 0
                range_val = max_val - min_val
                padding = max(range_val * 0.05, 0.01)  # At least 1% padding
                y_min = max(0, min_val - padding)
                y_max = min(1.05, max_val + padding)
                ax.set_ylim([y_min, y_max])
        elif ylim:
            ax.set_ylim(ylim)
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4, color='gray')
        if use_log_scale:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4, color='gray', which='both')  # Show both major and minor grid lines for log scale
        
        if show_legend and (backend_data or qvcache_data):
            # Place legend below the xlabel, outside the plot area
            # xlabel will be positioned between plot and legend
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=6, framealpha=0.95, 
                     frameon=True, ncol=2 if (backend_data and qvcache_data) else 1, edgecolor='gray', fancybox=False)
        
        plt.savefig(output_dir / filename, bbox_inches='tight', facecolor='white', format='pdf', 
                   pad_inches=0.05, dpi=300, transparent=False)
        plt.close()
        print(f"  ✓ {filename}")


def create_all_metrics_plot(backend_metrics: List[Dict], qvcache_metrics: List[Dict], output_dir: Path):
    """Create a single vertical chart with selected metrics comparing backend and QVCache."""
    backend_iterations = np.arange(len(backend_metrics)) if backend_metrics else []
    qvcache_iterations = np.arange(len(qvcache_metrics)) if qvcache_metrics else []
    
    # Extract data - handle missing keys gracefully
    backend_avg_latency = extract_metric(backend_metrics, 'avg_latency_ms')
    qvcache_avg_latency = extract_metric(qvcache_metrics, 'avg_latency_ms')
    
    backend_avg_hit_latency = extract_metric(backend_metrics, 'avg_hit_latency_ms', None)
    qvcache_avg_hit_latency = extract_metric(qvcache_metrics, 'avg_hit_latency_ms', None)
    
    backend_p50 = [m.get('tail_latency_ms', {}).get('p50', None) if 'tail_latency_ms' in m else None for m in backend_metrics]
    qvcache_p50 = [m.get('tail_latency_ms', {}).get('p50', None) if 'tail_latency_ms' in m else None for m in qvcache_metrics]
    
    backend_p99 = [m.get('tail_latency_ms', {}).get('p99', None) if 'tail_latency_ms' in m else None for m in backend_metrics]
    qvcache_p99 = [m.get('tail_latency_ms', {}).get('p99', None) if 'tail_latency_ms' in m else None for m in qvcache_metrics]
    
    backend_qps = extract_metric(backend_metrics, 'qps')
    qvcache_qps = extract_metric(qvcache_metrics, 'qps')
    
    backend_recall = extract_metric(backend_metrics, 'recall_all')
    qvcache_recall = extract_metric(qvcache_metrics, 'recall_all')
    
    # QVCache-only metrics
    qvcache_hit_ratios = extract_metric(qvcache_metrics, 'hit_ratio', None)
    qvcache_memory_active = extract_metric(qvcache_metrics, 'memory_active_vectors', None)
    
    # Create figure with 8 vertical subplots - make it bigger
    fig, axes = plt.subplots(8, 1, figsize=(2.2, 9.0), sharex=True)
    # Adjust left margin for log scale plots to accommodate tick labels
    # Leave extra space at bottom for legends and xlabel
    fig.subplots_adjust(hspace=0.30, left=0.22, right=0.95, top=0.98, bottom=0.25)
    
    # 1. Hit Ratio - QVCache only
    if qvcache_hit_ratios and any(x is not None for x in qvcache_hit_ratios):
        valid_indices = [i for i, val in enumerate(qvcache_hit_ratios) if val is not None and val >= 0]
        if valid_indices:
            valid_data = [qvcache_hit_ratios[i] for i in valid_indices]
            axes[0].plot([qvcache_iterations[i] for i in valid_indices], valid_data, 
                    linewidth=0.7, color='#2E86AB', linestyle='-', label='With QVCache')
    axes[0].set_ylabel('Hit Ratio', fontsize=6)
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    axes[0].set_title('Cache Hit Ratio', fontsize=6, fontweight='bold', pad=2)
    if qvcache_hit_ratios and any(x is not None for x in qvcache_hit_ratios):
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=1)
    
    # 2. Average Latency - Both (log scale)
    axes[1].set_yscale('log')
    axes[1].yaxis.set_major_formatter(FuncFormatter(format_log_tick))
    axes[1].yaxis.set_minor_locator(NullLocator())  # Disable minor ticks
    
    # Set custom tick locations: backend at average, QVCache at min, and 1ms reference
    tick_locations = []
    tick_labels = []
    
    # Add 1ms reference tick
    tick_locations.append(1.0)
    tick_labels.append('1')
    
    if backend_avg_latency and any(x is not None and x > 0 for x in backend_avg_latency):
        backend_values = [x for x in backend_avg_latency if x is not None and x > 0]
        if backend_values:
            backend_avg = np.mean(backend_values)
            tick_locations.append(backend_avg)
            tick_labels.append(format_log_tick(backend_avg, None))
    if qvcache_avg_latency and any(x is not None and x > 0 for x in qvcache_avg_latency):
        qvcache_values = [x for x in qvcache_avg_latency if x is not None and x > 0]
        if qvcache_values:
            qvcache_min = min(qvcache_values)
            tick_locations.append(qvcache_min)
            tick_labels.append(format_log_tick(qvcache_min, None))
    if tick_locations:
        axes[1].yaxis.set_major_locator(FixedLocator(tick_locations))
        axes[1].set_yticks(tick_locations)
        axes[1].set_yticklabels(tick_labels, fontsize=5)
    else:
        axes[1].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    
    axes[1].tick_params(axis='y', labelsize=5, pad=1, which='major')
    axes[1].tick_params(axis='y', which='minor', length=0)  # Hide minor ticks
    if backend_avg_latency and any(x is not None and x > 0 for x in backend_avg_latency):
        axes[1].plot(backend_iterations, backend_avg_latency, linewidth=0.7, color='#E63946', linestyle='--', label='Backend only')
    if qvcache_avg_latency and any(x is not None and x > 0 for x in qvcache_avg_latency):
        axes[1].plot(qvcache_iterations, qvcache_avg_latency, linewidth=0.7, color='#E63946', linestyle='-', label='With QVCache')
    axes[1].set_ylabel('Latency (ms)', fontsize=6)
    axes[1].set_title('Average Latency', fontsize=6, fontweight='bold', pad=2)
    axes[1].grid(True, alpha=0.25, linestyle='--', linewidth=0.35, which='both')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=2)
    
    # 3. Average Hit Latency - Both
    if backend_avg_hit_latency and any(x is not None for x in backend_avg_hit_latency):
        valid_indices = [i for i, val in enumerate(backend_avg_hit_latency) if val is not None]
        if valid_indices:
            axes[2].plot([backend_iterations[i] for i in valid_indices], 
                    [backend_avg_hit_latency[i] for i in valid_indices],
                    linewidth=0.7, color='#06A77D', linestyle='--', label='Backend only')
    if qvcache_avg_hit_latency and any(x is not None for x in qvcache_avg_hit_latency):
        valid_indices = [i for i, val in enumerate(qvcache_avg_hit_latency) if val is not None]
        if valid_indices:
            axes[2].plot([qvcache_iterations[i] for i in valid_indices], 
                    [qvcache_avg_hit_latency[i] for i in valid_indices],
                    linewidth=0.7, color='#06A77D', linestyle='-', label='With QVCache')
    axes[2].set_ylabel('Latency (ms)', fontsize=6)
    axes[2].set_title('Average Hit Latency', fontsize=6, fontweight='bold', pad=2)
    axes[2].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    if (backend_avg_hit_latency and any(x is not None for x in backend_avg_hit_latency)) or \
       (qvcache_avg_hit_latency and any(x is not None for x in qvcache_avg_hit_latency)):
        axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=2)
    
    # 4. P50 Latency - Both (log scale)
    axes[3].set_yscale('log')
    axes[3].yaxis.set_major_formatter(FuncFormatter(format_log_tick))
    axes[3].yaxis.set_minor_locator(NullLocator())  # Disable minor ticks
    
    # Set custom tick locations: backend at average, QVCache at min
    tick_locations = []
    tick_labels = []
    if backend_p50 and any(x is not None and x > 0 for x in backend_p50):
        backend_values = [x for x in backend_p50 if x is not None and x > 0]
        if backend_values:
            backend_avg = np.mean(backend_values)
            tick_locations.append(backend_avg)
            tick_labels.append(format_log_tick(backend_avg, None))
    if qvcache_p50 and any(x is not None and x > 0 for x in qvcache_p50):
        qvcache_values = [x for x in qvcache_p50 if x is not None and x > 0]
        if qvcache_values:
            qvcache_min = min(qvcache_values)
            tick_locations.append(qvcache_min)
            tick_labels.append(format_log_tick(qvcache_min, None))
    if tick_locations:
        axes[3].yaxis.set_major_locator(FixedLocator(tick_locations))
        axes[3].set_yticks(tick_locations)
        axes[3].set_yticklabels(tick_labels, fontsize=5)
    else:
        axes[3].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    
    axes[3].tick_params(axis='y', labelsize=5, pad=1, which='major')
    axes[3].tick_params(axis='y', which='minor', length=0)  # Hide minor ticks
    if backend_p50 and any(x is not None and x > 0 for x in backend_p50):
        valid_indices = [i for i, val in enumerate(backend_p50) if val is not None and val > 0]
        if valid_indices:
            axes[3].plot([backend_iterations[i] for i in valid_indices], 
                    [backend_p50[i] for i in valid_indices],
                    linewidth=0.7, color='#F77F00', linestyle='--', label='Backend only')
    if qvcache_p50 and any(x is not None and x > 0 for x in qvcache_p50):
        valid_indices = [i for i, val in enumerate(qvcache_p50) if val is not None and val > 0]
        if valid_indices:
            axes[3].plot([qvcache_iterations[i] for i in valid_indices], 
                    [qvcache_p50[i] for i in valid_indices],
                    linewidth=0.7, color='#F77F00', linestyle='-', label='With QVCache')
    axes[3].set_ylabel('Latency (ms)', fontsize=6)
    axes[3].set_title('P50 Latency', fontsize=6, fontweight='bold', pad=2)
    axes[3].grid(True, alpha=0.25, linestyle='--', linewidth=0.35, which='both')
    if (backend_p50 and any(x is not None and x > 0 for x in backend_p50)) or \
       (qvcache_p50 and any(x is not None and x > 0 for x in qvcache_p50)):
        axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=2)
    
    # 5. P99 Latency - Both (log scale)
    axes[4].set_yscale('log')
    axes[4].yaxis.set_major_formatter(FuncFormatter(format_log_tick))
    axes[4].yaxis.set_minor_locator(NullLocator())  # Disable minor ticks
    
    # Set custom tick locations: backend at average, QVCache at min
    tick_locations = []
    tick_labels = []
    if backend_p99 and any(x is not None and x > 0 for x in backend_p99):
        backend_values = [x for x in backend_p99 if x is not None and x > 0]
        if backend_values:
            backend_avg = np.mean(backend_values)
            tick_locations.append(backend_avg)
            tick_labels.append(format_log_tick(backend_avg, None))
    if qvcache_p99 and any(x is not None and x > 0 for x in qvcache_p99):
        qvcache_values = [x for x in qvcache_p99 if x is not None and x > 0]
        if qvcache_values:
            qvcache_min = min(qvcache_values)
            tick_locations.append(qvcache_min)
            tick_labels.append(format_log_tick(qvcache_min, None))
    if tick_locations:
        axes[4].yaxis.set_major_locator(FixedLocator(tick_locations))
        axes[4].set_yticks(tick_locations)
        axes[4].set_yticklabels(tick_labels, fontsize=5)
    else:
        axes[4].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    
    axes[4].tick_params(axis='y', labelsize=5, pad=1, which='major')
    axes[4].tick_params(axis='y', which='minor', length=0)  # Hide minor ticks
    if backend_p99 and any(x is not None and x > 0 for x in backend_p99):
        valid_indices = [i for i, val in enumerate(backend_p99) if val is not None and val > 0]
        if valid_indices:
            axes[4].plot([backend_iterations[i] for i in valid_indices], 
                    [backend_p99[i] for i in valid_indices],
                    linewidth=0.7, color='#A23B72', linestyle='--', label='Backend only')
    if qvcache_p99 and any(x is not None and x > 0 for x in qvcache_p99):
        valid_indices = [i for i, val in enumerate(qvcache_p99) if val is not None and val > 0]
        if valid_indices:
            axes[4].plot([qvcache_iterations[i] for i in valid_indices], 
                    [qvcache_p99[i] for i in valid_indices],
                    linewidth=0.7, color='#A23B72', linestyle='-', label='With QVCache')
    axes[4].set_ylabel('Latency (ms)', fontsize=6)
    axes[4].set_title('P99 Latency', fontsize=6, fontweight='bold', pad=2)
    axes[4].grid(True, alpha=0.25, linestyle='--', linewidth=0.35, which='both')
    if (backend_p99 and any(x is not None and x > 0 for x in backend_p99)) or \
       (qvcache_p99 and any(x is not None and x > 0 for x in qvcache_p99)):
        axes[4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=2)
    
    # 6. QPS - Both (log scale)
    axes[5].set_yscale('log')
    axes[5].yaxis.set_major_formatter(FuncFormatter(format_log_tick))
    axes[5].yaxis.set_minor_locator(NullLocator())  # Disable minor ticks
    
    # Set custom tick locations: backend at average, QVCache at max
    tick_locations = []
    tick_labels = []
    if backend_qps and any(x is not None and x > 0 for x in backend_qps):
        backend_values = [x for x in backend_qps if x is not None and x > 0]
        if backend_values:
            backend_avg = np.mean(backend_values)
            tick_locations.append(backend_avg)
            tick_labels.append(format_log_tick(backend_avg, None))
    if qvcache_qps and any(x is not None and x > 0 for x in qvcache_qps):
        qvcache_values = [x for x in qvcache_qps if x is not None and x > 0]
        if qvcache_values:
            qvcache_max = max(qvcache_values)
            tick_locations.append(qvcache_max)
            tick_labels.append(format_log_tick(qvcache_max, None))
    if tick_locations:
        axes[5].yaxis.set_major_locator(FixedLocator(tick_locations))
        axes[5].set_yticks(tick_locations)
        axes[5].set_yticklabels(tick_labels, fontsize=5)
    else:
        axes[5].yaxis.set_major_locator(LogLocator(base=10, numticks=4))
    
    axes[5].tick_params(axis='y', labelsize=5, pad=1, which='major')
    axes[5].tick_params(axis='y', which='minor', length=0)  # Hide minor ticks
    if backend_qps and any(x is not None and x > 0 for x in backend_qps):
        axes[5].plot(backend_iterations, backend_qps, linewidth=0.7, color='#FCBF49', linestyle='--', label='Backend only')
    if qvcache_qps and any(x is not None and x > 0 for x in qvcache_qps):
        axes[5].plot(qvcache_iterations, qvcache_qps, linewidth=0.7, color='#FCBF49', linestyle='-', label='With QVCache')
    axes[5].set_ylabel('QPS', fontsize=6)
    axes[5].grid(True, alpha=0.25, linestyle='--', linewidth=0.35, which='both')
    axes[5].set_title('Query Throughput', fontsize=6, fontweight='bold', pad=2)
    if (backend_qps and any(x is not None and x > 0 for x in backend_qps)) or \
       (qvcache_qps and any(x is not None and x > 0 for x in qvcache_qps)):
        axes[5].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=2)
    
    # 7. Memory Active Vectors - QVCache only
    if qvcache_memory_active and any(x is not None for x in qvcache_memory_active):
        axes[6].plot(qvcache_iterations, qvcache_memory_active, linewidth=0.7, color='#7209B7', linestyle='-', label='With QVCache')
    axes[6].set_ylabel('Vectors', fontsize=6)
    axes[6].set_title('Memory Active Vectors', fontsize=6, fontweight='bold', pad=2)
    axes[6].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    if qvcache_memory_active and any(x is not None for x in qvcache_memory_active):
        axes[6].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=1)
    
    # 8. Recall All - Both (with dynamic scale)
    if backend_recall and any(x is not None for x in backend_recall):
        axes[7].plot(backend_iterations, backend_recall, linewidth=0.7, color='#17A2B8', linestyle='--', label='Backend only')
    if qvcache_recall and any(x is not None for x in qvcache_recall):
        axes[7].plot(qvcache_iterations, qvcache_recall, linewidth=0.7, color='#17A2B8', linestyle='-', label='With QVCache')
    axes[7].set_xlabel('Splits', fontsize=6, labelpad=4)
    axes[7].set_ylabel('Recall', fontsize=6)
    
    # Dynamic scale for recall based on actual data range
    all_recall_values = []
    if backend_recall:
        all_recall_values.extend([x for x in backend_recall if x is not None])
    if qvcache_recall:
        all_recall_values.extend([x for x in qvcache_recall if x is not None])
    if all_recall_values:
        min_recall = min(all_recall_values)
        max_recall = max(all_recall_values)
        # Add 5% padding on each side, but ensure we don't go below 0
        range_recall = max_recall - min_recall
        padding = max(range_recall * 0.05, 0.01)  # At least 1% padding
        y_min = max(0, min_recall - padding)
        y_max = min(1.05, max_recall + padding)
        axes[7].set_ylim([y_min, y_max])
    else:
        axes[7].set_ylim([0, 1.05])
    
    axes[7].set_title('Recall', fontsize=6, fontweight='bold', pad=2)
    axes[7].grid(True, alpha=0.25, linestyle='--', linewidth=0.35)
    if (backend_recall and any(x is not None for x in backend_recall)) or \
       (qvcache_recall and any(x is not None for x in qvcache_recall)):
        axes[7].legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=5, framealpha=0.9, ncol=2)
    
    plt.savefig(output_dir / 'all_metrics.pdf', bbox_inches='tight', facecolor='white', format='pdf', pad_inches=0.05)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize and compare backend and QVCache experiment logs')
    parser.add_argument('--backend_log', type=str, default=None, help='Path to backend-only log file')
    parser.add_argument('--qvcache_log', type=str, default=None, help='Path to QVCache log file')
    parser.add_argument('--log', type=str, default=None, help='Path to log file (legacy, for single file)')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of iterations to visualize (default: all)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Parse log files
    backend_metrics = []
    qvcache_metrics = []
    
    if args.backend_log:
        print(f"Parsing backend log file: {args.backend_log}")
        backend_metrics = parse_log_file(args.backend_log)
        print(f"Found {len(backend_metrics)} split_metrics events in backend log")
    
    if args.qvcache_log:
        print(f"Parsing QVCache log file: {args.qvcache_log}")
        qvcache_metrics = parse_log_file(args.qvcache_log)
        print(f"Found {len(qvcache_metrics)} split_metrics events in QVCache log")
    
    # Legacy support: if only --log is provided, use it as QVCache log
    if args.log and not args.qvcache_log and not args.backend_log:
        print(f"Parsing log file: {args.log}")
        qvcache_metrics = parse_log_file(args.log)
        print(f"Found {len(qvcache_metrics)} split_metrics events")
    
    if not backend_metrics and not qvcache_metrics:
        print("No metrics found in log files!")
        return
    
    # Limit iterations if specified
    if args.max_iterations is not None and args.max_iterations > 0:
        if backend_metrics:
            backend_metrics = backend_metrics[:args.max_iterations]
            print(f"Limited backend metrics to first {len(backend_metrics)} iterations")
        if qvcache_metrics:
            qvcache_metrics = qvcache_metrics[:args.max_iterations]
            print(f"Limited QVCache metrics to first {len(qvcache_metrics)} iterations")
    
    print("Generating individual plots...")
    create_individual_plots(backend_metrics, qvcache_metrics, output_dir)
    
    print(f"\nPlots saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

