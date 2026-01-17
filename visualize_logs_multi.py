#!/usr/bin/env python3
"""
Multi-log visualization script for QVCache experiment logs.
Accepts multiple log files with custom legend names for each.
Produces publication-ready plots for paper submission.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter, LogFormatter, LogLocator, FixedLocator, NullLocator, MultipleLocator
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
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
    """Parse JSON log file and extract split_metrics or window_metrics events."""
    metrics = []
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
                if data.get('event') in ('split_metrics', 'window_metrics'):
                    metrics.append(data)
            except json.JSONDecodeError:
                continue
    return metrics


def extract_deviation_factor(log_path: str) -> float:
    """Extract deviation_factor from log file params event."""
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
                if data.get('event') == 'params' and 'deviation_factor' in data:
                    return data['deviation_factor']
            except json.JSONDecodeError:
                continue
    return None


def extract_number_of_mini_indexes(log_path: str) -> int:
    """Extract number_of_mini_indexes from log file params event."""
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
                if data.get('event') == 'params' and 'number_of_mini_indexes' in data:
                    return data['number_of_mini_indexes']
            except json.JSONDecodeError:
                continue
    return None


def format_legend_name(legend_name: str, deviation_factor: float = None, number_of_mini_indexes: int = None, format_type: str = None) -> str:
    """Format legend name for experiments.
    
    If format_type is "pca_dim", format as 'd_reduced' with subscript.
    If format_type is "buckets_per_dim", format as 'n_buckets' with subscript.
    If format_type is "cache_size", format as 'Total Cache Size = {value}'.
    If format_type is "noise_ratio", format as 'η = {value}'.
    If number_of_mini_indexes is provided, format as 'n_mini-index' with subscript.
    If deviation_factor is provided, format as 'D = {value}'.
    Otherwise, try to parse legend_name as a numeric value.
    Handles cases like "0.25", "025" (interpreted as 0.25), "05" (interpreted as 0.5).
    """
    # Handle PCA-specific format types
    if format_type == "pca_dim":
        try:
            value = int(legend_name) if legend_name.isdigit() else float(legend_name)
            return f'${{\\text{{d}}}}_{{\\text{{reduced}}}}$ = {value}'
        except (ValueError, AttributeError):
            return legend_name
    
    if format_type == "buckets_per_dim":
        try:
            value = int(legend_name) if legend_name.isdigit() else float(legend_name)
            return f'${{\\text{{n}}}}_{{\\text{{buckets}}}}$ = {value}'
        except (ValueError, AttributeError):
            return legend_name
    
    if format_type == "cache_size":
        # Format as "Cache Capacity = {value}" preserving the original format (e.g., "15k", "30k")
        return f'Cache Capacity = {legend_name}'
    
    if format_type == "noise_ratio":
        try:
            # Parse the value, handling both decimal and zero-padded formats
            if '.' in legend_name:
                value = float(legend_name)
            elif legend_name.isdigit() and len(legend_name) > 1 and legend_name.startswith('0'):
                # Handle zero-padded numbers like "025" -> 0.25, "05" -> 0.5, "005" -> 0.05
                leading_zeros = 0
                for char in legend_name:
                    if char == '0':
                        leading_zeros += 1
                    else:
                        break
                if leading_zeros > 0:
                    remaining_digits = legend_name[leading_zeros:]
                    if remaining_digits:
                        value = float('0.' + '0' * (leading_zeros - 1) + remaining_digits)
                    else:
                        value = 0.0
                else:
                    value = float(legend_name)
            else:
                value = float(legend_name)
            # Format as η = value using LaTeX notation
            return f'$\\eta$ = {value}'
        except (ValueError, AttributeError):
            return legend_name
    
    if number_of_mini_indexes is not None:
        # Use LaTeX subscript notation: n_mini-index = value
        return f'${{\\text{{n}}}}_{{\\text{{mini-index}}}}$ = {number_of_mini_indexes}'
    
    if deviation_factor is not None:
        return f'D = {deviation_factor}'
    
    # Try to parse legend_name as a numeric value
    try:
        # If it contains a decimal point, parse directly
        if '.' in legend_name:
            value = float(legend_name)
            return f'D = {value}'
        
        # Handle zero-padded integers like "025" -> 0.25, "05" -> 0.5, "005" -> 0.05
        if legend_name.isdigit():
            if len(legend_name) > 1 and legend_name.startswith('0'):
                # Convert zero-padded number to decimal
                # "025" -> 0.25, "05" -> 0.5, "005" -> 0.05
                # Count leading zeros
                leading_zeros = 0
                for char in legend_name:
                    if char == '0':
                        leading_zeros += 1
                    else:
                        break
                
                if leading_zeros > 0:
                    # Create decimal: "025" -> "0.25", "005" -> "0.05"
                    remaining_digits = legend_name[leading_zeros:]
                    if remaining_digits:
                        value = float('0.' + '0' * (leading_zeros - 1) + remaining_digits)
                    else:
                        value = 0.0
                    return f'D = {value}'
                else:
                    # Regular integer starting with 0 but not zero-padded (shouldn't happen)
                    value = float(legend_name)
                    return f'D = {value}'
            else:
                # Regular integer (no leading zero) - treat as mini index if no deviation_factor
                value = int(legend_name)
                # If it's a simple integer and no deviation_factor, format as mini index
                return f'${{\\text{{n}}}}_{{\\text{{mini-index}}}}$ = {value}'
    except (ValueError, AttributeError):
        pass
    
    # If not a numeric value, return as-is
    return legend_name


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


# Color palette for multiple series - high contrast colors for better visibility
COLORS = ['#E63946', '#0066FF', '#228B22', '#FF6600', '#9900FF', '#FFCC00', '#00CCFF', 
          '#FF00AA', '#0000CC', '#CC0000', '#00CC00', '#FF8800', '#AA00FF', '#FFD700']


def create_individual_plots(log_data_list: List[Tuple[List[Dict], str, str]], output_dir: Path, tick_interval: int = 3, max_legend_width: int = None):
    """
    Create individual plot files for each metric with comparison.
    
    Args:
        log_data_list: List of tuples (metrics, legend_name, color)
        tick_interval: Interval between x-axis ticks (default: 3)
        max_legend_width: Maximum number of legend items per row (default: 3, or None to use default behavior)
    """
    # Extract data for all log files
    all_metrics_data = {}
    all_iterations = {}
    
    for idx, (metrics, legend_name, color) in enumerate(log_data_list):
        iterations = np.arange(len(metrics)) if metrics else []
        all_iterations[legend_name] = iterations
        
        # Extract all metrics
        all_metrics_data[legend_name] = {
            'avg_latency_ms': extract_metric(metrics, 'avg_latency_ms'),
            'avg_hit_latency_ms': extract_metric(metrics, 'avg_hit_latency_ms', None),
            'p50': [m.get('tail_latency_ms', {}).get('p50', None) if 'tail_latency_ms' in m else None for m in metrics],
            'p99': [m.get('tail_latency_ms', {}).get('p99', None) if 'tail_latency_ms' in m else None for m in metrics],
            'qps': extract_metric(metrics, 'qps'),
            'recall_all': extract_metric(metrics, 'recall_all'),
            'hit_ratio': extract_metric(metrics, 'hit_ratio', None),
            'memory_active_vectors': extract_metric(metrics, 'memory_active_vectors', None),
            'pca_active_regions': extract_metric(metrics, 'pca_active_regions', None),
        }
    
    # Define plot configurations: (metric_key, ylabel, title, color_override, ylim, filename, show_legend, use_log_scale)
    plot_configs = [
        ('hit_ratio', 'Hit Ratio', 'Cache Hit Ratio', '#1f4e79', (-0.05, 1.05), 'hit_ratio.pdf', True, False),
        ('avg_hit_latency_ms', 'Latency (ms)', 'Average Hit Latency', '#548235', None, 'avg_hit_latency.pdf', True, False),
        ('p50', 'Latency (ms)', 'Latency (P50)', '#0066cc', None, 'p50_latency.pdf', True, False),
        ('qps', 'QPS', 'Query Throughput', '#bf8f00', None, 'qps.pdf', True, False),
        ('memory_active_vectors', 'Vectors', 'Vectors In Cache', '#5b2c6f', None, 'memory_active_vectors.pdf', True, False),
        ('pca_active_regions', 'Regions', 'PCA Active Regions', '#8B4513', None, 'pca_active_regions.pdf', True, False),
        ('recall_all', '10-Recall@10', '10-Recall@10', '#2e75b6', None, 'recall.pdf', True, False),
    ]
    
    for metric_key, ylabel, title, default_color, ylim, filename, show_legend, use_log_scale in plot_configs:
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
            ax.yaxis.set_major_formatter(FuncFormatter(format_log_tick))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='y', labelsize=6, pad=2, which='major')
            ax.tick_params(axis='y', which='minor', length=0)
        
        # Plot data for each log file
        colors_used = []
        for idx, (metrics, legend_name, color) in enumerate(log_data_list):
            data = all_metrics_data[legend_name][metric_key]
            iterations = all_iterations[legend_name]
            
            if data and any(x is not None for x in data):
                # For hit ratio, allow 0 values (0% hit rate is valid)
                # For other metrics, filter out 0 values
                if filename == 'hit_ratio.pdf':
                    valid_indices = [i for i, val in enumerate(data) if val is not None and val >= 0]
                else:
                    valid_indices = [i for i, val in enumerate(data) if val is not None and val > 0]
                
                if valid_indices:
                    valid_data = [data[i] for i in valid_indices]
                    valid_iterations = [iterations[i] for i in valid_indices]
                    
                    # Use provided color or default from palette
                    plot_color = color if color else COLORS[idx % len(COLORS)]
                    colors_used.append(plot_color)
                    
                    # Use different line styles for variety
                    linestyle = '-' if idx == 0 else ('--' if idx == 1 else (':' if idx == 2 else '-.'))
                    
                    ax.plot(valid_iterations, valid_data, linewidth=1.0, color=plot_color, 
                           linestyle=linestyle,
                           label=legend_name, alpha=0.9, zorder=10)
        
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlabel('Window Step', fontsize=7, labelpad=4)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=8)
        
        # Set x-axis limits to match data range (no extra tick at the end)
        # Find the maximum iteration value from actual plotted data
        max_iteration = -1
        for metrics, legend_name, _ in log_data_list:
            data = all_metrics_data[legend_name][metric_key]
            if filename == 'hit_ratio.pdf':
                valid_indices = [i for i, val in enumerate(data) if val is not None and val >= 0]
            else:
                valid_indices = [i for i, val in enumerate(data) if val is not None and val > 0]
            if valid_indices:
                max_iteration = max(max_iteration, max(valid_indices))
        
        # If no data, use length of iterations arrays
        if max_iteration < 0:
            max_iteration = max([len(all_iterations[legend_name]) for _, legend_name, _ in log_data_list]) - 1 if log_data_list else 0
        
        # Set x-axis limits to exactly match data range (no extra tick)
        if max_iteration >= 0:
            ax.set_xlim([-0.5, max_iteration + 0.5])
        
        # Set x-axis to show one tick per tick_interval records, plus the last tick if not a multiple of tick_interval
        if max_iteration >= 0:
            tick_positions = list(range(0, max_iteration + 1, tick_interval))
            if max_iteration % tick_interval != 0 and max_iteration not in tick_positions:
                tick_positions.append(max_iteration)
            ax.xaxis.set_major_locator(FixedLocator(tick_positions))
            # Set minor locator to show grid lines at every window step
            ax.xaxis.set_minor_locator(MultipleLocator(base=1))
        
        ax.tick_params(axis='x', labelsize=6, pad=2)
        ax.tick_params(axis='x', which='minor', length=0)  # Hide minor tick marks, keep grid lines
        if not use_log_scale:
            ax.tick_params(axis='y', labelsize=6, pad=2)
        
        # Special handling for hit ratio: set custom y-axis ticks
        if filename == 'hit_ratio.pdf':
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'], fontsize=6)
        
        # Special handling for P50 latency
        if filename == 'p50_latency.pdf':
            all_values = []
            for metrics, legend_name, _ in log_data_list:
                data = all_metrics_data[legend_name]['p50']
                all_values.extend([x for x in data if x is not None])
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                top_padding = max((max_val - 0) * 0.05, 0.01)
                bottom_padding = max((max_val - 0) * 0.08, 0.01)
                ax.set_ylim([-bottom_padding, max_val + top_padding])
                # Set custom ticks
                tick_locations = [min_val]
                num_ticks = 3
                for i in range(1, num_ticks + 1):
                    tick_val = min_val + (max_val - min_val) * i / (num_ticks + 1)
                    tick_locations.append(tick_val)
                tick_locations.append(max_val)
                tick_locations = sorted(set(tick_locations))
                tick_labels = [f'{tick:.2f}' if tick < 1 else f'{tick:.1f}' for tick in tick_locations]
                ax.set_yticks(tick_locations)
                ax.set_yticklabels(tick_labels, fontsize=6)
        
        # Special handling for QPS
        if filename == 'qps.pdf':
            all_values = []
            for metrics, legend_name, _ in log_data_list:
                data = all_metrics_data[legend_name]['qps']
                all_values.extend([x for x in data if x is not None])
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                range_val = max_val - min_val
                padding = max(range_val * 0.05, 0.01)
                y_min = max(0, min_val - padding)
                y_max = max_val + padding
                ax.set_ylim([y_min, y_max])
                # Set custom ticks
                num_ticks = 3
                tick_locations = []
                for i in range(num_ticks + 1):
                    tick_val = y_min + (y_max - y_min) * i / num_ticks
                    tick_locations.append(tick_val)
                tick_locations = sorted(set([t for t in tick_locations if abs(t) > 0.01]))
                tick_labels = [f'{tick:.0f}' if tick >= 1 else f'{tick:.2f}' for tick in tick_locations]
                ax.set_yticks(tick_locations)
                ax.set_yticklabels(tick_labels, fontsize=6)
        
        # Special handling for recall: use dynamic scale based on data range
        if filename == 'recall.pdf':
            all_values = []
            for metrics, legend_name, _ in log_data_list:
                data = all_metrics_data[legend_name]['recall_all']
                all_values.extend([x for x in data if x is not None])
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                range_val = max_val - min_val
                padding = max(range_val * 0.05, 0.01)
                y_min = max(0, min_val - padding)
                y_max = min(1.05, max_val + padding)
                ax.set_ylim([y_min, y_max])
        elif ylim:
            ax.set_ylim(ylim)
        
        # Enable grid for both major and minor ticks to show vertical lines at every window step
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4, color='gray', zorder=1, which='both')
        # Make minor grid lines lighter for vertical lines
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.2, color='gray', zorder=1, which='minor', axis='x')
        
        if show_legend and any(all_metrics_data[legend_name][metric_key] for _, legend_name, _ in log_data_list):
            num_series = sum(1 for _, legend_name, _ in log_data_list if all_metrics_data[legend_name][metric_key])
            # Maximum items per row (use max_legend_width if provided, otherwise default to 3)
            max_width = max_legend_width if max_legend_width is not None else 3
            ncol = min(max_width, num_series)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=6, framealpha=0.95, 
                     frameon=True, ncol=ncol, edgecolor='gray', fancybox=False)
        
        plt.savefig(output_dir / filename, bbox_inches='tight', facecolor='white', format='pdf', 
                   pad_inches=0.05, dpi=300, transparent=False)
        plt.close()
        print(f"  ✓ {filename}")


def create_combined_plot(log_data_list: List[Tuple[List[Dict], str, str]], output_dir: Path, 
                         metric_keys: List[str], combined_filename: str = 'combined.pdf', tick_interval: int = 3, max_legend_width: int = None):
    """
    Create a combined plot with multiple subplots arranged vertically.
    
    Args:
        log_data_list: List of tuples (metrics, legend_name, color)
        output_dir: Output directory for the plot
        metric_keys: List of metric keys to include in the combined plot
        combined_filename: Filename for the combined plot
        tick_interval: Interval between x-axis ticks (default: 3)
        max_legend_width: Maximum number of legend items per row (default: 3, or None to use default behavior)
    """
    # Extract data for all log files
    all_metrics_data = {}
    all_iterations = {}
    
    for idx, (metrics, legend_name, color) in enumerate(log_data_list):
        iterations = np.arange(len(metrics)) if metrics else []
        all_iterations[legend_name] = iterations
        
        # Extract all metrics
        all_metrics_data[legend_name] = {
            'avg_latency_ms': extract_metric(metrics, 'avg_latency_ms'),
            'avg_hit_latency_ms': extract_metric(metrics, 'avg_hit_latency_ms', None),
            'p50': [m.get('tail_latency_ms', {}).get('p50', None) if 'tail_latency_ms' in m else None for m in metrics],
            'p99': [m.get('tail_latency_ms', {}).get('p99', None) if 'tail_latency_ms' in m else None for m in metrics],
            'qps': extract_metric(metrics, 'qps'),
            'recall_all': extract_metric(metrics, 'recall_all'),
            'hit_ratio': extract_metric(metrics, 'hit_ratio', None),
            'memory_active_vectors': extract_metric(metrics, 'memory_active_vectors', None),
            'pca_active_regions': extract_metric(metrics, 'pca_active_regions', None),
        }
    
    # Map metric keys to their display info
    metric_info = {
        'hit_ratio': ('Hit Ratio', 'Hit Ratio', (-0.05, 1.05), False),
        'avg_hit_latency_ms': ('Latency (ms)', 'Average Hit Latency', None, False),
        'p50': ('Latency (ms)', 'Latency (P50)', None, False),
        'qps': ('QPS', 'Query Throughput', None, False),
        'memory_active_vectors': ('Vectors', 'Vectors In Cache', None, False),
        'pca_active_regions': ('Regions', 'PCA Active Regions', None, False),
        'recall_all': ('10-Recall@10', '10-Recall@10', None, False),
    }
    
    num_subplots = len(metric_keys)
    if num_subplots == 0:
        return
    
    # Create figure with vertical subplots
    # Each subplot should be the same size as individual plots (3.0, 2.2)
    # Individual plots use figsize=(3.0, 2.2)
    # To make each subplot the same size, we need to account for:
    # - Each subplot height: 2.2
    # - Spacing between subplots (hspace)
    # - Top and bottom margins
    individual_height = 2.2
    # Use hspace to control spacing between subplots
    # hspace is the height of the space between subplots as a fraction of average subplot height
    hspace = 0.2
    # Calculate total figure height - make subplots smaller/more compact
    # Reduce the height per subplot to make the combined plot more compact
    total_height = individual_height * num_subplots * 0.85  # 0.85 makes subplots smaller
    fig, axes = plt.subplots(num_subplots, 1, figsize=(3.0, total_height))
    if num_subplots == 1:
        axes = [axes]
    
    # Adjust spacing to match individual plots
    # Individual plots use: left=0.18, right=0.95, top=0.85, bottom=0.25 (no legend) or bottom=0.50 (with legend)
    # For combined plot, use similar horizontal margins, adjust vertical margins for multiple subplots
    fig.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15, hspace=hspace)
    
    for subplot_idx, metric_key in enumerate(metric_keys):
        if metric_key not in metric_info:
            continue
        
        ax = axes[subplot_idx]
        ylabel, title, ylim, use_log_scale = metric_info[metric_key]
        
        # Set log scale if requested
        if use_log_scale:
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(FuncFormatter(format_log_tick))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='y', labelsize=6, pad=2, which='major')
            ax.tick_params(axis='y', which='minor', length=0)
        
        # Plot data for each log file
        for idx, (metrics, legend_name, color) in enumerate(log_data_list):
            data = all_metrics_data[legend_name][metric_key]
            iterations = all_iterations[legend_name]
            
            if data and any(x is not None for x in data):
                # For hit ratio, allow 0 values (0% hit rate is valid)
                # For other metrics, filter out 0 values
                if metric_key == 'hit_ratio':
                    valid_indices = [i for i, val in enumerate(data) if val is not None and val >= 0]
                else:
                    valid_indices = [i for i, val in enumerate(data) if val is not None and val > 0]
                
                if valid_indices:
                    valid_data = [data[i] for i in valid_indices]
                    valid_iterations = [iterations[i] for i in valid_indices]
                    
                    # Use provided color or default from palette
                    plot_color = color if color else COLORS[idx % len(COLORS)]
                    
                    # Use different line styles for variety
                    linestyle = '-' if idx == 0 else ('--' if idx == 1 else (':' if idx == 2 else '-.'))
                    
                    ax.plot(valid_iterations, valid_data, linewidth=1.0, color=plot_color, 
                           linestyle=linestyle,
                           label=legend_name, alpha=0.9, zorder=10)
        
        ax.set_ylabel(ylabel, fontsize=7)
        if subplot_idx == num_subplots - 1:
            ax.set_xlabel('Window Step', fontsize=7, labelpad=4)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=8)
        
        # Set x-axis limits to match data range (no extra tick at the end)
        # Find the maximum iteration value from actual plotted data
        max_iteration = -1
        for metrics, legend_name, _ in log_data_list:
            data = all_metrics_data[legend_name][metric_key]
            if metric_key == 'hit_ratio':
                valid_indices = [i for i, val in enumerate(data) if val is not None and val >= 0]
            else:
                valid_indices = [i for i, val in enumerate(data) if val is not None and val > 0]
            if valid_indices:
                max_iteration = max(max_iteration, max(valid_indices))
        
        # If no data, use length of iterations arrays
        if max_iteration < 0:
            max_iteration = max([len(all_iterations[legend_name]) for _, legend_name, _ in log_data_list]) - 1 if log_data_list else 0
        
        # Set x-axis limits to exactly match data range (no extra tick)
        if max_iteration >= 0:
            ax.set_xlim([-0.5, max_iteration + 0.5])
        
        # Set x-axis to show one tick per tick_interval records, plus the last tick if not a multiple of tick_interval
        # (applies to all subplots via sharex if needed)
        if max_iteration >= 0:
            tick_positions = list(range(0, max_iteration + 1, tick_interval))
            if max_iteration % tick_interval != 0 and max_iteration not in tick_positions:
                tick_positions.append(max_iteration)
            ax.xaxis.set_major_locator(FixedLocator(tick_positions))
            # Set minor locator to show grid lines at every window step
            ax.xaxis.set_minor_locator(MultipleLocator(base=1))
            # Hide minor tick marks, keep grid lines
            ax.tick_params(axis='x', which='minor', length=0)
        
        ax.tick_params(axis='x', labelsize=6, pad=2)
        if not use_log_scale:
            ax.tick_params(axis='y', labelsize=6, pad=2)
        
        # Special handling for hit ratio
        if metric_key == 'hit_ratio':
            ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
            ax.set_yticklabels(['0.0', '0.25', '0.50', '0.75', '1.0'], fontsize=6)
        
        # Set ylim if specified
        if ylim:
            ax.set_ylim(ylim)
        
        # Enable grid for both major and minor ticks to show vertical lines at every window step
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.4, color='gray', zorder=1, which='both')
        # Make minor grid lines lighter for vertical lines
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.2, color='gray', zorder=1, which='minor', axis='x')
        
        # Add legend only to the last subplot
        if subplot_idx == num_subplots - 1:
            num_series = sum(1 for _, legend_name, _ in log_data_list if all_metrics_data[legend_name][metric_key])
            if num_series > 0:
                # Maximum items per row (use max_legend_width if provided, otherwise default to 3)
                max_width = max_legend_width if max_legend_width is not None else 3
                ncol = min(max_width, num_series)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fontsize=6, framealpha=0.95, 
                         frameon=True, ncol=ncol, edgecolor='gray', fancybox=False)
    
    plt.savefig(output_dir / combined_filename, bbox_inches='tight', facecolor='white', format='pdf', 
               pad_inches=0.05, dpi=300, transparent=False)
    plt.close()
    print(f"  ✓ {combined_filename}")


def main():
    parser = argparse.ArgumentParser(description='Visualize multiple QVCache experiment logs with custom legend names')
    parser.add_argument('--logs', type=str, nargs='+', required=True, 
                       help='Paths to log files (space-separated, e.g., --logs log1.log log2.log log3.log)')
    parser.add_argument('--legends', type=str, nargs='+', required=True,
                       help='Legend names for each log file (space-separated, must match number of log files)')
    parser.add_argument('--colors', type=str, nargs='*', default=None,
                       help='Optional colors for each log file (hex codes, e.g., #FF0000). If not provided, uses default palette')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--max_iterations', type=int, default=None, 
                       help='Maximum number of iterations to visualize (default: all)')
    parser.add_argument('--format-type', type=str, default=None,
                       choices=['pca_dim', 'buckets_per_dim', 'cache_size', 'noise_ratio'],
                       help='Format type for legend names: "pca_dim" for PCA dimension, "buckets_per_dim" for buckets per dimension, "cache_size" for total cache size, "noise_ratio" for noise ratio (η)')
    parser.add_argument('--tick-interval', type=int, default=3,
                       help='Interval between x-axis ticks (default: 3)')
    parser.add_argument('--max-legend-width', type=int, default=None,
                       help='Maximum number of legend items per row (default: 3, or None to use default behavior)')
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.logs) != len(args.legends):
        print(f"Error: Number of log files ({len(args.logs)}) must match number of legend names ({len(args.legends)})")
        return
    
    if args.colors and len(args.colors) != len(args.logs):
        print(f"Warning: Number of colors ({len(args.colors) if args.colors else 0}) doesn't match number of logs ({len(args.logs)}). Using default palette.")
        args.colors = None
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse log files
    log_data_list = []
    for idx, log_path in enumerate(args.logs):
        print(f"Parsing log file {idx+1}/{len(args.logs)}: {log_path}")
        metrics = parse_log_file(log_path)
        print(f"Found {len(metrics)} split_metrics events")
        
        # Limit iterations if specified
        if args.max_iterations is not None and args.max_iterations > 0:
            metrics = metrics[:args.max_iterations]
            print(f"Limited to first {len(metrics)} iterations")
        
        # Try to extract parameters from log file
        deviation_factor = extract_deviation_factor(log_path)
        number_of_mini_indexes = extract_number_of_mini_indexes(log_path)
        
        # Format legend name (prioritize format_type, then number_of_mini_indexes, then deviation_factor, then parse legend_name)
        legend_name = args.legends[idx]
        formatted_legend = format_legend_name(legend_name, deviation_factor, number_of_mini_indexes, args.format_type)
        
        color = args.colors[idx] if args.colors else None
        log_data_list.append((metrics, formatted_legend, color))
    
    if not any(metrics for metrics, _, _ in log_data_list):
        print("No metrics found in any log files!")
        return
    
    print("\nGenerating individual plots...")
    create_individual_plots(log_data_list, output_dir, tick_interval=args.tick_interval, max_legend_width=args.max_legend_width)
    
    print(f"\nPlots saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

