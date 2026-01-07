#!/usr/bin/env python3
"""
Multi-log visualization script for QVCache experiment logs with K value support.
Accepts multiple log files with custom legend names for each.
Supports pairing backend and qvcache logs with matching colors.
Produces publication-ready plots for paper submission.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter, LogFormatter, LogLocator, FixedLocator, NullLocator, MultipleLocator
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse
import re

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


def parse_log_file(log_path: str) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Parse JSON log file and extract split_metrics or window_metrics events. Also extract K value."""
    metrics = []
    k_value = None
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
                if data.get('event') in ('split_metrics', 'window_metrics'):
                    metrics.append(data)
                    # Extract K value from first metrics event
                    if k_value is None and 'K' in data:
                        k_value = data['K']
            except json.JSONDecodeError:
                continue
    return metrics, k_value


def extract_k_from_filename(log_path: str) -> Optional[int]:
    """Extract k value from log filename (e.g., backend_1.log -> 1, qvcache_10.log -> 10)."""
    filename = Path(log_path).name
    # Match pattern like backend_1.log, qvcache_10.log, etc.
    match = re.search(r'_(\d+)\.log$', filename)
    if match:
        return int(match.group(1))
    return None


def detect_log_type(log_path: str) -> str:
    """Detect if log is backend or qvcache based on filename."""
    filename = Path(log_path).name.lower()
    if 'backend' in filename:
        return 'backend'
    elif 'qvcache' in filename:
        return 'qvcache'
    return 'unknown'


def format_legend_name_for_comparison(log_path: str, k_value: Optional[int], log_type: str) -> str:
    """Format legend name for all_comparison mode."""
    if k_value is None:
        k_value = extract_k_from_filename(log_path)
    
    if log_type == 'backend':
        if k_value is not None:
            return f'Backend only (k={k_value})'
        else:
            return 'Backend only'
    elif log_type == 'qvcache':
        if k_value is not None:
            return f'With QVCache (k={k_value})'
        else:
            return 'With QVCache'
    else:
        # Fallback: use filename
        return Path(log_path).stem


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


def get_color_and_linestyle_for_comparison(log_data_list: List[Tuple[List[Dict], str, str, Optional[int], str]]) -> List[Tuple[str, str]]:
    """
    Get color and linestyle for all_comparison mode.
    Pairs backend and qvcache with same k value using same color, but different linestyles.
    
    Returns list of (color, linestyle) tuples.
    """
    # Group by k value
    k_groups = {}
    for idx, (metrics, legend_name, color, k_value, log_type) in enumerate(log_data_list):
        # Use k_value directly, or extract from legend name as fallback
        if k_value is None:
            # Try to extract from legend name (e.g., "Backend only (k=1)" -> 1)
            match = re.search(r'k=(\d+)', legend_name)
            if match:
                k_value = int(match.group(1))
            else:
                k_value = idx  # Fallback to index
        
        if k_value not in k_groups:
            k_groups[k_value] = []
        k_groups[k_value].append((idx, log_type))
    
    # Assign colors: one color per k value
    k_colors = {}
    sorted_k_values = sorted([k for k in k_groups.keys() if k is not None])
    for i, k_val in enumerate(sorted_k_values):
        k_colors[k_val] = COLORS[i % len(COLORS)]
    
    # Build result list
    result = []
    for idx, (metrics, legend_name, color, k_value, log_type) in enumerate(log_data_list):
        # Use k_value directly, or extract from legend name as fallback
        if k_value is None:
            # Try to extract from legend name (e.g., "Backend only (k=1)" -> 1)
            match = re.search(r'k=(\d+)', legend_name)
            if match:
                k_value = int(match.group(1))
            else:
                k_value = idx  # Fallback to index
        
        assigned_color = k_colors.get(k_value, COLORS[idx % len(COLORS)])
        # Backend: dashed, QVCache: solid
        linestyle = '--' if log_type == 'backend' else '-'
        result.append((assigned_color, linestyle))
    
    return result


def create_individual_plots(log_data_list: List[Tuple[List[Dict], str, str, Optional[int], str]], 
                           output_dir: Path, tick_interval: int = 3, max_legend_width: int = None,
                           is_all_comparison: bool = False, k_values: List[Optional[int]] = None):
    """
    Create individual plot files for each metric with comparison.
    
    Args:
        log_data_list: List of tuples (metrics, legend_name, color, k_value, log_type)
        tick_interval: Interval between x-axis ticks (default: 3)
        max_legend_width: Maximum number of legend items per row (default: 3, or None to use default behavior)
        is_all_comparison: Whether this is an all_comparison plot (affects color/linestyle pairing)
        k_values: List of K values for each log (for dynamic recall labels)
    """
    # Extract data for all log files
    all_metrics_data = {}
    all_iterations = {}
    
    for idx, (metrics, legend_name, color, k_value, log_type) in enumerate(log_data_list):
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
            'k_value': k_value,  # Store k value for recall label
        }
    
    # Get color and linestyle assignments for all_comparison mode
    color_linestyle_pairs = None
    if is_all_comparison:
        color_linestyle_pairs = get_color_and_linestyle_for_comparison(log_data_list)
    
    # Determine recall label - y-axis uses "k-Recall@k" format, title is just "Recall"
    recall_ylabel = 'k-Recall@k'
    recall_title = 'Recall'
    
    plot_configs = [
        ('hit_ratio', 'Hit Ratio', 'Cache Hit Ratio', '#1f4e79', (-0.05, 1.05), 'hit_ratio.pdf', True, False),
        ('avg_hit_latency_ms', 'Latency (ms)', 'Average Hit Latency', '#548235', None, 'avg_hit_latency.pdf', True, False),
        ('p50', 'Latency (ms)', 'Latency (P50)', '#0066cc', None, 'p50_latency.pdf', True, False),
        ('qps', 'QPS', 'Query Throughput', '#bf8f00', None, 'qps.pdf', True, False),
        ('memory_active_vectors', 'Vectors', 'Vectors In Cache', '#5b2c6f', None, 'memory_active_vectors.pdf', True, False),
        ('pca_active_regions', 'Regions', 'PCA Active Regions', '#8B4513', None, 'pca_active_regions.pdf', True, False),
        ('recall_all', recall_ylabel, recall_title, '#2e75b6', None, 'recall.pdf', True, False),
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
        for idx, (metrics, legend_name, color, k_value, log_type) in enumerate(log_data_list):
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
                    
                    # Use color and linestyle from pairing if all_comparison mode
                    if is_all_comparison and color_linestyle_pairs:
                        plot_color, linestyle = color_linestyle_pairs[idx]
                    else:
                        # Use provided color or default from palette
                        plot_color = color if color else COLORS[idx % len(COLORS)]
                        # Use different line styles for variety
                        linestyle = '-' if idx == 0 else ('--' if idx == 1 else (':' if idx == 2 else '-.'))
                    
                    colors_used.append(plot_color)
                    
                    ax.plot(valid_iterations, valid_data, linewidth=1.0, color=plot_color, 
                           linestyle=linestyle,
                           label=legend_name, alpha=0.9, zorder=10)
        
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlabel('iteration', fontsize=7, labelpad=4)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=8)
        
        # Set x-axis limits to match data range (no extra tick at the end)
        # Find the maximum iteration value from actual plotted data
        max_iteration = -1
        for metrics, legend_name, _, _, _ in log_data_list:
            data = all_metrics_data[legend_name][metric_key]
            if filename == 'hit_ratio.pdf':
                valid_indices = [i for i, val in enumerate(data) if val is not None and val >= 0]
            else:
                valid_indices = [i for i, val in enumerate(data) if val is not None and val > 0]
            if valid_indices:
                max_iteration = max(max_iteration, max(valid_indices))
        
        # If no data, use length of iterations arrays
        if max_iteration < 0:
            max_iteration = max([len(all_iterations[legend_name]) for _, legend_name, _, _, _ in log_data_list]) - 1 if log_data_list else 0
        
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
            for metrics, legend_name, _, _, _ in log_data_list:
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
            for metrics, legend_name, _, _, _ in log_data_list:
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
            for metrics, legend_name, _, _, _ in log_data_list:
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
        
        if show_legend and any(all_metrics_data[legend_name][metric_key] for _, legend_name, _, _, _ in log_data_list):
            num_series = sum(1 for _, legend_name, _, _, _ in log_data_list if all_metrics_data[legend_name][metric_key])
            # Maximum items per row (use max_legend_width if provided, otherwise default to 3)
            max_width = max_legend_width if max_legend_width is not None else 3
            
            # For all_comparison mode, set ncol to number of backend items to ensure backend fills top row
            if is_all_comparison:
                num_backend = sum(1 for _, legend_name, _, _, log_type in log_data_list 
                                 if log_type == 'backend' and all_metrics_data[legend_name][metric_key])
                if num_backend > 0:
                    ncol = num_backend  # This ensures backend items fill the first row
                else:
                    ncol = min(max_width, num_series)
            else:
                ncol = min(max_width, num_series)
            
            # Get handles and labels - they should already be in the correct order since log_data_list is reordered
            # But we need to ensure they match log_data_list order for all_comparison mode
            handles, labels = ax.get_legend_handles_labels()
            
            if is_all_comparison and len(handles) > 0:
                # Create ordered list based on log_data_list order (which is already backend first, qvcache second)
                # Map labels to handles
                label_to_handle = dict(zip(labels, handles))
                
                # Build handles and labels in the order they appear in log_data_list
                ordered_handles = []
                ordered_labels = []
                for metrics, legend_name, _, _, _ in log_data_list:
                    if legend_name in label_to_handle:
                        ordered_handles.append(label_to_handle[legend_name])
                        ordered_labels.append(legend_name)
                
                handles = ordered_handles
                labels = ordered_labels
            
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.40), fontsize=6, framealpha=0.95, 
                     frameon=True, ncol=ncol, edgecolor='gray', fancybox=False)
        
        plt.savefig(output_dir / filename, bbox_inches='tight', facecolor='white', format='pdf', 
                   pad_inches=0.05, dpi=300, transparent=False)
        plt.close()
        print(f"  ✓ {filename}")


def main():
    parser = argparse.ArgumentParser(description='Visualize multiple QVCache experiment logs with K value support and paired colors')
    parser.add_argument('--logs', type=str, nargs='+', required=True, 
                       help='Paths to log files (space-separated, e.g., --logs log1.log log2.log log3.log)')
    parser.add_argument('--legends', type=str, nargs='+', required=True,
                       help='Legend names for each log file (space-separated, must match number of log files). Use "auto" to auto-format for all_comparison mode.')
    parser.add_argument('--colors', type=str, nargs='*', default=None,
                       help='Optional colors for each log file (hex codes, e.g., #FF0000). If not provided, uses default palette or pairs for all_comparison')
    parser.add_argument('--output', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--max_iterations', type=int, default=None, 
                       help='Maximum number of iterations to visualize (default: all)')
    parser.add_argument('--tick-interval', type=int, default=3,
                       help='Interval between x-axis ticks (default: 3)')
    parser.add_argument('--max-legend-width', type=int, default=None,
                       help='Maximum number of legend items per row (default: 3, or None to use default behavior)')
    parser.add_argument('--all-comparison', action='store_true',
                       help='Enable all_comparison mode: auto-format legends and pair colors for backend/qvcache')
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.logs) != len(args.legends):
        print(f"Error: Number of log files ({len(args.logs)}) must match number of legend names ({len(args.legends)})")
        return
    
    if args.colors and len(args.colors) != len(args.logs):
        print(f"Warning: Number of colors ({len(args.colors) if args.colors else 0}) doesn't match number of logs ({len(args.logs)}). Using default palette.")
        args.colors = None
    
    # Detect if all_comparison mode (if --all-comparison flag is set, or if "auto" in legends, or if both backend and qvcache logs are present)
    has_backend = any('backend' in Path(log_path).name.lower() for log_path in args.logs)
    has_qvcache = any('qvcache' in Path(log_path).name.lower() for log_path in args.logs)
    is_all_comparison = args.all_comparison or any(leg == 'auto' for leg in args.legends) or (has_backend and has_qvcache)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Parse log files
    log_data_list = []
    k_values = []
    for idx, log_path in enumerate(args.logs):
        print(f"Parsing log file {idx+1}/{len(args.logs)}: {log_path}")
        metrics, k_value = parse_log_file(log_path)
        k_values.append(k_value)
        print(f"Found {len(metrics)} metrics events, K={k_value}")
        
        # Limit iterations if specified
        if args.max_iterations is not None and args.max_iterations > 0:
            metrics = metrics[:args.max_iterations]
            print(f"Limited to first {len(metrics)} iterations")
        
        # Determine log type
        log_type = detect_log_type(log_path)
        
        # Format legend name
        if is_all_comparison:
            # Auto-format if in all_comparison mode (unless explicitly provided)
            if args.legends[idx] == 'auto' or (has_backend and has_qvcache):
                legend_name = format_legend_name_for_comparison(log_path, k_value, log_type)
            else:
                legend_name = args.legends[idx]
        else:
            legend_name = args.legends[idx]
        
        color = args.colors[idx] if args.colors else None
        log_data_list.append((metrics, legend_name, color, k_value, log_type))
    
    # Reorder log_data_list for all_comparison mode: backend logs first, then qvcache logs
    # This ensures backend legends appear in the top row of the legend panel
    if is_all_comparison:
        # Separate by log type
        backend_logs = [data for data in log_data_list if data[4] == 'backend']
        qvcache_logs = [data for data in log_data_list if data[4] == 'qvcache']
        other_logs = [data for data in log_data_list if data[4] not in ['backend', 'qvcache']]
        
        # Sort backend and qvcache logs by k value to maintain consistent ordering (k=1, k=10, k=100)
        backend_logs.sort(key=lambda x: x[3] if x[3] is not None else float('inf'))
        qvcache_logs.sort(key=lambda x: x[3] if x[3] is not None else float('inf'))
        
        # Reorder: backend first, then qvcache, then others
        log_data_list = backend_logs + qvcache_logs + other_logs
        
        # Reorder k_values to match the new order
        k_values = [data[3] for data in log_data_list]
    
    if not any(metrics for metrics, _, _, _, _ in log_data_list):
        print("No metrics found in any log files!")
        return
    
    print("\nGenerating individual plots...")
    create_individual_plots(log_data_list, output_dir, tick_interval=args.tick_interval, 
                           max_legend_width=args.max_legend_width, is_all_comparison=is_all_comparison,
                           k_values=k_values)
    
    print(f"\nPlots saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

