#!/usr/bin/env python3
"""
Visualize neighbor overlap analysis results with average and standard deviation.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def parse_log_file(log_file):
    """Parse the analysis log file to extract noise ratios, averages, and std devs."""
    noise_ratios = []
    avg_overlaps = []
    std_devs = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    # Find the summary section
    in_summary = False
    for line in lines:
        if "Summary" in line:
            in_summary = True
            continue
        if in_summary and "Analysis complete" in line:
            break
        if in_summary and "|" in line and not "Noise Ratio" in line and not "---" in line:
            # Parse the data line
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    noise_ratio = float(parts[0])
                    avg_overlap = float(parts[1])
                    std_dev = float(parts[2])
                    noise_ratios.append(noise_ratio)
                    avg_overlaps.append(avg_overlap)
                    std_devs.append(std_dev)
                except ValueError:
                    continue
    
    return np.array(noise_ratios), np.array(avg_overlaps), np.array(std_devs)


def create_visualization(noise_ratios, avg_overlaps, std_devs, output_file="neighbor_overlap_analysis.pdf"):
    """Create a minimal, paper-ready visualization of the neighbor overlap analysis."""
    
    # Set up the figure with minimal style
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    
    # Color scheme - minimal and professional
    primary_color = '#1f77b4'  # Standard blue
    fill_color = '#ff7f0e'  # Orange
    
    # Main plot: Average overlap with confidence band (convert to percentage)
    ax.plot(noise_ratios, avg_overlaps * 100, 
             linewidth=1.5, color=primary_color, marker='o', 
             markersize=3, markeredgewidth=0.5,
             markeredgecolor=primary_color, label='Average Overlap', zorder=3)
    
    # Shaded confidence band (std dev) - convert to percentage
    upper_bound = np.minimum(avg_overlaps + std_devs, 1.0) * 100
    lower_bound = np.maximum(avg_overlaps - std_devs, 0.0) * 100
    ax.fill_between(noise_ratios, lower_bound, upper_bound, 
                     alpha=0.25, color=fill_color, label='Standard Deviation', zorder=1)
    
    # Annotate the value at 0.01 with x-axis projection
    mask_001 = np.abs(noise_ratios - 0.01) <= 0.001
    if np.any(mask_001):
        idx_001 = np.where(mask_001)[0][0]
        y_val_001 = avg_overlaps[idx_001] * 100
        x_val_001 = 0.01
        
        # Annotate the y-value with noise ratio included in the text
        ax.annotate(f'0.01: {y_val_001:.2f}%',
                    xy=(x_val_001, y_val_001),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'),
                    zorder=5)
    
    # Styling for main plot - minimal
    ax.set_xlabel('Noise Ratio', fontsize=10)
    ax.set_ylabel('Average Neighbor Overlap (%)', fontsize=10)
    ax.set_xlim(-0.01, max(noise_ratios) + 0.01)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=False, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(labelsize=9)
    
    # Overall figure styling
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {output_file}")
    
    # Also save as PNG for easier viewing
    png_file = output_file.replace('.pdf', '.png')
    plt.savefig(png_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {png_file}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize neighbor overlap analysis results"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="analysis.log",
        help="Path to the analysis log file (default: analysis.log)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="neighbor_overlap_analysis.pdf",
        help="Output file path (default: neighbor_overlap_analysis.pdf)"
    )
    
    args = parser.parse_args()
    
    # Parse the log file
    print(f"Reading data from: {args.log_file}")
    noise_ratios, avg_overlaps, std_devs = parse_log_file(args.log_file)
    
    print(f"Found {len(noise_ratios)} data points")
    print(f"Noise ratio range: {noise_ratios[0]:.3f} to {noise_ratios[-1]:.3f}")
    print(f"Average overlap range: {avg_overlaps.min():.4f} to {avg_overlaps.max():.4f}")
    print(f"Std dev range: {std_devs.min():.4f} to {std_devs.max():.4f}")
    
    # Create visualization
    create_visualization(noise_ratios, avg_overlaps, std_devs, args.output)
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

