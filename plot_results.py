#!/usr/bin/env python3
"""
Generate bar graph comparing multi_sort performance across different thread counts.
Reads CSV data from multi_sort_results.csv and creates a bar chart showing:
- Total time per thread count
- Phase 1 and Phase 2 times in separate colors
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def read_csv_data(csv_filename):
    """Read timing data from CSV file."""
    threads = []
    total_times = []
    phase1_times = []
    phase2_times = []
    
    if not os.path.exists(csv_filename):
        print(f"Error: {csv_filename} not found!")
        print("Run multi_sort with ENABLE_CSV_LOGGING=1 to generate data.")
        sys.exit(1)
    
    with open(csv_filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            threads.append(int(row['threads']))
            total_times.append(float(row['total_time']))
            phase1_times.append(float(row['phase1_time']))
            phase2_times.append(float(row['phase2_time']))
    
    # Sort by thread count
    data = sorted(zip(threads, total_times, phase1_times, phase2_times))
    threads, total_times, phase1_times, phase2_times = zip(*data)
    
    return list(threads), list(total_times), list(phase1_times), list(phase2_times)

def plot_results(csv_filename='multi_sort_results.csv', output_file='multi_sort_comparison.png'):
    """Create bar graph comparing performance across thread counts."""
    
    threads, total_times, phase1_times, phase2_times = read_csv_data(csv_filename)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X positions for bars
    x = np.arange(len(threads))
    width = 0.6  # Width of bars
    
    # Create stacked bars - Phase 1 on bottom, Phase 2 on top
    bars1 = ax.bar(x, phase1_times, width, label='Phase 1 (Sort)', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, phase2_times, width, bottom=phase1_times, 
                   label='Phase 2 (Merge)', color='#A23B72', alpha=0.8)
    
    # Customize axes
    ax.set_xlabel('Number of Threads', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in threads])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Add title
    ax.set_title('Multi-Sort Performance: Thread Count Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add Phase 1 labels inside the blue bars (middle of Phase 1 section)
    for i, (x_pos, phase1) in enumerate(zip(x, phase1_times)):
        if phase1 > 0:  # Only label if bar has height
            y_pos = phase1 / 2  # Middle of Phase 1 bar
            ax.text(x_pos, y_pos, f'{phase1:.2f}s', ha='center', va='center', 
                   fontsize=9, color='white', fontweight='bold')
    
    # Add Phase 2 labels inside the purple bars (middle of Phase 2 section)
    for i, (x_pos, phase1, phase2) in enumerate(zip(x, phase1_times, phase2_times)):
        if phase2 > 0:  # Only label if bar has height
            y_pos = phase1 + phase2 / 2  # Middle of Phase 2 bar
            ax.text(x_pos, y_pos, f'{phase2:.2f}s', ha='center', va='center', 
                   fontsize=9, color='white', fontweight='bold')
    
    # Add total time labels on top of stacked bars
    for i, (x_pos, total) in enumerate(zip(x, total_times)):
        ax.text(x_pos, total, f'{total:.2f}s', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='black')
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_file}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\n=== Performance Summary ===")
    print(f"{'Threads':<10} {'Total (s)':<12} {'Phase 1 (s)':<14} {'Phase 2 (s)':<14} {'Speedup':<10}")
    print("-" * 60)
    baseline = total_times[0]  # First (usually 1 thread)
    for i, t in enumerate(threads):
        speedup = baseline / total_times[i]
        print(f"{t:<10} {total_times[i]:<12.3f} {phase1_times[i]:<14.3f} "
              f"{phase2_times[i]:<14.3f} {speedup:<10.2f}x")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'multi_sort_results.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'multi_sort_comparison.png'
    
    plot_results(csv_file, output_file)

