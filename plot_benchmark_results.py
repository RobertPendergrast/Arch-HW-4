#!/usr/bin/env python3
"""
Generate benchmark comparison plots for merge sort implementations.

Creates one bar chart per data distribution, comparing the 4 sorting algorithms:
  - Base merge sort
  - Cache-optimized merge sort
  - Multithreaded merge sort
  - SIMD-accelerated merge sort

Also creates a combined summary plot showing all distributions.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from collections import defaultdict

# Algorithm display order and colors
ALGORITHMS = ['Base', 'Cache-Optimized', 'Multithreaded', 'SIMD']
COLORS = {
    'Base': '#2E86AB',           # Blue
    'Cache-Optimized': '#A23B72', # Purple
    'Multithreaded': '#F18F01',   # Orange
    'SIMD': '#C73E1D',            # Red
}

# Distribution display names
DIST_NAMES = {
    'uniform': 'Uniform Random',
    'normal': 'Normal Distribution',
    'pareto': 'Pareto (Long-tail)',
    'sorted': 'Already Sorted',
    'reverse': 'Reverse Sorted',
    'nearly': 'Nearly Sorted (95%)',
}


def read_results(csv_file='benchmark_results.csv'):
    """Read benchmark results from CSV file."""
    results = defaultdict(dict)  # {distribution: {algorithm_name: time}}
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found!")
        print("Run ./run_all_benchmarks.sh first to generate results.")
        sys.exit(1)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist = row['distribution']
            algo_name = row['algorithm_name']
            status = row.get('status', 'OK')
            
            if status == 'OK' and row['time_seconds']:
                results[dist][algo_name] = float(row['time_seconds'])
            else:
                results[dist][algo_name] = None  # Failed test
    
    return results


def plot_single_distribution(dist, times, output_file):
    """Create a bar chart for a single distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get times in algorithm order
    algo_times = []
    algo_labels = []
    algo_colors = []
    
    for algo in ALGORITHMS:
        if algo in times:
            time = times[algo]
            algo_times.append(time if time is not None else 0)
            algo_labels.append(algo)
            algo_colors.append(COLORS[algo])
    
    if not algo_times:
        print(f"  No data for {dist}, skipping...")
        plt.close()
        return
    
    x = np.arange(len(algo_labels))
    bars = ax.bar(x, algo_times, color=algo_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, algo_times)):
        if time > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time:.1f}s', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.5,
                   'FAIL', ha='center', va='bottom', 
                   fontsize=10, color='red', fontweight='bold')
    
    # Customize axes
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algo_labels, fontsize=10)
    
    # Title
    dist_name = DIST_NAMES.get(dist, dist.title())
    ax.set_title(f'Merge Sort Performance: {dist_name}\n(1 Billion Elements)',
                fontsize=14, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_combined_summary(results, output_file='benchmark_summary.png'):
    """Create a combined summary plot with all distributions."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    distributions = list(results.keys())
    n_dist = len(distributions)
    n_algo = len(ALGORITHMS)
    
    if n_dist == 0:
        print("No data to plot!")
        return
    
    # Set up bar positions
    bar_width = 0.2
    x = np.arange(n_dist)
    
    # Plot bars for each algorithm
    for i, algo in enumerate(ALGORITHMS):
        times = []
        for dist in distributions:
            t = results[dist].get(algo)
            times.append(t if t is not None else 0)
        
        offset = (i - n_algo/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, times, bar_width, 
                     label=algo, color=COLORS[algo], alpha=0.85,
                     edgecolor='black', linewidth=0.5)
    
    # Customize axes
    ax.set_xlabel('Data Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DIST_NAMES.get(d, d.title()) for d in distributions], 
                       fontsize=10, rotation=15, ha='right')
    
    # Title
    ax.set_title('Merge Sort Performance Comparison\n(1 Billion Elements, Multiple Distributions)',
                fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_speedup_chart(results, output_file='benchmark_speedup.png'):
    """Create a speedup chart comparing algorithms to the base implementation."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    distributions = list(results.keys())
    n_dist = len(distributions)
    
    if n_dist == 0:
        print("No data to plot!")
        return
    
    # Calculate speedups relative to base
    speedups = defaultdict(list)
    for dist in distributions:
        base_time = results[dist].get('Base')
        if base_time and base_time > 0:
            for algo in ALGORITHMS:
                algo_time = results[dist].get(algo)
                if algo_time and algo_time > 0:
                    speedups[algo].append(base_time / algo_time)
                else:
                    speedups[algo].append(0)
        else:
            for algo in ALGORITHMS:
                speedups[algo].append(0)
    
    # Set up bar positions
    bar_width = 0.2
    x = np.arange(n_dist)
    
    # Plot bars for each algorithm
    for i, algo in enumerate(ALGORITHMS):
        offset = (i - len(ALGORITHMS)/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, speedups[algo], bar_width,
                     label=algo, color=COLORS[algo], alpha=0.85,
                     edgecolor='black', linewidth=0.5)
    
    # Add baseline reference
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Baseline')
    
    # Customize axes
    ax.set_xlabel('Data Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (relative to Base)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([DIST_NAMES.get(d, d.title()) for d in distributions],
                       fontsize=10, rotation=15, ha='right')
    
    # Title
    ax.set_title('Speedup vs Base Merge Sort\n(1 Billion Elements)',
                fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def print_summary_table(results):
    """Print a summary table to console."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Header
    header = f"{'Distribution':<20}"
    for algo in ALGORITHMS:
        header += f" {algo:<15}"
    print(header)
    print("-" * 80)
    
    # Data rows
    for dist in results:
        dist_name = DIST_NAMES.get(dist, dist.title())[:18]
        row = f"{dist_name:<20}"
        for algo in ALGORITHMS:
            time = results[dist].get(algo)
            if time is not None:
                row += f" {time:<15.2f}"
            else:
                row += f" {'FAIL':<15}"
        print(row)
    
    print("=" * 80)
    
    # Calculate and print averages
    print("\nAverage Times:")
    for algo in ALGORITHMS:
        times = [results[d].get(algo) for d in results if results[d].get(algo) is not None]
        if times:
            avg = sum(times) / len(times)
            print(f"  {algo}: {avg:.2f}s")


def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results.csv'
    
    print(f"\nReading results from {csv_file}...")
    results = read_results(csv_file)
    
    if not results:
        print("No benchmark results found!")
        return 1
    
    print(f"Found results for {len(results)} distributions\n")
    
    # Print summary table
    print_summary_table(results)
    
    # Generate individual distribution plots
    print("\nGenerating plots...")
    for dist in results:
        output_file = f'benchmark_{dist}.png'
        plot_single_distribution(dist, results[dist], output_file)
    
    # Generate combined summary plot
    plot_combined_summary(results, 'benchmark_summary.png')
    
    # Generate speedup chart
    plot_speedup_chart(results, 'benchmark_speedup.png')
    
    print("\nDone!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
