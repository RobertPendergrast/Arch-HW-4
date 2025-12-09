#!/usr/bin/env python3
"""
Benchmark script for sorting executables.
Tests across orders of magnitude from 1K to 10GB elements.
Tests multiple data distributions.
Generates test data if it doesn't exist.
"""

import argparse
import os
import subprocess
import sys
import time
import re
from pathlib import Path

# Test sizes: orders of magnitude from 1K to ~2.5B (10GB)
# Each size is (num_elements, human_readable_name)
TEST_SIZES = [
    (1_000, "1K"),
    (10_000, "10K"),
    (100_000, "100K"),
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
    (1_000_000_000, "1B"),
    (2_500_000_000, "2.5B (10GB)"),
]

# All available distributions
ALL_DISTRIBUTIONS = ['uniform', 'normal', 'pareto', 'sorted', 'reverse', 'nearly', 'same']

# Default distributions to test (representative subset)
DEFAULT_DISTRIBUTIONS = ['uniform', 'sorted', 'same']

# Timeout per test (seconds) - scales with size
BASE_TIMEOUT = 30  # seconds for 1M elements
TIMEOUT_PER_BILLION = 300  # additional seconds per billion elements


def get_timeout(size: int) -> int:
    """Calculate appropriate timeout based on array size."""
    return int(BASE_TIMEOUT + (size / 1_000_000_000) * TIMEOUT_PER_BILLION)


def human_size(num_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def ensure_data_exists(size: int, dist: str) -> str:
    """Generate test data if it doesn't exist. Returns path to data file."""
    data_path = f"datasets/{dist}_{size}.bin"
    
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path)
        expected_size = 8 + size * 4  # 8-byte header + 4 bytes per element
        if file_size == expected_size:
            print(f"  [✓] Data exists: {data_path} ({human_size(file_size)})")
            return data_path
        else:
            print(f"  [!] Data file size mismatch, regenerating...")
    
    print(f"  [→] Generating {data_path}...")
    start = time.time()
    
    result = subprocess.run(
        ["python3", "generate_data.py", str(size), dist],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"  [✗] Failed to generate data: {result.stderr}")
        return None
    
    elapsed = time.time() - start
    file_size = os.path.getsize(data_path)
    print(f"  [✓] Generated in {elapsed:.1f}s ({human_size(file_size)})")
    
    return data_path


def parse_sort_output(output: str) -> dict:
    """Parse the sorting program output for timing info."""
    result = {
        'total_time': None,
        'passes': [],
        'throughput': None,
        'success': False,
    }
    
    # Look for "Sorting took X.XXX seconds"
    match = re.search(r'Sorting took ([\d.]+) seconds', output)
    if match:
        result['total_time'] = float(match.group(1))
    
    # Look for pass timings: "Pass X (bits Y-Z): X.XXX sec (X.X GB/s)"
    for match in re.finditer(r'Pass \d+.*?: ([\d.]+) sec \(([\d.]+) GB/s\)', output):
        result['passes'].append({
            'time': float(match.group(1)),
            'throughput': float(match.group(2))
        })
    
    # Calculate average throughput from passes
    if result['passes']:
        result['throughput'] = sum(p['throughput'] for p in result['passes']) / len(result['passes'])
    
    # Check for success
    if 'sorted successfully' in output.lower() or 'hash check passed' in output.lower():
        result['success'] = True
    
    return result


def run_benchmark(executable: str, data_path: str, size: int) -> dict:
    """Run the sorting executable and return results."""
    timeout = get_timeout(size)
    output_path = "/tmp/benchmark_output.bin"
    
    cmd = [f"./{executable}", data_path, output_path]
    
    try:
        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        wall_time = time.time() - start
        
        parsed = parse_sort_output(result.stdout)
        parsed['wall_time'] = wall_time
        parsed['returncode'] = result.returncode
        parsed['stdout'] = result.stdout
        parsed['stderr'] = result.stderr
        
        if result.returncode != 0:
            parsed['error'] = f"Exit code {result.returncode}"
        
        return parsed
        
    except subprocess.TimeoutExpired:
        return {
            'error': f"Timeout ({timeout}s)",
            'wall_time': timeout,
            'success': False
        }
    except FileNotFoundError:
        return {
            'error': f"Executable not found: {executable}",
            'success': False
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


def print_results_table(results: list, single_dist: bool = True):
    """Print results in a formatted table."""
    if single_dist:
        print("\n" + "=" * 85)
        print(f"{'Size':<12} {'Elements':<14} {'Time (s)':<12} {'Throughput':<14} {'Status':<12}")
        print("=" * 85)
        
        for r in results:
            size_str = r['size_name']
            elements = f"{r['elements']:,}"
            
            if r.get('error'):
                time_str = "-"
                throughput_str = "-"
                status = f"FAIL: {r['error'][:20]}"
            elif r.get('total_time'):
                time_str = f"{r['total_time']:.3f}"
                if r.get('throughput'):
                    throughput_str = f"{r['throughput']:.1f} GB/s"
                else:
                    # Calculate from size and time
                    bytes_processed = r['elements'] * 4 * 2  # read + write
                    throughput = bytes_processed / r['total_time'] / 1e9
                    throughput_str = f"~{throughput:.1f} GB/s"
                status = "OK" if r.get('success') else "UNKNOWN"
            else:
                time_str = f"{r.get('wall_time', 0):.3f}"
                throughput_str = "-"
                status = "NO OUTPUT"
            
            print(f"{size_str:<12} {elements:<14} {time_str:<12} {throughput_str:<14} {status:<12}")
        
        print("=" * 85)


def print_multi_dist_table(all_results: dict, test_sizes: list, distributions: list):
    """Print comparison table across multiple distributions."""
    # Build header
    dist_width = 12
    header = f"{'Size':<8}"
    for dist in distributions:
        header += f" {dist:<{dist_width}}"
    
    print("\n" + "=" * (8 + len(distributions) * (dist_width + 1)))
    print("TIMING COMPARISON (seconds)")
    print("=" * (8 + len(distributions) * (dist_width + 1)))
    print(header)
    print("-" * (8 + len(distributions) * (dist_width + 1)))
    
    for size, size_name in test_sizes:
        row = f"{size_name:<8}"
        for dist in distributions:
            key = (size, dist)
            if key in all_results:
                r = all_results[key]
                if r.get('total_time'):
                    row += f" {r['total_time']:<{dist_width}.3f}"
                elif r.get('error'):
                    row += f" {'FAIL':<{dist_width}}"
                else:
                    row += f" {'-':<{dist_width}}"
            else:
                row += f" {'-':<{dist_width}}"
        print(row)
    
    print("=" * (8 + len(distributions) * (dist_width + 1)))
    
    # Throughput table
    print("\nTHROUGHPUT COMPARISON (GB/s)")
    print("=" * (8 + len(distributions) * (dist_width + 1)))
    print(header)
    print("-" * (8 + len(distributions) * (dist_width + 1)))
    
    for size, size_name in test_sizes:
        row = f"{size_name:<8}"
        for dist in distributions:
            key = (size, dist)
            if key in all_results:
                r = all_results[key]
                if r.get('throughput'):
                    row += f" {r['throughput']:<{dist_width}.1f}"
                elif r.get('total_time') and r['total_time'] > 0:
                    # Calculate throughput
                    bytes_processed = size * 4 * 2
                    tp = bytes_processed / r['total_time'] / 1e9
                    row += f" {tp:<{dist_width}.1f}"
                else:
                    row += f" {'-':<{dist_width}}"
            else:
                row += f" {'-':<{dist_width}}"
        print(row)
    
    print("=" * (8 + len(distributions) * (dist_width + 1)))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sorting executables across various array sizes and distributions"
    )
    parser.add_argument(
        "executable",
        nargs="?",
        default="./sort_simd_main",
        help="Sorting executable to benchmark (default: ./sort_simd_main)"
    )
    parser.add_argument(
        "--dist", "-d",
        nargs='+',
        default=None,
        choices=ALL_DISTRIBUTIONS,
        help=f"Distribution(s) to test. Can specify multiple. (default: {DEFAULT_DISTRIBUTIONS})"
    )
    parser.add_argument(
        "--all-dists", "-a",
        action="store_true",
        help="Test all available distributions"
    )
    parser.add_argument(
        "--min-size", "-m",
        type=int,
        default=0,
        help="Minimum size index to test (0-7, default: 0)"
    )
    parser.add_argument(
        "--max-size", "-M",
        type=int,
        default=len(TEST_SIZES) - 1,
        help=f"Maximum size index to test (0-{len(TEST_SIZES)-1}, default: {len(TEST_SIZES)-1})"
    )
    parser.add_argument(
        "--list-sizes", "-l",
        action="store_true",
        help="List available test sizes and exit"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full output from sorting program"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output during benchmarking"
    )
    
    args = parser.parse_args()
    
    if args.list_sizes:
        print("Available test sizes:")
        for i, (size, name) in enumerate(TEST_SIZES):
            bytes_size = size * 4
            print(f"  {i}: {name:<12} ({size:>15,} elements, {human_size(bytes_size):>10})")
        print("\nAvailable distributions:")
        print(f"  {', '.join(ALL_DISTRIBUTIONS)}")
        return 0
    
    # Check executable exists
    if not os.path.exists(args.executable):
        print(f"Error: Executable '{args.executable}' not found.")
        print("Try running 'make' first to build the sorting programs.")
        return 1
    
    # Determine which distributions to test
    if args.all_dists:
        distributions = ALL_DISTRIBUTIONS
    elif args.dist:
        distributions = args.dist
    else:
        distributions = DEFAULT_DISTRIBUTIONS
    
    # Ensure datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    
    # Filter test sizes based on arguments
    test_sizes = TEST_SIZES[args.min_size:args.max_size + 1]
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {args.executable}")
    print(f"Distributions: {', '.join(distributions)}")
    print(f"Test sizes: {len(test_sizes)} ({test_sizes[0][1]} to {test_sizes[-1][1]})")
    print(f"Total tests: {len(test_sizes) * len(distributions)}")
    print(f"{'='*60}")
    
    # Store all results keyed by (size, distribution)
    all_results = {}
    
    for dist in distributions:
        print(f"\n{'─'*60}")
        print(f"Distribution: {dist.upper()}")
        print(f"{'─'*60}")
        
        for size, size_name in test_sizes:
            if not args.quiet:
                print(f"\n[{dist}/{size_name}] {size:,} elements ({human_size(size * 4)})...")
            
            # Ensure test data exists
            data_path = ensure_data_exists(size, dist)
            if not data_path:
                all_results[(size, dist)] = {
                    'elements': size,
                    'size_name': size_name,
                    'dist': dist,
                    'error': "Data generation failed"
                }
                continue
            
            # Run benchmark
            if not args.quiet:
                print(f"  [→] Running {args.executable}...")
            result = run_benchmark(args.executable, data_path, size)
            result['elements'] = size
            result['size_name'] = size_name
            result['dist'] = dist
            
            if args.verbose and result.get('stdout'):
                print("  --- Output ---")
                for line in result['stdout'].strip().split('\n'):
                    print(f"  | {line}")
                print("  --- End ---")
            
            if result.get('error'):
                print(f"  [✗] {result['error']}")
            elif result.get('total_time'):
                if not args.quiet:
                    print(f"  [✓] {result['total_time']:.3f}s", end="")
                    if result.get('throughput'):
                        print(f" ({result['throughput']:.1f} GB/s)")
                    else:
                        print()
                else:
                    # One-line summary in quiet mode
                    tp = result.get('throughput', 0)
                    print(f"  {dist}/{size_name}: {result['total_time']:.3f}s ({tp:.1f} GB/s)")
            
            all_results[(size, dist)] = result
    
    # Print comparison tables
    if len(distributions) > 1:
        print_multi_dist_table(all_results, test_sizes, distributions)
    else:
        # Single distribution - use simple table
        results = [all_results[(s, distributions[0])] for s, _ in test_sizes if (s, distributions[0]) in all_results]
        print_results_table(results, single_dist=True)
    
    # Print CSV-style output for easy copy-paste
    print("\nCSV output (for plotting):")
    print("distribution,elements,time_seconds,throughput_gbps")
    for dist in distributions:
        for size, _ in test_sizes:
            key = (size, dist)
            if key in all_results:
                r = all_results[key]
                total_time = r.get('total_time')
                if total_time is not None:
                    tp = r.get('throughput') or 0
                    print(f"{dist},{size},{total_time:.4f},{tp:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
