#!/usr/bin/env python3
"""
Benchmark script for sorting executables.
Tests across orders of magnitude from 1K to 10GB elements.
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

# Default distribution for benchmarking
DEFAULT_DIST = "uniform"

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


def print_results_table(results: list):
    """Print results in a formatted table."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sorting executables across various array sizes"
    )
    parser.add_argument(
        "executable",
        help="Name of the sorting executable (e.g., sort_radix, sort_simd)"
    )
    parser.add_argument(
        "--dist", "-d",
        default=DEFAULT_DIST,
        choices=['uniform', 'normal', 'pareto', 'sorted', 'reverse', 'nearly', 'same'],
        help=f"Data distribution to test (default: {DEFAULT_DIST})"
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
    
    args = parser.parse_args()
    
    if args.list_sizes:
        print("Available test sizes:")
        for i, (size, name) in enumerate(TEST_SIZES):
            bytes_size = size * 4
            print(f"  {i}: {name:<12} ({size:>15,} elements, {human_size(bytes_size):>10})")
        return 0
    
    # Check executable exists
    if not os.path.exists(args.executable):
        print(f"Error: Executable '{args.executable}' not found.")
        print("Try running 'make' first to build the sorting programs.")
        return 1
    
    # Ensure datasets directory exists
    os.makedirs("datasets", exist_ok=True)
    
    # Filter test sizes based on arguments
    test_sizes = TEST_SIZES[args.min_size:args.max_size + 1]
    
    print(f"\nBenchmarking: {args.executable}")
    print(f"Distribution: {args.dist}")
    print(f"Test sizes: {len(test_sizes)} ({test_sizes[0][1]} to {test_sizes[-1][1]})")
    print("-" * 60)
    
    results = []
    
    for size, size_name in test_sizes:
        print(f"\n[{size_name}] Testing {size:,} elements ({human_size(size * 4)})...")
        
        # Ensure test data exists
        data_path = ensure_data_exists(size, args.dist)
        if not data_path:
            results.append({
                'elements': size,
                'size_name': size_name,
                'error': "Data generation failed"
            })
            continue
        
        # Run benchmark
        print(f"  [→] Running {args.executable}...")
        result = run_benchmark(args.executable, data_path, size)
        result['elements'] = size
        result['size_name'] = size_name
        
        if args.verbose and result.get('stdout'):
            print("  --- Output ---")
            for line in result['stdout'].strip().split('\n'):
                print(f"  | {line}")
            print("  --- End ---")
        
        if result.get('error'):
            print(f"  [✗] {result['error']}")
        elif result.get('total_time'):
            print(f"  [✓] Completed in {result['total_time']:.3f}s", end="")
            if result.get('throughput'):
                print(f" ({result['throughput']:.1f} GB/s)")
            else:
                print()
        
        results.append(result)
    
    # Print summary table
    print_results_table(results)
    
    # Print CSV-style output for easy copy-paste
    print("\nCSV output (for plotting):")
    print("elements,time_seconds,throughput_gbps")
    for r in results:
        if r.get('total_time'):
            tp = r.get('throughput', 0)
            print(f"{r['elements']},{r['total_time']:.4f},{tp:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
