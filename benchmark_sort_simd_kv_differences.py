#!/usr/bin/env python3
"""
Benchmark `sort_simd_kv` across the available distributions and record
stability/variation metrics for each dataset file.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_DISTRIBUTIONS = [
    "uniform",
    "normal",
    "pareto",
    "sorted",
    "reverse",
    "nearly",
    "same",
]

DEFAULT_SIZES = [1_000_000]

CSV_HEADERS = [
    "distribution",
    "size",
    "elements",
    "total_time",
    "computed_throughput_gbps",
    "equal_pairs",
    "stability_violations",
    "violation_ratio",
    "stability_pct",
    "success",
    "error",
]

DATA_DIR = Path("datasets")
GENERATOR = Path("generate_data.py")
TMP_OUTPUT = Path("/tmp/sort_simd_kv_out.bin")


def ensure_dataset(size: int, dist: str, python_executable: str) -> tuple[Path | None, str | None]:
    """Generate or reuse the binary dataset for <dist>_<size>.bin."""
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / f"{dist}_{size}.bin"

    expected = 8 + size * 4
    if target.exists() and target.stat().st_size == expected:
        return target, None

    print(f"  [generate] {dist}_{size}.bin")
    cmd = [python_executable, str(GENERATOR), str(size), dist]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        return None, f"data generation failed: {message}"

    if not target.exists():
        return None, "data generation exited cleanly but file is missing"

    if target.stat().st_size != expected:
        target.unlink(missing_ok=True)
        return None, "generated file has unexpected size"

    return target, None


def parse_sort_output(stdout: str) -> dict:
    """Extract timing and stability metrics from the executable output."""
    metrics: dict[str, float | int | None] = {
        "total_time": None,
        "equal_pairs": None,
        "stability_violations": None,
        "stability_pct": None,
    }

    if not stdout:
        return metrics

    match = re.search(r"Sorting took ([\d.]+) seconds", stdout)
    if match:
        metrics["total_time"] = float(match.group(1))

    equal_pairs = re.search(r"Adjacent pairs with equal keys:\s+(\d+)", stdout)
    if equal_pairs:
        metrics["equal_pairs"] = int(equal_pairs.group(1))

    violations = re.search(r"Stability violations:\s+(\d+)", stdout)
    if violations:
        metrics["stability_violations"] = int(violations.group(1))

    stability_score = re.search(r"Stability score:\s+([\d.]+)%", stdout)
    if stability_score:
        metrics["stability_pct"] = float(stability_score.group(1))

    return metrics


def run_sort(executable: str, data_path: Path, timeout: int) -> tuple[subprocess.CompletedProcess | None, float, str | None]:
    """Run `sort_simd_kv` and return the process result, wall time and error message."""
    TMP_OUTPUT.unlink(missing_ok=True)
    cmd = [executable, str(data_path), str(TMP_OUTPUT)]
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        wall_time = time.time() - start
        return result, wall_time, None
    except subprocess.TimeoutExpired:
        return None, timeout, f"timeout after {timeout}s"
    except FileNotFoundError as exc:
        return None, 0.0, f"executable not found: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record stability metrics from sort_simd_kv for generated datasets."
    )
    parser.add_argument(
        "--executable",
        "-e",
        default="./sort_simd_kv",
        help="Path to the sort_simd_kv binary",
    )
    parser.add_argument(
        "--dists",
        "-d",
        nargs="+",
        default=DEFAULT_DISTRIBUTIONS,
        choices=DEFAULT_DISTRIBUTIONS,
        help="Distributions to test",
    )
    parser.add_argument(
        "--sizes",
        "-s",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Element counts for each dataset",
    )
    parser.add_argument(
        "--csv",
        "-c",
        default="sort_simd_kv_differences.csv",
        help="Path to write the CSV result table",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout (seconds) per sorting run",
    )

    args = parser.parse_args()
    executable = args.executable
    csv_path = Path(args.csv)
    sizes = sorted(set(args.sizes))
    distributions = args.dists

    if not Path(executable).exists():
        parser.error(f"executable '{executable}' not found")

    python_executable = sys.executable or "python3"
    rows: list[dict[str, str | float | int | None | bool]] = []

    print(f"Benchmarking {executable}")
    print(f"Distributions: {', '.join(distributions)}")
    print(f"Sizes: {', '.join(str(s) for s in sizes)}")
    print(f"CSV -> {csv_path}")

    for dist in distributions:
        for size in sizes:
            print(f"\n=== {dist.upper()} / {size:,} elements ===")
            data_path, gen_error = ensure_dataset(size, dist, python_executable)
            if gen_error:
                print(f"  [✗] {gen_error}")
                rows.append(
                    {
                        "distribution": dist,
                        "size": size,
                        "elements": size,
                        "total_time": None,
                        "computed_throughput_gbps": None,
                        "equal_pairs": None,
                        "stability_violations": None,
                        "violation_ratio": None,
                        "stability_pct": None,
                        "success": False,
                        "error": gen_error,
                    }
                )
                continue

            result, wall_time, exec_error = run_sort(executable, data_path, args.timeout)
            if exec_error:
                print(f"  [✗] {exec_error}")
                rows.append(
                    {
                        "distribution": dist,
                        "size": size,
                        "elements": size,
                        "total_time": None,
                        "computed_throughput_gbps": None,
                        "equal_pairs": None,
                        "stability_violations": None,
                        "violation_ratio": None,
                        "stability_pct": None,
                        "success": False,
                        "error": exec_error,
                    }
                )
                continue

            assert result is not None  # mypy hint
            success = result.returncode == 0
            metrics = parse_sort_output(result.stdout)
            total_time = metrics["total_time"]
            throughput = (
                (size * 4 * 2) / total_time / 1e9
                if total_time and total_time > 0
                else None
            )

            equal_pairs = metrics["equal_pairs"]
            violations = metrics["stability_violations"]
            if equal_pairs and equal_pairs > 0 and violations is not None:
                violation_ratio = violations / equal_pairs
            elif equal_pairs == 0:
                violation_ratio = 0.0
            else:
                violation_ratio = None

            err_msg = None
            if result.returncode != 0:
                err_msg = (
                    result.stderr.strip() or result.stdout.strip() or "non-zero exit"
                )

            row = {
                "distribution": dist,
                "size": size,
                "elements": size,
                "total_time": total_time,
                "computed_throughput_gbps": throughput,
                "equal_pairs": equal_pairs,
                "stability_violations": violations,
                "violation_ratio": violation_ratio,
                "stability_pct": metrics["stability_pct"],
                "success": success,
                "error": err_msg,
            }
            rows.append(row)

            if total_time:
                status = f"{total_time:.3f}s"
                if throughput:
                    status += f" ({throughput:.1f} GB/s)"
            else:
                status = "no timing"

            stability_text = (
                f"violations={violations}"
                if violations is not None
                else "violations=n/a"
            )
            print(f"  [✓] {status} | {stability_text}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_HEADERS})

    print(f"\nResults written to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
