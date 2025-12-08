import struct
import sys
import time
from pathlib import Path

import numpy as np

DISTRIBUTIONS = ['uniform', 'normal', 'pareto', 'sorted', 'reverse', 'nearly']


def write_binary_array(filename: str, data: np.ndarray) -> float:
    """Write numpy array to binary file (uint64 header + uint32 data)."""
    data = data.astype(np.uint32)
    size = len(data)
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'wb') as f:
        f.write(struct.pack('<Q', size))
        data.astype('<u4').tofile(f)
    
    return Path(filename).stat().st_size / (1024 * 1024)


def generate_uniform(size: int) -> np.ndarray:
    """Uniformly random uint32 values."""
    return np.random.randint(0, 2**32, size=size, dtype=np.uint32)


def generate_normal(size: int) -> np.ndarray:
    """Normal distribution centered at 2^31."""
    data = np.random.normal(2**31, 2**30, size)
    return np.clip(data, 0, 2**32 - 1).astype(np.uint32)


def generate_pareto(size: int) -> np.ndarray:
    """Pareto (long-tail) distribution."""
    data = (np.random.pareto(1.5, size) + 1) * 1000
    return np.clip(data, 0, 2**32 - 1).astype(np.uint32)


def generate_sorted(size: int) -> np.ndarray:
    """Sorted ascending: 0, 1, 2, ..., size-1."""
    return np.arange(size, dtype=np.uint32)


def generate_reverse(size: int) -> np.ndarray:
    """Sorted descending: size-1, ..., 1, 0."""
    return np.arange(size, dtype=np.uint32)[::-1].copy()


def generate_nearly(size: int) -> np.ndarray:
    """Nearly sorted (95% sorted, 5% swapped)."""
    data = np.arange(size, dtype=np.uint32)
    n_swaps = int(size * 0.05)
    for _ in range(n_swaps):
        i, j = np.random.randint(0, size, 2)
        data[i], data[j] = data[j], data[i]
    return data


GENERATORS = {
    'uniform': generate_uniform,
    'normal': generate_normal,
    'pareto': generate_pareto,
    'sorted': generate_sorted,
    'reverse': generate_reverse,
    'nearly': generate_nearly,
}


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <size> <distribution>")
        print(f"Distributions: {', '.join(DISTRIBUTIONS)}")
        sys.exit(1)
    
    size = int(sys.argv[1])
    dist = sys.argv[2]
    
    if dist not in DISTRIBUTIONS:
        print(f"Unknown distribution: {dist}")
        print(f"Valid options: {', '.join(DISTRIBUTIONS)}")
        sys.exit(1)
    
    np.random.seed(42)
    
    output_path = f"datasets/{dist}_{size}.bin"
    
    print(f"Generating {dist} ({size:,} elements)...")
    
    start = time.time()
    data = GENERATORS[dist](size)
    file_size_mb = write_binary_array(output_path, data)
    elapsed = time.time() - start
    
    print(f"Done: {output_path} ({file_size_mb:.2f} MB, {elapsed:.3f}s)")
