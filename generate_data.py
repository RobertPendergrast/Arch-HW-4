import struct
import sys
import time
from pathlib import Path

import numpy as np

DISTRIBUTIONS = ['uniform', 'normal', 'pareto', 'sorted', 'reverse', 'nearly']

# Chunk size: 16M elements = 64 MB per chunk
CHUNK_SIZE = 16 * 1024 * 1024


def write_header(f, size: int):
    """Write the uint64 size header."""
    f.write(struct.pack('<Q', size))


def write_chunk(f, data: np.ndarray):
    """Write a chunk of uint32 data."""
    data.astype('<u4').tofile(f)


def generate_uniform_chunked(f, size: int):
    """Uniformly random uint32 values, written in chunks."""
    remaining = size
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        data = np.random.randint(0, 2**32, size=chunk_size, dtype=np.uint32)
        write_chunk(f, data)
        remaining -= chunk_size


def generate_normal_chunked(f, size: int):
    """Normal distribution centered at 2^31, written in chunks."""
    remaining = size
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        data = np.random.normal(2**31, 2**30, chunk_size)
        data = np.clip(data, 0, 2**32 - 1).astype(np.uint32)
        write_chunk(f, data)
        remaining -= chunk_size


def generate_pareto_chunked(f, size: int):
    """Pareto (long-tail) distribution, written in chunks."""
    remaining = size
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        data = (np.random.pareto(1.5, chunk_size) + 1) * 1000
        data = np.clip(data, 0, 2**32 - 1).astype(np.uint32)
        write_chunk(f, data)
        remaining -= chunk_size


def generate_sorted_chunked(f, size: int):
    """Sorted ascending: 0, 1, 2, ..., size-1, written in chunks."""
    offset = 0
    remaining = size
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        data = np.arange(offset, offset + chunk_size, dtype=np.uint32)
        write_chunk(f, data)
        offset += chunk_size
        remaining -= chunk_size


def generate_reverse_chunked(f, size: int):
    """Sorted descending: size-1, ..., 1, 0, written in chunks."""
    offset = size - 1
    remaining = size
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        data = np.arange(offset, offset - chunk_size, -1, dtype=np.uint32)
        write_chunk(f, data)
        offset -= chunk_size
        remaining -= chunk_size


def generate_nearly_chunked(f, size: int):
    """Nearly sorted (95% sorted, 5% swapped), written in chunks."""
    # Generate sorted chunks with local perturbations
    offset = 0
    remaining = size
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        data = np.arange(offset, offset + chunk_size, dtype=np.uint32)
        # Swap 5% of elements within this chunk
        n_swaps = int(chunk_size * 0.05)
        for _ in range(n_swaps):
            i, j = np.random.randint(0, chunk_size, 2)
            data[i], data[j] = data[j], data[i]
        write_chunk(f, data)
        offset += chunk_size
        remaining -= chunk_size


def generate_same_chunked(f, size: int):
    """All same value (42), written in chunks."""
    remaining = size
    while remaining > 0:
        chunk_size = min(CHUNK_SIZE, remaining)
        data = np.full(chunk_size, 42, dtype=np.uint32)
        write_chunk(f, data)
        remaining -= chunk_size


GENERATORS = {
    'uniform': generate_uniform_chunked,
    'normal': generate_normal_chunked,
    'pareto': generate_pareto_chunked,
    'sorted': generate_sorted_chunked,
    'reverse': generate_reverse_chunked,
    'nearly': generate_nearly_chunked,
    'same': generate_same_chunked,
}


def generate_file(output_path: str, size: int, dist: str) -> float:
    """Generate data file with chunked writing to minimize RAM usage."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        write_header(f, size)
        GENERATORS[dist](f, size)
    
    return Path(output_path).stat().st_size / (1024 * 1024)


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
    
    output_path = f"/mnt/data/{dist}_{size}.bin"
    
    print(f"Generating {dist} ({size:,} elements) in {CHUNK_SIZE:,}-element chunks...")
    
    start = time.time()
    file_size_mb = generate_file(output_path, size, dist)
    elapsed = time.time() - start
    
    print(f"Done: {output_path} ({file_size_mb:.2f} MB, {elapsed:.3f}s)")
