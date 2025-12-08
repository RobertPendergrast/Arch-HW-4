CC = gcc
CFLAGS = -Wall -O3 -mavx512f -mavx512bw -march=native -funroll-loops -ffast-math
LDFLAGS = -pthread

# ============================================
# Main sorting implementations (4 versions)
# ============================================

# 1. Base: Basic merge sort (sorting.c)
# 2. Cache-optimized: Improved base cases with insertion sort (improved_split.c)
# 3. Multithreaded: Parallel chunk sorting with hierarchical merge (multi_sort.c)
# 4. SIMD: AVX-512 vectorized sorting network with OpenMP (sort_simd.c)

# ============================================
# Build object files
# ============================================

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o utils.o

merge.o: merge.c merge.h
	$(CC) $(CFLAGS) -c merge.c -o merge.o

# ============================================
# Main sorting executables
# ============================================

# Base merge sort
sorting: sorting.c utils.o
	$(CC) $(CFLAGS) sorting.c utils.o -o sorting $(LDFLAGS)

# Cache-optimized merge sort
improved_split: improved_split.c utils.o
	$(CC) $(CFLAGS) improved_split.c utils.o -o improved_split $(LDFLAGS)

# Multithreaded merge sort
multi_sort: multi_sort.c utils.o
	$(CC) $(CFLAGS) multi_sort.c utils.o -o multi_sort $(LDFLAGS)

# SIMD-accelerated merge sort (requires AVX-512)
sort_simd: sort_simd.c utils.o merge.o
	$(CC) $(CFLAGS) -fopenmp sort_simd.c utils.o merge.o -o sort_simd $(LDFLAGS)

# Non-SIMD version (same parallelization as sort_simd but scalar operations)
sort_no_simd: sort_no_simd.c utils.o
	$(CC) $(CFLAGS) -fopenmp sort_no_simd.c utils.o -o sort_no_simd $(LDFLAGS)

# ============================================
# Convenience targets
# ============================================

# Build all 5 main sorting implementations
all: sorting improved_split multi_sort sort_simd sort_no_simd
	@echo "Built all 5 sorting implementations:"
	@echo "  sorting        - Base merge sort"
	@echo "  improved_split - Cache-optimized merge sort"
	@echo "  multi_sort     - Multithreaded merge sort"
	@echo "  sort_simd      - SIMD-accelerated merge sort"
	@echo "  sort_no_simd   - Non-SIMD (scalar) with same parallelization as sort_simd"

# Clean build artifacts
clean:
	rm -f utils.o merge.o sorting improved_split multi_sort sort_simd sort_no_simd

# Clean everything including datasets and results
clean-all: clean
	rm -rf datasets/
	rm -f benchmark_results.csv benchmark_results_*.csv
	rm -f *.png

# ============================================
# Benchmarking
# ============================================

# Run the full benchmark suite
benchmark: all
	./run_all_benchmarks.sh

# Generate plots from results
plot:
	python3 plot_benchmark_results.py

.PHONY: all clean clean-all benchmark plot
