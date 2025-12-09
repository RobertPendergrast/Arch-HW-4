### Cache Optimized, Multithreaded, SIMD Instructions Merge Sorting Algorithm

**Sorting.c/.h**: Main Merge Sort algorithm with cache optimized multi threading

**merge.c/.h**: Functions that take advantage of SIMD instructions and implement bitonic sort used in *Sorting.c* 

**utils.c/.h**: Utility functions to read and write arrays, verify sortedness and correctness

**generate_data.py**: Python script to generate arrays of various sizes and data distributions

**run_all_benchmarks.sh**: Bash script to automate the testing process and test: variations of our optimizations, different data distributions, different data sizes

**sort_simd_kv.c/.h**: Implements key value sort used in *Sorting.c*, about half as fast as the non-KV due to memory overhead

**sort_simd_main.c**: Main driver program for keys-only sorting, reads/writes binary files, verifies correctness via hash comparison

**sort_simd_kv_main.c**: Main driver program for key-value sorting, includes stability violation detection and equal key pair counting

**test_merge.c**: Unit tests for merge and bitonic sort functions in *merge.c*, validates correctness of SIMD merge operations

**benchmark_perf.py**: Python benchmarking script that tests sorting across orders of magnitude (1K to 10GB elements) with multiple data distributions

**benchmark_sort_simd_kv_differences.py**: Python script to benchmark the key-value sorter, records stability metrics, timing, and throughput across distributions

**makefile**: Build configuration using GCC with AVX-512 SIMD flags, OpenMP, and aggressive optimizations (-O3, -march=native)
