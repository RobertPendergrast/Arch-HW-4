### Cache Optimized, Multithreaded, SIMD Instructions Merge Sorting Algorithm

*Sorting.c/.h*: Main Merge Sort algorithm with cache optimized multi threading

*merge.c/.h*: Functions that take advantage of SIMD instructions and implement bitonic sort used in *Sorting.c* 

*utils.c/.h*: Utility functions to read and write arrays, verify sortedness and correctness

*generate_data.py*: Python script to generate arrays of various sizes and data distributions

*run_all_benchmarks.sh*: Bash script to automate the testing process and test: variations of our optimizations, different data distributions, different data sizes

*sort_simd_kv.c/.h*: Implements key value sort used in *Sorting.c*, about half of fast as the non-KV due to memory overhead. 

*