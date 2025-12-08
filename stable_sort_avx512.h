#ifndef STABLE_SORT_AVX512_H
#define STABLE_SORT_AVX512_H

#include <stdint.h>
#include <stddef.h>

/*
 * Stable merge sort using AVX-512 acceleration.
 * 
 * Guarantees: Elements with equal keys maintain their original relative order.
 * 
 * Performance:
 * - Uses insertion sort for small chunks (stable, cache-friendly)
 * - Uses galloping optimization for merge phase (like Timsort)
 * - Uses AVX-512 for bulk memory copies
 * - Detects already-sorted boundaries to skip unnecessary work
 */

// Standard version - allocates temp buffer internally
void stable_merge_sort_avx512(uint32_t *arr, size_t n);

// Version with pre-allocated buffer (avoids allocation overhead)
// temp must be at least n elements and preferably 64-byte aligned
void stable_merge_sort_avx512_with_buffer(uint32_t *arr, size_t n, uint32_t *temp);

#ifdef _OPENMP
// Parallel version using OpenMP
void stable_merge_sort_avx512_parallel(uint32_t *arr, size_t n, int num_threads);
#endif

#endif // STABLE_SORT_AVX512_H
