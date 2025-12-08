/**
 * sort_no_simd.c
 * 
 * Non-SIMD version of sort_simd.c for fair comparison.
 * Uses the same:
 *   - OpenMP parallelization
 *   - Parallel merge with median split
 *   - L3 cache blocking
 * 
 * But replaces SIMD sorting/merging with scalar operations (like multi_sort.c).
 * 
 * KEY DIFFERENCE from previous version: Uses recursive merge sort instead of
 * insertion sort for base cases. Insertion sort is O(n²) which is catastrophically
 * slow even for n=64 when you have 15 million chunks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "utils.h"

// Number of threads for OpenMP parallelization (same as sort_simd.c)
#define NUM_THREADS 16

// L3 cache size: 32 MiB = 8M uint32_t elements (same as sort_simd.c)
#define L3_CHUNK_ELEMENTS (4 * 1024 * 1024)

// Threshold for parallel merge (same as sort_simd.c)
#define PARALLEL_MERGE_THRESHOLD (64 * 1024)

// ============== Scalar Merge Function ==============

/**
 * Merge two sorted subarrays [left..mid) and [mid..right) into one.
 * Same as multi_sort.c
 */
static void merge(uint32_t *arr, uint32_t *tmp, size_t left, size_t mid, size_t right) {
    size_t i = left, j = mid, k = 0;
    
    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    while (i < mid)   tmp[k++] = arr[i++];
    while (j < right) tmp[k++] = arr[j++];
    
    memcpy(&arr[left], tmp, (right - left) * sizeof(uint32_t));
}

/**
 * Merge two separate sorted arrays into output buffer.
 * Used for parallel merge and cross-chunk merging.
 */
static void merge_arrays(uint32_t *left, size_t left_size, 
                         uint32_t *right, size_t right_size, 
                         uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    
    while (i < left_size && j < right_size) {
        if (left[i] <= right[j]) {
            out[k++] = left[i++];
        } else {
            out[k++] = right[j++];
        }
    }
    while (i < left_size) out[k++] = left[i++];
    while (j < right_size) out[k++] = right[j++];
}

// ============== Recursive Merge Sort (like multi_sort.c) ==============

/**
 * Standard recursive merge sort.
 * O(n log n) - MUCH faster than insertion sort for chunks of any size!
 */
static void merge_sort_recursive(uint32_t *arr, uint32_t *tmp, size_t left, size_t right) {
    if (right - left < 2) return;
    
    size_t mid = left + (right - left) / 2;
    merge_sort_recursive(arr, tmp, left, mid);
    merge_sort_recursive(arr, tmp, mid, right);
    merge(arr, tmp, left, mid, right);
}

// ============== Parallel Merge with Median Split ==============

/**
 * Find split point for parallel merge using binary search.
 * Same algorithm as sort_simd.c
 */
static void find_merge_split(
    uint32_t *left, size_t left_size,
    uint32_t *right, size_t right_size,
    size_t *out_i, size_t *out_j
) {
    size_t total = left_size + right_size;
    size_t target = total / 2;
    
    size_t lo = (target > right_size) ? (target - right_size) : 0;
    size_t hi = (target < left_size) ? target : left_size;
    
    while (lo < hi) {
        size_t i = lo + (hi - lo) / 2;
        size_t j = target - i;
        
        if (j > 0 && i < left_size && right[j - 1] > left[i]) {
            lo = i + 1;
        } else if (i > 0 && j < right_size && left[i - 1] > right[j]) {
            hi = i;
        } else {
            *out_i = i;
            *out_j = j;
            return;
        }
    }
    
    *out_i = lo;
    *out_j = target - lo;
}

/**
 * Parallel merge implementation using OpenMP tasks.
 * Same structure as sort_simd.c
 */
static void parallel_merge_impl(
    uint32_t *left, size_t left_size,
    uint32_t *right, size_t right_size,
    uint32_t *out,
    int depth
) {
    size_t total = left_size + right_size;
    
    // Base case: small enough or no more parallelism
    if (depth <= 0 || total < PARALLEL_MERGE_THRESHOLD) {
        merge_arrays(left, left_size, right, right_size, out);
        return;
    }
    
    // Find split point
    size_t i, j;
    find_merge_split(left, left_size, right, right_size, &i, &j);
    
    size_t out_split = i + j;
    
    // Spawn parallel tasks for the two halves
    #pragma omp task
    parallel_merge_impl(left, i, right, j, out, depth - 1);
    
    #pragma omp task
    parallel_merge_impl(left + i, left_size - i, 
                        right + j, right_size - j,
                        out + out_split, depth - 1);
    
    #pragma omp taskwait
}

/**
 * Public interface for parallel merge.
 * Same as sort_simd.c
 */
static void parallel_merge(
    uint32_t *left, size_t left_size,
    uint32_t *right, size_t right_size,
    uint32_t *out
) {
    int depth = 0;
    for (int t = NUM_THREADS; t > 1; t /= 2) depth++;
    depth += 1;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_merge_impl(left, left_size, right, right_size, out, depth);
        }
    }
}

// Helper for timing
static inline double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ============== Main Sort Function ==============

/**
 * Main merge sort with L3 cache blocking.
 * Same structure as sort_simd.c:
 *   1. Divide into L3-sized chunks
 *   2. Sort each chunk with all threads collaborating
 *   3. Merge chunks using parallel merge
 */
void basic_merge_sort(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    double t_start, t_end;
    
    omp_set_num_threads(NUM_THREADS);
    
    // Allocate temp buffer
    uint32_t *temp = NULL;
    if (posix_memalign((void**)&temp, 64, size * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for temp buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // Warmup temp buffer
    t_start = get_time_sec();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 4096 / sizeof(uint32_t)) {
        temp[i] = 0;
    }
    t_end = get_time_sec();
    printf("  [Warmup] temp buffer (page faults): %.3f sec\n", t_end - t_start);
    
    // ========== Phase 1: Sort each L3-sized chunk ==========
    t_start = get_time_sec();
    size_t num_chunks = (size + L3_CHUNK_ELEMENTS - 1) / L3_CHUNK_ELEMENTS;
    
    // Sort chunks in parallel - each thread gets its own chunk
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < num_chunks; c++) {
        size_t start = c * L3_CHUNK_ELEMENTS;
        size_t chunk_size = (start + L3_CHUNK_ELEMENTS <= size) ? L3_CHUNK_ELEMENTS : (size - start);
        
        // Each thread has its own temp region
        merge_sort_recursive(arr + start, temp + start, 0, chunk_size);
    }
    t_end = get_time_sec();
    printf("  [Phase 1] Sort %zu L3 chunks (4M elements each): %.3f sec (%d threads)\n", 
           num_chunks, t_end - t_start, NUM_THREADS);
    
    // ========== Phase 2: Merge L3-sized chunks together ==========
    if (size > L3_CHUNK_ELEMENTS) {
        uint32_t *src = arr;
        uint32_t *dst = temp;
        size_t width = L3_CHUNK_ELEMENTS;
        
        // Do parallel merges while num_pairs >= NUM_THREADS
        while (width < size) {
            size_t num_pairs = (size + 2 * width - 1) / (2 * width);
            
            if (num_pairs < (size_t)NUM_THREADS) {
                printf("  [Phase 2] Stopping parallel-for merge at width %zu (%zu pairs < %d threads)\n", 
                       width, num_pairs, NUM_THREADS);
                break;
            }
            
            t_start = get_time_sec();
            
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t p = 0; p < num_pairs; p++) {
                size_t left_start = p * 2 * width;
                if (left_start >= size) continue;
                
                size_t left_size = (left_start + width <= size) ? width : (size - left_start);
                size_t right_start = left_start + left_size;
                
                if (right_start >= size) {
                    memcpy(dst + left_start, src + left_start, left_size * sizeof(uint32_t));
                } else {
                    size_t right_size = (right_start + width <= size) ? width : (size - right_start);
                    merge_arrays(src + left_start, left_size, 
                               src + right_start, right_size, 
                               dst + left_start);
                }
            }
            t_end = get_time_sec();
            double throughput = (size * sizeof(uint32_t)) / (t_end - t_start) / 1e9;
            printf("  [Phase 2] Merge width %10zu: %.3f sec (%zu parallel merges, %.2f GB/s)\n", 
                   width, t_end - t_start, num_pairs, throughput);
            
            uint32_t *swap = src;
            src = dst;
            dst = swap;
            width *= 2;
        }
        
        // Finish remaining merges with PARALLEL MERGE
        while (width < size) {
            t_start = get_time_sec();
            size_t num_pairs = (size + 2 * width - 1) / (2 * width);
            
            for (size_t p = 0; p < num_pairs; p++) {
                size_t left_start = p * 2 * width;
                if (left_start >= size) continue;
                
                size_t left_size = (left_start + width <= size) ? width : (size - left_start);
                size_t right_start = left_start + left_size;
                
                if (right_start >= size) {
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < left_size; i += 4096) {
                        size_t copy_size = (i + 4096 <= left_size) ? 4096 : (left_size - i);
                        memcpy(dst + left_start + i, src + left_start + i, copy_size * sizeof(uint32_t));
                    }
                } else {
                    size_t right_size = (right_start + width <= size) ? width : (size - right_start);
                    parallel_merge(src + left_start, left_size, 
                                   src + right_start, right_size, 
                                   dst + left_start);
                }
            }
            
            t_end = get_time_sec();
            double throughput = (size * sizeof(uint32_t)) / (t_end - t_start) / 1e9;
            printf("  [Phase 2] Merge width %10zu: %.3f sec (%zu parallel merges, %d threads, %.2f GB/s)\n", 
                   width, t_end - t_start, num_pairs, NUM_THREADS, throughput);
            
            uint32_t *swap = src;
            src = dst;
            dst = swap;
            width *= 2;
        }
        
        // Copy result back to arr if needed
        if (src != arr) {
            t_start = get_time_sec();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < size; i += 4096) {
                size_t copy_size = (i + 4096 <= size) ? 4096 : (size - i);
                memcpy(arr + i, src + i, copy_size * sizeof(uint32_t));
            }
            t_end = get_time_sec();
            printf("  [Final ] Copy back: %.3f sec\n", t_end - t_start);
        }
    }
    
    free(temp);
}

// ============== Main ==============

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) {
        return 1;
    }

    printf("Read %lu elements from %s\n", size, argv[1]);

    // Compute hash before sorting
    uint64_t xor_before, sum_before;
    compute_hash(arr, size, &xor_before, &sum_before);
    printf("Input hash: XOR=0x%016lx SUM=0x%016lx\n", xor_before, sum_before);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    basic_merge_sort(arr, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);

    // Compute hash after sorting
    uint64_t xor_after, sum_after;
    compute_hash(arr, size, &xor_after, &sum_after);
    printf("Output hash: XOR=0x%016lx SUM=0x%016lx\n", xor_after, sum_after);

    if (xor_before != xor_after || sum_before != sum_after) {
        printf("Error: Hash mismatch! Elements were lost or corrupted.\n");
        free(arr);
        return 1;
    }
    printf("Hash check passed: all elements preserved.\n");

    if (verify_sortedness(arr, size)) {
        printf("Array sorted successfully!\n");
    } else {
        printf("Error: Array is not sorted correctly!\n");
        free(arr);
        return 1;
    }

    free(arr);
    return 0;
}


