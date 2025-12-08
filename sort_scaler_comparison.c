#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include "utils.h"
#include "merge.h"

// Number of threads for OpenMP parallelization
#define NUM_THREADS 16

// Base case: use insertion sort for chunks this size or smaller
#define BASE_CASE_SIZE 64

// Threshold: when number of merges drops below this, switch to parallel merge
#define PARALLEL_MERGE_THRESHOLD NUM_THREADS

// ============== SIMD-optimized insertion sort for base case (stable) ==============
// Uses SIMD for the shift operation when inserting elements
static inline void insertion_sort_simd(uint32_t *arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        uint32_t key = arr[i];
        if (arr[i - 1] <= key) continue;  // Already in place (common case)
        
        // Binary search for insertion point (more efficient for larger gaps)
        size_t lo = 0, hi = i;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (arr[mid] <= key) {  // <= for stability
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        
        // Shift elements [lo, i) right by 1 using memmove
        // memmove is optimized and often uses SIMD internally
        memmove(arr + lo + 1, arr + lo, (i - lo) * sizeof(uint32_t));
        arr[lo] = key;
    }
}

// ============== Branchless stable merge with galloping ==============
// Uses branchless selection + galloping for long runs + SIMD bulk copies

// Binary search: find first position where arr[pos] > value (upper bound)
static inline size_t gallop_upper(uint32_t *arr, size_t size, uint32_t value) {
    size_t lo = 0, hi = size;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] <= value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Binary search: find first position where arr[pos] >= value (lower bound)  
static inline size_t gallop_lower(uint32_t *arr, size_t size, uint32_t value) {
    size_t lo = 0, hi = size;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// SIMD memcpy for aligned data (16 elements = 64 bytes at a time)
static inline void simd_copy_aligned(uint32_t *dst, uint32_t *src, size_t count) {
    size_t i = 0;
    // Process 16 elements at a time with AVX-512
    for (; i + 16 <= count; i += 16) {
        __m512i v = _mm512_loadu_si512((__m512i*)(src + i));
        _mm512_storeu_si512((__m512i*)(dst + i), v);
    }
    // Process 8 elements with AVX2
    for (; i + 8 <= count; i += 8) {
        __m256i v = _mm256_loadu_si256((__m256i*)(src + i));
        _mm256_storeu_si256((__m256i*)(dst + i), v);
    }
    // Remainder
    for (; i < count; i++) {
        dst[i] = src[i];
    }
}

// Stable merge with galloping mode for handling runs
// When one side has many consecutive smaller elements, we binary search
// to find how many, then bulk copy them
static void stable_merge_gallop(uint32_t *left, size_t size_left,
                                uint32_t *right, size_t size_right,
                                uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    
    // Galloping threshold - switch to gallop mode after this many consecutive wins
    const size_t GALLOP_THRESHOLD = 7;
    
    size_t left_wins = 0, right_wins = 0;
    
    while (i < size_left && j < size_right) {
        if (left[i] <= right[j]) {
            out[k++] = left[i++];
            left_wins++;
            right_wins = 0;
            
            // If left is winning a lot, gallop
            if (left_wins >= GALLOP_THRESHOLD && i < size_left) {
                // Find how many more from left are <= right[j]
                size_t run = gallop_upper(left + i, size_left - i, right[j]);
                if (run > 0) {
                    simd_copy_aligned(out + k, left + i, run);
                    k += run;
                    i += run;
                }
                left_wins = 0;
            }
        } else {
            out[k++] = right[j++];
            right_wins++;
            left_wins = 0;
            
            // If right is winning a lot, gallop
            if (right_wins >= GALLOP_THRESHOLD && j < size_right) {
                // Find how many more from right are < left[i] (strict < for stability)
                size_t run = gallop_lower(right + j, size_right - j, left[i]);
                if (run > 0) {
                    simd_copy_aligned(out + k, right + j, run);
                    k += run;
                    j += run;
                }
                right_wins = 0;
            }
        }
    }
    
    // Copy remaining elements with SIMD
    if (i < size_left) {
        simd_copy_aligned(out + k, left + i, size_left - i);
    }
    if (j < size_right) {
        simd_copy_aligned(out + k, right + j, size_right - j);
    }
}

// Simple branchless merge for small sizes (avoids gallop overhead)
static void stable_merge_simple(uint32_t *left, size_t size_left,
                                uint32_t *right, size_t size_right,
                                uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    
    // Main merge loop - use branchless selection
    while (i < size_left && j < size_right) {
        uint32_t l = left[i];
        uint32_t r = right[j];
        // Branchless: if l <= r, take left (stable); else take right
        int take_left = (l <= r);
        out[k++] = take_left ? l : r;
        i += take_left;
        j += !take_left;
    }
    
    // Copy remaining
    if (i < size_left) {
        simd_copy_aligned(out + k, left + i, size_left - i);
    }
    if (j < size_right) {
        simd_copy_aligned(out + k, right + j, size_right - j);
    }
}

// Dispatcher: choose merge strategy based on size
static inline void stable_merge(uint32_t *left, size_t size_left,
                                uint32_t *right, size_t size_right,
                                uint32_t *out) {
    // For small merges, simple is faster (less overhead)
    if (size_left + size_right < 512) {
        stable_merge_simple(left, size_left, right, size_right, out);
    } else {
        stable_merge_gallop(left, size_left, right, size_right, out);
    }
}

// ============== Parallel merge using median splitting ==============

// Given two sorted arrays, find a split point that divides the combined 
// output at position 'target_pos' into left[0..left_split) + right[0..right_split)
static void find_split_point(uint32_t *left, size_t left_size,
                             uint32_t *right, size_t right_size,
                             size_t target_pos,
                             size_t *left_split, size_t *right_split) {
    // Binary search to find the split
    size_t lo = (target_pos > right_size) ? target_pos - right_size : 0;
    size_t hi = (target_pos < left_size) ? target_pos : left_size;
    
    while (lo < hi) {
        size_t i = lo + (hi - lo) / 2;
        size_t j = target_pos - i;
        
        if (j > 0 && i < left_size && right[j - 1] > left[i]) {
            lo = i + 1;
        } else if (i > 0 && j < right_size && left[i - 1] > right[j]) {
            hi = i;
        } else {
            lo = i;
            break;
        }
    }
    
    *left_split = lo;
    *right_split = target_pos - lo;
}

// Parallel merge: merge two sorted arrays using multiple threads
static void parallel_merge(uint32_t *left, size_t left_size,
                          uint32_t *right, size_t right_size,
                          uint32_t *out, int num_threads) {
    size_t total_size = left_size + right_size;
    
    // For small merges, just use serial
    if (total_size < 100000 || num_threads <= 1) {
        stable_merge(left, left_size, right, right_size, out);
        return;
    }
    
    // Limit threads to avoid too-small segments
    size_t min_segment = 50000;
    int effective_threads = (total_size / min_segment);
    if (effective_threads < 1) effective_threads = 1;
    if (effective_threads > num_threads) effective_threads = num_threads;
    
    // Compute split points for each thread
    size_t *left_splits = (size_t*)alloca((effective_threads + 1) * sizeof(size_t));
    size_t *right_splits = (size_t*)alloca((effective_threads + 1) * sizeof(size_t));
    
    left_splits[0] = 0;
    right_splits[0] = 0;
    left_splits[effective_threads] = left_size;
    right_splits[effective_threads] = right_size;
    
    // Find intermediate split points
    for (int t = 1; t < effective_threads; t++) {
        size_t target_pos = (total_size * t) / effective_threads;
        find_split_point(left, left_size, right, right_size, target_pos,
                        &left_splits[t], &right_splits[t]);
    }
    
    // Parallel merge each segment
    #pragma omp parallel for num_threads(effective_threads) schedule(static)
    for (int t = 0; t < effective_threads; t++) {
        size_t l_start = left_splits[t];
        size_t l_end = left_splits[t + 1];
        size_t r_start = right_splits[t];
        size_t r_end = right_splits[t + 1];
        size_t o_start = l_start + r_start;
        
        stable_merge(left + l_start, l_end - l_start,
                    right + r_start, r_end - r_start,
                    out + o_start);
    }
}

// ============== Bottom-up merge sort ==============

void sort_array(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    omp_set_num_threads(NUM_THREADS);
    
    struct timespec stage_start, stage_end;
    double stage_time;
    
    // ============== STAGE 0: Base case - insertion sort chunks ==============
    clock_gettime(CLOCK_MONOTONIC, &stage_start);
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < size; i += BASE_CASE_SIZE) {
        size_t chunk_size = (i + BASE_CASE_SIZE <= size) ? BASE_CASE_SIZE : (size - i);
        insertion_sort_simd(arr + i, chunk_size);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &stage_end);
    stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
                (stage_end.tv_nsec - stage_start.tv_nsec) / 1e9;
    printf("Stage  0: insertion sort %zu-element chunks, time=%.6f sec\n", 
           (size_t)BASE_CASE_SIZE, stage_time);
    
    // Allocate temporary buffer (aligned for SIMD)
    uint32_t *temp = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        return;
    }
    
    // Pre-touch temp buffer to force OS to allocate physical pages
    clock_gettime(CLOCK_MONOTONIC, &stage_start);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 4096 / sizeof(uint32_t)) {
        temp[i] = 0;
    }
    clock_gettime(CLOCK_MONOTONIC, &stage_end);
    stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
                (stage_end.tv_nsec - stage_start.tv_nsec) / 1e9;
    printf("         temp buffer warmup (page faults), time=%.6f sec\n", stage_time);
    
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    int stage = 1;
    
    // Bottom-up merge passes
    for (size_t run_size = BASE_CASE_SIZE; run_size < size; run_size *= 2) {
        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        
        size_t num_merges = (size + 2 * run_size - 1) / (2 * run_size);
        
        if (num_merges >= PARALLEL_MERGE_THRESHOLD) {
            // Many merges: one merge per thread task
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < size; i += 2 * run_size) {
                size_t left_start = i;
                size_t left_end = (i + run_size < size) ? i + run_size : size;
                size_t right_start = left_end;
                size_t right_end = (i + 2 * run_size < size) ? i + 2 * run_size : size;
                
                size_t left_size = left_end - left_start;
                size_t right_size = right_end - right_start;
                
                if (right_size == 0) {
                    simd_copy_aligned(dst + left_start, src + left_start, left_size);
                } else {
                    stable_merge(src + left_start, left_size,
                               src + right_start, right_size,
                               dst + left_start);
                }
            }
        } else {
            // Few merges: use parallel merge for EACH merge with ALL threads
            // Process merges sequentially, but each merge uses all NUM_THREADS
            for (size_t i = 0; i < size; i += 2 * run_size) {
                size_t left_start = i;
                size_t left_end = (i + run_size < size) ? i + run_size : size;
                size_t right_start = left_end;
                size_t right_end = (i + 2 * run_size < size) ? i + 2 * run_size : size;
                
                size_t left_size = left_end - left_start;
                size_t right_size = right_end - right_start;
                
                if (right_size == 0) {
                    simd_copy_aligned(dst + left_start, src + left_start, left_size);
                } else {
                    parallel_merge(src + left_start, left_size,
                                  src + right_start, right_size,
                                  dst + left_start, NUM_THREADS);
                }
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &stage_end);
        stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
                    (stage_end.tv_nsec - stage_start.tv_nsec) / 1e9;
        
        printf("Stage %2d: run_size=%10zu, merges=%6zu, time=%.6f sec%s\n", 
               stage, run_size, num_merges, stage_time,
               num_merges < PARALLEL_MERGE_THRESHOLD ? " (parallel merge)" : "");
        stage++;
        
        // Swap buffers
        uint32_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    // If result ended up in temp buffer, copy back to arr
    if (src != arr) {
        memcpy(arr, src, size * sizeof(uint32_t));
    }
    
    free(temp);
}

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

    uint64_t xor_before, sum_before;
    compute_hash(arr, size, &xor_before, &sum_before);
    printf("Input hash: XOR=0x%016lx SUM=0x%016lx\n", xor_before, sum_before);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sort_array(arr, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);

    uint64_t xor_after, sum_after;
    compute_hash(arr, size, &xor_after, &sum_after);
    printf("Output hash: XOR=0x%016lx SUM=0x%016lx\n", xor_after, sum_after);

    if (xor_before != xor_after || sum_before != sum_after) {
        printf("Error: Hash mismatch! Elements were lost or corrupted during sorting.\n");
        free(arr);
        return 1;
    }
    printf("Hash check passed: all elements preserved.\n");

    if (size <= 10) {
        print_array(arr, size);
    }

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
