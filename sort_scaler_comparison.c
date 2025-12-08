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

// Threshold: when number of merges drops below this, switch to parallel merge
#define PARALLEL_MERGE_THRESHOLD NUM_THREADS

// ============== Stable scalar merge (uses <= for stability) ==============
static void stable_merge(uint32_t *left, size_t size_left,
                         uint32_t *right, size_t size_right,
                         uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    while (i < size_left && j < size_right) {
        // Use <= for stability: when equal, take from left (preserves original order)
        if (left[i] <= right[j]) {
            out[k++] = left[i++];
        } else {
            out[k++] = right[j++];
        }
    }
    // Copy remaining elements
    if (i < size_left) {
        memcpy(out + k, left + i, (size_left - i) * sizeof(uint32_t));
    }
    if (j < size_right) {
        memcpy(out + k, right + j, (size_right - j) * sizeof(uint32_t));
    }
}

// ============== Binary search utilities for parallel merge ==============

// Find the position in sorted array where value should be inserted (lower bound)
// Returns index such that all elements before it are < value
static size_t lower_bound(uint32_t *arr, size_t size, uint32_t value) {
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

// Find upper bound - first position where element > value
static size_t upper_bound(uint32_t *arr, size_t size, uint32_t value) {
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

// ============== Parallel merge using median splitting ==============

// Structure to hold parallel merge task info
typedef struct {
    uint32_t *left;
    size_t left_start;
    size_t left_end;
    uint32_t *right;
    size_t right_start;
    size_t right_end;
    uint32_t *out;
    size_t out_start;
} merge_segment_t;

// Given two sorted arrays, find a pivot that splits the combined output at position 'target_pos'
// Returns the split points in left and right arrays
static void find_split_point(uint32_t *left, size_t left_size,
                             uint32_t *right, size_t right_size,
                             size_t target_pos,
                             size_t *left_split, size_t *right_split) {
    // Binary search to find the split
    // We need to find left_split and right_split such that:
    //   left_split + right_split = target_pos
    //   left[left_split-1] <= right[right_split] (if indices valid)
    //   right[right_split-1] <= left[left_split] (if indices valid)
    
    size_t lo = (target_pos > right_size) ? target_pos - right_size : 0;
    size_t hi = (target_pos < left_size) ? target_pos : left_size;
    
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        size_t j = target_pos - mid;
        
        // Check if this is a valid split point
        // left[mid-1] <= right[j] and right[j-1] < left[mid]
        // (using < instead of <= for the second condition to ensure stability)
        
        if (j > 0 && mid < left_size && right[j - 1] > left[mid]) {
            // right[j-1] too big, need more from left
            lo = mid + 1;
        } else if (mid > 0 && j < right_size && left[mid - 1] > right[j]) {
            // left[mid-1] too big, need less from left
            hi = mid;
        } else {
            // Found valid split
            *left_split = mid;
            *right_split = j;
            return;
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
    if (total_size < 1000 || num_threads <= 1) {
        stable_merge(left, left_size, right, right_size, out);
        return;
    }
    
    // Compute split points for each thread
    size_t *left_splits = (size_t*)malloc((num_threads + 1) * sizeof(size_t));
    size_t *right_splits = (size_t*)malloc((num_threads + 1) * sizeof(size_t));
    size_t *out_starts = (size_t*)malloc((num_threads + 1) * sizeof(size_t));
    
    // First and last splits are boundaries
    left_splits[0] = 0;
    right_splits[0] = 0;
    out_starts[0] = 0;
    left_splits[num_threads] = left_size;
    right_splits[num_threads] = right_size;
    out_starts[num_threads] = total_size;
    
    // Find intermediate split points
    for (int t = 1; t < num_threads; t++) {
        size_t target_pos = (total_size * t) / num_threads;
        find_split_point(left, left_size, right, right_size, target_pos,
                        &left_splits[t], &right_splits[t]);
        out_starts[t] = left_splits[t] + right_splits[t];
    }
    
    // Parallel merge each segment
    #pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < num_threads; t++) {
        size_t l_start = left_splits[t];
        size_t l_end = left_splits[t + 1];
        size_t r_start = right_splits[t];
        size_t r_end = right_splits[t + 1];
        size_t o_start = out_starts[t];
        
        stable_merge(left + l_start, l_end - l_start,
                    right + r_start, r_end - r_start,
                    out + o_start);
    }
    
    free(left_splits);
    free(right_splits);
    free(out_starts);
}

// ============== Bottom-up merge sort ==============

void sort_array(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    omp_set_num_threads(NUM_THREADS);
    
    // Allocate temporary buffer (aligned for potential SIMD use)
    uint32_t *temp = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        return;
    }
    
    // Pointers for ping-pong buffering
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    struct timespec stage_start, stage_end;
    int stage = 0;
    
    // Bottom-up: start with runs of size 1, double each iteration
    for (size_t run_size = 1; run_size < size; run_size *= 2) {
        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        
        // Calculate number of merges at this level
        size_t num_merges = (size + 2 * run_size - 1) / (2 * run_size);
        
        if (num_merges >= PARALLEL_MERGE_THRESHOLD) {
            // Many merges: run merges in parallel (one per thread)
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < size; i += 2 * run_size) {
                size_t left_start = i;
                size_t left_end = (i + run_size < size) ? i + run_size : size;
                size_t right_start = left_end;
                size_t right_end = (i + 2 * run_size < size) ? i + 2 * run_size : size;
                
                size_t left_size = left_end - left_start;
                size_t right_size = right_end - right_start;
                
                if (right_size == 0) {
                    // Only left part exists, just copy
                    memcpy(dst + left_start, src + left_start, left_size * sizeof(uint32_t));
                } else {
                    // Merge left and right parts
                    stable_merge(src + left_start, left_size,
                               src + right_start, right_size,
                               dst + left_start);
                }
            }
        } else {
            // Few merges: use parallel merge for each merge operation
            // Calculate threads per merge
            int threads_per_merge = NUM_THREADS / num_merges;
            if (threads_per_merge < 1) threads_per_merge = 1;
            
            // Process merges - can still parallelize across merges if we have threads left
            size_t merge_idx = 0;
            for (size_t i = 0; i < size; i += 2 * run_size) {
                size_t left_start = i;
                size_t left_end = (i + run_size < size) ? i + run_size : size;
                size_t right_start = left_end;
                size_t right_end = (i + 2 * run_size < size) ? i + 2 * run_size : size;
                
                size_t left_size = left_end - left_start;
                size_t right_size = right_end - right_start;
                
                if (right_size == 0) {
                    memcpy(dst + left_start, src + left_start, left_size * sizeof(uint32_t));
                } else {
                    // Use parallel merge with multiple threads
                    parallel_merge(src + left_start, left_size,
                                  src + right_start, right_size,
                                  dst + left_start, threads_per_merge);
                }
                merge_idx++;
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &stage_end);
        double stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
                           (stage_end.tv_nsec - stage_start.tv_nsec) / 1e9;
        
        printf("Stage %2d: run_size=%8zu, merges=%6zu, time=%.6f sec\n", 
               stage, run_size, num_merges, stage_time);
        stage++;
        
        // Swap buffers for next iteration
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

    // Read array from input file
    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) {
        return 1;
    }

    printf("Read %lu elements from %s\n", size, argv[1]);

    // Compute hash before sorting (order-independent)
    uint64_t xor_before, sum_before;
    compute_hash(arr, size, &xor_before, &sum_before);
    printf("Input hash: XOR=0x%016lx SUM=0x%016lx\n", xor_before, sum_before);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sort_array(arr, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);

    // Compute hash after sorting
    uint64_t xor_after, sum_after;
    compute_hash(arr, size, &xor_after, &sum_after);
    printf("Output hash: XOR=0x%016lx SUM=0x%016lx\n", xor_after, sum_after);

    // Verify hashes match (same elements, just reordered)
    if (xor_before != xor_after || sum_before != sum_after) {
        printf("Error: Hash mismatch! Elements were lost or corrupted during sorting.\n");
        free(arr);
        return 1;
    }
    printf("Hash check passed: all elements preserved.\n");

    // Only print array if small enough
    if (size <= 10) {
        print_array(arr, size);
    }

    // Verify the array is sorted
    if (verify_sortedness(arr, size)) {
        printf("Array sorted successfully!\n");
    } else {
        printf("Error: Array is not sorted correctly!\n");
        free(arr);
        return 1;
    }

    // Write sorted array to output file
    // if (write_array_to_file(argv[2], arr, size) != 0) {
    //     free(arr);
    //     return 1;
    // }

    free(arr);
    return 0;
}
