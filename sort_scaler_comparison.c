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

// ============== Insertion sort for base case (stable) ==============
static inline void insertion_sort(uint32_t *arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        uint32_t key = arr[i];
        size_t j = i;
        // Use > (not >=) for stability - equal elements stay in original order
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

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

// ============== Parallel merge using median splitting ==============

// Given two sorted arrays, find a split point that divides the combined 
// output at position 'target_pos' into left[0..left_split) + right[0..right_split)
static void find_split_point(uint32_t *left, size_t left_size,
                             uint32_t *right, size_t right_size,
                             size_t target_pos,
                             size_t *left_split, size_t *right_split) {
    // Binary search to find the split
    // We need left_split + right_split = target_pos
    // Constraint: left[left_split-1] <= right[right_split] (if both valid)
    //             right[right_split-1] <= left[left_split] (if both valid)
    
    // Bounds for left_split
    size_t lo = (target_pos > right_size) ? target_pos - right_size : 0;
    size_t hi = (target_pos < left_size) ? target_pos : left_size;
    
    while (lo < hi) {
        size_t i = lo + (hi - lo) / 2;  // candidate left_split
        size_t j = target_pos - i;       // corresponding right_split
        
        // Check if this is a valid split
        if (j > 0 && i < left_size && right[j - 1] > left[i]) {
            // right[j-1] > left[i]: need more from left
            lo = i + 1;
        } else if (i > 0 && j < right_size && left[i - 1] > right[j]) {
            // left[i-1] > right[j]: need less from left
            hi = i;
        } else {
            // Valid split found
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
    if (total_size < 10000 || num_threads <= 1) {
        stable_merge(left, left_size, right, right_size, out);
        return;
    }
    
    // Compute split points for each thread
    size_t left_splits[NUM_THREADS + 1];
    size_t right_splits[NUM_THREADS + 1];
    
    // First and last splits are boundaries
    left_splits[0] = 0;
    right_splits[0] = 0;
    left_splits[num_threads] = left_size;
    right_splits[num_threads] = right_size;
    
    // Find intermediate split points (can be done in parallel)
    #pragma omp parallel for num_threads(num_threads)
    for (int t = 1; t < num_threads; t++) {
        size_t target_pos = (total_size * t) / num_threads;
        find_split_point(left, left_size, right, right_size, target_pos,
                        &left_splits[t], &right_splits[t]);
    }
    
    // Parallel merge each segment
    #pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < num_threads; t++) {
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

// Structure for parallel merge task
typedef struct {
    uint32_t *left;
    size_t left_size;
    uint32_t *right;
    size_t right_size;
    uint32_t *out;
    int num_threads;
} merge_task_t;

// L3 cache size: 32 MiB = 8M uint32_t elements
// All threads work on ONE chunk of this size at a time (cache locality)
#define L3_CHUNK_ELEMENTS (8 * 1024 * 1024)

// Helper for timing
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ============== Sort a single L3 chunk with ALL threads ==============
// All threads collaborate on the SAME chunk, keeping data hot in L3 cache
static void sort_l3_chunk(uint32_t *arr, size_t chunk_size, uint32_t *temp) {
    // Step 1: Base case sort - ALL threads work on this chunk's base cases
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < chunk_size; i += BASE_CASE_SIZE) {
        size_t block_size = (i + BASE_CASE_SIZE <= chunk_size) ? BASE_CASE_SIZE : (chunk_size - i);
        insertion_sort(arr + i, block_size);
    }
    
    // Step 2: Merge passes WITHIN this chunk (stays in L3 cache)
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    for (size_t width = BASE_CASE_SIZE; width < chunk_size; width *= 2) {
        size_t num_pairs = (chunk_size + 2 * width - 1) / (2 * width);
        
        if (num_pairs > 1) {
            // Multiple pairs: parallelize across merges
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t p = 0; p < num_pairs; p++) {
                size_t left_start = p * 2 * width;
                if (left_start >= chunk_size) continue;
                
                size_t left_size = (left_start + width <= chunk_size) ? width : (chunk_size - left_start);
                size_t right_start = left_start + left_size;
                
                if (right_start >= chunk_size) {
                    memcpy(dst + left_start, src + left_start, left_size * sizeof(uint32_t));
                } else {
                    size_t right_size = (right_start + width <= chunk_size) ? width : (chunk_size - right_start);
                    stable_merge(src + left_start, left_size,
                               src + right_start, right_size,
                               dst + left_start);
                }
            }
        } else {
            // Single pair: use parallel merge (all threads on one merge)
            size_t left_size = (width <= chunk_size) ? width : chunk_size;
            size_t right_start = left_size;
            
            if (right_start < chunk_size) {
                size_t right_size = chunk_size - right_start;
                parallel_merge(src, left_size, src + right_start, right_size, dst, NUM_THREADS);
            } else {
                memcpy(dst, src, chunk_size * sizeof(uint32_t));
            }
        }
        
        // Swap src and dst
        uint32_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    // Copy result back to arr if needed
    if (src != arr) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < chunk_size; i += 4096) {
            size_t copy_size = (i + 4096 <= chunk_size) ? 4096 : (chunk_size - i);
            memcpy(arr + i, src + i, copy_size * sizeof(uint32_t));
        }
    }
}

// ============== Bottom-up merge sort with L3 cache blocking ==============

void sort_array(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    omp_set_num_threads(NUM_THREADS);
    
    double t_start, t_end;
    
    // Allocate temporary buffer (aligned for potential SIMD use)
    uint32_t *temp = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        return;
    }
    
    // Pre-touch temp buffer to force OS to allocate physical pages
    t_start = get_time_sec();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 4096 / sizeof(uint32_t)) {
        temp[i] = 0;  // Touch one element per page (4KB pages)
    }
    t_end = get_time_sec();
    printf("         temp buffer warmup (page faults), time=%.6f sec\n", t_end - t_start);
    
    // ========== PHASE 1: Sort each L3-sized chunk ==========
    // Process ONE chunk at a time, but ALL threads work on that chunk
    // This keeps data hot in L3 cache during all merge passes within the chunk
    t_start = get_time_sec();
    size_t num_chunks = (size + L3_CHUNK_ELEMENTS - 1) / L3_CHUNK_ELEMENTS;
    
    for (size_t c = 0; c < num_chunks; c++) {
        size_t start = c * L3_CHUNK_ELEMENTS;
        size_t chunk_size = (start + L3_CHUNK_ELEMENTS <= size) ? L3_CHUNK_ELEMENTS : (size - start);
        // All threads collaborate on THIS chunk (data stays in L3)
        sort_l3_chunk(arr + start, chunk_size, temp + start);
    }
    t_end = get_time_sec();
    printf("[Phase 1] Sort %zu L3 chunks (%zuM elements each): %.3f sec (%d threads, cache-focused)\n",
           num_chunks, L3_CHUNK_ELEMENTS / (1024 * 1024), t_end - t_start, NUM_THREADS);
    
    // ========== PHASE 2: Merge L3-sized chunks together ==========
    // These merges are memory-bound (exceed L3), but we minimize the number of passes
    if (size > L3_CHUNK_ELEMENTS) {
        uint32_t *src = arr;
        uint32_t *dst = temp;
        
        for (size_t width = L3_CHUNK_ELEMENTS; width < size; width *= 2) {
            t_start = get_time_sec();
            
            size_t num_pairs = (size + 2 * width - 1) / (2 * width);
            
            if (num_pairs >= (size_t)NUM_THREADS) {
                // Many pairs: each thread handles one merge
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
                        stable_merge(src + left_start, left_size,
                                   src + right_start, right_size,
                                   dst + left_start);
                    }
                }
            } else {
                // Few pairs: use parallel merge within each pair
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
                                      dst + left_start, NUM_THREADS);
                    }
                }
            }
            
            t_end = get_time_sec();
            double throughput = (size * sizeof(uint32_t)) / (t_end - t_start) / 1e9;
            printf("[Phase 2] Merge width %10zu: %.3f sec (%zu merges, %.2f GB/s)\n",
                   width, t_end - t_start, num_pairs, throughput);
            
            // Swap src and dst
            uint32_t *swap = src;
            src = dst;
            dst = swap;
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
            printf("[Final  ] Copy back: %.3f sec\n", t_end - t_start);
        }
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
