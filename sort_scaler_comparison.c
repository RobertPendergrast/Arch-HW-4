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

// ============== Stable branchless merge (uses <= for stability) ==============
// Uses conditional moves to avoid branch mispredictions (~15-20 cycle penalty each)
static void stable_merge(uint32_t *left, size_t size_left,
                         uint32_t *right, size_t size_right,
                         uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    
    // Main branchless loop - avoid branch misprediction penalty
    while (i < size_left && j < size_right) {
        uint32_t l_val = left[i];
        uint32_t r_val = right[j];
        
        // Compute mask: 0xFFFFFFFF if take_left, 0x00000000 otherwise
        // Use unsigned comparison for stability: l_val <= r_val means take left
        uint32_t cmp = (l_val <= r_val);  // 1 or 0
        uint32_t mask = -cmp;              // 0xFFFFFFFF or 0x00000000
        
        // Branchless select: (l_val & mask) | (r_val & ~mask)
        out[k] = (l_val & mask) | (r_val & ~mask);
        
        // Branchless index updates
        i += cmp;
        j += (1 - cmp);
        k++;
    }
    
    // Copy remaining elements
    if (i < size_left) {
        memcpy(out + k, left + i, (size_left - i) * sizeof(uint32_t));
    }
    if (j < size_right) {
        memcpy(out + k, right + j, (size_right - j) * sizeof(uint32_t));
    }
}

// ============== Interleaved merge: 4 independent merges per thread ==============
// Gives CPU more independent work to hide latency
typedef struct {
    uint32_t *left, *right, *out;
    size_t size_left, size_right;
    size_t i, j, k;
} merge_state_t;

static inline void finish_merge(merge_state_t *m) {
    while (m->i < m->size_left && m->j < m->size_right) {
        uint32_t l = m->left[m->i], r = m->right[m->j];
        uint32_t cmp = (l <= r);
        uint32_t mask = -cmp;
        m->out[m->k++] = (l & mask) | (r & ~mask);
        m->i += cmp; m->j += (1 - cmp);
    }
    if (m->i < m->size_left) 
        memcpy(m->out + m->k, m->left + m->i, (m->size_left - m->i) * sizeof(uint32_t));
    if (m->j < m->size_right) 
        memcpy(m->out + m->k, m->right + m->j, (m->size_right - m->j) * sizeof(uint32_t));
}

static void stable_merge_interleaved_4(
    uint32_t *left0, size_t size_left0, uint32_t *right0, size_t size_right0, uint32_t *out0,
    uint32_t *left1, size_t size_left1, uint32_t *right1, size_t size_right1, uint32_t *out1,
    uint32_t *left2, size_t size_left2, uint32_t *right2, size_t size_right2, uint32_t *out2,
    uint32_t *left3, size_t size_left3, uint32_t *right3, size_t size_right3, uint32_t *out3
) {
    size_t i0 = 0, j0 = 0, k0 = 0;
    size_t i1 = 0, j1 = 0, k1 = 0;
    size_t i2 = 0, j2 = 0, k2 = 0;
    size_t i3 = 0, j3 = 0, k3 = 0;
    
    // Interleaved loop - work on all 4 merges alternately
    while ((i0 < size_left0 && j0 < size_right0) && 
           (i1 < size_left1 && j1 < size_right1) &&
           (i2 < size_left2 && j2 < size_right2) &&
           (i3 < size_left3 && j3 < size_right3)) {
        // Merge 0
        uint32_t l0 = left0[i0], r0 = right0[j0];
        uint32_t cmp0 = (l0 <= r0);
        uint32_t mask0 = -cmp0;
        out0[k0] = (l0 & mask0) | (r0 & ~mask0);
        i0 += cmp0; j0 += (1 - cmp0); k0++;
        
        // Merge 1
        uint32_t l1 = left1[i1], r1 = right1[j1];
        uint32_t cmp1 = (l1 <= r1);
        uint32_t mask1 = -cmp1;
        out1[k1] = (l1 & mask1) | (r1 & ~mask1);
        i1 += cmp1; j1 += (1 - cmp1); k1++;
        
        // Merge 2
        uint32_t l2 = left2[i2], r2 = right2[j2];
        uint32_t cmp2 = (l2 <= r2);
        uint32_t mask2 = -cmp2;
        out2[k2] = (l2 & mask2) | (r2 & ~mask2);
        i2 += cmp2; j2 += (1 - cmp2); k2++;
        
        // Merge 3
        uint32_t l3 = left3[i3], r3 = right3[j3];
        uint32_t cmp3 = (l3 <= r3);
        uint32_t mask3 = -cmp3;
        out3[k3] = (l3 & mask3) | (r3 & ~mask3);
        i3 += cmp3; j3 += (1 - cmp3); k3++;
    }
    
    // Finish remaining merges
    merge_state_t m0 = {left0, right0, out0, size_left0, size_right0, i0, j0, k0};
    merge_state_t m1 = {left1, right1, out1, size_left1, size_right1, i1, j1, k1};
    merge_state_t m2 = {left2, right2, out2, size_left2, size_right2, i2, j2, k2};
    merge_state_t m3 = {left3, right3, out3, size_left3, size_right3, i3, j3, k3};
    finish_merge(&m0);
    finish_merge(&m1);
    finish_merge(&m2);
    finish_merge(&m3);
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

// ============== Bottom-up merge sort ==============

void sort_array(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    omp_set_num_threads(NUM_THREADS);
    
    struct timespec stage_start, stage_end;
    
    // ============== STAGE 0: Base case - insertion sort chunks ==============
    clock_gettime(CLOCK_MONOTONIC, &stage_start);
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < size; i += BASE_CASE_SIZE) {
        size_t chunk_size = (i + BASE_CASE_SIZE <= size) ? BASE_CASE_SIZE : (size - i);
        insertion_sort(arr + i, chunk_size);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &stage_end);
    double stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
                       (stage_end.tv_nsec - stage_start.tv_nsec) / 1e9;
    printf("Stage  0: insertion sort %zu-element chunks, time=%.6f sec\n", 
           (size_t)BASE_CASE_SIZE, stage_time);
    
    // Allocate temporary buffer (aligned for potential SIMD use)
    uint32_t *temp = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temporary buffer\n");
        return;
    }
    
    // Pre-touch temp buffer to force OS to allocate physical pages
    // This avoids page fault overhead during the first merge stage
    clock_gettime(CLOCK_MONOTONIC, &stage_start);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 4096 / sizeof(uint32_t)) {
        temp[i] = 0;  // Touch one element per page (4KB pages)
    }
    clock_gettime(CLOCK_MONOTONIC, &stage_end);
    stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
                (stage_end.tv_nsec - stage_start.tv_nsec) / 1e9;
    printf("         temp buffer warmup (page faults), time=%.6f sec\n", stage_time);
    
    // Pointers for ping-pong buffering
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    int stage = 1;
    
    // Bottom-up: start with runs of BASE_CASE_SIZE, double each iteration
    for (size_t run_size = BASE_CASE_SIZE; run_size < size; run_size *= 2) {
        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        
        // Calculate number of merges at this level
        size_t num_merges = (size + 2 * run_size - 1) / (2 * run_size);
        
        if (num_merges >= PARALLEL_MERGE_THRESHOLD) {
            // Many merges: use interleaved merge (4 merges per thread iteration)
            // This hides latency by giving CPU 4 independent work streams
            size_t quad_stride = 8 * run_size;  // Each iteration handles 4 merge pairs
            
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < size; i += quad_stride) {
                // Calculate all 4 merge pairs
                size_t base[4] = {i, i + 2*run_size, i + 4*run_size, i + 6*run_size};
                size_t left_start[4], left_size[4], right_start[4], right_size[4];
                int valid[4] = {0, 0, 0, 0};
                
                for (int m = 0; m < 4; m++) {
                    if (base[m] < size) {
                        left_start[m] = base[m];
                        size_t left_end = (base[m] + run_size < size) ? base[m] + run_size : size;
                        left_size[m] = left_end - left_start[m];
                        right_start[m] = left_end;
                        size_t right_end = (base[m] + 2*run_size < size) ? base[m] + 2*run_size : size;
                        right_size[m] = (right_start[m] < size) ? right_end - right_start[m] : 0;
                        valid[m] = (right_size[m] > 0);  // Valid merge if right side exists
                    }
                }
                
                // Check if all 4 are valid merges
                if (valid[0] && valid[1] && valid[2] && valid[3]) {
                    stable_merge_interleaved_4(
                        src + left_start[0], left_size[0], src + right_start[0], right_size[0], dst + left_start[0],
                        src + left_start[1], left_size[1], src + right_start[1], right_size[1], dst + left_start[1],
                        src + left_start[2], left_size[2], src + right_start[2], right_size[2], dst + left_start[2],
                        src + left_start[3], left_size[3], src + right_start[3], right_size[3], dst + left_start[3]
                    );
                } else {
                    // Handle remaining merges individually
                    for (int m = 0; m < 4; m++) {
                        if (base[m] >= size) continue;
                        if (right_size[m] == 0) {
                            memcpy(dst + left_start[m], src + left_start[m], left_size[m] * sizeof(uint32_t));
                        } else {
                            stable_merge(src + left_start[m], left_size[m],
                                       src + right_start[m], right_size[m],
                                       dst + left_start[m]);
                        }
                    }
                }
            }
        } else {
            // Few merges: use multiple threads PER merge
            // Collect all merge tasks first
            merge_task_t tasks[num_merges];
            size_t task_idx = 0;
            
            for (size_t i = 0; i < size; i += 2 * run_size) {
                size_t left_start = i;
                size_t left_end = (i + run_size < size) ? i + run_size : size;
                size_t right_start = left_end;
                size_t right_end = (i + 2 * run_size < size) ? i + 2 * run_size : size;
                
                tasks[task_idx].left = src + left_start;
                tasks[task_idx].left_size = left_end - left_start;
                tasks[task_idx].right = src + right_start;
                tasks[task_idx].right_size = right_end - right_start;
                tasks[task_idx].out = dst + left_start;
                task_idx++;
            }
            
            // Calculate threads per merge
            int threads_per_merge = NUM_THREADS / num_merges;
            if (threads_per_merge < 2) threads_per_merge = 2;
            if (threads_per_merge > NUM_THREADS) threads_per_merge = NUM_THREADS;
            
            // For very few merges (1-4), process sequentially but with parallel merge
            if (num_merges <= 4) {
                for (size_t t = 0; t < num_merges; t++) {
                    if (tasks[t].right_size == 0) {
                        memcpy(tasks[t].out, tasks[t].left, 
                               tasks[t].left_size * sizeof(uint32_t));
                    } else {
                        parallel_merge(tasks[t].left, tasks[t].left_size,
                                      tasks[t].right, tasks[t].right_size,
                                      tasks[t].out, NUM_THREADS);
                    }
                }
            } else {
                // For moderate number of merges (5-15), use nested parallelism
                // Outer loop: parallelize across merges
                // Inner: each merge uses threads_per_merge threads
                omp_set_nested(1);
                #pragma omp parallel for num_threads(num_merges) schedule(static, 1)
                for (size_t t = 0; t < num_merges; t++) {
                    if (tasks[t].right_size == 0) {
                        memcpy(tasks[t].out, tasks[t].left, 
                               tasks[t].left_size * sizeof(uint32_t));
                    } else {
                        parallel_merge(tasks[t].left, tasks[t].left_size,
                                      tasks[t].right, tasks[t].right_size,
                                      tasks[t].out, threads_per_merge);
                    }
                }
                omp_set_nested(0);
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &stage_end);
        stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
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
