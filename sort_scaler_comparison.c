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
// With prefetching to hide memory latency
static void stable_merge(uint32_t *left, size_t size_left,
                         uint32_t *right, size_t size_right,
                         uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    
    // Initial prefetch
    _mm_prefetch((const char*)(left), _MM_HINT_T0);
    _mm_prefetch((const char*)(right), _MM_HINT_T0);
    
    // Main branchless loop - avoid branch misprediction penalty
    while (i < size_left && j < size_right) {
        // Prefetch ahead every 16 elements
        if ((k & 15) == 0) {
            _mm_prefetch((const char*)(left + i + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(right + j + 64), _MM_HINT_T0);
        }
        
        uint32_t l_val = left[i];
        uint32_t r_val = right[j];
        
        // Compute mask: 0xFFFFFFFF if take_left, 0x00000000 otherwise
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

// ============== 4-way merge: merge 4 sorted runs into 1 ==============
// Reduces number of merge stages from log2(N) to log4(N)
// Stable: when equal, picks from earlier run (lower index)
static void stable_merge_4way(
    uint32_t *run0, size_t size0,
    uint32_t *run1, size_t size1,
    uint32_t *run2, size_t size2,
    uint32_t *run3, size_t size3,
    uint32_t *out
) {
    size_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, k = 0;
    size_t total = size0 + size1 + size2 + size3;
    
    // Prefetch all 4 runs
    _mm_prefetch((const char*)run0, _MM_HINT_T0);
    _mm_prefetch((const char*)run1, _MM_HINT_T0);
    _mm_prefetch((const char*)run2, _MM_HINT_T0);
    _mm_prefetch((const char*)run3, _MM_HINT_T0);
    
    // Main loop: find minimum of 4 runs
    while (i0 < size0 && i1 < size1 && i2 < size2 && i3 < size3) {
        // Prefetch ahead
        if ((k & 15) == 0) {
            _mm_prefetch((const char*)(run0 + i0 + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(run1 + i1 + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(run2 + i2 + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(run3 + i3 + 64), _MM_HINT_T0);
        }
        
        uint32_t v0 = run0[i0], v1 = run1[i1], v2 = run2[i2], v3 = run3[i3];
        
        // Tournament: find minimum with stability (prefer lower-indexed run on tie)
        // First round: compare pairs
        int cmp01 = (v0 <= v1);  // 1 if v0 wins (or tie)
        int cmp23 = (v2 <= v3);  // 1 if v2 wins (or tie)
        
        uint32_t win01 = cmp01 ? v0 : v1;
        uint32_t win23 = cmp23 ? v2 : v3;
        
        // Final round: winner of winners
        int cmp_final = (win01 <= win23);  // 1 if left pair wins (or tie)
        
        // Determine which run won
        int winner;
        if (cmp_final) {
            winner = cmp01 ? 0 : 1;
        } else {
            winner = cmp23 ? 2 : 3;
        }
        
        // Output and advance
        out[k++] = (winner == 0) ? v0 : (winner == 1) ? v1 : (winner == 2) ? v2 : v3;
        i0 += (winner == 0);
        i1 += (winner == 1);
        i2 += (winner == 2);
        i3 += (winner == 3);
    }
    
    // When one run exhausts, fall back to 3-way, then 2-way, then copy
    // Collect remaining runs into array for easier handling
    uint32_t *runs[4] = {run0 + i0, run1 + i1, run2 + i2, run3 + i3};
    size_t sizes[4] = {size0 - i0, size1 - i1, size2 - i2, size3 - i3};
    
    // Count non-empty runs and compact
    int active[4], num_active = 0;
    for (int r = 0; r < 4; r++) {
        if (sizes[r] > 0) active[num_active++] = r;
    }
    
    // Handle remaining based on how many runs are left
    if (num_active == 3) {
        // 3-way merge remainder
        int a = active[0], b = active[1], c = active[2];
        size_t ia = 0, ib = 0, ic = 0;
        while (ia < sizes[a] && ib < sizes[b] && ic < sizes[c]) {
            uint32_t va = runs[a][ia], vb = runs[b][ib], vc = runs[c][ic];
            int ab = (va <= vb), ac = (va <= vc), bc = (vb <= vc);
            if (ab && ac) { out[k++] = va; ia++; }
            else if (!ab && bc) { out[k++] = vb; ib++; }
            else { out[k++] = vc; ic++; }
        }
        // Update for 2-way
        runs[a] += ia; sizes[a] -= ia;
        runs[b] += ib; sizes[b] -= ib;
        runs[c] += ic; sizes[c] -= ic;
        num_active = 0;
        if (sizes[a] > 0) active[num_active++] = a;
        if (sizes[b] > 0) active[num_active++] = b;
        if (sizes[c] > 0) active[num_active++] = c;
    }
    
    if (num_active == 2) {
        // 2-way merge remainder
        int a = active[0], b = active[1];
        stable_merge(runs[a], sizes[a], runs[b], sizes[b], out + k);
        k += sizes[a] + sizes[b];
    } else if (num_active == 1) {
        // Just copy
        memcpy(out + k, runs[active[0]], sizes[active[0]] * sizeof(uint32_t));
    }
}

// ============== Interleaved merge: 2 independent merges per thread ==============
// Gives CPU more independent work to hide latency
// With prefetching to hide memory latency
#define PREFETCH_DIST 64  // Prefetch 64 elements ahead (~256 bytes, 4 cache lines)

static void stable_merge_interleaved_2(
    uint32_t *left0, size_t size_left0, uint32_t *right0, size_t size_right0, uint32_t *out0,
    uint32_t *left1, size_t size_left1, uint32_t *right1, size_t size_right1, uint32_t *out1
) {
    size_t i0 = 0, j0 = 0, k0 = 0;
    size_t i1 = 0, j1 = 0, k1 = 0;
    
    // Initial prefetch burst
    _mm_prefetch((const char*)(left0), _MM_HINT_T0);
    _mm_prefetch((const char*)(right0), _MM_HINT_T0);
    _mm_prefetch((const char*)(left1), _MM_HINT_T0);
    _mm_prefetch((const char*)(right1), _MM_HINT_T0);
    
    // Interleaved loop - work on both merges alternately
    while ((i0 < size_left0 && j0 < size_right0) && 
           (i1 < size_left1 && j1 < size_right1)) {
        // Prefetch ahead every 16 iterations (1 cache line worth of output)
        if ((k0 & 15) == 0) {
            _mm_prefetch((const char*)(left0 + i0 + PREFETCH_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(right0 + j0 + PREFETCH_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(left1 + i1 + PREFETCH_DIST), _MM_HINT_T0);
            _mm_prefetch((const char*)(right1 + j1 + PREFETCH_DIST), _MM_HINT_T0);
        }
        
        // Merge 0: one step
        uint32_t l0 = left0[i0], r0 = right0[j0];
        uint32_t cmp0 = (l0 <= r0);
        uint32_t mask0 = -cmp0;
        out0[k0] = (l0 & mask0) | (r0 & ~mask0);
        i0 += cmp0; j0 += (1 - cmp0); k0++;
        
        // Merge 1: one step (independent, can execute in parallel)
        uint32_t l1 = left1[i1], r1 = right1[j1];
        uint32_t cmp1 = (l1 <= r1);
        uint32_t mask1 = -cmp1;
        out1[k1] = (l1 & mask1) | (r1 & ~mask1);
        i1 += cmp1; j1 += (1 - cmp1); k1++;
    }
    
    // Finish merge 0
    while (i0 < size_left0 && j0 < size_right0) {
        uint32_t l0 = left0[i0], r0 = right0[j0];
        uint32_t cmp0 = (l0 <= r0);
        uint32_t mask0 = -cmp0;
        out0[k0++] = (l0 & mask0) | (r0 & ~mask0);
        i0 += cmp0; j0 += (1 - cmp0);
    }
    if (i0 < size_left0) memcpy(out0 + k0, left0 + i0, (size_left0 - i0) * sizeof(uint32_t));
    if (j0 < size_right0) memcpy(out0 + k0, right0 + j0, (size_right0 - j0) * sizeof(uint32_t));
    
    // Finish merge 1
    while (i1 < size_left1 && j1 < size_right1) {
        uint32_t l1 = left1[i1], r1 = right1[j1];
        uint32_t cmp1 = (l1 <= r1);
        uint32_t mask1 = -cmp1;
        out1[k1++] = (l1 & mask1) | (r1 & ~mask1);
        i1 += cmp1; j1 += (1 - cmp1);
    }
    if (i1 < size_left1) memcpy(out1 + k1, left1 + i1, (size_left1 - i1) * sizeof(uint32_t));
    if (j1 < size_right1) memcpy(out1 + k1, right1 + j1, (size_right1 - j1) * sizeof(uint32_t));
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
    
    // Bottom-up: start with runs of BASE_CASE_SIZE, QUADRUPLE each iteration (4-way merge)
    for (size_t run_size = BASE_CASE_SIZE; run_size < size; run_size *= 4) {
        clock_gettime(CLOCK_MONOTONIC, &stage_start);
        
        // Calculate number of 4-way merges at this level
        size_t num_merges = (size + 4 * run_size - 1) / (4 * run_size);
        
        if (num_merges >= PARALLEL_MERGE_THRESHOLD) {
            // Many merges: parallelize across 4-way merges
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < size; i += 4 * run_size) {
                // Calculate the 4 runs for this 4-way merge
                size_t start0 = i;
                size_t start1 = i + run_size;
                size_t start2 = i + 2 * run_size;
                size_t start3 = i + 3 * run_size;
                
                size_t size0 = (start0 < size) ? ((start1 < size) ? run_size : size - start0) : 0;
                size_t size1 = (start1 < size) ? ((start2 < size) ? run_size : size - start1) : 0;
                size_t size2 = (start2 < size) ? ((start3 < size) ? run_size : size - start2) : 0;
                size_t size3 = (start3 < size) ? ((start3 + run_size < size) ? run_size : size - start3) : 0;
                
                // Count how many runs we actually have
                int num_runs = (size0 > 0) + (size1 > 0) + (size2 > 0) + (size3 > 0);
                
                if (num_runs == 4) {
                    // Full 4-way merge
                    stable_merge_4way(src + start0, size0, src + start1, size1,
                                     src + start2, size2, src + start3, size3,
                                     dst + start0);
                } else if (num_runs == 3) {
                    // 3 runs: do 2-way merge of first two, then 2-way with third
                    uint32_t *temp_buf = aligned_alloc(64, (size0 + size1) * sizeof(uint32_t));
                    stable_merge(src + start0, size0, src + start1, size1, temp_buf);
                    stable_merge(temp_buf, size0 + size1, src + start2, size2, dst + start0);
                    free(temp_buf);
                } else if (num_runs == 2) {
                    // 2 runs: standard 2-way merge
                    stable_merge(src + start0, size0, src + start1, size1, dst + start0);
                } else if (num_runs == 1) {
                    // Just copy
                    memcpy(dst + start0, src + start0, size0 * sizeof(uint32_t));
                }
            }
        } else {
            // Few 4-way merges: process them with parallel 2-way merge internally
            for (size_t i = 0; i < size; i += 4 * run_size) {
                size_t start0 = i;
                size_t start1 = i + run_size;
                size_t start2 = i + 2 * run_size;
                size_t start3 = i + 3 * run_size;
                
                size_t size0 = (start0 < size) ? ((start1 < size) ? run_size : size - start0) : 0;
                size_t size1 = (start1 < size) ? ((start2 < size) ? run_size : size - start1) : 0;
                size_t size2 = (start2 < size) ? ((start3 < size) ? run_size : size - start2) : 0;
                size_t size3 = (start3 < size) ? ((start3 + run_size < size) ? run_size : size - start3) : 0;
                
                int num_runs = (size0 > 0) + (size1 > 0) + (size2 > 0) + (size3 > 0);
                
                if (num_runs == 4) {
                    // 4-way merge using parallel 2-way merges
                    // First: merge (0,1) and (2,3) in parallel
                    uint32_t *temp1 = aligned_alloc(64, (size0 + size1) * sizeof(uint32_t));
                    uint32_t *temp2 = aligned_alloc(64, (size2 + size3) * sizeof(uint32_t));
                    
                    #pragma omp parallel sections
                    {
                        #pragma omp section
                        parallel_merge(src + start0, size0, src + start1, size1, temp1, NUM_THREADS/2);
                        #pragma omp section
                        parallel_merge(src + start2, size2, src + start3, size3, temp2, NUM_THREADS/2);
                    }
                    
                    // Then: merge the two results
                    parallel_merge(temp1, size0 + size1, temp2, size2 + size3, dst + start0, NUM_THREADS);
                    
                    free(temp1);
                    free(temp2);
                } else if (num_runs == 3) {
                    uint32_t *temp1 = aligned_alloc(64, (size0 + size1) * sizeof(uint32_t));
                    parallel_merge(src + start0, size0, src + start1, size1, temp1, NUM_THREADS);
                    parallel_merge(temp1, size0 + size1, src + start2, size2, dst + start0, NUM_THREADS);
                    free(temp1);
                } else if (num_runs == 2) {
                    parallel_merge(src + start0, size0, src + start1, size1, dst + start0, NUM_THREADS);
                } else if (num_runs == 1) {
                    memcpy(dst + start0, src + start0, size0 * sizeof(uint32_t));
                }
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &stage_end);
        stage_time = (stage_end.tv_sec - stage_start.tv_sec) + 
                    (stage_end.tv_nsec - stage_start.tv_nsec) / 1e9;
        
        printf("Stage %2d: run_size=%8zu, 4way_merges=%6zu, time=%.6f sec\n", 
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
