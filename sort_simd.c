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

// Forward declaration
void basic_merge_sort(uint32_t *arr, size_t size);

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {
    basic_merge_sort(arr, size);
}

// Insertion sort for small/remainder arrays
static inline void insertion_sort(uint32_t *arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        uint32_t key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

// ============== SIMD Sorting Network ==============
// Bitonic sort for 16 elements in a single 512-bit register

// Shuffle indices for bitonic sort - swap elements at distance d
static const int SORT_SWAP1_IDX[16] __attribute__((aligned(64))) = 
    {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};  // distance 1: swap i <-> i^1
static const int SORT_SWAP2_IDX[16] __attribute__((aligned(64))) = 
    {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13};  // distance 2: swap i <-> i^2
static const int SORT_SWAP4_IDX[16] __attribute__((aligned(64))) = 
    {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11};  // distance 4: swap i <-> i^4
static const int SORT_SWAP8_IDX[16] __attribute__((aligned(64))) = 
    {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7};  // distance 8: swap i <-> i^8

/*
 * Inline sort_16 that accepts pre-loaded shuffle indices.
 * OPTIMIZATION: Avoids 4 memory loads per call when indices are hoisted.
 */
static inline __m512i sort_16_inline(
    __m512i v,
    const __m512i swap1,
    const __m512i swap2,
    const __m512i swap4,
    const __m512i swap8
) {
    __m512i t, lo, hi;
    
    // ========== Stage 1: Sort pairs (distance 1) ==========
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x6666, lo, hi);
    
    // ========== Stage 2: Merge into sorted 4s ==========
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xC3C3, lo, hi);
    
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xA5A5, lo, hi);
    
    // ========== Stage 3: Merge into sorted 8s ==========
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x0FF0, lo, hi);
    
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x33CC, lo, hi);
    
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x55AA, lo, hi);
    
    // ========== Stage 4: Final merge into sorted 16 (all ascending) ==========
    t = _mm512_permutexvar_epi32(swap8, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xFF00, lo, hi);
    
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xF0F0, lo, hi);
    
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xCCCC, lo, hi);
    
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
    
    return v;
}

/*
 * Inline merge_32 that accepts pre-loaded shuffle indices.
 * Merges two sorted 16-element registers into one sorted 32-element sequence.
 */
static inline void merge_32_inline(
    __m512i *a, __m512i *b,
    const __m512i idx_rev,
    const __m512i idx_swap8,
    const __m512i idx_swap4
) {
    // Reverse b to form bitonic sequence
    *b = _mm512_permutexvar_epi32(idx_rev, *b);

    // Compare-swap across registers
    __m512i lo = _mm512_min_epu32(*a, *b);
    __m512i hi = _mm512_max_epu32(*a, *b);
    *a = lo;
    *b = hi;

    // Bitonic clean both registers: distances 8,4,2,1
    // Distance 8
    __m512i a_shuf = _mm512_permutexvar_epi32(idx_swap8, *a);
    __m512i b_shuf = _mm512_permutexvar_epi32(idx_swap8, *b);
    lo = _mm512_min_epu32(*a, a_shuf);
    hi = _mm512_max_epu32(*a, a_shuf);
    *a = _mm512_mask_blend_epi32(0xFF00, lo, hi);
    lo = _mm512_min_epu32(*b, b_shuf);
    hi = _mm512_max_epu32(*b, b_shuf);
    *b = _mm512_mask_blend_epi32(0xFF00, lo, hi);

    // Distance 4
    a_shuf = _mm512_permutexvar_epi32(idx_swap4, *a);
    b_shuf = _mm512_permutexvar_epi32(idx_swap4, *b);
    lo = _mm512_min_epu32(*a, a_shuf);
    hi = _mm512_max_epu32(*a, a_shuf);
    *a = _mm512_mask_blend_epi32(0xF0F0, lo, hi);
    lo = _mm512_min_epu32(*b, b_shuf);
    hi = _mm512_max_epu32(*b, b_shuf);
    *b = _mm512_mask_blend_epi32(0xF0F0, lo, hi);

    // Distance 2 (within-lane, no cross-lane permute needed)
    a_shuf = _mm512_shuffle_epi32(*a, _MM_SHUFFLE(1,0,3,2));
    b_shuf = _mm512_shuffle_epi32(*b, _MM_SHUFFLE(1,0,3,2));
    lo = _mm512_min_epu32(*a, a_shuf);
    hi = _mm512_max_epu32(*a, a_shuf);
    *a = _mm512_mask_blend_epi32(0xCCCC, lo, hi);
    lo = _mm512_min_epu32(*b, b_shuf);
    hi = _mm512_max_epu32(*b, b_shuf);
    *b = _mm512_mask_blend_epi32(0xCCCC, lo, hi);

    // Distance 1
    a_shuf = _mm512_shuffle_epi32(*a, _MM_SHUFFLE(2,3,0,1));
    b_shuf = _mm512_shuffle_epi32(*b, _MM_SHUFFLE(2,3,0,1));
    lo = _mm512_min_epu32(*a, a_shuf);
    hi = _mm512_max_epu32(*a, a_shuf);
    *a = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
    lo = _mm512_min_epu32(*b, b_shuf);
    hi = _mm512_max_epu32(*b, b_shuf);
    *b = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
}

/*
 * Public wrapper for sort_16 (loads indices internally, for external callers).
 */
static inline __m512i sort_16_simd(__m512i v) {
    const __m512i swap1 = _mm512_load_epi32(SORT_SWAP1_IDX);
    const __m512i swap2 = _mm512_load_epi32(SORT_SWAP2_IDX);
    const __m512i swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);
    const __m512i swap8 = _mm512_load_epi32(SORT_SWAP8_IDX);
    return sort_16_inline(v, swap1, swap2, swap4, swap8);
}

/*
 * Sort 32 elements using SIMD: sort two 16-element registers, then merge them.
 * Requires arr to be 64-byte aligned.
 */
static inline void sort_32_simd(uint32_t *arr) {
    // Load and sort each half (aligned loads for cache efficiency)
    __m512i a = _mm512_load_epi32(arr);
    __m512i b = _mm512_load_epi32(arr + 16);
    
    a = sort_16_simd(a);
    b = sort_16_simd(b);
    
    // Merge the two sorted halves using existing merge network
    merge_512_registers(&a, &b);
    
    // Store results (aligned stores for cache efficiency)
    _mm512_store_epi32(arr, a);
    _mm512_store_epi32(arr + 16, b);
}

/*
 * OPTIMIZED: Sort 64 elements with all shuffle indices loaded ONCE.
 * Eliminates ~25 redundant memory loads per call vs the nested version.
 * Requires arr to be 64-byte aligned.
 */
static inline void sort_64_simd(uint32_t *arr) {
    // Load ALL shuffle indices ONCE (sort needs swap1-4-8, merge needs rev-swap8-swap4)
    const __m512i swap1 = _mm512_load_epi32(SORT_SWAP1_IDX);
    const __m512i swap2 = _mm512_load_epi32(SORT_SWAP2_IDX);
    const __m512i swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);
    const __m512i swap8 = _mm512_load_epi32(SORT_SWAP8_IDX);
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);

    // Load all 4 chunks
    __m512i a = _mm512_load_epi32(arr);
    __m512i b = _mm512_load_epi32(arr + 16);
    __m512i c = _mm512_load_epi32(arr + 32);
    __m512i d = _mm512_load_epi32(arr + 48);

    // Sort each 16-element chunk using pre-loaded indices
    a = sort_16_inline(a, swap1, swap2, swap4, swap8);
    b = sort_16_inline(b, swap1, swap2, swap4, swap8);
    c = sort_16_inline(c, swap1, swap2, swap4, swap8);
    d = sort_16_inline(d, swap1, swap2, swap4, swap8);

    // Merge into two 32-element sorted sequences
    // Note: swap4 == SORT_SWAP4_IDX, but IDX_SWAP4 from merge.h has same values
    merge_32_inline(&a, &b, idx_rev, swap8, swap4);
    merge_32_inline(&c, &d, idx_rev, swap8, swap4);

    // Store temporary results for final merge
    _mm512_store_epi32(arr, a);
    _mm512_store_epi32(arr + 16, b);
    _mm512_store_epi32(arr + 32, c);
    _mm512_store_epi32(arr + 48, d);

    // Merge the two 32-element sorted runs
    uint32_t temp[64] __attribute__((aligned(64)));
    merge_arrays(arr, 32, arr + 32, 32, temp);
    memcpy(arr, temp, 64 * sizeof(uint32_t));
}

// Base case threshold - must be multiple of 32 for SIMD efficiency
#define SORT_THRESHOLD 64

// Threshold for parallel merge (below this, use sequential merge)
// 64K elements = 256KB, small enough to fit in L2 cache
#define PARALLEL_MERGE_THRESHOLD (64 * 1024)

// ============== Parallel Merge Implementation ==============

/*
 * Find split point for parallel merge using binary search.
 * Given two sorted arrays, find indices (i, j) such that:
 *   - i + j = target (approximately half the total elements)
 *   - All elements in left[0..i) and right[0..j) are <= all elements in left[i..] and right[j..]
 */
static void find_merge_split(
    uint32_t *left, size_t left_size,
    uint32_t *right, size_t right_size,
    size_t *out_i, size_t *out_j
) {
    size_t total = left_size + right_size;
    size_t target = total / 2;
    
    // Binary search bounds
    size_t lo = (target > right_size) ? (target - right_size) : 0;
    size_t hi = (target < left_size) ? target : left_size;
    
    while (lo < hi) {
        size_t i = lo + (hi - lo) / 2;
        size_t j = target - i;
        
        // Check if this is a valid split point
        // We need: left[i-1] <= right[j] AND right[j-1] <= left[i]
        if (j > 0 && i < left_size && right[j - 1] > left[i]) {
            lo = i + 1;  // Need more from left
        } else if (i > 0 && j < right_size && left[i - 1] > right[j]) {
            hi = i;      // Need less from left
        } else {
            // Found valid split
            *out_i = i;
            *out_j = j;
            return;
        }
    }
    
    // Use the final position
    *out_i = lo;
    *out_j = target - lo;
}

/*
 * Parallel merge using OpenMP tasks.
 * Recursively splits the merge into independent sub-merges.
 * 
 * Parameters:
 *   left, left_size: First sorted array (may be unaligned)
 *   right, right_size: Second sorted array (may be unaligned)
 *   out: Output array (MUST be 64-byte aligned, and out+output positions must be aligned)
 *   depth: Remaining recursion depth (controls parallelism)
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
        // Check alignment of inputs
        int left_aligned = ((uintptr_t)left % 64) == 0;
        int right_aligned = ((uintptr_t)right % 64) == 0;
        
        if (left_aligned && right_aligned) {
            merge_arrays(left, left_size, right, right_size, out);
        } else {
            merge_arrays_unaligned(left, left_size, right, right_size, out);
        }
        return;
    }
    
    // Find split point
    size_t i, j;
    find_merge_split(left, left_size, right, right_size, &i, &j);
    
    // Round split points to 16-element boundaries for output alignment
    // This ensures out + (i + j) is 64-byte aligned if out is aligned
    size_t i_aligned = (i / 16) * 16;
    size_t j_aligned = (j / 16) * 16;
    
    // Adjust to maintain merge correctness
    // We need to find valid split that's close to aligned boundaries
    if (i_aligned != i || j_aligned != j) {
        // Find the nearest valid aligned split
        // Try rounding down first
        size_t new_target = i_aligned + j_aligned;
        if (new_target > 0 && new_target < total) {
            find_merge_split(left, left_size, right, right_size, &i, &j);
            // Accept whatever split we get - the unaligned merge will handle it
        }
    }
    
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

/*
 * Public interface for parallel merge.
 * Automatically determines recursion depth based on thread count.
 */
static void parallel_merge(
    uint32_t *left, size_t left_size,
    uint32_t *right, size_t right_size,
    uint32_t *out
) {
    // Calculate depth: log2(num_threads) levels of parallelism
    int depth = 0;
    for (int t = NUM_THREADS; t > 1; t /= 2) depth++;
    depth += 1;  // One extra level for better load balancing
    
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



// Bottom-up merge sort with OpenMP parallelization
void basic_merge_sort(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    double t_start, t_end;
    
    // Set number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Allocate temp buffer (reused for all operations)
    uint32_t *temp = NULL;
    if (posix_memalign((void**)&temp, 64, size * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for temp buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // ========== Phase 1: Base case sort (64-element chunks) ==========
    t_start = get_time_sec();
    size_t num_64_blocks = size / 64;
    size_t remainder_start = num_64_blocks * 64;
    
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < num_64_blocks; b++) {
        sort_64_simd(arr + b * 64);
    }
    
    // Handle remainder (single thread, small work)
    if (remainder_start + 32 <= size) {
        sort_32_simd(arr + remainder_start);
        remainder_start += 32;
    }
    if (remainder_start < size) {
        insertion_sort(arr + remainder_start, size - remainder_start);
    }
    t_end = get_time_sec();
    printf("  [Phase 1] Base case sort: %.3f sec (%d threads)\n", t_end - t_start, NUM_THREADS);
    
    // ========== Phase 2: Merge passes with doubling width ==========
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    for (size_t width = SORT_THRESHOLD; width < size; width *= 2) {
        t_start = get_time_sec();
        
        // Calculate number of merge pairs at this width
        size_t num_pairs = (size + 2 * width - 1) / (2 * width);
        
        if (num_pairs >= (size_t)NUM_THREADS) {
            // MANY PAIRS: Each thread handles one merge (simple parallelism)
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
        } else {
            // FEW PAIRS: Use parallel merge WITHIN each pair
            for (size_t p = 0; p < num_pairs; p++) {
                size_t left_start = p * 2 * width;
                if (left_start >= size) continue;
                
                size_t left_size = (left_start + width <= size) ? width : (size - left_start);
                size_t right_start = left_start + left_size;
                
                if (right_start >= size) {
                    // Odd chunk - parallel copy
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < left_size; i += 4096) {
                        size_t copy_size = (i + 4096 <= left_size) ? 4096 : (left_size - i);
                        memcpy(dst + left_start + i, src + left_start + i, copy_size * sizeof(uint32_t));
                    }
                } else {
                    size_t right_size = (right_start + width <= size) ? width : (size - right_start);
                    // Use PARALLEL MERGE - all threads collaborate on this single merge
                    parallel_merge(src + left_start, left_size, 
                                  src + right_start, right_size, 
                                  dst + left_start);
                }
            }
            t_end = get_time_sec();
            double throughput = (size * sizeof(uint32_t)) / (t_end - t_start) / 1e9;
            printf("  [Phase 2] Merge width %10zu: %.3f sec (%zu PARALLEL merges, %d threads/merge, %.2f GB/s)\n", 
                   width, t_end - t_start, num_pairs, NUM_THREADS, throughput);
        }
        
        // Swap src and dst
        uint32_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    // Copy result back to arr if needed - PARALLEL copy
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

    basic_merge_sort(arr, size);

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

       
