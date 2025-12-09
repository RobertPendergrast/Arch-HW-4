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
void basic_merge_sort_kv(uint32_t *arr, uint32_t *payload, size_t size);

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array_kv(uint32_t *arr, uint32_t *payload, size_t size) {
    basic_merge_sort_kv(arr, payload, size);
}

// ============================================================================
// KEY-VALUE SORTING IMPLEMENTATION
// ============================================================================
// This version tracks a payload array alongside keys for stability testing.
//
// DESIGN DIFFERENCES FROM NON-KV VERSION:
// 1. TIE-BREAKING: Original uses min/max intrinsics which select the second
//    operand when equal. KV version uses cmpgt+blend which selects the first
//    operand when equal. This means equal keys may end up in different positions.
//
// 2. MEMORY: Uses 2x the memory (separate key and payload arrays).
//
// 3. BANDWIDTH: 2x memory bandwidth usage since we load/store both arrays.
//
// 4. INSTRUCTION COUNT: More instructions per comparison (cmpgt+2*blend vs min+max).
// ============================================================================

// Insertion sort for small/remainder arrays - KV version
static inline void insertion_sort_kv(uint32_t *keys, uint32_t *payload, size_t size) {
    for (size_t i = 1; i < size; i++) {
        uint32_t key = keys[i];
        uint32_t pay = payload[i];
        size_t j = i;
        while (j > 0 && keys[j - 1] > key) {
            keys[j] = keys[j - 1];
            payload[j] = payload[j - 1];
            j--;
        }
        keys[j] = key;
        payload[j] = pay;
    }
}

// ============== SIMD Sorting Network (KV Version) ==============
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
 * KV version of sort_16_inline.
 * 
 * DIFFERENCE FROM ORIGINAL: Uses explicit cmpgt + blend instead of min/max.
 * When keys are equal, this version keeps the element from the FIRST operand,
 * while the original min/max keeps the element from the SECOND operand.
 */
static inline void sort_16_inline_kv(
    __m512i *key,
    __m512i *pay,
    const __m512i swap1,
    const __m512i swap2,
    const __m512i swap4,
    const __m512i swap8
) {
    __m512i key_t, pay_t, lo_key, hi_key, lo_pay, hi_pay;
    __mmask16 cmp;
    
    // Helper macro for compare-swap with payload tracking
    // mask specifies which positions should take from hi (1) vs lo (0)
    #define COMPARE_SWAP_KV(swap_idx, blend_mask) \
        key_t = _mm512_permutexvar_epi32(swap_idx, *key); \
        pay_t = _mm512_permutexvar_epi32(swap_idx, *pay); \
        cmp = _mm512_cmpgt_epu32_mask(*key, key_t); \
        lo_key = _mm512_mask_blend_epi32(cmp, *key, key_t); \
        hi_key = _mm512_mask_blend_epi32(cmp, key_t, *key); \
        lo_pay = _mm512_mask_blend_epi32(cmp, *pay, pay_t); \
        hi_pay = _mm512_mask_blend_epi32(cmp, pay_t, *pay); \
        *key = _mm512_mask_blend_epi32(blend_mask, lo_key, hi_key); \
        *pay = _mm512_mask_blend_epi32(blend_mask, lo_pay, hi_pay);
    
    // ========== Stage 1: Sort pairs (distance 1) ==========
    COMPARE_SWAP_KV(swap1, 0x6666)
    
    // ========== Stage 2: Merge into sorted 4s ==========
    COMPARE_SWAP_KV(swap2, 0xC3C3)
    COMPARE_SWAP_KV(swap1, 0xA5A5)
    
    // ========== Stage 3: Merge into sorted 8s ==========
    COMPARE_SWAP_KV(swap4, 0x0FF0)
    COMPARE_SWAP_KV(swap2, 0x33CC)
    COMPARE_SWAP_KV(swap1, 0x55AA)
    
    // ========== Stage 4: Final merge into sorted 16 (all ascending) ==========
    COMPARE_SWAP_KV(swap8, 0xFF00)
    COMPARE_SWAP_KV(swap4, 0xF0F0)
    COMPARE_SWAP_KV(swap2, 0xCCCC)
    COMPARE_SWAP_KV(swap1, 0xAAAA)
    
    #undef COMPARE_SWAP_KV
}

/*
 * KV version of merge_32_inline.
 * Merges two sorted 16-element register pairs into one sorted 32-element sequence.
 * 
 * DIFFERENCE FROM ORIGINAL: Same tie-breaking difference as sort_16_inline_kv.
 */
static inline void merge_32_inline_kv(
    __m512i *a_key, __m512i *b_key,
    __m512i *a_pay, __m512i *b_pay,
    const __m512i idx_rev,
    const __m512i idx_swap8,
    const __m512i idx_swap4
) {
    __m512i lo_key, hi_key, lo_pay, hi_pay;
    __mmask16 cmp;
    
    // Reverse b to form bitonic sequence
    *b_key = _mm512_permutexvar_epi32(idx_rev, *b_key);
    *b_pay = _mm512_permutexvar_epi32(idx_rev, *b_pay);

    // Compare-swap across registers
    cmp = _mm512_cmpgt_epu32_mask(*a_key, *b_key);
    lo_key = _mm512_mask_blend_epi32(cmp, *a_key, *b_key);
    hi_key = _mm512_mask_blend_epi32(cmp, *b_key, *a_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *a_pay, *b_pay);
    hi_pay = _mm512_mask_blend_epi32(cmp, *b_pay, *a_pay);
    *a_key = lo_key;
    *b_key = hi_key;
    *a_pay = lo_pay;
    *b_pay = hi_pay;

    // Bitonic clean both registers: distances 8,4,2,1
    
    // Helper macro for distance-based compare-swap
    #define CLEAN_STEP_KV(reg_key, reg_pay, shuf_key, shuf_pay, blend_mask) \
        cmp = _mm512_cmpgt_epu32_mask(*reg_key, shuf_key); \
        lo_key = _mm512_mask_blend_epi32(cmp, *reg_key, shuf_key); \
        hi_key = _mm512_mask_blend_epi32(cmp, shuf_key, *reg_key); \
        lo_pay = _mm512_mask_blend_epi32(cmp, *reg_pay, shuf_pay); \
        hi_pay = _mm512_mask_blend_epi32(cmp, shuf_pay, *reg_pay); \
        *reg_key = _mm512_mask_blend_epi32(blend_mask, lo_key, hi_key); \
        *reg_pay = _mm512_mask_blend_epi32(blend_mask, lo_pay, hi_pay);

    // Distance 8
    __m512i a_key_shuf = _mm512_permutexvar_epi32(idx_swap8, *a_key);
    __m512i b_key_shuf = _mm512_permutexvar_epi32(idx_swap8, *b_key);
    __m512i a_pay_shuf = _mm512_permutexvar_epi32(idx_swap8, *a_pay);
    __m512i b_pay_shuf = _mm512_permutexvar_epi32(idx_swap8, *b_pay);
    CLEAN_STEP_KV(a_key, a_pay, a_key_shuf, a_pay_shuf, 0xFF00)
    CLEAN_STEP_KV(b_key, b_pay, b_key_shuf, b_pay_shuf, 0xFF00)

    // Distance 4
    a_key_shuf = _mm512_permutexvar_epi32(idx_swap4, *a_key);
    b_key_shuf = _mm512_permutexvar_epi32(idx_swap4, *b_key);
    a_pay_shuf = _mm512_permutexvar_epi32(idx_swap4, *a_pay);
    b_pay_shuf = _mm512_permutexvar_epi32(idx_swap4, *b_pay);
    CLEAN_STEP_KV(a_key, a_pay, a_key_shuf, a_pay_shuf, 0xF0F0)
    CLEAN_STEP_KV(b_key, b_pay, b_key_shuf, b_pay_shuf, 0xF0F0)

    // Distance 2 (within-lane, no cross-lane permute needed)
    a_key_shuf = _mm512_shuffle_epi32(*a_key, _MM_SHUFFLE(1,0,3,2));
    b_key_shuf = _mm512_shuffle_epi32(*b_key, _MM_SHUFFLE(1,0,3,2));
    a_pay_shuf = _mm512_shuffle_epi32(*a_pay, _MM_SHUFFLE(1,0,3,2));
    b_pay_shuf = _mm512_shuffle_epi32(*b_pay, _MM_SHUFFLE(1,0,3,2));
    CLEAN_STEP_KV(a_key, a_pay, a_key_shuf, a_pay_shuf, 0xCCCC)
    CLEAN_STEP_KV(b_key, b_pay, b_key_shuf, b_pay_shuf, 0xCCCC)

    // Distance 1
    a_key_shuf = _mm512_shuffle_epi32(*a_key, _MM_SHUFFLE(2,3,0,1));
    b_key_shuf = _mm512_shuffle_epi32(*b_key, _MM_SHUFFLE(2,3,0,1));
    a_pay_shuf = _mm512_shuffle_epi32(*a_pay, _MM_SHUFFLE(2,3,0,1));
    b_pay_shuf = _mm512_shuffle_epi32(*b_pay, _MM_SHUFFLE(2,3,0,1));
    CLEAN_STEP_KV(a_key, a_pay, a_key_shuf, a_pay_shuf, 0xAAAA)
    CLEAN_STEP_KV(b_key, b_pay, b_key_shuf, b_pay_shuf, 0xAAAA)
    
    #undef CLEAN_STEP_KV
}

/*
 * Public wrapper for sort_16_kv (loads indices internally).
 */
static inline void sort_16_simd_kv(__m512i *key, __m512i *pay) {
    const __m512i swap1 = _mm512_load_epi32(SORT_SWAP1_IDX);
    const __m512i swap2 = _mm512_load_epi32(SORT_SWAP2_IDX);
    const __m512i swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);
    const __m512i swap8 = _mm512_load_epi32(SORT_SWAP8_IDX);
    sort_16_inline_kv(key, pay, swap1, swap2, swap4, swap8);
}

/*
 * Sort 32 elements using SIMD - KV version.
 * Requires arr to be 64-byte aligned.
 */
static inline void sort_32_simd_kv(uint32_t *keys, uint32_t *payload) {
    // Load keys and payload
    __m512i a_key = _mm512_load_epi32(keys);
    __m512i b_key = _mm512_load_epi32(keys + 16);
    __m512i a_pay = _mm512_load_epi32(payload);
    __m512i b_pay = _mm512_load_epi32(payload + 16);
    
    // Sort each half
    sort_16_simd_kv(&a_key, &a_pay);
    sort_16_simd_kv(&b_key, &b_pay);
    
    // Load merge indices
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(SORT_SWAP8_IDX);
    const __m512i idx_swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);
    
    // Merge the two sorted halves
    merge_32_inline_kv(&a_key, &b_key, &a_pay, &b_pay, idx_rev, idx_swap8, idx_swap4);
    
    // Store results
    _mm512_store_epi32(keys, a_key);
    _mm512_store_epi32(keys + 16, b_key);
    _mm512_store_epi32(payload, a_pay);
    _mm512_store_epi32(payload + 16, b_pay);
}

/*
 * Sort 64 elements with all shuffle indices loaded ONCE - KV version.
 * Requires arr to be 64-byte aligned.
 * 
 * DIFFERENCE FROM ORIGINAL: After SIMD sorting, uses merge_arrays_kv instead
 * of merge_arrays for the final 32+32 merge.
 */
static inline void sort_64_simd_kv(uint32_t *keys, uint32_t *payload) {
    // Load ALL shuffle indices ONCE
    const __m512i swap1 = _mm512_load_epi32(SORT_SWAP1_IDX);
    const __m512i swap2 = _mm512_load_epi32(SORT_SWAP2_IDX);
    const __m512i swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);
    const __m512i swap8 = _mm512_load_epi32(SORT_SWAP8_IDX);
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);

    // Load all 4 chunks (keys and payload)
    __m512i a_key = _mm512_load_epi32(keys);
    __m512i b_key = _mm512_load_epi32(keys + 16);
    __m512i c_key = _mm512_load_epi32(keys + 32);
    __m512i d_key = _mm512_load_epi32(keys + 48);
    __m512i a_pay = _mm512_load_epi32(payload);
    __m512i b_pay = _mm512_load_epi32(payload + 16);
    __m512i c_pay = _mm512_load_epi32(payload + 32);
    __m512i d_pay = _mm512_load_epi32(payload + 48);

    // Sort each 16-element chunk
    sort_16_inline_kv(&a_key, &a_pay, swap1, swap2, swap4, swap8);
    sort_16_inline_kv(&b_key, &b_pay, swap1, swap2, swap4, swap8);
    sort_16_inline_kv(&c_key, &c_pay, swap1, swap2, swap4, swap8);
    sort_16_inline_kv(&d_key, &d_pay, swap1, swap2, swap4, swap8);

    // Merge into two 32-element sorted sequences
    merge_32_inline_kv(&a_key, &b_key, &a_pay, &b_pay, idx_rev, swap8, swap4);
    merge_32_inline_kv(&c_key, &d_key, &c_pay, &d_pay, idx_rev, swap8, swap4);

    // Store temporary results for final merge
    _mm512_store_epi32(keys, a_key);
    _mm512_store_epi32(keys + 16, b_key);
    _mm512_store_epi32(keys + 32, c_key);
    _mm512_store_epi32(keys + 48, d_key);
    _mm512_store_epi32(payload, a_pay);
    _mm512_store_epi32(payload + 16, b_pay);
    _mm512_store_epi32(payload + 32, c_pay);
    _mm512_store_epi32(payload + 48, d_pay);

    // Merge the two 32-element sorted runs using KV merge
    uint32_t temp_key[64] __attribute__((aligned(64)));
    uint32_t temp_pay[64] __attribute__((aligned(64)));
    merge_arrays_kv(keys, payload, 32, keys + 32, payload + 32, 32, temp_key, temp_pay);
    memcpy(keys, temp_key, 64 * sizeof(uint32_t));
    memcpy(payload, temp_pay, 64 * sizeof(uint32_t));
}

// Base case threshold - must be multiple of 32 for SIMD efficiency
#define SORT_THRESHOLD 64

// L3 cache size: 32 MiB = 8M uint32_t elements
// For KV, we have 2x data, so use half the chunk size
#define L3_CHUNK_ELEMENTS (2 * 1024 * 1024)  // DIFFERENCE: Half of non-KV version

// Threshold for parallel merge (below this, use sequential merge)
#define PARALLEL_MERGE_THRESHOLD (64 * 1024)

// ============== Parallel Merge Implementation (KV Version) ==============

/*
 * Find split point for parallel merge using binary search.
 * Same as non-KV version - only looks at keys.
 */
static void find_merge_split_kv(
    uint32_t *left_key, size_t left_size,
    uint32_t *right_key, size_t right_size,
    size_t *out_i, size_t *out_j
) {
    size_t total = left_size + right_size;
    size_t target = total / 2;
    
    size_t lo = (target > right_size) ? (target - right_size) : 0;
    size_t hi = (target < left_size) ? target : left_size;
    
    while (lo < hi) {
        size_t i = lo + (hi - lo) / 2;
        size_t j = target - i;
        
        if (j > 0 && i < left_size && right_key[j - 1] > left_key[i]) {
            lo = i + 1;
        } else if (i > 0 && j < right_size && left_key[i - 1] > right_key[j]) {
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

// Scalar merge for KV (used by merge_arrays_kv internally, but we need a local version)
static void merge_local_kv_inline(
    uint32_t* left_key, uint32_t* left_pay,
    uint32_t* right_key, uint32_t* right_pay,
    uint32_t* out_key, uint32_t* out_pay,
    size_t size_left, size_t size_right
) {
    size_t i = 0, j = 0, k = 0;
    while (i < size_left && j < size_right) {
        if (left_key[i] <= right_key[j]) {
            out_key[k] = left_key[i];
            out_pay[k] = left_pay[i];
            i++; k++;
        } else {
            out_key[k] = right_key[j];
            out_pay[k] = right_pay[j];
            j++; k++;
        }
    }
    if (i < size_left) {
        memcpy(out_key + k, left_key + i, (size_left - i) * sizeof(uint32_t));
        memcpy(out_pay + k, left_pay + i, (size_left - i) * sizeof(uint32_t));
    } else if (j < size_right) {
        memcpy(out_key + k, right_key + j, (size_right - j) * sizeof(uint32_t));
        memcpy(out_pay + k, right_pay + j, (size_right - j) * sizeof(uint32_t));
    }
}

/*
 * Parallel merge for KV version.
 * 
 * DIFFERENCE FROM ORIGINAL: No merge_arrays_unaligned_kv exists yet, so we
 * fall back to scalar merge for unaligned inputs. This may be slower for
 * deeply nested parallel merges.
 */
static void parallel_merge_impl_kv(
    uint32_t *left_key, uint32_t *left_pay, size_t left_size,
    uint32_t *right_key, uint32_t *right_pay, size_t right_size,
    uint32_t *out_key, uint32_t *out_pay,
    int depth
) {
    size_t total = left_size + right_size;
    
    // Base case: small enough or no more parallelism
    if (depth <= 0 || total < PARALLEL_MERGE_THRESHOLD) {
        // Check alignment
        int left_aligned = ((uintptr_t)left_key % 64) == 0 && ((uintptr_t)left_pay % 64) == 0;
        int right_aligned = ((uintptr_t)right_key % 64) == 0 && ((uintptr_t)right_pay % 64) == 0;
        
        if (left_aligned && right_aligned) {
            merge_arrays_kv(left_key, left_pay, left_size,
                           right_key, right_pay, right_size,
                           out_key, out_pay);
        } else {
            merge_arrays_unaligned_kv(left_key, left_pay, left_size,
                                      right_key, right_pay, right_size,
                                      out_key, out_pay);
        }
        return;
    }
    
    // Find split point
    size_t i, j;
    find_merge_split_kv(left_key, left_size, right_key, right_size, &i, &j);
    
    size_t out_split = i + j;
    
    // Spawn parallel tasks for the two halves
    #pragma omp task
    parallel_merge_impl_kv(left_key, left_pay, i,
                           right_key, right_pay, j,
                           out_key, out_pay, depth - 1);
    
    #pragma omp task
    parallel_merge_impl_kv(left_key + i, left_pay + i, left_size - i,
                           right_key + j, right_pay + j, right_size - j,
                           out_key + out_split, out_pay + out_split, depth - 1);
    
    #pragma omp taskwait
}

static void parallel_merge_kv(
    uint32_t *left_key, uint32_t *left_pay, size_t left_size,
    uint32_t *right_key, uint32_t *right_pay, size_t right_size,
    uint32_t *out_key, uint32_t *out_pay
) {
    int depth = 0;
    for (int t = NUM_THREADS; t > 1; t /= 2) depth++;
    depth += 1;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_merge_impl_kv(left_key, left_pay, left_size,
                                   right_key, right_pay, right_size,
                                   out_key, out_pay, depth);
        }
    }
}

// Helper for timing
static inline double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Sort a single chunk with ALL threads collaborating - KV version
static void sort_chunk_parallel_kv(uint32_t *keys, uint32_t *payload, size_t chunk_size,
                                    uint32_t *temp_key, uint32_t *temp_pay) {
    // Step 1: Base case sort (64-element chunks) - PARALLEL within chunk
    size_t num_64_blocks = chunk_size / 64;
    size_t remainder_start = num_64_blocks * 64;
    
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < num_64_blocks; b++) {
        sort_64_simd_kv(keys + b * 64, payload + b * 64);
    }
    
    // Handle remainder (single thread, small work)
    if (remainder_start + 32 <= chunk_size) {
        sort_32_simd_kv(keys + remainder_start, payload + remainder_start);
        remainder_start += 32;
    }
    if (remainder_start < chunk_size) {
        insertion_sort_kv(keys + remainder_start, payload + remainder_start,
                          chunk_size - remainder_start);
    }
    
    // Step 2: Merge passes within this chunk
    uint32_t *src_key = keys;
    uint32_t *src_pay = payload;
    uint32_t *dst_key = temp_key;
    uint32_t *dst_pay = temp_pay;
    
    for (size_t width = SORT_THRESHOLD; width < chunk_size; width *= 2) {
        size_t num_pairs = (chunk_size + 2 * width - 1) / (2 * width);
        
        if (num_pairs > 1) {
            // MULTIPLE PAIRS: Use parallel for
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t p = 0; p < num_pairs; p++) {
                size_t left_start = p * 2 * width;
                if (left_start >= chunk_size) continue;
                
                size_t left_size = (left_start + width <= chunk_size) ? width : (chunk_size - left_start);
                size_t right_start = left_start + left_size;
                
                if (right_start >= chunk_size) {
                    memcpy(dst_key + left_start, src_key + left_start, left_size * sizeof(uint32_t));
                    memcpy(dst_pay + left_start, src_pay + left_start, left_size * sizeof(uint32_t));
                } else {
                    size_t right_size = (right_start + width <= chunk_size) ? width : (chunk_size - right_start);
                    // Use CACHED KV merge
                    merge_arrays_cached_kv(src_key + left_start, src_pay + left_start, left_size,
                                           src_key + right_start, src_pay + right_start, right_size,
                                           dst_key + left_start, dst_pay + left_start);
                }
            }
        } else {
            // SINGLE PAIR: Use parallel merge
            size_t left_size = (width <= chunk_size) ? width : chunk_size;
            size_t right_start = left_size;
            
            if (right_start < chunk_size) {
                size_t right_size = chunk_size - right_start;
                parallel_merge_kv(src_key, src_pay, left_size,
                                  src_key + right_start, src_pay + right_start, right_size,
                                  dst_key, dst_pay);
            } else {
                memcpy(dst_key, src_key, chunk_size * sizeof(uint32_t));
                memcpy(dst_pay, src_pay, chunk_size * sizeof(uint32_t));
            }
        }
        
        // Swap src and dst
        uint32_t *swap_key = src_key; src_key = dst_key; dst_key = swap_key;
        uint32_t *swap_pay = src_pay; src_pay = dst_pay; dst_pay = swap_pay;
    }
    
    // Copy result back if needed - PARALLEL copy
    if (src_key != keys) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < chunk_size; i += 4096) {
            size_t copy_size = (i + 4096 <= chunk_size) ? 4096 : (chunk_size - i);
            memcpy(keys + i, src_key + i, copy_size * sizeof(uint32_t));
            memcpy(payload + i, src_pay + i, copy_size * sizeof(uint32_t));
        }
    }
}

// Bottom-up merge sort with L3 cache blocking - KV version
void basic_merge_sort_kv(uint32_t *keys, uint32_t *payload, size_t size) {
    if (size <= 1) return;
    
    double t_start, t_end;
    
    // Set number of threads
    omp_set_num_threads(NUM_THREADS);
    
    // Allocate temp buffers for both keys and payload
    uint32_t *temp_key = NULL;
    uint32_t *temp_pay = NULL;
    if (posix_memalign((void**)&temp_key, 64, size * sizeof(uint32_t)) != 0 ||
        posix_memalign((void**)&temp_pay, 64, size * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for temp buffers\n");
        exit(EXIT_FAILURE);
    }
    
    // Warmup temp buffers
    t_start = get_time_sec();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 4096 / sizeof(uint32_t)) {
        temp_key[i] = 0;
        temp_pay[i] = 0;
    }
    t_end = get_time_sec();
    printf("  [Warmup] temp buffers (page faults): %.3f sec\n", t_end - t_start);
    
    // ========== Phase 1: Sort each L3-sized chunk ==========
    t_start = get_time_sec();
    size_t num_chunks = (size + L3_CHUNK_ELEMENTS - 1) / L3_CHUNK_ELEMENTS;
    
    for (size_t c = 0; c < num_chunks; c++) {
        size_t start = c * L3_CHUNK_ELEMENTS;
        size_t chunk_size = (start + L3_CHUNK_ELEMENTS <= size) ? L3_CHUNK_ELEMENTS : (size - start);
        sort_chunk_parallel_kv(keys + start, payload + start, chunk_size,
                               temp_key + start, temp_pay + start);
    }
    t_end = get_time_sec();
    printf("  [Phase 1] Sort %zu L3 chunks (%d elements each): %.3f sec (%d threads)\n", 
           num_chunks, L3_CHUNK_ELEMENTS, t_end - t_start, NUM_THREADS);
    
    // ========== Phase 2: Merge L3-sized chunks together ==========
    if (size > L3_CHUNK_ELEMENTS) {
        uint32_t *src_key = keys;
        uint32_t *src_pay = payload;
        uint32_t *dst_key = temp_key;
        uint32_t *dst_pay = temp_pay;
        size_t width = L3_CHUNK_ELEMENTS;
        
        while (width < size) {
            size_t num_pairs = (size + 2 * width - 1) / (2 * width);
            
            if (num_pairs < (size_t)NUM_THREADS) {
                printf("  [Phase 2] Stopping L3 merge at width %zu (%zu pairs < %d threads)\n", 
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
                    memcpy(dst_key + left_start, src_key + left_start, left_size * sizeof(uint32_t));
                    memcpy(dst_pay + left_start, src_pay + left_start, left_size * sizeof(uint32_t));
                } else {
                    size_t right_size = (right_start + width <= size) ? width : (size - right_start);
                    merge_arrays_kv(src_key + left_start, src_pay + left_start, left_size,
                                    src_key + right_start, src_pay + right_start, right_size,
                                    dst_key + left_start, dst_pay + left_start);
                }
            }
            t_end = get_time_sec();
            double throughput = (2 * size * sizeof(uint32_t)) / (t_end - t_start) / 1e9;  // 2x for KV
            printf("  [Phase 2] Merge width %10zu: %.3f sec (%zu parallel merges, %.2f GB/s)\n", 
                   width, t_end - t_start, num_pairs, throughput);
            
            // Swap
            uint32_t *swap = src_key; src_key = dst_key; dst_key = swap;
            swap = src_pay; src_pay = dst_pay; dst_pay = swap;
            width *= 2;
        }
        
        // Finish with parallel merge
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
                        memcpy(dst_key + left_start + i, src_key + left_start + i, copy_size * sizeof(uint32_t));
                        memcpy(dst_pay + left_start + i, src_pay + left_start + i, copy_size * sizeof(uint32_t));
                    }
                } else {
                    size_t right_size = (right_start + width <= size) ? width : (size - right_start);
                    parallel_merge_kv(src_key + left_start, src_pay + left_start, left_size,
                                      src_key + right_start, src_pay + right_start, right_size,
                                      dst_key + left_start, dst_pay + left_start);
                }
            }
            
            t_end = get_time_sec();
            double throughput = (2 * size * sizeof(uint32_t)) / (t_end - t_start) / 1e9;
            printf("  [Phase 2] Merge width %10zu: %.3f sec (%zu parallel merges, %d threads, %.2f GB/s)\n", 
                   width, t_end - t_start, num_pairs, NUM_THREADS, throughput);
            
            uint32_t *swap = src_key; src_key = dst_key; dst_key = swap;
            swap = src_pay; src_pay = dst_pay; dst_pay = swap;
            width *= 2;
        }
        
        // Copy result back if needed
        if (src_key != keys) {
            t_start = get_time_sec();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < size; i += 4096) {
                size_t copy_size = (i + 4096 <= size) ? 4096 : (size - i);
                memcpy(keys + i, src_key + i, copy_size * sizeof(uint32_t));
                memcpy(payload + i, src_pay + i, copy_size * sizeof(uint32_t));
            }
            t_end = get_time_sec();
            printf("  [Final ] Copy back: %.3f sec\n", t_end - t_start);
        }
    }
    
    free(temp_key);
    free(temp_pay);
}
