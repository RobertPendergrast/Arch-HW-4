#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include "utils.h"

// Number of threads for OpenMP parallelization
#define NUM_THREADS 16

// ============== STABLE SORT USING (VALUE, INDEX) KEYS ==============
// Each 32-bit value is packed with its 32-bit original index into a 64-bit key:
//   key = (value << 32) | index
// Sorting by key sorts by value first, then by index for ties → STABLE!
// With 64-bit elements, we get 8 elements per 512-bit register (vs 16 for 32-bit)

// Pack values with indices into 64-bit keys
static void pack_keys(const uint32_t *values, uint64_t *keys, size_t size) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        keys[i] = ((uint64_t)values[i] << 32) | (uint32_t)i;
    }
}

// Unpack 64-bit keys back to 32-bit values
static void unpack_keys(const uint64_t *keys, uint32_t *values, size_t size) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        values[i] = (uint32_t)(keys[i] >> 32);
    }
}

// ============== 64-bit SIMD Sorting Network (8 elements) ==============

// Shuffle indices for 8-element sorting network
static const long long SORT64_SWAP1_IDX[8] __attribute__((aligned(64))) = 
    {1,0,3,2,5,4,7,6};  // distance 1: swap i <-> i^1
static const long long SORT64_SWAP2_IDX[8] __attribute__((aligned(64))) = 
    {2,3,0,1,6,7,4,5};  // distance 2: swap i <-> i^2
static const long long SORT64_SWAP4_IDX[8] __attribute__((aligned(64))) = 
    {4,5,6,7,0,1,2,3};  // distance 4: swap i <-> i^4

// Reverse for merge
static const long long IDX64_REV[8] __attribute__((aligned(64))) = 
    {7,6,5,4,3,2,1,0};

/*
 * Bitonic sort for 8 uint64_t elements in a single __m512i register.
 * Uses unsigned 64-bit comparison for correct sorting of packed (value, index) keys.
 */
static inline __m512i sort_8_simd(__m512i v) {
    const __m512i swap1 = _mm512_load_epi64(SORT64_SWAP1_IDX);
    const __m512i swap2 = _mm512_load_epi64(SORT64_SWAP2_IDX);
    const __m512i swap4 = _mm512_load_epi64(SORT64_SWAP4_IDX);
    
    __m512i t, lo, hi;
    
    // Stage 1: Sort pairs (distance 1)
    t = _mm512_permutexvar_epi64(swap1, v);
    lo = _mm512_min_epu64(v, t);
    hi = _mm512_max_epu64(v, t);
    // Pairs 0,4: ascending, pairs 2,6: descending → mask 0b01100110 = 0x66
    v = _mm512_mask_blend_epi64(0x66, lo, hi);
    
    // Stage 2: Merge into sorted 4s
    // Step 2a: distance 2
    t = _mm512_permutexvar_epi64(swap2, v);
    lo = _mm512_min_epu64(v, t);
    hi = _mm512_max_epu64(v, t);
    // Groups 0-3: ascending, 4-7: descending → 0b11000011 = 0xC3
    v = _mm512_mask_blend_epi64(0xC3, lo, hi);
    
    // Step 2b: distance 1
    t = _mm512_permutexvar_epi64(swap1, v);
    lo = _mm512_min_epu64(v, t);
    hi = _mm512_max_epu64(v, t);
    // 0b10100101 = 0xA5
    v = _mm512_mask_blend_epi64(0xA5, lo, hi);
    
    // Stage 3: Final merge into sorted 8 (all ascending)
    // Step 3a: distance 4
    t = _mm512_permutexvar_epi64(swap4, v);
    lo = _mm512_min_epu64(v, t);
    hi = _mm512_max_epu64(v, t);
    // 0b11110000 = 0xF0
    v = _mm512_mask_blend_epi64(0xF0, lo, hi);
    
    // Step 3b: distance 2
    t = _mm512_permutexvar_epi64(swap2, v);
    lo = _mm512_min_epu64(v, t);
    hi = _mm512_max_epu64(v, t);
    // 0b11001100 = 0xCC
    v = _mm512_mask_blend_epi64(0xCC, lo, hi);
    
    // Step 3c: distance 1
    t = _mm512_permutexvar_epi64(swap1, v);
    lo = _mm512_min_epu64(v, t);
    hi = _mm512_max_epu64(v, t);
    // 0b10101010 = 0xAA
    v = _mm512_mask_blend_epi64(0xAA, lo, hi);
    
    return v;
}

/*
 * Merge two sorted 8-element registers into one sorted 16-element sequence.
 * Output: *a contains smallest 8, *b contains largest 8
 */
static inline void merge_8x8_registers(__m512i *a, __m512i *b) {
    const __m512i idx_rev = _mm512_load_epi64(IDX64_REV);
    const __m512i swap4 = _mm512_load_epi64(SORT64_SWAP4_IDX);
    const __m512i swap2 = _mm512_load_epi64(SORT64_SWAP2_IDX);
    const __m512i swap1 = _mm512_load_epi64(SORT64_SWAP1_IDX);
    
    // Reverse b to form bitonic sequence
    *b = _mm512_permutexvar_epi64(idx_rev, *b);
    
    // Compare-swap across registers (distance 8)
    __m512i lo = _mm512_min_epu64(*a, *b);
    __m512i hi = _mm512_max_epu64(*a, *b);
    *a = lo;
    *b = hi;
    
    // Bitonic clean both registers: distances 4, 2, 1
    // Distance 4
    __m512i a_shuf = _mm512_permutexvar_epi64(swap4, *a);
    __m512i b_shuf = _mm512_permutexvar_epi64(swap4, *b);
    lo = _mm512_min_epu64(*a, a_shuf);
    hi = _mm512_max_epu64(*a, a_shuf);
    *a = _mm512_mask_blend_epi64(0xF0, lo, hi);
    lo = _mm512_min_epu64(*b, b_shuf);
    hi = _mm512_max_epu64(*b, b_shuf);
    *b = _mm512_mask_blend_epi64(0xF0, lo, hi);
    
    // Distance 2
    a_shuf = _mm512_permutexvar_epi64(swap2, *a);
    b_shuf = _mm512_permutexvar_epi64(swap2, *b);
    lo = _mm512_min_epu64(*a, a_shuf);
    hi = _mm512_max_epu64(*a, a_shuf);
    *a = _mm512_mask_blend_epi64(0xCC, lo, hi);
    lo = _mm512_min_epu64(*b, b_shuf);
    hi = _mm512_max_epu64(*b, b_shuf);
    *b = _mm512_mask_blend_epi64(0xCC, lo, hi);
    
    // Distance 1
    a_shuf = _mm512_permutexvar_epi64(swap1, *a);
    b_shuf = _mm512_permutexvar_epi64(swap1, *b);
    lo = _mm512_min_epu64(*a, a_shuf);
    hi = _mm512_max_epu64(*a, a_shuf);
    *a = _mm512_mask_blend_epi64(0xAA, lo, hi);
    lo = _mm512_min_epu64(*b, b_shuf);
    hi = _mm512_max_epu64(*b, b_shuf);
    *b = _mm512_mask_blend_epi64(0xAA, lo, hi);
}

/*
 * Sort 16 elements (two registers) using SIMD.
 */
static inline void sort_16_64bit(uint64_t *arr) {
    __m512i a = _mm512_load_epi64(arr);
    __m512i b = _mm512_load_epi64(arr + 8);
    
    a = sort_8_simd(a);
    b = sort_8_simd(b);
    merge_8x8_registers(&a, &b);
    
    _mm512_store_epi64(arr, a);
    _mm512_store_epi64(arr + 8, b);
}

/*
 * Sort 32 elements (four registers) using SIMD.
 */
static inline void sort_32_64bit(uint64_t *arr) {
    // Sort each 16-element half
    sort_16_64bit(arr);
    sort_16_64bit(arr + 16);
    
    // Merge using scalar stable merge (simpler, still fast for 32 elements)
    uint64_t temp[32];
    size_t i = 0, j = 16, k = 0;
    while (i < 16 && j < 32) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    while (i < 16) temp[k++] = arr[i++];
    while (j < 32) temp[k++] = arr[j++];
    memcpy(arr, temp, 32 * sizeof(uint64_t));
}

// ============== Scalar Stable Merge for 64-bit keys ==============

static void merge_local_64(uint64_t *left, uint64_t *right, uint64_t *out,
                           size_t size_left, size_t size_right) {
    size_t i = 0, j = 0, k = 0;
    while (i < size_left && j < size_right) {
        if (left[i] <= right[j]) {  // <= for stability
            out[k++] = left[i++];
        } else {
            out[k++] = right[j++];
        }
    }
    if (i < size_left) {
        memcpy(out + k, left + i, (size_left - i) * sizeof(uint64_t));
    } else if (j < size_right) {
        memcpy(out + k, right + j, (size_right - j) * sizeof(uint64_t));
    }
}

// ============== SIMD Merge for 64-bit keys ==============

static void merge_arrays_64(uint64_t *left, size_t size_left,
                            uint64_t *right, size_t size_right,
                            uint64_t *out) {
    // Small arrays: use scalar stable merge
    if (size_left < 8 || size_right < 8) {
        merge_local_64(left, right, out, size_left, size_right);
        return;
    }
    
    const __m512i idx_rev = _mm512_load_epi64(IDX64_REV);
    const __m512i swap4 = _mm512_load_epi64(SORT64_SWAP4_IDX);
    const __m512i swap2 = _mm512_load_epi64(SORT64_SWAP2_IDX);
    const __m512i swap1 = _mm512_load_epi64(SORT64_SWAP1_IDX);
    
    // Load first chunks
    __m512i left_reg = _mm512_load_epi64(left);
    __m512i right_reg = _mm512_load_epi64(right);
    
    // Initial merge
    right_reg = _mm512_permutexvar_epi64(idx_rev, right_reg);
    __m512i lo = _mm512_min_epu64(left_reg, right_reg);
    __m512i hi = _mm512_max_epu64(left_reg, right_reg);
    left_reg = lo;
    right_reg = hi;
    
    // Bitonic clean left_reg
    __m512i shuf = _mm512_permutexvar_epi64(swap4, left_reg);
    lo = _mm512_min_epu64(left_reg, shuf);
    hi = _mm512_max_epu64(left_reg, shuf);
    left_reg = _mm512_mask_blend_epi64(0xF0, lo, hi);
    
    shuf = _mm512_permutexvar_epi64(swap2, left_reg);
    lo = _mm512_min_epu64(left_reg, shuf);
    hi = _mm512_max_epu64(left_reg, shuf);
    left_reg = _mm512_mask_blend_epi64(0xCC, lo, hi);
    
    shuf = _mm512_permutexvar_epi64(swap1, left_reg);
    lo = _mm512_min_epu64(left_reg, shuf);
    hi = _mm512_max_epu64(left_reg, shuf);
    left_reg = _mm512_mask_blend_epi64(0xAA, lo, hi);
    
    _mm512_store_epi64(out, left_reg);
    
    // Bitonic clean right_reg (for pending)
    shuf = _mm512_permutexvar_epi64(swap4, right_reg);
    lo = _mm512_min_epu64(right_reg, shuf);
    hi = _mm512_max_epu64(right_reg, shuf);
    right_reg = _mm512_mask_blend_epi64(0xF0, lo, hi);
    
    shuf = _mm512_permutexvar_epi64(swap2, right_reg);
    lo = _mm512_min_epu64(right_reg, shuf);
    hi = _mm512_max_epu64(right_reg, shuf);
    right_reg = _mm512_mask_blend_epi64(0xCC, lo, hi);
    
    shuf = _mm512_permutexvar_epi64(swap1, right_reg);
    lo = _mm512_min_epu64(right_reg, shuf);
    hi = _mm512_max_epu64(right_reg, shuf);
    right_reg = _mm512_mask_blend_epi64(0xAA, lo, hi);
    
    size_t left_idx = 8;
    size_t right_idx = 8;
    
    // Main SIMD merge loop
    while (left_idx + 8 <= size_left && right_idx + 8 <= size_right) {
        __m512i next_left = _mm512_load_epi64(left + left_idx);
        __m512i next_right = _mm512_load_epi64(right + right_idx);
        
        // Choose which chunk to merge next
        int take_left = left[left_idx] <= right[right_idx];
        __mmask8 mask = take_left ? 0xFF : 0x00;
        left_reg = _mm512_mask_blend_epi64(mask, next_right, next_left);
        
        left_idx += take_left * 8;
        right_idx += (!take_left) * 8;
        
        // Merge with pending (right_reg)
        left_reg = _mm512_permutexvar_epi64(idx_rev, left_reg);
        lo = _mm512_min_epu64(left_reg, right_reg);
        hi = _mm512_max_epu64(left_reg, right_reg);
        left_reg = lo;
        right_reg = hi;
        
        // Bitonic clean left_reg
        shuf = _mm512_permutexvar_epi64(swap4, left_reg);
        lo = _mm512_min_epu64(left_reg, shuf);
        hi = _mm512_max_epu64(left_reg, shuf);
        left_reg = _mm512_mask_blend_epi64(0xF0, lo, hi);
        
        shuf = _mm512_permutexvar_epi64(swap2, left_reg);
        lo = _mm512_min_epu64(left_reg, shuf);
        hi = _mm512_max_epu64(left_reg, shuf);
        left_reg = _mm512_mask_blend_epi64(0xCC, lo, hi);
        
        shuf = _mm512_permutexvar_epi64(swap1, left_reg);
        lo = _mm512_min_epu64(left_reg, shuf);
        hi = _mm512_max_epu64(left_reg, shuf);
        left_reg = _mm512_mask_blend_epi64(0xAA, lo, hi);
        
        _mm512_store_epi64(out + left_idx + right_idx - 16, left_reg);
        
        // Bitonic clean right_reg
        shuf = _mm512_permutexvar_epi64(swap4, right_reg);
        lo = _mm512_min_epu64(right_reg, shuf);
        hi = _mm512_max_epu64(right_reg, shuf);
        right_reg = _mm512_mask_blend_epi64(0xF0, lo, hi);
        
        shuf = _mm512_permutexvar_epi64(swap2, right_reg);
        lo = _mm512_min_epu64(right_reg, shuf);
        hi = _mm512_max_epu64(right_reg, shuf);
        right_reg = _mm512_mask_blend_epi64(0xCC, lo, hi);
        
        shuf = _mm512_permutexvar_epi64(swap1, right_reg);
        lo = _mm512_min_epu64(right_reg, shuf);
        hi = _mm512_max_epu64(right_reg, shuf);
        right_reg = _mm512_mask_blend_epi64(0xAA, lo, hi);
    }
    
    // Handle remainders with scalar merge
    size_t output_pos = left_idx + right_idx - 8;
    uint64_t pending[8];
    _mm512_storeu_epi64(pending, right_reg);
    
    size_t left_remaining = size_left - left_idx;
    size_t right_remaining = size_right - right_idx;
    
    if (left_remaining == 0 && right_remaining == 0) {
        _mm512_store_epi64(out + output_pos, right_reg);
    } else {
        uint64_t remainder_merged[16];
        if (left_remaining < right_remaining) {
            merge_local_64(left + left_idx, pending, remainder_merged, left_remaining, 8);
            merge_local_64(remainder_merged, right + right_idx, out + output_pos,
                          left_remaining + 8, right_remaining);
        } else {
            merge_local_64(right + right_idx, pending, remainder_merged, right_remaining, 8);
            merge_local_64(remainder_merged, left + left_idx, out + output_pos,
                          right_remaining + 8, left_remaining);
        }
    }
}

// ============== Insertion sort for small arrays ==============

static inline void insertion_sort_64(uint64_t *arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        uint64_t key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

// ============== Main Sort Implementation ==============

#define SORT_THRESHOLD 32  // Base case size (must be multiple of 16 for SIMD)
#define L3_CHUNK_ELEMENTS (4 * 1024 * 1024)  // 4M 64-bit elements = 32MB, fits in L3

static inline double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Sort a chunk in place
static void sort_chunk_parallel_64(uint64_t *arr, size_t chunk_size, uint64_t *temp) {
    // Step 1: Base case sort
    size_t num_32_blocks = chunk_size / 32;
    size_t remainder_start = num_32_blocks * 32;
    
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < num_32_blocks; b++) {
        sort_32_64bit(arr + b * 32);
    }
    
    // Handle remainder
    if (remainder_start + 16 <= chunk_size) {
        sort_16_64bit(arr + remainder_start);
        remainder_start += 16;
    }
    if (remainder_start < chunk_size) {
        insertion_sort_64(arr + remainder_start, chunk_size - remainder_start);
    }
    
    // Step 2: Merge passes
    uint64_t *src = arr;
    uint64_t *dst = temp;
    
    for (size_t width = SORT_THRESHOLD; width < chunk_size; width *= 2) {
        size_t num_pairs = (chunk_size + 2 * width - 1) / (2 * width);
        
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t p = 0; p < num_pairs; p++) {
            size_t left_start = p * 2 * width;
            if (left_start >= chunk_size) continue;
            
            size_t left_size = (left_start + width <= chunk_size) ? width : (chunk_size - left_start);
            size_t right_start = left_start + left_size;
            
            if (right_start >= chunk_size) {
                memcpy(dst + left_start, src + left_start, left_size * sizeof(uint64_t));
            } else {
                size_t right_size = (right_start + width <= chunk_size) ? width : (chunk_size - right_start);
                merge_arrays_64(src + left_start, left_size,
                               src + right_start, right_size,
                               dst + left_start);
            }
        }
        
        uint64_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    if (src != arr) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < chunk_size; i += 4096) {
            size_t copy_size = (i + 4096 <= chunk_size) ? 4096 : (chunk_size - i);
            memcpy(arr + i, src + i, copy_size * sizeof(uint64_t));
        }
    }
}

// Main stable sort function
void stable_merge_sort(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    double t_start, t_end;
    omp_set_num_threads(NUM_THREADS);
    
    // Allocate 64-bit key buffer
    uint64_t *keys = NULL;
    if (posix_memalign((void**)&keys, 64, size * sizeof(uint64_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for keys buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate temp buffer
    uint64_t *temp = NULL;
    if (posix_memalign((void**)&temp, 64, size * sizeof(uint64_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for temp buffer\n");
        free(keys);
        exit(EXIT_FAILURE);
    }
    
    // Pack values with indices
    t_start = get_time_sec();
    pack_keys(arr, keys, size);
    t_end = get_time_sec();
    printf("  [Pack   ] Pack (value, index) keys: %.3f sec\n", t_end - t_start);
    
    // Phase 1: Sort L3-sized chunks
    t_start = get_time_sec();
    size_t num_chunks = (size + L3_CHUNK_ELEMENTS - 1) / L3_CHUNK_ELEMENTS;
    
    for (size_t c = 0; c < num_chunks; c++) {
        size_t start = c * L3_CHUNK_ELEMENTS;
        size_t chunk_size = (start + L3_CHUNK_ELEMENTS <= size) ? L3_CHUNK_ELEMENTS : (size - start);
        sort_chunk_parallel_64(keys + start, chunk_size, temp + start);
    }
    t_end = get_time_sec();
    printf("  [Phase 1] Sort %zu L3 chunks (4M elements each): %.3f sec\n", num_chunks, t_end - t_start);
    
    // Phase 2: Merge chunks
    if (size > L3_CHUNK_ELEMENTS) {
        uint64_t *src = keys;
        uint64_t *dst = temp;
        
        for (size_t width = L3_CHUNK_ELEMENTS; width < size; width *= 2) {
            t_start = get_time_sec();
            size_t num_pairs = (size + 2 * width - 1) / (2 * width);
            
            #pragma omp parallel for schedule(dynamic, 1)
            for (size_t p = 0; p < num_pairs; p++) {
                size_t left_start = p * 2 * width;
                if (left_start >= size) continue;
                
                size_t left_size = (left_start + width <= size) ? width : (size - left_start);
                size_t right_start = left_start + left_size;
                
                if (right_start >= size) {
                    memcpy(dst + left_start, src + left_start, left_size * sizeof(uint64_t));
                } else {
                    size_t right_size = (right_start + width <= size) ? width : (size - right_start);
                    merge_arrays_64(src + left_start, left_size,
                                   src + right_start, right_size,
                                   dst + left_start);
                }
            }
            
            t_end = get_time_sec();
            double throughput = (size * sizeof(uint64_t)) / (t_end - t_start) / 1e9;
            printf("  [Phase 2] Merge width %10zu: %.3f sec (%zu merges, %.2f GB/s)\n",
                   width, t_end - t_start, num_pairs, throughput);
            
            uint64_t *swap = src;
            src = dst;
            dst = swap;
        }
        
        // Ensure final result is in keys
        if (src != keys) {
            memcpy(keys, src, size * sizeof(uint64_t));
        }
    }
    
    // Unpack back to values
    t_start = get_time_sec();
    unpack_keys(keys, arr, size);
    t_end = get_time_sec();
    printf("  [Unpack ] Unpack values: %.3f sec\n", t_end - t_start);
    
    free(keys);
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
    printf("Using STABLE sort with 64-bit (value, index) keys\n");
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    stable_merge_sort(arr, size);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);
    
    // Verify sorted
    if (verify_sortedness(arr, size)) {
        printf("Array sorted successfully! (STABLE)\n");
    } else {
        printf("Error: Array is not sorted correctly!\n");
        free(arr);
        return 1;
    }
    
    // Write output
    if (write_array_to_file(argv[2], arr, size) != 0) {
        free(arr);
        return 1;
    }
    
    free(arr);
    return 0;
}
