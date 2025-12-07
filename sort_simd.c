#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include "utils.h"
#include "merge.h"

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {

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
 * Bitonic sort for 16 uint32_t elements in a single __m512i register.
 * 
 * Uses standard bitonic sorting network structure:
 * - Compare-swap at distance d, where d = 2^k for each stage
 * - Alternating directions to build bitonic sequences, then final merge
 */
static inline __m512i sort_16_simd(__m512i v) {
    // Load shuffle indices for distance-d swaps
    const __m512i swap1 = _mm512_load_epi32(SORT_SWAP1_IDX);  // i <-> i^1
    const __m512i swap2 = _mm512_load_epi32(SORT_SWAP2_IDX);  // i <-> i^2
    const __m512i swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);  // i <-> i^4
    const __m512i swap8 = _mm512_load_epi32(SORT_SWAP8_IDX);  // i <-> i^8
    
    __m512i t, lo, hi;
    
    // ========== Stage 1: Sort pairs (distance 1) ==========
    // Compare (0,1), (2,3), (4,5), ... with alternating directions
    // Pairs at 0,4,8,12: ascending (bit pattern: keep min at lower index)
    // Pairs at 2,6,10,14: descending (keep max at lower index)
    // This creates bitonic sequences of length 4
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Positions: 0=lo,1=hi, 2=hi,3=lo, 4=lo,5=hi, 6=hi,7=lo, ...
    // Mask where bit=1 means take hi: 0110 0110 0110 0110 = 0x6666
    v = _mm512_mask_blend_epi32(0x6666, lo, hi);
    
    // ========== Stage 2: Merge into sorted 4s ==========
    // Step 2a: distance 2
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Groups 0-3,8-11: ascending (lo at 0,1; hi at 2,3)
    // Groups 4-7,12-15: descending (hi at 4,5; lo at 6,7)
    // Mask: 1100 0011 1100 0011 = 0xC3C3
    v = _mm512_mask_blend_epi32(0xC3C3, lo, hi);
    
    // Step 2b: distance 1
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Groups 0-3,8-11: ascending (10 10 pattern)
    // Groups 4-7,12-15: descending (01 01 pattern)
    // Mask: 1010 0101 1010 0101 = 0xA5A5
    v = _mm512_mask_blend_epi32(0xA5A5, lo, hi);
    
    // ========== Stage 3: Merge into sorted 8s ==========
    // Step 3a: distance 4
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Group 0-7: ascending (lo at 0-3, hi at 4-7) -> 11110000 = 0xF0
    // Group 8-15: descending (hi at 8-11, lo at 12-15) -> 00001111 = 0x0F
    // Mask: 0000 1111 1111 0000 = 0x0FF0
    v = _mm512_mask_blend_epi32(0x0FF0, lo, hi);
    
    // Step 3b: distance 2
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Group 0-7: ascending -> 1100 1100 = 0xCC
    // Group 8-15: descending -> 0011 0011 = 0x33
    // Mask: 0011 0011 1100 1100 = 0x33CC
    v = _mm512_mask_blend_epi32(0x33CC, lo, hi);
    
    // Step 3c: distance 1
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Group 0-7: ascending -> 1010 1010 = 0xAA
    // Group 8-15: descending -> 0101 0101 = 0x55
    // Mask: 0101 0101 1010 1010 = 0x55AA
    v = _mm512_mask_blend_epi32(0x55AA, lo, hi);
    
    // ========== Stage 4: Final merge into sorted 16 (all ascending) ==========
    // Step 4a: distance 8
    t = _mm512_permutexvar_epi32(swap8, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Ascending: lo at 0-7, hi at 8-15 -> mask = 0xFF00
    v = _mm512_mask_blend_epi32(0xFF00, lo, hi);
    
    // Step 4b: distance 4
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Ascending in both halves: 11110000 11110000 = 0xF0F0
    v = _mm512_mask_blend_epi32(0xF0F0, lo, hi);
    
    // Step 4c: distance 2
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Ascending: 1100 1100 1100 1100 = 0xCCCC
    v = _mm512_mask_blend_epi32(0xCCCC, lo, hi);
    
    // Step 4d: distance 1
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    // Ascending: 1010 1010 1010 1010 = 0xAAAA
    v = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
    
    return v;
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
 * Sort 64 elements using SIMD: sort four 16-element chunks, then merge.
 * Requires arr to be 64-byte aligned.
 */
static inline void sort_64_simd(uint32_t *arr) {
    // Sort each 32-element half
    sort_32_simd(arr);
    sort_32_simd(arr + 32);
    
    // Merge the two 32-element sorted runs
    // Use a 64-byte aligned temp buffer for cache-aligned SIMD operations
    uint32_t temp[64] __attribute__((aligned(64)));
    merge_arrays(arr, 32, arr + 32, 32, temp);
    memcpy(arr, temp, 64 * sizeof(uint32_t));
}

// Base case threshold - must be multiple of 32 for SIMD efficiency
#define SORT_THRESHOLD 64

// Helper for timing
static inline double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Bottom-up merge sort: O(1) allocations instead of O(N) allocations!
void basic_merge_sort(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    double t_start, t_end;
    int pass_num = 0;
    
    // Step 1: Sort all base-case chunks in place using SIMD
    t_start = get_time_sec();
    size_t i = 0;
    
    // Process full 64-element chunks with SIMD
    for (; i + 64 <= size; i += 64) {
        sort_64_simd(arr + i);
    }
    
    // Process remaining 32-element chunk if present
    if (i + 32 <= size) {
        sort_32_simd(arr + i);
        i += 32;
    }
    
    // Handle any remainder with insertion sort
    if (i < size) {
        insertion_sort(arr + i, size - i);
    }
    t_end = get_time_sec();
    printf("  [Pass %2d] Base case sort (chunks of %d): %.3f sec\n", 
           pass_num++, SORT_THRESHOLD, t_end - t_start);
    
    // Step 2: Allocate ONE temp buffer for all merges
    uint32_t *temp = NULL;
    if (posix_memalign((void**)&temp, 64, size * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for temp buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // Step 3: Bottom-up merge passes
    // Each pass doubles the size of sorted runs
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    for (size_t width = SORT_THRESHOLD; width < size; width *= 2) {
        t_start = get_time_sec();
        size_t num_merges = 0;
        
        // Merge adjacent pairs of runs
        size_t i = 0;
        while (i < size) {
            size_t left_start = i;
            size_t left_size = (left_start + width <= size) ? width : (size - left_start);
            size_t right_start = left_start + left_size;
            size_t right_size = 0;
            
            if (right_start < size) {
                right_size = (right_start + width <= size) ? width : (size - right_start);
            }
            
            if (right_size == 0) {
                // No right half - just copy left to dst
                memcpy(dst + left_start, src + left_start, left_size * sizeof(uint32_t));
            } else {
                // Merge left and right into dst
                merge_arrays(src + left_start, left_size, 
                           src + right_start, right_size, 
                           dst + left_start);
                num_merges++;
            }
            
            i = right_start + right_size;
        }
        
        t_end = get_time_sec();
        double throughput = (size * sizeof(uint32_t)) / (t_end - t_start) / 1e9;
        printf("  [Pass %2d] Merge width %10zu: %.3f sec (%zu merges, %.2f GB/s)\n", 
               pass_num++, width, t_end - t_start, num_merges, throughput);
        
        // Swap src and dst for next pass
        uint32_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    // If result ended up in temp, copy back to arr
    if (src != arr) {
        t_start = get_time_sec();
        memcpy(arr, src, size * sizeof(uint32_t));
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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    basic_merge_sort(arr, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);

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
    if (write_array_to_file(argv[2], arr, size) != 0) {
        free(arr);
        return 1;
    }

    free(arr);
    return 0;
}

       
