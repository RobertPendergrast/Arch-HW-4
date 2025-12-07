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

// Sorting-specific shuffle indices (not shared with merge)
static const int SORT_SWAP1_IDX[16] __attribute__((aligned(64))) = 
    {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};  // swap adjacent pairs
static const int SORT_SWAP2_IDX[16] __attribute__((aligned(64))) = 
    {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};  // reverse within groups of 4
static const int SORT_SWAP4_IDX[16] __attribute__((aligned(64))) = 
    {7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8};  // reverse within groups of 8

// IDX_REV (full reverse) is imported from merge.h - same as swap8

/*
 * Bitonic sort for 16 uint32_t elements in a single __m512i register.
 * Uses compare-swap network to fully sort the register.
 */
static inline __m512i sort_16_simd(__m512i v) {
    // Load shuffle indices
    const __m512i swap1 = _mm512_load_epi32(SORT_SWAP1_IDX);
    const __m512i swap2 = _mm512_load_epi32(SORT_SWAP2_IDX);
    const __m512i swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);
    const __m512i swap8 = _mm512_load_epi32(IDX_REV);  // full reverse, shared with merge.c
    
    __m512i t, lo, hi;
    
    // Stage 1: Sort pairs (build sorted 2s with alternating direction)
    // Pairs at positions (0,1), (2,3), ... with alternating asc/desc
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xAAAA, lo, hi);  // 0:asc, 1:desc pattern
    
    // Stage 2: Merge pairs into sorted 4s
    // Step 2a: distance 2 compare-swap
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xCCCC, lo, hi);
    
    // Step 2b: distance 1 compare-swap (clean)
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
    
    // Stage 3: Merge 4s into sorted 8s
    // Step 3a: distance 4 compare-swap
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xF0F0, lo, hi);
    
    // Step 3b: distance 2 compare-swap
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xCCCC, lo, hi);
    
    // Step 3c: distance 1 compare-swap
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
    
    // Stage 4: Merge 8s into sorted 16
    // Step 4a: distance 8 compare-swap
    t = _mm512_permutexvar_epi32(swap8, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xFF00, lo, hi);
    
    // Step 4b: distance 4 compare-swap
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xF0F0, lo, hi);
    
    // Step 4c: distance 2 compare-swap
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xCCCC, lo, hi);
    
    // Step 4d: distance 1 compare-swap
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
    
    return v;
}

/*
 * Sort 32 elements using SIMD: sort two 16-element registers, then merge them.
 */
static inline void sort_32_simd(uint32_t *arr) {
    // Load and sort each half
    __m512i a = _mm512_loadu_epi32(arr);
    __m512i b = _mm512_loadu_epi32(arr + 16);
    
    a = sort_16_simd(a);
    b = sort_16_simd(b);
    
    // Merge the two sorted halves using existing merge network
    merge_512_registers(&a, &b);
    
    // Store results
    _mm512_storeu_epi32(arr, a);
    _mm512_storeu_epi32(arr + 16, b);
}

/*
 * Sort 64 elements using SIMD: sort four 16-element chunks, then merge.
 */
static inline void sort_64_simd(uint32_t *arr) {
    // Sort each 32-element half
    sort_32_simd(arr);
    sort_32_simd(arr + 32);
    
    // Merge the two 32-element sorted runs
    // Use a small temp buffer for the merge
    uint32_t temp[64];
    merge_arrays(arr, 32, arr + 32, 32, temp);
    memcpy(arr, temp, 64 * sizeof(uint32_t));
}

// Base case threshold - must be multiple of 32 for SIMD efficiency
#define SORT_THRESHOLD 64

// Bottom-up merge sort: O(1) allocations instead of O(N) allocations!
void basic_merge_sort(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    // Step 1: Sort all base-case chunks in place using SIMD
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
            }
            
            i = right_start + right_size;
        }
        
        // Swap src and dst for next pass
        uint32_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    // If result ended up in temp, copy back to arr
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

       
