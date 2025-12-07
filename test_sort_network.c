#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "merge.h"

// Copy of the sorting network for testing
static const int SORT_SWAP1_IDX[16] __attribute__((aligned(64))) = 
    {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14};  // distance 1
static const int SORT_SWAP2_IDX[16] __attribute__((aligned(64))) = 
    {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13};  // distance 2
static const int SORT_SWAP4_IDX[16] __attribute__((aligned(64))) = 
    {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11};  // distance 4
static const int SORT_SWAP8_IDX[16] __attribute__((aligned(64))) = 
    {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7};  // distance 8

void print_vec(const char *label, __m512i v) {
    uint32_t arr[16];
    _mm512_storeu_epi32(arr, v);
    printf("%s: ", label);
    for (int i = 0; i < 16; i++) printf("%2u ", arr[i]);
    printf("\n");
}

// Test the sort_16 function step by step (with corrected masks)
__m512i sort_16_debug(__m512i v) {
    const __m512i swap1 = _mm512_load_epi32(SORT_SWAP1_IDX);
    const __m512i swap2 = _mm512_load_epi32(SORT_SWAP2_IDX);
    const __m512i swap4 = _mm512_load_epi32(SORT_SWAP4_IDX);
    const __m512i swap8 = _mm512_load_epi32(SORT_SWAP8_IDX);
    
    __m512i t, lo, hi;
    
    print_vec("Input   ", v);
    
    // Stage 1: Sort pairs with ALTERNATING direction (mask 0x6666)
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x6666, lo, hi);
    print_vec("Stage 1 ", v);
    
    // Stage 2a: distance 2 (mask 0xC3C3)
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xC3C3, lo, hi);
    print_vec("Stage 2a", v);
    
    // Stage 2b: distance 1 (mask 0xA5A5)
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xA5A5, lo, hi);
    print_vec("Stage 2b", v);
    
    // Stage 3a: distance 4 (mask 0x0FF0)
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x0FF0, lo, hi);
    print_vec("Stage 3a", v);
    
    // Stage 3b: distance 2 (mask 0x33CC)
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x33CC, lo, hi);
    print_vec("Stage 3b", v);
    
    // Stage 3c: distance 1 (mask 0x55AA)
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0x55AA, lo, hi);
    print_vec("Stage 3c", v);
    
    // Stage 4a: distance 8 (mask 0xFF00)
    t = _mm512_permutexvar_epi32(swap8, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xFF00, lo, hi);
    print_vec("Stage 4a", v);
    
    // Stage 4b: distance 4 (mask 0xF0F0)
    t = _mm512_permutexvar_epi32(swap4, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xF0F0, lo, hi);
    print_vec("Stage 4b", v);
    
    // Stage 4c: distance 2 (mask 0xCCCC)
    t = _mm512_permutexvar_epi32(swap2, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xCCCC, lo, hi);
    print_vec("Stage 4c", v);
    
    // Stage 4d: distance 1 (mask 0xAAAA)
    t = _mm512_permutexvar_epi32(swap1, v);
    lo = _mm512_min_epu32(v, t);
    hi = _mm512_max_epu32(v, t);
    v = _mm512_mask_blend_epi32(0xAAAA, lo, hi);
    print_vec("Stage 4d", v);
    
    return v;
}

int is_sorted_16(uint32_t *arr) {
    for (int i = 1; i < 16; i++) {
        if (arr[i] < arr[i-1]) return 0;
    }
    return 1;
}

int main() {
    printf("=== Test 1: Reverse order ===\n");
    uint32_t test1[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    __m512i v1 = _mm512_loadu_epi32(test1);
    v1 = sort_16_debug(v1);
    _mm512_storeu_epi32(test1, v1);
    printf("Sorted: %s\n\n", is_sorted_16(test1) ? "YES" : "NO");
    
    printf("=== Test 2: Random ===\n");
    uint32_t test2[16] = {5,12,3,8,15,1,9,6,14,2,11,4,13,7,10,0};
    __m512i v2 = _mm512_loadu_epi32(test2);
    v2 = sort_16_debug(v2);
    _mm512_storeu_epi32(test2, v2);
    printf("Sorted: %s\n\n", is_sorted_16(test2) ? "YES" : "NO");
    
    printf("=== Test 3: Already sorted ===\n");
    uint32_t test3[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    __m512i v3 = _mm512_loadu_epi32(test3);
    v3 = sort_16_debug(v3);
    _mm512_storeu_epi32(test3, v3);
    printf("Sorted: %s\n\n", is_sorted_16(test3) ? "YES" : "NO");
    
    printf("=== Test 4: All same ===\n");
    uint32_t test4[16] = {5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5};
    __m512i v4 = _mm512_loadu_epi32(test4);
    v4 = sort_16_debug(v4);
    _mm512_storeu_epi32(test4, v4);
    printf("Sorted: %s\n\n", is_sorted_16(test4) ? "YES" : "NO");

    return 0;
}
