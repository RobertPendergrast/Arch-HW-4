#include "merge.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to extract the 15th (last) element from a __m512i register
static inline uint32_t extract_last_512(const __m512i *reg) {
    // Element 15 is in lane 3 (bits 384-511), position 3 within that lane
    __m128i lane3 = _mm512_extracti32x4_epi32(*reg, 3);
    return (uint32_t)_mm_extract_epi32(lane3, 3);
}
// 128 Constants
// Control to reverse. First two bits (11) means the new first value is the old
// last value, etc...
const int _128_REV = 0b00011011;
const int _128_BLEND_1 = 0b1100;
const int _128_BLEND_2 = 0b1010;
const int _128_BLEND_3 = _128_BLEND_1;
const int _128_SHUFFLE_1 = 0b01001110;
const int _128_SHUFFLE_2 = 0b10110001;
const int _128_SHUFFLE_3 = _128_SHUFFLE_1;;
const int _128_SHUFFLE_3_L = 0b11011000;
const int _128_SHUFFLE_3_H = 0b01110010;

/*
 * Takes in two m128i registers and merges them in place.
 * NOTE: Uses unsigned comparison (epu32) for correct uint32_t sorting!
 */
void merge_128_registers(
    __m128i *left,
    __m128i *right
) {
    // Level 1 - Get min/max values (unsigned comparison)
    __m128i L1 = _mm_min_epu32(*left, *right);
    __m128i H1 = _mm_max_epu32(*left, *right);

    // Shuffle
    __m128i L1p = _mm_blend_epi32(L1, H1, _128_BLEND_1);
    __m128i H1p = _mm_blend_epi32(H1, L1, _128_BLEND_1);
    H1p = _mm_shuffle_epi32(H1p, _128_SHUFFLE_1);

    // Level 2 - Get min/max values (unsigned comparison)
    __m128i L2 = _mm_min_epu32(L1p, H1p);
    __m128i H2 = _mm_max_epu32(L1p, H1p);

    // Shuffle
    __m128i L2p = _mm_blend_epi32(L2, H2, _128_BLEND_2);
    __m128i H2p = _mm_blend_epi32(H2, L2, _128_BLEND_2);
    H2p = _mm_shuffle_epi32(H2p, _128_SHUFFLE_2);

    // Level 3 - Get min/max values (unsigned comparison)
    __m128i L3 = _mm_min_epu32(L2p, H2p);
    __m128i H3 = _mm_max_epu32(L2p, H2p);

    //Shuffle
    __m128i H3p =  _mm_shuffle_epi32(H3, _128_SHUFFLE_3);
    __m128i L3p =  _mm_blend_epi32(L3, H3p, _128_BLEND_3);
    H3p = _mm_blend_epi32(H3p, L3, _128_BLEND_3);
    H3p = _mm_shuffle_epi32(H3p, _128_SHUFFLE_3_H);
    L3p = _mm_shuffle_epi32(L3p, _128_SHUFFLE_3_L);

    // Output
    *left = L3p;
    *right = H3p;
}


// 512 Constants - blend masks for each level of bitonic merge
// Blend mask: 1 = take from second operand, 0 = take from first
const __mmask16 _512_BLEND_8  = 0xFF00;  // upper 8 from second: 0b1111111100000000
const __mmask16 _512_BLEND_4  = 0xF0F0;  // groups of 4:         0b1111000011110000
const __mmask16 _512_BLEND_2  = 0xCCCC;  // groups of 2:         0b1100110011001100
const __mmask16 _512_BLEND_1  = 0xAAAA;  // alternating:         0b1010101010101010

/*
 * Takes in two m512i registers and merges them in place.
 * Full 512-bit bitonic merge network - no fallback to 128-bit.
 * 
 * Algorithm: Standard bitonic merge
 * 1. Reverse right register to form bitonic sequence
 * 2. Compare-swap across registers (distance 16)
 * 3. Bitonic clean each register: distances 8, 4, 2, 1
 */
// Static shuffle indices for cross-lane permutations (avoids recreating each call)
static const int IDX_REV_ARR[16] __attribute__((aligned(64))) = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
static const int IDX_SWAP8_ARR[16] __attribute__((aligned(64))) = {7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8};
static const int IDX_SWAP4_ARR[16] __attribute__((aligned(64))) = {11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4};

inline __attribute__((always_inline)) void merge_512_registers(
    __m512i *left,
    __m512i *right
) {
    // Load shuffle indices from static arrays
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV_ARR);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8_ARR);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4_ARR);

    // Step 0: Reverse right register to form bitonic sequence
    *right = _mm512_permutexvar_epi32(idx_rev, *right);

    // Step 1: Compare-swap across registers (distance 16)
    // After this, left has smaller elements, right has larger elements
    // NOTE: Using epu32 (unsigned) for correct comparison of uint32_t values!
    __m512i lo = _mm512_min_epu32(*left, *right);
    __m512i hi = _mm512_max_epu32(*left, *right);
    *left = lo;
    *right = hi;

    // Now each register needs independent bitonic clean (distances 8,4,2,1)
    // We process both registers in parallel for better instruction-level parallelism
    
    // Step 2: Distance 8 - compare elements i with i+8
    __m512i left_shuf = _mm512_permutexvar_epi32(idx_swap8, *left);
    __m512i right_shuf = _mm512_permutexvar_epi32(idx_swap8, *right);
    
    lo = _mm512_min_epu32(*left, left_shuf);
    hi = _mm512_max_epu32(*left, left_shuf);
    *left = _mm512_mask_blend_epi32(_512_BLEND_8, lo, hi);
    
    lo = _mm512_min_epu32(*right, right_shuf);
    hi = _mm512_max_epu32(*right, right_shuf);
    *right = _mm512_mask_blend_epi32(_512_BLEND_8, lo, hi);

    // Step 3: Distance 4 - compare elements i with i+4 within each group of 8
    left_shuf = _mm512_permutexvar_epi32(idx_swap4, *left);
    right_shuf = _mm512_permutexvar_epi32(idx_swap4, *right);
    
    lo = _mm512_min_epu32(*left, left_shuf);
    hi = _mm512_max_epu32(*left, left_shuf);
    *left = _mm512_mask_blend_epi32(_512_BLEND_4, lo, hi);
    
    lo = _mm512_min_epu32(*right, right_shuf);
    hi = _mm512_max_epu32(*right, right_shuf);
    *right = _mm512_mask_blend_epi32(_512_BLEND_4, lo, hi);

    // Step 4: Distance 2 - compare elements i with i+2 within each group of 4
    // Can use within-lane shuffle (faster than cross-lane permute)
    // _MM_SHUFFLE(1,0,3,2) swaps pairs: [0,1,2,3] -> [2,3,0,1]
    left_shuf = _mm512_shuffle_epi32(*left, _MM_SHUFFLE(1,0,3,2));
    right_shuf = _mm512_shuffle_epi32(*right, _MM_SHUFFLE(1,0,3,2));
    
    lo = _mm512_min_epu32(*left, left_shuf);
    hi = _mm512_max_epu32(*left, left_shuf);
    *left = _mm512_mask_blend_epi32(_512_BLEND_2, lo, hi);
    
    lo = _mm512_min_epu32(*right, right_shuf);
    hi = _mm512_max_epu32(*right, right_shuf);
    *right = _mm512_mask_blend_epi32(_512_BLEND_2, lo, hi);

    // Step 5: Distance 1 - compare adjacent elements
    // _MM_SHUFFLE(2,3,0,1) swaps adjacent: [0,1,2,3] -> [1,0,3,2]
    left_shuf = _mm512_shuffle_epi32(*left, _MM_SHUFFLE(2,3,0,1));
    right_shuf = _mm512_shuffle_epi32(*right, _MM_SHUFFLE(2,3,0,1));
    
    lo = _mm512_min_epu32(*left, left_shuf);
    hi = _mm512_max_epu32(*left, left_shuf);
    *left = _mm512_mask_blend_epi32(_512_BLEND_1, lo, hi);
    
    lo = _mm512_min_epu32(*right, right_shuf);
    hi = _mm512_max_epu32(*right, right_shuf);
    *right = _mm512_mask_blend_epi32(_512_BLEND_1, lo, hi);
}

void merge_local(uint32_t* left, uint32_t* right, uint32_t* arr, int size_left, int size_right) {
    int i = 0;
    int j = 0;
    int k = 0;
    // Merge the two arrays into arr
    while (i < size_left && j < size_right) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }
    // Copy remaining elements with memcpy
    if (i < size_left) {
        memcpy(arr + k, left + i, (size_left - i) * sizeof(uint32_t));
    } else if (j < size_right) {
        memcpy(arr + k, right + j, (size_right - j) * sizeof(uint32_t));
    }
}

void merge_arrays(
    uint32_t *left,
    size_t size_left,
    uint32_t *right,
    size_t size_right,
    uint32_t *arr
) {
    // Small arrays: fall back to scalar merge
    if (size_left < 16 || size_right < 16) {
        merge_local(left, right, arr, size_left, size_right);
        return;
    }

    //creating a mm512i register from the left and right arrays
    __m512i left_reg = _mm512_loadu_epi32((__m512i*) left);
    __m512i right_reg = _mm512_loadu_epi32((__m512i*) right);
    merge_512_registers(&left_reg, &right_reg);
    _mm512_storeu_epi32(arr, left_reg);
    
    size_t right_idx = 16;
    size_t left_idx = 16;
    
    // Main SIMD loop: continue while BOTH sides can provide full 16-element chunks
    while (left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        if (left[left_idx] <= right[right_idx]) {
            left_reg = _mm512_loadu_epi32(left + left_idx);
            left_idx += 16;
        } else {
            left_reg = _mm512_loadu_epi32(right + right_idx);
            right_idx += 16;
        }
        merge_512_registers(&left_reg, &right_reg);
        _mm512_storeu_epi32(arr + left_idx + right_idx - 32, left_reg);
    }
    
    // At this point, at least one side cannot provide a full 16-element chunk.
    // We have three sorted sequences to merge:
    //   1. The 16 pending elements in right_reg
    //   2. left[left_idx : size_left]   (0 to size_left - left_idx elements)
    //   3. right[right_idx : size_right] (0 to size_right - right_idx elements)
    
    size_t left_remaining = size_left - left_idx;
    size_t right_remaining = size_right - right_idx;
    size_t output_pos = left_idx + right_idx - 16;  // Where pending elements should start
    
    // Extract the 16 pending elements from right_reg
    uint32_t pending[16];
    _mm512_storeu_epi32(pending, right_reg);
    
    // Merge left_remainder with right_remainder into a temp buffer
    // Max size: both could have up to (size - 16) elements if the other was exhausted early,
    // but typically this is small. Use dynamic allocation for safety.
    size_t remainder_size = left_remaining + right_remaining;
    
    if (remainder_size == 0) {
        // No remainders - just output the pending 16
        _mm512_storeu_epi32(arr + output_pos, right_reg);
    } else {
        // Stack buffer - max 30 elements (15 from each side worst case)
        uint32_t remainder_merged[32];
        
        // Merge the two remainders
        merge_local(left + left_idx, right + right_idx, remainder_merged, 
              left_remaining, right_remaining);
        
        // Merge pending 16 with the merged remainders into final output
        merge_local(pending, remainder_merged, arr + output_pos, 16, remainder_size);
    }
}