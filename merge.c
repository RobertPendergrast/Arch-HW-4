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
// Shared shuffle indices (exported via merge.h)
// Note: stored in memory order (elem0 first)
const int IDX_REV[16] __attribute__((aligned(64))) = {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
const int IDX_SWAP8[16] __attribute__((aligned(64))) = {8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7};
const int IDX_SWAP4[16] __attribute__((aligned(64))) = {4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11};

// Inline version that accepts pre-loaded indices to avoid repeated memory loads
static inline __attribute__((always_inline)) void merge_512_inline(
    __m512i *left,
    __m512i *right,
    const __m512i idx_rev,
    const __m512i idx_swap8,
    const __m512i idx_swap4
) {
    // Step 0: Reverse right register to form bitonic sequence
    *right = _mm512_permutexvar_epi32(idx_rev, *right);

    // Step 1: Compare-swap across registers (distance 16)
    __m512i lo = _mm512_min_epu32(*left, *right);
    __m512i hi = _mm512_max_epu32(*left, *right);
    *left = lo;
    *right = hi;

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

    // Step 4: Distance 2 - within-lane shuffle (faster than cross-lane permute)
    left_shuf = _mm512_shuffle_epi32(*left, _MM_SHUFFLE(1,0,3,2));
    right_shuf = _mm512_shuffle_epi32(*right, _MM_SHUFFLE(1,0,3,2));
    
    lo = _mm512_min_epu32(*left, left_shuf);
    hi = _mm512_max_epu32(*left, left_shuf);
    *left = _mm512_mask_blend_epi32(_512_BLEND_2, lo, hi);
    
    lo = _mm512_min_epu32(*right, right_shuf);
    hi = _mm512_max_epu32(*right, right_shuf);
    *right = _mm512_mask_blend_epi32(_512_BLEND_2, lo, hi);

    // Step 5: Distance 1 - compare adjacent elements
    left_shuf = _mm512_shuffle_epi32(*left, _MM_SHUFFLE(2,3,0,1));
    right_shuf = _mm512_shuffle_epi32(*right, _MM_SHUFFLE(2,3,0,1));
    
    lo = _mm512_min_epu32(*left, left_shuf);
    hi = _mm512_max_epu32(*left, left_shuf);
    *left = _mm512_mask_blend_epi32(_512_BLEND_1, lo, hi);
    
    lo = _mm512_min_epu32(*right, right_shuf);
    hi = _mm512_max_epu32(*right, right_shuf);
    *right = _mm512_mask_blend_epi32(_512_BLEND_1, lo, hi);
}

// Public wrapper that loads indices (for external callers)
inline __attribute__((always_inline)) void merge_512_registers(
    __m512i *left,
    __m512i *right
) {
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4);
    merge_512_inline(left, right, idx_rev, idx_swap8, idx_swap4);
}

// Key-Value version: sorts payload alongside keys without using payload for comparison
// Uses explicit comparison masks to track element movement
static inline __attribute__((always_inline)) void merge_512_inline_kv(
    __m512i *left_key,
    __m512i *right_key,
    __m512i *left_payload,
    __m512i *right_payload,
    const __m512i idx_rev,
    const __m512i idx_swap8,
    const __m512i idx_swap4
) {
    // Step 0: Reverse right registers to form bitonic sequence
    *right_key = _mm512_permutexvar_epi32(idx_rev, *right_key);
    *right_payload = _mm512_permutexvar_epi32(idx_rev, *right_payload);

    // Step 1: Compare-swap across registers (distance 16)
    // cmp = 1 where left > right
    __mmask16 cmp = _mm512_cmpgt_epu32_mask(*left_key, *right_key);
    __m512i lo_key = _mm512_mask_blend_epi32(cmp, *left_key, *right_key);
    __m512i hi_key = _mm512_mask_blend_epi32(cmp, *right_key, *left_key);
    __m512i lo_pay = _mm512_mask_blend_epi32(cmp, *left_payload, *right_payload);
    __m512i hi_pay = _mm512_mask_blend_epi32(cmp, *right_payload, *left_payload);
    *left_key = lo_key;
    *right_key = hi_key;
    *left_payload = lo_pay;
    *right_payload = hi_pay;

    // Step 2: Distance 8 - compare elements i with i+8
    __m512i left_key_shuf = _mm512_permutexvar_epi32(idx_swap8, *left_key);
    __m512i right_key_shuf = _mm512_permutexvar_epi32(idx_swap8, *right_key);
    __m512i left_pay_shuf = _mm512_permutexvar_epi32(idx_swap8, *left_payload);
    __m512i right_pay_shuf = _mm512_permutexvar_epi32(idx_swap8, *right_payload);
    
    cmp = _mm512_cmpgt_epu32_mask(*left_key, left_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *left_key, left_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, left_key_shuf, *left_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *left_payload, left_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, left_pay_shuf, *left_payload);
    *left_key = _mm512_mask_blend_epi32(_512_BLEND_8, lo_key, hi_key);
    *left_payload = _mm512_mask_blend_epi32(_512_BLEND_8, lo_pay, hi_pay);
    
    cmp = _mm512_cmpgt_epu32_mask(*right_key, right_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *right_key, right_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, right_key_shuf, *right_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *right_payload, right_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, right_pay_shuf, *right_payload);
    *right_key = _mm512_mask_blend_epi32(_512_BLEND_8, lo_key, hi_key);
    *right_payload = _mm512_mask_blend_epi32(_512_BLEND_8, lo_pay, hi_pay);

    // Step 3: Distance 4 - compare elements i with i+4 within each group of 8
    left_key_shuf = _mm512_permutexvar_epi32(idx_swap4, *left_key);
    right_key_shuf = _mm512_permutexvar_epi32(idx_swap4, *right_key);
    left_pay_shuf = _mm512_permutexvar_epi32(idx_swap4, *left_payload);
    right_pay_shuf = _mm512_permutexvar_epi32(idx_swap4, *right_payload);
    
    cmp = _mm512_cmpgt_epu32_mask(*left_key, left_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *left_key, left_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, left_key_shuf, *left_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *left_payload, left_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, left_pay_shuf, *left_payload);
    *left_key = _mm512_mask_blend_epi32(_512_BLEND_4, lo_key, hi_key);
    *left_payload = _mm512_mask_blend_epi32(_512_BLEND_4, lo_pay, hi_pay);
    
    cmp = _mm512_cmpgt_epu32_mask(*right_key, right_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *right_key, right_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, right_key_shuf, *right_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *right_payload, right_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, right_pay_shuf, *right_payload);
    *right_key = _mm512_mask_blend_epi32(_512_BLEND_4, lo_key, hi_key);
    *right_payload = _mm512_mask_blend_epi32(_512_BLEND_4, lo_pay, hi_pay);

    // Step 4: Distance 2 - within-lane shuffle
    left_key_shuf = _mm512_shuffle_epi32(*left_key, _MM_SHUFFLE(1,0,3,2));
    right_key_shuf = _mm512_shuffle_epi32(*right_key, _MM_SHUFFLE(1,0,3,2));
    left_pay_shuf = _mm512_shuffle_epi32(*left_payload, _MM_SHUFFLE(1,0,3,2));
    right_pay_shuf = _mm512_shuffle_epi32(*right_payload, _MM_SHUFFLE(1,0,3,2));
    
    cmp = _mm512_cmpgt_epu32_mask(*left_key, left_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *left_key, left_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, left_key_shuf, *left_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *left_payload, left_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, left_pay_shuf, *left_payload);
    *left_key = _mm512_mask_blend_epi32(_512_BLEND_2, lo_key, hi_key);
    *left_payload = _mm512_mask_blend_epi32(_512_BLEND_2, lo_pay, hi_pay);
    
    cmp = _mm512_cmpgt_epu32_mask(*right_key, right_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *right_key, right_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, right_key_shuf, *right_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *right_payload, right_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, right_pay_shuf, *right_payload);
    *right_key = _mm512_mask_blend_epi32(_512_BLEND_2, lo_key, hi_key);
    *right_payload = _mm512_mask_blend_epi32(_512_BLEND_2, lo_pay, hi_pay);

    // Step 5: Distance 1 - compare adjacent elements
    left_key_shuf = _mm512_shuffle_epi32(*left_key, _MM_SHUFFLE(2,3,0,1));
    right_key_shuf = _mm512_shuffle_epi32(*right_key, _MM_SHUFFLE(2,3,0,1));
    left_pay_shuf = _mm512_shuffle_epi32(*left_payload, _MM_SHUFFLE(2,3,0,1));
    right_pay_shuf = _mm512_shuffle_epi32(*right_payload, _MM_SHUFFLE(2,3,0,1));
    
    cmp = _mm512_cmpgt_epu32_mask(*left_key, left_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *left_key, left_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, left_key_shuf, *left_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *left_payload, left_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, left_pay_shuf, *left_payload);
    *left_key = _mm512_mask_blend_epi32(_512_BLEND_1, lo_key, hi_key);
    *left_payload = _mm512_mask_blend_epi32(_512_BLEND_1, lo_pay, hi_pay);
    
    cmp = _mm512_cmpgt_epu32_mask(*right_key, right_key_shuf);
    lo_key = _mm512_mask_blend_epi32(cmp, *right_key, right_key_shuf);
    hi_key = _mm512_mask_blend_epi32(cmp, right_key_shuf, *right_key);
    lo_pay = _mm512_mask_blend_epi32(cmp, *right_payload, right_pay_shuf);
    hi_pay = _mm512_mask_blend_epi32(cmp, right_pay_shuf, *right_payload);
    *right_key = _mm512_mask_blend_epi32(_512_BLEND_1, lo_key, hi_key);
    *right_payload = _mm512_mask_blend_epi32(_512_BLEND_1, lo_pay, hi_pay);
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

// Key-Value version of merge_local: sorts payload alongside keys
void merge_local_kv(
    uint32_t* left_key, uint32_t* left_payload,
    uint32_t* right_key, uint32_t* right_payload,
    uint32_t* arr_key, uint32_t* arr_payload,
    int size_left, int size_right
) {
    int i = 0;
    int j = 0;
    int k = 0;
    // Merge the two arrays into arr, comparing only keys
    while (i < size_left && j < size_right) {
        if (left_key[i] <= right_key[j]) {
            arr_key[k] = left_key[i];
            arr_payload[k] = left_payload[i];
            i++;
            k++;
        } else {
            arr_key[k] = right_key[j];
            arr_payload[k] = right_payload[j];
            j++;
            k++;
        }
    }
    // Copy remaining elements with memcpy
    if (i < size_left) {
        memcpy(arr_key + k, left_key + i, (size_left - i) * sizeof(uint32_t));
        memcpy(arr_payload + k, left_payload + i, (size_left - i) * sizeof(uint32_t));
    } else if (j < size_right) {
        memcpy(arr_key + k, right_key + j, (size_right - j) * sizeof(uint32_t));
        memcpy(arr_payload + k, right_payload + j, (size_right - j) * sizeof(uint32_t));
    }
}

// CACHED version: uses regular stores to keep data in L3 cache
// Use this for merges within L3-sized chunks where data will be reused soon
void merge_arrays_cached(
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

    // OPTIMIZATION: Load shuffle indices ONCE before the loop (bitonic merge)
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4);

    // Load from left and right arrays (aligned for cache efficiency)
    __m512i left_reg = _mm512_load_epi32((__m512i*) left);
    __m512i right_reg = _mm512_load_epi32((__m512i*) right);
    merge_512_inline(&left_reg, &right_reg, idx_rev, idx_swap8, idx_swap4);
    // REGULAR STORE: keeps data in cache for next merge pass
    _mm512_store_epi32(arr, left_reg);
    
    size_t right_idx = 16;
    size_t left_idx = 16;
    
    // Main SIMD loop: continue while BOTH sides can provide full 16-element chunks
    while (left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        // Prefetch ahead to hide memory latency
        _mm_prefetch((const char*)(left + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right + right_idx + 64), _MM_HINT_T0);
        
        // Speculatively load both next chunks (aligned for cache efficiency)
        __m512i next_left = _mm512_load_epi32(left + left_idx);
        __m512i next_right = _mm512_load_epi32(right + right_idx);
        
        // Compare first elements to decide which chunk to use
        unsigned int take_left = left[left_idx] <= right[right_idx];
        
        // Branchless select
        __mmask16 mask = take_left ? 0xFFFF : 0x0000;
        left_reg = _mm512_mask_blend_epi32(mask, next_right, next_left);
        
        // Branchless index update
        left_idx += take_left * 16;
        right_idx += (!take_left) * 16;
        
        merge_512_inline(&left_reg, &right_reg, idx_rev, idx_swap8, idx_swap4);
        // REGULAR STORE: keeps data in L3 cache
        _mm512_store_epi32(arr + left_idx + right_idx - 32, left_reg);
    }
    
    // Handle remainders
    size_t left_remaining = size_left - left_idx;
    size_t right_remaining = size_right - right_idx;
    size_t output_pos = left_idx + right_idx - 16;
    
    uint32_t pending[16];
    _mm512_storeu_epi32(pending, right_reg);
    
    size_t remainder_size = left_remaining + right_remaining;
    
    if (remainder_size == 0) {
        _mm512_store_epi32(arr + output_pos, right_reg);
    } else {
        uint32_t remainder_merged[32];
        
        if(left_remaining < right_remaining) {
            merge_local(left + left_idx, pending, remainder_merged, 
              left_remaining, 16);
            merge_local(remainder_merged, right + right_idx, arr + output_pos, 
              left_remaining + 16, right_remaining);
        } else {
            merge_local(right + right_idx, pending, remainder_merged, 
              right_remaining, 16);
            merge_local(remainder_merged, left + left_idx, arr + output_pos, 
              right_remaining + 16, left_remaining);
        }
    }
}

// STREAMING version: uses non-temporal stores to bypass cache
// Use this for large merges where data won't fit in cache anyway
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

    // OPTIMIZATION: Load shuffle indices ONCE before the loop (bitonic merge)
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4);

    // Load from left and right arrays (aligned for cache efficiency)
    __m512i left_reg = _mm512_load_epi32((__m512i*) left);
    __m512i right_reg = _mm512_load_epi32((__m512i*) right);
    merge_512_inline(&left_reg, &right_reg, idx_rev, idx_swap8, idx_swap4);
    // STREAMING STORE: bypasses cache since output won't be read until next merge pass
    _mm512_stream_si512((__m512i*)arr, left_reg);
    
    size_t right_idx = 16;
    size_t left_idx = 16;
    
    // Main SIMD loop: continue while BOTH sides can provide full 16-element chunks
    // Uses branchless selection to avoid branch mispredictions
    while (left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        // Prefetch ahead to hide memory latency
        _mm_prefetch((const char*)(left + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right + right_idx + 64), _MM_HINT_T0);
        
        // Speculatively load both next chunks (aligned for cache efficiency)
        __m512i next_left = _mm512_load_epi32(left + left_idx);
        __m512i next_right = _mm512_load_epi32(right + right_idx);
        
        // Compare first elements to decide which chunk to use
        unsigned int take_left = left[left_idx] <= right[right_idx];
        
        // Branchless select: if take_left, use next_left; else use next_right
        // Using mask blend: mask=0xFFFF means take from second arg (next_left)
        __mmask16 mask = take_left ? 0xFFFF : 0x0000;
        left_reg = _mm512_mask_blend_epi32(mask, next_right, next_left);
        
        // Branchless index update
        left_idx += take_left * 16;
        right_idx += (!take_left) * 16;
        
        // Use inline version with pre-loaded indices (bitonic merge)
        merge_512_inline(&left_reg, &right_reg, idx_rev, idx_swap8, idx_swap4);
        // STREAMING STORE: bypasses cache for better memory bandwidth
        _mm512_stream_si512((__m512i*)(arr + left_idx + right_idx - 32), left_reg);
    }
    
    // Memory fence to ensure streaming stores are visible before we continue
    _mm_sfence();
    
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
        // No remainders - just output the pending 16 (aligned store)
        _mm512_stream_si512((__m512i*)(arr + output_pos), right_reg);
        _mm_sfence();
    } else {
        // Stack buffer - max 30 elements (15 from each side worst case)
        uint32_t remainder_merged[32];
        
        // Merge the two remainders
        if(left_remaining < right_remaining) {
            merge_local(left + left_idx, pending, remainder_merged, 
              left_remaining, 16);
            merge_local(remainder_merged, right + right_idx, arr + output_pos, 
              left_remaining + 16, right_remaining);
        } else {
            merge_local(right + right_idx, pending, remainder_merged, 
              right_remaining, 16);
            merge_local(remainder_merged, left + left_idx, arr + output_pos, 
              right_remaining + 16, left_remaining);
        }
    }
}

// Unaligned-input version for parallel merge (inputs may be at arbitrary offsets)
// Uses alignment preamble: write a few elements with regular stores to reach
// 64-byte alignment, then use streaming stores for the rest.
void merge_arrays_unaligned(
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

    size_t left_idx = 0;
    size_t right_idx = 0;
    size_t out_idx = 0;
    
    // ===== ALIGNMENT PREAMBLE =====
    // Calculate how many elements until next 64-byte boundary
    size_t misalign_bytes = (uintptr_t)arr % 64;
    size_t preamble_count = 0;
    if (misalign_bytes != 0) {
        preamble_count = (64 - misalign_bytes) / sizeof(uint32_t);  // Elements to reach alignment
        
        // Scalar merge for preamble (small, so overhead is minimal)
        while (out_idx < preamble_count && left_idx < size_left && right_idx < size_right) {
            if (left[left_idx] <= right[right_idx]) {
                arr[out_idx++] = left[left_idx++];
            } else {
                arr[out_idx++] = right[right_idx++];
            }
        }
        // Handle case where one side exhausted during preamble
        while (out_idx < preamble_count && left_idx < size_left) {
            arr[out_idx++] = left[left_idx++];
        }
        while (out_idx < preamble_count && right_idx < size_right) {
            arr[out_idx++] = right[right_idx++];
        }
    }
    
    // Now arr + out_idx is 64-byte aligned!
    // Check if we have enough elements left for SIMD
    size_t left_remaining = size_left - left_idx;
    size_t right_remaining = size_right - right_idx;
    
    if (left_remaining < 16 || right_remaining < 16) {
        // Not enough for SIMD, finish with scalar
        merge_local(left + left_idx, right + right_idx, arr + out_idx, 
                    left_remaining, right_remaining);
        return;
    }
    
    // ===== ALIGNED SIMD MERGE WITH STREAMING STORES =====
    // Load shuffle indices ONCE before the loop (bitonic merge)
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4);

    // UNALIGNED loads from left and right (inputs may be at arbitrary offsets)
    __m512i left_reg = _mm512_loadu_epi32(left + left_idx);
    __m512i right_reg = _mm512_loadu_epi32(right + right_idx);
    merge_512_inline(&left_reg, &right_reg, idx_rev, idx_swap8, idx_swap4);
    
    // Output is now aligned - use streaming store!
    _mm512_stream_si512((__m512i*)(arr + out_idx), left_reg);
    
    left_idx += 16;
    right_idx += 16;
    out_idx += 16;
    
    // Main SIMD loop - output stays aligned since we write 64 bytes (16 elements) at a time
    while (left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        _mm_prefetch((const char*)(left + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right + right_idx + 64), _MM_HINT_T0);
        
        // UNALIGNED loads (inputs may not be aligned)
        __m512i next_left = _mm512_loadu_epi32(left + left_idx);
        __m512i next_right = _mm512_loadu_epi32(right + right_idx);
        
        unsigned int take_left = left[left_idx] <= right[right_idx];
        __mmask16 mask = take_left ? 0xFFFF : 0x0000;
        left_reg = _mm512_mask_blend_epi32(mask, next_right, next_left);
        
        left_idx += take_left * 16;
        right_idx += (!take_left) * 16;
        
        merge_512_inline(&left_reg, &right_reg, idx_rev, idx_swap8, idx_swap4);
        
        // STREAMING STORE - output is aligned!
        _mm512_stream_si512((__m512i*)(arr + out_idx), left_reg);
        out_idx += 16;
    }
    
    // Memory fence for streaming stores
    _mm_sfence();
    
    // ===== HANDLE REMAINDERS =====
    left_remaining = size_left - left_idx;
    right_remaining = size_right - right_idx;
    
    // Extract the 16 pending elements from right_reg
    uint32_t pending[16];
    _mm512_storeu_epi32(pending, right_reg);
    
    size_t remainder_size = left_remaining + right_remaining;
    
    if (remainder_size == 0) {
        // No remainders - just output the pending 16
        _mm512_stream_si512((__m512i*)(arr + out_idx), right_reg);
        _mm_sfence();
    } else {
        // Stack buffer for merging remainders
        uint32_t remainder_merged[32];
        
        if (left_remaining < right_remaining) {
            merge_local(left + left_idx, pending, remainder_merged, 
                        left_remaining, 16);
            merge_local(remainder_merged, right + right_idx, arr + out_idx, 
                        left_remaining + 16, right_remaining);
        } else {
            merge_local(right + right_idx, pending, remainder_merged, 
                        right_remaining, 16);
            merge_local(remainder_merged, left + left_idx, arr + out_idx, 
                        right_remaining + 16, left_remaining);
        }
    }
}

// ============== KEY-VALUE VERSIONS ==============
// These sort payload arrays alongside keys for stability testing

// CACHED KV version: uses regular stores to keep data in L3 cache
void merge_arrays_cached_kv(
    uint32_t *left_key,
    uint32_t *left_payload,
    size_t size_left,
    uint32_t *right_key,
    uint32_t *right_payload,
    size_t size_right,
    uint32_t *arr_key,
    uint32_t *arr_payload
) {
    // Small arrays: fall back to scalar merge
    if (size_left < 16 || size_right < 16) {
        merge_local_kv(left_key, left_payload, right_key, right_payload,
                       arr_key, arr_payload, size_left, size_right);
        return;
    }

    // OPTIMIZATION: Load shuffle indices ONCE before the loop (bitonic merge)
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4);

    // Load from left and right arrays (aligned for cache efficiency)
    __m512i left_key_reg = _mm512_load_epi32((__m512i*) left_key);
    __m512i right_key_reg = _mm512_load_epi32((__m512i*) right_key);
    __m512i left_pay_reg = _mm512_load_epi32((__m512i*) left_payload);
    __m512i right_pay_reg = _mm512_load_epi32((__m512i*) right_payload);
    
    merge_512_inline_kv(&left_key_reg, &right_key_reg, &left_pay_reg, &right_pay_reg,
                        idx_rev, idx_swap8, idx_swap4);
    
    // REGULAR STORE: keeps data in cache for next merge pass
    _mm512_store_epi32(arr_key, left_key_reg);
    _mm512_store_epi32(arr_payload, left_pay_reg);
    
    size_t right_idx = 16;
    size_t left_idx = 16;
    
    // Main SIMD loop: continue while BOTH sides can provide full 16-element chunks
    while (left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        // Prefetch ahead to hide memory latency
        _mm_prefetch((const char*)(left_key + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right_key + right_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(left_payload + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right_payload + right_idx + 64), _MM_HINT_T0);
        
        // Speculatively load both next chunks (aligned for cache efficiency)
        __m512i next_left_key = _mm512_load_epi32(left_key + left_idx);
        __m512i next_right_key = _mm512_load_epi32(right_key + right_idx);
        __m512i next_left_pay = _mm512_load_epi32(left_payload + left_idx);
        __m512i next_right_pay = _mm512_load_epi32(right_payload + right_idx);
        
        // Compare first elements to decide which chunk to use
        unsigned int take_left = left_key[left_idx] <= right_key[right_idx];
        
        // Branchless select
        __mmask16 mask = take_left ? 0xFFFF : 0x0000;
        left_key_reg = _mm512_mask_blend_epi32(mask, next_right_key, next_left_key);
        left_pay_reg = _mm512_mask_blend_epi32(mask, next_right_pay, next_left_pay);
        
        // Branchless index update
        left_idx += take_left * 16;
        right_idx += (!take_left) * 16;
        
        merge_512_inline_kv(&left_key_reg, &right_key_reg, &left_pay_reg, &right_pay_reg,
                            idx_rev, idx_swap8, idx_swap4);
        
        // REGULAR STORE: keeps data in L3 cache
        _mm512_store_epi32(arr_key + left_idx + right_idx - 32, left_key_reg);
        _mm512_store_epi32(arr_payload + left_idx + right_idx - 32, left_pay_reg);
    }
    
    // Handle remainders
    size_t left_remaining = size_left - left_idx;
    size_t right_remaining = size_right - right_idx;
    size_t output_pos = left_idx + right_idx - 16;
    
    uint32_t pending_key[16];
    uint32_t pending_pay[16];
    _mm512_storeu_epi32(pending_key, right_key_reg);
    _mm512_storeu_epi32(pending_pay, right_pay_reg);
    
    size_t remainder_size = left_remaining + right_remaining;
    
    if (remainder_size == 0) {
        _mm512_store_epi32(arr_key + output_pos, right_key_reg);
        _mm512_store_epi32(arr_payload + output_pos, right_pay_reg);
    } else {
        uint32_t remainder_key[32];
        uint32_t remainder_pay[32];
        
        if(left_remaining < right_remaining) {
            merge_local_kv(left_key + left_idx, left_payload + left_idx,
                          pending_key, pending_pay,
                          remainder_key, remainder_pay,
                          left_remaining, 16);
            merge_local_kv(remainder_key, remainder_pay,
                          right_key + right_idx, right_payload + right_idx,
                          arr_key + output_pos, arr_payload + output_pos,
                          left_remaining + 16, right_remaining);
        } else {
            merge_local_kv(right_key + right_idx, right_payload + right_idx,
                          pending_key, pending_pay,
                          remainder_key, remainder_pay,
                          right_remaining, 16);
            merge_local_kv(remainder_key, remainder_pay,
                          left_key + left_idx, left_payload + left_idx,
                          arr_key + output_pos, arr_payload + output_pos,
                          right_remaining + 16, left_remaining);
        }
    }
}

// STREAMING KV version: uses non-temporal stores to bypass cache
void merge_arrays_kv(
    uint32_t *left_key,
    uint32_t *left_payload,
    size_t size_left,
    uint32_t *right_key,
    uint32_t *right_payload,
    size_t size_right,
    uint32_t *arr_key,
    uint32_t *arr_payload
) {
    // Small arrays: fall back to scalar merge
    if (size_left < 16 || size_right < 16) {
        merge_local_kv(left_key, left_payload, right_key, right_payload,
                       arr_key, arr_payload, size_left, size_right);
        return;
    }

    // OPTIMIZATION: Load shuffle indices ONCE before the loop (bitonic merge)
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4);

    // Load from left and right arrays (aligned for cache efficiency)
    __m512i left_key_reg = _mm512_load_epi32((__m512i*) left_key);
    __m512i right_key_reg = _mm512_load_epi32((__m512i*) right_key);
    __m512i left_pay_reg = _mm512_load_epi32((__m512i*) left_payload);
    __m512i right_pay_reg = _mm512_load_epi32((__m512i*) right_payload);
    
    merge_512_inline_kv(&left_key_reg, &right_key_reg, &left_pay_reg, &right_pay_reg,
                        idx_rev, idx_swap8, idx_swap4);
    
    // STREAMING STORE: bypasses cache since output won't be read until next merge pass
    _mm512_stream_si512((__m512i*)arr_key, left_key_reg);
    _mm512_stream_si512((__m512i*)arr_payload, left_pay_reg);
    
    size_t right_idx = 16;
    size_t left_idx = 16;
    
    // Main SIMD loop: continue while BOTH sides can provide full 16-element chunks
    while (left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        // Prefetch ahead to hide memory latency
        _mm_prefetch((const char*)(left_key + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right_key + right_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(left_payload + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right_payload + right_idx + 64), _MM_HINT_T0);
        
        // Speculatively load both next chunks (aligned for cache efficiency)
        __m512i next_left_key = _mm512_load_epi32(left_key + left_idx);
        __m512i next_right_key = _mm512_load_epi32(right_key + right_idx);
        __m512i next_left_pay = _mm512_load_epi32(left_payload + left_idx);
        __m512i next_right_pay = _mm512_load_epi32(right_payload + right_idx);
        
        // Compare first elements to decide which chunk to use
        unsigned int take_left = left_key[left_idx] <= right_key[right_idx];
        
        // Branchless select
        __mmask16 mask = take_left ? 0xFFFF : 0x0000;
        left_key_reg = _mm512_mask_blend_epi32(mask, next_right_key, next_left_key);
        left_pay_reg = _mm512_mask_blend_epi32(mask, next_right_pay, next_left_pay);
        
        // Branchless index update
        left_idx += take_left * 16;
        right_idx += (!take_left) * 16;
        
        merge_512_inline_kv(&left_key_reg, &right_key_reg, &left_pay_reg, &right_pay_reg,
                            idx_rev, idx_swap8, idx_swap4);
        
        // STREAMING STORE: bypasses cache for better memory bandwidth
        _mm512_stream_si512((__m512i*)(arr_key + left_idx + right_idx - 32), left_key_reg);
        _mm512_stream_si512((__m512i*)(arr_payload + left_idx + right_idx - 32), left_pay_reg);
    }
    
    // Memory fence to ensure streaming stores are visible before we continue
    _mm_sfence();
    
    // Handle remainders
    size_t left_remaining = size_left - left_idx;
    size_t right_remaining = size_right - right_idx;
    size_t output_pos = left_idx + right_idx - 16;
    
    uint32_t pending_key[16];
    uint32_t pending_pay[16];
    _mm512_storeu_epi32(pending_key, right_key_reg);
    _mm512_storeu_epi32(pending_pay, right_pay_reg);
    
    size_t remainder_size = left_remaining + right_remaining;
    
    if (remainder_size == 0) {
        _mm512_stream_si512((__m512i*)(arr_key + output_pos), right_key_reg);
        _mm512_stream_si512((__m512i*)(arr_payload + output_pos), right_pay_reg);
        _mm_sfence();
    } else {
        uint32_t remainder_key[32];
        uint32_t remainder_pay[32];
        
        if(left_remaining < right_remaining) {
            merge_local_kv(left_key + left_idx, left_payload + left_idx,
                          pending_key, pending_pay,
                          remainder_key, remainder_pay,
                          left_remaining, 16);
            merge_local_kv(remainder_key, remainder_pay,
                          right_key + right_idx, right_payload + right_idx,
                          arr_key + output_pos, arr_payload + output_pos,
                          left_remaining + 16, right_remaining);
        } else {
            merge_local_kv(right_key + right_idx, right_payload + right_idx,
                          pending_key, pending_pay,
                          remainder_key, remainder_pay,
                          right_remaining, 16);
            merge_local_kv(remainder_key, remainder_pay,
                          left_key + left_idx, left_payload + left_idx,
                          arr_key + output_pos, arr_payload + output_pos,
                          right_remaining + 16, left_remaining);
        }
    }
}

// Unaligned-input KV version for parallel merge (inputs may be at arbitrary offsets)
// Uses alignment preamble: write a few elements with regular stores to reach
// 64-byte alignment, then use streaming stores for the rest.
void merge_arrays_unaligned_kv(
    uint32_t *left_key,
    uint32_t *left_payload,
    size_t size_left,
    uint32_t *right_key,
    uint32_t *right_payload,
    size_t size_right,
    uint32_t *arr_key,
    uint32_t *arr_payload
) {
    // Small arrays: fall back to scalar merge
    if (size_left < 16 || size_right < 16) {
        merge_local_kv(left_key, left_payload, right_key, right_payload,
                       arr_key, arr_payload, size_left, size_right);
        return;
    }

    size_t left_idx = 0;
    size_t right_idx = 0;
    size_t out_idx = 0;
    
    // ===== ALIGNMENT PREAMBLE =====
    // Calculate how many elements until next 64-byte boundary
    size_t misalign_bytes = (uintptr_t)arr_key % 64;
    size_t preamble_count = 0;
    if (misalign_bytes != 0) {
        preamble_count = (64 - misalign_bytes) / sizeof(uint32_t);
        
        // Scalar merge for preamble
        while (out_idx < preamble_count && left_idx < size_left && right_idx < size_right) {
            if (left_key[left_idx] <= right_key[right_idx]) {
                arr_key[out_idx] = left_key[left_idx];
                arr_payload[out_idx] = left_payload[left_idx];
                left_idx++;
            } else {
                arr_key[out_idx] = right_key[right_idx];
                arr_payload[out_idx] = right_payload[right_idx];
                right_idx++;
            }
            out_idx++;
        }
        // Handle case where one side exhausted during preamble
        while (out_idx < preamble_count && left_idx < size_left) {
            arr_key[out_idx] = left_key[left_idx];
            arr_payload[out_idx] = left_payload[left_idx];
            left_idx++;
            out_idx++;
        }
        while (out_idx < preamble_count && right_idx < size_right) {
            arr_key[out_idx] = right_key[right_idx];
            arr_payload[out_idx] = right_payload[right_idx];
            right_idx++;
            out_idx++;
        }
    }
    
    // Now arr_key + out_idx is 64-byte aligned!
    // Check if we have enough elements left for SIMD
    size_t left_remaining = size_left - left_idx;
    size_t right_remaining = size_right - right_idx;
    
    if (left_remaining < 16 || right_remaining < 16) {
        // Not enough for SIMD, finish with scalar
        merge_local_kv(left_key + left_idx, left_payload + left_idx,
                       right_key + right_idx, right_payload + right_idx,
                       arr_key + out_idx, arr_payload + out_idx,
                       left_remaining, right_remaining);
        return;
    }
    
    // ===== ALIGNED SIMD MERGE WITH STREAMING STORES =====
    // Load shuffle indices ONCE before the loop (bitonic merge)
    const __m512i idx_rev = _mm512_load_epi32(IDX_REV);
    const __m512i idx_swap8 = _mm512_load_epi32(IDX_SWAP8);
    const __m512i idx_swap4 = _mm512_load_epi32(IDX_SWAP4);

    // UNALIGNED loads from left and right (inputs may be at arbitrary offsets)
    __m512i left_key_reg = _mm512_loadu_epi32(left_key + left_idx);
    __m512i right_key_reg = _mm512_loadu_epi32(right_key + right_idx);
    __m512i left_pay_reg = _mm512_loadu_epi32(left_payload + left_idx);
    __m512i right_pay_reg = _mm512_loadu_epi32(right_payload + right_idx);
    
    merge_512_inline_kv(&left_key_reg, &right_key_reg, &left_pay_reg, &right_pay_reg,
                        idx_rev, idx_swap8, idx_swap4);
    
    // Output is now aligned - use streaming store!
    _mm512_stream_si512((__m512i*)(arr_key + out_idx), left_key_reg);
    _mm512_stream_si512((__m512i*)(arr_payload + out_idx), left_pay_reg);
    
    left_idx += 16;
    right_idx += 16;
    out_idx += 16;
    
    // Main SIMD loop - output stays aligned since we write 64 bytes at a time
    while (left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        _mm_prefetch((const char*)(left_key + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right_key + right_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(left_payload + left_idx + 64), _MM_HINT_T0);
        _mm_prefetch((const char*)(right_payload + right_idx + 64), _MM_HINT_T0);
        
        // UNALIGNED loads (inputs may not be aligned)
        __m512i next_left_key = _mm512_loadu_epi32(left_key + left_idx);
        __m512i next_right_key = _mm512_loadu_epi32(right_key + right_idx);
        __m512i next_left_pay = _mm512_loadu_epi32(left_payload + left_idx);
        __m512i next_right_pay = _mm512_loadu_epi32(right_payload + right_idx);
        
        unsigned int take_left = left_key[left_idx] <= right_key[right_idx];
        __mmask16 mask = take_left ? 0xFFFF : 0x0000;
        left_key_reg = _mm512_mask_blend_epi32(mask, next_right_key, next_left_key);
        left_pay_reg = _mm512_mask_blend_epi32(mask, next_right_pay, next_left_pay);
        
        left_idx += take_left * 16;
        right_idx += (!take_left) * 16;
        
        merge_512_inline_kv(&left_key_reg, &right_key_reg, &left_pay_reg, &right_pay_reg,
                            idx_rev, idx_swap8, idx_swap4);
        
        // STREAMING STORE - output is aligned!
        _mm512_stream_si512((__m512i*)(arr_key + out_idx), left_key_reg);
        _mm512_stream_si512((__m512i*)(arr_payload + out_idx), left_pay_reg);
        out_idx += 16;
    }
    
    // Memory fence for streaming stores
    _mm_sfence();
    
    // ===== HANDLE REMAINDERS =====
    left_remaining = size_left - left_idx;
    right_remaining = size_right - right_idx;
    
    uint32_t pending_key[16];
    uint32_t pending_pay[16];
    _mm512_storeu_epi32(pending_key, right_key_reg);
    _mm512_storeu_epi32(pending_pay, right_pay_reg);
    
    size_t remainder_size = left_remaining + right_remaining;
    
    if (remainder_size == 0) {
        _mm512_stream_si512((__m512i*)(arr_key + out_idx), right_key_reg);
        _mm512_stream_si512((__m512i*)(arr_payload + out_idx), right_pay_reg);
        _mm_sfence();
    } else {
        uint32_t remainder_key[32];
        uint32_t remainder_pay[32];
        
        if (left_remaining < right_remaining) {
            merge_local_kv(left_key + left_idx, left_payload + left_idx,
                           pending_key, pending_pay,
                           remainder_key, remainder_pay,
                           left_remaining, 16);
            merge_local_kv(remainder_key, remainder_pay,
                           right_key + right_idx, right_payload + right_idx,
                           arr_key + out_idx, arr_payload + out_idx,
                           left_remaining + 16, right_remaining);
        } else {
            merge_local_kv(right_key + right_idx, right_payload + right_idx,
                           pending_key, pending_pay,
                           remainder_key, remainder_pay,
                           right_remaining, 16);
            merge_local_kv(remainder_key, remainder_pay,
                           left_key + left_idx, left_payload + left_idx,
                           arr_key + out_idx, arr_payload + out_idx,
                           right_remaining + 16, left_remaining);
        }
    }
}