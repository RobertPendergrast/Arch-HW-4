#include "merge.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
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
 */
void merge_128_registers(
    __m128i *left,
    __m128i *right
) {
    // Reverse the *right register

    // Level 1
    // Get min/max values
    __m128i L1 = _mm_min_epi32(*left, *right);
    __m128i H1 = _mm_max_epi32(*left, *right);

    // Shuffle
    __m128i L1p = _mm_blend_epi32(L1, H1, _128_BLEND_1);
    __m128i H1p = _mm_blend_epi32(H1, L1, _128_BLEND_1);
    H1p = _mm_shuffle_epi32(H1p, _128_SHUFFLE_1);

    // Level 2
    // Get min/max values
    __m128i L2 = _mm_min_epi32(L1p, H1p);
    __m128i H2 = _mm_max_epi32(L1p, H1p);

    // Shuffle
    __m128i L2p = _mm_blend_epi32(L2, H2, _128_BLEND_2);
    __m128i H2p = _mm_blend_epi32(H2, L2, _128_BLEND_2);
    H2p = _mm_shuffle_epi32(H2p, _128_SHUFFLE_2);

    // Level 3
    // Get min/max values
    __m128i L3 = _mm_min_epi32(L2p, H2p);
    __m128i H3 = _mm_max_epi32(L2p, H2p);

    //Shuffle
    __m128i H3p =  _mm_shuffle_epi32(H3, _128_SHUFFLE_3);
    __m128i L3p =  _mm_blend_epi32(L3, H3p, _128_BLEND_3);
    H3p = _mm_blend_epi32(H3p, L3, _128_BLEND_3);
    H3p = _mm_shuffle_epi32(H3p, _128_SHUFFLE_3_H);
    L3p = _mm_shuffle_epi32(L3p, _128_SHUFFLE_3_L);

    // Reset
    // Note: This will eventually be removed by consolidating registers
    *left = L3p;
    *right = H3p;
}


// 512 Constants - moved to file scope for efficiency
const __mmask16 _512_BLEND_1 = 0b1111111100000000;
const __mmask16 _512_BLEND_2 = 0b1111000011110000;
const __mmask16 _512_BLEND_3 = 0b1100110011001100;  // For level 3: groups of 2
const __mmask16 _512_BLEND_4 = 0b1010101010101010;  // For level 4: alternating

// Shuffle indices - precomputed at file scope
// Note: _mm512_set_epi32 args are in reverse order (element 15 first)
static __m512i get_512_rev(void) {
    return _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
static __m512i get_512_shuffle_1(void) {
    return _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
}
static __m512i get_512_shuffle_2(void) {
    return _mm512_set_epi32(11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4);
}

/*
 * Takes in two m512i registers and merges them in place.
 * Fully inlined bitonic merge network - all 4 levels in 512-bit registers.
 */
static inline __attribute__((always_inline)) void merge_512_registers(
    __m512i *left,
    __m512i *right
) {
    // Get shuffle indices (compiler will optimize these)
    __m512i _512_REV = get_512_rev();
    __m512i _512_SHUFFLE_1 = get_512_shuffle_1();
    __m512i _512_SHUFFLE_2 = get_512_shuffle_2();

    // Level 0: Reverse the right register to form bitonic sequence
    *right = _mm512_permutexvar_epi32(_512_REV, *right);

    // Level 1: Compare 16 vs 16, split into two groups
    __m512i L1 = _mm512_min_epi32(*left, *right);
    __m512i H1 = _mm512_max_epi32(*left, *right);
    __m512i L1p = _mm512_mask_blend_epi32(_512_BLEND_1, L1, H1);
    __m512i H1p = _mm512_mask_blend_epi32(_512_BLEND_1, H1, L1);
    H1p = _mm512_permutexvar_epi32(_512_SHUFFLE_1, H1p);

    // Level 2: Compare 8 vs 8 within each group
    __m512i L2 = _mm512_min_epi32(L1p, H1p);
    __m512i H2 = _mm512_max_epi32(L1p, H1p);
    __m512i L2p = _mm512_mask_blend_epi32(_512_BLEND_2, L2, H2);
    __m512i H2p = _mm512_mask_blend_epi32(_512_BLEND_2, H2, L2);
    H2p = _mm512_permutexvar_epi32(_512_SHUFFLE_2, H2p);

    // Level 3: Compare 4 vs 4 (using within-lane shuffle - faster!)
    __m512i L3 = _mm512_min_epi32(L2p, H2p);
    __m512i H3 = _mm512_max_epi32(L2p, H2p);
    __m512i L3p = _mm512_mask_blend_epi32(_512_BLEND_3, L3, H3);
    __m512i H3p = _mm512_mask_blend_epi32(_512_BLEND_3, H3, L3);
    // _mm512_shuffle_epi32 applies same shuffle to each 128-bit lane
    H3p = _mm512_shuffle_epi32(H3p, _MM_SHUFFLE(1, 0, 3, 2));  // 0b01001110

    // Level 4: Compare 2 vs 2
    __m512i L4 = _mm512_min_epi32(L3p, H3p);
    __m512i H4 = _mm512_max_epi32(L3p, H3p);
    __m512i L4p = _mm512_mask_blend_epi32(_512_BLEND_4, L4, H4);
    __m512i H4p = _mm512_mask_blend_epi32(_512_BLEND_4, H4, L4);
    H4p = _mm512_shuffle_epi32(H4p, _MM_SHUFFLE(2, 3, 0, 1));  // 0b10110001

    // Level 5: Final compare and interleave to produce sorted output
    __m512i L5 = _mm512_min_epi32(L4p, H4p);
    __m512i H5 = _mm512_max_epi32(L4p, H4p);
    
    // Final shuffle and blend (same pattern as 128-bit version, applied per lane)
    __m512i H5s = _mm512_shuffle_epi32(H5, _MM_SHUFFLE(1, 0, 3, 2));  // 0b01001110
    __m512i L5p = _mm512_mask_blend_epi32(_512_BLEND_3, L5, H5s);
    __m512i H5p = _mm512_mask_blend_epi32(_512_BLEND_3, H5s, L5);
    L5p = _mm512_shuffle_epi32(L5p, _MM_SHUFFLE(3, 1, 2, 0));  // 0b11011000
    H5p = _mm512_shuffle_epi32(H5p, _MM_SHUFFLE(1, 3, 0, 2));  // 0b01110010

    // Combine lanes: L5p has lower 4 of each 8, H5p has upper 4 of each 8
    // Need to interleave: left gets lanes 0,1 from both, right gets lanes 2,3
    __m512i left_lo = _mm512_shuffle_i64x2(L5p, H5p, 0b01000100);  // lanes 0,1 from L5p, 0,1 from H5p
    __m512i left_hi = _mm512_shuffle_i64x2(L5p, H5p, 0b11101110);  // lanes 2,3 from L5p, 2,3 from H5p
    
    // Permute to get final order
    *left = _mm512_permutex2var_epi64(left_lo, 
        _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), left_hi);
    *right = _mm512_permutex2var_epi64(left_lo,
        _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), left_hi);
}

void merge(uint32_t* left, uint32_t* right, uint32_t* arr, int size_left, int size_right) {
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
    // Merge the rest in
    while (i < size_left) {
        arr[k++] = left[i++];
    }
    while (j< size_right) {
        arr[k++] = right[j++];
    }
}

void merge_arrays(
    uint32_t *left,
    size_t size_left,
    uint32_t *right,
    size_t size_right,
    uint32_t *arr
) {
    // For performance, try to make left and right cache line aligned, as well as arr
    // Also, try to make the size_left and size_right cache line aligned, but we can handle if not
    
    //TODO: take this line out once done debugging
    if(size_left < 16 || size_right < 16) {
        fprintf(stderr, "Error: Arrays are too small to merge (size_left=%zu, size_right=%zu)\n", size_left, size_right);
        exit(EXIT_FAILURE);
    }

    //creating a mm512i register from the left and right arrays
    __m512i left_reg = _mm512_loadu_epi32((__m512i*) left);
    __m512i right_reg = _mm512_loadu_epi32((__m512i*) right);

    //merging the two registers
    merge_512_registers(&left_reg, &right_reg);

    //storing the result in the arr array
    _mm512_storeu_epi32(arr, left_reg);
    int right_idx = 16;
    int left_idx = 16;
    while(left_idx + 16 <= size_left && right_idx + 16 <= size_right) {
        if(left[left_idx] <= right[right_idx]) {
            left_reg = _mm512_loadu_epi32(left + left_idx);
            left_idx += 16;
        }
        else {
            left_reg = _mm512_loadu_epi32(right + right_idx);
            right_idx += 16;
        }
        merge_512_registers(&left_reg, &right_reg);
        _mm512_storeu_epi32(arr + left_idx + right_idx - 32, left_reg);
    }
    if(left_idx == size_left) {
        printf("Left is done, merging right\n");
        int done = 0;
        while(right_idx + 16 <= size_right) {
            left_reg = _mm512_loadu_epi32(right + right_idx);  // Load into left_reg to preserve pending right_reg
            right_idx += 16;
            merge_512_registers(&left_reg, &right_reg);
            _mm512_storeu_epi32(arr + left_idx + right_idx - 32, left_reg);
            if(right_idx < size_right && right[right_idx] >= right_reg[15]) {
                //memcpy the rest and break
                _mm512_storeu_epi32(arr + left_idx + right_idx - 16, right_reg);
                memcpy(arr + left_idx + right_idx, right + right_idx, (size_right - right_idx) * sizeof(uint32_t));
                done = 1;
                break;
            }
            
        }
        if(done == 0){
            _mm512_storeu_epi32(arr + left_idx + right_idx - 16, right_reg);
        }
    }
    else if(right_idx == size_right) {
        printf("Right is done, merging left\n");
        int done = 0;
        while(left_idx + 16 <= size_left) {
            left_reg = _mm512_loadu_epi32(left + left_idx);
            left_idx += 16;
            merge_512_registers(&left_reg, &right_reg);
            _mm512_storeu_epi32(arr + left_idx + right_idx - 32, left_reg);
            if(left_idx < size_left && left[left_idx] >= right_reg[15]) {
                //memcpy the rest and break
                _mm512_storeu_epi32(arr + left_idx + right_idx - 16, right_reg);
                memcpy(arr + left_idx + right_idx, left + left_idx, (size_left - left_idx) * sizeof(uint32_t));
                done = 1;
                break;
            }
        }   
        if(done == 0){
            _mm512_storeu_epi32(arr + left_idx + right_idx - 16, right_reg);
        }
    }
    else{
        fprintf(stderr, "Error: Arrays are not cache line aligned for merge. Exiting.\n");
        exit(1);
    }
     
}