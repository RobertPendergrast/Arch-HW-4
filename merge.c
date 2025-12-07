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


// 512 Constants
const __mmask16 _512_BLEND_1 = 0b1111111100000000;
const __mmask16 _512_BLEND_2 = 0b1111000011110000;

/*
 * Takes in two m512i registers and merges them in place.
 */
void merge_512_registers(
    __m512i *left,
    __m512i *right
) {
    // Reverse the right register using a permutex
    __m512i _512_REV = _mm512_set_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    *right = _mm512_permutexvar_epi32(_512_REV, *right);
    // Level 1
    // Get min/max values
    __m512i L1 = _mm512_min_epi32(*left, *right);
    __m512i H1 = _mm512_max_epi32(*left, *right);
    //Shuffle
    __m512i L1p = _mm512_mask_blend_epi32(_512_BLEND_1, L1, H1);
    __m512i H1p = _mm512_mask_blend_epi32(_512_BLEND_1, H1, L1);
    __m512i _512_SHUFFLE_1 = _mm512_set_epi32(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
    );
    H1p = _mm512_permutexvar_epi32(_512_SHUFFLE_1, H1p);
    // Level 2
    // Get min/max values
    __m512i L2 = _mm512_min_epi32(L1p, H1p);
    __m512i H2 = _mm512_max_epi32(L1p, H1p);
    //Shuffle
    __m512i L2p = _mm512_mask_blend_epi32(_512_BLEND_2, L2, H2);
    __m512i H2p = _mm512_mask_blend_epi32(_512_BLEND_2, H2, L2);
    __m512i _512_SHUFFLE_2 = _mm512_set_epi32(
        11, 10, 9, 8, 15, 14, 13, 12, 3 , 2 , 1, 0, 7, 6, 5, 4
    );
    H2p = _mm512_permutexvar_epi32(_512_SHUFFLE_2, H2p);
    // Levels 3-4
    // Break each register up into 4 __mm128i registers and pass them into the
    // merge_128_registers function. Perhaps this should be put in an array...
    __m128i H3_0 = _mm512_extracti32x4_epi32(H2p, 0);
    __m128i H3_1 = _mm512_extracti32x4_epi32(H2p, 1);
    __m128i H3_2 = _mm512_extracti32x4_epi32(H2p, 2);
    __m128i H3_3 = _mm512_extracti32x4_epi32(H2p, 3);
    __m128i L3_0 = _mm512_extracti32x4_epi32(L2p, 0);
    __m128i L3_1 = _mm512_extracti32x4_epi32(L2p, 1);
    __m128i L3_2 = _mm512_extracti32x4_epi32(L2p, 2);
    __m128i L3_3 = _mm512_extracti32x4_epi32(L2p, 3);

    merge_128_registers(&L3_0, &H3_0);
    merge_128_registers(&L3_1, &H3_1);
    merge_128_registers(&L3_2, &H3_2);
    merge_128_registers(&L3_3, &H3_3);

    __m256i L3p_0 = _mm256_set_m128i(H3_0, L3_0); // Combines v1 and v2 into a m256i register
    __m256i L3p_1 = _mm256_set_m128i(H3_1, L3_1); // Combines v3 and v4 into a m256i register
    *left = _mm512_inserti64x4(*left, L3p_0, 0);
    *left = _mm512_inserti64x4(*left, L3p_1, 1);

    __m256i H3p_0 = _mm256_set_m128i(H3_2, L3_2); // Combines v1 and v2 into a m256i register
    __m256i H3p_1 = _mm256_set_m128i(H3_3, L3_3); // Combines v3 and v4 into a m256i register
    *right = _mm512_inserti64x4(*right, H3p_0, 0);
    *right = _mm512_inserti64x4(*right, H3p_1, 1);
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
            right_reg = _mm512_loadu_epi32(right + right_idx);
            right_idx += 16;
        }
        merge_512_registers(&left_reg, &right_reg);
        _mm512_storeu_epi32(arr + left_idx + right_idx - 32, left_reg);
    }
    if(left_idx == size_left) {
        while(right_idx + 16 <= size_right) {
            right_reg = _mm512_loadu_epi32(right + right_idx);
            right_idx += 16;
            merge_512_registers(&left_reg, &right_reg);
            _mm512_storeu_epi32(arr + left_idx + right_idx - 32, left_reg);
            if(right[right_idx] >= right_reg[15]) {
                //memcpy the rest and break
                _mm512_storeu_epi32(arr + left_idx + right_idx - 16, right_reg);
                memcpy(arr + left_idx + right_idx, right + right_idx, (size_right - right_idx) * sizeof(uint32_t));
                break;
            }
                

        }
    }
    else if(right_idx == size_right) {
        while(left_idx + 16 <= size_left) {
            left_reg = _mm512_loadu_epi32(left + left_idx);
            left_idx += 16;
            merge_512_registers(&left_reg, &right_reg);
            _mm512_storeu_epi32(arr + left_idx + right_idx - 32, left_reg);
            if(left[left_idx] >= left_reg[15]) {
                //memcpy the rest and break
                _mm512_storeu_epi32(arr + left_idx + right_idx - 16, left_reg);
                memcpy(arr + left_idx + right_idx, left + left_idx, (size_left - left_idx) * sizeof(uint32_t));
                break;
            }
        }
    }
    else{
        fprintf(stderr, "Error: Arrays are not cache line aligned for merge. Exiting.\n");
        exit(1);
    }
     
}