#include "merge.h"
#include "utils.h"
#include <stdio.h>

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
    *right = _mm_shuffle_epi32(*right, _128_REV);

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
    print_512_num(*right);
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
    print_512_num(L1p);
    print_512_num(H1p);
    // Level 2
    // Get min/max values
    //Shuffle
    // Levels 3-4
    // Break each register up into 4 __mm128i registers and pass them into the
    // merge_128_registers function. Perhaps this should be put in an array...
    __m128i H3_0 = _mm512_extracti32x4_epi32(*right, 0);
    __m128i H3_1 = _mm512_extracti32x4_epi32(*right, 1);
    __m128i H3_2 = _mm512_extracti32x4_epi32(*right, 2);
    __m128i H3_3 = _mm512_extracti32x4_epi32(*right, 3);
    __m128i L3_0 = _mm512_extracti32x4_epi32(*left, 0);
    __m128i L3_1 = _mm512_extracti32x4_epi32(*left, 1);
    __m128i L3_2 = _mm512_extracti32x4_epi32(*left, 2);
    __m128i L3_3 = _mm512_extracti32x4_epi32(*left, 3);
}
