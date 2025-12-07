#include "merge.h"
#include "utils.h"
#include <stdio.h>

// 128 Constants
// Control to reverse. First two bits (11) means the new first value is the old
// last value, etc...
const int 128_REV = 0b00011011;
const int 128_BLEND_1 = 0b1100;
const int 128_BLEND_2 = 0b1010;
const int 128_BLEND_3 = 128_BLEND_1;
const int 128_SHUFFLE_1 = 0b01001110;
const int 128_SHUFFLE_2 = 0b10110001;
const int 128_SHUFFLE_3 = 128_SHUFFLE_1;;
const int 128_SHUFFLE_3_L = 0b11011000;
const int 128_SHUFFLE_3_H = 0b01110010;

/*
 * Takes in two m128i registers and merges them in place.
 */
void merge_128_registers(
    __m128i *left,
    __m128i *right
) {
    // Reverse the *right register
    *right = _mm_shuffle_epi32(*right, 128_REV);

    // Level 1
    // Get min/max values
    __m128i L1 = _mm_min_epi32(*left, *right);
    __m128i H1 = _mm_max_epi32(*left, *right);

    // Shuffle 1
    __m128i L1p = _mm_blend_epi32(L1, H1, 128_BLEND_1);
    __m128i H1p = _mm_blend_epi32(H1, L1, 128_BLEND_1);
    H1p = _mm_shuffle_epi32(H1p, 128_SHUFFLE_1);

    // Level 2
    // Get min/max values
    __m128i L2 = _mm_min_epi32(L1p, H1p);
    __m128i H2 = _mm_max_epi32(L1p, H1p);

    // Shuffle 2
    __m128i L2p = _mm_blend_epi32(L2, H2, 128_BLEND_2);
    __m128i H2p = _mm_blend_epi32(H2, L2, 128_BLEND_2);
    H2p = _mm_shuffle_epi32(H2p, 128_SHUFFLE_2);

    // Level 3
    // Get min/max values
    __m128i L3 = _mm_min_epi32(L2p, H2p);
    __m128i H3 = _mm_max_epi32(L2p, H2p);

    //Shuffle 3
    __m128i H3p =  _mm_shuffle_epi32(H3, 128_SHUFFLE_3);
    __m128i L3p =  _mm_blend_epi32(L3, H3p, 128_BLEND_3);
    H3p = _mm_blend_epi32(H3p, L3, 128_BLEND_3);
    H3p = _mm_shuffle_epi32(H3p, 128_SHUFFLE_3_H);
    L3p = _mm_shuffle_epi32(L3p, 128_SHUFFLE_3_L);

    // Reset
    // Note: This will eventually be removed by consolidating registers
    *left = L3p;
    *right = H3p;
}


// 512 Constants
const __m512i 512_REV = _mm512_set_epi32(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
);


/*
 * Takes in two m512i registers and merges them in place.
 */
void merge_512_registers(
    __m512i *left,
    __m512i *right
) {
    // Reverse the right register using a permutex
    *right = _mm512_permutexvar_epi32(512_REV, *right);
    print_512_num(*right);
    // Level 1
    // Level 2
    // Levels 3-4
    // Break each register up into 4 __mm128i registers and pass them into the
    // merge_128_registers function
    __m128i H3_0 = _mm512_extracti32x4_epi32(*right, 0);
    __m128i H3_1 = _mm512_extracti32x4_epi32(*right, 1);
    __m128i H3_2 = _mm512_extracti32x4_epi32(*right, 2);
    __m128i H3_3 = _mm512_extracti32x4_epi32(*right, 3);
    __m128i L3_0 = _mm512_extracti32x4_epi32(*left, 0);
    __m128i L3_1 = _mm512_extracti32x4_epi32(*left, 1);
    __m128i L3_2 = _mm512_extracti32x4_epi32(*left, 2);
    __m128i L3_3 = _mm512_extracti32x4_epi32(*left, 3);
    print_128_num(L3_0);
    print_128_num(L3_1);
    print_128_num(L3_2);
    print_128_num(L3_3);
}
