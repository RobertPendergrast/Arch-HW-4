#include "merge.h"
#include "utils.h"
#include <stdio.h>

// Control to reverse. First two bits (11) means the new first value is the old
// last value, etc...
const int REV_CNTRL = 0b00011011;

const int BLEND_1 = 0b1100;
const int BLEND_2 = 0b1010;
const int BLEND_3 = BLEND_1;
const int SHUFFLE_1 = 0b01001110;
const int SHUFFLE_2 = 0b10110001;
const int SHUFFLE_3 = SHUFFLE_1;;
const int SHUFFLE_3_L = 0b11011000;
const int SHUFFLE_3_H = 0b01110010;

/*
 * Takes in two m128i registers and merges them in place.
 */
void merge_128_registers(
    __m128i left,
    __m128i right
) {
    // Reverse the right register
    right = _mm_shuffle_epi32(right, REV_CNTRL);

    // Level 1
    // Get min/max values
    __m128i L1 = _mm_min_epi32(left, right);
    __m128i H1 = _mm_max_epi32(left, right);

    // Shuffle 1
    __m128i L1p = _mm_blend_epi32(L1, H1, BLEND_1);
    __m128i H1p = _mm_blend_epi32(H1, L1, BLEND_1);
    H1p = _mm_shuffle_epi32(H1p, SHUFFLE_1);

    // Level 2
    // Get min/max values
    __m128i L2 = _mm_min_epi32(L1p, H1p);
    __m128i H2 = _mm_max_epi32(L1p, H1p);

    // Shuffle 2
    __m128i L2p = _mm_blend_epi32(L2, H2, BLEND_2);
    __m128i H2p = _mm_blend_epi32(H2, L2, BLEND_2);
    H2p = _mm_shuffle_epi32(H2p, SHUFFLE_2);

    // Level 3
    // Get min/max values
    __m128i L3 = _mm_min_epi32(L2p, H2p);
    __m128i H3 = _mm_max_epi32(L2p, H2p);

    //Shuffle 3
    __m128i H3p =  _mm_shuffle_epi32(H3, SHUFFLE_3);
    __m128i L3p =  _mm_blend_epi32(L3, H3p, BLEND_3);
    H3p = _mm_blend_epi32(H3p, L3, BLEND_3);
    H3p = _mm_shuffle_epi32(H3p, SHUFFLE_3_H);
    L3p = _mm_shuffle_epi32(L3p, SHUFFLE_3_L);

    // Reset
    // Note: This will eventually be removed by consolidating registers
    left = L3p;
    right = H3p;
}

/*
 * Takes in two m512i registers and merges them in place.
 */
void merge_512_registers(
    __m512i left,
    __m512i right
) {
    print_512_num(left);
    print_512_num(right);
}
