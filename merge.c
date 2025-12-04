#include "merge.h"
#include "utils.h"
#include <stdio.h>

// Control to reverse. First two bits (11) means the new first value is the old
// last value, etc...
const int REV_CNTRL = 0b00011011;

const int BLEND_1 = 0b1100;
const int BLEND_2 = 0b1010;
const int SHUFFLE_1 = 0b01001110;
const int SHUFFLE_0 = 0b11100001;
const int SHUFFLE_2 = 0b10110001;

/*
 * Takes in two m128i registers and merges them in place.
 */
void merge_128_registers(
    __m128i left,
    __m128i right
) {
    //print_128_num(left);
    //print_128_num(right);
    // Reverse the right register
    right = _mm_shuffle_epi32(right, REV_CNTRL);
    // Level 1
    // Get min values
    __m128i L1 = _mm_min_epi32(left, right);
    // Get max values
    __m128i H1 = _mm_max_epi32(left, right);

    // Shuffle 1
    __m128i L1p =  _mm_blend_epi32(L1, H1, BLEND_1);
    __m128i H1p =  _mm_blend_epi32(H1, L1, BLEND_1);
    H1p = _mm_shuffle_epi32(H1p, SHUFFLE_1);

    // Level 2
    // Get min values
    __m128i L2 = _mm_min_epi32(L1p, H1p);
    // Get max values
    __m128i H2 = _mm_max_epi32(L1p, H1p);

    __m128i L2p =  _mm_blend_epi32(L2, H2, BLEND_2);
    __m128i H2p =  _mm_blend_epi32(H2, L2, BLEND_2);
    H2p = _mm_shuffle_epi32(H2p, SHUFFLE_2);

    // Level 3
    // Get min values
    __m128i L3 = _mm_min_epi32(L2p, H2p);
    // Get max values
    __m128i H3 = _mm_max_epi32(L2p, H2p);
    print_128_num(L3);
    print_128_num(H3);
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
