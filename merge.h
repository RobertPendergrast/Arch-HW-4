#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>
#include <stddef.h>

#include <immintrin.h>

/*
 * Takes in two __m128i registers, each with 4 32bit integers and performs a
 * bitonic merge in place.
 */
void merge_128_registers(
    __m128i *left,
    __m128i *right
);

/*
 * Takes in two __m512i registers, each with 16 32bit integers and performs a
 * bitonic merge in place.
*/
void merge_512_registers(
    __m512i *left,
    __m512i *right
);


#endif
