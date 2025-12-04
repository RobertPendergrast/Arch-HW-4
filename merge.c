#include <string.h>
#include <stdio.h>

#include "merge.h"


void print_128_num(__m128i var)
{
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %i %i %i %i \n", 
           val[0], val[1], val[2], val[3]);
}

void print_512_num(__m512i var)
{
    uint32_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n", 
        val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
        val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]
    );
}

void merge_128_registers(
    __m128i left,
    __m128i right
) {
    print_128_num(left);
    print_128_num(right);
}

void merge_512_registers(
    __m512i left,
    __m512i right
) {
    print_512_num(left);
    print_512_num(right);
}
