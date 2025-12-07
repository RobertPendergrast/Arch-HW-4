#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "merge.h"
#include "utils.h"

void run_128(){
    uint32_t left[] = {1, 2, 5, 8};
    // size_t size_left = sizeof(left) / sizeof(uint32_t);
    uint32_t right[] = {3, 4, 6, 7};
    // size_t size_right = sizeof(right) / sizeof(uint32_t);
     __m128i left_reg = _mm_loadu_si128((__m128i*) left);
     __m128i right_reg = _mm_loadu_si128((__m128i*) right);
    printf("Before: \n");
    printf("Left: \n");
    print_128_num(left_reg);
    printf("Right: \n");
    print_128_num(right_reg);
    merge_128_registers(&left_reg, &right_reg);
    printf("After: \n");
    printf("Left: \n");
    print_128_num(left_reg);
    printf("Right: \n");
    print_128_num(right_reg);
}

void run_512(){
    // uint32_t left[] = {1, 4, 6, 7, 8, 12, 17, 18, 19, 20, 21, 23, 24 ,25, 27, 30};
    // uint32_t right[] = {2, 3, 5, 9, 10, 11, 13, 14, 15, 16, 22, 26, 28, 29, 31, 32};
    uint32_t left[] = {7, 1, 15, 4, 11, 2, 9, 16, 5, 10, 3, 14, 8, 12, 6, 13};
    uint32_t right[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

     __m512i left_reg = _mm512_loadu_epi32((__m512i*) left);
     __m512i right_reg = _mm512_loadu_epi32((__m512i*) right);

    printf("Before: \n");
    printf("Left: \n");
    print_512_num(left_reg);
    printf("Right: \n");
    print_512_num(right_reg);

    merge_512_registers(&left_reg, &right_reg);
    printf("After: \n");
    printf("Left: \n");
    print_512_num(left_reg);
    printf("Right: \n");
    print_512_num(right_reg);
}

int main(int argc, char *argv[]) {
    run_512();
}
