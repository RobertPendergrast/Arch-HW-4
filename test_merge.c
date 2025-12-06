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
    uint32_t left[] = {32, 24, 1, 17, 28, 3, 10, 15, 21, 4, 26, 12, 30, 8, 19, 2};
    // size_t size_left = sizeof(left) / sizeof(uint32_t);
    uint32_t right[] = {11, 25, 9, 14, 29, 5, 18, 6, 22, 13, 27, 7, 20, 31, 16, 23};
    // size_t size_right = sizeof(right) / sizeof(uint32_t);
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
