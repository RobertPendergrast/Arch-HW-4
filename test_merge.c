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
    uint32_t left[] = {1, 4, 6, 7, 8, 12, 17, 18, 19, 20, 21, 23, 24 ,25, 27, 30};
    uint32_t right[] = {2, 3, 5, 9, 10, 11, 13, 14, 15, 16, 22, 26, 28, 29, 31, 32};
    // uint32_t left[] = {7, 1, 15, 4, 11, 2, 9, 16, 5, 10, 3, 14, 8, 12, 6, 13};
    // uint32_t right[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

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

void run_merge_arrays(){
    printf("=== Test 1: 16 + 16 elements ===\n");
    uint32_t left[] = {1, 4, 6, 7, 8, 12, 17, 18, 19, 20, 21, 23, 24 ,25, 27, 30};
    uint32_t right[] = {2, 3, 5, 9, 10, 11, 13, 14, 15, 16, 22, 26, 28, 29, 31, 32};
    uint32_t arr[32];
    merge_arrays(left, 16, right, 16, arr);
    printf("After: \n");
    printf("Arr: \n");
    print_array(arr, 32);

    printf("\n=== Test 2: 32 + 32 elements ===\n");
    uint32_t left2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49};
    uint32_t right2[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
    printf("Left2 size: %zu\n", sizeof(left2) / sizeof(uint32_t));
    printf("Right2 size: %zu\n", sizeof(right2) / sizeof(uint32_t));
    uint32_t arr2[64];
    merge_arrays(left2, 32, right2, 32, arr2);
    printf("After: \n");
    printf("Arr: \n");
    print_array(arr2, 64);
    
    printf("\n=== Test 3: 33 + 33 elements (non-multiple of 16) ===\n");
    uint32_t left3[33], right3[33], arr3[66];
    // Fill with sorted odd numbers for left, even for right
    for (int i = 0; i < 33; i++) {
        left3[i] = 2*i + 1;   // 1, 3, 5, ..., 65
        right3[i] = 2*i + 2;  // 2, 4, 6, ..., 66
    }
    printf("Left3: last 5 elements: %u %u %u %u %u\n", left3[28], left3[29], left3[30], left3[31], left3[32]);
    printf("Right3: last 5 elements: %u %u %u %u %u\n", right3[28], right3[29], right3[30], right3[31], right3[32]);
    merge_arrays(left3, 33, right3, 33, arr3);
    printf("After: \n");
    printf("Arr3 (should be 1-66): \n");
    print_array(arr3, 66);
    
    // Verify correctness
    int correct = 1;
    for (int i = 0; i < 66; i++) {
        if (arr3[i] != (uint32_t)(i + 1)) {
            printf("ERROR at index %d: expected %d, got %u\n", i, i + 1, arr3[i]);
            correct = 0;
        }
    }
    if (correct) printf("Test 3 PASSED!\n");
}

int main(int argc, char *argv[]) {
    run_merge_arrays();
    return 0;
}
