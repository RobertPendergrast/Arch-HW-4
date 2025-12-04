#include <stdint.h>
#include <stdlib.h>

#include "merge.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    uint32_t left[] = {1, 2, 5, 8};
    // size_t size_left = sizeof(left) / sizeof(uint32_t);
    uint32_t right[] = {3, 4, 6, 7, 9};
    // size_t size_right = sizeof(right) / sizeof(uint32_t);
     __m128i left_reg = _mm_loadu_si128((__m128i*) left);
     __m128i right_reg = _mm_loadu_si128((__m128i*) right);
    merge_128_registers(left_reg, right_reg);
}
