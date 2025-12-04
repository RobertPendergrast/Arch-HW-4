#include "merge.h"
#include "utils.h"

/*
 * Takes in two m128i registers and merges them in place.
 */
void merge_128_registers(
    __m128i left,
    __m128i right
) {
    print_128_num(left);
    print_128_num(right);
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
