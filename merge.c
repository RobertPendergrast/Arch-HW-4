#include <stdint.h> 
#include <stdlib.h> 
#include <stdio.h>

#include "utils.h"

uint32_t* merge_arrays(
    uint32_t *arr_1,
    size_t size_1,
    uint32_t *arr_2,
    size_t size_2
) {
    uint32_t *merged_array = (uint32_t *)malloc(
        (size_1 + size_2) * sizeof(uint32_t)
    ); // Allocate memory for the sorted array
    // Do stuff here
    return merged_array;
}


int main(int argc, char *argv[]) {
    uint32_t test_arr_1[] = {1, 5, 8, 2};
    size_t size_1 = sizeof(test_arr_1);
    uint32_t test_arr_2[] = {4, 7, 3, 6};
    size_t size_2 = sizeof(test_arr_2);
    uint32_t* ret_arr = merge_arrays(
        test_arr_1,
        size_1,
        test_arr_2,
        size_2
    );
    print_array(ret_arr, size_1 + size_2);
}
