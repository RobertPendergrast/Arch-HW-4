#include <stdint.h> 
#include <stdlib.h> 
#include <stdio.h>

#include "utils.h"

void merge_arrays(
    uint32_t *arr_left,
    size_t size_left,
    uint32_t *arr_right,
    size_t size_right,
    uint32_t *merged_arr
) {
    int i, j, k = 0;
    // Merge the two arrays into arr
    while (i < size_left && i < size_right) {
        if (arr_left[j] <= arr_right[k]) {
            merged_arr[i] = arr_left[j];
            j++;
        } else {
            merged_arr[i] = arr_right[k];
            k++;
        }
        i++;
    }
    // Merge the rest in
    while (i < size_left) {
        merged_arr[i] = arr_left[i];
        i++;
    }
    while (i < size_right) {
        merged_arr[i] = arr_right[i];
        i++;
    }
}


int main(int argc, char *argv[]) {
    uint32_t test_arr_1[] = {1, 5, 8, 2};
    size_t size_1 = sizeof(test_arr_1);
    uint32_t test_arr_2[] = {4, 7, 3, 6};
    size_t size_2 = sizeof(test_arr_2);
    uint32_t* arr = malloc((size_1) + (size_2) * sizeof(uint32_t));
    merge_arrays(
        test_arr_1,
        size_1,
        test_arr_2,
        size_2,
        arr
    );
    print_array(arr, size_1 + size_2);
}
