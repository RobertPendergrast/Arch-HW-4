#include "merge.h"

void merge_arrays(
    uint32_t *arr_left,
    size_t size_left,
    uint32_t *arr_right,
    size_t size_right,
    uint32_t *merged_arr
) {
    int i = 0, j = 0, k = 0;
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

