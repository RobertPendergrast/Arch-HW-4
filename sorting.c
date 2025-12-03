#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include "utils.h"

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {

}

void merge(uint32_t* left, uint32_t* right, uint32_t* arr, int size_left, int size_right) {
    int i = 0;
    // Merge the two arrays into arr
    while (i < size_left && i < size_right) {
        if (left[i] <= right[i]) {
            arr[i] = left[i];
        } else {
            arr[i] = right[i];
        }
        i++;
    }
    // Merge the rest in
    while (i < size_left) {
        arr[i] = left[i];
        i++;
    }
    while (i< size_right) {
        arr[i] = right[i];
        i++;
    }
}

void basic_merge_sort(uint32_t *arr, size_t size) {
    // Basic Merge Sort Implementation
    if (size < 2) {
        return; // Nothing to sort
    }
    int middle = size / 2;
    int size_left = middle;
    int size_right = size - middle;

    uint32_t* left = malloc(size_left * sizeof(uint32_t));
    uint32_t* right = malloc(size_right * sizeof(uint32_t));

    memcpy(left, arr, size_left * sizeof(uint32_t));
    memcpy(right, arr + middle, size_right * sizeof(uint32_t));

    basic_merge_sort(left, size_left);
    basic_merge_sort(right, size_right);

    // Merge the two halves 
    merge(left, right, arr, size_left, size_right);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input_file output_file\n", argv[0]);
        return 1; 
    }

    // Read array from input file
    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) {
        return 1;
    }

    basic_merge_sort(arr, size);

    // Verify the array is sorted
    if (verify_sortedness(arr, size)) {
        printf("Array sorted successfully!\n");
    } else {
        printf("Error: Array is not sorted correctly!\n");
        free(arr);
        return 1;
    }

    // Write sorted array to output file
    if (write_array_to_file(argv[2], arr, size) != 0) {
        free(arr);
        return 1;
    }

    free(arr);
    return 0;
}

       
