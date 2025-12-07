#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include <time.h>
#include "utils.h"
#include "merge.h"

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {

}

// Insertion sort for small arrays (faster than merge sort for small n)
void insertion_sort(uint32_t *arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        uint32_t key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

// Base case threshold: use insertion sort for arrays smaller than this
#define SORT_THRESHOLD 2

void basic_merge_sort(uint32_t *arr, size_t size) {
    // Base case: use insertion sort for small arrays
    if (size <= SORT_THRESHOLD) {
        //insertion_sort(arr, size);
        return;
    }
    int middle = size / 2;
    int size_left = middle;
    int size_right = size - middle;

    // Ensure left and right arrays are cache line aligned (typically 64 bytes)
    uint32_t* left = NULL;
    uint32_t* right = NULL;
    if (posix_memalign((void**)&left, 64, size_left * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for left array\n");
        exit(EXIT_FAILURE);
    }
    if (posix_memalign((void**)&right, 64, size_right * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for right array\n");
        free(left);
        exit(EXIT_FAILURE);
    }

    memcpy(left, arr, size_left * sizeof(uint32_t));
    memcpy(right, arr + middle, size_right * sizeof(uint32_t));

    basic_merge_sort(left, size_left);
    basic_merge_sort(right, size_right);

    // Merge the two halves 
    merge_arrays(left, size_left, right, size_right, arr);
    free(left);
    free(right);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    // Read array from input file
    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) {
        return 1;
    }

    printf("Read %lu elements from %s\n", size, argv[1]);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    basic_merge_sort(arr, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);

    // Only print array if small enough
    if (size <= 10) {
        print_array(arr, size);
    }

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

       
