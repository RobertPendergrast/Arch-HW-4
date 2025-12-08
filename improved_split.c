#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include <time.h>
#include "utils.h"

void merge(uint32_t* left, uint32_t* right, uint32_t* arr, int size_left, int size_right) {
    int i = 0;
    int j = 0;
    int k = 0;
    // Merge the two arrays into arr
    while (i < size_left && j < size_right) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }
    // Merge the rest in
    while (i < size_left) {
        arr[k++] = left[i++];
    }
    while (j < size_right) {
        arr[k++] = right[j++];
    }
}

void sort_three(uint32_t *arr) {
    uint32_t temp = 0; 
    if (arr[0] > arr[1]) {
        temp = arr[0];
        arr[0] = arr[1];
        arr[1] = temp;
    }
    if (arr[2] < arr[1]) {
        temp = arr[1];
        arr[1] = arr[2];
        arr[2] = temp;
        if (arr[1] < arr[0]) {
            temp = arr[0];
            arr[0] = arr[1];
            arr[1] = temp;
        }
    }
}

void sort_two(uint32_t *arr) {
    uint32_t temp = 0;
    if(arr[1] < arr[0]) {
        temp = arr[0];
        arr[0] = arr[1];
        arr[1] = temp;
    }
}

void improved_split(uint32_t *arr, size_t size) {
    // Improved Merge Sort with optimized base cases
    if (size < 2) {
        return; // Nothing to sort
    } 
    if (size == 2) {
        sort_two(arr);
        return;
    } 
    if (size == 3) {
        sort_three(arr);
        return;
    }

    // Split the array
    int middle = size / 2;
    int size_left = middle;
    int size_right = size - middle;

    uint32_t* left = malloc(size_left * sizeof(uint32_t));
    uint32_t* right = malloc(size_right * sizeof(uint32_t));

    memcpy(left, arr, size_left * sizeof(uint32_t));
    memcpy(right, arr + middle, size_right * sizeof(uint32_t));

    improved_split(left, size_left);
    improved_split(right, size_right);

    // Merge the two halves 
    merge(left, right, arr, size_left, size_right);
    free(left);
    free(right);
}

// Main sorting function - matches interface expected by benchmark
void sort_array(uint32_t *arr, size_t size) {
    improved_split(arr, size);
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

    improved_split(arr, size);

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


