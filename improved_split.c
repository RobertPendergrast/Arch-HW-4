#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include "utils.h"

void merge(uint32_t* left, uint32_t* right, uint32_t* arr, int size_left, int size_right) {
    int i, j, k = 0;
    // Merge the two arrays into arr
    while (i < size_left && i < size_right) {
        if (left[j] <= right[k]) {
            arr[i] = left[j];
            j++;
        } else {
            arr[i] = right[k];
            k++;
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
    // Basic Merge Sort Implementation
    if (size < 2) {
        return; // Nothing to sort
    } if (size == 2) {
        sort_two(arr);
        return;
    } if (size == 3) {
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

    basic_merge_sort(left, size_left);
    basic_merge_sort(right, size_right);

    // Merge the two halves 
    merge(left, right, arr, size_left, size_right);
}

int main(int argc, char *argv[]) {
    // TODO: It would probably be better to read the array from a file instead of pass in as a command line argument
    if (argc < 3) {
        printf("Usage: %s <size_of_array> <array_elements>\n", argv[0]);
        return 1; 
    }

    //Initialise the array
    int size = atoi(argv[1]);

    uint32_t *sorted_arr = malloc(size * sizeof(uint32_t)); // Allocate memory for the sorted array

    printf("Basic Sorting Selected\n");
    improved_split(sorted_arr, size);

    // Print the sorted array
    for (int i = 0; i < size; i++) {
        printf("%d", sorted_arr[i]);
    }
    printf("\n");

    // Free and return
    free(sorted_arr);
    return 0;
}

       
