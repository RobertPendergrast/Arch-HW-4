#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <utils.h>

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {

}

void merge(int* left, int* right, int* arr, int size_left, int size_right) {
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

void basic_merge_sort(int *arr, size_t size) {
    // Basic Merge Sort Implementation
    if (size < 2) {
        return; // Nothing to sort
    }
    int middle = size / 2;
    int size_left = middle;
    int size_right = size - middle;

    int* left = malloc(size_left * sizeof(int));
    int* right = malloc(size_right * sizeof(int));

    memcpy(left, arr, size_left * sizeof(int));
    memcpy(right, arr + middle, size_right * sizeof(int));

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

     // Sort the copied array
    if (strcmp(argv[3], "-o")== 0){
        printf("Optimized Sorting Selected\n");
        sort_array(sorted_arr, size);
    } else {
        printf("Basic Sorting Selected\n");
        basic_merge_sort(sorted_arr, size);
    }

    // Print the sorted array
    for (int i = 0; i < size; i++) {
        printf("%d", sorted_arr[i]);
    }
    printf("\n");

    // Free and return
    free(sorted_arr);
    return 0;
}

       
