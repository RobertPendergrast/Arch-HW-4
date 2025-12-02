#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {

}

void baseline_sort_array(uint32_t *arr, size_t size) {
    // Basic Merge Sort Implementation
    if (size < 2) {
        return; // Nothing to sort
    }
    int middle = size / 2;
    baseline_sort_array(arr, middle);
    baseline_sort_array(arr + middle, size - middle);

    // Merge the two halves (This can definitely be optimized!)
    merge();
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
        baseline_sort_array(sorted_arr, size);
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

       
