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
    // TODO: It would probably be better to read the array from a file instead of pass in as a command line argument
    if (argc < 3) {
        printf("Usage: %s input_file output_file\n", argv[0]);
        return 1; 
    }

    //Size is the first 8 bytes of the input file
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error: Could not open file '%s'\n", argv[1]);
        return 1;
    }
    uint32_t size;
    fread(&size, sizeof(uint64_t), 1, file);
    if (size == 0) {
        printf("Error: Size is 0\n");
        return 1;
    }
    uint32_t *arr = malloc(size * sizeof(uint32_t));
    int read = fread(arr, sizeof(uint32_t), size, file);
    if (read != size) {
        printf("Error: Could not read array from file\n");
        return 1;
    }
    fclose(file);

    basic_merge_sort(arr, size);

    FILE *outfile = fopen(argv[2], "wb");
    if (!outfile) {
        printf("Error: Could not open output file '%s'\n", argv[2]);
        free(arr);
        return 1;
    }
    // Write the size first (as uint64_t for compatibility with reading code)
    uint64_t f_size = size;
    if (fwrite(&f_size, sizeof(uint64_t), 1, outfile) != 1) {
        printf("Error: Could not write size to output file\n");
        fclose(outfile);
        free(arr);
        return 1;
    }
    // Write the sorted array
    if (fwrite(arr, sizeof(uint32_t), size, outfile) != size) {
        printf("Error: Could not write array to output file\n");
        fclose(outfile);
        free(arr);
        return 1;
    }
    fclose(outfile);

    // Free and return
    free(arr);
    return 0;
}

       
