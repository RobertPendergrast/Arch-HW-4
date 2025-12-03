#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

uint32_t* read_array_from_file(const char *filename, uint64_t *out_size) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file '%s'\n", filename);
        return NULL;
    }

    // Read size (first 8 bytes as uint64_t)
    uint64_t size;
    if (fread(&size, sizeof(uint64_t), 1, file) != 1) {
        printf("Error: Could not read size from file\n");
        fclose(file);
        return NULL;
    }

    if (size == 0) {
        printf("Error: Size is 0\n");
        fclose(file);
        return NULL;
    }

    // Allocate array
    uint32_t *arr = malloc(size * sizeof(uint32_t));
    if (!arr) {
        printf("Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    // Read array elements
    size_t elements_read = fread(arr, sizeof(uint32_t), size, file);
    if (elements_read != size) {
        printf("Error: Expected %lu elements, but read %zu\n", size, elements_read);
        free(arr);
        fclose(file);
        return NULL;
    }

    fclose(file);
    *out_size = size;
    return arr;
}

int write_array_to_file(const char *filename, uint32_t *arr, uint64_t size) {
    FILE *outfile = fopen(filename, "wb");
    if (!outfile) {
        printf("Error: Could not open output file '%s'\n", filename);
        return -1;
    }

    // Write size first (as uint64_t)
    if (fwrite(&size, sizeof(uint64_t), 1, outfile) != 1) {
        printf("Error: Could not write size to output file\n");
        fclose(outfile);
        return -1;
    }

    // Write the array
    if (fwrite(arr, sizeof(uint32_t), size, outfile) != size) {
        printf("Error: Could not write array to output file\n");
        fclose(outfile);
        return -1;
    }

    fclose(outfile);
    return 0;
}

void print_array(uint32_t *arr, uint64_t size) {
    for (uint64_t i = 0; i < size; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");
}


int verify_sortedness(uint32_t *arr, uint64_t size) {
    for (uint64_t i = 1; i < size; i++) {
        if (arr[i] < arr[i - 1]) {
            printf("Array not sorted at index %lu: %u > %u\n", i - 1, arr[i - 1], arr[i]);
            return 0;
        }
    }
    return 1;
}
