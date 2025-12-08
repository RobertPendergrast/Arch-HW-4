#include "utils.h"

#include <string.h>
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

    // Allocate 64-byte aligned array for cache line and SIMD alignment
    uint32_t *arr = NULL;
    if (posix_memalign((void**)&arr, 64, size * sizeof(uint32_t)) != 0) {
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

/*
 * Creates an 4 long array of integers and copies the m128i register into them
 * before printing the values out.
 */
void print_128_num(__m128i var)
{
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %i %i %i %i \n", 
           val[0], val[1], val[2], val[3]);
}

/*
 * Creates an 16 long array of integers and copies the m512i register into them
 * before printing the values out.
 */
void print_512_num(__m512i var)
{
    uint32_t val[16];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i %i \n", 
        val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7],
        val[8], val[9], val[10], val[11], val[12], val[13], val[14], val[15]
    );
}

// Order-independent hash using XOR and sum (parallel)
// Returns two values: xor_hash and sum_hash
static void compute_hash(uint32_t *arr, size_t size, uint64_t *xor_out, uint64_t *sum_out) {
    uint64_t xor_hash = 0;
    uint64_t sum_hash = 0;
    
    #pragma omp parallel reduction(^:xor_hash) reduction(+:sum_hash)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++) {
            xor_hash ^= arr[i];
            sum_hash += arr[i];
        }
    }
    
    *xor_out = xor_hash;
    *sum_out = sum_hash;
}