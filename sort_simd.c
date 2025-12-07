#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include "utils.h"
#include "merge.h"

// Avoid making changes to this function skeleton, apart from data type changes if required
// In this starter code we have used uint32_t, feel free to change it to any other data type if required
void sort_array(uint32_t *arr, size_t size) {

}

// Insertion sort for small arrays
static inline void insertion_sort(uint32_t *arr, size_t size) {
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

// SIMD sorting network for exactly 32 elements (two 512-bit registers)
static inline void sort_32_simd(uint32_t *arr) {
    __m512i a = _mm512_loadu_epi32(arr);
    __m512i b = _mm512_loadu_epi32(arr + 16);
    
    // Sort each register individually using bitonic sort
    // This requires sorting network within each register first
    // For now, use merge to combine two sorted halves
    
    // Sort first 16: split into two 8s, sort, merge
    // Simplified: just merge the two registers (assumes inputs need full sort)
    // We'll use a simple approach: sort in place then merge
    
    // For a proper 32-element sort, we need a full sorting network
    // Fallback to insertion sort for the base case for correctness
    insertion_sort(arr, 32);
}

// Base case threshold - larger = fewer merge passes, but base sort matters more
#define SORT_THRESHOLD 64

// Bottom-up merge sort: O(1) allocations instead of O(N) allocations!
void basic_merge_sort(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    // Step 1: Sort all base-case chunks in place
    for (size_t i = 0; i < size; i += SORT_THRESHOLD) {
        size_t chunk_size = (i + SORT_THRESHOLD <= size) ? SORT_THRESHOLD : (size - i);
        insertion_sort(arr + i, chunk_size);
    }
    
    // Step 2: Allocate ONE temp buffer for all merges
    uint32_t *temp = NULL;
    if (posix_memalign((void**)&temp, 64, size * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for temp buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // Step 3: Bottom-up merge passes
    // Each pass doubles the size of sorted runs
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    for (size_t width = SORT_THRESHOLD; width < size; width *= 2) {
        // Merge adjacent pairs of runs
        size_t i = 0;
        while (i < size) {
            size_t left_start = i;
            size_t left_size = (left_start + width <= size) ? width : (size - left_start);
            size_t right_start = left_start + left_size;
            size_t right_size = 0;
            
            if (right_start < size) {
                right_size = (right_start + width <= size) ? width : (size - right_start);
            }
            
            if (right_size == 0) {
                // No right half - just copy left to dst
                memcpy(dst + left_start, src + left_start, left_size * sizeof(uint32_t));
            } else {
                // Merge left and right into dst
                merge_arrays(src + left_start, left_size, 
                           src + right_start, right_size, 
                           dst + left_start);
            }
            
            i = right_start + right_size;
        }
        
        // Swap src and dst for next pass
        uint32_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    // If result ended up in temp, copy back to arr
    if (src != arr) {
        memcpy(arr, src, size * sizeof(uint32_t));
    }
    
    free(temp);
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

       
