#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stddef.h>

// Read an array from a binary file
// File format: first 8 bytes = size (uint64_t), followed by size * uint32_t elements
// Returns a malloc'd array, caller must free
// Sets *out_size to the number of elements
uint32_t* read_array_from_file(const char *filename, uint64_t *out_size);

// Write an array to a binary file
// File format: first 8 bytes = size (uint64_t), followed by size * uint32_t elements
// Returns 0 on success, -1 on failure
int write_array_to_file(const char *filename, uint32_t *arr, uint64_t size);

// Print array elements to stdout
void print_array(uint32_t *arr, uint64_t size);

// Verify array is sorted in ascending order
// Returns 1 if sorted, 0 if not sorted
int verify_sortedness(uint32_t *arr, uint64_t size);

#endif
