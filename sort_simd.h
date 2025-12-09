#ifndef SORT_SIMD_H
#define SORT_SIMD_H

#include <stddef.h>
#include <stdint.h>

// Main sorting entry point exposed to callers.
void sort_array(uint32_t *arr, size_t size);

#endif // SORT_SIMD_H
