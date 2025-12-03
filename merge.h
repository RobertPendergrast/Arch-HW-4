#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>
#include <stddef.h>

/*
 * Takes in two uint32_t arrays and returns the merged uint32_t array
 */
void merge_arrays(
    uint32_t *arr_1,
    size_t size_1,
    uint32_t *arr_2,
    size_t size_2,
    uint32_t *merged_arr
);

#endif
