#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>
#include <stddef.h>

/*
 * Takes in two uint32_t arrays and returns the merged uint32_t array
 */
void merge_arrays(
    uint32_t *left,
    size_t size_left,
    uint32_t *right,
    size_t size_right,
    uint32_t *arr
);

#endif
