#ifndef MERGE_H
#define MERGE_H

#include <stdint.h>
#include <stddef.h>
#include <immintrin.h>

// ============== Shared SIMD shuffle indices ==============
// These are used by both sorting and merging networks

// Full reverse: [0,1,...,15] -> [15,14,...,0]
extern const int IDX_REV[16];

// Swap halves: [0-7,8-15] -> [8-15,0-7]
extern const int IDX_SWAP8[16];

// Swap within groups of 8: [0-3,4-7] -> [4-7,0-3] in each half
extern const int IDX_SWAP4[16];

// ============== Function declarations ==============

/*
 * Takes in two __m128i registers, each with 4 32bit integers and performs a
 * bitonic merge in place.
 */
void merge_128_registers(
    __m128i *left,
    __m128i *right
);

/*
 * Takes in two __m512i registers, each with 16 32bit integers and performs a
 * bitonic merge in place.
*/
void merge_512_registers(
    __m512i *left,
    __m512i *right
);

// STREAMING version: uses non-temporal stores, best for large out-of-cache merges
void merge_arrays(
    uint32_t *left,
    size_t size_left,
    uint32_t *right,
    size_t size_right,
    uint32_t *arr
);

// CACHED version: uses regular stores, best for L3-resident data that will be reused
void merge_arrays_cached(
    uint32_t *left,
    size_t size_left,
    uint32_t *right,
    size_t size_right,
    uint32_t *arr
);

// Variant that accepts unaligned inputs (for parallel merge splitting)
// Output MUST still be 64-byte aligned
void merge_arrays_unaligned(
    uint32_t *left,
    size_t size_left,
    uint32_t *right,
    size_t size_right,
    uint32_t *arr
);

// ============== KEY-VALUE VERSIONS (for stability testing) ==============
// These sort payload arrays alongside keys without using payload for comparison.
// Useful for tracking element movement to measure sorting stability.

// STREAMING KV version: uses non-temporal stores, best for large out-of-cache merges
void merge_arrays_kv(
    uint32_t *left_key,
    uint32_t *left_payload,
    size_t size_left,
    uint32_t *right_key,
    uint32_t *right_payload,
    size_t size_right,
    uint32_t *arr_key,
    uint32_t *arr_payload
);

// CACHED KV version: uses regular stores, best for L3-resident data that will be reused
void merge_arrays_cached_kv(
    uint32_t *left_key,
    uint32_t *left_payload,
    size_t size_left,
    uint32_t *right_key,
    uint32_t *right_payload,
    size_t size_right,
    uint32_t *arr_key,
    uint32_t *arr_payload
);

#endif
