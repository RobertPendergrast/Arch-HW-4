#ifndef SORT_SIMD_KV_H
#define SORT_SIMD_KV_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include "utils.h"
#include "merge.h"

void sort_array_kv(uint32_t *arr, uint32_t *payload, size_t size);

#endif // SORT_SIMD_KV_H
