#ifndef SORT_SIMD_H
#define SORT_SIMD_H

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

void sort_array(uint32_t *arr, size_t size);

#endif // SORT_SIMD_H