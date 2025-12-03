#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>

void print_array(uint32_t *arr, size_t size) {
    for (int i = 0; i < size; i++) {
        printf("%u, ", arr[i]);
    }
    printf("\n");
}