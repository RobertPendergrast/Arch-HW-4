#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>
#include "utils.h"
#include "pthread.h"
#include "merge.h"
#define MAX_DEPTH 5  // Limit parallel depth to avoid too many threads



struct thread_args {
    uint32_t *arr;
    size_t size;
    int depth;
};

void* threaded_devide_merge_sort_thread(void* arg);

void threaded_devide_merge_sort(uint32_t *arr, size_t size, int depth) {
    if (size < 2) {
        return; // Nothing to sort
    }
    int middle = size / 2;
    int size_left = middle;
    int size_right = size - middle;

    uint32_t* left = malloc(size_left * sizeof(uint32_t));
    uint32_t* right = malloc(size_right * sizeof(uint32_t));

    memcpy(left, arr, size_left * sizeof(uint32_t));
    memcpy(right, arr + middle, size_right * sizeof(uint32_t));
    if (depth < MAX_DEPTH && size > 1000) {
        pthread_t left_thread, right_thread;
        

        struct thread_args *left_args = malloc(sizeof(struct thread_args));
        left_args->arr = left;
        left_args->size = size_left;
        left_args->depth = depth + 1;
        pthread_create(&left_thread, NULL, threaded_devide_merge_sort_thread, left_args);

        struct thread_args *right_args = malloc(sizeof(struct thread_args));
        right_args->arr = right;
        right_args->size = size_right;
        right_args->depth = depth + 1;
        pthread_create(&right_thread, NULL, threaded_devide_merge_sort_thread, right_args);
        pthread_join(left_thread, NULL);
        pthread_join(right_thread, NULL);
    } else {
        threaded_devide_merge_sort(left, size_left, depth + 1);
        threaded_devide_merge_sort(right, size_right, depth + 1);
    }
    // Merge the two halves 
    merge_arrays(left, size_left, right, size_right, arr);
    free(left);
    free(right);
}

void* threaded_devide_merge_sort_thread(void* arg) {
    struct thread_args* args = (struct thread_args*)arg;
    threaded_devide_merge_sort(args->arr, args->size, args->depth);
    free(arg);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s input_file output_file\n", argv[0]);
        return 1; 
    }

    // Read array from input file
    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) {
        return 1;
    }

    threaded_devide_merge_sort(arr, size, 0);

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

       
