#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <pthread.h>
#include <string.h>

// Constant Definitions
#define L3_CACHE_SIZE 64 // For test purposes set to 16 elements
#define NUM_THREADS 4 // We will use 4 threads for merging
#define CACHE_LINE_SIZE 64 // Cache Line Size

// Struct to hold thread arguments
typedef struct {
    uint32_t *arr; // Pointer to array partition
    size_t size; // Size of the partition
} ThreadArgs;

void print_array(uint32_t *arr, uint64_t size) {
    for (uint64_t i = 0; i < size; i++) {
        printf("%u ", arr[i]);
    }
    printf("\n");
}

void merge(uint32_t* left, uint32_t* right, uint32_t* arr, int size_left, int size_right) {
    int i = 0;
    int j = 0;
    int k = 0;
    // Merge the two arrays into arr
    while (i < size_left && j < size_right) {
        if (left[i] <= right[j]) {
            arr[k++] = left[i++];
        } else {
            arr[k++] = right[j++];
        }
    }
    // Merge the rest in
    while (i < size_left) {
        arr[k++] = left[i++];
    }
    while (j< size_right) {
        arr[k++] = right[j++];
    }
}

// A single thread will use this sort function to sort a partition
void single_thread_sort(uint32_t *arr, size_t size) {
    // Basic Merge Sort Implementation
    if (size < 2) {
        return; // Nothing to sort
    }

    int middle = size / 2;
    int size_left = middle;
    int size_right = size - middle;

    // TODO: Use aligned allocation for better cache performance
    uint32_t* left = malloc(size_left * sizeof(uint32_t));
    uint32_t* right = malloc(size_right * sizeof(uint32_t));

    memcpy(left, arr, size_left * sizeof(uint32_t));
    memcpy(right, arr + middle, size_right * sizeof(uint32_t));

    single_thread_sort(left, size_left);
    single_thread_sort(right, size_right);

    // Merge the two halves 
    merge(left, right, arr, size_left, size_right);
}

void* single_thread_sort_wrapper(void *arg){
    ThreadArgs *args = (ThreadArgs *)arg;
    single_thread_sort(args->arr, args->size);
    return NULL;
}

// Split the array into N Partitions
void split_L3_cache_for_multi_threading(uint32_t *arr, size_t size){
    // Partition Size is L3 Cache Size
    size_t partition_size = L3_CACHE_SIZE / NUM_THREADS;
    pthread_t threads[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        // What the block has little data?
        if (i*partition_size > size){
            break;
        }
        ThreadArgs *args = malloc(sizeof(*args));
        args->arr = arr + i*partition_size;
        if (i == NUM_THREADS - 1) {
            partition_size = size - i*partition_size;
        } 
        if (size < partition_size){
            partition_size = size;
        }
        args->size = partition_size;  
        //print_array(args->arr, args->size);
        pthread_create(&threads[i],NULL,single_thread_sort_wrapper, args);
    }

    // Wait for all the threads to complete
    for (int i = 0; i< NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    return;
}

// Return a whole sorted l3 cache
// This might not be the full cache size though!
void sort_l3_cache_sized(uint32_t *arr, size_t size){
    
    // First split the array and sort them in parallel
    split_L3_cache_for_multi_threading(arr, size);

    // Then merge the sorted partitions
    // Right now just do pairwise merging
    size_t max_partition_size = L3_CACHE_SIZE / NUM_THREADS;
    int num_partitions = (size + max_partition_size -1) / max_partition_size; 
    
    for (int i = 1; i< num_partitions; i*=2) { // Logarithmic scaling

        // Merge pairs of partitions in place
        for (int j = 0; j < num_partitions; j += 2) {
            // TODO: Actually merge the partitions
        }  
    }
}

// Merge multiple cache sizes (rehash the k-way merge)
void merge_multiple_l3_cached_sizes(uint32_t *arr, size_t size){
    // TODO: Do this lmao
}

int main(int argc, char *argv[]) {
    // Read array from input file
    uint64_t size;
    //uint32_t *arr = read_array_from_file(argv[1], &size);
    uint32_t arr[] = {1, 4, 8, 5, 
                    16, 12, 7, 3, 
                    2, 11, 10, 9,
                    6, 15, 13, 14,
                
                    1, 4, 8, 5, 
                    16, 12, 7, 3, 
                    2, 11, 10, 9,
                    6, 15, 13, 14,
                
                    1, 4, 8, 5, 
                    16, 12, 7, 3, 
                    2, 11, 10, 9,
                    6, 15, 13, 14,
                
                    1, 4, 8, 5, 
                    16, 12, 7, 3, 
                    2, 11, 10, 9,
                    6, 15, 13, 14};  // 64 elements, each group of 16 will be sorted in a thread
    size = sizeof(arr) / sizeof(arr[0]);

    split_L3_cache_for_multi_threading(arr, size);

    print_array(arr,size);

    return 0;
}