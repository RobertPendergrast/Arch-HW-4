#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include "utils.h"

// L3 cache is 33MB (i think), so 16 threads with 2MB (basically) each maximizes cache locality
#define NUM_THREADS 16
#define CACHE_SIZE_MB 33
#define CHUNK_SIZE (CACHE_SIZE_MB / NUM_THREADS * 1024 * 1024 / sizeof(uint32_t))

// Helper to compute elapsed time in seconds
static double elapsed_sec(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// Thread task structs
typedef struct {
    uint32_t *arr;
    size_t start;
    size_t end;
} SortTask;

typedef struct {
    uint32_t *arr;
    uint32_t *tmp;
    size_t left, mid, right;
} MergeTask;

// Merge two sorted subarrays [left..mid) and [mid..right) into one
static void merge(uint32_t *arr, uint32_t *tmp, size_t left, size_t mid, size_t right) {
    size_t i = left, j = mid, k = 0;
    
    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    while (i < mid)   tmp[k++] = arr[i++];
    while (j < right) tmp[k++] = arr[j++];
    
    memcpy(&arr[left], tmp, (right - left) * sizeof(uint32_t));
}

// Recursive merge sort used by each thread to sort its chunk
static void merge_sort(uint32_t *arr, uint32_t *tmp, size_t left, size_t right) {
    if (right - left < 2) return;
    
    size_t mid = left + (right - left) / 2;
    merge_sort(arr, tmp, left, mid);
    merge_sort(arr, tmp, mid, right);
    merge(arr, tmp, left, mid, right);
}

// Thread entry point for Phase 1: sort a single chunk
static void *sort_thread_fn(void *arg) {
    SortTask *task = (SortTask *)arg;
    size_t size = task->end - task->start;
    
    uint32_t *tmp = malloc(size * sizeof(uint32_t));
    if (tmp) {
        merge_sort(task->arr, tmp, task->start, task->end);
        free(tmp);
    }
    return NULL;
}

// Thread entry point for Phase 2: merge two adjacent sorted regions
static void *merge_thread_fn(void *arg) {
    MergeTask *task = (MergeTask *)arg;
    merge(task->arr, task->tmp, task->left, task->mid, task->right);
    return NULL;
}

// Phase 1: Sort all chunks in parallel (8 threads at a time)
static size_t parallel_sort_chunks(uint32_t *arr, size_t n) {
    size_t num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for (size_t batch = 0; batch < num_chunks; batch += NUM_THREADS) {
        pthread_t threads[NUM_THREADS];
        SortTask tasks[NUM_THREADS];
        int active = 0;
        
        for (int t = 0; t < NUM_THREADS && (batch + t) < num_chunks; t++) {
            size_t idx = batch + t;
            tasks[active].arr = arr;
            tasks[active].start = idx * CHUNK_SIZE;
            tasks[active].end = (idx + 1) * CHUNK_SIZE;
            if (tasks[active].end > n) tasks[active].end = n;
            
            pthread_create(&threads[active], NULL, sort_thread_fn, &tasks[active]);
            active++;
        }
        
        for (int t = 0; t < active; t++) {
            pthread_join(threads[t], NULL);
        }
    }
    
    return num_chunks;
}

// Hierarchical merge with thread halving
// When num_chunks is odd, the last chunk is left alone (already sorted)
// and gets merged in the next level
static void parallel_merge_recursive(uint32_t *arr, size_t n, size_t num_chunks) {
    size_t chunk_size = CHUNK_SIZE;
    
    while (num_chunks > 1) {
        size_t num_merges = num_chunks / 2;
        
        //Reduce threads to use as needed
        int threads_to_use = (num_merges < NUM_THREADS) ? (int)num_merges : NUM_THREADS;
        
        printf("%3zu chunks -> %3zu chunks | Merges: %3zu | Threads: %d\n", 
               num_chunks, (num_chunks + 1) / 2, num_merges, threads_to_use);
        
        for (size_t batch = 0; batch < num_merges; batch += threads_to_use) {
            pthread_t threads[NUM_THREADS];
            MergeTask tasks[NUM_THREADS];
            uint32_t *tmps[NUM_THREADS];
            int active = 0;
            
            for (int t = 0; t < threads_to_use && (batch + t) < num_merges; t++) {
                size_t merge_idx = batch + t;
                size_t left = merge_idx * chunk_size * 2;
                size_t mid = left + chunk_size;
                size_t right = mid + chunk_size;
                
                // Skip if out of bounds
                if (mid >= n) continue;
                if (right > n) right = n;
                
                // Allocate temp buffer for this merge
                tmps[active] = malloc((right - left) * sizeof(uint32_t));
                tasks[active].arr = arr;
                tasks[active].tmp = tmps[active];
                tasks[active].left = left;
                tasks[active].mid = mid;
                tasks[active].right = right;
                
                pthread_create(&threads[active], NULL, merge_thread_fn, &tasks[active]);
                active++;
            }
            
            for (int t = 0; t < active; t++) {
                pthread_join(threads[t], NULL);
                free(tmps[t]);
            }
        }
        
        chunk_size *= 2;
        num_chunks = (num_chunks + 1) / 2;
    }
}

void sort_array(uint32_t *arr, size_t n) {
    struct timespec start1, end1, start2, end2;

    // Phase 1: Sort chunks
    printf("Starting thread level sort...\n");
    clock_gettime(CLOCK_MONOTONIC, &start1);
    size_t num_chunks = parallel_sort_chunks(arr, n);
    clock_gettime(CLOCK_MONOTONIC, &end1);
    double t1 = elapsed_sec(start1, end1);
    printf("%zu chunks sorted in %.3fs\n\n", num_chunks, t1);
    
    // Phase 2: Merge chunks
    printf("Starting recursive merge...\n");
    clock_gettime(CLOCK_MONOTONIC, &start2);
    parallel_merge_recursive(arr, n, num_chunks);
    clock_gettime(CLOCK_MONOTONIC, &end2);
    double t2 = elapsed_sec(start2, end2);
    printf("Merges completed in %.3fs\n\n", t2);

    printf("Total time: %.3fs\n", t1 + t2);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) return 1;

    printf("Read %lu elements from %s\n\n", size, argv[1]);
    
    sort_array(arr, size);

    if (verify_sortedness(arr, size)) {
        printf("Array sorted successfully!\n");
    } else {
        printf("ERROR: Array not sorted!\n");
        free(arr);
        return 1;
    }

    free(arr);
    return 0;
}
