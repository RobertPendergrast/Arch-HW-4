#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include "utils.h"

// Configuration
#define NUM_THREADS 8
#define CHUNK_8MB   (8 * 1024 * 1024 / sizeof(uint32_t))   // 2M elements
#define CHUNK_64MB  (64 * 1024 * 1024 / sizeof(uint32_t))  // 16M elements

static int g_k_value = 8;  // Default K for k-way merge

// ============================================================================
// Basic Merge Sort (used by each thread)
// ============================================================================

void merge(uint32_t *arr, uint32_t *tmp, size_t left, size_t mid, size_t right) {
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

void merge_sort(uint32_t *arr, uint32_t *tmp, size_t left, size_t right) {
    if (right - left < 2) return;
    
    size_t mid = left + (right - left) / 2;
    merge_sort(arr, tmp, left, mid);
    merge_sort(arr, tmp, mid, right);
    merge(arr, tmp, left, mid, right);
}

// ============================================================================
// Phase 1: Parallel Sort of 8MB Chunks
// ============================================================================

typedef struct {
    uint32_t *arr;
    size_t start;
    size_t end;
} SortTask;

void *sort_chunk_thread(void *arg) {
    SortTask *task = (SortTask *)arg;
    size_t size = task->end - task->start;
    
    uint32_t *tmp = malloc(size * sizeof(uint32_t));
    if (tmp) {
        merge_sort(task->arr, tmp, task->start, task->end);
        free(tmp);
    }
    return NULL;
}

double phase1_parallel_sort(uint32_t *arr, size_t n) {
    size_t num_chunks = (n + CHUNK_8MB - 1) / CHUNK_8MB;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Process chunks in batches of NUM_THREADS
    for (size_t batch = 0; batch < num_chunks; batch += NUM_THREADS) {
        pthread_t threads[NUM_THREADS];
        SortTask tasks[NUM_THREADS];
        int count = 0;
        
        for (int i = 0; i < NUM_THREADS && (batch + i) < num_chunks; i++) {
            size_t idx = batch + i;
            tasks[count].arr = arr;
            tasks[count].start = idx * CHUNK_8MB;
            tasks[count].end = (idx + 1) * CHUNK_8MB;
            if (tasks[count].end > n) tasks[count].end = n;
            
            pthread_create(&threads[count], NULL, sort_chunk_thread, &tasks[count]);
            count++;
        }
        
        for (int i = 0; i < count; i++) {
            pthread_join(threads[i], NULL);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Phase 1: Sorted %zu chunks (8MB each) in %.3fs\n", num_chunks, elapsed);
    return elapsed;
}

// ============================================================================
// Phase 2: Hierarchical Merge (8MB -> 16MB -> 32MB -> 64MB)
// ============================================================================

typedef struct {
    uint32_t *arr;
    uint32_t *tmp;
    size_t left, mid, right;
} MergeTask;

void *merge_chunk_thread(void *arg) {
    MergeTask *task = (MergeTask *)arg;
    merge(task->arr, task->tmp, task->left, task->mid, task->right);
    return NULL;
}

double phase2_hierarchical_merge(uint32_t *arr, size_t n) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Double chunk size each iteration: 8MB -> 16MB -> 32MB -> 64MB
    for (size_t chunk_size = CHUNK_8MB; chunk_size < CHUNK_64MB && chunk_size < n; chunk_size *= 2) {
        size_t num_merges = (n + chunk_size * 2 - 1) / (chunk_size * 2);
        
        for (size_t batch = 0; batch < num_merges; batch += NUM_THREADS) {
            pthread_t threads[NUM_THREADS];
            MergeTask tasks[NUM_THREADS];
            uint32_t *tmps[NUM_THREADS];
            int count = 0;
            
            for (int i = 0; i < NUM_THREADS && (batch + i) < num_merges; i++) {
                size_t left = (batch + i) * chunk_size * 2;
                size_t mid = left + chunk_size;
                size_t right = mid + chunk_size;
                
                if (mid >= n) continue;
                if (right > n) right = n;
                
                tmps[count] = malloc((right - left) * sizeof(uint32_t));
                tasks[count].arr = arr;
                tasks[count].tmp = tmps[count];
                tasks[count].left = left;
                tasks[count].mid = mid;
                tasks[count].right = right;
                
                pthread_create(&threads[count], NULL, merge_chunk_thread, &tasks[count]);
                count++;
            }
            
            for (int i = 0; i < count; i++) {
                pthread_join(threads[i], NULL);
                free(tmps[i]);
            }
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Phase 2: Hierarchical merge to 64MB in %.3fs\n", elapsed);
    return elapsed;
}

// ============================================================================
// Phase 3: K-Way Merge using Min-Heap
// ============================================================================

typedef struct {
    uint32_t value;
    size_t chunk_id;
} HeapNode;

void heap_sift_down(HeapNode *heap, size_t idx, size_t heap_size) {
    while (1) {
        size_t smallest = idx;
        size_t left = 2 * idx + 1;
        size_t right = 2 * idx + 2;
        
        if (left < heap_size && heap[left].value < heap[smallest].value)
            smallest = left;
        if (right < heap_size && heap[right].value < heap[smallest].value)
            smallest = right;
        
        if (smallest == idx) break;
        
        HeapNode tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

double phase3_kway_merge(uint32_t *arr, size_t n, int k) {
    size_t num_chunks = (n + CHUNK_64MB - 1) / CHUNK_64MB;
    
    if (num_chunks <= 1) {
        printf("Phase 3: Single chunk, no merge needed\n");
        return 0;
    }
    
    if ((size_t)k > num_chunks) k = num_chunks;
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Allocate output and tracking arrays
    uint32_t *output = malloc(n * sizeof(uint32_t));
    size_t *positions = malloc(num_chunks * sizeof(size_t));
    size_t *ends = malloc(num_chunks * sizeof(size_t));
    HeapNode *heap = malloc(num_chunks * sizeof(HeapNode));
    
    // Initialize chunk boundaries
    for (size_t i = 0; i < num_chunks; i++) {
        positions[i] = i * CHUNK_64MB;
        ends[i] = (i + 1) * CHUNK_64MB;
        if (ends[i] > n) ends[i] = n;
    }
    
    // Build initial heap with first element from each chunk
    size_t heap_size = 0;
    for (size_t i = 0; i < num_chunks; i++) {
        heap[heap_size].value = arr[positions[i]++];
        heap[heap_size].chunk_id = i;
        heap_size++;
    }
    
    // Heapify
    for (int i = heap_size / 2 - 1; i >= 0; i--) {
        heap_sift_down(heap, i, heap_size);
    }
    
    // Extract minimum and refill from same chunk
    size_t out_idx = 0;
    while (heap_size > 0) {
        output[out_idx++] = heap[0].value;
        size_t chunk_id = heap[0].chunk_id;
        
        if (positions[chunk_id] < ends[chunk_id]) {
            // Refill from same chunk
            heap[0].value = arr[positions[chunk_id]++];
            heap_sift_down(heap, 0, heap_size);
        } else {
            // Chunk exhausted, remove from heap
            heap[0] = heap[--heap_size];
            if (heap_size > 0) {
                heap_sift_down(heap, 0, heap_size);
            }
        }
    }
    
    // Copy result back
    memcpy(arr, output, n * sizeof(uint32_t));
    
    free(output);
    free(positions);
    free(ends);
    free(heap);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Phase 3: %d-way merge of %zu chunks in %.3fs\n", k, num_chunks, elapsed);
    return elapsed;
}

// ============================================================================
// Main Entry Point
// ============================================================================

void sort_array(uint32_t *arr, size_t n) {
    printf("\n=== Three-Phase Parallel Merge Sort ===\n");
    printf("Elements: %zu (%.2f GB), Threads: %d, K: %d\n\n", 
           n, n * 4.0 / 1024 / 1024 / 1024, NUM_THREADS, g_k_value);
    
    double t1 = phase1_parallel_sort(arr, n);
    double t2 = phase2_hierarchical_merge(arr, n);
    double t3 = phase3_kway_merge(arr, n, g_k_value);
    
    double total = t1 + t2 + t3;
    printf("\n=== Total: %.3fs (%.2f M elem/s) ===\n", total, n / total / 1e6);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_file> [output_file] [k_value]\n", argv[0]);
        return 1;
    }

    if (argc >= 4) g_k_value = atoi(argv[3]);
    if (g_k_value < 2) g_k_value = 8;

    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) return 1;

    printf("Read %lu elements from %s\n", size, argv[1]);
    
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
