#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include "utils.h"

#define NUM_THREADS 16
#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)  // 256 buckets
#define RADIX_MASK (RADIX_SIZE - 1)

// Software Write-Combining buffer size (one cache line = 16 uint32_t)
#define WC_BUFFER_SIZE 16

// Helper for timing
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ============== LSD Radix Sort (Stable) ==============
// Processes 8 bits at a time, 4 passes for 32-bit integers
// Each pass is stable, so overall sort is stable

void sort_array(uint32_t *arr, size_t size) {
    if (size <= 1) return;
    
    omp_set_num_threads(NUM_THREADS);
    
    double t_start, t_end;
    
    // Allocate output buffer
    uint32_t *temp = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temp buffer\n");
        return;
    }
    
    // Pre-touch temp buffer
    t_start = get_time_sec();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 1024) {
        temp[i] = 0;
    }
    t_end = get_time_sec();
    printf("Buffer warmup: %.3f sec\n", t_end - t_start);
    
    // Per-thread histograms to avoid false sharing
    // [thread][digit] layout
    size_t (*local_hist)[RADIX_SIZE] = aligned_alloc(64, NUM_THREADS * sizeof(*local_hist));
    
    // Global histogram and prefix sums
    size_t global_hist[RADIX_SIZE];
    size_t global_prefix[RADIX_SIZE];
    
    // Per-thread offsets for scatter phase
    size_t (*thread_offsets)[RADIX_SIZE] = aligned_alloc(64, NUM_THREADS * sizeof(*thread_offsets));
    
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    // Process each byte (4 passes for 32-bit integers)
    for (int pass = 0; pass < 4; pass++) {
        t_start = get_time_sec();
        int shift = pass * RADIX_BITS;
        
        // ===== Phase 1: Build local histograms in parallel =====
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            // Clear local histogram
            memset(local_hist[tid], 0, RADIX_SIZE * sizeof(size_t));
            
            // Each thread processes its static chunk
            size_t chunk_size = (size + nthreads - 1) / nthreads;
            size_t start = tid * chunk_size;
            size_t end = (start + chunk_size < size) ? start + chunk_size : size;
            
            // Count elements with prefetching and unrolling (4x)
            size_t i = start;
            for (; i + 4 <= end; i += 4) {
                // Prefetch ahead
                _mm_prefetch((const char*)(src + i + 64), _MM_HINT_T0);
                
                // Load 4 elements
                uint32_t v0 = src[i];
                uint32_t v1 = src[i + 1];
                uint32_t v2 = src[i + 2];
                uint32_t v3 = src[i + 3];
                
                // Extract digits
                uint32_t d0 = (v0 >> shift) & RADIX_MASK;
                uint32_t d1 = (v1 >> shift) & RADIX_MASK;
                uint32_t d2 = (v2 >> shift) & RADIX_MASK;
                uint32_t d3 = (v3 >> shift) & RADIX_MASK;
                
                // Update histogram
                local_hist[tid][d0]++;
                local_hist[tid][d1]++;
                local_hist[tid][d2]++;
                local_hist[tid][d3]++;
            }
            // Handle remainder
            for (; i < end; i++) {
                uint32_t digit = (src[i] >> shift) & RADIX_MASK;
                local_hist[tid][digit]++;
            }
        }
        
        // ===== Phase 2: Compute global histogram and prefix sum =====
        // Sum all local histograms
        memset(global_hist, 0, RADIX_SIZE * sizeof(size_t));
        for (int t = 0; t < NUM_THREADS; t++) {
            for (int d = 0; d < RADIX_SIZE; d++) {
                global_hist[d] += local_hist[t][d];
            }
        }
        
        // Compute prefix sum (exclusive)
        global_prefix[0] = 0;
        for (int d = 1; d < RADIX_SIZE; d++) {
            global_prefix[d] = global_prefix[d-1] + global_hist[d-1];
        }
        
        // ===== Phase 3: Compute per-thread offsets for stable scatter =====
        // For each digit, thread t's elements go after threads 0..t-1's elements
        for (int d = 0; d < RADIX_SIZE; d++) {
            size_t offset = global_prefix[d];
            for (int t = 0; t < NUM_THREADS; t++) {
                thread_offsets[t][d] = offset;
                offset += local_hist[t][d];
            }
        }
        
        // ===== Phase 4: Scatter elements to output with Write-Combining =====
        // Software write-combining: buffer writes locally, flush as cache lines
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            // Each thread processes its chunk in order (for stability)
            size_t chunk_size = (size + nthreads - 1) / nthreads;
            size_t start = tid * chunk_size;
            size_t end = (start + chunk_size < size) ? start + chunk_size : size;
            
            // Local copy of offsets (will be modified)
            size_t my_offsets[RADIX_SIZE];
            memcpy(my_offsets, thread_offsets[tid], RADIX_SIZE * sizeof(size_t));
            
            // Software write-combining buffers (one per bucket)
            // Aligned to cache line for efficient flushing
            uint32_t wc_buffers[RADIX_SIZE][WC_BUFFER_SIZE] __attribute__((aligned(64)));
            uint8_t wc_counts[RADIX_SIZE];
            memset(wc_counts, 0, RADIX_SIZE);
            
            // Scatter this thread's elements with write-combining
            for (size_t i = start; i < end; i++) {
                // Prefetch input ahead
                if ((i & 15) == 0) {
                    _mm_prefetch((const char*)(src + i + 64), _MM_HINT_T0);
                }
                
                uint32_t val = src[i];
                uint32_t digit = (val >> shift) & RADIX_MASK;
                
                // Add to write-combine buffer
                wc_buffers[digit][wc_counts[digit]++] = val;
                
                // Flush buffer when full (16 elements = 1 cache line)
                if (wc_counts[digit] == WC_BUFFER_SIZE) {
                    // Use SIMD store for full cache line
                    __m512i vec = _mm512_loadu_si512((__m512i*)wc_buffers[digit]);
                    _mm512_storeu_si512((__m512i*)(dst + my_offsets[digit]), vec);
                    my_offsets[digit] += WC_BUFFER_SIZE;
                    wc_counts[digit] = 0;
                }
            }
            
            // Flush remaining elements in buffers
            for (int d = 0; d < RADIX_SIZE; d++) {
                if (wc_counts[d] > 0) {
                    memcpy(dst + my_offsets[d], wc_buffers[d], wc_counts[d] * sizeof(uint32_t));
                }
            }
        }
        
        t_end = get_time_sec();
        double throughput = (size * sizeof(uint32_t) * 2) / (t_end - t_start) / 1e9;
        printf("Pass %d (bits %2d-%2d): %.3f sec (%.1f GB/s)\n", 
               pass, shift, shift + RADIX_BITS - 1, t_end - t_start, throughput);
        
        // Swap buffers
        uint32_t *swap = src;
        src = dst;
        dst = swap;
    }
    
    // If result is in temp, copy back
    if (src != arr) {
        t_start = get_time_sec();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; i++) {
            arr[i] = src[i];
        }
        t_end = get_time_sec();
        printf("Copy back: %.3f sec\n", t_end - t_start);
    }
    
    free(temp);
    free(local_hist);
    free(thread_offsets);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    // Read array from input file
    uint64_t size;
    uint32_t *arr = read_array_from_file(argv[1], &size);
    if (!arr) {
        return 1;
    }

    printf("Read %lu elements from %s\n", size, argv[1]);

    // Compute hash before sorting (order-independent)
    uint64_t xor_before, sum_before;
    compute_hash(arr, size, &xor_before, &sum_before);
    printf("Input hash: XOR=0x%016lx SUM=0x%016lx\n", xor_before, sum_before);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sort_array(arr, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);

    // Compute hash after sorting
    uint64_t xor_after, sum_after;
    compute_hash(arr, size, &xor_after, &sum_after);
    printf("Output hash: XOR=0x%016lx SUM=0x%016lx\n", xor_after, sum_after);

    // Verify hashes match (same elements, just reordered)
    if (xor_before != xor_after || sum_before != sum_after) {
        printf("Error: Hash mismatch! Elements were lost or corrupted during sorting.\n");
        free(arr);
        return 1;
    }
    printf("Hash check passed: all elements preserved.\n");

    // Verify the array is sorted
    if (verify_sortedness(arr, size)) {
        printf("Array sorted successfully!\n");
    } else {
        printf("Error: Array is not sorted correctly!\n");
        free(arr);
        return 1;
    }

    free(arr);
    return 0;
}
