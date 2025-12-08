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

// Software Write-Combining buffer size
// Larger buffers = fewer flushes, better write coalescing
// 256 elements = 1KB = 16 cache lines
#define WC_BUFFER_SIZE 256

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
        
        // ===== Phase 1: Build local histograms in parallel (SIMD) =====
        __m512i shift_vec = _mm512_set1_epi32(shift);
        __m512i mask_vec = _mm512_set1_epi32(RADIX_MASK);
        
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
            
            // SIMD histogram: process 16 elements at a time
            size_t i = start;
            for (; i + 16 <= end; i += 16) {
                // Prefetch ahead
                _mm_prefetch((const char*)(src + i + 128), _MM_HINT_T0);
                
                // Load 16 elements with SIMD
                __m512i vals = _mm512_loadu_si512((__m512i*)(src + i));
                
                // Extract 16 digits: (vals >> shift) & mask
                __m512i digits = _mm512_and_epi32(_mm512_srlv_epi32(vals, shift_vec), mask_vec);
                
                // Store digits to temp array and update histogram
                // (Can't easily vectorize histogram update due to conflicts)
                uint32_t digit_arr[16];
                _mm512_storeu_si512((__m512i*)digit_arr, digits);
                
                // Unrolled histogram update
                local_hist[tid][digit_arr[0]]++;
                local_hist[tid][digit_arr[1]]++;
                local_hist[tid][digit_arr[2]]++;
                local_hist[tid][digit_arr[3]]++;
                local_hist[tid][digit_arr[4]]++;
                local_hist[tid][digit_arr[5]]++;
                local_hist[tid][digit_arr[6]]++;
                local_hist[tid][digit_arr[7]]++;
                local_hist[tid][digit_arr[8]]++;
                local_hist[tid][digit_arr[9]]++;
                local_hist[tid][digit_arr[10]]++;
                local_hist[tid][digit_arr[11]]++;
                local_hist[tid][digit_arr[12]]++;
                local_hist[tid][digit_arr[13]]++;
                local_hist[tid][digit_arr[14]]++;
                local_hist[tid][digit_arr[15]]++;
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
            uint16_t wc_counts[RADIX_SIZE];  // uint16_t for larger buffer sizes
            memset(wc_counts, 0, RADIX_SIZE * sizeof(uint16_t));
            
            // Scatter with SIMD loading + write-combining
            size_t i = start;
            
            // Process 16 elements at a time with SIMD
            for (; i + 16 <= end; i += 16) {
                _mm_prefetch((const char*)(src + i + 128), _MM_HINT_T0);
                
                // SIMD load 16 values and extract digits
                __m512i vals = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i digits = _mm512_and_epi32(_mm512_srlv_epi32(vals, shift_vec), mask_vec);
                
                // Store to temp arrays for scatter
                uint32_t val_arr[16], digit_arr[16];
                _mm512_storeu_si512((__m512i*)val_arr, vals);
                _mm512_storeu_si512((__m512i*)digit_arr, digits);
                
                // Scatter each element to its bucket (unrolled)
                #define FLUSH_BUFFER(digit) do { \
                    /* Flush 16 cache lines (1KB) with SIMD - unrolled for speed */ \
                    uint32_t *buf = wc_buffers[digit]; \
                    uint32_t *out = dst + my_offsets[digit]; \
                    for (int _c = 0; _c < WC_BUFFER_SIZE; _c += 64) { \
                        __m512i v0 = _mm512_load_si512((__m512i*)(buf + _c + 0)); \
                        __m512i v1 = _mm512_load_si512((__m512i*)(buf + _c + 16)); \
                        __m512i v2 = _mm512_load_si512((__m512i*)(buf + _c + 32)); \
                        __m512i v3 = _mm512_load_si512((__m512i*)(buf + _c + 48)); \
                        _mm512_storeu_si512((__m512i*)(out + _c + 0), v0); \
                        _mm512_storeu_si512((__m512i*)(out + _c + 16), v1); \
                        _mm512_storeu_si512((__m512i*)(out + _c + 32), v2); \
                        _mm512_storeu_si512((__m512i*)(out + _c + 48), v3); \
                    } \
                    my_offsets[digit] += WC_BUFFER_SIZE; \
                    wc_counts[digit] = 0; \
                } while(0)
                
                #define SCATTER_ONE(idx) do { \
                    uint32_t d = digit_arr[idx]; \
                    wc_buffers[d][wc_counts[d]++] = val_arr[idx]; \
                    if (wc_counts[d] == WC_BUFFER_SIZE) { \
                        FLUSH_BUFFER(d); \
                    } \
                } while(0)
                
                SCATTER_ONE(0);  SCATTER_ONE(1);  SCATTER_ONE(2);  SCATTER_ONE(3);
                SCATTER_ONE(4);  SCATTER_ONE(5);  SCATTER_ONE(6);  SCATTER_ONE(7);
                SCATTER_ONE(8);  SCATTER_ONE(9);  SCATTER_ONE(10); SCATTER_ONE(11);
                SCATTER_ONE(12); SCATTER_ONE(13); SCATTER_ONE(14); SCATTER_ONE(15);
                
                #undef SCATTER_ONE
            }
            
            // Handle remainder
            for (; i < end; i++) {
                uint32_t val = src[i];
                uint32_t digit = (val >> shift) & RADIX_MASK;
                
                wc_buffers[digit][wc_counts[digit]++] = val;
                
                if (wc_counts[digit] == WC_BUFFER_SIZE) {
                    FLUSH_BUFFER(digit);
                }
            }
            
            #undef FLUSH_BUFFER
            
            // Flush remaining elements in buffers (use SIMD for full cache lines)
            for (int d = 0; d < RADIX_SIZE; d++) {
                if (wc_counts[d] > 0) {
                    uint32_t *buf = wc_buffers[d];
                    uint32_t *out = dst + my_offsets[d];
                    uint16_t cnt = wc_counts[d];
                    uint16_t j = 0;
                    
                    // SIMD copy for full 16-element chunks
                    for (; j + 16 <= cnt; j += 16) {
                        __m512i v = _mm512_load_si512((__m512i*)(buf + j));
                        _mm512_storeu_si512((__m512i*)(out + j), v);
                    }
                    // Scalar copy for remainder
                    for (; j < cnt; j++) {
                        out[j] = buf[j];
                    }
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
    
    // If result is in temp, copy back with SIMD
    if (src != arr) {
        t_start = get_time_sec();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; i += 16) {
            if (i + 16 <= size) {
                __m512i vec = _mm512_loadu_si512((__m512i*)(src + i));
                _mm512_storeu_si512((__m512i*)(arr + i), vec);
            } else {
                // Handle remainder
                for (size_t j = i; j < size; j++) {
                    arr[j] = src[j];
                }
            }
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
