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
// 256 elements = 1024 bytes = 16 cache lines (tuned for large L2)
#define WC_BUFFER_SIZE 256

// Alignment requirement: 16 elements = 64 bytes = 1 cache line
#define ALIGN_ELEMS 16
#define ALIGN_MASK (ALIGN_ELEMS - 1)

// ============== Streaming Store Flush ==============
// Two versions: fast aligned path (no checks) and general path (handles any alignment)

// FAST PATH: Output is 64-byte aligned, buffer is 64-byte aligned
// No alignment checks, pure streaming stores
static inline void flush_buffer_aligned(uint32_t *buf, uint32_t *out, int count) {
    int full_lines = count / 16;
    
    // Stream full cache lines (16 elements = 64 bytes each)
    for (int line = 0; line < full_lines; line++) {
        __m512i v = _mm512_load_si512((__m512i*)(buf + line * 16));
        _mm512_stream_si512((__m512i*)(out + line * 16), v);
    }
    
    // Handle tail (< 16 elements) with scalar stores
    int tail_start = full_lines * 16;
    for (int j = tail_start; j < count; j++) {
        out[j] = buf[j];
    }
}


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
        double t_pass_start = get_time_sec();
        double t_hist_start, t_hist_end;
        double t_prefix_start, t_prefix_end;
        double t_scatter_start, t_scatter_end;
        
        int shift = pass * RADIX_BITS;
        
        // ===== Phase 1: Build local histograms in parallel (SIMD) =====
        t_hist_start = get_time_sec();
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
            
            // Prefetch first few cache lines
            for (int p = 0; p < 8; p++) {
                _mm_prefetch((const char*)(src + start + p * 16), _MM_HINT_T0);
            }
            
            for (; i + 16 <= end; i += 16) {
                // Prefetch further ahead (512 bytes = 8 cache lines ahead)
                _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
                _mm_prefetch((const char*)(src + i + 320), _MM_HINT_T0);
                
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
        t_hist_end = get_time_sec();
        
        // ===== Phase 2: Compute global histogram and prefix sum =====
        t_prefix_start = get_time_sec();
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
        t_prefix_end = get_time_sec();
        
        // ===== Phase 4: Scatter elements to output with Write-Combining =====
        t_scatter_start = get_time_sec();
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
            uint16_t wc_counts[RADIX_SIZE];
            memset(wc_counts, 0, RADIX_SIZE * sizeof(uint16_t));
            
            // EARLY FLUSH STRATEGY: For unaligned buckets, do a small early flush to align
            // After early flush, offset becomes aligned and all subsequent flushes use fast path
            //
            // first_flush_threshold[d] = elements to accumulate before first flush
            //   - If already aligned: WC_BUFFER_SIZE (normal full flush)
            //   - If unaligned: (16 - offset%16) to reach alignment, then future flushes are aligned
            // bucket_aligned[d] = 1 after we've done the aligning flush
            
            uint16_t first_flush_threshold[RADIX_SIZE];
            uint8_t bucket_aligned[RADIX_SIZE];
            
            for (int d = 0; d < RADIX_SIZE; d++) {
                size_t offset = my_offsets[d];
                size_t misalign = offset & ALIGN_MASK;  // offset % 16
                
                if (misalign == 0) {
                    // Already aligned - use full buffer size, mark as aligned
                    first_flush_threshold[d] = WC_BUFFER_SIZE;
                    bucket_aligned[d] = 1;
                } else {
                    // Unaligned - flush early to align (1-15 elements)
                    first_flush_threshold[d] = ALIGN_ELEMS - misalign;
                    bucket_aligned[d] = 0;
                }
            }
            
            // Scatter with SIMD loading + write-combining
            size_t i = start;
            
            // Prefetch first cache lines
            for (int p = 0; p < 8; p++) {
                _mm_prefetch((const char*)(src + start + p * 16), _MM_HINT_T0);
            }
            
            // Process 16 elements at a time with SIMD
            for (; i + 16 <= end; i += 16) {
                // Prefetch further ahead
                _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
                _mm_prefetch((const char*)(src + i + 320), _MM_HINT_T0);
                
                // SIMD load 16 values and extract digits
                __m512i vals = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i digits = _mm512_and_epi32(_mm512_srlv_epi32(vals, shift_vec), mask_vec);
                
                // Store to temp arrays for scatter
                uint32_t val_arr[16], digit_arr[16];
                _mm512_storeu_si512((__m512i*)val_arr, vals);
                _mm512_storeu_si512((__m512i*)digit_arr, digits);
                
                // Scatter each element to its bucket (unrolled)
                // FLUSH_BUFFER: uses fast aligned path (bucket is always aligned after first flush)
                #define FLUSH_BUFFER_FULL(digit) do { \
                    flush_buffer_aligned(wc_buffers[digit], dst + my_offsets[digit], WC_BUFFER_SIZE); \
                    my_offsets[digit] += WC_BUFFER_SIZE; \
                    wc_counts[digit] = 0; \
                } while(0)
                
                // Early flush for alignment: small flush (1-15 elements) with scalar stores
                #define FLUSH_BUFFER_EARLY(digit) do { \
                    uint16_t cnt = wc_counts[digit]; \
                    uint32_t *out = dst + my_offsets[digit]; \
                    for (uint16_t _j = 0; _j < cnt; _j++) { \
                        out[_j] = wc_buffers[digit][_j]; \
                    } \
                    my_offsets[digit] += cnt; \
                    wc_counts[digit] = 0; \
                    bucket_aligned[digit] = 1; /* Now aligned for fast path! */ \
                    first_flush_threshold[digit] = WC_BUFFER_SIZE; \
                } while(0)
                
                #define SCATTER_ONE(idx) do { \
                    uint32_t d = digit_arr[idx]; \
                    wc_buffers[d][wc_counts[d]++] = val_arr[idx]; \
                    if (wc_counts[d] == first_flush_threshold[d]) { \
                        if (bucket_aligned[d]) { \
                            FLUSH_BUFFER_FULL(d); \
                        } else { \
                            FLUSH_BUFFER_EARLY(d); \
                        } \
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
                
                // Check if we hit the flush threshold (early or full)
                if (wc_counts[digit] == first_flush_threshold[digit]) {
                    if (bucket_aligned[digit]) {
                        FLUSH_BUFFER_FULL(digit);
                    } else {
                        FLUSH_BUFFER_EARLY(digit);
                    }
                }
            }
            
            #undef FLUSH_BUFFER_FULL
            #undef FLUSH_BUFFER_EARLY
            
            // Flush remaining elements in buffers
            // By now all buckets with >= early_threshold elements are aligned
            for (int d = 0; d < RADIX_SIZE; d++) {
                if (wc_counts[d] > 0) {
                    // If bucket is aligned, use fast path; else scalar (small remainder anyway)
                    if (bucket_aligned[d]) {
                        flush_buffer_aligned(wc_buffers[d], dst + my_offsets[d], wc_counts[d]);
                    } else {
                        // Never got enough elements to do early flush - just scalar copy
                        uint32_t *out = dst + my_offsets[d];
                        for (int j = 0; j < wc_counts[d]; j++) {
                            out[j] = wc_buffers[d][j];
                        }
                    }
                }
            }
            
            // IMPORTANT: Store fence ensures all streaming stores are globally visible
            // before we swap buffers and start reading from dst in the next pass
            _mm_sfence();
        }
        t_scatter_end = get_time_sec();
        
        double t_pass_total = t_scatter_end - t_pass_start;
        double t_hist = t_hist_end - t_hist_start;
        double t_prefix = t_prefix_end - t_prefix_start;
        double t_scatter = t_scatter_end - t_scatter_start;
        
        double throughput = (size * sizeof(uint32_t) * 2) / t_pass_total / 1e9;
        printf("Pass %d (bits %2d-%2d): %.3f sec (%.1f GB/s)  [hist: %.3fs, prefix: %.4fs, scatter: %.3fs]\n", 
               pass, shift, shift + RADIX_BITS - 1, t_pass_total, throughput,
               t_hist, t_prefix, t_scatter);
        
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
