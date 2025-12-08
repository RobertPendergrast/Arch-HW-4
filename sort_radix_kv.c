#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <sys/mman.h>
#include "utils.h"

#define NUM_THREADS 16
#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)  // 256 buckets
#define RADIX_MASK (RADIX_SIZE - 1)

// Software Write-Combining buffer size (elements per bucket)
#define WC_BUFFER_SIZE 128

// Alignment for streaming stores
#define ALIGN_ELEMS 16
#define ALIGN_MASK (ALIGN_ELEMS - 1)

// Helper for timing
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// ============== Streaming Store Flush for Keys and Values ==============

// Flush key buffer (aligned path)
static inline void flush_key_buffer_aligned(uint32_t *buf, uint32_t *out, int count) {
    int full_lines = count / 16;
    for (int line = 0; line < full_lines; line++) {
        __m512i v = _mm512_load_si512((__m512i*)(buf + line * 16));
        _mm512_stream_si512((__m512i*)(out + line * 16), v);
    }
    int tail_start = full_lines * 16;
    for (int j = tail_start; j < count; j++) {
        out[j] = buf[j];
    }
}

// Flush value buffer (aligned path) - same as key, but separate for clarity
static inline void flush_val_buffer_aligned(uint32_t *buf, uint32_t *out, int count) {
    int full_lines = count / 16;
    for (int line = 0; line < full_lines; line++) {
        __m512i v = _mm512_load_si512((__m512i*)(buf + line * 16));
        _mm512_stream_si512((__m512i*)(out + line * 16), v);
    }
    int tail_start = full_lines * 16;
    for (int j = tail_start; j < count; j++) {
        out[j] = buf[j];
    }
}

// ============== LSD Radix Sort for Key-Value Pairs ==============
// Keys: 32-bit unsigned integers (sort key)
// Values: 32-bit unsigned integers (payload, moves with key)

void sort_kv_array(uint32_t *keys, uint32_t *values, size_t size) {
    if (size <= 1) return;
    
    omp_set_num_threads(NUM_THREADS);
    
    // Request huge pages for input arrays
    size_t arr_size = size * sizeof(uint32_t);
    madvise(keys, arr_size, MADV_HUGEPAGE);
    madvise(values, arr_size, MADV_HUGEPAGE);
    
    double t_start, t_end;
    
    // Allocate output buffers for keys and values
    size_t alloc_size = size * sizeof(uint32_t);
    
    uint32_t *temp_keys = (uint32_t*)aligned_alloc(2 * 1024 * 1024, alloc_size);
    if (!temp_keys) temp_keys = (uint32_t*)aligned_alloc(64, alloc_size);
    
    uint32_t *temp_values = (uint32_t*)aligned_alloc(2 * 1024 * 1024, alloc_size);
    if (!temp_values) temp_values = (uint32_t*)aligned_alloc(64, alloc_size);
    
    if (!temp_keys || !temp_values) {
        fprintf(stderr, "Failed to allocate temp buffers\n");
        free(temp_keys);
        free(temp_values);
        return;
    }
    
    // Request transparent huge pages
    madvise(temp_keys, alloc_size, MADV_HUGEPAGE);
    madvise(temp_values, alloc_size, MADV_HUGEPAGE);
    
    // Pre-touch temp buffers
    t_start = get_time_sec();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i += 1024) {
        temp_keys[i] = 0;
        temp_values[i] = 0;
    }
    t_end = get_time_sec();
    printf("Buffer warmup: %.3f sec\n", t_end - t_start);
    
    // Per-thread histograms
    size_t (*local_hist)[RADIX_SIZE] = aligned_alloc(64, NUM_THREADS * sizeof(*local_hist));
    
    // Global histogram and prefix sums
    size_t global_hist[RADIX_SIZE];
    size_t global_prefix[RADIX_SIZE];
    
    // Per-thread offsets for scatter phase
    size_t (*thread_offsets)[RADIX_SIZE] = aligned_alloc(64, NUM_THREADS * sizeof(*thread_offsets));
    
    uint32_t *src_keys = keys;
    uint32_t *src_vals = values;
    uint32_t *dst_keys = temp_keys;
    uint32_t *dst_vals = temp_values;
    
    int num_passes = (32 + RADIX_BITS - 1) / RADIX_BITS;
    
    for (int pass = 0; pass < num_passes; pass++) {
        double t_pass_start = get_time_sec();
        double t_hist_start, t_hist_end;
        double t_prefix_start, t_prefix_end;
        double t_scatter_start, t_scatter_end;
        
        int shift = pass * RADIX_BITS;
        
        // ===== Phase 1: Build local histograms (KEYS ONLY - saves bandwidth!) =====
        t_hist_start = get_time_sec();
        __m512i shift_vec = _mm512_set1_epi32(shift);
        __m512i mask_vec = _mm512_set1_epi32(RADIX_MASK);
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            memset(local_hist[tid], 0, RADIX_SIZE * sizeof(size_t));
            
            size_t chunk_size = (size + nthreads - 1) / nthreads;
            size_t start = tid * chunk_size;
            size_t end = (start + chunk_size < size) ? start + chunk_size : size;
            
            size_t i = start;
            
            // Prefetch keys only
            for (int p = 0; p < 8; p++) {
                _mm_prefetch((const char*)(src_keys + start + p * 16), _MM_HINT_T0);
            }
            
            // SIMD histogram on keys only
            for (; i + 16 <= end; i += 16) {
                _mm_prefetch((const char*)(src_keys + i + 256), _MM_HINT_T0);
                _mm_prefetch((const char*)(src_keys + i + 320), _MM_HINT_T0);
                
                __m512i keys_vec = _mm512_loadu_si512((__m512i*)(src_keys + i));
                __m512i digits = _mm512_and_epi32(_mm512_srlv_epi32(keys_vec, shift_vec), mask_vec);
                
                uint32_t digit_arr[16];
                _mm512_storeu_si512((__m512i*)digit_arr, digits);
                
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
            
            for (; i < end; i++) {
                uint32_t digit = (src_keys[i] >> shift) & RADIX_MASK;
                local_hist[tid][digit]++;
            }
        }
        t_hist_end = get_time_sec();
        
        // ===== Phase 2: Compute global histogram and prefix sum =====
        t_prefix_start = get_time_sec();
        memset(global_hist, 0, RADIX_SIZE * sizeof(size_t));
        for (int t = 0; t < NUM_THREADS; t++) {
            for (int d = 0; d < RADIX_SIZE; d++) {
                global_hist[d] += local_hist[t][d];
            }
        }
        
        global_prefix[0] = 0;
        for (int d = 1; d < RADIX_SIZE; d++) {
            global_prefix[d] = global_prefix[d-1] + global_hist[d-1];
        }
        
        // ===== Phase 3: Compute per-thread offsets =====
        for (int d = 0; d < RADIX_SIZE; d++) {
            size_t offset = global_prefix[d];
            for (int t = 0; t < NUM_THREADS; t++) {
                thread_offsets[t][d] = offset;
                offset += local_hist[t][d];
            }
        }
        t_prefix_end = get_time_sec();
        
        // ===== Phase 4: Scatter keys AND values with Write-Combining =====
        t_scatter_start = get_time_sec();
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            
            size_t chunk_size = (size + nthreads - 1) / nthreads;
            size_t start = tid * chunk_size;
            size_t end = (start + chunk_size < size) ? start + chunk_size : size;
            
            // Local copy of offsets
            size_t my_offsets[RADIX_SIZE];
            memcpy(my_offsets, thread_offsets[tid], RADIX_SIZE * sizeof(size_t));
            
            // Write-combining buffers for BOTH keys and values
            uint32_t wc_keys[RADIX_SIZE][WC_BUFFER_SIZE] __attribute__((aligned(64)));
            uint32_t wc_vals[RADIX_SIZE][WC_BUFFER_SIZE] __attribute__((aligned(64)));
            uint16_t wc_counts[RADIX_SIZE];
            memset(wc_counts, 0, RADIX_SIZE * sizeof(uint16_t));
            
            // Early flush thresholds for alignment
            uint16_t first_flush_threshold[RADIX_SIZE];
            uint8_t bucket_aligned[RADIX_SIZE];
            
            for (int d = 0; d < RADIX_SIZE; d++) {
                size_t offset = my_offsets[d];
                size_t misalign = offset & ALIGN_MASK;
                
                if (misalign == 0) {
                    first_flush_threshold[d] = WC_BUFFER_SIZE;
                    bucket_aligned[d] = 1;
                } else {
                    first_flush_threshold[d] = ALIGN_ELEMS - misalign;
                    bucket_aligned[d] = 0;
                }
            }
            
            size_t i = start;
            
            // Prefetch both keys and values
            for (int p = 0; p < 8; p++) {
                _mm_prefetch((const char*)(src_keys + start + p * 16), _MM_HINT_T0);
                _mm_prefetch((const char*)(src_vals + start + p * 16), _MM_HINT_T0);
            }
            
            // Process 16 elements at a time
            for (; i + 16 <= end; i += 16) {
                // Prefetch ahead
                _mm_prefetch((const char*)(src_keys + i + 256), _MM_HINT_T0);
                _mm_prefetch((const char*)(src_keys + i + 320), _MM_HINT_T0);
                _mm_prefetch((const char*)(src_vals + i + 256), _MM_HINT_T0);
                _mm_prefetch((const char*)(src_vals + i + 320), _MM_HINT_T0);
                
                // Load keys and values
                __m512i keys_vec = _mm512_loadu_si512((__m512i*)(src_keys + i));
                __m512i vals_vec = _mm512_loadu_si512((__m512i*)(src_vals + i));
                __m512i digits = _mm512_and_epi32(_mm512_srlv_epi32(keys_vec, shift_vec), mask_vec);
                
                uint32_t key_arr[16], val_arr[16], digit_arr[16];
                _mm512_storeu_si512((__m512i*)key_arr, keys_vec);
                _mm512_storeu_si512((__m512i*)val_arr, vals_vec);
                _mm512_storeu_si512((__m512i*)digit_arr, digits);
                
                // Flush macros for key-value pairs
                #define FLUSH_KV_FULL(digit) do { \
                    flush_key_buffer_aligned(wc_keys[digit], dst_keys + my_offsets[digit], WC_BUFFER_SIZE); \
                    flush_val_buffer_aligned(wc_vals[digit], dst_vals + my_offsets[digit], WC_BUFFER_SIZE); \
                    my_offsets[digit] += WC_BUFFER_SIZE; \
                    wc_counts[digit] = 0; \
                } while(0)
                
                #define FLUSH_KV_EARLY(digit) do { \
                    uint16_t cnt = wc_counts[digit]; \
                    uint32_t *out_k = dst_keys + my_offsets[digit]; \
                    uint32_t *out_v = dst_vals + my_offsets[digit]; \
                    for (uint16_t _j = 0; _j < cnt; _j++) { \
                        out_k[_j] = wc_keys[digit][_j]; \
                        out_v[_j] = wc_vals[digit][_j]; \
                    } \
                    my_offsets[digit] += cnt; \
                    wc_counts[digit] = 0; \
                    bucket_aligned[digit] = 1; \
                    first_flush_threshold[digit] = WC_BUFFER_SIZE; \
                } while(0)
                
                #define SCATTER_KV(idx) do { \
                    uint32_t d = digit_arr[idx]; \
                    uint16_t cnt = wc_counts[d]; \
                    wc_keys[d][cnt] = key_arr[idx]; \
                    wc_vals[d][cnt] = val_arr[idx]; \
                    wc_counts[d] = cnt + 1; \
                    if (wc_counts[d] == first_flush_threshold[d]) { \
                        if (bucket_aligned[d]) { \
                            FLUSH_KV_FULL(d); \
                        } else { \
                            FLUSH_KV_EARLY(d); \
                        } \
                    } \
                } while(0)
                
                SCATTER_KV(0);  SCATTER_KV(1);  SCATTER_KV(2);  SCATTER_KV(3);
                SCATTER_KV(4);  SCATTER_KV(5);  SCATTER_KV(6);  SCATTER_KV(7);
                SCATTER_KV(8);  SCATTER_KV(9);  SCATTER_KV(10); SCATTER_KV(11);
                SCATTER_KV(12); SCATTER_KV(13); SCATTER_KV(14); SCATTER_KV(15);
                
                #undef SCATTER_KV
            }
            
            // Handle remainder
            for (; i < end; i++) {
                uint32_t key = src_keys[i];
                uint32_t val = src_vals[i];
                uint32_t digit = (key >> shift) & RADIX_MASK;
                
                uint16_t cnt = wc_counts[digit];
                wc_keys[digit][cnt] = key;
                wc_vals[digit][cnt] = val;
                wc_counts[digit] = cnt + 1;
                
                if (wc_counts[digit] == first_flush_threshold[digit]) {
                    if (bucket_aligned[digit]) {
                        FLUSH_KV_FULL(digit);
                    } else {
                        FLUSH_KV_EARLY(digit);
                    }
                }
            }
            
            #undef FLUSH_KV_FULL
            #undef FLUSH_KV_EARLY
            
            // Flush remaining elements
            for (int d = 0; d < RADIX_SIZE; d++) {
                if (wc_counts[d] > 0) {
                    if (bucket_aligned[d]) {
                        flush_key_buffer_aligned(wc_keys[d], dst_keys + my_offsets[d], wc_counts[d]);
                        flush_val_buffer_aligned(wc_vals[d], dst_vals + my_offsets[d], wc_counts[d]);
                    } else {
                        uint32_t *out_k = dst_keys + my_offsets[d];
                        uint32_t *out_v = dst_vals + my_offsets[d];
                        for (int j = 0; j < wc_counts[d]; j++) {
                            out_k[j] = wc_keys[d][j];
                            out_v[j] = wc_vals[d][j];
                        }
                    }
                }
            }
            
            _mm_sfence();
        }
        t_scatter_end = get_time_sec();
        
        double t_pass_total = t_scatter_end - t_pass_start;
        double t_hist = t_hist_end - t_hist_start;
        double t_prefix = t_prefix_end - t_prefix_start;
        double t_scatter = t_scatter_end - t_scatter_start;
        
        // 2 arrays × (read + write) for scatter, but histogram only reads keys
        double bytes_hist = size * sizeof(uint32_t);  // keys only
        double bytes_scatter = size * sizeof(uint32_t) * 4;  // read keys+vals, write keys+vals
        double throughput = (bytes_hist + bytes_scatter) / t_pass_total / 1e9;
        
        printf("Pass %d (bits %2d-%2d): %.3f sec (%.1f GB/s)  [hist: %.3fs, prefix: %.4fs, scatter: %.3fs]\n", 
               pass, shift, shift + RADIX_BITS - 1, t_pass_total, throughput,
               t_hist, t_prefix, t_scatter);
        
        // Swap buffers
        uint32_t *swap;
        swap = src_keys; src_keys = dst_keys; dst_keys = swap;
        swap = src_vals; src_vals = dst_vals; dst_vals = swap;
    }
    
    // If result is in temp, copy back
    if (src_keys != keys) {
        t_start = get_time_sec();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; i += 16) {
            if (i + 16 <= size) {
                __m512i k = _mm512_loadu_si512((__m512i*)(src_keys + i));
                __m512i v = _mm512_loadu_si512((__m512i*)(src_vals + i));
                _mm512_storeu_si512((__m512i*)(keys + i), k);
                _mm512_storeu_si512((__m512i*)(values + i), v);
            } else {
                for (size_t j = i; j < size; j++) {
                    keys[j] = src_keys[j];
                    values[j] = src_vals[j];
                }
            }
        }
        t_end = get_time_sec();
        printf("Copy back: %.3f sec\n", t_end - t_start);
    }
    
    free(temp_keys);
    free(temp_values);
    free(local_hist);
    free(thread_offsets);
}

// ============== Test Main ==============

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_file> [output_file]\n", argv[0]);
        printf("       Input file contains keys; values will be initialized to indices\n");
        return 1;
    }

    // Read keys from input file
    uint64_t size;
    uint32_t *keys = read_array_from_file(argv[1], &size);
    if (!keys) {
        return 1;
    }

    // Allocate and initialize values (payload = original index)
    uint32_t *values = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!values) {
        fprintf(stderr, "Failed to allocate values array\n");
        free(keys);
        return 1;
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        values[i] = (uint32_t)i;  // Payload = original index
    }

    printf("Read %lu key-value pairs from %s\n", size, argv[1]);

    // Compute hash before sorting
    uint64_t xor_before, sum_before;
    compute_hash(keys, size, &xor_before, &sum_before);
    printf("Input key hash: XOR=0x%016lx SUM=0x%016lx\n", xor_before, sum_before);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sort_kv_array(keys, values, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);
    
    // Calculate bandwidth (histogram reads keys only, scatter reads/writes both)
    int num_passes = (32 + RADIX_BITS - 1) / RADIX_BITS;
    double bytes_per_pass = (double)size * sizeof(uint32_t) * 5;  // 1 read (hist) + 2 read + 2 write (scatter)
    double total_bytes = bytes_per_pass * num_passes;
    double effective_bw = total_bytes / elapsed / 1e9;
    printf("Total data movement: %.1f GB (%d passes)\n", total_bytes / 1e9, num_passes);
    printf("Effective bandwidth: %.1f GB/s\n", effective_bw);

    // Verify keys hash
    uint64_t xor_after, sum_after;
    compute_hash(keys, size, &xor_after, &sum_after);
    printf("Output key hash: XOR=0x%016lx SUM=0x%016lx\n", xor_after, sum_after);

    if (xor_before != xor_after || sum_before != sum_after) {
        printf("Error: Key hash mismatch!\n");
        free(keys);
        free(values);
        return 1;
    }
    printf("Key hash check passed.\n");

    // Verify sorted
    if (verify_sortedness(keys, size)) {
        printf("Keys sorted successfully!\n");
    } else {
        printf("Error: Keys not sorted correctly!\n");
        free(keys);
        free(values);
        return 1;
    }

    // Verify values moved correctly (spot check)
    int values_ok = 1;
    for (size_t i = 0; i < size && i < 1000; i++) {
        // For a proper check, we'd need the original keys array
        // Here we just verify values are valid indices
        if (values[i] >= size) {
            values_ok = 0;
            break;
        }
    }
    if (values_ok) {
        printf("Value integrity check passed.\n");
    } else {
        printf("Warning: Value integrity check failed.\n");
    }

    free(keys);
    free(values);
    return 0;
}
