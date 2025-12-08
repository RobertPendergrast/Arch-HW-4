#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

#define GB (1024ULL * 1024 * 1024)

static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Test 1: Sequential read (measures read bandwidth)
double test_read(uint32_t *arr, size_t size, int nthreads) {
    volatile uint64_t sink = 0;  // Prevent optimization
    
    double start = get_time_sec();
    
    #pragma omp parallel num_threads(nthreads)
    {
        uint64_t local_sum = 0;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += 16) {
            __m512i v = _mm512_load_si512((__m512i*)(arr + i));
            // Reduce to prevent optimization
            local_sum += _mm512_reduce_add_epi64(_mm512_cvtepu32_epi64(_mm512_castsi512_si256(v)));
        }
        #pragma omp atomic
        sink += local_sum;
    }
    
    double elapsed = get_time_sec() - start;
    return (size * sizeof(uint32_t)) / elapsed / GB;
}

// Test 2: Sequential write with regular stores (measures write bandwidth)
double test_write_regular(uint32_t *arr, size_t size, int nthreads) {
    __m512i val = _mm512_set1_epi32(0x12345678);
    
    double start = get_time_sec();
    
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += 16) {
            _mm512_store_si512((__m512i*)(arr + i), val);
        }
    }
    
    double elapsed = get_time_sec() - start;
    return (size * sizeof(uint32_t)) / elapsed / GB;
}

// Test 3: Sequential write with streaming stores (measures streaming write bandwidth)
double test_write_streaming(uint32_t *arr, size_t size, int nthreads) {
    __m512i val = _mm512_set1_epi32(0x12345678);
    
    double start = get_time_sec();
    
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += 16) {
            _mm512_stream_si512((__m512i*)(arr + i), val);
        }
        // Each thread issues sfence
        _mm_sfence();
    }
    
    double elapsed = get_time_sec() - start;
    return (size * sizeof(uint32_t)) / elapsed / GB;
}

// Test 4: Copy (read + write, like radix sort scatter)
double test_copy(uint32_t *dst, uint32_t *src, size_t size, int nthreads) {
    double start = get_time_sec();
    
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += 16) {
            __m512i v = _mm512_load_si512((__m512i*)(src + i));
            _mm512_store_si512((__m512i*)(dst + i), v);
        }
    }
    
    double elapsed = get_time_sec() - start;
    // Read + Write = 2x data movement
    return (size * sizeof(uint32_t) * 2) / elapsed / GB;
}

// Test 5: Copy with streaming stores (best case for radix sort)
double test_copy_streaming(uint32_t *dst, uint32_t *src, size_t size, int nthreads) {
    double start = get_time_sec();
    
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += 16) {
            __m512i v = _mm512_load_si512((__m512i*)(src + i));
            _mm512_stream_si512((__m512i*)(dst + i), v);
        }
        _mm_sfence();
    }
    
    double elapsed = get_time_sec() - start;
    return (size * sizeof(uint32_t) * 2) / elapsed / GB;
}

// Test 6: Random scatter (worst case - simulates radix sort scatter pattern)
double test_random_scatter(uint32_t *dst, uint32_t *src, uint32_t *indices, size_t size, int nthreads) {
    double start = get_time_sec();
    
    #pragma omp parallel num_threads(nthreads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++) {
            dst[indices[i]] = src[i];
        }
    }
    
    double elapsed = get_time_sec() - start;
    return (size * sizeof(uint32_t) * 2) / elapsed / GB;
}

int main(int argc, char *argv[]) {
    size_t size = 256 * 1024 * 1024;  // 256M elements = 1GB
    int nthreads = 8;
    
    if (argc > 1) nthreads = atoi(argv[1]);
    if (argc > 2) size = (size_t)atol(argv[2]) * 1024 * 1024;
    
    printf("Memory Bandwidth Test\n");
    printf("======================\n");
    printf("Array size: %zu elements (%.2f GB)\n", size, (double)(size * 4) / GB);
    printf("Threads: %d\n\n", nthreads);
    
    // Allocate aligned buffers
    uint32_t *arr1 = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    uint32_t *arr2 = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    uint32_t *indices = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    
    if (!arr1 || !arr2 || !indices) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    // Initialize (parallel to distribute across NUMA nodes)
    printf("Initializing arrays...\n");
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (size_t i = 0; i < size; i++) {
        arr1[i] = i;
        arr2[i] = 0;
        // Random-ish scatter indices (simulate radix scatter pattern)
        indices[i] = (i * 2654435761UL) % size;  // Knuth multiplicative hash
    }
    
    printf("\nRunning tests (3 iterations each, best result shown):\n\n");
    
    // Run each test multiple times, report best
    #define ITERATIONS 3
    #define RUN_TEST(name, test_call) do { \
        double best = 0; \
        for (int iter = 0; iter < ITERATIONS; iter++) { \
            double bw = test_call; \
            if (bw > best) best = bw; \
        } \
        printf("%-35s %7.2f GB/s\n", name, best); \
    } while(0)
    
    RUN_TEST("Sequential Read:", test_read(arr1, size, nthreads));
    RUN_TEST("Sequential Write (regular):", test_write_regular(arr1, size, nthreads));
    RUN_TEST("Sequential Write (streaming):", test_write_streaming(arr1, size, nthreads));
    RUN_TEST("Copy (read + regular write):", test_copy(arr2, arr1, size, nthreads));
    RUN_TEST("Copy (read + streaming write):", test_copy_streaming(arr2, arr1, size, nthreads));
    RUN_TEST("Random Scatter (worst case):", test_random_scatter(arr2, arr1, indices, size, nthreads));
    
    printf("\n");
    printf("Interpretation for Radix Sort:\n");
    printf("==============================\n");
    printf("- Each radix pass does: READ input + WRITE output (scattered)\n");
    printf("- Best case: 'Copy (streaming)' bandwidth\n");
    printf("- Worst case: 'Random Scatter' bandwidth\n");
    printf("- Your radix sort should be between these two values\n");
    printf("- If close to 'Copy (streaming)', you're doing great!\n");
    printf("- The WC buffers help move from 'Random Scatter' toward 'Copy'\n");
    
    free(arr1);
    free(arr2);
    free(indices);
    
    return 0;
}
