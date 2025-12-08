/*
 * Benchmark wrapper for stable_sort_avx512.c
 * 
 * Tests the stable AVX-512 merge sort against large datasets.
 * Includes hash verification, timing, and multiple variant testing.
 * 
 * Usage: ./bench_stable_sort <input_file> [output_file] [--parallel N] [--iterations N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include "utils.h"
#include "stable_sort_avx512.h"

// Default number of threads for parallel version
#define DEFAULT_NUM_THREADS 16

// Helper for timing
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Order-independent hash using XOR and sum (parallel)
static void compute_hash(uint32_t *arr, size_t size, uint64_t *xor_out, uint64_t *sum_out) {
    uint64_t xor_hash = 0;
    uint64_t sum_hash = 0;
    
    #pragma omp parallel reduction(^:xor_hash) reduction(+:sum_hash)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++) {
            xor_hash ^= arr[i];
            sum_hash += arr[i];
        }
    }
    
    *xor_out = xor_hash;
    *sum_out = sum_hash;
}

// Copy array for repeated testing
static uint32_t* copy_array(const uint32_t *src, size_t size) {
    uint32_t *dst = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!dst) {
        dst = (uint32_t*)malloc(size * sizeof(uint32_t));
    }
    if (dst) {
        memcpy(dst, src, size * sizeof(uint32_t));
    }
    return dst;
}

// Print throughput stats
static void print_stats(const char *name, double elapsed, size_t size) {
    double throughput = (size * sizeof(uint32_t)) / elapsed / 1e9;  // GB/s
    double elements_per_sec = size / elapsed / 1e6;  // M elements/s
    printf("  %-30s: %.3f sec (%.2f GB/s, %.2f M elem/s)\n", 
           name, elapsed, throughput, elements_per_sec);
}

// Run a single benchmark
typedef void (*sort_func_t)(uint32_t*, size_t);
typedef void (*sort_func_buf_t)(uint32_t*, size_t, uint32_t*);
typedef void (*sort_func_parallel_t)(uint32_t*, size_t, int);

static double benchmark_sort(const uint32_t *original, size_t size, 
                             sort_func_t sort_fn, const char *name,
                             uint64_t expected_xor, uint64_t expected_sum,
                             int verify) {
    uint32_t *arr = copy_array(original, size);
    if (!arr) {
        fprintf(stderr, "Failed to allocate memory for benchmark\n");
        return -1.0;
    }
    
    double t_start = get_time_sec();
    sort_fn(arr, size);
    double t_end = get_time_sec();
    double elapsed = t_end - t_start;
    
    print_stats(name, elapsed, size);
    
    if (verify) {
        // Verify hash
        uint64_t xor_after, sum_after;
        compute_hash(arr, size, &xor_after, &sum_after);
        if (xor_after != expected_xor || sum_after != expected_sum) {
            printf("    ERROR: Hash mismatch! Elements were lost or corrupted.\n");
            free(arr);
            return -1.0;
        }
        
        // Verify sorted
        if (!verify_sortedness(arr, size)) {
            printf("    ERROR: Array is not sorted correctly!\n");
            free(arr);
            return -1.0;
        }
        printf("    Verified: hash match, array sorted\n");
    }
    
    free(arr);
    return elapsed;
}

static double benchmark_sort_with_buffer(const uint32_t *original, size_t size, 
                                         sort_func_buf_t sort_fn, const char *name,
                                         uint64_t expected_xor, uint64_t expected_sum,
                                         int verify) {
    uint32_t *arr = copy_array(original, size);
    uint32_t *temp = (uint32_t*)aligned_alloc(64, size * sizeof(uint32_t));
    if (!arr || !temp) {
        fprintf(stderr, "Failed to allocate memory for benchmark\n");
        free(arr);
        free(temp);
        return -1.0;
    }
    
    double t_start = get_time_sec();
    sort_fn(arr, size, temp);
    double t_end = get_time_sec();
    double elapsed = t_end - t_start;
    
    print_stats(name, elapsed, size);
    
    if (verify) {
        uint64_t xor_after, sum_after;
        compute_hash(arr, size, &xor_after, &sum_after);
        if (xor_after != expected_xor || sum_after != expected_sum) {
            printf("    ERROR: Hash mismatch! Elements were lost or corrupted.\n");
            free(arr);
            free(temp);
            return -1.0;
        }
        
        if (!verify_sortedness(arr, size)) {
            printf("    ERROR: Array is not sorted correctly!\n");
            free(arr);
            free(temp);
            return -1.0;
        }
        printf("    Verified: hash match, array sorted\n");
    }
    
    free(arr);
    free(temp);
    return elapsed;
}

#ifdef _OPENMP
static double benchmark_sort_parallel(const uint32_t *original, size_t size, 
                                      sort_func_parallel_t sort_fn, const char *name,
                                      uint64_t expected_xor, uint64_t expected_sum,
                                      int num_threads, int verify) {
    uint32_t *arr = copy_array(original, size);
    if (!arr) {
        fprintf(stderr, "Failed to allocate memory for benchmark\n");
        return -1.0;
    }
    
    double t_start = get_time_sec();
    sort_fn(arr, size, num_threads);
    double t_end = get_time_sec();
    double elapsed = t_end - t_start;
    
    char full_name[64];
    snprintf(full_name, sizeof(full_name), "%s (%d threads)", name, num_threads);
    print_stats(full_name, elapsed, size);
    
    if (verify) {
        uint64_t xor_after, sum_after;
        compute_hash(arr, size, &xor_after, &sum_after);
        if (xor_after != expected_xor || sum_after != expected_sum) {
            printf("    ERROR: Hash mismatch! Elements were lost or corrupted.\n");
            free(arr);
            return -1.0;
        }
        
        if (!verify_sortedness(arr, size)) {
            printf("    ERROR: Array is not sorted correctly!\n");
            free(arr);
            return -1.0;
        }
        printf("    Verified: hash match, array sorted\n");
    }
    
    free(arr);
    return elapsed;
}
#endif

static void print_usage(const char *prog) {
    printf("Usage: %s <input_file> [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -o, --output <file>     Write sorted output to file\n");
    printf("  -t, --threads <N>       Number of threads for parallel version (default: %d)\n", DEFAULT_NUM_THREADS);
    printf("  -i, --iterations <N>    Number of iterations for averaging (default: 1)\n");
    printf("  -v, --verify            Verify correctness (default: on first iteration)\n");
    printf("  -a, --all               Run all variants (single, buffer, parallel)\n");
    printf("  -s, --single            Run only single-threaded version\n");
    printf("  -p, --parallel-only     Run only parallel version\n");
    printf("  -h, --help              Show this help\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Parse arguments
    const char *input_file = NULL;
    const char *output_file = NULL;
    int num_threads = DEFAULT_NUM_THREADS;
    int iterations = 1;
    int run_all = 1;       // Default: run all variants
    int run_single = 0;
    int run_parallel_only = 0;
    int always_verify = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing output file\n"); return 1; }
            output_file = argv[i];
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing thread count\n"); return 1; }
            num_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (++i >= argc) { fprintf(stderr, "Missing iteration count\n"); return 1; }
            iterations = atoi(argv[i]);
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verify") == 0) {
            always_verify = 1;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--all") == 0) {
            run_all = 1;
            run_single = 0;
            run_parallel_only = 0;
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--single") == 0) {
            run_all = 0;
            run_single = 1;
            run_parallel_only = 0;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--parallel-only") == 0) {
            run_all = 0;
            run_single = 0;
            run_parallel_only = 1;
        } else if (argv[i][0] != '-') {
            if (!input_file) {
                input_file = argv[i];
            } else if (!output_file) {
                output_file = argv[i];
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!input_file) {
        fprintf(stderr, "Error: No input file specified\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Set OpenMP threads
    omp_set_num_threads(num_threads);
    
    // Read array from input file
    uint64_t size;
    uint32_t *original = read_array_from_file(input_file, &size);
    if (!original) {
        fprintf(stderr, "Failed to read input file: %s\n", input_file);
        return 1;
    }
    
    printf("=== Stable Sort AVX-512 Benchmark ===\n");
    printf("Input file: %s\n", input_file);
    printf("Array size: %lu elements (%.2f MB)\n", size, (size * sizeof(uint32_t)) / 1e6);
    printf("Iterations: %d\n", iterations);
    printf("Threads: %d\n", num_threads);
    printf("\n");
    
    // Compute hash before sorting (order-independent)
    uint64_t xor_before, sum_before;
    compute_hash(original, size, &xor_before, &sum_before);
    printf("Input hash: XOR=0x%016lx SUM=0x%016lx\n\n", xor_before, sum_before);
    
    // Results storage
    double time_single = 0, time_buffer = 0, time_parallel = 0;
    int errors = 0;
    
    // Run benchmarks
    for (int iter = 0; iter < iterations; iter++) {
        if (iterations > 1) {
            printf("--- Iteration %d/%d ---\n", iter + 1, iterations);
        }
        
        int verify = (iter == 0 || always_verify);
        
        if (run_all || run_single) {
            // Test single-threaded version
            double t = benchmark_sort(original, size, 
                                      stable_merge_sort_avx512,
                                      "stable_merge_sort_avx512",
                                      xor_before, sum_before, verify);
            if (t < 0) errors++;
            else time_single += t;
            
            // Test with pre-allocated buffer
            t = benchmark_sort_with_buffer(original, size,
                                           stable_merge_sort_avx512_with_buffer,
                                           "stable_sort (with buffer)",
                                           xor_before, sum_before, verify);
            if (t < 0) errors++;
            else time_buffer += t;
        }
        
#ifdef _OPENMP
        if (run_all || run_parallel_only) {
            // Test parallel version
            double t = benchmark_sort_parallel(original, size,
                                               stable_merge_sort_avx512_parallel,
                                               "stable_sort_parallel",
                                               xor_before, sum_before,
                                               num_threads, verify);
            if (t < 0) errors++;
            else time_parallel += t;
        }
#endif
        
        if (iterations > 1) printf("\n");
    }
    
    // Print summary
    if (iterations > 1) {
        printf("=== Summary (averaged over %d iterations) ===\n", iterations);
        if (run_all || run_single) {
            print_stats("stable_merge_sort_avx512", time_single / iterations, size);
            print_stats("stable_sort (with buffer)", time_buffer / iterations, size);
        }
#ifdef _OPENMP
        if (run_all || run_parallel_only) {
            char name[64];
            snprintf(name, sizeof(name), "stable_sort_parallel (%d threads)", num_threads);
            print_stats(name, time_parallel / iterations, size);
        }
#endif
    }
    
    // Write output if requested
    if (output_file && errors == 0) {
        printf("\nSorting final copy for output...\n");
        uint32_t *sorted = copy_array(original, size);
        if (sorted) {
#ifdef _OPENMP
            stable_merge_sort_avx512_parallel(sorted, size, num_threads);
#else
            stable_merge_sort_avx512(sorted, size);
#endif
            if (write_array_to_file(output_file, sorted, size) == 0) {
                printf("Sorted array written to: %s\n", output_file);
            } else {
                fprintf(stderr, "Failed to write output file: %s\n", output_file);
                errors++;
            }
            free(sorted);
        }
    }
    
    free(original);
    
    if (errors > 0) {
        printf("\n*** %d ERROR(S) DETECTED ***\n", errors);
        return 1;
    }
    
    printf("\nAll tests passed!\n");
    return 0;
}
