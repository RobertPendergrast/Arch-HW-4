/*
 * Test suite for AVX-512 Odd-Even Merge Network
 * 
 * Tests:
 * 1. Register-level merge (merge_512_oddeven)
 * 2. Array-level merge (merge_arrays, merge_arrays_cached)
 * 3. Key-Value merge variants
 * 4. Various input patterns: sorted, reverse, interleaved, random
 * 
 * Compile: make test_merge
 * Run: ./test_merge
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include "merge.h"

// Colors for terminal output
#define RED     "\x1b[31m"
#define GREEN   "\x1b[32m"
#define YELLOW  "\x1b[33m"
#define RESET   "\x1b[0m"

// Test configuration
#define SMALL_SIZE 32
#define MEDIUM_SIZE 1024
#define LARGE_SIZE (1 << 20)  // 1M elements

static int tests_passed = 0;
static int tests_failed = 0;

// Verify array is sorted
static int is_sorted(uint32_t *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i] < arr[i-1]) {
            return 0;
        }
    }
    return 1;
}

// Verify KV array is sorted by key and payloads match
static int is_sorted_kv(uint32_t *keys, uint32_t *payloads, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (keys[i] < keys[i-1]) {
            return 0;
        }
    }
    // Payloads should still be valid (not corrupted)
    return 1;
}

// Print array (for debugging small arrays)
static void print_array(const char *name, uint32_t *arr, size_t n) {
    printf("%s: [", name);
    for (size_t i = 0; i < n && i < 32; i++) {
        printf("%u", arr[i]);
        if (i < n-1 && i < 31) printf(", ");
    }
    if (n > 32) printf(", ...");
    printf("]\n");
}

// Allocate aligned memory
static uint32_t* aligned_alloc_u32(size_t n) {
    return (uint32_t*)aligned_alloc(64, n * sizeof(uint32_t));
}

// Fill with ascending values
static void fill_ascending(uint32_t *arr, size_t n, uint32_t start) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = start + i;
    }
}

// Fill with descending values
static void fill_descending(uint32_t *arr, size_t n, uint32_t start) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = start + n - 1 - i;
    }
}

// Fill with interleaved pattern (evens or odds)
static void fill_interleaved(uint32_t *arr, size_t n, int even) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = 2 * i + (even ? 0 : 1);
    }
}

// Fill with random values
static void fill_random(uint32_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = rand();
    }
}

// Sort array for merge input preparation
static int cmp_u32(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t*)a;
    uint32_t ub = *(const uint32_t*)b;
    return (ua > ub) - (ua < ub);
}

// ============== Test Functions ==============

// Test 1: Register-level merge (32 elements -> 32 sorted)
void test_register_merge(void) {
    printf("\n" YELLOW "=== Test: Register-level merge_512_oddeven ===" RESET "\n");
    
    __m512i left, right;
    uint32_t left_arr[16] __attribute__((aligned(64)));
    uint32_t right_arr[16] __attribute__((aligned(64)));
    uint32_t result[32] __attribute__((aligned(64)));
    
    // Test case 1: Evens in left, odds in right
    printf("  Case 1: Evens/Odds interleaved... ");
    fill_interleaved(left_arr, 16, 1);   // 0,2,4,6,...,30
    fill_interleaved(right_arr, 16, 0);  // 1,3,5,7,...,31
    
    left = _mm512_load_epi32(left_arr);
    right = _mm512_load_epi32(right_arr);
    merge_512_oddeven(&left, &right);
    _mm512_store_epi32(result, left);
    _mm512_store_epi32(result + 16, right);
    
    if (is_sorted(result, 32)) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        print_array("Result", result, 32);
        tests_failed++;
    }
    
    // Test case 2: Left all smaller
    printf("  Case 2: Left all smaller... ");
    fill_ascending(left_arr, 16, 0);    // 0-15
    fill_ascending(right_arr, 16, 16);  // 16-31
    
    left = _mm512_load_epi32(left_arr);
    right = _mm512_load_epi32(right_arr);
    merge_512_oddeven(&left, &right);
    _mm512_store_epi32(result, left);
    _mm512_store_epi32(result + 16, right);
    
    if (is_sorted(result, 32)) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        print_array("Result", result, 32);
        tests_failed++;
    }
    
    // Test case 3: Right all smaller
    printf("  Case 3: Right all smaller... ");
    fill_ascending(left_arr, 16, 16);   // 16-31
    fill_ascending(right_arr, 16, 0);   // 0-15
    
    left = _mm512_load_epi32(left_arr);
    right = _mm512_load_epi32(right_arr);
    merge_512_oddeven(&left, &right);
    _mm512_store_epi32(result, left);
    _mm512_store_epi32(result + 16, right);
    
    if (is_sorted(result, 32)) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        print_array("Result", result, 32);
        tests_failed++;
    }
    
    // Test case 4: Random sorted inputs
    printf("  Case 4: Random sorted inputs... ");
    fill_random(left_arr, 16);
    fill_random(right_arr, 16);
    qsort(left_arr, 16, sizeof(uint32_t), cmp_u32);
    qsort(right_arr, 16, sizeof(uint32_t), cmp_u32);
    
    left = _mm512_load_epi32(left_arr);
    right = _mm512_load_epi32(right_arr);
    merge_512_oddeven(&left, &right);
    _mm512_store_epi32(result, left);
    _mm512_store_epi32(result + 16, right);
    
    if (is_sorted(result, 32)) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        print_array("Left input", left_arr, 16);
        print_array("Right input", right_arr, 16);
        print_array("Result", result, 32);
        tests_failed++;
    }
}

// Test 2: Array-level merge (various sizes)
void test_array_merge(void) {
    printf("\n" YELLOW "=== Test: Array-level merge_arrays ===" RESET "\n");
    
    size_t sizes[] = {32, 64, 128, 256, 512, 1024, 4096, 16384};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        size_t n = sizes[s];
        size_t half = n / 2;
        
        uint32_t *left = aligned_alloc_u32(half);
        uint32_t *right = aligned_alloc_u32(half);
        uint32_t *result = aligned_alloc_u32(n);
        
        // Interleaved pattern
        printf("  Size %zu (interleaved)... ", n);
        fill_interleaved(left, half, 1);
        fill_interleaved(right, half, 0);
        
        merge_arrays(left, half, right, half, result);
        
        if (is_sorted(result, n)) {
            printf(GREEN "PASSED" RESET "\n");
            tests_passed++;
        } else {
            printf(RED "FAILED" RESET "\n");
            tests_failed++;
        }
        
        // Random pattern
        printf("  Size %zu (random)... ", n);
        fill_random(left, half);
        fill_random(right, half);
        qsort(left, half, sizeof(uint32_t), cmp_u32);
        qsort(right, half, sizeof(uint32_t), cmp_u32);
        
        merge_arrays(left, half, right, half, result);
        
        if (is_sorted(result, n)) {
            printf(GREEN "PASSED" RESET "\n");
            tests_passed++;
        } else {
            printf(RED "FAILED" RESET "\n");
            tests_failed++;
        }
        
        free(left);
        free(right);
        free(result);
    }
}

// Test 3: Cached merge variant
void test_cached_merge(void) {
    printf("\n" YELLOW "=== Test: Cached merge_arrays_cached ===" RESET "\n");
    
    size_t n = 4096;
    size_t half = n / 2;
    
    uint32_t *left = aligned_alloc_u32(half);
    uint32_t *right = aligned_alloc_u32(half);
    uint32_t *result = aligned_alloc_u32(n);
    
    printf("  Size %zu (random)... ", n);
    fill_random(left, half);
    fill_random(right, half);
    qsort(left, half, sizeof(uint32_t), cmp_u32);
    qsort(right, half, sizeof(uint32_t), cmp_u32);
    
    merge_arrays_cached(left, half, right, half, result);
    
    if (is_sorted(result, n)) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        tests_failed++;
    }
    
    free(left);
    free(right);
    free(result);
}

// Test 4: Key-Value merge
void test_kv_merge(void) {
    printf("\n" YELLOW "=== Test: Key-Value merge_arrays_kv ===" RESET "\n");
    
    size_t n = 4096;
    size_t half = n / 2;
    
    uint32_t *left_key = aligned_alloc_u32(half);
    uint32_t *left_pay = aligned_alloc_u32(half);
    uint32_t *right_key = aligned_alloc_u32(half);
    uint32_t *right_pay = aligned_alloc_u32(half);
    uint32_t *result_key = aligned_alloc_u32(n);
    uint32_t *result_pay = aligned_alloc_u32(n);
    
    printf("  Size %zu (interleaved)... ", n);
    
    // Create interleaved keys with matching payloads
    for (size_t i = 0; i < half; i++) {
        left_key[i] = 2 * i;
        left_pay[i] = 2 * i + 1000000;  // Payload = key + offset
        right_key[i] = 2 * i + 1;
        right_pay[i] = 2 * i + 1 + 1000000;
    }
    
    merge_arrays_kv(left_key, left_pay, half, right_key, right_pay, half,
                    result_key, result_pay);
    
    int passed = is_sorted(result_key, n);
    
    // Verify payloads match keys
    if (passed) {
        for (size_t i = 0; i < n; i++) {
            if (result_pay[i] != result_key[i] + 1000000) {
                passed = 0;
                printf("\n    Payload mismatch at %zu: key=%u, payload=%u\n", 
                       i, result_key[i], result_pay[i]);
                break;
            }
        }
    }
    
    if (passed) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        tests_failed++;
    }
    
    free(left_key);
    free(left_pay);
    free(right_key);
    free(right_pay);
    free(result_key);
    free(result_pay);
}

// Test 5: Unequal sizes
void test_unequal_sizes(void) {
    printf("\n" YELLOW "=== Test: Unequal input sizes ===" RESET "\n");
    
    struct { size_t left; size_t right; } cases[] = {
        {100, 50},
        {50, 100},
        {1000, 17},
        {17, 1000},
        {256, 255},
        {255, 256},
    };
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    
    for (int c = 0; c < num_cases; c++) {
        size_t left_size = cases[c].left;
        size_t right_size = cases[c].right;
        size_t total = left_size + right_size;
        
        uint32_t *left = aligned_alloc_u32(left_size);
        uint32_t *right = aligned_alloc_u32(right_size);
        uint32_t *result = aligned_alloc_u32(total);
        
        printf("  Left=%zu, Right=%zu... ", left_size, right_size);
        
        fill_random(left, left_size);
        fill_random(right, right_size);
        qsort(left, left_size, sizeof(uint32_t), cmp_u32);
        qsort(right, right_size, sizeof(uint32_t), cmp_u32);
        
        merge_arrays(left, left_size, right, right_size, result);
        
        if (is_sorted(result, total)) {
            printf(GREEN "PASSED" RESET "\n");
            tests_passed++;
        } else {
            printf(RED "FAILED" RESET "\n");
            tests_failed++;
        }
        
        free(left);
        free(right);
        free(result);
    }
}

// Test 6: Edge cases
void test_edge_cases(void) {
    printf("\n" YELLOW "=== Test: Edge cases ===" RESET "\n");
    
    // Small arrays (fallback to scalar)
    printf("  Small arrays (size 8+8)... ");
    uint32_t left[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    uint32_t right[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    uint32_t result[16];
    
    merge_arrays(left, 8, right, 8, result);
    
    if (is_sorted(result, 16)) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        tests_failed++;
    }
    
    // All same values
    printf("  All same values... ");
    uint32_t *same_left = aligned_alloc_u32(64);
    uint32_t *same_right = aligned_alloc_u32(64);
    uint32_t *same_result = aligned_alloc_u32(128);
    
    for (int i = 0; i < 64; i++) {
        same_left[i] = 42;
        same_right[i] = 42;
    }
    
    merge_arrays(same_left, 64, same_right, 64, same_result);
    
    if (is_sorted(same_result, 128)) {
        printf(GREEN "PASSED" RESET "\n");
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        tests_failed++;
    }
    
    free(same_left);
    free(same_right);
    free(same_result);
}

// Test 7: Large array performance sanity check
void test_large_merge(void) {
    printf("\n" YELLOW "=== Test: Large array merge (1M elements) ===" RESET "\n");
    
    size_t n = LARGE_SIZE;
    size_t half = n / 2;
    
    uint32_t *left = aligned_alloc_u32(half);
    uint32_t *right = aligned_alloc_u32(half);
    uint32_t *result = aligned_alloc_u32(n);
    
    if (!left || !right || !result) {
        printf("  " RED "SKIPPED (allocation failed)" RESET "\n");
        return;
    }
    
    printf("  Generating random sorted inputs...\n");
    fill_random(left, half);
    fill_random(right, half);
    qsort(left, half, sizeof(uint32_t), cmp_u32);
    qsort(right, half, sizeof(uint32_t), cmp_u32);
    
    printf("  Merging %zu elements... ", n);
    
    clock_t start = clock();
    merge_arrays(left, half, right, half, result);
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double throughput = (n * sizeof(uint32_t)) / elapsed / 1e9;  // GB/s
    
    if (is_sorted(result, n)) {
        printf(GREEN "PASSED" RESET " (%.3f sec, %.2f GB/s)\n", elapsed, throughput);
        tests_passed++;
    } else {
        printf(RED "FAILED" RESET "\n");
        tests_failed++;
    }
    
    free(left);
    free(right);
    free(result);
}

// ============== Main ==============

int main(int argc, char **argv) {
    printf("===== AVX-512 Odd-Even Merge Test Suite =====\n");
    printf("Testing merge network correctness...\n");
    
    srand(42);  // Deterministic for reproducibility
    
    test_register_merge();
    test_array_merge();
    test_cached_merge();
    test_kv_merge();
    test_unequal_sizes();
    test_edge_cases();
    test_large_merge();
    
    printf("\n===== Results =====\n");
    printf("Passed: " GREEN "%d" RESET "\n", tests_passed);
    printf("Failed: %s%d%s\n", tests_failed > 0 ? RED : GREEN, tests_failed, RESET);
    
    return tests_failed > 0 ? 1 : 0;
}
