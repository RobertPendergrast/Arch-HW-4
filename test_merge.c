#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "merge.h"
#include "utils.h"

// Test counters
static int tests_passed = 0;
static int tests_failed = 0;

// Helper to verify array is sorted
int is_sorted(uint32_t *arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        if (arr[i-1] > arr[i]) {
            return 0;
        }
    }
    return 1;
}

// Helper to verify merged array contains all expected values
int verify_merge_result(uint32_t *result, uint32_t *left, size_t left_size, 
                        uint32_t *right, size_t right_size) {
    size_t total = left_size + right_size;
    
    // Check sorted
    if (!is_sorted(result, total)) {
        return 0;
    }
    
    // Check all elements present (by counting)
    // Create a copy and sort both to compare
    uint32_t *expected = malloc(total * sizeof(uint32_t));
    memcpy(expected, left, left_size * sizeof(uint32_t));
    memcpy(expected + left_size, right, right_size * sizeof(uint32_t));
    
    // Simple bubble sort for verification (we know inputs are sorted, so just merge)
    uint32_t *temp = malloc(total * sizeof(uint32_t));
    size_t i = 0, j = 0, k = 0;
    while (i < left_size && j < right_size) {
        if (left[i] <= right[j]) temp[k++] = left[i++];
        else temp[k++] = right[j++];
    }
    while (i < left_size) temp[k++] = left[i++];
    while (j < right_size) temp[k++] = right[j++];
    
    int match = (memcmp(result, temp, total * sizeof(uint32_t)) == 0);
    
    free(expected);
    free(temp);
    return match;
}

// ============== merge_512_registers Tests ==============

int test_merge_512_basic() {
    printf("  test_merge_512_basic: ");
    
    // Interleaved sorted sequences
    uint32_t left[] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    uint32_t right[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    
    __m512i left_reg = _mm512_loadu_epi32(left);
    __m512i right_reg = _mm512_loadu_epi32(right);
    
    merge_512_registers(&left_reg, &right_reg);
    
    uint32_t result_left[16], result_right[16];
    _mm512_storeu_epi32(result_left, left_reg);
    _mm512_storeu_epi32(result_right, right_reg);
    
    // Check left has 1-16, right has 17-32
    int pass = 1;
    for (int i = 0; i < 16; i++) {
        if (result_left[i] != (uint32_t)(i + 1)) pass = 0;
        if (result_right[i] != (uint32_t)(i + 17)) pass = 0;
    }
    
    if (pass) {
        printf("PASSED\n");
        return 1;
    } else {
        printf("FAILED\n");
        printf("    Left:  "); for(int i=0; i<16; i++) printf("%u ", result_left[i]); printf("\n");
        printf("    Right: "); for(int i=0; i<16; i++) printf("%u ", result_right[i]); printf("\n");
        return 0;
    }
}

int test_merge_512_already_partitioned() {
    printf("  test_merge_512_already_partitioned: ");
    
    // Left is all smaller than right
    uint32_t left[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint32_t right[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    
    __m512i left_reg = _mm512_loadu_epi32(left);
    __m512i right_reg = _mm512_loadu_epi32(right);
    
    merge_512_registers(&left_reg, &right_reg);
    
    uint32_t result_left[16], result_right[16];
    _mm512_storeu_epi32(result_left, left_reg);
    _mm512_storeu_epi32(result_right, right_reg);
    
    int pass = 1;
    for (int i = 0; i < 16; i++) {
        if (result_left[i] != (uint32_t)(i + 1)) pass = 0;
        if (result_right[i] != (uint32_t)(i + 17)) pass = 0;
    }
    
    if (pass) { printf("PASSED\n"); return 1; }
    else {
        printf("FAILED\n");
        printf("    Left:  "); for(int i=0; i<16; i++) printf("%u ", result_left[i]); printf("\n");
        printf("    Right: "); for(int i=0; i<16; i++) printf("%u ", result_right[i]); printf("\n");
        return 0;
    }
}

int test_merge_512_reversed() {
    printf("  test_merge_512_reversed: ");
    
    // Left is all larger than right (should swap)
    uint32_t left[] = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    uint32_t right[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    
    __m512i left_reg = _mm512_loadu_epi32(left);
    __m512i right_reg = _mm512_loadu_epi32(right);
    
    merge_512_registers(&left_reg, &right_reg);
    
    uint32_t result_left[16], result_right[16];
    _mm512_storeu_epi32(result_left, left_reg);
    _mm512_storeu_epi32(result_right, right_reg);
    
    int pass = 1;
    for (int i = 0; i < 16; i++) {
        if (result_left[i] != (uint32_t)(i + 1)) pass = 0;
        if (result_right[i] != (uint32_t)(i + 17)) pass = 0;
    }
    
    if (pass) { printf("PASSED\n"); return 1; }
    else {
        printf("FAILED\n");
        printf("    Left:  "); for(int i=0; i<16; i++) printf("%u ", result_left[i]); printf("\n");
        printf("    Right: "); for(int i=0; i<16; i++) printf("%u ", result_right[i]); printf("\n");
        return 0;
    }
}

int test_merge_512_duplicates() {
    printf("  test_merge_512_duplicates: ");
    
    // Arrays with duplicate values
    uint32_t left[] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
    uint32_t right[] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8};
    
    __m512i left_reg = _mm512_loadu_epi32(left);
    __m512i right_reg = _mm512_loadu_epi32(right);
    
    merge_512_registers(&left_reg, &right_reg);
    
    uint32_t result_left[16], result_right[16];
    _mm512_storeu_epi32(result_left, left_reg);
    _mm512_storeu_epi32(result_right, right_reg);
    
    // Just verify sorted
    int pass = is_sorted(result_left, 16) && is_sorted(result_right, 16);
    // And that max of left <= min of right
    if (pass && result_left[15] > result_right[0]) pass = 0;
    
    if (pass) { printf("PASSED\n"); return 1; }
    else {
        printf("FAILED\n");
        printf("    Left:  "); for(int i=0; i<16; i++) printf("%u ", result_left[i]); printf("\n");
        printf("    Right: "); for(int i=0; i<16; i++) printf("%u ", result_right[i]); printf("\n");
        return 0;
    }
}

int test_merge_512_random() {
    printf("  test_merge_512_random: ");
    
    // Generate random sorted arrays
    uint32_t left[16], right[16];
    uint32_t val = rand() % 100;
    for (int i = 0; i < 16; i++) {
        left[i] = val;
        val += rand() % 10 + 1;
    }
    val = rand() % 100;
    for (int i = 0; i < 16; i++) {
        right[i] = val;
        val += rand() % 10 + 1;
    }
    
    __m512i left_reg = _mm512_loadu_epi32(left);
    __m512i right_reg = _mm512_loadu_epi32(right);
    
    merge_512_registers(&left_reg, &right_reg);
    
    uint32_t result_left[16], result_right[16];
    _mm512_storeu_epi32(result_left, left_reg);
    _mm512_storeu_epi32(result_right, right_reg);
    
    int pass = is_sorted(result_left, 16) && is_sorted(result_right, 16);
    if (pass && result_left[15] > result_right[0]) pass = 0;
    
    if (pass) { printf("PASSED\n"); return 1; }
    else {
        printf("FAILED\n");
        printf("    Input Left:  "); for(int i=0; i<16; i++) printf("%u ", left[i]); printf("\n");
        printf("    Input Right: "); for(int i=0; i<16; i++) printf("%u ", right[i]); printf("\n");
        printf("    Result Left:  "); for(int i=0; i<16; i++) printf("%u ", result_left[i]); printf("\n");
        printf("    Result Right: "); for(int i=0; i<16; i++) printf("%u ", result_right[i]); printf("\n");
        return 0;
    }
}

// ============== merge_arrays Tests ==============

int test_merge_arrays_16_16() {
    printf("  test_merge_arrays_16_16: ");
    
    uint32_t left[16], right[16], result[32];
    for (int i = 0; i < 16; i++) {
        left[i] = 2*i + 1;   // 1, 3, 5, ..., 31
        right[i] = 2*i + 2;  // 2, 4, 6, ..., 32
    }
    
    merge_arrays(left, 16, right, 16, result);
    
    int pass = 1;
    for (int i = 0; i < 32; i++) {
        if (result[i] != (uint32_t)(i + 1)) {
            pass = 0;
            printf("FAILED at index %d: expected %d, got %u\n", i, i+1, result[i]);
            break;
        }
    }
    
    if (pass) { printf("PASSED\n"); return 1; }
    return 0;
}

int test_merge_arrays_32_32() {
    printf("  test_merge_arrays_32_32: ");
    
    uint32_t left[32], right[32], result[64];
    for (int i = 0; i < 32; i++) {
        left[i] = 2*i + 1;   // 1, 3, 5, ..., 63
        right[i] = 2*i + 2;  // 2, 4, 6, ..., 64
    }
    
    merge_arrays(left, 32, right, 32, result);
    
    int pass = 1;
    for (int i = 0; i < 64; i++) {
        if (result[i] != (uint32_t)(i + 1)) {
            pass = 0;
            printf("FAILED at index %d: expected %d, got %u\n", i, i+1, result[i]);
            break;
        }
    }
    
    if (pass) { printf("PASSED\n"); return 1; }
    return 0;
}

int test_merge_arrays_33_33() {
    printf("  test_merge_arrays_33_33 (non-multiple of 16): ");
    
    uint32_t left[33], right[33], result[66];
    for (int i = 0; i < 33; i++) {
        left[i] = 2*i + 1;
        right[i] = 2*i + 2;
    }
    
    merge_arrays(left, 33, right, 33, result);
    
    int pass = 1;
    for (int i = 0; i < 66; i++) {
        if (result[i] != (uint32_t)(i + 1)) {
            pass = 0;
            printf("FAILED at index %d: expected %d, got %u\n", i, i+1, result[i]);
            break;
        }
    }
    
    if (pass) { printf("PASSED\n"); return 1; }
    return 0;
}

int test_merge_arrays_small() {
    printf("  test_merge_arrays_small (< 16 elements): ");
    
    uint32_t left[] = {1, 3, 5, 7, 9};
    uint32_t right[] = {2, 4, 6, 8, 10};
    uint32_t result[10];
    
    merge_arrays(left, 5, right, 5, result);
    
    int pass = 1;
    for (int i = 0; i < 10; i++) {
        if (result[i] != (uint32_t)(i + 1)) {
            pass = 0;
            printf("FAILED at index %d: expected %d, got %u\n", i, i+1, result[i]);
            break;
        }
    }
    
    if (pass) { printf("PASSED\n"); return 1; }
    return 0;
}

int test_merge_arrays_asymmetric() {
    printf("  test_merge_arrays_asymmetric (17 + 31): ");
    
    uint32_t *left = malloc(17 * sizeof(uint32_t));
    uint32_t *right = malloc(31 * sizeof(uint32_t));
    uint32_t *result = malloc(48 * sizeof(uint32_t));
    
    // Left has odd numbers 1-33
    for (int i = 0; i < 17; i++) left[i] = 2*i + 1;
    // Right has even numbers 2-62
    for (int i = 0; i < 31; i++) right[i] = 2*i + 2;
    
    merge_arrays(left, 17, right, 31, result);
    
    int pass = is_sorted(result, 48);
    if (pass) pass = verify_merge_result(result, left, 17, right, 31);
    
    free(left); free(right); free(result);
    
    if (pass) { printf("PASSED\n"); return 1; }
    else { printf("FAILED\n"); return 0; }
}

int test_merge_arrays_large() {
    printf("  test_merge_arrays_large (1000 + 1000): ");
    
    uint32_t *left = malloc(1000 * sizeof(uint32_t));
    uint32_t *right = malloc(1000 * sizeof(uint32_t));
    uint32_t *result = malloc(2000 * sizeof(uint32_t));
    
    for (int i = 0; i < 1000; i++) {
        left[i] = 2*i + 1;
        right[i] = 2*i + 2;
    }
    
    merge_arrays(left, 1000, right, 1000, result);
    
    int pass = 1;
    for (int i = 0; i < 2000; i++) {
        if (result[i] != (uint32_t)(i + 1)) {
            pass = 0;
            printf("FAILED at index %d: expected %d, got %u\n", i, i+1, result[i]);
            break;
        }
    }
    
    free(left); free(right); free(result);
    
    if (pass) { printf("PASSED\n"); return 1; }
    return 0;
}

int test_merge_arrays_random_large() {
    printf("  test_merge_arrays_random_large (5000 + 5000 random): ");
    
    size_t size = 5000;
    uint32_t *left = malloc(size * sizeof(uint32_t));
    uint32_t *right = malloc(size * sizeof(uint32_t));
    uint32_t *result = malloc(2 * size * sizeof(uint32_t));
    
    // Generate sorted random arrays
    uint32_t val = 0;
    for (size_t i = 0; i < size; i++) {
        val += rand() % 10 + 1;
        left[i] = val;
    }
    val = 0;
    for (size_t i = 0; i < size; i++) {
        val += rand() % 10 + 1;
        right[i] = val;
    }
    
    merge_arrays(left, size, right, size, result);
    
    int pass = is_sorted(result, 2 * size);
    if (pass) pass = verify_merge_result(result, left, size, right, size);
    
    if (!pass) {
        // Find first error
        for (size_t i = 1; i < 2*size; i++) {
            if (result[i-1] > result[i]) {
                printf("FAILED - not sorted at index %zu: %u > %u\n", i, result[i-1], result[i]);
                break;
            }
        }
    }
    
    free(left); free(right); free(result);
    
    if (pass) { printf("PASSED\n"); return 1; }
    return 0;
}

int test_merge_arrays_edge_sizes() {
    printf("  test_merge_arrays_edge_sizes: ");
    
    int sizes[] = {16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 100, 127, 128, 129};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int pass = 1;
    
    for (int si = 0; si < num_sizes && pass; si++) {
        for (int sj = 0; sj < num_sizes && pass; sj++) {
            int s1 = sizes[si], s2 = sizes[sj];
            
            uint32_t *left = malloc(s1 * sizeof(uint32_t));
            uint32_t *right = malloc(s2 * sizeof(uint32_t));
            uint32_t *result = malloc((s1 + s2) * sizeof(uint32_t));
            
            uint32_t val = 0;
            for (int i = 0; i < s1; i++) { val += rand() % 5 + 1; left[i] = val; }
            val = 0;
            for (int i = 0; i < s2; i++) { val += rand() % 5 + 1; right[i] = val; }
            
            merge_arrays(left, s1, right, s2, result);
            
            if (!is_sorted(result, s1 + s2)) {
                printf("FAILED for sizes %d + %d\n", s1, s2);
                pass = 0;
            }
            
            free(left); free(right); free(result);
        }
    }
    
    if (pass) { printf("PASSED (tested %d size combinations)\n", num_sizes * num_sizes); return 1; }
    return 0;
}

// ============== Full Sort Tests ==============

// Copy of insertion_sort from sort_simd.c
void insertion_sort_test(uint32_t *arr, size_t size) {
    for (size_t i = 1; i < size; i++) {
        uint32_t key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

#define SORT_THRESHOLD_TEST 32

void basic_merge_sort_test(uint32_t *arr, size_t size) {
    if (size <= SORT_THRESHOLD_TEST) {
        insertion_sort_test(arr, size);
        return;
    }
    size_t middle = size / 2;
    size_t size_left = middle;
    size_t size_right = size - middle;

    uint32_t* left = NULL;
    uint32_t* right = NULL;
    if (posix_memalign((void**)&left, 64, size_left * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for left array\n");
        exit(EXIT_FAILURE);
    }
    if (posix_memalign((void**)&right, 64, size_right * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for right array\n");
        free(left);
        exit(EXIT_FAILURE);
    }

    memcpy(left, arr, size_left * sizeof(uint32_t));
    memcpy(right, arr + middle, size_right * sizeof(uint32_t));

    basic_merge_sort_test(left, size_left);
    basic_merge_sort_test(right, size_right);

    merge_arrays(left, size_left, right, size_right, arr);
    free(left);
    free(right);
}

// Comparison function for qsort
int compare_uint32(const void *a, const void *b) {
    uint32_t ua = *(const uint32_t*)a;
    uint32_t ub = *(const uint32_t*)b;
    return (ua > ub) - (ua < ub);
}

int test_full_sort() {
    printf("  Testing full merge sort with various sizes:\n");
    int pass = 1;
    
    // Test sizes
    size_t sizes[] = {10, 32, 33, 64, 100, 128, 256, 500, 1000, 5000, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int si = 0; si < num_sizes && pass; si++) {
        size_t size = sizes[si];
        printf("    Size %zu: ", size);
        
        uint32_t *arr = malloc(size * sizeof(uint32_t));
        uint32_t *copy = malloc(size * sizeof(uint32_t));
        
        // Generate random data
        for (size_t i = 0; i < size; i++) {
            arr[i] = rand();
            copy[i] = arr[i];
        }
        
        // Sort using our merge sort
        basic_merge_sort_test(arr, size);
        
        // Verify sorted
        if (!is_sorted(arr, size)) {
            printf("FAILED - not sorted!\n");
            // Find first error
            for (size_t i = 1; i < size; i++) {
                if (arr[i-1] > arr[i]) {
                    printf("      First error at index %zu: %u > %u\n", i, arr[i-1], arr[i]);
                    break;
                }
            }
            pass = 0;
        } else {
            // Verify all elements present (sort the copy with qsort and compare)
            qsort(copy, size, sizeof(uint32_t), compare_uint32);
            
            if (memcmp(arr, copy, size * sizeof(uint32_t)) != 0) {
                printf("FAILED - wrong elements!\n");
                pass = 0;
            } else {
                printf("PASSED\n");
            }
        }
        
        free(arr);
        free(copy);
    }
    
    return pass;
}

// ============== Main ==============

int main(int argc, char *argv[]) {
    srand(42);  // Fixed seed for reproducibility
    
    printf("====== merge_512_registers Tests ======\n");
    tests_passed += test_merge_512_basic();
    tests_passed += test_merge_512_already_partitioned();
    tests_passed += test_merge_512_reversed();
    tests_passed += test_merge_512_duplicates();
    for (int i = 0; i < 5; i++) {
        tests_passed += test_merge_512_random();
    }
    
    printf("\n====== merge_arrays Tests ======\n");
    tests_passed += test_merge_arrays_16_16();
    tests_passed += test_merge_arrays_32_32();
    tests_passed += test_merge_arrays_33_33();
    tests_passed += test_merge_arrays_small();
    tests_passed += test_merge_arrays_asymmetric();
    tests_passed += test_merge_arrays_large();
    tests_passed += test_merge_arrays_random_large();
    tests_passed += test_merge_arrays_edge_sizes();
    
    printf("\n====== Full Sort Tests ======\n");
    tests_passed += test_full_sort();
    
    printf("\n====== Summary ======\n");
    int total = tests_passed + tests_failed;
    printf("Passed: %d / %d\n", tests_passed, tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf("ALL TESTS PASSED!\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED!\n");
        return 1;
    }
}
