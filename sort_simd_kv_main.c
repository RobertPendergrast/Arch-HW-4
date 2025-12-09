#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#include "sort_simd_kv.h"
#include "utils.h"

static size_t count_stability_violations(uint32_t *keys, uint32_t *payload, size_t size) {
    size_t violations = 0;

    for (size_t i = 0; i + 1 < size; i++) {
        if (keys[i] == keys[i + 1] && payload[i] > payload[i + 1]) {
            violations++;
        }
    }

    return violations;
}

static size_t count_equal_key_pairs(uint32_t *keys, size_t size) {
    size_t equal_pairs = 0;

    for (size_t i = 0; i + 1 < size; i++) {
        if (keys[i] == keys[i + 1]) {
            equal_pairs++;
        }
    }

    return equal_pairs;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    uint64_t size;
    uint32_t *keys = read_array_from_file(argv[1], &size);
    if (!keys) {
        return 1;
    }

    printf("Read %lu elements from %s\n", size, argv[1]);

    uint32_t *payload = NULL;
    if (posix_memalign((void**)&payload, 64, size * sizeof(uint32_t)) != 0) {
        fprintf(stderr, "posix_memalign failed for payload\n");
        free(keys);
        return 1;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        payload[i] = (uint32_t)i;
    }

    uint64_t xor_before, sum_before;
    compute_hash(keys, size, &xor_before, &sum_before);
    printf("Input hash: XOR=0x%016lx SUM=0x%016lx\n", xor_before, sum_before);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sort_array_kv(keys, payload, size);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Sorting took %.3f seconds\n", elapsed);

    uint64_t xor_after, sum_after;
    compute_hash(keys, size, &xor_after, &sum_after);
    printf("Output hash: XOR=0x%016lx SUM=0x%016lx\n", xor_after, sum_after);

    if (xor_before != xor_after || sum_before != sum_after) {
        printf("Error: Hash mismatch! Elements were lost or corrupted during sorting.\n");
        free(keys);
        free(payload);
        return 1;
    }
    printf("Hash check passed: all elements preserved.\n");

    if (verify_sortedness(keys, size)) {
        printf("Array sorted successfully!\n");
    } else {
        printf("Error: Array is not sorted correctly!\n");
        free(keys);
        free(payload);
        return 1;
    }

    size_t equal_pairs = count_equal_key_pairs(keys, size);
    size_t violations = count_stability_violations(keys, payload, size);

    printf("\n=== STABILITY ANALYSIS ===\n");
    printf("Adjacent pairs with equal keys: %zu\n", equal_pairs);
    printf("Stability violations: %zu\n", violations);
    if (equal_pairs > 0) {
        double stability_pct = 100.0 * (1.0 - (double)violations / equal_pairs);
        printf("Stability score: %.2f%% (%.2f%% of equal-key pairs preserved order)\n",
               stability_pct, stability_pct);
    } else {
        printf("No duplicate keys found - stability not applicable.\n");
    }

    free(keys);
    free(payload);
    return 0;
}
