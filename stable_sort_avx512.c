/*
 * Stable Merge Sort with AVX-512 Acceleration
 * 
 * Key insight: A stable sort must preserve relative order of equal elements.
 * This means we CANNOT use sorting networks (bitonic) for merging because
 * min/max operations scramble equal elements.
 * 
 * Instead, we use:
 * 1. Insertion sort for base case (naturally stable)
 * 2. Sequential merge with <= comparison (guarantees stability)
 * 3. AVX-512 for bulk memory operations (copies, already-sorted detection)
 * 4. Galloping optimization to find long runs and copy them with SIMD
 */

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

// ============== Configuration ==============
#define INSERTION_THRESHOLD 32   // Use insertion sort below this size
#define MIN_GALLOP 7             // Minimum run length to trigger galloping

// ============== SIMD Memory Operations ==============

// Fast SIMD memcpy for uint32_t arrays
static inline void simd_copy(uint32_t *restrict dst, 
                             const uint32_t *restrict src, 
                             size_t n) {
    size_t i = 0;
    
    // Copy 64 bytes (16 uint32_t) at a time
    for (; i + 16 <= n; i += 16) {
        __m512i v = _mm512_loadu_epi32(src + i);
        _mm512_storeu_epi32(dst + i, v);
    }
    
    // Handle remainder with masked operation
    if (i < n) {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1);
        __m512i v = _mm512_maskz_loadu_epi32(mask, src + i);
        _mm512_mask_storeu_epi32(dst + i, mask, v);
    }
}

// Check if array is already sorted (useful optimization)
static inline bool is_sorted(const uint32_t *arr, size_t n) {
    if (n <= 1) return true;
    
    size_t i = 0;
    
    // SIMD check: compare arr[i] with arr[i+1] for 15 pairs at a time
    for (; i + 16 <= n; i += 15) {
        __m512i curr = _mm512_loadu_epi32(arr + i);
        __m512i next = _mm512_loadu_epi32(arr + i + 1);
        // Check if any curr[j] > next[j] (would mean unsorted)
        __mmask16 unsorted = _mm512_cmpgt_epu32_mask(curr, next);
        // Only check first 15 pairs (last element of curr vs first of next chunk)
        if (unsorted & 0x7FFF) return false;
    }
    
    // Scalar check for remainder
    for (; i + 1 < n; i++) {
        if (arr[i] > arr[i + 1]) return false;
    }
    
    return true;
}

// ============== Stable Base Case: Binary Insertion Sort ==============
// Uses binary search to find insertion point, memmove for shifting
// memmove is optimized by compiler to use SIMD

static void insertion_sort(uint32_t *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        uint32_t key = arr[i];
        
        // Binary search for insertion point (upper_bound for stability)
        // Find first position where arr[pos] > key
        size_t lo = 0, hi = i;
        while (lo < hi) {
            size_t mid = lo + (hi - lo) / 2;
            if (arr[mid] <= key) {
                lo = mid + 1;  // <= ensures stability: equal elements stay in order
            } else {
                hi = mid;
            }
        }
        
        // Shift elements right to make room (SIMD-optimized by compiler)
        if (lo < i) {
            memmove(arr + lo + 1, arr + lo, (i - lo) * sizeof(uint32_t));
            arr[lo] = key;
        }
    }
}

// ============== Binary Search Helpers ==============

// Lower bound: first position where arr[pos] >= val
static inline size_t lower_bound(const uint32_t *arr, size_t n, uint32_t val) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Upper bound: first position where arr[pos] > val
static inline size_t upper_bound(const uint32_t *arr, size_t n, uint32_t val) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (arr[mid] <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ============== Stable Merge with Galloping ==============
// 
// Galloping (from Timsort): when one run is "winning" repeatedly,
// use binary search to find how many elements can be copied in bulk.
// This turns O(n) element-by-element copies into O(log n) search + O(n) SIMD copy.

static void stable_merge(
    const uint32_t *restrict left, size_t nl,
    const uint32_t *restrict right, size_t nr,
    uint32_t *restrict out
) {
    size_t li = 0, ri = 0, oi = 0;
    size_t left_wins = 0, right_wins = 0;
    
    while (li < nl && ri < nr) {
        // Prefetch ahead for better cache behavior
        if ((oi & 63) == 0) {
            _mm_prefetch((const char*)(left + li + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(right + ri + 64), _MM_HINT_T0);
        }
        
        if (left[li] <= right[ri]) {
            // Left wins - for stability, <= means left element comes first on tie
            out[oi++] = left[li++];
            left_wins++;
            right_wins = 0;
            
            // Check if we should gallop
            if (left_wins >= MIN_GALLOP && li < nl) {
                // Find how many more from left are <= right[ri]
                size_t remaining = nl - li;
                size_t count = upper_bound(left + li, remaining, right[ri]);
                
                if (count > 0) {
                    simd_copy(out + oi, left + li, count);
                    li += count;
                    oi += count;
                    left_wins = 0;  // Reset after gallop
                }
            }
        } else {
            // Right wins
            out[oi++] = right[ri++];
            right_wins++;
            left_wins = 0;
            
            // Check if we should gallop
            if (right_wins >= MIN_GALLOP && ri < nr) {
                // Find how many from right are < left[li] (strict < for stability)
                size_t remaining = nr - ri;
                size_t count = lower_bound(right + ri, remaining, left[li]);
                
                if (count > 0) {
                    simd_copy(out + oi, right + ri, count);
                    ri += count;
                    oi += count;
                    right_wins = 0;  // Reset after gallop
                }
            }
        }
    }
    
    // Copy remaining elements with SIMD
    if (li < nl) {
        simd_copy(out + oi, left + li, nl - li);
    }
    if (ri < nr) {
        simd_copy(out + oi, right + ri, nr - ri);
    }
}

// ============== Simple Stable Merge (no galloping) ==============
// For when arrays are small or galloping overhead isn't worth it

static void stable_merge_simple(
    const uint32_t *restrict left, size_t nl,
    const uint32_t *restrict right, size_t nr,
    uint32_t *restrict out
) {
    size_t li = 0, ri = 0, oi = 0;
    
    while (li < nl && ri < nr) {
        // <= ensures stability: equal elements from left come first
        if (left[li] <= right[ri]) {
            out[oi++] = left[li++];
        } else {
            out[oi++] = right[ri++];
        }
    }
    
    // Copy remaining
    if (li < nl) simd_copy(out + oi, left + li, nl - li);
    if (ri < nr) simd_copy(out + oi, right + ri, nr - ri);
}

// ============== Main Sort Function ==============

void stable_merge_sort_avx512(uint32_t *arr, size_t n) {
    if (n <= 1) return;
    
    // Small arrays: just use insertion sort
    if (n <= INSERTION_THRESHOLD) {
        insertion_sort(arr, n);
        return;
    }
    
    // Allocate temp buffer (64-byte aligned for AVX-512)
    uint32_t *temp = (uint32_t*)aligned_alloc(64, n * sizeof(uint32_t));
    if (!temp) {
        temp = (uint32_t*)malloc(n * sizeof(uint32_t));
        if (!temp) return;  // Allocation failed
    }
    
    // Phase 1: Sort small chunks with insertion sort (stable)
    for (size_t i = 0; i < n; i += INSERTION_THRESHOLD) {
        size_t chunk_size = (i + INSERTION_THRESHOLD <= n) ? 
                           INSERTION_THRESHOLD : (n - i);
        insertion_sort(arr + i, chunk_size);
    }
    
    // Phase 2: Bottom-up merge (avoids recursion overhead)
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    for (size_t width = INSERTION_THRESHOLD; width < n; width *= 2) {
        for (size_t i = 0; i < n; i += 2 * width) {
            size_t left_start = i;
            size_t mid = (i + width < n) ? (i + width) : n;
            size_t right_end = (i + 2 * width < n) ? (i + 2 * width) : n;
            
            size_t left_size = mid - left_start;
            size_t right_size = right_end - mid;
            
            if (right_size == 0) {
                // Only left part exists
                simd_copy(dst + left_start, src + left_start, left_size);
            } 
            else if (src[mid - 1] <= src[mid]) {
                // Already sorted across boundary - just copy both
                simd_copy(dst + left_start, src + left_start, left_size + right_size);
            }
            else {
                // Need to merge
                stable_merge(src + left_start, left_size,
                            src + mid, right_size,
                            dst + left_start);
            }
        }
        
        // Swap buffers
        uint32_t *tmp = src;
        src = dst;
        dst = tmp;
    }
    
    // If result ended up in temp, copy back to arr
    if (src != arr) {
        simd_copy(arr, src, n);
    }
    
    free(temp);
}

// ============== Version with pre-allocated buffer ==============
// Useful when sorting multiple arrays to avoid repeated allocation

void stable_merge_sort_avx512_with_buffer(uint32_t *arr, size_t n, uint32_t *temp) {
    if (n <= 1) return;
    
    if (n <= INSERTION_THRESHOLD) {
        insertion_sort(arr, n);
        return;
    }
    
    // Phase 1: Sort small chunks
    for (size_t i = 0; i < n; i += INSERTION_THRESHOLD) {
        size_t chunk_size = (i + INSERTION_THRESHOLD <= n) ? 
                           INSERTION_THRESHOLD : (n - i);
        insertion_sort(arr + i, chunk_size);
    }
    
    // Phase 2: Bottom-up merge
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    for (size_t width = INSERTION_THRESHOLD; width < n; width *= 2) {
        for (size_t i = 0; i < n; i += 2 * width) {
            size_t left_start = i;
            size_t mid = (i + width < n) ? (i + width) : n;
            size_t right_end = (i + 2 * width < n) ? (i + 2 * width) : n;
            
            size_t left_size = mid - left_start;
            size_t right_size = right_end - mid;
            
            if (right_size == 0) {
                simd_copy(dst + left_start, src + left_start, left_size);
            } 
            else if (src[mid - 1] <= src[mid]) {
                simd_copy(dst + left_start, src + left_start, left_size + right_size);
            }
            else {
                stable_merge(src + left_start, left_size,
                            src + mid, right_size,
                            dst + left_start);
            }
        }
        
        uint32_t *tmp = src;
        src = dst;
        dst = tmp;
    }
    
    if (src != arr) {
        simd_copy(arr, src, n);
    }
}

// ============== Parallel version using OpenMP ==============
#ifdef _OPENMP
#include <omp.h>

void stable_merge_sort_avx512_parallel(uint32_t *arr, size_t n, int num_threads) {
    if (n <= 1) return;
    
    if (n <= INSERTION_THRESHOLD) {
        insertion_sort(arr, n);
        return;
    }
    
    uint32_t *temp = (uint32_t*)aligned_alloc(64, n * sizeof(uint32_t));
    if (!temp) {
        temp = (uint32_t*)malloc(n * sizeof(uint32_t));
        if (!temp) return;
    }
    
    // Phase 1: Parallel insertion sort of chunks
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < n; i += INSERTION_THRESHOLD) {
        size_t chunk_size = (i + INSERTION_THRESHOLD <= n) ? 
                           INSERTION_THRESHOLD : (n - i);
        insertion_sort(arr + i, chunk_size);
    }
    
    // Phase 2: Parallel bottom-up merge
    uint32_t *src = arr;
    uint32_t *dst = temp;
    
    for (size_t width = INSERTION_THRESHOLD; width < n; width *= 2) {
        size_t num_merges = (n + 2 * width - 1) / (2 * width);
        
        // Parallelize when there are enough independent merges
        if (num_merges >= (size_t)num_threads) {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
            for (size_t i = 0; i < n; i += 2 * width) {
                size_t left_start = i;
                size_t mid = (i + width < n) ? (i + width) : n;
                size_t right_end = (i + 2 * width < n) ? (i + 2 * width) : n;
                
                size_t left_size = mid - left_start;
                size_t right_size = right_end - mid;
                
                if (right_size == 0) {
                    simd_copy(dst + left_start, src + left_start, left_size);
                } 
                else if (src[mid - 1] <= src[mid]) {
                    simd_copy(dst + left_start, src + left_start, left_size + right_size);
                }
                else {
                    stable_merge(src + left_start, left_size,
                                src + mid, right_size,
                                dst + left_start);
                }
            }
        } else {
            // Sequential when merges are too large/few
            for (size_t i = 0; i < n; i += 2 * width) {
                size_t left_start = i;
                size_t mid = (i + width < n) ? (i + width) : n;
                size_t right_end = (i + 2 * width < n) ? (i + 2 * width) : n;
                
                size_t left_size = mid - left_start;
                size_t right_size = right_end - mid;
                
                if (right_size == 0) {
                    simd_copy(dst + left_start, src + left_start, left_size);
                } 
                else if (src[mid - 1] <= src[mid]) {
                    simd_copy(dst + left_start, src + left_start, left_size + right_size);
                }
                else {
                    stable_merge(src + left_start, left_size,
                                src + mid, right_size,
                                dst + left_start);
                }
            }
        }
        
        uint32_t *tmp = src;
        src = dst;
        dst = tmp;
    }
    
    if (src != arr) {
        simd_copy(arr, src, n);
    }
    
    free(temp);
}
#endif
