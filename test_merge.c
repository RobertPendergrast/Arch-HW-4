#include <stdint.h>
#include <stdlib.h>

#include "merge.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    uint32_t test_arr_1[] = {1, 5, 8, 2};
    size_t size_1 = sizeof(test_arr_1);
    uint32_t test_arr_2[] = {4, 7, 3, 6};
    size_t size_2 = sizeof(test_arr_2);
    uint32_t* arr = malloc((size_1) + (size_2) * sizeof(uint32_t));
    merge_arrays(
        test_arr_1,
        size_1,
        test_arr_2,
        size_2,
        arr
    );
    print_array(arr, size_1 + size_2);
}
