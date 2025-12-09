CC = gcc
CFLAGS = -Wall -O3 -mavx512f -mavx512bw -march=native -funroll-loops -ffast-math -fopenmp
LDFLAGS = -pthread

all: sort_simd sort_simd_kv

# Object files
utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o utils.o

merge.o: merge.c merge.h
	$(CC) $(CFLAGS) -c merge.c -o merge.o

# sorting (keys only)
sorting.o: sorting.c sorting.h
	$(CC) $(CFLAGS) -c sorting.c -o sorting.o

sort_simd_main.o: sort_simd_main.c sorting.h
	$(CC) $(CFLAGS) -c sort_simd_main.c -o sort_simd_main.o

sort_simd_main: sorting.o sort_simd_main.o utils.o merge.o
	$(CC) $(CFLAGS) sort_simd_main.o sorting.o utils.o merge.o -o sort_simd_main $(LDFLAGS)

sort_simd: sort_simd_main
	ln -sf sort_simd_main sort_simd

# sort_simd_kv (key-value)
sort_simd_kv.o: sort_simd_kv.c sort_simd_kv.h
	$(CC) $(CFLAGS) -c sort_simd_kv.c -o sort_simd_kv.o

sort_simd_kv_main.o: sort_simd_kv_main.c sort_simd_kv.h
	$(CC) $(CFLAGS) -c sort_simd_kv_main.c -o sort_simd_kv_main.o

sort_simd_kv_main: sort_simd_kv.o sort_simd_kv_main.o utils.o merge.o
	$(CC) $(CFLAGS) sort_simd_kv_main.o sort_simd_kv.o utils.o merge.o -o sort_simd_kv_main $(LDFLAGS)

sort_simd_kv: sort_simd_kv_main
	ln -sf sort_simd_kv_main sort_simd_kv

# Tests
test_merge: test_merge.c merge.o utils.o
	$(CC) $(CFLAGS) test_merge.c merge.o utils.o -o test_merge $(LDFLAGS)

test_sort_network: test_sort_network.c merge.o
	$(CC) $(CFLAGS) test_sort_network.c merge.o -o test_sort_network $(LDFLAGS)

test: test_merge test_sort_network
	./test_merge
	./test_sort_network

clean:
	rm -f *.o sort_simd_main sort_simd sort_simd_kv_main sort_simd_kv test_merge test_sort_network
