CC = gcc
CFLAGS = -Wall -O3 -mavx512f -mavx512bw -march=native -funroll-loops -ffast-math
LDFLAGS = -pthread

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o utils.o

merge.o: merge.c merge.h
	$(CC) $(CFLAGS) -c merge.c -o merge.o

sorting: sorting.c utils.o
	$(CC) $(CFLAGS) sorting.c utils.o -o sorting $(LDFLAGS)

sort_simd: sort_simd.c utils.o merge.o
	$(CC) $(CFLAGS) -fopenmp sort_simd.c utils.o merge.o -o sort_simd $(LDFLAGS)

sort_fast_stable: sort_fast_stable.c utils.o
	$(CC) $(CFLAGS) -fopenmp sort_fast_stable.c utils.o -o sort_fast_stable $(LDFLAGS)

merge: merge.c
	$(CC) $(CFLAGS) merge.c -o merge $(LDFLAGS)

test_merge: test_merge.c merge.o utils.o
	$(CC) $(CFLAGS) test_merge.c merge.o utils.o -o test_merge $(LDFLAGS)

test_sort_network: test_sort_network.c merge.o
	$(CC) $(CFLAGS) test_sort_network.c merge.o -o test_sort_network $(LDFLAGS)

test_merge_run: test_merge
	./test_merge

threaded_devide: sorting_threaded_devide.c utils.o merge.o
	$(CC) $(CFLAGS) sorting_threaded_devide.c utils.o merge.o -o threaded_devide $(LDFLAGS)

improved_split: improved_split.c utils.o
	$(CC) $(CFLAGS) improved_split.c utils.o -o improved_split $(LDFLAGS)

multi_sort: multi_sort.c utils.o
	$(CC) $(CFLAGS_OPT) multi_sort.c utils.o -o multi_sort $(LDFLAGS)

stable_sort_avx512.o: stable_sort_avx512.c stable_sort_avx512.h
	$(CC) $(CFLAGS) -fopenmp -c stable_sort_avx512.c -o stable_sort_avx512.o

test_stable_sort: test_stable_sort.c stable_sort_avx512.o
	$(CC) $(CFLAGS) test_stable_sort.c stable_sort_avx512.o -o test_stable_sort $(LDFLAGS)

test_stable_sort_run: test_stable_sort
	./test_stable_sort

bench_stable_sort: bench_stable_sort.c stable_sort_avx512.o utils.o
	$(CC) $(CFLAGS) -fopenmp bench_stable_sort.c stable_sort_avx512.o utils.o -o bench_stable_sort $(LDFLAGS)

sort_scaler_comparison: sort_scaler_comparison.c utils.o merge.o
	$(CC) $(CFLAGS) -fopenmp sort_scaler_comparison.c utils.o merge.o -o sort_scaler_comparison $(LDFLAGS)

clean:
	rm -f utils.o merge.o stable_sort_avx512.o sorting merge threaded_devide improved_split multi_sort test_stable_sort sort_simd sort_fast_stable test_merge test_sort_network bench_stable_sort

github:
	-git commit -a
	git push origin main
