CC = gcc
CFLAGS = -Wall -O3 -mavx512f -mavx512bw -march=native -funroll-loops -ffast-math -fopenmp
LDFLAGS = -pthread

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o utils.o

merge.o: merge.c merge.h
	$(CC) $(CFLAGS) -c merge.c -o merge.o

sort_simd.o: sort_simd.c sort_simd.h
	$(CC) $(CFLAGS) -c sort_simd.c -o sort_simd.o

sort_simd_main.o: sort_simd_main.c sort_simd.h
	$(CC) $(CFLAGS) -c sort_simd_main.c -o sort_simd_main.o

sort_simd_main: sort_simd.o sort_simd_main.o utils.o merge.o
	$(CC) $(CFLAGS) sort_simd_main.o sort_simd.o utils.o merge.o -o sort_simd_main $(LDFLAGS)

sort_simd: sort_simd_main
	ln -sf sort_simd_main sort_simd

sort_simd_kv.o: sort_simd_kv.c sort_simd_kv.h
	$(CC) $(CFLAGS) -c sort_simd_kv.c -o sort_simd_kv.o

sort_simd_kv_main.o: sort_simd_kv_main.c sort_simd_kv.h
	$(CC) $(CFLAGS) -c sort_simd_kv_main.c -o sort_simd_kv_main.o

sort_simd_kv_main: sort_simd_kv.o sort_simd_kv_main.o utils.o merge.o
	$(CC) $(CFLAGS) sort_simd_kv_main.o sort_simd_kv.o utils.o merge.o -o sort_simd_kv_main $(LDFLAGS)

sort_simd_kv: sort_simd_kv_main
	ln -sf sort_simd_kv_main sort_simd_kv

merge: merge.c
	$(CC) $(CFLAGS) merge.c -o merge $(LDFLAGS)

test_merge: test_merge.c merge.o utils.o
	$(CC) $(CFLAGS) test_merge.c merge.o utils.o -o test_merge $(LDFLAGS)


test_merge_run: test_merge
	./test_merge



sort_radix: sort_radix.c utils.o
	$(CC) $(CFLAGS) sort_radix.c utils.o -o sort_radix $(LDFLAGS)

sort_radix_kv: sort_radix_kv.c utils.o
	$(CC) $(CFLAGS) sort_radix_kv.c utils.o -o sort_radix_kv $(LDFLAGS)

membw_test: membw_test.c
	$(CC) $(CFLAGS) membw_test.c -o membw_test $(LDFLAGS)

clean:
	rm -f utils.o merge.o sort_simd.o sort_simd sort_simd_main.o sort_simd_main sort_simd_kv.o sort_simd_kv_main.o sort_simd_kv_main sort_simd_kv sort_radix sort_radix_kv

github:
	-git commit -a
	git push origin main
