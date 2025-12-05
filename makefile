CC = gcc
CFLAGS = -Wall -O2 -mavx512f
LDFLAGS = -pthread

all: sorting merge threaded_devide improved_split

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o utils.o

merge.o: merge.c merge.h
	$(CC) $(CFLAGS) -c merge.c -o merge.o

sorting: sorting.c utils.o
	$(CC) $(CFLAGS) sorting.c utils.o -o sorting $(LDFLAGS)

merge: merge.c
	$(CC) $(CFLAGS) merge.c -o merge $(LDFLAGS)

test_merge: test_merge.c merge.o utils.o
	$(CC) $(CFLAGS) test_merge.c merge.o utils.o -o test_merge $(LDFLAGS)

threaded_devide: sorting_threaded_devide.c utils.o merge.o
	$(CC) $(CFLAGS) sorting_threaded_devide.c utils.o merge.o -o threaded_devide $(LDFLAGS)

improved_split: improved_split.c utils.o
	$(CC) $(CFLAGS) improved_split.c utils.o -o improved_split $(LDFLAGS)

optimized: optimized.c 
	$(CC) $(CFLAGS) optimized.c -o optimized $(LDFLAGS)

clean:
	rm -f utils.o merge.o sorting merge threaded_devide improved_split optimized

github:
	-git commit -a
	git push origin main
