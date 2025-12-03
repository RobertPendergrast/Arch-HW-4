CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -pthread

all: sorting merge threaded_devide improved_split

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c -o utils.o

merge.o: merge.c merge.h
	$(CC) $(CFLAGS) -c merge.c -o merge.o

sorting: sorting.c utils.o utils.h
	$(CC) $(CFLAGS) sorting.c utils.o -o sorting $(LDFLAGS)

merge: merge.c utils.o utils.h
	$(CC) $(CFLAGS) merge.c utils.o -o merge $(LDFLAGS)

threaded_devide: sorting_threaded_devide.c utils.o merge.o utils.h 
	$(CC) $(CFLAGS) sorting_threaded_devide.c utils.o merge.o -o threaded_devide $(LDFLAGS)

improved_split: improved_split.c utils.o utils.h
	$(CC) $(CFLAGS) improved_split.c utils.o -o improved_split $(LDFLAGS)

clean:
	rm -f utils.o merge.o sorting merge threaded_devide improved_split
