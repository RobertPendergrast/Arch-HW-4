merge.o: utils.c merge.c
	cc merge.c -o merge.o

threaded_devide: utils sorting_threaded_devide.c
	cc sorting_threaded_devide.c -o threaded_devide -pthread