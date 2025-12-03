utils: utils.c
	cc utils.c -o utils

merge: utils merge.c
	cc merge.c -o merge
