CFLAGS = -ansi -Wall -Wextra -Wno-unused-parameter -pedantic-errors -lm

hw1: kmeans.c
	$(CC) $(CFLAGS) kmeans.c -o hw1
