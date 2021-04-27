CFLAGS = -ansi -Wall -Wextra -Wno-unused-parameter -pedantic-errors -lm

O_DEBUG := 0
O_RELEASE := 0

ifeq ($(O_DEBUG),1)
	CFLAGS += -g
endif

ifeq ($(O_RELEASE),1)
	CFLAGS += -O3 -flto -march=native -pipe -fno-plt
endif

hw01: kmeans.c
	$(CC) $(CFLAGS) -o $@ $^

clean :
	$(RM) hw01

.PHONY : clean

zip: 325711942_213056674_project.zip
325711942_213056674_project.zip: tasks.py main.py clustering_algs.py mat_algs.py numpy_utils.py basic_utils.py
	zip $@ $^
