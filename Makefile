CFLAGS = -ansi -Wall -Wextra -Wno-unused-parameter -pedantic-errors -lm


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
