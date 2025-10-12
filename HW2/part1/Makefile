CC = gcc
CFLAGS = -std=gnu11 -Ofast -march=native -mtune=native -funroll-loops -flto \
         -fomit-frame-pointer -fno-asynchronous-unwind-tables -DNDEBUG \
         -Wall -Wextra -pthread
LDFLAGS = -pthread

all: pi.out
pi.out: pi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f pi.out
