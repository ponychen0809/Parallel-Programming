CC = gcc
CFLAGS = -std=gnu11 -Ofast -funroll-loops -flto \
         -fomit-frame-pointer -fno-asynchronous-unwind-tables -DNDEBUG \
         -Wall -Wextra -pthread -march=x86-64 -mtune=generic
LDFLAGS = -pthread

all: pi.out

pi.out: pi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f pi.out
