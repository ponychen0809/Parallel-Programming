CC = gcc
CFLAGS = -std=gnu11 -Ofast -funroll-loops -flto \
         -fomit-frame-pointer -fno-asynchronous-unwind-tables -DNDEBUG \
         -Wall -Wextra -pthread -mavx2 -march=x86-64-v3 -mtune=skylake
LDFLAGS = -pthread

all: pi.out
pi.out: pi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f pi.out
