CC = gcc
CFLAGS = -std=gnu11 -O3 -march=native -mtune=native -funroll-loops -DNDEBUG \
         -Wall -Wextra -pthread
LDFLAGS = -pthread

all: pi.out
pi.out: pi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f pi.out
