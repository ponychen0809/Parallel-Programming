CC      := gcc
CFLAGS  := -std=gnu11 -O3 -march=native -mtune=native -funroll-loops -flto \
           -fomit-frame-pointer -fno-exceptions -fno-asynchronous-unwind-tables -DNDEBUG
LDFLAGS := -flto -pthread

all: pi

pi: pi.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f pi
