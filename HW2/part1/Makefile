CC = gcc
CFLAGS = -O3 -march=native -fno-math-errno -ffast-math -Wall -Wextra -pthread
TARGET = pi.out

all: $(TARGET)

$(TARGET): pi.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) *.o
