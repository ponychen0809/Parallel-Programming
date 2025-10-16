CC      := gcc
CFLAGS  := -O3 -march=native -mtune=native -Wall -Wextra -pthread
LDFLAGS := -pthread
TARGET  := pi.out
SRC     := pi.c

all: $(TARGET)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	@./$(TARGET) 6 100000000

clean:
	$(RM) $(TARGET)
