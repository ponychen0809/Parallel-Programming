# single portable build
TARGET := pi.out
SRC    := pi.c

CFLAGS := -O3 -march=x86-64 -mtune=generic \
          -mno-avx -mno-avx2 -mno-fma \
          -fstrict-aliasing -fno-math-errno -fno-trapping-math \
          -DNDEBUG -Wall -Wextra -Wno-unused-parameter
LIBS   := -pthread

all: $(TARGET)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)
clean:
	rm -f $(TARGET) *.o
