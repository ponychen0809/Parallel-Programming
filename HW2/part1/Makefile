TARGET := pi
SRC    := pi.c

CFLAGS := -O3 -march=native -flto -fstrict-aliasing -fno-math-errno -fno-trapping-math \
          -DNDEBUG -Wall -Wextra -Wno-unused-parameter
LIBS   := -pthread

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o pi.out $< $(LIBS)

clean:
	rm -f $(TARGET) *.out
