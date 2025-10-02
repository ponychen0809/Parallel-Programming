# 編譯器
CC = gcc

# 編譯選項
# -O3: 最佳化
# -Wall -Wextra: 打開警告
# -pthread: 使用 Pthreads
CFLAGS = -O3 -Wall -Wextra -pthread -ffast-math

# 目標執行檔
TARGET = pi.out

# 預設規則（打 make 就會執行）
all: $(TARGET)

# 建立 pi.out
$(TARGET): pi.c
	$(CC) $(CFLAGS) -o $(TARGET) pi.c

# 清除編譯產物
clean:
	rm -f $(TARGET) *.o
