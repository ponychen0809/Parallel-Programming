# Makefile for pi.c (AVX2 vectorized Monte Carlo π)
# 用法：
#   make            → 編譯成 pi
#   make run N=100000000  → 執行 pi，輸入投點數
#   make clean       → 清除執行檔

# ---- 編譯器 ----
CC = gcc

# ---- 編譯選項 ----
# -O3           : 高度最佳化
# -march=native : 針對本機 CPU 最佳化
# -mavx2 -mfma  : 啟用 AVX2 / FMA
# -std=c11      : 使用 C11 標準
# -Wall -Wextra : 顯示所有警告
CFLAGS = -O3 -march=native -mavx2 -mfma -std=c11 -Wall -Wextra

# ---- 目標 ----
TARGET = pi
SRC = pi.c

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

# 執行指令，例如 make run N=100000000
N ?= 10000000
run: $(TARGET)
	./$(TARGET) $(N)

clean:
	rm -f $(TARGET)
