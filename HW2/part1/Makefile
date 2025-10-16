CC = gcc

# 共用最佳化旗標
COMMON = -std=gnu11 -Ofast -funroll-loops -flto \
         -fomit-frame-pointer -fno-asynchronous-unwind-tables -DNDEBUG \
         -Wall -Wextra -pthread

# ==== 預設：開啟 AVX2（評測機 i5-10500 支援） ====
CFLAGS = $(COMMON) -mavx2 -march=x86-64-v3 -mtune=skylake

# ---- 若在某些環境出現 Illegal instruction，改用下行可攜版 ----
# CFLAGS = $(COMMON) -march=x86-64 -mtune=generic

LDFLAGS = -pthread

all: pi.out

pi.out: pi.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f pi.out
