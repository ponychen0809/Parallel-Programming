// Single-thread Monte Carlo π (AVX2 vectorized, 8 points/batch)
// - AVX2: 每回合同時判斷 8 個點是否在圓內
// - LCG: 超快 64-bit 線性同餘 PRNG
// - U[0,1) 轉換：用位元拼裝把 32-bit 整數變成 [1,2) 浮點後 -1.0f -> [0,1)
// 介面：./pi_avx <num_tosses: long long>
// 輸出：只印 π 的估計值（六位小數）+ 換行

#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// ----------- 快速 PRNG：64-bit LCG（每次 1 乘 1 加） -----------
static inline __attribute__((always_inline))
uint64_t lcg64(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL + 1ULL);
    return *s;
}

// ----------- 混 seed（SplitMix64 風格） -----------
static inline __attribute__((always_inline))
uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ----------- 產生 8 個 32-bit 隨機整數（打包成 __m256i） -----------
static inline __attribute__((always_inline))
__m256i lcg8_u32(uint64_t *state) {
    alignas(32) uint32_t buf[8];
    // 逐一前進 LCG，填入 8 個 32-bit（取低 32 位即可）
    for (int i = 0; i < 8; ++i) buf[i] = (uint32_t)lcg64(state);
    return _mm256_load_si256((const __m256i*)buf);
}

// ----------- 把 8 個 uint32 映射為 U[0,1) 的 float -----------
static inline __attribute__((always_inline))
__m256 u01_from_u32(__m256i u32) {
    // 將 u32 的低 23 位當 mantissa，和 0x3F800000 (1.0f 的 exponent) OR
    // 得到 [1,2) 的 IEEE-754 float，再減 1.0f -> [0,1)
    const __m256i mant_mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i one_exp   = _mm256_set1_epi32(0x3F800000);
    __m256i mant = _mm256_and_si256(u32, mant_mask);
    __m256i bits = _mm256_or_si256(mant, one_exp);
    __m256  f    = _mm256_castsi256_ps(bits);
    return _mm256_sub_ps(f, _mm256_set1_ps(1.0f));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_tosses:long long>\n", argv[0]);
        return 1;
    }
    long long num_tosses = atoll(argv[1]);
    if (num_tosses < 0) { fprintf(stderr, "Invalid num_tosses.\n"); return 1; }
    if (num_tosses == 0) { printf("0.000000\n"); return 0; }

    // 準備 seed（時間 + PID -> mix）
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t base_seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid() << 16);
    uint64_t state = mix64(base_seed);
    if (state == 0) state = 0x106689D45497FDB5ULL;

    const long long batches = num_tosses / 8; // 主要向量化路徑：一次 8 點
    const int       tail    = (int)(num_tosses % 8);

    long long hits = 0;

    // 常數向量
    const __m256 one = _mm256_set1_ps(1.0f);

    // --- 向量化主迴圈：每批 8 個點 ---
    for (long long b = 0; b < batches; ++b) {
        // 產 8 個 U[0,1) 的 x、y
        __m256i ux = lcg8_u32(&state);
        __m256i uy = lcg8_u32(&state);
        __m256  x  = u01_from_u32(ux);
        __m256  y  = u01_from_u32(uy);

        // d = x*x + y*y
        __m256 xx = _mm256_mul_ps(x, x);
        __m256 yy = _mm256_mul_ps(y, y);
        __m256 d  = _mm256_add_ps(xx, yy);

        // 比較 d <= 1.0f → 8-bit mask（bit=1 表示在圓內）
        __m256 cmp = _mm256_cmp_ps(d, one, _CMP_LE_OS);
        int mask   = _mm256_movemask_ps(cmp);

        // 計算 mask 內 1 的個數（0..8）
        hits += __builtin_popcount((unsigned)mask);
    }

    // --- 尾端（不足 8 個）的標量處理（沿用同一 RNG & 轉換方法） ---
    for (int i = 0; i < tail; ++i) {
        uint32_t rx = (uint32_t)lcg64(&state);
        uint32_t ry = (uint32_t)lcg64(&state);

        // 單筆用同樣的位元拼裝法
        uint32_t bx = (rx & 0x007FFFFF) | 0x3F800000;
        uint32_t by = (ry & 0x007FFFFF) | 0x3F800000;
        float fx = *((float*)&bx) - 1.0f;  // U[0,1)
        float fy = *((float*)&by) - 1.0f;  // U[0,1)

        float d = fx*fx + fy*fy;
        hits += (d <= 1.0f);
    }

    double pi = 4.0 * (double)hits / (double)num_tosses;
    printf("%.6f\n", pi);
    return 0;
}
