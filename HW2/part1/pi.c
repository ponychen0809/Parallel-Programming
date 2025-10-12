// pi.c — Monte Carlo π with Pthreads + super-fast LCG RNG
// 整數幾何；2 RNG→2 點；unroll×4；64-bit 計算；cacheline padding
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

/* ---------- 超快 RNG：64-bit LCG ---------- */
static inline __attribute__((always_inline))
uint64_t fast_lcg(uint64_t *s) {
    // 常用高速 LCG；每次只做 1 乘 1 加
    *s = (*s * 6364136223846793005ULL + 1ULL);
    return *s;
}

/* ---------- 種子混合（SplitMix64 風格） ---------- */
static inline __attribute__((always_inline))
uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---------- 避免 false sharing：64B 對齊 + padding ---------- */
typedef struct {
    long long tosses;
    uint64_t  state;
    long long hits;
    char      pad[64];
} __attribute__((aligned(64))) Task;

/* ---------- 64-bit 平方和（不溢位） ---------- */
static inline __attribute__((always_inline))
uint64_t sqsum64(int32_t x, int32_t y) {
    int64_t X = (int64_t)x, Y = (int64_t)y;
    return (uint64_t)(X*X) + (uint64_t)(Y*Y);
}

/* ---------- worker：整數幾何 + 手動 unroll ×4 ---------- */
static void* worker(void *arg) {
    Task *t = (Task*)arg;
    long long n = t->tosses;
    long long hits = 0;
    uint64_t st = t->state;

    const int64_t  R  = 2147483647LL;                 // 2^31-1
    const uint64_t R2 = (uint64_t)R * (uint64_t)R;

    long long i = 0;
    // 主迴圈：一輪處理 4 個點（4 次 RNG → 8 個 32-bit → 4 點）
    for (; i + 3 < n; i += 4) {
        uint64_t r1 = fast_lcg(&st);
        uint64_t r2 = fast_lcg(&st);
        uint64_t r3 = fast_lcg(&st);
        uint64_t r4 = fast_lcg(&st);

        int32_t x1 = (int32_t)(r1 & 0xFFFFFFFFu), y1 = (int32_t)(r1 >> 32);
        int32_t x2 = (int32_t)(r2 & 0xFFFFFFFFu), y2 = (int32_t)(r2 >> 32);
        int32_t x3 = (int32_t)(r3 & 0xFFFFFFFFu), y3 = (int32_t)(r3 >> 32);
        int32_t x4 = (int32_t)(r4 & 0xFFFFFFFFu), y4 = (int32_t)(r4 >> 32);

        hits += (sqsum64(x1,y1) <= R2);
        hits += (sqsum64(x2,y2) <= R2);
        hits += (sqsum64(x3,y3) <= R2);
        hits += (sqsum64(x4,y4) <= R2);
    }
    // 收尾（2~3 點）
    for (; i + 1 < n; i += 2) {
        uint64_t r1 = fast_lcg(&st);
        uint64_t r2 = fast_lcg(&st);
        int32_t x1 = (int32_t)(r1 & 0xFFFFFFFFu), y1 = (int32_t)(r1 >> 32);
        int32_t x2 = (int32_t)(r2 & 0xFFFFFFFFu), y2 = (int32_t)(r2 >> 32);
        hits += (sqsum64(x1,y1) <= R2);
        hits += (sqsum64(x2,y2) <= R2);
    }
    // 單一剩餘
    if (i < n) {
        uint64_t r = fast_lcg(&st);
        int32_t x = (int32_t)(r & 0xFFFFFFFFu), y = (int32_t)(r >> 32);
        hits += (sqsum64(x,y) <= R2);
    }

    t->state = st;
    t->hits  = hits;
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads:int> <num_tosses:long long>\n", argv[0]);
        return 1;
    }
    int        num_threads = atoi(argv[1]);
    long long  num_tosses  = atoll(argv[2]);
    if (num_threads <= 0 || num_tosses < 0) {
        fprintf(stderr, "Invalid arguments.\n");
        return 1;
    }

    pthread_t *ths = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)num_threads);
    if (!ths) { perror("alloc ths"); return 1; }

    // C11 aligned_alloc：size 必須為對齊倍數 → 向上取整
    size_t need  = sizeof(Task) * (size_t)num_threads;
    size_t bytes = (need + 63) & ~((size_t)63);
    Task *tasks = (Task*)aligned_alloc(64, bytes);
    if (!tasks) { perror("aligned_alloc"); return 1; }

    long long base = num_tosses / num_threads;
    int       rem  = (int)(num_tosses % num_threads);

    // 基礎 seed：時間 + PID，再用 mix64 打散，確保各 thread 種子不同且 ≠ 0
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t base_seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid() << 16);

    for (int i = 0; i < num_threads; ++i) {
        tasks[i].tosses = base + (i < rem ? 1 : 0);
        uint64_t si = mix64(base_seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
        if (si == 0) si = 0x106689D45497FDB5ULL;
        tasks[i].state = si;
        tasks[i].hits  = 0;

        if (pthread_create(&ths[i], NULL, worker, &tasks[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    long long total_hits = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(ths[i], NULL);
        total_hits += tasks[i].hits;
    }

    free(ths);
    free(tasks);

    double pi = (num_tosses > 0) ? (4.0 * (double)total_hits / (double)num_tosses) : 0.0;
    printf("%.6f\n", pi);    // 僅數字 + 換行（評測要求）
    return 0;
}
