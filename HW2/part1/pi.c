// pi.c — Monte Carlo π with Pthreads + xorshift64*
// 整數幾何：兩次 RNG 產兩個點 (x1,y1),(x2,y2)；避免浮點轉換與 FPU 開銷
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

// ----- xorshift64* -----
static inline __attribute__((always_inline))
uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

// SplitMix64 風格種子混合（打散）
static inline __attribute__((always_inline))
uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// 避免 false sharing：64B 對齊 + padding
typedef struct {
    long long tosses;
    uint64_t  state;
    long long hits;
    char      pad[64];
} __attribute__((aligned(64))) Task;

// 每個 thread 的工作：整數幾何 + 兩點/迴圈
static void* worker(void *arg) {
    Task *t = (Task*)arg;
    long long n = t->tosses;
    long long hits = 0;
    uint64_t st = t->state;

    // 半徑 R=2^31-1；用 128-bit 做加總比較避免邊界顧慮
    const long long  R  = 2147483647LL;           // 2^31-1
    const __int128   R2 = (__int128)R * R;

    long long i = 0;
    for (; i + 1 < n; i += 2) {
        // 2 次 RNG → 4 個 32-bit → 兩個點
        uint64_t r1 = xs64(&st);
        uint64_t r2 = xs64(&st);

        int32_t x1 = (int32_t)(r1 & 0xFFFFFFFFu);
        int32_t y1 = (int32_t)(r1 >> 32);
        int32_t x2 = (int32_t)(r2 & 0xFFFFFFFFu);
        int32_t y2 = (int32_t)(r2 >> 32);

        __int128 s1 = (__int128)(int64_t)x1 * x1 + (__int128)(int64_t)y1 * y1;
        __int128 s2 = (__int128)(int64_t)x2 * x2 + (__int128)(int64_t)y2 * y2;

        hits += (s1 <= R2);
        hits += (s2 <= R2);
    }
    // 奇數收尾
    if (i < n) {
        uint64_t r = xs64(&st);
        int32_t x = (int32_t)(r & 0xFFFFFFFFu);
        int32_t y = (int32_t)(r >> 32);
        __int128 s = (__int128)(int64_t)x * x + (__int128)(int64_t)y * y;
        hits += (s <= R2);
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
    Task *tasks = NULL;
#if defined(_ISOC11_SOURCE) || __STDC_VERSION__ >= 201112L
    tasks = aligned_alloc(64, sizeof(Task) * (size_t)num_threads);
    if (!tasks) tasks = (Task*)malloc(sizeof(Task) * (size_t)num_threads);
#else
    if (posix_memalign((void**)&tasks, 64, sizeof(Task) * (size_t)num_threads) != 0)
        tasks = (Task*)malloc(sizeof(Task) * (size_t)num_threads);
#endif
    if (!ths || !tasks) { perror("alloc"); return 1; }

    long long base = num_tosses / num_threads;
    int       rem  = (int)(num_tosses % num_threads);

    // 基礎 seed：時間 + PID
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t base_seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid() << 16);

    for (int i = 0; i < num_threads; ++i) {
        tasks[i].tosses = base + (i < rem ? 1 : 0);
        uint64_t si = mix64(base_seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
        if (si == 0) si = 0x106689D45497FDB5ULL;   // xorshift 狀態不可為 0
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

    // 估計 π：仍為 4 * hits / tosses（正方形內均勻拋點不變）
    double pi = (num_tosses > 0) ? (4.0 * (double)total_hits / (double)num_tosses) : 0.0;
    printf("%.6f\n", pi);    // 僅數字 + 換行（評測要求）
    return 0;
}
