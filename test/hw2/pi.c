// pi.c -- Pthreads Monte Carlo Pi (LCG-64, MMIX constants)
// Usage: ./pi.out <num_threads:int> <num_tosses:long long>
// Output: one line with the estimated PI (e.g., 3.141592)

#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

typedef unsigned long long ull;

/* -------- splitmix64: 用於播種（高擾動，避免相關性） -------- */
static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* -------- 64-bit LCG (mod 2^64) — MMIX constants --------
   X_{n+1} = a * X_n + c (mod 2^64)
   a = 6364136223846793005, c = 1442695040888963407
   注意：低位統計性差，所以轉 double 時取高 53 bits。
*/
typedef struct { uint64_t s; } lcg64_state;

static inline uint64_t lcg64_next(lcg64_state *st) {
    st->s = st->s * 6364136223846793005ULL + 1442695040888963407ULL;
    return st->s;
}

/* 轉 [0,1) double：取高 53 bits / 2^53，避開低位相關性 */
static inline double u01_double_lcg(lcg64_state *st) {
    return (lcg64_next(st) >> 11) * (1.0 / 9007199254740992.0);
}

typedef struct {
    ull tosses;
    ull local_hits;
    lcg64_state rng;
} Task;

void* worker(void* arg) {
    Task* t = (Task*)arg;
    lcg64_state st = t->rng;
    ull hits = 0;

    // 小幅迴圈展開以降低分支與函式呼叫開銷
    ull i = 0, n = t->tosses;
    for (; i + 7 < n; i += 8) {
        double x0 = u01_double_lcg(&st), y0 = u01_double_lcg(&st);
        double x1 = u01_double_lcg(&st), y1 = u01_double_lcg(&st);
        double x2 = u01_double_lcg(&st), y2 = u01_double_lcg(&st);
        double x3 = u01_double_lcg(&st), y3 = u01_double_lcg(&st);
        if (x0*x0 + y0*y0 <= 1.0) ++hits;
        if (x1*x1 + y1*y1 <= 1.0) ++hits;
        if (x2*x2 + y2*y2 <= 1.0) ++hits;
        if (x3*x3 + y3*y3 <= 1.0) ++hits;

        double x4 = u01_double_lcg(&st), y4 = u01_double_lcg(&st);
        double x5 = u01_double_lcg(&st), y5 = u01_double_lcg(&st);
        double x6 = u01_double_lcg(&st), y6 = u01_double_lcg(&st);
        double x7 = u01_double_lcg(&st), y7 = u01_double_lcg(&st);
        if (x4*x4 + y4*y4 <= 1.0) ++hits;
        if (x5*x5 + y5*y5 <= 1.0) ++hits;
        if (x6*x6 + y6*y6 <= 1.0) ++hits;
        if (x7*x7 + y7*y7 <= 1.0) ++hits;
    }
    for (; i < n; ++i) {
        double x = u01_double_lcg(&st);
        double y = u01_double_lcg(&st);
        if (x*x + y*y <= 1.0) ++hits;
    }

    t->local_hits = hits;
    t->rng = st; // 保持嚴謹（非必要）
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads:int> <num_tosses:long long>\n", argv[0]);
        return 1;
    }
    int num_threads = atoi(argv[1]);
    if (num_threads <= 0) { fprintf(stderr, "num_threads must be > 0\n"); return 1; }

    char* endp = NULL;
    errno = 0;
    ull total_tosses = strtoull(argv[2], &endp, 10);
    if (errno || endp == argv[2]) { fprintf(stderr, "invalid num_tosses\n"); return 1; }

    pthread_t* th = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    Task* tasks    = (Task*)malloc(sizeof(Task) * num_threads);

    // 均分 tosses
    ull base = total_tosses / (ull)num_threads;
    ull rem  = total_tosses % (ull)num_threads;

    // 每執行緒獨立 seed（用 splitmix64 從全域種子衍生）
    uint64_t g = (uint64_t)time(NULL) ^ 0xA5A5A5A5A5A5A5A5ULL;
    for (int i = 0; i < num_threads; ++i) {
        tasks[i].tosses = base + (i < (int)rem ? 1 : 0);
        uint64_t mix = (uint64_t)i * 0x9E3779B97F4A7C15ULL ^ g;
        tasks[i].rng.s = splitmix64(&mix);
        tasks[i].local_hits = 0;
        pthread_create(&th[i], NULL, worker, &tasks[i]);
    }

    ull hits = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(th[i], NULL);
        hits += tasks[i].local_hits;
    }

    free(th);
    free(tasks);

    double pi = 4.0 * (double)hits / (double)total_tosses;
    printf("%.6f\n", pi); // 只印數字與換行
    return 0;
}
