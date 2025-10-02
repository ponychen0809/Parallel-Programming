// pi.c — Monte Carlo π with Pthreads + xorshift64* (fast RNG)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    long long tosses;        // 該 thread 要做的丟點次數
    uint64_t   state;        // 該 thread 的 PRNG 狀態（xorshift64*）
    long long hits;          // 命中次數
} Task;

/* --------- 快速 PRNG：xorshift64* + [0,1) 轉換 ---------- */
// xorshift64*：小、快、品質對本作業足夠
static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

// 把 64-bit 隨機值轉成 [0,1) 的 double（取高 53 位）
static inline double u01(uint64_t *s) {
    return (xs64(s) >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

/* --------- 64-bit seed 混合（SplitMix64 風格） ---------- */
static inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97f4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ----------------------- worker ------------------------- */
static void* worker(void *arg) {
    Task *t = (Task*)arg;
    long long local_hits = 0;
    uint64_t st = t->state;  // 放到暫存器，加速

    for (long long i = 0; i < t->tosses; ++i) {
        double x = u01(&st);
        double y = u01(&st);
        if (x*x + y*y <= 1.0) ++local_hits;
    }

    t->state = st;     // 回寫狀態（下次可延續用；本作業其實不必）
    t->hits  = local_hits;
    return NULL;
}

/* ------------------------ main -------------------------- */
int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads:int> <num_tosses:long long>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    long long num_tosses = atoll(argv[2]);
    if (num_threads <= 0 || num_tosses < 0) {
        fprintf(stderr, "Invalid arguments.\n");
        return 1;
    }

    pthread_t *threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
    Task      *tasks   = (Task*)malloc(sizeof(Task)      * num_threads);
    if (!threads || !tasks) { perror("malloc"); return 1; }

    long long base = num_tosses / num_threads;
    long long rem  = num_tosses % num_threads;

    // 產生基礎 seed：時間秒/奈秒 + PID
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t base_seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid() << 16);

    // 建立 threads
    for (int i = 0; i < num_threads; ++i) {
        tasks[i].tosses = base + (i < rem ? 1 : 0);

        // 每個 thread 用不同 seed（混得很散）
        uint64_t si = mix64(base_seed ^ (0x9E3779B97f4A7C15ULL * (uint64_t)(i + 1)));
        // xorshift64* 的 state 不能是 0
        if (si == 0) si = 0x106689D45497FDB5ULL;
        tasks[i].state = si;

        tasks[i].hits = 0;
        if (pthread_create(&threads[i], NULL, worker, &tasks[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    long long total_hits = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        total_hits += tasks[i].hits;
    }

    free(threads);
    free(tasks);

    double pi = (num_tosses > 0) ? (4.0 * (double)total_hits / (double)num_tosses) : 0.0;
    printf("%.6f\n", pi);   // 只輸出數字 + 換行（評測要求）
    return 0;
}
