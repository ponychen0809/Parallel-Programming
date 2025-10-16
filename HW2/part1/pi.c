// pi.c — Monte Carlo π with Pthreads + super-fast LCG RNG
// 整數幾何；LCG；unroll×8；cacheline padding；Linux thread affinity
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>   // pthread_setaffinity_np
#endif

static inline __attribute__((always_inline, hot))
uint64_t fast_lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL + 1ULL);  // 1 mul + 1 add
    return *s;
}

static inline __attribute__((always_inline, hot))
uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

typedef struct {
    long long tosses;
    uint64_t  state;
    long long hits;
    char      pad[64];
} __attribute__((aligned(64))) Task;

static inline __attribute__((always_inline, hot))
uint64_t sqsum64(int32_t x, int32_t y) {
    int64_t X = (int64_t)x, Y = (int64_t)y;
    return (uint64_t)(X*X) + (uint64_t)(Y*Y);
}

static void* worker(void *arg) {
    Task *t = (Task*)arg;
    long long n = t->tosses;
    long long hits = 0;
    uint64_t st = t->state;

    const int64_t  R  = 2147483647LL;                 // 2^31-1
    const uint64_t R2 = (uint64_t)R * (uint64_t)R;

    long long i = 0;
    for (; i + 7 < n; i += 8) {
        uint64_t r1 = fast_lcg(&st), r2 = fast_lcg(&st);
        uint64_t r3 = fast_lcg(&st), r4 = fast_lcg(&st);
        uint64_t r5 = fast_lcg(&st), r6 = fast_lcg(&st);
        uint64_t r7 = fast_lcg(&st), r8 = fast_lcg(&st);

        int32_t x1 = (int32_t)(r1), y1 = (int32_t)(r1 >> 32);
        int32_t x2 = (int32_t)(r2), y2 = (int32_t)(r2 >> 32);
        int32_t x3 = (int32_t)(r3), y3 = (int32_t)(r3 >> 32);
        int32_t x4 = (int32_t)(r4), y4 = (int32_t)(r4 >> 32);
        int32_t x5 = (int32_t)(r5), y5 = (int32_t)(r5 >> 32);
        int32_t x6 = (int32_t)(r6), y6 = (int32_t)(r6 >> 32);
        int32_t x7 = (int32_t)(r7), y7 = (int32_t)(r7 >> 32);
        int32_t x8 = (int32_t)(r8), y8 = (int32_t)(r8 >> 32);

        hits += (sqsum64(x1,y1) <= R2);
        hits += (sqsum64(x2,y2) <= R2);
        hits += (sqsum64(x3,y3) <= R2);
        hits += (sqsum64(x4,y4) <= R2);
        hits += (sqsum64(x5,y5) <= R2);
        hits += (sqsum64(x6,y6) <= R2);
        hits += (sqsum64(x7,y7) <= R2);
        hits += (sqsum64(x8,y8) <= R2);
    }

    for (; i + 1 < n; i += 2) {
        uint64_t r1 = fast_lcg(&st), r2 = fast_lcg(&st);
        int32_t x1 = (int32_t)(r1), y1 = (int32_t)(r1 >> 32);
        int32_t x2 = (int32_t)(r2), y2 = (int32_t)(r2 >> 32);
        hits += (sqsum64(x1,y1) <= R2);
        hits += (sqsum64(x2,y2) <= R2);
    }

    if (i < n) {
        uint64_t r = fast_lcg(&st);
        int32_t x = (int32_t)(r), y = (int32_t)(r >> 32);
        hits += (sqsum64(x,y) <= R2);
    }

    t->state = st;
    t->hits  = hits;
    return NULL;
}

#ifdef __linux__
// 將 pthread 綁定到某個 CPU（第 cpu_id 個）
static inline void bind_thread_to_cpu(pthread_t th, int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    // 忽略錯誤即可（在部分容器/系統上可能無法設置）
    pthread_setaffinity_np(th, sizeof(cpu_set_t), &cpuset);
}
#endif

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

    size_t need  = sizeof(Task) * (size_t)num_threads;
    size_t bytes = (need + 63) & ~((size_t)63);
    Task *tasks = (Task*)aligned_alloc(64, bytes);
    if (!tasks) { perror("aligned_alloc"); return 1; }

    long long base = num_tosses / num_threads;
    int       rem  = (int)(num_tosses % num_threads);

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t base_seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid() << 16);

#ifdef __linux__
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 1) ncpu = 1;
#endif

    for (int i = 0; i < num_threads; ++i) {
        tasks[i].tosses = base + (i < rem ? 1 : 0);
        uint64_t si = mix64(base_seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
        if (si == 0) si = 0x106689D45497FDB5ULL;   // LCG 狀態不可為 0
        tasks[i].state = si;
        tasks[i].hits  = 0;

        if (pthread_create(&ths[i], NULL, worker, &tasks[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
#ifdef __linux__
        // 綁定到不同 CPU，降低抖動（若系統允許）
        bind_thread_to_cpu(ths[i], (int)(i % ncpu));
#endif
    }

    long long total_hits = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(ths[i], NULL);
        total_hits += tasks[i].hits;
    }

    free(ths);
    free(tasks);

    double pi = (num_tosses > 0) ? (4.0 * (double)total_hits / (double)num_tosses) : 0.0;
    printf("%.6f\n", pi);    
    return 0;
}
