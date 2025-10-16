// pi.c — Monte Carlo π with Pthreads + super-fast LCG RNG (no SIMD)
// 整數幾何；LCG；unroll×16；cacheline padding；Duff's device 尾端處理
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

/* 直接用巨集消除函式呼叫開銷：1 mul + 1 add */
#define NEXT64(st) ( (st) = (st) * 6364136223846793005ULL + 1ULL )

/* 命中判定轉 0/1，讓編譯器更容易做無分支匯編 */
#define HIT(x,y,R2) ((uint64_t)((sqsum64((x),(y)) <= (R2))))

typedef struct {
    long long tosses;
    uint64_t  state;
    long long hits;
    char      pad[64];
} __attribute__((aligned(64))) Task;

/* 高擾動播種器（splitmix64 風格），避免種子彼此相關 */
static inline __attribute__((always_inline, hot))
uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* 64 位平方和，避免溢位（熱區請保持 inline） */
static inline __attribute__((always_inline, hot))
uint64_t sqsum64(int32_t x, int32_t y) {
    int64_t X = (int64_t)x, Y = (int64_t)y;
    return (uint64_t)(X*X) + (uint64_t)(Y*Y);
}

/* 工作者：×16 展開 + Duff's device 尾端處理 */
static void* worker(void *arg) {
    Task *t = (Task*)arg;
    long long n = t->tosses;
    long long hits = 0;
    uint64_t st = t->state;

    const int64_t  R  = 2147483647LL;                 // 2^31 - 1
    const uint64_t R2 = (uint64_t)R * (uint64_t)R;

    long long i = 0;

    /* 主迴圈：一次處理 16 組點 */
    for (; i + 15 < n; i += 16) {
        uint64_t r1 = NEXT64(st),  r2  = NEXT64(st),  r3  = NEXT64(st),  r4  = NEXT64(st);
        uint64_t r5 = NEXT64(st),  r6  = NEXT64(st),  r7  = NEXT64(st),  r8  = NEXT64(st);
        uint64_t r9 = NEXT64(st),  r10 = NEXT64(st),  r11 = NEXT64(st),  r12 = NEXT64(st);
        uint64_t r13= NEXT64(st),  r14 = NEXT64(st),  r15 = NEXT64(st),  r16 = NEXT64(st);

        int32_t x1=(int32_t)r1,   y1=(int32_t)(r1>>32);
        int32_t x2=(int32_t)r2,   y2=(int32_t)(r2>>32);
        int32_t x3=(int32_t)r3,   y3=(int32_t)(r3>>32);
        int32_t x4=(int32_t)r4,   y4=(int32_t)(r4>>32);
        int32_t x5=(int32_t)r5,   y5=(int32_t)(r5>>32);
        int32_t x6=(int32_t)r6,   y6=(int32_t)(r6>>32);
        int32_t x7=(int32_t)r7,   y7=(int32_t)(r7>>32);
        int32_t x8=(int32_t)r8,   y8=(int32_t)(r8>>32);
        int32_t x9=(int32_t)r9,   y9=(int32_t)(r9>>32);
        int32_t x10=(int32_t)r10, y10=(int32_t)(r10>>32);
        int32_t x11=(int32_t)r11, y11=(int32_t)(r11>>32);
        int32_t x12=(int32_t)r12, y12=(int32_t)(r12>>32);
        int32_t x13=(int32_t)r13, y13=(int32_t)(r13>>32);
        int32_t x14=(int32_t)r14, y14=(int32_t)(r14>>32);
        int32_t x15=(int32_t)r15, y15=(int32_t)(r15>>32);
        int32_t x16=(int32_t)r16, y16=(int32_t)(r16>>32);

        hits += HIT(x1,y1,R2)  + HIT(x2,y2,R2)  + HIT(x3,y3,R2)  + HIT(x4,y4,R2)
              + HIT(x5,y5,R2)  + HIT(x6,y6,R2)  + HIT(x7,y7,R2)  + HIT(x8,y8,R2)
              + HIT(x9,y9,R2)  + HIT(x10,y10,R2)+ HIT(x11,y11,R2)+ HIT(x12,y12,R2)
              + HIT(x13,y13,R2)+ HIT(x14,y14,R2)+ HIT(x15,y15,R2)+ HIT(x16,y16,R2);
    }

    /* 尾端處理：Duff's device（單一 switch，最少分支開銷） */
    int rem = (int)(n - i);
    switch (rem) {
        /* 注意：以下有意識的「貫穿」(fallthrough) 以一次性處理完餘數 */
        case 15: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case 14: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case 13: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case 12: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case 11: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case 10: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  9: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  8: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  7: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  6: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  5: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  4: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  3: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  2: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  1: { uint64_t r = NEXT64(st); int32_t x=(int32_t)r, y=(int32_t)(r>>32); hits += HIT(x,y,R2); } /* fallthrough */
        case  0: default: break;
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

    size_t need  = sizeof(Task) * (size_t)num_threads;
    size_t bytes = (need + 63) & ~((size_t)63);

    Task *tasks = NULL;
    if (posix_memalign((void**)&tasks, 64, bytes) != 0 || !tasks) {
        perror("posix_memalign");
        free(ths);
        return 1;
    }

    long long base = num_tosses / num_threads;
    int       rem  = (int)(num_tosses % num_threads);

    /* 簡單但足夠的 base seed（可改成 clock_gettime 混合） */
    uint64_t base_seed = (uint64_t)getpid();

    for (int i = 0; i < num_threads; ++i) {
        tasks[i].tosses = base + (i < rem ? 1 : 0);
        uint64_t si = mix64(base_seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
        if (si == 0) si = 0x106689D45497FDB5ULL;   // LCG 狀態不可為 0
        tasks[i].state = si;
        tasks[i].hits  = 0;

        if (pthread_create(&ths[i], NULL, worker, &tasks[i]) != 0) {
            perror("pthread_create");
            free(tasks);
            free(ths);
            return 1;
        }
    }

    long long total_hits = 0;
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(ths[i], NULL);
        total_hits += tasks[i].hits;
    }

    free(tasks);
    free(ths);

    double pi = (num_tosses > 0) ? (4.0 * (double)total_hits / (double)num_tosses) : 0.0;
    printf("%.6f\n", pi);
    return 0;
}
