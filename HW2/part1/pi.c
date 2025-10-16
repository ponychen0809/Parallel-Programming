// pi.c — Monte Carlo π with Pthreads + AVX2 8-lane vectorization
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <stdalign.h>   // ★ 必加：提供 alignas()

#ifdef __AVX2__
#include <immintrin.h>
#endif

// 64-bit LCG: 1 mul + 1 add
static inline __attribute__((always_inline, hot))
uint64_t fast_lcg(uint64_t *s) {
    *s = (*s * 6364136223846793005ULL + 1ULL);
    return *s;
}

// mix64: 擴散種子
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
    int64_t X = x, Y = y;
    return (uint64_t)(X*X) + (uint64_t)(Y*Y);
}

static void* worker(void *arg) {
    Task *t = (Task*)arg;
    long long n = t->tosses;
    uint64_t  st = t->state;
    long long hits = 0;

    static const int64_t  R  = 2147483647LL;
    static const uint64_t R2 = (uint64_t)R * (uint64_t)R;
    long long i = 0;

#ifdef __AVX2__
    const __m256i vR2 = _mm256_set1_epi64x((long long)R2);

    for (; i + 7 < n; i += 8) {
        uint64_t r1 = fast_lcg(&st), r2 = fast_lcg(&st);
        uint64_t r3 = fast_lcg(&st), r4 = fast_lcg(&st);
        uint64_t r5 = fast_lcg(&st), r6 = fast_lcg(&st);
        uint64_t r7 = fast_lcg(&st), r8 = fast_lcg(&st);

        alignas(32) int32_t x[8] = {
            (int32_t)r1,(int32_t)r2,(int32_t)r3,(int32_t)r4,
            (int32_t)r5,(int32_t)r6,(int32_t)r7,(int32_t)r8
        };
        alignas(32) int32_t y[8] = {
            (int32_t)(r1>>32),(int32_t)(r2>>32),(int32_t)(r3>>32),(int32_t)(r4>>32),
            (int32_t)(r5>>32),(int32_t)(r6>>32),(int32_t)(r7>>32),(int32_t)(r8>>32)
        };

        __m256i vx = _mm256_load_si256((const __m256i*)x);
        __m256i vy = _mm256_load_si256((const __m256i*)y);

        // 偶數 lanes: 0,2,4,6
        __m256i xx_even = _mm256_mul_epi32(vx, vx);
        __m256i yy_even = _mm256_mul_epi32(vy, vy);

        // 奇數 lanes: 移位後再乘
        __m256i vx_odd = _mm256_srli_si256(vx, 4);
        __m256i vy_odd = _mm256_srli_si256(vy, 4);
        __m256i xx_odd = _mm256_mul_epi32(vx_odd, vx_odd);
        __m256i yy_odd = _mm256_mul_epi32(vy_odd, vy_odd);

        __m256i sum_even = _mm256_add_epi64(xx_even, yy_even);
        __m256i sum_odd  = _mm256_add_epi64(xx_odd, yy_odd);

        __m256i m_even = _mm256_or_si256(
            _mm256_cmpgt_epi64(vR2, sum_even),
            _mm256_cmpeq_epi64(vR2, sum_even)
        );
        __m256i m_odd = _mm256_or_si256(
            _mm256_cmpgt_epi64(vR2, sum_odd),
            _mm256_cmpeq_epi64(vR2, sum_odd)
        );

        int me = _mm256_movemask_epi8(m_even);
        int mo = _mm256_movemask_epi8(m_odd);

        hits += ((me & 0x000000FF) == 0x000000FF);
        hits += ((me & 0x0000FF00) == 0x0000FF00);
        hits += ((me & 0x00FF0000) == 0x00FF0000);
        hits += ((me & 0xFF000000) == 0xFF000000);
        hits += ((mo & 0x000000FF) == 0x000000FF);
        hits += ((mo & 0x0000FF00) == 0x0000FF00);
        hits += ((mo & 0x00FF0000) == 0x00FF0000);
        hits += ((mo & 0xFF000000) == 0xFF000000);
    }
#endif

    // fallback / tail
    for (; i + 7 < n; i += 8) {
        uint64_t r1 = fast_lcg(&st), r2 = fast_lcg(&st);
        uint64_t r3 = fast_lcg(&st), r4 = fast_lcg(&st);
        uint64_t r5 = fast_lcg(&st), r6 = fast_lcg(&st);
        uint64_t r7 = fast_lcg(&st), r8 = fast_lcg(&st);
        int32_t x1=(int32_t)r1, y1=(int32_t)(r1>>32);
        int32_t x2=(int32_t)r2, y2=(int32_t)(r2>>32);
        int32_t x3=(int32_t)r3, y3=(int32_t)(r3>>32);
        int32_t x4=(int32_t)r4, y4=(int32_t)(r4>>32);
        int32_t x5=(int32_t)r5, y5=(int32_t)(r5>>32);
        int32_t x6=(int32_t)r6, y6=(int32_t)(r6>>32);
        int32_t x7=(int32_t)r7, y7=(int32_t)(r7>>32);
        int32_t x8=(int32_t)r8, y8=(int32_t)(r8>>32);
        hits += (sqsum64(x1,y1) <= R2) + (sqsum64(x2,y2) <= R2)
              + (sqsum64(x3,y3) <= R2) + (sqsum64(x4,y4) <= R2)
              + (sqsum64(x5,y5) <= R2) + (sqsum64(x6,y6) <= R2)
              + (sqsum64(x7,y7) <= R2) + (sqsum64(x8,y8) <= R2);
    }

    switch ((int)(n - i)) {
        case 7: { uint64_t r = fast_lcg(&st); hits += (sqsum64((int32_t)r,(int32_t)(r>>32)) <= R2); }
        case 6: { uint64_t r = fast_lcg(&st); hits += (sqsum64((int32_t)r,(int32_t)(r>>32)) <= R2); }
        case 5: { uint64_t r = fast_lcg(&st); hits += (sqsum64((int32_t)r,(int32_t)(r>>32)) <= R2); }
        case 4: { uint64_t r = fast_lcg(&st); hits += (sqsum64((int32_t)r,(int32_t)(r>>32)) <= R2); }
        case 3: { uint64_t r = fast_lcg(&st); hits += (sqsum64((int32_t)r,(int32_t)(r>>32)) <= R2); }
        case 2: { uint64_t r = fast_lcg(&st); hits += (sqsum64((int32_t)r,(int32_t)(r>>32)) <= R2); }
        case 1: { uint64_t r = fast_lcg(&st); hits += (sqsum64((int32_t)r,(int32_t)(r>>32)) <= R2); }
        default: break;
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
    int num_threads = atoi(argv[1]);
    long long num_tosses = atoll(argv[2]);
    if (num_threads <= 0 || num_tosses < 0) return 1;

    pthread_t *ths = malloc(sizeof(pthread_t)*num_threads);
    Task *tasks = aligned_alloc(64, sizeof(Task)*num_threads);
    if (!ths || !tasks) { perror("alloc"); return 1; }

    long long base = num_tosses / num_threads;
    int rem = num_tosses % num_threads;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t base_seed = ((uint64_t)ts.tv_sec<<32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)getpid()<<16);

    for (int i=0;i<num_threads;i++) {
        tasks[i].tosses = base + (i<rem);
        uint64_t si = mix64(base_seed ^ (0x9E3779B97F4A7C15ULL*(i+1ULL)));
        if (si==0) si = 0x106689D45497FDB5ULL;
        tasks[i].state = si;
        tasks[i].hits = 0;
        pthread_create(&ths[i],NULL,worker,&tasks[i]);
    }

    long long total=0;
    for (int i=0;i<num_threads;i++){
        pthread_join(ths[i],NULL);
        total += tasks[i].hits;
    }
    free(ths); free(tasks);

    double pi = (num_tosses>0)? 4.0*(double)total/(double)num_tosses:0.0;
    printf("%.6f\n",pi);
    return 0;
}
