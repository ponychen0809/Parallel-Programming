#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    long long tosses;       // 該 thread 要做的丟點次數
    unsigned short seed[3]; // 每個 thread 獨立隨機種子
    long long hits;         // 命中次數
} Task;

void* worker(void* arg) {
    Task* t = (Task*)arg;
    long long local_hits = 0;

    for (long long i = 0; i < t->tosses; i++) {
        double x = erand48(t->seed);
        double y = erand48(t->seed);
        if (x*x + y*y <= 1.0) {
            local_hits++;
        }
    }

    t->hits = local_hits;
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads> <num_tosses>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    long long num_tosses = atoll(argv[2]);

    pthread_t threads[num_threads];
    Task tasks[num_threads];

    long long base = num_tosses / num_threads;
    long long rem  = num_tosses % num_threads;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    unsigned long base_seed = (unsigned long)ts.tv_nsec ^ getpid();

    // 建立 thread
    for (int i = 0; i < num_threads; i++) {
        tasks[i].tosses = base + (i < rem ? 1 : 0);
        unsigned long s = base_seed ^ (i+1)*12345UL;
        tasks[i].seed[0] = (unsigned short)(s & 0xFFFF);
        tasks[i].seed[1] = (unsigned short)((s >> 16) & 0xFFFF);
        tasks[i].seed[2] = (unsigned short)((s >> 32) & 0xFFFF);
        tasks[i].hits = 0;
        pthread_create(&threads[i], NULL, worker, &tasks[i]);
    }

    // 等待 thread 結束
    long long total_hits = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_hits += tasks[i].hits;
    }

    double pi_estimate = 4.0 * (double)total_hits / (double)num_tosses;
    printf("%.6f\n", pi_estimate);

    return 0;
}
