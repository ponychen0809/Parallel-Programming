#include <array>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include "common/cycle_timer.h"

struct WorkerArgs
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
};

extern void mandelbrot_serial(float x0,
                              float y0,
                              float x1,
                              float y1,
                              int width,
                              int height,
                              int start_row,
                              int num_rows,
                              int max_iterations,
                              int *output);

//
// worker_thread_start --
//
// Thread entrypoint.
void worker_thread_start(WorkerArgs *const args)
{
    double t0 = CycleTimer::current_seconds();

    const int tid = args->threadId;         // 第幾個執行緒
    const int nth = args->numThreads;       // 總執行緒數
    const int H   = static_cast<int>(args->height);

    // 依列(row)等分，前面 rem 個執行緒各多拿 1 列
    const int base = (nth > 0) ? (H / nth) : H;
    const int rem  = (nth > 0) ? (H % nth) : 0;

    const int my_rows  = base + (tid < rem ? 1 : 0);
    const int my_start = tid * base + (tid < rem ? tid : rem);

    if (my_rows <= 0) return;  // 當執行緒比高度多時，部分 thread 無工作

    // 直接呼叫序列版計算自己負責的列區塊
    mandelbrot_serial(
        args->x0, args->y0, args->x1, args->y1,
        static_cast<int>(args->width),
        static_cast<int>(args->height),
        my_start, my_rows,
        args->maxIterations,
        args->output  // 共享同一塊 output；函式內用 j*width+i，全域索引，不會衝突
    );
    double t1 = CycleTimer::current_seconds();
printf("[thread %d] %.3f ms\n", args->threadId, (t1-t0)*1000.0);
}


//
// mandelbrot_thread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrot_thread(int num_threads,
                       float x0,
                       float y0,
                       float x1,
                       float y1,
                       int width,
                       int height,
                       int max_iterations,
                       int *output)
{
    static constexpr int max_threads = 32;

    if (num_threads > max_threads)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", max_threads);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::array<std::thread, max_threads> workers;
    std::array<WorkerArgs, max_threads> args = {};

    for (int i = 0; i < num_threads; i++)
    {
        // TODO FOR PP STUDENTS: You may or may not wish to modify
        // the per-thread arguments here.  The code below copies the
        // same arguments for each thread
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = max_iterations;
        args[i].numThreads = num_threads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < num_threads; i++)
    {
        workers[i] = std::thread(worker_thread_start, &args[i]);
    }

    worker_thread_start(&args[0]);

    // join worker threads
    for (int i = 1; i < num_threads; i++)
    {
        workers[i].join();
    }
}
