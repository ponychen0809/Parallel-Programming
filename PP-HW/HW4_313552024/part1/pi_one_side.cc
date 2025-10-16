#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 創建shared memory window
    long long int *total_count;
    if (world_rank == 0) {
        // Master
        MPI_Win_allocate(sizeof(long long int), sizeof(long long int), 
                        MPI_INFO_NULL, MPI_COMM_WORLD, &total_count, &win);
        *total_count = 0;
    } else {
        // Workers
        MPI_Win_allocate(0, sizeof(long long int), MPI_INFO_NULL, 
                        MPI_COMM_WORLD, &total_count, &win);
    }

    // 計算local的tosses
    long long int local_tosses = tosses / world_size;
    long long int local_count = 0;
    unsigned int seed = time(NULL) * (world_rank + 1);
    
    for (long long int i = 0; i < local_tosses; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }
    // 使用single-sided communication累加結果
    MPI_Win_fence(0, win);
    if (world_rank != 0) {
        MPI_Accumulate(&local_count, 1, MPI_LONG_LONG_INT, 
                      0, 0, 1, MPI_LONG_LONG_INT, 
                      MPI_SUM, win);
    } else {
        MPI_Accumulate(&local_count, 1, MPI_LONG_LONG_INT, 
                      0, 0, 1, MPI_LONG_LONG_INT, 
                      MPI_SUM, win);
    }
    MPI_Win_fence(0, win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result = 4.0 * (*total_count) / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}