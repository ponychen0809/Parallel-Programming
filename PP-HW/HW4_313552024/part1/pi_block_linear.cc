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

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // 使用MPI平行計算圓周率π
    // Main process(rank 0)負責收集其他process的結果
    long long int local_tosses = tosses / world_size;
    long long int local_count = 0;
    long long int total_count = 0;
    // 設定random seed
    unsigned int seed = time(NULL) * (world_rank + 1);
    
    // 計算每個process的local_count
    for (long long int i = 0; i < local_tosses; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    if (world_rank > 0)
    {
        // TODO: handle workers
        // 將local_count傳送給main process
        MPI_Send(&local_count, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: master
        // master處理自己的結果並接收其他process的結果
        total_count = local_count;
        long long int received_count;
        
        // 線性接收其他process的結果
        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&received_count, 1, MPI_LONG_LONG_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_count += received_count;
        }
        
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        // 計算最終的π值
        pi_result = 4.0 * total_count / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
