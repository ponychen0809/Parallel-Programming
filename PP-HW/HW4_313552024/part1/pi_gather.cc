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

    // TODO: MPI init
    // MPI初始化
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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

    // TODO: use MPI_Gather
    // 使用MPI_Gather收集所有結果
    long long int* all_counts = NULL;
    if (world_rank == 0) {
        all_counts = new long long int[world_size];
    }
    
    MPI_Gather(&local_count, 1, MPI_LONG_LONG_INT,
               all_counts, 1, MPI_LONG_LONG_INT,
               0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        // 計算總和
        long long int total_count = 0;
        for (int i = 0; i < world_size; i++) {
            total_count += all_counts[i];
        }
        
        // TODO: PI result
        // 計算π值
        pi_result = 4.0 * total_count / (double)tosses;

        // 清理記憶體
        delete[] all_counts;


        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    
    MPI_Finalize();
    return 0;
}
