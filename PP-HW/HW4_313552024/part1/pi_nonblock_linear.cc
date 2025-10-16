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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // 計算local的tosses
    long long int local_tosses = tosses / world_size;
    long long int local_count = 0;
    long long int total_count = 0;
    unsigned int seed = time(NULL) * (world_rank + 1);
    
    for (long long int i = 0; i < local_tosses; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    if (world_rank > 0)
    {
        // TODO: MPI workers
        // worker process使用non-blocking send
        MPI_Request request;
        MPI_Isend(&local_count, 1, MPI_LONG_LONG_INT, 0, 0, 
                  MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        // master process使用non-blocking receive
        total_count = local_count;
        long long int* received_counts = new long long int[world_size - 1];
        MPI_Request* requests = new MPI_Request[world_size - 1];
        // MPI_Request requests[];

        // 啟動所有non-blocking receive
        for (int i = 1; i < world_size; i++) {
            MPI_Irecv(&received_counts[i-1], 1, MPI_LONG_LONG_INT, 
                      i, 0, MPI_COMM_WORLD, &requests[i-1]);
        }

        // 等待所有接收完成
        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);

        // 計算總和
        for (int i = 0; i < world_size - 1; i++) {
            total_count += received_counts[i];
        }


        // 清理記憶體
        delete[] received_counts;
        delete[] requests;
    }

    if (world_rank == 0)
    {
        // TODO: PI result
        // 計算π值
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
