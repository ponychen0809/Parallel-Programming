#include <mpi.h>
#include <cstdio>

void matrix_multiply(const int n, const int m, const int l,
                    const int *a_mat, const int *b_mat) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 計算每個process要處理的行數
    int rows_per_proc = n / size;
    int extra_rows = n % size;
    int my_rows = (rank < extra_rows) ? rows_per_proc + 1 : rows_per_proc;
    int my_start_row = (rank < extra_rows) ? 
                      rank * (rows_per_proc + 1) :
                      rank * rows_per_proc + extra_rows;

    // 計算local的結果
    int *local_result = new int[my_rows * l];
    for (int i = 0; i < my_rows; i++) {
        for (int j = 0; j < l; j++) {
            long long sum = 0;
            for (int k = 0; k < m; k++) {
                sum += a_mat[(my_start_row + i) * m + k] * b_mat[k * l + j];
            }
            local_result[i * l + j] = sum;
        }
    }

    // 收集結果並輸出
    if (rank == 0) {
        // 先輸出自己的結果
        for (int i = 0; i < my_rows; i++) {
            for (int j = 0; j < l; j++) {
                printf("%d ", local_result[i * l + j]);
            }
            printf("\n");
        }

        // 接收並輸出其他process的結果
        for (int src = 1; src < size; src++) {
            int src_rows = (src < extra_rows) ? rows_per_proc + 1 : rows_per_proc;
            int *temp_result = new int[src_rows * l];
            MPI_Recv(temp_result, src_rows * l, MPI_INT, src, 0, 
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int i = 0; i < src_rows; i++) {
                for (int j = 0; j < l; j++) {
                    printf("%d ", temp_result[i * l + j]);
                }
                printf("\n");
            }
            delete[] temp_result;
        }
    } else {
        // 其他process發送結果給process 0
        MPI_Send(local_result, my_rows * l, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    delete[] local_result;
}