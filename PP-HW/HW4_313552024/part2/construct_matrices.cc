#include <fstream>

void construct_matrices(std::ifstream &in, int *n_ptr, int *m_ptr, int *l_ptr,
                       int **a_mat_ptr, int **b_mat_ptr) {
    // 讀取矩陣維度
    in >> *n_ptr >> *m_ptr >> *l_ptr;

    // 分配記憶體
    *a_mat_ptr = new int[(*n_ptr) * (*m_ptr)];
    *b_mat_ptr = new int[(*m_ptr) * (*l_ptr)];

    // 讀取矩陣A
    for (int i = 0; i < (*n_ptr) * (*m_ptr); i++) {
        in >> (*a_mat_ptr)[i];
    }

    // 讀取矩陣B
    for (int i = 0; i < (*m_ptr) * (*l_ptr); i++) {
        in >> (*b_mat_ptr)[i];
    }
}