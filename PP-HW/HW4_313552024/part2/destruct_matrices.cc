void destruct_matrices(int *a_mat, int *b_mat) {
    // 釋放記憶體
    delete[] a_mat;
    delete[] b_mat;
}