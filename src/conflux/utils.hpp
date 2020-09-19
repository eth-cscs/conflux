#pragma once
#include <omp.h>
#include <conflux/matrix_view.hpp>

namespace conflux {
// writes indices given in `indices` to the first column of matrix `mat`
// the previous values 
template <typename T, typename S>
void prepend_column(matrix_view<T> mat,
                    std::vector<S>& indices) {
    // put the elements of indices[i] -> column 0 of mat
    for (int i = 0; i < mat.n_rows; ++i) {
        // cast from S -> T
        auto el = indices[i];
        auto casted_el = static_cast<T>(el);
        assert(std::abs(casted_el - el) < 1e-12);
        mat(i, 0) = casted_el;
    }
}

// extracts the first column of matrix<T>, casts it to <S> and stores it in a vector
template <typename T, typename S>
std::vector<S> column(matrix_view<T> mat, int col) {
    std::vector<S> column;
    column.reserve(mat.n_rows);
    for (int i = 0; i < mat.n_rows; ++i) {
        const auto& el = mat(i, col);
        // cast from T -> S
        auto casted_el = static_cast<S>(el);
        assert(std::abs(casted_el - el) < 1e-12);
        column.push_back(casted_el);
    }
    return column;
}

// place perm[i]-th row from the input to the i-th row in the output
// perm is the final permutation (out-of-place)
template <typename T>
void inverse_permute_rows(T* in, T* out, 
                          int n_rows, int n_cols, int new_n_cols,
                          std::vector<int>& perm) {
    int row_offset = n_cols - new_n_cols;
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        auto src_offset = perm[i] * n_cols + row_offset;
        auto dst_offset = i * new_n_cols;
        std::copy_n(&in[src_offset], new_n_cols, 
                    &out[dst_offset]);
    }
}

// place i-th row from the input to the perm[i]-th row in the output
// perm is the final permutation (out-of-place)
template <typename T>
void permute_rows(T* in, T* out, 
                  int n_rows, int n_cols, int new_n_cols,
                  std::vector<int>& perm) {
    int row_offset = n_cols - new_n_cols;
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        auto src_offset = i * n_cols + row_offset;
        auto dst_offset = perm[i] * new_n_cols;
        std::copy_n(&in[src_offset], new_n_cols, 
                    &out[dst_offset]);
    }
}
}
