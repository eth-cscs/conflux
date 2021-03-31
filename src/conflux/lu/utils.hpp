#pragma once
#include <omp.h>
#include <mpi.h>
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <conflux/lu/matrix_view.hpp>

namespace conflux {
// writes indices given in `indices` to the first column of matrix `mat`
// the previous values 
template <typename T, typename S>
void prepend_column(matrix_view<T> mat,
                    S* indices, bool check_positive=true) {
    assert(mat.n_rows >= 0);
    assert(mat.n_cols >= 1);
    // put the elements of indices[i] -> column 0 of mat
    for (int i = 0; i < mat.n_rows; ++i) {
        // cast from S -> T
        auto el = indices[i];
        auto casted_el = static_cast<T>(el);
        assert(!check_positive || casted_el >= 0);
        assert(std::abs(casted_el - el) < 1e-12);
        mat(i, 0) = casted_el;
    }
}

// extracts the first column of matrix<T>, casts it to <S> and stores it in a vector
template <typename T, typename S>
void column(matrix_view<T> mat, int col, S* column) {
    assert(col >= 0);
    assert(mat.n_cols > col);
    for (int i = 0; i < mat.n_rows; ++i) {
        const auto& el = mat(i, col);
        // cast from T -> S
        auto casted_el = static_cast<S>(el);
        // assert(casted_el >= 0);
        assert(std::abs(casted_el - el) < 1e-12);
        column[i] = casted_el;
    }
}

// place i-th row from the input to the perm[i]-th row in the output
// perm is the final permutation (out-of-place)
// assumptions: see asserts below
// if out.n_cols < in.n_cols, then only last in.n_cols of out 
// will be copied during the permuting.
template <typename T>
void permute_rows(matrix_view<T> in, matrix_view<T> out,
                  std::vector<int>& out_perm) {
    assert(in.n_rows >= 0 && in.n_cols >=0 && out.n_rows >= 0 && out.n_cols >=0);
    if (in.n_rows == 0 || out.n_rows == 0) return;
    assert(in.n_rows >= out.n_rows);
    assert(in.n_cols >= out.n_cols);

    assert(in.n_rows <= out_perm.size());
    assert(in.layout() == out.layout());

    int col = in.n_cols - out.n_cols;
    if (in.layout() == order::row_major) {
        // let each thread copy i-th row to perm[i]-th row
#pragma omp parallel for shared(in, out, out_perm)
        for (int i = 0; i < in.n_rows; ++i) {
            T* in_ptr = &in(i, col);
            T* out_ptr = &out(out_perm[i], 0);
            std::copy_n(in_ptr, out.n_cols, out_ptr);
        }
    } else {
        // let each thread permute a single column
#pragma omp parallel for shared(in, out, out_perm)
        for (int c = col; c < in.n_cols; ++c) {
            for (int r = 0; r < in.n_rows; ++r) {
                T* in_ptr = &in(r, c);
                T* out_ptr = &out(out_perm[r], c-col);
                *out_ptr = *in_ptr;
            }
        }
    }
}

// place perm[i]-th row from the input to the i-th row in the output
// assumptions: see asserts below
// if out.n_cols < in.n_cols, then only last in.n_cols of out 
// will be copied during the permuting.
template <typename T>
void inverse_permute_rows(matrix_view<T> in, matrix_view<T> out,
                  std::vector<int>& in_perm) {
    assert(in.n_rows >= 0 && in.n_cols >= 0 && out.n_rows >= 0 && out.n_cols >= 0);
    if (in.n_rows == 0 || out.n_rows == 0) return;
    assert(in.n_rows <= in_perm.size());
    assert(in.n_cols >= out.n_cols);

    assert(out.n_rows <= in_perm.size());
    assert(in.layout() == out.layout());

    int col = in.n_cols - out.n_cols;
    if (in.layout() == order::row_major) {
        // let each thread copy i-th row to perm[i]-th row
#pragma omp parallel for shared(in, out, in_perm)
        for (int i = 0; i < out.n_rows; ++i) {
            T* in_ptr = &in(in_perm[i], col);
            T* out_ptr = &out(i, 0);
            std::copy_n(in_ptr, out.n_cols, out_ptr);
        }
    } else {
        // let each thread permute a single column
#pragma omp parallel for shared(in, out, in_perm)
        for (int c = 0; c < out.n_cols; ++c) {
            for (int r = 0; r < out.n_rows; ++r) {
                T* in_ptr = &in(in_perm[r], c+col);
                T* out_ptr = &out(r, c);
                *out_ptr = *in_ptr;
            }
        }
    }
}

// place perm[i]-th row from the input to the i-th row in the output
// perm is the final permutation
template <typename T>
void inverse_permute_rows(T* in, T* out,
                          int n_rows, int n_cols,
                          int new_n_rows, int new_n_cols,
                          order layout, std::vector<int>& perm) {
    assert(n_rows >= 0 && n_cols >= 0 && new_n_rows >= 0 && new_n_cols >= 0);
    if (n_rows == 0 || new_n_rows == 0) return;
    int in_stride = n_cols;
    int out_stride = new_n_cols;
    if (layout == order::col_major) {
        in_stride = n_rows;
        out_stride = new_n_rows;
    }

    matrix_view<T> in_mat(in, n_rows, n_cols, in_stride, layout);
    matrix_view<T> out_mat(out, new_n_rows, new_n_cols, out_stride, layout);

    inverse_permute_rows(in_mat, out_mat, perm);
}

// place i-th row from the input to the perm[i]-th row in the output
// perm is the final permutation
template <typename T>
void permute_rows(T* in, T* out,
                  int n_rows, int n_cols,
                  int new_n_rows, int new_n_cols,
                  order layout, std::vector<int>& perm) {
    assert(n_rows >= 0 && n_cols >= 0 && new_n_rows >= 0 && new_n_cols >= 0);
    if (n_rows == 0 || new_n_rows == 0) return;
    int in_stride = n_cols;
    int out_stride = new_n_cols;
    if (layout == order::col_major) {
        in_stride = n_rows;
        out_stride = new_n_rows;
    }

    matrix_view<T> in_mat(in, n_rows, n_cols, in_stride, layout);
    matrix_view<T> out_mat(out, new_n_rows, new_n_cols, out_stride, layout);

    permute_rows(in_mat, out_mat, perm);
}
}
