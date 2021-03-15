#pragma once
#include <iostream>
#include <iostream>
#include <iomanip>
#include <mpi.h>

namespace conflux {
template <typename T>
void print_matrix(T *pointer,
                  int row_start, int row_end,
                  int col_start, int col_end,
                  int stride) {
    for (int i = row_start; i < row_end; ++i) {
        //std::cout << "[" << i << "]:\t";
        printf("[%2u:] ", i);
        for (int j = col_start; j < col_end; ++j) {
            std::cout << pointer[i * stride + j] << ", \t";
        }
        std::cout << std::endl;
    }
}

template <>
void print_matrix<double>(double *pointer,
                          int row_start, int row_end,
                          int col_start, int col_end,
                          int stride) {
    for (int i = row_start; i < row_end; ++i) {
        printf("[%2u:] ", i);
        for (int j = col_start; j < col_end; ++j) {
            printf("%8.3f", pointer[i * stride + j]);
            // std::cout << pointer[i * stride + j] << ", \t";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print_matrix_all(T *pointer,
                      int row_start, int row_end,
                      int col_start, int col_end,
                      int stride,
                      int rank,
                      int P,
                      MPI_Comm comm) {
    for (int r = 0; r < P; ++r) {
        if (r == rank) {
            int pi, pj, pk;
            std::tie(pi, pj, pk) = p2X(comm, rank);
            std::cout << "Rank = " << pi << ", " << pj << ", " << pk << std::endl;
            for (int i = row_start; i < row_end; ++i) {
                for (int j = col_start; j < col_end; ++j) {
                    std::cout << pointer[i * stride + j] << ", \t";
                }
                std::cout << std::endl;
            }
        }
        MPI_Barrier(comm);
    }
}
}

