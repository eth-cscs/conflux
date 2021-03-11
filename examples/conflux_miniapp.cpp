#include <mpi.h>
#include <vector>
#include <iostream>

#include <cxxopts.hpp>
#include <conflux/lu/conflux_opt.hpp>
#include <conflux/lu/profiler.hpp>

int main(int argc, char *argv[]) {
    cxxopts::Options options("conflux miniapp", 
        "A miniapp computing: LU factorization of A, where dim(A)=N*N");
    options.add_options()
        ("M,rows",
            "number of rows of matrix A.", 
            cxxopts::value<int>()->default_value("1000"))
        ("N,cols",
            "number of cols of matrix A.", 
            cxxopts::value<int>()->default_value("1000"))
        ("b,block_size",
            "block size",
            cxxopts::value<int>()->default_value("256"))
        ("p, p_grid",
            "3D-process decomposition.", 
            cxxopts::value<std::vector<int>>()->default_value("-1,-1,-1"))
        ("l,print_limit",
            "limit for printing the final result.", 
            cxxopts::value<int>()->default_value("20"))
        ("r,n_rep",
            "number of repetitions.", 
            cxxopts::value<int>()->default_value("2"))
        ("h,help", "Print usage.")
    ;
    // for some reason, a recent version of cxxopts
    // requires a const char** for the second argument
    auto const_argv = const_cast<const char**>(argv);
    auto result = options.parse(argc, const_argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto M = result["M"].as<int>();
    auto N = result["N"].as<int>();
    auto b = result["block_size"].as<int>();
    auto n_rep = result["n_rep"].as<int>();
    auto print_limit = result["print_limit"].as<int>();
    auto p_grid = result["p_grid"].as<std::vector<int>>();

    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    conflux::lu_params<double> params;

    int newP = P;

    if (p_grid[0] <= 0 || p_grid[1] <= 0 || p_grid[2] <= 0) {
        params = conflux::lu_params<double>(M, N, b, P);
    } else {
        params = conflux::lu_params<double>(M, N, b, p_grid[0], p_grid[1], p_grid[2]);
        newP = p_grid[0] * p_grid[1] * p_grid[2];
    }

    if (newP < P) {
        std::vector<int> ranks;
        for (int i = 0; i < newP; ++i) {
            ranks.push_back(i);
        }
        comm = conflux::create_comm(comm, ranks);
    }

    if (rank == 0) {
        std::cout << "Rank: " << rank << ", M: " << params.M << ", N: " << params.N << ", P:" << params.P
                  << ", v:" << params.v << ", Px:" << params.Px << ", Py: " << params.Py << ", Pz: " << params.Pz
                  << ", Nt: " << params.Nt
                  << ", tA11x: " << params.tA11x << ", tA11y: " << params.tA11y << std::endl;
    }

    std::vector<double> C(params.M * params.N);
    std::vector<double> Perm(params.M * params.M);

    for (int i = 0; i < n_rep; ++i) {
        PC();
        conflux::LU_rep<double>(params.matrix.data(), 
                               C.data(), 
                               Perm.data(), 
                               params, 
                               MPI_COMM_WORLD);  
        // print the profiler data
        if (rank == 0) {
            PP();
        }

#ifdef CONFLUX_WITH_VALIDATION
        if (rank == 0) {
            auto M = params.M;
            auto N = params.N;
            std::vector<dtype> L(N*N);
            std::vector<dtype> U(N*N);
            for (auto i = 0; i < N; ++i) {
                for (auto j = 0; j < i; ++j) {
                    L[i * N + j] = C.data()[i * N + j];
                }
                L[i * N + i] = 1;
                for (auto j = i; j < N; ++j) {
                    U[i * N + j] = C.data()[i * N + j];
                }
            }

            // mm<dtype>(L, U, C, N, N, N);
            // gemm<dtype>(PP, params.matrix, C, -1.0, 1.0, N, N, N);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                        1.0, &L[0], N, &U[0], N, 0.0, C.data(), N);

            if (rank == 0 && std::max(M, N) < print_limit) {
                std::cout << "L:\n";
                conflux::print_matrix(&L[0], 0, M, 0, N, N);
                std::cout << "\nU:\n";
                conflux::print_matrix(&U[0], 0, M, 0, N, N);
                std::cout << "\nPerm:\n";
                conflux::print_matrix(Perm.data(), 0, M, 0, N, N);
                std::cout << "\nL*U:\n";
                conflux::print_matrix(C.data(), 0, M, 0, N, N);
            }

            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                        -1.0, Perm.data(), N, params.matrix.data(), N, 1.0, C.data(), N);
            if (rank == 0 && std::max(M, N) < print_limit){ 
                std::cout << "\nL*U - P*A:\n";
                conflux::print_matrix(C.data(), 0, M, 0, N, N);
            }
            dtype norm = 0;
            for (auto i = 0; i < M; ++i) {
                for (auto j = 0; j < i; ++j) {
                    norm += C[i * N + j] * C[i * N + j];
                }
            }
            norm = std::sqrt(norm);
            std::cout << "residual: " << norm << std::endl << std::flush;\
        }
#endif
    }

    // if not MPI_COMM_WORLD, deallocate it
    if (newP < P) {
        MPI_Comm_free(&comm);
    }
    MPI_Finalize();

    return 0;
}
