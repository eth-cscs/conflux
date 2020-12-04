#include <mpi.h>
#include <vector>
#include <iostream>

#include <cxxopts.hpp>
#include <conflux/conflux_opt.hpp>

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

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto gv = conflux::GlobalVars<dtype>(M, N, b, size);

    std::cout << "Rank: " << rank << ", M: " << gv.M << ", N: " << gv.N << ", P:" << gv.P
              << ", v:" << gv.v << ", Px:" << gv.Px << ", Py: " << gv.Py << ", Pz: " << gv.Pz
              << ", Nt: " << gv.Nt
              << ", tA11x: " << gv.tA11x << ", tA11y: " << gv.tA11y << std::endl;

    std::vector<double> C(gv.M * gv.N);
    std::vector<double> Perm(gv.M * gv.M);

    for (int i = 0; i < n_rep; ++i) {
        PC();
        conflux::LU_rep<dtype>(gv.matrix, 
                               C.data(), 
                               Perm.data(), 
                               gv, 
                               MPI_COMM_WORLD);
    }

    if (rank == 0) {
        PP();
    }

    MPI_Finalize();

    return 0;
}
