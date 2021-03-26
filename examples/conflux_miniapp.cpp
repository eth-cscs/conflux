#include <mpi.h>
#include <vector>
#include <iostream>

#include <cxxopts.hpp>
#include <conflux/lu/conflux_opt.hpp>
#include <conflux/lu/profiler.hpp>

#include <costa/grid2grid/transform.hpp>

#include "utils.hpp"

MPI_Comm subcommunicator(std::vector<int>& ranks, MPI_Comm comm = MPI_COMM_WORLD) {
    // original size
    int P;
    MPI_Comm_size(comm, &P);

    // original group
    MPI_Group group;
    MPI_Comm_group(comm, &group);

    // new comm and new group
    MPI_Comm newcomm;
    MPI_Group newcomm_group;

    // create reduced group
    MPI_Group_incl(group, ranks.size(), ranks.data(), &newcomm_group);
    // create reduced communicator
    MPI_Comm_create_group(comm, newcomm_group, 0, &newcomm);

    MPI_Group_free(&group);
    MPI_Group_free(&newcomm_group);

    return newcomm;
}

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
            cxxopts::value<int>()->default_value("30"))
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
    if (p_grid[0] <= 0 || p_grid[1] <= 0 || p_grid[2] <= 0) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "[ERROR] Use --p_grid=Px,Py,Pz to specify the process grid!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 0);
        }
    }
    conflux::lu_params<double> params(M, N, b, 
                                      p_grid[0], p_grid[1], p_grid[2], 
                                      MPI_COMM_WORLD);

    int rank, P;
    MPI_Comm_rank(params.lu_comm, &rank);
    MPI_Comm_size(params.lu_comm, &P);

    if (rank == 0) {
        std::cout << "Rank: " << rank << ", M: " << params.M << ", N: " << params.N << ", P:" << params.P
                  << ", v:" << params.v << ", Px:" << params.Px << ", Py: " << params.Py << ", Pz: " << params.Pz
                  << ", Nt: " << params.Nt
                  << ", tA11x: " << params.tA11x << ", tA11y: " << params.tA11y << std::endl;
    }

    // A = input, C = output
    std::vector<double>& A_buff = params.data;
    std::vector<double> C_buff(A_buff.size());

    // pivots
    std::vector<int> ipvt(params.Ml);
    for (int i = 0; i < n_rep; ++i) {
        PC();
        // reinitialize the matrix
        params.InitMatrix();
        conflux::LU_rep<double>(
                               params,
                               C_buff.data(),
                               ipvt.data()
                               );
    }

    // the bottom-base of processor grid
    // these ranks actually own the initial 
    // and final data
    std::vector<int> ranks;
    ranks.reserve(params.Px * params.Py);
    for (int ppi = 0; ppi < params.Px; ++ppi) {
        for (int ppj = 0; ppj < params.Py; ++ppj) {
            int rank = conflux::X2p(params.lu_comm, ppi, ppj, 0);
            ranks.push_back(rank);
        }
    }
    MPI_Comm comm = subcommunicator(ranks, params.lu_comm);

// TODO:
// 1. rank (and its coordinates) in lu_comm might be != rank (and coordinates) in comm
//    So, make sure to relabel ranks
// 2. permute rows with ipvt
    if (comm != MPI_COMM_NULL) {
        // create blacs grid from ranks owning data
        // observe that the ordering in comm might be different than in lu_comm
        int ctxt = Csys2blacs_handle(comm);
        char order = 'R';
        Cblacs_gridinit(&ctxt, &order, params.Px, params.Py);

        int sub_rank;
        MPI_Comm_rank(comm, &sub_rank);

        // conflux layouts for A and C
        auto A_layout = conflux::conflux_layout(A_buff.data(),
                                                params.M, params.N, params.v,
                                                'R',
                                                params.Px, params.Py,
                                                sub_rank);
        auto C_layout = conflux::conflux_layout(C_buff.data(),
                                                params.M, params.N, params.v,
                                                'R',
                                                params.Px, params.Py,
                                                sub_rank);

        // A and C copies in scalapack layout
        std::vector<double> A_scalapack_buff(A_buff.size());
        std::vector<double> C_scalapack_buff(A_buff.size());

        // L and U of the result
        std::vector<double> L_scalapack_buff(A_buff.size());
        std::vector<double> U_scalapack_buff(A_buff.size());

        auto A_scalapack_layout = conflux::conflux_layout(A_scalapack_buff.data(),
                                                params.M, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                sub_rank);
        auto C_scalapack_layout = conflux::conflux_layout(C_scalapack_buff.data(),
                                                params.M, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                sub_rank);

        auto L_scalapack_layout = conflux::conflux_layout(L_scalapack_buff.data(),
                                                params.M, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                sub_rank);
        auto U_scalapack_layout = conflux::conflux_layout(U_scalapack_buff.data(),
                                                params.M, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                sub_rank);

        // transform initial matrix conflux->scalapack
        costa::transform(A_layout, A_scalapack_layout, comm);

        // transform the resulting matrix conflux->scalapack
        // costa::transform(C_layout, C_scalapack_layout, comm);

        // Let L and U be identical copies of the result matrix C
        costa::transform(C_layout, L_scalapack_layout, comm);
        costa::transform(C_layout, U_scalapack_layout, comm);

        // annulate all elements of L above the diagonal
        auto discard_upper_half = [](int gi, int gj, double prev_value) -> double {
            // if above diagonal, return 0
            if (gj > gi) return 0.0;
            // otherwise, return the prev_value (i.e. left unchanged)
            return prev_value;
        };
        L_scalapack_layout.apply(discard_upper_half);

        // annulate all elements of U below the diagonal
        auto discard_lower_half = [](int gi, int gj, double prev_value) -> double {
            // if above diagonal, return 0
            if (gi <= gj) return 0.0;
            // otherwise, return the prev_value (i.e. left unchanged)
            return prev_value;
        };
        U_scalapack_layout.apply(discard_lower_half);

        // scalapack descriptors
        std::array<int, 9> desc_L;
        std::array<int, 9> desc_U;
        std::array<int, 9> desc_A;

        int info = 0;
        int k = std::min(params.M, params.N);
        int zero = 0;

        descinit_(&desc_L[0], &params.M, &k, 
                 &b, &b, &zero, &zero, &ctxt, &params.Ml, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }
        descinit_(&desc_U[0], &k, &params.N,
                 &b, &b, &zero, &zero, &ctxt, &params.Ml, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }
        descinit_(&desc_A[0], &params.M, &params.N,
                 &b, &b, &zero, &zero, &ctxt, &params.Ml, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }

        char no_trans = 'N';

        double one = 1.0;
        double minus_one = -1.0;

        int int_one = 1;

        // compute C = C - L*U
        pdgemm_(&no_trans, &no_trans,
                &params.M, &params.N, &k,
                &one, &L_scalapack_buff[0], &int_one, &int_one, &desc_L[0],
                &U_scalapack_buff[0], &int_one, &int_one, &desc_U[0], &minus_one,
                &A_scalapack_buff[0], &int_one, &int_one, &desc_A[0]);

        // annulate all elements of U below the diagonal
        auto sum_squares = [](double prev_value, double el) -> double {
            auto elem = std::abs(el);
            return elem*elem + prev_value;
        };

        // now we want to compute the Frobenius norm of C
        // first compute the local partial norms:
        double local_norm = A_scalapack_layout.accumulate(sum_squares, 0.0);

        double sum_local_norms = 0.0;

        MPI_Comm_rank(comm, &rank);

        int root = 0;

        MPI_Reduce(&local_norm, &sum_local_norms, 1, MPI_DOUBLE, 
                   MPI_SUM, root, comm);

        auto frobenius_norm = std::sqrt(sum_local_norms);

        if (sub_rank == 0) {
            std::cout << std::fixed;
            std::cout << std::setprecision(4);
            std::cout << "Total Frobenius norm = " << frobenius_norm << std::endl;
        }

        Cblacs_gridexit(ctxt);
        int dont_finalize_mpi = 1;
        Cblacs_exit(dont_finalize_mpi);
        MPI_Comm_free(&comm);
    }


    // print the profiler data
    if (rank == 0) {
        PP();
    }

    // free the communicators before finalize
    // since no MPI routine can be called afterwards
    params.free_comms();
    MPI_Finalize();

    return 0;
}
