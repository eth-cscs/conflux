#include <mpi.h>
#include <vector>
#include <iostream>

#include <cxxopts.hpp>
#include <conflux/lu/conflux_opt.hpp>
#include <conflux/lu/profiler.hpp>

#include <costa/grid2grid/transform.hpp>

#ifdef CONFLUX_WITH_VALIDATION
#include "utils.hpp"
#endif

std::tuple<int, int> rank_to_coord(MPI_Comm comm2D, int rank) {
    int coords[] = {-1, -1};
    MPI_Cart_coords(comm2D, rank, 2, coords);
    return {coords[0], coords[1]};
}

std::vector<int> translate_ranks(MPI_Comm comm1,
                                 MPI_Comm comm2,
                                 std::vector<int>& ranks1) {
    std::vector<int> ranks2(ranks1.size());

    MPI_Group group1;
    MPI_Comm_group(comm1, &group1);

    MPI_Group group2;
    MPI_Comm_group(comm2, &group2);

    MPI_Group_translate_ranks(group1, 
                              ranks1.size(), &ranks1[0], 
                              group2,
                              &ranks2[0]);
    return ranks2;
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

    bool print_full_matrices = std::max(M, N) < print_limit;

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
    std::vector<double> C_buff;
#ifdef CONFLUX_WITH_VALIDATION
    C_buff = std::vector<double>(A_buff.size());
#endif

    // pivots
    std::vector<int> pivotIndsBuff(params.M);
    for (int i = 0; i < n_rep; ++i) {
        PC();
        // reinitialize the matrix
        params.InitMatrix();
        conflux::LU_rep<double>(
                               params,
                               C_buff.data(),
                               pivotIndsBuff.data()
                               );
    }

#ifdef CONFLUX_WITH_VALIDATION
    if (params.pk == 0) {
        MPI_Comm comm = params.ij_comm;
        int rank, P;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &P);

        // relabel the ij_comm to row-major processor ordering
        // to be compatible with the Cblacs grid that
        // we will create with 'R'-ordering
        int my_pi, my_pj;
        std::tie(my_pi, my_pj) = rank_to_coord(comm, rank);
        int row_major_coord = my_pi * params.Py + my_pj;

        MPI_Comm row_major_comm;
        MPI_Comm_split(comm, 0, row_major_coord, &row_major_comm);

        // this is safe, because lu_params class is going to free the ij_comm
        comm = row_major_comm;
        rank = row_major_coord;

        // create blacs grid from ranks owning data
        // observe that the ordering in comm might be different than in lu_comm
        int ctxt = Csys2blacs_handle(comm);
        // Cblacs_gridmap(&ctxt, &ranks2[0], params.Px, params.Px, params.Py);
        char order = 'R';
        Cblacs_gridinit(&ctxt, &order, params.Px, params.Py);

        // conflux layouts for A and C
        auto A_layout = conflux::conflux_layout(A_buff.data(),
                                                params.M, params.N, params.v,
                                                'R',
                                                params.Px, params.Py,
                                                rank);
        auto C_layout = conflux::conflux_layout(C_buff.data(),
                                                params.M, params.N, params.v,
                                                'R',
                                                params.Px, params.Py,
                                                rank);

        int k = std::min(params.M, params.N);
        int kl = std::min(params.Ml, params.Nl);

        // A and C copies in scalapack layout
        std::vector<double> A_scalapack_buff(A_buff.size());
        std::vector<double> C_scalapack_buff(A_buff.size());

        // L and U of the result
        int lld_M = scalapack::max_leading_dimension(params.M, b, params.Px);
        int lld_N = scalapack::max_leading_dimension(params.N, b, params.Py);
        int lld_k = std::max(lld_M, lld_N);

        // scalapack descriptors
        std::array<int, 9> desc_P;
        std::array<int, 9> desc_PA;
        std::array<int, 9> desc_L;
        std::array<int, 9> desc_U;
        std::array<int, 9> desc_A;

        int info = 0;
        int zero = 0;

        descinit_(&desc_P[0], &params.M, &params.M, 
                 &b, &b, &zero, &zero, &ctxt, &lld_M, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }
        descinit_(&desc_PA[0], &params.M, &params.N, 
                 &b, &b, &zero, &zero, &ctxt, &lld_M, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }
        descinit_(&desc_L[0], &params.M, &k, 
                 &b, &b, &zero, &zero, &ctxt, &lld_M, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "error: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }

        descinit_(&desc_U[0], &k, &params.N,
                 &b, &b, &zero, &zero, &ctxt, &lld_k, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }
        descinit_(&desc_A[0], &params.M, &params.N,
                 &b, &b, &zero, &zero, &ctxt, &lld_M, &info);
        if (params.pi == 0 && params.pj == 0 && info != 0) {
            std::cout << "ERROR: descinit, argument: " << -info << " has an illegal value!" << std::endl;
        }

        std::vector<double> L_scalapack_buff(scalapack::local_buffer_size(&desc_L[0]));
        std::vector<double> U_scalapack_buff(scalapack::local_buffer_size(&desc_U[0]));
        std::vector<double> P_scalapack_buff(scalapack::local_buffer_size(&desc_P[0]));
        std::vector<double> PA_scalapack_buff(scalapack::local_buffer_size(&desc_PA[0]));

        std::vector<double> full_A_scalapack_buff;
        std::vector<double> full_C_scalapack_buff;
        std::vector<double> full_L_scalapack_buff;
        std::vector<double> full_U_scalapack_buff;
        std::vector<double> full_P_scalapack_buff;
        std::vector<double> full_PA_scalapack_buff;

        if (rank == 0 && print_full_matrices) {
            full_A_scalapack_buff = std::vector<double>(params.M * params.N);
            full_C_scalapack_buff = std::vector<double>(params.M * params.N);
            full_L_scalapack_buff = std::vector<double>(params.M * k);
            full_U_scalapack_buff = std::vector<double>(k * params.N);
            full_P_scalapack_buff = std::vector<double>(params.M * params.M);
            full_PA_scalapack_buff = std::vector<double>(params.M * params.N);
        }

        // A-matrix (result): M x N
        auto A_scalapack_layout = conflux::conflux_layout(A_scalapack_buff.data(),
                                                params.M, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                rank);
        // C-matrix (result): M x N
        auto C_scalapack_layout = conflux::conflux_layout(C_scalapack_buff.data(),
                                                params.M, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                rank);
        // M x M permutation matrix
        auto P_scalapack_layout = conflux::conflux_layout(P_scalapack_buff.data(),
                                                params.M, params.M, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                rank);
        // M x M permutation matrix
        auto PA_scalapack_layout = conflux::conflux_layout(PA_scalapack_buff.data(),
                                                params.M, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                rank);
        // L-matrix: M x min(M, N)
        auto L_scalapack_layout = conflux::conflux_layout(L_scalapack_buff.data(),
                                                params.M, k, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                rank);
        // U-matrix: min(M, N) x N
        auto U_scalapack_layout = conflux::conflux_layout(U_scalapack_buff.data(),
                                                k, params.N, params.v,
                                                'C',
                                                params.Px, params.Py,
                                                rank);

        // full matrices descriptors
        auto full_A_scalapack_layout = conflux::conflux_layout(full_A_scalapack_buff.data(),
                                           params.M, params.N, params.M,
                                           'C',
                                           1, 1,
                                           rank);
        auto full_C_scalapack_layout = conflux::conflux_layout(full_C_scalapack_buff.data(),
                                           params.M, params.N, params.M,
                                           'C',
                                           1, 1,
                                           rank);
        auto full_L_scalapack_layout = conflux::conflux_layout(full_L_scalapack_buff.data(),
                                           params.M, k, params.M,
                                           'C',
                                           1, 1,
                                           rank);
        auto full_U_scalapack_layout = conflux::conflux_layout(full_U_scalapack_buff.data(),
                                           k, params.N, k,
                                           'C',
                                           1, 1,
                                           rank);
        auto full_P_scalapack_layout = conflux::conflux_layout(full_P_scalapack_buff.data(),
                                                params.M, params.M, params.M,
                                                'C',
                                                1, 1,
                                                rank);
        auto full_PA_scalapack_layout = conflux::conflux_layout(full_PA_scalapack_buff.data(),
                                                params.M, params.M, params.M,
                                                'C',
                                                1, 1,
                                                rank);

        // transform initial matrix conflux->scalapack
        costa::transform(A_layout, A_scalapack_layout, comm);

        // Let L and U be identical copies of the result matrix C
        costa::transform(C_layout, L_scalapack_layout, comm);
        costa::transform(C_layout, U_scalapack_layout, comm);

        if (print_full_matrices) {
            // full matrix C
            costa::transform(C_layout, full_C_scalapack_layout, comm);
            // full matrix A
            costa::transform(A_layout, full_A_scalapack_layout, comm);
        }

        // annulate all elements of L above the diagonal
        auto discard_upper_half = [](int gi, int gj, double prev_value) -> double {
            // if above diagonal, return 0
            if (gj > gi) return 0.0;
            if (gj == gi) return 1.0;
            // otherwise, return the prev_value (i.e. left unchanged)
            return prev_value;
        };
        L_scalapack_layout.apply(discard_upper_half);

        // annulate all elements of U below the diagonal
        auto discard_lower_half = [](int gi, int gj, double prev_value) -> double {
            // if above diagonal, return 0
            if (gi > gj) return 0.0;
            // otherwise, return the prev_value (i.e. left unchanged)
            return prev_value;
        };
        U_scalapack_layout.apply(discard_lower_half);

        // set the permutation based on ipiv array
        auto set_permutation = [&pivotIndsBuff](int gi, int gj) -> double {
            // set (i, ipvt[i]) to 1
            // however, ipvt array is defined such that
            // it maps local i to global ipvt[i]
            return pivotIndsBuff[gi]==gj ? 1 : 0;
        };
        P_scalapack_layout.initialize(set_permutation);

        if (print_full_matrices) {
            costa::transform(L_scalapack_layout, full_L_scalapack_layout, comm);
            costa::transform(U_scalapack_layout, full_U_scalapack_layout, comm);
            costa::transform(P_scalapack_layout, full_P_scalapack_layout, comm);
        }

        char no_trans = 'N';

        double one = 1.0;
        double minus_one = -1.0;
        double d_zero = 0.0;

        int int_one = 1;

        MPI_Barrier(comm);

        // permute rows of A to get: PA
        pdgemm_(&no_trans, &no_trans,
                &params.M, &params.N, &params.M,
                &minus_one, // alpha
                &P_scalapack_buff[0], &int_one, &int_one, &desc_P[0],
                &A_scalapack_buff[0], &int_one, &int_one, &desc_A[0], 
                &d_zero, // beta
                &PA_scalapack_buff[0], &int_one, &int_one, &desc_PA[0]);

        // compute PA = PA - L*U
        pdgemm_(&no_trans, &no_trans,
                &params.M, &params.N, &k,
                &one,  // alpha
                &L_scalapack_buff[0], &int_one, &int_one, &desc_L[0],
                &U_scalapack_buff[0], &int_one, &int_one, &desc_U[0], 
                &one, // beta
                &PA_scalapack_buff[0], &int_one, &int_one, &desc_PA[0]);

        if (print_full_matrices) {
            costa::transform(PA_scalapack_layout, full_PA_scalapack_layout, comm);
        }

        if (print_full_matrices) {
            // local buffers
            for (int pii = 0; pii < params.Px; pii++) {
                for (int pjj = 0; pjj < params.Py; pjj++) {
                    int cur_rank = pii * params.Px + pjj;
                    if (rank == cur_rank) {
                        std::cout << "Rank [" << pii << ", " << pjj << "], local final result:\n";
                        conflux::print_matrix(C_buff.data(), 0, params.Ml, 0, params.Nl, params.Nl);
                        std::cout << std::flush;
                    }
                    MPI_Barrier(comm);
                }
            }

            MPI_Barrier(comm);

            if (rank == 0 && print_full_matrices) {
                // full matrix A
                std::cout << "full-A-matrix on rank 0" << std::endl;
                conflux::print_matrix(&full_A_scalapack_buff[0],
                                      0, params.M, 0, params.N,
                                      params.M, 'C');
                std::cout << "======================" << std::endl;
                // full matrix C
                std::cout << "full-C-matrix on rank 0" << std::endl;
                conflux::print_matrix(&full_C_scalapack_buff[0],
                                      0, params.M, 0, params.N,
                                      params.M, 'C');
                std::cout << "======================" << std::endl;
                std::cout << "full-L-matrix on rank 0" << std::endl;
                conflux::print_matrix(&full_L_scalapack_buff[0],
                                      0, params.M, 0, params.N,
                                      params.M, 'C');
                std::cout << "======================" << std::endl;
                std::cout << "full-U-matrix on rank 0" << std::endl;
                conflux::print_matrix(&full_U_scalapack_buff[0],
                                      0, params.M, 0, params.N,
                                      params.M, 'C');
                std::cout << "======================" << std::endl;
                std::cout << "full-P-matrix on rank 0" << std::endl;
                conflux::print_matrix(&full_P_scalapack_buff[0],
                                      0, params.M, 0, params.N,
                                      params.M, 'C');
                std::cout << "======================" << std::endl;
                std::cout << "full-Remainder-matrix on rank 0" << std::endl;
                conflux::print_matrix(&full_PA_scalapack_buff[0],
                                      0, params.M, 0, params.N,
                                      params.M, 'C');
            }
        }

        // compute local frobenius norm
        auto sum_squares = [](double prev_value, double el) -> double {
            auto elem = std::abs(el);
            return elem*elem + prev_value;
        };

        // now we want to compute the Frobenius norm of C
        // first compute the local partial norms:
        double local_norm = PA_scalapack_layout.accumulate(sum_squares, 0.0);

        double sum_local_norms = 0.0;

        MPI_Reduce(&local_norm, &sum_local_norms, 1, MPI_DOUBLE, 
                   MPI_SUM, 0, comm);

        auto frobenius_norm = std::sqrt(sum_local_norms);

        if (rank == 0) {
            std::cout << std::fixed;
            std::cout << std::setprecision(4);
            std::cout << "Total Frobenius norm = " << frobenius_norm << std::endl;
        }

        Cblacs_gridexit(ctxt);
        int dont_finalize_mpi = 1;
        Cblacs_exit(dont_finalize_mpi);
        MPI_Comm_free(&comm);
    }
#endif


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
