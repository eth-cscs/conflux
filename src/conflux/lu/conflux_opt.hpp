#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>  // has std::lcm
#include <random>
#include <tuple>
#include <unordered_map>
// blas backend
#ifdef __USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include <conflux/lu/lu_params.hpp>
#include <conflux/lu/memory_utils.hpp>
#include <conflux/lu/profiler.hpp>
#include <conflux/lu/utils.hpp>

namespace conflux {

template <typename T>
MPI_Win
create_window(MPI_Comm comm, T *pointer, size_t size, bool no_locks) {
    MPI_Info info;
    MPI_Info_create(&info);
    if (no_locks) {
        MPI_Info_set(info, "no_locks", "true");
    } else {
        MPI_Info_set(info, "no_locks", "false");
    }
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "accumulate_ordering", "none");

    MPI_Win win;
    MPI_Win_create(
        pointer, size * sizeof(T), sizeof(T), info, comm, &win);

    MPI_Info_free(&info);
    return win;
}

MPI_Comm create_comm(MPI_Comm &comm, std::vector<int> &ranks);

int l2g(int pi, int ind, int sqrtp1);

void g2l(int gind, int sqrtp1,
         int &out1, int &out2);

std::tuple<int, int, int> p2X(MPI_Comm comm3D, int rank);
std::tuple<int, int> p2X_2d(MPI_Comm comm2D, int rank);

int X2p(MPI_Comm comm3D, int pi, int pj, int pk);

int X2p(MPI_Comm comm2D, int pi, int pj);

template <typename T>
void print_matrix(T *pointer,
                  int row_start, int row_end,
                  int col_start, int col_end,
                  int stride, char order = 'R') {
    if (order == 'R') {
        for (int i = row_start; i < row_end; ++i) {
            //std::cout << "[" << i << "]:\t";
            printf("[%2u:] ", i);
            for (int j = col_start; j < col_end; ++j) {
                std::cout << pointer[i * stride + j] << ", \t";
            }
            std::cout << std::endl;
        }
    } else {
        for (int i = row_start; i < row_end; ++i) {
            //std::cout << "[" << i << "]:\t";
            printf("[%2u:] ", i);
            for (int j = col_start; j < col_end; ++j) {
                std::cout << pointer[j * stride + i] << ", \t";
            }
            std::cout << std::endl;
        }
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

template <typename T>
void print_map(std::unordered_map<T, std::vector<T>> m){
    for (const auto& x : m) {
            std::cout << x.first << ": [";
            for (auto i: x.second)
                std::cout << i << ", ";
             std::cout<< "]\n";
        }
}

template <typename T>
void print_map(std::unordered_map<T, T> m){
    for (const auto& x : m) {
            std::cout << x.first << ": " << x.second << "\n";
        }
}


int flipbit(int n, int k);

int butterfly_pair(int pi, int r, int Px);

template <typename T>
void LUP(int n_local_active_rows, int v, int stride,
         T *pivotBuff, T *candidatePivotBuff,
         std::vector<int> &ipiv, std::vector<int> &perm) {
    assert(n_local_active_rows >= 0);
//    if (n_local_active_rows == 0) return;
    // reset the values
    for (int i = 0; i < std::max(2 * v, n_local_active_rows); ++i) {
        perm[i] = i;
    }

    parallel_mcopy<T>(n_local_active_rows, v,
                      &candidatePivotBuff[0], stride,
                      &pivotBuff[0], v);

    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n_local_active_rows, v,
                   &pivotBuff[0], v, &ipiv[0]);

    // ipiv -> permutation
    // ipiv returned from blas is 1-based because of fortran compatibility
    for (int i = 0; i < std::min(v, n_local_active_rows); ++i) {
        std::swap(perm[i], perm[ipiv[i] - 1]);
    }
}

void analyze_pivots(int first_non_pivot_row,
                    int n_rows,
                    std::vector<int>& curPivots,
                    std::vector<bool>& pivots,
                    std::vector<int>& early_non_pivots,
                    std::vector<int>& late_pivots
                    );

template <typename T>
void push_pivots_up(std::vector<T> &in, std::vector<T> &temp,
                    int n_rows, int n_cols,
                    order layout,
                    std::vector<int> &curPivots,
                    int first_non_pivot_row,
                    std::vector<bool>& pivots,
                    std::vector<int>& early_non_pivots,
                    std::vector<int>& late_pivots
                    ) {
    if (n_rows == 0 || n_cols == 0 || first_non_pivot_row >= n_rows)
        return;

#pragma omp parallel for shared(curPivots, in, n_cols, temp)
    // copy first v pivots to temporary buffer
    for (int i = 0; i < curPivots[0]; ++i) {
        int pivot_row = curPivots[i + 1];
        std::copy_n(&in[pivot_row * n_cols],
                    n_cols,
                    &temp[i * n_cols]);
    }

    // copy early non_pivots to late pivots positions
#pragma omp parallel for shared(late_pivots, early_non_pivots, in, n_cols)
    for (int i = 0; i < early_non_pivots.size(); ++i) {
        int row = early_non_pivots[i];
        std::copy_n(&in[row * n_cols],
                    n_cols,
                    &in[late_pivots[i] * n_cols]);
    }

#pragma omp parallel for shared(first_non_pivot_row, curPivots, temp, n_cols, in)
    // overwrites first v rows with pivots
    for (int i = 0; i < curPivots[0]; ++i) {
        int pivot_row = curPivots[i + 1];
        std::copy_n(&temp[i * n_cols],
                    n_cols,
                    &in[(first_non_pivot_row + i) * n_cols]);
    }
}

template <typename T>
void tournament_rounds(
    int n_local_active_rows,
    int v,
    order layout,
    std::vector<T> &A00Buff,
    std::vector<T> &pivotBuff,
    std::vector<T> &candidatePivotBuff,
    std::vector<T> &candidatePivotBuffPerm,
    std::vector<int> &ipiv, std::vector<int> &perm,
    int n_rounds,
    int Px, int layrK,
    MPI_Comm lu_comm,
    int k) {
    int rank;
    MPI_Comm_rank(lu_comm, &rank);
    int pi, pj, pk;
    std::tie(pi, pj, pk) = p2X(lu_comm, rank);

    int req_id = 0;
    int n_reqs = ((Px & (Px - 1)) == 0) ? 2 : (Px + 2);
    // int n_reqs = Px+2;
    MPI_Request reqs[n_reqs];

    for (int r = 0; r < n_rounds; ++r) {
        // auto src_pi = std::min(flipbit(pi, r), Px - 1);
        auto src_pi = butterfly_pair(pi, r, Px);
        auto p_rcv = X2p(lu_comm, src_pi, pj, pk);
        req_id = 0;

        if (src_pi < pi) {
            MPI_Isend(&candidatePivotBuff[v * (v + 1)], v * (v + 1), MPI_DOUBLE,
                      p_rcv, 1, lu_comm, &reqs[req_id++]);
            MPI_Irecv(&candidatePivotBuff[0], v * (v + 1), MPI_DOUBLE,
                      p_rcv, 1, lu_comm, &reqs[req_id++]);
        } else {
            MPI_Isend(&candidatePivotBuff[0], v * (v + 1), MPI_DOUBLE,
                      p_rcv, 1, lu_comm, &reqs[req_id++]);
            MPI_Irecv(&candidatePivotBuff[v * (v + 1)], v * (v + 1), MPI_DOUBLE,
                      p_rcv, 1, lu_comm, &reqs[req_id++]);
        }

        // we may also need to send more than one pair of messages in case of Px not a power of two.
        // because src_pi = std::min(flipbit(pi, r), Px - 1), multiple ranks may need data from the last
        // rank pi = Px - 1.
        // first, check who wants something from us:
        // if Px not a power of 2
        if ((Px & (Px - 1)) != 0) {
            for (int ppi = 0; ppi < Px; ppi++) {
                //then it means that ppi wants something from us
                if (butterfly_pair(ppi, r, Px) == pi && ppi != src_pi) {
                    p_rcv = X2p(lu_comm, ppi, pj, pk);
                    MPI_Isend(&candidatePivotBuff[v * (v + 1)], v * (v + 1), MPI_DOUBLE,
                              p_rcv, 1, lu_comm, &reqs[req_id++]);
                }
            }
        }
        MPI_Waitall(req_id, &reqs[0], MPI_STATUSES_IGNORE);

        // if (n_local_active_rows > 0) {
            // MPI_Request_free(&reqs[0]);

            // if (pi == 1) {
            //    std::cout << "pi: " << pi << ", src_pi: " << src_pi << ", tournament round " << r << "/" << n_rounds << ", before LUP. candidatePivBuff\n" << std::flush;
            //    print_matrix(candidatePivotBuff.data(), 0, 2*v, 0, v+1, v+1);
            //             std::cout << "\n\n" << std::flush;
            // }
            // candidatePivotBuff := input
            LUP(2 * v, v, v + 1, &pivotBuff[0], &candidatePivotBuff[1], ipiv, perm);

            // if final round
            if (r == n_rounds - 1) {
                inverse_permute_rows(&candidatePivotBuff[0],
                                     &candidatePivotBuffPerm[0],
                                     2 * v, v + 1, v, v + 1, layout, perm);

                candidatePivotBuff.swap(candidatePivotBuffPerm);

                // if (k > 12) {
                // std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ", pivotBuff:\n" << std::flush;
                //         print_matrix(pivotBuff.data(), 0, v, 0, v, v);
                //         std::cout << "\n\n" << std::flush;
                // }

                // just the top v rows
                parallel_mcopy(v, v,
                               &pivotBuff[0], v,
                               &A00Buff[0], v);
            } else {
                src_pi = butterfly_pair(pi, r + 1, Px);  //std::min(flipbit(pi, r+1), Px - 1);
                if (src_pi < pi) {
                    inverse_permute_rows(&candidatePivotBuff[0],
                                         &candidatePivotBuffPerm[v * (v + 1)],
                                         2 * v, v + 1, v, v + 1, layout, perm);
                    candidatePivotBuff.swap(candidatePivotBuffPerm);
                } else {
                    inverse_permute_rows(&candidatePivotBuff[0],
                                         &candidatePivotBuffPerm[0],
                                         2 * v, v + 1, v, v + 1, layout, perm);
                    candidatePivotBuff.swap(candidatePivotBuffPerm);
                }
            }

            // if (pi == 1) {
            //    std::cout << "pi: " << pi << ", src_pi: " << src_pi << ", tournament round " << r << "/" << n_rounds << ", AFTER LUP. candidatePivBuff\n" << std::flush;
            //    print_matrix(candidatePivotBuff.data(), 0, 2*v, 0, v+1, v+1);
            //             std::cout << "\n\n" << std::flush;
            //     std::cout << "pivotBuff\n" << std::flush;
            //    print_matrix(pivotBuff.data(), 0, 2*v, 0, v, v);
            //             std::cout << "\n\n" << std::flush;
            // }
        // }
    }
}

std::pair<
    std::unordered_map<int, std::vector<int>>,
    std::unordered_map<int, std::vector<int>>>
g2lnoTile(std::vector<int> &grows, int size, int Px, int v);

template <class T>
void LU_rep(lu_params<T>& gv,
            T* C,
            int* permutation) {
    PC();
    PE(init);
    int M, N, P, Px, Py, Pz, v, nlayr, Mt, Nt, tA11x, tA11y;
    N = gv.N;
    M = gv.M;
    P = gv.P;
    Px = gv.Px;
    Py = gv.Py;
    Pz = gv.Pz;
    v = gv.v;
    nlayr = gv.nlayr;
    Nt = gv.Nt;
    tA11x = gv.tA11x;
    tA11y = gv.tA11y;
    // local n
    int Ml = gv.Ml;
    int Nl = gv.Nl;

    int pi = gv.pi;
    int pj = gv.pj;
    int pk = gv.pk;
    int rank = gv.rank;

    MPI_Comm lu_comm = gv.lu_comm;
    MPI_Comm jk_comm = gv.jk_comm;
    MPI_Comm ik_comm = gv.ik_comm;
    MPI_Comm k_comm = gv.k_comm;

#ifdef DEBUG
    MPI_Comm i_comm;
    int keep_dims_i[] = {1, 0, 0};
    MPI_Cart_sub(lu_comm, keep_dims_i, &i_comm);
#endif

    auto chosen_step = Nt; // - 1;
    // auto debug_level = 0;
    //auto chosen_step = 90;
    auto debug_level = 0;

    int print_rank = X2p(lu_comm, 0, 0, 0);

    // Create buffers
    std::vector<T> A00Buff(v * v);

    // A10 => M
    // A01 => N
    // A11 => M x N
    std::vector<T> A10Buff(Ml * v);
    std::vector<T> A10BuffTemp(Ml * v);
    std::vector<T> A10BuffRcv(Ml * nlayr);

    std::vector<T> A01Buff(v * Nl);
    std::vector<T> A01BuffTemp(v * Nl);
    std::vector<T> A01BuffRcv(nlayr * Nl);

    std::vector<T> A11Buff = gv.data;
    std::vector<T> A10resultBuff(Ml * Nl);
    std::vector<T> A11BuffTemp(Ml * Nl);

    //TODO: can we handle such a big global pivots vector?
    std::vector<int> pivotIndsBuff(M);
#ifdef CONFLUX_WITH_VALIDATION
    MPI_Win res_Win = create_window(lu_comm,
                                    C,
                                    Ml*Nl,
                                    true);
    auto ScaLAPACKResultBuff = C;
    MPI_Win_fence(0, res_Win);
    // TODO: This is DEFINITELY suboptimal. We will use two buffers
    // (pivotIndsBuff and ipvt_g) to recreate local scalapack ipvt
    // Hopefully, it won't impact performance much.
    std::vector<int> ipvt_g(M);
    std::vector<int> ipvt(Ml);
    // initially, ipvt_g contains consecutive integers (0, 1, ..., M-1)
    // then, we will recreate this ipvt permutations 
    std::iota (std::begin(ipvt_g), std::end(ipvt_g), 0); 
#endif

    // global row indices
    std::vector<int> gri(Ml);
    std::unordered_map<int, int> igri;
    std::vector<int> griTemp(Ml);
    for (int i = 0; i < Ml; ++i) {
        auto lrow = i;
        // # we are in the local tile:
        auto lT = lrow / v;
        // # and inside this tile it is a row number:
        auto lR = lrow % v;
        // # which is a global tile:
        auto gT = lT * Px + pi;
        gri[i] = lR + gT * v;
        igri[gri[i]] = i;
    }

    int n_local_active_rows = Ml;
    int first_non_pivot_row = 0;

    std::vector<T> pivotBuff(Ml * v);

    std::vector<T> candidatePivotBuff(Ml * (v + 1));
    std::vector<T> candidatePivotBuffPerm(Ml * (v + 1));
    std::vector<int> perm(std::max(2 * v, Ml));  // rows
    std::vector<int> ipiv(std::max(2 * v, Ml));

    std::vector<bool> pivots(Ml);
    std::vector<int> early_non_pivots;
    early_non_pivots.reserve(v);
    std::vector<int> late_pivots;
    late_pivots.reserve(v);

    std::vector<int> trsm_1_dspls(Py * Pz);
    std::vector<int> trsm_1_counts(Py * Pz);

    std::vector<int> trsm_2_dspls(Px * Pz);
    std::vector<int> trsm_2_counts(Px * Pz);

    int jk_rank = X2p(jk_comm, pj, pk);
    int ik_rank = X2p(ik_comm, pi, pk);

    // 0 = num of pivots
    // 1..v+1 = pivots
    // v+1 .. 2v+1 = curPivOrder
    // 2v+1 .. 3v+1 = pivotIndsBuff
    std::vector<int> curPivots(v+1);
    std::vector<int> curPivOrder(v);
    for (int i = 0; i < curPivOrder.size(); ++i) {
        curPivOrder[i] = i;
    }
    // global pivot indices
    std::vector<int> gpivots(v);

    // GLOBAL result buffer
    // For debug only!
    std::cout << std::setprecision(3);

    // RNG
    std::mt19937_64 eng(gv.seed);
    std::uniform_int_distribution<int> dist(0, Pz - 1);

    // # ------------------------------------------------------------------- #
    // # ------------------ INITIAL DATA DISTRIBUTION ---------------------- #
    // # ------------------------------------------------------------------- #

    PE(fence_create);
    MPI_Win A01Win = create_window(lu_comm,
                                   A01Buff.data(),
                                   A01Buff.size(),
                                   true);

    // Sync all windows
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01Win);
    PL();
    std::vector<int> timers(8);

    auto layout = order::row_major;

#ifdef DEBUG
    std::vector<int> activeRows(Px);
#endif

    /*
# ---------------------------------------------- #
# ----------------- MAIN LOOP ------------------ #
# 0. reduce first tile column from A11buff to PivotA11ReductionBuff
# 1. coalesce PivotA11ReductionBuff to PivotBuff and scatter to A10buff
# 2. find v pivots and compute A00
# 3. reduce pivot rows from A11buff to PivotA11ReductionBuff
# 4. scatter PivotA01ReductionBuff to A01Buff
# 5. compute A10 and broadcast it to A10BuffRecv
# 6. compute A01 and broadcast it to A01BuffRecv
# 7. compute A11
# ---------------------------------------------- #
*/

    MPI_Barrier(lu_comm);
    auto t1 = std::chrono::high_resolution_clock::now();

    // # now k is a step number
    for (auto k = 0; k < Nt; ++k) {
        bool last_step = k == Nt - 1;
#ifdef DEBUG
        if (debug_level > 1 && print_rank == rank) {
            std::cout << "Iteration = " << k << std::endl;
        }
        MPI_Barrier(lu_comm);
#endif
        if (k == chosen_step + 1)
            break;

        // global current offset
        auto off = k * v;
        // local current offset
        auto loff = (k / Py) * v;  // sqrtp1 = 2, k = 157

        // # in this step, layrK is the "lucky" one to receive all reduces
        auto layrK = 0;  // dist(eng);

        // layrK = 0;
        // if (k == 0) layrK = 0;
        // if (k == 1) layrK = 1;
        // if (doprint && printrank) std::cout << "layrK: " << layrK << std::endl << std::flush;
        // # ----------------------------------------------------------------- #
        // # 0. reduce first tile column from A11buff to PivotA11ReductionBuff #
        // # ----------------------------------------------------------------- #
        // MPI_Barrier(lu_comm);
        auto ts = std::chrono::high_resolution_clock::now();
        // # Currently, we dump everything to processors in layer pk == 0, and only this layer choose pivots
        // # that is, each processor [pi, pj, pk] sends to [pi, pj, layK]
        // # note that processors in layer pk == 0 locally copy their data from A11buff to PivotA11ReductionBuff

        // flush the buffer
        curPivots[0] = 0;

        assert(n_local_active_rows >= 0);
        assert(first_non_pivot_row <= Ml);
        #ifdef DEBUG
        if (n_local_active_rows <= 0) {
            assert(curPivots[0] == 0);
        }
        #endif

#if DEBUG
        if (k == chosen_step && debug_level > 1) {
            std::cout << "Matrix A10BuffRcv = " << std::endl;
            print_matrix_all(A10BuffRcv.data(), 0, n_local_active_rows, 
                                                0, nlayr, nlayr, 
                                                rank, P, lu_comm);
            std::cout << "Matrix A10BuffTemp = " << std::endl;
            print_matrix_all(A10BuffTemp.data(), 0, n_local_active_rows, 
                                                0, nlayr, nlayr, 
                                                rank, P, lu_comm);
        }
        assert(n_local_active_rows >= 0);
        assert(first_non_pivot_row <= Ml);

        assert(has_valid_data(&A10Buff[0], 
                              first_non_pivot_row, Ml, 
                              0, v, v));
        assert(has_valid_data(A00Buff.data(), 0, v, 0, v, v));
        assert(has_valid_data(&A01Buff[0], 
                              0, v,
                              0, Nl-loff, Nl-loff));
        assert(has_valid_data(&A11Buff[0], 
                              first_non_pivot_row, Ml,
                              loff, Nl, Nl))
#endif

        PE(step0_padding);
        if (n_local_active_rows < v) {
            int padding_start = std::max(0, n_local_active_rows) * (v + 1);
            int padding_end = v * (v + 1);
            std::fill(candidatePivotBuff.begin() + padding_start,
                      candidatePivotBuff.begin() + padding_end, 0);
            std::fill(candidatePivotBuffPerm.begin() + padding_start,
                      candidatePivotBuffPerm.begin() + padding_end, 0);
            // std::fill(gri.begin() + first_non_pivot_row, gri.end(), -1);
        }
        PL();

        // # reduce first tile column. In this part, only pj == k % sqrtp1 participate:

       // if (pj == k % Py && n_local_active_rows > 0) {
            PE(step0_copy);
            parallel_mcopy<T>(n_local_active_rows, v,
                              &A11Buff[first_non_pivot_row * Nl + loff], Nl,
                              &A10Buff[first_non_pivot_row * v], v);
            PL();
#ifdef DEBUG
            if (debug_level > 0) {
                if (k == chosen_step) {
                    if (rank == print_rank) {
                        std::cout << "Step 0, A10Buff before reduction." << std::endl;
                        print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
                    }
                }
            }
#endif

            PE(step0_reduce);
            if (pk == layrK) {
                MPI_Reduce(MPI_IN_PLACE, &A10Buff[first_non_pivot_row * v],
                           n_local_active_rows * v,
                           MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            } else {
                MPI_Reduce(&A10Buff[first_non_pivot_row * v],
                           &A10Buff[first_non_pivot_row * v],
                           n_local_active_rows * v,
                           MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            }
            PL();
     //   }

#ifdef DEBUG
        if (debug_level > 1) {
            MPI_Barrier(lu_comm);
            if (k == chosen_step) {
                if (rank == print_rank) {
                    std::cout << "Step 0, A10Buff after reduction." << std::endl;
                    print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
                }
                // std::exit(0);
            }
        }
#endif

        // MPI_Barrier(lu_comm);
        auto te = std::chrono::high_resolution_clock::now();
        timers[0] += std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();

        // # --------------------------------------------------------------------- #
        // # 1. coalesce PivotA11ReductionBuff to PivotBuff and scatter to A10buff #
        // # --------------------------------------------------------------------- #
        ts = te;
        // # ---------------- FIRST STEP ----------------- #
        // # in first step, we do pivot on the whole PivotBuff array (may be larger than [2v, v]
        // # local computation step
        // sqrtp1 many roots
        // sqrtp1 x c  many receivers
#ifdef DEBUG
        if (debug_level > 0) {
            if (chosen_step == k) {
                if (pi == 0 && pj == 1 && pk == 0) {
                    std::cout << "GRI before tournament" << std::endl;
                    print_matrix(gri.data(),
                                 0, 1,
                                 0, Ml,
                                 Ml);
                }
            }
        }
#endif
        MPI_Request A00_req[2];
        int n_A00_reqs = 0;
        assert(n_local_active_rows + first_non_pivot_row == Ml);
        auto min_perm_size = std::min(N - k * v, v);
        if (pj == k % Py && pk == layrK) {
            auto max_perm_size = std::max(n_local_active_rows, v);

          //  if (n_local_active_rows > 0) {
                PE(step1_A10copy);
                parallel_mcopy<T>(n_local_active_rows, v,
                                  &A10Buff[first_non_pivot_row * v], v,
                                  &candidatePivotBuff[1], v + 1);
                // glue the gri elements to the first column of candidatePivotBuff
                prepend_column(matrix_view<T>(&candidatePivotBuff[0],
                                              n_local_active_rows, v + 1, v + 1,
                                              layout),
                               &gri[first_non_pivot_row]);
                PL();
#ifdef DEBUG
                if (debug_level > 0) {
                    // TODO: before anything
                    if (chosen_step == k) {
                        std::cout << "candidatePivotBuff BEFORE ANYTHING " << pi << std::endl;
                        print_matrix(candidatePivotBuff.data(),
                                     0, n_local_active_rows, 0, v + 1, v + 1);
                    }
                }
#endif
                // # tricky part! to preserve the order of the rows between swapping pairs (e.g., if ranks 0 and 1 exchange their
                // # candidate rows), we want to preserve that candidates of rank 0 are always above rank 1 candidates. Otherwise,
                // # we can get inconsistent results. That's why,in each communication pair, higher rank puts his candidates below:

                // # find with which rank we will communicate
                // # ANOTHER tricky part ! If sqrtp1 is not 2^n, then we will not have a nice butterfly communication graph.
                // # that's why with the flipBit strategy, src_pi can actually be larger than sqrtp1
                auto src_pi = std::min(flipbit(pi, 0), Px - 1);

                PE(step1_lup)
                LUP(n_local_active_rows, v, v + 1, &pivotBuff[0], &candidatePivotBuff[1], ipiv, perm);
                PL();

                // TODO: after first LUP and swap
#ifdef DEBUG
                if (debug_level > 0) {
                    if (chosen_step == k) {
                        std::cout << "candidatePivotBuff BEFORE FIRST LUP AND SWAP " << pi << std::endl;
                        print_matrix(candidatePivotBuff.data(),
                                     0, n_local_active_rows, 0, v + 1, v + 1);
                    }
                }
#endif

                PE(step1_rowpermute);
                if (src_pi < pi) {
                    inverse_permute_rows(&candidatePivotBuff[0], &candidatePivotBuffPerm[v * (v + 1)],
                                         max_perm_size, v + 1, v, v + 1, layout, perm);
                    candidatePivotBuff.swap(candidatePivotBuffPerm);
                } else {
                    inverse_permute_rows(&candidatePivotBuff[0], &candidatePivotBuffPerm[0],
                                         max_perm_size, v + 1, v, v + 1, layout, perm);
                    candidatePivotBuff.swap(candidatePivotBuffPerm);
                }
                PL();
      //      }

            // TODO: after first LUP and swap
#ifdef DEBUG
            if (debug_level > 0) {
                if (chosen_step == k) {
                    std::cout << "candidatePivotBuff AFTER FIRST LUP AND SWAP " << pi << std::endl;
                    print_matrix(candidatePivotBuff.data(),
                                 0, n_local_active_rows, 0, v + 1, v + 1);
                }
            }

            if (k == chosen_step && debug_level > 1) {
                std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], n_local_active_rows: "
                          << n_local_active_rows << ", candidatePivotBuff: \n"
                          << std::flush;
                print_matrix(candidatePivotBuff.data(), 0, Ml, 0, v + 1, v + 1);
                std::cout << "\n\n"
                          << std::flush;
            }
#endif
            // std::cout << "Matrices permuted" << std::endl;

            // # ------------- REMAINING STEPS -------------- #
            // # now we do numRounds parallel steps which synchronization after each step
            PE(step1_pivoting);
            auto numRounds = int(std::ceil(std::log2(Px)));

            tournament_rounds(
                n_local_active_rows,
                v,
                layout,
                A00Buff,
                pivotBuff,
                candidatePivotBuff,
                candidatePivotBuffPerm,
                ipiv, perm,
                numRounds,
                Px, layrK,
                lu_comm,
                k);
#ifdef DEBUG
            // after the tournament pivoting is finished, A00 has global pivots
            // that are independent of n_local_active_rows
            // no global pivot should be 0
            for (int i = 0; i < v; ++i){
                assert(std::abs(A00Buff[i*v + i]) > 0.001);
            }
#endif

            // std::cout << "tournament rounds finished" << std::endl;
// extract the first col of candidatePivotBuff first v elements of the first column of candidatePivotBuff
            // first v rows
            // v+1 is the number of cols
            // std::cout << "candidatePivotBuff:" << std::endl;;
            // print_matrix(candidatePivotBuff.data(), 0, v, 0, v+1, v+1);

            //if (n_local_active_rows > 0) {
            column<T, int>(matrix_view<T>(&candidatePivotBuff[0],
                                         min_perm_size, v + 1, v + 1,
                                         layout
                                         ),
                           0,
                           &gpivots[0]);
            PL();

#ifdef DEBUG
                if (k == chosen_step && debug_level > -1 && rank == print_rank) {
                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k: " << k << ", n_local_active_rows: "
                              << n_local_active_rows << ", candidatePivotBuff after tournament pivoting: \n"
                              << std::flush;
                    print_matrix(candidatePivotBuff.data(), 0, Ml, 0, v + 1, v + 1);
                    std::cout << "\n\n"
                              << std::flush;

                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k: " << k << ", gpivots: \n" << std::flush;
                    print_matrix(gpivots.data(), 0, 1, 0, N, N);
                    std::cout << "\n\n"
                              << std::flush;

                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k: " << k << ", lpivots: \n" << std::flush;
                    print_map(lpivots);
                    std::cout << "\n\n"
                              << std::flush;

                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k: " << k << ", lpivots_offsets: \n" << std::flush;
                    print_map(loffsets);
                    std::cout << "\n\n"
                              << std::flush;
                }
#endif

            PE(step1_A00Buff_isend);
            // send A00 to pi = k % sqrtp1 && pk = layrK
            // pj = k % sqrtp1; pk = layrK
            if (pi < Py) {
#ifdef DEBUG
                if (debug_level > 1) {
                    if (k == chosen_step) {
                        std::cout << "Isend: (" << pi << ", " << pj << ", " << pk << ")->(" << k % Px << ", " << pi << ", " << layrK << ")" << std::endl;
                        std::cout << "k = " << k << ", A00Buff = " << std::endl;
                        print_matrix(A00Buff.data(), 0, v, 0, v, v);
                    }
                }
#endif
                auto p_rcv = X2p(lu_comm, k % Px, pi, layrK);
                if (p_rcv != rank) {
                    MPI_Isend(&A00Buff[0], v * v, MPI_DOUBLE,
                              p_rcv, 50, lu_comm, &A00_req[n_A00_reqs++]);
                }
            }
            PL();
        }
        // (pi, k % sqrtp1, layrK) -> (k % sqrtp1, pi, layrK)
        // # Receiving A00Buff:
        PE(step1_A00Buff_irecv);
        if (pj < Px && pi == k % Px && pi < Py && pk == layrK) {
            // std::cout << "Irecv: (" << pj << ", " << pi << ", " << layrK << ")->(" << pi << ", " << pj << ", " << pk << ")" << std::endl;
            auto p_send = X2p(lu_comm, pj, pi, layrK);
            if (p_send != rank) {
                MPI_Irecv(&A00Buff[0], v * v, MPI_DOUBLE,
                          p_send, 50, lu_comm, &A00_req[n_A00_reqs++]);
            }
        }
        PL();

#ifdef DEBUG
        MPI_Barrier(lu_comm);
        
        if (debug_level > 1 && k == chosen_step && rank == print_rank) {
                std::cout << "After ircv. Rank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ", A00Buff = " << std::endl;
                print_matrix(A00Buff.data(), 0, v, 0, v, v);
        }
#endif

        // COMMUNICATION
        // MPI_Request reqs_pivots[4];
        // the one who entered this is the root
        auto root = X2p(jk_comm, k % Py, layrK);

        PE(step1_curPivots);
        MPI_Bcast(&gpivots[0], min_perm_size, MPI_INT, root, jk_comm);
        PL();

        std::unordered_map<int, std::vector<int>> lpivots;
        std::unordered_map<int, std::vector<int>> loffsets;
        std::tie(lpivots, loffsets) = g2lnoTile(gpivots, min_perm_size, Px, v);

        // locally set curPivots
        /*
             because the thing is that n_local_active_rows is BEFORE tournament pivoting
             so you entered the tournnament with empty hands but at least 
             you should tell others what was the outcome of the tournament. 
             So other ranks produced A00, gpivots, etc. and this information has to be propagated further
             */
        curPivots[0] = lpivots[pi].size();
        // if (curPivots[0] > 0) {
        std::copy_n(&lpivots[pi][0], curPivots[0], &curPivots[1]);
        std::copy_n(&loffsets[pi][0], curPivots[0], &curPivOrder[0]);
        // curPivOrder = loffsets[pi];
        std::copy_n(&gpivots[0], v, &pivotIndsBuff[k * v]);

        // PL();

        // assert(curPivots[0] <= v && curPivots[0] >= 0);

        // wait for both broadcasts
        // MPI_Waitall(4, reqs_pivots, MPI_STATUSES_IGNORE);
        // # ---------------------------------------------- #
        // # 2. reduce pivot rows from A11buff to PivotA01ReductionBuff #
        // # ---------------------------------------------- #
        ts = te;
        // curPivots = pivotIndsBuff[k * v: (k + 1) * v]
        // # Currently, we dump everything to processors in layer pk == 0, pi == k % sqrtp1
        // # and only this strip distributes reduced pivot rows
        // # so layer pk == 0 do a LOCAL copy from A11Buff to PivotBuff, other layers do the communication
        // # that is, each processor [pi, pj, pk] sends to [pi, pj, 0]
        // update the row mask

#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (debug_level > 1) {
            if (chosen_step == k) {
                if (pi == 0 && pj == 1 && pk == 0) {
                    std::cout << "A11 before pushing pivots up" << std::endl;
                    print_matrix(A11Buff.data(),
                                 0, Ml,
                                 0, Nl,
                                 Nl);
                    std::cout << "GRI before pushing pivots up" << std::endl;
                    print_matrix(gri.data(),
                                 0, 1,
                                 0, Ml,
                                 Ml);
                    std::cout << "curPivots before pushing pivots up" << std::endl;
                    print_matrix(curPivots.data(),
                                 0, 1,
                                 0, v + 1,
                                 v + 1);
                    std::cout << "first non pivot row = " << first_non_pivot_row << std::endl;
                }
            }
        }
        MPI_Barrier(lu_comm);
#endif

#ifdef DEBUG
        assert(n_local_active_rows >= 0);

        MPI_Allgather(&n_local_active_rows, 1, MPI_INT,
                      &activeRows[0], 1, MPI_INT, i_comm);

        int sum_rows = 0;
        for (auto& el : activeRows) {
            if (el > 0) sum_rows += el;
        }

        if (sum_rows != N - k*v) {
            if (pj == 0) {
                std::cout << "Rank [" << pi << ", " << pj << "," << pk << "], k: " 
                << k << ", sum_rows = " << sum_rows << ", N-kv = " << N-k*v 
                << ", n_local_active_rows: " << n_local_active_rows << std::endl;
                std::cout << "Active rows: \n";
                print_matrix(activeRows.data(), 0, 1, 0, Px, Px);
            }
        }
        assert(sum_rows == N - k*v);

        if (k == chosen_step && debug_level > 1) {
            for (int p = 0; p < P; ++p) {
                if (rank == p) {
                    std::cout << "[Rank " << p << "], step: " << k << ", sum_rows = " << sum_rows << ", N-kv = " << N-k*v << std::endl;
                    std::cout << "Nl = " << Nl << ", n_local_active_rows = " << n_local_active_rows << ", first_non_pivot_row = " << first_non_pivot_row << std::endl;
                    std::cout << "curPivots: " << curPivots[0] << " | ";
                    for (int i = 0; i < curPivots[0]; ++i) {
                        std::cout << curPivots[i+1] << ", ";
                    }
                    std::cout << std::endl;
                    std::cout << "======================" << std::endl;
                }
                MPI_Barrier(lu_comm);
            }
        }
#endif

        PE(step1_curPivots);
        // MPI_Wait(&curPivots_bcast_req, MPI_STATUS_IGNORE);
        for (int i = 0; i < curPivots[0]; ++i) {
            auto pivot_row = igri[curPivots[i+1]];
            if (pivot_row < first_non_pivot_row || pivot_row >= Ml) {
                std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ", curPivots[0]: " << curPivots[0] << ", first_non_pivot_row: " << first_non_pivot_row << ", pivot_row: " << pivot_row  << ", n_rows: " << Ml << std::endl;;
                std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ", curPivots:\n";
                print_matrix(curPivots.data(), 0, 1, 0, 2*v+1, 2*v+1);
                std::cout << "\nRank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ", igri:\n";
                print_map(igri);
            }
            assert(first_non_pivot_row >= Ml 
                   || 
                  (first_non_pivot_row <= pivot_row && pivot_row < Ml));
            curPivots[i + 1] = pivot_row;
        }
        PL();

        PE(step2_pushingpivots);

        analyze_pivots(first_non_pivot_row, Ml,
                       curPivots, pivots, early_non_pivots, late_pivots);

        push_pivots_up<T>(A11Buff, A11BuffTemp,
                          Ml, Nl,
                          layout, curPivots,
                          first_non_pivot_row,
                          pivots,
                          early_non_pivots,
                          late_pivots
                          );
#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Pushed pivots up in A11Buff" << std::endl;
        }
#endif

#ifdef CONFLUX_WITH_VALIDATION
        push_pivots_up<T>(A10resultBuff, A11BuffTemp,
                          Ml, Nl,
                          layout, curPivots,
                          first_non_pivot_row,
                          pivots,
                          early_non_pivots,
                          late_pivots
                          );
#endif

        push_pivots_up<T>(A10Buff, A10BuffTemp,
                          Ml, v,
                          layout, curPivots,
                          first_non_pivot_row,
                          pivots,
                          early_non_pivots,
                          late_pivots
                          );

#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Pushed pivots up in A10Buff" << std::endl;
        }
#endif

        push_pivots_up<int>(gri, griTemp,
                            Ml, 1,
                            layout, curPivots,
                            first_non_pivot_row,
                            pivots,
                            early_non_pivots,
                            late_pivots
                            );
        PL();

#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Finished pushing pivots up" << std::endl;
        }
        if (debug_level > 1) {
            if (chosen_step == k) {
                if (pi == 0 && pj == 1 && pk == 0) {
                    std::cout << "A11 after pushing pivots up" << std::endl;
                    print_matrix(A11Buff.data(),
                                 0, Ml,
                                 0, Nl,
                                 Nl);
                    std::cout << "GRI after pushing pivots up" << std::endl;
                    print_matrix(gri.data(),
                                 0, 1,
                                 0, Ml,
                                 Ml);
                }
            }
        }
        MPI_Barrier(lu_comm);
#endif

        first_non_pivot_row += curPivots[0];
        n_local_active_rows -= curPivots[0];

        assert(n_local_active_rows >= 0);

        for (int i = 0; i < Ml; ++i) {
            igri[gri[i]] = i;
        }

        // for A01Buff
        // TODO: NOW: reduce pivot rows: curPivots[0] x (Nl-loff)
        //
      //  if (n_local_active_rows > 0) {
        if (Pz > 1) {
            PE(step2_localcopy);
            // we have curPivots[0] pivot rows to copy from A11Buff to A01Buff
            // But - the precise row location in A01Buff is determined by the curPivOrder,
            // so i-th pivot row goes to curPivOrder[i] row in A01Buff
            // HOWEVER. To coalesce the reduction operation, and make A01Buff for reduction dense and not sparse,
            // we put them in top curPivots[0] of A01BuffTemp. And then, only after the reduction took place, we
            // use MPI_Put to properly distribute in correct order pivot rows from A01BuffTemp to A01Buff
#pragma omp parallel for shared(curPivots, first_non_pivot_row, A11Buff, Nl, loff, A01BuffTemp)
            for (int i = 0; i < curPivots[0]; ++i) {
                // if (pi == 0 && pj == 1 && pk == 0 && (k % Px) == 0){
                //     std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "]. curPivOrder: \n";
                //             print_matrix(curPivOrder.data(), 0, 1,
                //                         0, v, v);
                // }
                int pivot_row = first_non_pivot_row - curPivots[0] + i;
                std::copy_n(&A11Buff[pivot_row * Nl + loff], Nl - loff, &A01BuffTemp[i * (Nl - loff)]);
            }

            // if (pi == 0 && pj == 1 && pk == 0 && (k % Px) == 0){
            //     std::cout << "A01Buff before reduce. Rank [" << pi << ", " << pj << ", " << pk << "]:" << std::endl << std::flush;
            //     print_matrix(A01Buff.data(), 0, v, 0, Nl, Nl);
            // }
            // MPI_Barrier(lu_comm);

            PL();

            PE(step2_reduce);
            if (pk == layrK) {
                MPI_Reduce(MPI_IN_PLACE, &A01BuffTemp[0],
                           curPivots[0] * (Nl - loff),
                           MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            } else {
                MPI_Reduce(&A01BuffTemp[0], &A01BuffTemp[0],
                           curPivots[0] * (Nl - loff),
                           MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            }
            PL();
        }
  //      }

        // MPI_Barrier(lu_comm);

        te = std::chrono::high_resolution_clock::now();
        timers[2] += std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();

        ts = te;
#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Finished step 2" << std::endl;
        }
        if (debug_level > 0) {
            if (k == chosen_step) {
                if (rank == print_rank) {
                    std::cout << "Step 2 finished." << std::endl;
                    print_matrix(A11Buff.data(), 0, n_local_active_rows,
                                 0, Nl, Nl);
                }
                MPI_Barrier(lu_comm);
            }
        }
#endif

        // # -------------------------------------------------- #
        // # 3. distribute v pivot rows from A11buff to A01Buff #
        // # here, only processors pk == layrK participate      #
        // # -------------------------------------------------- #
        PE(step3_put);
        if (pk == layrK) {
            // curPivOrder[i] refers to the target
            auto p_rcv = X2p(lu_comm, k % Px, pj, layrK);

            // if (pi == 1 && pj == 1 && pk == 0 && (k % Px) == 0){
            //         std::cout << "Sending A01Buff. Rank [" << pi << ", " << pj << ", " << pk << "] -> ["
            //               << k % Px << ", " << pj << ", " << layrK << "], Sender's A01Buff: " << std::endl;
            //         print_matrix(A01Buff.data(), 0, v, 0, Nl, Nl);

            //     }

            for (int i = 0; i < curPivots[0]; ++i) {
                int piv_order = curPivOrder[i];
                assert(piv_order >= 0 && piv_order < v);
                auto dest_dspls = piv_order * (Nl - loff);
                int pivot_row = first_non_pivot_row - curPivots[0] + i;
                T* src_ptr = Pz > 1 ? &A01BuffTemp[i* (Nl-loff)] : &A11Buff[pivot_row * Nl + loff];
                MPI_Put(src_ptr, Nl - loff, MPI_DOUBLE,
                        p_rcv, dest_dspls, Nl - loff, MPI_DOUBLE,
                        A01Win);
            }
        }
        MPI_Win_fence(0, A01Win);

        PL();

        // MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[3] += std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();

#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Finished step 3" << std::endl;
        }
        if (debug_level > 0) {
            if (k == chosen_step) {
                if (rank == print_rank) {
                    std::cout << "Step 3 finished." << std::endl;
                    print_matrix(A01Buff.data(), 0, v, 0, Nl, Nl);
                }
                MPI_Barrier(lu_comm);
            }
        }
#endif

        ts = te;

        /*
            PE(step1_A00Buff_bcast);
            MPI_Wait(&A00_bcast_req, MPI_STATUS_IGNORE);
            PL();
        */
        PE(step1_A00Buff_waitall);
        if (n_A00_reqs > 0) {
            MPI_Waitall(n_A00_reqs, &A00_req[0], MPI_STATUSES_IGNORE);
#ifdef DEBUG
            // after the tournament pivoting is finished, A00 has global pivots
            // that are independent of n_local_active_rows
            // no global pivot should be 0
            for (int i = 0; i < v; ++i){
                assert(std::abs(A00Buff[i*v + i]) > 0.001);
            }
#endif
        }
        PL();

        // if (n_local_active_rows <= 0) continue;
        MPI_Request reqs[2];

        // # ---------------------------------------------- #
        // # 4. compute A10 and broadcast it to A10BuffRecv #
        // # ---------------------------------------------- #
        if (pk == layrK && pj == k % Py) {// && n_local_active_rows > 0) {
            // # this could basically be a sparse-dense A10 = A10 * U^(-1)   (BLAS tiangular solve) with A10 sparse and U dense
            // however, since we are ignoring the mask, it's dense, potentially with more computation than necessary.
#ifdef DEBUG
            if (debug_level > 1) {
                if (k == chosen_step) {
                    std::cout << "before trsm." << std::endl;
                    if (rank == print_rank) {
                        std::cout << "chosen_step = " << chosen_step << std::endl;
                        std::cout << "A00Buff = " << std::endl;
                        print_matrix(A00Buff.data(), 0, v, 0, v, v);
                        std::cout << "A10Buff = " << std::endl;
                        print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
                    }
                }
            }
#endif
            PE(step4_dtrsm);
            cblas_dtrsm(CblasRowMajor,  // side
                        CblasRight,     // uplo
                        CblasUpper,
                        CblasNoTrans,
                        CblasNonUnit,
                        n_local_active_rows,                //  M
                        v,                                  // N
                        1.0,                                // alpha
                        &A00Buff[0],                        // triangular A
                        v,                                  // leading dim triangular
                        &A10Buff[first_non_pivot_row * v],  // A11
                        v);
            PL();
#ifdef DEBUG
            if (debug_level > 1) {
                if (k == chosen_step) {
                    std::cout << "after trsm." << std::endl;

                    if (rank == print_rank) {
                        std::cout << "A10Buff after trsm" << std::endl;
                        print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
                    }
                }
            }
#endif

            PE(step4_reshuffling);

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
#pragma omp parallel for shared(A10Buff, A10BuffTemp, first_non_pivot_row, Ml, v, n_local_active_rows, nlayr)
            for (int pk_rcv = 0; pk_rcv < Pz; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv * nlayr;
                auto colEnd = (pk_rcv + 1) * nlayr;

                int offset = colStart * n_local_active_rows;
                mcopy(A10Buff.data(), &A10BuffTemp[offset],
                      first_non_pivot_row, Ml, colStart, colEnd, v,
                      0, n_local_active_rows, 0, nlayr, nlayr);
            }
            PL();
        }

        auto root_trsm_1 = X2p(jk_comm, k % Py, layrK);
        for (int p = 0; p < trsm_1_dspls.size(); ++p) {
            int ppj, ppk;
            std::tie(ppj, ppk) = p2X_2d(jk_comm, p);

            // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
            auto colStart = ppk * nlayr;
            // auto colEnd   = (pk_rcv+1)*nlayr;

            int offset = colStart * n_local_active_rows;
            int size = nlayr * n_local_active_rows;  // nlayr = v / c

            trsm_1_dspls[p] = offset;
            trsm_1_counts[p] = size;
        }

        PE(step4_comm);
        MPI_Iscatterv(&A10BuffTemp[0], 
                     &trsm_1_counts[0], 
                     &trsm_1_dspls[0], 
                     MPI_DOUBLE, 
                     &A10BuffRcv[0],
                     trsm_1_counts[jk_rank],
                     MPI_DOUBLE,
                     root_trsm_1,
                     jk_comm,
                     &reqs[0]
                     );
        PL();

#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Finished step 4" << std::endl;
        }
        if (debug_level > 1) {
            if (k == chosen_step) {
                if (rank == print_rank) {
                    std::cout << "Step 4 finished." << std::endl;
                    std::cout << "Matrix A10BuffRcv = " << std::endl;
                    print_matrix(A10BuffRcv.data(), 0, n_local_active_rows, 0, nlayr, nlayr);
                }
                MPI_Barrier(lu_comm);
            }
        }
#endif

        auto lld_A01 = Nl - loff;

        // # ---------------------------------------------- #
        // # 5. compute A01 and broadcast it to A01BuffRecv #
        // # ---------------------------------------------- #
        // # here, only ranks which own data in A01Buff (step 3) participate
        if (pk == layrK && pi == k % Px) {
#ifdef DEBUG
            if (debug_level > 1) {
                if (k == chosen_step) {
                    if (rank == print_rank) {
                        std::cout << "before trsm. Rank [" << pi << ", " << pj << ", " << pk << "]" << std::endl;
                        std::cout << "A00Buff = " << std::endl;
                        print_matrix(A00Buff.data(), 0, v, 0, v, v);
                        std::cout << "A01Buff = " << std::endl;
                        print_matrix(A01Buff.data(), 0, v, 0, Nl - loff, Nl - loff);
                    }
                }
            }
#endif

            PE(step5_dtrsm);
            // # this is a dense-dense A01 =  L^(-1) * A01
            cblas_dtrsm(CblasRowMajor,  // side
                        CblasLeft,
                        CblasLower,
                        CblasNoTrans,
                        CblasUnit,
                        v,            //  M
                        Nl - loff,    // N
                        1.0,          // alpha
                        &A00Buff[0],  // triangular A
                        v,            // leading dim triangular
                        &A01Buff[0],  // A01
                        lld_A01);     // leading dim of A01
            PL();

#ifdef DEBUG
            if (debug_level > 1) {
                if (k == chosen_step) {
                    if (rank == print_rank) {
                        std::cout << "AFTER trsm. Rank [" << pi << ", " << pj << ", " << pk << "]" << std::endl;
                        std::cout << "A01Buff = " << std::endl;
                        print_matrix(A01Buff.data(), 0, v, 0, Nl - loff, Nl - loff);
                    }
                }
            }
#endif

        }

        auto root_trsm_2 = X2p(ik_comm, k % Px, layrK);

        for (int p = 0; p < trsm_2_dspls.size(); ++p) {
            int ppi, ppk;
            std::tie(ppi, ppk) = p2X_2d(ik_comm, p);

            const int n_cols = Nl - loff;
            auto rowStart = ppk * nlayr;
            int offset = rowStart * n_cols;
            int size = nlayr * n_cols;

            trsm_2_dspls[p] = offset;
            trsm_2_counts[p] = size;
        }

        PE(step5_comm);
        MPI_Iscatterv(&A01Buff[0], 
                     &trsm_2_counts[0], 
                     &trsm_2_dspls[0], 
                     MPI_DOUBLE, 
                     &A01BuffRcv[0],
                     trsm_2_counts[ik_rank],
                     MPI_DOUBLE,
                     root_trsm_2,
                     ik_comm,
                     &reqs[1]);
        PL();

#ifdef DEBUG
        if (debug_level > 0) {
            if (k == chosen_step) {
                if (rank == print_rank) {
                    std::cout << "Step 5 finished." << std::endl;

                    std::cout << "A01BuffRcv = " << std::endl;
                    print_matrix(A01BuffRcv.data(), 0, nlayr, 0, Nl, Nl);

                    std::cout << "A11 (before) = " << std::endl;
                    print_matrix(A11Buff.data(), 0, Ml,
                                 0, Nl, Nl);
                }
                MPI_Barrier(lu_comm);
            }
        }
#endif

        ts = te;

        MPI_Waitall(2, &reqs[0], MPI_STATUSES_IGNORE);

        // # ---------------------------------------------- #
        // # 7. compute A11  ------------------------------ #
        // # ---------------------------------------------- #
        // # filter which rows of this tile should be processed:
        // rows = A11MaskBuff[p]
        // assumptions:
        // 1. we don't do the filtering
        // 2. A10BuffRcv is column-major
        // 3. A01BuffTemp is densified and leading dimensions = Nl-loff, row-major
        PE(step6_dgemm);
    //    if (n_local_active_rows > 0) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        n_local_active_rows, Nl - loff, nlayr,
                        -1.0, &A10BuffRcv[0], nlayr,
                        &A01BuffRcv[0], Nl - loff,
                        1.0, &A11Buff[first_non_pivot_row * Nl + loff], Nl);
      //  }
        PL();

        te = std::chrono::high_resolution_clock::now();
        timers[6] += std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();

#ifdef DEBUG
        if (debug_level > 0) {
            if (k == chosen_step) {
                if (rank == print_rank) {
                    std::cout << "A11Buff after computeA11:" << std::endl;
                    print_matrix(A11Buff.data(), 0, Ml,
                                 0, Nl,
                                 Nl);
                    std::cout << "A10Buff after storing the results back:" << std::endl;
                    print_matrix(A10Buff.data(), 0, Ml,
                                 0, v,
                                 v);
                    std::cout << "A10BuffRcv after storing the results back:" << std::endl;
                    print_matrix(A10BuffRcv.data(), 0, Ml,
                                 0, nlayr,
                                 nlayr);
                }
            }
        }
#endif

#ifdef CONFLUX_WITH_VALIDATION
        // INEFFICIENT! But should be cheap
        for (int i = 0; i < v; i++) {
            std::swap(ipvt_g[k*v + i], ipvt_g[pivotIndsBuff[k*v + i]]);            
        }
        if (pi == k % Px) {
            for (int i = 0; i < v; i++) {
                ipvt[i + loff] = pivotIndsBuff[k*v + i];
            }
        }

        int locK = k / Py;
        // A10
        if (k > 0) {
            // Since we are looking at the past A10Buff from previous iterations, all ranks on pk = layrK hold data for storing
            if (pk == layrK) { 
                // the data is in A10Buffs, but we need to reshuffle it properly
                // again, we will put it row by row
                // our rank pi has curPivots[0] pivots in this round. Therefore, it has to store curPivots[0] rows from A10Buff
                // from previous iteration.
                for (int ii = 0; ii < curPivots[0]; ii++) {
                    int i = curPivOrder[ii];  // ii sis the ii'th pivot in this round. i is its row location
                    int A10_row_offset = (ii + first_non_pivot_row - curPivots[0]) * Nl;
                    int g_dest_row = i + k * v;
                    // check which rank will be the owner of this pivot row after row swapping
                    int dest_pi = k % Px;
                    int dest_p = X2p(lu_comm, dest_pi, pj, layrK);
                    int dest_row_offset = (i + locK * v) * Nl;
                    if (dest_pi > pj) {
                        MPI_Put(&A10resultBuff[A10_row_offset], (locK+1) * v, MPI_DOUBLE,
                            dest_p, dest_row_offset, (locK+1) * v, MPI_DOUBLE,
                            res_Win);
                    }
                    else {
                        MPI_Put(&A10resultBuff[A10_row_offset], locK * v, MPI_DOUBLE,
                            dest_p, dest_row_offset, locK * v, MPI_DOUBLE,
                            res_Win);
                    }
                }
            }
        }

        #ifdef DEBUG
        // MPI_Win_fence(0, res_Win);    
        // if (k > 0 && pi == 2 && pj == 0 && pk == layrK && ScaLAPACKResultBuff[3] > -6.4) { 
        //         std::cout << "\nk= " << k <<", rank [" << pi << ", " << pj << ", " << pk << "] (" << rank << ") " << 
        //                 ", ScaLAPACKResultBuff: \n";
        //                 print_matrix(ScaLAPACKResultBuff.data(), 0, Nl,
        //                         0, Nl,
        //                         Nl);
        // }
        #endif

        #ifdef DEBUG
        if (pj == k % Py && debug_level > 1) {
            std::cout << "\nk= " << k <<", rank [" << pi << ", " << pj << ", " << pk << "] (" << rank << ") " << 
                     ", n_local_active_rows: " << n_local_active_rows << "\n";
        }
        #endif

        // A01 and A00: these are the ranks that own the pivot data in this round
        if (pk == layrK && pi == k % Px) {
            // A01Buff (and therefore, our current v pivots)
            if (k < Nt - 1) {
                // due to the column densification of A01, we now need to know how many columns does A01 have
                int A01cols = Nl - v * (k / Py);
                // offsets in resultBuff
                int rowOffset = Nl * v * locK;
                int colOffset;
                if (pi > pj) {
                    colOffset = v * (locK + 1);
                    parallel_mcopy<T>(v, Nl - loff - v,
                                  &A01Buff[v], A01cols,
                                  &ScaLAPACKResultBuff[rowOffset + colOffset], Nl);
                }
                else {
                    colOffset = v * locK;
                    parallel_mcopy<T>(v, Nl - loff,
                                  &A01Buff[0], A01cols,
                                  &ScaLAPACKResultBuff[rowOffset + colOffset], Nl);
                }
            }

            // # -- A00 -- #
            // now additionally, we filter ranks by pj, so finally, only a single rank will store A00 in this round
            if (pj == k % Py) {                    
                // offsets in resultBuff
                int rowOffset = Nl * v * locK;
                int colOffset = v * locK;                
                parallel_mcopy<T>(v, v,
                                    &A00Buff[0], v,
                                    &ScaLAPACKResultBuff[rowOffset + colOffset], Nl);
            }

        }

        // storing the final result back
        // storing back A10
        // ranks entering the pivoting are the same ranks
        // that have to update A10
        //
        // the only ranks that need to receive A00 buffer
        // are the one participating in dtrsm(A01Buff)
        PE(storingresults)
        if (pj == k % Py && pk == layrK) {
            // condensed A10 to non-condensed result buff
            // n_local_active_rows already reduced beforehand
#pragma omp parallel for shared(first_non_pivot_row, curPivots, Ml, A10Buff, A10resultBuff, Nl, loff, v)
            for (int i = first_non_pivot_row - curPivots[0]; i < Ml; ++i) {
                std::copy_n(&A10Buff[i * v], v, &A10resultBuff[i * Nl + loff]);
            }
        }
        PL();

#ifdef DEBUG
        if (debug_level > 0) {
            if (k == chosen_step) {
                if (rank == print_rank) {
                    std::cout << "A01Buff after storing the results back:" << std::endl;
                    print_matrix(A01Buff.data(), 0, v,
                                 0, Nl,
                                 Nl);
                    std::cout << "A01BuffRcv after storing the results back:" << std::endl;
                    print_matrix(A01BuffRcv.data(), 0, nlayr,
                                 0, Nl,
                                 Nl);
                    std::cout << "A11Buff after storing the results back:" << std::endl;
                    print_matrix(A11Buff.data(), 0, Ml,
                                 0, Nl,
                                 Nl);
                }
                if (pi == 1 && pj == 0 && pk == 0) {
                    std::cout << "Superstep: " << k << std::endl;
                    std::cout << "A00Buff after storing the results back:" << std::endl;
                    print_matrix(A00Buff.data(),
                                 0, v,
                                 0, v,
                                 v);
                }
            }
        }
#endif
#endif
    }

    MPI_Barrier(lu_comm);
    if (rank == print_rank) {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        std::cout << "Runtime: " << double(duration) / 1000000 << " seconds" << std::endl;

        for (auto i = 0; i < 8; ++i) {
            std::cout << "Runtime " << i << ": " << double(timers[i]) / 1000000 << " seconds" << std::endl;
        }
    }

    MPI_Win_free(&A01Win);
#ifdef DEBUG
    MPI_Comm_free(&i_comm);
#endif

#ifdef CONFLUX_WITH_VALIDATION
    std::copy(pivotIndsBuff.begin(), pivotIndsBuff.end(), permutation);
    MPI_Win_fence(0, res_Win);
#else
#endif
}
}  // namespace conflux

// namespace conflux
