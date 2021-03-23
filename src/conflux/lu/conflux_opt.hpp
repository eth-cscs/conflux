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
int l2g(int pi, int ind, int sqrtp1) {
    return ind * sqrtp1 + pi;
}

void g2l(int gind, int sqrtp1,
         int &out1, int &out2) {
    out1 = gind % sqrtp1;
    out2 = (int)(gind / sqrtp1);
}

void g2lA10(int gti, int P, int &p, int &lti) {
    lti = (int)(gti / P);
    p = gti % P;
}

int l2gA10(int p, int lti, int P) {
    return lti * P + p;
}

void gr2gt(int gri, int v, int &gti, int &lri) {
    gti = (int)(gri / v);
    lri = gri % v;
}

std::tuple<int, int, int> p2X(MPI_Comm comm3D, int rank) {
    int coords[] = {-1, -1, -1};
    MPI_Cart_coords(comm3D, rank, 3, coords);
    return {coords[0], coords[1], coords[2]};
}

int X2p(MPI_Comm comm3D, int pi, int pj, int pk) {
    int coords[] = {pi, pj, pk};
    int rank;
    MPI_Cart_rank(comm3D, coords, &rank);
    return rank;
}

int X2p(MPI_Comm comm2D, int pi, int pj) {
    int coords[] = {pi, pj};
    int rank;
    MPI_Cart_rank(comm2D, coords, &rank);
    return rank;
}

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

int flipbit(int n, int k) {
    return n ^ (1ll << k);
}

int butterfly_pair(int pi, int r, int Px) {
    auto src_pi = flipbit(pi, r);
    if (src_pi >= Px) {
        if (r == 0)
            src_pi = pi;
        else {
            src_pi = flipbit(src_pi, r - 1);
            if (src_pi >= Px)
                src_pi = Px - 1;
        }
    }
    return src_pi;
    // return std::min(flipbit(pi, r), Px - 1);
}

// taken from COSMA
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

MPI_Comm create_comm(MPI_Comm &comm, std::vector<int> &ranks) {
    MPI_Comm newcomm;
    MPI_Group subgroup;

    MPI_Group comm_group;
    MPI_Comm_group(comm, &comm_group);

    MPI_Group_incl(comm_group, ranks.size(), ranks.data(), &subgroup);
    MPI_Comm_create_group(comm, subgroup, 0, &newcomm);

    MPI_Group_free(&subgroup);
    MPI_Group_free(&comm_group);

    return newcomm;
}

template <typename T>
void LUP(int n_local_active_rows, int v, int stride,
         T *pivotBuff, T *candidatePivotBuff,
         std::vector<int> &ipiv, std::vector<int> &perm) {
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
                    ) {
    if (first_non_pivot_row >= n_rows)
        return;
    // clear up from previous step
    early_non_pivots.clear();
    late_pivots.clear();

    for (int i = first_non_pivot_row; i < n_rows; ++i) {
        pivots[i] = false;
    }

    // map pivots to bottom rows (after non_pivot_rows)
    for (int i = 0; i < curPivots[0]; ++i) {
        int pivot_row = curPivots[i + 1];
        assert(first_non_pivot_row <= pivot_row && pivot_row < n_rows);
        // pivot_row = curPivots[i+1] <= Nl
        pivots[pivot_row] = true;
    }

    // ----------------------
    // extract non pivots from first v rows -> early non pivots
    // extract pivots from the rest of rows -> late pivots

    // extract from first pivot-rows those which are non-pivots
    for (int i = first_non_pivot_row;
         i < std::min(first_non_pivot_row + curPivots[0], n_rows); ++i) {
        if (!pivots[i]) {
            early_non_pivots.push_back(i);
        }
    }

    // extract from the rest, those which are pivots
    for (int i = first_non_pivot_row + curPivots[0];
         i < n_rows; ++i) {
        if (pivots[i]) {
            late_pivots.push_back(i);
        }
    }

    // std::cout << "late pivots = " << late_pivots.size() << std::endl;
    // std::cout << "early non pivots = " << early_non_pivots.size() << std::endl;
    assert(late_pivots.size() == early_non_pivots.size());
}

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

    for (int r = 0; r < n_rounds; ++r) {
        // auto src_pi = std::min(flipbit(pi, r), Px - 1);
        auto src_pi = butterfly_pair(pi, r, Px);
        auto p_rcv = X2p(lu_comm, src_pi, pj, pk);

        int req_id = 0;
        int n_reqs = (Px & (Px - 1) == 0) ? 2 : (Px + 2);
        // int n_reqs = Px+2;
        MPI_Request reqs[n_reqs];

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
        if (Px & (Px - 1) != 0) {
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
    }
}

std::pair<
    std::unordered_map<int, std::vector<int>>,
    std::unordered_map<int, std::vector<int>>>
g2lnoTile(std::vector<int> &grows, int Px, int v) {
    std::unordered_map<int, std::vector<int>> lrows;
    std::unordered_map<int, std::vector<int>> loffsets;

    for (unsigned i = 0u; i < grows.size(); ++i) {
        auto growi = grows[i];
        // # we are in the global tile:
        auto gT = growi / v;
        // # which is owned by:
        auto pOwn = int(gT % Px);
        // # and this is a local tile:
        auto lT = gT / Px;
        // # and inside this tile it is a row number:
        auto lR = growi % v;
        // # which is a No-Tile row number:
        auto lRNT = int(lR + lT * v);
        // lrows[pOwn].push_back(lRNT);
        lrows[pOwn].push_back(growi);
        loffsets[pOwn].push_back(i);
    }

    return {lrows, loffsets};
}

template <class T>
std::vector<T> LU_rep(T* C, // C is only used when CONFLUX_WITH_VALIDATION
                      T *PP, // C is only used when CONFLUX_WITH_VALIDATION
                      std::size_t *ipvt,
                      lu_params<T> &gv) {
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
    MPI_Comm k_comm = gv.k_comm;

    auto chosen_step = Nt - 1;
    // auto debug_level = 0;
    //auto chosen_step = 90;
    auto debug_level = 0;

    int print_rank = X2p(lu_comm, 0, 1, 0);

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

    std::vector<T>& A11Buff = gv.data;
    std::vector<T> A10resultBuff(Ml * Nl);
    std::vector<T> A11BuffTemp(Ml * Nl);

    //TODO: can we handle such a big global pivots vector?
    std::vector<T> pivotIndsBuff(M);
#ifdef FINAL_SCALAPACK_LAYOUT
    std::vector<T> ScaLAPACKResultBuff(Ml * Nl);
    MPI_Win res_Win = create_window(lu_comm,
                                    ScaLAPACKResultBuff.data(),
                                    ScaLAPACKResultBuff.size(),
                                    true);
    MPI_Win_fence(0, res_Win);
    // TODO: This is DEFINITELY suboptimal. We will use two buffers
    // (pivotIndsBuff and ipvt_g) to recreate local scalapack ipvt
    // Hopefully, it won't impact performance much.
    std::vector<T> ipvt_g(M);
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

    // 0 = num of pivots
    // 1..v+1 = pivots
    // v+1 .. 2v+1 = curPivOrder
    std::vector<int> curPivots(v + 1 + v);
    for (int i = v + 1; i < curPivots.size(); ++i) {
        curPivots[i] = i;
    }

    // GLOBAL result buffer
    // For debug only!
    std::cout << std::setprecision(3);

#ifdef CONFLUX_WITH_VALIDATION
    MPI_Win B_Win = create_window(lu_comm,
                                  &C[0],
                                  M * N,
                                  true);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, B_Win);
#endif

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
        if (debug_level > 1) {
            std::cout << "Iteration = " << k << std::endl;
            MPI_Barrier(lu_comm);
        }
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

        PE(step0_padding);
        if (n_local_active_rows < v) {
            int padding_start = n_local_active_rows * (v + 1);
            int padding_end = v * (v + 1);
            std::fill(candidatePivotBuff.begin() + padding_start,
                      candidatePivotBuff.begin() + padding_end, 0);
            std::fill(candidatePivotBuffPerm.begin() + padding_start,
                      candidatePivotBuffPerm.begin() + padding_end, 0);
            std::fill(gri.begin() + n_local_active_rows, gri.begin() + v, -1);
        }
        PL();

        // # reduce first tile column. In this part, only pj == k % sqrtp1 participate:
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

        if (pj == k % Py) {
            PE(step0_copy);
            parallel_mcopy<T>(n_local_active_rows, v,
                              &A11Buff[first_non_pivot_row * Nl + loff], Nl,
                              &A10Buff[first_non_pivot_row * v], v);
            PL();

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
        }

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
        if (pj == k % Py && pk == layrK) {
            auto min_perm_size = std::min(N - k * v, v);
            auto max_perm_size = std::max(n_local_active_rows, v);

            PE(step1_A10copy);
            parallel_mcopy<T>(n_local_active_rows, v,
                              &A10Buff[first_non_pivot_row * v], v,
                              &candidatePivotBuff[1], v + 1);
            assert(n_local_active_rows + first_non_pivot_row == Ml);
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
            if (k == chosen_step && debug_level > 1) {
                std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], n_local_active_rows: "
                          << n_local_active_rows << ", candidatePivotBuff after: \n"
                          << std::flush;
                print_matrix(candidatePivotBuff.data(), 0, Ml, 0, v + 1, v + 1);
                std::cout << "\n\n"
                          << std::flush;

                std::cout << "candidatePivotBuff FINAL VALUE " << pi << std::endl;
                print_matrix(candidatePivotBuff.data(),
                             0, n_local_active_rows, 0, v + 1, v + 1);
            }
#endif
            // std::cout << "tournament rounds finished" << std::endl;

            // extract the first col of candidatePivotBuff
            // first v elements of the first column of candidatePivotBuff
            // first v rows
            // v+1 is the number of cols
            // std::cout << "candidatePivotBuff:" << std::endl;;
            // print_matrix(candidatePivotBuff.data(), 0, v, 0, v+1, v+1);
            auto gpivots = column<T, int>(matrix_view<T>(&candidatePivotBuff[0],
                                                         min_perm_size, v + 1, v + 1,
                                                         layout),
                                          0);

            std::unordered_map<int, std::vector<int>> lpivots;
            std::unordered_map<int, std::vector<int>> loffsets;
            std::tie(lpivots, loffsets) = g2lnoTile(gpivots, Px, v);

            // locally set curPivots
            /*
                 because the thing is that n_local_active_rows is BEFORE tournament pivoting
                 so you entered the tournnament with empty hands but at least 
                 you should tell others what was the outcome of the tournament. 
                 So other ranks produced A00, gpivots, etc. and this information has to be propagated further
                 */
            curPivots[0] = lpivots[pi].size();
            std::copy_n(&lpivots[pi][0], curPivots[0], &curPivots[1]);
            std::copy_n(&loffsets[pi][0], curPivots[0], &curPivots[v + 1]);
            // curPivOrder = loffsets[pi];
            std::copy_n(&gpivots[0], v, &pivotIndsBuff[k * v]);
            PL();

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
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Finished tournament pivoting" << std::endl;
        }
        if (debug_level > 1) {
            if (k == chosen_step) {
                std::cout << "After ircv. Rank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ", A00Buff = " << std::endl;
                print_matrix(A00Buff.data(), 0, v, 0, v, v);
            }
        }
#endif

        // COMMUNICATION
        // MPI_Request reqs_pivots[4];
        // the one who entered this is the root
        auto root = X2p(jk_comm, k % Py, layrK);

        // # Sending A00Buff:
        /*
            PE(step1_A00Buff_bcast);
            MPI_Request A00_bcast_req;
            MPI_Ibcast(&A00Buff[0], v * v, MPI_DOUBLE, root, jk_comm, &A00_bcast_req);
            PL();
            */

        // sending pivotIndsBuff
        PE(step1_pivotIndsBuff);
        MPI_Request pivotIndsBuff_bcast_req;
        MPI_Ibcast(&pivotIndsBuff[k * v], v, MPI_DOUBLE, root, jk_comm, &pivotIndsBuff_bcast_req);
        PL();

        // PE(step1_barrier);
        // MPI_Barrier(lu_comm);
        // PL();

        PE(step1_curPivots);
        MPI_Bcast(&curPivots[0], 2 * v + 1, MPI_INT, root, jk_comm);
        PL();

        /*
            MPI_Request curPivOrder_bcast_req;
            MPI_Ibcast(&curPivOrder[0], v, MPI_INT, root, jk_comm, &curPivOrder_bcast_req);
            */
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
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Before pushing pivots up" << std::endl;
        }
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

        PE(step1_curPivots);
        // MPI_Wait(&curPivots_bcast_req, MPI_STATUS_IGNORE);
        for (int i = 0; i < curPivots[0]; ++i) {
            curPivots[i + 1] = igri[curPivots[i + 1]];
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

        for (int i = first_non_pivot_row; i < Ml; ++i) {
            igri[gri[i]] = i;
        }

        if (n_local_active_rows < 0) continue;

        // for A01Buff
        // TODO: NOW: reduce pivot rows: curPivots[0] x (Nl-loff)
        //
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

        /*
            PE(step1_curPivOrder);
            MPI_Wait(&curPivOrder_bcast_req, MPI_STATUS_IGNORE);
            PL();
            */
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
                auto dest_dspls = curPivots[v + 1 + i] * (Nl - loff);
                MPI_Put(&A01BuffTemp[i * (Nl - loff)], Nl - loff, MPI_DOUBLE,
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

        MPI_Request reqs[Pz * (Px + Py) + 2];
        int req_id = 0;

        ts = te;

        /*
            PE(step1_A00Buff_bcast);
            MPI_Wait(&A00_bcast_req, MPI_STATUS_IGNORE);
            PL();
            */
        PE(step1_A00Buff_waitall);
        if (n_A00_reqs > 0) {
            MPI_Waitall(n_A00_reqs, &A00_req[0], MPI_STATUSES_IGNORE);
        }
        PL();

        // RECEIVE FROM STEP 4
        auto p_send = X2p(lu_comm, pi, k % Py, layrK);
        int size = nlayr * n_local_active_rows;  // nlayr = v / c

        if (p_send != rank) {
            PE(step4_comm);
            MPI_Irecv(&A10BuffRcv[0], size, MPI_DOUBLE,
                      p_send, 5, lu_comm, &reqs[req_id]);
            ++req_id;
            PL();
        }

        // # ---------------------------------------------- #
        // # 4. compute A10 and broadcast it to A10BuffRecv #
        // # ---------------------------------------------- #
        if (pk == layrK && pj == k % Py) {
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

            PE(step4_comm);
            for (int pk_rcv = 0; pk_rcv < Pz; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv * nlayr;
                // auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                int size = nlayr * n_local_active_rows;  // nlayr = v / c

                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int pj_rcv = 0; pj_rcv < Py; ++pj_rcv) {
                    auto p_rcv = X2p(lu_comm, pi, pj_rcv, pk_rcv);
                    if (rank != p_rcv) {
                        MPI_Isend(&A10BuffTemp[offset], size, MPI_DOUBLE,
                                  p_rcv, 5, lu_comm, &reqs[req_id]);
                        ++req_id;
                    }
                }
            }
            auto colStart = pk * nlayr;
            int offset = colStart * n_local_active_rows;
            /*
                parallel_mcopy(nlayr, n_local_active_rows,
                              &A10BuffTemp[offset], v,
                              &A10BuffRcv[0], nlayr);
                              */
            std::copy_n(&A10BuffTemp[offset], nlayr * n_local_active_rows, &A10BuffRcv[0]);
            PL();
        }

        te = std::chrono::high_resolution_clock::now();
        timers[4] += std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();

#ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step && rank == print_rank) {
            std::cout << "Finished step 4" << std::endl;
        }
        if (debug_level > 0) {
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

        // RECEIVE FROM STEP 5
        p_send = X2p(lu_comm, k % Px, pj, layrK);
        size = nlayr * (Nl - loff);  // nlayr = v / c
        // if non-local, receive it
        if (p_send != rank) {
            PE(step5_irecv);
            MPI_Irecv(&A01BuffRcv[0], size, MPI_DOUBLE,
                      p_send, 6, lu_comm, &reqs[req_id]);
            ++req_id;
            PL();
        }

        ts = te;

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

            PE(step5_reshuffling);
            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for (int pk_rcv = 0; pk_rcv < Pz; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                auto rowStart = pk_rcv * nlayr;
                // auto rowEnd = (pk_rcv + 1) * nlayr;
                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                const int n_cols = Nl - loff;
                for (int pi_rcv = 0; pi_rcv < Px; ++pi_rcv) {
                    auto p_rcv = X2p(lu_comm, pi_rcv, pj, pk_rcv);
                    if (rank != p_rcv) {
                        PL();
                        PE(step5_isend);
                        MPI_Isend(&A01Buff[rowStart * n_cols],
                                  nlayr * n_cols, MPI_DOUBLE,
                                  p_rcv, 6, lu_comm, &reqs[req_id]);
                        ++req_id;
                        PL();
                        PE(step5_reshuffling);
                    }
                }
            }
            // perform the local copy outside of MPI
            PL();
            PE(step5_localreshuffling)
            const int row_start = pk * nlayr;
            const int n_cols = Nl - loff;
            parallel_mcopy(nlayr, n_cols,
                           &A01Buff[row_start * n_cols], n_cols,
                           &A01BuffRcv[0], n_cols);
            /*
                std::copy_n(&A01Buff[row_start*n_cols],
                            nlayr*n_cols,
                            &A01BuffRcv[0]);
                            */
            PL();
        }

        PE(step5_waitall);
        MPI_Waitall(req_id, reqs, MPI_STATUSES_IGNORE);
        PL();

        te = std::chrono::high_resolution_clock::now();
        timers[5] += std::chrono::duration_cast<std::chrono::microseconds>(te - ts).count();

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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_local_active_rows, Nl - loff, nlayr,
                    -1.0, &A10BuffRcv[0], nlayr,
                    &A01BuffRcv[0], Nl - loff,
                    1.0, &A11Buff[first_non_pivot_row * Nl + loff], Nl);
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

#ifdef FINAL_SCALAPACK_LAYOUT
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
                    int i = curPivots[v + 1 + ii];  // ii sis the ii'th pivot in this round. i is its row location
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
        MPI_Win_fence(0, res_Win);    
        if (k > 0 && pi == 2 && pj == 0 && pk == layrK && ScaLAPACKResultBuff[3] > -6.4) { 
                std::cout << "\nk= " << k <<", rank [" << pi << ", " << pj << ", " << pk << "] (" << rank << ") " << 
                        ", ScaLAPACKResultBuff: \n";
                        print_matrix(ScaLAPACKResultBuff.data(), 0, Nl,
                                0, Nl,
                                Nl);
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
#endif

        // storing the final result back
        // storing back A10
        // ranks entering the pivoting are the same ranks
        // that have to update A10
        //
        // the only ranks that need to receive A00 buffer
        // are the one participating in dtrsm(A01Buff)
#ifdef CONFLUX_WITH_VALIDATION
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

        // MPI_Wait(&pivotIndsBuff_bcast_req, MPI_STATUS_IGNORE);

        PE(step1_pivotIndsBuff);
        MPI_Wait(&pivotIndsBuff_bcast_req, MPI_STATUS_IGNORE);
        PL();

        // # ----------------------------------------------------------------- #
        // # ------------------------- DEBUG ONLY ---------------------------- #
        // # ----------- STORING BACK RESULTS FOR VERIFICATION --------------- #

        // # -- A10 -- #
        // Storing A10 is funny. Even though A10 contains final results, it is not "finally permuted". The final result
        // will have the same data, but permuted according to future pivots. Therefore, because our final result B is already
        // L*U*P permuted (so it has pivots already put on the diagonal), we fill B only by columns (we know that after k iterations
        // top v*k rows of B will contain v*k pivots and will be untouched).

        // Sooo, the plan for A10 is to keep all the data from all the steps in A10resultBuff and keep on permuting it as we proceed
        // in the next iterations, and flush only top rows of A10resultBuff which correspond to already chosen pivots, so they will
        // be untouched.

        // in k-th iteration, we look at the PREVIOUS iteration (k-1) and store these rows of A10resultBuff, which correspond to the
        // pivots which were chosen in this round. Think of it like that:
        // in round k, we have found v pivot rows, which will be put on the diagonal of B. Now we fill all the data to the right
        // of this diagonal with current A01Buff, and to the left of this diagonal with previous A10ResultBuff.
        if (k > 0) {
            // Since we are looking at the past A10Buff from previous iterations, all ranks on pk = layrK hold data for storing
            MPI_Win_fence(0, B_Win);
            if (pk == layrK) {  // && pj == (k-1) % Py) {
                                // the data is in A10Buffs, but we need to reshuffle it properly
                                // if (k == chosen_step) {
#ifdef DEBUG
                if (debug_level > 1 && k == chosen_step) {
                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "]. curPivOrder: \n";
                    print_matrix(&curPivots[v + 1], 0, 1,
                                 0, v, v);

                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "]. A10resultBuff: \n";
                    print_matrix(A10resultBuff.data(), 0, Ml,
                                 0, Nl, Nl);

                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "]. curPivots[0]: " << curPivots[0] << "\n";
                    print_matrix(A10resultBuff.data(), 0, Ml,
                                 0, Nl, Nl);
                }
#endif
                // }

                // this is the start column tile of the GLOBAL output matrix B
                int local_tile_end = (k - 1) / Py;

                // again, we will put it row by row
                // our rank pi has curPivots[0] pivots in this round. Therefore, it has to store curPivots[0] rows from A10Buff
                // from previous iteration.
                for (int ii = 0; ii < curPivots[0]; ii++) {
                    int i = curPivots[v + 1 + ii];  // ii is the ii'th pivot in this round. i is its row location
                    int A10_row_offset = (ii + first_non_pivot_row - curPivots[0]) * Nl;
                    int B_row_offset = N * (i + k * v);
                    // we now loop over all tiles (going horizontally)
                    for (int j = 0; j < local_tile_end + 1; j++) {
                        // ok, sooooo, j is a local tile. The gloal column should be:
                        int B_col_offset = (j * Py + pj) * v;
#ifdef DEBUG
                        if (k == chosen_step && debug_level > 1) {  // pi == 0 && pj == 0 && pk == 0 &&
                            std::cout << "\n\nRank [" << pi << ", " << pj << ", " << pk << "]. "
                                      << "curPivots[0]: " << curPivots[0] << ", curPivOrder[ii]: " << curPivots[v + 1 + ii]
                                      << ", local_tile_end: " << local_tile_end << ", B_row_offset: "
                                      << B_row_offset << ", B_col_offset: " << B_col_offset
                                      << ", A10_row_offset: " << A10_row_offset + j * v << "\n"
                                      << std::flush;
                        }
#endif
                        MPI_Put(&A10resultBuff[A10_row_offset + j * v], v, MPI_DOUBLE,
                                0, B_row_offset + B_col_offset, v, MPI_DOUBLE,
                                B_Win);
                    }
                }
            }
        }

        // # -- A01 -- #
        if (k < Nt - 1) {
            // Ranks who own the final data: (pk == layrK && pi == k % Px)
            MPI_Win_fence(0, B_Win);
            if (pk == layrK && pi == k % Px) {
                // Cool. Now we need a proper column offset. Imagine that you have, e.g., Py = 4 ranks in y-dimension.
                // Then, depending on the iteration (k), some ranks will already have to skip their first v rows
                // (the onces that were processed). So after Py steps of the outermost k loop, every rank was processed
                int local_tile_offset = 0;
                if (k % Py >= pj) {
                    // then it means that our rank was already processed in this big round (one big round is Py iterations of k loop)
                    local_tile_offset++;
                }
                // this is the start column tile of the GLOBAL output matrix B
                int global_tile_offset = k / Py + local_tile_offset;

                // due to the column densification of A01, we now need to know how many columns does A01 have
                int A01cols = Nl - v * (k / Py);

                // if (k == chosen_step) {
#ifdef DEBUG
                if (debug_level > 0 && k > 12) {
                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ". A01: \n";
                    print_matrix(A01Buff.data(), 0, v,
                                 0, A01cols, A01cols);
                }
#endif
                // }

                // again, we will put it row by row
                for (int i = 0; i < v; i++) {
                    // we now loop over all tiles (going horizontally)
                    for (int j = global_tile_offset; j < tA11y; j++) {
                        int B_row_offset = N * (i + k * v);
                        // ok, sooooo, j is a local tile. The gloal column shoud be:
                        int B_col_offset = (j * Py + pj) * v;
#ifdef DEBUG
                        if (debug_level > 0 && pi == 1 && pj == 1 && pk == 0 && k == chosen_step) {
                            std::cout << "local_tile_offset: " << local_tile_offset << ", B_row_offset: "
                                      << B_row_offset << ", B_col_offset: " << B_col_offset << "\n"
                                      << std::flush;
                        }
#endif
                        MPI_Put(&A01Buff[i * A01cols + (j - k / Py) * v], v, MPI_DOUBLE,
                                0, B_row_offset + B_col_offset, v, MPI_DOUBLE,
                                B_Win);
                    }
                }
            }
        }

        // # -- A00 -- #
        // All the ranks which participated in this tournament round (pj == k % Py) own the same A00.
        // We just take an arbirary rank (pi == 0 and pk == 0)
        MPI_Barrier(lu_comm);
        MPI_Win_fence(0, B_Win);
        if (pi == 0 && pj == k % Py && pk == 0) {
            // we will put it row by row, since A00 is v x v, and global B is N x N, so it has a different stride
            for (int i = 0; i < v; i++) {
                int B_row_offset = N * (i + k * v);
                int B_col_offset = k * v;

#ifdef DEBUG
                if (debug_level > 1 && chosen_step == k) {
                    std::cout << "Rank [" << pi << ", " << pj << ", " << pk << "], k = " << k << ", A00Buff" << std::endl;
                    print_matrix(A00Buff.data(), 0, v,
                                 0, v, v);
                }
#endif

                MPI_Put(&A00Buff[i * v], v, MPI_DOUBLE,
                        0, B_row_offset + B_col_offset, v, MPI_DOUBLE,
                        B_Win);
            }
        }

        // Printing global B
        MPI_Win_fence(0, B_Win);

#ifdef DEBUG
        if (k == chosen_step) {
            if (debug_level > 0) {
                if (rank == 0) {
                    std::cout << "GLOBAL result matrix B" << std::endl;
                }
                MPI_Barrier(lu_comm);
                if (rank == 0) {
                    print_matrix(&C[0], 0, M,
                                 0, N, N);
                }
            }
        }

        if (debug_level > 0) {
            if (rank == print_rank) {
                std::cout << "k = " << k << ", pivotIndsBuff: \n";
                print_matrix(pivotIndsBuff.data(), 0, 1, 0, N, N);
                std::cout << "\n\n";
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
#ifdef CONFLUX_WITH_VALIDATION
    MPI_Win_free(&B_Win);
    // std::copy(B.begin(), B.end(), C);
    MPI_Barrier(lu_comm);
    for (auto i = 0; i < N; ++i) {
        int idx = int(pivotIndsBuff[i]);
        //std::copy(B.begin() + idx * N, B.begin() + (idx + 1) * N, C + i * N);
        PP[i * N + idx] = 1;
    }
#endif

#ifdef FINAL_SCALAPACK_LAYOUT
    MPI_Win_fence(0, res_Win);
    return ScaLAPACKResultBuff;
#else
    return {};
#endif

}
}  // namespace conflux

// namespace conflux
