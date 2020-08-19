#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>  // has std::lcm
#include <random>

#include <omp.h>
#include <mpi.h>
#include <mkl.h>

#include "profiler.hpp"

#define dtype double
#define mtype MPI_DOUBLE

// #def DEBUG_PRINT

template <class T>
void mcopy(T* src, T* dst,
           int ssrow, int serow, int sscol, int secol, int sstride,
           int dsrow, int derow, int dscol, int decol, int dstride) {
    PE(mcopy);
    auto srow = ssrow;
    auto drow = dsrow;
    for (auto i = 0; i < serow - ssrow; ++i) {
        std::copy(&src[srow * sstride + sscol],
                  &src[srow * sstride + secol],
                  &dst[drow * dstride + dscol]);
        srow++;
        drow++;
    }
    PL();
}

long long l2g(long long pi, long long ind, long long sqrtp1) {
    return ind * sqrtp1 + pi;
}

void g2l(long long gind, long long sqrtp1,
         long long& out1, long long& out2){
    out1 = gind % sqrtp1;
    out2 = (long long) (gind / sqrtp1);
}

void g2lA10(long long gti, long long P, long long& p, long long& lti) {
    lti = (long long) (gti / P);
    p = gti % P;
}

long long l2gA10(long long p, long long lti, long long P) {
    return lti * P + p;
}

void gr2gt(long long gri, long long v, long long& gti, long long& lri) {
    gti = (long long) (gri / v);
    lri = gri % v;
}

void p2X(long long p, long long p1, long long sqrtp1,
         long long& pi, long long& pj, long long&pk) {
    pk = (long long) (p / p1);
    p -= pk * p1;
    pj = (long long) (p / sqrtp1);
    pi = p % sqrtp1;
}

long long X2p(long long pi, long long pj, long long pk,
              long long p1, long long sqrtp1) {
    return pi + sqrtp1 * pj + p1 * pk;
}

double ModelCommCost(long long ppp, long long c) {
    return 1.0 / (ppp * c);
}

void CalculateDecomposition(long long P,
                            long long& best_ppp,
                            long long& best_c) {
    long long p13 = (long long) (std::pow(P + 1, 1.0 / 3));
    long long ppp = (long long) (std::sqrt(P));
    long long c = 1ll;
    best_ppp = ppp;
    best_c = c;
    double bestCost = ModelCommCost(ppp, c);
    while (c <= p13) {
        long long P1 = (long long )(P / c);
        ppp = (long long) (std::sqrt(P1));
        double cost = ModelCommCost(ppp, c);
        if (cost < bestCost) {
            bestCost = cost;
            best_ppp = ppp;
            best_c = c;
        }
        c++;
    }
    assert(best_ppp * best_ppp * best_c <= P);
}

template <class T>
class GlobalVars {

private:

    void CalculateParameters(long long inpN, long long inpP) {
        CalculateDecomposition(inpP, sqrtp1, c);
        // v = std::lcm(sqrtp1, c);
        // v = 64;
        // long long nLocalTiles = (long long) (std::ceil((double) inpN / (v * sqrtp1)));
        // N = v * sqrtp1 * nLocalTiles;
        // std::cout << sqrtp1 << " " << c << std::endl << std::flush;
        // std::cout << v << " " << nLocalTiles << std::endl << std::flush;
        v = 2;
        N = 16;
        sqrtp1 = 2;
        P = 8;
        p1 = 4;
        c = 2;
    }

    void InitMatrix() {
         if (N == 16) {
             matrix = new T[N * N]{1, 8, 2, 7, 3, 8, 2, 4, 8, 7, 5, 5, 1, 4, 4, 9,
                                   8, 4, 9, 2, 8, 6, 9, 9, 3, 7, 7, 7, 8, 7, 2, 8,
                                   3, 5, 4, 8, 9, 2, 7, 1, 2, 2, 7, 9, 8, 2, 1, 3,
                                   6, 4, 1, 5, 3, 7, 9, 1, 1, 3, 2, 9, 9, 5, 1, 9,
                                   8, 7, 1, 2, 9, 1, 1, 9, 3, 5, 8, 8, 5, 5, 3, 3,
                                   4, 2, 9, 3, 7, 3, 4, 5, 1, 9, 7, 7, 2, 4, 5, 2,
                                   1, 9, 8, 3, 5, 5, 1, 3, 6, 8, 3, 4, 3, 9, 1, 9,
                                   3, 9, 2, 7, 9, 2, 3, 9, 8, 6, 3, 5, 5, 2, 2, 9,
                                   9, 9, 5, 4, 3, 4, 6, 6, 9, 2, 1, 5, 6, 9, 5, 7,
                                   3, 2, 4, 5, 2, 4, 5, 3, 6, 5, 2, 6, 2, 7, 8, 2,
                                   4, 4, 4, 5, 2, 5, 3, 4, 1, 7, 8, 1, 8, 8, 5, 4,
                                   4, 5, 9, 5, 7, 9, 2, 9, 4, 6, 4, 3, 5, 8, 1, 2,
                                   7, 8, 1, 4, 7, 6, 5, 7, 1, 2, 7, 3, 8, 1, 4, 4,
                                   7, 6, 7, 8, 2, 2, 4, 6, 6, 8, 3, 6, 5, 2, 6, 5,
                                   4, 5, 1, 5, 3, 7, 4, 4, 7, 5, 8, 2, 4, 7, 1, 7,
                                   8, 3, 2, 4, 3, 8, 1, 6, 9, 6, 3, 6, 4, 8, 7, 8};
         } else if (N == 32) {
             matrix = new T[N * N] {9.0, 4.0, 8.0, 8.0, 3.0, 8.0, 0.0, 5.0, 2.0, 1.0, 0.0, 6.0, 3.0, 7.0, 0.0, 3.0, 5.0, 7.0, 3.0, 6.0, 8.0, 6.0, 2.0, 0.0, 8.0, 0.0, 8.0, 5.0, 9.0, 7.0, 9.0, 3.0,
 7.0, 4.0, 4.0, 6.0, 8.0, 9.0, 7.0, 4.0, 4.0, 7.0, 2.0, 1.0, 3.0, 2.0, 2.0, 2.0, 0.0, 0.0, 9.0, 4.0, 3.0, 6.0, 2.0, 9.0, 7.0, 0.0, 4.0, 8.0, 9.0, 4.0, 6.0, 1.0,
 9.0, 2.0, 9.0, 6.0, 6.0, 5.0, 2.0, 1.0, 2.0, 1.0, 7.0, 3.0, 0.0, 9.0, 8.0, 9.0, 9.0, 1.0, 3.0, 7.0, 6.0, 1.0, 8.0, 2.0, 2.0, 5.0, 5.0, 5.0, 0.0, 8.0, 2.0, 1.0,
 8.0, 9.0, 8.0, 8.0, 6.0, 5.0, 0.0, 4.0, 3.0, 2.0, 7.0, 4.0, 0.0, 2.0, 6.0, 0.0, 8.0, 4.0, 4.0, 5.0, 8.0, 3.0, 6.0, 5.0, 2.0, 8.0, 7.0, 6.0, 8.0, 8.0, 7.0, 8.0,
 6.0, 6.0, 6.0, 7.0, 1.0, 8.0, 8.0, 0.0, 8.0, 1.0, 3.0, 7.0, 1.0, 8.0, 8.0, 5.0, 0.0, 2.0, 6.0, 9.0, 6.0, 2.0, 6.0, 5.0, 7.0, 1.0, 7.0, 5.0, 9.0, 3.0, 6.0, 9.0,
 1.0, 9.0, 6.0, 0.0, 3.0, 7.0, 0.0, 5.0, 3.0, 6.0, 0.0, 8.0, 9.0, 9.0, 7.0, 1.0, 7.0, 0.0, 0.0, 3.0, 4.0, 7.0, 6.0, 4.0, 2.0, 9.0, 4.0, 4.0, 1.0, 7.0, 6.0, 2.0,
 0.0, 6.0, 6.0, 2.0, 9.0, 1.0, 4.0, 9.0, 4.0, 6.0, 3.0, 2.0, 9.0, 4.0, 8.0, 2.0, 2.0, 0.0, 6.0, 3.0, 8.0, 4.0, 9.0, 1.0, 8.0, 7.0, 7.0, 8.0, 7.0, 6.0, 1.0, 0.0,
 9.0, 6.0, 7.0, 4.0, 1.0, 1.0, 6.0, 4.0, 2.0, 4.0, 0.0, 5.0, 2.0, 7.0, 3.0, 4.0, 0.0, 0.0, 3.0, 4.0, 6.0, 2.0, 6.0, 8.0, 7.0, 0.0, 4.0, 1.0, 2.0, 9.0, 1.0, 4.0,
 6.0, 7.0, 5.0, 0.0, 3.0, 5.0, 0.0, 3.0, 0.0, 0.0, 3.0, 1.0, 5.0, 6.0, 8.0, 2.0, 1.0, 1.0, 6.0, 7.0, 0.0, 9.0, 0.0, 5.0, 7.0, 8.0, 7.0, 8.0, 3.0, 8.0, 0.0, 8.0,
 5.0, 8.0, 4.0, 6.0, 5.0, 7.0, 0.0, 0.0, 2.0, 1.0, 8.0, 2.0, 9.0, 3.0, 1.0, 7.0, 6.0, 4.0, 5.0, 7.0, 2.0, 9.0, 9.0, 6.0, 1.0, 6.0, 0.0, 0.0, 2.0, 4.0, 8.0, 7.0,
 7.0, 4.0, 3.0, 3.0, 9.0, 0.0, 8.0, 5.0, 4.0, 7.0, 4.0, 8.0, 9.0, 4.0, 2.0, 5.0, 9.0, 2.0, 6.0, 6.0, 7.0, 1.0, 7.0, 9.0, 1.0, 2.0, 9.0, 1.0, 8.0, 4.0, 2.0, 8.0,
 4.0, 5.0, 3.0, 5.0, 1.0, 3.0, 9.0, 2.0, 6.0, 3.0, 7.0, 1.0, 9.0, 4.0, 2.0, 0.0, 1.0, 5.0, 3.0, 8.0, 4.0, 2.0, 6.0, 7.0, 1.0, 1.0, 0.0, 7.0, 6.0, 4.0, 8.0, 8.0,
 5.0, 8.0, 2.0, 1.0, 2.0, 0.0, 5.0, 9.0, 0.0, 1.0, 4.0, 9.0, 3.0, 5.0, 0.0, 1.0, 9.0, 9.0, 0.0, 9.0, 6.0, 8.0, 4.0, 5.0, 4.0, 6.0, 1.0, 0.0, 3.0, 7.0, 2.0, 6.0,
 9.0, 0.0, 6.0, 4.0, 8.0, 1.0, 6.0, 8.0, 9.0, 6.0, 4.0, 6.0, 8.0, 5.0, 0.0, 9.0, 6.0, 6.0, 2.0, 6.0, 3.0, 6.0, 1.0, 6.0, 9.0, 0.0, 9.0, 4.0, 8.0, 7.0, 5.0, 7.0,
 8.0, 4.0, 3.0, 6.0, 8.0, 7.0, 7.0, 4.0, 8.0, 1.0, 5.0, 0.0, 3.0, 3.0, 3.0, 6.0, 3.0, 4.0, 2.0, 3.0, 2.0, 0.0, 6.0, 6.0, 6.0, 4.0, 3.0, 8.0, 5.0, 4.0, 0.0, 3.0,
 3.0, 3.0, 5.0, 5.0, 6.0, 7.0, 8.0, 7.0, 9.0, 0.0, 1.0, 0.0, 6.0, 8.0, 2.0, 9.0, 0.0, 9.0, 3.0, 1.0, 4.0, 2.0, 2.0, 3.0, 8.0, 5.0, 3.0, 6.0, 7.0, 2.0, 4.0, 1.0,
 1.0, 6.0, 1.0, 5.0, 7.0, 1.0, 5.0, 2.0, 9.0, 4.0, 8.0, 5.0, 0.0, 6.0, 9.0, 6.0, 8.0, 8.0, 2.0, 2.0, 6.0, 4.0, 8.0, 9.0, 3.0, 2.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0,
 6.0, 4.0, 5.0, 1.0, 7.0, 8.0, 2.0, 0.0, 0.0, 6.0, 6.0, 5.0, 2.0, 3.0, 5.0, 4.0, 9.0, 1.0, 6.0, 4.0, 4.0, 7.0, 6.0, 9.0, 1.0, 1.0, 7.0, 5.0, 2.0, 0.0, 0.0, 8.0,
 1.0, 3.0, 2.0, 3.0, 0.0, 5.0, 0.0, 8.0, 2.0, 5.0, 8.0, 6.0, 5.0, 3.0, 3.0, 6.0, 9.0, 6.0, 5.0, 7.0, 4.0, 0.0, 5.0, 9.0, 1.0, 6.0, 2.0, 5.0, 0.0, 4.0, 7.0, 3.0,
 6.0, 7.0, 9.0, 2.0, 3.0, 1.0, 9.0, 9.0, 5.0, 8.0, 5.0, 6.0, 0.0, 7.0, 1.0, 8.0, 7.0, 7.0, 0.0, 3.0, 2.0, 3.0, 0.0, 9.0, 5.0, 3.0, 3.0, 4.0, 6.0, 5.0, 9.0, 4.0,
 9.0, 8.0, 2.0, 9.0, 1.0, 8.0, 3.0, 8.0, 8.0, 8.0, 7.0, 3.0, 0.0, 4.0, 1.0, 6.0, 3.0, 9.0, 6.0, 8.0, 1.0, 8.0, 9.0, 4.0, 6.0, 7.0, 1.0, 5.0, 3.0, 1.0, 3.0, 0.0,
 0.0, 1.0, 9.0, 5.0, 9.0, 4.0, 3.0, 5.0, 4.0, 1.0, 6.0, 2.0, 6.0, 6.0, 1.0, 0.0, 7.0, 4.0, 0.0, 9.0, 0.0, 6.0, 9.0, 2.0, 1.0, 1.0, 3.0, 1.0, 6.0, 0.0, 5.0, 9.0,
 8.0, 6.0, 3.0, 6.0, 5.0, 4.0, 1.0, 8.0, 4.0, 1.0, 3.0, 4.0, 8.0, 7.0, 7.0, 0.0, 4.0, 4.0, 0.0, 2.0, 7.0, 1.0, 5.0, 2.0, 0.0, 2.0, 9.0, 8.0, 9.0, 4.0, 1.0, 5.0,
 4.0, 8.0, 0.0, 4.0, 1.0, 3.0, 7.0, 4.0, 3.0, 3.0, 4.0, 7.0, 8.0, 9.0, 7.0, 3.0, 6.0, 4.0, 2.0, 8.0, 0.0, 9.0, 4.0, 6.0, 6.0, 8.0, 6.0, 6.0, 0.0, 5.0, 1.0, 7.0,
 5.0, 6.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 9.0, 7.0, 3.0, 2.0, 3.0, 7.0, 6.0, 1.0, 1.0, 0.0, 6.0, 7.0, 2.0, 0.0, 0.0, 9.0, 2.0, 7.0, 6.0, 3.0, 2.0, 1.0, 6.0, 7.0,
 6.0, 5.0, 0.0, 9.0, 7.0, 2.0, 9.0, 6.0, 5.0, 7.0, 8.0, 6.0, 1.0, 3.0, 9.0, 2.0, 3.0, 4.0, 4.0, 6.0, 9.0, 2.0, 1.0, 1.0, 8.0, 6.0, 2.0, 8.0, 8.0, 8.0, 9.0, 2.0,
 7.0, 4.0, 8.0, 7.0, 7.0, 6.0, 1.0, 5.0, 9.0, 9.0, 0.0, 1.0, 1.0, 7.0, 8.0, 2.0, 5.0, 8.0, 7.0, 5.0, 5.0, 5.0, 2.0, 5.0, 6.0, 8.0, 6.0, 7.0, 1.0, 4.0, 0.0, 2.0,
 7.0, 9.0, 0.0, 4.0, 8.0, 2.0, 5.0, 7.0, 6.0, 1.0, 3.0, 7.0, 5.0, 0.0, 7.0, 0.0, 7.0, 2.0, 9.0, 3.0, 3.0, 1.0, 3.0, 8.0, 9.0, 3.0, 4.0, 7.0, 8.0, 5.0, 3.0, 4.0,
 6.0, 0.0, 6.0, 3.0, 7.0, 0.0, 5.0, 4.0, 6.0, 0.0, 5.0, 5.0, 5.0, 6.0, 6.0, 8.0, 2.0, 8.0, 4.0, 0.0, 0.0, 3.0, 7.0, 7.0, 7.0, 5.0, 4.0, 1.0, 3.0, 4.0, 0.0, 2.0,
 5.0, 7.0, 9.0, 9.0, 6.0, 4.0, 6.0, 7.0, 1.0, 4.0, 8.0, 3.0, 5.0, 5.0, 1.0, 3.0, 3.0, 0.0, 0.0, 8.0, 2.0, 5.0, 2.0, 9.0, 2.0, 4.0, 8.0, 8.0, 1.0, 8.0, 4.0, 4.0,
 1.0, 0.0, 7.0, 4.0, 4.0, 7.0, 7.0, 1.0, 6.0, 1.0, 7.0, 6.0, 9.0, 0.0, 0.0, 2.0, 2.0, 2.0, 9.0, 2.0, 2.0, 7.0, 4.0, 7.0, 0.0, 4.0, 0.0, 0.0, 9.0, 1.0, 5.0, 4.0,
 3.0, 8.0, 0.0, 6.0, 9.0, 5.0, 9.0, 0.0, 4.0, 2.0, 7.0, 9.0, 2.0, 6.0, 1.0, 5.0, 4.0, 9.0, 6.0, 3.0, 1.0, 1.0, 2.0, 2.0, 8.0, 5.0, 5.0, 1.0, 8.0, 7.0, 0.0, 7.0};
     } else {
        matrix = new T[N * N];

        std::mt19937_64 eng(seed);
        std::uniform_real_distribution<T> dist;
        std::generate(matrix, matrix + N * N, std::bind(dist, eng));
    }
}

public:
    long long N, P;
    long long p1, sqrtp1, c;
    long long v, nlayr, Nt, t, tA11, tA10;
    long long seed;
    T* matrix;

    GlobalVars(long long inpN=16, long long inpP=8, long long inpSeed=42) {

        CalculateParameters(inpN, inpP);
        P = sqrtp1 * sqrtp1 * c;
        nlayr = (long long)((v + c-1) / c);
        p1 = sqrtp1 * sqrtp1;

        seed = inpSeed;
        InitMatrix();

        Nt = (long long) (std::ceil((double) N / v));
        t = (long long) (std::ceil((double) Nt / sqrtp1)) + 1ll;
        tA11 = (long long) (std::ceil((double) Nt / sqrtp1));
        tA10 = (long long) (std::ceil((double) Nt / P));
    }

    ~GlobalVars() {
        delete matrix;
    }
};


long long flipbit(long long n, long long k) {
    return n ^ (1ll << k);
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

MPI_Comm create_comm(MPI_Comm& comm, std::vector<int>& ranks) {
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

MPI_Comm pivot_buff_reduction_comm(int p1, int sqrtp1, MPI_Comm comm) {
    // global communicator
    int rank, P;
    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &rank);

    // coordinates of current rank
    long long pi, pj, pk;
    p2X(rank, p1, sqrtp1, pi, pj, pk);

    long long pi_max = sqrtp1;
    long long pj_max = sqrtp1;
    long long pk_max = P / (pi_max * pj_max);

    std::vector<int> ranks;

    for (long long new_pk = 0; new_pk < pk_max; ++new_pk) {
        auto p = X2p(pi, pj, new_pk, p1, sqrtp1);
        ranks.push_back(p);
    }

    return create_comm(comm, ranks);
}

template <typename T> 
void print_matrix(T* pointer, 
                  int row_start, int row_end,
                  int col_start, int col_end,
                  int stride) {
    for (int i = row_start; i < row_end; ++i) {
        for (int j = col_start; j < col_end; ++j) {
            std::cout << pointer[i * stride + j] << ", ";
        }
        std::cout << std::endl;
    }
}

template <typename T> 
void print_matrix_all(T* pointer, 
                  int row_start, int row_end,
                  int col_start, int col_end,
                  int stride,
                  int rank,
                  int P,
                  MPI_Comm comm) {
    for (int r = 0; r < P; ++r) {
        if (r == rank) {
            std::cout << "Rank = " << rank << std::endl;
            for (int i = row_start; i < row_end; ++i) {
                for (int j = col_start; j < col_end; ++j) {
                    std::cout << pointer[i * stride + j] << ", ";
                }
                std::cout << std::endl;
            }
        }
        MPI_Barrier(comm);
    }
}

template <typename T>
void remove_pivotal_rows(std::vector<T>& mat,
                         int n_rows, int n_cols,
                         std::vector<T>& mat_temp,
                         std::vector<int>& pivots) {
    // check which rows should be extracted
    std::vector<int> kept_rows;
    int prev_pivot = -1;
    for (int i = 0; i < pivots[0]; ++i) {
        int pivot = pivots[i + 1];
        for (int j = prev_pivot + 1; j < pivot; ++j) {
            kept_rows.push_back(j);
        }
        prev_pivot = pivot;
    }

    // iterate from the last pivot to the end of the rows
    for (int i = prev_pivot + 1; i < n_rows; ++i) {
        kept_rows.push_back(i);
    }

    // extract kept_rows to temp
#pragma omp parallel for
    for (int i = 0; i < kept_rows.size(); ++i) {
        const auto& row = kept_rows[i];
        std::copy_n(&mat[row * n_cols], n_cols, &mat_temp[i * n_cols]);
    }

    // swap temp with mat
    mat.swap(mat_temp);
}

template <class T>
void LU_rep(T*& A, T*& C, T*& PP, GlobalVars<T>& gv, int rank, int size) {

    long long N, P, p1, sqrtp1, c, v, nlayr, Nt, tA11, tA10;
    N = gv.N;
    P = gv.P;
    p1 = gv.p1;
    sqrtp1 = gv.sqrtp1;
    c = gv.c;
    v = gv.v;
    nlayr = gv.nlayr;
    Nt = gv.Nt;
    tA11 = gv.tA11;
    tA10 = gv.tA10;
    // local n
    auto Nl = tA11 * v;

    // Make new communicator for P ranks
    std::vector<int> participating_ranks(P);
    for (auto i = 0; i < P; ++i) {
        participating_ranks[i] = i;
    }

    MPI_Comm lu_comm = MPI_COMM_WORLD;

    std::vector<T> B(N * N);
    std::copy(A, A + N * N, B.data());

    // Perm = np.eye(N);
    std::vector<int> Perm(N * N);
    for (int i = 0; i < N; ++i) {
        Perm[i * N + i] = 1;
    }

    // Create buffers
    std::vector<T> A00Buff(v * v);
    std::vector<T> A10Buff(Nl * v);
    std::vector<T> A10BuffRcv(Nl * nlayr);
    std::vector<T> A01Buff(v * Nl);
    std::vector<T> A01BuffTemp(v * Nl);
    std::vector<T> A01BuffRcv(nlayr * Nl);
    std::vector<T> A11Buff(Nl * Nl);
    std::vector<T> A10BuffTemp(Nl * v);
    std::vector<T> A11BuffTemp(Nl * Nl);

    int n_local_active_rows = Nl;

    std::vector<int> curPivots(Nl + 1);
    std::vector<int> ipiv(v);
    std::vector<int> curPivOrder(v);
    for (int i = 0; i < v; ++i) {
        curPivOrder[i] = i;
    }

    std::vector<T> pivotBuff(Nl * v);

    // RNG
    std::mt19937_64 eng(gv.seed);
    std::uniform_int_distribution<long long> dist(0, c-1);

    // # ------------------------------------------------------------------- #
    // # ------------------ INITIAL DATA DISTRIBUTION ---------------------- #
    // # ------------------------------------------------------------------- #
    // # get 3d processor decomposition coordinates
    long long pi, pj, pk;
    p2X(rank, p1, sqrtp1, pi, pj, pk);

    // # we distribute only A11, as anything else depends on the first pivots

    // # ----- A11 ------ #
    // # only layer pk == 0 owns initial data
    if (pk == 0) {
        for (auto lti = 0;  lti < tA11; ++lti) {
            auto gti = l2g(pi, lti, sqrtp1);
            for (auto ltj = 0; ltj < tA11; ++ltj) {
                auto gtj = l2g(pj, ltj, sqrtp1);
                mcopy(&B[0], &A11Buff[0],
                      gti * v, (gti + 1) * v, gtj * v, (gtj + 1) * v, N,
                      lti * v, (lti + 1) * v, ltj * v, (ltj + 1) * v, Nl);
            }
        }
    }
    if (rank == 0) {
        print_matrix(A11Buff.data(), 0, Nl, 0, Nl, Nl);
    }
    std::cout << "Allocated." << std::endl;
    MPI_Barrier(lu_comm);


    // Create windows
    MPI_Win A00Win = create_window<T>(lu_comm,
                                   A00Buff.data(),
                                   A00Buff.size(),
                                   true);
    MPI_Win A11Win = create_window(lu_comm,
                                   A11Buff.data(),
                                   A11Buff.size(),
                                   true);
    MPI_Win A10Win = create_window(lu_comm,
                                   A10Buff.data(),
                                   A10Buff.size(),
                                   true);
    MPI_Win A10RcvWin = create_window(lu_comm,
                                      A10BuffRcv.data(),
                                      A10BuffRcv.size(),
                                      true);
    MPI_Win A01Win = create_window(lu_comm,
                                   A01Buff.data(),
                                   A01Buff.size(),
                                   true);
    MPI_Win A01RcvWin = create_window(lu_comm,
                                      A01BuffRcv.data(),
                                      A01BuffRcv.size(),
                                      true);
    MPI_Win curPivotsWin = create_window<int>(lu_comm,
                                     curPivots.data(),
                                     curPivots.size(),
                                     true);

    // Sync all windows
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A00Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A11Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A10Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A10RcvWin);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01RcvWin);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, curPivotsWin);

    long long timers[8] = {0};

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
        if (rank == 0) {
            std::cout << "Iteration = " << k << std::endl;
        }
        // if (k == 1) break;

        auto doprint = (k == 0);
        auto printrank = (rank == 0);

        // global current offset
        auto off = k * v;
        // local current offset
        auto loff = (k / sqrtp1) * v; // sqrtp1 = 2, k = 157

        // # in this step, layrK is the "lucky" one to receive all reduces
        auto layrK = 0; // dist(eng);

        // layrK = 0;
        // if (k == 0) layrK = 0;
        // if (k == 1) layrK = 1;
        // if (doprint && printrank) std::cout << "layrK: " << layrK << std::endl << std::flush;
        // # ----------------------------------------------------------------- #
        // # 0. reduce first tile column from A11buff to PivotA11ReductionBuff #
        // # ----------------------------------------------------------------- #
        MPI_Barrier(lu_comm);
        auto ts = std::chrono::high_resolution_clock::now();
        // # Currently, we dump everything to processors in layer pk == 0, and only this layer choose pivots
        // # that is, each processor [pi, pj, pk] sends to [pi, pj, layK]
        // # note that processors in layer pk == 0 locally copy their data from A11buff to PivotA11ReductionBuff
        p2X(rank, p1, sqrtp1, pi, pj, pk);

        // flush the buffer
        curPivots[0] = 0;

        // # reduce first tile column. In this part, only pj == k % sqrtp1 participate:
        if (pj == k % sqrtp1) {
            auto p_rcv = X2p(pi, pj, layrK, p1, sqrtp1);
            // here it's guaranteed that rank != p_rcv because pk != layrK
            // transpose matrix A11Buff for easier sending
            std::cout << "Step 0, before." << std::endl;
            if (pk == layrK) {
                mkl_domatcopy('R', 'N',
                               n_local_active_rows, v,
                               1.0,
                               &A11Buff[loff], Nl,
                               &A10Buff[0], v); 
            } else {
                MPI_Accumulate(&A10Buff[0], n_local_active_rows * v, 
                               MPI_DOUBLE,
                               p_rcv, 0, n_local_active_rows * v, 
                               MPI_DOUBLE,
                               MPI_SUM, A10Win);
            }
            if (rank == 0) {
                print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
            }

            std::cout << "Step 0, after." << std::endl;
        }

        MPI_Win_fence(0, A10Win);

        MPI_Barrier(lu_comm);

        std::cout << "Step 0 finished." << std::endl;
        MPI_Barrier(lu_comm);
        auto te = std::chrono::high_resolution_clock::now();
        timers[0] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        // # --------------------------------------------------------------------- #
        // # 1. coalesce PivotA11ReductionBuff to PivotBuff and scatter to A10buff #
        // # --------------------------------------------------------------------- #
        MPI_Barrier(lu_comm);
        ts = std::chrono::high_resolution_clock::now();

        int zero = 0;
        if (k % sqrtp1 == pi && k % sqrtp1 == pj && pk == 0) {
            // # filter A11buff by masked pivot rows (local) and densify it
            mkl_domatcopy('R', 'N',
                           n_local_active_rows, v,
                           1.0,
                           &A10Buff[0], v,
                           &pivotBuff[0], v);
            std::cout << "pivotBuff after copy" << std::endl;
            print_matrix(pivotBuff.data(), 0, n_local_active_rows, 0, v, v);

            int info = 0;
            int v_int = (int) v;
            std::cout << "before LU" << std::endl;
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n_local_active_rows, v, pivotBuff.data(), v, ipiv.data());
            std::cout << "After LU decomposition, the result = " << std::endl;
            print_matrix(pivotBuff.data(), 0, n_local_active_rows, 0, v, v);
            std::cout << "ipiv = " << std::endl;
            std::cout << ipiv[0] << ", " << ipiv[1] << std::endl;

            for (int i = 0; i < Nl; ++i) {
                curPivots[i+1] = i;
            }
            curPivots[0] = v;

            for (int i = 0; i < ipiv.size(); ++i) {
                auto& a = curPivots[i+1];
                auto& b = curPivots[ipiv[i]];
                std::swap(a, b);
            }


            std::sort(&curPivots[1], &curPivots[v + 1]);
            print_matrix(curPivots.data(), 0, 1,  0, v+1, v+1);

            mkl_domatcopy('R', 'N',
                           v, v,
                           1.0,
                           &pivotBuff[0], v,
                           &A00Buff[0], v);

            std::cout << "A00Buff = " << std::endl;
            print_matrix(A00Buff.data(), 0, v, 0, v, v);


            // # In round robin fashion, only one rank chooses pivots among its local rows
            // # -------------------------- #
            // # !!!!! COMMUNICATION !!!!!! #
            std::cout << "rank = " << rank << std::endl;
            std::cout << "before sending pivots" << std::endl;
            // # Sending pivots:
            for (int pj_rcv = 0; pj_rcv < sqrtp1; ++pj_rcv) {
                for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                    auto p_rcv = X2p(pi, pj_rcv, pk_rcv, p1, sqrtp1);
                    if (rank != p_rcv) 
                        MPI_Put(&curPivots[0], v+1, MPI_INT, p_rcv, 0, v+1, MPI_INT, curPivotsWin);
                }
            }
            std::cout << "after sending pivots" << std::endl;

            std::cout << "before sending zero" << std::endl;
            for (int pi_rcv = 0; pi_rcv < sqrtp1; ++pi_rcv) {
                if (pi != pi_rcv) {
                    for (int pj_rcv = 0; pj_rcv < sqrtp1; ++pj_rcv) {
                        for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                            auto p_rcv = X2p(pi_rcv, pj_rcv, pk_rcv, p1, sqrtp1);
                            std::cout << "rank = " << p_rcv << std::endl;
                            MPI_Put(&zero, 1, MPI_INT, p_rcv, 0, 1, MPI_INT, curPivotsWin);
                        }
                    }
                }
            }
            std::cout << "after sending zero" << std::endl;

            // # Sending A00Buff:
            for (int pi_rcv = 0; pi_rcv < sqrtp1; ++pi_rcv) {
                for (int pj_rcv = 0; pj_rcv < sqrtp1; ++pj_rcv) {
                    for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                        auto p_rcv = X2p(pi_rcv, pj_rcv, pk_rcv, p1, sqrtp1);
                        if (rank != p_rcv) 
                            MPI_Put(&A00Buff[0], v * v, MPI_DOUBLE, p_rcv, 0, v * v, MPI_DOUBLE, A00Win);
                    }
                }
            }

        }

        MPI_Win_fence(0, curPivotsWin);
        MPI_Win_fence(0, A00Win);

        // std::sort(curPivots.begin() + 1, curPivots.end());

        if (rank == 1) {
            std::cout << "After sending A00Buff, rank 1 has:" << std::endl;;
            print_matrix(A00Buff.data(), 0, v, 0, v, v);
        }

        MPI_Barrier(lu_comm);
        std::cout << "Step 1 finished." << std::endl;
        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[2] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        // # ---------------------------------------------- #
        // # 2. reduce pivot rows from A11buff to PivotA01ReductionBuff #
        // # ---------------------------------------------- #
        MPI_Barrier(lu_comm);
        ts = std::chrono::high_resolution_clock::now();
        p2X(rank, p1, sqrtp1, pi, pj, pk);
        // curPivots = pivotIndsBuff[k * v: (k + 1) * v]
        // # Currently, we dump everything to processors in layer pk == 0, pi == k % sqrtp1
        // # and only this strip distributes reduced pivot rows
        // # so layer pk == 0 do a LOCAL copy from A11Buff to PivotBuff, other layers do the communication
        // # that is, each processor [pi, pj, pk] sends to [pi, pj, 0]
        // update the row mask
        if (pk != layrK) {
            auto dspls = 0;
            auto origin_ptr = &A11Buff[dspls];
            auto p_rcv = X2p(pi, pj, layrK, p1, sqrtp1);
            MPI_Accumulate(origin_ptr, A11Buff.size(), MPI_DOUBLE,
                           p_rcv, dspls, A11Buff.size(), MPI_DOUBLE,
                           MPI_SUM, A11Win);
        }
        MPI_Win_fence(0, A11Win);
        MPI_Barrier(lu_comm);
        if (rank == 0) {
            std::cout << "Step 2 finished." << std::endl;
            print_matrix(A11Buff.data(), 0, Nl, 0, Nl, Nl);
        }
        MPI_Barrier(lu_comm);


        // # -------------------------------------------------- #
        // # 3. distribute v pivot rows from A11buff to A01Buff #
        // # here, only processors pk == layrK participate      #
        // # -------------------------------------------------- #
        if (pk == layrK) {
            auto p_rcv = X2p(k % sqrtp1, pj, layrK, p1, sqrtp1);
            for (int i = 0; i < curPivots[0]; ++i) {
                auto pivot = curPivots[i+1];
                auto offset = curPivOrder[i];
                auto origin_dspls = pivot * Nl + loff;
                auto origin_ptr = &A11Buff[origin_dspls];
                auto size = Nl - loff;
                auto dest_dspls = offset * Nl + loff;
                MPI_Put(origin_ptr, size, MPI_DOUBLE,
                        p_rcv, dest_dspls, size, MPI_DOUBLE,
                        A01Win);
            }
        }

        MPI_Win_fence(0, A01Win);
        MPI_Barrier(lu_comm);

        if (rank == 0) {
            std::cout << "Step 3 finished." << std::endl;
            print_matrix(A01Buff.data(), 0, v, 0, Nl, Nl);
        }
        MPI_Barrier(lu_comm);

        te = std::chrono::high_resolution_clock::now();
        timers[3] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        // remove pivotal rows from matrix A10Buff and A11Buff
        // A10Buff
        if (rank == 0) {
            std::cout << "A10Buff before row swapping" << std::endl;
            print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
        }

        // we want to push at the end the following rows:
        // ipiv -> series of row swaps 5, 5
        // curPivots -> 4, 0 (4th row with second-last and 0th row with the one)
        // last = n_local_active_rows
        // last(A11) = n_local_active_rows
        remove_pivotal_rows(A10Buff, n_local_active_rows, v, A10BuffTemp, curPivots);
        remove_pivotal_rows(A11Buff, n_local_active_rows, Nl, A11BuffTemp, curPivots);
        n_local_active_rows -= curPivots[0];

        if (rank == 0) {
            std::cout << "A10Buff after row swapping" << std::endl;
            print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
        }

        // # ---------------------------------------------- #
        // # 4. compute A10 and broadcast it to A10BuffRecv #
        // # ---------------------------------------------- #
        if (pk == layrK && pj == k % sqrtp1) {
            // # this could basically be a sparse-dense A10 = A10 * U^(-1)   (BLAS tiangular solve) with A10 sparse and U dense
            // however, since we are ignoring the mask, it's dense, potentially with more computation than necessary.
            std::cout << "before trsm." << std::endl;
            cblas_dtrsm(CblasRowMajor, // side
                   CblasRight, // uplo
                   CblasUpper,
                   CblasNoTrans,
                   CblasNonUnit,
                   n_local_active_rows, //  M
                   v,  // N
                   1.0, // alpha
                   &A00Buff[0], // triangular A
                   v, // leading dim triangular
                   &A10Buff[0], // A11
                   v);
            std::cout << "after trsm." << std::endl;

            if (rank == 0) {
                std::cout << "A10Buff after trsm" << std::endl;
                print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
            }

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv*nlayr;
                auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                int size = nlayr * n_local_active_rows; // nlayr = v / c

                // copy [colStart, colEnd) columns of A10Buff -> A10BuffTemp densely
                mcopy(A10Buff.data(), A10BuffTemp.data(), 
                      0, n_local_active_rows, colStart, colEnd, v,
                      0, n_local_active_rows, 0, nlayr, nlayr);

                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int pj_rcv = 0; pj_rcv <  sqrtp1; ++pj_rcv) {
                    auto p_rcv = X2p(pi, pj_rcv, pk_rcv, p1, sqrtp1);
                    // MPI_Put(&A11Buff[offset]), size, MPI_DOUBLE,
                    //     p_rcv, 0, size, MPI_DOUBLE, A11Win);
                    std::cout << "before put in ." << p_rcv << std::endl;
                    MPI_Put(&A10BuffTemp[0], size, MPI_DOUBLE,
                        p_rcv, 0, size, MPI_DOUBLE, A10RcvWin);
                    std::cout << "after put in ." << p_rcv << std::endl;
                }
            }
        }

        MPI_Win_fence(0, A10RcvWin);
        MPI_Barrier(lu_comm);
        std::cout << "Step 4 finished." << std::endl;
        MPI_Barrier(lu_comm);

        auto lld_A01 = Nl;
        // # ---------------------------------------------- #
        // # 5. compute A01 and broadcast it to A01BuffRecv #
        // # ---------------------------------------------- #
        // # here, only ranks which own data in A01Buff (step 3) participate
        if (pk == layrK && pi == k % sqrtp1) {
            // # this is a dense-dense A01 =  L^(-1) * A01
            cblas_dtrsm(CblasRowMajor, // side
                   CblasLeft,
                   CblasLower,
                   CblasNoTrans,
                   CblasUnit,
                   v, //  M
                   Nl - loff,  // N
                   1.0, // alpha
                   &A00Buff[0], // triangular A
                   v, // leading dim triangular
                   &A01Buff[loff], // A01
                   lld_A01); // leading dim of A01

            // # local reshuffle before broadcast
            // pack all the data for each rank
            // extract rows [rowStart, rowEnd) and cols [loff, Nl)
#pragma omp parallel for
            for(int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                auto rowStart = pk_rcv * nlayr;
                auto rowEnd = (pk_rcv + 1) * nlayr;
                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int row = rowStart; row < rowEnd; ++row) {
                    const int n_cols = Nl - loff;
                    std::copy_n(&A01Buff[row * lld_A01 + loff], // A01Buff[row, loff]
                                n_cols,
                                &A01BuffTemp[row * n_cols]);
                    // A01BuffTemp has received leading dimension
                }
            }

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for(int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                auto rowStart = pk_rcv * nlayr;
                auto rowEnd = (pk_rcv + 1) * nlayr;
                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for(int pi_rcv = 0; pi_rcv < sqrtp1; ++pi_rcv) {
                    const int n_cols = Nl - loff;
                    auto p_rcv = X2p(pi_rcv, pj, pk_rcv, p1, sqrtp1);
                    MPI_Put(&A01BuffTemp[rowStart * n_cols],
                            nlayr * n_cols, MPI_DOUBLE,
                            p_rcv, 0, nlayr * n_cols, 
                            MPI_DOUBLE, A01RcvWin);
                }
            }
        }

        MPI_Win_fence(0, A01RcvWin);
        MPI_Barrier(lu_comm);
        std::cout << "Step 5 finished." << std::endl;
        MPI_Barrier(lu_comm);

        // # ---------------------------------------------- #
        // # 6. compute A11  ------------------------------ #
        // # ---------------------------------------------- #
        // # filter which rows of this tile should be processed:
        // rows = A11MaskBuff[p]
        // assumptions:
        // 1. we don't do the filtering
        // 2. A10BuffRcv is column-major
        // 3. A01BuffTemp is densified and leading dimensions = Nl-loff, row-major
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_local_active_rows, Nl - loff, nlayr,
                    -1.0, &A10BuffRcv[0], nlayr,
                    &A01BuffRcv[0], Nl-loff,
                    1.0, &A11Buff[loff], Nl);
        MPI_Barrier(lu_comm);
        for (int i = 0; i < P; ++i) {
            if (rank == i) {
                std::cout << "rank " << rank << ", A11Buff after computeA11:" << std::endl;
                print_matrix(A11Buff.data(), 0, n_local_active_rows,
                                             0, Nl,
                                             Nl);
            }
            MPI_Barrier(lu_comm);
        }
        std::exit(0);
    }

    // # recreate the permutation matrix
    std::vector<T> Permutation(N * N);
    for (int i = 0; i < N; ++i) {
        auto row = ipiv[i];
        std::copy_n(&B[row * N], N, &C[i * N]);
        std::copy_n(&Perm[row * N], N, &Permutation[i * N]);
    }

    if (rank == 0) {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        std::cout << "Runtime: " << double(duration) / 1000000 << " seconds" << std::endl;

        for (auto i = 0; i < 7; ++i) {
            std::cout << "Runtime " << i + 1 << ": " << double(timers[i]) / 1000000 << " seconds" << std::endl;
        }
    }

    // Delete all windows
    MPI_Win_free(&A00Win);
    MPI_Win_free(&A11Win);
    MPI_Win_free(&A10Win);
    MPI_Win_free(&A10RcvWin);
    MPI_Win_free(&A01Win);
    MPI_Win_free(&A01RcvWin);
    MPI_Win_free(&curPivotsWin);
}

int main(int argc, char **argv) {

    // GlobalVars<dtype> gb = GlobalVars<dtype>(4096, 32);
    // return 0;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (!rank && argc != 2) {
      std::cout << "USAGE: conflux <N>" << std::endl;
      return 1;
    }

    GlobalVars<dtype> gv = GlobalVars<dtype>(atoi(argv[1]), size);
    std::cout << "Rank: " << rank << " N: " << gv.N << ", P:" << gv.P
              << ", v:" << gv.v << ", c:" << gv.c
              << ", sqrtp1: " << gv.sqrtp1 <<  ", Nt: " << gv.Nt
              << ", tA10 " << gv.tA10 << ", tA11: " << gv.tA11 << std::endl;

    long long p, pi, pj, pk;
    p2X(rank, gv.p1, gv.sqrtp1, pi, pj, pk);
    p = X2p(pi, pj, pk, gv.p1, gv.sqrtp1);

    // std::cout << "Rank: " << rank << ", coords: (" << pi << ", " << pj
    //           << ", " << pk << "), p: " << p << std::endl;

    // if (rank == 0) {
    //     for (auto i = 0; i < gv.N; ++i) {
    //         for (auto j = 0; j < gv.N; ++j) std::cout << gv.matrix[i*gv.N+j] << " ";
    //         std::cout << std::endl << std::flush;
    //     }
    // }

    dtype* C = new dtype[gv.N * gv.N]{0};
    dtype* Perm = new dtype[gv.N * gv.N]{0};

    for (int i = 0; i < 1; ++i) {
        PC();
        LU_rep<dtype>(gv.matrix, C, Perm, gv, rank, size);
    }

    if (rank == 0) {
        PP();
    }

    MPI_Finalize();

    delete C;
    delete Perm;

    return 0;
}
