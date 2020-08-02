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
        v = std::lcm(sqrtp1, c);
        // v = 64;
        long long nLocalTiles = (long long) (std::ceil((double) inpN / (v * sqrtp1)));
        N = v * sqrtp1 * nLocalTiles;
        // std::cout << sqrtp1 << " " << c << std::endl << std::flush;
        // std::cout << v << " " << nLocalTiles << std::endl << std::flush;
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
        nlayr = (long long)(v / c);
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

template <class T>
void LUP(std::vector<T>& PivotBuff, T* L, T* U, T* Perm, int v) {
    // global global_vars
    // v = global_vars['v']
    int n = v;
    int m = PivotBuff.size() / n;

    [Perm, L, U] = la.lu(inpA)
    origA = Perm.T @ inpA
    [m, n] = inpA.shape
    res = (L +
           np.concatenate((U, np.zeros([m-n,n])), axis = 0) -
           np.eye(m, n))[:v,:]
    return [origA[:v,], res[:v,], Perm]
}

template <class T>
void EmptyPivotMPI(int rank, int k, T* PivotBuff, MPI_Win& PivotWin,
                   T* A00Buff, MPI_Win& A00Win, int* pivots, MPI_Win& pivotsWin,
                   bool* A10Mask, bool* A11Mask, int layrK,
                   GlobalVars<T>& gv) {
    PE(tpivoting_other);
    auto doprint = (k == -1);
    auto printrank = (rank == -1);

    long long N, P, v, p1, sqrtp1, c, tA10, tA11;
    N = gv.N;
    P = gv.P;
    v = gv.v;
    p1 = gv.p1;
    sqrtp1 = gv.sqrtp1;
    c = gv.c;
    tA10 = gv.tA10;
    tA11 = gv.tA11;
    auto Nl = tA11 * v;

    // # ---------------- FIRST STEP ----------------- #
    // # in first step, we do pivot on the whole PivotBuff array (may be larger than [2v, v]
    // # local computation step
    long long pi, pj, pk;
    p2X(rank, p1, sqrtp1, pi, pj, pk);
    if (k % sqrtp1 != pi || k % sqrtp1 != pj || pk != 0)
        return;

    // data = densified A11Buff with respect to A11MaskBuff
    auto& rows = A11MaskBuff;
    std::vector<T> data;
    // filter A11buff by masked pivot rows (local) and densify it
    for (const auto& row : rows) {
        auto col_start = (k / sqrtp1) * v;
        auto col_end = (k / sqrtp1 + 1) * v;
        data.insert(dense_A11Buff.end(),
                             &A11Buff[row * Nl + col_start],
                             &A11Buff[row * Nl + col_end]);
    }

    // flush the buffer
    std::fill_n(PivotBuff.data(), PivotBuff.size(), 0);
    std::copy_n(data.data(), data.size(), PivotBuff.data());

    // In round robin fashion, only one rank chooses pivots among its local rows
    if (k % sqrtp1 == pi) {
        [PivotBuff[p, :v, :], A00Buff[p], Perm] = LUP(PivotBuff[p, :len(data)])

        grows = l2gnoTile(np.nonzero(rows)[0], pi)
        gpivots = (grows @ Perm)[:v]
        [lpivots, loffsets] = g2lnoTile(gpivots)

        // # locally set curPivots
        curPivots[p, 0] = len(lpivots[pi])
        curPivots[p, 1 : 1 + len(lpivots[pi])] = lpivots[pi]
        curPivOrder[p, :len(loffsets[pi])] = loffsets[pi]
        pivotIndsBuff[p, k*v:(k+1)*v] = gpivots

        auto src_pi = std::min(flipbit(pi, 0ll), sqrtp1 - 1ll);
        auto src_p = X2p(src_pi, pj, pk, p1, sqrtp1);
        PL();
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
    int* participating_ranks = new int[P];
    for (auto i = 0; i < P; ++i) {
        participating_ranks[i] = i;
    }

    MPI_Group world_group, lu_group;
    MPI_Comm lu_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_incl(world_group, P, participating_ranks, &lu_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, lu_group, 0, &lu_comm);

    delete participating_ranks;
    if (rank >= P) {
        return;
    }

    std::vector<T> B(N * N);
    std::copy(A, A + N * N, B.data());

    // Perm = np.eye(N);
    std::vector<int> Perm(N * N);
    for (int i = 0; i < N; ++i) {
        Perm[i * N + i] = 1;
    }


    // Create buffers
    std::vector<T> A00Buff(v * v);
    std::vector<T> A10Buff(tA10 * v * v);
    std::vector<T> A10BuffRcv(Nl * nlayr);
    std::vector<T> A01Buff(v * Nl);
    std::vector<T> A01BuffTemp(v * Nl);
    std::vector<T> A01BuffRcv(nlayr * Nl);
    std::vector<T> A11Buff(Nl * Nl);
    std::vector<T> A11BuffTransposed(Nl * Nl);

    std::vector<bool> A11MaskBuff(Nl, true);

    std::vector<T> PivotBuff(v * std::max(2ll, tA11) * v);

    std::vector<T> PivotA11ReductionBuff(tA11 * v * v);
    std::vector<int> pivotIndsBuff(N, -1);

    std::vector<int> curPivots(v + 1);
    std::vector<int> curPivOrder(v);

    // Create windows
    MPI_Win A00Win = create_window(lu_comm,
                                   A00Buff.data(),
                                   A00Buff.size(),
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
    MPI_Win A01TempWin = create_window(lu_comm,
                                   A01BuffTemp.data(),
                                   A01BuffTemp.size(),
                                   true);
    MPI_Win A01RcvWin = create_window(lu_comm,
                                      A01BuffRcv.data(),
                                      A01BuffRcv.size(),
                                      true);
    MPI_Win A11Win = create_window(lu_comm,
                                   A11Buff.data(),
                                   A11Buff.size(),
                                   true);
    MPI_Win A11TransposedWin = create_window(lu_comm,
                                   A11BuffTransposed.data(),
                                   A11BuffTransposed.size(),
                                   true);
    MPI_Win PivotWin = create_window(lu_comm,
                                     PivotBuff.data(),
                                     PivotBuff.size(),
                                     true);
    MPI_Win pivotsWin = create_window(lu_comm,
                                      pivotIndsBuff.data(),
                                      pivotIndsBuff.size(),
                                      true);
    MPI_Win PivotA11Win = create_window(lu_comm,
                                        PivotA11ReductionBuff.data(),
                                        PivotA11ReductionBuff.size(),
                                        true);

    // Sync all windows
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A00Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A10Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A10RcvWin);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01RcvWin);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A11Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A11TransposedWin);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, PivotWin);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, PivotA11Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, pivotsWin);

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
                // A11Buff[lti, ltj] = B[gti * v: (gti + 1) * v, gtj * v: (gtj + 1) * v]
                mcopy(B, A11Buff + (lti * tA11 + ltj) * v * v,
                      gti * v, (gti + 1) * v, gtj * v, (gtj + 1) * v, N,
                      0, v, 0, v, v);
                assert(B[gti * v * N + gtj * v] == A11Buff[(lti * tA11 + ltj) * v * v]);
            }
        }
    }

    long long timers[8] = {0};

    // # ---------------------------------------------- #
    // # ----------------- MAIN LOOP ------------------ #
    // # 1. reduce first tile column from A11buff to PivotA11ReductionBuff
    // # 2. coalesce PivotA11ReductionBuff to PivotBuff and scatter to A10buff
    // # 3. find v pivots and compute A00
    // # 4. reduce pivot rows from A11buff to PivotA11ReductionBuff
    // # 5. scatter PivotA01ReductionBuff to A01Buff
    // # 6. compute A10 and broadcast it to A10BuffRecv
    // # 7. compute A01 and broadcast it to A01BuffRecv
    // # 8. compute A11
    // # ---------------------------------------------- #

    MPI_Barrier(lu_comm);
    auto t1 = std::chrono::high_resolution_clock::now();

    // # now k is a step number
    for (auto k = 0; k < Nt; ++k) {
        // if (k == 1) break;

        auto doprint = (k == 0);
        auto printrank = (rank == 0);

        // global current offset
        auto off = k * v;
        // local current offset
        auto loff = (k / sqrtp1) * v;

        // # in this step, layrK is the "lucky" one to receive all reduces
        auto layrK = dist(eng);

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
        auto p_rcv = X2p(pi, pj, layrK, p1, sqrtp1);

        // flush the buffer
        curPivots[0] = 0;

        // # reduce first tile column. In this part, only pj == k % sqrtp1 participate:
        if (pk != layrK && pj == k % sqrtp1) {
            // rows = A11MaskBuff
            // here it's guaranteed that rank != p_rcv because pk != layrK
            /*
             * if we want to take the mask into account:
            for (const auto& rows : A11MaskBuff) {
                auto dspls = rows * Nl + loff;
                auto origin_ptr = &A11Buff[dspls];
                MPI_Accumulate(origin_ptr, v, MPI_DOUBLE,
                               p_rcv, dspls, v, MPI_DOUBLE,
                               MPI_SUM, A11Win);
            }
            */
            auto dspls = 0;
            auto origin_ptr = &A11Buff[dspls];
            MPI_Accumulate(origin_ptr, A11Buff.size(), MPI_DOUBLE,
                           p_rcv, dspls, A11Buff.size(), MPI_DOUBLE,
                           MPI_SUM, A11Win);
        }

        MPI_Win_fence(0, A11Win);

        MPI_Barrier(lu_comm);
        auto te = std::chrono::high_resolution_clock::now();
        timers[0] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();


        // # --------------------------------------------------------------------- #
        // # 1. coalesce PivotA11ReductionBuff to PivotBuff and scatter to A10buff #
        // # --------------------------------------------------------------------- #
        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[1] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        //# flush the buffer
        if (pj == k % sqrtp1 && pk == layrK) {
            std::fill(PivotA11ReductionBuff,
                      PivotA11ReductionBuff + tA11 * v * v, 0);
        }

        MPI_Barrier(lu_comm);
        ts = std::chrono::high_resolution_clock::now();
        EmptyPivotMPI(rank, k, PivotBuff, PivotWin, A00Buff, A00Win,
                      pivotIndsBuff, pivotsWin, A10MaskBuff, A11MaskBuff,
                      layrK, gv, luA, origA, Perm, ipiv, p);
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
        auto p_rcv = X2p(pi, pj, layrK, p1, sqrtp1);
        // update the row mask
        if (pk != layrK) {
            auto dspls = 0;
            auto origin_ptr = &A11Buff[dspls];
            MPI_Accumulate(origin_ptr, A11Buff.size(), MPI_DOUBLE,
                           p_rcv, dspls, A11Buff.size(), MPI_DOUBLE,
                           MPI_SUM, A11Win);
        }

        // # -------------------------------------------------- #
        // # 3. distribute v pivot rows from A11buff to A01Buff #
        // # here, only processors pk == layrK participate      #
        // # -------------------------------------------------- #
        if (pk == layrK) {
            auto p_rcv = X2p(k % sqrtp1, pj, layrK, p1, sqrtp1);
            for (int i = 0; i < curPivots[0]; ++i) {
                pivot = curPivots[i+1]
                offset = curPivOrder[i]
                auto origin_dspls = pivot * Nl + loff;
                auto origin_ptr = &A11Buff[origin_dspls];
                auto size = Nl - loff;
                auto dest_dspls = offset * Nl + loff;
                MPI_Put(origin_ptr, size, MPI_DOUBLE,
                        p_rcv, dest_dspls, MPI_DOUBLE,
                        A01Win);
            }
        }

        MPI_Win_fence(0, A01Win);

        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[3] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        // # ---------------------------------------------- #
        // # 4. compute A10 and broadcast it to A10BuffRecv #
        // # ---------------------------------------------- #
        if (pk == layrK && pj == k % sqrtp1) {
            // # this could basically be a sparse-dense A10 = A10 * U^(-1)   (BLAS tiangular solve) with A10 sparse and U dense
            // however, since we are ignoring the mask, it's dense, potentially with more computation than necessary.
            cblas_dtrsm(CblasRowMajor, // side
                   CblasRight, // uplo
                   CblasUpper,
                   CblasNoTrans,
                   CblasNonUnit,
                   Nl, //  M
                   v,  // N
                   1.0, // alpha
                   A00Buff.data(), // triangular A
                   v, // leading dim triangular
                   &A11Buff[loff], // A11
                   v);

            // transpose matrix A11Buff for easier sending
            mkl_domatcopy('R', 'T', Nl, Nl, 1.0, A11Buff.data(), Nl, A11BuffTransposed.data(), Nl);

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                colStart = loff + pk_rcv*nlayr;
                colEnd   = loff + (pk_rcv+1)*nlayr;

                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int pj_rcv = 0; pj_rcv <  sqrtp1; ++pj_rcv) {
                    p_rcv = X2p(pi, pj_rcv, pk_rcv, p1, sqrtp1)
                    int offset = colStart * Nl;
                    int size = nlayr;
                    // ASSUMES: receiving ranks only receive from a single rank
                    MPI_Put(&A11BuffTransposed[offset]), size, MPI_DOUBLE,
                        p_rcv, 0, size, MPI_DOUBLE, A11TransposedWin);
                }
            }
        }

        MPI_Win_fence(0, A11TransposedWin);

        // ranks which are on the receiving side of step 4
        if (pk != layrK && pj != k % sqrtp1) {
            // transpose back what is received
            mkl_domatcopy('C', 'T', Nl, Nl, 1.0,
                    A11BuffTransposed.data(), Nl, A10BuffRcv.data(), Nl);
        }

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
                   Nl,  // N
                   1.0, // alpha
                   A00Buff.data(), // triangular A
                   v, // leading dim triangular
                   &A01Buff[loff], // A01
                   Nl); // leading dim of A01

            // # local reshuffle before broadcast
            // pack all the data for each rank
            // extract rows [rowStart, rowEnd) and cols [loff, Nl)
#pragma omp parallel for
            for(int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                rowStart = pk_rcv * nlayr
                rowEnd = (pk_rcv + 1) * nlayr
                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int row = rowStart; row < rowEnd; ++row) {
                    const int n_cols = Nl - loff;
                    std::copy_n(&A01Buff[row * Nl + loff],
                                n_cols,
                                &A01BuffTemp[row * n_cols]);
                }
            }

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for(int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                rowStart = pk_rcv * nlayr
                rowEnd = (pk_rcv + 1) * nlayr
                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for(int pi_rcv = 0; pi_rcv < sqrtp1; ++pi_rcv) {
                    const int n_cols = Nl - loff;
                    p_rcv = X2p(pi_rcv, pj, pk_rcv, p1, sqrtp1);
                    MPI_Put(A01BuffTemp.data(), nlayr * n_cols, MPI_DOUBLE,
                            p_rcv, 0, nlayr * n_cols, MPI_DOUBLE, A01TempWin);
                }
            }
        }

        MPI_Win_fence(0, A01TempWin);

        // ranks which are on the receiving side of step 5
        if (pk != layrK && pi != k % sqrtp1) {
            // unpack the received data
            // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
            // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
#pragma omp parallel for
            for (int row = 0; row < nlayr; ++row) {
                auto rowStart = pk * nlayr;
                const int n_cols = Nl - loff;
                std::copy_n(&A01BuffTemp[row * n_cols],
                            n_cols,
                            &A01BuffRcv[row * Nl + loff]);
            }
        }

        // # ---------------------------------------------- #
        // # 6. compute A11  ------------------------------ #
        // # ---------------------------------------------- #
        // # filter which rows of this tile should be processed:
        // rows = A11MaskBuff[p]
        A11Buff[p, rows,  loff:] -= A10BuffRcv[p, rows] @ A01BuffRcv[p, :, loff:]
    }

    // # recreate the permutation matrix
    PP = np.zeros([N,N])
    C = np.zeros([N,N])
    for i in range(N):
        C[i, :] = B[pivotIndsBuff[0,i], :]
        PP[i, :] = Perm[pivotIndsBuff[0, i], :]

    if (rank == 0) {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        std::cout << "Runtime: " << double(duration) / 1000000 << " seconds" << std::endl;

        for (auto i = 0; i < 7; ++i) {
            std::cout << "Runtime " << i + 1 << ": " << double(timers[i]) / 1000000 << " seconds" << std::endl;
        }

        MPI_Reduce(MPI_IN_PLACE, comm_count, 7, MPI_LONG_LONG, MPI_SUM, 0, lu_comm);
        MPI_Reduce(MPI_IN_PLACE, local_comm, 7, MPI_LONG_LONG, MPI_SUM, 0, lu_comm);
        long long total_count = 0;
        long long total_local = 0;
        for (auto i = 0; i < 7; ++i) {
            std::cout << "Step " << i << ": " << comm_count[i]
                      << " (" << comm_count[i] + local_comm[i]
                      << " with local comm) elements" << std::endl << std::flush;
            total_count += comm_count[i];
            total_local += local_comm[i];
        }
        std::cout << "Total:  " << total_count << " (" << total_count + total_local
                  << " with local comm) elements" << std::endl << std::flush;

        for (auto i = 0; i < N; ++i) {
            auto idx = pivotIndsBuff[i];
            std::copy(B + idx * N, B + (idx + 1) * N, C + i * N);
            PP[i * N + idx] = 1;
        }

    } else {
        MPI_Reduce(comm_count, NULL, 7, MPI_LONG_LONG, MPI_SUM, 0, lu_comm);
        MPI_Reduce(local_comm, NULL, 7, MPI_LONG_LONG, MPI_SUM, 0, lu_comm);
    }

    // Delete all windows
    MPI_Win_free(&A00Win);
    MPI_Win_free(&A10Win);
    MPI_Win_free(&A10RcvWin);
    MPI_Win_free(&A01Win);
    MPI_Win_free(&A01TempWin);
    MPI_Win_free(&A01RcvWin);
    MPI_Win_free(&A11Win);
    MPI_Win_free(&A11TransposedWin);
    MPI_Win_free(&PivotWin);
    MPI_Win_free(&PivotA11Win);
    MPI_Win_free(&pivotsWin);

    // Delete communicator
    MPI_Group_free(&world_group);
    MPI_Group_free(&lu_group);
    MPI_Comm_free(&lu_comm);
}

int main(int argc, char **argv) {

    // GlobalVars<dtype> gb = GlobalVars<dtype>(4096, 32);
    // return 0;

    MPI_Init(NULL, NULL);
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
    dtype* PP = new dtype[gv.N * gv.N]{0};

    for (int i = 0; i < 1; ++i) {
        PC();
        LU_rep<dtype>(gv.matrix, C, PP, gv, rank, size);
    }

    if (rank == 0) {
        PP();
    }

    #ifdef VALIDATE

    if (rank == 0) {
        auto N = gv.N;
        dtype* U = new dtype[N * N]{0};
        dtype* L = new dtype[N * N] {0};
        for (auto i = 0; i < N; ++i) {
            for (auto j = 0; j < i; ++j) {
                L[i * N + j] = C[i * N + j];
            }
            L[i * N + i] = 1;
            for (auto j = i; j < N; ++j) {
                U[i * N + j] = C[i * N + j];
            }
        }
        // mm<dtype>(L, U, C, N, N, N);
        // gemm<dtype>(PP, gv.matrix, C, -1.0, 1.0, N, N, N);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                    1.0, L, N, U, N, 0.0, C, N);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                    -1.0, PP, N, gv.matrix, N, 1.0, C, N);
        dtype norm = 0;
        for (auto i = 0; i < N; ++i) {
            for (auto j = 0; j < i; ++j) {
                norm += C[i * N + j] * C[i * N + j];
            }
        }
        norm = std::sqrt(norm);
        std::cout << "residual: " << norm << std::endl << std::flush;\
        delete U;
        delete L;
    }

    #endif

    MPI_Finalize();

    delete C;
    delete PP;

    return 0;
}
