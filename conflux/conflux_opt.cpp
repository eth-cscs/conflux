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
        // CalculateDecomposition(inpP, sqrtp1, c);
        // v = std::lcm(sqrtp1, c);
        // v = 64;
        // long long nLocalTiles = (long long) (std::ceil((double) inpN / (v * sqrtp1)));
        // N = v * sqrtp1 * nLocalTiles;
        // std::cout << sqrtp1 << " " << c << std::endl << std::flush;
        // std::cout << v << " " << nLocalTiles << std::endl << std::flush;
        N = 16;
        Nt = 8;
        P = 8;
        c = 2;
        sqrtp1 = 2;
        tA10 = 1;
        tA11 = 4;
        v = 2;
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
    std::vector<T> A01BuffRcv(nlayr * Nl);
    std::vector<T> A11Buff(Nl * Nl);

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
    MPI_Win A01RcvWin = create_window(lu_comm,
                                      A01BuffRcv.data(),
                                      A01BuffRcv.size(),
                                      true);
    MPI_Win A11Win = create_window(lu_comm,
                                   A11Buff.data(),
                                   A11Buff.size(),
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

                // A11BuffWin -> A11Buff
                // A01BuffWin -> A01Buff
                // Options:
                // 1. MPI_Put(A01Buff )
                // 2. broadcasting all
                // 3. what ranks are taking part
                A01Buff[p_rcv, offset, loff:] = np.copy(A11Buff[p, pivot, loff:])

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

        ////////////////////////////////
        if (pi == k % sqrtp1 && pk == layrK) {
            // # flush the buffer
            std::fill(PivotA11ReductionBuff + (int)(k / sqrtp1) * v * v,
                      PivotA11ReductionBuff + tA11 * v * v, 0);
        }

        // # ---------------------------------------------- #
        // # 6. compute A10 and broadcast it to A10BuffRecv #
        // # ---------------------------------------------- #
        MPI_Barrier(lu_comm);
        ts = std::chrono::high_resolution_clock::now();
        p2X(rank, p1, sqrtp1, pi, pj, pk);

        // Comp A10
        #pragma omp parallel for
        for (auto ltiA10 = 0; ltiA10 < tA10; ++ltiA10) {
            int tid = omp_get_thread_num();
            PE(copy);
            std::copy(A10Buff + ltiA10 * v * v, A10Buff + (ltiA10 + 1) * v * v,
                      tmpA11 + tid * v * v);
            PL();
        }

        #pragma omp parallel for
        for (auto ltiA10 = 0; ltiA10 < tA10; ++ltiA10) {
            int tid = omp_get_thread_num();
            PE(dtrsm);
            cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                        v, v, 1.0, A00Buff, v, tmpA11 + tid * v * v, v);
            PL();
        }
        #pragma omp parallel for
        for (auto ltiA10 = 0; ltiA10 < tA10; ++ltiA10) {
            int tid = omp_get_thread_num();
            for (auto ii = 0; ii < v; ++ii) {
                if (A10MaskBuff[ltiA10 * v + ii]) {
                    std::copy_n(&tmpA11[tid * v * v + ii * v],
                                v,
                                &A10Buff[(ltiA10 * v + ii) * v]);
                }
            }
        }
        // # here we always go through all rows, and then filter it by the remainingMask array
        for (auto ltiA10 = 0; ltiA10 < tA10; ++ltiA10) {
            // # this is updating A10 based on A00
            // # initial data for A10 is already waiting for us (from initial redundant decomposition or from
            // # the previous reduction of A11)
            // # A00 is waiting for us too. Good to go!

            // # we will compute global tile:
            long long gti = l2gA10(rank, ltiA10, P);
            if (gti >= Nt) {
                break;
            }
            // # which is local tile in A11:
            long long tmp, lti_rcv;
            g2l(gti, sqrtp1, tmp, lti_rcv);

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for (auto pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                for (auto pj_rcv = 0; pj_rcv < sqrtp1; ++pj_rcv) {
                    auto p_rcv = X2p(pi, pj_rcv, pk_rcv, p1, sqrtp1);
                    // # A10buffRcv[p_rcv, lti_rcv, rows] = A10buff[p, ltiA10, rows, pk_rcv*nlayr : (pk_rcv+1)*nlayr]
                    for (auto i = 0; i < v; ++i) {
                        auto row = A10MaskBuff[ltiA10 * v + i];
                        if (row) {
                            if (rank != p_rcv) {
                                MPI_Put(&A10Buff[(ltiA10 * v + i) * v + pk_rcv * nlayr],
                                        nlayr, mtype, p_rcv,
                                        nlayr*(i + v*lti_rcv), nlayr, mtype, A10RcvWin);
                                comm_count[5] += nlayr;
                            }
                        }
                    }
                }
            }
        }

        // # here we always go through all rows, and then filter it by the remainingMask array
#pragma omp parallel for
        for (auto ltiA10 = 0; ltiA10 < tA10; ++ltiA10) {
            // # this is updating A10 based on A00
            // # initial data for A10 is already waiting for us (from initial redundant decomposition or from
            // # the previous reduction of A11)
            // # A00 is waiting for us too. Good to go!

            // # we will compute global tile:
            long long gti = l2gA10(rank, ltiA10, P);
            if (gti < Nt) {
                // # which is local tile in A11:
                long long tmp, lti_rcv;
                g2l(gti, sqrtp1, tmp, lti_rcv);

                // # -- BROADCAST -- #
                // # after compute, send it to sqrt(p1) * c processors
                for (auto pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                    for (auto pj_rcv = 0; pj_rcv < sqrtp1; ++pj_rcv) {
                        auto p_rcv = X2p(pi, pj_rcv, pk_rcv, p1, sqrtp1);
                        // # A10buffRcv[p_rcv, lti_rcv, rows] = A10buff[p, ltiA10, rows, pk_rcv*nlayr : (pk_rcv+1)*nlayr]
                        for (auto i = 0; i < v; ++i) {
                            auto row = A10MaskBuff[ltiA10 * v + i];
                            if (row) {
                                if (rank == p_rcv) {
                                    std::copy(A10Buff + (ltiA10 * v + i) * v + pk_rcv * nlayr,
                                              A10Buff + (ltiA10 * v + i) * v + (pk_rcv + 1) * nlayr,
                                              A10BuffRcv + (lti_rcv * v + i) * nlayr);
                                    local_comm[5] += nlayr;
                                }
                            }
                        }
                    }
                }
            }
        }

        // # ---------------------------------------------- #
        // # 7. compute A01 and broadcast it to A01BuffRecv #
        // # ---------------------------------------------- #

        // Comp A01
        #pragma omp parallel for
        for (auto ltjA01 = k / P; ltjA01 < tA10; ++ltjA01) {
            // cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
            //             v, v, 1.0, A00Buff, v, &A01Buff[ltjA01], v);
            for (auto ik = 0; ik < v - 1; ++ik) {
                for (auto ii = ik + 1; ii < v; ++ii) {
                    for (auto ij = 0; ij < v; ++ij) {
                        A01Buff[(ltjA01 * v + ii) * v + ij] -=
                            A00Buff[ii *v + ik] * A01Buff[(ltjA01 * v + ik) * v + ij];
                    }
                }
            }
        }

        for (auto ltjA01 = k / P; ltjA01 < tA10; ++ltjA01) {
            // # we are computing the global tile:
            auto gtj = l2gA10(rank, ltjA01, P);
            if (gtj <= k) {
                continue;
            }
            if (gtj >= Nt) {
                break;
            }
            // # which is local tile in A11:
            long long pj_rcv, ltj_rcv;
            g2l(gtj, sqrtp1, pj_rcv, ltj_rcv);

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for (auto pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                for (auto pi_rcv = 0; pi_rcv < sqrtp1; ++pi_rcv) {
                    auto p_rcv = X2p(pi_rcv, pj_rcv, pk_rcv, p1, sqrtp1);
                    if (p_rcv != rank) {
                        MPI_Put(&A01Buff[(ltjA01 * v + pk_rcv * nlayr) * v],
                                         nlayr * v, mtype, p_rcv,
                                         ltj_rcv*nlayr*v, nlayr * v, mtype, A01RcvWin);
                        comm_count[6] += nlayr * v;
                    }
                }
            }
        }

#pragma omp parallel for
        for (auto ltjA01 = k / P; ltjA01 < tA10; ++ltjA01) {
            // # we are computing the global tile:
            auto gtj = l2gA10(rank, ltjA01, P);
            if (gtj <= k) {
                continue;
            }
            if (gtj < Nt) {
                // # which is local tile in A11:
                long long pj_rcv, ltj_rcv;
                g2l(gtj, sqrtp1, pj_rcv, ltj_rcv);

                // # -- BROADCAST -- #
                // # after compute, send it to sqrt(p1) * c processors
                for (auto pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                    for (auto pi_rcv = 0; pi_rcv < sqrtp1; ++pi_rcv) {
                        auto p_rcv = X2p(pi_rcv, pj_rcv, pk_rcv, p1, sqrtp1);
                        if (p_rcv == rank) {
                            mcopy(A01Buff + ltjA01 * v * v,
                                  A01BuffRcv + ltj_rcv * nlayr * v,
                                  pk_rcv * nlayr, (pk_rcv + 1) * nlayr, 0, v, v,
                                  0, nlayr, 0, v, v);
                            local_comm[6] += nlayr * v;
                        }
                    }
                }
            }
        }

        MPI_Win_fence(0, A01RcvWin);
        MPI_Win_fence(0, A10RcvWin);
        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[5] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        #ifdef DEBUG_PRINT
        if (doprint && printrank) {
            for (auto i = 0; i < tA10 * v * v; ++i) {
                std::cout << A10Buff[i] << ", ";
            }
            std::cout << std::endl << std::flush;
            for (auto i = 0; i < tA11 * v * nlayr; ++i) {
                std::cout << A10BuffRcv[i] << ", ";
            }
            std::cout << std::endl << std::flush;
            for (auto i = 0; i < tA10 * v * v; ++i) {
                std::cout << A01Buff[i] << ", ";
            }
            std::cout << std::endl << std::flush;
            for (auto i = 0; i < tA11 * v * nlayr; ++i) {
                std::cout << A01BuffRcv[i] << ", ";
            }
            std::cout << std::endl << std::flush;
        }
        #endif

        MPI_Barrier(lu_comm);
        ts = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (auto lti = 0; lti < tA11; ++lti) {
            // # filter which rows of this tile should be processed:
            // rows = A11MaskBuff[lti, :]
            // A10 = A10BuffRcv[lti, rows]
            for (auto ltj = k / P; ltj < tA11; ++ltj) {
                int tid = omp_get_thread_num();
                // # Every processor computes only v/c "layers" in k dimension
                PE(dgemm);
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, v, v, nlayr,
                            1.0, &A10BuffRcv[lti * v * nlayr], nlayr,
                            &A01BuffRcv[ltj * nlayr * v], v,
                            0.0,  tmpA11 + tid * v * v, v);
                PL();
            }
        }
        #pragma omp parallel for collapse(2)
        for (auto lti = 0; lti < tA11; ++lti) {
            // # filter which rows of this tile should be processed:
            // rows = A11MaskBuff[lti, :]
            // A10 = A10BuffRcv[lti, rows]

            for (auto ltj = k / P; ltj < tA11; ++ltj) {
                int tid = omp_get_thread_num();
                PE(mask);
                for (auto ii = 0; ii < v; ++ii) {
                    auto row = A11MaskBuff[lti * v + ii];
                    if (row) {
                        for (auto ij = 0; ij < v; ++ij) {
                            A11Buff[((lti * tA11 + ltj) * v + ii) * v + ij] -= tmpA11[tid * v * v + ii * v + ij];
                        }
                    }
                }
                PL();
            }
        }

        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[6] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        if (rank == 0) {
            std::cout << "AFTER STEP " << k << std::endl;;
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    std::cout << B[i * N + j] << ", ";
                }
                std::cout << std::endl;
            }
        }

        MPI_Barrier(lu_comm);

        #ifdef DEBUG_PRINT
        if (doprint && printrank) {
            for (auto i = 0; i < tA11 * tA11 * v * v; ++i) {
                std::cout << A11Buff[i] << ", ";
            }
            std::cout << std::endl << std::flush;
        }
        #endif

        // # ----------------------------------------------------------------- #
        // # ------------------------- DEBUG ONLY ---------------------------- #
        // # ----------- STORING BACK RESULTS FOR VERIFICATION --------------- #

        #ifdef VALIDATE

        // # -- A10 -- #
        if (rank == 0) {
            std::copy(pivotIndsBuff, pivotIndsBuff + N, bufpivots);
        }
        MPI_Bcast(bufpivots, N, MPI_INT, 0, lu_comm);
        // remaining = np.setdiff1d(np.array(range(N)), bufpivots)
        int rem_num = 0;
        for (auto i = 0; i < N; ++i) {
            bool found = false;
            for (auto j = 0; j < N; ++j) {
                if (i == bufpivots[j]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                remaining[rem_num] = i;
                rem_num++;
            }
        }
        // tmpA10 = np.zeros([1,v])
        for (auto ltiA10 = 0; ltiA10 < tA10; ++ ltiA10) {
            for (auto r = 0; r < P; ++r) {
                if (rank == r) {
                    std::copy(A10Buff + ltiA10 * v * v,
                              A10Buff + (ltiA10 + 1) * v * v,
                              buf);
                    // buf[:] = A10Buff[ltiA10]
                }
                MPI_Bcast(buf, v * v, mtype, r, lu_comm);
                std::copy(buf, buf + v * v, tmpA10 + (ltiA10 * P + r) * v * v);
                // tmpA10 = np.concatenate((tmpA10, buf), axis = 0)
            }
        }
        // tmpA10 = tmpA10[1:, :]
        for (auto i = 0; i < rem_num; ++i) {
            auto rem = remaining[i];
            std::copy(tmpA10 + rem * v, tmpA10 + (rem + 1) * v, B + rem * N + off);
        }
        // B[remaining, off : off + v] = tmpA10[remaining]

        // # -- A00 and A01 -- #
        // tmpA01 = np.zeros([v,1])
        for (auto ltjA10 = 0; ltjA10 < tA10; ++ ltjA10) {
            for (auto r = 0; r < P; ++r) {
                auto gtj = l2gA10(r, ltjA10, P);
                if (gtj >= Nt) break;
                if (rank == r) {
                    std::copy(A01Buff + ltjA10 * v * v,
                              A01Buff + (ltjA10 + 1) * v * v,
                              buf);
                }
                MPI_Bcast(buf, v * v, mtype, r, lu_comm);
                mcopy(buf, tmpA01, 0, v, 0, v, v, 0, v,
                      (ltjA10 * P + r) * v, (ltjA10 * P + r + 1) * v, tA10 * P * v);
                // tmpA01 = np.concatenate((tmpA01, buf), axis = 1)
            }
        }
        // tmpA01 = tmpA01[:, (1+ (k+1)*v):]

        if (rank == 0) {
            std::copy(A00Buff, A00Buff + v * v, buf);
        }
        MPI_Bcast(buf, v * v, mtype, 0, lu_comm);
        if (rank == layrK) {
            std::copy(pivotIndsBuff, pivotIndsBuff + N, bufpivots);
        }
        MPI_Bcast(bufpivots, N, MPI_INT, layrK, lu_comm);
        // curPivots = bufpivots[off : off + v]
        for (auto i = 0; i < v; ++i) {
            // # B[curPivots[i], off : off + v] = A00buff[0, i, :]
            auto curpiv = bufpivots[off + i];
            if (curpiv == -1) curpiv = N - 1;
            // B[curPivots[i], off : off + v] = buf[i, :]
            // assert(i * v + v <= v * v);
            // assert(curpiv * N + off + v <= N * N);
            // if (rank == 0) std::cout << "(" << k << ", " << curpiv << ", " << off << ", " << i << "):" << std::flush;
            // if (rank == 0) std::cout << B[curpiv * N + off + i] << std::endl << std::flush;
            std::copy(buf + i * v, buf + i * v + v, B + curpiv * N + off);
            // B[curPivots[i], (off + v):] = tmpA01[i, :]
            std::copy(tmpA01 + i * tA10 * P * v + (k+1)*v,
                      tmpA01 + (i + 1) * tA10 * P * v,
                      B + curpiv * N + off + v);
        }

        MPI_Barrier(lu_comm);

        #endif

        // # ----------------------------------------------------------------- #
        // # ----------------------END OF DEBUG ONLY ------------------------- #
        // # ----------------------------------------------------------------- #
    }

    // std::cout << "Rank : " << rank << ", comm: " << comm_count << " elements" << std::endl;

    // long long count[1];
    // long long local[1];
    // count[0] = comm_count;
    // local[0] = local_comm;
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

        // for (auto i = 0; i < N; ++i) {
        //     for (auto j = 0; j < N; ++j) {
        //         std::cout << C[i * N + j] << " " << std::flush;
        //     }
        //     std::cout << std::endl << std::flush;
        // }
    } else {
        MPI_Reduce(comm_count, NULL, 7, MPI_LONG_LONG, MPI_SUM, 0, lu_comm);
        MPI_Reduce(local_comm, NULL, 7, MPI_LONG_LONG, MPI_SUM, 0, lu_comm);
    }

    // Delete all windows
    MPI_Win_free(&A00Win);
    MPI_Win_free(&A10Win);
    MPI_Win_free(&A10RcvWin);
    MPI_Win_free(&A01Win);
    MPI_Win_free(&A01RcvWin);
    MPI_Win_free(&A11Win);
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
