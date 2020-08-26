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
#include <tuple>

#include <omp.h>
#include <mpi.h>
#include <mkl.h>

#include "profiler.hpp"

#define dtype double
#define mtype MPI_DOUBLE

template <class T>
void mcopy(T* src, T* dst,
           int ssrow, int serow, int sscol, int secol, int sstride,
           int dsrow, int derow, int dscol, int decol, int dstride) {
    PE(mcopy);
    assert(serow-ssrow == derow-dsrow);
    assert(secol-sscol == decol-dscol);

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

int l2g(int pi, int ind, int sqrtp1) {
    return ind * sqrtp1 + pi;
}

void g2l(int gind, int sqrtp1,
         int& out1, int& out2){
    out1 = gind % sqrtp1;
    out2 = (int) (gind / sqrtp1);
}

void g2lA10(int gti, int P, int& p, int& lti) {
    lti = (int) (gti / P);
    p = gti % P;
}

int l2gA10(int p, int lti, int P) {
    return lti * P + p;
}

void gr2gt(int gri, int v, int& gti, int& lri) {
    gti = (int) (gri / v);
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

double ModelCommCost(int ppp, int c) {
    return 1.0 / (ppp * c);
}

void CalculateDecomposition(int P,
                            int& best_ppp,
                            int& best_c) {
    int p13 = (int) (std::pow(P + 1, 1.0 / 3));
    int ppp = (int) (std::sqrt(P));
    int c = 1ll;
    best_ppp = ppp;
    best_c = c;
    double bestCost = ModelCommCost(ppp, c);
    while (c <= p13) {
        int P1 = (int )(P / c);
        ppp = (int) (std::sqrt(P1));
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

    void CalculateParameters(int inpN, int v, int inpP) {
        CalculateDecomposition(inpP, sqrtp1, c);
        // v = std::lcm(sqrtp1, c);
        // v = 256;
        this->v = v;
        int nLocalTiles = (int) (std::ceil((double) inpN / (v * sqrtp1)));
        N = v * sqrtp1 * nLocalTiles;
        // std::cout << sqrtp1 << " " << c << std::endl << std::flush;
        // std::cout << v << " " << nLocalTiles << std::endl << std::flush;
        /*
        v = 2;
        N = 16;
        sqrtp1 = 2;
        P = 8;
        p1 = 4;
        c = 2;
        */
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
    int N, P;
    int p1, sqrtp1, c;
    int v, nlayr, Nt, t, tA11, tA10;
    int seed;
    T* matrix;

    GlobalVars(int inpN, int v, int inpP, int inpSeed=42) {

        CalculateParameters(inpN, v, inpP);
        P = sqrtp1 * sqrtp1 * c;
        nlayr = (int)((v + c-1) / c);
        p1 = sqrtp1 * sqrtp1;

        seed = inpSeed;
        InitMatrix();

        Nt = (int) (std::ceil((double) N / v));
        t = (int) (std::ceil((double) Nt / sqrtp1)) + 1ll;
        tA11 = (int) (std::ceil((double) Nt / sqrtp1));
        tA10 = (int) (std::ceil((double) Nt / P));
    }

    ~GlobalVars() {
        delete matrix;
    }
};


int flipbit(int n, int k) {
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

template <typename T> 
void print_matrix(T* pointer, 
                  int row_start, int row_end,
                  int col_start, int col_end,
                  int stride) {
    for (int i = row_start; i < row_end; ++i) {
        for (int j = col_start; j < col_end; ++j) {
            std::cout << pointer[i * stride + j] << ", \t";
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
void remove_pivotal_rows(std::vector<T>& mat,
                         int n_rows, int n_cols,
                         std::vector<T>& mat_temp,
                         std::vector<int> pivots) {
    auto n_pivots = pivots[0];
    std::sort(&pivots[1], &pivots[n_pivots+1]);

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
    for (unsigned i = 0; i < kept_rows.size(); ++i) {
        const auto& row = kept_rows[i];
        std::copy_n(&mat[row * n_cols], n_cols, &mat_temp[i * n_cols]);
    }

    // swap temp with mat
    mat.swap(mat_temp);
}

template <class T>
void LU_rep(T* A, T* C, T* PP, GlobalVars<T>& gv, MPI_Comm comm) {

    int N, P, p1, sqrtp1, c, v, nlayr, Nt, tA11;
    N = gv.N;
    P = gv.P;
    p1 = gv.p1;
    sqrtp1 = gv.sqrtp1;
    c = gv.c;
    v = gv.v;
    nlayr = gv.nlayr;
    Nt = gv.Nt;
    tA11 = gv.tA11;
    // tA10 = gv.tA10;
    // local n
    auto Nl = tA11 * v;

    MPI_Comm lu_comm;
    int dim[] = {sqrtp1, sqrtp1, c}; // 3D processor grid
    int period[] = {0, 0};
    int reorder = 1;
    MPI_Cart_create(comm, 3, dim, period, reorder, &lu_comm);

    int rank;
    MPI_Comm_rank(lu_comm, &rank);

    int print_rank = X2p(lu_comm, 0, 0, 0);

    MPI_Comm k_comm;
    int keep_dims_k[] = {0, 0, 1};
    MPI_Cart_sub(lu_comm, keep_dims_k, &k_comm);

    MPI_Comm jk_comm;
    int keep_dims_jk[] = {0, 1, 1};
    MPI_Cart_sub(lu_comm, keep_dims_jk, &jk_comm);

    std::vector<T> B(N * N);
    std::copy(A, A + N * N, B.data());

    // Perm = np.eye(N);
    std::vector<int> Perm(N * N);
    for (unsigned i = 0; i < N; ++i) {
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
    std::uniform_int_distribution<int> dist(0, c-1);

    // # ------------------------------------------------------------------- #
    // # ------------------ INITIAL DATA DISTRIBUTION ---------------------- #
    // # ------------------------------------------------------------------- #
    // # get 3d processor decomposition coordinates
    int pi, pj, pk;
    std::tie(pi, pj, pk) = p2X(lu_comm, rank);

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
#ifdef DEBUG
    if (rank == print_rank) {
        print_matrix(A11Buff.data(), 0, Nl, 0, Nl, Nl);
    }
    std::cout << "Allocated." << std::endl;
    MPI_Barrier(lu_comm);
#endif

    // Create windows
    MPI_Win A11Win = create_window(lu_comm,
                                   A11Buff.data(),
                                   A11Buff.size(),
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

    // Sync all windows
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A11Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A10RcvWin);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01Win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01RcvWin);

    int timers[8] = {0};

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

    auto chosen_step = 1;

    MPI_Barrier(lu_comm);
    auto t1 = std::chrono::high_resolution_clock::now();

    // # now k is a step number
    for (auto k = 0; k < Nt; ++k) {
        bool last_step = k == Nt-1;
#ifdef DEBUG
        std::cout << "Iteration = " << k << std::endl;
        MPI_Barrier(lu_comm);
#endif
        // if (k == 1) break;

        // global current offset
        // auto off = k * v;
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

        // flush the buffer
        curPivots[0] = 0;

        // # reduce first tile column. In this part, only pj == k % sqrtp1 participate:
#ifdef DEBUG
        if (k == chosen_step) {
            std::cout << "Step 0, A10Buff before reduction." << std::endl;
            print_matrix_all(A10Buff.data(), 0, n_local_active_rows, 0, v, v, rank, P, lu_comm);
        }
#endif

        if (pj == k % sqrtp1) {
            // int p_rcv = X2p(lu_comm, pi, pj, layrK);
            mkl_domatcopy('R', 'N',
                           n_local_active_rows, v,
                           1.0,
                           &A11Buff[loff], Nl,
                           &A10Buff[0], v); 

            if (pk == layrK) {
                MPI_Reduce(MPI_IN_PLACE, &A10Buff[0], n_local_active_rows * v,
                           MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            } else {
                MPI_Reduce(&A10Buff[0], &A10Buff[0], n_local_active_rows * v,
                           MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            }
        }

 #ifdef DEBUG
        MPI_Barrier(lu_comm);
        if (k == chosen_step) {
            std::cout << "Step 0, A10Buff after reduction." << std::endl;
            print_matrix_all(A10Buff.data(), 0, n_local_active_rows, 0, v, v, rank, P, lu_comm);
            std::exit(0);
        }
#endif


#ifdef DEBUG
        std::cout << "Step 0 finished." << std::endl;
        MPI_Barrier(lu_comm);
#endif
        MPI_Barrier(lu_comm);
        auto te = std::chrono::high_resolution_clock::now();
        timers[0] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        // # --------------------------------------------------------------------- #
        // # 1. coalesce PivotA11ReductionBuff to PivotBuff and scatter to A10buff #
        // # --------------------------------------------------------------------- #
        MPI_Barrier(lu_comm);
        ts = std::chrono::high_resolution_clock::now();

        int zero = 0;
        if (k % sqrtp1 == pi && k % sqrtp1 == pj && pk == layrK) {
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "A10Buff before copy" << std::endl;
                print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
            }
#endif
            // # filter A11buff by masked pivot rows (local) and densify it
            mkl_domatcopy('R', 'N',
                           n_local_active_rows, v,
                           1.0,
                           &A10Buff[0], v,
                           &pivotBuff[0], v);
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "pivotBuff after copy" << std::endl;
                print_matrix(pivotBuff.data(), 0, n_local_active_rows, 0, v, v);
            }
#endif

#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "Before LU decomposition, the result = " << std::endl;
                print_matrix(pivotBuff.data(), 0, n_local_active_rows, 0, v, v);
            }
#endif
            LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n_local_active_rows, v, pivotBuff.data(), v, ipiv.data());
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "After LU decomposition, the result = " << std::endl;
                print_matrix(pivotBuff.data(), 0, n_local_active_rows, 0, v, v);
                std::exit(0);
            }
#endif

            for (int i = 0; i < Nl; ++i) {
                curPivots[i+1] = i;
            }
            curPivots[0] = v;

            for (unsigned i = 0; i < ipiv.size(); ++i) {
                auto& a = curPivots[i+1];
                auto& b = curPivots[ipiv[i]];
                std::swap(a, b);
            }

#ifdef DEBUG
            if (k == chosen_step) {
                print_matrix(curPivots.data(), 0, 1,  0, v+1, v+1);
            }
#endif
            mkl_domatcopy('R', 'N',
                           v, v,
                           1.0,
                           &pivotBuff[0], v,
                           &A00Buff[0], v);
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "A00Buff = " << std::endl;
                print_matrix(A00Buff.data(), 0, v, 0, v, v);
            }
#endif
        } else if (pi != k % sqrtp1) {
            curPivots[0] = 0;
        }

        // BCast is a collective, must be called from all the rank
        if (!last_step) {
            auto root = X2p(lu_comm, k % sqrtp1, k % sqrtp1, 0);
            // # Sending A00Buff:
            MPI_Bcast(&A00Buff[0], v * v, MPI_DOUBLE, root, lu_comm);

            if (pi == k % sqrtp1) {
                auto pivot_root = X2p(jk_comm, k % sqrtp1, layrK);
                // # Sending pivots:
                MPI_Bcast(&curPivots[0], v+1, MPI_INT, pivot_root, jk_comm);
            }
        }

        if (last_step) break;


        // std::sort(curPivots.begin() + 1, curPivots.end());

#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == 1) {
                std::cout << "After sending A00Buff, rank 1 has:" << std::endl;;
                print_matrix(A00Buff.data(), 0, v, 0, v, v);
            }

            MPI_Barrier(lu_comm);
            std::cout << "Step 1 finished." << std::endl;
        }
#endif

        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[1] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

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
        if (pk != layrK) {
            auto p_rcv = X2p(lu_comm, pi, pj, layrK);
            for (int i = 0; i < curPivots[0]; ++i) {
                int pivot_row = curPivots[i+1];
                auto offset = loff + pivot_row * Nl;
                MPI_Accumulate(&A11Buff[offset], v, MPI_DOUBLE,
                               p_rcv, offset, v, MPI_DOUBLE,
                               MPI_SUM, A11Win);
            }
        }
        MPI_Win_fence(0, A11Win);
        MPI_Barrier(lu_comm);

        te = std::chrono::high_resolution_clock::now();
        timers[2] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        ts = te;
#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == print_rank) {
                std::cout << "Step 2 finished." << std::endl;
                print_matrix(A11Buff.data(), 0, n_local_active_rows, 0, Nl, Nl);
            }
            MPI_Barrier(lu_comm);
        }
#endif

        // # -------------------------------------------------- #
        // # 3. distribute v pivot rows from A11buff to A01Buff #
        // # here, only processors pk == layrK participate      #
        // # -------------------------------------------------- #
        if (pk == layrK) {
            auto p_rcv = X2p(lu_comm, k % sqrtp1, pj, layrK);
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
        te = std::chrono::high_resolution_clock::now();
        timers[3] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == print_rank) {
                std::cout << "Step 3 finished." << std::endl;
                print_matrix(A01Buff.data(), 0, v, 0, Nl, Nl);
            }
            MPI_Barrier(lu_comm);
        }
#endif

        ts = te;

        // remove pivotal rows from matrix A10Buff and A11Buff
        // A10Buff
#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == print_rank) {
                std::cout << "A10Buff before row swapping" << std::endl;
                print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
            }
        }
#endif

        // we want to push at the end the following rows:
        // ipiv -> series of row swaps 5, 5
        // curPivots -> 4, 0 (4th row with second-last and 0th row with the one)
        // last = n_local_active_rows
        // last(A11) = n_local_active_rows

        // # -------------------------------------------------- #
        // # 4. remove pivot rows from A10 and A11              #
        // # -------------------------------------------------- #

        remove_pivotal_rows(A10Buff, n_local_active_rows, v, A10BuffTemp, curPivots);
        remove_pivotal_rows(A11Buff, n_local_active_rows, Nl, A11BuffTemp, curPivots);
        n_local_active_rows -= curPivots[0];

        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[4] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == print_rank) {
                std::cout << "A10Buff after row swapping" << std::endl;
                print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
            }
        }
#endif

        ts = te;
        // # ---------------------------------------------- #
        // # 5. compute A10 and broadcast it to A10BuffRecv #
        // # ---------------------------------------------- #
        if (pk == layrK && pj == k % sqrtp1) {
            // # this could basically be a sparse-dense A10 = A10 * U^(-1)   (BLAS tiangular solve) with A10 sparse and U dense
            // however, since we are ignoring the mask, it's dense, potentially with more computation than necessary.
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "before trsm." << std::endl;
                if (rank == 2) {
                    std::cout << "chosen_step = " << chosen_step << std::endl;
                    std::cout << "A00Buff = " << std::endl;
                    print_matrix(A00Buff.data(), 0, v, 0, v, v);
                    std::cout << "A10Buff = " << std::endl;
                    print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
                }
            }
#endif
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
#ifdef DEBUG
        if (k == chosen_step) {
            std::cout << "after trsm." << std::endl;

            if (rank == print_rank) {
                std::cout << "A10Buff after trsm" << std::endl;
                print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
            }
        }
#endif

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
#pragma omp parallel for
            for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv*nlayr;
                auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                int size = nlayr * n_local_active_rows; // nlayr = v / c

                // copy [colStart, colEnd) columns of A10Buff -> A10BuffTemp densely
                mcopy(A10Buff.data(), &A10BuffTemp[offset], 
                      0, n_local_active_rows, colStart, colEnd, v,
                      0, n_local_active_rows, 0, nlayr, nlayr);
            }

            for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv*nlayr;
                auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                int size = nlayr * n_local_active_rows; // nlayr = v / c

                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int pj_rcv = 0; pj_rcv <  sqrtp1; ++pj_rcv) {
                    auto p_rcv = X2p(lu_comm, pi, pj_rcv, pk_rcv);
                    MPI_Put(&A10BuffTemp[offset], size, MPI_DOUBLE,
                        p_rcv, 0, size, MPI_DOUBLE, A10RcvWin);
                }
            }
        }

        MPI_Win_fence(0, A10RcvWin);
        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[5] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

#ifdef DEBUG
        if (k == chosen_step) {
            std::cout << "Step 4 finished." << std::endl;
            std::cout << "Matrix A10BuffRcv = " << std::endl;
            print_matrix_all(A10BuffRcv.data(), 0, Nl, 0, nlayr, nlayr,
                             rank, P, lu_comm);
            MPI_Barrier(lu_comm);
        }
#endif

        ts = te;

        auto lld_A01 = Nl;
        // # ---------------------------------------------- #
        // # 6. compute A01 and broadcast it to A01BuffRecv #
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
                mcopy(A01Buff.data(), A01BuffTemp.data(),
                      rowStart, rowEnd, loff, Nl, lld_A01,
                      rowStart, rowEnd, 0, Nl-loff, Nl-loff);
            }

            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for(int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                auto rowStart = pk_rcv * nlayr;
                // auto rowEnd = (pk_rcv + 1) * nlayr;
                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for(int pi_rcv = 0; pi_rcv < sqrtp1; ++pi_rcv) {
                    const int n_cols = Nl - loff;
                    auto p_rcv = X2p(lu_comm, pi_rcv, pj, pk_rcv);
                    MPI_Put(&A01BuffTemp[rowStart * n_cols],
                            nlayr * n_cols, MPI_DOUBLE,
                            p_rcv, 0, nlayr * n_cols, 
                            MPI_DOUBLE, A01RcvWin);
                }
            }
        }

        MPI_Win_fence(0, A01RcvWin);

        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[6] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

#ifdef DEBUG
        if (k == chosen_step) {
            std::cout << "Step 5 finished." << std::endl;
            std::cout << "A01BuffRcv = " << std::endl;
            print_matrix_all(A01BuffRcv.data(), 0, nlayr, 0, Nl, Nl,
                             rank, P, lu_comm);
            MPI_Barrier(lu_comm);
            std::cout << "A11 (before) = " << std::endl;
            print_matrix_all(A11Buff.data(), 0, n_local_active_rows, 0, Nl, Nl,
                             rank, P, lu_comm);
            MPI_Barrier(lu_comm);
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
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n_local_active_rows, Nl - loff, nlayr,
                    -1.0, &A10BuffRcv[0], nlayr,
                    &A01BuffRcv[0], Nl-loff,
                    1.0, &A11Buff[loff], Nl);
#ifdef DEBUG
        if (k == chosen_step) {
            for (int i = 0; i < P; ++i) {
                if (rank == i) {
                    std::cout << "rank " << X2p(lu_comm, pi, pj, pk) << ", A11Buff after computeA11:" << std::endl;
                    print_matrix(A11Buff.data(), 0, n_local_active_rows,
                                                 0, Nl,
                                                 Nl);
                }
                MPI_Barrier(lu_comm);
            }
            std::exit(0);
        }
#endif

        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[7] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();
    }

#ifdef DEBUG
    std::cout << "rank: " << X2p(lu_comm, pi, pj, pk) <<", Finished everything" << std::endl;
    MPI_Barrier(lu_comm);
#endif

    // # recreate the permutation matrix
    /*
    std::vector<T> Permutation(N * N);
    for (int i = 0; i < N; ++i) {
        auto row = ipiv[i];
        std::copy_n(&B[row * N], N, &C[i * N]);
        std::copy_n(&Perm[row * N], N, &Permutation[i * N]);
    }
    */

    MPI_Barrier(lu_comm);
    if (rank == print_rank) {
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        std::cout << "Runtime: " << double(duration) / 1000000 << " seconds" << std::endl;

        for (auto i = 0; i < 8; ++i) {
            std::cout << "Runtime " << i << ": " << double(timers[i]) / 1000000 << " seconds" << std::endl;
        }
    }

    // Delete all windows
    MPI_Win_free(&A11Win);
    MPI_Win_free(&A10RcvWin);
    MPI_Win_free(&A01Win);
    MPI_Win_free(&A01RcvWin);
}
