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
#include <unordered_map>

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
        std::vector<int>& pivots,
        std::vector<int>& l2c) {
    // check which rows should be extracted
    // mask[i] = 1 means i-th row should be kept
    std::vector<int> mask(n_rows, -1);
    for (int i = 0; i < pivots[0]; ++i) {
        auto pivot = l2c[pivots[i+1]];
        assert(pivot < n_rows);
        mask[pivot] = 0;
    }
    for (int i = 0; i < n_rows; ++i) {
        if (mask[i] == -1) {
            mask[i] = 1;
        }
    }

    // perform the prefix-sum (exclusive-scan)
    std::vector<int> prefix_sum(n_rows);
    for (int i = 1; i < n_rows; ++i) {
        prefix_sum[i] = prefix_sum[i-1] + mask[i-1];
    }

    // extract kept_rows to temp
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        if (mask[i] == 1) {
            std::copy_n(&mat[i * n_cols], n_cols, 
                    &mat_temp[prefix_sum[i] * n_cols]);
        }
    }

    // swap temp with mat
    mat.swap(mat_temp);
}

template <typename T>
void LUP(int n_local_active_rows, int v, int stride,
        T* pivotBuff, T* candidatePivotBuff,
        std::vector<int>& ipiv, std::vector<int>& perm) {
    // reset the values
    for (int i = 0; i < std::max(2*v, n_local_active_rows); ++i) {
        perm[i] = i;
    }

    mkl_domatcopy('R', 'N',
            n_local_active_rows, v,
            1.0,
            &candidatePivotBuff[0], stride,
            &pivotBuff[0], v);

    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n_local_active_rows, v, 
            &pivotBuff[0], v, &ipiv[0]);

    // ipiv -> permutation
    // ipiv returned from blas is 1-based because of fortran compatibility
    for (int i = 0; i < std::min(v, n_local_active_rows); ++i) {
        std::swap(perm[i], perm[ipiv[i]-1]);
    }
}

// place perm[i]-th row from the input to the i-th row in the output
// perm is the final permutation
template <typename T>
void inverse_permute_rows(T* in, T* out, 
                          int n_rows, int n_cols, int new_n_cols,
                          std::vector<int>& perm) {
    int row_offset = n_cols - new_n_cols;
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        auto src_offset = perm[i] * n_cols + row_offset;
        auto dst_offset = i * new_n_cols;
        std::copy_n(&in[src_offset], new_n_cols, 
                    &out[dst_offset]);
    }
}

// place i-th row from the input to the perm[i]-th row in the output
// perm is the final permutation
template <typename T>
void permute_rows(T* in, T* out, 
                  int n_rows, int n_cols, int new_n_cols,
                  std::vector<int>& perm) {
    int row_offset = n_cols - new_n_cols;
#pragma omp parallel for
    for (int i = 0; i < n_rows; ++i) {
        auto src_offset = i * n_cols + row_offset;
        auto dst_offset = perm[i] * new_n_cols;
        std::copy_n(&in[src_offset], new_n_cols, 
                    &out[dst_offset]);
    }
}

template <typename T>
void push_pivot_rows_below(std::vector<T>& in, std::vector<T>& temp,
                               int n_rows, int n_cols, int new_n_cols,
                               std::vector<int>& curPivots,
                               std::vector<int>& l2c) {
    if (n_rows == 0 || n_cols == 0) return;
    if (curPivots[0] == 0 && new_n_cols == n_cols) return;

    std::vector<int> perm(n_rows, -1);
    int non_pivot_rows = n_rows - curPivots[0];
    assert(n_rows >= curPivots[0]);

#ifdef DEBUG
    for (int i = 0; i < curPivots[0]; ++i) {
        std::cout << "non_pivots = " << non_pivot_rows << ", n_rows = " << n_rows << ", curPivots[" << i+1 << "] = " << l2c[curPivots[i+1]] << std::endl;
    }
#endif

    // map pivots to bottom rows (after non_pivot_rows)
    for (int i = 0; i < curPivots[0]; ++i) {
        int pivot_row = l2c[curPivots[i+1]];
        assert(pivot_row < n_rows);
        assert(pivot_row >= 0);
        assert(non_pivot_rows + i < n_rows);
        assert(non_pivot_rows + i >= 0);
        perm[pivot_row] = non_pivot_rows + i;
    }

    // map non_pivots to upper rows
    int index = 0;
    for (int i = 0; i < n_rows; ++i) {
        // if not a pivot row
        if (perm[i] == -1) {
            perm[i] = index;
            ++index;
        }
    }

    permute_rows(in.data(), temp.data(), n_rows, n_cols, new_n_cols, perm);

    // swap the non-permuted and permuted matrices
    in.swap(temp);
}

template <typename T>
std::vector<int> column(T* matrix, int n_rows, int stride, int col_id) {
    std::vector<int> col;
    col.reserve(n_rows);
    for (int i = 0; i < n_rows; ++i) {
        const auto& el = matrix[i * stride + col_id];
        auto int_el = (int) std::round(el);
        assert(std::abs(int_el - el) < 1e-12);
        if (int_el == -1) break;
        col.push_back(int_el);
    }
    return col;
}

template <typename T>
void tournament_rounds(
        int n_local_active_rows,
        int v,
        std::vector<T>& A00Buff,
        std::vector<T>& pivotBuff, 
        std::vector<T>& candidatePivotBuff,
        std::vector<T>& candidatePivotBuffPerm,
        std::vector<int>& ipiv, std::vector<int>& perm,
        int n_rounds, 
        int sqrtp1, int layrK,
        MPI_Comm lu_comm) {
    int rank;
    MPI_Comm_rank(lu_comm, &rank);
    int pi, pj, pk;
    std::tie(pi, pj, pk) = p2X(lu_comm, rank);

    for (int r = 0; r < n_rounds; ++r) {
        auto src_pi = std::min(flipbit(pi, r), sqrtp1 - 1);
        auto p_rcv = X2p(lu_comm, src_pi, pj, pk);

        // int req_id = 0;
        // MPI_Request reqs[2];

        if (src_pi < pi) {
            MPI_Send(&candidatePivotBuff[v*(v+1)], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm);
            MPI_Recv(&candidatePivotBuff[0], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm, MPI_STATUS_IGNORE);
            /*
            MPI_Isend(&candidatePivotBuff[v*(v+1)], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm, &reqs[req_id++]);
            MPI_Irecv(&candidatePivotBuff[0], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm, &reqs[req_id++]);
                    */
        } else {
            MPI_Recv(&candidatePivotBuff[v*(v+1)], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm, MPI_STATUS_IGNORE);
            MPI_Send(&candidatePivotBuff[0], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm);
            /*
            MPI_Isend(&candidatePivotBuff[0], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm, &reqs[req_id++]);
            MPI_Irecv(&candidatePivotBuff[v*(v+1)], v*(v+1), MPI_DOUBLE,
                    p_rcv, 1, lu_comm, &reqs[req_id++]);
                    */
        }

        if (n_local_active_rows == 0) continue;

        // TODO: after 0th round of communication
#ifdef DEBUG
        if (pi == 0) {
            std::cout << "candidatePivotBuff AFTER 0TH ROUND OF COMM" << std::endl;
            print_matrix(candidatePivotBuff.data(), 
                         0, n_local_active_rows, 0, v+1, v+1);
        }
#endif
        // pivotBuff := output
        // candidatePivotBuff := input
        LUP(2*v, v, v+1, &pivotBuff[0], &candidatePivotBuff[1], ipiv, perm);

        // if final round
        if (r == n_rounds - 1) {
            inverse_permute_rows(&candidatePivotBuff[0], 
                         &candidatePivotBuffPerm[0],
                         v, v+1, v+1, perm);

            candidatePivotBuff.swap(candidatePivotBuffPerm);

            // just the top v rows
            mkl_domatcopy('R', 'N',
                    v, v,
                    1.0,
                    &pivotBuff[0], v,
                    &A00Buff[0], v);
        } else {
            if (src_pi < pi) {
                inverse_permute_rows(&candidatePivotBuff[0], 
                             &candidatePivotBuffPerm[v*(v+1)],
                             v, v+1, v+1, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            } else {
                inverse_permute_rows(&candidatePivotBuff[0], 
                             &candidatePivotBuffPerm[0],
                             v, v+1, v+1, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            }
        }
    }
}

std::pair<
std::unordered_map<int, std::vector<int>>,
    std::unordered_map<int, std::vector<int>>
    >
    g2lnoTile(std::vector<int>& grows, int sqrtp1, int v) {
        std::unordered_map<int, std::vector<int>> lrows;
        std::unordered_map<int, std::vector<int>> loffsets;

        for (unsigned i = 0u; i < grows.size(); ++i) {
            auto growi = grows[i];
            // # we are in the global tile:
            auto gT = growi / v;
            // # which is owned by:
            auto pOwn = int(gT % sqrtp1);
            // # and this is a local tile:
            auto lT = gT / sqrtp1;
            // # and inside this tile it is a row number:
            auto lR = growi % v;
            // # which is a No-Tile row number:
            auto lRNT = int(lR + lT * v);
            lrows[pOwn].push_back(lRNT);
            loffsets[pOwn].push_back(i);
        }

        return {lrows, loffsets};
    }

template <class T>
void LU_rep(T* A, T* C, T* PP, GlobalVars<T>& gv, MPI_Comm comm) {
    PC();

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
    int period[] = {0, 0, 0};
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

    // # get 3d processor decomposition coordinates
    int pi, pj, pk;
    std::tie(pi, pj, pk) = p2X(lu_comm, rank);

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

    // global row indices
    std::vector<int> gri(Nl);
    std::vector<int> griTemp(Nl);
    for (int i = 0 ; i < Nl; ++i) {
        auto lrow = i;
        // # we are in the local tile:
        auto lT = lrow / v;
        // # and inside this tile it is a row number:
        auto lR = lrow % v;
        // # which is a global tile:
        auto gT = lT * sqrtp1 + pi;
        gri[i] = lR + gT * v;
    }

    std::vector<int> l2c(Nl);
    std::vector<bool> removed(Nl, false);
    for (int i = 0; i < Nl; ++i) {
        l2c[i] = i;
    }

    int n_local_active_rows = Nl;
    int n_local_active_cols = Nl;

    std::vector<T> pivotBuff(Nl * v);
    std::vector<T> pivotIndsBuff(N);
    std::vector<T> candidatePivotBuff(Nl * (v+1));
    std::vector<T> candidatePivotBuffPerm(Nl * (v+1));
    std::vector<int> perm(std::max(2*v, Nl));
    std::vector<int> ipiv(std::max(2*v, Nl));

    std::vector<int> curPivots(Nl + 1);
    std::vector<int> curPivOrder(v);
    for (int i = 0; i < v; ++i) {
        curPivOrder[i] = i;
    }

    // RNG
    std::mt19937_64 eng(gv.seed);
    std::uniform_int_distribution<int> dist(0, c-1);

    // # ------------------------------------------------------------------- #
    // # ------------------ INITIAL DATA DISTRIBUTION ---------------------- #
    // # ------------------------------------------------------------------- #

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

    MPI_Win A01Win = create_window(lu_comm,
            A01Buff.data(),
            A01Buff.size(),
            true);

    // Sync all windows
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01Win);

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

    auto chosen_step = Nt;

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
        for (int i = 0; i < v; ++i) {
            curPivOrder[i] = i;
        }

        if (n_local_active_rows < v) {
            std::fill(&candidatePivotBuff[n_local_active_rows * (v+1)], 
                      &candidatePivotBuff[v*(v+1)], 0);
            std::fill(&candidatePivotBuffPerm[n_local_active_rows * (v+1)], 
                      &candidatePivotBuffPerm[v*(v+1)], 0);
        }

        // # reduce first tile column. In this part, only pj == k % sqrtp1 participate:
#ifdef DEBUG
        if (k == chosen_step) {
            std::cout << "Step 0, A10Buff before reduction." << std::endl;
            print_matrix_all(A10Buff.data(), 0, n_local_active_rows, 0, v, v, rank, P, lu_comm);
        }
#endif

        if (pj == k % sqrtp1) {
            PE(step0_copy);
            // int p_rcv = X2p(lu_comm, pi, pj, layrK);
            mkl_domatcopy('R', 'N',
                    n_local_active_rows, v,
                    1.0,
                    &A11Buff[n_local_active_cols - Nl + loff], n_local_active_cols,
                    &A10Buff[0], v); 
            PL();

            PE(step0_reduce);
            if (pk == layrK) {
                MPI_Reduce(MPI_IN_PLACE, &A10Buff[0], n_local_active_rows * v,
                        MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            } else {
                MPI_Reduce(&A10Buff[0], &A10Buff[0], n_local_active_rows * v,
                        MPI_DOUBLE, MPI_SUM, layrK, k_comm);
            }
            PL();
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
        ts = te;
        // # ---------------- FIRST STEP ----------------- #
        // # in first step, we do pivot on the whole PivotBuff array (may be larger than [2v, v]
        // # local computation step
        // sqrtp1 many roots
        // sqrtp1 x c  many receivers
        if (pj == k % sqrtp1 && pk == layrK) {
            // std::cout << "rank = " << pi << ", " << pj << ", " << pk << std::endl;
            // std::cout << "A10Buff = " << std::endl;
            // print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
            mkl_domatcopy('R', 'N',
                           n_local_active_rows, v,
                           1.0,
                           &A10Buff[0], v,
                           &candidatePivotBuff[1], v+1);
            // glue the gri elements to the first column of candidatePivotBuff
            for (int i = 0; i < n_local_active_rows; ++i) {
                candidatePivotBuff[i * (v+1)] = gri[i];
            }
            // pad with zeros (global row index = -1)
            for (int i = n_local_active_rows; i < v; ++i) {
                candidatePivotBuff[i * (v+1)] = -1;
            }

#ifdef DEBUG
            // TODO: before anything
            if (pi == 0) {
                std::cout << "candidatePivotBuff BEFORE ANYTHING" << std::endl;
                print_matrix(candidatePivotBuff.data(), 
                             0, n_local_active_rows, 0, v+1, v+1);
            }
#endif
            // std::cout << "candidatePivotBuff after gluing:" << std::endl;
            // print_matrix(candidatePivotBuff.data(), 0, n_local_active_rows, 0, v+1, v+1);
            // # tricky part! to preserve the order of the rows between swapping pairs (e.g., if ranks 0 and 1 exchange their
            // # candidate rows), we want to preserve that candidates of rank 0 are always above rank 1 candidates. Otherwise,
            // # we can get inconsistent results. That's why,in each communication pair, higher rank puts his candidates below:

            // # find with which rank we will communicate
            // # ANOTHER tricky part ! If sqrtp1 is not 2^n, then we will not have a nice butterfly communication graph.
            // # that's why with the flipBit strategy, src_pi can actually be larger than sqrtp1
            auto src_pi = std::min(flipbit(pi, 0), sqrtp1 - 1);

            LUP(n_local_active_rows, v, v+1, &pivotBuff[0], &candidatePivotBuff[1], ipiv, perm);

            // auto perm_size = std::min(n_local_active_rows, v);
            auto perm_size = v;

            // TODO: after first LUP and swap
#ifdef DEBUG
            if (pi == 0) {
                std::cout << "candidatePivotBuff BEFORE FIRST LUP AND SWAP" << std::endl;
                print_matrix(candidatePivotBuff.data(), 
                             0, n_local_active_rows, 0, v+1, v+1);
            }
#endif

            if (src_pi < pi) {
                inverse_permute_rows(&candidatePivotBuff[0], &candidatePivotBuffPerm[v*(v+1)],
                             perm_size, v+1, v+1, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            } else {
                inverse_permute_rows(&candidatePivotBuff[0], &candidatePivotBuffPerm[0],
                             perm_size, v+1, v+1, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            }

            // TODO: after first LUP and swap
#ifdef DEBUG
            if (pi == 0) {
                std::cout << "candidatePivotBuff AFTER FIRST LUP AND SWAP" << std::endl;
                print_matrix(candidatePivotBuff.data(), 
                             0, n_local_active_rows, 0, v+1, v+1);
            }
#endif

            // std::cout << "Matrices permuted" << std::endl;

            // # ------------- REMAINING STEPS -------------- #
            // # now we do numRounds parallel steps which synchronization after each step
            auto numRounds = int(std::ceil(std::log2(sqrtp1)));
            tournament_rounds(
                    n_local_active_rows,
                    v, 
                    A00Buff,
                    pivotBuff, 
                    candidatePivotBuff,
                    candidatePivotBuffPerm,
                    ipiv, perm,
                    numRounds, 
                    sqrtp1, layrK, 
                    lu_comm);

                // TODO: final value (all superstep 0)
#ifdef DEBUG
            if (pi == 0) {
                std::cout << "candidatePivotBuff FINAL VALUE" << std::endl;
                print_matrix(candidatePivotBuff.data(), 
                             0, n_local_active_rows, 0, v+1, v+1);
            }
#endif
            // std::cout << "tournament rounds finished" << std::endl;

            // extract the first col of candidatePivotBuff
            // first v elements of the first column of candidatePivotBuff
            // first v rows
            // v+1 is the number of cols
            // std::cout << "candidatePivotBuff:" << std::endl;;
            // print_matrix(candidatePivotBuff.data(), 0, v, 0, v+1, v+1);
            auto gpivots = column(&candidatePivotBuff[0], perm_size, v+1, 0);

            std::unordered_map<int, std::vector<int>> lpivots;
            std::unordered_map<int, std::vector<int>> loffsets;
            std::tie(lpivots, loffsets) = g2lnoTile(gpivots, sqrtp1, v);
            // locally set curPivots
            if (n_local_active_rows > 0) {
                curPivots[0] = lpivots[pi].size();
                std::copy_n(&lpivots[pi][0], curPivots[0], &curPivots[1]);
                curPivOrder = loffsets[pi];
                std::copy_n(&gpivots[0], v, &pivotIndsBuff[k*v]);
            } else 
                curPivots[0] = 0;
        }

        // COMMUNICATION
        // MPI_Request reqs_pivots[4];
        // the one who entered this is the root
        auto root = X2p(jk_comm, k % sqrtp1, layrK);

        // # Sending pivots:
        MPI_Bcast(&curPivots[0], 1, MPI_INT, root, jk_comm);

        assert(curPivots[0] <= v && curPivots[0] >= 0);

        // # Sending A00Buff:
        MPI_Bcast(&A00Buff[0], v * v, MPI_DOUBLE, root, jk_comm);

        // sending pivotIndsBuff
        MPI_Bcast(&pivotIndsBuff[k*v], v, MPI_DOUBLE, root, jk_comm);

        // std::cout << "pivotIndsBuff bcast" << std::endl;
        MPI_Bcast(&curPivots[1], curPivots[0], MPI_INT, root, jk_comm);

        MPI_Bcast(&curPivOrder[0], curPivots[0], MPI_INT, root, jk_comm); //  &reqs_pivots[3]);

#ifdef DEBUG
        if (pi == 0 && pk == layrK && pj == k % sqrtp1) {
            std::cout << "pi = " << pi << ", n_local_active_rows = " << n_local_active_rows;;
            std::cout << ", pivots(Nl) = ";
            for (int i = 0; i < curPivots[0]; ++i) {
                std::cout << curPivots[i+1] << ", ";
            }
            std::cout << ", dp = ";
            for (int i = 0; i < Nl; ++i) {
                std::cout << l2c[i] << ", ";
                assert(l2c[i] == -1 || l2c[i] >= 0);
            }
            std::cout << std::endl;
        }
#endif

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
        PE(step2);

#ifdef DEBUG
        if (rank == 0) {
            std::cout << "before pushing pivots below" << std::endl;
            print_matrix(A11Buff.data(), 
                         0, n_local_active_rows, 
                         0, n_local_active_cols, 
                         n_local_active_cols);
        }
        MPI_Barrier(lu_comm);
#endif
        push_pivot_rows_below(A11Buff, A11BuffTemp, 
                              n_local_active_rows, n_local_active_cols, 
                              Nl-loff, curPivots, l2c);

        n_local_active_cols = Nl-loff;
#ifdef DEBUG
        if (rank == 0) {
            std::cout << "after pushing pivots below" << std::endl;
            print_matrix(A11Buff.data(), 
                         0, n_local_active_rows, 
                         0, n_local_active_cols, 
                         n_local_active_cols);
        }
#endif

        // MPI_Barrier(lu_comm);
        int non_pivot_rows = n_local_active_rows - curPivots[0];
        // A01Buff is at the bottom of A11Buff
        // (beneath all non-pivot rows) 
        // and contains all the pivot rows
        T* pivots_ptr = &A11Buff[non_pivot_rows * n_local_active_cols];

        if (pk == layrK) {
            MPI_Reduce(MPI_IN_PLACE, pivots_ptr, 
                       curPivots[0] * n_local_active_cols,
                       MPI_DOUBLE, MPI_SUM, layrK, k_comm);
        } else {
            MPI_Reduce(pivots_ptr, pivots_ptr,
                       curPivots[0] * n_local_active_cols,
                       MPI_DOUBLE, MPI_SUM, layrK, k_comm);
        }
        PL();

        // MPI_Barrier(lu_comm);

        te = std::chrono::high_resolution_clock::now();
        timers[2] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

        ts = te;
#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == print_rank) {
                std::cout << "Step 2 finished." << std::endl;
                print_matrix(A11Buff.data(), 0, n_local_active_rows, 
                             0, n_local_active_cols, n_local_active_cols);
            }
            MPI_Barrier(lu_comm);
        }
#endif

        // # -------------------------------------------------- #
        // # 3. distribute v pivot rows from A11buff to A01Buff #
        // # here, only processors pk == layrK participate      #
        // # -------------------------------------------------- #
        if (pk == layrK) {
            // curPivOrder[i] refers to the target
            auto p_rcv = X2p(lu_comm, k % sqrtp1, pj, layrK);
            for (int i = 0; i < curPivots[0]; ++i) {
                auto dest_dspls = curPivOrder[i] * n_local_active_cols;
                MPI_Put(pivots_ptr, n_local_active_cols, MPI_DOUBLE,
                        p_rcv, dest_dspls, n_local_active_cols, MPI_DOUBLE,
                        A01Win);
            }
        }
        MPI_Win_fence(0, A01Win);

        PL();

        // MPI_Barrier(lu_comm);
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
        // curPivots -> 4, 0 (4th row with second-last and 0th row with the last one)
        // last = n_local_active_rows
        // last(A11) = n_local_active_rows

        // # -------------------------------------------------- #
        // # 4. remove pivot rows from A10 and A11              #
        // # -------------------------------------------------- #
        //
        // prefix-sum[i]:= index in the compacted vector
        // stream - compactions algorithm

        PE(step4);
        remove_pivotal_rows(A10Buff, n_local_active_rows, v, A10BuffTemp, curPivots, l2c);
        remove_pivotal_rows(gri, n_local_active_rows, 1, griTemp, curPivots, l2c);

        for (int i = 0; i < curPivots[0]; ++i) {
            auto pivot_row = curPivots[i+1];
            assert(!removed[pivot_row]);
            l2c[pivot_row] = -1;
        }
        int shift = 0;
        for (int i = 0; i < Nl; ++i) {
            if (!removed[i] && l2c[i] < 0) {
                assert(l2c[i] == -1);
                --shift;
                removed[i] = true;
            } else if (!removed[i]) {
                l2c[i] += shift;
            }
        }
        n_local_active_rows -= curPivots[0];

#ifdef DEBUG
        if (k == chosen_step) {
            for (int i = 0 ; i < P; ++i) {
                if (rank == i) {
                    std::cout << "l2c = ";
                    for (int j = 0; j < Nl; ++j)
                        std::cout << l2c[j] << ", ";
                    std::cout << std::endl;
                    std::cout << "rank = " << pi << ", " << pj << ", " << pk << ", n_local_active_rows " << n_local_active_rows << std::endl;
                }
                MPI_Barrier(lu_comm);
            }
            std::cout << "-----------------------------" << std::endl;
        }
#endif
        PL();

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

        MPI_Request reqs[2 * c * sqrtp1 +  2];
        int req_id = 0;

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
            PE(step5_dtrsm);
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
            PL();
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "after trsm." << std::endl;

                if (rank == print_rank) {
                    std::cout << "A10Buff after trsm" << std::endl;
                    print_matrix(A10Buff.data(), 0, n_local_active_rows, 0, v, v);
                }
            }
#endif

            PE(step5_reshuffling);
            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
#pragma omp parallel for
            for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv*nlayr;
                auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                // int size = nlayr * n_local_active_rows; // nlayr = v / c

                // copy [colStart, colEnd) columns of A10Buff -> A10BuffTemp densely
                mcopy(A10Buff.data(), &A10BuffTemp[offset], 
                        0, n_local_active_rows, colStart, colEnd, v,
                        0, n_local_active_rows, 0, nlayr, nlayr);
            }
            PL();

            PE(step5_comm);
            for (int pk_rcv = 0; pk_rcv < c; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv*nlayr;
                // auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                int size = nlayr * n_local_active_rows; // nlayr = v / c

                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int pj_rcv = 0; pj_rcv <  sqrtp1; ++pj_rcv) {
                    auto p_rcv = X2p(lu_comm, pi, pj_rcv, pk_rcv);
                    MPI_Isend(&A10BuffTemp[offset], size, MPI_DOUBLE, 
                            p_rcv, 5, lu_comm, &reqs[req_id]);
                    ++req_id;
                }
            }
            PL();
        }

        auto p_send = X2p(lu_comm, pi, k % sqrtp1, layrK);
        int size = nlayr * n_local_active_rows; // nlayr = v / c
        if (size < 0) {
            std::cout << "weird size = " << size << ", nlayr = " << nlayr << ", active rowws = " << n_local_active_rows << std::endl;
        }
        MPI_Irecv(&A10BuffRcv[0], size, MPI_DOUBLE, 
                p_send, 5, lu_comm, &reqs[req_id]);
        ++req_id;

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

        auto lld_A01 = Nl-loff;
        // # ---------------------------------------------- #
        // # 6. compute A01 and broadcast it to A01BuffRecv #
        // # ---------------------------------------------- #
        // # here, only ranks which own data in A01Buff (step 3) participate
        if (pk == layrK && pi == k % sqrtp1) {
            PE(step6_dtrsm);
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
                    &A01Buff[0], // A01
                    lld_A01); // leading dim of A01
            PL();

            PE(step6_reshuffling);
            // # local reshuffle before broadcast
            // pack all the data for each rank
            // extract rows [rowStart, rowEnd) and cols [loff, Nl)
            /*
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
            */
            PL();

            PE(step6_comm);
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
                    MPI_Isend(&A01Buff[rowStart * n_cols],
                            nlayr * n_cols, MPI_DOUBLE,
                            p_rcv, 6, lu_comm, &reqs[req_id]);
                    ++req_id;
                }
            }
            PL();
        }

        p_send = X2p(lu_comm, k % sqrtp1, pj, layrK);
        size = nlayr * (Nl-loff); // nlayr = v / c
        MPI_Irecv(&A01BuffRcv[0], size, MPI_DOUBLE, 
                p_send, 6, lu_comm, &reqs[req_id]);
        ++req_id;

        PE(step56_recv);
        MPI_Waitall(req_id, reqs, MPI_STATUSES_IGNORE);
        PL();

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
            print_matrix_all(A11Buff.data(), 0, n_local_active_rows, 
                             0, n_local_active_cols, n_local_active_cols,
                             rank, P, lu_comm);
            MPI_Barrier(lu_comm);
        }
#endif

        PE(step7);
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
                n_local_active_rows, n_local_active_cols, nlayr,
                -1.0, &A10BuffRcv[0], nlayr,
                &A01BuffRcv[0], n_local_active_cols,
                1.0, &A11Buff[0], n_local_active_cols);
        PL();
#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == 0) {
                std::cout << "A11Buff after computeA11:" << std::endl;
            }
            print_matrix_all(A11Buff.data(), 0, n_local_active_rows,
                    0, n_local_active_cols,
                    n_local_active_cols,
                    rank, P, lu_comm);
            std::exit(0);
        }
#endif
        te = std::chrono::high_resolution_clock::now();
        timers[7] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, A01Win);


#ifdef DEBUG
    std::cout << "rank: " << X2p(lu_comm, pi, pj, pk) <<", Finished everything" << std::endl;
    MPI_Barrier(lu_comm);
#endif

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
    MPI_Win_free(&A01Win);

    for (int i = 0; i < P; ++i) {
        if (i == rank) {
            std::cout << "Rank = " << rank << std::endl;
            PP();
        }
        MPI_Barrier(lu_comm);
    }
}
