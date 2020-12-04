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
#include "utils.hpp"

#define dtype double
#define mtype MPI_DOUBLE

namespace conflux {
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

void CalculateDecomposition(
        int M, int N, int P,
        int& Px,
        int& Py,
        int& Pz) {
    int ratio = M/N;
    int p1 = (int) std::cbrt(P/ratio);
    Px = p1;
    Py = ratio * p1;
    Pz = P/(Px*Py);
    /*
    int p13 = (int) (std::pow(P + 1, 1.0 / 3));
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
    */
    // M x N 
}

template <class T>
class GlobalVars {

    private:

        void CalculateParameters(int inpM, int inpN, int v, int inpP) {
            CalculateDecomposition(inpM, inpN, inpP, Px, Py, Pz);
            // v = std::lcm(sqrtp1, c);
            // v = 256;
            this->v = v;
            int nLocalTilesx = (int) (std::ceil((double) inpM / (v * Px)));
            int nLocalTilesy = (int) (std::ceil((double) inpN / (v * Py)));
            M = v * Px * nLocalTilesx;
            M = v * Py * nLocalTilesy;
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
            if (N == 16 && M == 16) {
                matrix = new T[M * N]{
                    1, 8, 2, 7, 3, 8, 2, 4, 8, 7, 5, 5, 1, 4, 4, 9,
                    8, 4, 9, 2, 8, 6, 9, 9, 3, 7, 7, 7, 8, 7, 2, 8,
                    3, 5, 4, 8, 9, 2, 7, 1, 2, 2, 7, 9, 8, 2, 1, 3,
                    6, 4, 1, 5, 3, 7, 9, 1, 1, 3, 2, 9, 9, 5, 1, 9,
                    8, 7, 100, 2, 9, 1, 1, 9, 3, 5, 8, 8, 5, 5, 3, 3,
                    4, 2, 900, 3, 7, 3, 4, 5, 1, 9, 7, 7, 2, 4, 5, 2, //pi=0, pj=1 owner of 900
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
            } else if (N == 32 && M == 32) {
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
                matrix = new T[M * N];

                std::mt19937_64 eng(seed);
                std::uniform_real_distribution<T> dist;
                std::generate(matrix, matrix + M * N, std::bind(dist, eng));
            }
        }

    public:
        int M, N, P;
        // Px refers to rows
        // Py refers to cols
        // Pz refers to height
        int Px, Py, Pz;
        int v, nlayr, Mt, Nt, t, tA11x, tA11y;
        int seed;
        T* matrix;

        GlobalVars(int inpM, int inpN, int v, int inpP, int inpSeed=42) {

            CalculateParameters(inpM, inpN, v, inpP);
            M = inpM;
            N = inpN;
            P = Px * Py * Pz;
            nlayr = (int)((v + Pz-1) / Pz);

            seed = inpSeed;
            InitMatrix();

            Nt = (int) (std::ceil((double) N / v));
            Mt = (int) (std::ceil((double) M / v));
            t = (int) (std::ceil((double) Nt / Py)) + 1ll;
            tA11x = (int) (std::ceil((double) Mt / Px));
            tA11y = (int) (std::ceil((double) Nt / Py));
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

template <typename T>
void push_pivots_up(std::vector<T>& in, std::vector<T>& temp,
                    int n_rows, int n_cols,
                    order layout,
                    std::vector<int>& curPivots,
                    int first_non_pivot_row
                    ) {
    if (n_rows == 0 || n_cols == 0) return;

    std::vector<bool> pivots(n_rows, false);

    // map pivots to bottom rows (after non_pivot_rows)
    for (int i = 0; i < curPivots[0]; ++i) {
        int pivot_row = curPivots[i+1];
        // pivot_row = curPivots[i+1] <= Nl
        pivots[pivot_row] = true;
    }

    // ----------------------
    // v rows -> extract non pivots from first v rows
    // // rest of column:
    // extract pivots from the rest of rows -> late pivots

    // extract from first pivot-rows those which are non-pivots
    std::vector<int> early_non_pivots;
    for (int i = first_non_pivot_row; 
             i < first_non_pivot_row + curPivots[0]; ++i) {
        if (!pivots[i]) {
            early_non_pivots.push_back(i);
        }
    }

    // extract from the rest, those which are pivots
    std::vector<int> late_pivots;
    for (int i = first_non_pivot_row + curPivots[0]; 
             i < n_rows; ++i) {
        if (pivots[i]) {
            late_pivots.push_back(i);
        }
    }

    // copy first non_pivots from in to temp
#pragma omp parallel for
    for (int i = 0; i < early_non_pivots.size(); ++i) {
        int row = early_non_pivots[i];
        std::copy_n(&in[row * n_cols],
                    n_cols,
                    &temp[i * n_cols]);
    }

#pragma omp parallel for
    // overwrites first v rows with pivots
    for (int i = 0; i < curPivots[0]; ++i) {
        int pivot_row = curPivots[i+1];
        std::copy_n(&in[pivot_row * n_cols],
                    n_cols,
                    &in[(first_non_pivot_row + i) * n_cols]);
    }

    // std::cout << "late pivots = " << late_pivots.size() << std::endl;
    // std::cout << "early non pivots = " << early_non_pivots.size() << std::endl;
    assert(late_pivots.size() == early_non_pivots.size());

#pragma omp parallel for
    // copy non_pivots to late_pivots's positions from temp to in
    for (int i = 0; i < late_pivots.size(); ++i) {
        std::copy_n(&temp[i * n_cols], 
                    n_cols, 
                    &in[late_pivots[i] * n_cols]);
    }
}

template <typename T>
void tournament_rounds(
        int n_local_active_rows,
        int v,
        order layout,
        std::vector<T>& A00Buff,
        std::vector<T>& pivotBuff, 
        std::vector<T>& candidatePivotBuff,
        std::vector<T>& candidatePivotBuffPerm,
        std::vector<int>& ipiv, std::vector<int>& perm,
        int n_rounds, 
        int Px, int layrK,
        MPI_Comm lu_comm) {
    int rank;
    MPI_Comm_rank(lu_comm, &rank);
    int pi, pj, pk;
    std::tie(pi, pj, pk) = p2X(lu_comm, rank);

    for (int r = 0; r < n_rounds; ++r) {
        auto src_pi = std::min(flipbit(pi, r), Px - 1);
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
        // candidatePivotBuff := input
        LUP(2*v, v, v+1, &pivotBuff[0], &candidatePivotBuff[1], ipiv, perm);

        // if final round
        if (r == n_rounds - 1) {
            inverse_permute_rows(&candidatePivotBuff[0],
                         &candidatePivotBuffPerm[0],
                         2*v, v+1, v, v+1, layout, perm);

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
                             2*v, v+1, v, v+1, layout, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            } else {
                inverse_permute_rows(&candidatePivotBuff[0], 
                             &candidatePivotBuffPerm[0],
                             2*v, v+1, v, v+1, layout, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            }
        }
    }
}

std::pair<
std::unordered_map<int, std::vector<int>>,
    std::unordered_map<int, std::vector<int>>
    >
    g2lnoTile(std::vector<int>& grows, int Px, int v) {
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
void LU_rep(T* A, 
            T* C, 
            T* PP, 
            GlobalVars<T>& gv, 
            MPI_Comm comm) {
    PC();

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
    // tA10 = gv.tA10;
    // local n
    auto Ml = tA11x * v;
    auto Nl = tA11y * v;

    MPI_Comm lu_comm;
    int dim[] = {Px, Py, Pz}; // 3D processor grid
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

    std::vector<T> A11Buff(Ml * Nl);
    /*
    for (int i = 0; i < A11Buff.size(); ++i) {
        A11Buff[i] = 100 + i;
    }
    */
    std::vector<T> A11BuffTemp(Ml * Nl);

    // global row indices
    std::vector<int> gri(Ml);
    std::unordered_map<int, int> igri;
    std::vector<int> griTemp(Ml);
    for (int i = 0 ; i < Ml; ++i) {
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
    std::vector<T> pivotIndsBuff(M);
    std::vector<T> candidatePivotBuff(Ml * (v+1));
    std::vector<T> candidatePivotBuffPerm(Ml * (v+1));
    std::vector<int> perm(std::max(2*v, Ml)); // rows 
    std::vector<int> ipiv(std::max(2*v, Ml));

    std::vector<int> curPivots(Nl + 1);
    std::vector<int> curPivOrder(v);
    for (int i = 0; i < v; ++i) {
        curPivOrder[i] = i;
    }

    // RNG
    std::mt19937_64 eng(gv.seed);
    std::uniform_int_distribution<int> dist(0, Pz-1);

    // # ------------------------------------------------------------------- #
    // # ------------------ INITIAL DATA DISTRIBUTION ---------------------- #
    // # ------------------------------------------------------------------- #

    // # we distribute only A11, as anything else depends on the first pivots

    // # ----- A11 ------ #
    // # only layer pk == 0 owns initial data
    if (pk == 0) {
        for (auto lti = 0;  lti < tA11x; ++lti) {
            auto gti = l2g(pi, lti, Px);
            for (auto ltj = 0; ltj < tA11y; ++ltj) {
                auto gtj = l2g(pj, ltj, Py);
                mcopy(&A[0], &A11Buff[0],
                        gti * v, (gti + 1) * v, gtj * v, (gtj + 1) * v, N,
                        lti * v, (lti + 1) * v, ltj * v, (ltj + 1) * v, Nl);
            }
        }
    }

    MPI_Win A01Win = create_window(lu_comm,
            A01Buff.data(),
            A01Buff.size(),
            true);

    // Sync all windows
    MPI_Win_fence(MPI_MODE_NOPRECEDE, A01Win);

    int timers[8] = {0};

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

    auto chosen_step = 0;

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
        auto loff = (k / Py) * v; // sqrtp1 = 2, k = 157

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
        /*
        for (int i = 0; i < v; ++i) {
            curPivOrder[i] = i;
        }
        */

        if (n_local_active_rows < v) {
            int padding_start = n_local_active_rows * (v+1);
            int padding_end = v * (v+1);
            std::fill(candidatePivotBuff.begin() + padding_start ,
                      candidatePivotBuff.begin() + padding_end, 0);
            std::fill(candidatePivotBuffPerm.begin() + padding_start,
                      candidatePivotBuffPerm.begin() + padding_end, 0);
            std::fill(gri.begin() + n_local_active_rows, gri.begin() + v, -1);
        }

        // # reduce first tile column. In this part, only pj == k % sqrtp1 participate:
#ifdef DEBUG
        if (k == chosen_step) {
            std::cout << "Step 0, A10Buff before reduction." << std::endl;
            print_matrix_all(A10Buff.data(), 0, Ml, 0, v, v, rank, P, lu_comm);
        }
#endif

        if (pj == k % Py) {
            PE(step0_copy);
            // int p_rcv = X2p(lu_comm, pi, pj, layrK);
            mkl_domatcopy('R', 'N',
                    n_local_active_rows, v,
                    1.0,
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
        MPI_Barrier(lu_comm);
        if (k == chosen_step) {
            std::cout << "Step 0, A10Buff after reduction." << std::endl;
            print_matrix_all(A10Buff.data(), 0, Ml, 0, v, v, rank, P, lu_comm);
            // std::exit(0);
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
#ifdef DEBUG
        if (chosen_step == k) {
            if (pi == 0 && pj == 1 && pk == 0) {
                std::cout << "GRI before tournament" << std::endl;
                print_matrix(gri.data(), 
                             0, 1,
                             0, Ml,
                             Ml);
            }
        }
#endif
        MPI_Request A00_req[2];
        int n_A00_reqs = 0;
        if (pj == k % Py && pk == layrK) {
            auto min_perm_size = std::min(n_local_active_rows, v);
            auto max_perm_size = std::max(n_local_active_rows, v);

            mkl_domatcopy('R', 'N',
                           n_local_active_rows, v,
                           1.0,
                           &A10Buff[first_non_pivot_row * v], v,
                           &candidatePivotBuff[1], v+1);
            assert(n_local_active_rows + first_non_pivot_row == Ml);
            // glue the gri elements to the first column of candidatePivotBuff
            prepend_column(matrix_view<T>(&candidatePivotBuff[0],
                                       n_local_active_rows, v+1, v+1,
                                       layout),
                           &gri[first_non_pivot_row]);
#ifdef DEBUG
            // TODO: before anything
            if (chosen_step == k) {
                std::cout << "candidatePivotBuff BEFORE ANYTHING " << pi << std::endl;
                print_matrix(candidatePivotBuff.data(), 
                             0, n_local_active_rows, 0, v+1, v+1);
            }
#endif
            // # tricky part! to preserve the order of the rows between swapping pairs (e.g., if ranks 0 and 1 exchange their
            // # candidate rows), we want to preserve that candidates of rank 0 are always above rank 1 candidates. Otherwise,
            // # we can get inconsistent results. That's why,in each communication pair, higher rank puts his candidates below:

            // # find with which rank we will communicate
            // # ANOTHER tricky part ! If sqrtp1 is not 2^n, then we will not have a nice butterfly communication graph.
            // # that's why with the flipBit strategy, src_pi can actually be larger than sqrtp1
            auto src_pi = std::min(flipbit(pi, 0), Px - 1);

            LUP(n_local_active_rows, v, v+1, &pivotBuff[0], &candidatePivotBuff[1], ipiv, perm);

            // TODO: after first LUP and swap
#ifdef DEBUG
            if (chosen_step == k) {
                std::cout << "candidatePivotBuff BEFORE FIRST LUP AND SWAP " << pi << std::endl;
                print_matrix(candidatePivotBuff.data(), 
                             0, n_local_active_rows, 0, v+1, v+1);
            }
#endif

            if (src_pi < pi) {
                inverse_permute_rows(&candidatePivotBuff[0], &candidatePivotBuffPerm[v*(v+1)],
                             max_perm_size, v+1, v, v+1, layout, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            } else {
                inverse_permute_rows(&candidatePivotBuff[0], &candidatePivotBuffPerm[0],
                             max_perm_size, v+1, v, v+1, layout, perm);
                candidatePivotBuff.swap(candidatePivotBuffPerm);
            }

            // TODO: after first LUP and swap
#ifdef DEBUG
            if (chosen_step == k) {
                std::cout << "candidatePivotBuff AFTER FIRST LUP AND SWAP " << pi << std::endl;
                print_matrix(candidatePivotBuff.data(), 
                             0, n_local_active_rows, 0, v+1, v+1);
            }
#endif

            // std::cout << "Matrices permuted" << std::endl;

            // # ------------- REMAINING STEPS -------------- #
            // # now we do numRounds parallel steps which synchronization after each step
            // TODO: ask Greg (was sqrtp1)
            auto numRounds = int(std::ceil(std::log2(Py)));
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
                    lu_comm);

                // TODO: final value (all superstep 0)
#ifdef DEBUG
            if (chosen_step == k) {
                std::cout << "candidatePivotBuff FINAL VALUE " << pi << std::endl;
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
            auto gpivots = column<T, int>(matrix_view<T>(&candidatePivotBuff[0],
                                                 min_perm_size, v+1, v+1,
                                                 layout),
                                  0);

            std::unordered_map<int, std::vector<int>> lpivots;
            std::unordered_map<int, std::vector<int>> loffsets;
            std::tie(lpivots, loffsets) = g2lnoTile(gpivots, Px, v);
            // locally set curPivots
            if (n_local_active_rows > 0) {
                curPivots[0] = lpivots[pi].size();
                std::copy_n(&lpivots[pi][0], curPivots[0], &curPivots[1]);
                curPivOrder = loffsets[pi];
                std::copy_n(&gpivots[0], v, &pivotIndsBuff[k*v]);
            } else {
                curPivots[0] = 0;
            }

            // send A00 to pi = k % sqrtp1 && pk = layrK
            // pj = k % sqrtp1; pk = layrK
            if (pi < Py) {
                auto p_rcv = X2p(lu_comm, k % Px, pi, layrK);
                MPI_Isend(&A00Buff[0], v*v, MPI_DOUBLE,
                        p_rcv, 50, lu_comm, &A00_req[n_A00_reqs++]);
            }
        }

        // (pi, k % sqrtp1, layrK) -> (k % sqrtp1, pi, layrK)
        // # Receiving A00Buff:
        if (pj < Px && pi == k % Px && pi < Py && pk == layrK) {
            auto p_send = X2p(lu_comm, pj, pi, layrK);
            MPI_Irecv(&A00Buff[0], v*v, MPI_DOUBLE,
                    p_send, 50, lu_comm, &A00_req[n_A00_reqs++]);
        }

        if (n_A00_reqs > 0) {
            MPI_Waitall(n_A00_reqs, &A00_req[0], MPI_STATUSES_IGNORE);
        }

        // COMMUNICATION
        // MPI_Request reqs_pivots[4];
        // the one who entered this is the root
        auto root = X2p(jk_comm, k % Py, layrK);

        // # Sending A00Buff:
        // MPI_Bcast(&A00Buff[0], v * v, MPI_DOUBLE, root, jk_comm);

        // # Sending pivots:
        MPI_Bcast(&curPivots[0], 1, MPI_INT, root, jk_comm);

        assert(curPivots[0] <= v && curPivots[0] >= 0);

        // sending pivotIndsBuff
        MPI_Bcast(&pivotIndsBuff[k*v], v, MPI_DOUBLE, root, jk_comm);

        MPI_Bcast(&curPivots[1], curPivots[0], MPI_INT, root, jk_comm);

        MPI_Bcast(&curPivOrder[0], curPivots[0], MPI_INT, root, jk_comm); //  &reqs_pivots[3]);

        for (int i = 0; i < curPivots[0]; ++i) {
            curPivots[i+1] = igri[curPivots[i+1]];
        }

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
            }
        }
        MPI_Barrier(lu_comm);
#endif

        push_pivots_up<T>(A11Buff, A11BuffTemp,
                       Ml, Nl,
                       layout, curPivots,
                       first_non_pivot_row);

        push_pivots_up<T>(A10Buff, A10BuffTemp,
                       Ml, v,
                       layout, curPivots,
                       first_non_pivot_row);

        push_pivots_up<int>(gri, griTemp, 
                            Ml, 1,
                            layout, curPivots,
                            first_non_pivot_row);

        igri.clear();
        for (int i = 0; i < Ml; ++i) {
            igri[gri[i]] = i;
        }

#ifdef DEBUG
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
        MPI_Barrier(lu_comm);
#endif

        first_non_pivot_row += curPivots[0];
        n_local_active_rows -= curPivots[0];

        // A01Buff is at the top of A11Buff
        // and contains all the pivot rows

        // for A01Buff
        // TODO: NOW: reduce pivot rows: curPivots[0] x (Nl-loff)
        //
        for (int i = 0; i < curPivots[0]; ++i) {
            int pivot_row = first_non_pivot_row - curPivots[0] + i;
            std::copy_n(&A11Buff[pivot_row * Nl + loff], Nl-loff, &A01Buff[i*(Nl-loff)]);
        }

        if (pk == layrK) {
            MPI_Reduce(MPI_IN_PLACE, &A01Buff[0],
                       curPivots[0] * (Nl-loff),
                       MPI_DOUBLE, MPI_SUM, layrK, k_comm);
        } else {
            MPI_Reduce(&A01Buff[0], &A01Buff[0],
                       curPivots[0] * (Nl-loff),
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
                             0, Nl, Nl);
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
            auto p_rcv = X2p(lu_comm, k % Px, pj, layrK);
            for (int i = 0; i < curPivots[0]; ++i) {
                auto dest_dspls = curPivOrder[i] * (Nl-loff);
                MPI_Put(&A01Buff[i * (Nl-loff)], Nl-loff, MPI_DOUBLE,
                        p_rcv, dest_dspls, Nl-loff, MPI_DOUBLE,
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
                print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
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


        PL();

        MPI_Barrier(lu_comm);
        te = std::chrono::high_resolution_clock::now();
        timers[4] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == print_rank) {
                std::cout << "A10Buff after row swapping" << std::endl;
                print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
            }
        }
#endif

        // TODO: ask Greg (2 * c * sqrtp1)
        MPI_Request reqs[Pz * (Px + Py) +  2];
        int req_id = 0;

        ts = te;
    // # ---------------------------------------------- #
    // # 5. compute A10 and broadcast it to A10BuffRecv #
    // # ---------------------------------------------- #
        if (pk == layrK && pj == k % Py) {
            // # this could basically be a sparse-dense A10 = A10 * U^(-1)   (BLAS tiangular solve) with A10 sparse and U dense
            // however, since we are ignoring the mask, it's dense, potentially with more computation than necessary.
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "before trsm." << std::endl;
                if (rank == 0) {
                    std::cout << "chosen_step = " << chosen_step << std::endl;
                    std::cout << "A00Buff = " << std::endl;
                    print_matrix(A00Buff.data(), 0, v, 0, v, v);
                    std::cout << "A10Buff = " << std::endl;
                    print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
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
                    &A10Buff[first_non_pivot_row * v], // A11
                    v);
            PL();
#ifdef DEBUG
            if (k == chosen_step) {
                std::cout << "after trsm." << std::endl;

                if (rank == print_rank) {
                    std::cout << "A10Buff after trsm" << std::endl;
                    print_matrix(A10Buff.data(), 0, Ml, 0, v, v);
                }
            }
#endif

            PE(step5_reshuffling);
            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
#pragma omp parallel for
            for (int pk_rcv = 0; pk_rcv < Pz; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv*nlayr;
                auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                // int size = nlayr * n_local_active_rows; // nlayr = v / c

                // copy [colStart, colEnd) columns of A10Buff -> A10BuffTemp densely
                mcopy(A10Buff.data(), &A10BuffTemp[offset], 
                        first_non_pivot_row, Ml, colStart, colEnd, v,
                        0, n_local_active_rows, 0, nlayr, nlayr);
            }
            PL();

            PE(step5_comm);
            for (int pk_rcv = 0; pk_rcv < Pz; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                auto colStart = pk_rcv*nlayr;
                // auto colEnd   = (pk_rcv+1)*nlayr;

                int offset = colStart * n_local_active_rows;
                int size = nlayr * n_local_active_rows; // nlayr = v / c

                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for (int pj_rcv = 0; pj_rcv < Py; ++pj_rcv) {
                    auto p_rcv = X2p(lu_comm, pi, pj_rcv, pk_rcv);
                    MPI_Isend(&A10BuffTemp[offset], size, MPI_DOUBLE, 
                            p_rcv, 5, lu_comm, &reqs[req_id]);
                    ++req_id;
                }
            }
            PL();
        }

        auto p_send = X2p(lu_comm, pi, k % Py, layrK);
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
            print_matrix_all(A10BuffRcv.data(), 0, n_local_active_rows, 0, nlayr, nlayr,
                    rank, P, lu_comm);
            MPI_Barrier(lu_comm);
        }
#endif

        ts = te;

        auto lld_A01 = Nl - loff;
        // # ---------------------------------------------- #
        // # 6. compute A01 and broadcast it to A01BuffRecv #
        // # ---------------------------------------------- #
        // # here, only ranks which own data in A01Buff (step 3) participate
        if (pk == layrK && pi == k % Px) {
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
            PL();

            PE(step6_comm);
            // # -- BROADCAST -- #
            // # after compute, send it to sqrt(p1) * c processors
            for(int pk_rcv = 0; pk_rcv < Pz; ++pk_rcv) {
                // # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                auto rowStart = pk_rcv * nlayr;
                // auto rowEnd = (pk_rcv + 1) * nlayr;
                // # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for(int pi_rcv = 0; pi_rcv < Px; ++pi_rcv) {
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

        p_send = X2p(lu_comm, k % Px, pj, layrK);
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
            if (rank == 0) {
                std::cout << "Step 5 finished." << std::endl;
                std::cout << "A01BuffRcv = " << std::endl;
            }
            print_matrix_all(A01BuffRcv.data(), 0, nlayr, 0, Nl, Nl,
                    rank, P, lu_comm);
            MPI_Barrier(lu_comm);
            if (rank == 0) {
                std::cout << "A11 (before) = " << std::endl;
            }
            print_matrix_all(A11Buff.data(), 0, Ml, 
                             0, Nl, Nl,
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
                n_local_active_rows, Nl-loff, nlayr,
                -1.0, &A10BuffRcv[0], nlayr,
                &A01BuffRcv[0], Nl-loff,
                1.0, &A11Buff[first_non_pivot_row * Nl + loff], Nl);
        PL();
#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == 0) {
                std::cout << "A11Buff after computeA11:" << std::endl;
            }
            print_matrix_all(A11Buff.data(), 0, Ml,
                    0, Nl,
                    Nl,
                    rank, P, lu_comm);
        }
#endif
        te = std::chrono::high_resolution_clock::now();
        timers[7] += std::chrono::duration_cast<std::chrono::microseconds>( te - ts ).count();

#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == 0) {
                std::cout << "A10Buff after storing the results back:" << std::endl;
            }
            print_matrix_all(A10Buff.data(), 0, Ml,
                    0, v,
                    v,
                    rank, P, lu_comm);
            if (rank == 0) {
                std::cout << "A10BuffRcv after storing the results back:" << std::endl;
            }
            print_matrix_all(A10BuffRcv.data(), 0, Ml,
                    0, nlayr,
                    nlayr,
                    rank, P, lu_comm);
        }
#endif
        // storing the final result back
        // storing back A10
        // ranks entering the pivoting are the same ranks
        // that have to update A10
        //
        // the only ranks that need to receive A00 buffer
        // are the one participating in dtrsm(A01Buff)
        if (pj == k % Py && pk == layrK) {
            // condensed A10 to non-condensed result buff
            // n_local_active_rows already reduced beforehand
#pragma omp parallel for
            for (int i = first_non_pivot_row; i < Nl; ++i) {
                std::copy_n(&A10Buff[i * v], v, &A11Buff[i*Nl+loff]);
            }
        }

        /*
        // storing back A01 = v * n_local_active_cols
        // and storing back A00
        // A00 has to be communicated from pivot to (pi = k%sqrtp1, pk = layrk)
        // (the reversed pi and pj of the sender)
        if (pi == k % sqrtp1 && pk == layrK) {
            for (int i = 0; i < curPivots[0]; ++i) {
                std::copy_n(&A01Buff[i * (Nl-loff) + v],
                            Nl-loff-v,
                            // columns always move by v
                            &A11Buff[(first_non_pivot_row-curPivots[0] + i) * Nl + loff + v]);
                            // &A11Buff[loff + v]);
            }
        }
        */

#ifdef DEBUG
        if (k == chosen_step) {
            if (rank == 0) {
                std::cout << "A11Buff after storing the results back:" << std::endl;
            }
            print_matrix_all(A11Buff.data(), 0, Ml,
                    0, Nl,
                    Nl,
                    rank, P, lu_comm);
        }
        /*
        if (k == chosen_step) {
            std::exit(0);
        }
        */
        if (k == chosen_step) {
            if (pi == 1 && pj == 0 && pk == 0) {
                std::cout << "Superstep: " << k << std::endl;
                std::cout << "A00Buff after storing the results back:" << std::endl;
                print_matrix(A00Buff.data(), 
                        0, v,
                        0, v,
                        v);
            }
        }
#endif
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, A01Win);
    // Delete all windows
    MPI_Win_free(&A01Win);

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
}
}
