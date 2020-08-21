#include <mpi.h>
#include <conflux/conflux_opt.hpp>

int main(int argc, char **argv) {
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

    std::vector<double> C(gv.N * gv.N);
    std::vector<double> Perm(gv.N * gv.N);

    int n_rep = 2;

    for (int i = 0; i < n_rep; ++i) {
        PC();
        LU_rep<dtype>(gv.matrix, C.data(), Perm.data(), gv, rank, size);
    }

    if (rank == 0) {
        PP();
    }

    MPI_Finalize();

    return 0;
}
