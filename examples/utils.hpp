
/*
 * Everything in this file is taken from:
 * https://github.com/eth-cscs/COSMA
 *
 * This LICENCE of COSMA is BSD-3 Clause and here is a copy of it:
 *
 * BSD 3-Clause License

   Copyright (c) 2018, ETH Zürich.
   All rights reserved.

   Redistribution and use in source and binary forms, with or without modification,
   are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

   3. Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software without
      specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifdef __cplusplus
extern "C" {
#endif
    // Initialization
    void Cblacs_pinfo(int* mypnum, int* nprocs);
    void Cblacs_setup(int* mypnum, int* nprocs);
    void Cblacs_set(int ictxt, int what, int* val);
    void Cblacs_get(int ictxt, int what, int* val);
    void Cblacs_gridinit(int* ictxt, const char* order, int nprow, int npcol);
    void Cblacs_gridmap(int* ictxt, int* usermap, int ldup, int nprow, int npcol);

    // Finalization
    void Cblacs_freebuff(int ictxt, int wait);
    void Cblacs_gridexit(int ictxt);
    void Cblacs_exit(int NotDone);

    // Abort
    void Cblacs_abort(int ictxt, int errno);

    // Information
    void Cblacs_gridinfo(int ictxt, int* nprow, int* npcol, int* myrow, int* mycol);
    int Cblacs_pnum(int ictxt, int prow, int pcol);
    void Cblacs_pcoord(int ictxt, int nodenum, int* prow, int* pcol);

    // Barrier
    void Cblacs_barrier(int ictxt, char* scope);

    // MPI communicator <-> Blacs context
    MPI_Comm Cblacs2sys_handle(int ictxt);
    int Csys2blacs_handle(MPI_Comm mpi_comm);
    void Cfree_blacs_system_handle(int i_sys_ctxt);

    // matrix multiplication
    void descinit_(int* desc, const int* m, const int* n, const int* mb, const int* nb,
           const int* irsrc, const int* icsrc, const int* ictxt, const int* lld, int* info);

    void psgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
            const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
            float* c, const int* ic, const int* jc, const int* descc);

    void pdgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
            const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
            double* c, const int* ic, const int* jc, const int* descc);

    void pcgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
            const float* alpha, const float* a, const int* ia, const int* ja, const int* desca,
            const float* b, const int* ib, const int* jb, const int* descb, const float* beta,
            float* c, const int* ic, const int* jc, const int* descc);

    void pzgemm_(const char* trans_a, const char* trans_b, const int* m, const int* n, const int* k,
            const double* alpha, const double* a, const int* ia, const int* ja, const int* desca,
            const double* b, const int* ib, const int* jb, const int* descb, const double* beta,
            double* c, const int* ic, const int* jc, const int* descc);
#ifdef __cplusplus
}
#endif

namespace scalapack {
    int leading_dimension(const int* desc) {
        return desc[8];
    }

    // computes the number of rows or columns that the specified rank owns
    int numroc(int n, int nb, int proc_coord, int proc_src, int n_procs) {
        // Arguments:
        /*
           - n: global matrix dimension (rows or columns)
           - nb: corresponding dimension of a block
           - proc_coord: coordinate of the process for which we are querying
           - proc_src: process src
           - n_procs: total number of processes along this dimension
           */
        // number of whole blocks along this dimension
        int n_blocks = n / nb;

        // the offset of given process to the source process
        // make sure it stays positive
        int proc_offset = (n_procs + proc_coord - proc_src) % n_procs;

        // number of blocks per process (at least)
        // Can also be zero.
        int n_blocks_per_process = n_blocks/n_procs;
        // Number of rows or columns that each process has (at least).
        // Can also be zero.
        int n_rows_or_cols_per_process = n_blocks_per_process * nb;

        // each rank owns at least this base
        int n_rows_or_columns_total = n_rows_or_cols_per_process;

        // if there is a remainder, then the current
        // process might own some additional blocks
        int remainder = n_blocks % n_procs;

        // possible additional "whole" blocks that
        // the current rank owns
        n_rows_or_columns_total += proc_offset < remainder ? nb : 0;
        // possible additional "partial" blocks that
        // the current ranks owns
        n_rows_or_columns_total += proc_offset == remainder ? n % nb : 0;

        return n_rows_or_columns_total;
    }

    int local_buffer_size(const int* desc) {
        int lld = leading_dimension(desc);

        int n_cols = desc[3]; // global matrix size (columns)
        int nb_cols = desc[5]; // block size (columns)
        int src_proc = desc[7]; // processor src (columns)

        int ctxt = desc[1];

        int nprow, npcol, myrow, mycol;
        Cblacs_gridinfo(ctxt, &nprow, &npcol, &myrow, &mycol);

        int P = nprow * npcol;

        int n_local_cols = numroc(n_cols, nb_cols, mycol, src_proc, npcol);

        return lld * n_local_cols;
    }

    int min_leading_dimension(int n, int nb, int rank_grid_dim) {
        // Arguments:
        /*
           - n: global matrix dimension (rows or columns)
           - nb: corresponding dimension of a block
           - rank_grid_dim: total number of processes along this dimension
           */
        // number of blocks along this dimension
        int n_blocks = n / nb;

        // number of blocks per process (at least)
        // Can also be zero.
        int n_blocks_per_process = n_blocks/rank_grid_dim;
        // Number of rows or columns that each process has (at least).
        // Can also be zero.
        // each rank owns at least this many rows
        int min_n_rows_or_cols_per_process = n_blocks_per_process * nb;

        return min_n_rows_or_cols_per_process;
    }

    int max_leading_dimension(int n, int nb, int rank_grid_dim) {
        // Arguments:
        /*
           - n: global matrix dimension (rows or columns)
           - nb: corresponding dimension of a block
           - rank_grid_dim: total number of processes along this dimension
           */
        int lld = min_leading_dimension(n, nb, rank_grid_dim);
        int n_blocks = n / nb;
        int remainder = n_blocks % rank_grid_dim;
        lld += (remainder == 0) ? (n % nb) : nb;
        return lld;
    }

    // queries the grid blacs context to get the communication blacs context
    int get_comm_context(int grid_context) {
        int comm_context;
        int ten = 10;
        Cblacs_get(grid_context, ten, &comm_context);
        return comm_context;
    }

    // gets MPI_Comm from the grid blacs context
    MPI_Comm get_communicator(int grid_context) {
        int comm_context = get_comm_context(grid_context);
        MPI_Comm comm = Cblacs2sys_handle(comm_context);
        return comm;
    }
}

MPI_Comm subcommunicator(int new_P, MPI_Comm comm = MPI_COMM_WORLD) {
    // original size
    int P;
    MPI_Comm_size(comm, &P);

    // original group
    MPI_Group group;
    MPI_Comm_group(comm, &group);

    // new comm and new group
    MPI_Comm newcomm;
    MPI_Group newcomm_group;

    // ranks to exclude
    std::vector<int> exclude_ranks;
    for (int i = new_P; i < P; ++i) {
        exclude_ranks.push_back(i);
    }
    // create reduced group
    MPI_Group_excl(group, exclude_ranks.size(), exclude_ranks.data(), &newcomm_group);
    // create reduced communicator
    MPI_Comm_create_group(comm, newcomm_group, 0, &newcomm);

    MPI_Group_free(&group);
    MPI_Group_free(&newcomm_group);

    return newcomm;
}
