#include <conflux/lu/layout.hpp>
#include <complex>
#include <string>

// helper functions
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

std::vector<int> line_split(int N, int v) {
    std::vector<int> splits;
    splits.reserve(N/v + 1);
    for (int i = 0; i < N/v; ++i) {
        splits.push_back(i * v);
    }
    splits.push_back(N);
    return splits;
}

template <typename T>
costa::grid_layout<T> conflux::conflux_layout(T* data,
                                     int M, int N, int v, 
                                     char ordering,
                                     int Px, int Py,
                                     int rank) {
    ordering = std::toupper(ordering);
    assert(ordering == 'R' || ordering == 'C');
    // blacs grid on comm is row-major
    int pi = rank / Py;
    int pj = rank % Py;

    int Nt = (int)(std::ceil((double)N / v));
    int Mt = (int)(std::ceil((double)M / v));
    int t = (int)(std::ceil((double)Nt / Py)) + 1ll;
    int tA11x = (int)(std::ceil((double)Mt / Px));
    int tA11y = (int)(std::ceil((double)Nt / Py));
    int Ml = tA11x * v;
    int Nl = tA11y * v;

    // local blocks
    std::vector<costa::block_t> local_blocks;
    int n_local_blocks = tA11x * tA11y;
    local_blocks.reserve(n_local_blocks);

    for (int lti = 0; lti < tA11x; ++lti) {
        auto gti = lti * Px + pi;
        for (int ltj = 0; ltj < tA11y; ++ltj) {
            auto gtj = ltj * Py + pj;
            costa::block_t block;
            // pointer to the data of this tile
            block.data = &data[lti * v * Nl + ltj * v];
            // leading dimension
            block.ld = ordering=='R' ? Nl : Ml;
            // global coordinates of this block
            block.row = gti;
            block.col = gtj;
            local_blocks.push_back(block);
        }
    }

    std::vector<int> row_splits = line_split(M, v);
    std::vector<int> col_splits = line_split(N, v);

    std::vector<int> owners(Mt*Nt);

    for (int i = 0; i < Mt; ++i) {
        for (int j = 0; j < Nt; ++j) {
            int ij = i * Nt + j;
            int pi = i % Px;
            int pj = j % Py;
            // row-major ordering in p-grid assumed
            owners[ij] = pi * Py + pj; // X2p(comm, pi, pj);
        }
    }

    auto matrix = costa::custom_layout<T>(
            Mt, Nt, // num of global blocks
            &row_splits[0], &col_splits[0], // splits in the global matrix
            &owners[0], // ranks owning each tile
            n_local_blocks, // num of local blocks
            &local_blocks[0], // local blocks
            ordering // row-major ordering within blocks
            );

    return matrix;
}

template <typename T>
costa::grid_layout<T> conflux::conflux_layout(T* data,
                                     int M, int N, int v, 
                                     char ordering,
                                     MPI_Comm lu_comm) {
    ordering = std::toupper(ordering);
    assert(ordering == 'R' || ordering == 'C');
    int dims[3];
    int periods[3];
    int coords[3];
    MPI_Cart_get(lu_comm, 3, dims, periods, coords);
    int rank;
    MPI_Comm_rank(lu_comm, &rank);
    int pi = coords[0];
    int pj = coords[1];
    int pk = coords[2];
    int Px = dims[0];
    int Py = dims[1];

    int Nt = (int)(std::ceil((double)N / v));
    int Mt = (int)(std::ceil((double)M / v));
    int t = (int)(std::ceil((double)Nt / Py)) + 1ll;
    int tA11x = (int)(std::ceil((double)Mt / Px));
    int tA11y = (int)(std::ceil((double)Nt / Py));
    int Ml = tA11x * v;
    int Nl = tA11y * v;

    // local blocks
    std::vector<costa::block_t> local_blocks;
    int n_local_blocks = tA11x * tA11y;
    local_blocks.reserve(n_local_blocks);

    for (int lti = 0; lti < tA11x; ++lti) {
        auto gti = lti * Px + pi;
        for (int ltj = 0; ltj < tA11y; ++ltj) {
            auto gtj = ltj * Py + pj;
            costa::block_t block;
            // pointer to the data of this tile
            block.data = &data[lti * v * Nl + ltj * v];
            // leading dimension
            block.ld = ordering=='R' ? Nl : Ml;
            // global coordinates of this block
            block.row = gti;
            block.col = gtj;
            local_blocks.push_back(block);
        }
    }

    std::vector<int> row_splits = line_split(M, v);
    std::vector<int> col_splits = line_split(N, v);

    std::vector<int> owners(Mt*Nt);

    for (int i = 0; i < Mt; ++i) {
        for (int j = 0; j < Nt; ++j) {
            int ij = i * Nt + j;
            int pi = i % Px;
            int pj = j % Py;
            owners[ij] = X2p(lu_comm, pi, pj, 0);
        }
    }

    auto matrix = costa::custom_layout<T>(
            Mt, Nt, // num of global blocks
            &row_splits[0], &col_splits[0], // splits in the global matrix
            &owners[0], // ranks owning each tile
            n_local_blocks, // num of local blocks
            &local_blocks[0], // local blocks
            ordering // row-major ordering within blocks
            );

    return matrix;
}

// template instantiation for conflux_layout
template 
costa::grid_layout<double> conflux::conflux_layout(
            double* data,
            int M, int N, int v, 
            char ordering,
            MPI_Comm lu_comm);
template 
costa::grid_layout<float> conflux::conflux_layout(
            float* data,
            int M, int N, int v, 
            char ordering,
            MPI_Comm lu_comm);
template 
costa::grid_layout<std::complex<float>> conflux::conflux_layout(
            std::complex<float>* data,
            int M, int N, int v, 
            char ordering,
            MPI_Comm lu_comm);
template 
costa::grid_layout<std::complex<double>> conflux::conflux_layout(
            std::complex<double>* data,
            int M, int N, int v, 
            char ordering,
            MPI_Comm lu_comm);

// template instantiation for conflux_layout
template 
costa::grid_layout<double> conflux::conflux_layout(
            double* data,
            int M, int N, int v, 
            char ordering,
            int Px, int Py,
            int rank);
template 
costa::grid_layout<float> conflux::conflux_layout(
            float* data,
            int M, int N, int v, 
            char ordering,
            int Px, int Py,
            int rank);
template 
costa::grid_layout<std::complex<float>> conflux::conflux_layout(
            std::complex<float>* data,
            int M, int N, int v, 
            char ordering,
            int Px, int Py,
            int rank);
template 
costa::grid_layout<std::complex<double>> conflux::conflux_layout(
            std::complex<double>* data,
            int M, int N, int v, 
            char ordering,
            int Px, int Py,
            int rank);
