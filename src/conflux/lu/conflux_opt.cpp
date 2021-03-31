#include <conflux/lu/conflux_opt.hpp>

MPI_Comm conflux::create_comm(MPI_Comm &comm, std::vector<int> &ranks) {
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

int conflux::l2g(int pi, int ind, int sqrtp1) {
    return ind * sqrtp1 + pi;
}

void conflux::g2l(int gind, int sqrtp1,
         int &out1, int &out2) {
    out1 = gind % sqrtp1;
    out2 = (int)(gind / sqrtp1);
}

std::tuple<int, int, int> conflux::p2X(MPI_Comm comm3D, int rank) {
    int coords[] = {-1, -1, -1};
    MPI_Cart_coords(comm3D, rank, 3, coords);
    return {coords[0], coords[1], coords[2]};
}

std::tuple<int, int> conflux::p2X_2d(MPI_Comm comm2D, int rank) {
    int coords[] = {-1, -1};
    MPI_Cart_coords(comm2D, rank, 2, coords);
    return {coords[0], coords[1]};
}

int conflux::X2p(MPI_Comm comm3D, int pi, int pj, int pk) {
    int coords[] = {pi, pj, pk};
    int rank;
    MPI_Cart_rank(comm3D, coords, &rank);
    return rank;
}

int conflux::X2p(MPI_Comm comm2D, int pi, int pj) {
    int coords[] = {pi, pj};
    int rank;
    MPI_Cart_rank(comm2D, coords, &rank);
    return rank;
}

int conflux::flipbit(int n, int k) {
    return n ^ (1ll << k);
}

int conflux::butterfly_pair(int pi, int r, int Px) {
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

std::pair<
    std::unordered_map<int, std::vector<int>>,
    std::unordered_map<int, std::vector<int>>>
conflux::g2lnoTile(std::vector<int> &grows, int Px, int v) {
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

void conflux::analyze_pivots(int first_non_pivot_row,
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

template <>
void conflux::print_matrix<double>(double *pointer,
                          int row_start, int row_end,
                          int col_start, int col_end,
                          int stride, char order) {
    if (order == 'R') {
        for (int i = row_start; i < row_end; ++i) {
            printf("[%2u:] ", i);
            for (int j = col_start; j < col_end; ++j) {
                printf("%8.3f", pointer[i * stride + j]);
                // std::cout << pointer[i * stride + j] << ", \t";
            }
            std::cout << std::endl;
        }
    } else {
        for (int i = row_start; i < row_end; ++i) {
            printf("[%2u:] ", i);
            for (int j = col_start; j < col_end; ++j) {
                printf("%8.3f", pointer[j * stride + i]);
                // std::cout << pointer[i * stride + j] << ", \t";
            }
            std::cout << std::endl;
        }
    }
}
