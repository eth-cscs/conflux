#include <experimental/memory>

namespace conflux {

// specifies the local matrix data layout
enum order {
    row_major, col_major
};

// specifies the pivoting strategy
enum pivoting {
    empty, partial, tournament
};

// matrix_view is a view on the data that has been allocated from the outside.
// this class offers a way to view/access the data in row-major or col-major order
// regardless of how the data is physically stored in the memory.
template <typename T>
class matrix_view {
public:
    // pointer which memory is managed outside (an observer pointer)
    std::experimental::observer_ptr<T> data = nullptr;

    // dimensions of the view
    int n_rows = 0;
    int n_cols = 0;
    // row- and col-strides
    // (one of these has to be 1)
    int row_stride = 1;
    int col_stride = 1;

    matrix_view() = default;
    matrix_view(T* data,
                int n_rows, int n_cols,
                int stride,
                order layout)
    : data(data)
    , n_rows(n_rows)
    , n_cols(n_cols)
    {
        if (layout == order::row_major) {
            row_stride = stride;
            assert(row_stride >= n_cols);
        } else {
            col_stride = stride;
            assert(col_stride >= n_rows);
        }
    }
    matrix_view(T* data,
                int n_rows, int n_cols,
                order layout):
        matrix_view(data, n_rows, n_cols,
                    (layout == order::row_major ? n_cols : n_rows),
                    layout)
    {}

    // returns the size of the actual data
    // (without strides)
    size_t size() {
        return n_rows * n_cols;
    }

    // returns the element with (row, col)-coordinates
    T& operator() (int row, int col) { 
        assert(data != nullptr);
        assert(row >= 0 && row < n_rows);
        assert(col >= 0 && col < n_cols);
        return data.get()[row * row_stride + col * col_stride];
    }

    T operator() (int row, int col) const { 
        assert(data != nullptr);
        assert(row >= 0 && row < n_rows);
        assert(col >= 0 && col < n_cols);
        return data.get()[row * row_stride + col * col_stride];
    }

    // 2 matrix views are equal if they have equal:
    // n_rows, n_cols and a(i, j) == b(i, j) 
    // for all i < n_rows and j < n_cols
    bool operator==(const matrix_view &other) const {
        if (n_rows != other.n_rows || n_cols != other.n_cols) {
            return false;
        }
        for (int i = 0; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                if (std::abs(operator()(i, j) != other(i, j)) > 1e-12) {
                    return false;
                }
            }
        }
        return true;
    }

    // returns a new matrix_view that contains the submatrix
    // in a rectangle [(r1, c1), (r2, c2)), i.e. containing the points:
    // (i, j) s.t. r1 <= i < r2 and c1 <= j < c2
    matrix_view submatrix(int r1, int c1, int r2, int c2) {
        // r1, r2, c1, c2 must be within n_rows and n_cols
        assert(r1 >= 0 && r1 <= n_rows);
        assert(r2 >= 0 && r2 <= n_rows);
        assert(c1 >= 0 && c1 <= n_cols);
        assert(c2 >= 0 && c2 <= n_cols);
        // cannot be empty
        assert(r2 > r1);
        assert(c2 > c1);

        // one of the strides has to be 1
        assert(row_stride == 1 || col_stride == 1);
        int new_stride = std::max(row_stride, col_stride);

        // starting from (r1, c1) element
        T* new_data = &((*this)(r1, c1));

        return matrix_view(new_data, 
                           r2 - r1,
                           c2 - c1,
                           new_stride,
                           layout());
    }

    // checks whether the layout is row- or col-major order
    // if n_rows == 1 or n_cols == 1, the layouts are equivalent
    order layout() {
        assert(row_stride == 1 || col_stride == 1);
        if (row_stride == 1) {
            return order::col_major;
        }
        return order::row_major;
    }

    // transposes the view: the actual data is not reshuffled
    // it's only accessed in a different (transposed) way
    void transpose() {
        std::swap(n_rows, n_cols);
        std::swap(row_stride, col_stride);
    }

    // the matrix output as:
    // (x stands for stride containing irrelevant data)
    // 1 2 3 x x
    // 4 5 6 x x
    // 7 8 9 x x
    std::string to_string() {
        std::string result = "";
        for (int i = 0 ; i < n_rows; ++i) {
            for (int j = 0; j < n_cols; ++j) {
                result += std::to_string(operator()(i, j)) + ", ";
            }
            // fill the rest with x up to row_stride
            for (int j = n_cols; j < row_stride; ++j) {
                result += "x, ";
            }
            result += "\n";
        }
            // fill the rest with x up to col_stride
        for (int i = n_rows; i < col_stride; ++i) {
            for (int j = 0; j < std::max(n_cols, row_stride); ++j) {
                result += "x, ";
            }
            result += "\n";
        }
        return result;
    }
};
}
