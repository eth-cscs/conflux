#include <experimental/memory>

namespace conflux {

enum order {
    row_major, col_major
};

enum pivoting {
    empty, partial, tournament
};

template <typename T>
class matrix_view {
public:
    // pointer which memory is managed outside (an observer pointer)
    std::experimental::observer_ptr<T> data = nullptr;

    int n_rows = 0;
    int n_cols = 0;
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
    // for all i<n_rows and j < n_cols
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

    matrix_view submatrix(int r1, int c1, int r2, int c2) {
        assert(r1 >= 0 && r1 <= n_rows);
        assert(r2 >= 0 && r2 <= n_rows);
        assert(r2 > r1);
        assert(c1 >= 0 && c1 <= n_cols);
        assert(c2 >= 0 && c2 <= n_cols);
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

    order layout() {
        assert(row_stride == 1 || col_stride == 1);
        if (row_stride == 1) {
            return order::col_major;
        }
        return order::row_major;
    }

    void transpose() {
        std::swap(n_rows, n_cols);
        std::swap(row_stride, col_stride);
    }

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
