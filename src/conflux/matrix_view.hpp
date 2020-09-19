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

    void transpose() {
        std::swap(n_rows, n_cols);
        std::swap(row_stride, col_stride);
    }
};
}
