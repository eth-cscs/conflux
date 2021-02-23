template <typename T>
class matrix {
public:
    char order = 'R';
    int n_rows = 0;
    int n_cols = 0;

    matrix() = default;
    matrix(char order, int n_rows, int n_cols)
        : order(order)
        , n_rows(n_rows)
        , n_cols(n_cols)
    {
        data.resize()
    }

private:
    std::vector<T> data;
}
