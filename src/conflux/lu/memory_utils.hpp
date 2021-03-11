#pragma once
#include <omp.h>
#include <algorithm>
#include <cassert>

namespace conflux {
template <typename T>
void mcopy(T *src, T *dst,
           int ssrow, int serow, int sscol, int secol, int sstride,
           int dsrow, int derow, int dscol, int decol, int dstride) {
    assert(serow - ssrow == derow - dsrow);
    assert(secol - sscol == decol - dscol);

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

template <typename T>
void parallel_mcopy(int n_rows, int n_cols,
                    T* in, int in_stride,
                    T* out, int out_stride) {
#pragma omp parallel for shared(in_stride, out_stride, n_rows, in, out)
    for (int i = 0; i < n_rows; ++i) {
        std::copy_n(&in[i * in_stride], n_cols, &out[i * out_stride]);
    }
}
}
