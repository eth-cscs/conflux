## CONFLUX
Communication-Optimal LU-factorization Algorithm

## Building CONFLUX

The library can be built by doing the following:
```bash
###############
# get CONFLUX
###############
git clone --recursive https://github.com/eth-cscs/conflux && cd conflux

##############################
# build and install CONFLUX
##############################
mkdir build && cd build

# set up the compiler, e.g. with:
export CC=`which cc`
export CXX=`which CC`

# build the library with a chosen blas backend
cmake -DCONFLUX_BLAS=MKL ..
make -j 8
```
Other available blas backends include: `CRAY_LIBSCI, OPENBLAS, CUSTOM`.

## Profiling CONFLUX

In order to profile CONFLUX, the `cmake` should be run with the following option:
```bash
cmake -DCONFLUX_BLAS=MKL -DCONFLUX_WITH_PROFILING=ON ..
make -j 8
```
