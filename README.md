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

# build the library
cmake ..
make -j 8
```

## Profiling CONFLUX

In order to profile CONFLUX, the `cmake` should be run with the following option:
```bash
cmake -DCONFLUX_WITH_PROFILING=ON ..
make -j 8
```
