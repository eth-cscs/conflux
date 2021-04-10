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
cmake -DCONFLUX_BLAS=MKL -DCONFLUX_SCALAPACK=MKL ..
make -j 8
```
Other available blas backends include: `CRAY_LIBSCI, OPENBLAS, CUSTOM`.

### On Piz Daint supercomputer:

First run:
```bash
source ./scripts/piz_daint_cpu.sh
```
to load all the modules and then run `cmake` and `make` commands as shown above.

**Running cholesky**
Run cholesky on Piz Daint with the following command:
```
export OMP_NUM_THREADS=18 # set number of omp threads (optimally 18 on daint)
srun -N 8 -n 16 ./build/examples/cholesky_miniapp --dim=2048 --run=5
```
where *dim* is the matrix dimension and *run* the number of repetitions (excluding a mandatory warm up round). *N* and *n* describe the number of nodes and the number of ranks to run the program with, respectively. You can also specify the grid you want to use by specifying an optional parameter *grid=<x,y,z>* where x,y,z are the number of processors in x,y,z direction, respectively. Another optional parameter is *tile=<tile_size>* with which you can specify the tile size. These two optimal parameters provide optimal defaults but sometimes some manual fine tuning is needed for maximal performance.
## Profiling CONFLUX

In order to profile CONFLUX, the `cmake` should be run with the following option:
```bash
cmake -DCONFLUX_BLAS=MKL -DCONFLUX_SCALAPACK=MKL -DCONFLUX_WITH_PROFILING=ON ..
make -j 8
```
The profiler outputs the regions sorted by duration, e.g. after locally running:
```
mpirun -np 8 ./examples/conflux_miniapp -M 16 -N 16 -b 2
```
The output might looks something like:
```
_p_ REGION                     CALLS      THREAD        WALL       %
_p_ total                          -       0.130       0.130   100.0
_p_   step3                        -       0.054       0.054    41.6
_p_     put                        8       0.054       0.054    41.6
_p_   fence                        -       0.026       0.026    19.8
_p_     create                     1       0.015       0.015    11.8
_p_     destroy                    1       0.010       0.010     8.0
_p_   step5                        -       0.019       0.019    14.5
_p_     waitall                    8       0.019       0.019    14.5
_p_     dtrsm                      4       0.000       0.000     0.0
_p_     isend                     16       0.000       0.000     0.0
_p_     localcopy                  8       0.000       0.000     0.0
_p_     reshuffling               20       0.000       0.000     0.0
_p_     irecv                      8       0.000       0.000     0.0
_p_   step1                        -       0.015       0.015    11.6
_p_     curPivots                  8       0.006       0.006     4.5
_p_     barrier                    8       0.006       0.006     4.3
_p_     pivoting                   4       0.002       0.002     1.8
_p_     A00Buff                    -       0.001       0.001     0.8
_p_       bcast                    8       0.001       0.001     0.8
_p_       isend                    4       0.000       0.000     0.0
_p_       irecv                    8       0.000       0.000     0.0
_p_       waitall                  8       0.000       0.000     0.0
_p_     rowpermute                 4       0.000       0.000     0.2
_p_     lup                        4       0.000       0.000     0.0
_p_     A10copy                    4       0.000       0.000     0.0
_p_   step2                        -       0.014       0.014    11.0
_p_     reduce                     8       0.011       0.011     8.4
_p_     pushingpivots              8       0.003       0.003     2.7
_p_     localcopy                  8       0.000       0.000     0.0
_p_   step0                        -       0.001       0.001     1.0
_p_     reduce                     4       0.001       0.001     0.9
_p_     copy                       4       0.000       0.000     0.0
_p_   step4                        -       0.000       0.000     0.3
_p_     reshuffling                4       0.000       0.000     0.2
_p_     dtrsm                      4       0.000       0.000     0.0
_p_     comm                      12       0.000       0.000     0.0
_p_   storingresults               8       0.000       0.000     0.1
_p_   step6                        -       0.000       0.000     0.0
_p_     dgemm                      8       0.000       0.000     0.0
_p_   init                         1       0.000       0.000     0.0
_p_     A11copy                    1       0.000       0.000     0.0
```
