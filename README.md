## CONFLUX and CONFCHOX

Communication-Optimal LU-factorization (CONFLUX) and Cholesky Factorization (CONFCHOX) Algorithms. The repo is named CONFLUX, but it still includes both of these algorithms.

## Building the Libraries

The libraries can be built by doing the following:
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
> !! Note the --recursive flag !!
> 
The available blas backends include: `MKL, CRAY_LIBSCI, OPENBLAS, CUSTOM`. The scalapack backend is optional and can be set to `OFF, MKL, CRAY_LIBSCI, CUSTOM`.

## The Dependencies

This is a CMake project and requires a recent CMake(>=3.12).

External dependencies: 

- `MPI 3`: (required)
- `BLAS`: the blas library for local CPU computations, which can be one of the following:
     - `MKL` (default): BLAS API as provided by Intel MPI. Requires the environment variable `MKL_ROOT` to be set to the MKL's root directory.
     - `OPENBLAS`: BLAS API as provided by OPENBLAS. Requires the environment variable `OPENBLAS_ROOT` to point to the openblas installation.
     - `CRAY_LIBSCI`: `Cray-libsci` or `Cray-libsci_acc` (GPU-accelerated). 
     - `CUSTOM`: user-provided BLAS API. 
- `SCALAPACK` (optional, for scalapack wrappers): the scalapack library, which can be one of the following:
     - `MKL` (default)
     - `CRAY_LIBSCI`: `Cray-libsci` or `Cray-libsci_acc` (GPU-accelerated)
     - `CUSTOM`: user-provided BLAS API. Requires the variable `SCALAPACK_ROOT` to point to the scalapack installation.
     - `OFF`: turned off.

Some dependencies are bundled as submodules and need not be installed explicitly:
- `COSTA` - distributed matrix reshuffle and transpose algorithm.
- `semiprof`(optional) - profiling utlility.
- `gtest_mpi` - MPI utlility wrapper over GoogleTest (unit testing library)

## Building on Cray Systems

There are already prepared scripts for loading the necessary dependencies on Cray-Systems:

Cray XC40 (CPU-only version): source ./scripts/piz_daint_cpu.sh loads MKL and other neccessary modules.

## Running Cholesky Factorization

The cholesky factorization miniapp can be run as follows:
```
export OMP_NUM_THREADS=18 # set number of omp threads (optimally 18 on our Cray XC40 system)
srun -N 8 -n 16 ./build/examples/cholesky_miniapp --dim=2048 --run=5
```
where `dim` is the matrix dimension and `run` the number of repetitions (excluding a mandatory warm up round). `N` and `n` describe the number of nodes and the number of ranks to run the program with, respectively. You can also specify the grid you want to use by specifying an optional parameter `--grid=<Px,Py,Pz>` where `Px,Py,Pz` are the number of processors in the `x,y,z` direction, respectively. Another optional parameter is `--tile=<tile_size>` with which you can specify the tile size. These two optimal parameters provide optimal defaults but sometimes some manual fine tuning is needed for maximal performance.

## Running LU Factorization

The LU factorization miniapp can be run as follows:
```
export OMP_NUM_THREADS=18 # set number of omp threads (optimally 18 on our Cray XC40 system)
srun -N 8 -n 16 ./build/examples/conflux_miniapp -N 2048 -r 5
```
where the second `N` (=2048) is the matrix dimension and `r` is the number of repetitions (excluding a mandatory warm up round). `N` and `n` in the `srun` command describe the number of nodes and the total number of ranks to run the program with, respectively. You can also specify the grid you want to use by specifying an optional parameter `--p_grid=<Px,Py,Pz>` where `Px,Py,Pz` are the number of processors in the `x,y,z` direction, respectively. Another optional parameter is `-b=<tile_size>` specifying the tile size. The default parameters are chosen to yield the optimal performance in most configuration. However, in some configurations, manual tuning might yield more performance.

# Reproducing the Benchmarks

All the necessary launch scripts can be found in the folder `launch`. After building and compiling the project as described above, we used Python to produce the necessary `sbatch` scripts for launching the benchmarks. Now, you can run `python3 scripts/launch_on_daint.py` from the root folder which will launch all the benchmarking experiments.

If you want to run a specific experiment, you can use `sbatch <path_to_launchscript>` from the root folder.

## Estimated Time Needed for the Measurements

It is hard to estimate accurately how long experiments take, as it is highly dependant on the platform. However, on Piz Daint Supercomputer (Cray XC40), the experiment that ran the longest (excluding the queueing time) was `launch_weak_conflux_256` which took roughly 3.5 hours.

## Benchmark Outputs

The output files containing the timeouts can be found in `./data/benchmarks/`. Each algorithm (*confchox, conflux*) has one file for each number of ranks tested with the algorithm and the number of ranks displayed in the file name. All experiment outputs corresponding to this particular algorithm and number of ranks can be found in this file.

## Experiment Overview

The following table displays all combinations of matrix dimensions (all square matrices) and number of ranks that we ran experiments on. In particular, we give the interval of matrix sizes, meaning that we benchmarked all power of 2's within this interval including the boundaries:

| Number of Ranks | Matrix size interval 
--- | ---
4 | [2048, 65536]
8 | [4096, 65536]
16 | [4096, 131072]
32 | [8192, 131072]
64 | [8192, 262144]
128 | [16384, 262144]
256 | [32768, 524288]
512 | [65536, 524288]
1024 | [131072, 524288]

## Generating your own launchfiles

In order to generate your own run files, you need to edit `scripts/params_weak.ini`. Check the instructions within this file on how to fill it out. After having filled the file with parameters, you can generate run scripts with `python3 scripts/generate_launch_files_weak.py`. The scripts can be found in the `launch` folder and run as described above. For large jobs, you might have to adapt the time parameter in the launch file.

## Profiling the Library

In order to profile the code, the `cmake` should be run with the following option:
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
