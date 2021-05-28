#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p8 
#SBATCH --time=01:00:00 
#SBATCH --nodes=4 
#SBATCH --output=data/benchmarks/psychol-weak-p8-11:52:11.362568.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 
srun -N 4 -n 8 ./build/examples/conflux_miniapp -N 4096  -r 5 
srun -N 4 -n 8 ./build/examples/conflux_miniapp -N 8192  -r 5 
srun -N 4 -n 8 ./build/examples/conflux_miniapp -N 16384 -r 5 
srun -N 4 -n 8 ./build/examples/conflux_miniapp -N 32768 -r 5 
srun -N 4 -n 8 ./build/examples/conflux_miniapp -N 65536 -r 5 
