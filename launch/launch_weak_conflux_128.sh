#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p128 
#SBATCH --time=03:00:00 
#SBATCH --nodes=64 
#SBATCH --output=data/benchmarks/psychol-weak-p128-11:52:11.366309.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 64 -n 128 ./build/examples/conflux_miniapp -N 16384 -r 5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp -N 32768 -r 5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp -N 65536 -r 5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp -N 131072 -r 5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp -N 262144 -r 5 
