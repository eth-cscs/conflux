#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p64 
#SBATCH --time=02:20:00 
#SBATCH --nodes=32 
#SBATCH --output=data/benchmarks/psychol-weak-p64-11:52:11.366248.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 32 -n 64 ./build/examples/conflux_miniapp -N 8192 -r 5 
srun -N 32 -n 64 ./build/examples/conflux_miniapp -N 16384 -r 5 
srun -N 32 -n 64 ./build/examples/conflux_miniapp -N 32768 -r 5 
srun -N 32 -n 64 ./build/examples/conflux_miniapp -N 65536 -r 5 
srun -N 32 -n 64 ./build/examples/conflux_miniapp -N 131072 -r 5 
srun -N 32 -n 64 ./build/examples/conflux_miniapp -N 262144 -r 5 
