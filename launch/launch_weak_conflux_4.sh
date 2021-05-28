#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p4 
#SBATCH --time=01:00:00 
#SBATCH --nodes=2 
#SBATCH --output=data/benchmarks/psychol-weak-p4-11:52:11.361220.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 2 -n 4 ./build/examples/conflux_miniapp -N 2048 -r 5 
srun -N 2 -n 4 ./build/examples/conflux_miniapp -N 4096 -r 5 
srun -N 2 -n 4 ./build/examples/conflux_miniapp -N 8192 -r 5 
srun -N 2 -n 4 ./build/examples/conflux_miniapp -N 16384 -r 5 
srun -N 2 -n 4 ./build/examples/conflux_miniapp -N 32768 -r 5 
srun -N 2 -n 4 ./build/examples/conflux_miniapp -N 65536 -r 5 
