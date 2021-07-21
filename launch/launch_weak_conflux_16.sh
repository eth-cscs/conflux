#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p16 
#SBATCH --time=02:40:00 
#SBATCH --nodes=8 
#SBATCH --output=data/benchmarks/psychol-weak-p16-11:52:11.363898.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 8 -n 16 ./build/examples/conflux_miniapp -N 4096  -r 5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp -N 8192  -r 5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp -N 16384 -r 5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp -N 32768 -r 5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp -N 65536 -r 5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp -N 13107 -r 5 