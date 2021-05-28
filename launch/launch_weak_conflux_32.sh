#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p32 
#SBATCH --time=02:40:00 
#SBATCH --nodes=16 
#SBATCH --output=data/benchmarks/psychol-weak-p32-11:52:11.365461.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 16 -n 32 ./build/examples/conflux_miniapp -N 8192 -r 5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp -N 16384 -r 5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp -N 32768 -r 5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp -N 65536 -r 5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp -N 131072 -r 5 
