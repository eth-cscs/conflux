#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p64 
#SBATCH --time=01:10:00 
#SBATCH --nodes=32 
#SBATCH --output=data/benchmarks/psychol-weak-p64-11:52:11.357734.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 32 -n 64 ./build/examples/cholesky_miniapp --dim=8192 --run=5 
srun -N 32 -n 64 ./build/examples/cholesky_miniapp --dim=16384 --run=5 
srun -N 32 -n 64 ./build/examples/cholesky_miniapp --dim=32768 --run=5 
srun -N 32 -n 64 ./build/examples/cholesky_miniapp --dim=65536 --run=5 
srun -N 32 -n 64 ./build/examples/cholesky_miniapp --dim=131072 --run=5 
srun -N 32 -n 64 ./build/examples/cholesky_miniapp --dim=262144 --run=5 
