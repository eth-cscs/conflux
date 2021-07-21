#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p4 
#SBATCH --time=01:00:00 
#SBATCH --nodes=2 
#SBATCH --output=data/benchmarks/psychol-weak-p4-11:52:11.353877.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 2 -n 4 ./build/examples/cholesky_miniapp --dim=2048 --run=5 
srun -N 2 -n 4 ./build/examples/cholesky_miniapp --dim=4096 --run=5 
srun -N 2 -n 4 ./build/examples/cholesky_miniapp --dim=8192 --run=5 
srun -N 2 -n 4 ./build/examples/cholesky_miniapp --dim=16384 --run=5 
srun -N 2 -n 4 ./build/examples/cholesky_miniapp --dim=32768 --run=5 
srun -N 2 -n 4 ./build/examples/cholesky_miniapp --dim=65536 --run=5 
