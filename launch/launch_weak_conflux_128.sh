#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p128 
#SBATCH --time=03:00:00 
#SBATCH --nodes=64 
#SBATCH --output=data/benchmarks/psychol-weak-p128-11:52:11.366309.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 64 -n 128 ./build/examples/conflux_miniapp --dim=16384 --run=5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp --dim=32768 --run=5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp --dim=65536 --run=5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp --dim=131072 --run=5 
srun -N 64 -n 128 ./build/examples/conflux_miniapp --dim=262144 --run=5 
