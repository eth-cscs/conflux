#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p1024 
#SBATCH --time=04:00:00 
#SBATCH --nodes=512 
#SBATCH --output=data/benchmarks/psychol-weak-p1024-11:52:11.366458.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 512 -n 1024 ./build/examples/conflux_miniapp -N 131072 -r 5 
srun -N 512 -n 1024 ./build/examples/conflux_miniapp -N 262144 -r 5 
srun -N 512 -n 1024 ./build/examples/conflux_miniapp -N 524288 -r 5 
