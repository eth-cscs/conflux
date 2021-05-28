#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p256 
#SBATCH --time=08:00:00 
#SBATCH --nodes=128 
#SBATCH --output=data/benchmarks/psychol-weak-p256-11:52:11.366360.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 128 -n 256 ./build/examples/conflux_miniapp -N 32768 -r 5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp -N 65536 -r 5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp -N 131072 -r 5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp -N 262144 -r 5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp -N 524288 -r 5 
