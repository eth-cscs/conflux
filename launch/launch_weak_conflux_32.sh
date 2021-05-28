#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p32 
#SBATCH --time=02:40:00 
#SBATCH --nodes=16 
#SBATCH --output=data/benchmarks/psychol-weak-p32-11:52:11.365461.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 16 -n 32 ./build/examples/conflux_miniapp --dim=8192 --run=5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp --dim=16384 --run=5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp --dim=32768 --run=5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp --dim=65536 --run=5 
srun -N 16 -n 32 ./build/examples/conflux_miniapp --dim=131072 --run=5 
