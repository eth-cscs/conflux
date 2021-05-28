#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p16 
#SBATCH --time=02:40:00 
#SBATCH --nodes=8 
#SBATCH --output=data/benchmarks/psychol-weak-p16-11:52:11.363898.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 8 -n 16 ./build/examples/conflux_miniapp --dim=4096 --run=5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp --dim=8192 --run=5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp --dim=16384 --run=5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp --dim=32768 --run=5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp --dim=65536 --run=5 
srun -N 8 -n 16 ./build/examples/conflux_miniapp --dim=131072 --run=5 
