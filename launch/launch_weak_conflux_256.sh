#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p256 
#SBATCH --time=08:00:00 
#SBATCH --nodes=128 
#SBATCH --output=data/benchmarks/psychol-weak-p256-11:52:11.366360.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 128 -n 256 ./build/examples/conflux_miniapp --dim=32768 --run=5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp --dim=65536 --run=5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp --dim=131072 --run=5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp --dim=262144 --run=5 
srun -N 128 -n 256 ./build/examples/conflux_miniapp --dim=524288 --run=5 
