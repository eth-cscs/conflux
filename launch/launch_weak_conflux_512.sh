#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p512 
#SBATCH --time=06:00:00 
#SBATCH --nodes=256 
#SBATCH --output=data/benchmarks/psychol-weak-p512-11:52:11.366410.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 256 -n 512 ./build/examples/conflux_miniapp --dim=65536 --run=5 
srun -N 256 -n 512 ./build/examples/conflux_miniapp --dim=131072 --run=5 
srun -N 256 -n 512 ./build/examples/conflux_miniapp --dim=262144 --run=5 
srun -N 256 -n 512 ./build/examples/conflux_miniapp --dim=524288 --run=5 
