#!/bin/bash -l 
#SBATCH --job-name=psychol-weak-p512 
#SBATCH --time=06:00:00 
#SBATCH --nodes=256 
#SBATCH --output=data/benchmarks/psychol-weak-p512-11:52:11.366410.txt 
#SBATCH --constraint=mc 
#SBATCH --account=g34 

export OMP_NUM_THREADS=18 

srun -N 256 -n 512 ./build/examples/conflux_miniapp -N 65536 -r 5 
srun -N 256 -n 512 ./build/examples/conflux_miniapp -N 131072 -r 5 
srun -N 256 -n 512 ./build/examples/conflux_miniapp -N 262144 -r 5 
srun -N 256 -n 512 ./build/examples/conflux_miniapp -N 524288 -r 5 
