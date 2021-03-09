#!/bin/bash -l
#SBATCH --job-name=cholesky-factorization-16k
#SBATCH --time=00:10:00
#SBATCH --nodes=18
#SBATCH --output=cholesky-factorization-16k.txt
#SBATCH --constraint=mc
#SBATCH --account=g34

export OMP_NUM_THREADS=18

srun -N 18 -n 36 ./build/examples/cholesky_miniapp --dim=16384 --tile=512 --grid=3x3x4 --run=0
