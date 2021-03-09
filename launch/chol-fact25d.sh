#!/bin/bash -l
#SBATCH --job-name=cholesky-factorization-65k
#SBATCH --time=00:50:00
#SBATCH --nodes=32
#SBATCH --output=cholesky-factorization-65k.txt
#SBATCH --constraint=mc
#SBATCH --account=g34

export OMP_NUM_THREADS=18

srun -N 32 -n 64 ./build/examples/cholesky_miniapp --dim=65536 --tile=512 --grid=4x4x4 --run=0
