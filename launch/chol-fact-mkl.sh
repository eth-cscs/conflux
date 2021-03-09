#!/bin/bash -l
#SBATCH --job-name=cholesky-factorization-mkl-32k
#SBATCH --time=00:50:00
#SBATCH --nodes=32
#SBATCH --output=cholesky-factorization-mkl-32k.txt
#SBATCH --constraint=mc
#SBATCH --account=g34

export OMP_NUM_THREADS=18

srun -N 32 -n 64 ./build/examples/cholesky_scalapack --dim=32768 --tile=512 --grid=8x8 --run=0
