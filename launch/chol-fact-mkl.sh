#!/bin/bash -l
#SBATCH --job-name=cholesky-factorization-mkl-16k
#SBATCH --time=00:50:00
#SBATCH --nodes=8
#SBATCH --output=cholesky-factorization-mkl-16k.txt
#SBATCH --constraint=mc
#SBATCH --account=g34

export OMP_NUM_THREADS=18

srun -N 8 -n 16 ./build/examples/cholesky_scalapack --dim=16384 --tile=512 --grid=4x4 --run=0
