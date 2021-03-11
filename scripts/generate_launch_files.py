# This files generates bash scripts that launch jobs on piz daint according to params.ini format.

import os
import configparser
import ast
import sys
import numpy as np
import csv
import struct
from numpy import genfromtxt
import configparser
import math

path_to_launch = './launch/'
path_to_params = './scripts/params.ini'
cholesky_section = 'Cholesky25d'
scalapack_section = 'Scalapack'

def createBashPreface(P, algorithm):
    numNodes = math.ceil(P/2)
    return '#!/bin/bash -l \n\
#SBATCH --job-name=cholesky-matrix-run-%d-%s \n\
#SBATCH --time=00:10:00 \n\
#SBATCH --nodes=%d \n\
#SBATCH --output=cholesky-matrix-run-%d-%s \n\
#SBATCH --constraint=mc \n\
#SBATCH --account=g34 \n\n\
export OMP_NUM_THREADS=18 \n\n' % (P, algorithm, numNodes, P, algorithm)

# parse params.ini
def readConfig(section):
    config = configparser.ConfigParser()
    config.read(path_to_params)
    if not config.has_section(section):
        print("Please add a %s section", (section))
        raise Exception()
    try:
        N = ast.literal_eval(config[section]['N'])
    except:
        print("Please add at least one matrix size N=[] (%s)" %(section))
        raise
    try:
        v = ast.literal_eval(config[section]['V'])
    except:
        print("Please add at least one tile size V=[] (%s)" %(section))
        raise
    
    grids = dict()
    try:
        read_grids = ast.literal_eval(config[section]['grids'])
    except:
        print("Please add at least one grid grids=[[P, grid]] (%s)" %(section))
        raise

    # we separate it here for later file separation
    for g in read_grids:
        P = g[0]
        grid = g[1:]
        grids[P] = grid

    try:
        reps = ast.literal_eval(config[cholesky_section]['reps'])
    except:
        print("No number of repetitions found, using default 5. If you do not want this, add r= and the number of reps")
        reps = 5

    if len(N) ==0 or len(v) == 0 or len(grids) == 0:
        print("One of the arrays in params.ini is empty, please add values")
        raise Exception()
    
    return N, v, grids, reps
    


def generateLaunchFile(N, V, grids, reps, algorithm):
    for grid in grids:
        filename = path_to_launch + 'launch_%s_%d.sh' %(algorithm, grid)
        with open(filename, 'w') as f:
            numNodes = math.ceil(grid/2)
            f.write(createBashPreface(grid, algorithm))
            # next we iterate over all possibilities and write the bash script
            for cubes in grids[grid]:
                for n in N:
                    for v in V:
                        cmd = 'srun -N %d -n %d ./build/examples/cholesky_miniapp --dim=%d --tile=%d --grid=%s --run=%d \n' % (numNodes, grid, n, v, cubes, reps)
                        f.write(cmd)
    return

# We use the convention that we ALWAYS use n nodes and 2n ranks
# We might want to change that in future use
if __name__ == "__main__":

    # grids is a dict since for each processor size, we have to create a new launch file
    try:
        Ns, V, grids, reps = readConfig(cholesky_section)
        generateLaunchFile(Ns, V, grids, reps, 'psychol')
    except:
        pass
    
    try:
        Ns, V, grids, reps = readConfig(scalapack_section)
        generateLaunchFile(Ns, V, grids, reps, 'scalapack')
    except:
        pass
    

    

