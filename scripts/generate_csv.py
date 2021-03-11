# This file generates csv's from benchmarks in 

import os
import configparser
import ast
import sys
import numpy as np
import csv
import struct
from numpy import genfromtxt

path_to_launch = './launch/'
path_to_params = './scripts/params.ini'
cholesky_section = 'Cholesky25d'
scalapack_section = 'Scalapack'

def parseBenchmark(filename):
    durations = []
    communications = []
    
    # Check if file is empty
    if os.stat(filename).st_size == 0:
        print(f"{filename} is empty")
        return -1

    with open(filename, "rb") as f:
        buf = f.read(28)
        while buf != b"":
            measurement = struct.unpack('=iddQ', buf)
            
            # Get duration
            begin = measurement[1]
            end = measurement[2]
            duration = end - begin
            durations.append(duration)
            
            # Get comunicated/sent bytes
            communications.append(measurement[3])

            buf = f.read(28)
    
    communication = np.sum(communications)
    durations.append(communication)

    # np.mean gives the arithmetic mean
    return durations # last entry is sumemd up communication
#
# For a given configuration get the measurement for 
def saveToCsv(N, V, grids,reps, scalaPack=False):
    # Contains durations for all repetitions

    for n in N: # Input size
        for v in V: # Tile size
            for P in grids: # Grid layouts
                for grid in grids[P]:
                    if not scalaPack:
                        csvFile = f"data/benchmarks/cholesky25d/output/csv/benchmark_{n}_{v}_{grid}.csv"
                    else:
                        csvFile = f"data/benchmarks/scalapack/output/csv/benchmark_{n}_{v}_{grid}x0.csv"

                    with open(csvFile, 'w') as f:
                        write = csv.writer(f)
                        for i in range(reps): # 10 Repetitions
                            if not scalaPack:
                                benchmarkFile = f"data/benchmarks/cholesky25d/output/benchmark-{i}_{n}_{v}_{grid}.bin"
                            else: 
                                benchmarkFile = f"data/benchmarks/scalapack/output/benchmark-{i}_{n}_{v}_{grid}x0.bin"
                            data = parseBenchmark(benchmarkFile)
                            if data == -1:
                                f.flush()
                                print(f"{benchmarkFile} was empty")
                                continue

                            write.writerow(data)


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

# benchmark_<run>_<input-size>_<tile-size>_<grid>.bin
if __name__ == "__main__":

    # Cholesky2.5d
    # grids is a dict since for each processor size, we have to create a new launch file
    try:
        Ns, V, grids, reps = readConfig(cholesky_section)
        saveToCsv(Ns, V, grids, reps)
    except:
        pass
    
    
    # Scalapack
    try:
        Ns, V, grids, reps = readConfig(scalapack_section)
    except:
        pass





