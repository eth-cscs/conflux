import os
import configparser
import ast
import sys
import numpy as np
import csv
import struct
from numpy import genfromtxt

def getGridLayout(gridString):
    grid = gridString.split("x")

    grid[0] = int(grid[0])
    grid[1] = int(grid[1])

    return grid[0], grid[1]

#
# For a given configuration get the measurement for 
def checkCsvExists(N, V, grids,reps, scalaPack=False):
    # Contains durations for all repetitions
    for n in N: # Input size
        for v in V: # Tile size
            for grid in grids: # Grid layouts
                if not scalaPack:
                    csvFile = f"data/benchmarks/cholesky25d/output/csv/benchmark_{n}_{v}_{grid[1]}.csv"
                else:
                    csvFile = f"data/benchmarks/scalapack/output/csv/benchmark_{n}_{v}_{grid[1]}x0.csv"
                
                exists = os.path.exists(csvFile)
                if not exists:
                    print(f"N={n}, v={v}, grid={grid[1]}")
                    continue
                else:
                    measurements = genfromtxt(csvFile, delimiter=',')
                    if (len(measurements) < 10):
                        print(f"N={n}, v={v}, grid={grid[1]} => only {len(measurements)} reps")
                    if(measurements.ndim == 1):
                        print(f"N={n}, v={v}, grid={grid[1]} => only 1 reps")
def readConfig(filename, scalaPack=False):
    config = configparser.ConfigParser()
    config.read(filename)

    if not scalaPack:
        section = 'Cholesky25d'
    else:
        section = 'Scalapack'

    N = ast.literal_eval(config[section]['N'])
    V = ast.literal_eval(config[section]['V'])
    grids = ast.literal_eval(config[section]['grids'])
    reps = ast.literal_eval(config[section]['reps'])

    return N, V, grids, reps

# benchmark_<run>_<input-size>_<tile-size>_<grid>.bin
if __name__ == "__main__":
    k = len(sys.argv)

    if k == 1:
        print("Provide a config file: e.g. python helper.py params.ini")
        sys.exit()

    configFile = sys.argv[1]
    
    # Cholesky2.5d
    N, V, grids, reps = readConfig(configFile)
    checkCsvExists(N, V, grids, reps)
    
    # Scalapack
    N, V, grids, reps = readConfig(configFile, True)
    checkCsvExists(N, V, grids, reps, True)
    print("Checked all csv files.")
    
    if k > 3:
        print("Usage 1: python benchmark_helper.py <config>")
        sys.exit()
