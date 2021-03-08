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

def genereateRunCmd_cholesky25d(run,n,v,P,grid):
    #cmd = f"mpiexec -n {P} build/cholesky --dim={n} --tile={v} --grid={grid} --run={run} --expPath={expPath}"
    
    cmd = f"bsub -n {P} mpirun build/cholesky_benchmark --dim={n} --tile={v} --grid={grid} --run={run}"
    
    return cmd

def genereateRunCmd_scalapack(run,n,v,P,Px,Py):
    if n > 8192:
        memSize = getMemSize(n)
        cmd = f"bsub -R \"rusage[mem={memSize}]\" -n {P} mpirun build/cholesky_scalapack --dim={n} --tile={v}x{v} --grid={Px}x{Py} --run={run}"
    else:
        cmd = f"bsub -n {P} mpirun build/cholesky_scalapack --dim={n} --tile={v}x{v} --grid={Px}x{Py} --run={run}"
    
    return cmd

def generateRunBashScript(N, V, grids, reps, scalaPack = False):
    if not scalaPack:
        filename = "run_cholesky25d.sh"
    else:
        filename = "run_scalapack.sh"

    f = open(filename, "w")

    for i in range(reps): # Repetitions
        for n in N: # Input size
            for v in V: # Tile size
                for grid in grids: # Grid layouts
                    cmd = ""
                    if not scalaPack:
                        cmd = genereateRunCmd_cholesky25d(i,n,v,grid[0],grid[1])
                    else:
                        Px, Py = getGridLayout(grid[1])
                        cmd = genereateRunCmd_scalapack(i,n,v,grid[0],Px,Py)

                    f.write(cmd + "\n")
    f.close()



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
            duration = end - begin;
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
            for grid in grids: # Grid layouts
                if not scalaPack:
                    csvFile = f"data/benchmarks/cholesky25d/output/csv/benchmark_{n}_{v}_{grid[1]}.csv"
                else:
                    csvFile = f"data/benchmarks/scalapack/output/csv/benchmark_{n}_{v}_{grid[1]}x0.csv"

                with open(csvFile, 'w') as f:
                    write = csv.writer(f)
                    for i in range(reps): # 10 Repetitions
                        if not scalaPack:
                            benchmarkFile = f"data/benchmarks/cholesky25d/output/benchmark-{i}_{n}_{v}_{grid[1]}.bin"
                        else: 
                            benchmarkFile = f"data/benchmarks/scalapack/output/benchmark-{i}_{n}_{v}_{grid[1]}x0.bin"
                        data = parseBenchmark(benchmarkFile)
                        if data == -1:
                            f.flush()
                            print(f"{benchmarkFile} was empty")
                            continue

                        write.writerow(data)

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

    if k == 2:
        if sys.argv[1] == 'generateCsv':
            print("You need to provide a config file first: python benchmark_helper.py <config> generateCsv.")
            sys.exit()

    configFile = sys.argv[1]
    
    if k == 3:
        if sys.argv[2] == 'generateCsv':
            # Cholesky2.5d
            N, V, grids, reps = readConfig(configFile)
            saveToCsv(N, V, grids, reps)
            
            # Scalapack
            N, V, grids, reps = readConfig(configFile, True)
            saveToCsv(N, V, grids, reps, True)
            print("Turned benchmarks into csv files.")
            sys.exit()
        else:
            print("The second argument can only be 'generateCsv'")

    if k > 3:
        print("Usage 1: python benchmark_helper.py <config>")
        print("Usage 2: python benchmark_helper.py <config> generateCsv") 
        sys.exit()

    print("\nGenerating a run_cholesky25d.sh and run_scalapack.sh file.")
    print("This run.sh file contains the bjobs for euler for the given config.")
    print("It generates all possible combinations.")
    print("...")

    if k < 3:
        # Cholesky2.5d
        N, V, grids, reps = readConfig(configFile)
        generateRunBashScript(N, V, grids, reps)
        
        # Scalapack
        N, V, grids, reps = readConfig(configFile, True)
        generateRunBashScript(N, V, grids, reps, True)
    
    print("Your files were created. Run one or both scripts to submit the jobs.")
    print("Make sure to chmod +x <file> so you can actually execute it.")
    print("\nATTENTION: Inspect the file before you run it. Carefully.\n")
    print("RUN THIS FROM ROOT DIR!\n")




