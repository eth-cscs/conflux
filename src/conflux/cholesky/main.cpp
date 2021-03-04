/**
 * Copyright (C) 2020, ETH Zurich
 *
 * This product includes software developed at the Scalable Parallel Computing
 * Lab (SPCL) at ETH Zurich, headed by Prof. Torsten Hoefler. It was developped
 * as a part of the "Design of Parallel- and High-Performance Computing"
 * lecture. For more information, visit http://spcl.inf.ethz.ch/. Unless you 
 * have an agreement with ETH Zurich for a separate license, the following
 * terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along 
 * with this program. If not, see http://www.gnu.org/licenses.

 * @file main.cpp
 * 
 * @brief main program for testing our Cholesky factorization program
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 14.12.2020
 */


#include <unistd.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <sstream>

#include "CholeskyTypes.h"
#include "Cholesky.h"

// only include Processor in debug or benchmarking mode
#if defined(DEBUG) || defined(BENCHMARK)
#include "Processor.h"
#include <sstream>
#include <string>
#endif

// only include fstream, Processor and CholeskyProperties in DEBUG mode
#ifdef DEBUG
#include <fstream>
#include "CholeskyProperties.h"
#endif // DEBUG


// in debug or benchmarking mode, we require access to the global variable
// proc defined in Cholesky.cpp
#if defined(DEBUG) || defined(BENCHMARK)
extern conflux::Processor *proc;
#endif

// in debug mode we require access to the global variables defined in Cholesky.cpp
#ifdef DEBUG
extern conflux::CholeskyProperties *prop;
extern std::string pathToMatrix;
#endif // DEBUG

/**
 * @brief prints help for command line arguments of this program
 */
void printHelp() 
{
    std::cerr << "Error: invalid parameters were provided. Allowed are: \n"
              << "\t --help, -h: Display (this) help section. Write --help or -h\n"
              << "\t --dim, -d: dimension of the matrix (>0). Write e.g. --dim=128 or -d 128 \n"
              << "\t --tile, -v: (optional) tile size (>0). Write --tile=4 or -v 4 \n"
              << "\t --grid, -g: (optional) processor grid. Write --grid=2x2x2 or -g 2x2x2\n"
              << "\t --no-writeback, -x: (optional) don't write back result to file. Write --no-writeback or -x\n"
              #if defined(BENCHMARK)
              << "\t --run, -r: (optional, benchmarking) set current run. For benchmarking purposes.\n"
              #endif
              << "Try again.\n"
              << std::endl;
    return;
}

/**
 * @brief parses args from the command line that define a decomposition
 * 
 * @param argc the number of arguments from the shell
 * @param argv array with the arguments and their values
 * @param dim reference to the matrix dimension read in this function
 * @param tile reference to the tile size read in this funciton
 * @param grid pointer to the grid read in this function
 * @param store reference to the flag indicating whether or not to save output
 * @param run reference to the number of current run read in this function
 */
#if defined(BENCHMARK)
void parseArgs(int argc, char* argv[], uint32_t &dim, uint32_t &tile, 
               conflux::ProcCoord *grid, bool &store, uint32_t &run)
#else
void parseArgs(int argc, char* argv[], uint32_t &dim, uint32_t &tile, 
               conflux::ProcCoord *grid, bool &store)
#endif
{
    // defines the allowed command line options in their "long" form
    static const struct option long_args[] = {
        {"dim", required_argument, 0, 'd'},
        {"tile", required_argument, 0, 'v'},
        {"grid", required_argument, 0, 'g'},
        {"no-writeback", no_argument, 0, 'x'},
        {"help", no_argument, 0, 'h'},
        #if defined(BENCHMARK)
        {"run", required_argument, 0, 'r'},
        #endif
        0
    };

    int index = -1;
    int result;
    struct option *opt = nullptr;

    while (true) {
        index = -1;
        opt = nullptr;
        #if defined(BENCHMARK)
        result = getopt_long(argc, argv, "hd:v:g:xr:", long_args, &index);
        #else
        result = getopt_long(argc, argv, "hd:v:g:x", long_args, &index);
        #endif

        // break loop if all arguments were parsed
        if (result == -1) break;

        switch (result) {
            case 'h': // help section
                printHelp();
                exit(0);

            case 'd': // matrix dimensionality
                try {
                    dim = std::stoi(optarg);
                    if (dim <= 0) {
                        printHelp();
                        exit(-1);
                    }
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                break;

            case 'v': // tile size
                try {
                    tile = std::stoi(optarg);
                    if (tile <= 0) {
                        printHelp();
                        exit(-1);
                    }
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                break;

            case 'g': // processor grid
                try {
                    std::string gridStr = std::string(optarg);
                    size_t x0 = gridStr.find('x', 0);
                    size_t x1 = gridStr.find('x', x0+1);
                    grid[0] = std::stoi(gridStr.substr(0, x0));
                    grid[1] = std::stoi(gridStr.substr(x0+1, x1-x0-1));
                    grid[2] = std::stoi(gridStr.substr(x1+1));
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                if (grid[0] < 0 || grid[1] < 0 || grid[2] < 0) {
                    printHelp();
                    exit(-1);
                }
                break;

            case 'x': // no write back (e.g. for benchmarking)
                store = false;
                break;

            default: // fallback, indicates incorrect usage, print help and exit
                printHelp();
                exit(-1);

            #if defined(BENCHMARK)
            case 'r': // number of current run
                try {
                    run = std::stoi(optarg);
                    if (run < 0) {
                        printHelp();
                        exit(-1);
                    }
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                break;
            #endif
        }
    }
}

/**
 * @brief main function (for debugging purposes)
 * 
 * @param argc the number of command line arguments
 * @param argv an array of pointer to the command line arguments
 */
int main(int argc, char *argv[])
{
    // parse command line arguments
    uint32_t matrixDim = 0;
    uint32_t tileSize = 0;
    conflux::ProcCoord grid[3] = {0, 0, 0};
    bool storeMatrix = true;    
    
    #if defined(BENCHMARK)
    uint32_t run = 0;
    parseArgs(argc, argv, matrixDim, tileSize, grid, storeMatrix, run);
    #else
    parseArgs(argc, argv, matrixDim, tileSize, grid, storeMatrix);
    #endif

    // if matrix dimension is zero, we cannot proceed
    if (matrixDim == 0) {
        std::cerr << "Error: matrix dimension is a mandatory parameter. Try again." << std::endl;
        exit(-1);
    }

    // if both tile size and grid were provided, use these values, otherwise
    // compute optimal parameters
    if (tileSize > 0 && grid[0] > 0 && grid[1] > 0 && grid[2] > 0) {
        conflux::initialize(argc, argv, matrixDim, tileSize, grid);
    } else {
        conflux::initialize(argc, argv, matrixDim);
    }

    #ifdef BENCHMARK
    proc->benchmark->set_props("data/benchmarks/cholesky25d/output", run,
            matrixDim, tileSize, grid[0], grid[1], grid[2]);
    proc->benchmark->timer_start();
    #endif

    conflux::parallelCholesky();
    
    #ifdef BENCHMARK
    proc->benchmark->timer_stop();
    #endif 

    conflux::finalize(true);
    return 0;
}
