/**
 * @file cholesky_miniapp.cpp
 * 
 * @brief miniapp for the psyCHOL algorithm, used for benchmarking
 * 
 * @authors Anonymized Authors
 * 
 * @date 13.03.2021
 */

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>

#include <mpi.h>
#include <cxxopts.hpp>

#include "conflux/cholesky/CholeskyProfiler.h"
#include "conflux/cholesky/CholeskyTypes.h"
#include "conflux/cholesky/Cholesky.h"

/**
 * @brief prints the timings to the desired stream
 * @param out the desired stream (e.g. std::cout or a file)
 * @param timings the vector containing the timings
 * @param N the matrix dimension
 * @param v the tile size
 * @param grid the three-dimensional processor grid
 */
void printTimings(std::vector<double> &timings, std::ostream &out, int N, int v, conflux::ProcCoord grid[3])
{
    out << "==========================" << std::endl;
    out << "    PROBLEM PARAMETERS:" << std::endl;
    out << "==========================" << std::endl;
    out << "Matrix size: " << N << std::endl;
    out << "Tile size: " << v << std::endl;
    out << "Processor grid: " << grid[0] << "x" << grid[1] << "x" << grid[2] << std::endl;
    out << "Number of repetitions: " << timings.size() << std::endl;
    out << "--------------------------" << std::endl;
    out << "TIMINGS [ms] = ";
    for (auto &time : timings) {
        out << time << " ";
    }
    out << std::endl;
    out << "==========================" << std::endl;
}

/**
 * @brief main function (for debugging purposes)
 * 
 * @param argc the number of command line arguments
 * @param argv an array of pointer to the command line arguments
 * 
 * @returns 0 on success, <0 otherwise
 */
int main(int argc, char *argv[])
{   
    // set-up parser for the command line arguments
    cxxopts::Options options(
        "Cholesky Mini-App for Benchmarking",
        "This is a miniapp computing A =LL^T for A s.p.d. and dim(A)= N*N"
    );
    options.add_options()
        ("N,dim", "Dimension of input matrix", 
        cxxopts::value<uint32_t>()->default_value("65536"))
        ("v,tile", "Dimension of the tiles",
        cxxopts::value<uint32_t>()->default_value("256"))
        ("g,grid", "Processor grid to use for this factorization",
        cxxopts::value<std::vector<uint32_t>>())
        ("r,run", "The number of runs to perform",
        cxxopts::value<uint32_t>()->default_value("5"))
        ("h,help", "Print usage.");

    auto argv2 = const_cast<const char**>(argv);
    auto result = options.parse(argc, argv2);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return -1;
    }

    // parse the command line arguments
    uint32_t N = result["N"].as<uint32_t>();
    uint32_t v = result["v"].as<uint32_t>();
    uint32_t runs = result["r"].as<uint32_t>();
    std::vector<uint32_t> tmpGrid = result["g"].as<std::vector<uint32_t>>();
    conflux::ProcCoord grid[3] = {tmpGrid[0], tmpGrid[1], tmpGrid[2]};

    // if matrix dimension is non-positive, we cannot proceed
    if (N < 0) {
        std::cerr << "Error: Invalid matrix dimension." << std::endl;
        return -1;
    }

    // initialize the MPI environment
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // run the factorization once before benchmarking to make sure that the nodes
    // and threads per rank are already allocated when benchmarking
    conflux::initialize(argc, argv, N, v, grid);
    conflux::parallelCholesky();
    conflux::finalize(true);

#ifndef DEBUG
    // in non-debug mode, the benchmarking starts now after the warm-up was completed
    std::vector<double> timings;
    timings.reserve(runs);

    // perform the cholesky factorization *runs* times and report the time
    for (size_t i = 0; i < runs; ++i) {
        // initialize the factorization from the ground for every iteration
        conflux::initialize(argc, argv, N, v, grid);

        // clear profiler if enabled with compile flag
        if (rank == 0) {
            PC();
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // this is the part of the code that will be benchmarked
        auto start = std::chrono::high_resolution_clock::now();
        conflux::parallelCholesky();
        MPI_Barrier(MPI_COMM_WORLD);

        // print profiler information if enabled with compile flag
        if (rank == 0) {
            PP();
        }

        auto end = std::chrono::high_resolution_clock::now();
        double timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

        // finalize the library, and if successful, push timing back in the vector
        conflux::finalize(true);
        timings.push_back(timeMs);
    }

    // let rank output the timing values
    if (rank == 0) {
        printTimings(timings, std::cout, N, v, grid);

        // if you want to print the results to a file, uncomment the following:
        //std::ofstream output("data/benchmarks/your-filename.txt", std::ios::out);
        //printTimings(timings, output, N, v, grid);
        //output.close();
    }

#endif // DEBUG

    MPI_Finalize();

    return 0;
}
