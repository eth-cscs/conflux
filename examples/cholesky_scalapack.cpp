/**
 * Copyright (C) 2021, ETH Zurich
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
 * @file Cholesky_scalapack.cpp
 * 
 * @brief Implementation of cholesky decompositiong using netlibs scalapack
 *
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 04.01.2021
 */


/*
 * This code is heavily inspired by the following two sources:
 * 1. https://gist.github.com/leopoldcambier/be8e68906ecfd7f03edf0d809db37cc1
 * 2. https://andyspiros.wordpress.com/2011/07/08/an-example-of-blacs-with-c/
 *
 * Some pieces of the code are adapted codes from the above two sources.
 */


#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>
#include <getopt.h>

#include <mpi.h>

#include <conflux/cholesky/benchmark/Benchmark.h>
 
using namespace std;
 
extern "C" {
    /* Cblacs declarations */
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, const char*, int, int);
    void Cblacs_pcoord(int, int, int*, int*);
    void Cblacs_gridexit(int);
    void Cblacs_barrier(int, const char*);
    void Cdgerv2d(int, int, int, double*, int, int, int);
    void Cdgesd2d(int, int, int, double*, int, int, int);
 
    int numroc_(int*, int*, int*, int*, int*);
    
    void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);

    void pdpotrf_(char*, int*, double*, int*, int*, int*, int*); 
}
 
/**
 * @brief prints help for command line arguments of this program
 */
void printHelp() 
{
    std::cerr << "Error: invalid parameters were provided. Allowed are: \n"
              << "\t --help, -h: Display (this) help section. Write --help or -h\n"
              << "\t --dim, -d: dimension of the matrix (>0). Write e.g. --dim=128 or -d 128 \n"
              << "\t --tile, -v: tile grid ((x>0,y>0). Can be non-square. Write --tile=4x4 or -v 4x4 \n"
              << "\t --grid, -g: processor grid. Can be non-square. Write --grid=2x2 or -g 2x2\n"
              << "\t --run, -r: set current run. For benchmarking purposes.\n"
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
 */
void parseArgs(int argc, char* argv[], int &dim, int *tileGrid, int *procGrid,
               int &run)
{
    // defines the allowed command line options in their "long" form
    static const struct option long_args[] = {
        {"dim", required_argument, 0, 'd'},
        {"tile", required_argument, 0, 'v'},
        {"grid", required_argument, 0, 'g'},
        {"run", required_argument, 0, 'r'},
        {"help", no_argument, 0, 'h'},
        0
    };

    int index = -1;
    int result;
    struct option *opt = nullptr;

    while (true) {
        index = -1;
        opt = nullptr;
        result = getopt_long(argc, argv, "hd:v:g:r:", long_args, &index);

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
                    std::string tileStr = std::string(optarg);
                    size_t x0 = tileStr.find('x', 0);
                    size_t x1 = tileStr.find('x', x0+1);
                    tileGrid[0] = std::stoi(tileStr.substr(0, x0));
                    tileGrid[1] = std::stoi(tileStr.substr(x0+1, x1-x0-1));
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                if (tileGrid[0] < 0 || tileGrid[1] < 0) {
                    printHelp();
                    exit(-1);
                }
                break;

            case 'g': // processor grid
                try {
                    std::string gridStr = std::string(optarg);
                    size_t x0 = gridStr.find('x', 0);
                    size_t x1 = gridStr.find('x', x0+1);
                    procGrid[0] = std::stoi(gridStr.substr(0, x0));
                    procGrid[1] = std::stoi(gridStr.substr(x0+1, x1-x0-1));
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                if (procGrid[0] < 0 || procGrid[1] < 0) {
                    printHelp();
                    exit(-1);
                }
                break;
            
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

            default: // fallback, indicates incorrect usage, print help and exit
                printHelp();
                exit(-1);
        }
    }
}

int main(int argc, char **argv)
{
    /*
     * Init MPI
     */
    MPI_Init(&argc, &argv);
    
    int mpirank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    
    bool mpiroot = (mpirank == 0);

    /*
     * Helping vars
     */
    int iZERO = 0;
 
    /*
     * Command line arguments
     */
    int N = 0;
    uint64_t numOfDoubles;
    int tileGrid[2] = {0, 0};
    int procGrid[2] = {0, 0};
    int run = 0;

    double *A_glob = NULL, *A_loc = NULL;

    int Nb, Mb;

    // Parse command line arguments
    parseArgs(argc, argv, N, tileGrid, procGrid, run);
    
    Nb = tileGrid[0];
    Mb = tileGrid[1];
    numOfDoubles = (uint64_t) N * N;

    if (mpiroot) {
        /* Reserve space and read matrix (with transposition!) */
        A_glob  = new double[ numOfDoubles ];

        string fname(argv[1]);
	
        // Construct filename for input matrix
        std::stringstream pathToMatrix;
        pathToMatrix << "data/input_" << N << ".bin";

        // Read matrix

        //char *memblock = new char[N * N * sizeof(double)];
        std::ifstream MatrixStream(pathToMatrix.str().c_str(), std::ios::in | std::ios::binary);
        MatrixStream.read(reinterpret_cast<char*>(A_glob), sizeof(double) * (uint64_t) N * (uint64_t) N);
        MatrixStream.close();       
	
        //std::memcpy(A_glob, memblock, N * N * sizeof(double));
        //delete[] memblock;

        /* Print matrix */
        #ifdef DEBUG
        cout << "Matrix A:\n";
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                cout << setw(3) << *(A_glob + N*c + r) << " ";
            }
            cout << "\n";
        }
        cout << endl;
        #endif
    }

    /* Begin Cblas context */
    /* We assume that we have 4 processes and place them in a 2-by-2 grid */
    int ctxt, myid, myrow, mycol, numproc;
    int procrows = procGrid[1], proccols = procGrid[0];
    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctxt);
    Cblacs_gridinit(&ctxt, "Row-major", procrows, proccols);
    Cblacs_pcoord(ctxt, myid, &myrow, &mycol);
 
    /* Print grid pattern */
    #ifdef DEBUG
    if (myid == 0)
        cout << "Processes grid pattern:" << endl;
    for (int r = 0; r < procrows; ++r) {
        for (int c = 0; c < proccols; ++c) {
            Cblacs_barrier(ctxt, "All");
            if (myrow == r && mycol == c) {
                cout << myid << " " << flush;
            }
        }
        Cblacs_barrier(ctxt, "All");
        if (myid == 0)
            cout << endl;
    }
    #endif
 
    /*********************
     * DATA DISTRIBUTION *
     *********************/
 
    /* Broadcast of the matrix dimensions */
    int dimensions[3];
    if (mpiroot) {
        dimensions[0] = N;
        dimensions[1] = Nb;
        dimensions[2] = Mb;
    }
    MPI_Bcast(dimensions, 3, MPI_INT, 0, MPI_COMM_WORLD);
    N = dimensions[0];
    Nb = dimensions[1];
    Mb = dimensions[2];
 
    /* Reserve space for local matrices */
    // Number of rows and cols owned by the current process
    uint64_t nrows = numroc_(&N, &Nb, &myrow, &iZERO, &procrows);
    uint64_t ncols = numroc_(&N, &Mb, &mycol, &iZERO, &proccols);
    for (int id = 0; id < numproc; ++id) {
        Cblacs_barrier(ctxt, "All");
    }
    A_loc = new double[nrows*ncols];
    for (int i = 0; i < nrows*ncols; ++i) *(A_loc+i)=0.;
 
    /* Scatter matrix */
    int sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (int r = 0; r < N; r += Nb, sendr=(sendr+1)%procrows) {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        int nr = Nb;
        if (N-r < Nb)
            nr = N-r;
 
        for (int c = 0; c < N; c += Mb, sendc=(sendc+1)%proccols) {
            // Number of cols to be sent
            // Is this the last col block?
            int nc = Mb;
            if (N-c < Mb)
                nc = N-c;
 
            if (mpiroot) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                Cdgesd2d(ctxt, nr, nc, A_glob + (uint64_t) N*c+r, N, sendr, sendc);
            }
 
            if (myrow == sendr && mycol == sendc) {
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                Cdgerv2d(ctxt, nr, nc, A_loc + (uint64_t) nrows*recvc+recvr, nrows, 0, 0);
                recvc = (recvc+nc)%ncols;
            }
 
        }
 
        if (myrow == sendr)
            recvr = (recvr+nr)%nrows;
    }
////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////
    // CHOLESKY DECOMPOSITION //
    ////////////////////////////
   
    /* 
     * IMPORTANT: We only store the x-componnent of the tile size.
     * That's because while it would be possible to run jobs with non-square
     * tile size it doesn't make sense because it wouldn't be comparable
     * to our algorithm.
     */
    
    conflux::Benchmark *benchmark = new conflux::Benchmark();
    benchmark->set_props("data/benchmarks/scalapack/output", run, N, Nb,
            procGrid[0], procGrid[1], 0);

    // Distributed A_glob to A_loc.
    char uplo = 'L';
    char layout = 'R';

    // Create descriptor
    int descA[9];
    int info;
    int zero = 0;
    int ione = 1;
    int lddA = nrows > 1 ? nrows : 1;
    descinit_(descA, &N, &N, &Nb, &Mb, &zero, &zero, &ctxt, &lddA, &info);
    
    if (info != 0) {
        cout << "Error in descinit, info=" << info;
    }
    
    // Run dpotrf
    //double MPIt1 = MPI_Wtime();
    
    benchmark->timer_start();
    pdpotrf_(&uplo, &N, A_loc, &ione, &ione, descA, &info);
    benchmark->timer_stop();

    benchmark->finish();

    if (info != 0) {
        cout << "Error in potrf, info=" << info;
    }
//////////////////////////////////////////////////////////////////////////////// 
    /* Gather matrix */
    #ifdef DEBUG
    sendr = 0;
    for (int r = 0; r < N; r += Nb, sendr=(sendr+1)%procrows) {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        int nr = Nb;
        if (N-r < Nb)
            nr = N-r;
 
        for (int c = 0; c < N; c += Mb, sendc=(sendc+1)%proccols) {
            // Number of cols to be sent
            // Is this the last col block?
            int nc = Mb;
            if (N-c < Mb)
                nc = N-c;
 
            if (myrow == sendr && mycol == sendc) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                Cdgesd2d(ctxt, nr, nc, A_loc+nrows*recvc+recvr, nrows, 0, 0);
                recvc = (recvc+nc)%ncols;
            }
 
            if (mpiroot) {
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                Cdgerv2d(ctxt, nr, nc, A_glob+N*c+r, N, sendr, sendc);
            }
 
        }
 
        if (myrow == sendr)
            recvr = (recvr+nr)%nrows;
    }
    
    /* Print test matrix */ 
    if (mpiroot) {
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                if (c<=r) {
                    cout << setw(3) << *(A_glob+N*c+r) << " ";
                } else {
                    cout << setw(3) << 0.0 << " ";
                }
            }
            cout << endl;
        }
    }
    #endif 
    /************************************
     * END OF THE MOST INTERESTING PART *
     ************************************/
 
    /* Release resources */
    delete[] A_glob;
    delete[] A_loc;
    Cblacs_gridexit(ctxt);
    MPI_Finalize();
}

