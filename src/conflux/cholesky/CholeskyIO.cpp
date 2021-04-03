/**
 * Copyright (C) 2020-2021, ETH Zurich
 *
 * This product includes software developed at the Scalable Parallel Computing
 * Lab (SPCL) at ETH Zurich, headed by Prof. Torsten Hoefler. For more information,
 * visit http://spcl.inf.ethz.ch/. Unless you have an agreement with ETH Zurich 
 * for a separate license, the following terms and conditions apply:
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

 * @file CholeskyIO.h
 * 
 * @brief definition of CholeskyIO class 
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard
 * Contact: {saethrej, andrega}@ethz.ch
 * 
 * @date 01.03.2021
 */

#include <fstream>
#include <cstring>
#include <math.h>
#include <string>
#include <random>
#include <sstream>
#include <mpi.h>

#ifdef __USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include "CholeskyIO.h"

/******************************** Constants **********************************/
#define TILE_TAG_A00   1
#define TILE_TAG_A10   2
#define TILE_TAG_A01   3
#define TILE_TAG_A11   4
#define SEED 1

/**
 * @brief constructor of CholeskyIO object
 * 
 * @param prop pointer to object storing all of the Cholesky factorization's
 * properties that are used
 * @param proc pointer to a processor's local variables, including all buffers
 * @throws CholeskyException if the MPI environment was not initialized
 */
conflux::CholeskyIO::CholeskyIO(CholeskyProperties *prop, Processor *proc)
{
    // throw an exception if the MPI environment was not initialized yet or if
    // any of the supplied pointers is NULL
    int isInitialized;
    MPI_Initialized(&isInitialized);
    if (!isInitialized || prop == nullptr || proc == nullptr) {
        throw CholeskyException(CholeskyException::errorCode::FailedMPIInit);
    }

    // create a new MPI file handle
    this->fh = new MPI_File;

    // set the member variables
    this->proc = proc;
    this->prop = prop;

    // set the name of the potential matrix input file
    std::stringstream tmp;
    tmp << "data/input_" << prop->N << ".bin";
    this->inputFilePath = tmp.str();
}

/**
 * @brief destructor of the CholeskyIO class. Deletes the file handle
 * 
 * @note the destructor does not take care of the closing of the file 
 * associated with this object. Any call to CholeskyIO::openFile() must
 * be followed by a call to CholeskyIO::closeFile(), otherwise the behaviour
 * of the object is undefined.
 */
conflux::CholeskyIO::~CholeskyIO()
{
    // close the file if this was not done already
    // commented this out as it results in segfault
    //if (*fh != MPI_FILE_NULL) {
    //   MPI_File_close(fh);
    //}
    // delete file handle, but nothing else since this is done by another object
    delete fh;
}

/**
 * @brief creates a symmetric positive definite matrix in a distributed fashion
 * on all processes
 * 
 * Unlike @ref void parseAndDistributeInputMatrix(), this function does not
 * require a matrix to be computed before-hand. Each rank creates the same random
 * matrix R of a full tile size (v x v) with the provided seed value, computes 
 * RR^T to obtain symmetry. This tile is then copied to all locations that should
 * store matrix values in the first iteration of the Cholesky algorithm. To obtain
 * positive definiteness, the diagonal of the matrix is then strenghtened, since
 * a real symmetric matrix is positive definite if it is diagonally dominant.
 * 
 * @pre the MPI environment and global variables have been set-up, i.e. the 
 * void initialize() function in @ref Cholesky.cpp has been called.
 * @post the data is distributed to the processes according to algorithm 8
 */
void conflux::CholeskyIO::generateInputMatrixDistributed()
{
    // set random seed at each rank s.t. all ranks compute same tile
    srand(SEED);

    // create random tile R and compute RR^T
    double *random = new double[prop->vSquare];
    double *tile = new double[prop->vSquare];
    for (size_t i = 0; i < prop->vSquare; ++i) {
        random[i] = (double) rand() / RAND_MAX * 2 - 1;
    }

    cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, prop->v, prop->v, 1.0,
                random, prop->v, 0.0, tile, prop->v);
    delete[] random;
    
    // copy this tile into every tile that the processor owns
    // starting with A00...
    size_t numBytes = prop->vSquare * sizeof(double);
    std::memcpy(proc->A00, tile, numBytes);

    // ...then all tiles in A10..
    for (TileIndex i = 0; i < proc->maxIndexA10; ++i) {
        std::memcpy(proc->A10->get(i), tile, numBytes);
    }

    // ... and finally all tiles in A11, but only by processors on z-layer 0
    if (proc->pz == 0) {
        for (TileIndex i = 0; i < proc->maxIndexA11i; ++i) {
            for (TileIndex j = 0; j < proc->maxIndexA11j; ++j) {
                std::memcpy(proc->A11->get(i, j), tile, numBytes);
            }
        }
    }

    // compute the maximum magnitude of a row's elements in the tile, multiply
    // it by the number of tiles and a factor of 2 to be sure
    double max = -1;
    for (size_t i = 0; i < prop->v; ++i) {
        double cur = 0.0;
        for (size_t j = 0; j < prop->v; ++j) {
            cur += std::abs(tile[i * prop->v + j]);
        }
        if (cur > max) max = cur;
    }
    max = max * prop->Kappa * 2;

    // adapt the diagonal: update A00's diagonal at all ranks and tiles on the
    // diagonal of A11. The latter only concers processors where pz=0 and px=py.
    for (int i = 0; i < prop->v; ++i) {
        proc->A00[i * prop->v + i] = max;
    }

    if (proc->pz == 0 && proc->px == proc->py) {
        // iterate over all local tiles from A11, compute global indices and check
        // whether iGlob = jGlob. If so, strengthen the diagonal there.
        for (TileIndex i = 0; i < proc->maxIndexA11i; ++i) {
            for (TileIndex j = 0; j < proc->maxIndexA11j; ++j) {
                // skip non-diagonal tiles
                TileIndices glob = prop->localToGlobal(proc->px, proc->py, i, j);
                if (glob.i != glob.j) {
                    continue;
                }

                // strengthen the diagonal on a a "diagonal tile"
                double *tile = proc->A11->get(i, j);
                for (int k = 0; k < prop->v; ++k) {
                    tile[k * prop->v + k] = max;
                }
            }
        }
    }
}

/**
* @brief reads a symmetric spd matrix from a file and distributes it
*
* This function distributes an spd matrix read from a file to all processors
* according to the specification of algorithm 8. We assume that this work
* is carried out by the processor with global rank 0.
*
* @pre the MPI environment and global variables have been set-up, i.e. the
* void initialize() function in @ref Cholesky.cpp has been called.
* @post the data is distributed to the processes according to algorithm 8
*/
void conflux::CholeskyIO::parseAndDistributeMatrix()
{
    // calculate new matrix size if needed
    int mod = (prop->N % prop->v) ? (prop->v - (prop->N % prop->v)) : 0;
    int originalN = prop->N;

    // if N is not divisible by v, we need to adjust dimension by zero-padding
    prop->N = prop->N + mod;

    /******************Distribution***************************/
    if (proc->rank == 0) {
        // allocate space for the full matrix here, initialize char* pointers that point
        // to the individual bytes for reading the input binary stream
        double *fullMatrix = new double[uint64_t(prop->N) * uint64_t(prop->N)];
        char *memblock = reinterpret_cast<char*>(fullMatrix);
        char *ptr = memblock;
        //@TODO need to rethink exception handling
        std::ifstream MatrixStream(inputFilePath.c_str(), std::ios::in | std::ios::binary);
        if (MatrixStream.fail()) {
            throw CholeskyException(CholeskyException::errorCode::FileSystemInputProblem);
        }
        // read the full matrix
        if (!mod) {
            MatrixStream.read(memblock, sizeof(double) * prop->N * prop->N);
        }
        // read parts and do padding
        else {
            for (uint32_t i = 0; i < originalN; i++) {
                // read N
                MatrixStream.read(ptr, sizeof(double) * originalN);
                ptr = ptr + originalN * sizeof(double);
                // pad mod
                for (uint32_t j = 0; j < mod; j++) {
                    ptr[j] = 0;
                }
                ptr = ptr + mod;
            }
            // now we need to add mod more 0 rows
            for (size_t i = 0; i < mod * prop->N; ++i) {
                ptr[i] = 0;
            }
        }
        ptr = nullptr;
        MatrixStream.close();

        // copy A00 into the buffer from the full matrix
        for (size_t i = 0; i < prop->v; ++i) {
            std::memcpy(&proc->A00[i * prop->v], &fullMatrix[i * prop->N], prop->v * sizeof(double));
        }

        // we distribute A11, A10 and A00 blockingly
        // the number of tiles we distributed is the total number of tiles without everything processor 0 receivez
        // don't distribute to yourself
        //A10 starts with a v*N offset into the matrix
        uint32_t A10_offset = prop->v * prop->N;
        // and A11 with v+N + v
        for (ProcRank rec_procRank = 0; rec_procRank < prop->P; ++rec_procRank) {
            // get the processor coordinates

            GridProc rec_procCoord = prop->globalToGrid(rec_procRank);

            /******************A10 Distribution*************************/
            // distribute all A10s to the correct location
            // we send Kappa-2 tiles from A10 in round robin manner
            for (size_t i_L = 0; i_L < prop->Kappa - 1; ++i_L) {
                ProcIndexPair1D pair1D;
                pair1D.i = i_L;
                pair1D.p = rec_procRank;
                TileIndex i_G = prop->localToGlobal(pair1D);
                // outside of tiles of A10
                if (i_G > prop->Kappa - 1) break;

                // fill the block
                // the block starts at element 0 of tile (i_G,0) i.e at element i_G*N*v
                // note that the first element is the global index for the receiving processor
                if (rec_procRank != 0) {
                    double* A10_send = new double[(prop->vSquare) + 1];
                    A10_send[0] = static_cast<double>(i_L);
                    for (size_t i = 0; i < prop->v; ++i) {
                        std::memcpy(
                            &A10_send[i * prop->v + 1],
                            &fullMatrix[i_G * prop->v * prop->N + i * prop->N],
                            prop->v * sizeof(double)
                        );
                    }

                    //MPI_Isend(A10_send, v * v + 1, MPI_DOUBLE, rec_procRank, TILE_TAG_A10, MPI_COMM_WORLD, &A10_request);
                    MPI_Send(A10_send, prop->vSquare + 1, MPI_DOUBLE, rec_procRank, TILE_TAG_A10, MPI_COMM_WORLD);
                    delete[] A10_send;
                }
                // processor 0
                else {
                    for (size_t i = 0; i < prop->v; ++i) {
                        std::memcpy(
                            &(proc->A10->get(i_L)[prop->v * i]),
                            &fullMatrix[i_G * prop->v * prop->N + i * prop->N],
                            prop->v * sizeof(double)
                        );
                    }
                }
            }

            /****************************A11 Distribution******************************/
            // only consider the lowest processor layer in z direction
            if (rec_procCoord.pz == 0) {
                for (TileIndex i_L = 0; i_L < prop->Kappa - 1; ++i_L) {
                    for (TileIndex j_L = 0; j_L < prop->Kappa - 1; ++j_L) {
                        ProcIndexPair2D pair2D;
                        pair2D.i = i_L;
                        pair2D.j = j_L;
                        pair2D.px = rec_procCoord.px;
                        pair2D.py = rec_procCoord.py;
                        TileIndices global2DTile = prop->localToGlobal(pair2D);

                        // out of bounds check
                        if (global2DTile.i > prop->Kappa - 1 || global2DTile.j > prop->Kappa - 1) {
                            continue;
                        }

                        // send it to the correct location
                        // the block starts at element 0 of tile (i_G,j_G) i.e at element i_G*N*v + j_G * v
                        uint64_t blockStart = prop->N * global2DTile.i * prop->v + global2DTile.j * prop->v;
                        if (rec_procRank > 0) {
                            double* A11_send = new double[prop->vSquare + 2];
                            // first two elements are indices
                            A11_send[0] = static_cast<double>(i_L);
                            A11_send[1] = static_cast<double>(j_L);
                            // copy the correct tile into the buffer
                            for (size_t i = 0; i < prop->v; i++) {
                                std::memcpy(
                                    &A11_send[i * prop->v + 2],
                                    &fullMatrix[blockStart + i * prop->N],
                                    prop->v * sizeof(double)
                                );
                            }
                            //MPI_Isend(A11_send, v * v + 2, MPI_DOUBLE, rec_procRank, TILE_TAG_A11, MPI_COMM_WORLD, &A11_request);
                            MPI_Send(A11_send, prop->vSquare + 2, MPI_DOUBLE,
                                     rec_procRank, TILE_TAG_A11, MPI_COMM_WORLD);
                            delete[] A11_send;
                        } else {
                            for (size_t i = 0; i < prop->v; ++i) {
                                std::memcpy(
                                    &(proc->A11->get(i_L, j_L)[prop->v * i]),
                                    &fullMatrix[blockStart + i * prop->N],
                                    prop->v * sizeof(double)
                                );
                            }
                        }
                    }
                }
            }
        }

        // free the memory allocated for storing the entire matrix
        delete[] fullMatrix;
    } // processor 0

    /***************************Reception*************************/
    else {
        // we have only one non-blocking statement to wait for
        // i.e. broadcast
        // we iterate over our local tiles in A10 and A11
        // to receive our data blockingly to be able to store it
        for (TileIndex i = 0; i < proc->maxIndexA10; ++i) {
            double *tmpBuffer = new double[prop->vSquare + 1];
            MPI_Recv(tmpBuffer, prop->vSquare + 1, MPI_DOUBLE, 0, TILE_TAG_A10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            size_t A10_idx = static_cast<size_t>(tmpBuffer[0]);
            std::memcpy(proc->A10->get(A10_idx), &tmpBuffer[1], prop->vSquare * sizeof(double));
            delete[] tmpBuffer;
        }

        // we only receive A11 if we are on the lowest z-layer
        if (proc->pz == 0) {
            TileIndex numTilesInA11 = proc->maxIndexA11i * proc->maxIndexA11j;
            for (TileIndex i = 0; i < numTilesInA11; ++i) {
                double* tmpBuffer = new double[prop->vSquare + 2];
                MPI_Recv(tmpBuffer, prop->vSquare + 2, MPI_DOUBLE, 0, TILE_TAG_A11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int A11_i_idx = (int)tmpBuffer[0];
                int A11_j_idx = (int)tmpBuffer[1];
                std::memcpy(
                    (proc->A11->get(A11_i_idx, A11_j_idx)),
                    &tmpBuffer[2],
                    prop->vSquare * sizeof(double)
                );
                delete[] tmpBuffer;
            }
        }
    }
    // blocking broadcasting
    MPI_Bcast(proc->A00, prop->vSquare, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

/**
 * @brief Opens a file handle to enable dumping of the matrix
 * @param filename the name of the file and its path
 * 
 * @note The user is responsible that a file opened with a call to this method
 * is closed with a call to @ref CholeskyIO::closeFile()
 */
void conflux::CholeskyIO::openFile(std::string filename)
{
    MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, fh);
}

/**
* @brief closes the file associated with this object
*/
void conflux::CholeskyIO::closeFile()
{
    MPI_File_close(fh);
}

/**
 * @brief dumps the complete (remaining) matrix including A11 into a file
 * @param round the iteration that the algorithm is in while dumping
 * 
 * @pre The algorithm has not yet touched the distributed matrix
 * @post The input matrix is completely dumped to the file and the buffers can be altered
 */
void conflux::CholeskyIO::dumpMatrix()
{
    // dump the current, single tile column consisting of A00 and A10
    dumpSingleTileColumn(0);

    // dump A11 from the z-layer that currently owns A11
    if (proc->pz == 0) {
        dumpA11();
    }
}

/**
 * @brief dumps a specified tile column to the opened file
 * @param round the iteration that the algorithm is in while dumping
 */
void conflux::CholeskyIO::dumpSingleTileColumn(TileIndex round)
{
    // if we have an idle processor, let this one dump A00
    //if (prop->P > prop->Kappa - round - 1) {
     //   if (proc->rank == prop->Kappa - round - 1) {
      //      dumpA00(round);
       // }
    //}
    // otherwise let the last processor handle it
    if (proc->rank == 0){
        dumpA00(round);
    }

    // iterate over global tiles in this tile column, let processor that owns it
    // dump it to the corresponding position
    for (TileIndex iGlob = round + 1; iGlob < prop->Kappa; ++iGlob) {
        // get processor that owns tile and corresponding local tile index
        ProcIndexPair1D loc = prop->globalToLocal(iGlob);

        // only let tile-owning processor dump it
        if (proc->rank == loc.p) {
            dumpA10(loc, iGlob, round);
        }
    }
}

/**
 * @brief dumps the A00 tile of the specified round to the file
 * @param round the iteration that the Cholesky factorization algorithm is in
 */
void conflux::CholeskyIO::dumpA00(TileIndex round)
{
    // dump the tile row-by-row
    for (int h = 0; h < prop->v; h++) {
        MPI_Offset offset = round *prop->N * prop->v  + round*prop->v + h*prop->N;
        MPI_File_write_at(*fh, offset * sizeof(double), &(proc->A00[h*prop->v]), 
                          prop->v, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
}

/**
 * @brief dumps a specified tile in A10 owned by the executing process to a
 * a file.
 * @param loc rank of owning processor and corresponding local index of tile
 * @param iGlob the tile's global rank, used for offset calculation
 * @param round the iteration that the Cholesky algorithm is in
 */
void conflux::CholeskyIO::dumpA10(ProcIndexPair1D loc, TileIndex iGlob, TileIndex round)
{
    double *cur = proc->A10->get(loc.i);
    // we need different offset calculations depending on if we are transposing or not
    for (int h = 0; h < prop->v; h++){
        MPI_Offset offset = iGlob * prop->N * prop->v   // row start of current tile
                            + round * prop->v           // column
                            + h * prop->N;              // index
        MPI_File_write_at(*fh, offset * sizeof(double), &cur[h*prop->v], prop->v, 
                          MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
}

/**
 * @brief dump all data in A11 into the opened file
 */
void conflux::CholeskyIO::dumpA11()
{
    // iterate over all local tiles in A11 owned by this processor
    for (TileIndex i = 0; i < proc->maxIndexA11i; i++) {
        for (TileIndex j = 0; j < proc->maxIndexA11j; j++) {

            // find global index of current tile for file offset
            TileIndices glob = prop->localToGlobal(proc->px, proc->py, i,j);
            double *buf = proc->A11->get(i,j);

            // write the tile row-by-row into the file
            for (int h = 0; h < prop->v; h++) {
                MPI_Offset offset = glob.i * prop->N * prop->v + glob.j * prop->v + prop->N * h;
                MPI_File_write_at(*fh, offset * sizeof(double), &buf[h*prop->v], 
                                  prop->v, MPI_DOUBLE, MPI_STATUS_IGNORE);
            }
        }
    }
}
