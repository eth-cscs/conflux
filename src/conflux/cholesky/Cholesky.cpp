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

 * @file Cholesky.cpp
 * 
 * @brief implementation of near communication-optimal parallel Cholesky
 * factorization algorithm
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 14.11.2020
 */

#include <fstream>
#include <cstring>
#include <math.h>
#include <string>
#include <random>
#include <iostream>
#include <sstream>
#include <unordered_set>

#ifdef DEBUG
#include <unistd.h>
#endif

#include <mpi.h>

#ifdef __USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#define TESTRANK 0

//#include <semiprof/semiprof.hpp>

#include "CholeskyTypes.h"
#include "CholeskyProperties.h"
#include "CholeskyIO.h"
#include "Cholesky.h"
#include "Processor.h"
#include "TileMatrix.h"

// global variables (defined extern elsewhere)
conflux::Processor *proc;
conflux::CholeskyProperties *prop;
conflux::CholeskyIO *io;
std::string pathToMatrix;
MPI_Datatype MPI_SUBTILE, MPI_SUBTILE_RECV;

/**
 * @brief initializes environment and allocates local buffers for all processes
 * 
 * This function initializes the MPI environment, computes the optimal global
 * variables, such as: tile size (v), processor grid (PX, PY, PZ, PXY), the
 * number of tile in each direction (Kappa), etc. Finally, it allocates the
 * necessary buffers for each process on the heap.
 *
 * @param argc the number of command line arguments (from main fnc)
 * @param argv an array of pointers to command line arguments (from main fnc)
 * @param N the size of the matrix
 */
void conflux::initialize(int argc, char *argv[], uint32_t N)
{
    // throw an exception if MPI was not initialized
    int isInitialized;
    MPI_Initialized(&isInitialized);
    if (!isInitialized) {
        throw CholeskyException(CholeskyException::errorCode::FailedMPIInit);
    }
    
    int numProc;
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    // get the properties for the cholesky factorization algorithm
    prop = new CholeskyProperties(static_cast<ProcRank>(numProc), N);

    // get the processor information
    proc = new Processor(prop);

    // create and commit new type for subtiles
    MPI_Type_vector(prop->v, prop->l, prop->v, MPI_DOUBLE, &MPI_SUBTILE);
    MPI_Type_commit(&MPI_SUBTILE);
    MPI_Type_vector(1, prop->v*prop->l , prop->PX * prop->v * prop->l, MPI_DOUBLE, &MPI_SUBTILE_RECV);
    MPI_Type_commit(&MPI_SUBTILE_RECV);
    // create new CholeksyIO object 
    io = new CholeskyIO(prop, proc);

    // set path to matrix for this dimension
    std::stringstream tmp;
    tmp << "../data/input_" << N << ".bin";
    pathToMatrix = tmp.str();

    // create input matrix and dump it (only in debug mode)
    io->generateInputMatrixDistributed();
    #if DEBUG
    io->openFile(pathToMatrix);
    io->dumpMatrix();
    io->closeFile();
    #endif
}

/**
 * @brief initializes environment and allocates local buffers for all processes
 * 
 * This function intializes the MPI environment, sets the variables as specified
 * by the user via command line arguments, and allocates the necessary buffers
 * to store the matrix tiles on the heap.
 * 
 * @overload void initialize(int, char*[], uint32_t)
 * 
 * @param argc the number of command line arguments
 * @param argv an array of command line arguments
 * @param N the dimension of the matrix
 * @param v the tile size
 * @param grid pointer to the grid dimensions
 * 
 * @throws CholeskyException if MPI environment was not initialized
 */
void conflux::initialize(int argc, char *argv[], uint32_t N, uint32_t v, ProcCoord *grid)
{
    // throw an exception if MPI was not initialized
    int isInitialized;
    MPI_Initialized(&isInitialized);
    if (!isInitialized) {
        throw CholeskyException(CholeskyException::errorCode::FailedMPIInit);
    }

    int numProc;
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    // get the properties for the cholesky factorization algorithm
    prop = new CholeskyProperties(static_cast<ProcRank>(numProc), N, v,
                                  grid[0], grid[1], grid[2]);

    // get the processor information
    proc = new Processor(prop);

    // create and commit new type for subtiles
    MPI_Type_vector(prop->v, prop->l, prop->v, MPI_DOUBLE, &MPI_SUBTILE);
    MPI_Type_commit(&MPI_SUBTILE);
    MPI_Type_vector(1, prop->v*prop->l , prop->PX * prop->v * prop->l, MPI_DOUBLE, &MPI_SUBTILE_RECV);
    MPI_Type_commit(&MPI_SUBTILE_RECV);

    // create new CholeksyIO object 
    io = new CholeskyIO(prop, proc);

    // set path to matrix for this dimension
    std::stringstream tmp;
    tmp << "../data/input_" << N << ".bin";
    pathToMatrix = tmp.str();

    // create input matrix and dump it (only in debug mode)
    io->generateInputMatrixDistributed();
    #if DEBUG
    io->openFile(pathToMatrix);
    io->dumpMatrix();
    io->closeFile();
    #endif 
}

/**
 * @brief finalizes the computation and frees allocated if desired
 * 
 * @param clean flag indicating whether data buffers are to be freed
 */
void conflux::finalize(bool clean)
{
    // delete the CholeskyIO object
    delete io;

    // as the processor owns MPI communicators
    // we need to clean it before finalizing
    if (clean) {
        delete proc;
        delete prop;
    }
}

/** 
 * @brief computes the Cholesky factorization of the current A00 tile
 * @see algorithm 9 in algo description PDF
 * 
 * @param k the current iteration index (must not be modified)
 * @param world the global MPI communicator
 */
void choleskyA00(const conflux::TileIndex k, const MPI_Comm &world)
{
    //PE(choleskya00_dpotrf);
    // compute Cholesky factorization of A00 tile
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', prop->v, proc->A00, prop->v);
    //PL();
}

/**
 * @brief updates A10 tile column according to algorithm 10
 * @see algorithm 10 in algo description PDF
 * 
 * @note this function has been refactored to allow for synchronous sends,
 * which should lead to a performance increase compared to our previous, non-
 * blocking asynchronous approach. Hence, what happens in here is not congruent
 * to the descirption in algorithm 10 anymore:
 * 
 * 1.) The processor iterates over its tile-row indices in A11 to obtain
 *     representatives of A10, and over its tile-col indices in A11 to 
 *     obtain representatives of A01. These receives are only posted here and
 *     will actually take place later, as soons as the representatives have been
 *     sent.
 * 2.) The processor iterates over their its in the current iteration's A10,
 *     update the tile via a TRSM call.
 * 3.) The processor splits its updated tiles into sub-tiles and distributes
 *     them as representatives of A10 or A01 to different z-layers.
 * 
 * @param k the current iteration index (must not be modified)
 * @param world the global MPI communicator
 */
void updateA10(const conflux::TileIndex k, const MPI_Comm &world)
{
    //MPI_Barrier(world);
    // 1.) post receive statements to later receive sub-tile representatives
    //if(proc->rank == 0)std::cout << "**********" << k << std::endl;
    conflux::ProcRank firstHolder = prop->globalToLocal(k+1).p;
    conflux::ProcRank lastHolder = (prop->P % prop->Kappa-1) - 1;

    // post to later receive representatives of A10
    std::unordered_set<conflux::ProcRank> sentSet;
    for (conflux::TileIndex iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {
        // compute processor to receive from
        conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, iLoc, iLoc);
        conflux::ProcRank pSnd = (prop->globalToLocal(glob.i)).p;

        uint64_t numSenderTilesInA10 = (prop->Kappa - 1 - k) / prop->P;
        numSenderTilesInA10 += ((firstHolder <= lastHolder && pSnd >= firstHolder && pSnd <= lastHolder) || 
        (lastHolder < firstHolder && (pSnd <= lastHolder || pSnd >= firstHolder)) &&
         !(firstHolder == lastHolder + 1)) ? 1 : 0;

        // we need to figure out how much we receive from the sending processor
        // continue with next iteration if this row has a global index <= k
        // because in this case the tile is irrelevant (above the diagonal)
        if (glob.i <= k || numSenderTilesInA10 <= 0) continue;

        if(sentSet.find(pSnd) != sentSet.end()) {
            continue;
        }

        sentSet.insert(pSnd);

        //PE(updatea10_postIRecvA10);
        // receive the tile and store it in A10 receive buffer
        MPI_Request req;
        MPI_Irecv(proc->A10rcv->get(iLoc), numSenderTilesInA10, MPI_SUBTILE_RECV, pSnd, MPI_ANY_TAG,
                 world, &req);
        proc->reqUpdateA10[proc->cntUpdateA10++] = req;  
        //if(proc->rank == TESTRANK) std::cout << "Processor " << proc->px << " " <<proc->py << " wants to receive from " << pSnd << " in round " << k << std::endl;
        
        //PL();      
    }

    // post to later receive representatives of A01
    sentSet.clear();
    for (conflux::TileIndex jLoc = k / prop->PY; jLoc < proc->maxIndexA11j; ++jLoc) {
        // compute processor to receive from
        conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, jLoc, jLoc);
        conflux::ProcRank pSnd = (prop->globalToLocal(glob.j)).p;
        uint64_t numSenderTilesInA10 = (prop->Kappa - 1 - k) / prop->P;
        numSenderTilesInA10 += ((firstHolder <= lastHolder && pSnd >= firstHolder && pSnd <= lastHolder) || 
        (lastHolder < firstHolder && (pSnd <= lastHolder || pSnd >= firstHolder)) &&
         !(firstHolder == lastHolder + 1)) ? 1 : 0;

        // continue with next iteration if this col has a global index <= k,
        // because in this case the tile has already been handled.
        if (glob.j <= k) continue;

        if(sentSet.find(pSnd) != sentSet.end() || numSenderTilesInA10 <= 0) {
            continue;
        }

        sentSet.insert(pSnd);

        //PE(updatea10_postIrecvA01);
        // receive the tile and store it in A01 receive buffer
        MPI_Request req;
        MPI_Irecv(proc->A01rcv->get(jLoc), numSenderTilesInA10, MPI_SUBTILE_RECV, pSnd, MPI_ANY_TAG,
                 world, &req);
        proc->reqUpdateA10[proc->cntUpdateA10++] = req; 
        //PL();       
    } 

    // 2-3.) update local tiles, split them into sub-tiles and distribute
    // them among z-layers

    
    //MPI_Barrier(world);
    // iterate over processor's local tiles in A10

    // we use this counter as a helper to how know how much we are sending
    conflux::TileIndex iGlob;
    uint64_t tilesToBeSent = (prop->Kappa - 1 - k) / prop->P;
    //if (proc->rank == 0 && k == 15) std::cout << firstHolder << " " << lastHolder << std::endl;
                        //+ ((prop->Kappa - 1 - k - prop->P) > proc->rank ? 1 : 0);
    tilesToBeSent += ((firstHolder <= lastHolder && proc->rank >= firstHolder && proc->rank <= lastHolder) || 
        (lastHolder < firstHolder && (proc->rank <= lastHolder || proc->rank >= firstHolder)) && 
        !(firstHolder == lastHolder + 1)) ? 1 : 0;
    tilesToBeSent = prop->Kappa - 1 <= proc->rank ? 0 : tilesToBeSent;
    for (conflux::TileIndex iLoc = k / prop->P; iLoc < proc->maxIndexA10; ++iLoc) {
     // skip tiles that were already handled or out-of-bounds
        if (iGlob <= k) continue;
        if (iGlob >= prop->Kappa) break;
        
        // update tile in A10 by solving X*A=B system for X where A = A00
        // is an upper triangular matrix and B = A10. Result is written
        // back to B, i.e. into the A10 tile.
        double *tile = proc->A10->get(iLoc);
        //PE(updatea10_dtrsm);
        cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    prop->v, prop->v, 1.0, proc->A00, prop->v, tile, prop->v);
        //PL();
    }

    // we make use of the fact that this value does not change over the course of the full run
    // but for when it fails the boundary checks
    //if (sendcounter != tilesToBeSent) {
        //std::cout << "Tiles do not match: counter " << sendcounter << " tilestobesent " << tilesToBeSent << " rank " << proc->rank << " round " << k << std::endl;
   // }
    iGlob = prop->localToGlobal(proc->rank, k/prop->P);// % prop->P;
    //std::cout << iGlob << std::endl;
    conflux::ProcIndexPair2D tileOwners = prop->globalToLocal(iGlob, iGlob);
    double *tile = proc->A10->get(k/prop->P);
        // send tile synchronously as representative of A10 to all processors 
        // that own tile-rows with index iGlob, split into subtiles among Z-layer
    if (tilesToBeSent > 0) {
        for (conflux::ProcCoord pyRcv = 0; pyRcv < prop->PY; ++pyRcv) {
            for (conflux::ProcCoord pzRcv = 0; pzRcv < prop->PZ; ++pzRcv) {
                //PE(updatea10_sendA10);
                conflux::ProcRank pRcv = prop->gridToGlobal(tileOwners.px, pyRcv, pzRcv);
                //if(proc->rank == TESTRANK) std::cout << "Processor " <<  proc->rank << " to processor " << pRcv<< std::endl;
                MPI_Ssend(tile + prop->l * pzRcv, tilesToBeSent, MPI_SUBTILE, pRcv, iGlob, world);
                //std::cout << "Processor " << pRcv<< " will be sent " << tilesToBeSent << " from processor " << proc->rank  << " in round " << k << std::endl;

                //PL();
            }
        }

        // send tile synchronously as representative of A01 to all processors 
        // that own tile-cols with index iGlob, split into subtiles along Z-layer
        for (conflux::ProcCoord pxRcv = 0; pxRcv < prop->PX; ++pxRcv) {
            for (conflux::ProcCoord pzRcv = 0; pzRcv < prop->PZ; ++pzRcv) {
                //PE(updatea10_sendA01);
                 conflux::ProcRank pRcv = prop->gridToGlobal(pxRcv, tileOwners.py, pzRcv);
                //if (proc->rank == 0) {
                    //std::cout << "Processor " << proc->rank << " sends " << sendCounter << " to processor " << pRcv << std::endl;
                //}
                MPI_Ssend(tile + prop->l * pzRcv, tilesToBeSent, MPI_SUBTILE, pRcv, iGlob, world);
                //PL();        
            }
        }
    }

    //MPI_Barrier(world);
    // wait until all the data transfers have been completed
    // @ TODO: investigate if this wait all is still necessary
    //PE(updatea10_waitall);
    if (proc->cntUpdateA10 > 0) {
        //std::cout << "processor " << proc->rank << " before waitall in round " << k << std::endl;
        //MPI_Barrier(world);
        //std::cout << "processor " << proc->rank << " before waitall in round " << k << std::endl;
        MPI_Waitall(proc->cntUpdateA10, &(proc->reqUpdateA10[0]), MPI_STATUSES_IGNORE);
        //std::cout << "processor " << proc->rank << " after waitall in round " << k << std::endl;
    }
    //MPI_Barrier(world);
    //PL();
}

/**
 * @brief updates the processor's individual copies of tiles in A11
 * 
 * @note This function has been updated to allow for synchronous sends for performance
 * reasons. Hence, in contrast to algorithm 11 in the algo description PDF 
 * document, the receiving already took place in updateA10(). This function thus
 * solely performs the actual tile "low-rank" updates.
 * 
 * @see algorithm 11 in algo description PDF
 * @param k the current iteration index (must not be modified)
 * @param world the global MPI communicator
 */
void computeA11(const conflux::TileIndex k, const MPI_Comm &world)
{
    // iterate over all tiles below (inclusive) the diagonal that this processor
    // owns and update them via low-rank update.
    // TODO: algo descriptions says that indices start from k/P, which imo is wrong (saethrej)
    for (conflux::TileIndex iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {
        for (conflux::TileIndex jLoc = k / prop->PY; jLoc <= iLoc && jLoc < proc->maxIndexA11j; ++jLoc) {
            // compute global index and skip tile if at least one index is <= k
            conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, iLoc, jLoc);
            if (glob.i <= k || glob.j > glob.i || glob.j <= k) continue;

            // perform "low-rank" update (A11 <- A11 - A10 * A01^T)
            //PE(computea10_dgemm);
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,  // DGEMM information
                prop->v, prop->v, prop->l,                // dimension information
                -1.0, proc->A10rcv->get(iLoc), prop->l,   // information about A10 rep
                proc->A01rcv->get(jLoc), prop->l,         // information about A01 rep
                1.0, proc->A11->get(iLoc, jLoc), prop->v  // information about A11 tile to be updated
            );  
            //PL();
        }
    }       
}

/**
 * @brief reduces the current tile column (at index k+1).
 * 
 * In this part of the algorithm, the tile column at index (k+1) is reduced,
 * i.e. the corresponding versions of a tile that exist on processors along
 * one z-axis (i.e. these processors share px and py coordinates) are sent
 * to the processor in this group that lies on the reduction layer, which 
 * adds all the tile versions together and thus prepares them to be scattered
 * in the next step.
 * 
 * @see algorithm 12 in algo description PDF
 * @param k the current iteration index (must not be modified)
 * @param world the global MPI communicator
 */
void reduceA11(const conflux::TileIndex k, const MPI_Comm &world)
{
    // fix the index of the tile-column index to be reduced
    conflux::TileIndex jLoc = prop->globalToLocal(k+1, k+1).j;

    // only parts of all processors take part in the reduction
    // in particular, we only consider processors with certain y-coordinates for sending and receiving
    // the receiving, reduction executing processors will will have y and z coordinates
    // (k+1) % PY, (k+1) % PZ respectively
    conflux::ProcCoord pyRed = prop->globalToLocal(k+1, k+1).py;
    if (proc->py == pyRed) {

        conflux::ProcRank recvProcessorRank = prop->gridToGlobal(proc->px, proc->py, (k + 1) % prop->PZ);

        // maybe check if this is the correct loop boundary
        for (conflux::TileIndex iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {

            conflux::TileIndices globalIndices = prop->localToGlobal(proc->px, proc->py, iLoc, jLoc);
            // we dont care about old indices
            if (globalIndices.i <= k)  continue;

            // this process actually performs the reduction (and thus in place)
            //PE(reducea11_reduction);
            if (proc->rank == recvProcessorRank) {
                MPI_Reduce(MPI_IN_PLACE, proc->A11->get(iLoc, jLoc), prop->vSquare,
                       MPI_DOUBLE, MPI_SUM, (k+1) % prop->PZ, proc->zAxisComm);//, &req);

            // all other processes only send data    
            } else {
                MPI_Reduce(proc->A11->get(iLoc, jLoc), proc->A11->get(iLoc, jLoc), prop->vSquare,
                       MPI_DOUBLE, MPI_SUM, (k+1) % prop->PZ, proc->zAxisComm);//, &req);
            }
            //PL();
        }
    }

}

/**
 * @brief scatters tile-col (k+1) to all processors as A10 for next iteration
 * 
 * @note This function has been refactored to allow for synchronous sends, which
 * should lead to an increase in performance. Thus, the function now performs the
 * following steps:
 * 
 * 1.) the processor posts the receive statements for tiles in the next iteration's
 *     A00 that were reduced just before in reduceA11()
 * 2.) If the processor participated actively in the reduction (i.e. performing it),
 *     then it distributes the tiles of the new A10 that it currently owns, as the 
 *     first tile-column of A11, to the corresponding processors. If the processor
 *     owns the new A00, this tile is broadcasted
 * 
 * @see algorithm 13 in algo description PDF
 * @param k current iteration index (must not be changed)
 * @param world the global communicator
 */
void scatterA11(const conflux::TileIndex k, const MPI_Comm &world)
{
    // post receive statements for tiles to be scattered in this function, which
    // will be the tiles in next iteration's A10
    for (conflux::TileIndex iLocRecv = (k+1)/prop->P; iLocRecv < proc->maxIndexA10; iLocRecv++) {
        conflux::TileIndex iGlobRecv = prop->localToGlobal(proc->rank, iLocRecv);
        if (iGlobRecv <= k+1) continue;
        if (iGlobRecv >= prop->Kappa) break;
         // receive tile from A10 from scattering procedure
        // the indices match the sends postet in the same round 
        //PE(scattera11_postIrecv);
        conflux::ProcIndexPair2D owners = prop->globalToLocal(iGlobRecv, k+1);
        conflux::ProcCoord zOwner = static_cast<conflux::ProcCoord>((k + 1) % prop->PZ);
        conflux::ProcRank senderProc = prop->gridToGlobal(owners.px, owners.py, zOwner);
        MPI_Request req;
        MPI_Irecv(proc->A10->get(iLocRecv), prop->vSquare, MPI_DOUBLE, senderProc,
                    iLocRecv, world, &req);
        proc->reqScatterA11[proc->cntScatterA11++] = req;
        //PL();
    }

    // we need to extract which processor owns the A00 tile of this round
    conflux::ProcIndexPair2D rootProcessorPair = prop->globalToLocal(k + 1, k + 1);
    conflux::ProcRank rootProcessorRank = prop->gridToGlobal(rootProcessorPair.px, rootProcessorPair.py, (k + 1) % prop->PZ);
    // processor that owns next A00 has to copy the tile into its A00 buffer
    if (proc->rank == rootProcessorRank) {
        std::memcpy(
            proc->A00,
            proc->A11->get(rootProcessorPair.i, rootProcessorPair.j),
            prop->vSquare * sizeof(double)
        );
    }

    // only processors that participated actively in the reduction are scattering
    conflux::ProcCoord pyScat = prop->globalToLocal(k+1, k+1).py;
    if (proc->pz == (k + 1) % prop->PZ && proc->py == pyScat) {
        conflux::TileIndex jLoc = prop->globalToLocal(k+1, k+1).j;

        // is this loop boundary correct?
        for (conflux::ProcCoord iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {

            // compute global index and skip or break loop if limits are exceeded
            conflux::TileIndices globalTile = prop->localToGlobal(proc->px, proc->py, iLoc, jLoc);

            // break if global index is too large, and skip new A00 and too small global indices
            if (globalTile.i >= prop->Kappa) break; 
            if ((globalTile.i == k + 1 && globalTile.j == k + 1) || globalTile.i < k + 1 ) continue;

            //PE(scattera11_senda10);
            // send the A11 tiles that become A10 tiles in the next round
            conflux::ProcIndexPair1D A10pair = prop->globalToLocal(globalTile.i);
            MPI_Ssend(proc->A11->get(iLoc, jLoc), prop->vSquare, MPI_DOUBLE,
                     A10pair.p, A10pair.i, world);
            //PL();
        }
    }

    // compute how many tiles remain, and create a communicator that only broadcasts
    // the new A00 to the processors that actually need it.
    /*
    conflux::ProcRank maxProc = prop->Kappa - k - 1;
    int color = proc->rank < maxProc || proc->rank == rootProcessorRank ? 1 : 0;

    MPI_Comm bcastComm;
    // make sure the root processor rank has the largest id for easy identification
    if (proc->rank == rootProcessorRank) {
        MPI_Comm_split(world, color, maxProc+5, &bcastComm);
    } else {
        MPI_Comm_split(world, color, proc->rank, &bcastComm);
    }

    // only let processors whose color is 1 participate in the broadcast
    if (color == 1) {
        int newRootRank;
        MPI_Comm_size(bcastComm, &newRootRank);
        MPI_Bcast(proc->A00, prop->vSquare, MPI_DOUBLE, newRootRank-1, bcastComm);
    }
    */
    //std::cout << tmp.str() << std::flush;
    //PE(scattera11_bcast);
    //MPI_Request req;
    //if (proc->inBcastComm ) {
        // compute new rank of root processor
           // conflux::GridProc rootCord = prop->globalToGrid(rootProcessorRank);
           // int newRoot = rootCord.px + rootCord.pz * prop->PX;
            // broadcast in the new communicator
    MPI_Bcast(proc->A00, prop->vSquare, MPI_DOUBLE, rootProcessorRank, world);//, &req);
   // }

    //else if(!prop->smallerBroadcast) {
     //   MPI_Bcast(proc->A00, prop->vSquare, MPI_DOUBLE, rootProcessorRank, world);//, &req);
    //}

    //PL();
    //MPI_Bcast(proc->A00, prop->vSquare, MPI_DOUBLE, rootProcessorRank, world); //, &req);
    //proc->reqScatterA11[proc->cntScatterA11++] = req;
    //std::cout << tmp2.str() << std::flush;


    // wait for the scattering to be completed
    // @TODO investigate if this still needed (maybe blocking broadcast)
    //PE(scattera11_waitall);
    MPI_Waitall(proc->cntScatterA11, &(proc->reqScatterA11[0]), MPI_STATUSES_IGNORE);
    //if (proc->inBcastComm) {
    //    MPI_Wait(&req, MPI_STATUS_IGNORE);
    //}
    //PL();
    
    // currently we need a barrier here for some reason
    //PE(scattera11_barrier);
    //MPI_Barrier(world);
    //PL();
}

/** 
 * @brief computes the Cholesky factorization of A 
 * 
 * This function computes the Cholesky faintctorization of A, i.e. it returns a
 * lower triangular matrix L with A = LL^T. Note that the input matrix A must
 * be symmetric positive definite (spd).
 * Moreover, this function assumes that both the MPI environment was initialized
 * and the input matrix A was distributed and already resides in the correct
 * buffers at the processors.
 * 
 * @note This function will NOT return the result matrix. This is handled by a 
 * separate function call.
 * 
 * @pre The MPI environment was initialized, optimal execution parameters and 
 * processor grid was created, and the input matrix was distributed and exists
 * in the designated buffers on all the processors.
 * @post The matrix L is computed and exists, distributed among all the pro-
 * cessors on the grid. This distribution is known, i.e. the matrix can be
 * reconstructed entirely from the processors.
 */
void conflux::parallelCholesky()
{
    // create shortcut for MPI_COMM_WORLD
    MPI_Comm world = MPI_COMM_WORLD;
    
    // in debug mode, write the matrix back into a file in every round
    #ifdef DEBUG
    std::stringstream tmp;
    tmp << "../data/output_" << prop->N << ".bin";
    io->openFile(tmp.str());
    #endif //DEBUG
    /********************** START OF THE FACTORIZATION ***********************/

    // We perform the factorization tile-column-wise, hence loop over tile cols
    for (TileIndex k = 0; k < prop->Kappa; ++k) {
        //std::cout << k << std::endl;
        // reset the request counters
        proc->cntUpdateA10 = 0;
        proc->cntScatterA11 = 0;

        /************************ (1) CHOLESKY OF A00 ************************/
        //std::cout << tmp.str() << std::endl;
        choleskyA00(k, world);

        // return if this was the last iteration
        if (k == prop->Kappa - 1){
            // ... and dump the last A00 in DEBUG mode
            #ifdef DEBUG
            if (proc->rank == 0) {
                io->dumpSingleTileColumn(k);
            }
            io->closeFile();
            #endif // DEBUG
            return;
        }

        /************************ (2) UPDATE A10 *****************************/
        //std::cout << "Rank " << proc->rank << " started updateA10 in round " << k << std::endl;
        updateA10(k, world);

        // dump current tile column in DEBUG mode
        #ifdef DEBUG
        io->dumpSingleTileColumn(k);
        #endif // DEBUG

        /************************ (3) COMPUTE A11 ****************************/
        //std::cout << "Rank " << proc->rank << " started computeA11 in round " << k << std::endl;
        computeA11(k, world);

        /************************ (4) REDUCE A11 *****************************/
        //std::cout << "Rank " << proc->rank << " started reduceA11 in round " << k << std::endl;
        reduceA11(k, world);

        /************************ (5) SCATTER A10, A00 ***********************/


        // update the broadcast communicator if possible. In iteration k, there are
        // Kappa - k - 1 tiles in the current A10, an thus Kappa - k - 2 in the next
        // iteration for brodcasting
        if (prop->Kappa - k > 2) {
            proc->updateBroadcastCommunicator(prop->Kappa - k - 2);
        }
        scatterA11(k, world);        
    }
}
