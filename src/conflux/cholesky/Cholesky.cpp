/**
 * @file Cholesky.cpp
 * 
 * @brief implementation of near communication-optimal parallel Cholesky
 * factorization algorithm
 * 
 * @authors Anonymized Authors
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

#ifdef DEBUG
#include <unistd.h>
#endif

#include <mpi.h>

#ifdef __USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include "CholeskyProfiler.h"
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
MPI_Datatype MPI_SUBTILE;

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

    // create new CholeksyIO object 
    io = new CholeskyIO(prop, proc);

    // set path to matrix for this dimension
    std::stringstream tmp;
    tmp << "data/input_" << N << ".bin";
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
    // compute Cholesky factorization of A00 tile
    PE(choleskyA00_compute);
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', prop->v, proc->A00, prop->v);
    PL();
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
    // 1.) post receive statements to later receive sub-tile representatives
    // post to later receive representatives of A10
    for (conflux::TileIndex iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {
        // compute processor to receive from
        conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, iLoc, iLoc);
        conflux::ProcRank pSnd = (prop->globalToLocal(glob.i)).p;

        // continue with next iteration if this row has a global index <= k
        // because in this case the tile is irrelevant (above the diagonal)
        if (glob.i <= k) continue;
        PE(updateA10_postIrecvA10);

        // receive the tile and store it in A10 receive buffer
        MPI_Request req;
        //if(proc->rank == PINSPECT) std::cout << "Processor " << proc->rank << " with grid " << proc->px << " " << proc->py << " " << proc->pz << " receives from " << pSnd << " with loc index " << iLoc << " and with glob " << glob.i << std::endl;
        MPI_Irecv(proc->A10rcv->get(iLoc), prop->v * prop->l, MPI_DOUBLE, pSnd, glob.i,
                 world, &req);
        proc->reqUpdateA10[proc->cntUpdateA10++] = req; 
        PL();       
    }

    // post to later receive representatives of A01
    for (conflux::TileIndex jLoc = k / prop->PY; jLoc < proc->maxIndexA11j; ++jLoc) {
        // compute processor to receive from
        conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, jLoc, jLoc);
        conflux::ProcRank pSnd = (prop->globalToLocal(glob.j)).p;

        // continue with next iteration if this col has a global index <= k,
        // because in this case the tile has already been handled.
        if (glob.j <= k) continue;
        PE(updateA10_postIrecvA01);

        // receive the tile and store it in A01 receive buffer
        MPI_Request req;
        MPI_Irecv(proc->A01rcv->get(jLoc), prop->v * prop->l, MPI_DOUBLE, pSnd, glob.j,
                 world, &req);
        proc->reqUpdateA10[proc->cntUpdateA10++] = req;  
        PL();      
    }
    // 2-3.) update local tiles, split them into sub-tiles and distribute
    // them among z-layers

    // iterate over processor's local tiles in A10
    // iLocRecv is needed for the reception of the scattering
    for (conflux::TileIndex iLoc = k / prop->P; iLoc < proc->maxIndexA10; ++iLoc) {

        conflux::TileIndex iGlob = prop->localToGlobal(proc->rank, iLoc);

        // skip tiles that were already handled or out-of-bounds
        if (iGlob <= k) continue;
        if (iGlob >= prop->Kappa) break;
        PE(updateA10_dtrsm())
        
        // update tile in A10 by solving X*A=B system for X where A = A00
        // is an upper triangular matrix and B = A10. Result is written
        // back to B, i.e. into the A10 tile.
        //if (!proc->inBcastComm) {
            //std::cout << "Processor " << proc->rank << " wants to update A10 but has not received it in round " << k << " and global tile " << iGlob << std::endl;
        //}
        double *tile = proc->A10->get(iLoc);
        cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    prop->v, prop->v, 1.0, proc->A00, prop->v, tile, prop->v);
        PL();

        // determine processors that own tile-rows or -cols with index iGlob
        conflux::ProcIndexPair2D tileOwners = prop->globalToLocal(iGlob, iGlob);
        //if (iGlob == 4) std::cout << tileOwners.px << " " << tileOwners.py << std::endl;
        // send tile synchronously as representative of A10 to all processors 
        // that own tile-rows with index iGlob, split into subtiles among Z-layer
        for (conflux::ProcCoord pyRcv = 0; pyRcv < prop->PY; ++pyRcv) {
            for (conflux::ProcCoord pzRcv = 0; pzRcv < prop->PZ; ++pzRcv) {
                PE(updateA10_sendA10);
                conflux::ProcRank pRcv = prop->gridToGlobal(tileOwners.px, pyRcv, pzRcv);
                //if(pRcv == PINSPECT)std::cout << "Processor " << proc->rank << " sends to " << pRcv << " with coordinates " << tileOwners.px << " " <<pyRcv << " " << pzRcv <<  std::endl;
                MPI_Ssend(tile + pzRcv * prop->l, 1, MPI_SUBTILE, pRcv, iGlob, world);
                PL();
            }
        }

        // send tile synchronously as representative of A01 to all processors 
        // that own tile-cols with index iGlob, split into subtiles along Z-layer
        for (conflux::ProcCoord pxRcv = 0; pxRcv < prop->PX; ++pxRcv) {
            for (conflux::ProcCoord pzRcv = 0; pzRcv < prop->PZ; ++pzRcv) {
                PE(updateA10_sendA01);
                conflux::ProcRank pRcv = prop->gridToGlobal(pxRcv, tileOwners.py, pzRcv);            //std::cout << "Processor " << proc->rank << " sends to " << pRcv << std::endl;
                MPI_Ssend(tile + pzRcv * prop->l, 1, MPI_SUBTILE, pRcv, iGlob, world);          
                PL();
            }
        }
    }

    // wait until all the data transfers have been completed
    // @ TODO: investigate if this wait all is still necessary
    PE(updateA10_waitall);
    if (proc->cntUpdateA10 > 0) {
        //std::cout << "before waitall in round " << proc->rank << " " << k << std::endl;
        MPI_Waitall(proc->cntUpdateA10, &(proc->reqUpdateA10[0]), MPI_STATUSES_IGNORE);
    }
    PL();
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
    for (conflux::TileIndex iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {
        for (conflux::TileIndex jLoc = k / prop->PY; jLoc < proc->maxIndexA11j; ++jLoc) {
            // compute global index and skip tile if at least one index is <= k
            conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, iLoc, jLoc);
            if (glob.i <= k || glob.j > glob.i || glob.j <= k) continue;

            // perform "low-rank" update (A11 <- A11 - A10 * A01^T)
            PE(computeA11_dgemm);
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,  // DGEMM information
                prop->v, prop->v, prop->l,                // dimension information
                -1.0, proc->A10rcv->get(iLoc), prop->l,   // information about A10 rep
                proc->A01rcv->get(jLoc), prop->l,         // information about A01 rep
                1.0, proc->A11->get(iLoc, jLoc), prop->v  // information about A11 tile to be updated
            );
            PL();  
        }
    }       
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
void updateComputeA10(const conflux::TileIndex k, const MPI_Comm &world)
{
    // 1.) post receive statements to later receive sub-tile representatives
    // post to later receive representatives of A10
    for (conflux::TileIndex iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {
        // compute processor to receive from
        conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, iLoc, iLoc);
        conflux::ProcRank pSnd = (prop->globalToLocal(glob.i)).p;

        // continue with next iteration if this row has a global index <= k
        // because in this case the tile is irrelevant (above the diagonal)
        if (glob.i <= k) continue;
        PE(updateA10_postIrecvA10);

        // receive the tile and store it in A10 receive buffer
        MPI_Request req;
        //if(proc->rank == PINSPECT) std::cout << "Processor " << proc->rank << " with grid " << proc->px << " " << proc->py << " " << proc->pz << " receives from " << pSnd << " with loc index " << iLoc << " and with glob " << glob.i << std::endl;
        MPI_Irecv(proc->A10rcv->get(iLoc), prop->v * prop->l, MPI_DOUBLE, pSnd, glob.i,
                 world, &req);
        proc->reqUpdateA10[proc->cntUpdateA10++] = req;

        // get index information and store it in the info buffer
        conflux::TileInfo tmpInf = {conflux::TileType::TILE_A10, iLoc, proc->A10rcv->get(iLoc)};
        proc->tileInfos.push_back(tmpInf);
        PL();       
    }

    // post to later receive representatives of A01
    for (conflux::TileIndex jLoc = k / prop->PY; jLoc < proc->maxIndexA11j; ++jLoc) {
        // compute processor to receive from
        conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, jLoc, jLoc);
        conflux::ProcRank pSnd = (prop->globalToLocal(glob.j)).p;

        // continue with next iteration if this col has a global index <= k,
        // because in this case the tile has already been handled.
        if (glob.j <= k) continue;
        PE(updateA10_postIrecvA01);

        // receive the tile and store it in A01 receive buffer
        MPI_Request req;
        MPI_Irecv(proc->A01rcv->get(jLoc), prop->v * prop->l, MPI_DOUBLE, pSnd, glob.j,
                 world, &req);
        proc->reqUpdateA10[proc->cntUpdateA10++] = req;  
        
        // get index information and store it in the requst buffer
        conflux::TileInfo tmpInf = {conflux::TileType::TILE_A01, jLoc, proc->A01rcv->get(jLoc)};
        proc->tileInfos.push_back(tmpInf);
        PL();      
    }
    // 2-3.) update local tiles, split them into sub-tiles and distribute
    // them among z-layers

    // iterate over processor's local tiles in A10
    // iLocRecv is needed for the reception of the scattering
    for (conflux::TileIndex iLoc = k / prop->P; iLoc < proc->maxIndexA10; ++iLoc) {

        conflux::TileIndex iGlob = prop->localToGlobal(proc->rank, iLoc);

        // skip tiles that were already handled or out-of-bounds
        if (iGlob <= k) continue;
        if (iGlob >= prop->Kappa) break;
        PE(updateA10_dtrsm())
        
        // update tile in A10 by solving X*A=B system for X where A = A00
        // is an upper triangular matrix and B = A10. Result is written
        // back to B, i.e. into the A10 tile.
        //if (!proc->inBcastComm) {
            //std::cout << "Processor " << proc->rank << " wants to update A10 but has not received it in round " << k << " and global tile " << iGlob << std::endl;
        //}
        double *tile = proc->A10->get(iLoc);
        cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    prop->v, prop->v, 1.0, proc->A00, prop->v, tile, prop->v);
        PL();

        // determine processors that own tile-rows or -cols with index iGlob
        conflux::ProcIndexPair2D tileOwners = prop->globalToLocal(iGlob, iGlob);
        //if (iGlob == 4) std::cout << tileOwners.px << " " << tileOwners.py << std::endl;
        // send tile synchronously as representative of A10 to all processors 
        // that own tile-rows with index iGlob, split into subtiles among Z-layer
        for (conflux::ProcCoord pyRcv = 0; pyRcv < prop->PY; ++pyRcv) {
            for (conflux::ProcCoord pzRcv = 0; pzRcv < prop->PZ; ++pzRcv) {
                PE(updateA10_sendA10);
                conflux::ProcRank pRcv = prop->gridToGlobal(tileOwners.px, pyRcv, pzRcv);
                MPI_Request req;
                MPI_Isend(tile + pzRcv * prop->l, 1, MPI_SUBTILE, pRcv, iGlob, world, &req);
                proc->reqUpdateA10snd[proc->cntUpdateA10snd++] = req;
                PL();
            }
        }

        // send tile synchronously as representative of A01 to all processors 
        // that own tile-cols with index iGlob, split into subtiles along Z-layer
        for (conflux::ProcCoord pxRcv = 0; pxRcv < prop->PX; ++pxRcv) {
            for (conflux::ProcCoord pzRcv = 0; pzRcv < prop->PZ; ++pzRcv) {
                PE(updateA10_sendA01);
                conflux::ProcRank pRcv = prop->gridToGlobal(pxRcv, tileOwners.py, pzRcv);
                MPI_Request req;
                MPI_Isend(tile + pzRcv * prop->l, 1, MPI_SUBTILE, pRcv, iGlob, world, &req);  
                proc->reqUpdateA10snd[proc->cntUpdateA10snd++] = req;        
                PL();
            }
        }
    }

    // wait for any request to be completed, and then, if possible, already start with the
    // computation of one of the dgemms
    int idx, numGemms;
    while (true) {
        // wait for a request to finish, break if the idx is MPI_UNDEFINED
        MPI_Waitany(proc->cntUpdateA10, proc->reqUpdateA10.data(), &idx, MPI_STATUS_IGNORE);
        if (idx == MPI_UNDEFINED) {
            break;
        }
        // if we are in here, this means that it is a valid request that was not yet handled
        conflux::TileInfo info = proc->tileInfos[idx];

        // if the request that finished is for an A10 rep, loop over all tiles in A11 that 
        // depend on this tile, and update the flags. If a location has both inputs ready,
        // call dgemm
        if (info.type == conflux::TileType::TILE_A10){
            for (conflux::TileIndex jLoc = k / prop->PY; jLoc < proc->maxIndexA11j; ++jLoc) {

                // get the tile and set the A10 flag to true
                conflux::TileReady *tmp = proc->dgemmReadyFlags->get(info.idxLoc, jLoc);
                tmp->a10 = true;

                // compute global index and skip tile if at least one index is <= k, or above diagonal
                conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, info.idxLoc, jLoc);
                if (glob.i <= k || glob.j > glob.i || glob.j <= k) continue;

                if (!tmp->done && tmp->a01) {
                    PE(computeA11_dgemm)
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,  // DGEMM information
                        prop->v, prop->v, prop->l,                // information about dimension
                        -1.0, info.tilePtr, prop->l,              // information about A10 rep
                        proc->A01rcv->get(jLoc), prop->l,         // information about A01 rep
                        1.0, proc->A11->get(info.idxLoc, jLoc), prop->v   // information about A11 tile
                    );
                    PL();
                    tmp->done = true;
                }                  
            }
        
        // otherwise this request concerns an A01 rep. Do the opposite of the above case
        } else {
            for (conflux::TileIndex iLoc = k / prop->PX; iLoc < proc->maxIndexA11i; ++iLoc) {

                // get the tile and set the A01 flag to true
                conflux::TileReady *tmp = proc->dgemmReadyFlags->get(iLoc, info.idxLoc);
                tmp->a01 = true;

                // compute global index and skip tile if at least one index is <= k, or above diagonal
                conflux::TileIndices glob = prop->localToGlobal(proc->px, proc->py, iLoc, info.idxLoc);
                if (glob.i <= k || glob.j > glob.i || glob.j <= k) continue;

                if (!tmp->done && tmp->a10) {
                    PE(computeA11_dgemm)
                    cblas_dgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,  // DGEMM information
                        prop->v, prop->v, prop->l,                // information about dimension
                        -1.0, proc->A10rcv->get(iLoc), prop->l,   // information about A10 rep
                        info.tilePtr, prop->l,                    // information about A01 rep
                        1.0, proc->A11->get(iLoc, info.idxLoc), prop->v   // information about A11 tile
                    );
                    PL();
                    tmp->done = true;
                }
            }
        }
     }

    // set everything to zero for the next iteration. or simplicity, do the entire buffer
    PE(computeA11_memset);
    std::memset(
        proc->dgemmReadyFlags->get(0,0), 
        0x00, 
        sizeof(conflux::TileReady) * proc->maxIndexA11i * proc->maxIndexA11j
    );
    proc->tileInfos.clear();
    PL();

    // finally, wait for the send requests to complete
    MPI_Waitall(proc->cntUpdateA10snd, proc->reqUpdateA10snd.data(), MPI_STATUSES_IGNORE);
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
    // return immediately if there is only a single z-layer
    if (prop->PZ == 1) return;

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

            PE(reduceA11_reduction);
            // this process actually performs the reduction (and thus in place)
            if (proc->rank == recvProcessorRank) {
                MPI_Reduce(MPI_IN_PLACE, proc->A11->get(iLoc, jLoc), prop->vSquare,
                       MPI_DOUBLE, MPI_SUM, (k+1) % prop->PZ, proc->zAxisComm);//, &req);

            // all other processes only send data    
            } else {
                MPI_Reduce(proc->A11->get(iLoc, jLoc), proc->A11->get(iLoc, jLoc), prop->vSquare,
                       MPI_DOUBLE, MPI_SUM, (k+1) % prop->PZ, proc->zAxisComm);//, &req);
            }
            PL();
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
        PE(scatterA11_postIrecv);
        conflux::ProcIndexPair2D owners = prop->globalToLocal(iGlobRecv, k+1);
        conflux::ProcCoord zOwner = static_cast<conflux::ProcCoord>((k + 1) % prop->PZ);
        conflux::ProcRank senderProc = prop->gridToGlobal(owners.px, owners.py, zOwner);
        MPI_Request req;
        MPI_Irecv(proc->A10->get(iLocRecv), prop->vSquare, MPI_DOUBLE, senderProc,
                    iLocRecv, world, &req);
        proc->reqScatterA11[proc->cntScatterA11++] = req;
        PL();
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

            // send the A11 tiles that become A10 tiles in the next round
            PE(scatterA11_sendNewA10);
            conflux::ProcIndexPair1D A10pair = prop->globalToLocal(globalTile.i);
            MPI_Ssend(proc->A11->get(iLoc, jLoc), prop->vSquare, MPI_DOUBLE,
                     A10pair.p, A10pair.i, world);
            PL();
        }
    }
    //MPI_Barrier(world);
    // brodcast and receive A00 tile for next iteration
    PE(scatterA11_bcast);
    MPI_Request req;
    if (proc->inBcastComm) {
        conflux::GridProc rootCord = prop->globalToGrid(rootProcessorRank);
        int newRoot = proc->isWorldBroadcast ? rootProcessorRank : rootCord.px + prop->PX * rootCord.pz;
        MPI_Bcast(proc->A00, prop->vSquare, MPI_DOUBLE, newRoot, proc->bcastComm);
    }
    //proc->reqScatterA11[proc->cntScatterA11++] = req;
    PL();

    // wait for the scattering to be completed
    // @TODO investigate if this still needed (maybe blocking broadcast)
    PE(scatterA11_waitall);
    MPI_Waitall(proc->cntScatterA11, &(proc->reqScatterA11[0]), MPI_STATUSES_IGNORE);
    PL();

    //MPI_Barrier(world);
}

/** 
 * @brief computes the Cholesky factorization of A 

 * @note This is the version with overlapping dgemm
 * 
 * @pre The MPI environment was initialized, optimal execution parameters and 
 * processor grid was created, and the input matrix was distributed and exists
 * in the designated buffers on all the processors.
 * @post The matrix L is computed and exists, distributed among all the pro-
 * cessors on the grid. This distribution is known, i.e. the matrix can be
 * reconstructed entirely from the processors.
 */
void _parallelCholesky1()
{
    // create shortcut for MPI_COMM_WORLD
    MPI_Comm world = MPI_COMM_WORLD;
    
    // in debug mode, write the matrix back into a file in every round
    #ifdef DEBUG
    std::stringstream tmp;
    tmp << "data/output_" << prop->N << ".bin";
    io->openFile(tmp.str());
    #endif //DEBUG
    /********************** START OF THE FACTORIZATION ***********************/

    // We perform the factorization tile-column-wise, hence loop over tile cols
    for (conflux::TileIndex k = 0; k < prop->Kappa; ++k) {
        // reset the request counters
        proc->cntUpdateA10 = 0;
        proc->cntScatterA11 = 0;
        proc->cntUpdateA10snd = 0;

        /************************ (1) CHOLESKY OF A00 ************************/
        // we only need to compute the cholesky factorization if 
        if (proc->inBcastComm) {
            choleskyA00(k, world);
        }

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

        /******************* (2-3) UPDATE A10, COMPUTE A11 *******************/
        updateComputeA10(k, world);

        // dump current tile column in DEBUG mode
        #ifdef DEBUG
        io->dumpSingleTileColumn(k);
        #endif // DEBUG

        /************************ (4) REDUCE A11 *****************************/
        reduceA11(k, world);

        if (prop->Kappa-k > 2) {
            proc->updateBcastComm(prop->Kappa - k - 2);
        }

        /************************ (5) SCATTER A10, A00 ***********************/
        scatterA11(k, world);
    }
}

/** 
 * @brief computes the Cholesky factorization of A 

 * @note This is the version without overlapping.
 * 
 * @pre The MPI environment was initialized, optimal execution parameters and 
 * processor grid was created, and the input matrix was distributed and exists
 * in the designated buffers on all the processors.
 * @post The matrix L is computed and exists, distributed among all the pro-
 * cessors on the grid. This distribution is known, i.e. the matrix can be
 * reconstructed entirely from the processors.
 */
void _parallelCholesky2()
{
    // create shortcut for MPI_COMM_WORLD
    MPI_Comm world = MPI_COMM_WORLD;
    
    // in debug mode, write the matrix back into a file in every round
    #ifdef DEBUG
    std::stringstream tmp;
    tmp << "data/output_" << prop->N << ".bin";
    io->openFile(tmp.str());
    #endif //DEBUG
    /********************** START OF THE FACTORIZATION ***********************/

    // We perform the factorization tile-column-wise, hence loop over tile cols
    for (conflux::TileIndex k = 0; k < prop->Kappa; ++k) {
        // reset the request counters
        proc->cntUpdateA10 = 0;
        proc->cntScatterA11 = 0;

        /************************ (1) CHOLESKY OF A00 ************************/
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
        updateA10(k, world);

        // dump current tile column in DEBUG mode
        #ifdef DEBUG
        io->dumpSingleTileColumn(k);
        #endif // DEBUG

        /************************ (3) COMPUTE A11 ****************************/
        computeA11(k, world);

        /************************ (4) REDUCE A11 *****************************/
        reduceA11(k, world);

        if (prop->Kappa-k > 2) {
            proc->updateBcastComm(prop->Kappa - k - 2);
        }

        /************************ (5) SCATTER A10, A00 ***********************/
        scatterA11(k, world);
    }
}


void conflux::parallelCholesky()
{
    // decide whether to overlap or not
    switch (prop->P) {
        case 4:
            if (prop->N >= 1<<16) {
                _parallelCholesky2(); // non-overlapping version
            } else {
                _parallelCholesky1(); // overlapping version
            }
            break;

        case 8:
            if (prop->N >= 1<<16) {
                _parallelCholesky2(); // non-overlapping version
            } else {
                _parallelCholesky1(); // overlapping version
            }
            break;

        case 16:
            if (prop->N == 1<<13 || prop->N == 1<<16) {
                _parallelCholesky1(); // overlapping version
            } else {
                _parallelCholesky2(); // non-overlapping version
            }
            break;

        case 32:
            if (prop->N >= 1<<17) {
                _parallelCholesky2(); // non-overlapping version
            } else {
                _parallelCholesky1(); // overlapping version
            }
            break;

        case 64:
            if (prop->N >= 1<<18) {
                _parallelCholesky1(); // overlapping version
            } else {
                _parallelCholesky2(); // non-overlapping version
            }
            break;

        case 128:
            if (prop->N == 1<<14) {
                _parallelCholesky2(); // non-overlapping version
            } else {
                _parallelCholesky1(); // overlapping version
            }
            break;

        case 256:
            if (prop->N >= 1<<18) {
                _parallelCholesky1(); // overlapping version
            } else {
                _parallelCholesky2(); // non-overlapping version
            }
            break;

        case 512:
            _parallelCholesky2(); // non-overlapping version
            break;

        case 1024:
            _parallelCholesky2(); // non-overlapping version
            break;
        
        default:
            _parallelCholesky1(); // overlapping version
            break;
    }
}
