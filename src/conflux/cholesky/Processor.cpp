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

 * @file Processor.cpp
 * 
 * @brief implementation of the Processor class
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 23.11.2020
 */

#include <sstream>
#include <iostream>
#include <mpi.h>
#include <math.h>

#include "Processor.h"

/**
 * @brief creates a Processor object and allocates all the buffers
 * 
 * This constructor creates all data structures that a processor needs through-
 * out the entirety of the algorithm. This includes basic information like
 * rank, maximum indices, but also buffers. The latter are all allocated on the
 * heap during the call of the constructor.
 * 
 * @pre The MPI environment was initialized. Otherwise, an exception is thrown.
 * @post all processor properties are computed and all buffers are allocated
 * 
 * @param prop pointer to properties of the Cholesky algorithm
 */
conflux::Processor::Processor(CholeskyProperties *prop)
{
    // check if MPI environment was already initialized, throw exception if not
    int init;
    MPI_Initialized(&init);
    if (!init) {
        throw CholeskyException(CholeskyException::errorCode::FailedMPIInit);
    }

    // get rank and grid position
    int procRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    this->rank = static_cast<ProcRank>(procRank);
    this->grid = prop->globalToGrid(this->rank);
    this->px = this->grid.px;
    this->py = this->grid.py;
    this->pz = this->grid.pz;
    this->_numProc = prop->P;
    this->_PX = prop->PX;
    this->inBcastComm = false;

    // compute maximal local tile indices for this processor
    this->maxIndexA10 = (prop->Kappa - 1) / prop->P
                        + ((prop->Kappa - 1) % prop->P > this->rank ? 1 : 0);
    this->maxIndexA11i = (prop->Kappa - 1) / prop->PX
                         + ((prop->Kappa - 1) % prop->PX > this->px ? 1 : 0);
    this->maxIndexA11j = (prop->Kappa - 1) / prop->PY 
                         + ((prop->Kappa - 1) % prop->PY > this->py ? 1 : 0);

    // allocate permanent buffers for this processor
    this->A00 = new double[prop->v * prop->v];
    this->A10 = new TileMatrix(MatrixType::VECTOR, prop->v, prop->v, this->maxIndexA10);
    this->A11 = new TileMatrix(MatrixType::MATRIX, prop->v, prop->v, this->maxIndexA11i,
                               this->maxIndexA11j);

    // allocate temporary receive buffers
    this->A10rcv = new TileMatrix(MatrixType::VECTOR, prop->v, prop->l, this->maxIndexA11i);
    this->A01rcv = new TileMatrix(MatrixType::VECTOR, prop->v, prop->l, this->maxIndexA11j);

    // set the request counters to zero
    this->cntUpdateA10 = 0;
    this->cntScatterA11 = 0;

    // compute the upper bounds for the number of requests on sub-tiles
    this->sndBound = this->maxIndexA10 * (prop->PZ * (prop->PX + prop->PY));
    this->rcvBound = this->maxIndexA11i + this->maxIndexA11j;

    // reserve memory for the MPI request vectors. Note that all these sizes are
    // upper bounds, and not exact by any means. It is only important to 
    reqUpdateA10.resize(this->rcvBound);
    reqScatterA11.resize(this->maxIndexA10 + 1);
    
    // create a new communicator for all processors along the same z-axis as
    // the current processor, i.e. processors that share (px,py) coordinates
    // we define color = px * PY + py, i.e. rank on XY-plane in row-major order
    // and rank as the pz coordinate.
    MPI_Comm_split(MPI_COMM_WORLD, this->px * prop->PY + this->py, this->pz, &this->zAxisComm);


        // generate the broadcast commnicators
    uint32_t nextPowOf2 = 1 << (uint32_t) floor(log2(this->_numProc));
    uint32_t numTiles = prop->Kappa - 2;

    // if there are less tiles than the remaining number of processors, only use
    // at most twice as many processors for the broadcast as necessary
    if (numTiles != 0 && numTiles < nextPowOf2) {
        while (numTiles <= nextPowOf2 / 2) {
            nextPowOf2 /= 2;
        }
        _bcastSizes.push_back(nextPowOf2);
        MPI_Comm first;
        if (this->px == this->py) {
            MPI_Comm_split(MPI_COMM_WORLD, 1, this->px + (this->pz*this->_PX), &first);
            this->inBcastComm = true;
        } else {
            int color = this->rank < nextPowOf2 ? 1 : 0;
            this->inBcastComm = color ? true : false;
            MPI_Comm_split(MPI_COMM_WORLD, color, this->_numProc + this->rank, &first);
        }
        this->_bcastComms.push_back(first);
        nextPowOf2 /= 2;
    // otherwise we start with MPI_COMM_WORLD
    } else {
        MPI_Comm first = MPI_COMM_WORLD;
        this->inBcastComm = true;
        
        this->_bcastSizes.push_back(prop->P);
        this->_bcastComms.push_back(first);
        nextPowOf2 /= 2;
    }

    // generate all communicators as long as the size is at least 4
    while (nextPowOf2 >= 4) {
        this->_bcastSizes.push_back(nextPowOf2);
        //std::cout << nextPowOf2 << std::endl;
        MPI_Comm cur;
        if (this->px == this->py) {
            MPI_Comm_split(MPI_COMM_WORLD, 1, this->px + (this->pz*this->_PX), &cur);
        } else {
            int color = this->rank < nextPowOf2 ? 1 : 0;
            MPI_Comm_split(MPI_COMM_WORLD, color, this->_numProc + this->rank, &cur);
        }
        this->_bcastComms.push_back(cur);
        nextPowOf2 /= 2;
    }
    // set the current broadcast communicator
    this->bcastComm = this->_bcastComms[0];
    this->_curBcastIdx = 0;
}

/**
 * @brief destroys a processor object and frees all buffers
 */
conflux::Processor::~Processor()
{
    // delete permanent buffers
    delete[] A00;
    delete A10;
    delete A11;

    // delete temporary receive buffers
    delete A10rcv;
    delete A01rcv;

    // the processor with rank 0 within its local z-axis communicator (i.e. 
    // the one with pz=0) has to free the communicator at the end of the
    // execution.
    if (this->pz == 0) {
        MPI_Comm_free(&this->zAxisComm);
    }
}

/**
 * @brief updates the broadcast communicator and lets each processor 
 * determine whether it's still part of it.
 * 
 * @param rem the number or tiles that are remaining
 */
void conflux::Processor::updateBroadcastCommunicator(TileIndex rem)
{
    // return if a reduction of the broadcast size is not possible (anymore)
    if (_curBcastIdx == _bcastComms.size()-1 || rem > _bcastSizes[_curBcastIdx+1]) {
        return;
    }

    // otherwise it's possible, and we update the bcastComm and the flag
    // indicating whether a rank belongs to the communicator or not
    _curBcastIdx++;
    bcastComm = _bcastComms[_curBcastIdx];
    if(rank == 0) {
        std::cout << _bcastSizes[_curBcastIdx] << std::endl;
    }
    inBcastComm = (rank < _bcastSizes[_curBcastIdx] || px == py) ? true : false;

}
