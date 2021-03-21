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

 * @file Processor.h
 * 
 * @brief declaration of class that stores properties and all buffers of
 * a particular processor that participates in the algorithm
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 22.11.2020
 */

#ifndef PROCESSOR_H
#define PROCESSOR_H


#include <vector>
#include <mpi.h>

#include "CholeskyProperties.h"
#include "CholeskyTypes.h"
#include "TileMatrix.h"

namespace conflux {
    
/**
 * @brief an instance for a processor storing its properties and all its buffers
 */
class Processor 
{
public:

    // constructors and destructors
    Processor(CholeskyProperties *prop);
    ~Processor();

    // function to change to a new broadcast communicator
    void updateBroadcastCommunicator(TileIndex tilesRemaining);

    // basic processor information
    ProcRank rank; //!< global rank within MPI_COMM_WORLD
    GridProc grid; //!< grid coordinates as a struct
    ProcCoord px;  //!< processor's x-coordinate
    ProcCoord py;  //!< processor's y-coordinate
    ProcCoord pz;  //!< processor's z-coordinate
    TileIndex maxIndexA10;  //!< max local index in A10 for this processor
    TileIndex maxIndexA11i; //!< max local tile-row index in A11 for this proc.
    TileIndex maxIndexA11j; //!< max local tile-col index in A11 for this proc.

    // buffers for this processor
    double *A00;        //!< (v,v) buffer that stores A00 of current iteration
    TileMatrix *A10;    //!< vector of tiles (v,v) from A10
    TileMatrix *A11;    //!< matrix of (v,v) tiles from A11
    
    // receive and send buffers (temporary data) for this processor
    TileMatrix *A10rcv; //!< contains subtiles (v,l) of A10 to receive
    TileMatrix *A01rcv; //!< contains subtiles (v,l) of A01 to receive

    // More specific communicators (e.g. for reduction)
    MPI_Comm zAxisComm; //!< communicator for reduction along z-Axis
    MPI_Comm bcastComm; //!< current broadcast communicator
    bool inBcastComm; //!< flag indicating whether this rank is in current bcast comm

    // vectors to store the request objects during an iteration
    std::vector<MPI_Request> reqUpdateA10;  //!< requests during the updateA10 operation
    std::vector<MPI_Request> reqScatterA11; //!< requests during the scatter operation

    // various counters for these requests 
    int cntUpdateA10;  //!< counts the number of requests in update A10 operation
    int cntScatterA11; //!< counts the number of requests in scatter A11 operation

    // upper bounds for requests in sub-tile handling in update A10
    int sndBound; //!< upper bound for number of requests due to sending subtiles
    int rcvBound; //!< upper bound for number of requests due to receiving subtiles

private:
    // communicator handling for the broadcast
    ProcRank _numProc; //!< number of processors in the global communicator
    ProcCoord _PX; //!< the number of processors in x direction
    std::vector<MPI_Comm> _bcastComms; //!< vector of all broadcasts
    std::vector<uint32_t> _bcastSizes; //!< vector of sizes of bcasts
    uint32_t _curBcastIdx; //!< index of the current broadcast comm
};

} // namespace conflux

#endif // PROCESSOR_H
