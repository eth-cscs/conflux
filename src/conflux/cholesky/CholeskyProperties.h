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

 * @file CholeskyProperties.h
 * 
 * @brief cdeclaration of class that stores global properties for a particular
 * run of the parallel Cholesky factorization algorithm
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 22.11.2020
 */

#ifndef CHOLESKY_PROPERTIES_H
#define CHOLESKY_PROPERTIES_H

#include "CholeskyTypes.h"

namespace conflux {

/**
 * @brief class that stores properties for a run of the Cholesky factorization
 * algorithm.
 * 
 * All members are public an can be accessed directly, without the need for
 * getter and setter functions. This is to ensure faster execution due to the
 * overhead a function call would incur.
 */
class CholeskyProperties
{
public:

    // constructor and destructor
    CholeskyProperties(ProcRank numProc, uint32_t dim);
    CholeskyProperties(ProcRank numProc, uint32_t dim, uint32_t tileSize,
                       ProcCoord xGrid, ProcCoord yGrid, ProcCoord zGrid);
    ~CholeskyProperties();

    // conversion functions
    ProcRank gridToGlobal(ProcCoord px, ProcCoord py, ProcCoord pz);
    ProcRank gridToGlobal(GridProc grid);
    GridProc globalToGrid(ProcRank p);
    TileIndex localToGlobal(ProcRank p, TileIndex i);
    TileIndex localToGlobal(ProcIndexPair1D pair);
    TileIndices localToGlobal(ProcCoord px, ProcCoord py, TileIndex i, TileIndex j);
    TileIndices localToGlobal(ProcIndexPair2D pair);
    ProcIndexPair1D globalToLocal(TileIndex i);
    ProcIndexPair2D globalToLocal(TileIndex i, TileIndex j);
    ProcIndexPair2D globalToLocal(TileIndices ind);

    // public member fields
    ProcRank P;      //!< total number of processors
    ProcCoord PX;    //!< number of processors in x-direction
    ProcCoord PY;    //!< number of processors in y-direction
    ProcCoord PZ;    //!< number of processors in z-direction
    ProcRank PXY;    //!< number of processors on single XY-plane
    uint32_t v;      //!< tile size (in rows or cols)
    uint32_t vSquare; //!< number of items in a tile
    uint32_t N;      //!< input matrix dimension
    uint32_t l;      //!< number of columns in a sub-tile
    TileIndex Kappa; //!< number of tiles along row or cols-dimension of matrix
    bool smallerBroadcast; //!< this flag indicates whether or not broadcast of A00 should be sent to all processors
};

} // namespace conflux

#endif // CHOLESKY_PROPERTIES_H