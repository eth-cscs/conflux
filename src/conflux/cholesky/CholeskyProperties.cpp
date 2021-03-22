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

 * @file CholeskyProperties.cpp
 * 
 * @brief implementation of the CholeskyProperties class
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 22.11.2020
 */

#include <math.h>
#include "CholeskyProperties.h"

/**
 * @brief creates a CholeskyProperties instance
 * 
 * @param numProc the number of processors in the current MPI environment
 * @param matrixDim the dimension of the s.p.d. input matrix
 */
conflux::CholeskyProperties::CholeskyProperties(ProcRank numProc, uint32_t matrixDim)
{
    // throw an exception if there are less than 4 processors
    if (numProc < 4) {
        throw CholeskyException(CholeskyException::errorCode::InvalidProcessorNumber);
    }

    // TODO: create logic here to compute optimal grid out of available proc.
    this->N = matrixDim;
    this->P = 8;
    this->PX = 2;
    this->PY = 2;
    this->PZ = 2;
    this->PXY = this -> PX * this->PY;
    this->v = 32;
    this->l = this->v / this->PZ;
    this->Kappa = static_cast<TileIndex>(ceil((double) this->N / this->v));
}


/**
 * @brief creates a CholeskyProperties instance with settings specified by user
 * 
 * @param numProc number of processors in current MPI environment
 * @param dim the matrix dimension
 * @param tileSize size of tiles in matrix
 * @param xGrid number of processors in x-direction
 * @param yGrid number of processors in y-direction
 * @param zGrid number of processors in z-direction
 */
conflux::CholeskyProperties::CholeskyProperties(ProcRank numProc, uint32_t dim, uint32_t tileSize,
                       ProcCoord xGrid, ProcCoord yGrid, ProcCoord zGrid)
{
    // throw an exception if there are less than 4 processors or if grid
    // does not match number of processors
    if (numProc < 4) {
        throw CholeskyException(CholeskyException::errorCode::InvalidProcessorNumber);
    }
        
    if (xGrid * yGrid * zGrid != numProc) {
        throw CholeskyException(CholeskyException::errorCode::InvalidGridSize);
    }

    if (xGrid != yGrid) {
        throw CholeskyException(CholeskyException::errorCode::UnmatchingXYSizes);
    }

    if (tileSize % zGrid != 0) {
        throw CholeskyException(CholeskyException::errorCode::TilesNotDivisibleByZ);
    }

    // set parameters according to the specification by the user
    // TODO: if v is not evenly divisble by PZ, this yields a problem
    this->N = dim;
    this->P = numProc;
    this->PX = xGrid;
    this->PY = yGrid;
    this->PZ = zGrid;
    this->PXY = this->PX * this->PY;
    this->v = tileSize;
    this->vSquare = tileSize * tileSize;
    this->l = this->v / this->PZ;
    this->Kappa = static_cast<TileIndex>(ceil((double) this->N / this->v));

}

/**
 * @brief destroys a CholeskyProperties instance
 */
conflux::CholeskyProperties::~CholeskyProperties()
{
    // nothing to do here.
}

/**
 * @brief converts processor grid coordinates to global processor rank
 *
 * This function returns a processor the global coordinate from its grid
 * coordinate representation, i.e. for p = (px, py, pz) and inputs px, py,
 * and pz it returns p. The conversion is computed as follows:
 *              p = px + py * PY + pz * PXY
 * This conversion function induces an ordering on the processors in the grid,
 * e.g. (0,0,0) < (1,0,0) < (0,1,0) < (1,1,0) < (0,0,1) < (1,0,1) < (0,1,1) ...
 * 
 * @param px the processor's x-coordinate on the grid
 * @param py the processor's y-coordinate on the grid
 * @param pz the processor's z-coordinate on the grid
 * @return the processor's global rank p
 */ 
conflux::ProcRank conflux::CholeskyProperties::gridToGlobal(ProcCoord px, ProcCoord py, ProcCoord pz)
{
    return px + py * this->PY + pz * this->PXY;
}

/**
 * @brief converts processor grid coordinates to global processor rank
 * @overload ProcRank gridToGlobal(ProcCoord px, ProcCoord py, ProcCoord pz)
 * 
 * @param grid the processor's coordinates on the grid
 * @return the processor's global rank p
 */
conflux::ProcRank conflux::CholeskyProperties::gridToGlobal(GridProc grid)
{
    return grid.px + grid.py * this->PY + grid.pz * this->PXY;
}

/**
 * @brief convers global processor coordinates to grid coordinates
 * 
 * This function is the inverse function of <grid2global>"()".
 * 
 * @param p the global processor coordinate
 * @return the processor's grid coordinates as a struct
 */
conflux::GridProc conflux::CholeskyProperties::globalToGrid(ProcRank p)
{
    ProcCoord pz = p / this->PXY;
    p -= pz * this->PXY;
    ProcCoord py = p / this->PY;
    ProcCoord px = p % this->PY;
    return GridProc{px, py, pz};
}

/**
 * @brief converts a local tile index in A10 to a global tile index
 * 
 * @param p the global processor coordinate
 * @param i local tile index of the tile under consideration
 * @return the global tile index of the tile 
 */
conflux::TileIndex conflux::CholeskyProperties::localToGlobal(ProcRank p, TileIndex i)
{
    return i * this->P + p + 1;
}

/**
 * @brief converts a local tile index in A10 to a global tile index
 * 
 * @param pair processor-local-index pair of tile
 * @return the global tile index 
 */
conflux::TileIndex conflux::CholeskyProperties::localToGlobal(ProcIndexPair1D pair)
{
    return pair.i * this->P + pair.p + 1;
}

/**
 * @brief converts local indices for tile in A11 owned by (px,py) to global indices
 * 
 * @param px processor's x-coordinate on grid
 * @param py processor's y-coordinate on grid
 * @param i local tile-row index (for A11)
 * @param j local tile-column index (for A11)
 * @return global tile indices (pair)
 */
conflux::TileIndices conflux::CholeskyProperties::localToGlobal(ProcCoord px, ProcCoord py, TileIndex i, TileIndex j)
{
    return TileIndices{this->PX * i + px + 1, this->PY * j + py + 1};
}

/**
 * @brief converts local indices for tile in A11 owned by (px,py) to global indices
 * 
 * @param pair 2d pair of processor and local tile index
 * @return global tile indices (pair)
 */
conflux::TileIndices conflux::CholeskyProperties::localToGlobal(ProcIndexPair2D pair)
{
    return TileIndices{this->PX * pair.i + pair.px + 1, this->PY * pair.j + pair.py + 1};
}

/**
 * @brief computes p that owns this global tile (in A10) and corresponding 
 * local tile index
 * 
 * @param i global tile index (in A10)
 * @return pair (p, i_L) of processor p (global) and local tile index
 */
conflux::ProcIndexPair1D conflux::CholeskyProperties::globalToLocal(TileIndex i)
{
    return ProcIndexPair1D{(i-1) % this->P, (i-1) / this->P};
}

/**
 * @brief computes processor coordinates (px,py) that own this global tile in
 * A11 and their local tile indices for it
 * 
 * @param i tile-row index (for A11)
 * @param j tile-column index (for A11)
 * @return 2D processor-local-index pair (px,py,i_L,j_L)
 */
conflux::ProcIndexPair2D conflux::CholeskyProperties::globalToLocal(TileIndex i, TileIndex j)
{
    return ProcIndexPair2D{
        (i-1) % this->PX, 
        (j-1) % this->PY, 
        (i-1) / this->PX, 
        (j-1) / this->PY
    };
}

/**
 * @brief computes processor coordinates (px,py) that own this global tile in
 * A11 and their local tile indices for it
 * 
 * @param ind pair of global tile indices (for A11)
 * @return 2d processor-local-index pair (px,py,i_L,j_L)
 */
conflux::ProcIndexPair2D conflux::CholeskyProperties::globalToLocal(TileIndices ind)
{
    return ProcIndexPair2D{
        (ind.i-1) % this->PX,
        (ind.j-1) % this->PY, 
        (ind.i-1) / this->PX, 
        (ind.j-1) / this->PY
    };
}
