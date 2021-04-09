/**
 * @file CholeskyProperties.h
 * 
 * @brief cdeclaration of class that stores global properties for a particular
 * run of the parallel Cholesky factorization algorithm
 * 
 * @authors Anonymized Authors
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
};

} // namespace conflux

#endif // CHOLESKY_PROPERTIES_H