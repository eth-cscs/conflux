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

 * @file CholeskyTypes.h
 * 
 * @brief type definitions for the Cholesky factorization algorithm
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 22.11.2020
 */

#ifndef CHOLESKY_TYPES_H
#define CHOLESKY_TYPES_H

#include <stdint.h>
#include <stdexcept>

namespace conflux {

/**************************** Type Definitions *******************************/

typedef uint32_t ProcRank;  // data structure for global processor rank
typedef uint32_t ProcCoord; // data structure for processor coordinates
typedef uint32_t TileIndex; // data structure for tile indices

/** @brief data structure that holds 3d grid coordinates */
typedef struct GridProc {
    ProcCoord px; //!< processor's x-coordinate
    ProcCoord py; //!< processor's y-coordinate
    ProcCoord pz; //!< processor's z-coordinate
} GridProc; 

/** @brief data structure that holds a 2D tile index pair (i,j) */
typedef struct TileIndices {
    TileIndex i; //!< tile-row index
    TileIndex j; //!< tile-column index
} TileIndices;

/** @brief data structure that holds a 1D processor-index pair (p,i) */
typedef struct ProcIndexPair1D {
    ProcRank p; //!< global processor rank
    TileIndex i; //!< processor's index for this tile
} ProcIndexPair1D;

/** @brief data structure that holds a 2D processor index pair (px,py,i,j) */
typedef struct ProcIndexPair2D {
    ProcCoord px; //!< processor's x-coordinate
    ProcCoord py; //!< processor's y-coordinate
    TileIndex i;  //!< processor's tile-row index for this tile
    TileIndex j;  //!< processor's tile-column index for this tile
} ProcIndexPair2D;

/**************************** Exception Types ********************************/


/**
 * @brief Basic exception type for Cholesky factorization
 */
class CholeskyException : public std::logic_error
{
    /**
     * @brief displays an exception when something goes wrong
     */
    public:

        enum errorCode {
            InvalidGridSize,
            InvalidProcessorNumber,
            FileSystemInputProblem,
            FileSystemOutputProblem,
            FailedMPIInit,
            MatrixInsteadOfVector,
            VectorInsteadOfMatrix,
            UnmatchingXYSizes,
            TilesNotDivisibleByZ
        };
        CholeskyException(errorCode code) : CholeskyException(ErrorCodeToString(code)){}

    private:
        const char* ErrorCodeToString(errorCode code) {
            switch(code) {
                case InvalidGridSize:
                    return "The specified grid size does not match the number of processors.";
                case InvalidProcessorNumber:
                    return "The specified number of processors is too small, must be at least 4.";
                case FileSystemInputProblem:
                    return "There was a problem with the file system and a file to read something in couldn't be opened. Check if the input paths are correct or if a matrix of given size exists at the specified path.";
                case FileSystemOutputProblem:
                    return "There was a problem with the file system and a file to write something to could'nt be opened. Check if the output paths are correct.";
                case FailedMPIInit:
                    return "The MPI Environment was not initialised.";
                case MatrixInsteadOfVector:
                    return "Cannot access a matrix with only one index.";
                case VectorInsteadOfMatrix:
                    return "Cannot access a vector with two indices.";
                case UnmatchingXYSizes:
                    return "The number of processors in X direction must match the number of processors in Y direction.";
                case TilesNotDivisibleByZ:
                    return "The tile size must be divisible by the number of processors in z-direction.";
                default:
                    return "Unknown error";
            }
        }

        CholeskyException(const char* code) : std::logic_error(code){};
};

} // namespace conflux

#endif // CHOLESKY_TYPES_H