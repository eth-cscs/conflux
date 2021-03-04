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

 * @file TileMatrix.h
 * 
 * @brief declaration of class for matrices of tiles, where there is no need
 * for accessing individual elements of a single tile, but only the tile as
 * a whole, i.e. the pointer to it. More over, tiles are stored contiguously.
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 19.12.2020
 */

#ifndef TILEMATRIX_H
#define TILEMATRIX_H

#include <stdint.h>
#include <exception>

namespace conflux {

/**
 * @brief defines a type for a TileMatrix object, if it's a tile-column vector
 * only (e.g for A10snd and A01snd), which is 1-dimensional, or a tile matrix 
 * (e.g. for A11), which is 2-dimensional.
 */
enum class MatrixType {VECTOR, MATRIX};

/**
 * @brief class that represents a matrix of tiles. This is a contiguous buffer
 * that consists of multiple tiles that are stored either in a 1d vector or a
 * 2d matrix. Each tile can be accessed individually via the get() function.
 */
class TileMatrix
{
public:
    // constructors and destructors
    TileMatrix(MatrixType type, uint32_t rowsPerTile, uint32_t colsPerTile,
               uint32_t tileRows, uint32_t tileCols = 1);
    ~TileMatrix();

    // methods to access a tile in the tile matrix
    double* get(uint32_t i);
    double* get(uint32_t i, uint32_t j);

private:
    MatrixType type;        //!< type of the matrix
    double* data;           //!< pointer to the start of the tile matrix
    uint32_t rowsPerTile;   //!< number of rows per tile
    uint32_t colsPerTile;   //!< number of cols per tile
    uint32_t tileRows;      //!< number of rows in this tile matrix
    uint32_t tileCols;      //!< number of cols in this tile matrix
}; 

} // namespace conflux

#endif // TILEMATRIX_H