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
 * @brief header for class that enables debug IO functionality such as reading
 * input matrices and distributing them, creating an s.p.d. matrix in a distributed
 * manner 
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard
 * Contact: {saethrej, andrega}@ethz.ch
 * 
 * @date 01.03.2021
 */

#ifndef CHOLESKY_IO_H
#define CHOLESKY_IO_H

#include "CholeskyTypes.h"
#include "CholeskyProperties.h"
#include "Processor.h"
#include "TileMatrix.h"

// typedef struct ompi_file_t *MPI_File; // forward declaration

namespace conflux {

/**
 * @brief object that handles the creation of s.p.d. input matrices for the
 * Cholesky factorization and the dumping of the matrix to a file at any time
 * in the algorithm.
 * 
 * @note this is DEBUG functionality and can be used to assess correctness of
 * the implementation during development. It does not have to be included in 
 * a deployable version.
 */
class CholeskyIO
{
public:
    // constructor and destructor
    CholeskyIO(CholeskyProperties *prop, Processor *proc, MPI_Comm &mainComm);
    ~CholeskyIO();
    
    // functions to generate an input matrix
    void generateInputMatrixDistributed();
    void parseAndDistributeMatrix();
    
    // functions to dump a matrix to a file
    void openFile(std::string filename);
    void closeFile();
    void dumpMatrix();
    void dumpSingleTileColumn(TileIndex round);

private:
    // private member fields
    MPI_File *fh;               //!< file handle used for matrix dumping
    CholeskyProperties *prop;   //!< pointer to the execution's property class
    Processor *proc;            //!< pointer to this processor's local variables
    std::string inputFilePath;  //!< path to the input file (not used currently)
    MPI_Comm _mainComm;         //!< main communicator (differs from world in a few special parameter choices)
    
    void dumpA00(TileIndex round);
    void dumpA10(ProcIndexPair1D local, TileIndex global, TileIndex round);
    void dumpA11();
};

} // namespace conflux

#endif // CHOLESKY_IO_H
