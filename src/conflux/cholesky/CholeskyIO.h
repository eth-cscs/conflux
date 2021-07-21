/**
 * @file CholeskyIO.h
 * 
 * @brief header for class that enables debug IO functionality such as reading
 * input matrices and distributing them, creating an s.p.d. matrix in a distributed
 * manner 
 * 
 * @authors Anonymized Authors
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
    CholeskyIO(CholeskyProperties *prop, Processor *proc);
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
    
    void dumpA00(TileIndex round);
    void dumpA10(ProcIndexPair1D local, TileIndex global, TileIndex round);
    void dumpA11();
};

} // namespace conflux

#endif // CHOLESKY_IO_H
