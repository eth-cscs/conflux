/**
 * @file Processor.h
 * 
 * @brief declaration of class that stores properties and all buffers of
 * a particular processor that participates in the algorithm
 * 
 * @authors Anonymized Authors
 * 
 * @date 22.11.2020
 */

#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <vector>
#include <set>
#include <mpi.h>

#include "CholeskyProperties.h"
#include "CholeskyTypes.h"
#include "TileMatrix.h"

namespace conflux {

/**
 * @brief simple enum class used for checking where a tile belongs
 */
enum class TileType {TILE_A10, TILE_A01};

/**
 * @brief simple struct that stores the flags on whether tiles are ready
 * for a dgemm call which we can overlap
 */
typedef struct TileReady {
    bool a10; // flag indicating whether A10 rep is ready
    bool a01; // flag indicating whether A01 rep is ready
    bool done; // flag indicating whether dgemm was performed
} TileReady;

typedef struct TileInfo {
    TileType type;    // type of the tile, i.e. A10 or A01
    TileIndex idxLoc; // index of the tile in the corresponding rcv buffer
    double *tilePtr;  // pointer to the data of the tile
} TileInfo;
    
/**
 * @brief an instance for a processor storing its properties and all its buffers
 */
class Processor 
{
public:

    // constructors and destructors
    Processor(CholeskyProperties *prop);
    ~Processor();

    // method to update the broadcast communicator
    void updateBcastComm(uint32_t remTiles);

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
    TileMatrix<double> *A10;    //!< vector of tiles (v,v) from A10
    TileMatrix<double> *A11;    //!< matrix of (v,v) tiles from A11
    
    // receive and send buffers (temporary data) for this processor
    TileMatrix<double> *A10rcv; //!< contains subtiles (v,l) of A10 to receive
    TileMatrix<double> *A01rcv; //!< contains subtiles (v,l) of A01 to receive

    // More specific communicators (e.g. for reduction)
    MPI_Comm zAxisComm; //!< communicator for reduction along z-Axis

    // vectors to store the request objects during an iteration
    std::vector<MPI_Request> reqUpdateA10;  //!< requests during the updateA10 operation
    std::vector<MPI_Request> reqUpdateA10snd; // !< send requests during updateA10
    std::vector<MPI_Request> reqScatterA11; //!< requests during the scatter operation
    TileMatrix<TileReady> *dgemmReadyFlags; //!< stores whether both operands for dgemm are ready
    std::vector<TileInfo> tileInfos; //!< stores information on the requests in updateA10


    // various counters for these requests 
    int cntUpdateA10;    //!< counts the number of requests in update A10 operation
    int cntUpdateA10snd; //!< counts the number of send requests in updateA10
    int cntScatterA11;   //!< counts the number of requests in scatter A11 operation

    // upper bounds for requests in sub-tile handling in update A10
    int sndBound; //!< upper bound for number of requests due to sending subtiles
    int rcvBound; //!< upper bound for number of requests due to receiving subtiles

    // communicators for the broadcast of new A00
    MPI_Comm bcastComm; //!< communicator for the current broadcast
    bool inBcastComm; //!< flag indicating whether this processors is in bcast comm
    bool isWorldBroadcast;
    bool ownsDiagonal; //!< flag indicating if ths processor owns a diagonal tile

private:
    // private member functions
    void initializeBroadcastComms();
    // this creates a new communicator for a given tile size
    void createNewComm(uint64_t &broadCastSize);

    // private member fields
    std::vector<MPI_Comm> m_bcastComms; //!< vector of all bcast comms in this execution
    std::vector<uint32_t> m_bcastSizes; //!< vector of the theoretical bcast sizes (- A00 providers)
    std::vector<bool> m_inCurrentBcastComm; //!< vector of flags of membership
    std::vector<std::set<ProcRank>> m_tileOwners; //!< set of ranks that own tile for given remaining num tiles
    bool m_alwaysUseWorld; //!< this flags says that we do not use any optimization of the broadcast
    uint8_t m_curIdx; //!< current index for all of the above vectors
    CholeskyProperties *m_prop; //!< pointer to the cholesky properties object. DO NOT FREE IN THIS CLASS
};

} // namespace conflux

#endif // PROCESSOR_H
