/**
 * @file Processor.cpp
 * 
 * @brief implementation of the Processor class
 * 
 * @authors Anonymized Authors
 * 
 * @date 23.11.2020
 */

#include <sstream>
#include <mpi.h>
#include <math.h>
#include <iostream>
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

    // compute maximal local tile indices for this processor
    this->maxIndexA10 = (prop->Kappa - 1) / prop->P
                        + ((prop->Kappa - 1) % prop->P > this->rank ? 1 : 0);
    this->maxIndexA11i = (prop->Kappa - 1) / prop->PX
                         + ((prop->Kappa - 1) % prop->PX > this->px ? 1 : 0);
    this->maxIndexA11j = (prop->Kappa - 1) / prop->PY 
                         + ((prop->Kappa - 1) % prop->PY > this->py ? 1 : 0);

    // allocate permanent buffers for this processor
    this->A00 = new double[prop->v * prop->v];
    this->A10 = new TileMatrix<double>(MatrixType::VECTOR, prop->v, prop->v, this->maxIndexA10);
    this->A11 = new TileMatrix<double>(MatrixType::MATRIX, prop->v, prop->v, this->maxIndexA11i,
                               this->maxIndexA11j);

    // allocate temporary receive buffers
    this->A10rcv = new TileMatrix<double>(MatrixType::VECTOR, prop->v, prop->l, this->maxIndexA11i);
    this->A01rcv = new TileMatrix<double>(MatrixType::VECTOR, prop->v, prop->l, this->maxIndexA11j);

    // set the request counters to zero and initialze the dgemm ready flags
    this->cntUpdateA10 = 0;
    this->cntUpdateA10snd = 0;
    this->cntScatterA11 = 0;
    this->dgemmReadyFlags = new TileMatrix<TileReady>(MatrixType::MATRIX, 1, 1, this->maxIndexA11i, this->maxIndexA11j);
    for (TileIndex iLoc = 0; iLoc < this->maxIndexA11i; ++iLoc) {
        for (TileIndex jLoc = 0; jLoc < this->maxIndexA11j; ++jLoc) {
            TileReady *tmp = this->dgemmReadyFlags->get(iLoc, jLoc);
            tmp->a10 = false;
            tmp->a01 = false; 
        }
    }

    // compute the upper bounds for the number of requests on sub-tiles
    this->sndBound = this->maxIndexA10 * (prop->PZ * (prop->PX + prop->PY));
    this->rcvBound = this->maxIndexA11i + this->maxIndexA11j;

    // reserve memory for the MPI request vectors. Note that all these sizes are
    // upper bounds, and not exact by any means. It is only important to 
    this->reqUpdateA10.resize(this->rcvBound);
    this->reqUpdateA10snd.resize(this->sndBound);
    this->reqScatterA11.resize(this->maxIndexA10 + 1);
    this->tileInfos.reserve(this->rcvBound);
    
    // create a new communicator for all processors along the same z-axis as
    // the current processor, i.e. processors that share (px,py) coordinates
    // we define color = px * PY + py, i.e. rank on XY-plane in row-major order
    // and rank as the pz coordinate.
    MPI_Comm_split(MPI_COMM_WORLD, this->px * prop->PY + this->py, this->pz, &this->zAxisComm);

    // set the private pointer to the properties object
    this->m_prop = prop;

    initializeBroadcastComms();
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

    // delete the tile ready matrix
    delete dgemmReadyFlags;

    // the processor with rank 0 within its local z-axis communicator (i.e. 
    // the one with pz=0) has to free the communicator at the end of the
    // execution.
    if (this->pz == 0) {
        MPI_Comm_free(&this->zAxisComm);
    }
}

/**
 * @brief updates the broadcast communicator
 * @param remTiles the remaining number of tiles in A10
 */
void conflux::Processor::updateBcastComm(uint32_t remTiles)
{
    if (m_alwaysUseWorld || m_curIdx + 1 >= m_bcastComms.size() || remTiles > m_bcastSizes[m_curIdx + 1]) {
        return;
    }

    this->isWorldBroadcast = false;
    m_curIdx++;
    bcastComm = m_bcastComms[m_curIdx];
    inBcastComm = m_inCurrentBcastComm[m_curIdx];
}

/**
 * @brief initializes the broadcast communicators
 */
void conflux::Processor::initializeBroadcastComms()
{  
    ownsDiagonal = false;
    for (TileIndex i = 1; i < m_prop->Kappa; i++) {
        ProcIndexPair2D tmp = m_prop->globalToLocal(i,i);
        if (tmp.px == this->px && tmp.py == this->py) {
            ownsDiagonal = true;
            break;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // if we have 8 or less processors, it's not worth it to generate new communicators
    if (m_prop->P <= 8) {
        m_alwaysUseWorld = true;
        m_bcastComms.push_back(MPI_COMM_WORLD);
        m_inCurrentBcastComm.push_back(true);
        this->bcastComm = m_bcastComms[0];
        this->m_curIdx = 0;
        this->inBcastComm = m_inCurrentBcastComm[0];
        this->isWorldBroadcast = true;
        return;
    }


    m_alwaysUseWorld = false;
    uint64_t maxBroadcastSize = std::min(m_prop->P, m_prop->Kappa-2);
    if (maxBroadcastSize == m_prop->P) {
        maxBroadcastSize = 1 << (uint64_t)ceil(log2(maxBroadcastSize));
        m_inCurrentBcastComm.push_back(true);
        m_bcastComms.push_back(MPI_COMM_WORLD);

        std::set<ProcRank> dummySet;
        dummySet.insert(this->rank);
        m_tileOwners.push_back(dummySet);

        m_bcastSizes.push_back(m_prop->P);
        maxBroadcastSize /= 2;
        this->isWorldBroadcast = true;
    }  

    else {
        maxBroadcastSize = 1 << (uint64_t)ceil(log2(m_prop->Kappa - 2));
        if (maxBroadcastSize >= m_prop->P) {
            m_inCurrentBcastComm.push_back(true);
            m_bcastComms.push_back(MPI_COMM_WORLD);

            std::set<ProcRank> dummySet;
            dummySet.insert(this->rank);

            m_tileOwners.push_back(dummySet);
            m_bcastSizes.push_back(m_prop->P);
            maxBroadcastSize /= 2;
            this->isWorldBroadcast = true;
        }

        else {
            this->isWorldBroadcast = false;
            createNewComm(maxBroadcastSize);
        }
    }

    while (maxBroadcastSize >= 8) {
        //if (rank == 0) std::cout << maxBroadcastSize << std::endl;
        createNewComm(maxBroadcastSize);
    }
    this->bcastComm = m_bcastComms[0];
    this->m_curIdx = 0;
    this->inBcastComm = m_inCurrentBcastComm[0];
}

/**
 * @brief creates a new communicator for broadcasting
 * @note this function MUST NOT be called from outside the constructor
 * 
 * @param broadCastSize the new size of the broadcast (excl. the a00 owners)
 */
void conflux::Processor::createNewComm(uint64_t &broadCastSize) {
    TileIndex glob = m_prop->Kappa - 1;
    //if(rank == 0) std::cout << glob << std::endl;
    std::set<ProcRank> tmp_set;
    // we iterate backwards to find all owners in the current column
    for (int i = 0; i < broadCastSize; i++) {
        ProcRank owner = m_prop->globalToLocal(glob).p;
        tmp_set.insert(owner);
        glob--;
    }

    // we add everything on the diagonal to the communicator
    for (TileIndex j = 0; j < m_prop->Kappa; j++) {
        ProcIndexPair2D r = m_prop->globalToLocal(j,j);
        for (ProcRank k = 0; k < m_prop->PZ; k++) {
            tmp_set.insert(m_prop->gridToGlobal(r.px,r.py,k));
        }
    }

    MPI_Comm tmpComm;
    bool inBroadcast = tmp_set.find(this->rank) != tmp_set.end();
    int newRank = ownsDiagonal ? this->px + m_prop->PX * this->pz : m_prop->P + this->rank;
    MPI_Comm_split(MPI_COMM_WORLD, inBroadcast, newRank, &tmpComm);
    m_tileOwners.push_back(tmp_set);
    m_bcastSizes.push_back(broadCastSize);
    m_inCurrentBcastComm.push_back(inBroadcast);
    m_bcastComms.push_back(tmpComm);
    broadCastSize /= 2;
}











