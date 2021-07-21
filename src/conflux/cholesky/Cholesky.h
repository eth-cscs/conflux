/**
 * @file Cholesky.h
 * 
 * @brief header file for near-communication optimal parallel Cholesky 
 * decomposition.
 * 
 * @authors Anonymized Authors
 * 
 * @date 14.11.2020
 */

#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <stdint.h>

namespace conflux {

/*********************** Library Function Declarations ***********************/
void initialize(int argc, char *argv[], uint32_t N, uint32_t v, ProcCoord *grid);
void finalize(bool clean = false);
void parallelCholesky();

} // namespace conflux

#endif // CHOLESKY_H