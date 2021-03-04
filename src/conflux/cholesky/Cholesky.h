/**
 * Copyright (C) 2020-2021, ETH Zurich
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

 * @file Cholesky.h
 * 
 * @brief header file for near-communication optimal parallel Cholesky 
 * decomposition.
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 14.11.2020
 */

#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <stdint.h>

/*********************** Library Function Declarations ***********************/
void initialize(int argc, char *argv[], uint32_t N);
void initialize(int argc, char *argv[], uint32_t N, uint32_t v, ProcCoord *grid);
void finalize(bool clean = false);
void setOptimalParams();
void parallelCholesky();

#endif // CHOLESKY_H