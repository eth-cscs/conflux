/**
 * Copyright (C) 2020, ETH Zurich
 *
 * This product includes software developed at the Scalable Parallel Computing
 * Lab (SPCL) at ETH Zurich, headed by Prof. Torsten Hoefler. For more information
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
 * @file CholesykProfiler.h
 * 
 * @brief this header makes semiprof an optional dependency that does not need to 
 * be included when Conflux is shipped.
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard
 * Contact: {saethrej, andrega}@student.ethz.ch
 * 
 * @date 23.03.2021
 */

#ifndef CHOLESKY_PROFILER_H
#define CHOLESKY_PROFILER_H

// The header makes semiprof an optional dependency that needs not be shipped when conflux is installed.
//
#ifdef CONFLUX_WITH_PROFILING

#include <semiprof/semiprof.hpp>

// prints the profiler summary
#define PP() std::cout << semiprof::profiler_summary() << "\n"
// clears the profiler (counts and timings)
#define PC() semiprof::profiler_clear()

#else // the pre-processor will simply remove these statements

#define PE(name)
#define PL()
#define PP()
#define PC()
#endif

#endif // CHOLESKY_PROFILER_H