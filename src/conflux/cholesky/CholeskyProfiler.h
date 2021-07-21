/**
 * @file CholesykProfiler.h
 * 
 * @brief this header makes semiprof an optional dependency that does not need to 
 * be included when Conflux is shipped.
 * 
 * @authors Anonymized Authors
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