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

 * @file Benchmark.cpp
 * 
 * @brief implementation of the Benchmark class.
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 14.11.2020
 */

#include <stddef.h>
#include <iostream>
#include <stdint.h>
#include <sstream>
#include <mpi.h>

#include "Benchmark.h"

/**
 * @brief constructs a Benchmark object
 */
Benchmark::Benchmark()
{
    // Get MPI_COMM_WORLD properties
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Set buffer size
    buf_size = sizeof(int) + 2*sizeof(double) + sizeof(uint64_t); // No alignment!
    
    // Set starting offset
    offset = buf_size*rank;

    // Declare time measurements
    std::vector<time_measurement*> time_measurements;
    
    // Create datatype
    int blocklen[4] = { 1, 1, 1, 1 };
    MPI_Datatype type[4] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_UNSIGNED_LONG_LONG };
    // Setting the displacement! Note that the integer in the struct might be
    // padded! We don't want any padding. The padding depends on the compiler.
    // That's why we use offsetof().
    MPI_Aint disp [4];
    disp[0] = offsetof(time_measurement, rank);
    disp[1] = offsetof(time_measurement, begin);
    disp[2] = offsetof(time_measurement, end);
    disp[3] = offsetof(time_measurement, bytes_sent);

    MPI_Type_create_struct(4, blocklen, disp, type, &TIMEDATA);
    MPI_Type_commit(&TIMEDATA);    
}

/**
 * @brief dumps the benchmark data to a binary file that can later be read
 */
void Benchmark::finish()
{
    std::stringstream filename;
    filename << outputDir
             << "/benchmark-"
             << run << "_"
             << N << "_"
             << v << "_"
             << Px << "x" << Py << "x" << Pz << ".bin";


    // Open file
    MPI_File_open(MPI_COMM_WORLD, filename.str().data(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
           MPI_INFO_NULL, &fh);
    
    std::vector<time_measurement*>::iterator it;
    for (it = time_measurements.begin(); it < time_measurements.end(); it++) {
        MPI_File_write_at(fh, offset, *it, 1, TIMEDATA, &status);
        offset += buf_size*world_size; 
    }
    
    MPI_File_close(&fh);
}

/**
 * @brief start the timer associated with this object
 */
void Benchmark::timer_start()
{
    current_tm = new time_measurement();
    current_tm->rank = rank;
    current_tm->begin = MPI_Wtime();
}

/**
 * @brief stop the timer associated with this object
 */
void Benchmark::timer_stop()
{
    current_tm->end = MPI_Wtime();
    time_measurements.push_back(current_tm);
}

/**
 * @brief get the duration that the timer associated with this object measured
 * @returns the duration in s
 */
double Benchmark::timer_duration()
{
    double duration = current_tm->end - current_tm->begin;
    
    if (duration <= 0.0) {
        return -1.0;
    } else {
        return duration;
    }
}

/**
 * @brief set the properties of the Cholesky factorization to be benchmarked
 * @param outputDir_ the desired output directory
 * @param run_ the unique identifier of this run
 * @param N_ the matrix input dimension
 * @param v_ the tile size
 * @param Px_ the number of processors along an x-axis
 * @param Py_ the number of processors along a y-axis
 * @param Pz_ the number of processors along a z-axis
 */
void Benchmark::set_props(std::string outputDir_, uint32_t run_, uint32_t N_, uint32_t v_,
                          uint32_t Px_, uint32_t Py_, uint32_t Pz_) {
    outputDir = outputDir_;
    run = run_;
    N = N_;
    v = v_;
    Px = Px_;
    Py = Py_;
    Pz = Pz_;
}

/** 
 * @brief adds a number of bytes to the send counter
 * @param val the number of bytes to be added
 */
void Benchmark::add(uint64_t val)
{
    current_tm->bytes_sent += val;
}

/**
 * @brief destructor that deletes all measurements
 */
Benchmark::~Benchmark()
{
    std::vector<time_measurement*>::iterator it;
    for (it = time_measurements.begin(); it < time_measurements.end(); it++) {
        delete *it;
    }
}
