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

 * @file Benchmark.h
 * 
 * @brief header file for basic benchmarking funtionality
 * 
 * @authors Jens Eirik Saethre, Andre Gaillard, Pascal Müller, Marc Styger
 * Contact: {saethrej, andrega, stygerma, pamuelle}@student.ethz.ch
 * 
 * @date 14.11.2020
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <vector>
#include <stdint.h>
#include <string>
#include <mpi.h>

class Benchmark
{
private:
    // Time measurement
    struct time_measurement {
        int rank;
        double begin = 0.0;
        double end = 0.0;
        uint64_t bytes_sent = 0;
    };

    std::vector<time_measurement*> time_measurements;
    time_measurement* current_tm;

    // MPI properties
    int world_size;
    int rank;
    MPI_Datatype TIMEDATA;
    MPI_File fh;
    MPI_Offset offset;
    MPI_Status status;

    int buf_size;
    // Properties
    uint32_t run;
    uint32_t N;
    uint32_t v;
    uint32_t Px;
    uint32_t Py;
    uint32_t Pz;

    std::string outputDir = "output"; //default value

    uint64_t bytes_sent = 0;

public:
    Benchmark();
    ~Benchmark();

    char filename;

    void finish();

    // Time measurement methods
    void timer_start();
    void timer_stop();
    double timer_duration();
    void set_props(std::string outputDir_, uint32_t run_, uint32_t N_, uint32_t v_, uint32_t Px_, uint32_t Py_, uint32_t Pz_);

    // Communication measurement methos
    void add(uint64_t val);
};

#endif // BENCHMARK_H
