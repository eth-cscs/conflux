#pragma once
#include <mpi.h>
#include <costa/layout.hpp>

namespace conflux {

template <typename T>
costa::grid_layout<T> conflux_layout(T* pointer, 
                                     int M, int N, int v, 
                                     char ordering,
                                     int Px, int Py,
                                     int rank);
template <typename T>
costa::grid_layout<T> conflux_layout(T* pointer, 
                                     int M, int N, int v, 
                                     char ordering,
                                     MPI_Comm comm);
}
