cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(lu VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# Dependencies
#
find_package(MPI REQUIRED)

# MKL BLAS
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP SCALAPACK_32BIT_OMP)
set(BLAS_TARGET "mkl::blas_32bit_omp")
set(SCALAPACK_TARGET "mkl::scalapack_32bit_omp")

 if (CONFLUX_WITH_PROFILING)
    option(SEMIPROF_WITH_INSTALL "" ${CONFLUX_WITH_INSTALL})
    add_subdirectory(libs/semiprof)
endif ()

add_subdirectory(src/conflux)
