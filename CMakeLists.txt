cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(conflux VERSION 0.1 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# Dependencies
#
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# MKL BLAS
find_package(MKL REQUIRED COMPONENTS BLAS_32BIT_OMP SCALAPACK_32BIT_OMP)
set(BLAS_TARGET "mkl::blas_32bit_omp")
set(SCALAPACK_TARGET "mkl::scalapack_32bit_omp")

if (CONFLUX_WITH_PROFILING)
    option(SEMIPROF_WITH_INSTALL "" ${CONFLUX_WITH_INSTALL})
    add_subdirectory(libs/semiprof)
endif ()

# library
add_subdirectory(src/conflux)

# cxxopts
add_subdirectory(libs/cxxopts)

# miniapps
add_subdirectory(examples)
