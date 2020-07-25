cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(lu VERSION 0.1 LANGUAGES CXX)

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


add_executable(conflux "conflux/conflux.cpp")
target_compile_features(conflux PUBLIC cxx_std_17)
target_link_libraries(conflux PUBLIC MPI::MPI_CXX
                         PRIVATE ${BLAS_TARGET}
                                 ${SCALAPACK_TARGET}
                     )
target_compile_definitions(conflux PRIVATE "VALIDATE")

add_executable(conflux_opt "conflux/conflux_opt.cpp")
target_compile_features(conflux_opt PUBLIC cxx_std_17)
target_link_libraries(conflux_opt PUBLIC MPI::MPI_CXX
                         PRIVATE ${BLAS_TARGET}
                                 ${SCALAPACK_TARGET}
                     )
target_compile_definitions(conflux_opt PRIVATE "VALIDATE")

if(CONFLUX_WITH_PROFILING)
    target_link_libraries(conflux PRIVATE semiprof)
    target_compile_definitions(conflux PRIVATE CONFLUX_WITH_PROFILING)

    target_link_libraries(conflux_opt PRIVATE semiprof)
    target_compile_definitions(conflux_opt PRIVATE CONFLUX_WITH_PROFILING)
endif()

