set(INSTALLED_TARGETS_LIST "")
if(NOT TARGET conflux-cholesky)
    set(conflux_cholesky_src_files Cholesky.cpp
                                   CholeskyIO.cpp
                                   CholeskyProperties.cpp
                                   Processor.cpp
                                   #TileMatrix.cpp
                               )

    add_library(conflux-cholesky ${conflux_cholesky_src_files})

    target_include_directories(conflux-cholesky PUBLIC $<BUILD_INTERFACE:${conflux_SOURCE_DIR}/src>)
    target_compile_features(conflux-cholesky PUBLIC cxx_std_17)
    target_link_libraries(conflux-cholesky PUBLIC  MPI::MPI_CXX
                                                   OpenMP::OpenMP_CXX
                                                   ${BLAS_TARGET}
        )
    target_compile_definitions(conflux-cholesky PRIVATE ${BLAS_DEF})

    if (CONFLUX_WITH_BENCHMARKS)
        target_compile_definitions(conflux-cholesky PUBLIC "BENCHMARK")
    endif()

    if (CONFLUX_WITH_PROFILING)
        target_link_libraries(conflux-cholesky PUBLIC semiprof)
        target_compile_definitions(conflux-cholesky PUBLIC CONFLUX_WITH_PROFILING)
    endif()

    list(APPEND INSTALLED_TARGETS_LIST "conflux-cholesky")
endif()