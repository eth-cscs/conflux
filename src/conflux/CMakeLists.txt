set(INSTALLED_TARGETS_LIST "")
if(NOT TARGET conflux)
    set(conflux_src_files conflux_opt.cpp
                          blas.cpp)

    add_library(conflux ${conflux_src_files})

    target_include_directories(conflux PUBLIC $<BUILD_INTERFACE:${conflux_SOURCE_DIR}/src>)
    target_compile_features(conflux PUBLIC cxx_std_17)
    target_link_libraries(conflux PUBLIC  MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        PRIVATE ${BLAS_TARGET}
        )
    target_compile_definitions(conflux PRIVATE ${BLAS_DEF})

    if(CONFLUX_WITH_PROFILING)
        target_link_libraries(conflux PUBLIC semiprof)
        target_compile_definitions(conflux PUBLIC CONFLUX_WITH_PROFILING)
    endif()

    list(APPEND INSTALLED_TARGETS_LIST "cosma")
endif()


