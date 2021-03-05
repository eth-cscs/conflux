set(INSTALLED_TARGETS_LIST "")
if(NOT TARGET conflux-lu)
    set(conflux_lu_src_files conflux_opt.cpp
                          blas.cpp)

    add_library(conflux-lu ${conflux_lu_src_files})

    target_include_directories(conflux-lu PUBLIC $<BUILD_INTERFACE:${conflux_SOURCE_DIR}/src>)
    target_compile_features(conflux-lu PUBLIC cxx_std_17)
    target_link_libraries(conflux-lu PUBLIC  MPI::MPI_CXX
        OpenMP::OpenMP_CXX
        ${BLAS_TARGET}
        )
    target_compile_definitions(conflux-lu PRIVATE ${BLAS_DEF})

    if(CONFLUX_WITH_VALIDATION)
        target_compile_definitions(conflux-lu PUBLIC CONFLUX_WITH_VALIDATION)
    endif()

    if(CONFLUX_WITH_PROFILING)
        target_link_libraries(conflux-lu PUBLIC semiprof)
        target_compile_definitions(conflux-lu PUBLIC CONFLUX_WITH_PROFILING)
    endif()

    list(APPEND INSTALLED_TARGETS_LIST "conflux-lu")
endif()


