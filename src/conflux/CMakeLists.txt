add_executable(conflux_opt "conflux_opt.cpp")
target_include_directories(conflux_opt PUBLIC $<BUILD_INTERFACE:${conflux_SOURCE_DIR}/src>)
target_compile_features(conflux_opt PUBLIC cxx_std_17)
target_link_libraries(conflux_opt PUBLIC MPI::MPI_CXX
                         PRIVATE ${BLAS_TARGET}
                                 ${SCALAPACK_TARGET}
                     )
# target_compile_definitions(conflux_opt PRIVATE "VALIDATE")

if(CONFLUX_WITH_PROFILING)
    target_link_libraries(conflux_opt PRIVATE semiprof)
    target_compile_definitions(conflux_opt PRIVATE CONFLUX_WITH_PROFILING)
endif()

