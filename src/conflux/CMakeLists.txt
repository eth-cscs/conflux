set(conflux_src_files conflux_opt.cpp)

add_library(conflux ${conflux_src_files})
# set_target_properties(conflux PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(conflux PUBLIC $<BUILD_INTERFACE:${conflux_SOURCE_DIR}/src>)
target_compile_features(conflux PUBLIC cxx_std_14)
target_link_libraries(conflux PUBLIC  MPI::MPI_CXX
                              PRIVATE ${BLAS_TARGET}
                                      ${SCALAPACK_TARGET}
                     )
# target_compile_definitions(conflux_opt PRIVATE "VALIDATE")

if(CONFLUX_WITH_PROFILING)
    target_link_libraries(conflux PRIVATE semiprof)
    target_compile_definitions(conflux PRIVATE CONFLUX_WITH_PROFILING)
endif()

