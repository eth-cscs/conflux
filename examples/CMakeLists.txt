set(lu_executables "conflux_miniapp")

foreach(exec ${lu_executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE conflux-lu cxxopts)
    target_include_directories(${exec} PUBLIC "${conflux_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()

set(cholesky_executables "cholesky_miniapp" "cholesky_helper")
foreach(exec ${cholesky_executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE conflux-cholesky cxxopts)
    target_include_directories(${exec} PUBLIC "${conflux_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()

target_compile_definitions(cholesky_helper PRIVATE ${BLAS_DEF})

if(CONFLUX_SCALAPACK) 
    add_executable(cholesky_scalapack "cholesky_scalapack.cpp")
    target_link_libraries(cholesky_scalapack PRIVATE conflux-cholesky cxxopts ${SCALAPACK_TARGET})
    target_include_directories(cholesky_scalapack PUBLIC "${conflux_SOURCE_DIR}/libs/cxxopts/include/")
endif()


