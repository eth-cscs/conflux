set(lu_executables "conflux_miniapp")

foreach(exec ${lu_executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE conflux-lu cxxopts ${SCALAPACK_TARGET})
    target_include_directories(${exec} PUBLIC "${conflux_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()

set(cholesky_executables "cholesky_miniapp" "cholesky_helper")
foreach(exec ${cholesky_executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE conflux-cholesky cxxopts)
    target_include_directories(${exec} PUBLIC "${conflux_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()

target_compile_definitions(cholesky_helper PRIVATE ${BLAS_DEF})
