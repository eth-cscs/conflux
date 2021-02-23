set(executables "conflux_miniapp")

foreach(exec ${executables})
    add_executable(${exec} "${exec}.cpp")
    target_link_libraries(${exec} PRIVATE conflux-lu cxxopts)
    target_include_directories(${exec} PUBLIC "${conflux_SOURCE_DIR}/libs/cxxopts/include/")
endforeach()
