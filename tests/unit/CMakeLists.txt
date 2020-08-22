
set(unit_test_files
    test_mappers.cpp
    test.cpp
)

add_executable(unit EXCLUDE_FROM_ALL ${unit_test_files})
add_dependencies(tests unit)
target_include_directories(unit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(unit PRIVATE gtest conflux)
