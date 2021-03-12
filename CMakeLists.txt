cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Check if CONFLUX is a subproject
set(MASTER_PROJECT OFF)
if(NOT DEFINED PROJECT_NAME)
    set(MASTER_PROJECT ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/build_type.cmake)
include(cmake/adjust_mpiexec_flags.cmake)

# Options
#
if (NOT CONFLUX_SCALAPACK)
    set(CONFLUX_SCALAPACK "OFF" CACHE STRING
        "ScaLAPACK backend. Can be MKL, CRAY_LIBSCI, CUSTOM or OFF.")
endif()
set_property(CACHE CONFLUX_SCALAPACK PROPERTY STRINGS
    "OFF" "MKL" "CRAY_LIBSCI" "CUSTOM")

option(CONFLUX_WITH_TESTS "Generate the test target." ${MASTER_PROJECT})
option(CONFLUX_WITH_APPS "Generate the miniapp targets." ${MASTER_PROJECT})
option(CONFLUX_WITH_BENCHMARKS "Generate the benchmark targets." OFF)
option(CONFLUX_WITH_INSTALL "Enable installation." ${MASTER_PROJECT})
option(CONFLUX_WITH_PROFILING "Enable profiling." OFF)
option(CONFLUX_WITH_VALIDATION "Enable results validation." OFF)
option(BUILD_SHARED_LIBS "Build shared libraries." OFF)

set(CONFLUX_BLAS "MKL" CACHE STRING
    "Blas backend. Can be MKL, OPENBLAS, CRAY_LIBSCI, CUSTOM, CUDA or ROCM.")
set_property(CACHE CONFLUX_BLAS PROPERTY STRINGS
    "MKL" "OPENBLAS" "CRAY_LIBSCI" "CUSTOM" "CUDA" "ROCM")

set(CONFLUX_SCALAPACK "OFF" CACHE STRING
    "Blas backend. Can be MKL, CRAY_LIBSCI, CUSTOM or OFF.")
set_property(CACHE CONFLUX_SCALAPACK PROPERTY STRINGS
    "OFF" "MKL" "CRAY_LIBSCI" "CUSTOM")

# check if blas backend is valid
message(STATUS "Selected BLAS backend for CONFLUX: ${CONFLUX_BLAS}")
get_property(BACKEND_LIST CACHE CONFLUX_BLAS PROPERTY STRINGS)
if(NOT CONFLUX_BLAS IN_LIST BACKEND_LIST)
    message(FATAL_ERROR "Invalid value for CONFLUX_BLAS!")
endif()

# check if scalapack backend is valid
message(STATUS "Selected SCALAPACK backend for CONFLUX: ${CONFLUX_SCALAPACK}")
unset(BACKEND_LIST)
get_property(BACKEND_LIST CACHE CONFLUX_SCALAPACK PROPERTY STRINGS)
if(CONFLUX_SCALAPACK AND NOT CONFLUX_SCALAPACK IN_LIST BACKEND_LIST)
    message(FATAL_ERROR "Invalid value for CONFLUX_SCALAPACK!")
endif()

if (NOT ${CONFLUX_BLAS} STREQUAL "OPENBLAS")
    if (CONFLUX_SCALAPACK AND NOT ${CONFLUX_BLAS} STREQUAL ${CONFLUX_SCALAPACK})
        message(FATAL_ERROR "SCALAPACK backend MUST match BLAS backend if no OpenBLAS backend is used!")
    endif()
endif()

project(conflux VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# preserve rpaths when installing and make the install folder relocatable
# use `CMAKE_SKIP_INSTALL_RPATH` to skip this
# https://spack.readthedocs.io/en/latest/workflows.html#write-the-cmake-build
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
          "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
# skip RPATH if installed to system directories
if(isSystemDir STREQUAL "-1")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  if(APPLE)
    set(basePoint @loader_path)
  else()
    set(basePoint $ORIGIN)
  endif()
  file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
                            ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
  set(CMAKE_INSTALL_RPATH ${basePoint} ${basePoint}/${relDir})
endif()

# include this, so that all path variables are set up before adding the library subdirectory
if(CONFLUX_WITH_INSTALL)
    include(GNUInstallDirs)
endif()

# Dependencies
#
find_package(MPI REQUIRED COMPONENTS CXX)
adjust_mpiexec_flags()
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Bundled dependencies
#
if (CONFLUX_WITH_PROFILING)
    option(SEMIPROF_WITH_INSTALL "" ${CONFLUX_WITH_INSTALL})
    add_subdirectory(libs/semiprof)
endif ()

# Do not install cxxopts when including it.
if (CONFLUX_WITH_TESTS OR CONFLUX_WITH_APPS)
    option(CXXOPTS_BUILD_EXAMPLES OFF)
    option(CXXOPTS_BUILD_TESTS OFF)
    option(CXXOPTS_ENABLE_INSTALL OFF)
    add_subdirectory(libs/cxxopts)
endif()

# BLAS providers
#
set(BLAS_TARGET "")
set(BLAS_DEF "")

# BLAS LIBRARIES below can be used together with the GPU backend
# For this reason, we use MATCHES to also match entries like "MKL,CUDA" for BLAS
if (${CONFLUX_BLAS} MATCHES "MKL")
    find_package(MKL REQUIRED)
    set(BLAS_TARGET "mkl::mkl_intel_32bit_omp_dyn")
    set(BLAS_DEF "__USE_MKL")
elseif (${CONFLUX_BLAS} MATCHES "CRAY_LIBSCI")
    find_package(CRAY_LIBSCI REQUIRED)
    set(BLAS_TARGET "${CRAY_LIBSCI_LIBRARIES}")
    # set(BLAS_DEF "CONFLUX_CRAY_LIBSCI_BLAS")
elseif (${CONFLUX_BLAS} MATCHES "CUSTOM")
    find_package(BLAS REQUIRED)
    set(BLAS_TARGET "${BLAS_LIBRARIES}")
    # set(BLAS_DEF "CONFLUX_CUSTOM_BLAS")
elseif (${CONFLUX_BLAS} MATCHES "OPENBLAS")
    find_package(OpenBLAS REQUIRED)
    set(BLAS_TARGET "openblas")
    # set(BLAS_DEF "CONFLUX_OPEN_BLAS")
endif()

# (optional) SCALAPACK providers
#
set(SCALAPACK_TARGET "")
if (${CONFLUX_SCALAPACK} STREQUAL "MKL")
    find_package(MKL REQUIRED)
    set(SCALAPACK_TARGET "mkl::scalapack_${MPI_TYPE}_intel_32bit_omp_dyn")
elseif (${CONFLUX_SCALAPACK} STREQUAL "CRAY_LIBSCI")
    find_package(CRAY_LIBSCI REQUIRED)
    set(SCALAPACK_TARGET "${CRAY_LIBSCI_LIBRARIES}")
elseif (${CONFLUX_SCALAPACK} STREQUAL "CUSTOM")
    find_package(SCALAPACK REQUIRED)
    set(SCALAPACK_TARGET "${SCALAPACK_LIBRARIES}")
else  ()
    message(STATUS "Building with no SCALAPACK interface support.")
endif ()

# library
# lu decomposition
add_subdirectory(src/conflux/lu)

# cholesky decomposition
add_subdirectory(src/conflux/cholesky)

# testing
if(CONFLUX_WITH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# miniapps
if(CONFLUX_WITH_APPS)
    add_subdirectory(examples)
endif()


if(CONFLUX_WITH_INSTALL)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    install(DIRECTORY "${conflux_SOURCE_DIR}/src/conflux/lu"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.hpp")
    install(DIRECTORY "${conflux_SOURCE_DIR}/src/conflux/cholesky"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING
        PATTERN "*.hpp")

    write_basic_package_version_file(
        "${conflux_BINARY_DIR}/confluxConfigVersion.cmake"
        VERSION ${conflux_VERSION}
        COMPATIBILITY SameMajorVersion)

    configure_file("${conflux_SOURCE_DIR}/cmake/conflux.pc.in"
        "${conflux_BINARY_DIR}/conflux.pc"
        @ONLY)

    configure_file("${conflux_SOURCE_DIR}/cmake/confluxConfig.cmake.in"
        "${conflux_BINARY_DIR}/confluxConfig.cmake"
        @ONLY)

    install(FILES "${conflux_BINARY_DIR}/confluxConfig.cmake"
        "${conflux_BINARY_DIR}/confluxConfigVersion.cmake"
        "${conflux_SOURCE_DIR}/cmake/FindMKL.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/conflux")

    install(FILES "${conflux_BINARY_DIR}/conflux.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
endif()

