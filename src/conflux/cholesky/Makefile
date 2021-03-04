# compiler options
CXX = mpic++
CXXFLAGS = -O3 -march=native

# linker options
LDFLAGS = -llapacke -lcblas -lblas -llapack -lgfortran
LDFLAGS_EULER = -lopenblas
LDFLAGS_SCALAPACK = -lscalapack

# directories
SRC_DIR = src
DATA_DIR = data
BUILD_DIR = build
INCLUDE_DIR = include -I $(MKLROOT)\include
BENCH_DIR = benchmark
SCALAPACK_DIR = scalapack

# dependencies
DEPS = $(shell find src -type f \( -iname *.cpp -o -iname *.h \))

# default is locally compiling Cholesky and CholeskyHelper
default: local

# compiles and links Cholesky main program and the Cholesky Helper in local setting
local: $(DEPS)
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/*.cpp -o $(BUILD_DIR)/cholesky $(LDFLAGS) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/scripts/CholeskyHelper.cpp -o $(BUILD_DIR)/helper $(LDFLAGS) -I $(INCLUDE_DIR)/

# load required software stack and modules on Euler, then compile and links Cholesky main
# program and the Cholesky Helper in euler setting
euler: $(DEPS)
	$(CXX) $(CXXFLAGS) -DEULER $(SRC_DIR)/*.cpp -o $(BUILD_DIR)/cholesky $(LDFLAGS_EULER) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/scripts/CholeskyHelper.cpp -o $(BUILD_DIR)/helper $(LDFLAGS_EULER) -I $(INCLUDE_DIR)/

# compiles and links Cholesky main program and CholeskyHelper with DEBUG variable set locally
local_debug: $(DEPS)
	$(CXX) $(CXXFLAGS) -DDEBUG $(SRC_DIR)/*.cpp -o $(BUILD_DIR)/cholesky $(LDFLAGS) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/scripts/CholeskyHelper.cpp -o $(BUILD_DIR)/helper $(LDFLAGS) -I $(INCLUDE_DIR)/

# compiles and links Cholesky main program and CholeskyHelper with DEBUG variable set on Euler
euler_debug: $(DEPS)
	$(CXX) $(CXXFLAGS) -DEULER -DDEBUG $(SRC_DIR)/*.cpp -o $(BUILD_DIR)/cholesky $(LDFLAGS_EULER) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/scripts/CholeskyHelper.cpp -o $(BUILD_DIR)/helper $(LDFLAGS_EULER) -I $(INCLUDE_DIR)/

# compile like the local target but sets BENCHMARK variable - mainly for developing purposes
local_benchmark: $(DEPS)
	$(CXX) $(CXXFLAGS) -DBENCHMARK $(SRC_DIR)/*.cpp $(SRC_DIR)/$(BENCH_DIR)/*.cpp -o $(BUILD_DIR)/cholesky_benchmark $(LDFLAGS) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/scripts/CholeskyHelper.cpp -o $(BUILD_DIR)/helper $(LDFLAGS) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/$(SCALAPACK_DIR)/*.cpp $(SRC_DIR)/$(BENCH_DIR)/*.cpp -o $(BUILD_DIR)/cholesky_scalapack $(LDFLAGS_SCALAPACK) -I $(INCLUDE_DIR) -I $(SRC_DIR)/$(BENCH_DIR)

# benchmark on euler, sets BENCHMARK variable
euler_benchmark: $(DEPS)
	$(CXX) $(CXXFLAGS) -DEULER -DBENCHMARK $(SRC_DIR)/*.cpp ${SRC_DIR}/${BENCH_DIR}/*.cpp -o $(BUILD_DIR)/cholesky_benchmark $(LDFLAGS_EULER) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/scripts/CholeskyHelper.cpp -o $(BUILD_DIR)/helper $(LDFLAGS_EULER) -I $(INCLUDE_DIR)/
	$(CXX) $(CXXFLAGS) $(SRC_DIR)/$(SCALAPACK_DIR)/*.cpp $(SRC_DIR)/$(BENCH_DIR)/*.cpp -o $(BUILD_DIR)/cholesky_scalapack $(LDFLAGS_SCALAPACK) -I $(INCLUDE_DIR) -I $(SRC_DIR)/$(BENCH_DIR)

.PHONY: clean

# clean binaries and matrix data
clean:
	$(RM) data/*.bin
	$(RM) build/*

clean_benchmark:
	$(RM) data/*.bin
	$(RM) build/*
	$(RM) data/benchmarks/cholesky25d/output/*.bin
	$(RM) data/benchmarks/cholesky25d/output/csv/*.csv
	$(RM) data/benchmarks/scalapack/output/*.bin
	$(RM) data/benchmarks/scalapack/output/*.csv
