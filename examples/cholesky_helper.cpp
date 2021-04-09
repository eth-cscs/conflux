/**
 * @file CholeskyHelper.cpp
 * 
 * @brief utility program to generate SPD matrices as input for our algorithm.
 * This program addresses the short-coming of the python script of the same
 * name when it comes to very large matrices
 * 
 * @authors Anonymized Authors
 * 
 * @date 15.12.2020
 */

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <getopt.h>

#ifdef __USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

// defines the mode for the current execution 
enum Mode {GENERATE = 0, COMPARE = 1};

/**
 * @brief prints help for command line arguments of this program
 */
void printHelp() 
{
    std::cerr << "Error: invalid parameters were provided. Allowed are: \n"
              << "\t --help, -h: Display (this) help section. Write --help or -h\n"
              << "\t --generate, -g: to generate matrices of dim >0. Write e.g. --generate=128 or -g 128 \n"
              << "\t --nocholesky, -n: only computes input matrix (only for --generate), write --nocholesky or -n \n"
              << "\t --compare, -c: to compare matfices of dim >0. Write --compare=4 or -c 4 \n"
              << "Note you cannot enter both arguments at the same time. Try again.\n"
              << std::endl;
    return;
}

/**
 * @brief parses args from the command line
 * 
 * @param argc the number of arguments from the shell
 * @param argv array with the arguments and their values
 * @param dim reference to the matrix dimension read in this function
 * @param tile reference to the mode that the program should run in
 * @param computeChol reference to flag indicating whether to compute result_N.bin
 */
void parseArgs(int argc, char* argv[], uint32_t &dim, Mode &mode, bool &computeChol)
{
    // defines the allowed command line options in their "long" form
    static const struct option long_args[] = {
        {"generate", required_argument, 0, 'g'},
        {"compare", required_argument, 0, 'c'},
        {"nocholesky", no_argument, 0, 'n'},
        {"help", no_argument, 0, 'h'},
        0
    };
    int gen = -1;
    int com = -1;

    int index = -1;
    int result;
    struct option *opt = nullptr;

    while (true) {
        index = -1;
        opt = nullptr;
        result = getopt_long(argc, argv, "hg:c:n", long_args, &index);

        // break loop if all arguments were parsed
        if (result == -1) break;

        switch (result) {
            case 'h': // help section
                printHelp();
                exit(0);

            case 'g': // generate matrix
                try {
                    mode = GENERATE;
                    gen = 1;
                    dim = std::stoi(optarg);
                    if (dim <= 0) {
                        printHelp();
                        exit(-1);
                    }
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                break;

            case 'c': // compare matrices
                try {
                    mode = COMPARE;
                    com = 1;
                    dim = std::stoi(optarg);
                    if (dim <= 0) {
                        printHelp();
                        exit(-1);
                    }
                } catch (std::exception e) {
                    printHelp();
                    exit(-1);
                }
                break;
            
            case 'n': // no cholesky option for generate
                computeChol = false;
                break;


            default: // fallback, indicates incorrect usage, print help and exit
                printHelp();
                exit(-1);
        }
    }

    // only one option is allowed, if none or multiple, the program
    // can't run.
    if (gen * com > 0) {
        printHelp();
        exit(-1);
    }
}


/**
 * @brief compares the correct Choleksy factorized matrix to our implementation's
 * output and returns the matrix norm of the difference matrix
 * 
 * @param dim dimension of the matrices to be compared
 * @returns the matrix norm of the difference matrix
 */
double compare(uint32_t dim)
{
    // create strings for matrix paths
    std::stringstream res, out;
    res << "data/result_" << dim << ".bin";
    out << "data/output_" << dim << ".bin";

    // read result matrix if it exists, otherwise compute it
    double *result = new double[((uint64_t) dim) * dim]();
    struct stat buffer;
    
    if (stat(res.str().c_str(), &buffer) == 0) {
        // the file exists, read it into the result buffer
        std::cout << "A result file exists, using it for the comparison.." << std::endl;
        std::ifstream resultInput(res.str().c_str(), std::ios::in | std::ios::binary);
        if (resultInput.fail()) {
            throw std::logic_error("Reading result input matrix failed. Check the path or if matrix of given size exists.");
        }
        resultInput.read(reinterpret_cast<char*>(result), dim * dim * sizeof(double));
        resultInput.close();

    } else {
        // the file does not exist. read the input matrix file, compute its
        // Cholesky factorization, and store it into the result buffer
        std::cout << "No result file found, computing result from input file.." << std::endl;

        // read the input file
        std::stringstream inp;
        inp << "data/input_" << dim << ".bin";
        std::ifstream input(inp.str().c_str(), std::ios::in | std::ios::binary);
        if (input.fail()) {
            throw std::logic_error("Reading result input matrix failed. Check the path or if matrix of given size exists.");
        }
        input.read(reinterpret_cast<char*>(result), dim * dim * sizeof(double));
        input.close();

        // use LAPACKE's dpotrf function to compute the correct Cholesky decomposition
        LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', dim, result, dim);
        
        // zero out the entries above the diagonal (LAPACKE does not touch that region)
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                if (j > i) result[i * dim + j] = 0.0;
            }
        }
    }

    // read output matrix
    double *output = new double[((uint64_t) dim) * dim]();
    std::ifstream outputInput(out.str().c_str(), std::ios::in | std::ios::binary);
    if (outputInput.fail()) {
        throw std::logic_error("Reading cholesky output matrix failed. Check the path or if matrix of given size exists.");
    }
    outputInput.read(reinterpret_cast<char*>(output), dim * dim * sizeof(double));
    outputInput.close();
    // compute the difference matrix output - result. For the output matrix, the
    // elements above the diagonal have be zeroed first. rsults are stored in 
    // outpzt matrix.
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            if (j > i) {
                output[i * dim + j] = 0.0;
                result[i * dim + j] = 0.0;
            } 
            output[i * dim + j] -= result[i * dim + j];
        }
    }

    // compute the matrix norm with LAPACKE's DLANGE function and return norm
    double norm = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'F', static_cast<int>(dim), 
                                 static_cast<int>(dim), output, dim);

    std::cout << "||output-result|| = " << norm << std::endl;
    return norm;
}

/**
 * @brief generates an input matrix of dim*dim size and computes the correct
 * Cholesky factorization, storing it in a result matrix
 * 
 * @param dim dimension of the matrix
 * @param computeCholesky flag indicating whether cholesky should be computed
 */
void generate(uint32_t dim, bool computeCholesky) 
{
    // define file names
    std::stringstream inp, res;
    inp << "data/input_" << dim << ".bin";
    res << "data/result_" << dim << ".bin";

    // create a random matrix and the matrix that will store A^T * A + dim * I
    // the dim * I is added since we want our matrix to be diagonally dominant,
    // which implies that the input matrix will be s.p.d.
    double *random = new double[((uint64_t) dim) * dim]();
    double *input = new double[((uint64_t) dim) * dim]();
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            random[i * dim + j] = rand() / ((double) RAND_MAX + 1.0);
            if (i == j) input[i * dim + j] = dim * 1.0;
        }
    }
    
    // compute input = random^T * random + dim * I
    cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, dim, dim, 1.0,
                random, dim, 1.0, input, dim);
    delete[] random;
    
    // the above operation only fills out the lower half of input. For complete-
    // ness we will fill the matrixs upper half, even though this is technically
    // not necessary, since the algorithm never operates on tiles above the 
    // diagonal.
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = i+1; j < dim; ++j) {
            input[i * dim + j] = input[j * dim + i];
        }
    }

    // write the input matrix to a file
    std::ofstream inputOutput(inp.str().c_str(), std::ios::out | std::ios::binary);
    if (inputOutput.fail()) {
        throw std::logic_error("Could not write the generated matrix to a file. Check the path.");
    }
    inputOutput.write(reinterpret_cast<const char*>(input), dim * dim * sizeof(double));
    inputOutput.close();

    // return if cholesky should not be computed
    if (!computeCholesky) {
        return;
    }

    // use LAPACKE's dpotrf function to compute the correct Cholesky decomposition
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', dim, input, dim);
    
    // zero out the entries above the diagonal (LAPACKE does not touch that region)
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            if (j > i) input[i * dim + j] = 0.0;
        }
    }

    // write the result matrix to a file
    std::ofstream resultOutput(res.str().c_str(), std::ios::out | std::ios::binary);
    if (resultOutput.fail()) {
        throw std::logic_error("Could not write sequentiall decomposited matrix to file. Check the path");
    }
    resultOutput.write(reinterpret_cast<const char*>(input), dim * dim * sizeof(double));
    resultOutput.close();
}

/**
 * @brief utility program that can generate matrices of arbitrary size or compare
 * two such matrices (i.e. our algorithm's output with the correct result) and
 * report the norm of the difference matrix.
 * 
 * @param argc number of args from command line
 * @param argv the command line arguments
 */
int main(int argc, char *argv[])
{
    // parse command line arguments
    uint32_t matrixDim = 0;
    Mode mode;
    bool computeChol = true;
    parseArgs(argc, argv, matrixDim, mode, computeChol);

    // execute the desired option
    if (mode == GENERATE) {
        generate(matrixDim, computeChol);
    } else if (mode == COMPARE) {
        compare(matrixDim);
    } else {
        std::cerr << "No valid option. Goodbye." << std::endl;
    }

    return 0;
}
