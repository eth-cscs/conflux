#!/usr/bin/env python

'''
Project: Parallel Cholesky Factorization
Authors: Jens Eirik Saethre, André Gaillard, Pascal Müller, Marc Styger
Date: 15.12.2020
Description: utility script for the creation or comparison of matrices

Script can be used to either generate matrices or to compare them. Syntax:
- python3 src/scripts/cholesky_helper.py --generate 8192
- python3 src/scripts/cholesky_helper.py --compare 8192
The former generates data/input_8192.bin and data/result_8192.bin and the
latter compares data/output_8192.bin with data/result_8192.bin and prints
the matrix norm of the difference.
'''

import argparse
from sklearn.datasets import make_spd_matrix as gen_spd
import numpy as np
import struct
from sys import byteorder

def generate(dim: int) -> None:
    '''
    generates a spd matrix with specified dim, computes its Cholesky
    factorization and stores both matrices.

    Parameters
    ----------
        dim {int} -- dimension of the matrix
    '''

    matrix = gen_spd(dim)
    cholesky = np.linalg.cholesky(matrix)

    # write input matrix
    with open('data/input_{}.bin'.format(dim), 'wb+') as f:
        for i in range(dim):
            for j in range(dim):
                f.write(struct.pack('<d', matrix[i][j]))

    # write result cholesky matrix
    with open('data/result_{}.bin'.format(dim), 'wb+') as f:
        for i in range(dim):
            for j in range(dim):
                f.write(struct.pack('<d', cholesky[i][j]))

    return

def generate_no_cholesky(dim: int) -> None:
    '''
    generates a spd matrix with specified dim

    Parameters
    ----------
        dim {int} -- dimension of the matrix
    '''
    matrix = gen_spd(dim)

    # write input matrix
    with open('data/input_{}.bin'.format(dim), 'wb+') as f:
        for i in range(dim):
            for j in range(dim):
                f.write(struct.pack('<d', matrix[i][j]))
    return

def compare(dim: int) -> float:
    '''
    compares implementation's output to correct result and returns matrix
    norm of difference matrix.

    Parameters
    ----------
        dim {int} -- dimension of the matrix
    
    Returns
    -------
        {float} -- the matrix norm of the difference matrix

    Raises
    ------
        OSError -- when one of the files was not found
    '''
    # define space for matrix to be read
    output = np.ndarray(shape = (dim, dim))
    result = np.ndarray(shape = (dim, dim))
    
    try:
        # read output matrix (set entries above diagonal to 0)
        with open('data/output_{}.bin'.format(dim), 'rb+') as f:
            for i in range(dim):
                for j in range(dim):
                    output[i][j] = struct.unpack('<d', f.read(8))[0]
                    if (j > i):
                        output[i][j] = 0

        # read result matrix (also set entries above diagonal to 0 to ensure
        # interoperability between python and C++ programs.)
        with open('data/result_{}.bin'.format(dim), 'rb+') as f:
            for i in range(dim):
                for j in range(dim):
                    result[i][j] = struct.unpack('<d', f.read(8))[0]
                    if (j > i):
                        result[i][j] = 0

        norm = np.linalg.norm(output - result)
        print("||output-result|| = {}".format(norm))
        return norm
    
    except Exception as err:
        print(err)

def check_spd(dim: int) -> bool:
    '''
    checks if input_dim.bin is an spd matrix

    Parameters
    ----------
        dim {int} -- matrix dimension
    
    Returns
    -------
        {bool} -- true if matrix is spd
    '''
    input_matrix = np.ndarray(shape = (dim, dim))

    # try to compute the cholesky factorization, if it is possible, the
    # matrix is spd, otherwise it is not
    try:
        # read input matrix
        with open('data/input_{}.bin'.format(dim), 'rb+') as f:
            for i in range(dim):
                for j in range(dim):
                    input_matrix[i][j] = struct.unpack('<d', f.read(8))[0]

        np.linalg.cholesky(input_matrix)
        print(True)
        return True
    
    except np.linalg.LinAlgError as err:
        print(err)
        return False


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', type=int, help='generate matrix of this size')
    parser.add_argument('-c', '--compare', type=int, help='compares output to correct result for this size')
    parser.add_argument('-s', '--checkspd', type=int, help='checks if an input matrix is SPD')
    parser.add_argument('-n', '--nocholesky', action='store_true', help="only generates input matrix")
    args = vars(parser.parse_args())

    # can't have both arguments at the same time
    if args['generate'] is not None and args['compare'] is not None:
        print("Error: Can only generate or compare, not both. Try again.")
    elif args['generate'] is not None:
        if args['nocholesky'] is not None:
            generate_no_cholesky(args['generate'])
        else:
            generate(args['generate'])
    elif args['compare'] is not None:
        compare(args['compare'])
    elif args['checkspd'] is not None:
        check_spd(args['checkspd'])

