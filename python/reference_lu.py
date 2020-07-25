import numpy as np
from utils import *
import scipy.linalg as la
import warnings

warnings.filterwarnings('error')
np.seterr(all='warn')

def LUPv2(inpA):
    global global_vars
    v = global_vars['v']

    A = np.copy(inpA)
    [m0, n0] = np.shape(A)
    # first column of A holds original indices of rows
    n0 = n0-1
    Perm = np.eye(m0)

    for k in range(n0):
        pivotRow = np.argmax(np.abs(A[k:, k+1])) + k
        A[[k, pivotRow]] = A[[pivotRow, k]]
        Perm[[k, pivotRow]] = Perm[[pivotRow, k]]
        try:
            for i in range(k+1, m0):
                A[i,k+1] /= A[k,k+1]
                for j in range(k+2, n0+1):
                    A[i,j] -= A[i,k+1] * A[k,j]
        except Warning:
            tmp = 1

    # debugging only
    U = np.triu(A[:,1:])
    U = U[:n0, :]
    L = np.concatenate((np.tril(A[:,1:]), np.zeros([m0, m0-n0])), axis = 1) \
        - np.concatenate((np.concatenate((np.diag(np.diag(A[:,1:])), np.zeros([m0 - n0, n0])), axis = 0),
                          np.zeros([m0, m0 - n0])), axis=1)   \
        + np.eye(m0)
    L = L[:, :n0]
    res = np.dot(L,U) - np.dot(Perm,inpA[:,1:])
    res = res[~np.isnan(res).any(axis=1)]
    assert(np.linalg.norm(res) <= 1e-6)
    #end of debugging

    origA = np.dot(Perm,inpA)[:v, :]
    tmp = 1
    return [origA, A[:v, 1:]]


def LUP(inpA):
    # global global_vars
    # v = global_vars['v']
    [Perm, L, U] = la.lu(inpA)
    origA = Perm.T @ inpA
    [m, n] = inpA.shape
    res = (L +
           np.concatenate((U, np.zeros([m-n,n])), axis = 0) -
           np.eye(m, n))[:v,:]
    return [origA[:v,], res[:v,], Perm]


def LU(inpA):
    B = np.copy(inpA)
    [m, n] = inpA[:,1:].shape
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if B[i, k] != 0.0:
                lam = B[i, k+1] / B[k, k+1]
                B[i, k + 2:n+1] = B[i, k + 2:n+1] - lam * B[k, k + 2:n+1]
                B[i, k+1] = lam

    return [inpA[:v,], B[:v, ], np.eye(m)]


def LU_ref(W):
    B = np.copy(W[:, 1:])
    [m0, n0] = np.shape(B)

    for k in range(0, n0):
        for i in range(k+1, m0):
            B[i,k] /= B[k,k]
            for j in range(k+1, n0):
                B[i,j] -= B[i,k] * B[k,j]

    # debugging only
    U = np.triu(B)
    U = U[:n0, :]
    L = np.concatenate((np.tril(B), np.zeros([m0, m0-n0])), axis = 1) \
        - np.concatenate((np.concatenate((np.diag(np.diag(B)), np.zeros([m0 - n0, n0])), axis = 0),
                          np.zeros([m0, m0 - n0])), axis=1)   \
        + np.eye(m0)
    L = L[:, :n0]
    res = np.dot(L,U) - W[:,1:]
    #end of debugging
    return B