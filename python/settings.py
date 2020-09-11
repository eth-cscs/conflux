import math
import numpy as np
from numpy.random import RandomState
import math

# import dace
######################################
######## global parameters ###########
######################################
# dace_dtype = dace.float64
np_dtype = np.float64

inpN = 16
inpP = 8


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


def ModelCommCost(ppp, c):
    return 1 / (ppp * c)


def CalculateDecomposition(P):
    p13 = np.floor(np.power(P + 1, 1 / 3))
    ppp = np.floor(np.sqrt(P))
    c = 1
    best_ppp = ppp
    best_c = c
    bestCost = ModelCommCost(ppp, c)
    while (c <= p13):
        P1 = np.floor(P / c)
        ppp = np.floor(np.sqrt(P1))
        cost = ModelCommCost(ppp, c)
        if cost < bestCost:
            bestCost = cost
            best_ppp = ppp
            best_c = c
        c += 1
    assert (best_ppp * best_ppp * best_c <= P)
    return [best_ppp, best_c]


def CalculateParameters(inpN, P):
    [sqrtp1, c] = CalculateDecomposition(P)
    sqrtp1 = int(sqrtp1)
    v = lcm(sqrtp1, c)
    nLocalTiles = np.ceil(inpN / (v * sqrtp1))
    nLocalTiles = int(nLocalTiles)
    N = v * sqrtp1 * nLocalTiles
    return [sqrtp1, c, v, N]


[sqrtp1, c, v, N] = CalculateParameters(inpN, inpP)
P = sqrtp1 * sqrtp1 * c
nlayr = v // c
p1 = sqrtp1 * sqrtp1

prng = RandomState(43)
inpA = prng.rand(N, N)  # int(low=0, high=N, size=(N, N)).astype(np_dtype)
inpA = inpA @ inpA.transpose()

# # number of processors in 1 dimension - we'll make it full 3D decomposition
# ppp = int(8)
# # number of layers we want one processor computes in one execution of A11
# lrs = int(4)
# # number of tiles each processor will own
# nLocalTiles = 6
# P = ppp * ppp * ppp
# v = ppp * lrs
# N = v * ppp * nLocalTiles

# # P = 8
# # N = 16
# # v = 2

# # this is if we want full 3D decomposition: super huge memory, so that with extra resources we end up doing 3D
# M = N*N
# Mp = M

# prng = RandomState(42)
# inpA = prng.rand(N,N) #int(low=0, high=N, size=(N, N)).astype(np_dtype)
# # inpA = np.zeros([N,N])
# # for i in range(N):
# #     inpA[i,:] = np.array(range(N))
# #     inpA[i,:] = np.roll(inpA[i,:], i)
# #
# # offset = N / ppp + 1
# # for i in range(N):
# #     j = int(((1 + offset) * i) % N)
# #     inpA[[i, j],:] = inpA[[j, i],:]

# inpN = 1200
# inpP = 100


# def lcm(a, b):
#     return abs(a*b) // math.gcd(a, b)


# def ModelCommCost(ppp, c):
#     return 1/(ppp * c)


# def CalculateDecomposition(P):
#     p13 = np.floor(np.power(P + 1, 1 / 3))
#     ppp = np.floor(np.sqrt(P))
#     c = 1
#     best_ppp = ppp
#     best_c = c
#     bestCost = ModelCommCost(ppp, c)
#     while (c <= p13):
#         P1 = np.floor(P / c)
#         ppp = np.floor(np.sqrt(P1))
#         cost = ModelCommCost(ppp, c)
#         if cost < bestCost:
#             bestCost = cost
#             best_ppp = ppp
#             best_c = c
#         c += 1
#     assert(best_ppp*best_ppp*best_c <= P)
#     return [best_ppp, best_c]


# def CalculateParameters(inpN, P):
#     [sqrtp1, c] = CalculateDecomposition(P)
#     sqrtp1 = int(sqrtp1)
#     v = lcm(sqrtp1,c)
#     nLocalTiles = np.ceil(inpN / (v * sqrtp1))
#     nLocalTiles = int(nLocalTiles)
#     N = v * sqrtp1 * nLocalTiles
#     return [sqrtp1, c, v, N]


# [sqrtp1, c, v, N] = CalculateParameters(inpN, inpP)
# P = sqrtp1 * sqrtp1 * c
# nlayr = v // c
# p1 = sqrtp1 * sqrtp1


# prng = RandomState(43)
# inpA = prng.rand(N,N) #int(low=0, high=N, size=(N, N)).astype(np_dtype)

# inpA = np.array([[73, 79, 56, 47, 44, 63, 75, 53],
#                  [79, 138, 64, 94, 100, 114, 96, 77],
#                  [56, 64, 51, 42, 43, 48, 62, 44],
#                  [47, 94, 42, 76, 74, 85, 55, 54],
#                  [44, 100, 43, 74, 87, 86, 66, 54],
#                  [63, 114, 48, 85, 86, 110, 76, 63],
#                  [75, 96, 62, 55, 66, 76, 97, 57],
#                  [53, 77, 44, 54, 54, 63, 57, 53]]).astype(np_dtype)

# # inpA = np.array([[1, 8, 2, 7, 3, 8, 2, 4, 8, 7, 5, 5, 1, 4, 4, 9],
# #                 [8, 4, 9, 2, 8, 6, 9, 9, 3, 7, 7, 7, 8, 7, 2, 8],
# #                 [3, 5, 4, 8, 9, 2, 7, 1, 2, 2, 7, 9, 8, 2, 1, 3],
# #                 [6, 4, 1, 5, 3, 7, 9, 1, 1, 3, 2, 9, 9, 5, 1, 9],
# #                 [8, 7, 1, 2, 9, 1, 1, 9, 3, 5, 8, 8, 5, 5, 3, 3],
# #                 [4, 2, 9, 3, 7, 3, 4, 5, 1, 9, 7, 7, 2, 4, 5, 2],
# #                 [1, 9, 8, 3, 5, 5, 1, 3, 6, 8, 3, 4, 3, 9, 1, 9],
# #                 [3, 9, 2, 7, 9, 2, 3, 9, 8, 6, 3, 5, 5, 2, 2, 9],
# #                 [9, 9, 5, 4, 3, 4, 6, 6, 9, 2, 1, 5, 6, 9, 5, 7],
# #                 [3, 2, 4, 5, 2, 4, 5, 3, 6, 5, 2, 6, 2, 7, 8, 2],
# #                 [4, 4, 4, 5, 2, 5, 3, 4, 1, 7, 8, 1, 8, 8, 5, 4],
# #                 [4, 5, 9, 5, 7, 9, 2, 9, 4, 6, 4, 3, 5, 8, 1, 2],
# #                 [7, 8, 1, 4, 7, 6, 5, 7, 1, 2, 7, 3, 8, 1, 4, 4],
# #                 [7, 6, 7, 8, 2, 2, 4, 6, 6, 8, 3, 6, 5, 2, 6, 5],
# #                 [4, 5, 1, 5, 3, 7, 4, 4, 7, 5, 8, 2, 4, 7, 1, 7],
# #                 [8, 3, 2, 4, 3, 8, 1, 6, 9, 6, 3, 6, 4, 8, 7, 8]]).astype(np_dtype)

# [n, tmp] = np.shape(inpA)
# N = n
# p1 = max(np.floor(n * n / M).astype(int), np.ceil(np.power(P,(2/3))).astype(int))
# sqrtp1 = np.floor(np.sqrt(p1)).astype(int)
# c = np.ceil(P / p1).astype(int)
# t = int(np.ceil(np.ceil(n / v) / sqrtp1) + 1)
# # how many layers each processor compute in each step in A11
# nlayr = v // c


# # Nt is the total number of tiles in one dimension
Nt = math.ceil(N / v)
# # t is number of tiles in one dimension one processor owns. Each tile is [v,v]
t = int(np.ceil(Nt / sqrtp1))
tA11 = int(np.ceil(Nt / sqrtp1))
tA10 = int(np.ceil(Nt / P))

######################################
#### end of global parameters ########
######################################
