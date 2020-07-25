import math
import numpy as np
from numpy.random import RandomState
from typing import Tuple
from settings import *
from collections import defaultdict
# Infix copied from http://code.activestate.com/recipes/384122/

# definition of an Infix operator class
# this recipe also works in jython
# calling sequence for the infix is either:
#  x |op| y
# or:
# x <<op>> y

class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)


# ceiling as this infix lambda
l = Infix(lambda x,y : (x - 1) // y + 1)


def ceil(a,b):
    return (a - 1) // b + 1


def l2g(pi, ind) -> int:
    return ind * sqrtp1 + pi


# input: global tile index (in one dimension)
# return [pi, lti]
# where pi is processor number (in that direction) owning this global tile
# and lti is the local tile index (in that dimension)
# that is: B[gind, _] is owned by Aloc[ X2p( pi, _, _), lti, _ ]
def g2l(gind) -> Tuple[int, int]:
    return [gind % sqrtp1, gind // sqrtp1]


# input: global ROW indices
# returns: dictionary of owning ranks and their LOCAL ROW indices
# AND dictionary of owning ranks and which pivot row it it
def g2lnoTile(grows):
    lrows = defaultdict(list)
    loffsets = defaultdict(list)
    i = 0
    for grow in grows:
        # we are in the global tile:
        gT = grow // v
        # which is owned by:
        pOwn = int(gT % sqrtp1)
        # and this is a local tile:
        lT = gT // sqrtp1
        # and inside this tile it is a row number:
        lR = grow % v
        # which is a No-Tile row number:
        lRNT = int(lR + lT * v)
        lrows[pOwn].append(lRNT)
        loffsets[pOwn].append(i)
        i += 1
    return [lrows, loffsets]


# input: local ROW indices lrows of processor pi
# return dictionary of owning ranks and their LOCAL ROW indices
def l2gnoTile(lrows, pi):
    grows = []
    for lrow in lrows:
        # we are in the local tile:
        lT = lrow // v
        # and inside this tile it is a row number:
        lR = lrow % v
        # which is a global tile:
        gT = lT * sqrtp1 + pi
        # which gives a global row index:
        grows.append(lR + gT * v)
    return np.array(grows)


def g2lA10(gti):
    lti = gti // P
    p = gti % P
    return [p, lti]


def l2gA10(p,lti):
    gti = lti * P + p
    return gti


# global row index to global tile index and row inside the tile
def gr2gt(gri):
    gti = int(gri // v)
    lri = int(gri % v)
    return [gti, lri]



def p2X(p):
    pk = np.floor(p / p1).astype(int)
    p -= pk * p1
    pj = np.floor(p / np.sqrt(p1)).astype(int)
    pi = np.mod(p, np.sqrt(p1)).astype(int)
    return [pi, pj, pk]


def X2p(pi, pj, pk):
    return (pi + np.sqrt(p1) * pj + p1 * pk).astype(int)

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b)



global_vars = {}

def init_globals(inpN=16, inpP=8, seed=42):
# def init_globals(ppp=2, lrs=1, nLocalTiles=4, N=16, P=8, seed=42):
    global global_vars

    [sqrtp1, c, v, N] = CalculateParameters(inpN, inpP)
    P = sqrtp1 * sqrtp1 * c
    nlayr = v // c
    p1 = sqrtp1 * sqrtp1

    # global_vars['ppp'] = ppp
    # global_vars['lrs'] = lrs
    # global_vars['nLocalTiles'] = nLocalTiles
    global_vars['N'] = N
    global_vars['n'] = N
    global_vars['P'] = P

    # v = ppp * lrs
    M = N * N
    Mp = M

    if seed:
        prng = RandomState(seed)
        inpA = prng.rand(N, N)
    else:
        seed = 42
        inpA = None

    tmpA = np.array([[1, 8, 2, 7, 3, 8, 2, 4, 8, 7, 5, 5, 1, 4, 4, 9],
                     [8, 4, 9, 2, 8, 6, 9, 9, 3, 7, 7, 7, 8, 7, 2, 8],
                     [3, 5, 4, 8, 9, 2, 7, 1, 2, 2, 7, 9, 8, 2, 1, 3],
                     [6, 4, 1, 5, 3, 7, 9, 1, 1, 3, 2, 9, 9, 5, 1, 9],
                     [8, 7, 1, 2, 9, 1, 1, 9, 3, 5, 8, 8, 5, 5, 3, 3],
                     [4, 2, 9, 3, 7, 3, 4, 5, 1, 9, 7, 7, 2, 4, 5, 2],
                     [1, 9, 8, 3, 5, 5, 1, 3, 6, 8, 3, 4, 3, 9, 1, 9],
                     [3, 9, 2, 7, 9, 2, 3, 9, 8, 6, 3, 5, 5, 2, 2, 9],
                     [9, 9, 5, 4, 3, 4, 6, 6, 9, 2, 1, 5, 6, 9, 5, 7],
                     [3, 2, 4, 5, 2, 4, 5, 3, 6, 5, 2, 6, 2, 7, 8, 2],
                     [4, 4, 4, 5, 2, 5, 3, 4, 1, 7, 8, 1, 8, 8, 5, 4],
                     [4, 5, 9, 5, 7, 9, 2, 9, 4, 6, 4, 3, 5, 8, 1, 2],
                     [7, 8, 1, 4, 7, 6, 5, 7, 1, 2, 7, 3, 8, 1, 4, 4],
                     [7, 6, 7, 8, 2, 2, 4, 6, 6, 8, 3, 6, 5, 2, 6, 5],
                     [4, 5, 1, 5, 3, 7, 4, 4, 7, 5, 8, 2, 4, 7, 1, 7],
                     [8, 3, 2, 4, 3, 8, 1, 6, 9, 6, 3, 6, 4, 8, 7, 8]]).astype(np.float64)

    if N == 16:
        inpA = tmpA

    # if N == 32:
    #     inpA = (prng.rand(N, N) * 10).astype(np.int64).astype(np.float64)

    # p1 = max(math.floor(N * N / M), math.ceil(math.pow(P,(2/3))))
    # sqrtp1 = math.floor(math.sqrt(p1))
    # c = math.ceil(P / p1)
    # nlayr = v // c
    Nt = math.ceil(N / v)
    t = math.ceil(Nt / sqrtp1) + 1
    tA11 = math.ceil(Nt / sqrtp1)
    tA10 = math.ceil(Nt / P)

    global_vars['seed'] = seed
    global_vars['P'] = inpP
    global_vars['v'] = v
    global_vars['M'] = M
    global_vars['Mp'] = Mp
    global_vars['inpA'] = inpA
    global_vars['p1'] = p1
    global_vars['sqrtp1'] = sqrtp1
    global_vars['c'] = c
    global_vars['nlayr'] = nlayr
    global_vars['Nt'] = Nt
    global_vars['t'] = t
    global_vars['tA11'] = tA11
    global_vars['tA10'] = tA10
