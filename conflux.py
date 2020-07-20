import numpy as np
from numpy.random import RandomState
import scipy.linalg as la
from scipy.linalg import solve_triangular as trsm
from pivoting import *
import utils
from collections import defaultdict

np.set_printoptions(precision=4, suppress=True)
parrange = range

def LU_rep(A, measureComm, commCounter, commRcvCounter):
    global global_vars

    # seed = global_vars['seed']
    # prng = RandomState(seed)
    #
    N = global_vars['N']
    P = global_vars['P']
    v = global_vars['v']
    sqrtp1 = global_vars['sqrtp1']
    c = global_vars['c']
    nlayr = global_vars['nlayr']
    Nt = global_vars['Nt']
    tA11 = global_vars['tA11']
    tA10 = global_vars['tA10']
    # local n
    Nl = tA11 * v

    B = np.copy(A)
    Perm = np.eye(N)

    A00Buff = np.zeros([P, v, v])
    A10Buff = np.zeros([P, tA10*v, v])
    A10BuffRcv = np.zeros([P, Nl, nlayr])
    A01Buff = np.zeros([P, v, Nl])
    A01BuffRcv = np.zeros([P, nlayr, Nl])
    A11Buff = np.zeros([P, Nl, Nl])

    A11MaskBuff = np.ones([P, Nl]).astype(bool)

    PivotBuff = np.zeros([P, v * max(2, tA11), v])
    # the "+1" in the number of columns is for a number of current pivots for this rank
    curPivots = np.zeros([P, v+1]).astype(int)
    # indicates the position to which local pivots will be sent in A01Buff
    curPivOrder = np.zeros([P, v]).astype(int)
    PivotA11ReductionBuff = np.zeros([P, tA11 * v, v])
    pivotIndsBuff = -1 * np.ones([P, N]).astype(int)

    # ------------------------------------------------------------------- #
    # ------------------ INITIAL DATA DISTRIBUTION ---------------------- #
    # ------------------------------------------------------------------- #
    for p in range(P):
        # get 3d processor decomposition coordinates
        [pi, pj, pk] = p2X(p)

        # we distribute only A11, as anything else depends on the first pivots

        # ----- A11 ------ #
        # only layer pk == 0 owns initial data
        if pk == 0:
            for lti in range(tA11):
                gti = l2g(pi, lti)
                for ltj in range(tA11):
                    gtj = l2g(pj, ltj)
                    A11Buff[p, lti*v : (lti+1)*v, ltj*v : (ltj+1)*v] = \
                        B[gti * v: (gti + 1) * v, gtj * v: (gtj + 1) * v]

    # ---------------------------------------------- #
    # ----------------- MAIN LOOP ------------------ #
    # 0. reduce first tile column from A11buff to PivotA11ReductionBuff
    # 1. coalesce PivotA11ReductionBuff to PivotBuff and scatter to A10buff
    # 2. find v pivots and compute A00
    # 3. reduce pivot rows from A11buff to PivotA11ReductionBuff
    # 4. scatter PivotA01ReductionBuff to A01Buff
    # 5. compute A10 and broadcast it to A10BuffRecv
    # 6. compute A01 and broadcast it to A01BuffRecv
    # 7. compute A11
    # ---------------------------------------------- #

# now k is a step number
    for k in range(Nt):
        # global current offset
        goff = k * v
        # local current offset
        loff = (k // sqrtp1) * v

        # in this step, layrK is the "lucky" one to receive all reduces
        # layrK = int(prng.randint(c, size = 1))
        layrK = 0

        # ----------------------------------------------------------------- #
        # 0. reduce first tile column of A11buff                            #
        # ----------------------------------------------------------------- #
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            # Currently, we dump everything to processors in layer pk == layrK, and only this layer choose pivots
            # that is, each processor [pi, pj, pk] sends to [pi, pj, layrK]
            p_rcv = X2p(pi, pj, layrK)

            # flush the buffer:
            curPivots[p, 0] = 0

            if pk == layrK:
                continue

            # reduce first tile column. In this part, only pj == k % sqrtp1 participate:
            if pj == k % sqrtp1:
                # filter which rows of this tile should be reduced:
                rows = A11MaskBuff[p]
                A11Buff[p_rcv, rows, loff : loff + v] += A11Buff[p, rows, loff : loff + v]

                # --- comm counters --- #
                if measureComm:
                    if p_rcv != p:
                        data_size = np.size(A11Buff[p, rows, loff : loff + v])
                        commCounter[p_rcv] += data_size
                        commRcvCounter[p_rcv, 0] += data_size
                # - end comm counters - #

        # ---------------------------------------------- #
        # 1. find v pivots and compute A00 ------------- #
        # ---------------------------------------------- #

        EmptyPivot(k, PivotBuff, A11Buff, A00Buff, A11MaskBuff, curPivots, curPivOrder, pivotIndsBuff, layrK,
                   measureComm, commCounter, commRcvCounter)
        # TournPivot(k, PivotBuff, A00Buff, pivotIndsBuff, A10MaskBuff, A11MaskBuff, layrK, measureComm, commCounter,
        #            commRcvCounter)

        # ---------------------------------------------- #
        # 2. reduce v pivot rows to  A11buff             #
        # ---------------------------------------------- #
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            # Currently, we dump everything to processors in layer pk == layrK, and only this layer choose pivots
            # that is, each processor [pi, pj, pk] sends to [pi, pj, layrK]
            p_rcv = X2p(pi, pj, layrK)

            # update the row mask:
            # number of pivots to be updated
            numPivots = curPivots[p][0]
            A11MaskBuff[p][curPivots[p][1: (1 + numPivots)]] = False

            if pk == layrK:
                continue

            for pivot in curPivots[p, 1:]:
                A11Buff[p_rcv, pivot, loff: ] += A11Buff[p, pivot, loff: ]

                # --- comm counters --- #
                if measureComm:
                    if p_rcv != p:
                        data_size = np.size(A11Buff[p, pivot, loff: ])
                        commCounter[p_rcv] += data_size
                        commRcvCounter[p_rcv, 2] += data_size
                # - end comm counters - #

        # -------------------------------------------------- #
        # 3. distribute v pivot rows from A11buff to A01Buff #
        # here, only processors pk == layrK participate      #
        # -------------------------------------------------- #
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pk != layrK:
                continue

            # we sent v pivot rows, which may be spread over all ranks, to ranks with pi == k % sqrt1 and pk == layrK
            p_rcv = X2p(k % sqrtp1, pj, layrK)

            for i in range(curPivots[p,0]):
                pivot = curPivots[p, i+1]
                offset = curPivOrder[p, i]
                A01Buff[p_rcv, offset, loff:] = np.copy(A11Buff[p, pivot, loff:])

                # --- comm counters --- #
                if measureComm:
                    if p_rcv != p:
                        data_size = np.size(A11Buff[p, pivot, loff:])
                        commCounter[p_rcv] += data_size
                        commRcvCounter[p_rcv, 2] += data_size
                # - end comm counters - #

        # ---------------------------------------------- #
        # 4. compute A10 and broadcast it to A10BuffRecv #
        # ---------------------------------------------- #
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pk != layrK or pj != k % sqrtp1:
                continue
            A00U = np.triu(A00Buff[p])

            # filter A11buff by masked pivot rows (local)
            rows = A11MaskBuff[p]
            # this is basically a sparse-dense A10 = A10 * U^(-1)   (BLAS tiangular solve) with A10 sparse and U dense
            A11Buff[p, rows, loff: loff + v] = A11Buff[p, rows, loff: loff + v] @ np.linalg.inv(A00U)

            # -- BROADCAST -- #
            # after compute, send it to sqrt(p1) * c processors
            for pk_rcv in range(c):
                # for the receive layer pk_rcv, its A10BuffRcv is formed by the following columns of A11Buff[p]
                colStart = loff + pk_rcv*nlayr
                colEnd   = loff + (pk_rcv+1)*nlayr
                # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for pj_rcv in range(sqrtp1):
                    p_rcv = X2p(pi, pj_rcv, pk_rcv)
                    A10BuffRcv[p_rcv, rows] = np.copy(A11Buff[p, rows, colStart : colEnd])

                    # --- comm counters --- #
                    if measureComm:
                        if p_rcv != p:
                            data_size = np.size(A11Buff[p, rows, colStart : colEnd])
                            commCounter[p_rcv] += data_size
                            commRcvCounter[p_rcv, 5] += data_size
                    # - end comm counters - #

        # ---------------------------------------------- #
        # 5. compute A01 and broadcast it to A01BuffRecv #
        # ---------------------------------------------- #
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            # here, only ranks which own data in A01Buff (step 3) participate
            if pk != layrK or pi != k % sqrtp1:
                continue

            A00L = np.tril(A00Buff[p], k=-1) + np.eye(v)
            # this is a dense-dense A01 =  L^(-1) * A01
            A01Buff[p, :, loff: ] = np.linalg.inv(A00L) @  A01Buff[p, :, loff: ]

            # -- BROADCAST -- #
            # after compute, send it to sqrt(p1) * c processors
            for pk_rcv in range(c):
                # for the receive layer pk_rcv, its A01BuffRcv is formed by the following rows of A01Buff[p]
                rowStart = pk_rcv * nlayr
                rowEnd = (pk_rcv + 1) * nlayr
                # all pjs receive the same data A11Buff[p, rows, colStart : colEnd]
                for pi_rcv in range(sqrtp1):
                    p_rcv = X2p(pi_rcv, pj, pk_rcv)
                    A01BuffRcv[p_rcv, :, loff: ] = np.copy(A01Buff[p, rowStart:rowEnd, loff:])

                    # --- comm counters --- #
                    if measureComm:
                        if p_rcv != p:
                            data_size = np.size(A01Buff[p, rowStart:rowEnd, (k+1) * v:])
                            commCounter[p_rcv] += data_size
                            commRcvCounter[p_rcv, 5] += data_size
                    # - end comm counters - #

        # ----------------------------------------------------------------- #
        # ------------------------- DEBUG ONLY ---------------------------- #
        # ----------- STORING BACK RESULTS FOR VERIFICATION --------------- #

        # remaining = np.setdiff1d(np.array(range(N)), pivotIndsBuff[0])
        pivots = pivotIndsBuff[0, goff: goff + v]

        # storing back A10
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pj != k % sqrtp1 or pk != layrK:
                continue
            grinds = l2gnoTile(np.arange(0,Nl)[A11MaskBuff[p]], pi)
            if grinds.any():
                B[grinds, goff: goff + v] = np.copy(A11Buff[p, A11MaskBuff[p], loff: loff + v])

        # storing back A01
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pi != k % sqrtp1 or pk != layrK:
                continue
            if k % sqrtp1 > pj:
                soff = loff + v
            else:
                soff = loff
            gcinds = l2gnoTile(np.arange(soff, Nl), pj)
            if gcinds.any():
                B[pivots[:, np.newaxis], gcinds] = np.copy(A01Buff[p, :, soff: ])

        # storing back A00
        B[pivots, goff: goff + v] = np.copy(A00Buff[0])
        a = 1

        # ----------------------------------------------------------------- #
        # ----------------------END OF DEBUG ONLY ------------------------- #
        # ----------------------------------------------------------------- #

        # ---------------------------------------------- #
        # 6. compute A11  ------------------------------ #
        # ---------------------------------------------- #
        for p in range(P):
            # filter which rows of this tile should be processed:
            rows = A11MaskBuff[p]
            A11Buff[p, rows,  loff:] -= A10BuffRcv[p, rows] @ A01BuffRcv[p, :, loff:]

        a = 1

    # recreate the permutation matrix
    PP = np.zeros([N,N])
    C = np.zeros([N,N])
    for i in range(N):
        C[i, :] = B[pivotIndsBuff[0,i], :]
        PP[i, :] = Perm[pivotIndsBuff[0, i], :]
    return [C, PP]



def main():
    init_globals(inpN=inpN, inpP=inpP)
    global global_vars
    inpA = global_vars['inpA']
    N = global_vars['N']
    P = global_vars['P']
    v = global_vars['v']
    sqrtp1 = global_vars['sqrtp1']
    c = global_vars['c']
    Nt = global_vars['Nt']
    tA10 = global_vars['tA10']
    tA11 = global_vars['tA11']

    # ---- comm Vol debug only ---- #
    measureComm = True
    commCounter = np.zeros([P]).astype(int)
    commRcvCounter = np.zeros([P, 8]).astype(int)
    # -- end comm Vol debug only -- #

    lrows = [1,2]
    grows = [2,3,5,9, 11, 12]
    p1 = 0
    p2 = 7
    [pi1, pj1, pk1] = p2X(p1)
    [pi2, pj2, pk2] = p2X(p2)
    lrowsOut = g2lnoTile(grows)
    growsOut = l2gnoTile(lrows, pi1)
    growsOut2 = l2gnoTile(lrows, pi2)


    [B, Perm] = LU_rep(inpA, measureComm, commCounter, commRcvCounter)

    U = np.triu(B)
    U = U[:N, :]
    L = np.tril(B) - np.diag(np.diag(B)) + np.eye(N)
    res = L @ U - Perm @ inpA
    residual = np.linalg.norm(res)
    print('------------')
    print('residual = ' + str(residual))

    # ---- commVol debugging ---- #
    maxComm = np.max(commCounter)
    minComm = np.min(commCounter)
    totalComm = np.sum(commCounter)

    p_maxComm = np.argmax(commCounter)
    p_minComm = np.argmin(commCounter)
    print('N: ' + str(N) + ', P: ' + str(P) + ', v: ' + str(v) + ', c: ' +
          str(c) + ', sqrtp1: ' + str(sqrtp1) + ', Nt: ' + str(Nt) + ', tA10: ' +
          str(tA10) + ', tA11: ' + str(tA11))
    print('maxComm: ' + str(maxComm))
    print('minComm: ' + str(minComm))
    print('totalComm: ' + str(totalComm))
    print('p_maxComm: ' + str(p_maxComm))
    print('p_minComm: ' + str(p_minComm))
    # -- end commVol debugging -- #

    assert np.linalg.norm(res) <= 1e-5


if __name__ == '__main__':
    main()
