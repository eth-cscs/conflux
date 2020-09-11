import numpy as np
#from mpi4py import MPI
import mpi4py
from reference_lu import *
from utils import *
import math


def flipbit(n, k):
    return n ^ (1 << k)


# empty pivoting strategy.
# in step k, k-th rank chooses his first v rows as pivots
def EmptyPivot(k, PivotBuff, A11Buff, A00Buff, A11MaskBuff, curPivots, curPivOrder, pivotIndsBuff, layrK,
               measureComm, commCounter, commRcvCounter):
    global global_vars
    P = global_vars['P']
    v = global_vars['v']
    sqrtp1 = global_vars['sqrtp1']
    c = global_vars['c']

    for p in range(P):
        [pi, pj, pk] = p2X(p)
        if k % sqrtp1 != pi or k % sqrtp1 != pj or pk != 0:
            continue
        # filter A11buff by masked pivot rows (local) and densify it
        rows = A11MaskBuff[p]
        data = A11Buff[p, rows, (k // sqrtp1) *v : (k // sqrtp1 + 1) *v]

        # flush the buffer
        PivotBuff[p] = np.zeros(PivotBuff[p].shape)
        PivotBuff[p, :len(data)] = data

        # In round robin fashion, only one rank chooses pivots among its local rows
        if k % sqrtp1 == pi:
            [PivotBuff[p, :v, :], A00Buff[p], Perm] = LUPnoTile(PivotBuff[p, :len(data)])
            grows = l2gnoTile(np.nonzero(rows)[0], pi)
            gpivots = (grows @ Perm)[:v]
            [lpivots, loffsets] = g2lnoTile(gpivots)

            # locally set curPivots
            curPivots[p, 0] = len(lpivots[pi])
            curPivots[p, 1 : 1 + len(lpivots[pi])] = lpivots[pi]
            curPivOrder[p, :len(loffsets[pi])] = loffsets[pi]
            pivotIndsBuff[p, k*v:(k+1)*v] = gpivots

            # -------------------------- #
            # !!!!! COMMUNICATION !!!!!! #
            # Sending pivots:
            for pi_rcv in lpivots:
                for pj_rcv in range(sqrtp1):
                    for pk_rcv in range(c):
                        p_rcv = X2p(pi_rcv, pj_rcv, pk_rcv)
                        curPivots[p_rcv, 0] = len(lpivots[pi_rcv])
                        curPivots[p_rcv, 1: 1 +len(lpivots[pi_rcv])] = np.copy(lpivots[pi_rcv])
                        curPivOrder[p_rcv, :len(loffsets[pi_rcv])] = np.copy(loffsets[pi_rcv])
            # Sending A00Buff:
            for pi_rcv in range(sqrtp1):
                for pj_rcv in range(sqrtp1):
                    for pk_rcv in range(c):
                        p_rcv = X2p(pi_rcv, pj_rcv, pk_rcv)
                        A00Buff[p_rcv] = np.copy(A00Buff[p])
                        pivotIndsBuff[p_rcv, k * v:(k + 1) * v] = gpivots

    return



# finds v consecutive pivots
# !!!!!!!!!!!! this is a collective call. Parallelism is further down in this function
# note that there is no p in the function argument
# pivots : np.zeros(1,N) which gets gradually filled up with consecutive chosen pivot row indices
# remaining : range(N) holds which rows were not chosen as pivots yet.
# With each step, we reduce the number of remaining rows.

def TournPivotNoTile(k, PivotBuff, A11Buff, A00Buff, A11MaskBuff, growsBuff, curPivots, curPivOrder, pivotIndsBuff,
                     layrK, measureComm, commCounter, commRcvCounter):
    global global_vars
    P = global_vars['P']
    v = global_vars['v']
    sqrtp1 = global_vars['sqrtp1']
    c = global_vars['c']

    # ---------------- FIRST STEP ----------------- #
    # in first step, we do pivot on the whole PivotBuff array (may be larger than [2v, v]
    # local computation step
    for p in range(P):
        [pi, pj, pk] = p2X(p)
        if pj != k % sqrtp1 or pk != layrK:
            continue

        # grows holds global row indices of current processor's local active rows
        rows = A11MaskBuff[p]
        grows = l2gnoTile(np.nonzero(rows)[0], pi)
        growsBuff[p, :len(grows)] = grows
        data = A11Buff[p, rows, (k // sqrtp1) * v: (k // sqrtp1 + 1) * v]

        # flush the buffer
        PivotBuff[p] = np.zeros(PivotBuff[p].shape)
        PivotBuff[p, :len(data)] = data

        # tricky part! to preserve the order of the rows between swapping pairs (e.g., if ranks 0 and 1 exchange their
        # candidate rows), we want to preserve that candidates of rank 0 are always above rank 1 candidates. Otherwise,
        # we can get inconsistent results. That's why,in each communication pair, higher rank puts his candidates below:

        # find with which rank we will communicate
        # ANOTHER tricky part ! If sqrtp1 is not 2^n, then we will not have a nice butterfly communication graph.
        # that's why with the flipBit strategy, src_pi can actually be larger than sqrtp1
        src_pi = min(flipbit(pi, 0), sqrtp1 - 1)
        src_p = X2p(src_pi, pj, pk)

        if src_p < p:
            # move my candidates below
            [PivotBuff[p, v: 2*v, :], _, Perm] = LUPnoTile(PivotBuff[p])
            growsBuff[p, v: 2*v] = (growsBuff[p] @ Perm)[:v]
        else:
            [PivotBuff[p, :v, :], _, Perm] = LUPnoTile(PivotBuff[p])
            growsBuff[p, :v] = (growsBuff[p] @ Perm)[:v]

    # ------------- REMAINING STEPS -------------- #
    # now we do numRounds parallel steps which synchronization after each step
    numRounds = int(math.ceil(np.log2(sqrtp1)))

    for r in range(numRounds):
        # COMMUNICATION #
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pj != k % sqrtp1 or pk != layrK:
                continue
            # find with which rank we will communicate
            src_pi = min(flipbit(pi, r), sqrtp1 - 1)
            src_p = X2p(src_pi, pj, pk)

            # see comment above for the communication pattern:
            if src_p > p:
                PivotBuff[p, v:2 * v, :] = np.copy(PivotBuff[src_p, v: 2*v, :])
                growsBuff[p, v:2*v] = np.copy(growsBuff[src_p, v: 2*v])
            else:
                PivotBuff[p, :v, :] = np.copy(PivotBuff[src_p, :v, :])
                growsBuff[p, :v] = np.copy(growsBuff[src_p, :v])

            # --- comm counters --- #
            if measureComm:
                if src_p != p:
                    data_size = np.size(PivotBuff[p, :v, :])
                    commCounter[p] += data_size
                    commRcvCounter[p, 2] += data_size
            # - end comm counters - #

        # local computation step
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pj != k % sqrtp1 or pk != layrK:
                continue
            # find local pivots
            if r == numRounds - 1:
                [PivotBuff[p, :v, :], A00Buff[p], Perm] = LUPnoTile(PivotBuff[p, :2*v])
                growsBuff[p, :v] = (growsBuff[p, :2 * v] @ Perm)[:v]
            else:
                src_pi = min(flipbit(pi, r+1), sqrtp1 - 1)
                src_p = X2p(src_pi, pj, pk)
                if src_p < p:
                    # move my candidates below
                    [PivotBuff[p, v:2 * v, :], _, Perm] = LUPnoTile(PivotBuff[p, :2*v])
                    growsBuff[p, v: 2 * v] = (growsBuff[p, :2*v] @ Perm)[:v]
                else:
                    [PivotBuff[p, :v, :], _, Perm] = LUPnoTile(PivotBuff[p, :2*v])
                    growsBuff[p, :v] = (growsBuff[p, :2 * v] @ Perm)[:v]

# distribute A00buff
# !! COMMUNICATION !!
    for p in range(P):
        [pi, pj, pk] = p2X(p)
        if pj != k % sqrtp1 or pk != layrK:
            continue
        [lpivots, loffsets] = g2lnoTile(growsBuff[p, :v])

        # locally set curPivots
        curPivots[p, 0] = len(lpivots[pi])
        curPivots[p, 1: 1 + len(lpivots[pi])] = lpivots[pi]
        curPivOrder[p, :len(loffsets[pi])] = loffsets[pi]
        pivotIndsBuff[p, k * v:(k + 1) * v] = growsBuff[p, :v]

        # -------------------------- #
        # !!!!! COMMUNICATION !!!!!! #
        # Sending pivots:
        for pj_rcv in range(sqrtp1):
            for pk_rcv in range(c):
                p_rcv = X2p(pi, pj_rcv, pk_rcv)
                curPivots[p_rcv, 0] = len(lpivots[pi])
                curPivots[p_rcv, 1: 1 + len(lpivots[pi])] = np.copy(lpivots[pi])
                curPivOrder[p_rcv, :len(loffsets[pi])] = np.copy(loffsets[pi])
        # Sending A00Buff:
        for pj_rcv in range(sqrtp1):
            for pk_rcv in range(c):
                p_rcv = X2p(pi, pj_rcv, pk_rcv)
                A00Buff[p_rcv] = np.copy(A00Buff[p])
                pivotIndsBuff[p_rcv, k * v:(k + 1) * v] = growsBuff[p, :v]
    return




# finds v consecutive pivots
# !!!!!!!!!!!! this is a collective call. Parallelism is further down in this function
# note that there is no p in the function argument
# pivots : np.zeros(1,N) which gets gradually filled up with consecutive chosen pivot row indices
# remaining : range(N) holds which rows were not chosen as pivots yet.
# With each step, we reduce the number of remaining rows.

def TournPivot(k, PivotBuff, A00buff, pivots, A10Mask, A11Mask, layrK, measureComm, commCounter, commRcvCounter):
    global global_vars
    P = global_vars['P']
    v = global_vars['v']
    sqrtp1 = global_vars['sqrtp1']
    c = global_vars['c']

    # ---------------- FIRST STEP ----------------- #
    # in first step, we do pivot on the whole PivotBuff array (may be larger than [2v, v]
    # local computation step
    for p in range(P):
        [pi, pj, pk] = p2X(p)
        if pj != k % sqrtp1 or pk != layrK:
            continue

        # tricky part! to preserve the order of the rows between swapping pairs (e.g., if ranks 0 and 1 exchange their
        # candidate rows), we want to preserve that candidates of rank 0 are always above rank 1 candidates. Otherwise,
        # we can get inconsistent results. That's why,in each communication pair, higher rank puts his candidates below:

        # find with which rank we will communicate
        # ANOTHER tricky part ! If sqrtp1 is not 2^n, then we will not have a nice butterfly communication graph.
        # that's why with the flipBit strategy, src_pi can actually be larger than sqrtp1
        src_pi = min(flipbit(pi, 0), sqrtp1 - 1)
        src_p = X2p(src_pi, pj, pk)

        if src_p < p:
            # move my candidates below
            [PivotBuff[p, v: 2*v, :], _] = LUP(PivotBuff[p])
        else:
            [PivotBuff[p, :v, :], _] = LUP(PivotBuff[p])

    # ------------- REMAINING STEPS -------------- #
    # now we do numRounds parallel steps which synchronization after each step
    numRounds = int(math.ceil(np.log2(sqrtp1)))

    for r in range(numRounds):
        # COMMUNICATION #
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pj != k % sqrtp1 or pk != layrK:
                continue
            # find with which rank we will communicate
            src_pi = min(flipbit(pi, r), sqrtp1 - 1)
            src_p = X2p(src_pi, pj, pk)

            # see comment above for the communication pattern:
            if src_p > p:
                PivotBuff[p, v:2 * v, :] = np.copy(PivotBuff[src_p, v: 2*v, :])
            else:
                PivotBuff[p, :v, :] = np.copy(PivotBuff[src_p, :v, :])

            # --- comm counters --- #
            if measureComm:
                if src_p != p:
                    data_size = np.size(PivotBuff[p, :v, :])
                    commCounter[p] += data_size
                    commRcvCounter[p, 2] += data_size
            # - end comm counters - #

        # local computation step
        for p in range(P):
            [pi, pj, pk] = p2X(p)
            if pj != k % sqrtp1 or pk != layrK:
                continue
            # find local pivots
            if r == numRounds - 1:
                [PivotBuff[p, :v, :], A00buff[p]] = LUP(PivotBuff[p, :2*v])
            else:
                src_pi = min(flipbit(pi, r+1), sqrtp1 - 1)
                src_p = X2p(src_pi, pj, pk)
                if src_p < p:
                    # move my candidates below
                    [PivotBuff[p, v:2 * v, :], _] = LUP(PivotBuff[p, :2*v])
                else:
                    [PivotBuff[p, :v, :], _] = LUP(PivotBuff[p, :2*v])

# distribute A00buff
# !! COMMUNICATION !!
    for p in range(P):
        [pi, pj, pk] = p2X(p)
        if pj != k % sqrtp1 or pk != layrK:
            continue

        curPivots = PivotBuff[p, :v, 0].astype(int)
        pivots[p, k * v: (k + 1) * v] = curPivots

        for destpj in range(sqrtp1):
            for destpk in range(c):
                destp = X2p(pi,destpj, destpk)
                A00buff[destp] = np.copy(A00buff[p])
                pivots[destp, k * v: (k + 1) * v] = np.copy(pivots[p, k * v: (k + 1) * v])

                # --- comm counters --- #
                if measureComm:
                    if destp != p:
                        data_size = np.size(A00buff[p]) + np.size(pivots[p, k * v: (k + 1) * v])
                        commCounter[destp] += data_size
                        commRcvCounter[destp, 2] += data_size
                # - end comm counters - #

    # now every processor locally updates its local masks. No communication involved
    for p in range(P):
        [pi, pj, pk] = p2X(p)
        curPivots = pivots[p, k * v: (k + 1) * v]
        for piv in curPivots:
            if 114 == piv:
                tmp = 1
            # translating global row index (piv) to global tile index and row index inside the tile
            [gti, lri] = gr2gt(piv)
            [pown, lti] = g2lA10(gti)
            if p == pown:
                A10Mask[p, lti, lri] = 0
            [pown, lti] = g2l(gti)

            if pi == pown:
                A11Mask[p, lti, lri] = 0
    return



choosePivot = TournPivot
