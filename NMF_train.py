import numpy as np
from scipy.sparse import coo_matrix, load_npz
import math


def train(R, iterations, k, alpha=0.0001, beta=0.001, target_error=None):
    """
    R is scipy.sparse.coo_matrix
    The error doesn't compute zero term in R
    R = P * Q
    (m*n) = (m*k) * (k*n)
    alpha: learning rate
    beta: regular term
    """
    assert isinstance(R, coo_matrix)

    def _sum_kPQ(i, j):
        sum = 0
        for kk in range(k):
            sum = sum + P[i, kk] * Q[kk, j]
        return sum


    m, n = np.shape(R)
    P = np.mat(np.random.random([m, k]))
    Q = np.mat(np.random.random([k, n]))
    R_indices = np.mat([R.row, R.col]).transpose()
    R_data = R.data
    error = R.copy()
    print("compute first error...")
    for c in range(R_indices.shape[0]):
        i = R_indices[c, 0]
        j = R_indices[c, 1]
        error[i, j] = R_data[c] - _sum_kPQ(i, j)
        if (c + 1) % 100000 == 0:
            print("error iter %d in %d" % ((c + 1), R_indices.shape[0]))

    error = R.copy()
    print("begin to compute first error...")
    for c in range(R_indices.shape[0]):
        i = R_indices[c, 0]
        j = R_indices[c, 1]
        error.data[c] = R_data[c] - _sum_kPQ(i, j)
        if (c+1) % 100000 == 0:
            print("finish error iter %d in %d" % ((c+1), R_indices.shape[0]))

    for i_count in range(iterations):
        print("iteration %d..." % (i_count+1))
        print("begin to iterate...")
        for c in range(R_indices.shape[0]):
            i = R_indices[c, 0]
            j = R_indices[c, 1]
            P_next = P.copy()
            Q_next = Q.copy()
            e = error.data[c]
            for kk in range(k):
                P_next[i, kk] = P[i, kk] - 2*alpha*(e*Q[kk, j] - beta*P[i, kk])
                Q_next[kk, j] = Q[kk, j] - 2*alpha*(e*P[i, kk] - beta*Q[kk, j])
                P = P_next
                Q = Q_next
            if (c+1) % 1000 == 0:
                print("finish point iter %d in %d" % ((c+1), R_indices.shape[0]))
        print("begin to compute loss...")
        loss = 0
        for c in range(R_indices.shape[0]):
            i = R_indices[c, 0]
            j = R_indices[c, 1]
            error.data[c] = R_data[c] - _sum_kPQ(i, j)
            loss = loss + math.fabs(error.data[c])
            if (c+1) % 100000 == 0:
                print("finish loss iter %d in %d" % ((c+1), R_indices.shape[0]))

        if target_error is not None:
            if loss < target_error:
                print("finish training at iteration %d" % (i_count + 1))
                print("loss: %0.4f" % loss)
                break
        print("iteration %d loss: %0.4f" % ((i_count + 1), loss))
    return P, Q


if __name__ == "__main__":
    R = load_npz("data/r-100000.npz").tocoo()
    P, Q = train(R, 1, k=50)
    np.savez("./data/p-100000.npz", P)
    np.savez("./data/q-100000.npz", Q)
    # P = np.load("./data/p-1000000.npz")['arr_0']
