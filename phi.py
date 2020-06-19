import numpy as np
from scipy import linalg


def phi(Q, xi):
    '''
    Lengths of springs
    :param Q:
    :param xi:
    :return:
    '''
    return np.sqrt(np.sum(np.square(np.matmul(Q.transpose(), xi)), axis=1))


def K(Q, xi):
    '''
    Normalized directions of springs
    :param Q:
    :param xi:
    :return:
    '''
    d = xi.shape[1]
    return np.divide(
        np.matmul(Q.transpose(), xi),
        np.tile(phi(Q, xi), (d, 1)).transpose()
    )


def d_xi_phi(Q, xi):
    '''
    :param Q:
    :param xi:
    :return: m x n x d array representing D_xi phi
    '''
    d = xi.shape[1]
    Q1 = np.expand_dims(Q.transpose(), axis=2)
    Q2 = np.tile(Q1, (1, 1, d))

    n = Q.shape[0]
    N1 = np.swapaxes(np.expand_dims(K(Q, xi), axis=2), 1, 2)
    N2 = np.tile(N1, (1, n, 1))

    return np.multiply(N2, Q2)


def tensor_to_matrix(t):
    """
    :param t: m x n x d numpy array
    :return: m x (nd) numpy array: ( t[m x n x (1)]  t[m x n x (2)] ... t[m x n x (d)])
    """
    (m, n, d) = t.shape
    result = np.zeros((m,n*d))
    for k in range(0, d):
        result[:, range(n*k, n*(k+1))] = t[:, :, k]
    return result


def matrix_to_tensor(matrix, d):
    """
    :param matrix: m x (nd) numpy array: ((m x n x 1) (m x n x 2) ... (m x n x d)
    :param d: 3nd dimension size
    :return: m x n x d numpy array
    """
    m = matrix.shape[0]
    n = matrix.shape[1]//d
    result=np.zeros((m, n, d))
    for k in range(0, d):
        result[:, :, k] = matrix[:, range(n*k, n*(k+1))]
    return result


