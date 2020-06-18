import numpy as np
from scipy import linalg
def Phi(Q,xi):
    '''
    Lengths of springs
    :param Q:
    :param xi:
    :return:
    '''
    return np.sqrt(np.sum(np.square(np.matmul(Q.transpose(),xi)), axis=1))

def K(Q, xi):
    '''
    Normalized directions of springs
    :param Q:
    :param xi:
    :return:
    '''
    d=xi.shape[1]
    return np.divide(
        np.matmul(Q.transpose(),xi),
        np.tile(Phi(Q,xi),(d,1)).transpose()
    )

def DPhi(Q,xi):
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

    return np.multiply(N2,Q2)

def DPhiMat(Q,xi):
    '''
    :param Q:
    :param xi:
    :return: m x (nd) matrix representing D_xi phi of concatenated m x n slices
    '''
    DP=DPhi(Q,xi)
    d=DP.shape[2]
    result=DP[:,:,0]
    for k in range(1,d):
        result = np.concatenate((result,DP[:,:,k]), axis=1)
    return result


def kernelBasisDPhiMat(Q, xi):
    return linalg.null_space(DPhiMat(Q,xi))


