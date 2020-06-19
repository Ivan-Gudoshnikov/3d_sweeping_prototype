import numpy as np
import scipy


def tensor_to_matrix(t):
    """
    :param t: m x n x d numpy array
    :return: m x (nd) numpy array: ( t[m x n x (1)]  t[m x n x (2)] ... t[m x n x (d)])
    """
    (m, n, d) = t.shape
    result = np.zeros((m, n * d))
    for k in range(0, d):
        result[:, range(n * k, n * (k + 1))] = t[:, :, k]
    return result


def matrix_to_tensor(matrix, d):
    """
    :param matrix: m x (nd) numpy array: ((m x n x 1) (m x n x 2) ... (m x n x d)
    :param d: 3nd dimension size
    :return: m x n x d numpy array
    """
    m = matrix.shape[0]
    n = matrix.shape[1] // d
    result = np.zeros((m, n, d))
    for k in range(0, d):
        result[:, :, k] = matrix[:, range(n * k, n * (k + 1))]
    return result


class ElastoplasticProcess:
    def __init__(self, Q, a, cminus, cplus, d, q, rho, d_xi_rho, d_t_rho):
        # basic properties

        # incidence matrix
        self.Q = Q

        # amount of springs
        self.m = Q.shape[1]
        # amount of nodes
        self.n = Q.shape[0]
        # spatial dimension
        self.d = d
        #rnak of the additional constraint
        self.q = q

        self.A = np.diag(a)
        self.cminus = cminus
        self.cplus = cplus

        # function rho and its derivatives| rho: xi, t -> rho(xi,t)
        self.rho = rho
        self.d_xi_rho = d_xi_rho
        self.d_t_rho = d_t_rho

    def phi(self, xi):
        """
        Lengths of springs
        :param xi:
        :return:
        """
        return np.sqrt(np.sum(np.square(np.matmul(self.Q.T, xi)), axis=1))

    def K(self, xi):
        """
        Normalized directions of springs
        :param xi:
        :return:
        """
        return np.divide(
            np.matmul(self.Q.T, xi),
            np.tile(self.phi(xi), (self.d, 1)).T
        )

    def d_xi_phi(self, xi):
        """
        :param xi:
        :return: m x n x d array representing D_xi phi
        """
        Q1 = np.expand_dims(self.Q.T, axis=2)
        Q2 = np.tile(Q1, (1, 1, self.d))

        N1 = np.swapaxes(np.expand_dims(self.K(xi), axis=2), 1, 2)
        N2 = np.tile(N1, (1, self.n, 1))

        return np.multiply(N2, Q2)

    def ker_d_xi_rho(self, xi, t):
        """
        basis in the nullspace of d_xi_rho
        :param xi:
        :param t:
        :return: (dim Ker d_xi_rho) x n x d array
        """
        return matrix_to_tensor(scipy.linalg.null_space(tensor_to_matrix(self.d_xi_rho(xi, t))).T, self.d)

    def ker_d_xi_phi(self, xi):
        """
        basis in the nullspace of d_xi_phi
        :param xi:
        :return: (dim Ker d_xi_phi) x n x d array
        """
        return matrix_to_tensor(scipy.linalg.null_space(tensor_to_matrix(self.d_xi_phi(xi))).T, self.d)

    def dim_intersection_nullspaces(self, xi):
        """
        should always be 0!
        :param xi:
        :return:
        """
        return self.n*self.d-np.linalg.matrix_rank(
            np.vstack((tensor_to_matrix(self.d_xi_rho(xi, 0)), tensor_to_matrix(self.d_xi_phi(xi)))))
    def u_basis(self,xi,t):
        result = np.matmul(tensor_to_matrix(self.d_xi_phi(xi)), tensor_to_matrix(self.ker_d_xi_rho(xi, t)).T)
        if np.linalg.matrix_rank(result)!= self.n*self.d - self.q:
            raise NameError("Constraint rho is not enough for that phi(xi)")
        return result

    def R(self,xi,t):
        return np.linalg.pinv(tensor_to_matrix(self.d_xi_rho(xi,t)))



