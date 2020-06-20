import numpy as np
import scipy

def matrix_to_vector(matrix):
    """
    :param matrix: n x d matrix
    :return: vector ( [d x (1)]  [d x 2] ... [d x (n)])
    """
    (n, d) = matrix.shape
    result = np.zeros(n*d)
    for j in range(0,n):
        result[range(d*j, d*(j+1))] = matrix[j, :]
    return result

def vector_to_matrix(vector, d):
    """    
    :param vector: vector ( [d x (1)]  [d x 2] ... [d x (n)])
    :param d: 
    :return: matrix: n x d matrix
    """
    n = vector.shape[0] // d
    result = np.zeros((n, d))
    for j in range(0,n):
        result[j, :] = vector[range(d*j, d*(j+1))]
    return result

def tensor_to_matrix(t):
    """
    :param t: m x n x d numpy array
    :return: m x (nd) numpy array: ( t[m x (1) x d]  t[m x (2) x d] ... t[m x (n) x d])
    """
    (m, n, d) = t.shape
    result = np.zeros((m, n * d))
    for j in range(0, n):
        result[:, range(d * j, d * (j + 1))] = t[:, j, :]
    return result


def matrix_to_tensor(matrix, d):
    """
    :param matrix: m x (nd) numpy array: m x (nd) numpy array: ( t[m x (1) x d]  t[m x (2) x d] ... t[m x (n) x d])
    :param d: 3nd dimension size
    :return: m x n x d numpy array
    """
    m = matrix.shape[0]
    n = matrix.shape[1] // d
    result = np.zeros((m, n, d))
    for j in range(0, n):
        result[:, j, :] = matrix[:, range(d * j, d * (j + 1))]
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
        xi_mat=vector_to_matrix(xi,self.d)
        return np.sqrt(np.sum(np.square(np.matmul(self.Q.T, xi_mat)), axis=1))

    def K(self, xi):
        """
        Normalized directions of springs
        :param xi as a vector
        :return: m x d matrix
        """
        xi_mat = vector_to_matrix(xi, self.d)
        return np.divide(
            np.matmul(self.Q.T, xi_mat),
            np.tile(self.phi(xi), (self.d, 1)).T
        )

    def d_xi_phi(self, xi):
        """
        :param xi:
        :return: m x (nd) array representing D_xi phi
        """

        Q1 = np.expand_dims(self.Q.T, axis=2)
        Q2 = np.tile(Q1, (1, 1, self.d))

        N1 = np.swapaxes(np.expand_dims(self.K(xi), axis=2), 1, 2)
        N2 = np.tile(N1, (1, self.n, 1))

        return tensor_to_matrix(np.multiply(N2, Q2))

    def ker_d_xi_rho(self, xi, t):
        """
        basis in the nullspace of d_xi_rho
        :param xi:
        :param t:
        :return:  (nd)x(dim Ker d_xi_rho) array
        """
        return scipy.linalg.null_space(self.d_xi_rho(xi, t))

    def ker_d_xi_phi(self, xi):
        """
        basis in the nullspace of d_xi_phi
        :param xi:
        :return: (nd)x(dim Ker d_xi_phi) array
        """
        return scipy.linalg.null_space(self.d_xi_phi(xi))

    def dim_intersection_nullspaces(self, xi):
        """
        should always be 0!
        :param xi:
        :return:
        """
        return self.n*self.d- np.linalg.matrix_rank(np.vstack((self.d_xi_rho(xi, 0), self.d_xi_phi(xi))))

    def u_basis(self,xi,t):
        result = np.matmul(self.d_xi_phi(xi), self.ker_d_xi_rho(xi, t))
        if np.linalg.matrix_rank(result)!= self.n*self.d - self.q:
            raise NameError("Constraint rho is not enough for that phi(xi)")
        return result

    def R(self,xi,t):
        return np.linalg.pinv(self.d_xi_rho(xi,t))



