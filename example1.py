import numpy as np
from elastoplastic_process import ElastoplasticProcess
import phi
import scipy


Q = np.array([[ 1, 1, 0, 0, 0],
              [-1, 0, 1, 1, 0],
              [ 0,-1,-1, 0, 1],
              [ 0, 0, 0,-1,-1]])

xi0 = np.array([[ 0., 0.],
                [-1., 1.],
                [ 1., 1.],
                [ 2., 0.]])

rho = lambda xi, t: np.array([xi[0,0],
                              xi[1,2],
                              xi[4,1],
                              xi[4,2]-t-2])

d_xi_rho = lambda xi, t: np.array([[[1,0],
                                    [0,0],
                                    [0,0],
                                    [0,0]],
                                   [[0,1],
                                    [0,0],
                                    [0,0],
                                    [0,0]],
                                   [[0,0],
                                    [0,0],
                                    [0,0],
                                    [1,0]],
                                   [[0,0],
                                    [0,0],
                                    [0,0],
                                    [0,1]]])

d_t_rho = lambda xi, t: np.array([0,0,0,-1])

a = np.array([1., 1., 1., 1., 1.])
cminus = np.array([-1.,-1.1,-1.15,-1.25,-1.05])
cplus =  np.array([ 1., 1.1, 1.15, 1.25, 1.05])

#spatial dimension
d=2

example1 = ElastoplasticProcess(Q, a, cminus, cplus, d, rho, d_xi_rho, d_t_rho)
print(phi.tensor_to_matrix(example1.d_xi_phi(xi0)))
print(phi.tensor_to_matrix(example1.d_xi_phi(xi0)).shape)
print(np.linalg.matrix_rank(phi.tensor_to_matrix(example1.d_xi_phi(xi0))))
print("d_xi_rho_matrix=")
print(phi.tensor_to_matrix(example1.d_xi_rho(xi0,0)))
print("d_xi_rho_matrix_nullspace=")
print(scipy.linalg.null_space(phi.tensor_to_matrix(example1.d_xi_rho(xi0,0))))
print(example1.ker_d_xi_rho(xi0,0))