import numpy as np
import scipy.linalg
from elastoplastic_process import ElastoplasticProcess, tensor_to_matrix,matrix_to_tensor


Q = np.array([[ 1, 1, 0, 0, 0],
              [-1, 0, 1, 1, 0],
              [ 0,-1,-1, 0, 1],
              [ 0, 0, 0,-1,-1]])

xi0 = np.array([0., 0.,   -1., 1.,   1., 1.,   0, 2.])
t0 = 0

rho = lambda xi, t: np.array([xi[0],
                              xi[1],
                              xi[6],
                              xi[7]-t-2])

d_xi_rho = lambda xi, t: np.array([[1,0,  0,0,  0,0,  0,0],
                                   [0,1,  0,0,  0,0,  0,0],
                                   [0,0,  0,0,  0,0,  1,0],
                                   [0,0,  0,0,  0,0,  0,1]])

d_t_rho = lambda xi, t: np.array([0,0,0,-1])

a = np.array([1., 1., 1., 1., 1.])
cminus = np.array([-1.,-1.1,-1.15,-1.25,-1.05])
cplus =  np.array([ 1., 1.1, 1.15, 1.25, 1.05])

#spatial dimension
d=2
q=4

example1 = ElastoplasticProcess(Q, a, cminus, cplus, d,q, rho, d_xi_rho, d_t_rho)
print("phi(xi0)=")
print(example1.phi(xi0))

print("K(xi0)=")
print(example1.K(xi0))

print("d_xi_phi=")
print(example1.d_xi_phi(xi0))
print("d_xi_phi.shape")
print(example1.d_xi_phi(xi0).shape)
print(np.linalg.matrix_rank(example1.d_xi_phi(xi0)))
print("d_xi_rho_matrix=")
print(example1.d_xi_rho(xi0,0))
print("d_xi_rho_matrix_nullspace=")
print(example1.ker_d_xi_rho(xi0,0))

print("d_xi_phi_matrix_nullspace=")
print(example1.ker_d_xi_phi(xi0))

print("dimension of the intersection:")
print(example1.dim_intersection_nullspaces(xi0))

print("basis in U:")
#print(example1.d_xi_phi(xi0))
#print(example1.ker_d_xi_rho(xi0,0))

print(example1.u_basis(xi0,0))
print(np.linalg.matrix_rank(example1.u_basis(xi0,0)))

print("R=")
print(example1.R(xi0,t0))

print("d_xi_rho * R = ")
print(np.matmul(example1.d_xi_rho(xi0,t0), example1.R(xi0,t0)))

