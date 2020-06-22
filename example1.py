import numpy as np
import scipy.linalg
from elastoplastic_process import ElastoplasticProcess, tensor_to_matrix,matrix_to_tensor
from quadprog_interface import McGibbonQuadprog


Q = np.array([[ 1, 1, 0, 0, 0],
              [-1, 0, 1, 1, 0],
              [ 0,-1,-1, 0, 1],
              [ 0, 0, 0,-1,-1]])

xi0 = np.array([0., 0.,   -1., 1.,   1., 1.,   0, 2.])
t0 = 0
dt=0.05

tmax = 4
nsteps= int((tmax-t0)//dt)

e0 = np.array([0., 0., 0., 0., 0.])

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
cplus  = np.array([ 1., 1.1, 1.15, 1.25, 1.05])

#spatial dimension
d=2
q=4


#external forces at nodes
f = lambda t: np.array([0,0,  0,0,  0,0,  0,0])



example1 = ElastoplasticProcess(Q, a, cminus, cplus, d,q, rho, d_xi_rho, d_t_rho, f)
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
print(example1.d_xi_rho(xi0,t0))
print("d_xi_rho_matrix_nullspace=")
print(example1.ker_d_xi_rho(xi0,t0))

print("d_xi_phi_matrix_nullspace=")
print(example1.ker_d_xi_phi(xi0))

print("dimension of the intersection:")
print(example1.dim_intersection_nullspaces(xi0))

print("H=")
print(example1.H(xi0,t0))


print("basis in U:")
#print(example1.d_xi_phi(xi0))
#print(example1.ker_d_xi_rho(xi0,0))

print(example1.u_basis(xi0,0))
print(np.linalg.matrix_rank(example1.u_basis(xi0,0)))

print("basis in V:")
print(example1.v_basis(xi0,0))

print("R=")
print(example1.R(xi0,t0))

print("d_xi_rho * R = ")
print(np.matmul(example1.d_xi_rho(xi0,t0), example1.R(xi0,t0)))

print("p_u_coords=")
print(example1.p_u_coords(xi0,t0))

print("p_v_coords=")
print(example1.p_v_coords(xi0,t0))

print("ubasis*p_u_coord + vbasis*p_v_coords - I")
print(np.matmul(example1.u_basis(xi0,t0),example1.p_u_coords(xi0,t0))+np.matmul(example1.v_basis(xi0,t0),example1.p_v_coords(xi0,t0))-np.identity(example1.m))

print("g_u_coords=")
print(example1.g_v_coords(xi0,t0))

print("g=")
print(np.matmul(example1.v_basis(xi0,t0),example1.g_v_coords(xi0,t0)))

print("h_u_coords=")
print(example1.h_u_coords(xi0, t0, f(t0)))

print("h=")
print(np.matmul(example1.u_basis(xi0, t0), example1.h_u_coords(xi0,t0, f(t0))))

print("v_basis and v_orth:")
print(example1.v_orth(xi0, t0).dot(example1.v_basis(xi0, t0)))
print(example1.v_basis(xi0, t0).T.dot(example1.v_orth(xi0, t0).T))

moving_set1 = example1.moving_set(xi0, t0)
print("Moving set is ", moving_set1)
print(e0 in moving_set1)

(xi1,e1)=example1.solve_system_step(xi0,e0,t0,dt)
print("Step1:")
print("xi =")
print(xi1)
print("e1=")
print(e1)

(T,XI,E) = example1.solve(xi0,e0,t0, dt, nsteps)


