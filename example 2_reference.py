import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from elastoplastic_process import ElastoplasticProcess
from springs_view import SpringsView

Q = np.array([[ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [-1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
              [ 0,-1,-1, 0, 0, 1, 1, 0, 0, 0],
              [ 0, 0, 0,-1, 0,-1, 0, 1, 1, 0],
              [ 0, 0, 0, 0,-1, 0,-1,-1, 0, 1],
              [ 0, 0, 0, 0, 0, 0, 0, 0,-1,-1]])

xi0 = np.array([0., 0.,   -1., 1.,   1., 1.,   -1., 2.,   1., 2.,   0., 3.])
t0 = 0
dt=0.0002

#tmax = 4
#nsteps= int((tmax-t0)//dt)
nsteps = 6000

e0 = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

rho = lambda xi, t: np.array([xi[0],
                              xi[1],
                              xi[10],
                              xi[11]-t-3])

d_xi_rho = lambda xi, t: np.array([[1,0,  0,0,  0,0,  0,0,  0,0,  0,0],
                                   [0,1,  0,0,  0,0,  0,0,  0,0,  0,0],
                                   [0,0,  0,0,  0,0,  0,0,  0,0,  1,0],
                                   [0,0,  0,0,  0,0,  0,0,  0,0,  0,1]])

d_t_rho = lambda xi, t: np.array([0,0,0,-1])

a = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
cminus = np.array([-1, -1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -1, -1])
cplus  = np.array([ 1,  1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  1,  1])

#spatial dimension
d=2
q=4

#external forces at nodes
f = lambda t: np.array([0,0,  0,0,  0,0,  0,0,  0,0,  0,0])

example1 = ElastoplasticProcess(Q, a, cminus, cplus, d, q, rho, d_xi_rho, d_t_rho, f)

xi_ref = xi0
t_ref = 0
(T, E) = example1.solve_fixed_spaces_e_only(xi0, e0,t0, dt, nsteps, xi_ref, t_ref)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi_ref, axis=1),(1,T.shape[0]))

SpringsView(T,XI,E, example1,((-3,3),(-1,8)))

plt.show()




