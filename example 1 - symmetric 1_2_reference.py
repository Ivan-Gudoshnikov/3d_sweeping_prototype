import numpy as np
import matplotlib.pyplot as plt
from solver.elastoplastic_process import ElastoplasticProcess
from solver.springs_view import SpringsView

Q = np.array([[ 1, 1, 0, 0, 0],
              [-1, 0, 1, 1, 0],
              [ 0,-1,-1, 0, 1],
              [ 0, 0, 0,-1,-1]])

xi0 = np.array([0., 0.,   -1., 1.,   1., 1.,   0, 2.])
t0 = 0
dt=0.0005

#tmax = 4
#nsteps= int((tmax-t0)//dt)
nsteps = 1000

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
cminus = np.array([-2,-2,-0.1,-2,-2])
cplus  = np.array([ 2, 2, 0.1, 2, 2])

#spatial dimension
d=2
q=4

#external forces at nodes
f = lambda t: np.array([0,0,  0,0,  0,0,  0,0])


example1 = ElastoplasticProcess(Q, a, cminus, cplus, d, q, rho, d_xi_rho, d_t_rho, f)

xi_ref = xi0
t_ref = 0
(T, XI, E, X, P, N, DOT_P_CONE_COORDS,P_ALT) = example1.solve_fixed_spaces(xi0,e0,t0, dt, nsteps, xi_ref, t_ref) #example1.solve(xi0, e0, t0, dt, nsteps)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

figP, axP = plt.subplots()
axP.plot(T, P_ALT.T)
axP.set(title="P_ALT")

SpringsView(T,XI,E, example1,((-3,3),(-1,7)))
#SpringsView(T,XI,E, example1,((-3,3),(-1,7)), "example 1 symmetric 1_2.mp4")

plt.show()




