import math
from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
import numpy as np
import matplotlib.pyplot as plt

from solver.sp_view_linearized import SweepingViewLinearized
from solver.springs_view import SpringsView

d=2

Q=np.array([[ 1,  0,  1,  0,  1,  0,  1,  0,  0,  0],
            [ 0,  1, -1,  0,  0,  1,  0,  1,  0,  0],
            [-1,  0,  0,  1,  0, -1,  0,  0,  1,  0],
            [ 0, -1,  0, -1, -1,  0,  0,  0,  0,  1],
            [ 0,  0,  0,  0,  0,  0, -1, -1,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0, -1, -1]])

xi0 = np.array([2., -1., 2., 1., 4., -1., 4., 1., 0., 0., 6., 0.])
n=Q.shape[0]
m=Q.shape[1]

a=np.ones(10)

cplus = 0.001* np.array([1., 1., 1., 1., 1./np.sqrt(2), 1./np.sqrt(2), 10., 10., 10., 10.])
cminus = -cplus

d=2
q=4



R = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])


r_prime_vect=np.array([0,0,-0.1,0])
r_prime = lambda t: r_prime_vect
r = lambda t: r_prime_vect*t
f = lambda t: np.zeros(12)


t0 = 0
dt = 0.0001
nsteps = 800

process = Elastoplastic_process_linearized(Q, xi0, a, cminus, cplus, d, R, r, f, r_prime)


t_ref = 0
e0= np.zeros(10)

#(T, E, Y, Sigma, Rho) = process.solve_e_catch_up(e0, t0, dt, nsteps) #solve the sweeping process in R^m (slower)
(T, E, Y_V, Sigma, Rho) = process.solve_e_in_V_catch_up(e0, t0, dt, nsteps) #solve the sweeping process in R^{dim V},(faster)
(T_leapfrog, E_leapfrog, E_V_leapfrog, Sigma_leapfrog, Rho_leapfrog) = process.solve_e_in_V_leapfrog(e0, t0, 1e-12)

figSigma, axSigma = plt.subplots()
axSigma.plot(T, Sigma.T)
axSigma.set(title="Sigma")

figRho, axRho = plt.subplots()
axRho.plot(T, Rho.T)
axRho.set(title="Rho")


XI = np.tile(np.expand_dims(xi0, axis=1),(1,T.shape[0]))

#SpringsView(T,XI,E, process,((-1,7),(-2,2)),time_text_coords=(-0.5,-1.5),"example3_new_two_squares_disp.mp4",20) #to save the movie in a file
SpringsView(T,XI,E, process,((-1,7),(-2,2)),time_text_coords=(-0.5,-1.5))

SweepingViewLinearized(T, E, E_leapfrog, process, ((-0.008, 0.008), (-0.008, 0.008)))

plt.show()