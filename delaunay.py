import math

from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
from solver.springs_view import SpringsView
from solver.yang_loader import Yang_loader
import numpy as np
import matplotlib.pyplot as plt
from solver.springs_view_static import SpringsViewStatic

eps=1e-10

def a_func(orig_coords, termin_coords):
    base_stiffess = 1
    return base_stiffess

def cplus_func(orig_coords, termin_coords):
    base_yeld_stress=0.001
    return base_yeld_stress

def cminus_func(orig_coords, termin_coords):
    return -cplus_func(orig_coords, termin_coords)


def width_prime(t):
    return 0.1

def height_prime(t):
    return 0.0

def width_change(t):
    return 0.1*t

def height_change(t):
    return 0.0



loader = Yang_loader("DHU_NetworkData/stealth_0.3/config1/Iconfig.txt", "DHU_NetworkData/stealth_0.3/config1/delaunay_connectivity_matrix.txt",a_func, cminus_func, cplus_func,width_change, width_prime,height_change, height_prime)
t0 = 0
dt=0.001
nsteps=400

xi_ref = loader.get_xi()
e0 = loader.get_e0()
t_ref = 0

process = Elastoplastic_process_linearized(loader.Q, xi_ref, loader.a, loader.cminus, loader.cplus, loader.d, loader.d_xi_rho_mat, loader.r, loader.f, loader.d_t_r(xi_ref,t0))


(T, E, Y_V, Sigma, Rho) = process.solve_e_in_V_catch_up(e0, t0, dt, nsteps)
(T_leapfrog, E_leapfrog, E_V_leapfrog, Sigma_leapfrog, Rho_leapfrog) = process.solve_e_in_V_leapfrog(e0, t0, 1e-9)

XI= np.tile(np.expand_dims(xi_ref, axis=1),(1,T.shape[0]))
#SpringsViewStatic(t0, xi_ref, loader.get_e0(), process,  ((-0.1,1.1),(-0.1,1.1)))
#SpringsView(T_lite,XI_leapfrog ,E_lite, process, ((-0.1,1.1),(-0.1,1.1)),"delaunay_0_3_config1.mp4",5)
SpringsView(T,XI, E, process, ((-0.1,1.1),(-0.1,1.1)),time_text_coords=(-0.09,-0.09))

XI_leapfrog=np.tile(np.expand_dims(xi_ref, axis=1),(1,T_leapfrog.shape[0]))
SpringsView(T_leapfrog,XI_leapfrog, E_leapfrog, process, ((-0.1,1.1),(-0.1,1.1)),time_text_coords=(-0.09,-0.09))


figSigma, axSigma = plt.subplots()
axSigma.plot(T, Sigma.T)
axSigma.set(title="Sigma")

summed_rho_horizontal_links_x = np.zeros_like(T)
summed_rho_horizontal_links_y = np.zeros_like(T)
summed_rho_vertical_links_x = np.zeros_like(T)
summed_rho_vertical_links_y = np.zeros_like(T)


l_h=len(loader.boundary_cond.duplicate_horizontal_nodes)
l_v=len(loader.boundary_cond.duplicate_vertical_nodes)
for i in range(T.shape[0]):
    summed_rho_horizontal_links_x[i] = np.sum(Rho[[2+2*j for j in range(l_h)], i])
    summed_rho_horizontal_links_y[i] = np.sum(Rho[[3 + 2 * j for j in range(l_h)], i])
    summed_rho_vertical_links_x[i] = np.sum(Rho[[2 + 2*l_h + 2 * j for j in range(l_v)], i])
    summed_rho_vertical_links_y[i] = np.sum(Rho[[3 + 2*l_h + 2 * j for j in range(l_v)], i])

figRho_h_x, axRho_h_x = plt.subplots()
axRho_h_x.set(title="Rho: horizontal links, sum of x components")

figRho_h_y, axRho_h_y = plt.subplots()
axRho_h_y.set(title="Rho: horizontal links, sum of y components")

figRho_v_x, axRho_v_x = plt.subplots()
axRho_v_x.set(title="Rho: vertical links, sum of x components")

figRho_v_y, axRho_v_y = plt.subplots()
axRho_v_y.set(title="Rho: vertical links, sum of y components")

axRho_h_x.plot(T, summed_rho_horizontal_links_x)
axRho_h_y.plot(T, summed_rho_horizontal_links_y)
axRho_v_x.plot(T, summed_rho_vertical_links_x)
axRho_v_y.plot(T, summed_rho_vertical_links_y)
plt.show()