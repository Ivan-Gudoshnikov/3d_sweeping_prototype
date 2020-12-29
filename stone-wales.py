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
def width_cnahge(t):
    return 0.1*t

def height_prime(t):
    return 0.0

def height_cnahge(t):
    return 0.0


loader = Yang_loader("configs_600/vertex_relax_0.12.txt", "configs_600/connectivity_matrix_0.12.txt",a_func, cminus_func, cplus_func,width_cnahge, width_prime,height_cnahge, height_prime)
process= loader.get_elastoplastic_process()
process_lite = Elastoplastic_process_linearized(loader.Q, loader.a, loader.cminus, loader.cplus, loader.d, loader.d_xi_rho_mat, loader.r, loader.f, demand_enough_constraints=False)

t0 = 0
dt=0.001
nsteps=200

xi_ref = loader.get_xi()
t_ref = 0



(T_lite,E_lite,Y_lite)= process_lite.solve_e_catch_up(loader.get_e0(), t0, dt, nsteps, xi_ref)
XI_rep = np.tile(np.expand_dims(xi_ref, axis=1),(1,T_lite.shape[0]))
#(T_leapfrog, E_leapfrog) = process.leapfrog(loader.get_e0(),t0,xi_ref,t_ref)

#SpringsViewStatic(t0, xi_ref, loader.get_e0(), process,  ((-2,32),(-2,32)))


SpringsView(T_lite,XI_rep ,E_lite, process, ((-2,32),(-2,32)))

plt.show()