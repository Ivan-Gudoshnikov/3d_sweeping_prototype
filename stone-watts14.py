import math
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


def add_boundary_cond_func(coords, loader :Yang_loader):
    velocity = None
    if np.abs(coords[1]- loader.get_min_y_in_xi())< eps:
        velocity = lambda t: (0,0)

    if np.abs(coords[1]- loader.get_max_y_in_xi())< eps:
        velocity = lambda t: (0, 0.1)

    force = lambda t: (0, 0)

    return velocity, force


loader = Yang_loader("configs_600/vertex_relax_0.12.txt", "configs_600/connectivity_matrix_0.12.txt",a_func, cminus_func, cplus_func, add_boundary_cond_func)
process= loader.get_elastoplastic_process()
t0 = 0
xi_ref = loader.get_xi()
t_ref = 0

#(T_leapfrog, E_leapfrog) = process.leapfrog(loader.get_e0(),t0,xi_ref,t_ref)

SpringsViewStatic(t0, xi_ref, loader.get_e0(), process,  ((-2,32),(-2,32)))
plt.show()