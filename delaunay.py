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


def width_prime(t):
    return 0.1

def height_prime(t):
    return 0.0


loader = Yang_loader("DHU_NetworkData/stealth_0.3/config1/Iconfig.txt", "DHU_NetworkData/stealth_0.3/config1/delaunay_connectivity_matrix.txt",a_func, cminus_func, cplus_func, width_prime, height_prime)
process= loader.get_elastoplastic_process()
t0 = 0
xi_ref = loader.get_xi()
t_ref = 0

#(T_leapfrog, E_leapfrog) = process.leapfrog(loader.get_e0(),t0,xi_ref,t_ref)

SpringsViewStatic(t0, xi_ref, loader.get_e0(), process,  ((-0.1,1.1),(-0.1,1.1)))
plt.show()