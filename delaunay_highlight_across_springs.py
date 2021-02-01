import math

from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
from solver.plot_boundary_connections_reactions import plot_boundary_connections_reaction
from solver.plot_boundary_springs_stresses import plot_boundary_springs_stresses
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
dt=0.0005
nsteps=800

xi0 = loader.xi
e0 = loader.e0
t_ref = 0

process = Elastoplastic_process_linearized(loader.Q, xi0, loader.a, loader.cminus, loader.cplus, loader.d,
                                           loader.boundary_condition.R, loader.boundary_condition.r, loader.boundary_condition.f, loader.boundary_condition.r_prime)
e0mod1 = e0.copy()
e0mod2 = e0.copy()
for i in loader.connections_across_X:
    e0mod1[i] = process.K[i, i] * process.cplus[i]


for i in loader.connections_across_Y:
    e0mod2[i] = process.K[i, i] * process.cplus[i]

lim=((-0.1,1.1),(-0.1,1.1))
SpringsViewStatic(t0, process.xi0, e0mod1, process, lim, highlight=None)
SpringsViewStatic(t0, process.xi0, e0mod2, process, lim, highlight=None)
plt.show()