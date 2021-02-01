import math

from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
from solver.plot_boundary_connections_reactions import plot_boundary_connections_reaction
from solver.plot_boundary_springs_stresses import plot_boundary_springs_stresses
from solver.springs_view import SpringsView
from solver.yang_loader import Yang_loader
import numpy as np
import matplotlib.pyplot as plt
from solver.springs_view_static import SpringsViewStatic
from matplotlib.lines import Line2D

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

lim = ((-0.1,1.1),(-0.1,1.1))
springs_view1 = SpringsViewStatic(t0, process.xi0, e0, process, lim)

for n1, n2 in loader.duplicate_horizontal_nodes.items():
    n1_x = xi0[2 * n1]
    n1_y = xi0[2 * n1 + 1]
    n2_x = xi0[2 * n2]
    n2_y = xi0[2 * n2 + 1]

    springs_view1.ax.add_line(Line2D([n1_x, n2_x],[n1_y, n2_y], marker=None, color="k", linewidth=2))

springs_view2 = SpringsViewStatic(t0, process.xi0, e0, process, lim)

for n1, n2 in loader.duplicate_vertical_nodes.items():
    n1_x = xi0[2 * n1]
    n1_y = xi0[2 * n1 + 1]
    n2_x = xi0[2 * n2]
    n2_y = xi0[2 * n2 + 1]

    springs_view2.ax.add_line(Line2D([n1_x, n2_x],[n1_y, n2_y], marker=None, color="k", linewidth=2))

plt.show()


