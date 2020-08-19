from triangular_grid import TriangularGrid
import numpy as np
import matplotlib.pyplot as plt
from solver.springs_view import SpringsView

n1 = 12
n2 = 10

def a_func(orig, termin):
    base_stiffess = 1
    return base_stiffess

def cplus_func(orig, termin):
    base_yeld_stress=0.001
    return base_yeld_stress

def cminus_func(orig, termin):
    return -cplus_func(orig, termin)

def is_node_func(coords):
    return True

def add_springs_func(orig):
    (i,j) = orig

    termins = []
    if i < n1-1:
        termins.append((i+1,j))
    if j < n2-1:
        termins.append((i,j+1))
    if (i < n1-1) and (j < n2-1):
        termins.append((i+1, j + 1))
    return termins

def add_boundary_cond_func(coords):
    velocity = None
    if coords[0] == 0:
        velocity = lambda t: (0,0)

    if coords[0] == n1-1:
        velocity = lambda t: (0.1, 0)

    force = lambda t: (0, 0)

    return velocity, force



example3grid = TriangularGrid(n1, n2, 0.5, is_node_func, add_springs_func, a_func, cminus_func,cplus_func, add_boundary_cond_func)

example3 = example3grid.get_elastoplastic_process()


t0 = 0
dt = 0.0001
nsteps = 2500

xi_ref = example3grid.xi
t_ref = 0
(T, E) = example3.solve_fixed_spaces_e_only(example3grid.xi, example3grid.e0,t0, dt, nsteps, xi_ref, t_ref)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi_ref, axis=1),(1,T.shape[0]))

SpringsView(T,XI,E, example3,((-3,7),(-1,8)),"example4_skewed_homo_disp.mp4",20)
plt.show()