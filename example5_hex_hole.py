import math
from solver.grid import Grid
import matplotlib.pyplot as plt
from solver.springs_view import SpringsView
import numpy as np

n2 = 5
n1 = 5

def a_func(orig, termin):
    base_stiffess = 1
    return base_stiffess

def cplus_func(orig, termin):
    base_yeld_stress=0.001
    return base_yeld_stress

def cminus_func(orig, termin):
    return -cplus_func(orig, termin)

def is_node_func(coords):
    result = True
    if coords[0] == 2 and coords[1] == 2:
        result = False

    if abs(coords[0]-2) > 2 or abs(coords[1]-2) > 2:
        result = False

    if coords[0]+2 < coords[1] or coords[0]-2 > coords[1]:
        result = False
    return result

def xi_func(coords):
    delta=1
    (i,j)=coords
    return (i * delta - j * delta / 2., j * delta * math.sqrt(3) / 2)

def add_springs_func(orig):
    (i,j) = orig

    termins = []
    if (i < n1-1) and is_node_func((i+1,j)):
        termins.append((i+1,j))
    if (j < n2-1) and is_node_func((i,j+1)):
        termins.append((i,j+1))
    if (i < n1-1) and (j < n2-1) and is_node_func((i+1, j + 1)):
        termins.append((i+1, j + 1))
    return termins

def add_boundary_cond_func(coords):
    velocity = None
    if coords[0] == 0 and coords[1] == 0:
        velocity = lambda t: (0,0)

    if coords[0] == 4 and coords[1] == 4:
        velocity = lambda t: (0.1, 0.1)

    force = lambda t: (0, 0)

    return velocity, force



example3grid = Grid(n1, n2, is_node_func, xi_func, add_springs_func, a_func, cminus_func,cplus_func, add_boundary_cond_func)

example3 = example3grid.get_elastoplastic_process()


t0 = 0
dt = 0.0001
nsteps = 800

xi_ref = example3grid.xi
t_ref = 0
(T, E) = example3.solve_fixed_spaces_e_only(example3grid.xi, example3grid.e0,t0, dt, nsteps, xi_ref, t_ref)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi_ref, axis=1),(1,T.shape[0]))

SpringsView(T,XI,E, example3,((-3,7),(-1,8)),"example5_hex_hole.mp4",20)
#SpringsView(T,XI,E, example3,((-3,7),(-1,8)))
plt.show()