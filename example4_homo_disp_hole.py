import math

from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
from solver.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from solver.springs_view import SpringsView

n2 = 15
n1 = n2 + n2//2

def a_func(orig, termin):
    base_stiffess = 1
    return base_stiffess

def cplus_func(orig, termin):
    base_yeld_stress=0.001
    return base_yeld_stress

def cminus_func(orig, termin):
    return -cplus_func(orig, termin)

def is_node_func(coords):
    result = (coords[0] - coords[1]//2 > -1) and (2*coords[0] - coords[1] < 2*(n2-1))
    if (coords[0]>7 and coords[0]<11) and coords[1]==5:
        result =False

    if (coords[0]>7 and coords[0]<12) and coords[1]==6:
        result =False

    if (coords[0]>7 and coords[0]<13) and coords[1]==7:
        result =False

    if (coords[0]>8 and coords[0]<13) and coords[1]==8:
        result =False

    if (coords[0] > 9 and coords[0]<13) and coords[1]==9:
        result =False
    return result

def xi_func(coords):
    delta=0.5
    (i,j)=coords
    return (i * delta - j * delta / 2., j * delta * math.sqrt(3) / 2)

def add_springs_func(orig):
    (i,j) = orig

    termins = []
    if (i < n1-1) and (j>0) and (j<n2-1) and is_node_func((i+1,j)):
        termins.append((i+1,j))
    if (j < n2-1) and is_node_func((i,j+1)):
        termins.append((i,j+1))
    if (i < n1-1) and (j < n2-1) and is_node_func((i+1, j + 1)):
        termins.append((i+1, j + 1))
    return termins


def r_and_r_prime_component_functions(node_coords, component):
    """
    Set up displacement boundary condition at the node with grid coordinates node_coords
    :param node_coords: coordinates of the node in terms of the grid (integers)
    :param component: 0 is x, 1 is y; allows to turn on/off constraints on individual components
    :return: Function of time, representing the component of the displacement and its derivative at the node
    """
    func = None
    if node_coords[1] == 0:
        func = lambda t: (0,0)
    elif node_coords[1] == n2-1:
        if component == 0:
            func = lambda t: (0,0)
        else:
            rate = 0.1
            func = lambda t: (rate*t, rate)
    return func

def force(node_coords):
    """
    Set up force boundary condition at the node with grid coordinates node_coords
    :param node_coords: coordinates of the node in terms of the grid (integers)
    :return: Vector function (touple-valued), representing the component of the displacement velocity at the node
    """
    force = lambda t: (0,0)
    return force

grid = Grid(n1, n2, is_node_func, xi_func, add_springs_func, a_func, cminus_func,cplus_func, r_and_r_prime_component_functions, force)
process = Elastoplastic_process_linearized(grid.Q, grid.xi, grid.a, grid.cminus, grid.cplus, grid.d,
                                           grid.boundary_condition.R, grid.boundary_condition.r, grid.boundary_condition.f, grid.boundary_condition.r_prime)


t0 = 0
dt = 0.001
nsteps = 250

e0=grid.e0
xi0=grid.xi

#(T, E, Y, Sigma, Rho) = process.solve_e_catch_up(e0, t0, dt, nsteps) #solve the sweeping process in R^m (slower)
(T, E, Y_V, Sigma, Rho) = process.solve_e_in_V_catch_up(e0, t0, dt, nsteps) #solve the sweeping process in R^{dim V},(faster)
(T_leapfrog, E_leapfrog, E_V_leapfrog, Sigma_leapfrog, Rho_leapfrog) = process.solve_e_in_V_leapfrog(e0, t0, 1e-9)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi0, axis=1),(1,T.shape[0]))
#SpringsView(T,XI,E, process,((-1,8),(-1,8)),time_text_coords=(0.,-0.5),"example4_skewed_homo_disp.mp4",20) #to save the movie in a file
SpringsView(T,XI,E, process,((-1,8),(-1,8)),time_text_coords=(0.,-0.5))


XI_leapfrog=np.tile(np.expand_dims(xi0, axis=1),(1,T_leapfrog.shape[0]))
SpringsView(T_leapfrog,XI_leapfrog, E_leapfrog, process, ((-1,8),(-1,8)),time_text_coords=(0.,-0.5))
plt.show()