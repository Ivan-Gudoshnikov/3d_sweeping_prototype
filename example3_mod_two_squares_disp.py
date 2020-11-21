import math

from solver.elastoplastic_process import ElastoplasticProcess
from solver.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from solver.springs_view import SpringsView

n1 = 3
n2 = 2


def a_func(orig, termin):
    base_stiffess = 1.
    if (orig[0] == termin[0]) or (orig[1] == termin[1]):
        return base_stiffess #non-diagonal springs have stiffness 2
    else:
        return base_stiffess/2 #diagonal springs have stiffness 1

def cplus_func(orig, termin):
    base_yeld_stress=0.001

    if (orig[0] == termin[0]) or (orig[1] == termin[1]):
        yeld_stress=base_yeld_stress #non-diagonal springs
    else:
        yeld_stress = base_yeld_stress/math.sqrt(2)  #diagonal sptings

    return yeld_stress

def cminus_func(orig, termin):
    return -cplus_func(orig, termin)

def is_node_func(coords):
    result = True
    return result


def xi_func(coords):
    delta=2
    (i,j)=coords
    return (i * delta, j * delta)


def add_springs_func(orig):
    (i,j) = orig

    termins = []
    if (i < n1-1) and is_node_func((i+1,j)):
        termins.append((i+1,j))
    if (j < n2-1) and is_node_func((i,j+1)):
        termins.append((i,j+1))
    if (i < n1-1) and (j < n2-1) and is_node_func((i+1, j + 1)):
        termins.append((i+1, j + 1))
    if (i < n1-1) and (j > 0) and is_node_func((i + 1, j - 1)):
        termins.append((i + 1, j - 1))
    return termins

def add_disp_boundary_cond_func(node_coords, component):
    """
    Set up displacement boundary condition at the node with grid coordinates node_coords
    :param node_coords: coordinates of the node in terms of the grid (integers)
    :param component:
    :return: Scalar function, representing the component of the displacement velocity at the node
    """
    v = None
    if node_coords[0] == 0 and node_coords[1] == 0: #both components for the node to be 0
        v = lambda t: 0

    if node_coords[0] == 2 and node_coords[1] == 1 and component == 0: # x-compontnt of the velocity of the node is 0.1, y-component is unrestricted
        v = lambda t: 0.2

    #if node_coords[0] == 2 and node_coords[1] == 1 and component == 1:
    #    v = lambda t: 0.1

    return v

def add_force_boundary_cond_func(node_coords):
    """
    Set up displacement boundary condition at the node with grid coordinates node_coords
    :param node_coords: coordinates of the node in terms of the grid (integers)
    :return: Vector function (touple-valued), representing the component of the displacement velocity at the node
    """
    force = lambda t: (0,0)
    return force


example3grid = Grid(n1, n2, is_node_func, xi_func, add_springs_func, a_func, cminus_func,cplus_func, add_disp_boundary_cond_func, add_force_boundary_cond_func)

example3 = example3grid.get_elastoplastic_process()

Q_mod=example3grid.Q
Q_mod=np.hstack((Q_mod, np.zeros((6,4))))
Q_mod=np.vstack((Q_mod, np.zeros((2,15))))
Q_mod[0,11]=1
Q_mod[6,11]=-1
Q_mod[1,12]=1
Q_mod[6,12]=-1
Q_mod[4,13]=1
Q_mod[7,13]=-1
Q_mod[5,14]=1
Q_mod[7,14]=-1

a_mod=example3grid.a
a_mod=np.hstack((a_mod, np.array([0.5,0.5,0.5,0.5])))
cplus_mod=example3grid.cplus
cplus_mod=np.hstack((cplus_mod, np.ones(4)*0.03/math.sqrt(2)))
cminus_mod=-cplus_mod

d=2
q_mod=4
d_xi_rho_mod_mat=np.zeros((q_mod, 8*2))
d_xi_rho_mod_mat[0,12]=1
d_xi_rho_mod_mat[1,13]=1
d_xi_rho_mod_mat[2,14]=1
d_xi_rho_mod_mat[3,15]=1
d_xi_rho_mod = lambda xi, t: d_xi_rho_mod_mat
d_t_rho_mod = lambda xi, t: np.array([0,0,-0.1,0])
f_mod = lambda t: np.zeros(16)


process_mod = ElastoplasticProcess(Q_mod, a_mod, cminus_mod, cplus_mod, d, q_mod, None, d_xi_rho_mod, d_t_rho_mod, f_mod)
t0 = 0
dt = 0.0001
nsteps = 2000


xi_mod =  example3grid.xi
xi_mod=np.hstack((xi_mod, np.array([-2.,1.,6.,1.])))

t_ref = 0
e0_mod= np.zeros(15)

(T, E) = process_mod.solve_fixed_spaces_e_only(xi_mod, e0_mod,t0, dt, nsteps, xi_mod, t_ref)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi_mod, axis=1),(1,T.shape[0]))

#SpringsView(T,XI,E, example3,((-3,7),(-1,8)),"example3_new_two_squares_disp.mp4",20)
SpringsView(T,XI,E, process_mod,((-3,7),(-1,8)))
plt.show()