import math

from solver.elastoplastic_process import ElastoplasticProcess
from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
from solver.grid import Grid
import numpy as np
import matplotlib.pyplot as plt

from solver.sp_view import SweepingView
from solver.springs_view import SpringsView

n1 = 2
n2 = 2


def a_func(orig, termin):
    base_stiffess = 1.
    if (orig[0] == termin[0]) or (orig[1] == termin[1]):
        return base_stiffess #non-diagonal springs have stiffness 1
    else:
        return base_stiffess/2 #diagonal springs have stiffness 0.5

def cplus_func(orig, termin):
    base_yeld_stress=0.001

    if (orig[0] == termin[0]) or (orig[1] == termin[1]):
        yeld_stress=base_yeld_stress #non-diagonal springs
    else:
        yeld_stress = base_yeld_stress/math.sqrt(2)  #diagonal springs
        #yeld_stress = base_yeld_stress/2.5

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

#geberating a sinle square with diagonal springs
examplegrid = Grid(n1, n2, is_node_func, xi_func, add_springs_func, a_func, cminus_func,cplus_func, add_disp_boundary_cond_func, add_force_boundary_cond_func)
example = examplegrid.get_elastoplastic_process()

#adding springs on the sides
Q_mod=examplegrid.Q
Q_mod=np.vstack((Q_mod, np.zeros((2,6))))
Q_mod=np.hstack((Q_mod, np.array([[1., 0., 0., 0.],
                                  [0., 1., 0., 0.],
                                  [0., 0., 1., 0.],
                                  [0., 0., 0., 1.],
                                  [-1., -1., 0., 0.],
                                  [0., 0., -1., -1.]])))

#a_mod=examplegrid.a
#a_mod=np.hstack((a_mod, np.array([0.5,0.5,0.5,0.5])))
#a_mod=np.hstack((a_mod, np.array([1,1,1,1])))
a_mod=np.ones(10)

cplus_mod=examplegrid.cplus
cplus_mod=np.hstack((cplus_mod, np.ones(4)*0.01))
cminus_mod=-cplus_mod

d=2
q_mod=4
d_xi_rho_mod_mat=np.zeros((q_mod, 6*2))
d_xi_rho_mod_mat[0,8]=1
d_xi_rho_mod_mat[1,9]=1
d_xi_rho_mod_mat[2,10]=1
d_xi_rho_mod_mat[3,11]=1
d_xi_rho_mod = lambda xi, t: d_xi_rho_mod_mat
d_t_rho_mod = lambda xi, t: np.array([0,0,-0.1,0])
f_mod = lambda t: np.zeros(12)

r_lite = lambda t: np.array([0,0,-0.1*t,0])


t0 = 0
dt = 0.0001
nsteps = 800


xi_mod =  examplegrid.xi
xi_mod=np.hstack((xi_mod, np.array([-2.,1.,4.,1.])))

process_mod = ElastoplasticProcess(Q_mod, a_mod, cminus_mod, cplus_mod, d, q_mod, None, d_xi_rho_mod, d_t_rho_mod, f_mod)
process_mod_lite = Elastoplastic_process_linearized(Q_mod, xi_mod, a_mod, cminus_mod, cplus_mod, d, d_xi_rho_mod_mat, r_lite, f_mod)

t_ref = 0
e0_mod=process_mod_lite.vbasis @ np.array([-0.0015,0.0008])

(T, E) = process_mod.solve_fixed_spaces_e_only(xi_mod, e0_mod,t0, dt, nsteps, xi_mod, t_ref)
(T_leapfrog, E_leapfrog) = process_mod.leapfrog(e0_mod,t0,xi_mod,t_ref)
(T_lite,E_lite,Y_lite,Sigma_lite, Rho_lite)= process_mod_lite.solve_e_catch_up(e0_mod, t0, dt, nsteps)



figY, axY = plt.subplots()
figSigma, axSigma = plt.subplots()
axSigma.plot(T_lite, Sigma_lite.T)
axSigma.set(title="Sigma")

axY.plot(T_lite, Y_lite.T)
axY.set(title="Y")


XI = np.tile(np.expand_dims(xi_mod, axis=1),(1,T.shape[0]))

#SpringsView(T,XI,E, example3,((-3,7),(-1,8)),"example3_new_two_squares_disp.mp4",20)
SpringsView(T,XI,E_lite, process_mod,((-3,7),(-1,8)))

SweepingView(T, XI, E_lite, E_leapfrog,  process_mod,((-0.008, 0.008),(-0.008,0.008)))


figRho, axRho = plt.subplots()
axRho.plot(T_lite, Rho_lite.T)
axRho.set(title="Rho")
plt.show()