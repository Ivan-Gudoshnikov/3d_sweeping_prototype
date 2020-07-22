from square_grid import SquareGrid
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from elastoplastic_process import ElastoplasticProcess
from springs_view import SpringsView
import math


n1=13
n2=9

def a_func(orig, termin):
    base_stiffess = 1
    if (abs(orig[0] - 4)+ abs(orig[1] - 4)<=2 and abs(termin[0] - 4)+ abs(termin[1] - 4)<=2) or (abs(orig[0] - 8) + abs(orig[1] - 4) <= 2 and abs(termin[0] - 8) + abs(termin[1] - 4) <= 2):
        base_stiffess = 5

    if (orig[0] == termin[0]) or (orig[1] == termin[1]):
        stiffness=2*base_stiffess #non-diagonal springs have stiffness 2x
    else:
        stiffness=base_stiffess  #diagonal springs have stiffness 1x
    return stiffness

def cplus_func(orig, termin):
    base_yeld_stress=0.001

    if (orig[0] == termin[0]) or (orig[1] == termin[1]):
        yeld_stress=base_yeld_stress #non-diagonal springs
    else:
        yeld_stress = base_yeld_stress/math.sqrt(2)  #diagonal sptings

    return yeld_stress

def cminus_func(orig, termin):
    return -cplus_func(orig, termin)




example3grid = SquareGrid(n1, n2, 0.5, 0.5, a_func, cminus_func,cplus_func, SquareGrid.HoldLeftStressRight())

bc=example3grid.HoldLeftDisplacementRight()
example3 = example3grid.get_elastoplastic_process()


t0 = 0
dt = 0.00001
nsteps = 1500

xi_ref = example3grid.xi
t_ref = 0
(T, E) = example3.solve_fixed_spaces_e_only(example3grid.xi, example3grid.e0,t0, dt, nsteps, xi_ref, t_ref)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi_ref, axis=1),(1,T.shape[0]))

SpringsView(T,XI,E, example3,((-3,7),(-1,8)),"example3_non-homo_forces.mp4",20)
#SpringsView(T,XI,E, example3,((-3,7),(-1,8)))
plt.show()