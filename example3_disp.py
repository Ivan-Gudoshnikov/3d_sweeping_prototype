from square_grid import SquareGrid
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from elastoplastic_process import ElastoplasticProcess
from springs_view import SpringsView



n1=10
n2=10
def a_func(orig, termin):
    if (orig[0] == termin[0]) or (orig[1] == termin[1]):
        return 2. #non-diagonal springs have stiffness 2
    else:
        return 1. #diagonal sptings have stiffness 1

cminus_func =  lambda orig, termin: -0.1
cplus_func = lambda orig, termin: 0.1


example3grid = SquareGrid(n1, n2, 0.5, 0.5, a_func, cminus_func,cplus_func, SquareGrid.HoldLeftDisplacementRight())

bc=example3grid.HoldLeftDisplacementRight()
example3 = example3grid.get_elastoplastic_process()


t0 = 0
dt = 0.005
nsteps = 3400

xi_ref = example3grid.xi
t_ref = 0
(T, E) = example3.solve_fixed_spaces_e_only(example3grid.xi, example3grid.e0,t0, dt, nsteps, xi_ref, t_ref)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi_ref, axis=1),(1,T.shape[0]))

SpringsView(T,XI,E, example3,((-3,7),(-1,8)))

plt.show()