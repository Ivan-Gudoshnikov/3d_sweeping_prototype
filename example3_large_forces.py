from square_grid import SquareGrid
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from elastoplastic_process import ElastoplasticProcess
from springs_view import SpringsView

example3grid = SquareGrid(20,20, 0.5, 0.5, lambda conn: 1., lambda conn: -0.1, lambda conn: 0.1, SquareGrid.HoldLeftStressRight())

example3 = example3grid.get_elastoplastic_process()
t0 = 0
dt = 0.01
nsteps = 170

xi_ref = example3grid.xi
t_ref = 0
(T, E) = example3.solve_fixed_spaces_e_only(example3grid.xi, example3grid.e0,t0, dt, nsteps, xi_ref, t_ref)
np.save("ex3_forces_20x20_T", T)
np.save("ex3_forces_20x20_E", E)

figE, axE = plt.subplots()
axE.plot(T, E.T)
axE.set(title="E")

XI = np.tile(np.expand_dims(xi_ref, axis=1),(1,T.shape[0]))

SpringsView(T,XI,E, example3,((-3,12),(-1,12)))

plt.show()