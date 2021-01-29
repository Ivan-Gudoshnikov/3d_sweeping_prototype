from solver import time_event_pick
from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
from solver.yang_loader import Yang_loader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_boundary_springs_stresses(loader, process, T,T_leapfrog, E, E_leapfrog, Sigma, Rho, grey_y_interval, lim):

    figSigma_springs, axSigma_springs = plt.subplots()
    axSigma_springs.set(title="Sum of stresses of springs across X and springs across Y")
    for i in range(T_leapfrog.shape[0]):
        t = T_leapfrog[i]
        if t>0  and t<T[-1]:
            axSigma_springs.add_line(Line2D([t, t], grey_y_interval, marker=None, color="lightgrey", gid=i, picker=True))

    on_pick = time_event_pick.get_callback(process, E_leapfrog, T_leapfrog, lim)
    figSigma_springs.canvas.callbacks.connect('pick_event', on_pick)

    Sigma_h = Sigma[loader.connections_across_X, :]
    Sigma_v = Sigma[loader.connections_across_Y, :]
    sum_h = np.sum(Sigma_h, 0)
    sum_v = np.sum(Sigma_v, 0)

    axSigma_springs.plot(T, sum_h, label="springs across x")
    axSigma_springs.plot(T, sum_v, label="springs across y")
    axSigma_springs.legend()