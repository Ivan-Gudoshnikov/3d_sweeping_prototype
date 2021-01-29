from solver import time_event_pick
from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
from solver.springs_view_static import SpringsViewStatic
from solver.yang_loader import Yang_loader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_boundary_connections_reaction(loader, process, T,T_leapfrog, E, E_leapfrog, Sigma, Rho, grey_y_interval_main, grey_y_interval_minor, lim):
    figRho_main, axRho_main = plt.subplots()
    axRho_main.set(title="Rho: horizontal links - sum of x components and vertical links - sum of y components")

    figRho_h_y, axRho_h_y = plt.subplots()
    axRho_h_y.set(title="Rho: horizontal links - sum of y components")

    figRho_v_x, axRho_v_x = plt.subplots()
    axRho_v_x.set(title="Rho: vertical links - sum of x components")

    for i in range(T_leapfrog.shape[0]):
        t = T_leapfrog[i]
        if t>0  and t<T[-1]:
            axRho_main.add_line(Line2D([t, t], grey_y_interval_main, marker=None, color="lightgrey", gid=i, picker=True))
            axRho_h_y.add_line(Line2D([t, t], grey_y_interval_minor, marker=None, color="lightgrey", gid=i, picker=True))
            axRho_v_x.add_line(Line2D([t, t], grey_y_interval_minor, marker=None, color="lightgrey", gid=i, picker=True))

    on_pick = time_event_pick.get_callback(process, E_leapfrog, T_leapfrog, lim)

    figRho_main.canvas.callbacks.connect('pick_event', on_pick)
    figRho_h_y.canvas.callbacks.connect('pick_event', on_pick)
    figRho_v_x.canvas.callbacks.connect('pick_event', on_pick)

    l_h = len(loader.duplicate_horizontal_nodes)
    l_v = len(loader.duplicate_vertical_nodes)
    rho_horizontal_links_x = Rho[[2 + 2 * j for j in range(l_h)], :]
    rho_horizontal_links_y = Rho[[3 + 2 * j for j in range(l_h)], :]
    rho_vertical_links_x   = Rho[[2 + 2 * j + 2 * l_h for j in range(l_v)], :]
    rho_vertical_links_y   = Rho[[3 + 2 * j + 2 * l_h for j in range(l_v)], :]

    np.savetxt("delaunay_h_y.txt", rho_horizontal_links_y,fmt="%+.18e")
    np.savetxt("delaunay_v_x.txt", rho_vertical_links_x,fmt="%+.18e")

    sum_rho_horizontal_links_x = np.sum(rho_horizontal_links_x,0)
    sum_rho_horizontal_links_y = np.sum(rho_horizontal_links_y,0)
    sum_rho_vertical_links_x = np.sum(rho_vertical_links_x,0)
    sum_rho_vertical_links_y = np.sum(rho_vertical_links_y,0)

    np.savetxt("delaunay_diff.txt",sum_rho_horizontal_links_y - sum_rho_vertical_links_x)


    axRho_main.plot(T, sum_rho_horizontal_links_x,label="horizontal connections, x-component")
    axRho_main.plot(T, sum_rho_vertical_links_y,label="vertical connections,y-component")
    axRho_main.legend()

    axRho_h_y.plot(T, sum_rho_horizontal_links_y)
    axRho_v_x.plot(T, sum_rho_vertical_links_x)