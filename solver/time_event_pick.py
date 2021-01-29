import numpy as np
from solver.springs_view_static import SpringsViewStatic
import matplotlib.pyplot as plt

#############################################################################
# auxilliary function to make the lines denoting the time moments interactive
#############################################################################

def get_callback(process, E_leapfrog, T_leapfrog, lim):
    def on_pick(event):
        i = event.artist.get_gid()
        active_i = process.e_bounds_box.get_active_box_faces(E_leapfrog[:, i], eps=None)
        if i > 0:
            active_prev = process.e_bounds_box.get_active_box_faces(E_leapfrog[:, i - 1], eps=None)
            diff_edges = np.where(np.invert(active_i == active_prev))[0]
            diff_nodes = []
            for j in diff_edges:
                diff_nodes.append(process.connections[j][0])
                diff_nodes.append(process.connections[j][1])
        else:
            diff_nodes = None

        SpringsViewStatic(T_leapfrog[i], process.xi0, E_leapfrog[:, i], process, lim, highlight=diff_nodes)
        plt.show()

    return on_pick