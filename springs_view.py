import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from elastoplastic_process import ElastoplasticProcess, vector_to_matrix
import matplotlib.animation as animation


class SpringsView:
    def __init__(self, T, XI, E, problem: ElastoplasticProcess, lim, filename=None):
        if problem.get_d()!=2:
            raise NameError("3d networks are not supported yet")

        self.T = T
        self.XI = XI
        self.E = E
        self.problem=problem
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlim=lim[0], ylim=lim[1], aspect='equal')

        self.nodes_markers = Line2D([0, 1], [0, 1], marker="o", linestyle="None", markerfacecolor="k",
                                    markeredgecolor="k", markersize=5)
        self.ax.add_line(self.nodes_markers)

        self.springs_lines = []
        for i in range(problem.get_m()):
            self.springs_lines.append(Line2D([0, 1], [0, 1], marker=None, color="k", linewidth=1))
            self.ax.add_line(self.springs_lines[-1])  # add the last created line to the axes

        self.artists = self.springs_lines.copy()
        self.artists.append(self.nodes_markers)

        def init_animation():
            return self.artists

        def update_animation(i):
            xi = vector_to_matrix(self.XI[:, i], self.problem.get_d())

            self.nodes_markers.set_data(xi[:,0], xi[:,1])
            active = self.problem.e_bounds_box.get_active_box_faces(E[:,i], eps=None) #None means the same eps as in the computations

            for i in range(problem.get_m()):
                xdata = [xi[self.problem.connections[i][0], 0], xi[self.problem.connections[i][1], 0]]
                ydata = [xi[self.problem.connections[i][0], 1], xi[self.problem.connections[i][1], 1]]
                self.springs_lines[i].set_data(xdata, ydata)
                if active[i] == 0:
                    thickness = 1
                else:
                    thickness = 4
                self.springs_lines[i].set_linewidth(thickness)
            return self.artists

        self.ani = animation.FuncAnimation(self.fig, update_animation, init_func=init_animation, frames=self.T.shape[0], interval=1, blit=True, repeat=True)
        if filename is not None:
            self.ani.save(filename)
