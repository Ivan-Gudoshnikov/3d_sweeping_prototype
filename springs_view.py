import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from elastoplastic_process import ElastoplasticProcess, vector_to_matrix


class SpringsView:
    def __init__(self, T, XI, E, problem: ElastoplasticProcess, lim):
        self.T = T
        self.XI = XI
        self.E = E
        self.problem=problem
        self.i = 0
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlim=lim[0], ylim=lim[1], aspect='equal')

        xi = vector_to_matrix(XI[:, self.i], self.problem.get_d())
        if self.problem.get_d() == 2:
            self.nodes_markers = Line2D([0,1],[0,1], marker="o", linestyle="None", markerfacecolor="k",
                                        markeredgecolor="k", markersize=5)
            self.ax.add_line(self.nodes_markers)

            self.springs_lines = []
            for i in range(problem.get_m()):
                self.springs_lines.append(Line2D([0,1],[0,1], marker=None, color="k", linewidth=1))
                self.ax.add_line(self.springs_lines[-1])        #add the last created line to the axes

            self.nodes_markers.set_data(xi[:,0], xi[:,1])
            for i in range(problem.get_m()):
                xdata = [xi[self.problem.connections[i][0], 0], xi[self.problem.connections[i][1], 0]]
                ydata = [xi[self.problem.connections[i][0], 1], xi[self.problem.connections[i][1], 1]]
                self.springs_lines[i].set_data(xdata, ydata)


        plt.show()
