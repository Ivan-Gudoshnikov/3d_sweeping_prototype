import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from solver.elastoplastic_process import ElastoplasticProcess, vector_to_matrix
import matplotlib.animation as animation


class SpringsViewStatic:
    def __init__(self, t, xi, e, problem: ElastoplasticProcess, lim, filename=None, fps=None):
        if problem.get_d()!=2:
            raise NameError("3d networks are not supported yet")

        self.problem = problem
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlim=lim[0], ylim=lim[1], aspect='equal')

        self.nodes_markers = Line2D([0, 1], [0, 1], marker="None", linestyle="None", markerfacecolor="k",
                                    markeredgecolor="k", markersize=5)
        self.ax.add_line(self.nodes_markers)

        self.springs_lines = []
        for i in range(problem.get_m()):
            self.springs_lines.append(Line2D([0, 1], [0, 1], marker=None, color="k", linewidth=1))
            self.ax.add_line(self.springs_lines[-1])  # add the last created line to the axes

        self.time_text = plt.text(lim[0][0]+0.1,-0.8,"T="+ format(t, '.6f'))

        xi = vector_to_matrix(xi[:], self.problem.get_d())
        self.nodes_markers.set_data(xi[:,0], xi[:,1])
        active = self.problem.e_bounds_box.get_active_box_faces(e[:], eps=None) #None means the same eps as in the computations

        for j in range(problem.get_m()):
                xdata = [xi[self.problem.connections[j][0], 0], xi[self.problem.connections[j][1], 0]]
                ydata = [xi[self.problem.connections[j][0], 1], xi[self.problem.connections[j][1], 1]]
                self.springs_lines[j].set_data(xdata, ydata)

                #show stress level:
                if e[j] >= 0:
                    hue1 = e[j]/self.problem.get_elastic_bounds()[1][j]
                else:
                    hue1 = e[j]/self.problem.get_elastic_bounds()[0][j]
                hue2 = (1-hue1)*0.25 #linear interpolation between green(0.25) and red (0)

                #show stiffnesses:
                #hue2 = self.problem.get_A()[j,j]/12

                self.springs_lines[j].set_color(matplotlib.colors.hsv_to_rgb((abs(hue2),1,0.9)))


                if active[j] == 0:
                    thickness = 2
                else:
                    thickness = 4
                self.springs_lines[j].set_linewidth(thickness)

