import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from solver.elastoplastic_process import ElastoplasticProcess, vector_to_matrix
import matplotlib.animation as animation


class SpringsView:
    def __init__(self, T, XI, E, problem, lim,time_text_coords, filename=None, fps=None):
        if problem.d!=2:
            raise NameError("3d networks are not supported yet")
        self.T = T
        self.XI = XI
        self.E = E
        self.problem = problem
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlim=lim[0], ylim=lim[1], aspect='equal')

        self.nodes_markers = Line2D([0, 1], [0, 1], marker="None", linestyle="None", markerfacecolor="k",
                                    markeredgecolor="k", markersize=5)
        self.ax.add_line(self.nodes_markers)

        self.springs_lines = []
        for i in range(problem.m):
            self.springs_lines.append(Line2D([0, 1], [0, 1], marker=None, color="k", linewidth=1))
            self.ax.add_line(self.springs_lines[-1])  # add the last created line to the axes

        self.artists = self.springs_lines.copy()
        self.artists.append(self.nodes_markers)

        self.time_text = plt.text(time_text_coords[0],time_text_coords[1],"T=")
        self.artists.append(self.time_text)

        def init_animation():
            return self.artists

        def update_animation(i):
            self.time_text.set_text("T=" + format(self.T[i], '.6f'))
            xi = vector_to_matrix(self.XI[:, i], self.problem.d)

            self.nodes_markers.set_data(xi[:,0], xi[:,1])
            active = self.problem.e_bounds_box.get_active_box_faces(E[:, i], eps=None) #None means the same eps as in the computations

            for j in range(problem.m):
                xdata = [xi[self.problem.connections[j][0], 0], xi[self.problem.connections[j][1], 0]]
                ydata = [xi[self.problem.connections[j][0], 1], xi[self.problem.connections[j][1], 1]]
                self.springs_lines[j].set_data(xdata, ydata)

                #show stress level:
                if E[j,i] >= 0:
                    hue1 = E[j,i]/(self.problem.K[j,j]*self.problem.cplus[j])
                else:
                    hue1 = E[j,i]/(self.problem.K[j,j]*self.problem.cminus[j])
                hue2 = (1-hue1)*0.25 #linear interpolation between green(0.25) and red (0)

                #show stiffnesses:
                #hue2 = self.problem.get_A()[j,j]/12

                self.springs_lines[j].set_color(matplotlib.colors.hsv_to_rgb((abs(hue2),1,0.9)))


                if active[j] == 0:
                    thickness = 2
                else:
                    thickness = 4
                self.springs_lines[j].set_linewidth(thickness)
            return self.artists

        self.ani = animation.FuncAnimation(self.fig, update_animation, init_func=init_animation, frames=self.T.shape[0], interval=1, blit=True, repeat=True)
        if filename is not None:
            self.ani.save(filename, fps=fps)
