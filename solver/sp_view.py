#from ._mkl_bootstrap import _load_mkl_win
#_load_mkl_win()
#del _load_mkl_win

import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from solver.elastoplastic_process import ElastoplasticProcess, vector_to_matrix
import matplotlib.animation as animation
import numpy as np
import pypoman


class SweepingView:
    def __init__(self, T, XI, E, points, problem: ElastoplasticProcess, lim, filename=None, fps=None):
        if problem.get_d()!=2:
            raise NameError("3d networks are not supported yet")
        self.T = T
        self.XI = XI
        self.E = E
        self.problem = problem
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlim=lim[0], ylim=lim[1], aspect='equal')

        vertices, self.v_basis, self.p_v_coords =  self.get_vertices2d(0, XI[:,0])
        self.moving_set_polygon = Polygon(vertices)
        self.ax.add_patch(self.moving_set_polygon)


        solVcoords = np.matmul(self.p_v_coords, self.E)
        self.sol_line=Line2D(solVcoords[0,:], solVcoords[1,:],color='k')
        self.ax.add_line(self.sol_line)

        points_V=np.matmul(self.p_v_coords, points)
        self.markers = Line2D([points_V[0,:],points_V[0,:]],[points_V[1,:],points_V[1,:]], marker='o',markerfacecolor='r', markeredgecolor='r', linestyle='None')
        self.ax.add_line(self.markers)

        #self.nodes_markers = Line2D([0, 1], [0, 1], marker="None", linestyle="None", markerfacecolor="k",
        #                            markeredgecolor="k", markersize=5)
        #self.ax.add_line(self.nodes_markers)

        #self.springs_lines = []
        #for i in range(problem.get_m()):
        #    self.springs_lines.append(Line2D([0, 1], [0, 1], marker=None, color="k", linewidth=1))
        #    self.ax.add_line(self.springs_lines[-1])  # add the last created line to the axes

        #self.artists = self.springs_lines.copy()
        #self.artists.append(self.nodes_markers)



    def get_vertices2d(self,  t_ref, xi_ref):
        d_xi_phi = self.problem.d_xi_phi(xi_ref)
        d_xi_rho = self.problem.d_xi_rho(xi_ref, t_ref)
        [p_u_coords, p_v_coords] = self.problem.p_u_and_p_v_coords(d_xi_phi, d_xi_rho)
        u_basis = self.problem.u_basis(d_xi_phi, d_xi_rho)
        v_basis = self.problem.v_basis(d_xi_phi, d_xi_rho)
        H = self.problem.H(d_xi_phi, d_xi_rho)
        R = self.problem.R(d_xi_rho)
        h_u_coords = self.problem.h_u_coords(p_u_coords, H, self.problem.f(t_ref))
        moving_set = self.problem.moving_set(p_u_coords, h_u_coords)
        d_t_rho = self.problem.d_t_rho(xi_ref, t_ref)
        g_v_coords = self.problem.g_v_coords(p_v_coords, d_xi_phi, R, d_t_rho)
        V_constr = np.matmul(moving_set.A, v_basis)

        vertices = pypoman.compute_polytope_vertices(V_constr, moving_set.b)
        vertices[2], vertices[3] = vertices[3], vertices[2]
        return np.array(vertices), v_basis, p_v_coords





