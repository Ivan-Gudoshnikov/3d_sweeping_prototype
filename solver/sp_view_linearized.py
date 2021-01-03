#from ._mkl_bootstrap import _load_mkl_win
#_load_mkl_win()
#del _load_mkl_win

import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from solver.elastoplastic_process_linearized import Elastoplastic_process_linearized
import matplotlib.animation as animation
import numpy as np
import pypoman



class SweepingViewLinearized:
    def __init__(self, T, E, points, process: Elastoplastic_process_linearized, lim, filename=None, fps=None):
        if process.d!=2:
            raise NameError("3d networks are not supported yet")
        self.T = T
        self.E = E
        self.process = process
        self.fig, self.ax = plt.subplots()
        self.ax.grid()
        self.ax.set(xlim=lim[0], ylim=lim[1], aspect='equal')

        vertices =  self.get_vertices2d(0)
        self.moving_set_polygon = Polygon(vertices)
        self.ax.add_patch(self.moving_set_polygon)

        solVcoords = np.matmul(self.process.P_V_coords, self.E)
        self.sol_line=Line2D(solVcoords[0,:], solVcoords[1,:],color='k')
        self.ax.add_line(self.sol_line)

        points_V=np.matmul(self.process.P_V_coords, points)
        self.markers = Line2D([points_V[0,:],points_V[0,:]],[points_V[1,:],points_V[1,:]], marker='o',markerfacecolor='r', markeredgecolor='r', linestyle='None')
        self.ax.add_line(self.markers)

    def get_vertices2d(self,  t_ref):
        A = np.vstack((self.process.normals_in_V, -self.process.normals_in_V))
        eplusminus = np.hstack((self.process.Kinv @ self.process.cplus, -self.process.Kinv @ self.process.cminus))
        box_offset_by_force = - self.process.F @ self.process.f(t_ref)

        vertices = pypoman.compute_polytope_vertices(A,
                                                     eplusminus + np.hstack((box_offset_by_force , -box_offset_by_force)))
        vertices.sort(key=lambda vert: np.arctan2(vert[1], vert[0]))
        return np.array(vertices)





