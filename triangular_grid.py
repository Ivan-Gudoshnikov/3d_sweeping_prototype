import numpy as np
import math
from elastoplastic_process import ElastoplasticProcess
from boundary_conditions import BoundaryConditions


class TriangularGrid:
    def __init__(self, n1, n2, delta, is_node_func,  add_springs_func, a_func, cminus_func, cplus_func, add_boundary_cond_func):
        self.n1 = n1
        self.n2 = n2

        self.d = 2
        self.node_id_by_coord = -1*np.ones((n1, n2)).astype(int)
        self.node_coord_by_id = []
        self.connections = []

        k = 0
        for i in range(n1):
            for j in range(n2):
                if is_node_func((i,j)):
                    self.node_id_by_coord[i, j] = k
                    self.node_coord_by_id.append((i,j))
                    k = k + 1
        self.n = k
        self.xi = np.zeros(self.n * self.d)
        self.Q = np.zeros((self.n,0))
        for k in range(self.n):
            (i, j) = self.node_coord_by_id[k]
            self.xi[2*k] = i * delta - j * delta / 2.
            self.xi[2*k + 1] = j * delta * math.sqrt(3) / 2
            termins = add_springs_func((i, j))
            for k1 in range(len(termins)):
                (i1,j1)=termins[k1]
                edge_vect = np.zeros((self.n, 1))
                edge_vect[k,0] = 1
                edge_vect[self.node_id_by_coord[i1, j1], 0] = -1
                self.Q = np.append(self.Q, edge_vect, axis=1)
                self.connections.append((k,self.node_id_by_coord[i1,j1]))



        self.m = self.Q.shape[1]

        self.a=np.zeros(self.m)
        self.cminus = np.zeros(self.m)
        self.cplus = np.zeros(self.m)
        for i in range(self.m):
            self.a[i] = a_func(self.node_coord_by_id[self.connections[i][0]], self.node_coord_by_id[self.connections[i][1]])
            self.cminus[i] = cminus_func(self.node_coord_by_id[self.connections[i][0]], self.node_coord_by_id[self.connections[i][1]])
            self.cplus[i] = cplus_func(self.node_coord_by_id[self.connections[i][0]], self.node_coord_by_id[self.connections[i][1]])

        boundary_cond = NodeWise(self, add_boundary_cond_func)

        #TODO: rho setup API
        #hardcoded rho

        self.rho = lambda xi, t: boundary_cond.rho(self, xi,t)
        self.d_xi_rho = lambda xi, t: boundary_cond.d_xi_rho(self, xi, t)
        self.d_t_rho = lambda xi, t: boundary_cond.d_t_rho(self, xi, t)
        self.f = lambda t: boundary_cond.f(self, t)

        self.e0 = np.zeros(self.m)

        self.q = boundary_cond.q(self)

    def get_elastoplastic_process(self):
        return ElastoplasticProcess(self.Q, self.a, self.cminus, self.cplus, self.d, self.q, self.rho, self.d_xi_rho, self.d_t_rho, self.f)


class NodeWise(BoundaryConditions):
    def __init__(self, outer: TriangularGrid, add_boundary_cond_func):
        self.add_boundary_cond_func=add_boundary_cond_func
        self.q_val = 0
        self.d_xi_rho_mat = np.zeros((0,outer.n*outer.d))
        self.rho_list = []
        for k in range(outer.n):
            (i, j) = outer.node_coord_by_id[k]
            (v, f) = add_boundary_cond_func((i, j))
            if v is not None:
                self.d_xi_rho_mat = np.vstack((self.d_xi_rho_mat, np.zeros((2, outer.n*outer.d))))
                self.d_xi_rho_mat[self.q_val, k*outer.d] = 1
                self.d_xi_rho_mat[self.q_val+1, k * outer.d+1] = 1
                self.rho_list.append((i, j))
                self.q_val = self.q_val + outer.d

    def rho(self, outer, xi, t):
        raise ValueError("Not implemented")

    def d_xi_rho(self, outer, xi, t):
        return self.d_xi_rho_mat

    def d_t_rho(self, outer, xi, t):
        velocities = np.zeros(self.q_val)
        for r in range(len(self.rho_list)):
            (v, f) = self.add_boundary_cond_func(self.rho_list[r])
            velocities[outer.d * r] = v(t)[0]
            velocities[outer.d * r + 1] = v(t)[1]
        return velocities

    def f(self, outer, t):
        forces = np.zeros(outer.n*outer.d)
        for k in range(outer.n):
            (v, f) = self.add_boundary_cond_func(outer.node_coord_by_id[k])
            forces[outer.d * k] = f(t)[0]
            forces[outer.d * k + 1] = f(t)[1]
        return forces

    def q(self, outer):
        return self.q_val


