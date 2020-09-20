import numpy as np

from solver.boundary_conditions import BoundaryConditions
from solver.elastoplastic_process import ElastoplasticProcess

class Yang_loader:
    def __init__(self, vertices_filepath, adjacency_filepath, a_func, cminus_func, cplus_func, add_boundary_cond_func):
        file1 = open(vertices_filepath,"r")
        lines = file1.readlines()
        file1.close()
        self.d = 2

        #first 3 lines are service
        self.n=len(lines)-3
        self.xi = np.zeros(self.n*self.d)
        for i in range(self.n):
            xi_x = float(lines[i+3].split()[0])
            xi_y = float(lines[i+3].split()[1])
            self.xi[2*i] = xi_x
            self.xi[2*i+1] = xi_y
        #print(xi)

        file2 = open(adjacency_filepath,"r")
        adjacency = np.loadtxt(file2)
        file2.close()
        #print(adjacency)

        if self.n!= adjacency.shape[0] or self.n!= adjacency.shape[1]:
            raise(NameError("Dimensions of adjacency matrix does not match the amount of nodes!"))


        self.connections = []
        self.Q = np.zeros((self.n,0))
        for i in range(1,self.n):
            for j in range(i+1,self.n):
                if adjacency[i,j]==1:
                    edge_vect = np.zeros((self.n, 1))
                    edge_vect[i, 0] = 1
                    edge_vect[j, 0] = -1
                    self.Q = np.append(self.Q, edge_vect, axis=1)
                    self.connections.append((i, j))
        self.m = self.Q.shape[1]
        self.a = np.zeros(self.m)
        self.cminus = np.zeros(self.m)
        self.cplus = np.zeros(self.m)
        for i in range(self.m):
            orig=self.connections[i][0]
            termin=self.connections[i][1]
            orig_coords=(self.xi[2*orig], self.xi[2*orig+1])
            termin_coords=(self.xi[2*termin], self.xi[2*termin+1])
            self.a[i] = a_func(orig_coords,termin_coords)
            self.cminus[i] = cminus_func(orig_coords,termin_coords)
            self.cplus[i] = cplus_func(orig_coords,termin_coords)

        boundary_cond = NodeWise(self, add_boundary_cond_func)

        # TODO: rho setup API
        # hardcoded rho

        self.rho = lambda xi, t: boundary_cond.rho(self, xi, t)
        self.d_xi_rho = lambda xi, t: boundary_cond.d_xi_rho(self, xi, t)
        self.d_t_rho = lambda xi, t: boundary_cond.d_t_rho(self, xi, t)
        self.f = lambda t: boundary_cond.f(self, t)

        self.e0 = np.zeros(self.m)

        self.q = boundary_cond.q(self)

    def get_elastoplastic_process(self):
        return ElastoplasticProcess(self.Q, self.a, self.cminus, self.cplus, self.d, self.q, self.rho, self.d_xi_rho,
                                    self.d_t_rho, self.f)

    def get_xi(self):
        return self.xi

    def get_e0(self):
        return self.e0

    def get_min_x_in_xi(self):
        min = np.Inf
        for i in range(self.n):
            if self.xi[2*i] < min:
                min = self.xi[2*i]
        return min

    def get_min_y_in_xi(self):
        min = np.Inf
        for i in range(self.n):
            if self.xi[2 * i+1] < min:
                min = self.xi[2 * i + 1]
        return min

    def get_max_x_in_xi(self):
        max = np.NINF
        for i in range(self.n):
            if self.xi[2 * i] > max:
                max = self.xi[2 * i]
        return max

    def get_max_y_in_xi(self):
        max = np.NINF
        for i in range(self.n):
            if self.xi[2 * i+1] > max:
                max = self.xi[2 * i+1]
        return max


class NodeWise(BoundaryConditions):
        def __init__(self, grid: Yang_loader, add_boundary_cond_func):
            self.add_boundary_cond_func = add_boundary_cond_func
            self.q_val = 0
            self.d_xi_rho_mat = np.zeros((0, grid.n * grid.d))
            self.rho_list = []
            for k in range(grid.n):
                coords=(grid.xi[2*k],grid.xi[2*k + 1])
                (v, f) = add_boundary_cond_func(coords, grid)
                if v is not None:
                    coords = (grid.xi[2*k], grid.xi[2*k+1])
                    self.d_xi_rho_mat = np.vstack((self.d_xi_rho_mat, np.zeros((2, grid.n * grid.d))))
                    self.d_xi_rho_mat[self.q_val, k * grid.d] = 1
                    self.d_xi_rho_mat[self.q_val + 1, k * grid.d + 1] = 1
                    self.rho_list.append(coords)
                    self.q_val = self.q_val + grid.d

        def rho(self, grid, xi, t):
            raise ValueError("Not implemented")

        def d_xi_rho(self, grid, xi, t):
            return self.d_xi_rho_mat

        def d_t_rho(self, grid, xi, t):
            velocities = np.zeros(self.q_val)
            for r in range(len(self.rho_list)):
                (v, f) = self.add_boundary_cond_func(self.rho_list[r], grid)
                velocities[grid.d * r] = v(t)[0]
                velocities[grid.d * r + 1] = v(t)[1]
            return velocities

        def f(self, grid, t):
            forces = np.zeros(grid.n * grid.d)
            for k in range(grid.n):
                coords = (self.xi[2 * k], self.xi[2 * k + 1])
                (v, f) = self.add_boundary_cond_func(coords, grid)
                forces[grid.d * k] = f(t)[0]
                forces[grid.d * k + 1] = f(t)[1]
            return forces

        def q(self, grid):
            return self.q_val






