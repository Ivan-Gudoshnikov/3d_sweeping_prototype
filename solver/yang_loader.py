import numpy as np

from solver.boundary_conditions import BoundaryConditions
from solver.elastoplastic_process import ElastoplasticProcess

class Yang_loader:
    def __init__(self, vertices_filepath, adjacency_filepath, a_func, cminus_func, cplus_func, width_prime, height_prime):
        self.d = 2
        self.width_prime = width_prime #derivatives of the dispacement loadings
        self.height_prime = height_prime

        #open vertex coords file
        file1 = open(vertices_filepath, "r")
        lines = file1.readlines()
        file1.close()


        #first 3 lines in the file are service headers
        self.n = int(lines[0].split()[0]) #the fist element of the first line is a number of vertices

        #auume the simulation box is rectangular with rectangular axes
        self.width = float(lines[1].split()[0]) #the first element of the second line is a width of the simulation box
        self.height = float(lines[2].split()[1]) #the second element of the third line is a height of the simulation box

        self.xi = np.zeros(self.n*self.d)
        for i in range(self.n):
            xi_x = float(lines[i+3].split()[0])
            xi_y = float(lines[i+3].split()[1])
            self.xi[2*i] = xi_x
            self.xi[2*i+1] = xi_y
        #print(xi)

        #adjacency matrix file
        file2 = open(adjacency_filepath,"r")
        adjacency = np.loadtxt(file2)
        file2.close()
        #print(adjacency)

        if self.n!= adjacency.shape[0] or self.n!= adjacency.shape[1]:
            raise(NameError("Dimensions of adjacency matrix does not match the amount of nodes!"))

        self.connections = []
        self.Q = np.zeros((self.n,0))


        for i in range(0,self.n):
            for j in range(i+1,self.n):
                if adjacency[i,j]==1:
                    edge_vect = np.zeros((self.n, 1))
                    edge_vect[i, 0] = 1
                    edge_vect[j, 0] = -1
                    self.Q = np.append(self.Q, edge_vect, axis=1)
                    self.connections.append((i, j))


        #parameters of the network
        self.m = self.Q.shape[1]
        self.a = np.zeros(self.m)
        self.cminus = np.zeros(self.m)
        self.cplus = np.zeros(self.m)

        across_cutoff_x = self.width/2.  # value to distinguish the connections across the simulation box
        across_cutoff_y = self.height/2.
        self.connections_across_X = []
        self.connections_across_Y = []

        for i in range(self.m):
            orig=self.connections[i][0]
            termin=self.connections[i][1]
            orig_coords=(self.xi[2*orig], self.xi[2*orig+1])
            termin_coords=(self.xi[2*termin], self.xi[2*termin+1])
            self.a[i] = a_func(orig_coords,termin_coords)
            self.cminus[i] = cminus_func(orig_coords,termin_coords)
            self.cplus[i] = cplus_func(orig_coords,termin_coords)

            if np.abs(orig_coords[0] - termin_coords[0]) > across_cutoff_x:
                self.connections_across_X.append(i)
            if np.abs(orig_coords[1] - termin_coords[1]) > across_cutoff_y:
                self.connections_across_Y.append(i)

        boundary_cond = PeriodicBoundaryConditions(grid=self)


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

class PeriodicBoundaryConditions(BoundaryConditions):
    def __init__(self, grid: Yang_loader):
        m = grid.m

        # HORIZONTAL "LONG" CONNECTIONS:
        left_side_nodes = {} #pairs of node numbers, list of "long" connections
        self.duplicate_horizontal_nodes = {}
        #
        for j in grid.connections_across_X:
            #getting nodes ids
            orig = grid.connections[j][0]
            termin = grid.connections[j][1]
            #getting nodes coords
            orig_coords = (grid.xi[2 * orig], grid.xi[2 * orig + 1])
            termin_coords = (grid.xi[2 * termin], grid.xi[2 * termin + 1])
            #forming the list of "left" nodes(to be duplicated) and adjacent "long" edges
            if termin_coords[0] - orig_coords[0] > 0:
                #orig is the "left" node
                if orig not in left_side_nodes:
                    left_side_nodes[orig] = [j]
                else:
                    left_side_nodes[orig].append(j)
            else:
                # termin is the "left" node
                if termin not in left_side_nodes:
                    left_side_nodes[termin] = [j]
                else:
                    left_side_nodes[termin].append(j)

        #updating grid.connections and grid.Q with the new node number n
        for node in left_side_nodes:
            node_vect = np.zeros((1, m))
            for edge in left_side_nodes[node]:
                grid.connections[edge] = (grid.n, grid.connections[edge][1]) if grid.Q[node, edge]==1 else (grid.connections[edge][0],grid.n) #modify connections depending if current node is orig or termin
                node_vect[0,edge] = grid.Q[node, edge]
                grid.Q[node, edge] = 0

            self.duplicate_horizontal_nodes[node] = grid.n
            grid.Q = np.append(grid.Q, node_vect, axis=0)
            grid.xi = np.append(grid.xi, np.array([grid.xi[2*node] + grid.width, grid.xi[2 * node + 1]]))
            grid.n = grid.n+1

        # VERTICAL "LONG" CONNECTIONS:
        bottom_side_nodes = {}  # pairs of node number, list of "long" connections
        self.duplicate_vertical_nodes = {}
        for j in grid.connections_across_Y:
            orig = grid.connections[j][0]
            termin = grid.connections[j][1]
            orig_coords = (grid.xi[2 * orig], grid.xi[2 * orig + 1])
            termin_coords = (grid.xi[2 * termin], grid.xi[2 * termin + 1])
            if termin_coords[1] - orig_coords[1] > 0:
                # orig is the "bottom" node
                if orig not in bottom_side_nodes:
                    bottom_side_nodes[orig] = [j]
                else:
                    bottom_side_nodes[orig].append(j)
            else:
                # termin is the "bottom" node
                if termin not in bottom_side_nodes:
                    bottom_side_nodes[termin] = [j]
                else:
                    bottom_side_nodes[termin].append(j)

        for node in bottom_side_nodes:
            node_vect = np.zeros((1, m))
            for edge in bottom_side_nodes[node]:
                grid.connections[edge] = (grid.n, grid.connections[edge][1]) if grid.Q[node, edge] == 1 else (grid.connections[edge][0], grid.n)  # modify connections depending if current node is orig or termin
                node_vect[0, edge] = grid.Q[node, edge]
                grid.Q[node, edge] = 0

            self.duplicate_vertical_nodes[node] = grid.n
            grid.Q = np.append(grid.Q, node_vect, axis=0)
            grid.xi = np.append(grid.xi, np.array([grid.xi[2 * node], grid.xi[2 * node + 1]+grid.height]))
            grid.n = grid.n + 1


        # DERIVATIVE MATRICES
        self.d_xi_rho_mat = np.zeros((2, grid.n * grid.d))
        self.d_xi_rho_mat[0, 0] = 1 # fix the position of the first vertex
        self.d_xi_rho_mat[1, 1] = 1
        for node in self.duplicate_horizontal_nodes:
            vect = np.zeros((2, grid.n * grid.d))
            vect[0, 2 * node] = -1            #x coordinate of the left node
            vect[1, 2 * node + 1 ] = -1       #y coordinate of the left node
            vect[0, 2 * self.duplicate_horizontal_nodes[node]] = 1         #x coordinate of the right node
            vect[1, 2 * self.duplicate_horizontal_nodes[node] + 1] = 1     #y coordinate of the right node
            self.d_xi_rho_mat = np.append(self.d_xi_rho_mat, vect, axis=0)
        for node in self.duplicate_vertical_nodes:
            vect = np.zeros((2, grid.n * grid.d))
            vect[0, 2 * node] = -1  # x coordinate of the bottom node
            vect[1, 2 * node + 1] = -1  # y coordinate of the bottom node
            vect[0, 2 * self.duplicate_vertical_nodes[node]] = 1  # x coordinate of the top node
            vect[1, 2 * self.duplicate_vertical_nodes[node] + 1] = 1  # y coordinate the top node
            self.d_xi_rho_mat = np.append(self.d_xi_rho_mat, vect, axis=0)

    def d_t_rho(self, grid, xi, t):
        result = np.zeros(2)
        for node in self.duplicate_horizontal_nodes:
            result = np.append(result, np.array([-grid.width_prime(t), 0]))
        for node in self.duplicate_vertical_nodes:
            result = np.append(result, np.array([0, -grid.height_prime(t)]))
        return result

    def d_xi_rho(self, grid, xi, t):
        return self.d_xi_rho_mat

    def rho(self, grid, xi, t):
        raise ValueError("Not implemented")

    def f(self, grid, t):
        forces = np.zeros(grid.n * grid.d)
        return forces

    def q(self, grid):
        return self.d_xi_rho_mat.shape[0]



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






