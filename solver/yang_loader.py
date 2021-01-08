import numpy as np

from solver.boundary_conditions import AffineBoundaryCondition
from solver.elastoplastic_process import ElastoplasticProcess

class Yang_loader:
    def __init__(self, vertices_filepath, adjacency_filepath, a_func, cminus_func, cplus_func,width_change, width_prime,height_change,height_prime):
        self.d = 2
        self.width_change = width_change
        self.height_change = height_change

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

        for i in range(self.m):
            orig=self.connections[i][0]
            termin=self.connections[i][1]
            orig_coords=(self.xi[2*orig], self.xi[2*orig+1])
            termin_coords=(self.xi[2*termin], self.xi[2*termin+1])
            self.a[i] = a_func(orig_coords,termin_coords)
            self.cminus[i] = cminus_func(orig_coords,termin_coords)
            self.cplus[i] = cplus_func(orig_coords,termin_coords)

        self.e0 = np.zeros(self.m)
        self.boundary_condition = self.get_periodic_boundary_condition(width_change, width_prime,height_change,height_prime)


    def get_periodic_boundary_condition(self,width_change, width_prime,height_change,height_prime):
        ## BOUNDARY CONDITION
        across_cutoff_x = self.width/2.  # value to distinguish the connections across the simulation box
        across_cutoff_y = self.height/2.
        self.connections_across_X = []
        self.connections_across_Y = []

        for i in range(self.m):
            orig = self.connections[i][0]
            termin = self.connections[i][1]
            orig_coords = (self.xi[2 * orig], self.xi[2 * orig + 1])
            termin_coords = (self.xi[2 * termin], self.xi[2 * termin + 1])

            if np.abs(orig_coords[0] - termin_coords[0]) > across_cutoff_x:
                self.connections_across_X.append(i)
            if np.abs(orig_coords[1] - termin_coords[1]) > across_cutoff_y:
                self.connections_across_Y.append(i)

        left_side_nodes = {} #pairs of node numbers, list of "long" connections
        self.duplicate_horizontal_nodes = {}

        for j in self.connections_across_X:
            #getting nodes ids
            orig = self.connections[j][0]
            termin = self.connections[j][1]
            #getting nodes coords
            orig_coords = (self.xi[2 * orig], self.xi[2 * orig + 1])
            termin_coords = (self.xi[2 * termin], self.xi[2 * termin + 1])
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
            node_vect = np.zeros((1, self.m))
            for edge in left_side_nodes[node]:
                self.connections[edge] = (self.n, self.connections[edge][1]) if self.Q[node, edge]==1 else (self.connections[edge][0],self.n) #modify connections depending if current node is orig or termin
                node_vect[0,edge] = self.Q[node, edge]
                self.Q[node, edge] = 0

            self.duplicate_horizontal_nodes[node] = self.n
            self.Q = np.append(self.Q, node_vect, axis=0)
            self.xi = np.append(self.xi, np.array([self.xi[2*node] + self.width, self.xi[2 * node + 1]]))
            self.n = self.n+1

        # VERTICAL "LONG" CONNECTIONS:
        bottom_side_nodes = {}  # pairs of node number, list of "long" connections
        self.duplicate_vertical_nodes = {}
        for j in self.connections_across_Y:
            orig = self.connections[j][0]
            termin = self.connections[j][1]
            orig_coords = (self.xi[2 * orig], self.xi[2 * orig + 1])
            termin_coords = (self.xi[2 * termin], self.xi[2 * termin + 1])
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
            node_vect = np.zeros((1, self.m))
            for edge in bottom_side_nodes[node]:
                self.connections[edge] = (self.n, self.connections[edge][1]) if self.Q[node, edge] == 1 else (self.connections[edge][0], self.n)  # modify connections depending if current node is orig or termin
                node_vect[0, edge] = self.Q[node, edge]
                self.Q[node, edge] = 0

            self.duplicate_vertical_nodes[node] = self.n
            self.Q = np.append(self.Q, node_vect, axis=0)
            self.xi = np.append(self.xi, np.array([self.xi[2 * node], self.xi[2 * node + 1]+self.height]))
            self.n = self.n + 1

        # R MATRIX
        R = np.zeros((2, self.n * self.d))
        R[0, 0] = 1 # fix the position of the first vertex
        R[1, 1] = 1
        for node in self.duplicate_horizontal_nodes:
            vect = np.zeros((2, self.n * self.d))
            vect[0, 2 * node] = -1            #x coordinate of the left node
            vect[1, 2 * node + 1 ] = -1       #y coordinate of the left node
            vect[0, 2 * self.duplicate_horizontal_nodes[node]] = 1         #x coordinate of the right node
            vect[1, 2 * self.duplicate_horizontal_nodes[node] + 1] = 1     #y coordinate of the right node
            R = np.append(R, vect, axis=0)
        for node in self.duplicate_vertical_nodes:
            vect = np.zeros((2, self.n * self.d))
            vect[0, 2 * node] = -1  # x coordinate of the bottom node
            vect[1, 2 * node + 1] = -1  # y coordinate of the bottom node
            vect[0, 2 * self.duplicate_vertical_nodes[node]] = 1  # x coordinate of the top node
            vect[1, 2 * self.duplicate_vertical_nodes[node] + 1] = 1  # y coordinate the top node
            R = np.append(R, vect, axis=0)

        def r(t):
            result = np.zeros(2)
            for node in self.duplicate_horizontal_nodes:
                result = np.append(result, np.array([-width_change(t), 0]))
            for node in self.duplicate_vertical_nodes:
                result = np.append(result, np.array([0, -height_change(t)]))
            return result

        def r_prime(t):
            result = np.zeros(2)
            for node in self.duplicate_horizontal_nodes:
                result = np.append(result, np.array([-width_prime(t), 0]))
            for node in self.duplicate_vertical_nodes:
                result = np.append(result, np.array([0, -height_prime(t)]))
            return result

        def f(t):
            forces = np.zeros(self.n * self.d)
            return forces

        return AffineBoundaryCondition(R.shape[0], R, r, r_prime, f)






