import numpy as np

from solver.boundary_conditions import AffineBoundaryCondition
from solver.elastoplastic_process import ElastoplasticProcess

class Grid:
    def __init__(self, n1, n2, is_node_func, xi_func, add_springs_func, a_func, cminus_func, cplus_func, r_and_r_prime_component_functions, force_func):
        self.n1 = n1
        self.n2 = n2

        self.d = 2
        self.node_id_by_coord = -1*np.ones((n1, n2)).astype(int) #-1 means no node
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
            xi=xi_func((i,j))
            self.xi[2*k] =  xi[0]
            self.xi[2*k+1] = xi[1]
            termins = add_springs_func((i, j))
            for k1 in range(len(termins)):
                (i1,j1)=termins[k1]
                edge_vect = np.zeros((self.n, 1))
                edge_vect[k,0] = 1
                edge_vect[self.node_id_by_coord[i1, j1], 0] = -1
                self.Q = np.append(self.Q, edge_vect, axis=1)
                self.connections.append((k,self.node_id_by_coord[i1,j1]))

        self.m = self.Q.shape[1]
        self.e0 = np.zeros(self.m)

        self.a=np.zeros(self.m)
        self.cminus = np.zeros(self.m)
        self.cplus = np.zeros(self.m)
        for i in range(self.m):
            self.a[i] = a_func(self.node_coord_by_id[self.connections[i][0]], self.node_coord_by_id[self.connections[i][1]])
            self.cminus[i] = cminus_func(self.node_coord_by_id[self.connections[i][0]], self.node_coord_by_id[self.connections[i][1]])
            self.cplus[i] = cplus_func(self.node_coord_by_id[self.connections[i][0]], self.node_coord_by_id[self.connections[i][1]])

        self.boundary_condition = self.get_node_wise_boundary_condition(r_and_r_prime_component_functions, force_func)

    def get_node_wise_boundary_condition(self, r_and_r_prime_component_functions, force_func):
        ## BOUNDARY CONDITION
        q_val = 0
        R = np.zeros((0, self.n * self.d))
        self.rho_list = []
        for k in range(self.n):
            (i, j) = self.node_coord_by_id[k]
            for component in range(self.d):
                r_func = r_and_r_prime_component_functions((i,j), component) #checking that there is a constraint on that node
                if r_func is not None:
                #a new row of R which corresponds to the node and the component
                    R = np.vstack((R, np.zeros((1, self.n * self.d))))
                    R[q_val, k * self.d + component] = 1
                    self.rho_list.append(((i, j), component, r_func))
                    q_val = q_val + 1
        def r(t):
            r = np.zeros(q_val)
            counter = 0
            for constr in self.rho_list:
                r_func = constr[2]
                r[counter] = -r_func(t)[0]
                counter = counter + 1
            return r

        def r_prime(t):
            r = np.zeros(q_val)
            counter = 0
            for constr in self.rho_list:
                r_func = constr[2]
                r[counter] = -r_func(t)[1]
                counter = counter + 1
            return r

        def f(t):
            forces = np.zeros(self.n * self.d)
            for k in range(self.n):
                f = force_func(self.node_coord_by_id[k])
                forces[self.d * k] = f(t)[0]
                forces[self.d * k + 1] = f(t)[1]
            return forces

        return AffineBoundaryCondition(q_val, R, r, r_prime, f)






