import numpy as np
from elastoplastic_process import ElastoplasticProcess
from abc import ABC, abstractmethod

class SquareGrid:
    class BoundaryConditions(ABC):

        @abstractmethod
        def rho(self, outer, xi,t):
            pass

        @abstractmethod
        def d_xi_rho(self, outer, xi, t):
            pass

        @abstractmethod
        def d_t_rho(self,outer, xi, t):
            pass

        @abstractmethod
        def f(self, outer, t):
            pass

        @abstractmethod
        def q(self, outer):
            pass

    def __init__(self, n1, n2, delta_x, delta_y, a_func, cminus_func, cplus_func, boundary_cond: BoundaryConditions):
        self.n1=n1
        self.n2=n2

        self.d = 2

        self.nodes = np.zeros((n1,n2)).astype(int)

        self.connections = []

        k = 0
        for i in range(n1):
            for j in range(n2):
                self.nodes[i, j] = k
                k = k + 1

        self.n = k
        self.xi = np.zeros(self.n * self.d)
        k = 0
        for i in range(n1):
            for j in range(n2):
                self.xi[k] = i * delta_x
                self.xi[k+1] = j * delta_y
                k = k + 2

        self.Q = np.zeros((self.n, 4*(n1-1)*(n2-1)+ (n1-1)+(n2-1))).astype(int)
        k = 0

        for i in range(n1-1):
            for j in range(n2-1):
                #bottom neighbour
                self.Q[self.nodes[i, j], k]=1
                self.Q[self.nodes[i, j+1], k] = -1
                self.connections.append((self.nodes[i, j], self.nodes[i, j+1]))
                k = k + 1


                #right neighbour
                self.Q[self.nodes[i, j], k] = 1
                self.Q[self.nodes[i+1, j], k] = -1
                self.connections.append((self.nodes[i, j], self.nodes[i+1, j]))
                k = k + 1

                #right-bottom neighbour
                self.Q[self.nodes[i, j], k] = 1
                self.Q[self.nodes[i + 1, j + 1], k] = -1
                self.connections.append((self.nodes[i, j], self.nodes[i + 1, j + 1]))
                k = k + 1

                # right-to-bottom connection
                self.Q[self.nodes[i, j + 1], k] = 1
                self.Q[self.nodes[i + 1, j], k] = -1
                self.connections.append((self.nodes[i, j + 1], self.nodes[i + 1, j]))
                k = k + 1

        for i in range(n1-1):
            #right connection
            self.Q[self.nodes[i, n2-1], k] = 1
            self.Q[self.nodes[i + 1, n2-1], k] = -1
            self.connections.append((self.nodes[i, n2-1], self.nodes[i + 1, n2-1]))
            k = k + 1

        for j in range(n2 - 1):
            #bottom connection
            self.Q[self.nodes[n1 - 1, j], k] = 1
            self.Q[self.nodes[n1 - 1, j + 1], k] = -1
            self.connections.append((self.nodes[n1 - 1, j], self.nodes[n1 - 1, j + 1]))
            k = k + 1

        self.m = k

        self.a=np.zeros(self.m)
        self.cminus = np.zeros(self.m)
        self.cplus = np.zeros(self.m)
        for i in range(self.m):
            self.a[i] = a_func(self.connections[i])
            self.cminus[i] = cminus_func(self.connections[i])
            self.cplus[i] = cplus_func(self.connections[i])

        #TODO: rho setup API
        #hardcoded rho

        self.rho = lambda xi, t: boundary_cond.rho(self, xi,t)
        self.d_xi_rho = lambda xi, t: boundary_cond.d_xi_rho(self, xi, t)
        self.d_t_rho = lambda xi, t: boundary_cond.d_t_rho(self, xi, t)
        self.f = lambda t: boundary_cond.f(self, t)

        self.e0 = np.zeros(self.m)

        self.q = boundary_cond.q(self)


    class HoldLeftStressRight(BoundaryConditions):
        rate = 0.1
        def rho(self, outer, xi, t):
            return xi[range(outer.d*outer.n2)]-outer.xi[range(outer.d*outer.n2)]

        def d_xi_rho(self, outer, xi, t):
            M = np.zeros((outer.d * outer.n2, outer.d * outer.n))
            for i in range(outer.d * outer.n2):
                M[i, i] = 1
            return M

        def d_t_rho(self,outer, xi, t):
            return np.zeros(outer.d* outer.n2)

        def f(self, outer, t):
            f = np.zeros(outer.d * outer.n)
            for i in range(outer.n2):
                f[outer.d * outer.n - 2 * i - 2] = self.rate * t  # pulling only x-component
            return f

        def q(self, outer):
            return outer.n2 * outer.d


    class HoldLeftDisplacementRight(BoundaryConditions):
        rate = 0.1
        def rho(self, outer, xi, t):
            tho = np.zeros(outer.d * outer.n2) #alteranting 1 and 0
            for i in range(outer.n2):
                tho[2*i] = 1

            return np.hstack((xi[range(outer.d*outer.n2)]-outer.xi[range(outer.d*outer.n2)],
                             xi[range(outer.d*outer.n - outer.d*outer.n2, outer.d*outer.n)]-outer.xi[range(outer.d*outer.n - outer.d*outer.n2, outer.d*outer.n)] - self.rate*t*tho))

        def d_xi_rho(self, outer, xi, t):
            M1 = np.zeros((outer.d * outer.n2, outer.d * outer.n))
            for i in range(outer.d * outer.n2):
                M1[i, i] = 1

            M2 = np.zeros((outer.d * outer.n2, outer.d * outer.n))
            for i in range(outer.d * outer.n2):
                M2[outer.d * outer.n2 - i - 1, outer.d * outer.n - i - 1] = 1
            return np.vstack((M1, M2))

        def d_t_rho(self,outer, xi, t):
            dtr = np.zeros(2*outer.d*outer.n2)
            for i in range(outer.n2):
                dtr[outer.d*outer.n2+2*i] = self.rate
            return dtr

        def f(self, outer, t):
            f = np.zeros(outer.d * outer.n)
            return f

        def q(self, outer):
            return 2*outer.n2 * outer.d

    def get_elastoplastic_process(self):
        return ElastoplasticProcess(self.Q, self.a, self.cminus, self.cplus, self.d, self.q, self.rho, self.d_xi_rho, self.d_t_rho, self.f)






