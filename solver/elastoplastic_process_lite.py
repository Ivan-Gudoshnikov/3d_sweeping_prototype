import sys

import numpy as np
import scipy
from solver.convex import Polytope, Box
from solver.quadprog_interface import McGibbonQuadprog

class Elastoplastic_process_lite:
    def __init__(self, Q, k, cminus, cplus, d,R, r,f, demand_enough_constraints = True):
        """
        :param Q: incidence matrix
        :param k: vector of stiffnesses
        :param cminus: vector of min stresses
        :param cplus: vector of max stresses
        :param d: amount of spatial dimensions
        :param R: matrix from additional constraint R(zeta-xi0) + r(t)=0
        :param r: function r(t) from the additional constraint
        :param f: external force applied at the nodes - function of t
        """

        self.Q = Q

        self.k=k
        self.K=np.diag(k)
        self.Kinv = np.diag(1 / k)

        # amount of springs
        self.m = Q.shape[1]
        # amount of nodes
        self.n = Q.shape[0]
        # spatial dimension
        self.d = d
        #amount of additional constraints
        self.q = R.shape[0]

        if R.shape[1]!=self.n*self.d:
            raise NameError("Wrong matrix dimension of R !")
        if np.linalg.matrix_rank(R)!=self.q:
            raise NameError("Excessive constraints in R !")

        self.R=R
        self.r=r
        self.f=f

        self.cminus = cminus
        self.cplus = cplus

        #explicit list of edges with endpoints from Q
        self.connections = []
        for i in range(self.m):
            spring_tuple = (np.where(self.Q[:, i] == 1)[0][0], np.where(self.Q[:, i] == -1)[0][0])
            self.connections.append(spring_tuple)

        self.demand_enough_constraints = demand_enough_constraints


    def phi(self, xi0):
        result = np.zeros(self.m)
        J = range(self.n)
        K = range(self.d)
        for i in range(self.m):
            result[i]=np.sqrt(np.sum([np.square(np.dot(xi0[[self.d * j + k for j in J]], self.Q[:, i])) for k in K]))
        return result


    def d_xi_phi(self, xi0):
        phi = self.phi(xi0)
        result = np.zeros((self.m, self.n*self.d))
        J = range(self.n)
        K = range(self.d)
        for i in range(self.m):
            for j in J:
                for k in K:
                    result[i, self.d*j+k] = np.dot(xi0[[self.d * j1 + k for j1 in J]], self.Q[:, i])*self.Q[j,i]/phi[i]
        return result

    def ubasis_vbasis(self, xi0):
        """
        Orthonormalized basis in U
        :param xi0:
        :return:
        """
        R0 = scipy.linalg.null_space(self.R)
        ubasis_candidate = self.d_xi_phi(xi0) @ R0
        if self.demand_enough_constraints:
            if np.linalg.matrix_rank(ubasis_candidate)<ubasis_candidate.shape[1]:  #if columns of result are not linearly independent, throw an error
                raise NameError("Not enough constraints!")

        ubasis = scipy.linalg.orth(ubasis_candidate)

        if np.linalg.matrix_rank(ubasis_candidate)!= np.linalg.matrix_rank(ubasis):
            raise NameError("Something is terribly wrong here")

        vbasis = scipy.linalg.null_space((self.K @ ubasis).T)
        if ubasis.shape[1]+vbasis.shape[1]!=self.m:
            raise NameError("Dimensions of U and V do not add up!")

        return ubasis, vbasis

    def PU_PV_in_coords(self, xi0):
        (U, V) = self.ubasis_vbasis(xi0)
        stacked = np.hstack((U, V))
        stacked_inv = np.linalg.inv(stacked)
        return stacked_inv[range(0, U.shape[1]), :], stacked_inv[range(U.shape[1], self.m), :]

    def g_vcoord_function(self,xi0):
        Rp=np.linalg.pinv(self.R)

        (PU_coords, PV_coords)= self.PU_PV_in_coords(xi0)
        return lambda t: PV_coords @ self.d_xi_phi(xi0) @ Rp @ self.r(t)

    def h_ucoord_function(self, xi0):
        # To compute H take the "upper" part of the pseudo-inverse matrix of the combined  matrix [(d_xi_phi)^T  R^T]
        H = np.linalg.pinv(np.hstack((self.d_xi_phi(xi0).T, self.R.T)))[range(0, self.m), :]
        (PU_coords, PV_coords) = self.PU_PV_in_coords(xi0)
        return lambda t: PU_coords @ self.Kinv @ H @ self.f(t)

    def C_moving_set_function(self,xi0):
        (ubasis,vbasis) = self.ubasis_vbasis(xi0)
        g_vcoord = self.g_vcoord_function(xi0)
        h_ucoord = self.h_ucoord_function(xi0)

        Aeq = (self.K @ ubasis).T #orthogonal constraints to V
        beq=np.zeros(Aeq.shape[0])
        A=np.vstack((np.identity(self.m), -np.identity(self.m)))

        return lambda t: Polytope(A,
                                  np.hstack((self.Kinv @ self.cplus + vbasis @ g_vcoord(t) + ubasis @ h_ucoord(t),
                                            -(self.Kinv @ self.cminus + vbasis @ g_vcoord(t) + ubasis @ h_ucoord(t)))),
                                  Aeq, beq)


    def solve_e_catch_up(self, e0, t0, dt, nsteps, xi0):
        T = np.zeros(nsteps + 1)
        Y = np.zeros((self.m, nsteps + 1))
        E = np.zeros((self.m, nsteps + 1))

        (ubasis, vbasis) = self.ubasis_vbasis(xi0)
        g_vcoord = self.g_vcoord_function(xi0)
        h_ucoord = self.h_ucoord_function(xi0)

        y0= e0 + vbasis @ g_vcoord (t0) - ubasis @ h_ucoord(t0)

        T[0] = t0
        Y[:, 0] = y0[:]
        E[:, 0] = e0[:]

        C_moving_set = self.C_moving_set_function(xi0)

        print("Starting the catch-up method with m = ", self.m, ", n = ", self.n, ", d = ", self.d, ", q = ", self.q, ", dim V = ", vbasis.shape[1])
        for i in range(nsteps):
            t_0 = T[i]
            y_0 = Y[:, i]
            t_1 = t_0+dt
            try:
                y_1 = C_moving_set(t_1).projection(self.K, y_0, McGibbonQuadprog())
            except ValueError:
                raise NameError("Can't perform a step number "+str(i))
            T[i+1] = t_1
            Y[:,i+1] = y_1
            E[:,i+1] = y_1 - vbasis @ g_vcoord (t_1) + ubasis @ h_ucoord(t_0)
            sys.stdout.write("\r Completed step "+str(i+1)+" of "+ str(nsteps))
            sys.stdout.flush()
        return T, E, Y





