import sys

import numpy as np
import scipy
from solver.convex import Polytope, Box
from solver.quadprog_interface import McGibbonQuadprog


def phi(Q, xi, d):
    n=Q.shape[0]
    m=Q.shape[1]
    result = np.zeros(m)
    J = range(n)
    K = range(d)
    for i in range(m):
        result[i] = np.sqrt(np.sum([np.square(np.dot(xi[[d * j + k for j in J]], Q[:, i])) for k in K]))
    return result

def d_xi_phi(Q, xi0,d):
    n = Q.shape[0]
    m = Q.shape[1]
    phi0 = phi(Q,xi0,d)
    result = np.zeros((m, n*d))
    J = range(n)
    K = range(d)
    for i in range(m):
        for j in J:
            for k in K:
                result[i, d*j+k] = np.dot(xi0[[d * j1 + k for j1 in J]], Q[:, i])*Q[j,i]/phi0[i]
    return result

class Elastoplastic_process_linearized:
    def __init__(self,  Q, xi0, k, cminus, cplus, d,R, r,f,r_prime = None, demand_enough_constraints = True):
        """
        :param Q: incidence matrix
        :param k: vector of stiffnesses
        :param cminus: vector of min stresses
        :param cplus: vector of max stresses
        :param d: amount of spatial dimensions
        :param R: matrix from additional constraint R(zeta-xi0) + r(t)=0
        :param r: function r(t) from the additional constraint
        :param f: external force applied at the nodes - function of t
        :param r_prime: fixed vector for leapfrog algorithm. None if non-constant and the leapfrog won't be used
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
            raise NameError("Dependent constraints in R !")

        self.R=R
        self.r=r
        self.r_prime=r_prime
        self.f=f

        self.cminus = cminus
        self.cplus = cplus
        self.e_bounds_box = Box(np.matmul(self.Kinv, self.cminus), np.matmul(self.Kinv, self.cplus))

        #explicit list of edges with endpoints, computed from Q
        self.connections = []
        for i in range(self.m):
            spring_tuple = (np.where(self.Q[:, i] == 1)[0][0], np.where(self.Q[:, i] == -1)[0][0])
            self.connections.append(spring_tuple)

        self.demand_enough_constraints = demand_enough_constraints

        self.xi0=xi0

        self.d_xi_phi = d_xi_phi(Q, xi0, d)

        self.R0 = scipy.linalg.null_space(self.R)
        self.Rp = np.linalg.pinv(self.R)

        # To compute H take the "upper" part of the pseudo-inverse matrix of the combined  matrix [(d_xi_phi)^T  R^T]
        self.H = np.linalg.pinv(np.hstack((self.d_xi_phi.T, self.R.T)))[range(0, self.m), :]

        #COMPUTING ubasis and vbasis:

        ubasis_candidate = self.d_xi_phi @ self.R0
        if self.demand_enough_constraints:
            if np.linalg.matrix_rank(ubasis_candidate) < ubasis_candidate.shape[1]:  # if columns of result are not linearly independent, throw an error
                raise NameError("Not enough constraints!")
        #ubasis = scipy.linalg.orth(ubasis_candidate)
        self.ubasis = ubasis_candidate
        self.vbasis = scipy.linalg.null_space((self.K @ self.ubasis).T)
        if self.ubasis.shape[1] + self.vbasis.shape[1] != self.m:
            raise NameError("Dimensions of U and V do not add up!")

        print("Pre-computations are complete.")
        print("m = ", self.m, ", n = ", self.n, ", d = ", self.d, ", q = ", self.q)
        print("dim U = ", self.ubasis.shape[1], " dim V = ", self.vbasis.shape[1])

        #COMPUTING P_U, P_V in coodinates of U and V:

        stacked_inv = np.linalg.inv(np.hstack((self.ubasis, self.vbasis)))
        self.P_U_coords = stacked_inv[range(0, self.ubasis.shape[1]), :]
        self.P_V_coords = stacked_inv[range(self.ubasis.shape[1], self.m), :]

        #pre-computing values for the sweeping process
        self.G_V = self.P_V_coords @ self.d_xi_phi @ self.Rp
        self.G = self.vbasis @ self.G_V

        self.F = self.ubasis @ self.P_U_coords @ self.Kinv @ self.H

        #values for the process in the coordinate space of V
        self.S = self.vbasis.T @ self.K @ self.vbasis
        self.normals_in_V = (self.P_V_coords @ self.Kinv).T @ self.S # normal vectors tho the moving set in the coordinate space of V, (in sence of standard dot product)


    def solve_e_catch_up(self, e0, t0, dt, nsteps):
        """
        Time-stepping catch-up in R^m
        :param e0:
        :param t0:
        :param dt:
        :param nsteps:
        :return:
        """
        print("Starting the catch-up method.")

        T = np.zeros(nsteps + 1)
        Y = np.zeros((self.m, nsteps + 1))

        #pre-calculations for the moving set
        Aeq = (self.K @ self.ubasis).T  # orthogonal constraints to V
        beq = np.zeros(Aeq.shape[0])
        A = np.vstack((np.identity(self.m), -np.identity(self.m)))

        eplusminus = np.hstack((self.Kinv @ self.cplus, -self.Kinv @ self.cminus))

        y0= e0 + self.G @ self.r(t0) - self.F @ self.f(t0)

        T[0] = t0
        Y[:, 0] = y0[:]

        for i in range(nsteps):
            t_0 = T[i]
            y_0 = Y[:, i]
            t_1 = t_0+dt

            box_offset_1= self.G @ self.r(t_1) - self.F @ self.f(t_1)

            moving_set_1 = Polytope(A,
                                  eplusminus + np.hstack((box_offset_1, -box_offset_1)),
                                  Aeq, beq)
            try:
                y_1 = moving_set_1.projection(self.K, y_0, McGibbonQuadprog())
            except ValueError:
                raise NameError("Can't perform a step number "+str(i))
            T[i+1] = t_1
            Y[:,i+1] = y_1

            sys.stdout.write("\r Completed step "+str(i+1)+" of "+ str(nsteps))
            sys.stdout.flush()
        print("")

        E = np.zeros((self.m, nsteps + 1))
        Sigma = np.zeros((self.m, nsteps + 1)) #stresses
        Rho = np.zeros((self.q, nsteps + 1)) #reaction force values

        for i in range(nsteps+1): #can be vectorized, provoded that r and f are vectorized
            E[:, i] = Y[i] - self.G @ self.r(T[i]) + self.F @ self.f(T[i])
            Sigma[:, i] = (self.K @ E[:, i])[:]
            Rho[:, i] = self.Rp.T @ self.d_xi_phi.T @ (-Sigma[:, i] + self.H @ self.f(T[i]))

        return T, E, Y, Sigma, Rho

    def solve_e_in_V_catch_up(self, e0, t0, dt, nsteps):
        """
        Time-stepping catch-up in the coordinate space of V, much faster then in R^m
        :param e0:
        :param t0:
        :param dt:
        :param nsteps:
        :return:
        """
        print("Starting the catch-up method in coordinate space of V")
        T = np.zeros(nsteps + 1)
        Y_V= np.zeros((self.vbasis.shape[1], nsteps + 1))

        eplusminus = np.hstack((self.Kinv @ self.cplus, -self.Kinv @ self.cminus))

        A=np.vstack((self.normals_in_V, -self.normals_in_V))

        y_v0= self.P_V_coords @ e0 + self.G_V @ self.r(t0)

        T[0] = t0
        Y_V[:,0]=y_v0[:]

        for i in range(nsteps):
            t_0 = T[i]
            y_v_0 = Y_V[:, i]
            t_1 = t_0+dt

            box_offset_1= self.G @ self.r(t_1) - self.F @ self.f(t_1)

            moving_set_1 = Polytope(A,
                                  eplusminus + np.hstack((box_offset_1, -box_offset_1)),
                                  None, None)
            try:
                y_v_1 = moving_set_1.projection(self.S, y_v_0, McGibbonQuadprog())
            except ValueError:
                raise NameError("Can't perform a step number "+str(i))
            T[i+1] = t_1
            Y_V[:,i+1] = y_v_1
            sys.stdout.write("\r Completed step "+str(i+1)+" of "+ str(nsteps))
            sys.stdout.flush()
        print("")

        E = np.zeros((self.m, nsteps + 1))
        Sigma = np.zeros((self.m, nsteps + 1)) #stresses
        Rho = np.zeros((self.q, nsteps + 1)) #reaction force values

        for i in range(nsteps+1): #can be vectorized, provoded that r and f are vectorized
            E[:, i] = self.vbasis @ Y_V[:,i] - self.G @ self.r(T[i]) + self.F @ self.f(T[i])
            Sigma[:, i] = (self.K @ E[:, i])[:]
            Rho[:, i] = self.Rp.T @ self.d_xi_phi.T @ (-Sigma[:, i] + self.H @ self.f(T[i]))

        return T, E, Y_V, Sigma, Rho

    def solve_e_in_V_leapfrog(self, e0, t0, eps):
        """
        Event-based "leapfrog" algorithm, (so far) implemented under the assumption that f(0)=f(t)=const and r'(0)=r'(t) = const for all t
        :param e0:
        :param t0:
        :return:
        """
        print("Starting the event-based \"leapfrog\" method in coordinate space of V")
        T = np.zeros(1)
        E_V=np.zeros((self.vbasis.shape[1],1))

        T[0] = t0
        E_V[:,0] = self.P_V_coords @ e0

        A = np.vstack((self.normals_in_V, -self.normals_in_V))
        eplusminus = np.hstack((self.Kinv @ self.cplus, -self.Kinv @ self.cminus))
        default_direction = - self.G_V @ self.r_prime
        #assuming f=const
        box_offset_by_force =  - self.F @ self.f(t0)
        #C_V (t) = moving_set_shape + G_V @ self.r(t)
        moving_set_shape = Polytope(A,
                                  eplusminus + np.hstack((box_offset_by_force , -box_offset_by_force)),
                                  None, None)
        i = 0
        while True:
            e_v_0=E_V[:,i]
            tcone = moving_set_shape.tangent_cone(e_v_0)
            direction = tcone.projection(self.S, default_direction, McGibbonQuadprog())
            if np.linalg.norm(direction) <eps:
                break
            t_1, e_v_1 = moving_set_shape.first_intersection_with_boundary(e_v_0, direction)
            T = np.append(T, t_1+T[i])
            E_V = np.append(E_V,np.expand_dims(e_v_1,1), 1)
            i = i + 1
            sys.stdout.write("\r Completed step " + str(i))
            sys.stdout.flush()

        print("")
        nsteps = i
        E = np.zeros((self.m, nsteps+1))
        Sigma = np.zeros((self.m, nsteps+1))
        Rho = np.zeros((self.q, nsteps+1))
        for i in range(nsteps+1):
            E[:,i] = self.vbasis @ E_V[:,i] - box_offset_by_force
            Sigma[:, i] = (self.K @ E[:, i])[:]
            Rho[:, i] = self.Rp.T @ self.d_xi_phi.T @ (-Sigma[:, i] + self.H @ self.f(T[i]))

        return T, E, E_V, Sigma, Rho


















