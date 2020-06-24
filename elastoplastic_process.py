import sys

import numpy as np
import scipy
from convex import Polytope, Box
from quadprog_interface import QuadprogInterface, McGibbonQuadprog

def matrix_to_vector(matrix):
    """
    :param matrix: n x d matrix
    :return: vector ( [d x (1)]  [d x 2] ... [d x (n)])
    """
    (n, d) = matrix.shape
    result = np.zeros(n*d)
    for j in range(0,n):
        result[range(d*j, d*(j+1))] = matrix[j, :]
    return result

def vector_to_matrix(vector, d):
    """    
    :param vector: vector ( [d x (1)]  [d x 2] ... [d x (n)])
    :param d: 
    :return: matrix: n x d matrix
    """
    n = vector.shape[0] // d
    result = np.zeros((n, d))
    for j in range(0,n):
        result[j, :] = vector[range(d*j, d*(j+1))]
    return result

def tensor_to_matrix(t):
    """
    :param t: m x n x d numpy array
    :return: m x (nd) numpy array: ( t[m x (1) x d]  t[m x (2) x d] ... t[m x (n) x d])
    """
    (m, n, d) = t.shape
    result = np.zeros((m, n * d))
    for j in range(0, n):
        result[:, range(d * j, d * (j + 1))] = t[:, j, :]
    return result


def matrix_to_tensor(matrix, d):
    """
    :param matrix: m x (nd) numpy array: m x (nd) numpy array: ( t[m x (1) x d]  t[m x (2) x d] ... t[m x (n) x d])
    :param d: 3nd dimension size
    :return: m x n x d numpy array
    """
    m = matrix.shape[0]
    n = matrix.shape[1] // d
    result = np.zeros((m, n, d))
    for j in range(0, n):
        result[:, j, :] = matrix[:, range(d * j, d * (j + 1))]
    return result


class ElastoplasticProcess:
    def __init__(self, Q, a, cminus, cplus, d, q, rho, d_xi_rho, d_t_rho, f):
        # basic properties

        # incidence matrix
        self.Q = Q

        # amount of springs
        self.m = Q.shape[1]
        # amount of nodes
        self.n = Q.shape[0]
        # spatial dimension
        self.d = d
        #rnak of the additional constraint
        self.q = q

        self.A = np.diag(a)
        self.Ainv = np.diag(1/a)


        self.cminus = cminus
        self.cplus = cplus

        # function rho and its derivatives| rho: xi, t -> rho(xi,t)
        self.rho = rho
        self.d_xi_rho = d_xi_rho
        self.d_t_rho = d_t_rho

        #external forces
        self.f=f

        self.connections=[]
        for i in range(self.m):
            spring_tuple = (np.where(self.Q[:, i] == 1)[0][0], np.where(self.Q[:, i] == -1)[0][0])
            self.connections.append(spring_tuple)

        self.e_bounds_box=Box(np.matmul(self.Ainv,self.cminus), np.matmul(self.Ainv,self.cplus))


    def get_connections(self):
        return self.connections
    def get_Q(self):
        return self.Q
    def get_A(self):
        return self.A
    def get_Ainv(self):
        return self.Ainv
    def get_m(self):
        return self.m
    def get_n(self):
        return self.n
    def get_d(self):
        return self.d
    def get_q(self):
        return self.q
    def get_stress_bounds(self):
        return (self.cminus, self.cplus)
    def get_elastic_bounds(self):
        return (np.matmul(self.Ainv, self.cminus), np.matmul(self.Ainv, self.cplus))


    def phi(self, xi):
        """
        Lengths of springs
        :param xi:
        :return:
        """
        xi_mat=vector_to_matrix(xi,self.d)
        return np.sqrt(np.sum(np.square(np.matmul(self.Q.T, xi_mat)), axis=1))

    def K(self, xi):
        """
        Normalized directions of springs
        :param xi as a vector
        :return: m x d matrix
        """
        xi_mat = vector_to_matrix(xi, self.d)
        return np.divide(
            np.matmul(self.Q.T, xi_mat),
            np.tile(self.phi(xi), (self.d, 1)).T
        )

    def d_xi_phi(self, xi):
        """
        :param xi:
        :return: m x (nd) array representing D_xi phi
        """

        Q1 = np.expand_dims(self.Q.T, axis=2)
        Q2 = np.tile(Q1, (1, 1, self.d))

        N1 = np.swapaxes(np.expand_dims(self.K(xi), axis=2), 1, 2)
        N2 = np.tile(N1, (1, self.n, 1))

        return tensor_to_matrix(np.multiply(N2, Q2))

    def ker_d_xi_rho(self, xi, t):
        """
        basis in the nullspace of d_xi_rho
        :param xi:
        :param t:
        :return:  (nd)x(dim Ker d_xi_rho) array
        """
        return scipy.linalg.null_space(self.d_xi_rho(xi, t))

    def ker_d_xi_phi(self, xi):
        """
        basis in the nullspace of d_xi_phi
        :param xi:
        :return: (nd)x(dim Ker d_xi_phi) array
        """
        return scipy.linalg.null_space(self.d_xi_phi(xi))

    def H(self,xi,t):
        """
        :param xi:
        :param t:
        :return: external force term minus the reactions of rho
        """
        # Take the "upper" part of the pseudo-inverse matrix of the combined  matrix [(d_xi_phi)^T  (d_xi_rho)^T], which is of full row rank

        M1 = self.d_xi_phi(xi).T
        M2 = self.d_xi_rho(xi, t).T
        return np.linalg.pinv(np.hstack((M1,M2)))[range(0,self.m), :]


    def dim_intersection_nullspaces(self, xi):
        """
        should always be 0!
        :param xi:
        :return:
        """
        return self.n*self.d- np.linalg.matrix_rank(np.vstack((self.d_xi_rho(xi, 0), self.d_xi_phi(xi))))

    def u_basis(self,xi,t):
        """
        A very spceific, generally not orthogonal, basis in U, such that
        the coordinates of u\in U  in it are the same as the coordinates in L^{-1}u in the ker_d_xi_rho basis
        :param xi:
        :param t:
        :return:
        """
        result = np.matmul(self.d_xi_phi(xi), self.ker_d_xi_rho(xi, t))
        if np.linalg.matrix_rank(result) != self.n*self.d - self.q:
            raise NameError("Constraint rho is not enough for that phi(xi)")
        return result

    def v_basis(self, xi, t):
        return scipy.linalg.null_space(np.matmul(self.A, self.u_basis(xi,t)).T)


    def R(self,xi,t):
        return np.linalg.pinv(self.d_xi_rho(xi,t))

    def p_u_coords(self,xi,t):
        """
        Projection matrix on U(xi,t) orthogonal to V(xi,t), in terms of coordinates in u_basis
        Inefficient as a separate calculation, but this is a prototype
        :param xi:
        :param t:
        :return:
        """
        M1=self.u_basis(xi, t)
        M2=self.v_basis(xi, t)

        inv= np.linalg.inv(np.hstack((M1,M2)))
        return inv[range(0, M1.shape[1]),:]



    def p_v_coords(self, xi, t):
        """
        Projection matrix on V(xi,t) orthogonal to U(xi,t), in terms of coordinates in terms of v_basis
        :param xi:
        :param t:
        :return:
        """
        M1=self.u_basis(xi, t)
        M2=self.v_basis(xi, t)

        inv= np.linalg.inv(np.hstack((M1,M2)))
        return inv[range(M1.shape[1],self.m),:]

    def g_v_coords(self,xi,t):
        """
        Function g expressed in the coordinates of v_basis
        :param xi:
        :param t:
        :return:
        """
        return np.matmul(np.matmul(np.matmul(self.p_v_coords(xi,t), self.d_xi_phi(xi)), self.R(xi,t)), self.d_t_rho(xi,t))

    def h_u_coords(self,xi,t, fval):
        """
        Function h expressed in the coordinates of u_basis
        :param xi:
        :param t:
        :param fval: external forces vector of the size (nd)
        :return:
        """
        return np.matmul(np.matmul(np.matmul(self.p_u_coords(xi,t), self.Ainv),self.H(xi,t)), fval)

    def v_orth(self,xi,t):
        """
        Set of constraints which define V
        :param xi:
        :param t:
        :return:
        """
        return scipy.linalg.null_space(self.v_basis(xi,t).T).T

    def moving_set(self, xi, t):
        """
        :param xi:
        :param t:
        :return:
        """
        A = np.vstack((self.A, -self.A))
        b = np.hstack((self.cplus, -self.cminus))
        #Aeq = self.v_orth(xi, t)
        #beq = - np.matmul(Aeq, np.matmul(self.u_basis(xi, t), self.h_u_coords(xi, t)))

        Aeq = self.p_u_coords(xi, t)
        beq = self.h_u_coords(xi, t, self.f(t))

        return Polytope(A, b, Aeq, beq)

    def dot_xi_and_p(self, xi, t, e, dot_e):
        """
        part of the formula for dot xi, whic gives u - component
        :param xi:
        :param t:
        :param e:
        :param dot_e:
        :return: (dot_xi, cone , dot_p_cone_coords),
        cone - active normals to the box
        dot_p_cone_coords is the decomposition of dot_p in terms the active normals

        """
        N = self.e_bounds_box.normal_cone(self.A, e).N

        if N is not None:
            u_basis = self.u_basis(xi,t)
            Aeq = np.hstack((u_basis, -N))
            beq = dot_e + np.matmul(np.matmul(self.d_xi_phi(xi), self.R(xi,t)), self.d_t_rho(xi,t))
            l_size = N.shape[1]
            dim_u = u_basis.shape[1]
            A = np.hstack((np.zeros((l_size,dim_u)), - np.identity(l_size)))
            b = np.zeros(l_size)

            #TODO: alternative ways to find \dot xi from its constraints?\
            param_set = Polytope(A,b, Aeq,beq)

            u_and_l_params = param_set.projection(np.identity(dim_u+l_size), np.zeros(dim_u+l_size), McGibbonQuadprog())
            #u_and_l_params = param_set.linprog(np.hstack((np.zeros(dim_u), np.ones(l_size)))) #linprog is actually more strict

            u_coords = u_and_l_params[range(0, dim_u)]
            dot_p_cone_coords = u_and_l_params[range(dim_u,dim_u+l_size)]
        else:
            #if dot p = 0 we have  - dot x = - dot e in U + np.matmul(np.matmul(self.d_xi_phi(xi), self.R(xi,t)), self.d_t_rho(xi,t)) due to sweeping process
            # take projection to fund u-coords
            u_coords = np.matmul(self.p_u_coords(xi, t), dot_e + np.matmul(np.matmul(self.d_xi_phi(xi), self.R(xi,t)), self.d_t_rho(xi,t)))
            dot_p_cone_coords = None
        return np.matmul(self.ker_d_xi_rho(xi, t), u_coords) - np.matmul(self.R(xi,t), self.d_t_rho(xi,t)), N, dot_p_cone_coords

    def solve_system_step(self, xi_0, e_0, t_0, dt):
        t_1 = t_0+dt
        #see Tahar Haddad. Differential Inclusion Governed by a State Dependent Sweeping Process
        #International Journal of Difference Equations, Volume 8, Number 1, pp. 63–70 (2013)
        e_1 = self.moving_set(xi_0, t_1, ).projection(self.A,
                                                    e_0 - dt*np.matmul(self.v_basis(xi_0,t_0), self.g_v_coords(xi_0,t_0)),
                                                    McGibbonQuadprog())
        dot_e = (e_1-e_0)/dt
        (dot_xi, cone, dot_p_cone_coords) = self.dot_xi_and_p(xi_0, t_0, e_0, dot_e)
        xi_1 = xi_0 + dt*dot_xi
        return xi_1, e_1, cone, dot_p_cone_coords

    def solve(self, xi0,e0,t0, dt, nsteps):
        print("Starting the problem with m =", self.m, ", n =", self.n, ", d =", self.d, ", q = ", self.q)
        T =  np.zeros(nsteps+1)
        XI = np.zeros((self.n*self.d, nsteps+1))
        E = np.zeros((self.m, nsteps + 1))
        X = np.zeros((self.m, nsteps + 1))
        P = np.zeros((self.m, nsteps + 1))
        N=[]
        DOT_P_CONE_COORDS=[]

        T[0] = t0
        XI[:,0] = xi0[:]
        E[:,0] = e0[:]
        X[:,0] = self.phi(xi0)
        P[:,0] = X[:,0] - E[:,0]

        for i in range(nsteps):
            t_0  = T[i]
            xi_0 = XI[:,i]
            e_0  = E[:,i]

            t_1=t_0+dt
            try:
                (xi_1,e_1,cone,  dot_p_cone_coords)=self.solve_system_step(xi_0, e_0, t_0, dt)
            except ValueError:
                raise NameError("Can't perform a step number "+str(i))
            T[i+1] = t_1
            XI[:,i+1] = xi_1
            E[:,i+1] = e_1
            X[:,i+1] = self.phi(xi_1)
            P[:, i+1] = X[:,i+1] - E[:,i+1]
            N.append(cone)
            DOT_P_CONE_COORDS.append(dot_p_cone_coords)

            sys.stdout.write("\r Completed step "+str(i+1)+" of "+ str(nsteps))
            sys.stdout.flush()
        return T, XI, E, X, P, N, DOT_P_CONE_COORDS

















