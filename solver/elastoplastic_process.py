import sys

import numpy as np
import scipy
from solver.convex import Polytope, Box
from solver.quadprog_interface import McGibbonQuadprog

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
        """
        :param Q: incidence matrix
        :param a: vector of stiffness values
        :param cminus: vector of min stresses
        :param cplus: vector of max stresses
        :param d: amount of spatial dimensions
        :param q: amount of rho-constraints
        :param rho: rho-constraint(displacement boundary condition) - a function of xi,t
        :param d_xi_rho: Jacobi matrix of rho-constraint w.r.to xi  - a function of xi,t
        :param d_t_rho: time-derivative of rho-constraint - a fucntion of xi, t with values -  vectos of leg q
        :param f: external force applied at the nodes - function of t
        """
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

    def ker_d_xi_rho(self, d_xi_rho):
        """
        basis in the nullspace of d_xi_rho
        :param d_xi_rho: value of the derivative
        :return:  (nd)x(dim Ker d_xi_rho) array
        """
        return scipy.linalg.null_space(d_xi_rho)

    def ker_d_xi_phi(self, d_xi_phi):
        """
        basis in the nullspace of d_xi_phi
        :param d_xi_phi: value of the derivaltive
        :return: (nd)x(dim Ker d_xi_phi) array
        """
        return scipy.linalg.null_space(d_xi_phi)

    def H(self, d_xi_phi, d_xi_rho):
        """
        :param d_xi_phi
        :param d_xi_rho
        :return: external force term minus the reactions of rho
        """
        # Take the "upper" part of the pseudo-inverse matrix of the combined  matrix [(d_xi_phi)^T  (d_xi_rho)^T], which is of full row rank

        M1 = d_xi_phi.T
        M2 = d_xi_rho.T
        return np.linalg.pinv(np.hstack((M1,M2)))[range(0,self.m), :]


    def dim_intersection_nullspaces(self, d_xi_phi, d_xi_rho):
        """
        should always be 0!
        :param d_xi_phi
        :param d_xi_rho
        :return:
        """
        return self.n*self.d - np.linalg.matrix_rank(np.vstack((d_xi_rho, d_xi_phi)))

    def u_basis(self, d_xi_phi, d_xi_rho):
        """
        A very spceific, generally not orthogonal, basis in U, such that
        the coordinates of u\in U  in it are the same as the coordinates in L^{-1}u in the ker_d_xi_rho basis
        :param d_xi_phi
        :param d_xi_rho
        :return:
        """
        result = np.matmul(d_xi_phi, self.ker_d_xi_rho(d_xi_rho))
        if np.linalg.matrix_rank(result) != self.n*self.d - self.q:
            raise NameError("Constraint rho is not enough for that phi(xi)")
        return result

    def v_basis(self, d_xi_phi, d_xi_rho):
        return scipy.linalg.null_space(np.matmul(self.A, self.u_basis(d_xi_phi, d_xi_rho)).T)


    def R(self, d_xi_rho):
        return np.linalg.pinv(d_xi_rho)

    def p_u_and_p_v_coords(self,d_xi_phi, d_xi_rho):
        """
        To compute together is more efficient
        :param d_xi_phi:
        :param d_xi_rho:
        :return:
        """
        M1 = self.u_basis(d_xi_phi, d_xi_rho)
        M2 = self.v_basis(d_xi_phi, d_xi_rho)
        inv = np.linalg.inv(np.hstack((M1, M2)))
        return inv[range(0, M1.shape[1]), :], inv[range(M1.shape[1],self.m),:]

    def p_u_coords(self, d_xi_phi, d_xi_rho):
        """
        Projection matrix on U(xi,t) orthogonal to V(xi,t), in terms of coordinates in u_basis
        Inefficient as a separate calculation if both p_u, p_v are computed
        :param d_xi_phi
        :param d_xi_rho
        :return:
        """
        return self.p_u_and_p_v_coords(d_xi_phi,d_xi_rho)[0]

    def p_v_coords(self, d_xi_phi, d_xi_rho):
        """
        Projection matrix on V(xi,t) orthogonal to U(xi,t), in terms of coordinates in terms of v_basis
        Inefficient as a separate calculation if both p_u, p_v are computed
        :param d_xi_phi
        :param d_xi_rho
        :return:
        """
        return self.p_u_and_p_v_coords(d_xi_phi,d_xi_rho)[1]

    def g_v_coords(self, p_v_coords, d_xi_phi, R, d_t_rho):
        """
        Function g expressed in the coordinates of v_basis
        The arguments are like this to allow computation without repetitions
        :param p_v_coords:
        :param d_xi_phi:
        :param R
        :param d_t_rho
        :return:
        """

        return np.matmul(np.matmul(np.matmul(p_v_coords, d_xi_phi), R), d_t_rho)

    def h_u_coords(self, p_u_coords, H, fval):
        """
        Function h expressed in the coordinates of u_basis
        The arguments are like this to allow computation without repetitions
        :param p_u_coords
        :param H
        :param fval: external forces vector of the size (nd)
        :return:
        """
        return np.matmul(np.matmul(np.matmul(p_u_coords, self.Ainv), H), fval)

    def moving_set(self, p_u_coords, h_u_coords):
        """
        :param xi:
        :param t:
        :return:
        """
        A = np.vstack((self.A, -self.A))
        b = np.hstack((self.cplus, -self.cminus))

        Aeq = p_u_coords
        beq = h_u_coords

        return Polytope(A, b, Aeq, beq)

    def dot_xi_and_p(self, e, dot_e, d_xi_phi, R, d_t_rho, u_basis, ker_d_xi_rho, p_u_coords):
        """
        part of the formula for dot xi, which gives u - component
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
            Aeq = np.hstack((u_basis, -N))
            beq = dot_e + np.matmul(np.matmul(d_xi_phi, R,), d_t_rho)
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
            u_coords = np.matmul(p_u_coords, dot_e + np.matmul(np.matmul(d_xi_phi, R), d_t_rho))
            dot_p_cone_coords = None
        return np.matmul(ker_d_xi_rho, u_coords) - np.matmul(R, d_t_rho), N, dot_p_cone_coords

    def solve_system_step_naive(self, xi_0, e_0, t_0, dt):
        t_1 = t_0+dt
        #see Tahar Haddad. Differential Inclusion Governed by a State Dependent Sweeping Process
        #International Journal of Difference Equations, Volume 8, Number 1, pp. 63–70 (2013)


        d_xi_phi_0 = self.d_xi_phi(xi_0)
        d_xi_rho_0_0 = self.d_xi_rho(xi_0, t_0)
        d_xi_rho_0_1 = self.d_xi_rho(xi_0, t_1)
        p_u_coords_0_1 = self.p_u_coords(d_xi_phi_0, d_xi_rho_0_1)

        H_0_1 = self.H(d_xi_phi_0,d_xi_rho_0_1)
        h_u_coords_0_1 = self.h_u_coords(p_u_coords_0_1,H_0_1, self.f(t_1))

        moving_set = self.moving_set(p_u_coords_0_1, h_u_coords_0_1)

        v_basis_0_0 = self.v_basis(d_xi_phi_0, d_xi_rho_0_0)
        R_0_0 = self.R(d_xi_rho_0_0)
        p_v_coords_0_0 = self.p_v_coords(d_xi_phi_0, d_xi_rho_0_0)
        d_t_rho_0 = self.d_t_rho(xi_0, t_0)
        g_v_coords_0_0 = self.g_v_coords(p_v_coords_0_0,d_xi_phi_0,R_0_0,d_t_rho_0)

        e_1 = moving_set.projection(self.A, e_0 - dt*np.matmul(v_basis_0_0, g_v_coords_0_0), McGibbonQuadprog())
        #Old version:
        #e_1 = self.moving_set(xi_0, t_1).projection(self.A,
        #                                            e_0 - dt*np.matmul(self.v_basis(xi_0,t_0), self.g_v_coords(xi_0,t_0)),
        #                                            McGibbonQuadprog())
        dot_e = (e_1-e_0)/dt
        #Old version:
        #(dot_xi, cone, dot_p_cone_coords) = self.dot_xi_and_p(xi_0, t_0, e_0, dot_e)
        u_basis_0_0 = self.u_basis(d_xi_phi_0,d_xi_rho_0_0)
        ker_d_xi_rho_0_0 = self.ker_d_xi_rho(d_xi_rho_0_0)
        p_u_coords_0_0 = self.p_u_coords(d_xi_phi_0, d_xi_rho_0_0)

        (dot_xi, cone, dot_p_cone_coords) = self.dot_xi_and_p(e_0, dot_e, d_xi_phi_0, R_0_0, d_t_rho_0, u_basis_0_0,
                                                              ker_d_xi_rho_0_0, p_u_coords_0_0)

        xi_1 = xi_0 + dt*dot_xi
        return xi_1, e_1, cone, dot_p_cone_coords

    def solve_system_step_naive_reference(self, xi_0, e_0, t_0, dt, xi_ref, d_xi_phi, d_xi_rho, p_u_coords, p_v_coords, u_basis, v_basis, H, R, ker_d_xi_rho):
        t_1 = t_0 + dt
        h_u_coords=self.h_u_coords(p_u_coords,H, self.f(t_1))
        moving_set = self.moving_set(p_u_coords, h_u_coords)
        d_t_rho = self.d_t_rho(xi_ref, t_0)
        g_v_coords = self.g_v_coords(p_v_coords, d_xi_phi, R, d_t_rho)

        e_1 = moving_set.projection(self.A, e_0 - dt * np.matmul(v_basis, g_v_coords), McGibbonQuadprog())
        dot_e = (e_1 - e_0) / dt
        (dot_xi, cone, dot_p_cone_coords) = self.dot_xi_and_p(e_0, dot_e, d_xi_phi, R, d_t_rho, u_basis, ker_d_xi_rho, p_u_coords)
        xi_1 = xi_0 + dt * dot_xi

        return xi_1, e_1, cone, dot_p_cone_coords


    def solve_fixed_spaces_e_only(self, xi0, e0,t0, dt, nsteps, xi_ref, t_ref):
        print("Starting the fixed spaces problem with m =", self.m, ", n =", self.n, ", d =", self.d, ", q = ", self.q)
        T = np.zeros(nsteps + 1)
        E = np.zeros((self.m, nsteps + 1))
        T[0] = t0
        E[:, 0] = e0[:]

        d_xi_phi = self.d_xi_phi(xi_ref)
        d_xi_rho = self.d_xi_rho(xi_ref, t_ref)
        [p_u_coords, p_v_coords] = self.p_u_and_p_v_coords(d_xi_phi, d_xi_rho)
        u_basis = self.u_basis(d_xi_phi, d_xi_rho)
        v_basis = self.v_basis(d_xi_phi, d_xi_rho)
        print("dim V = ", np.linalg.matrix_rank(v_basis))
        H = self.H(d_xi_phi, d_xi_rho)
        R = self.R(d_xi_rho)
        ker_d_xi_rho = self.ker_d_xi_rho(d_xi_rho)

        for i in range(nsteps):
            t_0  = T[i]
            e_0  = E[:, i]
            t_1=t_0+dt

            try:
                h_u_coords = self.h_u_coords(p_u_coords, H, self.f(t_1))
                moving_set = self.moving_set(p_u_coords, h_u_coords)
                d_t_rho = self.d_t_rho(xi_ref, t_0)
                g_v_coords = self.g_v_coords(p_v_coords, d_xi_phi, R, d_t_rho)
                e_1 = moving_set.projection(self.A, e_0 - dt * np.matmul(v_basis, g_v_coords), McGibbonQuadprog())
            except ValueError:
                raise NameError("Can't perform a step number "+str(i))
            T[i+1] = t_1
            E[:,i+1] = e_1
            sys.stdout.write("\r Completed step "+str(i+1)+" of "+ str(nsteps))
            sys.stdout.flush()
        return T, E

    def leapfrog_step(self, e0, t0,  xi_ref, t_ref):
        d_xi_phi = self.d_xi_phi(xi_ref)
        d_xi_rho = self.d_xi_rho(xi_ref, t_ref)
        [p_u_coords, p_v_coords] = self.p_u_and_p_v_coords(d_xi_phi, d_xi_rho)
        u_basis = self.u_basis(d_xi_phi, d_xi_rho)
        v_basis = self.v_basis(d_xi_phi, d_xi_rho)
        print("dim V = ", np.linalg.matrix_rank(v_basis))
        H = self.H(d_xi_phi, d_xi_rho)
        R = self.R(d_xi_rho)
        h_u_coords = self.h_u_coords(p_u_coords, H, self.f(t_ref))
        moving_set = self.moving_set(p_u_coords, h_u_coords)
        d_t_rho = self.d_t_rho(xi_ref, t_ref)
        g_v_coords = self.g_v_coords(p_v_coords, d_xi_phi, R, d_t_rho)
        tcone = moving_set.tangent_cone(e0)
        direction = tcone.projection(self.A, -np.matmul(v_basis, g_v_coords), McGibbonQuadprog())
        return moving_set.first_intersection_with_boundary(e0, direction)




    def solve_fixed_spaces(self, xi0,e0,t0, dt, nsteps, xi_ref, t_ref):
        """
        :param xi0:
        :param e0:
        :param t0:
        :param dt:
        :param nsteps:
        :param xi_ref:
        :param t_ref:
        :return:
        """
        print("Starting the fixed spaces problem with m =", self.m, ", n =", self.n, ", d =", self.d, ", q = ", self.q)
        T =  np.zeros(nsteps+1)
        XI = np.zeros((self.n*self.d, nsteps+1))
        E = np.zeros((self.m, nsteps + 1))
        X = np.zeros((self.m, nsteps + 1))
        P = np.zeros((self.m, nsteps + 1))
        P_ALT=np.zeros((self.m, nsteps + 1))
        N=[]
        DOT_P_CONE_COORDS=[]

        T[0] = t0
        XI[:,0] = xi0[:]
        E[:,0] = e0[:]
        X[:,0] = self.phi(xi0)
        P[:,0] = X[:,0] - E[:,0]
        P_ALT[:,0] = X[:,0] - E[:,0]



        d_xi_phi = self.d_xi_phi(xi_ref)
        d_xi_rho = self.d_xi_rho(xi_ref, t_ref)
        [p_u_coords, p_v_coords] = self.p_u_and_p_v_coords(d_xi_phi, d_xi_rho)
        u_basis = self.u_basis(d_xi_phi, d_xi_rho)
        v_basis = self.v_basis(d_xi_phi, d_xi_rho)
        H = self.H(d_xi_phi, d_xi_rho)
        R = self.R(d_xi_rho)
        ker_d_xi_rho = self.ker_d_xi_rho(d_xi_rho)

        for i in range(nsteps):
            t_0  = T[i]
            xi_0 = XI[:, i]
            e_0  = E[:, i]
            t_1=t_0+dt

            try:
                (xi_1, e_1, cone, dot_p_cone_coords) = self.solve_system_step_naive_reference(xi_0, e_0, t_0, dt, xi_ref, d_xi_phi, d_xi_rho, p_u_coords, p_v_coords, u_basis, v_basis, H, R, ker_d_xi_rho)
            except ValueError:
                raise NameError("Can't perform a step number "+str(i))
            T[i+1] = t_1
            XI[:,i+1] = xi_1
            E[:,i+1] = e_1
            X[:,i+1] = self.phi(xi_1)
            P[:, i+1] = X[:,i+1] - E[:,i+1]
            N.append(cone)
            DOT_P_CONE_COORDS.append(dot_p_cone_coords)
            if cone is None:
                P_ALT[:, i+1] = P_ALT[:,i]
            else:
                P_ALT[:, i+1] = P_ALT[:,i] + dt*np.matmul(cone, dot_p_cone_coords)

            sys.stdout.write("\r Completed step "+str(i+1)+" of "+ str(nsteps))
            sys.stdout.flush()
        return T, XI, E, X, P, N, DOT_P_CONE_COORDS, P_ALT



    def solve(self, xi0,e0,t0, dt, nsteps):
        """
        :param xi0:
        :param e0:
        :param t0:
        :param dt:
        :param nsteps:
        :param xi_ref: If note none will use as a fixed reference state
        :param t_ref: reference time for to compute d_xi_rho
        :return:
        """
        print("Starting the problem with m =", self.m, ", n =", self.n, ", d =", self.d, ", q = ", self.q)
        print("dim V = ", np.linalg.matrix_rank(self.v_basis(self.d_xi_phi(xi0), self.d_xi_rho(xi0,t0))))
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
                (xi_1, e_1, cone, dot_p_cone_coords) = self.solve_system_step_naive(xi_0, e_0, t_0, dt)
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


    def discrepancies_in_solution(self, T,XI,E,X, P, N, DOT_P_CONE_COORDS):
        nsteps=(T.shape[0])-1
        P_err=np.zeros((self.m, nsteps+1))
        for i in range(nsteps):
            P_err[:,i] = None if N[i] is None else (P[:,i+1]-P[:,i])/(T[i+1]-T[i]) - np.matmul(N[i], DOT_P_CONE_COORDS[i])

        return P_err



















