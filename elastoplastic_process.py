import numpy as np
import phi
import scipy



class ElastoplasticProcess:
    def __init__(self, Q, a, cminus, cplus, d, rho, d_xi_rho, d_t_rho):
        #basic properties
        self.Q=Q

        self.m=Q.shape[1]
        self.n=Q.shape[0]


        self.A = np.diag(a)
        self.cminus = cminus
        self.cplus = cplus

        #spatial dimension
        self.d=d

        #function rho and its derivatives| rho: xi, t -> rho(xi,t)
        self.rho = rho
        self.d_xi_rho = d_xi_rho
        self.d_t_rho = d_t_rho


        #distance function phi
        self.phi = lambda xi: phi.phi(self.Q, xi)
        self.K = lambda xi: phi.K(self.Q, xi)
        self.d_xi_phi = lambda xi: phi.d_xi_phi(self.Q, xi)

        #basis in the nullspace of d_xi_rho
        self.ker_d_xi_rho = lambda xi, t: phi.matrix_to_tensor(scipy.linalg.null_space(phi.tensor_to_matrix(self.d_xi_rho(xi, t))).T,self.d)



        # Kernel of d_xi_rho:



        #L, R









