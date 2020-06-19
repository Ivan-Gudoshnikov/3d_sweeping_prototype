import numpy as np
import phi



class ElastoplasticProcess:
    def __init__(self, Q, a, cminus, cplus, d_xi_rho, d_t_rho):
        #basic properties
        self.Q=Q

        self.m=Q.shape[1]
        self.n=Q.shape[0]


        self.A = np.diag(a)
        self.cminus = cminus
        self.cplus = cplus

        #function rho
        self.d_xi_rho = d_xi_rho
        self.d_t_rho = d_t_rho


        #distance function phi
        self.phi= lambda xi: phi.phi(self.Q,xi)
        self.K= lambda xi: phi.K(self.Q,xi)
        self.d_xi_phi= lambda xi: phi.d_xi_phi(self.Q,xi)

        #L, R









