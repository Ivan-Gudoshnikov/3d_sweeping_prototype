import numpy as np

class BoundaryCondition:
    def __init__(self, q, rho, d_xi_rho, d_t_rho, f):
        """
        :param q: positive int
        :param rho: function: xi,t |-> rho value
        :param d_xi_rho: function: xi, t |-> martix q x (nd)
        :param d_t_rho: function xi,t |-> vector from R^q
        :param f: function: t |-> force vector from R^{nd}
        """
        self.q=q
        self.rho=rho
        self.d_xi_rho=d_xi_rho
        self.d_t_rho=d_t_rho
        self.f=f

class AffineBoundaryCondition(BoundaryCondition):
    """
    particular type of boundary condition with affine rho(xi,t)= R xi +r(t) = R(zeta + xi_0) +r(t)
    """
    def __init__(self, q, R, r, r_prime, f):
        self.R=R
        self.r=r
        self.r_prime=r_prime
        super().__init__(q=q, rho=lambda xi, t: R @ xi + r(t), d_xi_rho = lambda xi, t: R, d_t_rho = lambda xi, t: r_prime(t),f=f)
