from abc import ABC, abstractmethod

class BoundaryConditions(ABC):

    @abstractmethod
    def rho(self, outer, xi, t):
        pass

    @abstractmethod
    def d_xi_rho(self, outer, xi, t):
        pass

    @abstractmethod
    def d_t_rho(self, outer, xi, t):
        pass

    @abstractmethod
    def f(self, outer, t):
        pass

    @abstractmethod
    def q(self, outer):
        pass