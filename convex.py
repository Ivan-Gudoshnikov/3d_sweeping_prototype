from abc import ABC, abstractmethod
import numpy as np

from quadprog_interface import QuadprogInterface, McGibbonQuadprog


class ConvexSet(ABC):

    @abstractmethod
    def projection(self, H, x, method):
        """
        abstract method for finding a closest point to a set
        :param H: inner product x^T.H.y  defines the norm
        :param x: the point to project
        :param method: link to a respective method to use
        :return: the closest point
        """
        pass

class Polytope(ConvexSet):
    def __init__(self, A,b,Aeq,beq):
        """
        :param A:
        :param b:
        :param Aeq:
        :param beq:
        """
        self.A=A
        self.b=b
        self.Aeq=Aeq
        self.beq=beq

    def projection(self, H, x, method: QuadprogInterface):
        """
        method for finding a closest point to a set using given implementation of a quadratic programming algorithm
        :param H: inner product x^T.H.y  defines the norm
        :param x: the point to project
        :param method: link to a implementation of a quadprog algorithm
        :return: the closest point
        """

        # (proj-x)^T.H.(proj-x) = proj^T.H.proj - 2.x^T.H.proj + x^T.H.x
        # 1/2 proj^T.H.proj + (- H.x)^T.proj
        #  1/2 x^T.H.x + c^T.x

        x = method.quadprog(H, -np.matmul(H,x), self.A, self.b, self.Aeq, self.beq)
        return x


def test_polytope_projection():
    Aeq=np.array([[1., 1., 1.]])
    beq=np.array([1.])
    A=-np.identity(3)
    b=np.array([0., 0., 0.])

    poly1=Polytope(A,b,Aeq,beq)

    H1=np.identity(3)
    x1=np.array([2, 0.1, -0.1])
    print(poly1.projection(H1,x1,McGibbonQuadprog()))

    x2=np.array([0.1, 10, 3])
    print(poly1.projection(H1, x2, McGibbonQuadprog()))

    x3=np.array([-0.2, -0.1, 3])
    print(poly1.projection(H1, x3, McGibbonQuadprog()))

    x4 = np.array([-10, -10, -10])
    print(poly1.projection(H1, x4, McGibbonQuadprog()))


test_polytope_projection()




