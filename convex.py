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


    @abstractmethod
    def __contains__(self, x):
        """
        Implement a check if a point x is from the set
        :param x:
        :return:
        """
        pass

    @abstractmethod
    def normal_cone(self, H, x):
        """
        the normal cone to the set at x in the sense of the inner product  x^T.H.y
        :param H: inner product x^T.H.y  needed to construct the normal cone
        :param x:
        :return: ConvexCone
        """
        pass

class ConvexCone(ABC):
    """
    TODO: implement it as a subclass of ConvexSet
    """

class FiniteCone(ConvexCone):
    """
    TODO: implement it as a subclass of Polytope, by deriving the constraints from N
    """
    def __init__(self, N):
        """
        :param N: each column is a generating direction
        """
        self.N=N


class Polytope(ConvexSet):
    eps = 1e-10

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

    def __contains__(self, x):
        if self.A is not None:
            ineq = all(np.matmul(self.A,x) <=self.b)
        else:
            ineq = True
        if self.Aeq is not None:
            eq = (np.linalg.norm(np.matmul(self.Aeq, x)-self.beq, ord=1) < self.eps)
        else:
            eq = True
        return ineq and eq

    def normal_cone(self, H, x):
        raise NameError("NOT IMPLEMENTED YET: To find the constraints of the normal cone to a polytope we need to solve a constraint enumeration problem ")


class Box(Polytope):
    def __init__(self, bmin, bmax):
        self.bmin = bmin
        self.bmax = bmax
        n = bmin.shape[0]
        super().__init__(A=np.vstack((np.identity(n), -np.identity(n))), b=np.hstack((bmax,-bmin)), Aeq=None, beq=None)

    def normal_cone(self, H, x):
        """
        H actually don't matter for a box
        :param H:
        :param x:
        :return:
        """
        if x not in self:
            raise NameError("x is not from the set!")

        n=x.shape[0]
        N = None
        for i in range(0,n):
            if self.bmax[i]-x[i]<self.eps:
                if N is None:
                    N = np.expand_dims(np.identity(n)[:,i], axis=1)
                else:
                    N = np.hstack((N, np.expand_dims(np.identity(n)[:,i], axis=1)))
        for i in range(0,n):
            if  x[i] - self.bmin[i] < self.eps:
                if N is None:
                    N = - np.expand_dims(np.identity(n)[:,i], axis=1)
                else:
                    N = np.hstack((N, np.expand_dims(-np.identity(n)[:,i], axis=1)))

        return FiniteCone(N)


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

    print("--------")
    box1=Box(bmin=np.array([-5.,-5.]), bmax=np.array([1.,2.]))
    H2=np.diag([1.,1.])

    x1=np.array([2,3])
    p1=box1.projection(H2, x1, McGibbonQuadprog())
    print(p1)
    print(box1.normal_cone(H2,p1).N)

    x2=np.array([-6,-10])
    p2 = box1.projection(H2, x2, McGibbonQuadprog())
    print(p2)
    print(box1.normal_cone(H2, p2).N)

    x3=np.array([-6,7])
    p3 = box1.projection(H2, x3, McGibbonQuadprog())
    print(p3)
    print(box1.normal_cone(H2, p3).N)


    x4=np.array([0.1,0.1])
    p4 = box1.projection(H2, x4, McGibbonQuadprog())
    print(p4)
    print(box1.normal_cone(H2, p4).N)

    x5=np.array([10,0.21])
    p5 = box1.projection(H2, x5, McGibbonQuadprog())
    print(p5)
    print(box1.normal_cone(H2, p5).N)



test_polytope_projection()




