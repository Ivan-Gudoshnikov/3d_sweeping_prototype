from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize

from solver.quadprog_interface import QuadprogInterface, McGibbonQuadprog


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

    @abstractmethod
    def tangent_cone(self,  x):
        """
        :param x:
        :return:
        """


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
    eps = 1e-8

    def __init__(self, A,b,Aeq,beq):
        """
        :param A:
        :param b:
        :param Aeq:
        :param beq:
        """
        self.n=A.shape[1]
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

        x, active = method.quadprog(H, -H @ x, self.A, self.b, self.Aeq, self.beq)
        return x

    def linprog(self, c):
        result = scipy.optimize.linprog(c, self.A, self.b, self.Aeq, self.beq)
        if not result[2]:
            raise NameError("Can't do linprog, errorcode", result[6])
        return result[0]

    def __contains__(self, x):
        if x.shape[0]!=self.n:
            raise NameError("x is of wrong dimension!")
        if self.A is not None:
            ineq = all(np.matmul(self.A,x) -self.b <=self.eps)
        else:
            ineq = True
        if self.Aeq is not None:
            eq = (np.linalg.norm(np.matmul(self.Aeq, x)-self.beq, ord=np.inf) < self.eps)
        else:
            eq = True
        return ineq and eq

    def normal_cone(self, H, x):
        raise NameError("NOT IMPLEMENTED YET: To find the constraints of the normal cone to a polytope we need to solve a constraint enumeration problem ")

    def tangent_cone(self,  x):
        """
        :param x:
        :return:
        """
        if x not in self:
            raise NameError("x is not from the set!")
        cone_Aeq = self.Aeq
        if self.Aeq is not None:
            cone_beq = np.zeros_like(self.beq)
        else:
            cone_beq = None

        if self.A is not None:
            ineq = np.abs(np.matmul(self.A,x) -self.b) <=self.eps
            cone_A = self.A[ineq, :]
            cone_b = np.zeros(np.sum(ineq))
        else:
            cone_A = None
            cone_b = None
        return Polytope(cone_A, cone_b, cone_Aeq, cone_beq)

    def first_intersection_with_boundary(self, x0, xprime):
        if x0 not in self:
            raise NameError("x0 outside of the polytope")
        if np.linalg.norm(np.matmul(self.Aeq, xprime)) > self.eps:
            raise NameError("xprime takes outside of the polytope")

        # find all moments t_i when equality a_i (x0 + t_i xprime) = b_i holds for a row a_i of A
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.divide(self.b - np.matmul(self.A, x0), np.matmul(self.A, xprime))
            #if the nominator is zero, then we are already at the respective facet and we should not count for an intersection with it
            #if the denominator is zero, then xprime is parallel to the respective facet, inf value of the ratio is ok
            t[np.abs(self.b - np.matmul(self.A, x0)) <=self.eps] = np.Inf

        min = np.Inf
        for i in range(t.shape[0]):
            if t[i]>0 and t[i] < min:
                nmin=i
                min = t[i]
        if min == np.Inf:
            raise NameError("The polytope is unbounded in the direction xprime")

        return t[nmin], x0 + t[nmin]*xprime

class Box(Polytope):
    def __init__(self, bmin, bmax):
        self.bmin = bmin
        self.bmax = bmax
        n = bmin.shape[0]
        super().__init__(A=np.vstack((np.identity(n), -np.identity(n))), b=np.hstack((bmax,-bmin)), Aeq=None, beq=None)

    def get_active_box_faces(self, x, eps=None):
        if eps is None:
            eps = self.eps

        if x.shape[0] != self.n:
            raise NameError("x is of wrong dimension!")

        active = np.zeros_like(x)
        for i in range(self.n):
            if self.bmax[i]-x[i] < eps:
                active[i] = 1
            if x[i] - self.bmin[i] < eps:
                if active[i] == 1:
                    raise NameError("Equality constraints is a box are not supported yet")
                active[i] = -1
        return active

    def normal_cone(self, H, x, eps=None):
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

        active = self.get_active_box_faces(x,eps)

        for i in range(n):
            if active[i] == 1:
                if N is None:
                    N = np.expand_dims(np.identity(n)[:,i], axis=1)
                else:
                    N = np.hstack((N, np.expand_dims(np.identity(n)[:,i], axis=1)))
            if active[i] == -1:
                if N is None:
                    N = np.expand_dims(-np.identity(n)[:,i], axis=1)
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
    x=np.array([2, 0.1, -0.1])
    p,a = poly1.projection(H1,x,McGibbonQuadprog())
    print(p,a)

    x=np.array([0.1, 10, 3])
    p, a = poly1.projection(H1, x, McGibbonQuadprog())
    print(p, a)

    x=np.array([-0.2, -0.1, 3])
    p, a = poly1.projection(H1, x, McGibbonQuadprog())
    print(p, a)

    x = np.array([-10, -10, -10])
    p, a = poly1.projection(H1, x, McGibbonQuadprog())
    print(p, a)


    print("--------")
    box1 = Box(bmin=np.array([-5.,-5.]), bmax=np.array([1.,2.]))
    H2=np.diag([1.,1.])

    x=np.array([2,3])
    p, a=box1.projection(H2, x, McGibbonQuadprog())
    print(p, a)
    print(box1.normal_cone(H2,p).N)

    x=np.array([-6,-10])
    p, a = box1.projection(H2, x, McGibbonQuadprog())
    print(p, a)
    print(box1.normal_cone(H2, p).N)


    x=np.array([-6,7])
    p, a = box1.projection(H2, x, McGibbonQuadprog())
    print(p, a)
    print(box1.normal_cone(H2, p).N)


    x=np.array([0.1,0.1])
    p, a = box1.projection(H2, x, McGibbonQuadprog())
    print(p, a)
    print(box1.normal_cone(H2, p).N)


    x=np.array([10,0.21])
    p, a = box1.projection(H2, x, McGibbonQuadprog())
    print(p, a)
    print(box1.normal_cone(H2, p).N)


if __name__ == "__main__":
    test_polytope_projection()




