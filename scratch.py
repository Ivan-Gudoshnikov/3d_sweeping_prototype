import numpy as np
import scipy.linalg

n=5
d=3
m=2

xi0=np.array([0.1,0.2,0.3,
              0.4,0.5,0.6,
              0.7,0.8,0.9,
              1.0,1.1,1.2,
              1.3,1.4,1.5])
Q=np.array([[1,0],
            [0,0],
            [-1,1],
            [0,0],
            [0,-1]])

J = range(n)
K = range(d)
for i in range(m):
    print(np.sqrt(np.sum([np.square(np.dot(xi0[[d * j + k for j in J]], Q[:, i])) for k in K])))

print(scipy.linalg.null_space(Q.T))
print(scipy.linalg.orth(Q))

A=np.array([[1,2,3],
            [4,5,6]])
B=np.array([[-1,0,0],
            [0,-1,0],
            [0,0,-1]])
C=np.array([[2,0],
            [0,7],
            [0,5]])
print(A@B@C)
d=np.array([0.1,0.2,0.3])
print(B@d)
