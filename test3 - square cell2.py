import numpy as np
import phi
import scipy


Q = np.array([[ 1, 0, 0,-1, 1, 0],
              [-1, 1, 0, 0, 0, 1],
              [ 0,-1, 1, 0,-1, 0],
              [ 0, 0,-1, 1, 0,-1]])


xi=np.array([[-1, 0],
             [0 ,1],
             [2, 1],
             [1, 0]])


print("xi=")
print(xi)
print("Q=")
print(Q)
print("Phi=")
print(phi.Phi(Q,xi))
print("N=")
print(phi.K(Q, xi))


DP=phi.DPhi(Q,xi)
print("DPhi=")
print(DP)

d = DP.shape[2]
for k in range(d):
    print(k,"=k, DP[:,:,k]=")
    print(DP[:, :, k])
DPmat=phi.DPhiMat(Q,xi)
print("DPmat=")
print(DPmat)
print("rank DPmat=")
print(np.linalg.matrix_rank(DPmat))

print("Ker DPmat=")
kerBasis=phi.kernelBasisDPhiMat(Q,xi)
print(kerBasis)
print("dim Ker DPmat=")
print(np.linalg.matrix_rank(kerBasis))

print("Ker DPmat*")
print(scipy.linalg.null_space(np.transpose(DPmat)))


