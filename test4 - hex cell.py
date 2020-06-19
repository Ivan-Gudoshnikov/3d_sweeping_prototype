import numpy as np
import phi


Q = np.array([[ 1, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0],
              [-1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [ 0,-1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [ 0, 0,-1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
              [ 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 1, 0],
              [ 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 1],
              [ 0, 0, 0, 0, 0, 0,-1,-1,-1,-1,-1,-1]])


xi=np.array([[-2, 0],
             [-1 ,1],
             [ 1, 1],
             [ 2, 0],
             [ 1,-1],
             [-1,-1],
             [ 0, 0]])


print("xi=")
print(xi)
print("Q=")
print(Q)
print("Phi=")
print(phi.phi(Q, xi))
print("N=")
print(phi.K(Q, xi))


DP=phi.d_xi_phi(Q, xi)
print("DPhi=")
print(DP)

d = DP.shape[2]
for k in range(d):
    print(k,"=k, DP[:,:,k]=")
    print(DP[:, :, k])

d_phi_mat = phi.tensor_to_matrix(phi.d_xi_phi(Q, xi))
print("DPmat=")
print(d_phi_mat)

print("rank DPmat=")
print(np.linalg.matrix_rank(d_phi_mat))
