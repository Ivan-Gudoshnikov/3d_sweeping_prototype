import numpy as np
import  elastoplastic_process


print(int(0.135 // 0.1)% 2)
dt = 0.0001
nsteps = 2100
print(dt*nsteps)


u=[1]

print(len(u))
v=np.zeros((0,3))

v2=np.array([[1,1,1]])
v3=np.array([[2,2,2]])
v=np.append(v,v2,0)
v=np.append(v,v3,0)
print(v)

print(6//2)