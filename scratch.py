import numpy as np

a=np.array([1., 2., -1])
b=np.array([2., 0., 0.])
np.seterr(divide='ignore')
print(np.divide(a,b))
print(np.Inf>0)
print(np.NINF>0)

c=np.array([1,-1,0,2])
print(c.shape[0])

print(int(2.65))

