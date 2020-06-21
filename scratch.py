import numpy as np
import  elastoplastic_process



class MyClass:
    def __init__(self, a):
        self.a=a    #some variable

        self.add = lambda b: self.a+b


my_object=MyClass(2)
print(my_object.add(2)) #prints 4

print(np.zeros((5)))

m1=np.array(np.array([[1, 2],
                      [3, 4],
                      [5, 6]]))

v=elastoplastic_process.matrix_to_vector(m1)
print(v)
print(elastoplastic_process.vector_to_matrix(v,2))


v1=np.array([-1,-2])
print(np.matmul(m1,v1))

print(1/v)

g=np.array([-1,4,-2,5,0,10])
print(v<=g)
print(all(v<=g))

u=np.array([[]])
print(u.shape)

print(np.identity(3)[1,:])

print(np.hstack((np.identity(3), np.expand_dims(np.identity(3)[:,1], axis=1))))