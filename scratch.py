import numpy as np
import  elastoplastic_process



class MyClass:
    def __init__(self, a):
        self.a=a    #some variable

        self.add = lambda b: self.a+b


my_object=MyClass(2)
print(my_object.add(2)) #prints 4

print(np.zeros((5)))

v=elastoplastic_process.matrix_to_vector(np.array([[1, 2],
                                                   [3, 4],
                                                   [5, 6]]))
print(v)
print(elastoplastic_process.vector_to_matrix(v,2))