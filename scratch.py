import numpy as np




class MyClass:
    def __init__(self, a):
        self.a=a    #some variable

        self.add = lambda b: self.a+b


my_object=MyClass(2)
print(my_object.add(2)) #prints 4