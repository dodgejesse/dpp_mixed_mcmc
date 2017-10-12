import numpy

class Cube_Sampler():
    def __init__(self, d):
        self.d = d

    # returns a numpy array of dimension nxd
    def __call__(self, n):
        return numpy.random.rand(n,self.d)

    
