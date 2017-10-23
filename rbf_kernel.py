import sklearn.metrics
import numpy as np

class RBF_Kernel():
    def __call__(self,B,B_prime,g=None):
        g=1.0/(B.shape[1]*B.shape[0])
        # default for gamma is 1/d
        L_Y = sklearn.metrics.pairwise.rbf_kernel(B,B_prime,gamma=g)
        return L_Y

# if x_i-y_i > epsilon, set it to 1 (the max value)

class RBF_Clipped_Kernel():
    def __call__(self, B, B_prime, epsilon=None, gamma=None):
        #import pdb; pdb.set_trace()
        d = len(B[0])
        L_Y = []
        if epsilon is None:
            # epsilon = 1-1.0/len(B)
            # to keep expected number of non-zero dimension distances to p:
            # 1/2 +- sqrt(1+p/d)/2  <--- might be wrong
            # to keep the expected number of non-zero distances to 2:
            # 1-sqrt(1-2/d)
            # another idea: set so only half of the points influence a given point, in expect
            epsilon = 1-np.sqrt(1-2.0/d)
        if gamma is None:
            gamma = 1.0/d
        for i in range(len(B_prime)):
            to_be_summed = np.square((B-B[i])*(abs(B-B[i])<epsilon)+(abs(B-B[i])>epsilon))
            L_Y.append(np.exp(-gamma*np.sum(to_be_summed,axis=1)))
            
        return L_Y
